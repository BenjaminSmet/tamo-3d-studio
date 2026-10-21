"""The generation engines, each behind one adapter, so models can be added or replaced.

An adapter turns a job into a command and reads progress from the command's output.
The command writes everything into the run folder, ending with run.json.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

PIPELINES = ("512", "1024", "1024_cascade", "1536_cascade")
TURNED_VIEWS = ("side", "three_quarter")


@dataclass
class Progress:
    stage: str
    fraction: float


BAR = re.compile(r"^\s*(.+?):\s+(\d{1,3})%\|")


def advance(line: str, current: Progress, stages: list, bars: list) -> Progress | None:
    """The new progress if this output line moves it on, else None.

    stages: (marker, stage, fraction) for messages that mark a step. bars: (name, stage, start,
    end) for sampler progress bars, in the order they run; a bar fills the first band whose name
    it starts with that isn't finished yet, so a bar that restarts at 0% (a second pass with the
    same name) moves into the next band.
    """
    for marker, stage, fraction in stages:
        if marker in line and fraction > current.fraction:
            return Progress(stage, fraction)
    match = BAR.search(line)
    if match:
        name, step = match.group(1), min(int(match.group(2)), 100) / 100
        for bar, stage, start, end in bars:
            if name.startswith(bar) and end > current.fraction:
                if stage != current.stage and step >= 1:
                    break   # tqdm redraws a finished bar at 100%; that isn't the next pass starting
                fraction = max(start + step * (end - start), current.fraction)
                if fraction > current.fraction + 0.005 or stage != current.stage:
                    return Progress(stage, fraction)
                break
    return None


class Trellis2Engine:
    """TRELLIS.2 through ai-engine/phase1/run_phase1.py (Espalier-2)."""

    name = "trellis2"
    run_name = ""
    views = ()
    needs_turned_view = False
    defaults = {"texture": True, "pipeline": "1024_cascade", "seed": 42, "smooth": 1.0, "crop": True}

    # The script's and Espalier-2's own messages, in order, and how far along each one is, in
    # time, from a timed 1024_cascade textured run on an RTX 5070 Ti (about 3.5 minutes; loading
    # the models alone takes about 100 s and prints nothing, so it gets a big step of its own).
    STAGES = [
        ("Removing background", "Removing the background", 0.01),
        ("Cropped to head", "Cropping to head and neck", 0.12),
        ("No shoulders found", "Cropping to head and neck", 0.12),
        ("GPU memory in use before generation", "Loading the models", 0.13),
        ("generated {", "Building the mesh", 0.75),
        ("Running Dual Contouring", "Building the mesh", 0.77),
        ("Simplifying", "Simplifying the mesh", 0.80),
        ("Parameterizing new mesh", "Unwrapping the texture", 0.84),
        ("Sampling attributes", "Baking the texture", 0.88),
        ("exported /", "Saving the model", 0.91),
        ("Exporting the MetaHuman", "Exporting for MetaHuman", 0.93),
        ("Result:", "Finishing", 0.99),
    ]
    # Espalier-2's sampler progress bars, in the order they run, each with its own band. The
    # cascade pipelines sample the shape twice (low, then high resolution); 512 and 1024 once,
    # in which case the second shape band is skipped.
    BARS = [
        ("Sampling sparse structure", "Generating: rough shape", 0.51, 0.53),
        ("Sampling shape SLat", "Generating: shape", 0.55, 0.57),
        ("Sampling shape SLat", "Generating: shape detail", 0.57, 0.66),
        ("Sampling texture SLat", "Generating: texture", 0.66, 0.73),
    ]
    def __init__(self, settings: Settings):
        self.settings = settings

    def found(self) -> bool:
        return self.settings.engine_python.is_file() and self.settings.engine_script.is_file()

    def command(self, images: dict[str, Path], project: str, actor: str, run_dir: Path, options: dict) -> list[str]:
        cmd = [
            # -u: unbuffered, or the script's output reaches the server in 8 KB bursts and progress stalls.
            str(self.settings.engine_python), "-u", str(self.settings.engine_script),
            "--image", str(images["front"]), "--actor", actor, "--project", project,
            "--library", str(self.settings.library), "--espalier", str(self.settings.espalier),
            "--run-dir", str(run_dir),
            "--pipeline", options["pipeline"], "--seed", str(options["seed"]), "--smooth", str(options["smooth"]),
        ]
        if options["texture"]:
            cmd.append("--texture")
        if not options["crop"]:
            cmd.append("--no-crop")
        return cmd

    def progress(self, line: str, current: Progress) -> Progress | None:
        return advance(line, current, self.STAGES, self.BARS)

    def validate(self, options: dict) -> dict:
        merged = {**self.defaults, **{k: v for k, v in options.items() if v is not None}}
        unknown = set(merged) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        if merged["pipeline"] not in PIPELINES:
            raise ValueError(f"pipeline must be one of {', '.join(PIPELINES)}")
        merged["texture"] = bool(merged["texture"])
        merged["crop"] = bool(merged["crop"])
        merged["seed"] = int(merged["seed"])
        merged["smooth"] = float(merged["smooth"])
        if not 0 <= merged["smooth"] <= 3:
            raise ValueError("smooth must be between 0 and 3")
        return merged


class Pixal3DEngine:
    """Pixal3D (multi-view) through ai-engine/phase4/run_multiview.py: the front image plus a
    side and/or three-quarter one, in their own copy of Espalier-2."""

    name = "pixal3d"
    run_name = "multiview-"
    views = TURNED_VIEWS
    needs_turned_view = True
    defaults = {"texture": True, "pipeline": "1024_cascade", "seed": 42, "smooth": 1.0,
                "side_id": None, "three_quarter_id": None}

    # Timed like TRELLIS.2's, from a 1024_cascade textured run on an RTX 5070 Ti (about 4.7
    # minutes: 20 s cutting out and framing, 70 s loading, 65 s sampling, 110 s building and
    # baking the mesh, 15 s for the MetaHuman export).
    STAGES = [
        ("Removing background", "Removing the background", 0.01),
        ("Cropped to head and neck", "Framing the views", 0.07),
        ("GPU memory in use before generation", "Loading the models", 0.09),
        ("generated {", "Building the mesh", 0.56),
        ("Running Dual Contouring", "Building the mesh", 0.60),
        ("Simplifying", "Simplifying the mesh", 0.66),
        ("Parameterizing new mesh", "Unwrapping the texture", 0.70),
        ("Sampling attributes", "Baking the texture", 0.80),
        ("exported /", "Saving the model", 0.92),
        ("Exporting the MetaHuman", "Exporting for MetaHuman", 0.93),
        ("Result:", "Finishing", 0.99),
    ]
    # The high-resolution pass names its resolution ("..., 1024)"), so bars match by prefix.
    BARS = [
        ("Sampling sparse structure (proj)", "Generating: rough shape", 0.33, 0.35),
        ("Sampling shape SLat (proj)", "Generating: shape", 0.35, 0.37),
        ("Sampling HR shape SLat", "Generating: shape detail", 0.37, 0.46),
        ("Sampling texture SLat (proj)", "Generating: texture", 0.46, 0.52),
    ]

    def __init__(self, settings: Settings):
        self.settings = settings

    def found(self) -> bool:
        python, script = self.settings.multiview_python, self.settings.multiview_script
        return bool(python and script and python.is_file() and script.is_file())

    def command(self, images: dict[str, Path], project: str, actor: str, run_dir: Path, options: dict) -> list[str]:
        cmd = [
            str(self.settings.multiview_python), "-u", str(self.settings.multiview_script),
            "--front", str(images["front"]), "--actor", actor, "--project", project,
            "--library", str(self.settings.library), "--espalier", str(self.settings.pixal3d),
            "--run-dir", str(run_dir),
            "--pipeline", options["pipeline"], "--seed", str(options["seed"]), "--smooth", str(options["smooth"]),
        ]
        for view in TURNED_VIEWS:
            if view in images:
                cmd += [f"--{view.replace('_', '-')}", str(images[view])]
        if options["texture"]:
            cmd.append("--texture")
        return cmd

    def progress(self, line: str, current: Progress) -> Progress | None:
        return advance(line, current, self.STAGES, self.BARS)

    def validate(self, options: dict) -> dict:
        merged = {**self.defaults, **{k: v for k, v in options.items() if v is not None}}
        unknown = set(merged) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        if merged["pipeline"] not in ("1024_cascade", "1536_cascade"):
            raise ValueError("pipeline must be 1024_cascade or 1536_cascade for multi-view")
        merged["texture"] = bool(merged["texture"])
        merged["seed"] = int(merged["seed"])
        merged["smooth"] = float(merged["smooth"])
        if not 0 <= merged["smooth"] <= 3:
            raise ValueError("smooth must be between 0 and 3")
        return merged


class GnmEngine:
    """Google's GNM face model through ai-engine/phase5/run_facemodel.py, in its own Python: a
    front-only TRELLIS.2 head first, then GNM fitted to the front, side and three-quarter photos,
    with the TRELLIS.2 head's face shape added on top. The side views are optional."""

    name = "gnm"
    run_name = "facemodel-"
    views = TURNED_VIEWS
    needs_turned_view = False
    defaults = {"texture": False, "pipeline": "1024_cascade", "seed": 42, "side_id": None, "three_quarter_id": None}

    # Timed on an RTX 5070 Ti (about 5.5 minutes): TRELLIS.2 without a texture about 2 minutes,
    # most of it loading the models, which prints nothing; the fit about 2.5 minutes (most of it in
    # five stages of loosening the face model, each printed when done); the hybrid about 30 s and
    # the MetaHuman export a few seconds.
    STAGES = [
        ("Building the rough head with TRELLIS.2", "Building a rough head", 0.01),
        ("Cropped to head", "Building a rough head", 0.04),
        ("No shoulders found", "Building a rough head", 0.04),
        ("GPU memory in use before generation", "Loading the models", 0.05),
        ("generated {", "Building the rough mesh", 0.36),
        ("exported /", "Saving the rough head", 0.40),
        ("Removing background from the turned views", "Cutting out the side views", 0.41),
        ("Fitting the face model", "Fitting the face model: cameras", 0.45),
        ("[rigid]", "Fitting the face model: head shape", 0.47),
        ("[identity]", "Fitting the face model: head shape", 0.51),
        ("[v1]", "Fitting the face model: face detail", 0.58),
        ("[dense reg 5.0]", "Fitting the face model: face detail", 0.64),
        ("[dense reg 2.0]", "Fitting the face model: face detail", 0.70),
        ("[dense reg 1.0]", "Fitting the face model: face detail", 0.75),
        ("[dense reg 0.5]", "Fitting the face model: face detail", 0.80),
        ("[dense reg 0.25]", "Fitting the face model: face detail", 0.86),
        ("Adding the face shape", "Adding the face shape", 0.87),
        ("Exporting the MetaHuman", "Exporting for MetaHuman", 0.98),
        ("Face model run finished", "Finishing", 0.99),
    ]
    BARS = [
        ("Sampling sparse structure", "Generating: rough shape", 0.28, 0.29),
        ("Sampling shape SLat", "Generating: shape", 0.29, 0.30),
        ("Sampling shape SLat", "Generating: shape detail", 0.30, 0.34),
    ]

    def __init__(self, settings: Settings):
        self.settings = settings

    def found(self) -> bool:
        python, script = self.settings.facemodel_python, self.settings.facemodel_script
        return bool(python and script and python.is_file() and script.is_file() and self.settings.engine_python.is_file())

    def command(self, images: dict[str, Path], project: str, actor: str, run_dir: Path, options: dict) -> list[str]:
        cmd = [
            str(self.settings.facemodel_python), "-u", str(self.settings.facemodel_script),
            "--front", str(images["front"]), "--actor", actor, "--project", project,
            "--library", str(self.settings.library), "--espalier", str(self.settings.espalier),
            "--engine-python", str(self.settings.engine_python), "--run-dir", str(run_dir),
            "--pipeline", options["pipeline"], "--seed", str(options["seed"]),
        ]
        for view in TURNED_VIEWS:
            if view in images:
                cmd += [f"--{view.replace('_', '-')}", str(images[view])]
        return cmd

    def progress(self, line: str, current: Progress) -> Progress | None:
        return advance(line, current, self.STAGES, self.BARS)

    def validate(self, options: dict) -> dict:
        merged = {**self.defaults, **{k: v for k, v in options.items() if v is not None}}
        unknown = set(merged) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        if merged["pipeline"] not in PIPELINES:
            raise ValueError(f"pipeline must be one of {', '.join(PIPELINES)}")
        if merged["texture"]:
            raise ValueError("The face model builds shape only; MetaHuman makes the skin.")
        merged["seed"] = int(merged["seed"])
        return merged
