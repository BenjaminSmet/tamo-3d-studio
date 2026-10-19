"""The generation engine behind one adapter, so a future model can replace TRELLIS.2.

An adapter turns a job into a command and reads progress from the command's output.
The command writes everything into the run folder, ending with run.json.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

PIPELINES = ("512", "1024", "1024_cascade", "1536_cascade")


@dataclass
class Progress:
    stage: str
    fraction: float


class Trellis2Engine:
    """TRELLIS.2 through ai-engine/phase1/run_phase1.py (Espalier-2)."""

    name = "trellis2"
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
    BAR = re.compile(r"^\s*(.+?):\s+(\d{1,3})%\|")

    def __init__(self, settings: Settings):
        self.settings = settings

    def command(self, image: Path, project: str, actor: str, run_dir: Path, options: dict) -> list[str]:
        cmd = [
            # -u: unbuffered, or the script's output reaches the server in 8 KB bursts and progress stalls.
            str(self.settings.engine_python), "-u", str(self.settings.engine_script),
            "--image", str(image), "--actor", actor, "--project", project,
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
        """The new progress if this output line moves it on, else None."""
        for marker, stage, fraction in self.STAGES:
            if marker in line and fraction > current.fraction:
                return Progress(stage, fraction)
        # A sampler bar fills the first band with its name that isn't finished yet, so a bar
        # that restarts at 0% (the cascade's second shape pass) moves into the next band.
        match = self.BAR.search(line)
        if match:
            name, step = match.group(1), min(int(match.group(2)), 100) / 100
            for bar, stage, start, end in self.BARS:
                if bar == name and end > current.fraction:
                    if stage != current.stage and step >= 1:
                        break   # tqdm redraws a finished bar at 100%; that isn't the next pass starting
                    fraction = max(start + step * (end - start), current.fraction)
                    if fraction > current.fraction + 0.005 or stage != current.stage:
                        return Progress(stage, fraction)
                    break
        return None

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
