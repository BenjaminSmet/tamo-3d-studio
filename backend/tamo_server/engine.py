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

    # The script's own messages, in order, and roughly how far along each one is.
    STAGES = [
        ("Removing background", "Removing the background", 0.03),
        ("Cropped to head", "Cropping to head and neck", 0.08),
        ("No shoulders found", "Cropping to head and neck", 0.08),
        ("GPU memory in use before generation", "Generating the 3D head", 0.10),
        ("Exporting the MetaHuman", "Exporting for MetaHuman", 0.92),
        ("Result:", "Finishing", 0.99),
    ]
    GENERATE_START, GENERATE_END = 0.10, 0.90
    PERCENT = re.compile(r"(\d{1,3})%\|")

    def __init__(self, settings: Settings):
        self.settings = settings

    def command(self, image: Path, project: str, actor: str, run_dir: Path, options: dict) -> list[str]:
        cmd = [
            str(self.settings.engine_python), str(self.settings.engine_script),
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
            if marker in line and fraction >= current.fraction:
                return Progress(stage, fraction)
        # Espalier-2's progress bars, mapped into the generation band. The stages inside it
        # (shape, then texture) each restart at 0%, so only ever move forward.
        if current.stage == "Generating the 3D head":
            match = self.PERCENT.search(line)
            if match:
                step = min(int(match.group(1)), 100) / 100
                fraction = self.GENERATE_START + step * (self.GENERATE_END - self.GENERATE_START)
                if fraction > current.fraction + 0.005:
                    return Progress(current.stage, fraction)
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
