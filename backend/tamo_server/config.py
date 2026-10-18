"""Paths and engine location, from environment variables with defaults for Benjamin's WSL setup."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    library: Path           # library/projects
    database: Path          # database/studio.sqlite
    logs: Path              # one log per job
    engine_python: Path     # Espalier-2's Python, which has torch and trimesh
    engine_script: Path     # run_phase1.py, or a stand-in for tests
    espalier: Path
    app_dist: Path          # built interface, served at / when present

    @classmethod
    def from_env(cls) -> "Settings":
        def path(name: str, default: Path) -> Path:
            return Path(os.environ.get(name, default)).expanduser()

        espalier = path("TAMO_ESPALIER", Path.home() / "espalier")
        database = path("TAMO_DATABASE", REPO_ROOT / "database" / "studio.sqlite")
        return cls(
            library=path("TAMO_LIBRARY", REPO_ROOT / "library" / "projects"),
            database=database,
            logs=path("TAMO_LOGS", database.parent / "logs"),
            engine_python=path("TAMO_ENGINE_PYTHON", espalier / ".venv" / "bin" / "python"),
            engine_script=path("TAMO_ENGINE_SCRIPT", REPO_ROOT / "ai-engine" / "phase1" / "run_phase1.py"),
            espalier=espalier,
            app_dist=path("TAMO_APP_DIST", REPO_ROOT / "app" / "dist"),
        )
