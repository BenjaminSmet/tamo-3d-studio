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
    # Multi-view (Pixal3D) lives in its own copy of Espalier-2; see docs/multiview.md.
    multiview_python: Path | None = None
    multiview_script: Path | None = None
    pixal3d: Path | None = None
    # The face model (GNM) has its own Python 3.12 environment; see docs/facemodel.md.
    facemodel_python: Path | None = None
    facemodel_script: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        def path(name: str, default: Path) -> Path:
            return Path(os.environ.get(name, default)).expanduser()

        espalier = path("TAMO_ESPALIER", Path.home() / "espalier")
        pixal3d = path("TAMO_PIXAL3D", Path.home() / "espalier-pixal3d")
        database = path("TAMO_DATABASE", REPO_ROOT / "database" / "studio.sqlite")
        return cls(
            library=path("TAMO_LIBRARY", REPO_ROOT / "library" / "projects"),
            database=database,
            logs=path("TAMO_LOGS", database.parent / "logs"),
            engine_python=path("TAMO_ENGINE_PYTHON", espalier / ".venv" / "bin" / "python"),
            engine_script=path("TAMO_ENGINE_SCRIPT", REPO_ROOT / "ai-engine" / "phase1" / "run_phase1.py"),
            espalier=espalier,
            app_dist=path("TAMO_APP_DIST", REPO_ROOT / "app" / "dist"),
            multiview_python=path("TAMO_MULTIVIEW_PYTHON", pixal3d / ".venv" / "bin" / "python"),
            multiview_script=path("TAMO_MULTIVIEW_SCRIPT", REPO_ROOT / "ai-engine" / "phase4" / "run_multiview.py"),
            pixal3d=pixal3d,
            facemodel_python=path("TAMO_FACEMODEL_PYTHON", Path.home() / "tamo-gnm" / ".venv" / "bin" / "python"),
            facemodel_script=path("TAMO_FACEMODEL_SCRIPT", REPO_ROOT / "ai-engine" / "phase5" / "run_facemodel.py"),
        )
