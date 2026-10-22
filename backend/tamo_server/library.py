"""The project library on disk: folder names, reference uploads, and finding existing runs.

    library/projects/<project>/actors/<actor>/references/
                                             /generations/<timestamp>-<mode>/
    library/trash/<timestamp>/<project>/...   deleted items, same layout, until you empty it
"""

import io
import json
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .db import Database, now

VIEWS = ("front", "three_quarter", "side", "other_side", "back", "look", "unsorted")
IMAGE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_SIDE = 8192
MIN_SIDE = 256


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    if not slug:
        raise ValueError("The name needs at least one letter or digit.")
    return slug


def actor_dir(library: Path, project_slug: str, actor_slug: str) -> Path:
    return library / project_slug / "actors" / actor_slug


def new_run_dir(library: Path, project_slug: str, actor_slug: str, mode: str) -> Path:
    """A generations/ folder name that doesn't exist yet; runs are never overwritten."""
    base = actor_dir(library, project_slug, actor_slug) / "generations" / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}"
    candidate, n = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{n}")
        n += 1
    return candidate


def trash_dir(library: Path) -> Path:
    """Next to library/projects, so scanning the projects never brings deleted items back."""
    return library.parent / "trash"


def move_to_trash(library: Path, path: Path) -> Path | None:
    """Moves a file or folder into library/trash/<timestamp>/, keeping its place in the library.

    Nothing is erased: to undo, move it back and press Rescan. Returns where it went, or None if it was already gone.
    """
    if not path.exists():
        return None
    batch = trash_dir(library) / f"{datetime.now():%Y%m%d-%H%M%S}"
    try:
        target = batch / path.resolve().relative_to(library.resolve())
    except ValueError:   # outside the library (shouldn't happen); keep just its name
        target = batch / path.name
    n = 2
    while target.exists():
        target = target.with_name(f"{target.name}-{n}")
        n += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


class InvalidImage(ValueError):
    pass


def check_image(data: bytes) -> tuple[str, int, int]:
    """Returns (extension, width, height), or raises InvalidImage. Runs before anything is saved."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage("The image is larger than 50 MB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            fmt, (width, height) = image.format, image.size
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImage(f"That file isn't a readable image ({error}).") from error
    if fmt not in IMAGE_FORMATS:
        raise InvalidImage("Use a PNG, JPEG or WebP image.")
    if max(width, height) > MAX_SIDE:
        raise InvalidImage(f"The image is larger than {MAX_SIDE} pixels on a side.")
    if min(width, height) < MIN_SIDE:
        raise InvalidImage(f"The image is smaller than {MIN_SIDE} pixels on a side.")
    return IMAGE_FORMATS[fmt], width, height


def save_reference(library: Path, project_slug: str, actor_slug: str, view: str, original_name: str,
                   data: bytes) -> tuple[Path, int, int]:
    extension, width, height = check_image(data)
    references = actor_dir(library, project_slug, actor_slug) / "references"
    references.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_")[:60] or "image"
    path = references / f"{view}-{datetime.now():%Y%m%d-%H%M%S}-{stem}{extension}"
    n = 2
    while path.exists():
        path = path.with_stem(f"{path.stem}-{n}")
        n += 1
    path.write_bytes(data)
    return path, width, height


@lru_cache(maxsize=4096)
def windows_path(path: str) -> str | None:
    """The \\\\wsl.localhost\\... path Explorer can open, when running under WSL."""
    if not shutil.which("wslpath"):
        return None
    result = subprocess.run(["wslpath", "-w", path], capture_output=True, text=True)
    return result.stdout.strip() or None


def ensure_project(db: Database, slug: str, name: str | None = None) -> int:
    row = db.one("SELECT id FROM projects WHERE slug = ?", (slug,))
    if row:
        return row["id"]
    return db.insert("projects", slug=slug, name=name or slug.replace("_", " ").title(), created=now())


def ensure_actor(db: Database, project_id: int, slug: str, name: str | None = None) -> int:
    row = db.one("SELECT id FROM actors WHERE project_id = ? AND slug = ?", (project_id, slug))
    if row:
        return row["id"]
    return db.insert("actors", project_id=project_id, slug=slug,
                     name=name or slug.replace("_", " ").title(), created=now())


def scan(db: Database, library: Path) -> int:
    """Registers projects, actors, references and finished runs already on disk. Returns how many runs were new."""
    added = 0
    if not library.is_dir():
        return added
    for project in sorted(p for p in library.iterdir() if p.is_dir()):
        actors = project / "actors"
        if not actors.is_dir():
            continue
        project_id = ensure_project(db, project.name)
        for actor in sorted(a for a in actors.iterdir() if a.is_dir()):
            actor_id = ensure_actor(db, project_id, actor.name)
            references = actor / "references"
            if references.is_dir():
                for image in sorted(references.iterdir()):
                    if image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                        continue
                    if db.one("SELECT id FROM reference_images WHERE path = ?", (str(image),)):
                        continue
                    try:
                        with Image.open(image) as opened:
                            width, height = opened.size
                    except OSError:
                        continue
                    view = image.name.split("-", 1)[0] if image.name.split("-", 1)[0] in VIEWS else "unsorted"
                    db.insert("reference_images", actor_id=actor_id, view=view, path=str(image),
                              width=width, height=height, created=now())
            generations = actor / "generations"
            if not generations.is_dir():
                continue
            for run_dir in sorted(g for g in generations.iterdir() if (g / "run.json").is_file()):
                if db.one("SELECT id FROM generations WHERE run_dir = ?", (str(run_dir),)):
                    continue
                try:
                    record = json.loads((run_dir / "run.json").read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                db.insert("generations", actor_id=actor_id, job_id=None, run_dir=str(run_dir),
                          record=json.dumps(record), created=record.get("started") or now())
                added += 1
    return added
