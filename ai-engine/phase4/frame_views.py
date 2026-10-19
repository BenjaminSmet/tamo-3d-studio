"""Frame several cut-out views of one head the same way, with cameras Pixal3D understands.

Pixal3D builds the head from posed views: each image is a camera looking at the same
3D box, so every view has to show that box at the same size and in the same place.
Each view is cropped square from the top of the head to the collarbone, with the
head's own silhouette centred, so the same 3D box fills the same share of every frame.

Cameras follow Pixal3D's convention (Blender/NeRF): world Z up, the front camera at
(0, -d, 0) looking at the origin, each camera looking along its own -Z with +Y up.
Azimuth turns the camera around Z, counter-clockwise seen from above, so +90 is a camera
at +X: it sees the actor's left cheek, with the face pointing to the left of the image.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from crop_head import ALPHA_THRESHOLD, BELOW_NECK, MARGIN, find_neck_row  # noqa: E402

# Pixal3D's shipped orbit rig: 20 degree field of view, 3.1192 units away, at eye level.
FOV_DEG = 20.0
DISTANCE = 3.1192
SIZE = 1024


def camera_to_world(azimuth_deg: float, elevation_deg: float = 0.0, distance: float = DISTANCE) -> list:
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    position = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)]) * distance
    back = position / np.linalg.norm(position)            # the camera's +Z points away from the head
    right = np.cross(np.array([0.0, 0.0, 1.0]), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = right, up, back, position
    return np.round(matrix, 6).tolist()


def head_box(rgba: Image.Image) -> dict:
    """Top of the head, the neck, and the head's left and right edges, in pixels."""
    alpha = np.asarray(rgba.getchannel("A")) >= ALPHA_THRESHOLD
    rows = np.flatnonzero(alpha.any(axis=1))
    if len(rows) == 0:
        raise ValueError("The cut-out is empty.")
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    neck = find_neck_row(alpha.sum(axis=1), top, bottom)
    if neck is None:
        raise ValueError("Couldn't find the neck; the view needs the head and some shoulder.")
    cols = np.flatnonzero(alpha[top:neck].any(axis=0))
    neck_cols = np.flatnonzero(alpha[neck])
    return {"top": top, "neck": neck, "bottom": bottom, "left": int(cols[0]), "right": int(cols[-1]) + 1,
            "neck_centre": float(neck_cols.mean()) if len(neck_cols) else None}


def facing(box: dict) -> str:
    """Which way a turned head faces in the image: the face sticks out past the neck, the skull less so."""
    centre = (box["left"] + box["right"]) / 2
    return "left" if box["neck_centre"] is not None and box["neck_centre"] > centre else "right"


def frame(rgba: Image.Image, box: dict) -> Image.Image:
    """Square crop, sized by the head's height alone so it's the same 3D size in every view."""
    head_height = box["neck"] - box["top"]
    crop_bottom = min(box["bottom"], box["neck"] + int(BELOW_NECK * head_height))
    side = int((crop_bottom - box["top"]) * (1 + 2 * MARGIN))
    cx, cy = (box["left"] + box["right"]) // 2, (box["top"] + crop_bottom) // 2
    left, top = cx - side // 2, cy - side // 2
    cropped = rgba.crop((left, top, left + side, top + side))
    cut = crop_bottom - top
    if 0 < cut < side:   # nothing below the collarbone, as in Phase 1
        a = np.asarray(cropped.getchannel("A")).copy()
        a[cut:] = 0
        cropped.putalpha(Image.fromarray(a))
    return cropped.resize((SIZE, SIZE), Image.LANCZOS)


def write_views(views: list[tuple[str, Image.Image, float]], out_dir: Path, fov_deg: float = FOV_DEG) -> dict:
    """views: (name, RGBA cut-out, azimuth), front first. Writes the images and transforms.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, report = [], []
    for i, (name, rgba, azimuth) in enumerate(views):
        box = head_box(rgba)
        file_name = f"view{i:02d}_{name}.png"
        frame(rgba, box).save(out_dir / file_name)
        frames.append({"file_path": file_name, "name": name, "transform_matrix": camera_to_world(azimuth)})
        report.append({"name": name, "azimuth": azimuth, "head_height_px": box["neck"] - box["top"],
                       "facing": facing(box)})
    (out_dir / "transforms.json").write_text(json.dumps(
        {"camera_angle_x": math.radians(fov_deg), "mesh_scale": 1.0, "frames": frames}, indent=2))
    return {"fov_deg": fov_deg, "views": report}
