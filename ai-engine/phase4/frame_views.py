"""Frame several cut-out views of one head the same way, with cameras Pixal3D understands.

Pixal3D builds the head from posed views: each image is a camera looking at the same
3D box, so every view has to show that box at the same size and in the same place.
Each view is cropped square and scaled by the height from the top of the head to the
bottom of the chin, which stays the same from every side at eye level; the top of the
head sits on the same row and the head's silhouette is centred, so the same 3D box fills
the same share of every frame. The narrowest point of the neck is no good for this: in a
front view it sits level with the lower face, in a profile well down the throat.

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


def chin_row(alpha: np.ndarray, top: int, neck: int, direction: str) -> int | None:
    """The bottom of the chin in a turned view, or None if the face edge shows no chin.

    Walking down the face's edge (the side the face points to), the edge comes forward to
    the nose and chin, then jumps back under the jaw to the throat; the chin is where that
    jump starts.
    """
    height = neck - top
    rows = np.arange(top, neck)
    filled = alpha[top:neck]
    has = filled.any(axis=1)
    if not has.all():
        return None
    if direction == "left":
        edge = -filled.argmax(axis=1).astype(float)                        # bigger = further forward
    else:
        edge = (filled.shape[1] - filled[:, ::-1].argmax(axis=1)).astype(float)
    edge = np.convolve(edge, np.ones(5) / 5, mode="same")
    start = int(0.45 * height)                                              # below the eyes
    nose = start + int(np.argmax(edge[start:int(0.85 * height)]))
    step = max(3, int(0.04 * height))
    back = edge[nose:-step] - edge[nose + step:]                            # how far the edge falls back
    if len(back) == 0:
        return None
    jump = int(np.argmax(back))
    head_width = filled.sum(axis=1).max()
    if back[jump] < 0.08 * head_width:
        return None
    return int(rows[nose + jump + step // 2])


def front_chin_row(rgba: Image.Image, alpha: np.ndarray, top: int, neck: int) -> int | None:
    """The bottom of the chin in a front view, or None if it doesn't show clearly.

    From the front the chin sits inside the silhouette, in front of the neck, so the outline
    can't show it; its lit underside meeting the shadowed throat can. The narrowest point of
    the neck's outline sits level with the lower face, so from just above it look down a
    central strip half the neck wide for the strongest light-to-dark step. Starting higher
    would catch the shadows under the nose and lips instead.
    """
    height = neck - top
    start = neck - int(0.05 * height)
    end = min(alpha.shape[0] - 1, neck + int(0.25 * height))
    neck_cols = np.flatnonzero(alpha[neck])
    if len(neck_cols) == 0 or end - start < 8:
        return None
    centre, half = neck_cols.mean(), 0.25 * (neck_cols[-1] - neck_cols[0])
    c0, c1 = int(centre - half), int(centre + half) + 1
    rgb = np.asarray(rgba.convert("RGB"), dtype=float)[start:end, c0:c1]
    inside = alpha[start:end, c0:c1]
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    count = inside.sum(axis=1)
    if (count == 0).any():
        return None
    rows = (luma * inside).sum(axis=1) / count
    rows = np.convolve(np.pad(rows, 2, mode="edge"), np.ones(5) / 5, mode="valid")   # no dip at the ends
    k = max(2, int(0.015 * height))
    drop = rows[:-2 * k] - rows[2 * k:]                                                # light above, dark below
    if len(drop) == 0:
        return None
    best = int(np.argmax(drop))
    if drop[best] <= 0 or drop[best] < 0.09 * (rows.max() - rows.min() + 1e-6):
        return None
    return start + best + k


def head_box(rgba: Image.Image, direction: str | None = None) -> dict:
    """Top of the head, the neck, the chin, and the head's left and right edges, in pixels.

    direction: None for a front view (the chin from front_chin_row, else the narrowest point
    of the neck, which sits just under it), or "left"/"right" for a turned view facing that way.
    """
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
    box = {"top": top, "neck": neck, "bottom": bottom, "left": int(cols[0]), "right": int(cols[-1]) + 1,
           "neck_centre": float(neck_cols.mean()) if len(neck_cols) else None}
    chin = chin_row(alpha, top, neck, direction) if direction else front_chin_row(rgba, alpha, top, neck)
    box["chin"] = chin if chin is not None else neck
    box["chin_found"] = chin is not None
    if chin is not None:
        cols = np.flatnonzero(alpha[top:chin].any(axis=0))
        box["left"], box["right"] = int(cols[0]), int(cols[-1]) + 1
    return box


def facing(box: dict) -> str:
    """Which way a turned head faces in the image: the face sticks out past the neck, the skull less so."""
    centre = (box["left"] + box["right"]) / 2
    return "left" if box["neck_centre"] is not None and box["neck_centre"] > centre else "right"


def frame(rgba: Image.Image, box: dict) -> Image.Image:
    """Square crop, sized by the top of the head to the chin alone, never by where the photo
    ends, so the head is the same 3D size and height in every view."""
    head_height = box["chin"] - box["top"]
    crop_bottom = box["chin"] + int(BELOW_NECK * head_height)
    side = int((crop_bottom - box["top"]) * (1 + 2 * MARGIN))
    cx = (box["left"] + box["right"]) // 2
    left, top = cx - side // 2, box["top"] - int(MARGIN * (crop_bottom - box["top"]))
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
        box = head_box(rgba, None if azimuth == 0 else ("left" if azimuth > 0 else "right"))
        file_name = f"view{i:02d}_{name}.png"
        frame(rgba, box).save(out_dir / file_name)
        frames.append({"file_path": file_name, "name": name, "transform_matrix": camera_to_world(azimuth)})
        report.append({"name": name, "azimuth": azimuth, "head_height_px": box["chin"] - box["top"],
                       "chin_found": box["chin_found"], "facing": facing(box)})
    (out_dir / "transforms.json").write_text(json.dumps(
        {"camera_angle_x": math.radians(fov_deg), "mesh_scale": 1.0, "frames": frames}, indent=2))
    return {"fov_deg": fov_deg, "views": report}
