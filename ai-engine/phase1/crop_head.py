"""Crop a cut-out portrait (RGBA) square on the head and neck.

TRELLIS.2 scales whatever is in the cut-out to fill its 3D box, so shoulders and
shirt take detail away from the face. Framing from the top of the head to the
collarbone gave clearly better heads than a headshot down to the chest.

The neck is found from the alpha mask alone: walking down from the top of the
head, the silhouette widens to the head's widest point, narrows at the neck,
then widens again at the shoulders.
"""

import argparse

import numpy as np
from PIL import Image

ALPHA_THRESHOLD = 128
# How far below the narrowest point of the neck to keep, in head heights.
# 0.35 reaches roughly the collarbone.
BELOW_NECK = 0.35
MARGIN = 0.04


def find_neck_row(widths: np.ndarray, top: int, bottom: int) -> int | None:
    """Returns the row where the neck is narrowest, or None if the silhouette has no head shape."""
    w = np.convolve(widths[top:bottom].astype(float), np.ones(9) / 9, mode="same")

    # The head's widest row: the silhouette has widened and then clearly narrowed again.
    peak = 0
    for i in range(len(w)):
        if w[i] > w[peak]:
            peak = i
        elif w[i] < 0.85 * w[peak]:
            break
    else:
        return None

    # The neck: the narrowest row before the shoulders spread out.
    neck = peak
    for i in range(peak, len(w)):
        if w[i] < w[neck]:
            neck = i
        elif w[i] > 1.5 * w[neck]:
            break
    return top + neck


def crop_head_and_neck(rgba: Image.Image) -> tuple[Image.Image, bool]:
    """Returns the square crop, and whether a crop was made.

    A cut-out that already ends at the neck just gets squared up.
    """
    alpha = np.asarray(rgba.getchannel("A")) >= ALPHA_THRESHOLD
    rows = np.flatnonzero(alpha.any(axis=1))
    if len(rows) == 0:
        return rgba, False
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    widths = alpha.sum(axis=1)

    neck = find_neck_row(widths, top, bottom)
    if neck is None:
        return rgba, False

    head_height = neck - top
    crop_bottom = min(bottom, neck + int(BELOW_NECK * head_height))
    cols = np.flatnonzero(alpha[top:crop_bottom].any(axis=0))
    left, right = int(cols[0]), int(cols[-1]) + 1

    side = int(max(crop_bottom - top, right - left) * (1 + 2 * MARGIN))
    cx = (left + right) // 2
    cy = (top + crop_bottom) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)

    # Pixels outside the original image come out transparent; everything below
    # the crop line is cleared so shoulders don't bleed back in.
    cropped = rgba.crop(box)
    cut = crop_bottom - box[1]
    if 0 < cut < side:
        a = np.asarray(cropped.getchannel("A")).copy()
        a[cut:] = 0
        cropped.putalpha(Image.fromarray(a))
    return cropped, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="RGBA cut-out")
    parser.add_argument("output")
    args = parser.parse_args()

    cropped, changed = crop_head_and_neck(Image.open(args.image).convert("RGBA"))
    cropped.save(args.output)
    print("Cropped to head and neck." if changed else "No shoulders found; left uncropped.")


if __name__ == "__main__":
    main()
