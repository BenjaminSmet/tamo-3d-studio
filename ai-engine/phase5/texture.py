"""Skin texture (albedo) for the face model head, from the actor's photos.

One 4K albedo image in GNM's skin UV tile (UDIM 1001, already 0-1), the way 3DMM texture fitting does it:
- Photos first. Each texel of the head is projected into every photo (front, three-quarter, side, other side,
  back: whichever the run has) through the cameras the fit found (fit/cameras.json; photos are always projected
  as taken, never mirrored), weighted by visibility (a depth test against the fitted mesh), cos^4 of the view angle
  and a fall-off near silhouettes and photo edges.
- Shine out first: pixels brighter and less saturated than the skin around them take that skin's colour.
- Delighting, all photos together: luminance = albedo x ambient occlusion x a 2nd-order spherical-harmonics
  light per photo, fitted robustly on plain skin, alternating between one shared albedo and each photo's light
  so that overlapping photos must agree. The fitted shading is divided out; the hue is kept.
- Each turned photo matched to its neighbour in their overlap (median per channel, one spread gain), starting
  from the front.
- Multi-band (Laplacian pyramid) blending in UV space, so colour blends wide and detail stays sharp. Each photo
  also comes in mirrored (the other side of the face), at the coarse levels and half weight only: tone
  continuity for a side no photo sees, without copying moles or freckles across. A real photo of that side
  outweighs it.
- What no photo sees at all (the back of the head without a back photo, under the chin) gets a smooth push-pull
  extension of the photos' skin tone, as do the eye sockets and the gaps between UV islands. The eyeballs get
  a flat colour from a patch in the tile's unused bottom strip; MetaHuman brings its own eyes.

Writes albedo.png (sRGB), model_textured.glb (the hybrid head with UVs and the texture) and texture.json into
the run folder. Needs fit/ (fit.npz, keep_indices.npy, cameras.json) and hybrid/gnm_head.glb. Runs on the CPU
in about a minute, with about 7 GB of memory.

    python texture.py --run <face model run folder>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from gnm.shape import gnm_pytorch as G

from fit_gnm import LANDMARKER

SIZE = 4096
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
LEVELS = 7              # pyramid levels of the blend (4096 down to 32)
MIRROR_W = 0.5          # mirrored photos weigh half as much as direct ones...
MIRROR_FROM = 3         # ...and only from pyramid level 3 up (detail finer than ~16 texels stays direct)
S_LO, S_HI = 0.4, 2.0   # how far the fitted shading (and exposure) may brighten or darken a photo
RIDGE = np.array([0, 0.3, 0.3, 0.3, 1.5, 1.5, 1.5, 1.5, 1.5])   # one photo sees only part of the sphere of normals
EYE_UV = (0.5, 0.01)    # the tile's bottom strip (v < 0.04) holds no skin
EYE_SRGB = (205, 200, 192)
SOCK_LIN = (0.05, 0.012, 0.012)   # inside of the mouth
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]


def to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb8(x):
    x = np.clip(x, 0, 1)
    return np.round(np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)


def lum(c):
    return c @ LUM


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def raster_ids(polys, shape):
    """Index of the triangle covering each pixel (-1 where none); later triangles are drawn on top.
    polys: (P, 3, 2) continuous pixel coordinates (pixel centres at i + 0.5)."""
    img = np.zeros(shape + (3,), np.uint8)
    pts = np.round((polys - 0.5) * 16).astype(np.int32)
    for k, p in enumerate(pts):
        i = k + 1
        cv2.fillPoly(img, [p], (i & 255, (i >> 8) & 255, (i >> 16) & 255), lineType=cv2.LINE_8, shift=4)
    ids = img[..., 0].astype(np.int64) | (img[..., 1].astype(np.int64) << 8) | (img[..., 2].astype(np.int64) << 16)
    return ids - 1


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    w = np.clip(np.stack([1 - l1 - l2, l1, l2], 1), 0, None)
    return w / np.maximum(w.sum(1, keepdims=True), 1e-12)


def push_pull(img, known):
    """Fill where known == 0 with a smooth extension of the known colours (pyramid push-pull)."""
    known = known.astype(np.float32)
    pyr = []
    c, k = img * known[..., None], known
    while min(k.shape) > 8:
        pyr.append((c, k))
        c, k = cv2.pyrDown(c), cv2.pyrDown(k)
    est = c / np.maximum(k, 1e-6)[..., None]
    for c_l, k_l in reversed(pyr):
        up = cv2.pyrUp(est, dstsize=(c_l.shape[1], c_l.shape[0]))
        a = np.clip(k_l * 2, 0, 1)[..., None]
        est = np.where(k_l[..., None] > 1e-6, c_l / np.maximum(k_l, 1e-6)[..., None], up) * a + up * (1 - a)
    return est.astype(np.float32)


def lap_pyramid(img):
    g = [img]
    for _ in range(LEVELS):
        g.append(cv2.pyrDown(g[-1]))
    return [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(LEVELS)] + [g[-1]]


def gauss_pyramid(w):
    g = [w]
    for _ in range(LEVELS):
        g.append(cv2.pyrDown(g[-1]))
    return g


def collapse(lap):
    img = lap[-1]
    for l in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=(l.shape[1], l.shape[0])) + l
    return img


def sh9(n):
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], 1).astype(np.float32)


def fit_shading(Y, basis, ao, w):
    """Y ~ ao * (basis @ l), robust (Huber IRLS), weighted, ridge on the higher SH bands. Returns l."""
    A = (basis * ao[:, None]).astype(np.float64)
    Y = Y.astype(np.float64)
    wt = w.astype(np.float64)
    for _ in range(6):
        AtA = (A * wt[:, None]).T @ A
        l = np.linalg.solve(AtA + np.diag(RIDGE) * np.trace(AtA) / 9, (A * wt[:, None]).T @ Y)
        res = Y - A @ l
        k = 1.345 * (1.4826 * np.median(np.abs(res)) + 1e-6)
        wt = w * np.where(np.abs(res) <= k, 1.0, k / np.maximum(np.abs(res), 1e-9))
    return l


def vertex_ao(V, tris, normals, n_dirs=96, res=640, tol=0.002):
    """Cosine-weighted ambient occlusion per vertex, from depth maps in n_dirs directions."""
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    num, den = np.zeros(len(V)), np.zeros(len(V))
    used = np.unique(tris)
    for d in dirs:
        a = np.cross(d, [0, 1, 0] if abs(d[1]) < 0.9 else [1, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        P = np.stack([V @ a, V @ b], 1)
        lo, span = P[used].min(0), (P[used].max(0) - P[used].min(0)).max() * 1.02
        uv = (P - lo) / span * res
        depth = V @ d                                              # larger = nearer the light
        ot = tris[np.argsort(depth[tris].mean(1))]                 # far first
        ids = raster_ids(uv[ot], (res, res))
        px = np.clip(uv[used].astype(int), 0, res - 1)
        occ = ids[px[:, 1], px[:, 0]]
        tri = ot[np.maximum(occ, 0)]
        zb = (depth[tri] * barycentric(uv[used], uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]])).sum(1)
        vis = (occ < 0) | (depth[used] >= zb - tol)
        c = np.clip(normals[used] @ d, 0, None)
        num[used] += c * vis
        den[used] += c
    return (num / np.maximum(den, 1e-9)).astype(np.float32)


def suppress_shine(rgb):
    """Specular highlights (brighter and less saturated than the local skin colour) take the local colour."""
    loc = to_lin(cv2.medianBlur(to_srgb8(rgb), 31).astype(np.float32))
    r = lum(rgb) / np.maximum(lum(loc), 1e-5)
    sat = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1e-5)
    satl = (loc.max(-1) - loc.min(-1)) / np.maximum(loc.max(-1), 1e-5)
    m = smoothstep(1.06, 1.3, r) * np.clip((satl - sat) / (0.35 * satl + 1e-5), 0, 1)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)[..., None]
    return rgb * (1 - m) + loc * m, float(m.mean())


def eye_polygons(rgba):
    """MediaPipe's eye-opening contours in a photo (pixels), or [] when no face is found (profiles often)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    rgb = np.asarray(Image.alpha_composite(Image.new("RGBA", (rgba.shape[1], rgba.shape[0]), (128, 128, 128, 255)),
                                           Image.fromarray(rgba)).convert("RGB"))
    det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER)), num_faces=1))
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return []
    h, w = rgb.shape[:2]
    pts = np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]])
    return [pts[LEFT_EYE], pts[RIGHT_EYE]]


def match_stats(src, ref):
    """Medians per channel, and one spread gain from the luminance (separate channel gains tint the shadows)."""
    ms, mr = np.median(src, 0), np.median(ref, 0)
    ls, lr = lum(src), lum(ref)
    q = lambda x: np.subtract(*np.percentile(x, [84, 16])) / 2
    g = float(np.clip(q(lr) / max(q(ls), 1e-6) * (np.median(ls) / max(np.median(lr), 1e-6)), 0.85, 1.2))
    return ms, mr, g


def apply_match(x, ms, mr, g):
    """Per-channel gain to the reference median, then the relative spread about it scaled by g."""
    y = x * (mr / np.maximum(ms, 1e-6))
    return np.maximum(mr * (1 + (y / np.maximum(mr, 1e-6) - 1) * g), 0.05 * mr).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Skin texture (albedo) for a face model run.")
    ap.add_argument("--run", required=True, type=Path, help="the face model run folder")
    ap.add_argument("--size", type=int, default=SIZE)
    args = ap.parse_args()
    run, fit = args.run, args.run / "fit"
    S = args.size
    t0 = time.time()
    timing = {}
    stage = lambda text: print(text, flush=True)

    def done(key, since):
        timing[key] = round(time.time() - since, 1)

    stage("Baking the skin texture...")
    model = G.GNM.from_remote(version=G.GNMMajorVersion.V3, variant=G.GNMVariant.HEAD)
    gm = lambda n: np.asarray(model.vertex_group_mask(n)).astype(bool)
    tris_all = np.asarray(model.triangles)
    tiles = np.asarray(model.triangle_udim_tiles)
    tuv = np.asarray(model.triangle_uvs_udim, dtype=np.float64)
    V = np.load(fit / "fit.npz")["vertices"]
    keep = np.load(fit / "keep_indices.npy")
    kmask = np.zeros(len(V), bool)
    kmask[keep] = True
    cameras = json.loads((fit / "cameras.json").read_text())
    H = V.copy()
    H[keep] = np.asarray(trimesh.load(run / "hybrid" / "gnm_head.glb", force="mesh", process=False).vertices)
    kept_t = kmask[tris_all].all(1)
    skin_t = np.flatnonzero(kept_t & (tiles == 1001))
    eye_t = np.flatnonzero(kept_t & (tiles != 1001))
    kept_tris = tris_all[kept_t]
    normals = trimesh.Trimesh(V, kept_tris, process=False).vertex_normals

    # ---- texels of the skin tile
    t1 = time.time()
    pix = np.stack([tuv[skin_t][..., 0] * S, (1 - tuv[skin_t][..., 1]) * S], -1)
    ids = raster_ids(pix, (S, S))
    ys, xs = np.nonzero(ids >= 0)
    tid = ids[ys, xs]
    del ids
    bw = barycentric(np.stack([xs + 0.5, ys + 0.5], 1), pix[tid, 0], pix[tid, 1], pix[tid, 2]).astype(np.float32)
    corner = tris_all[skin_t][tid].astype(np.int32)
    del tid
    interp = lambda X: sum(bw[:, k, None] * X.astype(np.float32)[corner[:, k]] for k in range(3))
    Pf, N = interp(V), interp(normals)                               # on the fit, where the cameras were fitted
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    grp = lambda *names: np.any([gm(n) for n in names], 0)
    in_sock = grp("mouth_sock")[corner].all(1)
    in_socket = grp("eye_sockets")[corner].all(1)
    in_orbit = grp("left_orbital_region", "right_orbital_region")[corner].any(1)
    not_plain = grp("upper_lip", "lower_lip", "mouth_sock", "eye_sockets", "left_brow_region", "right_brow_region",
                    "ears")[corner].any(1)
    exterior = grp("skin_exterior")[corner].all(1)
    n_tex = len(xs)

    def to_image(vals, fill=0.0):
        img = np.full((S, S) + vals.shape[1:], fill, np.float32)
        img[ys, xs] = vals
        return img
    done("texels", t1)

    t1 = time.time()
    stage("Skin texture: ambient occlusion...")
    # softened: a photo's fill light reaches into concave parts more than pure ambient occlusion says
    AO = np.clip(interp(vertex_ao(V, kept_tris, normals)[:, None])[:, 0], 0.05, 1) ** 0.6
    done("ambient_occlusion", t1)

    # ---- the photos, projected
    t1 = time.time()
    stage("Skin texture: projecting the photos...")
    cols, weights, mirrored, info_views = {}, {}, {}, {}
    flip = np.array([-1, 1, 1], np.float32)
    for name, cam in cameras.items():
        R, t, f, w, h = np.array(cam["R"], np.float32), np.array(cam["t"], np.float32), cam["f"], cam["w"], cam["h"]
        rgba = np.asarray(Image.open(cam["image"]).convert("RGBA"))
        if rgba.shape[:2] != (h, w):
            sys.exit(f"{cam['image']} is not {w}x{h}")
        rgb, shine = suppress_shine(to_lin(rgba[..., :3].astype(np.float32)))
        alpha = rgba[..., 3] >= 128
        eyes = eye_polygons(rgba)
        if eyes:
            # the eye openings, grown past the lashes and the lid margin, so the fill takes skin, not lash
            hole = np.zeros((h, w), np.uint8)
            cv2.fillPoly(hole, [np.round(e).astype(np.int32) for e in eyes], 1)
            r_ = max(3, int(round(0.012 * max(h, w))))
            alpha &= cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))) == 0
        edge = np.clip(cv2.distanceTransform(alpha.astype(np.uint8), cv2.DIST_L2, 5) / 6.0, 0, 1).astype(np.float32)
        # depth buffer of the fitted mesh in this camera: front-most triangle per pixel, exact depth there
        Vc = V @ R.T + t
        Vuv = np.stack([f * Vc[:, 0] / Vc[:, 2] + w / 2, f * Vc[:, 1] / Vc[:, 2] + h / 2], 1)
        ot = kept_tris[np.argsort(-Vc[kept_tris, 2].mean(1))]
        zid = raster_ids(Vuv[ot], (h, w))
        C = -R.T @ t

        def sample(P, Nn):
            """Photo colour and weight for surface points P with normals Nn."""
            Xc = P @ R.T + t
            u = f * Xc[:, 0] / Xc[:, 2] + w / 2
            v = f * Xc[:, 1] / Xc[:, 2] + h / 2
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            occ = zid[np.clip(v.astype(int), 0, h - 1), np.clip(u.astype(int), 0, w - 1)]
            tri = ot[np.maximum(occ, 0)]
            zb = (Vc[tri, 2] * barycentric(np.stack([u, v], 1), Vuv[tri[:, 0]], Vuv[tri[:, 1]], Vuv[tri[:, 2]])).sum(1)
            vis = inside & (occ >= 0) & (Xc[:, 2] <= zb + 0.002)
            del tri
            d = C - P
            ndv = np.clip((Nn * d / np.linalg.norm(d, axis=1, keepdims=True)).sum(1), 0, 1)
            mapx, mapy = to_image((u - 0.5).astype(np.float32), -10), to_image((v - 0.5).astype(np.float32), -10)
            c = cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[ys, xs]
            a_edge = cv2.remap(edge, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[ys, xs]
            del mapx, mapy
            # feathered at occlusion boundaries: visibility blurred in UV, kept only where visible
            vis_soft = cv2.GaussianBlur(to_image(vis.astype(np.float32)), (0, 0), S / 128)[ys, xs]
            wg = vis * vis_soft ** 2 * a_edge * smoothstep(0.05, 0.3, ndv) * ndv ** 4
            wg[in_sock | in_socket] = 0
            return c, wg

        col, wgt = sample(Pf, N)
        col_m, wgt_m = sample(Pf * flip, N * flip)          # through the head's mirror image
        del zid
        cols[name], weights[name] = col.astype(np.float32), wgt.astype(np.float32)
        mirrored[name] = (col_m.astype(np.float32), (MIRROR_W * wgt_m).astype(np.float32))
        info_views[name] = {"seen": round(float((wgt > 0.05).mean()), 3), "shine": round(shine, 4)}
    done("photos", t1)

    # ---- delighting, all photos together (scale fixed by the front photo)
    t1 = time.time()
    stage("Skin texture: taking out the lighting...")
    rng = np.random.default_rng(0)
    B = sh9(N)
    fit_idx = {}
    for name in cols:
        idx = np.flatnonzero((weights[name] > 0.2) & exterior & ~not_plain)
        fit_idx[name] = rng.choice(idx, 200000, replace=False) if len(idx) > 200000 else idx
    Y = {k: lum(c) for k, c in cols.items()}
    A = np.ones(n_tex, np.float32)
    lights, ref = {}, "front" if "front" in cols else next(iter(cols))
    for _ in range(5):
        for name in cols:
            i_ = fit_idx[name]
            lights[name] = fit_shading(Y[name][i_] / np.maximum(A[i_], 1e-4), B[i_], AO[i_], weights[name][i_])
        shade = {k: np.clip(AO * (B @ l), 1e-4, None) for k, l in lights.items()}
        norm = np.median(shade[ref][fit_idx[ref]])
        num = sum(weights[k] * Y[k] / (shade[k] / norm) for k in cols)
        den_ = sum(weights[k] for k in cols)
        seen_any = den_ > 1e-3
        A = np.where(seen_any, num / np.maximum(den_, 1e-6), np.median(num[seen_any] / den_[seen_any])).astype(np.float32)
    for name in cols:
        s = np.clip(shade[name] / norm, S_LO, S_HI)
        cols[name] = (cols[name] / s[:, None]).astype(np.float32)
        s_m = np.clip(np.clip(AO * (sh9(N * flip) @ lights[name]), 1e-4, None) / norm, S_LO, S_HI)
        mirrored[name] = ((mirrored[name][0] / s_m[:, None]).astype(np.float32), mirrored[name][1])
        info_views[name]["light"] = np.round(lights[name] / norm, 3).tolist()
    del A, Y, shade, B

    # colour-matched between neighbouring views, where both see the skin well: starting from the front, each
    # view in turn to the already matched view it shares the most skin with (three-quarter to front, side to
    # three-quarter, ...); a view sharing too little skin with all of them (often the back) stays as it is
    overlap = lambda a_, b_: (weights[a_] > 0.15) & (weights[b_] > 0.15) & ~not_plain
    matched, todo = [ref], [k for k in cols if k != ref]
    while todo:
        name, ref_, both = max(((k, m, overlap(k, m)) for k in todo for m in matched), key=lambda x: x[2].sum())
        todo.remove(name)
        matched.append(name)
        if both.sum() > 2000:
            ms, mr, g = match_stats(cols[name][both], cols[ref_][both])
            mr = ms * np.clip(mr / np.maximum(ms, 1e-6), 0.75, 1.33)
            cols[name] = apply_match(cols[name], ms, mr, g)
            mirrored[name] = (apply_match(mirrored[name][0], ms, mr, g), mirrored[name][1])
            info_views[name]["matched_to"] = ref_
    den = sum(weights.values())
    done("delight", t1)

    # ---- what no photo sees (the back of the head, under the chin): a smooth extension of the photos' skin
    t1 = time.time()
    stage("Skin texture: filling in unseen skin...")
    photo_mean = sum(weights[k][:, None] * cols[k] for k in cols) / np.maximum(den, 1e-6)[:, None]
    seen = (den > 0.05) & ~(in_sock | in_socket)
    base = push_pull(to_image(photo_mean), to_image(seen.astype(np.float32)))
    base = cv2.GaussianBlur(base, (0, 0), S / 256)[ys, xs]          # no copied detail, only the tone
    base_ok = ~(in_orbit | in_socket | in_sock)
    del photo_mean
    done("unseen_fill", t1)

    # ---- multi-band blend in UV space, normalised per level (the mirrored photos join at the coarse levels)
    t1 = time.time()
    stage("Skin texture: blending...")
    texel_img = to_image(np.ones(n_tex, np.float32))
    w_base = np.where(base_ok, 0.001, 0.0).astype(np.float32)       # only where no photo sees
    unknown = (den + w_base) < 1e-4
    sources = [(k, cols[k], weights[k], 0) for k in cols]
    sources += [(f"{k}_mirror", c, w_, MIRROR_FROM) for k, (c, w_) in mirrored.items()]
    sources += [("fill", base, w_base, 0)]
    num, wsum = None, None
    for name, c, wv, lvl0 in sources:
        cimg = push_pull(to_image(c), to_image((wv > 0).astype(np.float32)))
        W = to_image(wv)
        if name == "fill":
            W[texel_img == 0] = 1.0                                      # gaps between islands: any smooth colour
            W[ys[unknown], xs[unknown]] = 1.0
        lp, gp = lap_pyramid(cimg), gauss_pyramid(W)
        del cimg, W
        if num is None:
            num, wsum = [np.zeros_like(l) for l in lp], [np.zeros_like(g) for g in gp]
        for i in range(lvl0, len(lp)):
            num[i] += lp[i] * gp[i][..., None]
            wsum[i] += gp[i]
        del lp, gp
    col = collapse([n_ / np.maximum(w_, 1e-6)[..., None] for n_, w_ in zip(num, wsum)])[ys, xs]
    del num, wsum, mirrored

    # fills: eye surroundings no photo sees, unknown texels, then the gaps between islands
    refill = in_socket | (in_orbit & (den < 0.3)) | unknown
    col[in_sock] = SOCK_LIN
    known = to_image((~refill).astype(np.float32))
    img = push_pull(to_image(col), known)
    img = push_pull(img, (known > 0) | (to_image(refill.astype(np.float32)) > 0))
    out8 = to_srgb8(img)
    out8[int((1 - 0.02) * S):] = EYE_SRGB
    Image.fromarray(out8).save(run / "albedo.png")
    done("blend", t1)

    # ---- the hybrid head with per-corner UVs and the texture (eyeballs on the flat patch)
    stage("Skin texture: writing the textured head...")
    ftris = np.concatenate([tris_all[skin_t], tris_all[eye_t]])
    cuv = np.concatenate([tuv[skin_t], np.broadcast_to(np.array(EYE_UV), (len(eye_t), 3, 2))]).reshape(-1, 2)
    uniq, inv = np.unique(np.column_stack([ftris.reshape(-1), np.round(cuv * 1e6)]).astype(np.int64), axis=0,
                          return_inverse=True)
    inv = inv.reshape(-1)
    first = np.zeros(len(uniq), int)
    first[inv] = np.arange(len(inv))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(out8), metallicFactor=0.0,
                                                   roughnessFactor=0.6)
    trimesh.Trimesh(H[uniq[:, 0]], inv.reshape(-1, 3), process=False,
                    visual=trimesh.visual.TextureVisuals(uv=cuv[first], material=material)).export(run / "model_textured.glb")
    elapsed = time.time() - t0
    (run / "texture.json").write_text(json.dumps({
        "seconds": round(elapsed, 1), "size": S, "texels": int(n_tex), "views": info_views, "timing": timing},
        indent=1))
    print(f"Skin texture done in {elapsed:.0f} s: {run / 'albedo.png'}", flush=True)


if __name__ == "__main__":
    main()
