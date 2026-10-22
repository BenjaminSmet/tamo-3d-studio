"""Fit Google's GNM head model (v3, head variant) to an actor's portrait cut-outs.

Inputs are RGBA cut-outs (background removed) for a front view and optionally a
three-quarter and a side view, all with the camera on the actor's left for turned views
(the face points to the left of the image), as the multi-view pipeline uses.

Unknowns: GNM head identity (170), a few lower-face expression components (strongly
regularised), a small neck rotation, and per view a camera (rotation, translation, focal).
Losses: MediaPipe landmarks mapped to iBUG-68 against GNM's HEAD_SPARSE_68 (front and
three-quarter), and a renderer-free silhouette chamfer (all views) between the projected
skin outline and the photo's head outline above the chin, ignoring outline pixels on the
image border (a photo that crops the skull says nothing about the skull there).

    python fit_gnm.py --actor amara --front F.png --three-quarter T.png --side S.png
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from gnm.shape import gnm_pytorch as G

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase4"))
import frame_views  # noqa: E402

# MediaPipe Face Landmarker (Apache 2.0), downloaded by install.sh.
LANDMARKER = Path(os.environ.get("TAMO_LANDMARKER", Path.home() / "tamo-gnm" / "models" / "face_landmarker.task"))

MP_TO_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66,
            107, 336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385,
            387, 263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
LM_WEIGHT = np.ones(68)
LM_WEIGHT[0:17] = 0.0          # jaw contour: the silhouette handles it
LM_WEIGHT[17:27] = 0.5         # brows
SIL_W = 2.0
AZIMUTH = {"front": 0.0, "three_quarter": 45.0, "side": 90.0, "back": 180.0}
BACK_SIL = 0.5                 # the back's outline weighs half: hair, a collar or a bad cut-out mustn't pull the skull
torch.set_default_dtype(torch.float64)


def mediapipe_68(rgb: np.ndarray, model: Path) -> np.ndarray | None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    det = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)), num_faces=1))
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    return pts[MP_TO_68]


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def axis_angle(v: torch.Tensor) -> torch.Tensor:
    theta = torch.sqrt((v * v).sum() + 1e-12)
    k = v / theta
    K = torch.stack([torch.stack([torch.zeros(()), -k[2], k[1]]),
                     torch.stack([k[2], torch.zeros(()), -k[0]]),
                     torch.stack([-k[1], k[0], torch.zeros(())])])
    return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


class View:
    """One photo and its camera. kind is front, three_quarter, side or back (the camera's starting azimuth and
    what the photo can show); name tells views of the same kind apart, e.g. side and other_side, both handed in
    facing left. A back view has no face points: it is fitted from the head's outline above the neck only."""

    def __init__(self, name: str, path: Path, mp_model: Path, kind: str | None = None):
        self.name = name
        self.kind = kind = kind or name
        rgba = Image.open(path).convert("RGBA")
        self.rgba = rgba
        self.w, self.h = rgba.size
        a = np.asarray(rgba.getchannel("A")) >= frame_views.ALPHA_THRESHOLD
        self.alpha = a
        if kind == "back":
            box = frame_views.head_box(rgba, "left")
            box["chin"] = box["neck"]        # no chin to see: the outline down to the neck
        else:
            box = frame_views.head_box(rgba, None if kind == "front" else "left")
        self.box = box
        self.sil_w = BACK_SIL if kind == "back" else 1.0
        # photo outline above the chin, without pixels on the image border
        edge = a & ~cv2.erode(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.nonzero(edge)
        keep = (ys < box["chin"]) & (xs > 1) & (xs < self.w - 2) & (ys > 1)
        pts = np.stack([xs[keep], ys[keep]], 1).astype(float)
        if len(pts) > 1500:
            pts = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
        self.outline = torch.tensor(pts)
        rgb = np.asarray(Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba).convert("RGB"))
        self.rgb = rgb
        self.lm = mediapipe_68(rgb, mp_model) if kind in ("front", "three_quarter") else None
        # In a profile MediaPipe is unreliable; anchor the nose tip and chin from the face's edge instead
        # (the face points to the image's left: the edge is the first filled column of each row).
        self.anchors = None
        if kind == "side":
            top, chin = box["top"], box["chin"]
            hgt = chin - top
            rows = np.arange(top + int(0.45 * hgt), top + int(0.80 * hgt))
            left_cols = np.array([np.flatnonzero(a[r])[0] for r in rows])
            nose_r = rows[int(np.argmin(left_cols))]
            chin_cols = np.flatnonzero(a[chin - 2])
            self.anchors = {30: (float(left_cols.min()), float(nose_r)), 8: (float(chin_cols[0]), float(chin - 2))}
        # camera unknowns: rotation delta, translation, log focal
        self.rot = torch.zeros(3, requires_grad=True)
        self.R0 = torch.tensor(np.diag([1.0, -1.0, -1.0]) @ rot_y(-AZIMUTH[kind]))
        f0 = (max(self.w, self.h) / 2) / math.tan(math.radians(20) / 2)
        self.logf = torch.tensor(math.log(f0), requires_grad=True)
        self.t = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)

    def R(self):
        return axis_angle(self.rot) @ self.R0

    def project(self, X: torch.Tensor):
        Xc = X @ self.R().T + self.t
        f = torch.exp(self.logf)
        u = f * Xc[:, 0] / Xc[:, 2] + self.w / 2
        v = f * Xc[:, 1] / Xc[:, 2] + self.h / 2
        return torch.stack([u, v], 1), Xc[:, 2]

    def focal_bounds(self):
        lo = (max(self.w, self.h) / 2) / math.tan(math.radians(40) / 2)
        hi = (max(self.w, self.h) / 2) / math.tan(math.radians(8) / 2)
        return math.log(lo), math.log(hi)


def silhouette_vertices(uv: np.ndarray, tris: np.ndarray, w: int, h: int, skin_idx: np.ndarray) -> np.ndarray:
    """Indices (into uv) of skin vertices that sit on the projected mask's outline."""
    mask = np.zeros((h, w), np.uint8)
    poly = np.round(uv[tris]).astype(np.int32)
    cv2.fillPoly(mask, list(poly), 1)
    # Outer contour only: the skin mesh has holes at the eyes and mouth whose rims aren't silhouette.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    border = np.zeros_like(mask)
    cv2.drawContours(border, contours, -1, 1, 1)
    dist = cv2.distanceTransform((1 - border).astype(np.uint8), cv2.DIST_L2, 3)
    p = np.round(uv[skin_idx]).astype(int)
    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    near = np.zeros(len(skin_idx), bool)
    near[inside] = dist[p[inside, 1], p[inside, 0]] < 2.0
    return skin_idx[near]


def chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(a, b)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--front", required=True, type=Path)
    ap.add_argument("--three-quarter", type=Path)
    ap.add_argument("--side", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--expr", type=int, default=20, help="lower-face expression components to free")
    args = ap.parse_args()
    out = args.out / args.actor
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    model = G.GNM.from_remote(version=G.GNMMajorVersion.V3, variant=G.GNMVariant.HEAD).double()
    names = list(model.identity_names)
    head_ids = [i for i, n in enumerate(names) if n.startswith("head_")]
    expr_names = list(model.expression_names)
    expr_ids = [i for i, n in enumerate(expr_names) if n.startswith("lower_face_region")][:args.expr]
    def group_mask(name):
        m = model.vertex_group_mask(name)
        return np.asarray(m.detach().cpu() if torch.is_tensor(m) else m).astype(bool)
    skin = group_mask("skin_exterior")
    skin_idx = np.flatnonzero(skin)
    tris_all = np.asarray(model.triangles.detach().cpu() if torch.is_tensor(model.triangles) else model.triangles)
    skin_tris = tris_all[skin[tris_all].all(1)]
    njoints = model.num_joints
    neck_j = list(model.joint_names).index("neck")

    mp_model = LANDMARKER
    views = [View("front", args.front, mp_model)]
    if args.three_quarter:
        views.append(View("three_quarter", args.three_quarter, mp_model))
    if args.side:
        views.append(View("side", args.side, mp_model))
    for v in views:
        print(f"{v.name}: {v.w}x{v.h}, landmarks {'yes' if v.lm is not None else 'no'}, outline pts {len(v.outline)}, "
              f"crown {v.box['top']} chin {v.box['chin']}")

    ident = torch.zeros(len(head_ids), requires_grad=True)
    expr = torch.zeros(len(expr_ids), requires_grad=True)
    neck = torch.zeros(3, requires_grad=True)
    lm_w = torch.tensor(LM_WEIGHT)

    def geometry():
        full_id = torch.zeros(model.identity_dim)
        full_id = full_id.index_put((torch.tensor(head_ids),), ident)
        full_ex = torch.zeros(model.expression_dim)
        full_ex = full_ex.index_put((torch.tensor(expr_ids),), expr)
        rots = torch.zeros(njoints, 3)
        rots = rots.index_put((torch.tensor([neck_j]),), neck[None])
        verts, lms = model.vertices_and_landmarks(G.GNMLandmarksType.HEAD_SPARSE_68, identity=full_id[None],
                                                  expression=full_ex[None], rotations=rots[None])
        return verts[0], lms[0]

    # Initial translation per view: put the head's centre in front of the camera at the distance that matches
    # the photo's crown-chin height.
    with torch.no_grad():
        V0, L0 = geometry()
        head_h = float(V0[:, 1].max() - L0[8, 1])
        centre = V0[skin_idx].mean(0)
        for v in views:
            f = math.exp(float(v.logf))
            px_h = v.box["chin"] - v.box["top"]
            dist = f * head_h / px_h
            Xc = centre @ v.R().T
            mid_v = (v.box["top"] + v.box["chin"]) / 2
            mid_u = (v.box["left"] + v.box["right"]) / 2
            v.t.data = torch.tensor([(mid_u - v.w / 2) * dist / f - float(Xc[0]),
                                     (mid_v - v.h / 2) * dist / f - float(Xc[1]),
                                     dist - float(Xc[2])])

    def losses(V, L, use_sil, iteration):
        total, parts = 0.0, {}
        for v in views:
            uv, z = v.project(V)
            if v.lm is not None:
                luv, _ = v.project(L)
                err = ((luv - torch.tensor(v.lm)) ** 2).sum(1)
                lm_loss = (lm_w * err).sum() / lm_w.sum()
                total = total + lm_loss
                parts[f"lm_{v.name}"] = float(torch.sqrt((lm_w * err).sum() / lm_w.sum()).detach())
            if v.anchors:
                luv, _ = v.project(L)
                idx = list(v.anchors)
                target = torch.tensor([v.anchors[i] for i in idx])
                err = ((luv[idx] - target) ** 2).sum(1)
                total = total + err.mean()
                parts[f"anchor_{v.name}"] = float(torch.sqrt(err.mean()).detach())
            # keep each camera near its nominal angle and a portrait lens
            f20 = (max(v.w, v.h) / 2) / math.tan(math.radians(20) / 2)
            total = total + 2000.0 * (v.rot ** 2).sum() + 200.0 * (v.logf - math.log(f20)) ** 2
            if use_sil:
                if iteration % 5 == 0 or not hasattr(v, "sil_idx"):
                    v.sil_idx = silhouette_vertices(uv.detach().numpy(), skin_tris, v.w, v.h, skin_idx)
                    # only the part above the projected chin
                    chin_v = float(v.project(L[8:9])[0][0, 1])
                    sel = uv.detach().numpy()[v.sil_idx][:, 1] < min(chin_v, v.box["chin"])
                    v.sil_idx = v.sil_idx[sel]
                if len(v.sil_idx) > 10 and len(v.outline) > 10:
                    c = chamfer(uv[v.sil_idx], v.outline)
                    total = total + SIL_W * c ** 2
                    parts[f"sil_{v.name}"] = float(c)
        return total, parts

    def run(params, iters, use_sil, reg_id, reg_ex, lr, label, use_lm_only_views=False):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(iters):
            opt.zero_grad()
            V, L = geometry()
            data, parts = losses(V, L, use_sil, it)
            reg = reg_id * (ident ** 2).sum() + reg_ex * (expr ** 2).sum() + 50.0 * (neck ** 2).sum()
            loss = data + reg
            loss.backward()
            opt.step()
            with torch.no_grad():
                for v in views:
                    lo, hi = v.focal_bounds()
                    v.logf.clamp_(lo, hi)
                neck.clamp_(-0.2, 0.2)
            if it % 50 == 0 or it == iters - 1:
                print(f"[{label}] it {it:4d} loss {float(loss):10.2f} " +
                      " ".join(f"{k} {val:.2f}" for k, val in parts.items()), flush=True)
        return parts

    cams = [p for v in views for p in (v.rot, v.t, v.logf)]
    # 1) rigid: cameras from landmarks (front/3q) and silhouette for the side
    run(cams, 300, use_sil=True, reg_id=1e6, reg_ex=1e6, lr=0.01, label="rigid")
    # 2) identity + landmarks, strong regularisation
    run(cams + [ident, expr, neck], 300, use_sil=False, reg_id=20.0, reg_ex=500.0, lr=0.03, label="identity")
    # 3) add silhouettes, loosen identity
    parts = run(cams + [ident, expr, neck], 600, use_sil=True, reg_id=5.0, reg_ex=200.0, lr=0.02, label="silhouette")
    elapsed = time.time() - t_start

    with torch.no_grad():
        V, L = geometry()
    V = V.numpy()
    np.savez(out / "fit.npz", identity=ident.detach().numpy(), expression=expr.detach().numpy(),
             neck=neck.detach().numpy(), vertices=V)
    report = {"seconds": round(elapsed, 1), "errors_px": parts,
              "identity_abs_max": float(ident.abs().max()), "identity_rms": float(ident.pow(2).mean().sqrt()),
              "expression_rms": float(expr.pow(2).mean().sqrt()),
              "views": {v.name: {"fov_deg": round(2 * math.degrees(math.atan((max(v.w, v.h) / 2) / math.exp(float(v.logf)))), 2)}
                        for v in views}}
    (out / "fit.json").write_text(json.dumps(report, indent=1))
    print("REPORT", json.dumps(report))

    # overlays
    for v in views:
        img = Image.fromarray(v.rgb.copy())
        d = ImageDraw.Draw(img)
        with torch.no_grad():
            uv, _ = v.project(torch.tensor(V))
            luv, _ = v.project(L)
        uv, luv = uv.numpy(), luv.numpy()
        mask = np.zeros((v.h, v.w), np.uint8)
        cv2.fillPoly(mask, list(np.round(uv[skin_tris]).astype(np.int32)), 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            d.line([tuple(p[0]) for p in cnt] + [tuple(cnt[0][0])], fill=(0, 255, 255), width=2)
        o = v.outline.numpy()
        for x, y in o[::3]:
            d.point((x, y), fill=(255, 0, 255))
        if v.lm is not None:
            for (x, y) in v.lm:
                d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 60, 60))
        for i, (x, y) in enumerate(luv):
            if v.lm is not None and LM_WEIGHT[i] == 0:
                continue
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(60, 255, 60))
        img.save(out / f"overlay_{v.name}.png")

    # mesh: skin + eyeballs, glTF Y-up, face +Z, metres (GNM's own convention)
    import trimesh
    keep_groups = ["skin", "left_eye", "right_eye"]
    keep = np.zeros(len(V), bool)
    for gname in keep_groups:
        keep |= group_mask(gname)
    tris = tris_all[keep[tris_all].all(1)]
    remap = -np.ones(len(V), int)
    remap[np.flatnonzero(keep)] = np.arange(keep.sum())
    mesh = trimesh.Trimesh(V[keep], remap[tris], process=False)
    mesh.export(out / "gnm_head.glb")
    # vertex colours from the front photo on vertices facing its camera
    front = views[0]
    with torch.no_grad():
        uv, _ = front.project(torch.tensor(V))
        Rf = front.R().numpy()
    uv = uv.numpy()
    normals = mesh.vertex_normals
    view_dir = (Rf.T @ np.array([0, 0, -1.0]))      # direction from head towards camera, in world
    facing = normals @ view_dir > 0.25
    cols = np.tile(np.array([150, 110, 85, 255], np.uint8), (len(mesh.vertices), 1))
    uvk = uv[keep]
    ok = facing & (uvk[:, 0] >= 0) & (uvk[:, 0] < front.w) & (uvk[:, 1] >= 0) & (uvk[:, 1] < front.h)
    px = np.clip(np.round(uvk[ok]).astype(int), 0, [front.w - 1, front.h - 1])
    cols[ok, :3] = front.rgb[px[:, 1], px[:, 0]]
    coloured = trimesh.Trimesh(V[keep], remap[tris], vertex_colors=cols, process=False)
    coloured.export(out / "gnm_head_coloured.glb")
    print(f"done in {elapsed:.0f} s; wrote {out}")


if __name__ == "__main__":
    main()
