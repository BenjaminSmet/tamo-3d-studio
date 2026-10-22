"""Hybrid head: the GNM fit's clean skull and topology, plus the face shape of our TRELLIS head.

The TRELLIS head is aligned to the GNM fit (top of head and nose tip first, then scaled ICP on
the face). Each GNM face vertex then moves along its normal to the TRELLIS surface (the
closest point's offset along the normal, clamped to +-1 cm). The offset field is smoothed and
faded to zero at the forehead above the brows, the ears, the temples and the jaw behind the
cheeks. Never inside the mouth seam or on the eyelids; on the lips themselves only outward,
so the lips stay at least as full and the seam as deep as GNM's.

    python hybrid.py --fit fit/ --trellis model.glb --out hybrid/ [--turn 180]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from gnm.shape import gnm_pytorch as G

CLAMP = 0.01


def graph_neighbours(n, tris):
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    return edges[:, 0], edges[:, 1], np.bincount(edges[:, 0], minlength=n).astype(float)


def smooth_scalar(x, i, j, deg, iters, fixed=None):
    x = x.copy()
    for _ in range(iters):
        acc = np.zeros_like(x)
        np.add.at(acc, i, x[j])
        avg = acc / np.maximum(deg, 1)
        x = 0.5 * x + 0.5 * avg
        if fixed is not None:
            x[fixed] = 0.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, type=Path, help="the --out folder of fit_gnm_v2.py")
    ap.add_argument("--trellis", required=True, type=Path)
    ap.add_argument("--turn", type=float, default=0.0, help="degrees about Y to face +Z (Pixal3D GLBs need 180)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lip-edge", type=float, default=0.8,
                    help="mm the lip border (vermilion edge) stands out, so MetaHuman's tracking sees the lips (0 = off)")
    args = ap.parse_args()
    src, out = args.fit, args.out
    out.mkdir(parents=True, exist_ok=True)

    model = G.GNM.from_remote(version=G.GNMMajorVersion.V3, variant=G.GNMVariant.HEAD)
    gm = lambda n: np.asarray(model.vertex_group_mask(n)).astype(bool)
    V = np.load(src / "fit.npz")["vertices"]
    tris = np.asarray(model.triangles)
    n = len(V)
    gnm_mesh = trimesh.Trimesh(V, tris, process=False)
    normals = gnm_mesh.vertex_normals

    # TRELLIS head into GNM space
    th = trimesh.load(args.trellis, force="mesh")
    T = np.asarray(th.vertices, float)
    if args.turn:
        a = np.radians(args.turn)
        T = T @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]).T

    def top_and_nose(P, face_band):
        top = P[:, 1].max()
        cx = np.median(P[:, 0])
        band = P[(np.abs(P[:, 0] - cx) < 0.02 * (P[:, 1].max() - P[:, 1].min())) & face_band(P)]
        return top, band[band[:, 2].argmax()]

    skin_ext = gm("skin_exterior")
    g_top, g_nose = top_and_nose(V[skin_ext], lambda P: (P[:, 1] > 0.2) & (P[:, 1] < 0.33))
    hT = T[:, 1].max() - T[:, 1].min()
    t_top, t_nose = top_and_nose(T, lambda P: (P[:, 1] > T[:, 1].max() - 0.55 * hT) & (P[:, 1] < T[:, 1].max() - 0.25 * hT))
    s = (g_top - g_nose[1]) / (t_top - t_nose[1])
    T = (T - t_nose) * s + g_nose
    trellis = trimesh.Trimesh(T, np.asarray(th.faces), process=False)

    face = gm("hockey_mask")
    icp_pts = V[face]
    M, _, cost = trimesh.registration.icp(icp_pts, trellis, max_iterations=60, scale=True, reflection=False)
    trellis.apply_transform(np.linalg.inv(M))                    # ICP maps GNM->TRELLIS; bring TRELLIS to GNM
    print(f"aligned TRELLIS head: scale {s:.4f}, ICP cost {cost:.6f}")

    # offsets along the normal, clamped
    cp, dist, _ = trimesh.proximity.closest_point(trellis, V)
    d = ((cp - V) * normals).sum(1)
    lateral = np.linalg.norm((cp - V) - d[:, None] * normals, axis=1)
    valid = (np.abs(d) <= 2 * CLAMP) & (lateral < 0.006)
    d = np.clip(d, -CLAMP, CLAMP) * valid

    # mask: face regions only; forehead above the brows, ears, temples and parotid (jaw behind the cheeks) fade out
    i, j, deg = graph_neighbours(n, tris)
    allow = np.zeros(n, bool)
    for g in ("nose_region", "left_cheek_region", "right_cheek_region", "left_infraorbital_region",
              "right_infraorbital_region", "left_zygomatic_region", "right_zygomatic_region", "chin_region",
              "upper_lip_region", "lower_lip_region", "upper_lip", "lower_lip", "middle_brow_region"):
        allow |= gm(g)
    forbid = gm("mouth_sock") | gm("eye_sockets") | gm("eye_interiors") | gm("left_orbital_region") | gm("right_orbital_region")
    lips = gm("upper_lip") | gm("lower_lip")
    # the mouth seam: lip vertices next to the mouth sock, grown by two rings
    seam = np.zeros(n, bool)
    sock = gm("mouth_sock")
    seam[i[sock[j] & lips[i]]] = True
    for _ in range(2):
        grow = seam.copy()
        grow[i[seam[j]]] = True
        seam = grow & lips
    forbid |= seam
    weight = allow.astype(float)
    weight[forbid] = 0.0
    weight = smooth_scalar(weight, i, j, deg, 25)               # feather the edges
    weight[forbid] = 0.0
    weight[~skin_ext] = 0.0

    d = smooth_scalar(d, i, j, deg, 15)
    d[lips] = np.maximum(d[lips], 0.0)                           # lips only ever get fuller
    offset = (weight * d)[:, None] * normals
    H = V + offset

    # lip definition: GNM's lip border is a soft roll that MetaHuman's lip tracking barely sees. Raise the
    # border ring (the "white roll"), sink the skin ring just outside it a little and keep the lips' body slightly
    # forward, so the outline reads as a crisp edge. Never at the mouth seam.
    edge = args.lip_edge / 1000
    if edge > 0:
        skin = skin_ext & ~lips & ~sock
        border = np.zeros(n, bool)
        border[i[lips[i] & skin[j]]] = True
        border &= ~seam
        outer = np.zeros(n, bool)
        outer[j[border[i] & skin[j]]] = True
        body = lips & ~border & ~seam
        lift = np.zeros(n)
        lift[border], lift[outer], lift[body] = 1.0, -0.4, 0.5
        lift = smooth_scalar(lift, i, j, deg, 2)
        lift[seam | sock | ~skin_ext] = 0.0
        H = H + (edge * lift)[:, None] * trimesh.Trimesh(H, tris, process=False).vertex_normals
        print(f"lip edge: {np.count_nonzero(border)} border vertices raised {args.lip_edge:.1f} mm")
    offset = H - V
    moved = np.linalg.norm(offset, axis=1)
    print(f"offset: {np.count_nonzero(moved > 1e-4)} vertices moved, median {1000 * np.median(moved[moved > 1e-4]):.2f} mm, "
          f"max {1000 * moved.max():.2f} mm")

    keep = np.load(src / "keep_indices.npy")
    mask = np.zeros(n, bool)
    mask[keep] = True
    ftris = tris[mask[tris].all(1)]
    remap = -np.ones(n, int)
    remap[keep] = np.arange(len(keep))
    trimesh.Trimesh(H[keep], remap[ftris], process=False).export(out / "gnm_head.glb")
    json.dump({"scale": float(s), "icp_cost": float(cost), "moved_vertices": int(np.count_nonzero(moved > 1e-4)),
               "median_mm": float(1000 * np.median(moved[moved > 1e-4])), "max_mm": float(1000 * moved.max()), "lip_edge_mm": args.lip_edge},
              open(out / "hybrid.json", "w"), indent=1)
    print(f"wrote {out / 'gnm_head.glb'}")


if __name__ == "__main__":
    main()
