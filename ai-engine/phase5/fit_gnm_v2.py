"""GNM fit v2: v1's fit (68 landmarks + silhouettes), then MediaPipe's dense face points.

The dense correspondence comes from the v1 head: each MediaPipe point of the front photo is
projected onto the fitted GNM mesh (front-facing triangles only) and its barycentric
coordinates are kept fixed. Lips and nose weigh three times as much; the face oval is left
to the silhouette. The identity prior is then loosened stage by stage, stopping when the
silhouette error or the head's asymmetry grows, with more lower-face expression components
freed under regularisation.

    python fit_gnm_v2.py --front F.png --three-quarter T.png --side S.png --out fit/

Writes fit.npz (all GNM vertices), keep_indices.npy (skin and eyeballs), gnm_head.glb, fit.json
and overlay_<view>.png into --out. Turned views must face the left of the image.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import fit_gnm as F
from gnm.shape import gnm_pytorch as G

torch.set_default_dtype(torch.float64)


def mediapipe_dense(rgb: np.ndarray, model: Path) -> np.ndarray | None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    det = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)), num_faces=1))
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])[:468]   # irises dropped


def region_sets():
    from mediapipe.python.solutions import face_mesh_connections as c
    pts = lambda s: sorted({i for e in s for i in e})
    return {"oval": pts(c.FACEMESH_FACE_OVAL), "lips": pts(c.FACEMESH_LIPS), "nose": pts(c.FACEMESH_NOSE)}


def barycentric_correspondence(uv, V, tris, normals, view_dir, points):
    """For each 2D point, a front-facing triangle under it and barycentric coordinates (or -1)."""
    t2 = uv[tris]                                                   # T,3,2
    facing = (normals[tris].mean(1) @ view_dir) > 0.1
    depth = (V[tris] @ view_dir).mean(1)                            # larger = closer to the camera
    a, b, c = t2[:, 0], t2[:, 1], t2[:, 2]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    out_t, out_w = [], []
    for p in points:
        v2 = p - a
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / (den + 1e-12)
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / (den + 1e-12)
        l0 = 1 - l1 - l2
        inside = facing & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6) & (np.abs(den) > 1e-9)
        idx = np.flatnonzero(inside)
        if len(idx) == 0:
            out_t.append(-1); out_w.append((0, 0, 0)); continue
        best = idx[np.argmax(depth[idx])]
        out_t.append(best); out_w.append((l0[best], l1[best], l2[best]))
    return np.array(out_t), np.array(out_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True, type=Path)
    ap.add_argument("--three-quarter", type=Path)
    ap.add_argument("--side", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expr", type=int, default=60)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model = G.GNM.from_remote(version=G.GNMMajorVersion.V3, variant=G.GNMVariant.HEAD).double()
    names = list(model.identity_names)
    head_ids = [i for i, n in enumerate(names) if n.startswith("head_")]
    expr_ids = [i for i, n in enumerate(model.expression_names) if n.startswith("lower_face_region")][:args.expr]
    gm = lambda n: np.asarray(model.vertex_group_mask(n)).astype(bool)
    skin = gm("skin_exterior")
    skin_idx = np.flatnonzero(skin)
    tris_all = np.asarray(model.triangles)
    skin_tris = tris_all[skin[tris_all].all(1)]
    mirror = torch.tensor(np.asarray(model.mirror_indices).astype(np.int64))
    neck_j = list(model.joint_names).index("neck")
    mp_model = F.LANDMARKER
    regions = region_sets()

    views = [F.View("front", args.front, mp_model)]
    if args.three_quarter:
        views.append(F.View("three_quarter", args.three_quarter, mp_model))
    if args.side:
        views.append(F.View("side", args.side, mp_model))
    for v in views:
        v.dense = mediapipe_dense(v.rgb, mp_model) if v.name != "side" else None

    ident = torch.zeros(len(head_ids), requires_grad=True)
    expr = torch.zeros(len(expr_ids), requires_grad=True)
    neck = torch.zeros(3, requires_grad=True)
    lm_w = torch.tensor(F.LM_WEIGHT)
    tri_t = torch.tensor(tris_all)

    def geometry():
        fid = torch.zeros(model.identity_dim).index_put((torch.tensor(head_ids),), ident)
        fex = torch.zeros(model.expression_dim).index_put((torch.tensor(expr_ids),), expr)
        rots = torch.zeros(model.num_joints, 3).index_put((torch.tensor([neck_j]),), neck[None])
        V, L = model.vertices_and_landmarks(G.GNMLandmarksType.HEAD_SPARSE_68, identity=fid[None],
                                            expression=fex[None], rotations=rots[None])
        return V[0], L[0]

    with torch.no_grad():
        V0, L0 = geometry()
        head_h = float(V0[:, 1].max() - L0[8, 1])
        centre = V0[skin_idx].mean(0)
        for v in views:
            f = math.exp(float(v.logf))
            dist = f * head_h / (v.box["chin"] - v.box["top"])
            Xc = centre @ v.R().T
            mu, mv = (v.box["left"] + v.box["right"]) / 2, (v.box["top"] + v.box["chin"]) / 2
            v.t.data = torch.tensor([(mu - v.w / 2) * dist / f - float(Xc[0]), (mv - v.h / 2) * dist / f - float(Xc[1]),
                                     dist - float(Xc[2])])

    dense = {"tri": None, "w": None, "weights": None}

    def losses(V, L, use_sil, it, use_dense):
        total, parts = 0.0, {}
        for v in views:
            uv, _ = v.project(V)
            if v.lm is not None and not use_dense:
                luv, _ = v.project(L)
                err = ((luv - torch.tensor(v.lm)) ** 2).sum(1)
                total = total + (lm_w * err).sum() / lm_w.sum()
                parts[f"lm_{v.name}"] = float(torch.sqrt((lm_w * err).sum() / lm_w.sum()).detach())
            if use_dense and v.dense is not None:
                ok = dense["ok"]
                tri = tri_t[dense["tri"][ok]]
                w = torch.tensor(dense["w"][ok])
                P = (V[tri] * w[:, :, None]).sum(1)
                puv, _ = v.project(P)
                err = ((puv - torch.tensor(v.dense[ok])) ** 2).sum(1)
                ww = torch.tensor(dense["weights"][ok])
                total = total + (ww * err).sum() / ww.sum()
                parts[f"dense_{v.name}"] = float(torch.sqrt((ww * err).sum() / ww.sum()).detach())
                if dense.get("lip_vid") is not None:
                    luv, _ = v.project(V[torch.tensor(dense["lip_vid"])])
                    le = ((luv - torch.tensor(v.dense[dense["lip_mp"]])) ** 2).sum(1)
                    total = total + 3.0 * le.mean()
                    parts[f"lips_{v.name}"] = float(torch.sqrt(le.mean()).detach())
            if v.anchors:
                luv, _ = v.project(L)
                idx = list(v.anchors)
                e = ((luv[idx] - torch.tensor([v.anchors[i] for i in idx])) ** 2).sum(1)
                total = total + e.mean()
                parts[f"anchor_{v.name}"] = float(torch.sqrt(e.mean()).detach())
            f20 = (max(v.w, v.h) / 2) / math.tan(math.radians(20) / 2)
            total = total + 2000.0 * (v.rot ** 2).sum() + 200.0 * (v.logf - math.log(f20)) ** 2
            if use_sil:
                if it % 5 == 0 or not hasattr(v, "sil_idx"):
                    v.sil_idx = F.silhouette_vertices(uv.detach().numpy(), skin_tris, v.w, v.h, skin_idx)
                    chin_v = float(v.project(L[8:9])[0][0, 1])
                    v.sil_idx = v.sil_idx[uv.detach().numpy()[v.sil_idx][:, 1] < min(chin_v, v.box["chin"])]
                if len(v.sil_idx) > 10:
                    c = F.chamfer(uv[v.sil_idx], v.outline)
                    total = total + F.SIL_W * c ** 2
                    parts[f"sil_{v.name}"] = float(c.detach())
        return total, parts

    def asym(_posed=None):
        """Mean left/right mismatch of the unposed shape (identity and expression only), in mm."""
        fid = torch.zeros(model.identity_dim).index_put((torch.tensor(head_ids),), ident)
        fex = torch.zeros(model.expression_dim).index_put((torch.tensor(expr_ids),), expr)
        V = model(identity=fid[None], expression=fex[None])[0]
        Vm = V[mirror] * torch.tensor([-1.0, 1.0, 1.0])
        return float((V[skin_idx] - Vm[skin_idx]).norm(dim=1).mean() * 1000)       # mm

    def run(params, iters, use_sil, reg_id, reg_ex, lr, label, use_dense=False):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(iters):
            opt.zero_grad()
            V, L = geometry()
            data, parts = losses(V, L, use_sil, it, use_dense)
            loss = data + reg_id * (ident ** 2).sum() + reg_ex * (expr ** 2).sum() + 50.0 * (neck ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for v in views:
                    v.logf.clamp_(*v.focal_bounds())
                neck.clamp_(-0.2, 0.2)
            if it == iters - 1:
                print(f"[{label}] " + " ".join(f"{k} {x:.2f}" for k, x in parts.items()), flush=True)
        return parts

    cams = [p for v in views for p in (v.rot, v.t, v.logf)]
    run(cams, 300, True, 1e6, 1e6, 0.01, "rigid")
    run(cams + [ident, expr, neck], 300, False, 20.0, 500.0, 0.03, "identity")
    base = run(cams + [ident, expr, neck], 600, True, 5.0, 200.0, 0.02, "v1")

    # dense correspondence on the v1 head, front view
    front = views[0]
    with torch.no_grad():
        V, _ = geometry()
        uv, _ = front.project(V)
        Rf = front.R().numpy()
    Vn = V.numpy()
    import trimesh
    normals = trimesh.Trimesh(Vn, tris_all, process=False).vertex_normals
    view_dir = Rf.T @ np.array([0, 0, -1.0])
    tri, w = barycentric_correspondence(uv.numpy(), Vn, tris_all, normals, view_dir, front.dense)
    weights = np.ones(len(front.dense))
    weights[regions["oval"]] = 0.0
    weights[regions["lips"]] *= 3.0
    weights[regions["nose"]] *= 3.0
    # Lips: semantic, not from the v1 surface (v1's lips are thin, so its surface would pin the photo's lip line
    # to the skin around the lips). Outer lip line -> GNM's vermilion border, inner line -> its mouth seam,
    # each to the nearest such vertex in the front view.
    OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    lipsg = gm("upper_lip") | gm("lower_lip")
    sock = gm("mouth_sock")
    e = np.concatenate([tris_all[:, [0, 1]], tris_all[:, [1, 2]], tris_all[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    border = np.zeros(len(Vn), bool)
    border[e[lipsg[e[:, 0]] & ~lipsg[e[:, 1]] & ~sock[e[:, 1]], 0]] = True
    seamv = np.zeros(len(Vn), bool)
    seamv[e[lipsg[e[:, 0]] & sock[e[:, 1]], 0]] = True
    facing_v = normals @ view_dir > 0.0
    uvn = uv.numpy()
    lip_vid, lip_mp = [], []
    for mp_ids, cand in ((OUTER, border & facing_v), (INNER, seamv)):      # the seam faces into the mouth
        cidx = np.flatnonzero(cand)
        for k in mp_ids:
            dd = np.linalg.norm(uvn[cidx] - front.dense[k], axis=1)
            lip_vid.append(cidx[np.argmin(dd)])
            lip_mp.append(k)
    dense["lip_vid"], dense["lip_mp"] = np.array(lip_vid), np.array(lip_mp)
    weights[regions["lips"]] = 0.0                                   # handled by the semantic lip term
    ok = (tri >= 0) & (weights > 0)
    dense.update(tri=tri, w=w, weights=weights, ok=ok)
    print(f"lip correspondences: {len(lip_vid)} (border verts {border.sum()}, seam verts {seamv.sum()})")
    print(f"dense correspondences: {ok.sum()} of {len(front.dense)} (lips {np.isin(np.flatnonzero(ok), regions['lips']).sum()}, "
          f"nose {np.isin(np.flatnonzero(ok), regions['nose']).sum()})")

    # loosen the identity prior in stages; stop when the silhouette or symmetry gets worse
    with torch.no_grad():
        ref_asym = asym(geometry()[0])
    ref_sil = np.mean([base[k] for k in base if k.startswith("sil_")])
    history = [{"reg_id": 5.0, "sil": ref_sil, "asym_mm": ref_asym, **base}]
    state = None
    for reg in (5.0, 2.0, 1.0, 0.5, 0.25):
        saved = (ident.detach().clone(), expr.detach().clone(), neck.detach().clone(),
                 [(v.rot.detach().clone(), v.t.detach().clone(), v.logf.detach().clone()) for v in views])
        parts = run(cams + [ident, expr, neck], 400, True, reg, 100.0, 0.02, f"dense reg {reg}", use_dense=True)
        with torch.no_grad():
            a = asym(geometry()[0])
        sil = np.mean([parts[k] for k in parts if k.startswith("sil_")])
        history.append({"reg_id": reg, "sil": sil, "asym_mm": a, **parts})
        print(f"   reg {reg}: silhouette {sil:.2f} px, asymmetry {a:.2f} mm")
        if sil > 1.15 * ref_sil or a > ref_asym + 1.5:
            print("   stopping: silhouette or symmetry got worse; keeping the previous stage")
            ident.data, expr.data, neck.data = saved[0], saved[1], saved[2]
            for v, (r, t, lf) in zip(views, saved[3]):
                v.rot.data, v.t.data, v.logf.data = r, t, lf
            history[-1]["rejected"] = True
            break
        state = parts
    elapsed = time.time() - t0

    with torch.no_grad():
        V, L = geometry()
        final_parts = losses(V, L, True, 0, True)[1]
    V = V.numpy()
    report = {"seconds": round(elapsed, 1), "final": final_parts, "history": history,
              "identity_rms": float(ident.pow(2).mean().sqrt()), "identity_abs_max": float(ident.abs().max()),
              "expression_rms": float(expr.pow(2).mean().sqrt())}
    (out / "fit.json").write_text(json.dumps(report, indent=1))
    np.savez(out / "fit.npz", vertices=V, identity=ident.detach().numpy(), expression=expr.detach().numpy())
    print("REPORT", json.dumps({k: report[k] for k in ("seconds", "final", "identity_rms", "identity_abs_max", "expression_rms")}))

    # overlays (dense points: red = MediaPipe, green = model)
    ok = dense["ok"]
    for v in views:
        img = Image.fromarray(v.rgb.copy())
        d = ImageDraw.Draw(img)
        with torch.no_grad():
            uvv, _ = v.project(torch.tensor(V))
        uvv = uvv.numpy()
        mask = np.zeros((v.h, v.w), np.uint8)
        cv2.fillPoly(mask, list(np.round(uvv[skin_tris]).astype(np.int32)), 1)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cs:
            d.line([tuple(p[0]) for p in cnt] + [tuple(cnt[0][0])], fill=(0, 255, 255), width=2)
        if v.dense is not None:
            tri = tris_all[dense["tri"][ok]]
            P = (V[tri] * dense["w"][ok][:, :, None]).sum(1)
            with torch.no_grad():
                puv = v.project(torch.tensor(P))[0].numpy()
            for (x, y) in v.dense[ok]:
                d.point((x, y), fill=(255, 60, 60))
            for (x, y) in puv:
                d.point((x, y), fill=(60, 255, 60))
        img.save(out / f"overlay_{v.name}.png")

    keep = np.zeros(len(V), bool)
    for g in ("skin", "left_eye", "right_eye"):
        keep |= gm(g)
    tris = tris_all[keep[tris_all].all(1)]
    remap = -np.ones(len(V), int)
    remap[np.flatnonzero(keep)] = np.arange(keep.sum())
    trimesh.Trimesh(V[keep], remap[tris], process=False).export(out / "gnm_head.glb")
    np.save(out / "keep_indices.npy", np.flatnonzero(keep))
    print(f"done in {elapsed:.0f} s -> {out}")


if __name__ == "__main__":
    main()
