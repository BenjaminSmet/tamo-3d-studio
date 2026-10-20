"""Export a generated head as an FBX that MetaHuman's "From Custom Mesh" can fit.

MetaHuman auto-frames the place where its own head sits and tracks facial
features there, so the mesh has to be the right size and in the right place:
centimetres, Z up, face toward -Y, the top of the head and the nose tip on
MetaHuman's. The shoulders are cut off below the neck and the mesh is reduced
to about 200k faces; the texture is embedded in the FBX, which imports in
Unreal at Uniform Scale 1.0.

Run it with Python and it starts Blender in the background (found on PATH, in
$BLENDER, or in the usual install folders, or given with --blender):

    python export_metahuman.py model.glb out.fbx

or inside Blender directly:

    blender -b --python export_metahuman.py -- model.glb out.fbx
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

# Measured on SM_MH_Head (MetaHumanSDK/TemplateAssets, UE 5.8), exported to FBX and read in
# Blender: metres, Z up, face toward -Y. Unreal shows the same numbers in centimetres.
MH_TOP_Z = 1.7323
MH_NOSE = (0.0, -0.1452, 1.5835)

TARGET_FACES = 200_000
# How far below the narrowest point of the neck to cut, in head heights (top of head to neck).
BELOW_NECK = 0.12
# The cut never goes below where the outline has widened this much past the neck (shoulders).
SHOULDER_WIDENING = 1.35

# Smoothing at --smooth 1.0. TRELLIS.2 leaves bumps and flat panels on the skull and turns eyebrows
# and lashes into raised ridges about a centimetre wide, which MetaHuman's fit copies. Taubin
# (shrink-free) smoothing keeps features that wide, so this is plain Laplacian smoothing: it flattens
# the ridges into the skin around them, while the skull, being gently curved, shrinks by only about
# a millimetre, and the alignment below is measured after smoothing. The skull, forehead and back of
# the head get the full pass; the face below the brows a light one, so the nose, lips and jaw keep
# their shape.
SMOOTH_ITERATIONS = 100
SMOOTH_LAMBDA = 0.5
SMOOTH_FACE_WEIGHT = 0.1


def find_blender(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("BLENDER"), shutil.which("blender"), shutil.which("blender.exe")]
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        "/mnt/c/Program Files/Blender Foundation/Blender*/blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
    ]
    for pattern in patterns:
        candidates += sorted(glob.glob(pattern), reverse=True)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    sys.exit("Blender not found. Install it or pass --blender <path to blender>.")


def launch(argv: list[str]) -> None:
    """Outside Blender: start Blender in the background on this script."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="model.glb from a Phase 1 run")
    parser.add_argument("output", help="FBX to write")
    parser.add_argument("--faces", type=int, default=TARGET_FACES)
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="smoothing strength: 0 off, 1 removes skull bumps and brow ridges, 2 stronger")
    parser.add_argument("--turn", type=float, default=0.0,
                        help="degrees to turn the model about the vertical axis first, for models whose "
                             "face does not point toward -Y after import (Pixal3D: 180)")
    parser.add_argument("--blender", help="path to the Blender executable")
    args = parser.parse_args(argv)

    blender = find_blender(args.blender)
    paths = [os.path.abspath(__file__), os.path.abspath(args.model), os.path.abspath(args.output)]
    if blender.endswith(".exe") and sys.platform != "win32":
        # Windows Blender started from WSL needs Windows paths.
        paths = [subprocess.check_output(["wslpath", "-w", p], text=True).strip() for p in paths]
    script, model, output = paths
    cmd = [blender, "-b", "--factory-startup", "--python", script, "--", model, output, str(args.faces),
           str(args.smooth), str(args.turn)]
    sys.exit(subprocess.run(cmd).returncode)


def neck_cut_z(points, top_z: float) -> float:
    """The height to cut at: a little below the narrowest point of the neck, above the shoulders."""
    import numpy as np

    lo = points[:, 2].min()
    bands = np.linspace(top_z, lo, 200)
    step = bands[0] - bands[1]
    widths = []
    for z in bands:
        sel = points[np.abs(points[:, 2] - z) < step]
        widths.append(np.ptp(sel[:, 0]) if len(sel) else 0.0)
    widths = np.convolve(widths, np.ones(5) / 5, mode="same")

    peak = int(np.argmax(widths[: len(widths) // 2]))   # widest part of the head
    neck = peak
    for i in range(peak, len(widths)):
        if widths[i] < widths[neck]:
            neck = i
        elif widths[i] > SHOULDER_WIDENING * widths[neck]:
            break
    neck_z = bands[neck]
    cut = neck_z - BELOW_NECK * (top_z - neck_z)
    # Stay above the shoulders.
    for i in range(neck, len(widths)):
        if widths[i] > SHOULDER_WIDENING * widths[neck]:
            cut = max(cut, bands[i])
            break
    return float(cut)


def nose_tip(points):
    """The most forward (-Y) point near the centre line, in the face region."""
    import numpy as np

    lo, hi = points.min(0), points.max(0)
    h = hi[2] - lo[2]
    cx = (lo[0] + hi[0]) / 2
    face = points[(np.abs(points[:, 0] - cx) < 0.03 * h) & (points[:, 2] > lo[2] + 0.25 * h)
                  & (points[:, 2] < lo[2] + 0.85 * h)]
    return face[face[:, 1].argmin()], cx


def smooth(points, edges, boundary, strength: float):
    """Laplacian smoothing, full on the skull and back of the head, light on the face. Returns new points."""
    import numpy as np

    nose, _ = nose_tip(points)
    top_z = points[:, 2].max()
    # Height from the nose tip (0) to the top of the head (1); the brows sit around 0.35.
    f = (points[:, 2] - nose[2]) / (top_z - nose[2])
    weight = np.interp(f, [0.1, 0.3, 0.45], [SMOOTH_FACE_WEIGHT, 0.6, 1.0])
    depth = np.ptp(points[:, 1])
    weight = np.where(points[:, 1] > nose[1] + 0.55 * depth, 1.0, weight)   # behind the ears
    weight[boundary] = 0.0                                                    # keep the neck cut in place

    i, j = edges[:, 0], edges[:, 1]
    degree = np.bincount(np.concatenate([i, j]), minlength=len(points)).astype(float)
    degree[degree == 0] = 1.0

    def laplacian(p):
        acc = np.zeros_like(p)
        np.add.at(acc, i, p[j])
        np.add.at(acc, j, p[i])
        return acc / degree[:, None] - p

    p = points.copy()
    w = weight[:, None]
    for _ in range(round(SMOOTH_ITERATIONS * strength)):
        p += SMOOTH_LAMBDA * w * laplacian(p)
    return p


def export(model: str, output: str, faces: int, smooth_strength: float = 1.0, turn: float = 0.0) -> None:
    """Inside Blender."""
    import bmesh
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=model)   # glTF Y up becomes Blender Z up, face toward -Y
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if len(meshes) != 1:
        sys.exit(f"Expected one mesh in {model}, found {len(meshes)}")
    ob = meshes[0]
    # Bake the world transform, parents included (Pixal3D's GLB has a node above the mesh), into the
    # mesh data, so everything below works in world coordinates.
    ob.data.transform(ob.matrix_world)
    ob.parent = None
    ob.matrix_world = Matrix.Identity(4)
    for other in [o for o in bpy.context.scene.objects if o is not ob]:
        bpy.data.objects.remove(other)
    if turn:
        ob.data.transform(Matrix.Rotation(math.radians(turn), 4, "Z"))
    ob.data.update()
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)

    # glTF splits vertices along UV seams; join them so the surface is one piece (UVs are kept per corner).
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bm.to_mesh(ob.data)
    bm.free()

    points = np.array([v.co[:] for v in ob.data.vertices])
    top_z = points[:, 2].max()
    cut = neck_cut_z(points, top_z)

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, cut),
                           plane_no=(0, 0, 1), clear_inner=True)
    bm.to_mesh(ob.data)
    bm.free()

    before = len(ob.data.polygons)
    if before > faces:
        decimate = ob.modifiers.new("decimate", "DECIMATE")
        decimate.ratio = faces / before
        bpy.ops.object.modifier_apply(modifier=decimate.name)
    for polygon in ob.data.polygons:
        polygon.use_smooth = True

    if smooth_strength > 0:
        points = np.array([v.co[:] for v in ob.data.vertices])
        edges = np.array([e.vertices[:] for e in ob.data.edges])
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        boundary = np.array([v.index for v in bm.verts if v.is_boundary], dtype=int)
        bm.free()
        smoothed = smooth(points, edges, boundary, smooth_strength)
        ob.data.vertices.foreach_set("co", smoothed.ravel())
        ob.data.update()
        moved = np.linalg.norm(smoothed - points, axis=1) / np.ptp(points[:, 2])
        print(f"Smoothing moved vertices by {100 * np.median(moved):.3f}% of the height (median), "
              f"{100 * np.percentile(moved, 99):.3f}% (99th percentile); {len(edges)} edges, "
              f"{len(boundary)} boundary vertices")

    points = np.array([v.co[:] for v in ob.data.vertices])
    top_z = points[:, 2].max()
    nose, cx = nose_tip(points)
    scale = (MH_TOP_Z - MH_NOSE[2]) / (top_z - nose[2])
    ob.scale = (scale, scale, scale)
    ob.location = Vector(np.array(MH_NOSE) - scale * np.array([cx, nose[1], nose[2]]))
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    bpy.ops.export_scene.fbx(filepath=output, path_mode="COPY", embed_textures=True, apply_unit_scale=True,
                             global_scale=1.0, axis_forward="-Z", axis_up="Y", mesh_smooth_type="FACE",
                             bake_space_transform=True)
    points = np.array([v.co[:] for v in ob.data.vertices])
    print(f"Exported {output}: {before} -> {len(ob.data.polygons)} faces, scale {scale:.4f}, "
          f"height {100 * np.ptp(points[:, 2]):.1f} cm, ear to ear {100 * np.ptp(points[:, 0]):.1f} cm")


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        launch(sys.argv[1:])
    else:
        args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        if len(args) < 2:
            sys.exit("usage: blender -b --python export_metahuman.py -- model.glb out.fbx [faces] [smooth] [turn]")
        export(args[0], args[1], int(args[2]) if len(args) > 2 else TARGET_FACES,
               float(args[3]) if len(args) > 3 else 1.0, float(args[4]) if len(args) > 4 else 0.0)
