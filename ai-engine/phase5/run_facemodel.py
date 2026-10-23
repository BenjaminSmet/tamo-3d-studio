"""Phase 5: a clean head from Google's GNM face model, shaped by the actor's photos.

TRELLIS.2 and Pixal3D copy what they see (lash lines, moles) and guess what they don't, so
their heads come out with bumps on the skull and a flat mouth that MetaHuman's teeth poke
through. GNM (Apache 2.0) is a head model learned from 3D scans: it can only make real
heads, with a smooth skull, eyes and lips with depth over a proper mouth seam.

One run:
1. A front-only TRELLIS.2 head (Phase 1, in trellis/, shape only), for the hybrid in step 4.
2. Cut-outs of the other photos (three-quarter, side, other three-quarter, other side, back). GNM is symmetric, so
   the fit takes every turned view facing left: one that faces right is mirrored for the fit, and
   its fitted camera is reflected back afterwards (through the fitted head's own mirror map, as the
   neck turns the head off x = 0), so the
   texture always projects the photo as taken (a mole stays on its own cheek).
3. GNM fitted to the photos (fit_gnm_v2.py): MediaPipe face points on the front and
   three-quarter photos, the head's outline in every photo (the back's at half weight).
4. The hybrid (hybrid.py): GNM's skull and topology plus the TRELLIS.2 head's face shape,
   clamped to 1 cm and kept off the mouth seam and the eyes.
5. The skin texture (texture.py): albedo.png, 4K, baked from the photos.
6. model.glb (textured), model.obj (shape only) and head_metahuman.fbx with the albedo embedded
   (no smoothing: GNM's surface is already clean). If the texture fails, the untextured head is
   exported as before and run.json says "texture": "failed". --no-texture skips step 5, as before
   texturing existed.

Same run folder and run.json as Phase 1, so the app shows it as a version. Runs with the
face model's own Python (see docs/facemodel.md); TRELLIS.2 and the cut-outs run in
Espalier-2's, given with --engine-python:

    ~/tamo-gnm/.venv/bin/python ai-engine/phase5/run_facemodel.py --actor nia_okafor \\
        --front front.png --side side.png --three-quarter three_quarter.png [--other-three-quarter ot.png] [--other-side o.png] [--back b.png]
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent / "phase1"
sys.path.insert(0, str(PHASE1))
sys.path.insert(0, str(HERE.parent / "phase4"))
from run_phase1 import REPO_ROOT, git_revision, glb_to_obj, run_logged  # noqa: E402

from frame_views import facing, head_box  # noqa: E402


def cut_out(engine_python: Path, image: Path, out: Path, log: Path) -> Image.Image:
    with Image.open(image) as opened:
        if opened.mode == "RGBA" and opened.getchannel("A").getextrema()[0] < 128:
            opened.save(out)   # already cut out
            return Image.open(out).convert("RGBA")
    if run_logged([str(engine_python), str(PHASE1 / "prepare_image.py"), str(image), str(out)], log) != 0:
        sys.exit(f"Background removal failed for {image.name}. See {log}")
    return Image.open(out).convert("RGBA")


def unmirror_cameras(cameras_json: Path, originals: dict[str, Path], mirror_map: dict | None = None) -> None:
    """Cameras fitted to mirrored photos, given the photo as taken. With D = diag(-1, 1, 1), x_cam = D R D X + D t
    puts a point X at w - u where the fitted camera put D X at u: it sees the head's reflection D H. The posed head
    is not its own reflection (the neck turns it off x = 0), so the fit's mirror map H = A (D H) + b brings it back:
    R' = D R D A^T, t' = D t - R' b."""
    cams = json.loads(cameras_json.read_text())
    D = [-1.0, 1.0, 1.0]
    A = mirror_map["A"] if mirror_map else [[float(i == j) for j in range(3)] for i in range(3)]
    b = mirror_map["b"] if mirror_map else [0.0, 0.0, 0.0]
    for name, original in originals.items():
        if name in cams:
            c = cams[name]
            Rr = [[D[i] * c["R"][i][j] * D[j] for j in range(3)] for i in range(3)]
            R = [[sum(Rr[i][k] * A[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
            c["R"] = R
            c["t"] = [D[i] * c["t"][i] - sum(R[i][k] * b[k] for k in range(3)) for i in range(3)]
            c["image"], c["mirrored_for_fit"] = str(original), True
    cameras_json.write_text(json.dumps(cams, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: photos to a clean GNM head for MetaHuman.")
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--side", type=Path)
    parser.add_argument("--three-quarter", type=Path)
    parser.add_argument("--other-three-quarter", type=Path, help="the other three-quarter view")
    parser.add_argument("--other-side", type=Path, help="the other profile")
    parser.add_argument("--back", type=Path, help="the back of the head")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--project", default="tamo")
    parser.add_argument("--library", type=Path, default=REPO_ROOT / "library" / "projects")
    parser.add_argument("--espalier", type=Path, default=Path.home() / "espalier")
    parser.add_argument("--engine-python", type=Path, default=Path.home() / "espalier" / ".venv" / "bin" / "python",
                        help="Espalier-2's Python, for TRELLIS.2 and the background removal")
    parser.add_argument("--pipeline", default="1024_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-texture", action="store_true", help="shape only: no skin texture")
    parser.add_argument("--no-metahuman", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    inputs = [("front", args.front), ("three_quarter", args.three_quarter), ("side", args.side),
              ("other_three_quarter", args.other_three_quarter), ("other_side", args.other_side), ("back", args.back)]
    inputs = [(name, path.resolve()) for name, path in inputs if path]
    for _, path in inputs:
        if not path.is_file():
            sys.exit(f"Image not found: {path}")
    if not args.engine_python.is_file():
        sys.exit(f"Espalier-2's Python not found at {args.engine_python}. See docs/phase1-install.md.")

    texture = not args.no_texture
    mode = "textured" if texture else "geometry"
    actor_dir = args.library / args.project / "actors" / args.actor
    run_dir = args.run_dir or actor_dir / "generations" / f"{datetime.now():%Y%m%d-%H%M%S}-facemodel-{mode}"
    run_dir.mkdir(parents=True)
    log = run_dir / "espalier.log"
    record = {
        "model": "gnm",
        "source_images": {name: str(path) for name, path in inputs},
        "mode": mode,
        "pipeline": args.pipeline,
        "seed": args.seed,
        "smooth": 0.0,
        "facemodel_revision": git_revision(REPO_ROOT),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    start = time.monotonic()

    def finish(result: str) -> None:
        record.update(result=result, seconds=round(time.monotonic() - start))
        (run_dir / "run.json").write_text(json.dumps(record, indent=2))
        print("Face model run finished.")
        print(f"Result:       {result}")
        print(f"Time:         {record['seconds']} s")
        print(f"Output:       {run_dir}")
        if result != "ok":
            sys.exit(f"Generation failed. The last lines of {log} say why.")

    print("Building the rough head with TRELLIS.2...", flush=True)
    trellis = run_dir / "trellis"
    cmd = [str(args.engine_python), "-u", str(PHASE1 / "run_phase1.py"), "--image", str(args.front.resolve()),
           "--actor", args.actor, "--project", args.project, "--library", str(args.library),
           "--espalier", str(args.espalier), "--run-dir", str(trellis), "--pipeline", args.pipeline,
           "--seed", str(args.seed), "--no-metahuman"]
    if run_logged(cmd, log) != 0 or not (trellis / "model.glb").is_file():
        print("TRELLIS.2 failed; see trellis/ in the run folder.")
        finish("failed")
    # The input image the app shows for this version, and TRELLIS.2's GPU use (the fit runs on the CPU).
    shutil.copy2(trellis / "input_rgba.png", run_dir / "input_rgba.png")
    trellis_record = json.loads((trellis / "run.json").read_text())
    for key in ("vram_baseline_mib", "vram_peak_mib", "vram_peak_above_baseline_mib"):
        if key in trellis_record:
            record[key] = trellis_record[key]

    print("Removing background from the turned views...", flush=True)
    views = run_dir / "views"
    views.mkdir()
    shutil.copy2(trellis / "input_cutout.png", views / "front.png")
    mirrored = {}
    for name, path in inputs[1:]:
        original = run_dir / f"cutout_{name}.png"
        rgba = cut_out(args.engine_python, path, original, log)
        if name != "back":                  # the back has no face to point anywhere
            direction = facing(head_box(rgba))
            if direction == "right":
                rgba = ImageOps.mirror(rgba)   # the fit takes turned faces pointing left
                mirrored[name] = original
            print(f"{name.replace('_', '-').capitalize()} view faces {direction}.")
        rgba.save(views / f"{name}.png")

    print("Fitting the face model to the photos...", flush=True)
    fit = run_dir / "fit"
    cmd = [sys.executable, "-u", str(HERE / "fit_gnm_v2.py"), "--front", str(views / "front.png"), "--out", str(fit)]
    for name, _ in inputs[1:]:
        cmd += [f"--{name.replace('_', '-')}", str(views / f"{name}.png")]
    if run_logged(cmd, log) != 0:
        finish("failed")
    fit_report = json.loads((fit / "fit.json").read_text())
    record["fit"] = fit_report["final"]
    unmirror_cameras(fit / "cameras.json", mirrored, fit_report.get("mirror_map"))

    print("Adding the face shape from the TRELLIS.2 head...", flush=True)
    hybrid = run_dir / "hybrid"
    cmd = [sys.executable, "-u", str(HERE / "hybrid.py"), "--fit", str(fit), "--trellis", str(trellis / "model.glb"),
           "--out", str(hybrid)]
    if run_logged(cmd, log) != 0:
        finish("failed")
    record["hybrid"] = json.loads((hybrid / "hybrid.json").read_text())
    # Only trellis/model.glb is needed from here on; Espalier-2's raw latents and the OBJ copy are most of the
    # run's size.
    for leftover in ("raw.pt", "model.obj"):
        (trellis / leftover).unlink(missing_ok=True)

    glb = run_dir / "model.glb"
    shutil.copy2(hybrid / "gnm_head.glb", glb)
    glb_to_obj(glb, run_dir / "model.obj")                    # shape only, also when textured
    if texture:
        if run_logged([sys.executable, "-u", str(HERE / "texture.py"), "--run", str(run_dir)], log) == 0 \
                and (run_dir / "model_textured.glb").is_file():
            (run_dir / "model_textured.glb").replace(glb)    # the app's viewer shows the skin
            record["texture"] = "ok"
            record["albedo"] = str(run_dir / "albedo.png")
        else:
            print(f"The skin texture failed; exporting the head without it. See {log}")
            record["texture"] = "failed"
    if not args.no_metahuman:
        print("Exporting the MetaHuman-aligned head with Blender...", flush=True)
        fbx = run_dir / "head_metahuman.fbx"
        cmd = [sys.executable, str(PHASE1 / "export_metahuman.py"), str(glb), str(fbx), "--smooth", "0"]
        if run_logged(cmd, log) == 0 and fbx.exists():
            record["metahuman_fbx"] = str(fbx)
        else:
            print(f"MetaHuman export failed; the GLB and OBJ are still there. See {log}")
    finish("ok")


if __name__ == "__main__":
    main()
