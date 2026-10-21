"""Phase 4: build a head from a front photo plus side and three-quarter photos, with Pixal3D.

Same output folder and files as Phase 1 (run_phase1.py), so the app shows the result as a
version and the MetaHuman export is the same; the views Pixal3D saw are kept in views/.

Pixal3D runs in its own copy of Espalier-2 (~/espalier-pixal3d), so the TRELLIS.2 setup is
never touched:

    ~/espalier-pixal3d/.venv/bin/python ai-engine/phase4/run_multiview.py --actor nia_okafor \\
        --front front.png --side side.png --three-quarter three_quarter.png --texture

Which way a side or three-quarter photo faces is read from the photo; --side-facing and
--three-quarter-facing (left or right, as seen in the image) override that.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent / "phase1"
sys.path.insert(0, str(PHASE1))
from run_phase1 import REPO_ROOT, VramMonitor, git_revision, glb_to_obj, run_logged  # noqa: E402

from frame_views import FOV_DEG, facing, head_box, write_views  # noqa: E402

# Azimuth for a turned view whose face points to the left of the image (a camera on the
# actor's left); a face pointing right is the mirror.
TURN = {"side": 90.0, "three_quarter": 45.0}
# Pixal3D's model.glb faces the other way from TRELLIS.2's (+Y in Blender, not -Y), so the
# MetaHuman export turns it round first.
PIXAL3D_TURN = 180.0


def cut_out(image: Path, out: Path, log: Path) -> Image.Image:
    with Image.open(image) as opened:
        if opened.mode == "RGBA" and opened.getchannel("A").getextrema()[0] < 128:
            opened.save(out)   # already cut out
            return Image.open(out).convert("RGBA")
    if run_logged([sys.executable, str(PHASE1 / "prepare_image.py"), str(image), str(out)], log) != 0:
        sys.exit(f"Background removal failed for {image.name}. See {log}")
    return Image.open(out).convert("RGBA")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: front + side (+ three-quarter) photos to a head mesh.")
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--side", type=Path)
    parser.add_argument("--three-quarter", type=Path)
    parser.add_argument("--side-facing", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--three-quarter-facing", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--project", default="tamo")
    parser.add_argument("--library", type=Path, default=REPO_ROOT / "library" / "projects")
    parser.add_argument("--espalier", type=Path, default=Path.home() / "espalier-pixal3d")
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--pipeline", default="1024_cascade", choices=["1024_cascade", "1536_cascade"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="horizontal field of view of every view, degrees")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--no-metahuman", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--views-only", action="store_true",
                        help="cut out and frame the views, then stop before generating (to check the framing)")
    args = parser.parse_args()

    inputs = [("front", args.front, "auto"), ("side", args.side, args.side_facing),
              ("three_quarter", args.three_quarter, args.three_quarter_facing)]
    inputs = [(name, path.resolve(), how) for name, path, how in inputs if path]
    for _, path, _ in inputs:
        if not path.is_file():
            sys.exit(f"Image not found: {path}")
    espalier = args.espalier / "espalier"
    if not espalier.is_file():
        sys.exit(f"Espalier-2 with Pixal3D not found at {args.espalier}. See docs/multiview.md.")

    mode = "textured" if args.texture else "geometry"
    actor_dir = args.library / args.project / "actors" / args.actor
    run_dir = args.run_dir or actor_dir / "generations" / f"{datetime.now():%Y%m%d-%H%M%S}-multiview-{mode}"
    run_dir.mkdir(parents=True)
    log = run_dir / "espalier.log"
    record = {
        "model": "pixal3d",
        "source_images": {name: str(path) for name, path, _ in inputs},
        "mode": mode,
        "pipeline": args.pipeline,
        "seed": args.seed,
        "smooth": args.smooth,
        "espalier_revision": git_revision(args.espalier),
        "started": datetime.now().isoformat(timespec="seconds"),
    }

    print("Removing background with BiRefNet...")
    views = []
    for name, path, how in inputs:
        rgba = cut_out(path, run_dir / f"cutout_{name}.png", log)
        azimuth = 0.0
        if name != "front":
            direction = facing(head_box(rgba)) if how == "auto" else how
            azimuth = TURN[name] if direction == "left" else -TURN[name]
            print(f"{name.replace('_', '-').capitalize()} view faces {direction}: camera at {azimuth:+.0f} degrees.")
        views.append((name, rgba, azimuth))
    try:
        framing = write_views(views, run_dir / "views", args.fov)
    except ValueError as error:
        sys.exit(str(error))
    record["views"] = framing
    heights = [v["head_height_px"] for v in framing["views"]]
    print("Cropped to head and neck.")
    for v in framing["views"]:
        if not v["chin_found"]:
            print(f"Couldn't find the chin in the {v['name']} view; sized it from the neck instead. "
                  f"Check {run_dir / 'views'}.")
    # The input image the app shows for this version.
    Image.open(run_dir / "views" / "view00_front.png").save(run_dir / "input_rgba.png")
    if args.views_only:
        (run_dir / "run.json").write_text(json.dumps(record, indent=2))
        print(f"Views written to {run_dir / 'views'}; head heights (top of head to chin) {heights} px.")
        return

    cmd = [str(espalier), str(run_dir / "views"), str(run_dir), "--model", "pixal3d", "--no-rembg",
           "--pipeline", args.pipeline, "--seed", str(args.seed)]
    if not args.texture:
        cmd.append("--no-texture")
    baseline = VramMonitor.read_mib()
    print(f"GPU memory in use before generation: {baseline} MiB")
    start = time.monotonic()
    with VramMonitor(run_dir / "vram.csv") as monitor:
        exit_code = run_logged(cmd, log)
    record.update({
        "seconds": round(time.monotonic() - start),
        "vram_baseline_mib": baseline,
        "vram_peak_mib": monitor.peak_mib,
        "vram_peak_above_baseline_mib": monitor.peak_mib - baseline,
        "espalier_exit_code": exit_code,
        "head_heights_px": heights,
    })

    glb = run_dir / "model.glb"
    if exit_code == 0 and glb.exists():
        glb_to_obj(glb, run_dir / "model.obj")
        record["result"] = "ok"
        if not args.no_metahuman:
            print("Exporting the MetaHuman-aligned head with Blender...")
            fbx = run_dir / "head_metahuman.fbx"
            cmd = [sys.executable, str(PHASE1 / "export_metahuman.py"), str(glb), str(fbx), "--smooth", str(args.smooth),
                   "--turn", str(PIXAL3D_TURN)]
            if run_logged(cmd, log) == 0 and fbx.exists():
                record["metahuman_fbx"] = str(fbx)
            else:
                print(f"MetaHuman export failed; the GLB and OBJ are still there. See {log}")
    else:
        record["result"] = "failed"
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))

    print()
    print(f"Result:       {record['result']}")
    print(f"Time:         {record['seconds']} s")
    print(f"Peak GPU use: {monitor.peak_mib} MiB ({monitor.peak_mib - baseline} MiB above baseline)")
    print(f"Output:       {run_dir}")
    if record["result"] != "ok":
        sys.exit(f"Generation failed. The last lines of {log} say why.")


if __name__ == "__main__":
    main()
