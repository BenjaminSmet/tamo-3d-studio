"""Stands in for run_phase1.py, run_multiview.py and run_facemodel.py in tests: prints the same
messages and writes a run folder. Multi-view is the call with --front, the face model the call
with --engine-python.

FAKE_ENGINE=fail makes it fail; FAKE_ENGINE=slow makes it wait so it can be cancelled.
FAKE_GLB copies a real model in, for trying the interface without a GPU.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--image")
parser.add_argument("--front")
parser.add_argument("--side")
parser.add_argument("--three-quarter")
parser.add_argument("--other-side")
parser.add_argument("--back")
parser.add_argument("--actor")
parser.add_argument("--project")
parser.add_argument("--library")
parser.add_argument("--espalier")
parser.add_argument("--engine-python")
parser.add_argument("--run-dir", type=Path)
parser.add_argument("--pipeline")
parser.add_argument("--seed")
parser.add_argument("--smooth", default="0")
parser.add_argument("--texture", action="store_true")
parser.add_argument("--no-texture", action="store_true")
parser.add_argument("--no-crop", action="store_true")
args = parser.parse_args()

mode = os.environ.get("FAKE_ENGINE", "ok")
args.run_dir.mkdir(parents=True)
print("Removing background with BiRefNet...", flush=True)
print("Cropped to head and neck.", flush=True)
print("GPU memory in use before generation: 1200 MiB", flush=True)
# Espalier-2's sampler bars, as in a 1024_cascade run: the shape is sampled twice.
for bar in ("Sampling sparse structure", "Sampling shape SLat", "Sampling shape SLat", "Sampling texture SLat"):
    for percent in (10, 50, 100, 100):   # tqdm redraws the finished bar
        sys.stdout.write(f"\r{bar}: {percent}%|#####| {percent}/100")
        sys.stdout.flush()
        time.sleep(0.02 if mode != "slow" else 0.5)
    print(flush=True)
if mode == "fail":
    print("CUDA out of memory", flush=True)
    sys.exit(1)
glb = os.environ.get("FAKE_GLB")
(args.run_dir / "model.glb").write_bytes(Path(glb).read_bytes() if glb else b"glTF")
print("Exporting the MetaHuman-aligned head with Blender...", flush=True)
(args.run_dir / "head_metahuman.fbx").write_bytes(b"FBX")
facemodel_texture = args.engine_python and not args.no_texture
if facemodel_texture:
    (args.run_dir / "albedo.png").write_bytes(b"PNG")
record = {"model": "gnm" if args.engine_python else "pixal3d", "source_images": {"front": args.front, "side": args.side,
                                                 "three_quarter": args.three_quarter}} if args.front else {}
(args.run_dir / "run.json").write_text(json.dumps({
    **record, "source_image": args.image or args.front, "mode": "textured" if args.texture or facemodel_texture else "geometry",
    "pipeline": args.pipeline, "seed": int(args.seed), "smooth": float(args.smooth),
    "started": "2026-10-18T21:00:00", "seconds": 1, "result": "ok",
}))
print("Result:       ok", flush=True)
