"""Stands in for run_phase1.py in tests: prints the same messages and writes a run folder.

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
parser.add_argument("--actor")
parser.add_argument("--project")
parser.add_argument("--library")
parser.add_argument("--espalier")
parser.add_argument("--run-dir", type=Path)
parser.add_argument("--pipeline")
parser.add_argument("--seed")
parser.add_argument("--smooth")
parser.add_argument("--texture", action="store_true")
parser.add_argument("--no-crop", action="store_true")
args = parser.parse_args()

mode = os.environ.get("FAKE_ENGINE", "ok")
args.run_dir.mkdir(parents=True)
print("Removing background with BiRefNet...", flush=True)
print("Cropped to head and neck.", flush=True)
print("GPU memory in use before generation: 1200 MiB", flush=True)
for percent in (10, 50, 100):
    sys.stdout.write(f"\rSampling: {percent}%|#####| {percent}/100")
    sys.stdout.flush()
    time.sleep(0.05 if mode != "slow" else 2)
print(flush=True)
if mode == "fail":
    print("CUDA out of memory", flush=True)
    sys.exit(1)
glb = os.environ.get("FAKE_GLB")
(args.run_dir / "model.glb").write_bytes(Path(glb).read_bytes() if glb else b"glTF")
print("Exporting the MetaHuman-aligned head with Blender...", flush=True)
(args.run_dir / "head_metahuman.fbx").write_bytes(b"FBX")
(args.run_dir / "run.json").write_text(json.dumps({
    "source_image": args.image, "mode": "textured" if args.texture else "geometry",
    "pipeline": args.pipeline, "seed": int(args.seed), "smooth": float(args.smooth),
    "started": "2026-10-18T21:00:00", "seconds": 1, "result": "ok",
}))
print("Result:       ok", flush=True)
