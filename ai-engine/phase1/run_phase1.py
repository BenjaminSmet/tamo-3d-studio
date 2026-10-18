"""Phase 1: turn one actor portrait into a head mesh with TRELLIS.2 (Espalier-2).

Each run gets its own folder under the actor's generations/ and is never overwritten:

    library/projects/<project>/actors/<actor>/generations/<timestamp>-<mode>/
        input_cutout.png background removed
        input_rgba.png   cropped to head and neck, what TRELLIS.2 sees
        model.glb        from Espalier-2
        model.obj        same mesh, untouched
        head_metahuman.fbx  head and neck, sized and placed like a MetaHuman head (needs Blender)
        espalier.log     full output
        vram.csv         GPU memory every half second
        run.json         settings, timings, peak memory, result

Run it with Espalier-2's Python so trimesh and torch are available:

    ~/espalier/.venv/bin/python ai-engine/phase1/run_phase1.py --image front.png --actor anna
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from crop_head import crop_head_and_neck

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class VramMonitor:
    """Polls nvidia-smi for total GPU memory in use, including the Windows desktop."""

    def __init__(self, csv_path: Path, interval: float = 0.5):
        self.csv_path = csv_path
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def read_mib() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.splitlines()[0].strip())

    def _run(self) -> None:
        start = time.monotonic()
        with self.csv_path.open("w") as f:
            f.write("seconds,used_mib\n")
            while not self._stop.is_set():
                used = self.read_mib()
                self.peak_mib = max(self.peak_mib, used)
                f.write(f"{time.monotonic() - start:.1f},{used}\n")
                f.flush()
                self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def git_revision(path: Path) -> str | None:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or None


def run_logged(cmd: list[str], log_path: Path) -> int:
    """Runs cmd, echoing its output and appending it to log_path."""
    with log_path.open("a") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def has_alpha(path: Path) -> bool:
    with Image.open(path) as image:
        return image.mode in ("RGBA", "LA") or "transparency" in image.info


def glb_to_obj(glb: Path, obj: Path) -> None:
    import trimesh

    mesh = trimesh.load(glb, force="mesh")
    mesh.export(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: portrait to head mesh.")
    parser.add_argument("--image", required=True, type=Path, help="the actor's master front portrait")
    parser.add_argument("--actor", required=True, help="folder name for the actor, e.g. anna")
    parser.add_argument("--project", default="tamo")
    parser.add_argument("--library", type=Path, default=REPO_ROOT / "library" / "projects")
    parser.add_argument("--espalier", type=Path, default=Path.home() / "espalier")
    parser.add_argument("--texture", action="store_true", help="also bake a texture (default: geometry only)")
    parser.add_argument("--pipeline", default="1024_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-background", action="store_true",
                        help="skip background removal; the image must already have an alpha channel")
    parser.add_argument("--no-crop", action="store_true",
                        help="keep the full cut-out instead of cropping to head and neck")
    parser.add_argument("--no-metahuman", action="store_true",
                        help="skip the MetaHuman-aligned FBX (needs Blender)")
    parser.add_argument("--smooth", type=float, default=1.0, help="smoothing for the MetaHuman FBX, 0 = off")
    parser.add_argument("--run-dir", type=Path,
                        help="folder for this run (must not exist yet); default: a new timestamped folder")
    args = parser.parse_args()

    image = args.image.resolve()
    if not image.is_file():
        sys.exit(f"Image not found: {image}")
    espalier = args.espalier / "espalier"
    if not espalier.is_file():
        sys.exit(f"Espalier-2 not found at {args.espalier}. See docs/phase1-install.md.")

    actor_dir = args.library / args.project / "actors" / args.actor
    mode = "textured" if args.texture else "geometry"
    run_dir = args.run_dir or actor_dir / "generations" / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}"
    run_dir.mkdir(parents=True)
    references = actor_dir / "references"
    references.mkdir(exist_ok=True)
    if not (references / image.name).exists():
        shutil.copy2(image, references / image.name)

    log = run_dir / "espalier.log"
    record = {
        "source_image": str(references / image.name),
        "mode": mode,
        "pipeline": args.pipeline,
        "seed": args.seed,
        "smooth": args.smooth,
        "espalier_revision": git_revision(args.espalier),
        "started": datetime.now().isoformat(timespec="seconds"),
    }

    cutout = run_dir / "input_cutout.png"
    if args.keep_background:
        if not has_alpha(image):
            sys.exit("--keep-background needs an image with an alpha channel.")
        shutil.copy2(image, cutout)
    else:
        print("Removing background with BiRefNet...")
        if run_logged([sys.executable, str(HERE / "prepare_image.py"), str(image), str(cutout)], log) != 0:
            sys.exit(f"Background removal failed. See {log}")

    rgba = run_dir / "input_rgba.png"
    cropped = False
    with Image.open(cutout) as full:
        if args.no_crop:
            full.save(rgba)
        else:
            framed, cropped = crop_head_and_neck(full.convert("RGBA"))
            framed.save(rgba)
    record["cropped_to_head_and_neck"] = cropped
    if not args.no_crop:
        print("Cropped to head and neck." if cropped else "No shoulders found; using the full cut-out.")

    cmd = [str(espalier), str(rgba), str(run_dir), "--no-rembg",
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
    })

    glb = run_dir / "model.glb"
    if exit_code == 0 and glb.exists():
        glb_to_obj(glb, run_dir / "model.obj")
        record["result"] = "ok"
        if not args.no_metahuman:
            print("Exporting the MetaHuman-aligned head with Blender...")
            fbx = run_dir / "head_metahuman.fbx"
            cmd = [sys.executable, str(HERE / "export_metahuman.py"), str(glb), str(fbx), "--smooth", str(args.smooth)]
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
