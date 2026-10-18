# Tamo 3D Studio

A free, fully local app that turns actor images into clean, MetaHuman-ready head meshes.

Phase 1 (proof of quality) passed: a head generated from one portrait fits in Mesh to
MetaHuman and looks like the actor. The project is now in **Phase 3: the desktop app**.

## The app

In the Ubuntu terminal, run `./studio.sh`, then open <http://localhost:8765> in your
browser. See [docs/app.md](docs/app.md).

## Layout

```text
app/                  # React interface (Tauri packaging later)
backend/              # FastAPI server and job queue
ai-engine/
├── adapters/         # one adapter per model (Phase 2+)
├── models/           # downloaded weights (not committed)
└── phase1/           # Phase 1 scripts
library/projects/     # your images and meshes (not committed)
database/             # studio.sqlite (not committed)
docs/
```

## Phase 1 script

The app runs this same pipeline. To use it from the command line:

1. Install the engine: [docs/phase1-install.md](docs/phase1-install.md). For the portrait, follow
   [docs/shape-reference.md](docs/shape-reference.md).
2. Generate a head:

   ```bash
   ~/espalier/.venv/bin/python ai-engine/phase1/run_phase1.py \
       --image /mnt/c/Users/<you>/Pictures/actor-front.png --actor <actor-name>
   ```

   The portrait is cut out and cropped square to the head and neck automatically
   (`--no-crop` turns that off), then generated at `1024_cascade`.
   Every run creates a new folder under
   `library/projects/<project>/actors/<actor>/generations/` with the mesh (GLB and OBJ),
   the full log, and `run.json` with settings, timings and peak video memory.
   Nothing is ever overwritten.
3. Bring it into MetaHuman: each run also writes `head_metahuman.fbx` (needs Blender
   installed; `--no-metahuman` skips it). It is cut at the neck, about 200k faces,
   smoothed (`--smooth`, default 1.0, removes skull bumps and brow ridges), and sized and placed
   exactly like MetaHuman's own head. Import it in Unreal with the default
   settings (Uniform Scale 1.0) and don't move it, then in the MetaHuman Character use
   **Import → From Custom Mesh** (not From Template). For an older run:

   ```bash
   python3 ai-engine/phase1/export_metahuman.py <run folder>/model.glb head_metahuman.fbx
   ```
