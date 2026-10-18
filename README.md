# Tamo 3D Studio

A free, fully local app that turns actor images into clean, MetaHuman-ready head meshes.

The project is in **Phase 1: proof of quality**. There is no app yet, only a script. The
question Phase 1 answers is whether TRELLIS.2 produces a head from one portrait that
conforms cleanly in Mesh to MetaHuman and still looks like the actor.

## Layout

```text
app/                  # Tauri shell + React interface (Phase 3)
backend/              # FastAPI server, job queue, exporters (Phase 3)
ai-engine/
├── adapters/         # one adapter per model (Phase 2+)
├── models/           # downloaded weights (not committed)
└── phase1/           # Phase 1 scripts
library/projects/     # your images and meshes (not committed)
database/             # studio.sqlite (Phase 3)
docs/
```

## Phase 1

1. Install the engine: [docs/phase1-install.md](docs/phase1-install.md)
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
   installed; `--no-metahuman` skips it). It is cut at the neck, about 200k faces, and
   sized and placed exactly like MetaHuman's own head. Import it in Unreal with the default
   settings (Uniform Scale 1.0) and don't move it, then in the MetaHuman Character use
   **Import → From Custom Mesh** (not From Template). For an older run:

   ```bash
   python3 ai-engine/phase1/export_metahuman.py <run folder>/model.glb head_metahuman.fbx
   ```
