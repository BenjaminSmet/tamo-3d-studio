# Tamo 3D Studio app

The app has two parts that run on your PC only:

- **Engine server** (`backend/`): a small FastAPI server inside Ubuntu (WSL). It keeps projects,
  actors, reference images and versions in `database/studio.sqlite`, and runs generations as
  background jobs, one at a time because they share the GPU. Jobs keep going if you close the
  browser, and a job that was running when the server stopped starts again next time.
- **Interface** (`app/`): React, TypeScript and Tailwind, with a Three.js viewer. For now it opens
  in your Windows browser; packaging it as a Windows app comes later.

## Start it

In the Ubuntu terminal:

```bash
cd ~/tamo-3d-studio
./studio.sh
```

Then open <http://localhost:8765> in your Windows browser. Ctrl+C in the terminal stops it.

The first start sets up the server's own Python environment (separate from Espalier-2's, which
it never changes) and builds the interface. Building needs Node.js 20.19 or newer inside
Ubuntu, once:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

Your existing runs under `library/projects/` show up automatically as versions.

## Using it

1. Pick or create a project, then add an actor.
2. Drop in images and say which view each one is (front, three-quarter, side, back, or final
   look). Only the front image is used to build the head today; the rest are kept for
   multi-view later. Follow [shape-reference.md](shape-reference.md) for the front image.
3. Press **Generate head**. Progress shows in the viewer and in the queue on the left.
4. Turn the head in the viewer (textured or clay), and download **MetaHuman head (FBX)**.
   In Unreal, import it with default settings, then MetaHuman Character → Import →
   From Custom Mesh.
5. Every run is a new version; mark the best one as preferred.

Keep Unreal closed while a head is generating; they share the GPU.

## Development

```bash
# server tests (uses a stand-in engine, no GPU needed)
uv venv backend/.venv && uv pip install --python backend/.venv/bin/python -r backend/requirements-dev.txt
cd backend && .venv/bin/python -m pytest -q

# interface with live reload on http://localhost:5173 (talks to the server on :8765)
cd app && npm install && npm run dev
```
