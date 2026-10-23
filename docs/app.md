# Tamo 3D Studio app

The app has three parts that run on your PC only:

- **Engine server** (`backend/`): a small FastAPI server inside Ubuntu (WSL). It keeps projects,
  actors, reference images and versions in `database/studio.sqlite`, and runs generations as
  background jobs, one at a time because they share the GPU. Jobs keep going if you close the
  browser, and a job that was running when the server stopped starts again next time.
- **Interface** (`app/`): React, TypeScript and Tailwind, with a Three.js viewer, served by the
  engine server.
- **Windows app** (`desktop/`): a Tauri window. Opening it starts `./studio.sh` in Ubuntu out of
  sight and shows the interface once the engine answers; closing it stops the engine (a
  running generation goes back to the queue). It looks for the repo at `~/tamo-3d-studio` in
  `Ubuntu-24.04`; the `TAMO_WSL_REPO` and `TAMO_WSL_DISTRO` Windows environment variables
  change that. The engine's output goes to
  `%LOCALAPPDATA%\studio.tamo.app\logs\engine.log`.

## Install the Windows app

GitHub builds the installer whenever `desktop/` changes: open the latest run of the
[Windows app workflow](https://github.com/BenjaminSmet/tamo-3d-studio/actions/workflows/windows-app.yml),
download **Tamo-3D-Studio-installer** under Artifacts, unzip it and run the setup. It isn't
signed, so Windows SmartScreen warns the first time: **More info → Run anyway**. It installs
for your user only, with a Start menu entry.

## Update

Press **Check for updates** in the top bar, then **Update**. It pulls the latest code from
GitHub, restarts the engine (rebuilding the interface if needed) and reloads the window.
It waits if a generation is queued or running, and refuses rather than overwrite changes
made on this PC. Only when the Windows app itself changes does the top bar ask you to
install the new one.

## Start it without the Windows app

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
   look). Follow [shape-reference.md](shape-reference.md) for the front image.
3. Pick the **Method**. **Front only (TRELLIS.2)** uses just the front image.
   **Multi-view (Pixal3D)** also uses the Side and Three-quarter images and can follow the
   profile (nose, lips, chin, back of the skull) more closely, but only with turned views
   from the same shoot as the front, each showing the whole head; a photo that cuts off the
   back of the skull gives a squeezed, tall head, so the run stops with a message instead.
   Multi-view needs Pixal3D installed ([multiview.md](multiview.md)).
   **Face model (GNM)** fits Google's head model to the front photo and, when there are
   some, the Side, Three-quarter, Other side and Back images: a smooth skull and real lips, so
   MetaHuman's solve has no bumps and no teeth poking through. It is the default when it's
   installed ([facemodel.md](facemodel.md)), and takes about 7 minutes. **Skin texture**, on by
   default, bakes `albedo.png` from the photos so the head shows the skin in MetaHuman while you
   place the markers; untick it for shape only (about 1.5 minutes quicker).
   Press **Generate head**. Progress shows in the viewer and in the queue on the left.
4. Turn the head in the viewer (textured or clay), and download **MetaHuman head (FBX)**.
   In Unreal, import it with default settings, then MetaHuman Character → Import →
   From Custom Mesh.
5. Every run is a new version; mark the best one as preferred.
6. To delete an actor, an image or a version, use its **Delete** link and confirm. Nothing
   is erased: it moves to `library/trash/<date-time>/` with the same folders
   (`\\wsl.localhost\Ubuntu-24.04\home\ben\tamo-3d-studio\library\trash` in Explorer).
   To undo, move it back and restart the app. Empty the trash yourself when you like.
   Anything with a generation queued or running can't be deleted until you cancel it.

Keep Unreal closed while a head is generating; they share the GPU.

## Development

```bash
# server tests (uses a stand-in engine, no GPU needed)
uv venv backend/.venv && uv pip install --python backend/.venv/bin/python -r backend/requirements-dev.txt
cd backend && .venv/bin/python -m pytest -q

# interface with live reload on http://localhost:5173 (talks to the server on :8765)
cd app && npm install && npm run dev
```
