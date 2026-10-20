# Multi-view heads (Phase 4)

A single front photo leaves the profile (nose, jaw, ears, back of the head) to guesswork.
Phase 4 adds side and three-quarter photos with [Pixal3D](https://github.com/TencentARC/Pixal3D)
(MIT licence for code and weights), which builds on TRELLIS.2 and takes several posed views.

## Install

Pixal3D goes in its own copy of Espalier-2, so the TRELLIS.2 setup the app uses stays as it is.
Its weights take about 43 GB.

```bash
git clone <espalier remote> ~/espalier-pixal3d
cd ~/espalier-pixal3d
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export MAX_JOBS=4
./install.sh --pixal3d
```

`MAX_JOBS=4` keeps the CUDA extension build within 16 GB of RAM: with ninja installed it
otherwise starts one compile per CPU thread and runs out of memory. The build takes about
20 minutes.

## Photos

Take the side and three-quarter photos alongside the front shape photo
([shape-reference.md](shape-reference.md)): same identity, light and expression, camera at eye
level, head to collarbone, plain background. The side photo is a true profile (camera at 90
degrees), the three-quarter one at about 45 degrees. Either side works; the script reads which
way the face points.

## Run

In the app, pick **Multi-view (Pixal3D)** as the method (see [app.md](app.md)). From the command line:

```bash
~/espalier-pixal3d/.venv/bin/python ai-engine/phase4/run_multiview.py --actor nia_okafor \
    --front front.png --side side.png --three-quarter three_quarter.png --texture
```

Each view is cut out and cropped square from the top of the head to the collarbone, sized by
the head's height so the head is the same size in every view. The cameras form Pixal3D's
standard orbit (20 degree field of view, eye level, front camera first); `views/` in the run
folder holds exactly what Pixal3D saw, with `transforms.json`. The run folder is the same as
Phase 1's, including `head_metahuman.fbx`.

A run takes about 6 minutes, most of it loading the model, and needs about 11 GB of GPU memory
(TRELLIS.2 needs about 7 GB), so keep Unreal closed. Pixal3D's own background remover
(RMBG-2.0) is gated and licensed for non-commercial use only, so the script cuts the views out
itself and always runs Pixal3D with `--no-rembg`. Pixal3D's `model.glb` faces the opposite way
from TRELLIS.2's; the MetaHuman export turns it round, so `head_metahuman.fbx` faces the usual
way.

## First results (Nia, 2026-10-20)

With matching head sizes in every view, both multi-view heads (from the triptych sheet and
from the app's references) beat the front-only TRELLIS.2 head on shape: the forehead slope,
nose, lips, chin and the long back of the skull follow the side photo. The texture is worse
(eye make-up and brows bake in darker), so use the multi-view `head_metahuman.fbx` for the
shape, since MetaHuman builds its own skin. Views from one shoot work best; earrings come out
as small blobs.
