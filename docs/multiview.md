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
./install.sh --pixal3d
```

## Photos

Take the side and three-quarter photos alongside the front shape photo
([shape-reference.md](shape-reference.md)): same identity, light and expression, camera at eye
level, head to collarbone, plain background. The side photo is a true profile (camera at 90
degrees), the three-quarter one at about 45 degrees. Either side works; the script reads which
way the face points.

## Run

```bash
~/espalier-pixal3d/.venv/bin/python ai-engine/phase4/run_multiview.py --actor nia_okafor \
    --front front.png --side side.png --three-quarter three_quarter.png --texture
```

Each view is cut out and cropped square from the top of the head to the collarbone, sized by
the head's height so the head is the same size in every view. The cameras form Pixal3D's
standard orbit (20 degree field of view, eye level, front camera first); `views/` in the run
folder holds exactly what Pixal3D saw, with `transforms.json`. The run folder is the same as
Phase 1's, including `head_metahuman.fbx`.
