# Face model heads (Phase 5)

TRELLIS.2 and Pixal3D are general image-to-3D models: they copy what they see (lash lines,
moles) and guess what they don't. Their heads come out with bumps and dents on the skull and a
flat, thin mouth, and MetaHuman copies both: after the solve the skull is lumpy and the teeth
poke through the lips.

The face model method builds the head from [GNM](https://github.com/google/GNM), Google's head
model learned from 3D scans (Apache 2.0, code and weights). It can only make real heads: a
smooth skull, eyes and closed lips with depth over a proper mouth seam, about 25k faces of clean
topology. The photos decide its shape, and a front-only TRELLIS.2 head adds a little of the
face's own shape on top.

## Install

Everything goes in its own folder and Python 3.12 environment, `~/tamo-gnm`, so Espalier-2 is
never touched. It runs on the CPU, so Unreal can stay open during the fit (but not during the
TRELLIS.2 step).

```bash
bash ai-engine/phase5/install.sh
```

It clones GNM, installs PyTorch (CPU) and GNM's PyTorch backend without its TensorFlow
dependency, MediaPipe 0.10.14 with its Face Landmarker model (Apache 2.0), and downloads GNM's
weights into `~/.cache/gnm/models`. Restart the app, then pick **Face model
(GNM)** as the Method.

## How a run works

1. A front-only TRELLIS.2 head without a texture (`trellis/` in the run folder), about 2
   minutes on the GPU (about 4 GB).
2. The other photos (three-quarter, side, other side, back) are cut out. One that faces right is
   mirrored for the fit, since GNM is symmetric; its camera is turned back afterwards, so the skin
   texture always uses the photo as taken.
3. GNM is fitted to the photos (`fit_gnm_v2.py`, about 2.5 minutes on the CPU). Each photo gets
   its own camera. MediaPipe's face points on the front and three-quarter photos place the eyes,
   nose and lips, and the outline of the head in every photo shapes the skull, above the chin
   only and never where the photo cuts the head off. The lips are matched by their edges (the
   outer lip line to GNM's lip border, the inner one to its mouth seam), so they keep their
   fullness. In a profile the line under the jaw, from the chin back to the neck, is followed
   too (yellow in the overlay), so the under-chin doesn't hang below the photo's. A back photo
   adds its outline at half weight. The fit starts close to the average head and loosens in
   stages.
   `fit/overlay_<view>.png` shows the fit on each photo: cyan is the head's outline, red the
   photo's face points, green the model's.
4. The hybrid (`hybrid.py`) moves GNM's face toward the TRELLIS.2 head's surface along its
   normals, by at most 1 cm, smoothed and faded out at the forehead, ears, temples and jaw.
   It never touches the mouth seam or the eyes, and it only makes the lips fuller.
   The lip border is then raised 1.5 mm (`--lip-edge`), with the skin just outside it a little
   lower, so the lip outline reads as a crisp edge for MetaHuman's tracking.
5. With **Skin texture** on (on by default in the app and on the command line, which skips it when given
   `--no-texture`; about 1.5 minutes more and 7 GB of memory), the
   skin is baked from the photos (`texture.py`): projected through the fitted cameras, shine and
   the photos' lighting taken out, the views colour-matched and blended. `albedo.png` (4K) is
   written, and `head_metahuman.fbx` carries it.
6. `model.glb`, `model.obj` and `head_metahuman.fbx`, exported without smoothing.

Side, three-quarter, other-side and back photos are optional, but without them the depth of the
skull comes from GNM's average head, and the skin texture guesses whatever no photo shows (the
far cheek, the back of the head). They work best from the same shoot as the front, in the same
soft, even light, large (2000 px or more across the face), the whole head in frame, bald or with
the hair tied back.

From the command line:

```bash
~/tamo-gnm/.venv/bin/python ai-engine/phase5/run_facemodel.py --actor nia_okafor \
    --front front.png --side side.png --three-quarter three_quarter.png [--other-side o.png] [--back b.png] \
    [--no-texture]
```

## In MetaHuman

Import `head_metahuman.fbx` with default settings and use From Custom Mesh, as before. MetaHuman
takes only the shape from it: `albedo.png` is laid out on our head's UVs, not MetaHuman's, so it
can't go into MetaHuman's texture overrides as it is. The
eyes stay close to GNM's average shape on purpose, since the eye area is where image models add
lash ridges: tune the eyes in MetaHuman with its eye curves.

## First results (Nia and Amara, 2026-10-21)

The first MetaHuman built from a hybrid head came out really close to the photo, with a
clean skull and closed lips. The lips are fuller than GNM's average but still a little
thinner than the photos.

With the front photo alone (Nia), the fit to the photo is as good, but the skull comes out
about 8 mm shorter front to back, the profile flatter and the chin weaker: add side and
three-quarter photos when you have them.
