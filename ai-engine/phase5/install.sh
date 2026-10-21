#!/usr/bin/env bash
# Sets up the face model (GNM) in its own Python 3.12 environment at ~/tamo-gnm (or $TAMO_GNM),
# separate from Espalier-2. CPU only; about 1.5 GB. Safe to run again.
set -euo pipefail

HOME_DIR="${TAMO_GNM:-$HOME/tamo-gnm}"
mkdir -p "$HOME_DIR/models"
cd "$HOME_DIR"

if [ ! -d GNM ]; then
  git clone --depth 1 https://github.com/google/GNM.git GNM
fi
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv
fi
PY=.venv/bin/python
uv pip install -q --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
# --no-deps: GNM's package also requires TensorFlow, which only its TensorFlow backend uses.
uv pip install -q --python "$PY" --no-deps -e GNM/gnm/shape
uv pip install -q --python "$PY" absl-py etils immutabledict typing_extensions numpy scipy trimesh opt-einsum \
  typeguard opencv-python-headless pillow rtree
# 0.10.14: newer MediaPipe needs libGLESv2 (sudo apt install libgles2); this one runs on the CPU without it.
# It also provides the face mesh region lists the fit uses.
uv pip install -q --python "$PY" "mediapipe==0.10.14"
if [ ! -f models/face_landmarker.task ]; then
  curl -sSL -o models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
fi
# Downloads GNM's weights (huggingface.co/google/gnm-v3) into ~/.cache/gnm/models.
"$PY" -c "from gnm.shape import gnm_pytorch as G; G.GNM.from_remote(version=G.GNMMajorVersion.V3, variant=G.GNMVariant.HEAD); print('GNM ready')"
echo "Face model installed in $HOME_DIR. Restart the app to see the Face model method."
