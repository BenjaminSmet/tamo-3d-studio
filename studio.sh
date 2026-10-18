#!/usr/bin/env bash
# Starts Tamo 3D Studio inside WSL: sets up the server's own Python environment (separate
# from Espalier-2's), builds the interface when it has changed, and serves both on
# http://localhost:8765, which the Windows browser can open.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  echo "uv not found. It came with the Espalier-2 install; open a new terminal or see docs/app.md." >&2
  exit 1
fi
if [ ! -x backend/.venv/bin/python ]; then
  uv venv --quiet backend/.venv --python 3.12
fi
uv pip install --quiet --python backend/.venv/bin/python -r backend/requirements.txt

if [ ! -f app/dist/index.html ] || [ -n "$(find app/src app/index.html app/package.json -newer app/dist/index.html -print -quit)" ]; then
  if ! command -v npm >/dev/null; then
    echo "Node.js not found in Ubuntu; it's needed once to build the interface. See docs/app.md." >&2
    exit 1
  fi
  echo "Building the interface..."
  (cd app && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent)
fi

echo "Tamo 3D Studio: open http://localhost:8765 in your browser. Ctrl+C stops it."
cd backend
exec .venv/bin/uvicorn --factory tamo_server.main:create_app --host 127.0.0.1 --port 8765 --log-level warning
