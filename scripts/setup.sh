#!/usr/bin/env bash
# CorSeaCare_yolo — one-shot setup for macOS (Apple Silicon).
# Installs the light engine (no PyTorch) + Label Studio for annotation.
# Real-model backends (Ultralytics / SAM2) are installed separately — see the end.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is required. Install it with:  brew install uv" >&2
  exit 1
fi

echo "==> Installing the CorSeaCare engine (+ dev tools) into .venv ..."
uv sync --extra dev

echo "==> Installing Label Studio (annotation UI) as a uv tool ..."
uv tool install label-studio || echo "   (label-studio already installed — skipping)"

echo
echo "Done. Quick check:"
echo "    uv run pytest -q"
echo
echo "To run/train the REAL models (pulls in PyTorch, ~GB — use a good connection):"
echo '    uv pip install "ultralytics>=8.3" "sam-2 @ git+https://github.com/facebookresearch/sam2.git"'
echo
echo "Then see README.md (Annotate your own sieve photos) and docs/ANNOTATION_GUIDE.md."
