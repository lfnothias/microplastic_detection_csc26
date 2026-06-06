#!/usr/bin/env bash
# One command to start the annotation stack: CORS image server (:8081) in the background,
# then Label Studio (:8080) in the foreground. Ctrl-C stops Label Studio; the image server
# keeps running (kill it with: pkill -f serve_images.py).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"

if ! curl -s -o /dev/null http://localhost:8081/ 2>/dev/null; then
  echo "==> starting CORS image server at http://localhost:8081 (background) ..."
  ("$PY" "$REPO/scripts/serve_images.py" >/dev/null 2>&1 &)
  sleep 1
else
  echo "==> image server already running at http://localhost:8081"
fi

echo "==> starting Label Studio at http://localhost:8080 ..."
exec "$REPO/scripts/launch_label_studio.sh"
