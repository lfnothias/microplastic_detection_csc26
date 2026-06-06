#!/usr/bin/env bash
# Launch Label Studio for CorSeaCare annotation.
# Images are served separately by scripts/serve_images.py (HTTP+CORS at :8081) and referenced
# by plain http:// URLs in the task files — so NO local-files config is needed.
# Tip: `make annotate` starts the image server AND Label Studio together.
set -euo pipefail
export COLLECT_ANALYTICS=false          # quiet startup (no analytics / error reporting)
export LABEL_STUDIO_SENTRY_DSN=""

echo "Label Studio -> http://localhost:8080"
echo "Make sure the image server is running too:  make serve   (http://localhost:8081)"
echo
echo "Set up the project (once):"
echo "  automatic:  export LABEL_STUDIO_API_KEY=<token from Account & Settings>"
echo "              make ls-project        # creates project + 6-class config (+ TASKS=<file> to import)"
echo "  manual:     new project -> Labeling Setup -> Custom template ->"
echo "              paste configs/label_studio_config.xml, then Import a tasks .json"
exec "$(command -v label-studio || echo "$HOME/.local/bin/label-studio")" start
