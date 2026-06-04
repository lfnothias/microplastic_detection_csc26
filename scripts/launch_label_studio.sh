#!/usr/bin/env bash
# Launch Label Studio for CorSeaCare annotation, with local image serving enabled
# so data/corseacare/*.JPG load via /data/local-files/?d=corseacare/<name>.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$REPO/data"
# Reduce startup network chatter (boat connection): no analytics, no error reporting.
export COLLECT_ANALYTICS=false
export LABEL_STUDIO_SENTRY_DSN=""
echo "Local files document root: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
echo "After it starts (http://localhost:8080):"
echo "  1. Create project 'corseacare'"
echo "  2. Labeling Setup -> Custom template -> paste configs/label_studio_config.xml"
echo "  3. Import data/corseacare_preann/ls_tasks.json (candidate boxes appear as predictions)"
exec "$HOME/.local/bin/label-studio" start
