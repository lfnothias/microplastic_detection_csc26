"""Restore Label Studio projects from the portable snapshots in labelstudio/.

For each snapshot it creates a project (title + labeling config) via the Label Studio API and
imports its tasks, predictions and annotations. Idempotent-ish: pass --skip-existing to leave
projects whose title already exists untouched (default re-creates a fresh copy).

Prerequisites:
  1. Label Studio running + the image server (so http://localhost:8081/... resolves):
         make annotate
  2. An API token:
         export LABEL_STUDIO_API_KEY=<token>   # LS -> Account & Settings -> Access Token

    uv run python scripts/restore_label_studio.py [--skip-existing] [--only TITLE ...]

Override the server URL with LABEL_STUDIO_URL (default http://localhost:8080).
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "labelstudio"


# LS >= 1.23 disables legacy "Token <key>" auth by default; newer personal access tokens use
# "Bearer <jwt>". Try legacy first, fall back to Bearer on 401 so either token type works.
_AUTH_SCHEMES = ["Token", "Bearer"]


def _api(method, path, token, base, payload=None):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    last = None
    for scheme in _AUTH_SCHEMES:
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"{scheme} {token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as r:
                body = r.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            last = (e.code, e.read().decode()[:300])
            if e.code in (401, 403):
                continue  # wrong scheme — try the next one
            break
    code, detail = last or ("?", "")
    hint = ("\nHint: on Label Studio >= 1.23 the legacy 'Access Token' is disabled — create a "
            "Personal Access Token (Account & Settings -> Personal Access Token) and export it as "
            "LABEL_STUDIO_API_KEY, or re-enable legacy tokens in Organization settings.") if code in (401, 403) else ""
    raise SystemExit(f"API {method} {path} failed ({code}): {detail}{hint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave projects whose title already exists untouched")
    ap.add_argument("--only", nargs="*", default=None, help="restore only these project titles")
    args = ap.parse_args()

    token = os.environ.get("LABEL_STUDIO_API_KEY")
    if not token:
        sys.exit("Set LABEL_STUDIO_API_KEY (LS -> Account & Settings -> Access Token).")
    base = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")

    index_path = SNAP / "index.json"
    files = ([SNAP / e["file"] for e in json.loads(index_path.read_text())]
             if index_path.exists() else sorted(SNAP.glob("*.json")))
    files = [f for f in files if f.name != "index.json"]

    existing = {p["title"] for p in _api("GET", "/api/projects?page_size=1000", token, base).get("results", [])}

    for f in files:
        snap = json.loads(f.read_text())
        title = snap["title"]
        if args.only and title not in args.only:
            continue
        if args.skip_existing and title in existing:
            print(f"  skip (exists): {title}")
            continue
        proj = _api("POST", "/api/projects", token, base,
                    {"title": title, "label_config": snap["label_config"]})
        pid = proj["id"]
        tasks = snap["tasks"]
        # LS import accepts the task list with embedded predictions + annotations.
        _api("POST", f"/api/projects/{pid}/import", token, base, tasks)
        n_pred = sum(len(t.get("predictions", [])) for t in tasks)
        n_ann = sum(len(t.get("annotations", [])) for t in tasks)
        print(f"  restored '{title}' (id {pid}): {len(tasks)} tasks, {n_pred} predictions, {n_ann} annotations")

    print("\nDone. Open Label Studio to review.")


if __name__ == "__main__":
    main()
