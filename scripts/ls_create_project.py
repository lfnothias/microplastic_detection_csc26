"""Create a Label Studio project for CorSeaCare via the API — automates the annotation setup.

Avoids clicking through the UI: creates a project with the 6-class labeling config and (optionally)
imports a tasks file (e.g. pre-annotations).

Prereqs:
  1. Label Studio running (./scripts/launch_label_studio.sh or `make label-studio`).
  2. The image server running (`make serve`) so http://localhost:8081/... URLs in the tasks load.
  3. An API token: Label Studio -> Account & Settings -> Access Token, then:
         export LABEL_STUDIO_API_KEY=<token>

Usage:
    .venv/bin/python scripts/ls_create_project.py \
        [--title corseacare] [--url http://localhost:8080] [--tasks <tasks.json>]
"""
import argparse
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "label_studio_config.xml"


def api(url, key, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url.rstrip("/") + path, data=body,
        headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Label Studio API error {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach Label Studio at {url} — is it running? ({e.reason})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default="corseacare")
    ap.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    key = os.environ.get("LABEL_STUDIO_API_KEY")
    if not key:
        raise SystemExit("Set LABEL_STUDIO_API_KEY (Label Studio -> Account & Settings -> Access Token).")
    if not CONFIG.exists():
        raise SystemExit(f"missing labeling config: {CONFIG}")

    proj = api(args.url, key, "/api/projects",
               {"title": args.title, "label_config": CONFIG.read_text()})
    pid = proj["id"]
    print(f"created project '{args.title}' (id={pid}) with the 6-class config")

    if args.tasks:
        tasks = json.loads(Path(args.tasks).read_text())
        res = api(args.url, key, f"/api/projects/{pid}/import", tasks)
        print(f"imported {res.get('task_count', len(tasks))} tasks from {args.tasks}")
    print(f"open: {args.url}/projects/{pid}/data")


if __name__ == "__main__":
    main()
