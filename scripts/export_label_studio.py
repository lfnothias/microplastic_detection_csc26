"""Export every Label Studio project to a portable JSON snapshot under labelstudio/.

Reads the local Label Studio sqlite DB directly (no running server / API token needed) and
writes one snapshot per project: labeling config + tasks (data + predictions + annotations).
Image references are already in the form http://localhost:8081/corseacare/<file>, which the
shipped image server (`make annotate`) reproduces on any clone — so snapshots are portable.

No secrets are exported (no users, tokens, or absolute paths). Re-import with
`scripts/restore_label_studio.py` (`make ls-restore`).

    uv run python scripts/export_label_studio.py

Override the DB location with LS_DB=/path/to/label_studio.sqlite3.
"""
import json
import os
import sqlite3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labelstudio"
DEFAULT_DB = Path.home() / "Library" / "Application Support" / "label-studio" / "label_studio.sqlite3"


def _loads(blob):
    """LS stores result columns as JSON text (sometimes NULL); decode defensively."""
    if not blob:
        return []
    if isinstance(blob, (list, dict)):
        return blob
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return []


def _slug(title):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_") or "project"


def main():
    db_path = Path(os.environ.get("LS_DB", DEFAULT_DB))
    if not db_path.exists():
        raise SystemExit(f"Label Studio DB not found: {db_path} (set LS_DB=...)")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    OUT.mkdir(parents=True, exist_ok=True)

    projects = con.execute("SELECT id, title, label_config FROM project ORDER BY id").fetchall()
    index = []
    for proj in projects:
        tasks_out = []
        tasks = con.execute(
            "SELECT id, data FROM task WHERE project_id=? ORDER BY id", (proj["id"],)
        ).fetchall()
        n_pred = n_ann = 0
        for t in tasks:
            preds = con.execute(
                "SELECT result, score, model_version FROM prediction WHERE task_id=?", (t["id"],)
            ).fetchall()
            anns = con.execute(
                "SELECT result, was_cancelled, ground_truth FROM task_completion WHERE task_id=?",
                (t["id"],),
            ).fetchall()
            task = {"data": _loads(t["data"])}
            if preds:
                task["predictions"] = [
                    {"result": _loads(p["result"]),
                     "score": p["score"],
                     "model_version": p["model_version"]}
                    for p in preds
                ]
                n_pred += len(preds)
            if anns:
                task["annotations"] = [
                    {"result": _loads(a["result"]),
                     "was_cancelled": bool(a["was_cancelled"]),
                     "ground_truth": bool(a["ground_truth"])}
                    for a in anns
                ]
                n_ann += len(anns)
            tasks_out.append(task)

        snapshot = {"title": proj["title"], "label_config": proj["label_config"], "tasks": tasks_out}
        fname = f"{proj['id']:02d}_{_slug(proj['title'])}.json"
        (OUT / fname).write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
        index.append({"file": fname, "title": proj["title"],
                      "tasks": len(tasks_out), "predictions": n_pred, "annotations": n_ann})
        print(f"  {fname}: {len(tasks_out)} tasks, {n_pred} predictions, {n_ann} annotations")

    (OUT / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"\n{len(projects)} project snapshots -> {OUT}")


if __name__ == "__main__":
    main()
