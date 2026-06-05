"""Export Label Studio annotations (incl. DRAFTS) from the LS sqlite to a YOLO-det dataset.

Reads `tasks_annotationdraft` (works even when annotations aren't "Submitted"), maps the old
`matiere_organique` label → `autre`, and writes a 6-class YOLO dataset under
data/ls_export_yolo/ (gitignored). Submitted annotations (task_completion) are preferred when
present, falling back to the latest draft per task.

    .venv/bin/python scripts/export_from_ls.py
"""
import sqlite3
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path.home() / "Library/Application Support/label-studio/label_studio.sqlite3"
IMAGES = ROOT / "data" / "corseacare"
OUT = ROOT / "data" / "ls_export_yolo"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
ALIAS = {"matiere_organique": "autre"}


def _regions_to_lines(regions):
    lines = []
    for r in regions:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        lab = ALIAS.get(v["rectanglelabels"][0], v["rectanglelabels"][0])
        if lab not in CLASSES:
            print(f"  WARN unknown label '{lab}' skipped")
            continue
        cid = CLASSES.index(lab)
        cx, cy = (v["x"] + v["width"] / 2) / 100, (v["y"] + v["height"] / 2) / 100
        w, h = v["width"] / 100, v["height"] / 100
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()
    tid2name = {}
    for tid, data in cur.execute("SELECT id, data FROM task"):
        d = json.loads(data) if isinstance(data, str) else data
        tid2name[tid] = d.get("image", "").rsplit("/", 1)[-1]
    # submitted annotations first, then latest draft
    results = {}
    for tid, res in cur.execute("SELECT task_id, result FROM task_completion ORDER BY updated_at"):
        results[tid] = json.loads(res) if isinstance(res, str) else res
    for tid, res in cur.execute("SELECT task_id, result FROM tasks_annotationdraft ORDER BY updated_at"):
        results.setdefault(tid, json.loads(res) if isinstance(res, str) else res)

    items = []
    for tid, regions in results.items():
        name = tid2name.get(tid)
        if not name or not (IMAGES / name).exists():
            print(f"skip task {tid}: image missing ({name})")
            continue
        items.append((name, _regions_to_lines(regions)))
    items.sort()
    n_val = min(2, len(items) - 1) if len(items) > 2 else 1

    for i, (name, lines) in enumerate(items):
        split = "val" if i < n_val else "train"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(IMAGES / name, OUT / "images" / split / name)
        (OUT / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames: [{', '.join(CLASSES)}]\n")

    hist = Counter()
    for _, lines in items:
        for ln in lines:
            hist[CLASSES[int(ln.split()[0])]] += 1
    print(f"{len(items)} images, {sum(len(l) for _, l in items)} boxes -> {OUT} ({n_val} val)")
    print("class counts:", dict(hist))


if __name__ == "__main__":
    main()
