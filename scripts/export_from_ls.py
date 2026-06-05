"""Export Label Studio annotations (incl. DRAFTS) from the LS sqlite to a YOLO-det dataset.

Reads `tasks_annotationdraft` (works even when annotations aren't "Submitted"), maps the old
`matiere_organique` label → `autre`, and writes a 6-class YOLO dataset under
data/ls_export_yolo/ (gitignored). Submitted annotations (task_completion) are preferred when
present, falling back to the latest draft per task.

    .venv/bin/python scripts/export_from_ls.py
"""
import sqlite3
import json
import csv
import shutil
from collections import Counter, defaultdict
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
        labs = v.get("rectanglelabels") or []
        if not labs:
            continue                       # box drawn without a class assigned — skip
        lab = ALIAS.get(labs[0], labs[0])
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
    # split: honor an explicit `split` column in samples.csv (train/val); otherwise fall back to
    # the sample-aware heuristic (hold out the 2 fewest-view samples). Explicit is preferred so
    # the val set is in-domain (your TAMIS sieves) rather than the foreign MANTA collection.
    smap, split_col = {}, {}
    mpath = ROOT / "samples.csv"
    if mpath.exists():
        for row in csv.DictReader(open(mpath)):
            smap[row["image"]] = row["sample_id"]
            s = (row.get("split") or "").strip().lower()
            if s in ("train", "val"):
                split_col[row["image"]] = s
    sid = lambda n: smap.get(n, n)

    if any(v == "val" for v in split_col.values()):
        split_of = lambda n: split_col.get(n, "train")          # explicit; unmarked -> train
    else:
        by_sample = defaultdict(list)
        for name, _ in items:
            by_sample[sid(name)].append(name)
        val_set = {s for s, _ in sorted(by_sample.items(), key=lambda kv: (len(kv[1]), kv[0]))[:2]}
        split_of = lambda n: "val" if sid(n) in val_set else "train"
    val_samples = sorted({sid(n) for n, _ in items if split_of(n) == "val"})
    if OUT.exists():
        shutil.rmtree(OUT)

    for name, lines in items:
        split = split_of(name)
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
    n_train = sum(1 for n, _ in items if split_of(n) == "train")
    print(f"{len(items)} images, {sum(len(l) for _, l in items)} boxes -> {OUT}")
    print(f"split: {n_train} train / {len(items) - n_train} val images | held-out val samples: {sorted(val_samples)}")
    print("class counts:", dict(hist))


if __name__ == "__main__":
    main()
