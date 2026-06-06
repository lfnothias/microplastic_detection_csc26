"""Build ONE clean Label Studio import for reviewing enriched annotations.

Per enriched image, the task carries TWO prediction layers so they are distinguishable in LS:
  - "existing_GT"        = the boxes you already annotated
  - "vision_added"       = the quasi-certain additions (conf >= THR) the model+vision found missed
Import into a FRESH project, review (focus on the 'vision_added' layer), accept, Submit, export.

    .venv/bin/python scripts/make_review_import.py [conf_threshold]
Output: data/enrich/ls_review.json
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "enrich"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
LS_BASE = "http://localhost:8081/corseacare/"
STEM2IMG_CSV = ROOT / "samples.csv"


def rect(x1, y1, x2, y2, cls, W, H):
    return {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": W, "original_height": H, "image_rotation": 0,
            "value": {"x": x1 / W * 100, "y": y1 / H * 100,
                      "width": (x2 - x1) / W * 100, "height": (y2 - y1) / H * 100,
                      "rotation": 0, "rectanglelabels": [cls]}}


def main(thr):
    idx = json.loads((D / "index.json").read_text())
    dims = json.loads((D / "dims.json").read_text())
    import csv
    stem2img = {Path(r["image"]).stem: r["image"] for r in csv.DictReader(open(STEM2IMG_CSV))}

    add = defaultdict(list)
    for ents in idx.values():
        for e in ents:
            if e["conf"] >= thr:
                add[e["image"]].append((e["xyxy"], e["pred_class"]))

    tasks = []
    for im in sorted(dims):
        stem = Path(im).stem
        W, H = dims[im]
        gtf = D / "gt" / f"{stem}.txt"
        gt_results = []
        for ln in (gtf.read_text().splitlines() if gtf.exists() else []):
            p = ln.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = int(p[0]), *(float(x) for x in p[1:])
            gt_results.append(rect((cx - w / 2) * W, (cy - h / 2) * H,
                                   (cx + w / 2) * W, (cy + h / 2) * H, CLASSES[c], W, H))
        add_results = [rect(x1, y1, x2, y2, cls, W, H) for (x1, y1, x2, y2), cls in add.get(im, [])]
        # single prediction layer = all boxes pre-drawn (your GT + the recovered missed particles)
        tasks.append({"data": {"image": LS_BASE + im},
                      "predictions": [{"model_version": "GT+ajouts",
                                       "result": gt_results + add_results}]})
    (D / "ls_review.json").write_text(json.dumps(tasks, indent=2))
    print(f"{len(tasks)} tâches -> {D / 'ls_review.json'}")
    for t in tasks:
        im = t["data"]["image"].rsplit("/", 1)[-1]
        ng = len(t["predictions"][0]["result"])
        na = len(t["predictions"][1]["result"])
        print(f"  {im}: {ng} GT + {na} ajouts à réviser")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.6)
