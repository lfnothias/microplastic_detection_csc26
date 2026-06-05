"""Merge vision-model verdicts on FP crops into a Label Studio re-annotation import.

Reads <dir>/index.json (montage -> [{id,image,xyxy,...}]) and <dir>/verdicts.json
(montage -> [{id,verdict,class}]). For every crop judged 'real', emits a Label Studio rectangle
pre-annotation (box + vision class) so the expert reviews/accepts the recovered particles. Also
prints the FP-recovery stats and the REVISED per-image box count + class ratio (GT + recovered).

    .venv/bin/python scripts/make_reannot_import.py [dir]     (default data/reannot_B)
Output: <dir>/ls_reannot.json
"""
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
GT_DIR = ROOT / "data" / "ls_export_yolo" / "labels" / "val"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
LS_BASE = "http://localhost:8081/corseacare/"


def main(d):
    d = Path(d)
    index = json.loads((d / "index.json").read_text())
    verdicts = json.loads((d / "verdicts.json").read_text())
    rec = defaultdict(list)          # image -> [(xyxy, cls)]
    n_real = n_noise = 0
    by_cls = Counter()
    for mf, ents in index.items():
        vmap = {v["id"]: v for v in verdicts.get(mf, [])}
        for e in ents:
            v = vmap.get(e["id"])
            if not v:
                continue
            if v["verdict"] == "real":
                rec[e["image"]].append((e["xyxy"], v["class"]))
                n_real += 1
                by_cls[v["class"]] += 1
            else:
                n_noise += 1

    tasks = []
    for im, boxes in rec.items():
        img = cv2.imread(str(IMAGES / im))
        H, W = img.shape[:2]
        results = []
        for (x1, y1, x2, y2), cls in boxes:
            results.append({"type": "rectanglelabels", "from_name": "label", "to_name": "image",
                            "original_width": W, "original_height": H, "image_rotation": 0,
                            "value": {"x": x1 / W * 100, "y": y1 / H * 100,
                                      "width": (x2 - x1) / W * 100, "height": (y2 - y1) / H * 100,
                                      "rotation": 0, "rectanglelabels": [cls]}})
        tasks.append({"data": {"image": LS_BASE + im},
                      "predictions": [{"model_version": "vision-reannot", "result": results}]})
    (d / "ls_reannot.json").write_text(json.dumps(tasks, indent=2))

    print(f"FP jugés : {n_real} VRAIES particules récupérées, {n_noise} bruit "
          f"({n_real + n_noise} total, {n_real / (n_real + n_noise) * 100:.0f}% réelles)")
    print("Classes récupérées :", dict(by_cls))
    print("\nStats RÉVISÉES par image (GT annoté + récupéré) :")
    for im, boxes in rec.items():
        gt = Counter()
        gf = GT_DIR / f"{Path(im).stem}.txt"
        if gf.exists():
            for ln in gf.read_text().splitlines():
                p = ln.split()
                if len(p) == 5:
                    gt[CLASSES[int(p[0])]] += 1
        addc = Counter(c for _, c in boxes)
        merged = gt + addc
        tot_gt, tot_new = sum(gt.values()), sum((gt + addc).values())
        plast = sum(merged[c] for c in CLASSES if c != "autre")
        org = merged["autre"]
        print(f"  {im}: GT={tot_gt} -> révisé={tot_new}  (+{sum(addc.values())} récupérées)")
        print(f"     classes : " + ", ".join(f"{c}={merged[c]}" for c in CLASSES if merged[c]))
        print(f"     PLASTIQUE={plast} ({plast / tot_new * 100:.0f}%)  "
              f"ORGANIQUE={org} ({org / tot_new * 100:.0f}%)")
    print(f"\n-> {d / 'ls_reannot.json'}  (importer dans Label Studio pour révision)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "reannot_B"))
