"""Confidence-split active-learning pre-annotation on UNANNOTATED sieve photos.

Reads a model's tiled detections (predict_tiled counts.csv) for photos that have NO annotation,
and splits each predicted box by confidence:
  - SURE       (conf >= --high, default 0.6): auto-kept as YOLO pseudo-labels (data/active/sure/).
  - UNCERTAIN  (--low <= conf < --high)      : sent to Label Studio WITH its pseudo-label, for an
                                               expert to verify / correct / complete.
  - dropped    (conf < --low).
Also writes per-image overlays (sure = green, uncertain = orange) and a summary.

    .venv/bin/python scripts/confidence_split.py [counts.csv] [--high 0.6] [--low 0.25]
Output: data/active/{sure/<stem>.txt, ls_review.json, overlays/<img>, summary.csv, hist.txt}
"""
import sys
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
ANN = ROOT / "annotations"
OUT = ROOT / "data" / "active"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
LS_BASE = "http://localhost:8081/corseacare/"
SURE_C, UNC_C = (0, 200, 0), (0, 140, 255)   # green / orange (BGR)


def main(counts, high, low):
    annotated = {p.stem for p in ANN.glob("*.txt")}
    dets = defaultdict(list)
    confs = []
    for r in csv.DictReader(open(counts)):
        if Path(r["image"]).stem in annotated:   # only UNannotated photos
            continue
        c = float(r["conf"])
        confs.append(c)
        if c < low:
            continue
        dets[r["image"]].append((r["class"], c, [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]))
    if not dets:
        print("Aucune photo non annotée trouvée dans counts.csv (toutes ont déjà un label ?).")
        return

    for sub in ("sure", "overlays"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    tasks, summ = [], []
    tot_sure = tot_unc = 0
    for im in sorted(dets):
        img = cv2.imread(str(IMAGES / im))
        H, W = img.shape[:2]
        stem = Path(im).stem
        sure = [(cls, b) for cls, c, b in dets[im] if c >= high]
        unc = [(cls, c, b) for cls, c, b in dets[im] if c < high]
        tot_sure += len(sure); tot_unc += len(unc)

        # SURE -> YOLO pseudo-labels, kept aside
        lines = []
        for cls, (x1, y1, x2, y2) in sure:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (OUT / "sure" / f"{stem}.txt").write_text("\n".join(lines))

        # UNCERTAIN -> Label Studio review with pseudo-label
        res = [{"type": "rectanglelabels", "from_name": "label", "to_name": "image",
                "original_width": W, "original_height": H, "image_rotation": 0,
                "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                          "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [cls]}}
               for cls, c, (x1, y1, x2, y2) in unc]
        tasks.append({"data": {"image": LS_BASE + im},
                      "predictions": [{"model_version": "uncertain-pseudolabel", "result": res}]})

        # overlay: sure green, uncertain orange
        ov = img.copy()
        for cls, (x1, y1, x2, y2) in sure:
            cv2.rectangle(ov, (x1, y1), (x2, y2), SURE_C, 3)
        for cls, c, (x1, y1, x2, y2) in unc:
            cv2.rectangle(ov, (x1, y1), (x2, y2), UNC_C, 3)
        s = 1400 / max(H, W)
        cv2.imwrite(str(OUT / "overlays" / im), cv2.resize(ov, (int(W * s), int(H * s))))

        sc = Counter(cls for cls, _ in sure); uc = Counter(cls for cls, _, _ in unc)
        summ.append((im, len(sure), len(unc), dict(sc), dict(uc)))

    (OUT / "ls_review.json").write_text(json.dumps(tasks, indent=2))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "n_sure", "n_uncertain", "sure_by_class", "uncertain_by_class"])
        for im, ns, nu, sc, uc in summ:
            w.writerow([im, ns, nu, sc, uc])

    confs.sort()
    bins = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist = {f"[{a:.2f},{b:.2f})": sum(1 for x in confs if a <= x < b) for a, b in zip(bins, bins[1:])}
    (OUT / "hist.txt").write_text(json.dumps(hist, indent=2))
    print(f"{len(dets)} photos non annotées | conf>=high({high}) -> {tot_sure} SÛRES (gardées de côté) | "
          f"low({low})<=conf<high -> {tot_unc} INCERTAINES (à réviser dans Label Studio)")
    print("histogramme de confiance:", hist)
    print(f"-> data/active/ : sure/ (pseudo-labels auto), ls_review.json (révision), overlays/, summary.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    high = float(args[args.index("--high") + 1]) if "--high" in args else 0.6
    low = float(args[args.index("--low") + 1]) if "--low" in args else 0.25
    pos = [a for a in args if not a.startswith("--") and a not in (str(high), str(low))]
    counts = pos[0] if pos else str(ROOT / "data" / "corseacare_pred_tiled" / "counts.csv")
    main(counts, high, low)
