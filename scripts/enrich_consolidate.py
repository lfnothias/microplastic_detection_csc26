"""Enrichment step 2 — consolidate quasi-certain candidates into enriched annotations.

Quasi-certain rule (reproducible): keep detector candidates with conf >= THRESHOLD (default 0.6)
that don't match GT — high model confidence + absent from GT = very likely a real MISSED
particle. Borderline (lower conf) candidates are dropped. Writes enriched YOLO labels
(GT + additions) and a Label Studio import for human review, and reports per-image count/ratio
before vs after.

    .venv/bin/python scripts/enrich_consolidate.py [conf_threshold]
Output: data/enrich/{labels_enriched/<stem>.txt, ls_enrich.json}
"""
import sys
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "enrich"
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
LS_BASE = "http://localhost:8081/corseacare/"


def main(thr):
    idx = json.loads((D / "index.json").read_text())
    dims = json.loads((D / "dims.json").read_text())
    stem2img = {Path(r["image"]).stem: r["image"] for r in csv.DictReader(open(ROOT / "samples.csv"))}

    add = defaultdict(list)        # image -> [(xyxy, cls)]
    for ents in idx.values():
        for e in ents:
            if e["conf"] >= thr:
                add[e["image"]].append((e["xyxy"], e["pred_class"]))
    (D / "labels_enriched").mkdir(exist_ok=True)

    tasks = []
    tot_gt = tot_add = 0
    print(f"=== Consolidation enrichie (règle quasi-certain : conf >= {thr}) ===")
    for im in sorted(dims):
        stem = Path(im).stem
        W, H = dims[im]
        gtf = D / "gt" / f"{stem}.txt"
        gt_lines = gtf.read_text().splitlines() if gtf.exists() else []
        gt = Counter(CLASSES[int(ln.split()[0])] for ln in gt_lines if len(ln.split()) == 5)

        new_lines = list(gt_lines)
        results = []
        for (x1, y1, x2, y2), cls in add.get(im, []):
            cid = CLASSES.index(cls)
            cx, cy = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            new_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            results.append({"type": "rectanglelabels", "from_name": "label", "to_name": "image",
                            "original_width": W, "original_height": H, "image_rotation": 0,
                            "value": {"x": x1 / W * 100, "y": y1 / H * 100,
                                      "width": (x2 - x1) / W * 100, "height": (y2 - y1) / H * 100,
                                      "rotation": 0, "rectanglelabels": [cls]}})
        (D / "labels_enriched" / f"{stem}.txt").write_text("\n".join(new_lines))
        tasks.append({"data": {"image": LS_BASE + im},
                      "predictions": [{"model_version": f"enrich-conf{thr}", "result": results}]})

        addc = Counter(c for _, c in add.get(im, []))
        merged = gt + addc
        ng, na = sum(gt.values()), sum(addc.values())
        tot_gt += ng
        tot_add += na
        plast = sum(merged[c] for c in CLASSES if c != "autre")
        org = merged["autre"]
        tn = ng + na
        print(f"\n  {im}: {ng} -> {tn}  (+{na} quasi-certaines)")
        print(f"     ajouts par classe : {dict(addc)}")
        print(f"     révisé : PLASTIQUE={plast} ({plast / tn * 100:.0f}%)  ORGANIQUE={org} ({org / tn * 100:.0f}%)")
    (D / "ls_enrich.json").write_text(json.dumps(tasks, indent=2))
    print(f"\nTOTAL : {tot_gt} -> {tot_gt + tot_add}  (+{tot_add} ajouts quasi-certains)")
    print(f"-> labels enrichis dans {D / 'labels_enriched'}")
    print(f"-> import Label Studio (révision) : {D / 'ls_enrich.json'}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.6)
