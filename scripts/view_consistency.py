"""Test B — inter-view consistency.

Different photos of the SAME sieve are the same material reshuffled, so a reliable model should
predict ~the same particle count and the same per-class ratios across views. This reads a
per-detection CSV (image,class,...) such as data/corseacare_pred_tiled/counts.csv, groups the
views of one sample_id (from samples.csv), and reports per-view totals/ratios plus the dispersion
(coefficient of variation, CV) — no annotations needed. Low CV = consistent/repeatable.

    .venv/bin/python scripts/view_consistency.py TAMIS_B [counts.csv]
"""
import sys
import csv
import statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cv(xs):
    m = statistics.fmean(xs) if xs else 0.0
    return (statistics.pstdev(xs) / m * 100) if m else 0.0


def main(sample_id, counts_csv):
    smap = {r["image"]: r["sample_id"] for r in csv.DictReader(open(ROOT / "samples.csv"))}
    per_img = defaultdict(Counter)
    for r in csv.DictReader(open(counts_csv)):
        if smap.get(r["image"]) == sample_id:
            per_img[r["image"]][r["class"]] += 1
    if not per_img:
        print(f"no predictions for {sample_id} in {counts_csv}")
        return
    images = sorted(per_img)
    classes = sorted({c for cnt in per_img.values() for c in cnt})
    totals = [sum(per_img[im].values()) for im in images]

    print(f"=== {sample_id}: {len(images)} vues (même matériel, re-mélangé) ===")
    print(f"{'view':24} {'total':>6}  " + " ".join(f"{c[:6]:>7}" for c in classes))
    for im in images:
        t = sum(per_img[im].values())
        props = " ".join(f"{(per_img[im][c] / t * 100 if t else 0):6.1f}%" for c in classes)
        print(f"{im:24} {t:6d}  {props}")

    print(f"\nNOMBRE de particules : moyenne={statistics.fmean(totals):.1f} "
          f"sd={statistics.pstdev(totals):.1f}  CV={_cv(totals):.0f}%")
    print("RATIO par classe (proportion %) sur les vues :")
    for c in classes:
        props = [(per_img[im][c] / sum(per_img[im].values()) * 100 if sum(per_img[im].values()) else 0)
                 for im in images]
        print(f"  {c:10} moyenne={statistics.fmean(props):5.1f}%  CV={_cv(props):4.0f}%")
    print("\n(CV bas = cohérent entre vues = comptage répétable ; CV élevé = peu fiable.)")


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "TAMIS_B"
    cc = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "corseacare_pred_tiled" / "counts.csv")
    main(sid, cc)
