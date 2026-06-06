"""Part 2 — can v6/v7 recover the right count + class ratios on the UNANNOTATED views of a sieve?

For each multi-view sieve, the annotated view's ground truth is the reference; the other views
are the same material physically re-distributed (water-mixed). A reliable model should predict a
similar particle count and similar per-class / plastic-vs-organic ratios on those unannotated
views. Reads predict_tiled counts CSVs for v6 and v7.

    .venv/bin/python scripts/results_consistency.py /tmp/counts_v6.csv /tmp/counts_v7.csv
"""
import sys
import csv
import statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fragment", "fibre", "film", "mousse", "pellet", "autre"]
PLASTIC = {"fragment", "fibre", "film", "mousse", "pellet"}
ANN = ROOT / "annotations"
TAMIS = ["TAMIS_DSLR", "TAMIS_B", "TAMIS_C"]


def stem(name):
    return Path(name).stem


def load_counts(path):
    per = defaultdict(Counter)
    for r in csv.DictReader(open(path)):
        per[r["image"]][r["class"]] += 1
    return per


def gt_counts(name):
    f = ANN / f"{stem(name)}.txt"
    c = Counter()
    if f.exists():
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                c[CLASSES[int(p[0])]] += 1
    return c


def ratios(counter):
    t = sum(counter.values())
    r5 = {k: round(counter[k] / t * 100) for k in CLASSES if counter[k]} if t else {}
    plast = sum(counter[k] for k in PLASTIC)
    pa = (round(plast / t * 100), round(counter["autre"] / t * 100)) if t else (0, 0)
    return t, r5, pa


def cv(xs):
    m = statistics.fmean(xs) if xs else 0
    return round(statistics.pstdev(xs) / m * 100) if m else 0


def main(v6_csv, v7_csv):
    s2imgs = defaultdict(list)
    annotated = {stem(p.name) for p in ANN.glob("*.txt")}
    for r in csv.DictReader(open(ROOT / "samples.csv")):
        s2imgs[r["sample_id"]].append(r["image"])
    models = {"v6": load_counts(v6_csv), "v7": load_counts(v7_csv)}

    for sid in TAMIS:
        imgs = sorted(s2imgs.get(sid, []))
        ann = [i for i in imgs if stem(i) in annotated]
        unann = [i for i in imgs if stem(i) not in annotated]
        if not ann or not unann:
            continue
        gt = Counter()
        gt_view_totals = []
        for a in ann:
            gca = gt_counts(a)
            gt += gca
            gt_view_totals.append(sum(gca.values()))
        _, gt5, gtpa = ratios(gt)
        gt_pv = statistics.fmean(gt_view_totals)
        print(f"\n### {sid}  ({len(unann)} vues non annotées ; référence = vue(s) annotée(s) {[stem(a) for a in ann]})")
        print(f"- **GT (par vue)** : ~{gt_pv:.0f} particules | 5-classes {gt5} | plastique/autre {gtpa[0]}%/{gtpa[1]}%")
        for mname, per in models.items():
            totals = [sum(per[i].values()) for i in unann if i in per]
            agg = Counter()
            for i in unann:
                agg += per.get(i, Counter())
            _, r5, pa = ratios(agg)
            if totals:
                print(f"- **{mname}** (vues non annotées) : compte moyen {statistics.fmean(totals):.0f} "
                      f"(min {min(totals)}, max {max(totals)}, CV {cv(totals)}%) | "
                      f"5-classes {r5} | plastique/autre {pa[0]}%/{pa[1]}%")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "/tmp/counts_v6.csv"
    b = sys.argv[2] if len(sys.argv) > 2 else "/tmp/counts_v7.csv"
    main(a, b)
