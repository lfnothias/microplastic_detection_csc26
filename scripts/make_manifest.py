"""Generate / update samples.csv — maps each photo in data/corseacare/ to a sample_id
(one physical sieve = one sample; multiple views share a sample_id).

Idempotent: preserves existing rows (your edits) and only adds NEW images with a best-guess
sample_id you then correct. Also the home for per-sample metadata (date, gps, location...).

    .venv/bin/python scripts/make_manifest.py
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
MANIFEST = ROOT / "samples.csv"
FIELDS = ["image", "sample_id", "sieve_mm", "date", "location", "gps", "notes"]


def guess(name):
    """Best-guess (sample_id, date). EDIT sample_id by hand afterwards — these are heuristics."""
    if name.upper().startswith("DT5A"):
        return "TAMIS_DSLR", "2026-06-05"
    if name.startswith("20260605_145"):
        return "TAMIS_A", "2026-06-05"          # 14:57 burst (Tamis 1, 3 views)
    if name.startswith("20260605_15"):
        return "TAMIS_B", "2026-06-05"          # 15:xx burst (swisstransfer set)
    if name.startswith("20260605_"):
        return "TAMIS_" + name.split("_")[1][:4], "2026-06-05"
    return "MANTA_" + Path(name).stem[:8], ""   # original UUID mission photos: one each


def main():
    existing = {}
    if MANIFEST.exists():
        for row in csv.DictReader(open(MANIFEST)):
            existing[row["image"]] = {k: row.get(k, "") for k in FIELDS}
    images = sorted(p.name for p in IMAGES.glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows, added = [], 0
    for name in images:
        if name in existing:
            rows.append(existing[name])
        else:
            sid, date = guess(name)
            rows.append({"image": name, "sample_id": sid, "date": date,
                         "location": "", "gps": "", "notes": ""})
            added += 1
    rows.sort(key=lambda r: (r["sample_id"], r["image"]))
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images ({added} new) -> {MANIFEST}")
    for sid, n in sorted(Counter(r["sample_id"] for r in rows).items()):
        print(f"  {sid}: {n} view(s)")


if __name__ == "__main__":
    main()
