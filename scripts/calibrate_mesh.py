"""Mesh-as-ruler calibration — use the sieve's wire grid as a built-in scale.

The wire mesh is a regular periodic pattern; its pitch in pixels is recovered by FFT
(see corseacare.calib). Because "1 mm / 2 mm sieve" is the *aperture* but the detectable
period is the *pitch* (aperture + wire), we calibrate the true pitch ONCE per mesh size
against a ruler, then read px/mm straight off the mesh for every other photo.

Subcommands
-----------
  probe   IMG [--min 6 --max 90]      print detected period (px) + confidence, for tuning
  calibrate                           read data/mesh_refs.csv -> mesh_calibration.json
  apply   [--min-conf 3.0]            read calibration + samples.csv -> write px_per_mm col

Reference file  data/mesh_refs.csv   columns: image,sieve_mm,px_per_mm
  px_per_mm is measured by hand on a ruler-containing photo: open it, read the pixel
  distance between two marks N mm apart, px_per_mm = pixels / N.

    .venv/bin/python scripts/calibrate_mesh.py probe data/corseacare/IMG_8891.jpg
    .venv/bin/python scripts/calibrate_mesh.py calibrate
    .venv/bin/python scripts/calibrate_mesh.py apply
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from corseacare.calib import detect_mesh_period, solve_pitch_mm, px_per_mm_from_period

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
MANIFEST = ROOT / "samples.csv"
REFS = ROOT / "data" / "mesh_refs.csv"
CALIB = ROOT / "data" / "mesh_calibration.json"
FIELDS = ["image", "sample_id", "sieve_mm", "px_per_mm", "date", "location", "gps", "notes"]
MIN_PX, MAX_PX = 6.0, 90.0


def mesh_window(path, frac=0.5):
    """Grayscale central square crop inside the sieve, where the mesh is visible."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    cx, cy, r = w // 2, h // 2, int(0.42 * min(h, w))
    ds = 500.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * ds)), max(1, int(h * ds))))
    sh, sw = small.shape[:2]
    g = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=sh, param1=100,
                               param2=40, minRadius=int(0.22 * min(sh, sw)),
                               maxRadius=int(0.50 * min(sh, sw)))
    if circles is not None:
        cx, cy, r = (np.around(circles[0][0]).astype(float) / ds).astype(int)
    half = int(r * frac)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    crop = img[y0:cy + half, x0:cx + half]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _period(path):
    return detect_mesh_period(mesh_window(path), min_px=MIN_PX, max_px=MAX_PX)


def cmd_probe(args):
    p, conf = detect_mesh_period(mesh_window(args.image), min_px=args.min, max_px=args.max)
    print(f"{Path(args.image).name}: period={p:.1f} px  confidence={conf:.1f}"
          f"  {'(weak — mesh may be occluded/blurred)' if conf < 3 else ''}")


def cmd_calibrate(args):
    if not REFS.exists():
        raise SystemExit(f"missing {REFS} — create it with columns: image,sieve_mm,px_per_mm")
    by_sieve = defaultdict(list)
    for row in csv.DictReader(open(REFS)):
        img = IMAGES / row["image"]
        period, conf = _period(img)
        pitch = solve_pitch_mm(period, float(row["px_per_mm"]))
        by_sieve[str(row["sieve_mm"])].append((pitch, conf, row["image"]))
        print(f"  {row['image']}: sieve={row['sieve_mm']}mm period={period:.1f}px "
              f"conf={conf:.1f} -> pitch={pitch:.3f}mm")
    out = {}
    for sieve, recs in sorted(by_sieve.items()):
        pitches = [p for p, _, _ in recs]
        out[sieve] = {
            "pitch_mm": round(statistics.fmean(pitches), 4),
            "pitch_sd": round(statistics.pstdev(pitches), 4) if len(pitches) > 1 else 0.0,
            "n_refs": len(pitches),
            "aperture_mm": float(sieve),
        }
    CALIB.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CALIB}:")
    for s, v in out.items():
        wire = v["pitch_mm"] - v["aperture_mm"]
        print(f"  {s}mm sieve: pitch={v['pitch_mm']}mm (±{v['pitch_sd']}, n={v['n_refs']}) "
              f"=> implied wire ~{wire:.2f}mm")


def _load_manifest():
    rows = list(csv.DictReader(open(MANIFEST))) if MANIFEST.exists() else []
    for r in rows:
        for f in FIELDS:
            r.setdefault(f, "")
    return rows


def cmd_apply(args):
    if not CALIB.exists():
        raise SystemExit(f"missing {CALIB} — run `calibrate` first")
    pitch = {k: v["pitch_mm"] for k, v in json.loads(CALIB.read_text()).items()}
    rows = _load_manifest()
    done = weak = skipped = 0
    for r in rows:
        sieve = str(r.get("sieve_mm", "")).strip()
        if sieve not in pitch:
            skipped += 1
            continue
        img = IMAGES / r["image"]
        if not img.exists():
            skipped += 1
            continue
        period, conf = _period(img)
        if conf < args.min_conf or period <= 0:
            print(f"  WEAK {r['image']}: conf={conf:.1f} -> px_per_mm left blank")
            r["px_per_mm"] = ""
            weak += 1
            continue
        r["px_per_mm"] = round(px_per_mm_from_period(period, pitch[sieve]), 3)
        done += 1
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    print(f"px_per_mm written for {done} photos | {weak} weak/blank | {skipped} skipped "
          f"(no sieve_mm or calibration) -> {MANIFEST}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("image")
    p.add_argument("--min", type=float, default=MIN_PX); p.add_argument("--max", type=float, default=MAX_PX)
    p.set_defaults(func=cmd_probe)
    sub.add_parser("calibrate").set_defaults(func=cmd_calibrate)
    a = sub.add_parser("apply"); a.add_argument("--min-conf", type=float, default=3.0)
    a.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
