import json
from pathlib import Path
import cv2
import pandas as pd
import typer

from corseacare.config import Config, load_config
from corseacare.runner import build_pipeline
from corseacare.report import summarize

app = typer.Typer(help="CorSeaCare: count marine plastic particles in RGB photos.")


def _cfg(config: str, mm_per_px: float) -> Config:
    cfg = load_config(config) if config else Config()
    cfg.mm_per_px = mm_per_px
    return cfg


@app.command()
def predict(image: str, out: str = "overlay.png", config: str = "", mm_per_px: float = 0.1,
            tiled: bool = False):
    """Run on a single image, save an overlay and print the count."""
    from corseacare.viz import draw_overlay
    cfg = _cfg(config, mm_per_px)
    pipe = build_pipeline(cfg, mm_per_px, tiled=tiled)
    img = cv2.imread(image)
    r = pipe.run(img)
    cv2.imwrite(out, draw_overlay(img, r["detections"], r["masks"]))
    typer.echo(f"count={r['count']} overlay={out}")


@app.command()
def count(folder: str, out: str = "counts.csv", config: str = "", mm_per_px: float = 0.1,
          manifest: str = "samples.csv", tiled: bool = False):
    """Batch-process a folder, write per-particle CSV with sample metadata columns.

    Per-photo scale: if `manifest` has a calibrated `px_per_mm` for an image (from
    `calibrate_mesh.py`), that photo is measured at its own mm/px; otherwise the global
    `--mm-per-px` is used. Calibrated photos are reported in mm; the rest fall back.
    """
    from corseacare.calib import load_mm_per_px_map
    cfg = _cfg(config, mm_per_px)
    pipe = build_pipeline(cfg, mm_per_px, tiled=tiled)
    scale_map = load_mm_per_px_map(manifest)
    rows, n_cal = [], 0
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = cv2.imread(str(p))
        per_img = scale_map.get(p.name)
        if per_img is not None:
            n_cal += 1
        r = pipe.run(img, mm_per_px=per_img)        # None -> pipeline's global scale
        for rec in r["records"]:
            rec["image"] = p.name
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out, index=False)
    typer.echo(f"wrote {len(rows)} particles to {out} "
               f"({n_cal} photos used mesh-calibrated mm/px, rest used --mm-per-px {mm_per_px})")


@app.command()
def report(particles: str = "counts.csv", manifest: str = "samples.csv",
           out_json: str = "report.json", out_csv: str = "report_by_sample.csv",
           tow_volume_m3: float = 0.0, volume_model: str = "area_power",
           sum_views: bool = False):
    """Aggregate a per-particle CSV into per-sample summaries (counts, sizes, area, volume est.).

    Multiple photos of one sieve are reshuffled views of the SAME material, so by default they
    are NOT summed: the representative (median-count) view is summarised. Pass --sum-views only
    if each image is genuinely distinct material.
    """
    df = pd.read_csv(particles)
    smap = {}
    if Path(manifest).exists():
        m = pd.read_csv(manifest)
        smap = dict(zip(m["image"], m["sample_id"]))
    df["sample_id"] = df["image"].map(lambda im: smap.get(im, im))

    summaries = {}
    for sid, grp in df.groupby("sample_id"):
        images = list(grp["image"].unique())
        if len(images) > 1 and not sum_views:
            counts = grp.groupby("image").size().sort_values()
            rep = counts.index[len(counts) // 2]           # median-count view
            recs = grp[grp["image"] == rep].to_dict("records")
            note = f"{len(images)} views; representative='{rep}' (median count); not summed"
        else:
            recs = grp.to_dict("records")
            note = f"{len(images)} view(s); summed" if sum_views else "single view"
        s = summarize(recs, tow_volume_m3=tow_volume_m3 or None, volume_model=volume_model)
        s["sample_id"] = sid
        s["note"] = note
        summaries[sid] = s

    Path(out_json).write_text(json.dumps(summaries, indent=2))
    flat = []
    for sid, s in summaries.items():
        row = {"sample_id": sid, "n_particles": s["n_particles"],
               "size_mean_mm": round(s["size_mm"]["mean"], 3),
               "size_median_mm": round(s["size_mm"]["median"], 3),
               "area_total_mm2": round(s["area_mm2"]["total"], 2),
               "volume_est_mm3": round(s["volume_estimate"]["total_mm3"], 2)}
        for cls, n in s["counts_by_class"].items():
            row[f"n_{cls}"] = n
        if "concentration_per_m3" in s:
            row["conc_per_m3"] = round(s["concentration_per_m3"], 2)
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_csv, index=False)
    typer.echo(f"{len(summaries)} samples -> {out_json}, {out_csv}")


if __name__ == "__main__":
    app()
