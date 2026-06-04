from pathlib import Path
import cv2
import pandas as pd
import typer

from corseacare.config import Config, load_config
from corseacare.runner import build_pipeline

app = typer.Typer(help="CorSeaCare: count marine plastic particles in RGB photos.")


def _cfg(config: str, mm_per_px: float) -> Config:
    cfg = load_config(config) if config else Config()
    cfg.mm_per_px = mm_per_px
    return cfg


@app.command()
def predict(image: str, out: str = "overlay.png", config: str = "", mm_per_px: float = 0.1):
    """Run on a single image, save an overlay and print the count."""
    from corseacare.viz import draw_overlay
    cfg = _cfg(config, mm_per_px)
    pipe = build_pipeline(cfg, mm_per_px)
    img = cv2.imread(image)
    r = pipe.run(img)
    cv2.imwrite(out, draw_overlay(img, r["detections"], r["masks"]))
    typer.echo(f"count={r['count']} overlay={out}")


@app.command()
def count(folder: str, out: str = "counts.csv", config: str = "", mm_per_px: float = 0.1):
    """Batch-process a folder, write per-particle CSV with sample metadata columns."""
    cfg = _cfg(config, mm_per_px)
    pipe = build_pipeline(cfg, mm_per_px)
    rows = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = cv2.imread(str(p))
        r = pipe.run(img)
        for rec in r["records"]:
            rec["image"] = p.name
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out, index=False)
    typer.echo(f"wrote {len(rows)} particles to {out}")


if __name__ == "__main__":
    app()
