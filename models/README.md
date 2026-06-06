# Pretrained weights (Git LFS)

YOLO11n detectors (6 classes: `fragment fibre film mousse pellet autre`), tiled inference.
AGPL-3.0 (Ultralytics-derived). Fetch with `git lfs pull`.

| file | trained on | held-out | use |
|------|-----------|----------|-----|
| **`corseacare_tiles_v7.pt`** | MANTA + TAMIS_DSLR + TAMIS_C (enriched annotations) | TAMIS_B | **recommended** |
| `corseacare_tiles_v6.pt` | same sieves, original annotations | TAMIS_B | for the v6-vs-v7 comparison in [docs/RESULTS_v2.md](../docs/RESULTS_v2.md) |

## Inference (no training needed)
```bash
git lfs pull   # if not already
# tiled detection -> overlays + counts.csv  (recall-oriented: conf 0.12)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/predict_tiled.py models/corseacare_tiles_v7.pt 0.12
# active-learning split (sure vs borderline) for new unannotated photos:
uv run python scripts/confidence_split.py data/corseacare_pred_tiled/counts.csv --high 0.6 --low 0.12
```
Default detection geometry (baked in `Config` / `predict_tiled`): tile 640, overlap 0.5,
ROI margin 1.0, max box 4 %. See [docs/RESULTS_v2.md](../docs/RESULTS_v2.md) for performance.

> Note: held-out performance is modest (mAP@50 ≈ 0.37 on a typical sieve) and recall is the
> limiting factor — these weights are a research proof-of-concept, not a calibrated instrument.
