# CorSeaCare_yolo

**Open tool to detect, count, and characterise marine plastic particles** in photographs of
Manta-net sieve samples — built for the **CorSeaCare** citizen-science expedition
(association *Mare Vivu*, around Corsica).

Given a photo of a sieve, it detects each particle, classifies it into **5 plastic
morphotypes** (`fragment`, `fibre`, `film`, `mousse`, `pellet`) plus a loose **`autre`**
(organic / indeterminate), **counts** them, and measures **size** (mm, via a scale reference)
and **colour**. It runs **locally on a Mac** (Apple Silicon / MPS) and is built around a
**human-in-the-loop** annotation workflow so volunteers can grow the training set.

> **Status — research proof-of-concept.** The full pipeline and tooling work end-to-end
> (annotate → train → detect → count). Detection *accuracy* is currently limited by the small
> number of annotated sieves; it improves as more sieves are annotated. We report this
> honestly — see [Limitations](#limitations).

---

## How it works

- **Tier 1 — supervised detection (YOLO11):** one model detects + classifies + counts every
  particle (the 5 plastic classes + `autre`), with per-particle size and colour.
- **Tier 2 — discovery clustering (optional):** the `autre` residual is grouped into proposed
  clusters (masked colour/shape/size + DINOv2 features) for optional expert curation.
- **Tiling (SAHI):** sieve photos are huge (12 MP) and particles are tiny (~1–2 % of the
  frame), so images are sliced into overlapping tiles for both training and inference.
- **Assisted annotation:** a saturation-based pre-annotator proposes candidate boxes in
  Label Studio so you correct rather than draw from scratch; the trained model then
  pre-annotates the next batch (active-learning loop).

## Install (macOS, Apple Silicon)

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/) (`brew install uv`), `git`.

```bash
git clone https://github.com/<your-org>/CorSeaCare_yolo.git
cd CorSeaCare_yolo
./scripts/setup.sh          # installs the engine (no torch) + Label Studio
```
`setup.sh` runs `uv sync --extra dev` (provisions Python 3.12, installs the light engine) and
`uv tool install label-studio`. To **run/train the real models** (pulls in PyTorch, ~GB — do
this on a good connection):
```bash
uv pip install "ultralytics>=8.3" "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

## Quick try — no data, no models needed

```bash
uv run pytest                                   # the engine test-suite
# generate a synthetic dataset and run the whole pipeline end-to-end:
uv run python scripts/make_test_dataset.py
CORSEACARE_FAKE=1 uv run corseacare count data/test_run/images --out counts.csv --mm-per-px 0.1
```
`CORSEACARE_FAKE=1` swaps in deterministic stub detectors so you can exercise the CLI/app
without downloading any model.

## Annotate your own sieve photos

Put photos in `data/corseacare/`, then (two terminals):
```bash
uv run python scripts/serve_images.py     # image server  (http://localhost:8081)
./scripts/launch_label_studio.sh          # Label Studio   (http://localhost:8080)
```
Pre-annotate, import into Label Studio, correct, **Submit**, then train on your Mac. Full
walkthrough: **[docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)**.

Train from your annotations and run the model:
```bash
uv run python scripts/export_from_ls.py   # Label Studio -> YOLO dataset (split by sample)
uv run python scripts/tile_dataset.py     # slice into tiles
# ... train on the tiles (see ANNOTATION_GUIDE), then:
uv run python scripts/predict_tiled.py    # tiled inference -> overlays + counts
```

## Data & privacy

Your **photos, annotations, weights and datasets live under `data/` (and `runs/`) and are
git-ignored** — they are *not* part of the public repository. Each user supplies their own
images. `samples.csv` (which photo belongs to which physical sieve) is the only data-adjacent
file that is tracked, and it contains filenames + sample IDs only.

## Repository layout

```
src/corseacare/   engine: types, config, mask, features, cluster, metrics, measure, viz, pipeline/
scripts/          pipeline tools (pre-annotate, tile, train-from-LS, predict, cluster, manifest…)
app/server.py     Streamlit app (single image + batch, overlays, CSV export)
configs/          class list + Label Studio labeling config
docs/             ANNOTATION_GUIDE.md and notes
tests/            unit tests (run with `uv run pytest`)
samples.csv       photo -> sample_id (sieve) manifest
```

## Limitations

- **Accuracy scales with annotated sieves.** With only a handful of distinct sieves, the model
  fits them but generalises poorly to unseen sieves (honest, leakage-free val). More annotated
  *distinct* samples is the main lever.
- **Particles are tiny** in 12 MP photos → tiling is required; very dense sieves are hard.
- **Polymer type is not inferred** from RGB (needs FTIR/Raman) — only morphotype/colour/size.
- Sizes are reported in **mm** via the sieve **mesh as a built-in ruler**
  (`scripts/calibrate_mesh.py`): the wire grid is a known periodic scale, so px→mm is
  auto-derived per photo after a one-time pitch calibration against a physical ruler. Where
  the mesh is too occluded to detect, that photo's scale is left blank rather than guessed.

## Contributing

Citizen-science contributions (photos, annotations, code) welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License & credits

**AGPL-3.0** (see [LICENSE](LICENSE)). Built on: Ultralytics YOLO (AGPL-3.0), SAM 2
(Apache-2.0), DINOv2 (Apache-2.0), Label Studio (Apache-2.0), OpenCV, scikit-image,
scikit-learn. See [NOTICE.md](NOTICE.md). Part of the CorSeaCare mission (Mare Vivu; partners
CNRS / Ifremer / Sorbonne Université).

## Citation

See [CITATION.cff](CITATION.cff).
