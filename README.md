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

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/) (`brew install uv`), `git`, and
[`git-lfs`](https://git-lfs.com) (`brew install git-lfs`) for the dataset images.

```bash
git lfs install
git clone https://github.com/lfnothias/microplastic_detection_csc26.git
cd CorSeaCare_yolo && git lfs pull          # fetch the sieve images (Git LFS)
./scripts/setup.sh                          # installs the engine (no torch) + Label Studio
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

Put photos in `data/corseacare/`, build the manifest, and start the annotation stack:
```bash
uv run python scripts/make_manifest.py    # -> samples.csv (edit sample_id / sieve_mm / split; template: samples.csv.example)
make annotate                             # image server (:8081) + Label Studio (:8080) together
```
Set up the Label Studio project — **automatically** (recommended):
```bash
export LABEL_STUDIO_API_KEY=<token>       # Label Studio -> Account & Settings -> Access Token
make ls-project TASKS=data/corseacare_preann/ls_tasks.json   # project + 6-class config + import pre-annotations
```
…or manually: new project → Labeling Setup → paste `configs/label_studio_config.xml` → Import a tasks `.json`.
Then correct the boxes, **Submit**, and train. Full walkthrough:
**[docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)**.

Train and run the model. **To reproduce on the shipped dataset** (no Label Studio needed) start
with `build_yolo_dataset.py`; if you annotated your *own* photos, use `export_from_ls.py` instead:
```bash
uv run python scripts/build_yolo_dataset.py  # shipped images + annotations -> YOLO dataset (reproduce)
# (or from your own Label Studio project:)   uv run python scripts/export_from_ls.py
uv run python scripts/tile_dataset.py     # slice into tiles
uv run python scripts/train_tiles.py      # train YOLO11n on the tiles (MPS, early-stops)
uv run python scripts/predict_tiled.py    # tiled inference (sieve-ROI gated) -> overlays + counts

# sizes in mm: calibrate the mesh ruler ONCE (needs data/mesh_refs.csv), then it flows automatically
uv run python scripts/calibrate_mesh.py calibrate   # -> mesh_calibration.json (pitch per mesh)
uv run python scripts/calibrate_mesh.py apply       # -> px_per_mm column in samples.csv
uv run corseacare count data/corseacare --out counts.csv --tiled   # tiled detect+segment+measure (per-photo mm/px)
uv run corseacare report --particles counts.csv                    # per-sample summary (see Outputs)
```

## Outputs

`corseacare report` aggregates the per-particle CSV into a **per-sample** summary
(`report.json` + a flat `report_by_sample.csv`):

- **counts per class** (`fragment`/`fibre`/`film`/`mousse`/`pellet`/`autre`) and **per colour**;
- **size** — mean/median max-Feret (mm) and a **size-class histogram** (`<1`, `1–2`, `2–5`,
  `5–10`, `>10` mm, aligned with the 1 mm/2 mm sieve fractions);
- **projected area** (mm², total and per class) — the robust 2-D quantity;
- a **volume estimate** (mm³) clearly labelled as an estimate with its shape assumption — a 2-D
  photo cannot measure true volume;
- **concentration** (particles/m³) when a tow volume is supplied (`--tow-volume-m3`).

Multiple photos of one sieve are reshuffled views of the *same* material, so they are **not
summed** — the representative (median-count) view is summarised (override with `--sum-views`).
Size/colour require the segmentation pipeline (`corseacare count`); a detection-only CSV yields
class counts only. Polymer type is **not** inferred (needs FTIR/Raman).

## Vision-assisted (re-)annotation

Annotation is never complete — small particles get missed. Two model-assisted loops close the
gap (foundation models *propose*, you *review* in Label Studio):

- **Fresh pre-annotation:** `preannotate_corseacare.py` (candidate boxes) →
  `make_candidate_crops.py` (numbered montages) → a vision model classifies each crop →
  `ls_merge.py` writes class suggestions back into Label Studio.
- **Re-annotation of missed particles:** `extract_fp.py` takes a trained model's *false
  positives* on a held-out sieve (many are real un-boxed particles), crops them into montages;
  a vision model judges each (real particle + class, or noise); `make_reannot_import.py` emits a
  Label Studio import of the confirmed boxes to add to the annotation.

On `TAMIS_B` this recovered **109 missed particles (74 % of the model's "false positives")**,
revising the annotation 214 → 323 boxes and the model's real precision from 47 % to ~86 %.

- **Quasi-certain enrichment (reproducible):** `enrich_extract.py` + `enrich_consolidate.py`
  add only detector candidates with high confidence (conf ≥ 0.6) absent from the ground truth —
  objectively likely-real missed particles, dropping borderline. Across 4 sieves this added 109
  quasi-certain boxes (847 → 956). Writes enriched YOLO labels + a Label Studio import for review.

Evaluation tooling: `eval_binary.py` (plastic vs organic), `view_consistency.py` (inter-view
reproducibility). Full results: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Dataset & privacy

The **sieve photos and annotations are published** (CC0) so results are reproducible:
`data/corseacare/*.jpg,*.JPG` (via **Git LFS**), `annotations/*.txt`, and `samples.csv` — see
**[DATASET.md](DATASET.md)**. Rebuild the YOLO dataset from them with
`scripts/build_yolo_dataset.py` (no Label Studio needed).

**Working artifacts stay git-ignored**: model runs (`data/runs/`, `runs/`), RAW originals
(`data/raw_dng/`), tiles, montages and other derived files. To work on your own photos, drop
them in `data/corseacare/` and regenerate `samples.csv` with `make manifest` (template:
[`samples.csv.example`](samples.csv.example)).

## Repository layout

```
src/corseacare/   engine: types, config, mask, features, cluster, metrics, measure, viz, pipeline/
scripts/          pipeline tools (pre-annotate, tile, train-from-LS, predict, cluster, manifest…)
app/server.py     Streamlit app (single image + batch, overlays, CSV export)
configs/          class list + Label Studio labeling config
docs/             ANNOTATION_GUIDE.md and notes
tests/            unit tests (run with `uv run pytest`)
samples.csv.example  template for the photo -> sample_id manifest (your samples.csv is git-ignored)
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

**Code: AGPL-3.0** (see [LICENSE](LICENSE)). **Dataset (images + annotations): CC0** — see
[DATASET.md](DATASET.md). Built on: Ultralytics YOLO (AGPL-3.0), SAM 2
(Apache-2.0), DINOv2 (Apache-2.0), Label Studio (Apache-2.0), OpenCV, scikit-image,
scikit-learn. See [NOTICE.md](NOTICE.md). Part of the CorSeaCare mission (Mare Vivu; partners
CNRS / Ifremer / Sorbonne Université).

## Citation

See [CITATION.cff](CITATION.cff).
