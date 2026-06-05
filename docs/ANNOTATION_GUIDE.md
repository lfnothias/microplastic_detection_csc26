# Annotating the real CorSeaCare photos in Label Studio

15 Manta-net photos live in `data/corseacare/` (gitignored). A saturation-based
pre-annotator (`scripts/preannotate_corseacare.py`) proposes candidate boxes around the
coloured particles so you correct rather than draw from scratch. Its outputs are under
`data/corseacare_preann/` (gitignored): `overlays/` (previews), `ls_tasks.json` (Label
Studio import with predictions), and `yolo/` (a rough draft YOLO label set).

> The candidate boxes are *starting points*, all tentatively labelled `fragment`. Per box:
> **fix the class** (fragment / fibre / film / mousse / pellet / `matiere_organique`), adjust
> the box, **add** missed grey/white/transparent particles (colour-based detection can't see
> them), and **delete** false boxes (mesh texture, reflections; or relabel green algae/wood as
> `matiere_organique`).

## 1. Start two servers (two terminals)

**Terminal A — image server** (serves the photos to Label Studio over HTTP+CORS; reliable):
```bash
cd ~/git/CorSeaCare_yolo
.venv/bin/python scripts/serve_images.py     # http://localhost:8081  (keep it open)
```

**Terminal B — Label Studio:**
```bash
cd ~/git/CorSeaCare_yolo
./scripts/launch_label_studio.sh             # http://localhost:8080
```
Opens http://localhost:8080 (create a local account on first run — stays on your machine).

## 2. Create the project

1. **Create Project** → name `corseacare`.
2. **Labeling Setup → Custom template** → paste `configs/label_studio_config.xml` → **Save**.

## 3. Import photos + pre-annotations

- **Import** → upload `data/corseacare_preann/ls_tasks.json`.
- Tasks reference images via `http://localhost:8081/corseacare/<name>` and carry the candidate
  boxes as **predictions** — they appear pre-drawn.
- **If images don't load:** make sure Terminal A (`serve_images.py`) is running. Test it:
  `curl -I http://localhost:8081/corseacare/<one-file>.JPG` should return `200`.

## 4. Tips

- **Scale:** these 15 photos have no ruler, so sizes are in pixels until calibrated. Add a
  scale reference to future photos to get mm.
- **Colour is automatic** — don't encode it; the pipeline computes it from the mask.
- Keep `matiere_organique` on green filaments / wood — it teaches the model to reject organics.
- Start with the dense colourful sieves (best return on effort).

## 5. Export → train on the M4

1. **Export → YOLO** → `images/` + `labels/` + `classes.txt`.
2. Point `scripts/train_local.py` `DATA` at a `data.yaml` (6 classes) for that export, then
   `.venv/bin/python scripts/train_local.py` (fine-tunes YOLO11n on MPS).
3. Run the Streamlit app / CLI with the trained weights on new photos.

## 6. Making the candidate boxes better (optional)

Two pre-annotators exist; they're **complementary** (tested on these photos):

- **Saturation** (`scripts/preannotate_corseacare.py`, the default `ls_tasks.json`) — great for
  the **coloured** plastic fragments (the majority); misses grey/white/transparent ones.
- **SAM2 automatic** (`scripts/preannotate_sam2.py` → `data/corseacare_preann_sam2/`) — catches
  **large and grey/translucent** fragments the colour method misses, but misses small dense
  particles and is slow on the mesh. Run e.g.:
  `PYTORCH_ENABLE_MPS_FALLBACK=1 CORSEACARE_SAM2_POINTS=24 .venv/bin/python scripts/preannotate_sam2.py`

**Honest bottom line:** no heuristic nails this textured-mesh scene. The best annotator is the
**trained model itself** — hand-label a *seed* (a few images, using the candidates as a head
start), train once (`train_from_label_studio.py`), then let the model pre-annotate the rest and
just correct it (active-learning loop). Heuristics only reduce the seed effort.

**Claude-vision classification** (`scripts/make_candidate_crops.py` makes numbered crop
montages): Claude reliably tells **plastic vs organic** (e.g. it flags the green algae filaments
in the sparse sieves as `matiere_organique`); per-particle *morphotype* from tiny crops is noisy.
Ask to run it as a workflow over the montages to auto-suggest classes you then confirm.

## Claude few-shot classification (optional accelerator)

Build the reference guide (`scripts/build_reference_guide.py <examples_dir> guide.png`) and candidate
montages (`scripts/make_candidate_crops.py`), then run the workflow `scripts/classify_fewshot.workflow.js`
(say "workflow" to authorize it) with `args = {guide, montages:[...]}`. Map the returned per-box classes
back into `ls_tasks.json` with `corseacare.ls_merge.apply_class_suggestions`, re-import in Label Studio,
and **validate least-confident-first**. Morphotype suggestions are hints; plastic-vs-organic is reliable.
