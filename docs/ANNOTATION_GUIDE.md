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

## 1. Launch Label Studio (local image serving)

Installed at `~/.local/bin/label-studio` (v1.23). From a terminal:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/holobiomicslab/git/CorSeaCare_yolo/data
label-studio start
```

Opens http://localhost:8080 (create a local account on first run — stays on your machine).

## 2. Create the project

1. **Create Project** → name `corseacare`.
2. **Labeling Setup → Custom template** → paste `configs/label_studio_config.xml` → **Save**.

## 3. Import photos + pre-annotations

- **Import** → upload `data/corseacare_preann/ls_tasks.json`.
- Tasks reference images via `/data/local-files/?d=corseacare/<name>` (resolves thanks to the
  two `export`s above) and carry the candidate boxes as **predictions** — they appear pre-drawn.
- If images don't load: re-check the two `export` lines and that the document root is
  `.../CorSeaCare_yolo/data` (not `.../data/corseacare`).

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
