# CorSeaCare_yolo

Visual recognition for **counting marine microplastic particles** in photographs —
particles of multiple colours and sizes, mixed with organic matter, collected at sea.

This repository currently holds the **literature and resource base** for the project.
Code (annotation configs, training/inference pipeline) is to follow.

## Contents

| File | What it is |
|------|------------|
| [`references.bib`](references.bib) | 30 references (BibTeX), 5 sections — peer-reviewed entries fetched from Crossref/doi.org, grey literature curated by hand |
| [`GITHUB_PROJECTS.md`](GITHUB_PROJECTS.md) | Curated GitHub repos doing microplastic detection / counting / segmentation |

## Problem framing

The hard part is **not detection but discrimination**: on an RGB photo, colour and
morphology alone do not reliably separate a plastic particle from an organic fragment.
Two strategies:

1. **RGB-only** detection/segmentation on raw photos — fast to set up, more false positives.
2. **Nile Red fluorescence staining** (after organic-matter digestion) — plastics fluoresce,
   organics largely do not → much cleaner segmentation. Practical state of the art.

> **Appearance gives morphotype + colour only — NOT polymer type.** Polymer identification
> (PE / PP / PET) still requires FTIR or Raman on a sub-sample.

## Annotation plan

- Annotate the **real mixed scenes** (not isolated per-category photos) — matches the
  deployment distribution.
- **Schema:** morphotype = class (fragment / fibre / film / foam / pellet) **+ an explicit
  `organic_matter` negative class**; **colour = per-instance attribute, not a class**
  (avoids morphotype × colour explosion); size auto-computed from the mask
  (include a **scale reference** in frame).
- Bootstrap annotation with **SAM / µSAM** pre-segmentation, then correct.
- Tools: **Label Studio** or **CVAT** locally (offline-capable, SAM-assisted) at sea;
  **Roboflow** when online.

## Compute plan (field constraint: MacBook Pro 13", weak GPU, often offline at sea)

| Step | Where |
|------|-------|
| Annotation | Local, offline OK |
| Training / fine-tuning | **Cloud** (e.g. Colab) when connected / onshore — *not* on the MacBook |
| Inference (counting) | Local — a small YOLO runs fine on the MacBook |

## Candidate models

- **YOLOv8 / YOLO11-seg** — pragmatic default (count + size + colour, easy deployment).
- **YOLO + SAM2** — SAM2 segments, YOLO classifies/counts (recommended starting pipeline).
- **Mask R-CNN** — finer masks, heavier.
- **RT-DETRv2** — better on dense/overlapping particles, heavier to train/deploy.

See `references.bib` §2–3 and `GITHUB_PROJECTS.md` §1 for sources and code templates.

## Next steps

- [ ] Decide imaging modality (RGB macro vs Nile Red fluorescence)
- [ ] Fix the annotation class list + write a one-page annotation guide
- [ ] Build a pilot annotated set (~20–50 mixed-scene photos)
- [ ] Fine-tune a YOLO11-seg baseline in Colab; evaluate counting error on a held-out set

---

*Bibliography compiled 2026-06-04. Citation metadata fetched from Crossref/doi.org — verify
before use in a manuscript.*
