# CorSeaCare dataset (images + annotations)

This repository ships the sieve photographs and their annotations so the results are fully
reproducible. Large images are stored with **Git LFS**.

## License — CC0 1.0 (public domain)

The **images** (`data/corseacare/*.jpg`, `*.JPG`) and **annotations** (`annotations/*.txt`,
`samples.csv`) are released under **[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)**
(no rights reserved). The *software* remains AGPL-3.0 (see [LICENSE](LICENSE)).

## Contents

| Path | What |
|------|------|
| `data/corseacare/*.jpg,*.JPG` | 38 sieve photographs (Git LFS) |
| `annotations/<stem>.txt` | YOLO-format boxes (class cx cy w h, normalized) for the annotated images |
| `samples.csv` | manifest: `image → sample_id` (physical sieve), `split` (train/val), `sieve_mm` (1/2 mm mesh) |
| `configs/label_studio_config.xml` | 6-class labeling config (`fragment fibre film mousse pellet autre`) |

## Provenance

- **`TAMIS_*`** samples: CorSeaCare expedition (association *Mare Vivu*, Corsica). Each
  `sample_id` is one physical sieve; multiple photos of a sample are physical re-distributions
  (water-mixed) of the same material.
- **`MANTA_*`** samples: an **external public Manta-net collection** included as auxiliary
  training data. *Maintainers: add the original source/citation here.*
- Photographs may show hands/sieves; published with consent of the people involved.

## Reproduce from the published data (no Label Studio needed)

```bash
brew install git-lfs && git lfs install
git clone https://github.com/lfnothias/microplastic_detection_csc26.git
cd microplastic_detection_csc26 && git lfs pull   # fetch the images
./scripts/setup.sh                          # engine + deps
uv pip install "ultralytics>=8.3" "sam-2 @ git+https://github.com/facebookresearch/sam2.git"

uv run python scripts/build_yolo_dataset.py # images + annotations + split -> data/ls_export_yolo
uv run python scripts/tile_dataset.py       # slice into tiles
uv run python scripts/train_tiles.py        # train YOLO11n (held-out by sieve)
```

Results for the shipped split are in [docs/RESULTS.md](docs/RESULTS.md) and
[docs/MANUSCRIPT.md](docs/MANUSCRIPT.md).
