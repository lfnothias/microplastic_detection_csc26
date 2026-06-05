# Third-party notices

CorSeaCare_yolo is distributed under **AGPL-3.0-or-later** (see [LICENSE](LICENSE)). It
builds on, and at runtime can use, the following third-party components, each under its own
license. This list is informational; consult each project for authoritative terms.

| Component | Role | License |
|-----------|------|---------|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | detection model (Tier 1) | AGPL-3.0 |
| [SAM 2](https://github.com/facebookresearch/sam2) | optional segmentation / annotation assist | Apache-2.0 |
| [DINOv2](https://github.com/facebookresearch/dinov2) | optional embeddings (Tier-2 clustering) | Apache-2.0 |
| [Label Studio](https://github.com/HumanSignal/label-studio) | annotation UI | Apache-2.0 |
| [OpenCV](https://opencv.org) | image processing | Apache-2.0 |
| [scikit-image](https://scikit-image.org) | measurement / morphology | BSD-3-Clause |
| [scikit-learn](https://scikit-learn.org) | HDBSCAN clustering, scaling | BSD-3-Clause |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [pandas](https://pandas.pydata.org) | tabular output | BSD-3-Clause |
| [Pillow](https://python-pillow.org) | image I/O | MIT-CMU / HPND |
| [Streamlit](https://streamlit.io) | local app | Apache-2.0 |
| [Typer](https://typer.tiangolo.com) | CLI | MIT |

**Note on Ultralytics (AGPL-3.0):** because the YOLO backend is AGPL, this project is
likewise released under AGPL-3.0-or-later. If you deploy a network service built on it, the
AGPL's network-use clause applies.
