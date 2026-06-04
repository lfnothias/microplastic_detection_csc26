# GitHub projects — microplastic detection / counting / segmentation

Curated 2026-06-04 via `gh search repos` (62 repos screened) + web search.
Star counts and dates are a snapshot; re-run `gh repo view <repo>` for current state.

> ⚠️ Most of these are small academic / student repos (few stars, thin docs).
> They are valuable as **code references, pipelines and datasets to adapt**, not as
> maintained libraries. Always check the license before reusing code or data.

---

## 1. Most relevant — detection & counting from images (YOLO / Faster-RCNN / SAM)

| Repo | Stars | Lang | What it does |
|------|-------|------|--------------|
| [AniLeo-01/Plastic-In-River-Detection](https://github.com/AniLeo-01/Plastic-In-River-Detection) | ★8 | Jupyter | Plastic waste detection in rivers with **YOLOv8** (closest "detect + count in water" template) |
| [MisterKrabz/Microplastic-Segmentation-Research](https://github.com/MisterKrabz/Microplastic-Segmentation-Research) | ★1 | Python | **YOLOv8 + SAM2** coupled to recognise & segment micro/nanoplastics — matches our recommended pipeline |
| [simonspurs/Microplastic-](https://github.com/simonspurs/Microplastic-) | ★0 | — | Full **YOLOv11n-seg** workflow (soil): dataset prep, annotation files, training scripts, eval, viz |
| [PARINEETH2004/Microplastic_yolov11_FasterRCNN](https://github.com/PARINEETH2004/Microplastic_yolov11_FasterRCNN) | ★0 | Jupyter | Side-by-side **YOLOv11 vs Faster-RCNN** |
| [wawswa/microplastic-yolov11](https://github.com/wawswa/microplastic-yolov11) | ★0 | — | Microplastic object detection with **YOLOv11x** |
| [zayedarju/microplastic-detection](https://github.com/zayedarju/microplastic-detection) | ★1 | Python | **YOLOv5** notebook to detect and **count** instances (code behind RSC Adv. 2025, Bin Zahir Arju et al.) |
| [vKenjo/ML_FINAL-PROJ](https://github.com/vKenjo/ML_FINAL-PROJ) | ★1 | Jupyter | YOLOv8 for microplastic detection (student project) |

## 2. Code tied to papers in our bibliography (`references.bib`)

| Repo | Stars | Linked paper |
|------|-------|--------------|
| [arsanchai-su/microplastic-detection](https://github.com/arsanchai-su/microplastic-detection) | ★0 | Akkajit, Alahi & Sukkuea 2024 (*Reg. Stud. Mar. Sci.*) — YOLOv8 / YOLO-NAS, marine |
| [zayedarju/microplastic-detection](https://github.com/zayedarju/microplastic-detection) | ★1 | Bin Zahir Arju et al. 2025 (*RSC Adv.*) — low-cost on-site detection |
| [jalexs82/mp-sim-learn](https://github.com/jalexs82/mp-sim-learn) | ★2 | Smolen et al. 2025 (*PNAS*) — similarity learning on µFTIR spectra |

## 3. Segmentation & counting (U-Net / attention / GAN)

| Repo | Stars | Lang | What it does |
|------|-------|------|--------------|
| [axel-slid/Microplastic-Segmentation-GAN](https://github.com/axel-slid/Microplastic-Segmentation-GAN) | ★4 | Python | GAN-based microplastic segmentation |
| [MahatKC/MicroplasticCounting](https://github.com/MahatKC/MicroplasticCounting) | ★3 | Python | **U-Net for counting** microplastics in microscopy images |
| [Karthikreddy1010/Microplastic_Image_Segmentation](https://github.com/Karthikreddy1010/Microplastic_Image_Segmentation) | ★0 | Jupyter | U-Net + attention, FCN-VGG; **labelled via Label Studio** |
| [XRIST0PH0R0S/Microplastics-segmentation-classification-UNET-VGG16-LR](https://github.com/XRIST0PH0R0S/Microplastics-segmentation-classification-UNET-VGG16-LR) | ★1 | Jupyter | U-Net + VGG16 + LR (BSc thesis, Vilnius) |
| [audrius-savickas/microplastics-segmentation-counting-experiments](https://github.com/audrius-savickas/microplastics-segmentation-counting-experiments) | ★1 | Jupyter | Segmentation + counting experiments |

## 4. Quantitative analysis / counting pipelines

| Repo | Stars | What it does |
|------|-------|--------------|
| [abhinavrajgupta/Quantitative-Analysis-of-Microplastics](https://github.com/abhinavrajgupta/Quantitative-Analysis-of-Microplastics) | ★0 | Detection + classification + **count, size, morphology** at the microscopy stage |
| [andreschristen/bmca](https://github.com/andreschristen/bmca) | ★1 | **Bayesian** Microplastics Count Analyses |
| [S-Bonillas/Image_Recognition_for_Microplastics](https://github.com/S-Bonillas/Image_Recognition_for_Microplastics) | ★0 | ML / CV to detect microplastics in water samples |

## 5. Edge / portable — relevant to on-boat, low-GPU deployment

| Repo | Stars | What it does |
|------|-------|--------------|
| [Ajinkya8472/Iot-Integrated-AI-System-for-Microplastic-Detection-in-Water-Samples](https://github.com/Ajinkya8472/Iot-Integrated-AI-System-for-Microplastic-Detection-in-Water-Samples) | ★1 | **Raspberry Pi 4 edge-AI**, runs locally, no cloud — closest to a field/boat setup |
| [tanurivamsi/MicroplasticDetection_ESP32](https://github.com/tanurivamsi/MicroplasticDetection_ESP32) | ★0 | ESP32-based microplastic detection |
| [ASHWINKUMAR2903/Portable-Microplastic-Detection-System](https://github.com/ASHWINKUMAR2903/Portable-Microplastic-Detection-System) | ★0 | Portable detection system |

## 6. Datasets

| Repo | Stars | What it provides |
|------|-------|------------------|
| [Moore-Institute-4-Plastic-Pollution-Res/Microplastic_Data_Portal](https://github.com/Moore-Institute-4-Plastic-Pollution-Res/Microplastic_Data_Portal) | — | Open-source microplastics **data portal** |
| [ymzhu19eee/dataset_microplastics](https://github.com/ymzhu19eee/dataset_microplastics) | — | Microplastic particles dataset (digital inline **holography**) |
| [HighTempGamer/DATASET_MICROPLASTICS_YOLOv8](https://github.com/HighTempGamer/DATASET_MICROPLASTICS_YOLOv8) | — | Dataset pre-formatted for **YOLOv8** |
| [xf530xf/MP-SSM](https://github.com/xf530xf/MP-SSM) | — | Microplastic detection datasets |
| [DOI-USGS/great-lakes-microplastics](https://github.com/DOI-USGS/great-lakes-microplastics) | — | USGS microplastics field data (Great Lakes) |

Also see the **sewage segmentation+detection dataset** paper (Lee et al. 2024, *Microplastics*) and
the **Roboflow Universe** microplastics datasets — both in `references.bib`.

## 7. Spectral / hyperspectral / holographic classification (related, not RGB photos)

These identify **polymer type** from spectra/holography — complementary to visual counting,
relevant if you later add FTIR/Raman/hyperspectral validation.

| Repo | Stars | What it does |
|------|-------|--------------|
| [petroshatt/hyperplastics](https://github.com/petroshatt/hyperplastics) | ★2 | ML + chemometrics, **hyperspectral** imaging |
| [Gmiaojy/WMViT3](https://github.com/Gmiaojy/WMViT3) | ★3 | Polarimetric holography, lightweight **wavelet-enhanced ViT** |
| [jalexs82/mp-sim-learn](https://github.com/jalexs82/mp-sim-learn) | ★2 | Similarity learning on **µFTIR** spectra (PNAS 2025) |
| [CheLamVien/Microplastic-classification](https://github.com/CheLamVien/Microplastic-classification) | ★3 | LR + NN on spectra |
| [deprecated-work/rgb-to-hyper](https://github.com/deprecated-work/rgb-to-hyper) | ★1 | RGB→hyperspectral for water microplastics |

## 8. Marine-context / aggregation

| Repo | Stars | What it does |
|------|-------|--------------|
| [ShaliniAnandaPhD/Sea_Sifter](https://github.com/ShaliniAnandaPhD/Sea_Sifter) | ★3 | Marine microplastic contamination data + mapping + NLP recommendations |
| [smit-sms/Plastic-Detection-in-River](https://github.com/smit-sms/Plastic-Detection-in-River) | — | Streamlit app, YOLO plastic-in-river |

---

## Takeaways for CorSeaCare_yolo

- **Best starting templates:** `MisterKrabz/Microplastic-Segmentation-Research` (YOLOv8+SAM2,
  exactly our recommended pipeline) and `simonspurs/Microplastic-` (complete YOLOv11n-seg
  workflow with annotation files and training scripts).
- **Paper-backed code:** `arsanchai-su/microplastic-detection` and `zayedarju/microplastic-detection`
  correspond to peer-reviewed marine / low-cost detection papers already in `references.bib`.
- **Edge/boat reference:** `Ajinkya8472/Iot-Integrated-AI-System...` shows a fully local Raspberry-Pi
  inference setup — a model for at-sea, no-cloud counting.
- No single repo is a turnkey solution for *marine particles mixed with organic matter*; expect to
  adapt a YOLO(v8/v11)-seg or YOLO+SAM2 pipeline and build your own annotated dataset.
