"""Tier-2 discovery clustering of the 'autre' (unidentified) detections — propose clusters to name.

Takes a model's detections (predict_tiled counts.csv), keeps the `autre` boxes, computes HYBRID
features — physical (masked colour / size / shape via features.py) + optional DINOv2 embedding —
and clusters them with HDBSCAN. Outputs, under data/clusters/:
  - montage_cluster_<k>.png   example crops per proposed cluster (to eyeball + name)
  - ls_clusters.json          Label Studio import: each `autre` box labeled by its cluster, so you
                              open the sieve photo, SEE the clusters, and name them
  - cluster_config.xml        matching labeling config (cluster_0..N + noise)
  - cluster_summary.csv       per-cluster signature (n, dominant colour, mean size/shape) + a blank
                              `name` column to fill; rename via apply_cluster_names.py afterwards

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/cluster_autre.py [counts.csv] [--no-emb] [--min N]
"""
import sys
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import cv2

from corseacare.mask import particle_mask
from corseacare.features import masked_features, feature_vector
from corseacare.measure import classify_colour
from corseacare.cluster import hybrid_matrix, cluster_hdbscan
from corseacare.calib import load_mm_per_px_map

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
OUT = ROOT / "data" / "clusters"
CELL, COLS, PAD, MAX_EX = 110, 8, 0.25, 32
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#800000", "#aaffc3",
           "#808000", "#ffd8b1", "#000075", "#a9a9a9"]


def load_dino():
    try:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(dev).eval()
        return m, dev
    except Exception as e:
        print(f"  DINOv2 indisponible ({type(e).__name__}); features PHYSIQUES seulement.")
        return None, None


def embed(model, dev, crops):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            b = crops[i:i + 32]
            t = torch.stack([torch.from_numpy(
                cv2.cvtColor(cv2.resize(c, (98, 98)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                for c in b])
            t = ((t - mean) / std).to(dev)
            out.append(model(t).cpu().numpy())
    return np.vstack(out)


def crop_box(img, b):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = b
    pw, ph = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    c = img[y1:y2, x1:x2]
    return c if c.size else np.zeros((CELL, CELL, 3), np.uint8)


def main(counts, use_emb=True, min_cs=40):
    rows = [r for r in csv.DictReader(open(counts)) if r["class"] == "autre"]
    print(f"{len(rows)} détections 'autre' à clusteriser")
    scale = load_mm_per_px_map(ROOT / "samples.csv")
    cache, crops, feats, meta = {}, [], [], []
    for r in rows:
        im = r["image"]
        if im not in cache:
            cache[im] = cv2.imread(str(IMAGES / im))
        img = cache[im]
        if img is None:
            continue
        b = [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]
        c = crop_box(img, b)
        m = particle_mask(c)
        fd = masked_features(c, m, scale.get(im, 0.1))
        crops.append(c); feats.append(feature_vector(fd))
        meta.append({"image": im, "box": b, "feat": fd})
    F = np.array(feats)

    emb = None
    if use_emb:
        model, dev = load_dino()
        if model is not None:
            print("  calcul des embeddings DINOv2 ...")
            emb = embed(model, dev, crops)
    from corseacare.cluster import standardize
    if emb is not None:
        from sklearn.decomposition import PCA
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)   # L2-normalize (cosine-like)
        k = min(40, e.shape[0], e.shape[1])
        E = standardize(PCA(n_components=k, random_state=0).fit_transform(e))
        X = np.hstack([E, 0.5 * standardize(F)])                        # appearance-driven + light physical
    else:
        X = standardize(F)
    labels = cluster_hdbscan(X, min_cluster_size=min_cs)
    uniq = sorted(set(labels))
    clusters = [c for c in uniq if c >= 0]
    print(f"-> {len(clusters)} clusters + {int((labels == -1).sum())} bruit "
          f"(min_cluster_size={min_cs}, features={'physiques+DINOv2' if emb is not None else 'physiques'})")

    if OUT.exists():
        import shutil; shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # per-cluster montages (up to MAX_EX examples)
    by_cluster = defaultdict(list)
    for i, lab in enumerate(labels):
        by_cluster[int(lab)].append(i)
    for k in clusters:
        idx = by_cluster[k][:MAX_EX]
        rows_n = (len(idx) + COLS - 1) // COLS
        canvas = np.full((rows_n * CELL, COLS * CELL, 3), 30, np.uint8)
        for j, i in enumerate(idx):
            cell = cv2.resize(crops[i], (CELL, CELL))
            rr, cc = divmod(j, COLS)
            canvas[rr * CELL:(rr + 1) * CELL, cc * CELL:(cc + 1) * CELL] = cell
        cv2.putText(canvas, f"cluster_{k}  (n={len(by_cluster[k])})", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"montage_cluster_{k}.png"), canvas)

    # Label Studio import: per image, autre boxes labeled by cluster
    def lab_name(k):
        return f"cluster_{k}" if k >= 0 else "noise"
    per_img = defaultdict(list)
    for i, m in enumerate(meta):
        per_img[m["image"]].append((m["box"], lab_name(int(labels[i]))))
    tasks = []
    for im, items in per_img.items():
        H, W = cache[im].shape[:2]
        res = [{"type": "rectanglelabels", "from_name": "label", "to_name": "image",
                "original_width": W, "original_height": H, "image_rotation": 0,
                "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                          "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [lab]}}
               for (x1, y1, x2, y2), lab in items]
        tasks.append({"data": {"image": "http://localhost:8081/corseacare/" + im},
                      "predictions": [{"model_version": "autre-clusters", "result": res}]})
    (OUT / "ls_clusters.json").write_text(json.dumps(tasks, indent=2))

    # labeling config
    labels_xml = "".join(
        f'    <Label value="cluster_{k}" background="{PALETTE[k % len(PALETTE)]}"/>\n' for k in clusters)
    labels_xml += '    <Label value="noise" background="#000000"/>\n'
    (OUT / "cluster_config.xml").write_text(
        '<View>\n  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n' + labels_xml +
        "  </RectangleLabels>\n</View>\n")

    # cluster summary (signature + blank name to fill)
    with open(OUT / "cluster_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n", "dominant_colour", "mean_area_mm2", "mean_aspect",
                    "mean_solidity", "name_to_fill"])
        for k in clusters:
            ms = [meta[i]["feat"] for i in by_cluster[k]]
            hue = np.median([d["hue"] for d in ms]); sat = np.median([d["sat"] for d in ms])
            val = np.median([d["val"] for d in ms])
            col = classify_colour(hue, sat, val)
            w.writerow([f"cluster_{k}", len(by_cluster[k]), col,
                        round(float(np.median([d["area_mm2"] for d in ms])), 3),
                        round(float(np.median([d["aspect"] for d in ms])), 2),
                        round(float(np.median([d["solidity"] for d in ms])), 2), ""])
    print(f"-> montages + ls_clusters.json + cluster_config.xml + cluster_summary.csv dans {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    use_emb = "--no-emb" not in args
    min_cs = 40
    if "--min" in args:
        min_cs = int(args[args.index("--min") + 1])
    pos = [a for a in args if not a.startswith("--") and a != str(min_cs)]
    counts = pos[0] if pos else str(ROOT / "data" / "corseacare_pred_tiled" / "counts.csv")
    main(counts, use_emb, min_cs)
