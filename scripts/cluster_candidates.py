"""Cluster candidate particles with DINOv2 embeddings + HDBSCAN, for grouped annotation.

Reads a pre-annotation ls_tasks.json, crops each candidate, embeds every crop with DINOv2
(ViT-S/14, on MPS), L2-normalises, and clusters with HDBSCAN (robust, no k, outliers -> -1).
Writes per-cluster montages so you can judge coherence and label a whole cluster at once.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/cluster_candidates.py [preann_dir]

Outputs under <preann>/clusters/:
  cluster_<id>_<page>.png   montage of that cluster's crops (id -1 = noise/outliers)
  assignments.json          {global_id: {"image","xyxy","cluster"}}
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
CELL, COLS, PER_PAGE = 120, 8, 64
MIN_CLUSTER = int(os.environ.get("CORSEACARE_MIN_CLUSTER", "6"))
PAD_FRAC = float(os.environ.get("CORSEACARE_CROP_PAD", "0.05"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_candidates(preann):
    tasks = json.loads((Path(preann) / "ls_tasks.json").read_text())
    crops, meta = [], []
    for t in tasks:
        name = t["data"]["image"].rsplit("/", 1)[-1]
        img = cv2.imread(str(IMAGES / name))
        if img is None:
            continue
        H, W = img.shape[:2]
        for r in t["predictions"][0]["result"]:
            v = r["value"]
            x1, y1 = int(v["x"] / 100 * W), int(v["y"] / 100 * H)
            x2, y2 = int(x1 + v["width"] / 100 * W), int(y1 + v["height"] / 100 * H)
            px, py = int((x2 - x1) * PAD_FRAC), int((y2 - y1) * PAD_FRAC)
            c = img[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
            if c.size == 0:
                continue
            crops.append(c)
            meta.append({"image": name, "xyxy": [x1, y1, x2, y2]})
    return crops, meta


def to_square(c, size=224, padval=128):
    """Pad to square (neutral grey) then resize — preserves aspect (fibre vs fragment)."""
    h, w = c.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), padval, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return cv2.resize(sq, (size, size))


def embed(crops):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEVICE)
    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            batch = []
            for c in crops[i:i + 32]:
                rgb = cv2.cvtColor(to_square(c), cv2.COLOR_BGR2RGB)
                batch.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
            x = ((torch.stack(batch) - MEAN) / STD).to(DEVICE)
            feats.append(model(x).cpu())
    f = torch.cat(feats).numpy()
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def montage(crops, ids, label, out):
    for page in range((len(ids) + PER_PAGE - 1) // PER_PAGE):
        chunk = ids[page * PER_PAGE:(page + 1) * PER_PAGE]
        rows = (len(chunk) + COLS - 1) // COLS
        canvas = np.full((rows * CELL, COLS * CELL, 3), 25, np.uint8)
        for k, gid in enumerate(chunk):
            cell = cv2.resize(crops[gid], (CELL, CELL))
            r, cc = divmod(k, COLS)
            canvas[r * CELL:(r + 1) * CELL, cc * CELL:(cc + 1) * CELL] = cell
        cv2.imwrite(str(out / f"cluster_{label}_{page}.png"), canvas)


def main(preann):
    preann = Path(preann)
    out = preann / "clusters"; out.mkdir(parents=True, exist_ok=True)
    crops, meta = load_candidates(preann)
    print(f"{len(crops)} candidate crops; embedding with DINOv2 on {DEVICE}...")
    emb = embed(crops)
    from sklearn.cluster import HDBSCAN
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER, metric="euclidean").fit_predict(emb)
    for m, l in zip(meta, labels):
        m["cluster"] = int(l)
    (out / "assignments.json").write_text(json.dumps(
        {i: m for i, m in enumerate(meta)}, indent=2))
    uniq = sorted(set(labels))
    print(f"clusters: {len([u for u in uniq if u >= 0])} + noise; sizes:")
    for l in uniq:
        ids = [i for i, x in enumerate(labels) if x == l]
        print(f"  cluster {l:>3}: {len(ids)} crops")
        montage(crops, ids, l, out)
    print(f"montages -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "corseacare_preann"))
