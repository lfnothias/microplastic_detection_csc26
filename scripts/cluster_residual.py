"""Discovery clustering of the 'autre' residual: per-sieve HDBSCAN on hybrid features
(masked colour/shape/size + masked DINOv2 embedding), with an optional global pass.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/cluster_residual.py [preann_dir] [--global]
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

from corseacare.mask import particle_mask
from corseacare.features import masked_features, feature_vector
from corseacare.cluster import hybrid_matrix, cluster_hdbscan, group_by_image

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "corseacare"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CELL, COLS, PER_PAGE, MM_PER_PX = 120, 8, 64, 0.1


def load(preann):
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
            c = img[max(0, y1):y2, max(0, x1):x2]
            if c.size:
                crops.append(c); meta.append({"image": name, "xyxy": [x1, y1, x2, y2]})
    return crops, meta


def masked_square(crop, mask, size=224, pad=128):
    out = np.full_like(crop, pad); out[mask > 0] = crop[mask > 0]
    h, w = out.shape[:2]; s = max(h, w)
    sq = np.full((s, s, 3), pad, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = out
    return cv2.resize(sq, (size, size))


def embed_masked(crops, masks):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEVICE)
    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            batch = [torch.from_numpy(cv2.cvtColor(masked_square(c, m), cv2.COLOR_BGR2RGB))
                     .permute(2, 0, 1).float() / 255.0
                     for c, m in zip(crops[i:i + 32], masks[i:i + 32])]
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
            r, cc = divmod(k, COLS)
            canvas[r * CELL:(r + 1) * CELL, cc * CELL:(cc + 1) * CELL] = cv2.resize(crops[gid], (CELL, CELL))
        cv2.imwrite(str(out / f"{label}_{page}.png"), canvas)


def main(preann, do_global):
    preann = Path(preann); out = preann / "residual_clusters"; out.mkdir(parents=True, exist_ok=True)
    crops, meta = load(preann)
    masks = [particle_mask(c) for c in crops]
    fvecs = np.array([feature_vector(masked_features(c, m, MM_PER_PX)) for c, m in zip(crops, masks)])
    emb = embed_masked(crops, masks)
    X = hybrid_matrix(fvecs, emb)
    for img, idxs in group_by_image(meta).items():
        if len(idxs) < 5:
            continue
        sub = cluster_hdbscan(X[idxs], min_cluster_size=4)
        for lab in sorted(set(sub)):
            members = [idxs[j] for j, l in enumerate(sub) if l == lab]
            montage(crops, members, f"{Path(img).stem}_c{lab}", out)
        print(f"{img}: {len([l for l in set(sub) if l >= 0])} clusters (+noise) from {len(idxs)} particles")
    if do_global:
        glab = cluster_hdbscan(X, min_cluster_size=6)
        for lab in sorted(set(glab)):
            montage(crops, [i for i, l in enumerate(glab) if l == lab], f"GLOBAL_c{lab}", out)
        print(f"GLOBAL: {len([l for l in set(glab) if l >= 0])} clusters (+noise)")
    print(f"montages -> {out}")


if __name__ == "__main__":
    cli = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(cli[0] if cli else str(ROOT / "data" / "corseacare_preann"), "--global" in sys.argv)
