import numpy as np


def counting_error(pred_counts, gt_counts) -> dict:
    p = np.asarray(pred_counts, float); g = np.asarray(gt_counts, float)
    mae = float(np.mean(np.abs(p - g)))
    nz = g != 0
    mape = float(np.mean(np.abs((p[nz] - g[nz]) / g[nz])) * 100) if nz.any() else float("nan")
    return {"mae": mae, "mape": mape}


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def detection_pr_at_iou(preds, gts, iou_thr=0.5) -> dict:
    matched = set(); tp = 0
    for p in preds:
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = _iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp += 1; matched.add(best_j)
    fp = len(preds) - tp; fn = len(gts) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
