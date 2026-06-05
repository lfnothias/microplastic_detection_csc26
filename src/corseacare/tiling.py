"""Tiling (SAHI-style) helpers for tiny objects in large photos.

Slice a big image into overlapping tiles, remap YOLO boxes into each tile, offset tile
detections back to full-image coordinates, and merge with NMS.
"""
import numpy as np


def tile_origins(W, H, tile, overlap=0.2):
    """Top-left (x0, y0) of each tile covering a WxH image (last tile flush to the edge)."""
    stride = max(1, int(tile * (1 - overlap)))

    def axis(n):
        if n <= tile:
            return [0]
        xs = list(range(0, n - tile + 1, stride))
        if xs[-1] != n - tile:
            xs.append(n - tile)
        return xs

    return [(x, y) for y in axis(H) for x in axis(W)]


def remap_boxes_to_tile(boxes_norm, W, H, x0, y0, tile, min_visible=0.3):
    """Full-image YOLO boxes (cls,cx,cy,w,h normalised) -> tile-normalised boxes.

    Keeps a box only if at least `min_visible` of its area falls inside the tile; clips it.
    """
    out = []
    for cls, cx, cy, w, h in boxes_norm:
        bx1, by1 = (cx - w / 2) * W, (cy - h / 2) * H
        bx2, by2 = (cx + w / 2) * W, (cy + h / 2) * H
        ix1, iy1 = max(bx1, x0), max(by1, y0)
        ix2, iy2 = min(bx2, x0 + tile), min(by2, y0 + tile)
        iw, ih = ix2 - ix1, iy2 - iy1
        barea = (bx2 - bx1) * (by2 - by1)
        if iw <= 0 or ih <= 0 or barea <= 0 or (iw * ih) / barea < min_visible:
            continue
        out.append((int(cls), ((ix1 + ix2) / 2 - x0) / tile, ((iy1 + iy2) / 2 - y0) / tile,
                    iw / tile, ih / tile))
    return out


def offset_boxes_from_tile(dets, x0, y0):
    """Tile-pixel detections (cls,conf,x1,y1,x2,y2) -> full-image-pixel."""
    return [(c, cf, x1 + x0, y1 + y0, x2 + x0, y2 + y0) for (c, cf, x1, y1, x2, y2) in dets]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, iou_thr=0.5):
    """Greedy NMS over (cls,conf,x1,y1,x2,y2), class-agnostic. Returns kept dets."""
    kept = []
    for d in sorted(dets, key=lambda t: t[1], reverse=True):
        if all(_iou(d[2:], k[2:]) < iou_thr for k in kept):
            kept.append(d)
    return kept
