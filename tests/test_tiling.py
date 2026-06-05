import numpy as np
from corseacare.tiling import (tile_origins, remap_boxes_to_tile, offset_boxes_from_tile, nms,
                               tiled_detect)


def test_tiled_detect_offsets_and_merges():
    img = np.zeros((40, 40, 3), np.uint8)
    merged = tiled_detect(img, lambda t: [(0, 0.9, 5, 5, 10, 10)], tile=20, overlap=0.0, iou=0.5)
    assert len(merged) == 4                       # one box per tile, non-overlapping
    centres = {(round((b[2] + b[4]) / 2, 1), round((b[3] + b[5]) / 2, 1)) for b in merged}
    assert centres == {(7.5, 7.5), (27.5, 7.5), (7.5, 27.5), (27.5, 27.5)}


def test_tiled_detect_pads_small_image_to_tile():
    img = np.zeros((15, 15, 3), np.uint8)         # smaller than tile -> must be padded
    sizes = []
    tiled_detect(img, lambda t: (sizes.append(t.shape[:2]) or []), tile=20, overlap=0.0)
    assert sizes == [(20, 20)]


def test_tile_origins_cover_edges():
    origins = tile_origins(1000, 1000, 640, overlap=0.2)
    xs = sorted({x for x, y in origins})
    assert xs[0] == 0 and xs[-1] == 1000 - 640        # flush to right/bottom edge


def test_tile_origins_small_image_single_tile():
    assert tile_origins(400, 300, 640) == [(0, 0)]


def test_remap_box_inside_tile():
    # full-image box centred at (0.5,0.5), 10% size, on a 1000x1000 image
    out = remap_boxes_to_tile([(0, 0.5, 0.5, 0.1, 0.1)], 1000, 1000, x0=400, y0=400, tile=200)
    assert len(out) == 1
    cls, cx, cy, w, h = out[0]
    assert cls == 0 and abs(cx - 0.5) < 1e-6 and abs(w - 0.5) < 1e-6   # 100px box in 200px tile


def test_remap_box_outside_dropped():
    assert remap_boxes_to_tile([(0, 0.9, 0.9, 0.05, 0.05)], 1000, 1000, 0, 0, 200) == []


def test_offset_boxes():
    assert offset_boxes_from_tile([(0, 0.9, 10, 10, 20, 20)], 100, 50) == [(0, 0.9, 110, 60, 120, 70)]


def test_nms_dedups_overlap():
    a = (0, 0.9, 0, 0, 10, 10)
    b = (0, 0.5, 1, 1, 11, 11)          # ~IoU high with a
    c = (0, 0.8, 50, 50, 60, 60)        # disjoint
    kept = nms([a, b, c], iou_thr=0.5)
    assert a in kept and c in kept and b not in kept
