from corseacare.metrics import counting_error, detection_pr_at_iou


def test_counting_error_mae_mape():
    out = counting_error(pred_counts=[10, 8], gt_counts=[12, 8])
    assert out["mae"] == 1.0
    assert round(out["mape"], 2) == round((2 / 12) * 100 / 2, 2)


def test_detection_pr_perfect_match():
    preds = [(0, 0, 10, 10)]
    gts = [(0, 0, 10, 10)]
    out = detection_pr_at_iou(preds, gts, iou_thr=0.5)
    assert out["precision"] == 1.0 and out["recall"] == 1.0


def test_detection_pr_false_positive():
    out = detection_pr_at_iou([(0, 0, 10, 10), (50, 50, 60, 60)], [(0, 0, 10, 10)], 0.5)
    assert out["precision"] == 0.5 and out["recall"] == 1.0
