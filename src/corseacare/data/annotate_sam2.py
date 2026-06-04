from corseacare.pipeline.segment_sam2 import Segmenter
from corseacare.data.convert import masks_to_yolo_lines


def boxes_to_yolo_for_review(image_bgr, detections, segmenter: Segmenter) -> list[str]:
    """Refine human-drawn boxes into masks with SAM2, emit YOLO-det lines for review."""
    masks = segmenter.segment(image_bgr, detections)
    h, w = image_bgr.shape[:2]
    class_ids = [d.class_id for d in detections]
    return masks_to_yolo_lines(masks, class_ids, w, h)
