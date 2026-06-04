import numpy as np
import cv2


def draw_overlay(image_bgr: np.ndarray, detections, masks, alpha: float = 0.4) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()
    for det, mask in zip(detections, masks):
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=-1)
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{det.class_name} {det.confidence:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
