from typing import Optional

import cv2
import numpy as np


def apply_crop(
    frame_bgr: np.ndarray,
    crop_cfg: Optional[dict] = None,
) -> np.ndarray:

    if crop_cfg is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    top = int(h * crop_cfg.get("top", 0.0))
    bottom = int(h * crop_cfg.get("bottom", 1.0))
    left = int(w * crop_cfg.get("left", 0.0))
    right = int(w * crop_cfg.get("right", 1.0))

    top = max(0, min(top, h))
    bottom = max(top + 1, min(bottom, h))
    left = max(0, min(left, w))
    right = max(left + 1, min(right, w))

    return frame_bgr[top:bottom, left:right]