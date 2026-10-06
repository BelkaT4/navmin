"""Edge-structure quality score for VT-4 ACQUIRE."""
from __future__ import annotations

import cv2
import numpy as np

from navmin.vision.processors.vt11.features.common import candidate_patch_and_background, clamp01


def edge_score(frame_bgr, candidate) -> float:
    p = candidate_patch_and_background(frame_bgr, candidate)
    gray = p.gray
    if gray.size < 4:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    mean_grad = float(np.mean(magnitude)) / 255.0

    # Boundary strength: object boundary should differ from local background.
    obj_mean = float(np.mean(gray))
    bg_mean = float(np.mean(p.background))
    boundary_term = abs(obj_mean - bg_mean) / 255.0
    return clamp01(0.65 * min(1.0, mean_grad * 4.0) + 0.35 * min(1.0, boundary_term * 3.0))
