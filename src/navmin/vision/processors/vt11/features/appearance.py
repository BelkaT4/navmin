"""Generic single-frame appearance saliency score for VT-4 ACQUIRE.

This is deliberately not a trained identity descriptor. VT-5 will create the
reference bank needed for target-to-candidate appearance distance.
"""
from __future__ import annotations

import cv2
import numpy as np

from navmin.vision.processors.vt11.features.common import candidate_patch_and_background, clamp01


def appearance_score(frame_bgr, candidate) -> float:
    p = candidate_patch_and_background(frame_bgr, candidate)
    gray = p.gray.astype(np.float32)
    if gray.size == 0:
        return 0.0
    bg_mean = float(np.mean(p.background))
    bg_std = float(np.std(p.background))
    deviation = np.abs(gray - bg_mean)
    saliency = float(np.mean(deviation)) / (bg_std + 8.0)

    # Coarse normalized structure term: visible spatial variation is useful,
    # but saturation prevents texture from overwhelming the click prior.
    normalized = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA)
    structure = float(np.std(normalized)) / 64.0
    return clamp01(0.70 * min(1.0, saliency / 3.0) + 0.30 * min(1.0, structure))
