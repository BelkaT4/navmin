"""Local object/background contrast score for VT-4 ACQUIRE."""
from __future__ import annotations

import numpy as np

from navmin.vision.processors.vt11.features.common import candidate_patch_and_background, clamp01


def contrast_score(frame_bgr, candidate) -> float:
    p = candidate_patch_and_background(frame_bgr, candidate)
    obj_mean = float(np.mean(p.gray))
    bg_mean = float(np.mean(p.background))
    bg_std = float(np.std(p.background))
    z = abs(obj_mean - bg_mean) / (bg_std + 3.0)
    return clamp01(z / (z + 2.5))
