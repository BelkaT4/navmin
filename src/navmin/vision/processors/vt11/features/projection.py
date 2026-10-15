"""Projection concentration score for VT-4 ACQUIRE."""
from __future__ import annotations

import numpy as np

from navmin.vision.processors.vt11.features.common import candidate_patch_and_background, clamp01


def _projection_compactness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 1e-9 or values.size <= 1:
        return 0.0
    pos = np.linspace(-1.0, 1.0, values.size)
    centroid = float((values * pos).sum() / total)
    spread = float((values * np.abs(pos - centroid)).sum() / total)
    # A compact localized peak has low normalized spread.
    return clamp01(1.0 - spread)


def projection_score(frame_bgr, candidate) -> float:
    p = candidate_patch_and_background(frame_bgr, candidate)
    gray = p.gray.astype(np.float32)
    bg_mean = float(np.mean(p.background))
    deviation = np.abs(gray - bg_mean)
    if deviation.size == 0:
        return 0.0
    row = deviation.sum(axis=1)
    col = deviation.sum(axis=0)
    concentration = 0.5 * (_projection_compactness(row) + _projection_compactness(col))
    signal = float(np.mean(deviation)) / 64.0
    return clamp01(0.75 * concentration + 0.25 * min(1.0, signal))
