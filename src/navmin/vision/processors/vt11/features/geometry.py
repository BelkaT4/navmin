"""Candidate geometry plausibility score for VT-4 ACQUIRE."""
from __future__ import annotations

from math import exp, log

from navmin.vision.processors.vt11.features.common import clamp01


def geometry_score(_frame_bgr, candidate) -> float:
    b = candidate.bbox
    box_area = max(1, int(b.width) * int(b.height))
    fill = clamp01(float(candidate.area_px) / box_area)
    aspect = max(1e-6, float(b.width) / max(1.0, float(b.height)))
    # Very elongated one-pixel-like components are less plausible as the
    # compact local object hypothesis used by ACQUIRE. The penalty is gentle.
    aspect_score = exp(-abs(log(aspect)) / 2.0)
    return clamp01(0.65 * fill + 0.35 * aspect_score)
