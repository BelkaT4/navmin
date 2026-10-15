"""Operator-only runtime distance state; independent from aiming and turret."""

from __future__ import annotations

import math
from enum import Enum


class DistanceMeasurementMode(Enum):
    CONSTANT = "constant"
    STEREO_EXACT = "stereo-exact"
    STEREO_TIMESTAMP = "stereo-timestamp"


def quantize_constant_distance(value_m: float) -> int:
    """Clamp to 10..500 m, ties go upwards, with a 10 m grid."""
    if isinstance(value_m, bool) or not isinstance(value_m, (int, float)):
        raise TypeError("constant distance must be numeric")
    if not math.isfinite(value_m):
        raise ValueError("constant distance must be finite")
    clamped = max(10.0, min(500.0, float(value_m)))
    return min(500, max(10, math.floor(clamped / 10.0 + 0.5) * 10))


__all__ = ["DistanceMeasurementMode", "quantize_constant_distance"]
