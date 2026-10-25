"""Monotone relative-motion calibration curves in logical angular units."""

from itertools import pairwise
from math import isfinite

type MotionCurve = tuple[tuple[float, float], ...]


def validate_motion_curve(curve: MotionCurve, limit: float) -> None:
    if not curve:
        return
    if len(curve) < 3 or (0.0, 0.0) not in curve:
        raise ValueError("motion curve needs negative, zero and positive anchors")
    if curve[0][0] >= 0 or curve[-1][0] <= 0:
        raise ValueError("motion curve must cover both directions")
    for measured, command in curve:
        if not all(isfinite(v) for v in (measured, command)):
            raise ValueError("motion curve values must be finite")
        if measured * command <= 0 and (measured, command) != (0, 0):
            raise ValueError("motion curve must preserve direction")
        if abs(command) > limit:
            raise ValueError("motion curve command exceeds relative move limit")
    if any(a[0] >= b[0] or a[1] >= b[1] for a, b in pairwise(curve)):
        raise ValueError("motion curve coordinates must be strictly increasing")


def interpolate_motion(curve: MotionCurve, value: float) -> float:
    if not isfinite(value):
        raise ValueError("motion angle must be finite")
    if not curve:
        return value
    if not curve[0][0] <= value <= curve[-1][0]:
        raise ValueError("angle outside measured calibration range")
    for (x0, y0), (x1, y1) in pairwise(curve):
        if x0 <= value <= x1:
            return y0 + (value - x0) * (y1 - y0) / (x1 - x0)
    raise ValueError("invalid motion curve")
