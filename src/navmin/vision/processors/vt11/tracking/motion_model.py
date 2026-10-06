"""Transparent motion models for VT-7.

VT-7 keeps the constant-velocity assumption, but makes it frame-aware and
adds an uncertainty estimate. This avoids over-estimating velocity after a
missed frame and provides a deterministic basis for a dynamic search gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class MotionEstimate:
    vx_px_per_frame: float
    vy_px_per_frame: float
    predicted_xy: tuple[float, float]


@dataclass(frozen=True)
class MotionPoint:
    frame_index: int
    x: float
    y: float

    @property
    def xy(self) -> tuple[float, float]:
        return float(self.x), float(self.y)


@dataclass(frozen=True)
class EnhancedMotionEstimate:
    vx_px_per_frame: float
    vy_px_per_frame: float
    speed_px_per_frame: float
    predicted_xy: tuple[float, float]
    uncertainty_px: float
    target_frame_index: int


def robust_velocity(
    history: Sequence[tuple[float, float]],
    *,
    window_frames: int = 5,
) -> tuple[float, float]:
    """Legacy VT-6 velocity estimator for consecutive samples."""
    if window_frames < 2:
        raise ValueError("window_frames must be >= 2")
    if len(history) < 2:
        return 0.0, 0.0
    recent = list(history[-window_frames:])
    dx = [recent[i][0] - recent[i - 1][0] for i in range(1, len(recent))]
    dy = [recent[i][1] - recent[i - 1][1] for i in range(1, len(recent))]
    return float(median(dx)), float(median(dy))


def predict_position(
    history: Sequence[tuple[float, float]],
    *,
    fallback: tuple[float, float],
    window_frames: int = 5,
    steps_ahead: int = 1,
) -> MotionEstimate:
    """Legacy VT-6 prediction retained for compatibility/tests."""
    if steps_ahead < 1:
        raise ValueError("steps_ahead must be >= 1")
    if not history:
        return MotionEstimate(0.0, 0.0, (float(fallback[0]), float(fallback[1])))
    vx, vy = robust_velocity(history, window_frames=window_frames)
    x, y = history[-1]
    predicted = (float(x + vx * steps_ahead), float(y + vy * steps_ahead))
    return MotionEstimate(vx, vy, predicted)


def _recent_timed_velocities(
    history: Sequence[MotionPoint],
    *,
    window_points: int,
) -> list[tuple[float, float]]:
    if window_points < 2:
        raise ValueError("window_points must be >= 2")
    recent = list(history[-window_points:])
    velocities: list[tuple[float, float]] = []
    for a, b in zip(recent, recent[1:]):
        dt = int(b.frame_index) - int(a.frame_index)
        if dt <= 0:
            continue
        velocities.append(((b.x - a.x) / dt, (b.y - a.y) / dt))
    return velocities


def robust_velocity_timed(
    history: Sequence[MotionPoint],
    *,
    window_points: int = 5,
) -> tuple[float, float]:
    """Median pixel velocity accounting for gaps between accepted frames."""
    velocities = _recent_timed_velocities(history, window_points=window_points)
    if not velocities:
        return 0.0, 0.0
    return (
        float(median(v[0] for v in velocities)),
        float(median(v[1] for v in velocities)),
    )


def motion_uncertainty_timed(
    history: Sequence[MotionPoint],
    *,
    window_points: int = 5,
    steps_ahead: int = 1,
) -> float:
    """Robust forecast uncertainty in pixels from velocity residuals.

    The returned quantity grows with the forecast horizon. It is deliberately
    simple and interpretable: median radial residual of recent velocity
    samples multiplied by the number of frames being extrapolated.
    """
    if steps_ahead < 1:
        raise ValueError("steps_ahead must be >= 1")
    velocities = _recent_timed_velocities(history, window_points=window_points)
    if len(velocities) < 2:
        return 0.0
    vx = float(median(v[0] for v in velocities))
    vy = float(median(v[1] for v in velocities))
    residuals = [hypot(v[0] - vx, v[1] - vy) for v in velocities]
    return float(median(residuals) * steps_ahead)


def predict_position_timed(
    history: Sequence[MotionPoint],
    *,
    target_frame_index: int,
    fallback: tuple[float, float],
    window_points: int = 5,
) -> EnhancedMotionEstimate:
    """Predict a target center for an explicit frame index."""
    if not history:
        return EnhancedMotionEstimate(
            0.0,
            0.0,
            0.0,
            (float(fallback[0]), float(fallback[1])),
            0.0,
            int(target_frame_index),
        )
    last = history[-1]
    steps = int(target_frame_index) - int(last.frame_index)
    if steps < 1:
        raise ValueError("target_frame_index must be after the last history frame")
    vx, vy = robust_velocity_timed(history, window_points=window_points)
    predicted = (float(last.x + vx * steps), float(last.y + vy * steps))
    uncertainty = motion_uncertainty_timed(
        history,
        window_points=window_points,
        steps_ahead=steps,
    )
    return EnhancedMotionEstimate(
        vx_px_per_frame=vx,
        vy_px_per_frame=vy,
        speed_px_per_frame=float(hypot(vx, vy)),
        predicted_xy=predicted,
        uncertainty_px=uncertainty,
        target_frame_index=int(target_frame_index),
    )


def dynamic_search_radius(
    *,
    base_radius_px: float,
    min_radius_px: float,
    max_radius_px: float,
    speed_px_per_frame: float,
    uncertainty_px: float,
    miss_count: int,
    speed_gain: float,
    uncertainty_gain: float,
    miss_growth_px: float,
) -> int:
    """Return a clamped deterministic search radius for VT-7."""
    if min_radius_px <= 0 or max_radius_px <= 0 or base_radius_px <= 0:
        raise ValueError("search radii must be > 0")
    if min_radius_px > max_radius_px:
        raise ValueError("min_radius_px must be <= max_radius_px")
    if miss_count < 0:
        raise ValueError("miss_count must be >= 0")
    if min(speed_gain, uncertainty_gain, miss_growth_px) < 0:
        raise ValueError("dynamic gate gains must be >= 0")
    raw = (
        float(base_radius_px)
        + float(speed_gain) * max(0.0, float(speed_px_per_frame))
        + float(uncertainty_gain) * max(0.0, float(uncertainty_px))
        + float(miss_growth_px) * miss_count
    )
    return int(round(min(max(raw, float(min_radius_px)), float(max_radius_px))))
