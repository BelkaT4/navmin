"""Manual, single-axis measurements using a stationary scene landmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, isfinite, sqrt
from statistics import median

from navmin.config.models import AppConfig, AxisMechanicsConfig
from navmin.contracts import CameraModel, CameraRole
from navmin.motion_curve import MotionCurve, interpolate_motion, validate_motion_curve


@dataclass(frozen=True)
class MotionObservation:
    axis: str
    camera: CameraRole
    generation: int
    before_frame_id: int
    after_frame_id: int
    commanded_deg: float
    measured_deg: float
    reversal: bool | None

    @property
    def error_deg(self) -> float:
        return self.commanded_deg - self.measured_deg


@dataclass(frozen=True)
class MotionEstimate:
    gain: float
    rms_deg: float
    backlash_positive_deg: float | None
    backlash_negative_deg: float | None
    curve: MotionCurve = ()

    @property
    def correction(self) -> float:
        return 1.0 / self.gain


def axis_angle(model: CameraModel, x: float, y: float, axis: str) -> float:
    """Projected ray angle for rotation about one aligned mechanical axis."""
    if axis not in ("x", "y"):
        raise ValueError("Неизвестная ось")
    ray = model.pixel_to_ray(x, y)
    if not all(isfinite(v) for v in (ray.x, ray.y, ray.z)) or ray.z <= 0:
        raise ValueError("Недопустимый луч камеры")
    return degrees(atan2(ray.x, ray.z) if axis == "x" else -atan2(ray.y, ray.z))


def estimate_motion(observations: list[MotionObservation], axis: str) -> MotionEstimate:
    samples = [s for s in observations if s.axis == axis]
    # The initial approach has unknown mechanical preload. Reversal measurements
    # must not bias the steps/revolution fit toward compensating lost motion.
    straight = [s for s in samples if s.reversal is False]
    for sign in (-1, 1):
        amplitudes = {
            abs(s.commanded_deg) for s in straight if s.commanded_deg * sign > 0
        }
        if len(amplitudes) < 3 or max(amplitudes) < 2 * min(amplitudes):
            raise ValueError(
                "Нужны минимум три разных угла без реверса в каждую сторону; диапазон ≥2×"
            )
    if any(
        not isfinite(s.commanded_deg)
        or not isfinite(s.measured_deg)
        or abs(s.commanded_deg) < 0.1
        or s.commanded_deg * s.measured_deg <= 0
        for s in straight
    ):
        raise ValueError("Нет измеримого движения либо неверное направление")
    gain = sum(s.commanded_deg * s.measured_deg for s in straight) / sum(
        s.commanded_deg**2 for s in straight
    )
    if not 0.25 <= gain <= 4.0:
        raise ValueError("Масштаб вне диапазона 0.25–4: проверьте механику и точки")
    rms = sqrt(
        sum((s.measured_deg - gain * s.commanded_deg) ** 2 for s in straight)
        / len(straight)
    )
    commands = sorted({s.commanded_deg for s in straight})
    curve = tuple(
        sorted(
            [
                (
                    median(
                        s.measured_deg for s in straight if s.commanded_deg == command
                    ),
                    command,
                )
                for command in commands
            ]
            + [(0.0, 0.0)]
        )
    )
    try:
        validate_motion_curve(curve, max(abs(c) for c in commands))
    except ValueError as exc:
        raise ValueError(
            "Замеры немонотонны: повторите серию, проверьте точки и механику"
        ) from exc
    # Invert the empirical curve to predict actual motion at a given raw command.
    forward = tuple((command, measured) for measured, command in curve)

    def backlash(sign: int) -> float | None:
        turns = [
            s
            for s in samples
            if s.reversal
            and s.commanded_deg * sign > 0
            and forward[0][0] <= s.commanded_deg <= forward[-1][0]
        ]
        if len(turns) < 2:
            return None
        # Signed deficit remains visible: a negative value signals measurement
        # error/model mismatch, and must not silently become zero backlash.
        return median(
            sign * (interpolate_motion(forward, s.commanded_deg) - s.measured_deg)
            for s in turns
        )

    return MotionEstimate(gain, rms, backlash(1), backlash(-1), curve)


def corrected_config(
    config: AppConfig,
    axis: str,
    baseline: AxisMechanicsConfig,
    estimate: MotionEstimate,
) -> AppConfig:
    if axis not in ("x", "y"):
        raise ValueError("Неизвестная ось")
    if getattr(config.turret.axes, axis) != baseline:
        raise ValueError("Механика файла отличается от настроек текущего запуска")
    if not estimate.curve or baseline.relative_motion_curve:
        raise ValueError(
            "Нужна измеренная кривая и исходная механика без активной кривой"
        )
    validate_motion_curve(estimate.curve, baseline.max_relative_move_deg)
    axes = replace(
        config.turret.axes,
        **{
            axis: replace(baseline, relative_motion_curve=estimate.curve),
        },
    )
    return replace(config, turret=replace(config.turret, axes=axes))
