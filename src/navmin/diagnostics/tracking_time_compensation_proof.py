"""Stage F production tracking-time compensation synthetic proof.

The proof drives the production ``TrackingEstimator`` with a real
``PlatformMotionSnapshot``.  Unlike the Stage E oracle comparison, this module
verifies the production path selected for implementation: calibrated video
latency plus a platform-angle OLS slope over the estimator's own retained
observation timestamps.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from navmin.config import load_config
from navmin.config.models import TrackingControlConfig
from navmin.contracts import (
    CameraRole,
    PlatformMotionSnapshot,
    TargetRef,
    TrackingError,
)
from navmin.core.tracking_estimator import TrackingEstimator
from navmin.turret.platform_motion import CommandMotionModel

_BASE_NS = 30_000_000_000
_FPS = 20
_OBSERVATION_START_MS = 300
_OBSERVATION_END_MS = 2_500
_TARGET = TargetRef(CameraRole.OVERVIEW, generation=1, track_id=1)


@dataclass(frozen=True, slots=True)
class ProductionCompensationScenario:
    name: str
    target_rate_deg_s: float
    velocity_commands: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class ProductionCompensationCase:
    scenario: str
    true_video_latency_ms: int
    configured_video_latency_ms: int
    valid_frame_count: int
    peak_abs_target_rate_error_deg_s: float
    rms_target_rate_error_deg_s: float
    peak_signed_target_rate_error_deg_s: float


@dataclass(frozen=True, slots=True)
class ProductionCompensationProofResult:
    status: str
    acceleration_x_deg_s2: float
    window_ms: int
    fps: int
    cases: tuple[ProductionCompensationCase, ...]
    perfect_calibration_max_error_deg_s: float
    minus_25ms_peak_error_deg_s: float
    plus_25ms_peak_error_deg_s: float
    zero_latency_window_aligned_peak_error_deg_s: float
    invalid_platform_is_safe: bool

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SCENARIOS = (
    ProductionCompensationScenario(
        "stationary_target_accelerate",
        0.0,
        ((800, 40.0),),
    ),
    ProductionCompensationScenario(
        "moving_target_accelerate",
        10.0,
        ((800, 40.0),),
    ),
    ProductionCompensationScenario(
        "stationary_target_brake",
        0.0,
        ((200, 40.0), (900, 0.0)),
    ),
    ProductionCompensationScenario(
        "stationary_target_reverse",
        0.0,
        ((200, 40.0), (900, -40.0)),
    ),
)


def _tracking_config(
    *,
    video_latency_ms: int,
    window_ms: int,
) -> TrackingControlConfig:
    path = Path(__file__).resolve().parents[3] / "config" / "defaults" / "navmin.json"
    base = load_config(path).tracking_control
    return replace(
        base,
        velocity_estimator=replace(
            base.velocity_estimator,
            window_ms=window_ms,
            min_span_ms=min(100, window_ms),
            min_samples=4,
            max_samples=16,
            max_fit_rms_deg=1.0,
            max_target_rate_x_deg_s=90.0,
            max_target_rate_y_deg_s=90.0,
        ),
        platform_motion=replace(
            base.platform_motion,
            video_latency_ms=video_latency_ms,
        ),
    )


def _platform_snapshot(
    scenario: ProductionCompensationScenario,
    *,
    acceleration_x_deg_s2: float,
) -> PlatformMotionSnapshot:
    model = CommandMotionModel(
        acceleration_x_deg_s2=acceleration_x_deg_s2,
        acceleration_y_deg_s2=acceleration_x_deg_s2,
    )
    model.mark_known_zero(timestamp_ns=_BASE_NS)
    for time_ms, velocity_deg_s in scenario.velocity_commands:
        model.record_velocity_command(
            timestamp_ns=_BASE_NS + time_ms * 1_000_000,
            commanded_rate_x_deg_s=velocity_deg_s,
            commanded_rate_y_deg_s=0.0,
        )
    return model.snapshot()


def _run_case(
    scenario: ProductionCompensationScenario,
    *,
    acceleration_x_deg_s2: float,
    true_video_latency_ms: int,
    configured_video_latency_ms: int,
    window_ms: int,
) -> ProductionCompensationCase:
    if true_video_latency_ms < 0 or configured_video_latency_ms < 0:
        raise ValueError("video latency must be non-negative")
    snapshot = _platform_snapshot(
        scenario,
        acceleration_x_deg_s2=acceleration_x_deg_s2,
    )
    estimator = TrackingEstimator(
        _tracking_config(
            video_latency_ms=configured_video_latency_ms,
            window_ms=window_ms,
        )
    )
    errors: list[float] = []
    period_ms = round(1000 / _FPS)
    for receive_ms in range(
        _OBSERVATION_START_MS,
        _OBSERVATION_END_MS + 1,
        period_ms,
    ):
        capture_ms = receive_ms - true_video_latency_ms
        if capture_ms < 0:
            continue
        platform_angle = snapshot.angle_at(_BASE_NS + capture_ms * 1_000_000)
        if platform_angle is None or platform_angle.angle_x_deg is None:
            raise RuntimeError("synthetic platform angle unavailable")
        target_angle_deg = 5.0 + scenario.target_rate_deg_s * capture_ms / 1000.0
        diagnostics = estimator.observe(
            TrackingError(
                target=_TARGET,
                error_x_deg=target_angle_deg - platform_angle.angle_x_deg,
                error_y_deg=0.0,
                timestamp_ns=_BASE_NS + receive_ms * 1_000_000,
            ),
            snapshot,
        )
        axis = diagnostics.x
        if axis.valid:
            if axis.target_rate_deg_s is None:
                raise RuntimeError("valid estimator output has no target rate")
            errors.append(axis.target_rate_deg_s - scenario.target_rate_deg_s)

    if not errors:
        raise RuntimeError("production compensation case produced no valid samples")
    peak_signed = max(errors, key=abs)
    return ProductionCompensationCase(
        scenario=scenario.name,
        true_video_latency_ms=true_video_latency_ms,
        configured_video_latency_ms=configured_video_latency_ms,
        valid_frame_count=len(errors),
        peak_abs_target_rate_error_deg_s=max(abs(value) for value in errors),
        rms_target_rate_error_deg_s=math.sqrt(
            sum(value * value for value in errors) / len(errors)
        ),
        peak_signed_target_rate_error_deg_s=peak_signed,
    )


def run_production_tracking_time_compensation_proof(
    *,
    acceleration_x_deg_s2: float = 150.0,
    true_video_latency_ms: int = 150,
    window_ms: int = 200,
) -> ProductionCompensationProofResult:
    """Exercise the production candidate over acceleration/braking/reversal."""
    if acceleration_x_deg_s2 <= 0.0 or not math.isfinite(acceleration_x_deg_s2):
        raise ValueError("acceleration_x_deg_s2 must be finite and > 0")
    if true_video_latency_ms < 0:
        raise ValueError("true_video_latency_ms must be non-negative")
    if window_ms <= 0:
        raise ValueError("window_ms must be > 0")

    cases: list[ProductionCompensationCase] = []
    for scenario in _SCENARIOS:
        cases.append(
            _run_case(
                scenario,
                acceleration_x_deg_s2=acceleration_x_deg_s2,
                true_video_latency_ms=true_video_latency_ms,
                configured_video_latency_ms=true_video_latency_ms,
                window_ms=window_ms,
            )
        )

    accel = _SCENARIOS[0]
    minus_25 = _run_case(
        accel,
        acceleration_x_deg_s2=acceleration_x_deg_s2,
        true_video_latency_ms=true_video_latency_ms,
        configured_video_latency_ms=max(0, true_video_latency_ms - 25),
        window_ms=window_ms,
    )
    plus_25 = _run_case(
        accel,
        acceleration_x_deg_s2=acceleration_x_deg_s2,
        true_video_latency_ms=true_video_latency_ms,
        configured_video_latency_ms=true_video_latency_ms + 25,
        window_ms=window_ms,
    )
    zero_latency = _run_case(
        accel,
        acceleration_x_deg_s2=acceleration_x_deg_s2,
        true_video_latency_ms=0,
        configured_video_latency_ms=0,
        window_ms=window_ms,
    )

    invalid_model = CommandMotionModel(
        acceleration_x_deg_s2=acceleration_x_deg_s2,
        acceleration_y_deg_s2=acceleration_x_deg_s2,
    ).snapshot()
    invalid_estimator = TrackingEstimator(
        _tracking_config(video_latency_ms=0, window_ms=window_ms)
    )
    invalid_diagnostics = None
    for time_ms in (0, 50, 100, 150):
        invalid_diagnostics = invalid_estimator.observe(
            TrackingError(
                _TARGET,
                float(time_ms) / 100.0,
                0.0,
                _BASE_NS + time_ms * 1_000_000,
            ),
            invalid_model,
        )
    invalid_safe = bool(
        invalid_diagnostics is not None
        and not invalid_diagnostics.x.valid
        and invalid_diagnostics.x.target_rate_deg_s is None
        and invalid_diagnostics.x.prediction_offset_deg == 0.0
    )

    perfect_max = max(case.peak_abs_target_rate_error_deg_s for case in cases)
    tolerance = 1e-9
    calibration_tolerance = acceleration_x_deg_s2 * 0.025 + 0.05
    passed = (
        perfect_max <= tolerance
        and zero_latency.peak_abs_target_rate_error_deg_s <= tolerance
        and minus_25.peak_abs_target_rate_error_deg_s <= calibration_tolerance
        and plus_25.peak_abs_target_rate_error_deg_s <= calibration_tolerance
        and invalid_safe
    )
    return ProductionCompensationProofResult(
        status="PASS" if passed else "FAIL",
        acceleration_x_deg_s2=acceleration_x_deg_s2,
        window_ms=window_ms,
        fps=_FPS,
        cases=tuple(cases),
        perfect_calibration_max_error_deg_s=perfect_max,
        minus_25ms_peak_error_deg_s=minus_25.peak_abs_target_rate_error_deg_s,
        plus_25ms_peak_error_deg_s=plus_25.peak_abs_target_rate_error_deg_s,
        zero_latency_window_aligned_peak_error_deg_s=(
            zero_latency.peak_abs_target_rate_error_deg_s
        ),
        invalid_platform_is_safe=invalid_safe,
    )


def write_production_tracking_time_compensation_proof_json(
    result: ProductionCompensationProofResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    path.write_text(payload, encoding="utf-8")
    return path


__all__ = [
    "ProductionCompensationCase",
    "ProductionCompensationProofResult",
    "ProductionCompensationScenario",
    "run_production_tracking_time_compensation_proof",
    "write_production_tracking_time_compensation_proof_json",
]
