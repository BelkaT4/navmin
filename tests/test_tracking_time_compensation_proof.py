from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from navmin.config import load_config
from navmin.contracts import CameraRole, TargetRef, TrackingError
from navmin.core.tracking_estimator import (
    TrackingEstimateInvalidReason,
    TrackingEstimator,
)
from navmin.diagnostics.tracking_time_compensation_proof import (
    run_production_tracking_time_compensation_proof,
)
from navmin.turret.platform_motion import CommandMotionModel


def _tracking_config(*, video_latency_ms: int):
    path = Path(__file__).resolve().parents[1] / "config" / "defaults" / "navmin.json"
    base = load_config(path).tracking_control
    return replace(
        base,
        platform_motion=replace(
            base.platform_motion,
            video_latency_ms=video_latency_ms,
        ),
    )


def test_production_compensation_removes_dynamic_false_target_rate() -> None:
    result = run_production_tracking_time_compensation_proof()

    assert result.passed
    assert result.perfect_calibration_max_error_deg_s == pytest.approx(0.0, abs=1e-9)
    assert result.zero_latency_window_aligned_peak_error_deg_s == pytest.approx(
        0.0, abs=1e-9
    )
    assert result.invalid_platform_is_safe
    assert {case.scenario for case in result.cases} == {
        "stationary_target_accelerate",
        "moving_target_accelerate",
        "stationary_target_brake",
        "stationary_target_reverse",
    }


def test_production_compensation_is_bounded_for_25ms_calibration_error() -> None:
    result = run_production_tracking_time_compensation_proof(
        acceleration_x_deg_s2=150.0,
        true_video_latency_ms=150,
    )

    assert result.minus_25ms_peak_error_deg_s == pytest.approx(3.75, abs=0.05)
    assert result.plus_25ms_peak_error_deg_s == pytest.approx(3.75, abs=0.05)


def test_platform_latency_config_change_resets_estimator_history() -> None:
    estimator = TrackingEstimator(_tracking_config(video_latency_ms=0))
    target = TargetRef(CameraRole.OVERVIEW, 1, 1)
    model = CommandMotionModel(
        acceleration_x_deg_s2=100.0,
        acceleration_y_deg_s2=100.0,
    )
    model.mark_known_zero(timestamp_ns=1_000_000_000)
    snapshot = model.snapshot()
    for offset_ms in (0, 50, 100, 150):
        estimator.observe(
            TrackingError(
                target,
                1.0,
                0.0,
                1_000_000_000 + offset_ms * 1_000_000,
            ),
            snapshot,
        )
    assert estimator.sample_count == 4

    estimator.replace_config(_tracking_config(video_latency_ms=25))

    assert estimator.sample_count == 0
    assert estimator.diagnostics.target == target
    assert (
        estimator.diagnostics.x.invalid_reason
        is TrackingEstimateInvalidReason.INSUFFICIENT_SAMPLES
    )


def test_snapshot_coverage_failure_suppresses_rate_effects() -> None:
    estimator = TrackingEstimator(_tracking_config(video_latency_ms=150))
    target = TargetRef(CameraRole.OVERVIEW, 1, 1)
    model = CommandMotionModel(
        acceleration_x_deg_s2=100.0,
        acceleration_y_deg_s2=100.0,
    )
    # The model starts too late to cover corrected timestamps.
    model.mark_known_zero(timestamp_ns=2_000_000_000)
    snapshot = model.snapshot()
    diagnostics = None
    for offset_ms in (0, 50, 100, 150):
        diagnostics = estimator.observe(
            TrackingError(
                target,
                float(offset_ms) / 100.0,
                0.0,
                2_000_000_000 + offset_ms * 1_000_000,
            ),
            snapshot,
        )

    assert diagnostics is not None
    assert not diagnostics.x.valid
    assert (
        diagnostics.x.invalid_reason
        is TrackingEstimateInvalidReason.PLATFORM_UNAVAILABLE
    )
    assert diagnostics.x.target_rate_deg_s is None
    assert diagnostics.x.prediction_offset_deg == 0.0


def test_window_alignment_uses_same_nonuniform_observation_timestamps_as_error_fit() -> None:
    latency_ms = 150
    estimator = TrackingEstimator(_tracking_config(video_latency_ms=latency_ms))
    target = TargetRef(CameraRole.OVERVIEW, 1, 1)
    model = CommandMotionModel(
        acceleration_x_deg_s2=150.0,
        acceleration_y_deg_s2=150.0,
    )
    base_ns = 5_000_000_000
    model.mark_known_zero(timestamp_ns=base_ns)
    model.record_velocity_command(
        timestamp_ns=base_ns + 800_000_000,
        commanded_rate_x_deg_s=40.0,
        commanded_rate_y_deg_s=0.0,
    )
    snapshot = model.snapshot()

    diagnostics = None
    for receive_ms in (850, 897, 956, 1008, 1069, 1117, 1174):
        capture_ms = receive_ms - latency_ms
        angle = snapshot.angle_at(base_ns + capture_ms * 1_000_000)
        assert angle is not None and angle.angle_x_deg is not None
        diagnostics = estimator.observe(
            TrackingError(
                target,
                5.0 - angle.angle_x_deg,
                0.0,
                base_ns + receive_ms * 1_000_000,
            ),
            snapshot,
        )

    assert diagnostics is not None
    assert diagnostics.x.valid
    assert diagnostics.x.target_rate_deg_s == pytest.approx(0.0, abs=1e-9)
