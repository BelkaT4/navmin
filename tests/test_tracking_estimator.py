from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from navmin.config import load_config
from navmin.contracts import CameraRole, PlatformMotionRate, TargetRef, TrackingError
from navmin.core.tracking_estimator import (
    TrackingEstimateInvalidReason,
    TrackingEstimator,
)

BASE_NS = 10_000_000_000
TARGET = TargetRef(CameraRole.OVERVIEW, 1, 7)


def _default_tracking_control_config():
    defaults = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "navmin.json"
    )
    return load_config(defaults).tracking_control


def _config(**estimator_changes):
    config = _default_tracking_control_config()
    estimator = replace(config.velocity_estimator, **estimator_changes)
    return replace(config, velocity_estimator=estimator)


def _observe(
    estimator: TrackingEstimator,
    time_ms: int,
    error_x: float,
    error_y: float,
    *,
    target: TargetRef = TARGET,
    platform_x: float | None = 0.0,
    platform_y: float | None = 0.0,
):
    timestamp_ns = BASE_NS + time_ms * 1_000_000
    error = TrackingError(target, error_x, error_y, timestamp_ns)
    platform = PlatformMotionRate(timestamp_ns, platform_x, platform_y)
    return estimator.observe(error, platform)


def test_linear_fit_uses_nonuniform_observation_timestamps_for_both_axes() -> None:
    config = _config(max_target_rate_x_deg_s=100.0, max_target_rate_y_deg_s=100.0)
    prediction = replace(
        config.prediction,
        enabled=True,
        horizon_ms=100,
        max_angle_x_deg=0.5,
        max_angle_y_deg=0.5,
    )
    estimator = TrackingEstimator(replace(config, prediction=prediction))

    diagnostics = None
    for time_ms in (0, 40, 110, 200):
        time_s = time_ms / 1000.0
        diagnostics = _observe(
            estimator,
            time_ms,
            2.0 + 10.0 * time_s,
            -1.0 - 4.0 * time_s,
            platform_x=2.0,
            platform_y=-3.0,
        )

    assert diagnostics is not None
    assert diagnostics.x.valid
    assert diagnostics.y.valid
    assert diagnostics.x.relative_rate_deg_s == pytest.approx(10.0)
    assert diagnostics.y.relative_rate_deg_s == pytest.approx(-4.0)
    assert diagnostics.x.platform_rate_deg_s == 2.0
    assert diagnostics.y.platform_rate_deg_s == -3.0
    assert diagnostics.x.target_rate_deg_s == pytest.approx(12.0)
    assert diagnostics.y.target_rate_deg_s == pytest.approx(-7.0)
    assert diagnostics.x.fit_rms_deg == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.y.fit_rms_deg == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.x.prediction_offset_deg == 0.5
    assert diagnostics.y.prediction_offset_deg == -0.5


def test_minimum_sample_count_and_span_are_independent_gates() -> None:
    estimator = TrackingEstimator(_config())
    for time_ms in (0, 100, 200):
        diagnostics = _observe(estimator, time_ms, time_ms / 100.0, 0.0)
    assert (
        diagnostics.x.invalid_reason
        is TrackingEstimateInvalidReason.INSUFFICIENT_SAMPLES
    )

    estimator = TrackingEstimator(_config(min_samples=2))
    _observe(estimator, 0, 0.0, 0.0)
    diagnostics = _observe(estimator, 50, 1.0, 1.0)
    assert (
        diagnostics.x.invalid_reason is TrackingEstimateInvalidReason.INSUFFICIENT_SPAN
    )


def test_window_evicts_samples_strictly_older_than_cutoff() -> None:
    estimator = TrackingEstimator(
        _config(window_ms=100, min_span_ms=1, min_samples=2, max_samples=10)
    )
    _observe(estimator, 0, 0.0, 0.0)
    _observe(estimator, 50, 0.5, 0.5)
    diagnostics = _observe(estimator, 101, 1.01, 1.01)

    assert diagnostics.x.sample_count == 2
    assert diagnostics.x.actual_span_ms == 51.0
    assert diagnostics.x.valid


def test_max_sample_capacity_can_make_span_invalid_at_high_rate() -> None:
    estimator = TrackingEstimator(
        _config(window_ms=200, min_span_ms=100, min_samples=2, max_samples=4)
    )
    for time_ms in (0, 30, 60, 90, 120):
        diagnostics = _observe(estimator, time_ms, time_ms / 100.0, 0.0)

    assert diagnostics.x.sample_count == 4
    assert diagnostics.x.actual_span_ms == 90.0
    assert (
        diagnostics.x.invalid_reason is TrackingEstimateInvalidReason.INSUFFICIENT_SPAN
    )


def test_fit_rms_is_per_axis_and_bad_fit_does_not_clear_history() -> None:
    estimator = TrackingEstimator(
        _config(
            window_ms=400,
            min_samples=4,
            min_span_ms=100,
            max_samples=8,
            max_fit_rms_deg=0.1,
        )
    )
    y_values = (0.0, 10.0, -10.0, 0.0)
    for index, time_ms in enumerate((0, 100, 200, 300)):
        diagnostics = _observe(
            estimator,
            time_ms,
            0.01 * time_ms,
            y_values[index],
        )

    assert diagnostics.x.valid
    assert diagnostics.y.invalid_reason is TrackingEstimateInvalidReason.FIT_RMS
    assert diagnostics.y.fit_rms_deg is not None
    assert diagnostics.y.fit_rms_deg > 0.1
    assert diagnostics.y.sample_count == 4

    diagnostics = _observe(estimator, 400, 4.0, 0.0)
    assert diagnostics.y.sample_count == 5
    assert estimator.sample_count == 5


def test_target_rate_limit_invalidates_without_clamping_and_axes_are_independent() -> (
    None
):
    estimator = TrackingEstimator(
        _config(
            window_ms=400,
            max_target_rate_x_deg_s=50.0,
            max_target_rate_y_deg_s=50.0,
        )
    )
    for time_ms in (0, 100, 200, 300):
        time_s = time_ms / 1000.0
        diagnostics = _observe(
            estimator,
            time_ms,
            100.0 * time_s,
            10.0 * time_s,
        )

    assert not diagnostics.x.valid
    assert (
        diagnostics.x.invalid_reason is TrackingEstimateInvalidReason.TARGET_RATE_LIMIT
    )
    assert diagnostics.x.target_rate_deg_s == pytest.approx(100.0)
    assert diagnostics.y.valid
    assert diagnostics.y.target_rate_deg_s == pytest.approx(10.0)


def test_out_of_order_observation_is_rejected_without_destroying_history() -> None:
    estimator = TrackingEstimator(_config(window_ms=400, max_samples=8))
    for time_ms in (0, 100, 200, 300):
        diagnostics = _observe(estimator, time_ms, time_ms / 100.0, 0.0)
    assert diagnostics.x.valid

    rejected = _observe(estimator, 250, 99.0, 99.0)
    assert rejected.x.invalid_reason is TrackingEstimateInvalidReason.TIMESTAMP_INVALID
    assert rejected.y.invalid_reason is TrackingEstimateInvalidReason.TIMESTAMP_INVALID
    assert rejected.x.prediction_offset_deg == 0.0
    assert rejected.x.sample_count == 4
    assert estimator.sample_count == 4

    recovered = _observe(estimator, 400, 4.0, 0.0)
    assert recovered.x.valid
    assert recovered.x.sample_count == 5
    assert recovered.x.relative_rate_deg_s == pytest.approx(10.0)


def test_platform_unavailable_invalidates_rate_without_clearing_fit() -> None:
    estimator = TrackingEstimator(_config(window_ms=400))
    for time_ms in (0, 100, 200, 300):
        timestamp_ns = BASE_NS + time_ms * 1_000_000
        diagnostics = estimator.observe(
            TrackingError(TARGET, time_ms / 100.0, 0.0, timestamp_ns),
            None,
        )

    assert diagnostics.x.relative_rate_deg_s == pytest.approx(10.0)
    assert diagnostics.x.target_rate_deg_s is None
    assert (
        diagnostics.x.invalid_reason
        is TrackingEstimateInvalidReason.PLATFORM_UNAVAILABLE
    )
    assert diagnostics.x.prediction_offset_deg == 0.0
    assert estimator.sample_count == 4


def test_estimator_config_change_resets_history_but_control_only_change_does_not() -> (
    None
):
    config = _config(window_ms=400, max_samples=8)
    estimator = TrackingEstimator(config)
    for time_ms in (0, 100, 200, 300):
        _observe(estimator, time_ms, time_ms / 100.0, 0.0)

    control_only = replace(
        config,
        feed_forward=replace(config.feed_forward, enabled=True, gain_x=1.0),
        prediction=replace(config.prediction, enabled=True, horizon_ms=50),
    )
    estimator.replace_config(control_only)
    assert estimator.sample_count == 4

    estimator_changed = replace(
        control_only,
        velocity_estimator=replace(control_only.velocity_estimator, window_ms=250),
    )
    estimator.replace_config(estimator_changed)
    assert estimator.sample_count == 0
    assert estimator.diagnostics.target == TARGET
    assert (
        estimator.diagnostics.x.invalid_reason
        is TrackingEstimateInvalidReason.INSUFFICIENT_SAMPLES
    )


def test_new_target_starts_new_observation_history() -> None:
    estimator = TrackingEstimator(_config())
    for time_ms in (0, 100, 200, 300):
        _observe(estimator, time_ms, time_ms / 100.0, 0.0)

    new_target = TargetRef(CameraRole.OVERVIEW, 1, 8)
    diagnostics = _observe(
        estimator,
        400,
        0.0,
        0.0,
        target=new_target,
    )

    assert diagnostics.target == new_target
    assert diagnostics.x.sample_count == 1
    assert (
        diagnostics.x.invalid_reason
        is TrackingEstimateInvalidReason.INSUFFICIENT_SAMPLES
    )
