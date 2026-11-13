from __future__ import annotations

import cv2
import numpy as np
import pytest

from navmin.contracts import BBox, CameraRay
from navmin.diagnostics.video_latency_calibration import (
    OpticalFlowLatencyConfig,
    OpticalFlowRateObservation,
    OpticalFlowRateObservationY,
    VideoLatencyCalibrationError,
    estimate_optical_flow_rate_x,
    estimate_optical_flow_rate_y,
    estimate_video_latency,
    estimate_video_latency_y,
    run_synthetic_latency_calibration,
    run_video_latency_calibration_proof,
)
from navmin.turret.platform_motion import CommandMotionModel


class _SimpleCameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay((x_px - 79.5) / 500.0, (y_px - 59.5) / 500.0, 1.0)


def test_synthetic_optical_flow_recovers_known_latency() -> None:
    result, estimate = run_synthetic_latency_calibration(
        known_latency_ms=150,
        seed=123,
    )

    assert result.valid
    assert result.estimated_latency_ms == pytest.approx(150, abs=5)
    assert result.absolute_error_ms is not None and result.absolute_error_ms <= 5
    assert result.flow_observations >= 70
    assert result.median_flow_inliers >= 20
    assert estimate.best_score is not None
    assert estimate.best_score.correlation >= 0.99


def test_calibration_proof_reports_mean_and_spread() -> None:
    result = run_video_latency_calibration_proof()

    assert result.passed
    assert result.aggregate.valid_run_count == 5
    assert result.aggregate.mean_latency_ms == pytest.approx(150.0, abs=5.0)
    assert result.aggregate.median_latency_ms == pytest.approx(150.0, abs=5.0)
    assert result.aggregate.stddev_latency_ms is not None
    assert result.aggregate.stddev_latency_ms > 0.0
    assert result.max_absolute_estimation_error_ms is not None
    assert result.max_absolute_estimation_error_ms <= 5
    assert result.insufficient_excitation_rejected


def test_latency_estimator_rejects_unexcited_platform() -> None:
    model = CommandMotionModel(
        acceleration_x_deg_s2=40.0,
        acceleration_y_deg_s2=40.0,
    )
    base = 1_000_000_000
    model.mark_known_zero(timestamp_ns=base)
    observations = tuple(
        OpticalFlowRateObservation(
            receive_midpoint_timestamp_ns=base + index * 50_000_000,
            source_interval_ns=50_000_000,
            background_rate_x_deg_s=0.0,
            translation_x_px=0.0,
            inliers=80,
            tracked_points=100,
        )
        for index in range(1, 70)
    )

    result = estimate_video_latency(observations, model.snapshot())

    assert not result.valid
    assert result.reason == "insufficient-excitation"
    assert result.estimated_latency_ms is None


def test_optical_flow_supports_target_exclusion_mask() -> None:
    rng = np.random.default_rng(42)
    previous = np.clip(rng.normal(127.0, 50.0, (120, 160)), 0, 255).astype(np.uint8)
    previous = np.repeat(previous[:, :, None], 3, axis=2)
    transform = np.asarray(((1.0, 0.0, -4.0), (0.0, 1.0, 0.0)), dtype=np.float32)
    current = cv2.warpAffine(
        previous,
        transform,
        (160, 120),
        borderMode=cv2.BORDER_REFLECT101,
    )
    # A foreground rectangle is deliberately excluded from feature selection.
    previous[35:85, 55:105] = 255
    current[35:85, 70:120] = 0

    observation = estimate_optical_flow_rate_x(
        previous,
        current,
        previous_receive_timestamp_ns=1_000_000_000,
        current_receive_timestamp_ns=1_050_000_000,
        source_interval_ns=50_000_000,
        camera_model=_SimpleCameraModel(),
        exclusions=(BBox(50, 30, 75, 60),),
        config=OpticalFlowLatencyConfig(min_inliers=8),
    )

    assert observation.inliers >= 8
    assert observation.translation_x_px == pytest.approx(-4.0, abs=0.8)
    assert observation.background_rate_x_deg_s < 0.0


def test_optical_flow_rejects_featureless_frames() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    with pytest.raises(VideoLatencyCalibrationError, match="background features"):
        estimate_optical_flow_rate_x(
            frame,
            frame.copy(),
            previous_receive_timestamp_ns=1_000_000_000,
            current_receive_timestamp_ns=1_050_000_000,
            source_interval_ns=50_000_000,
            camera_model=_SimpleCameraModel(),
        )


def test_optical_flow_y_uses_logical_vertical_angle_convention() -> None:
    rng = np.random.default_rng(77)
    previous = np.clip(rng.normal(127.0, 50.0, (120, 160)), 0, 255).astype(np.uint8)
    previous = np.repeat(previous[:, :, None], 3, axis=2)
    transform = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 4.0)), dtype=np.float32)
    current = cv2.warpAffine(
        previous, transform, (160, 120), borderMode=cv2.BORDER_REFLECT101
    )

    observation = estimate_optical_flow_rate_y(
        previous,
        current,
        previous_receive_timestamp_ns=1_000_000_000,
        current_receive_timestamp_ns=1_050_000_000,
        source_interval_ns=50_000_000,
        camera_model=_SimpleCameraModel(),
        config=OpticalFlowLatencyConfig(min_inliers=8),
    )

    assert observation.translation_y_px == pytest.approx(4.0, abs=0.8)
    # Positive image Y means the background appears lower; logical turret pitch
    # uses the opposite sign, matching Aiming._turret_angles.
    assert observation.background_rate_y_deg_s < 0.0


def test_y_latency_estimator_uses_platform_y_not_x() -> None:
    model = CommandMotionModel(
        acceleration_x_deg_s2=1000.0,
        acceleration_y_deg_s2=1000.0,
    )
    base = 1_000_000_000
    model.mark_known_zero(timestamp_ns=base)
    model.record_velocity_command(
        timestamp_ns=base,
        commanded_rate_x_deg_s=0.0,
        commanded_rate_y_deg_s=10.0,
    )
    model.record_velocity_command(
        timestamp_ns=base + 500_000_000,
        commanded_rate_x_deg_s=0.0,
        commanded_rate_y_deg_s=-10.0,
    )
    latency_ms = 100
    observations = []
    for index in range(8, 28):
        receive_ns = base + index * 50_000_000
        motion = model.snapshot().rate_at(receive_ns - latency_ms * 1_000_000)
        assert motion is not None and motion.rate_y_deg_s is not None
        observations.append(
            OpticalFlowRateObservationY(
                receive_midpoint_timestamp_ns=receive_ns,
                source_interval_ns=50_000_000,
                background_rate_y_deg_s=-motion.rate_y_deg_s,
                translation_y_px=0.0,
                inliers=80,
                tracked_points=100,
            )
        )

    result = estimate_video_latency_y(
        tuple(observations),
        model.snapshot(),
        config=OpticalFlowLatencyConfig(
            search_min_ms=0,
            search_max_ms=200,
            search_step_ms=5,
            min_observations=8,
            min_correlation=0.9,
            max_normalized_rmse=0.2,
        ),
    )

    assert result.valid
    assert result.estimated_latency_ms == pytest.approx(latency_ms, abs=5)
