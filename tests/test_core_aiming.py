from __future__ import annotations

import math

import numpy as np
import pytest

from navmin.config.models import AimingConfig, AimPointConfig, AimPointsConfig
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    FramePacket,
    TargetRef,
    TrackedObject,
)
from navmin.core.aiming import Aiming


class _RayModel:
    def __init__(self, *, cx: float = 50.0, cy: float = 40.0) -> None:
        self.cx = cx
        self.cy = cy
        self.calls: list[tuple[float, float]] = []

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        self.calls.append((x_px, y_px))
        return CameraRay((x_px - self.cx) / 50.0, (y_px - self.cy) / 40.0, 1.0)


class _NonLinearRayModel(_RayModel):
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        self.calls.append((x_px, y_px))
        dx = (x_px - self.cx) / 10.0
        dy = (y_px - self.cy) / 10.0
        return CameraRay(dx * dx * dx, dy * dy * dy, 1.0)


def _config(
    *,
    overview: AimPointConfig | None = None,
    stereo_left: AimPointConfig | None = None,
) -> AimingConfig:
    return AimingConfig(
        target_lost_timeout_ms=500,
        aim_points=AimPointsConfig(
            overview=overview or AimPointConfig(50, 40),
            stereo_left=stereo_left or AimPointConfig(50, 40),
        ),
    )


def _frame(
    camera: CameraRole = CameraRole.OVERVIEW,
    *,
    width: int = 101,
    height: int = 81,
    timestamp_ns: int = 1234,
) -> FramePacket:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.flags.writeable = False
    return FramePacket(camera, 1, 7, None, timestamp_ns, image)


def test_aim_point_uses_configured_pixel_and_independent_center_fallback() -> None:
    aiming = Aiming(
        _config(
            overview=AimPointConfig(17, None),
            stereo_left=AimPointConfig(None, 23),
        )
    )

    assert aiming.aim_point(CameraRole.OVERVIEW, _frame()) == (17.0, 40.0)
    assert aiming.aim_point(CameraRole.STEREO_LEFT, _frame(CameraRole.STEREO_LEFT)) == (
        50.0,
        23.0,
    )


def test_click_on_aim_point_is_zero_delta() -> None:
    model = _RayModel()
    frame = _frame()
    command = Aiming(_config()).relative_command(
        camera_model=model,
        frame=frame,
        x_px=50.0,
        y_px=40.0,
    )

    assert command.delta_x_deg == 0.0
    assert command.delta_y_deg == 0.0


def test_logical_axis_signs_are_right_positive_and_image_down_negative() -> None:
    aiming = Aiming(_config())
    model = _RayModel()
    frame = _frame()

    right = aiming.relative_command(camera_model=model, frame=frame, x_px=60, y_px=40)
    left = aiming.relative_command(camera_model=model, frame=frame, x_px=40, y_px=40)
    down = aiming.relative_command(camera_model=model, frame=frame, x_px=50, y_px=50)
    up = aiming.relative_command(camera_model=model, frame=frame, x_px=50, y_px=30)

    assert right.delta_x_deg > 0.0
    assert left.delta_x_deg < 0.0
    assert down.delta_y_deg < 0.0
    assert up.delta_y_deg > 0.0


def test_angular_delta_comes_from_camera_model_rays_not_linear_pixels() -> None:
    aiming = Aiming(_config())
    model = _NonLinearRayModel()
    frame = _frame()

    near = aiming.relative_command(camera_model=model, frame=frame, x_px=55, y_px=40)
    far = aiming.relative_command(camera_model=model, frame=frame, x_px=60, y_px=40)

    assert model.calls == [(55, 40), (50.0, 40.0), (60, 40), (50.0, 40.0)]
    assert not math.isclose(far.delta_x_deg, near.delta_x_deg * 2.0)


def test_tracked_center_uses_current_bbox_center() -> None:
    tracked = TrackedObject(3, BBox(10, 20, 8, 6), 200.0, -100.0, 5)

    assert Aiming(_config()).tracked_center(tracked) == (14.0, 23.0)


def test_tracking_error_is_independent_of_tracked_pixel_velocity() -> None:
    target = TargetRef(CameraRole.OVERVIEW, 1, 3)
    stationary = TrackedObject(3, BBox(10, 20, 8, 6), 0.0, 0.0, 5)
    moving = TrackedObject(3, BBox(10, 20, 8, 6), 200.0, -100.0, 5)
    model = _RayModel()
    frame = _frame()

    stationary_error = Aiming(_config()).tracking_error(
        camera_model=model,
        frame=frame,
        tracked=stationary,
        target=target,
    )
    moving_error = Aiming(_config()).tracking_error(
        camera_model=model,
        frame=frame,
        tracked=moving,
        target=target,
    )

    assert moving_error == stationary_error


def test_tracking_error_contains_target_frame_timestamp_and_finite_angles() -> None:
    target = TargetRef(CameraRole.OVERVIEW, 1, 3)
    tracked = TrackedObject(3, BBox(50, 40, 10, 8), 4.0, -2.0, 8)
    model = _RayModel()
    frame = _frame(timestamp_ns=987654321)

    error = Aiming(_config()).tracking_error(
        camera_model=model,
        frame=frame,
        tracked=tracked,
        target=target,
    )

    assert error.target == target
    assert error.timestamp_ns == frame.receive_timestamp_ns
    assert math.isfinite(error.error_x_deg)
    assert math.isfinite(error.error_y_deg)


def test_motion_calibration_separates_scale_from_directional_backlash() -> None:
    from navmin.core.motion_calibration import MotionObservation, estimate_motion

    samples = [
        MotionObservation('x', CameraRole.OVERVIEW, 1, 1, 2, command, measured, reversal)
        for command, measured, reversal in [
            (5, 1, None), (2, 1.6, False), (5, 4, False), (10, 8, False),
            (-2, -1.6, False), (-5, -4, False), (-10, -8, False),
            (5, 3.7, True), (10, 7.7, True),
            (-5, -3.5, True), (-10, -7.5, True),
        ]
    ]
    estimate = estimate_motion(samples, 'x')
    assert estimate.gain == pytest.approx(0.8)
    assert estimate.correction == pytest.approx(1.25)
    assert estimate.rms_deg == pytest.approx(0)
    assert estimate.backlash_positive_deg == pytest.approx(0.3)
    assert estimate.backlash_negative_deg == pytest.approx(0.5)
    with pytest.raises(ValueError):
        estimate_motion(samples[:3], 'x')


def test_motion_calibration_config_uses_startup_baseline_and_preserves_other_fields() -> None:
    from dataclasses import replace
    from pathlib import Path

    from navmin.config import load_config
    from navmin.core.motion_calibration import MotionEstimate, corrected_config

    config = load_config(Path(__file__).resolve().parents[1] / 'config/defaults/navmin.json')
    baseline = config.turret.axes.x
    estimate = MotionEstimate(0.8, 0, None, None, ((-0.8, -1.0), (0.0, 0.0), (0.8, 1.0)))
    corrected = corrected_config(config, 'x', baseline, estimate)
    assert corrected.turret.axes.x.relative_motion_curve == estimate.curve
    assert corrected.turret.axes.x.full_steps_per_revolution == baseline.full_steps_per_revolution
    assert corrected.turret.axes.y == config.turret.axes.y
    assert corrected.aiming == config.aiming
    assert corrected.turret.axes.x.microstep_divider == baseline.microstep_divider
    with pytest.raises(ValueError):
        corrected_config(corrected, 'x', baseline, MotionEstimate(0.8, 0, None, None))
    with pytest.raises(ValueError):
        corrected_config(config, 'x', replace(baseline, invert=not baseline.invert), MotionEstimate(1, 0, None, None))


def test_nonlinear_motion_curve_reconstructs_angles_without_extrapolation() -> None:
    from navmin.core.motion_calibration import MotionObservation, estimate_motion
    from navmin.motion_curve import interpolate_motion

    samples = [
        MotionObservation('x', CameraRole.OVERVIEW, 1, 1, 2, command, measured, False)
        for command, measured in [(-10, -7), (-5, -4.5), (-2, -1.9), (2, 1.8), (5, 4), (10, 6)]
    ]
    estimate = estimate_motion(samples, 'x')
    assert estimate.rms_deg > 0.1  # A single scale cannot explain these measurements.
    assert interpolate_motion(estimate.curve, 1.8) == pytest.approx(2)
    assert interpolate_motion(estimate.curve, 6) == pytest.approx(10)
    assert interpolate_motion(estimate.curve, 5) == pytest.approx(7.5)
    assert interpolate_motion(estimate.curve, -5.75) == pytest.approx(-7.5)
    assert interpolate_motion(estimate.curve, 0) == 0
    with pytest.raises(ValueError):
        interpolate_motion(estimate.curve, 6.01)
    with pytest.raises(ValueError):
        interpolate_motion(estimate.curve, -7.01)
    samples[-1] = MotionObservation('x', CameraRole.OVERVIEW, 1, 1, 2, 10, 3, False)
    with pytest.raises(ValueError):
        estimate_motion(samples, 'x')
