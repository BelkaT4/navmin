from __future__ import annotations

import math

import numpy as np

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
    lead_time_ms: int = 150,
    overview: AimPointConfig | None = None,
    stereo_left: AimPointConfig | None = None,
) -> AimingConfig:
    return AimingConfig(
        lead_time_ms=lead_time_ms,
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


def test_lead_time_zero_uses_bbox_center() -> None:
    tracked = TrackedObject(3, BBox(10, 20, 8, 6), 200.0, -100.0, 5)

    assert Aiming(_config(lead_time_ms=0)).lead_point(tracked) == (14.0, 23.0)


def test_positive_velocity_and_lead_time_shift_lead_point() -> None:
    tracked = TrackedObject(3, BBox(10, 20, 8, 6), 20.0, 10.0, 5)

    assert Aiming(_config(lead_time_ms=250)).lead_point(tracked) == (19.0, 25.5)


def test_tracking_error_contains_target_frame_timestamp_and_finite_angles() -> None:
    target = TargetRef(CameraRole.OVERVIEW, 1, 3)
    tracked = TrackedObject(3, BBox(50, 40, 10, 8), 4.0, -2.0, 8)
    model = _RayModel()
    frame = _frame(timestamp_ns=987654321)

    error = Aiming(_config(lead_time_ms=100)).tracking_error(
        camera_model=model,
        frame=frame,
        tracked=tracked,
        target=target,
    )

    assert error.target == target
    assert error.timestamp_ns == frame.receive_timestamp_ns
    assert math.isfinite(error.error_x_deg)
    assert math.isfinite(error.error_y_deg)
