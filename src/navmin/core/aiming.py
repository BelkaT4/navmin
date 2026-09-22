"""Generation-bound working-frame geometry for Core aiming."""

from __future__ import annotations

import math

from navmin.config.models import AimingConfig, AimPointConfig
from navmin.contracts import (
    CameraModel,
    CameraRay,
    CameraRole,
    FramePacket,
    MoveRelativeCommand,
    TargetRef,
    TrackedObject,
    TrackingError,
)

type PixelPoint = tuple[float, float]
type TurretAngles = tuple[float, float]


class Aiming:
    """Convert corrected working-frame pixels into logical Turret angles."""

    def __init__(self, config: AimingConfig) -> None:
        if not isinstance(config, AimingConfig):
            raise TypeError("config must be AimingConfig")
        self._config = config

    @property
    def config(self) -> AimingConfig:
        return self._config

    def replace_config(self, config: AimingConfig) -> None:
        """Replace owner-local dynamic aiming settings as one immutable snapshot."""
        if not isinstance(config, AimingConfig):
            raise TypeError("config must be AimingConfig")
        self._config = config

    def aim_point(self, camera: CameraRole, frame: FramePacket) -> PixelPoint:
        point = self._aim_point_config(camera)
        height, width = frame.image.shape[:2]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        x_px = center_x if point.x_px is None else float(point.x_px)
        y_px = center_y if point.y_px is None else float(point.y_px)
        return x_px, y_px

    def relative_command(
        self,
        *,
        camera_model: CameraModel,
        frame: FramePacket,
        x_px: float,
        y_px: float,
    ) -> MoveRelativeCommand:
        delta_x, delta_y = self.angular_delta(
            camera=frame.camera,
            camera_model=camera_model,
            frame=frame,
            target=(x_px, y_px),
        )
        return MoveRelativeCommand(delta_x_deg=delta_x, delta_y_deg=delta_y)

    def lead_point(self, tracked: TrackedObject) -> PixelPoint:
        lead_time_s = self._config.lead_time_ms / 1000.0
        center_x = tracked.bbox.x + tracked.bbox.width / 2.0
        center_y = tracked.bbox.y + tracked.bbox.height / 2.0
        return (
            center_x + tracked.velocity_x_px_s * lead_time_s,
            center_y + tracked.velocity_y_px_s * lead_time_s,
        )

    def tracking_error(
        self,
        *,
        camera_model: CameraModel,
        frame: FramePacket,
        tracked: TrackedObject,
        target: TargetRef,
    ) -> TrackingError:
        lead = self.lead_point(tracked)
        error_x, error_y = self.angular_delta(
            camera=frame.camera,
            camera_model=camera_model,
            frame=frame,
            target=lead,
        )
        return TrackingError(
            target=target,
            error_x_deg=error_x,
            error_y_deg=error_y,
            timestamp_ns=frame.receive_timestamp_ns,
        )

    def angular_delta(
        self,
        *,
        camera: CameraRole,
        camera_model: CameraModel,
        frame: FramePacket,
        target: PixelPoint,
    ) -> TurretAngles:
        aim_x, aim_y = self.aim_point(camera, frame)
        target_angles = self._turret_angles(camera_model.pixel_to_ray(*target))
        aim_angles = self._turret_angles(camera_model.pixel_to_ray(aim_x, aim_y))
        delta_x = target_angles[0] - aim_angles[0]
        delta_y = target_angles[1] - aim_angles[1]
        if not math.isfinite(delta_x) or not math.isfinite(delta_y):
            raise ValueError("aiming produced a non-finite angular delta")
        return delta_x, delta_y

    def _aim_point_config(self, camera: CameraRole) -> AimPointConfig:
        if camera is CameraRole.OVERVIEW:
            return self._config.aim_points.overview
        if camera is CameraRole.STEREO_LEFT:
            return self._config.aim_points.stereo_left
        raise ValueError("Stereo Right is not a normal aiming camera")

    @staticmethod
    def _turret_angles(ray: CameraRay) -> TurretAngles:
        if not all(math.isfinite(value) for value in (ray.x, ray.y, ray.z)):
            raise ValueError("CameraModel returned a non-finite ray")
        if ray.x == 0.0 and ray.y == 0.0 and ray.z == 0.0:
            raise ValueError("CameraModel returned a zero-length ray")
        horizontal = math.degrees(math.atan2(ray.x, ray.z))
        vertical = -math.degrees(math.atan2(ray.y, math.hypot(ray.x, ray.z)))
        return horizontal, vertical


__all__ = ["Aiming", "PixelPoint", "TurretAngles"]
