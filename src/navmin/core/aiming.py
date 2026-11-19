"""Generation-bound working-frame geometry for Core aiming."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class FocalScaleCameraModel:
    """Scale focal lengths of an existing working-frame model without remapping."""

    base: CameraModel
    focal_scale: float

    def __post_init__(self) -> None:
        scale = self.focal_scale
        if isinstance(scale, bool) or not isinstance(scale, int | float):
            raise TypeError("focal_scale must be a number")
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError("focal_scale must be a finite number > 0")

    @property
    def projection_matrix(self):
        projection = getattr(self.base, "projection_matrix", None)
        if projection is None:
            return None
        scale = float(self.focal_scale)
        rows = [list(row) for row in projection]
        rows[0][0] *= scale
        rows[1][1] *= scale
        return tuple(tuple(float(item) for item in row) for row in rows)

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        ray = self.base.pixel_to_ray(x_px, y_px)
        scale = float(self.focal_scale)
        x = ray.x / scale
        y = ray.y / scale
        z = ray.z
        norm = math.sqrt(x * x + y * y + z * z)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("scaled camera ray must be finite and non-zero")
        return CameraRay(x / norm, y / norm, z / norm)

    def ray_to_pixel(self, ray: CameraRay) -> tuple[float, float]:
        if not isinstance(ray, CameraRay):
            raise TypeError("ray must be CameraRay")
        scale = float(self.focal_scale)
        return self.base.ray_to_pixel(
            CameraRay(ray.x * scale, ray.y * scale, ray.z)
        )


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

    @staticmethod
    def tracked_center(tracked: TrackedObject) -> PixelPoint:
        """Return the current observed bbox center without extrapolation."""
        return (
            tracked.bbox.x + tracked.bbox.width / 2.0,
            tracked.bbox.y + tracked.bbox.height / 2.0,
        )

    def tracking_error(
        self,
        *,
        camera_model: CameraModel,
        frame: FramePacket,
        tracked: TrackedObject,
        target: TargetRef,
        observation_timestamp_ns: int | None = None,
    ) -> TrackingError:
        center = self.tracked_center(tracked)
        error_x, error_y = self.angular_delta(
            camera=frame.camera,
            camera_model=camera_model,
            frame=frame,
            target=center,
        )
        timestamp_ns = (
            frame.receive_timestamp_ns
            if observation_timestamp_ns is None
            else observation_timestamp_ns
        )
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("observation_timestamp_ns must be an integer >= 0")
        return TrackingError(
            target=target,
            error_x_deg=error_x,
            error_y_deg=error_y,
            timestamp_ns=timestamp_ns,
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
        if camera is CameraRole.STEREO_RIGHT:
            return self._config.aim_points.stereo_right
        raise ValueError(f"unsupported aiming camera: {camera.value}")

    @staticmethod
    def _turret_angles(ray: CameraRay) -> TurretAngles:
        if not all(math.isfinite(value) for value in (ray.x, ray.y, ray.z)):
            raise ValueError("CameraModel returned a non-finite ray")
        if ray.x == 0.0 and ray.y == 0.0 and ray.z == 0.0:
            raise ValueError("CameraModel returned a zero-length ray")
        horizontal = math.degrees(math.atan2(ray.x, ray.z))
        vertical = -math.degrees(math.atan2(ray.y, math.hypot(ray.x, ray.z)))
        return horizontal, vertical


__all__ = ["Aiming", "FocalScaleCameraModel", "PixelPoint", "TurretAngles"]
