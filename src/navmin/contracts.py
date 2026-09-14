"""Shared immutable contracts used across NavMin modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

import numpy as np


class CameraRole(Enum):
    OVERVIEW = "overview"
    STEREO_LEFT = "stereo_left"
    STEREO_RIGHT = "stereo_right"


@dataclass(frozen=True)
class FramePacket:
    camera: CameraRole
    generation: int
    frame_id: int
    capture_id: int | None
    receive_timestamp_ns: int
    image: np.ndarray


@dataclass(frozen=True)
class CameraRay:
    x: float
    y: float
    z: float


class CameraModel(Protocol):
    """Geometry model for one corrected working-frame camera session."""

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        """Return a normalized camera ray for a working-frame pixel."""
        ...


@dataclass(frozen=True)
class CameraSessionStarted:
    camera: CameraRole
    generation: int
    camera_model: CameraModel
    timestamp_ns: int


class CameraState(Enum):
    STARTING = "starting"
    ONLINE = "online"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class CameraStatus:
    camera: CameraRole
    state: CameraState
    generation: int | None
    last_receive_timestamp_ns: int | None
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    bbox: BBox
    velocity_x_px_s: float
    velocity_y_px_s: float
    age_frames: int


@dataclass(frozen=True)
class VisionResult:
    frame: FramePacket
    tracked_objects: tuple[TrackedObject, ...]
    processing_time_ns: int


@dataclass(frozen=True)
class TargetRef:
    camera: CameraRole
    generation: int
    track_id: int


class DistanceSource(Enum):
    STEREO = "stereo"
    MANUAL = "manual"


@dataclass(frozen=True)
class DistanceResult:
    target: TargetRef
    distance_m: float
    source: DistanceSource
    source_frame_id: int | None
    capture_id: int | None
    measured_timestamp_ns: int


@dataclass(frozen=True)
class MoveRelativeCommand:
    delta_x_deg: float
    delta_y_deg: float


class TurretControlMode(Enum):
    RELATIVE = "relative"
    TRACKING = "tracking"


@dataclass(frozen=True)
class TrackingError:
    target: TargetRef
    error_x_deg: float
    error_y_deg: float
    timestamp_ns: int


@dataclass(frozen=True)
class AxisVelocitySetpoint:
    velocity_x_deg_s: float
    velocity_y_deg_s: float
    timestamp_ns: int


class TurretConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    ERROR = "error"


class MotorState(Enum):
    UNKNOWN = "unknown"
    OFF = "off"
    ON = "on"


@dataclass(frozen=True)
class TurretState:
    connection_state: TurretConnectionState
    motor_state: MotorState
    control_mode: TurretControlMode
    max_speed_x_deg_s: float
    max_speed_y_deg_s: float
    acceleration_x_deg_s2: float
    acceleration_y_deg_s2: float


T = TypeVar("T")


@dataclass(frozen=True)
class ConfigUpdate(Generic[T]):
    revision: int
    config: T
