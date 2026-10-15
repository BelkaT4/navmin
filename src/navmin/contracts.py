"""Shared immutable contracts used across NavMin modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol

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


@dataclass(frozen=True)
class TargetSelectionRequest:
    selection_id: int
    camera: CameraRole
    generation: int
    frame_id: int
    x_px: float
    y_px: float


@dataclass(frozen=True)
class ClearTargetSelection:
    camera: CameraRole
    generation: int


VisionTargetControlCommand = TargetSelectionRequest | ClearTargetSelection


class TargetAcquisitionFailureReason(Enum):
    NO_TARGET_AT_POINT = "no_target_at_point"
    STALE_SELECTION = "stale_selection"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACQUISITION_FAILED = "acquisition_failed"


@dataclass(frozen=True)
class TargetAcquired:
    selection_id: int
    target: TargetRef
    request_frame_id: int
    decision_frame_id: int
    decision_frame_timestamp_ns: int


@dataclass(frozen=True)
class TargetAcquisitionFailed:
    selection_id: int
    camera: CameraRole
    generation: int
    request_frame_id: int
    reason: TargetAcquisitionFailureReason
    decision_frame_id: int | None
    decision_frame_timestamp_ns: int | None


class TargetTrackingEndReason(Enum):
    LOST_UNRECOVERED = "lost_unrecovered"


@dataclass(frozen=True)
class TargetTrackingEnded:
    target: TargetRef
    reason: TargetTrackingEndReason
    decision_frame_id: int
    decision_frame_timestamp_ns: int


TargetLifecycleEvent = TargetAcquired | TargetAcquisitionFailed | TargetTrackingEnded


class DistanceSource(Enum):
    STEREO = "stereo"
    MANUAL = "manual"


class PlatformMotionSource(Enum):
    COMMAND_MODEL = "command-model"


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
class TrackingControlSample:
    target: TargetRef
    real_error_x_deg: float
    real_error_y_deg: float
    target_rate_x_deg_s: float | None
    target_rate_y_deg_s: float | None
    prediction_offset_x_deg: float
    prediction_offset_y_deg: float
    rate_valid_x: bool
    rate_valid_y: bool
    observation_timestamp_ns: int


@dataclass(frozen=True)
class PlatformMotionRate:
    timestamp_ns: int
    rate_x_deg_s: float | None
    rate_y_deg_s: float | None


@dataclass(frozen=True)
class PlatformMotionAnchor:
    timestamp_ns: int
    modeled_rate_x_deg_s: float
    modeled_rate_y_deg_s: float
    commanded_rate_x_deg_s: float
    commanded_rate_y_deg_s: float
    acceleration_x_deg_s2: float
    acceleration_y_deg_s2: float


@dataclass(frozen=True)
class PlatformMotionSnapshot:
    source: PlatformMotionSource
    valid: bool
    anchors: tuple[PlatformMotionAnchor, ...]

    def rate_at(self, timestamp_ns: int) -> PlatformMotionRate | None:
        """Reconstruct command-modeled platform rate at an observation time."""
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
            or not self.valid
            or not self.anchors
            or timestamp_ns < self.anchors[0].timestamp_ns
        ):
            return None

        anchor = self.anchors[0]
        for candidate in self.anchors[1:]:
            if candidate.timestamp_ns > timestamp_ns:
                break
            anchor = candidate
        elapsed_s = (timestamp_ns - anchor.timestamp_ns) / 1_000_000_000.0
        if not all(
            isfinite(value)
            for value in (
                anchor.modeled_rate_x_deg_s,
                anchor.modeled_rate_y_deg_s,
                anchor.commanded_rate_x_deg_s,
                anchor.commanded_rate_y_deg_s,
                anchor.acceleration_x_deg_s2,
                anchor.acceleration_y_deg_s2,
            )
        ):
            return None
        return PlatformMotionRate(
            timestamp_ns=timestamp_ns,
            rate_x_deg_s=_approach_rate(
                anchor.modeled_rate_x_deg_s,
                anchor.commanded_rate_x_deg_s,
                anchor.acceleration_x_deg_s2 * elapsed_s,
            ),
            rate_y_deg_s=_approach_rate(
                anchor.modeled_rate_y_deg_s,
                anchor.commanded_rate_y_deg_s,
                anchor.acceleration_y_deg_s2 * elapsed_s,
            ),
        )


def _approach_rate(current: float, target: float, maximum_delta: float) -> float:
    if current < target:
        return min(target, current + maximum_delta)
    return max(target, current - maximum_delta)


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
    max_speed_x_deg_s: float | None
    max_speed_y_deg_s: float | None
    acceleration_x_deg_s2: float | None
    acceleration_y_deg_s2: float | None


@dataclass(frozen=True)
class ConfigUpdate[T]:
    revision: int
    config: T
