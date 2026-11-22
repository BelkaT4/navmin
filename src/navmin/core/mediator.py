"""Core mediator for coordinated Vision target lifecycle and Turret intents."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite, radians, tan
from threading import Lock
from time import monotonic_ns
from typing import Protocol

from navmin.camera_time_sync import (
    CameraClockObservation,
    CameraClockSessionError,
    CameraClockTracker,
    CameraPcTimeBasis,
)
from navmin.concurrency import LatestValue, TargetControlChannel
from navmin.config.models import AimingConfig, TrackingControlConfig, UiConfig
from navmin.contracts import (
    CameraModel,
    CameraRole,
    CameraSessionStarted,
    CameraState,
    CameraStatus,
    ClearTargetSelection,
    ConfigUpdate,
    DistanceResult,
    DistanceSource,
    FramePacket,
    FrameTimestampSource,
    MotorState,
    MoveRelativeCommand,
    PlatformMotionSnapshot,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetCandidateSelectionRequest,
    TargetCandidateSet,
    TargetLifecycleEvent,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TrackedObject,
    TrackingControlSample,
    TrackingError,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.diagnostic_logging import emit_event
from navmin.time_sync import ClockMappingState, ClockSampleOutcome
from navmin.vision.stereo_target_join import StereoPairingKind

from .aiming import Aiming, FocalScaleCameraModel
from .distance import DistanceMeasurementMode, quantize_constant_distance
from .session_gate import CameraSessionGate
from .tracking_estimator import (
    TrackingControlVisualization,
    TrackingEstimator,
    TrackingEstimatorDiagnostics,
)

_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)
_OBSERVABLE_CAMERAS = (*_NORMAL_CAMERAS, CameraRole.STEREO_RIGHT)
_DISCONNECTED_CAMERA_STATES = (
    CameraState.RECONNECTING,
    CameraState.ERROR,
    CameraState.STOPPED,
)
_LOGGER = logging.getLogger(__name__)
_CAMERA_CLOCK_DEBUG_INTERVAL_NS = 1_000_000_000


class StereoDistanceRejectionReason(Enum):
    """Existing Core revalidation decisions exposed for diagnostics."""

    MODE_MISMATCH = "mode_mismatch"
    INTENT_REVISION_CHANGED = "intent_revision_changed"
    TARGET_CHANGED = "target_changed"
    GENERATION_INVALID = "generation_invalid"
    MAIN_CAMERA_MISMATCH = "main_camera_mismatch"
    STALE = "stale"
    SHUTDOWN = "shutdown"
    INVALID_CANDIDATE = "invalid_candidate"


@dataclass(frozen=True, slots=True)
class StereoDistancePublicationDiagnostics:
    candidates_received: int
    accepted: int
    rejected: int
    rejected_mode_mismatch: int
    rejected_intent_revision_changed: int
    rejected_target_changed: int
    rejected_generation_invalid: int
    rejected_main_camera_mismatch: int
    rejected_stale: int
    rejected_shutdown: int
    rejected_invalid_candidate: int
    stale_invalidations: int


class TurretPort(Protocol):
    """Owner-local typed subset of the existing TurretWorker boundary."""

    @property
    def current_state(self) -> TurretState: ...

    def submit_move_relative(self, command: MoveRelativeCommand) -> None: ...

    def submit_tracking_control(self, sample: TrackingControlSample) -> None: ...

    def invalidate_tracking_control(self) -> None: ...

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot: ...

    def set_control_mode(self, mode: TurretControlMode) -> bool: ...

    def stop_motion(self) -> bool: ...

    def motor_on(self) -> bool: ...

    def motor_off(self) -> bool: ...

    def request_emergency(self) -> None: ...


class MotionCalibrationCorrectionPort(Protocol):
    """Camera-aware correction kept above the camera-agnostic Turret boundary."""

    def correct(
        self,
        camera: CameraRole,
        command: MoveRelativeCommand,
    ) -> MoveRelativeCommand: ...


class StereoDistanceCandidatePort(Protocol):
    """Narrow internal result shape produced by the stereo distance worker."""

    target: TargetRef
    intent_revision: int
    distance_m: float
    source_frame_id: int
    capture_id: int | None
    measured_timestamp_ns: int
    pairing_kind: StereoPairingKind


class StereoTargetJoinPort(Protocol):
    """Narrow boundary for selected stereo pair/target frame joining."""

    def start_generation(self, generation: int) -> None: ...

    def set_target(
        self,
        target: TargetRef | None,
        *,
        intent_revision: int = 0,
    ) -> None: ...

    def offer_vision_result(self, result: VisionResult) -> object | None: ...


class Mediator:
    """Coordinate accepted Vision state, target lifecycle, Aiming, and Turret."""

    def __init__(
        self,
        *,
        aiming_config: AimingConfig,
        tracking_control_config: TrackingControlConfig,
        tracking_gap_timeout_ms: int,
        ui_config: UiConfig,
        turret: TurretPort,
        target_controls: Mapping[CameraRole, TargetControlChannel],
        vision_processing_enabled: Mapping[CameraRole, bool] | None = None,
        session_gate: CameraSessionGate | None = None,
        stereo_target_joiner: StereoTargetJoinPort | None = None,
        stereo_distance_results: LatestValue[StereoDistanceCandidatePort] | None = None,
        initial_manual_distance_m: float = 100.0,
        distance_stale_timeout_ms: int = 300,
        clock_ns: Callable[[], int] = monotonic_ns,
        motion_calibration: MotionCalibrationCorrectionPort | None = None,
        tracking_stereo_left_nominal_hfov_deg: float | None = None,
        camera_motion_control_roles: Collection[CameraRole] | None = None,
    ) -> None:
        if ui_config.default_camera not in _NORMAL_CAMERAS:
            raise ValueError("ui.default_camera must be Overview or Stereo Left")
        if set(target_controls) != set(_NORMAL_CAMERAS):
            raise ValueError("target_controls must contain Overview and Stereo Left")
        if any(
            channel.camera is not camera
            for camera, channel in target_controls.items()
        ):
            raise ValueError("target-control mapping key must match channel.camera")
        enabled = (
            {camera: True for camera in _NORMAL_CAMERAS}
            if vision_processing_enabled is None
            else dict(vision_processing_enabled)
        )
        if set(enabled) != set(_NORMAL_CAMERAS):
            raise ValueError(
                "vision_processing_enabled must contain Overview and Stereo Left"
            )
        if (
            isinstance(tracking_gap_timeout_ms, bool)
            or not isinstance(tracking_gap_timeout_ms, int)
            or tracking_gap_timeout_ms <= 0
        ):
            raise ValueError("tracking_gap_timeout_ms must be a positive integer")

        if (
            isinstance(distance_stale_timeout_ms, bool)
            or not isinstance(distance_stale_timeout_ms, int)
            or distance_stale_timeout_ms <= 0
        ):
            raise ValueError("distance_stale_timeout_ms must be a positive integer")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if tracking_stereo_left_nominal_hfov_deg is not None:
            self._validate_tracking_hfov(tracking_stereo_left_nominal_hfov_deg)
        motion_roles = (
            frozenset(_NORMAL_CAMERAS)
            if camera_motion_control_roles is None
            else frozenset(camera_motion_control_roles)
        )
        if not motion_roles.issubset(_NORMAL_CAMERAS):
            raise ValueError(
                "camera_motion_control_roles may contain only Overview and Stereo Left"
            )

        self._turret = turret
        self._target_controls = dict(target_controls)
        self._vision_processing_enabled = enabled
        self._camera_motion_control_roles = motion_roles
        self._session_gate = session_gate or CameraSessionGate()
        self._stereo_target_joiner = stereo_target_joiner
        self._stereo_distance_results = stereo_distance_results
        self._stereo_distance_result_revision = 0
        self._aiming = Aiming(aiming_config)
        self._tracking_estimator = TrackingEstimator(tracking_control_config)
        self._camera_clock_trackers = {
            camera: CameraClockTracker(camera) for camera in _OBSERVABLE_CAMERAS
        }
        self._camera_clock_observations: dict[
            CameraRole, CameraClockObservation | None
        ] = {camera: None for camera in _OBSERVABLE_CAMERAS}
        self._camera_clock_log_session: dict[
            CameraRole, tuple[int, FrameTimestampSource] | None
        ] = {camera: None for camera in _OBSERVABLE_CAMERAS}
        self._camera_clock_log_state: dict[
            CameraRole, tuple[ClockMappingState, str | None] | None
        ] = {camera: None for camera in _OBSERVABLE_CAMERAS}
        self._camera_clock_log_last_snapshot_receive_ns: dict[
            CameraRole, int | None
        ] = {camera: None for camera in _OBSERVABLE_CAMERAS}
        self._use_camera_time_for_tracking = False
        self._tracking_time_basis: CameraPcTimeBasis | None = None
        self._tracking_time_fallback_reason: str | None = None
        self._tracking_control_visualization: TrackingControlVisualization | None = None
        self._tracking_control_config_revision = 0
        self._tracking_gap_timeout_ns = tracking_gap_timeout_ms * 1_000_000
        self._main_camera = ui_config.default_camera
        self._pending_selection: TargetSelectionRequest | None = None
        self._selected_target: TargetRef | None = None
        self._distance_mode = DistanceMeasurementMode.CONSTANT
        self._manual_distance_m = quantize_constant_distance(initial_manual_distance_m)
        self._distance_stale_timeout_ns = distance_stale_timeout_ms * 1_000_000
        self._clock_ns = clock_ns
        self._motion_calibration = motion_calibration
        self._tracking_stereo_left_nominal_hfov_deg = (
            None
            if tracking_stereo_left_nominal_hfov_deg is None
            else float(tracking_stereo_left_nominal_hfov_deg)
        )
        self._tracking_stereo_left_hfov_deg = (
            self._tracking_stereo_left_nominal_hfov_deg
        )
        self._distance_intent_revision = 0
        self._current_distance: DistanceResult | None = None
        self._distance_shutdown = False
        self._distance_diagnostics_lock = Lock()
        self._distance_candidates_received = 0
        self._distance_candidates_accepted = 0
        self._distance_rejections = {
            reason: 0 for reason in StereoDistanceRejectionReason
        }
        self._distance_stale_invalidations = 0
        self._next_selection_id = 1
        self._next_operator_action_id = 1
        self._tracking_hfov_revision = 0
        self._tracking_estimator_reset_reason: str | None = None
        self._latest_results: dict[CameraRole, VisionResult | None] = {
            camera: None for camera in _OBSERVABLE_CAMERAS
        }
        self._turret_state = turret.current_state
        self._pending_control_mode: TurretControlMode | None = None

    @property
    def session_gate(self) -> CameraSessionGate:
        return self._session_gate

    @property
    def aiming(self) -> Aiming:
        return self._aiming

    @property
    def camera_motion_control_roles(self) -> frozenset[CameraRole]:
        return self._camera_motion_control_roles

    def camera_motion_control_allowed(self, camera: CameraRole) -> bool:
        if not isinstance(camera, CameraRole):
            raise TypeError("camera must be CameraRole")
        return camera in self._camera_motion_control_roles

    @property
    def tracking_estimator(self) -> TrackingEstimator:
        return self._tracking_estimator

    @property
    def use_camera_time_for_tracking(self) -> bool:
        return self._use_camera_time_for_tracking

    def set_use_camera_time_for_tracking(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if enabled is self._use_camera_time_for_tracking:
            return
        self._use_camera_time_for_tracking = enabled
        self._tracking_time_basis = None
        self._tracking_time_fallback_reason = None
        self._tracking_estimator.reset_history()
        self._tracking_estimator_reset_reason = "camera_time_mode_changed"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()
        _LOGGER.info(
            "Camera source time for tracking control %s",
            "enabled" if enabled else "disabled",
        )

    def camera_clock_observation(
        self, camera: CameraRole
    ) -> CameraClockObservation | None:
        if not isinstance(camera, CameraRole):
            raise TypeError("camera must be CameraRole")
        return self._camera_clock_observations.get(camera)

    @property
    def tracking_diagnostics(self) -> TrackingEstimatorDiagnostics:
        return self._tracking_estimator.diagnostics

    @property
    def tracking_control_visualization(self) -> TrackingControlVisualization | None:
        return self._tracking_control_visualization

    @property
    def tracking_stereo_left_hfov_deg(self) -> float | None:
        return self._tracking_stereo_left_hfov_deg

    @property
    def tracking_stereo_left_nominal_hfov_deg(self) -> float | None:
        return self._tracking_stereo_left_nominal_hfov_deg

    def set_tracking_stereo_left_hfov_deg(self, hfov_deg: float) -> None:
        """Apply a temporary tracking-only focal scale for Stereo Left."""
        nominal = self._tracking_stereo_left_nominal_hfov_deg
        if nominal is None:
            raise RuntimeError("Stereo Left tracking HFOV tuning is unavailable")
        value = self._validate_tracking_hfov(hfov_deg)
        if value == self._tracking_stereo_left_hfov_deg:
            return
        old_value = self._tracking_stereo_left_hfov_deg
        self._tracking_stereo_left_hfov_deg = value
        self._tracking_hfov_revision += 1
        self._tracking_estimator.reset_history()
        self._tracking_estimator_reset_reason = "tracking_hfov_changed"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()
        _LOGGER.info(
            "Stereo Left tracking HFOV updated nominal=%.6f effective=%.6f "
            "focal_scale=%.6f",
            nominal,
            value,
            self._tracking_stereo_left_focal_scale(),
        )
        emit_event(
            _LOGGER,
            "CONFIG_CHANGE",
            source_instance="core:mediator",
            revision=self._tracking_hfov_revision,
            owner="core.aiming",
            parameter="tracking.stereo_left_hfov_deg",
            old_value={
                "nominal_hfov_deg": nominal,
                "effective_hfov_deg": old_value,
            },
            new_value={
                "nominal_hfov_deg": nominal,
                "effective_hfov_deg": value,
                "focal_scale": self._tracking_stereo_left_focal_scale(),
            },
            apply_result="applied",
            reason=None,
        )

    def tracking_camera_model(
        self,
        camera: CameraRole,
        generation: int,
    ) -> CameraModel | None:
        """Return the camera model used only by the tracking-control path."""
        model = self._session_gate.camera_model(camera, generation)
        if model is None or camera is not CameraRole.STEREO_LEFT:
            return model
        scale = self._tracking_stereo_left_focal_scale()
        if scale == 1.0:
            return model
        return FocalScaleCameraModel(model, scale)

    @staticmethod
    def _validate_tracking_hfov(hfov_deg: float) -> float:
        if isinstance(hfov_deg, bool) or not isinstance(hfov_deg, int | float):
            raise TypeError("tracking HFOV must be a number")
        value = float(hfov_deg)
        if not isfinite(value) or not 0.1 <= value < 179.0:
            raise ValueError("tracking HFOV must be in [0.1, 179.0) degrees")
        return value

    def _tracking_stereo_left_focal_scale(self) -> float:
        nominal = self._tracking_stereo_left_nominal_hfov_deg
        effective = self._tracking_stereo_left_hfov_deg
        if nominal is None or effective is None or effective == nominal:
            return 1.0
        return tan(radians(nominal) / 2.0) / tan(radians(effective) / 2.0)

    @property
    def main_camera(self) -> CameraRole:
        return self._main_camera

    @property
    def pending_selection(self) -> TargetSelectionRequest | None:
        return self._pending_selection

    @property
    def selected_target(self) -> TargetRef | None:
        return self._selected_target

    @property
    def distance_mode(self) -> DistanceMeasurementMode:
        return self._distance_mode

    @property
    def manual_distance_m(self) -> int:
        return self._manual_distance_m

    @property
    def current_distance(self) -> DistanceResult | None:
        self._consume_stereo_distance_candidate()
        distance = self._current_distance
        if (
            self._main_camera is not CameraRole.STEREO_LEFT
            or self._selected_target is None
            or distance is None
            or distance.target != self._selected_target
        ):
            return None
        if (
            distance.source is DistanceSource.STEREO
            and self._clock_ns() - distance.measured_timestamp_ns
            > self._distance_stale_timeout_ns
        ):
            self._current_distance = None
            with self._distance_diagnostics_lock:
                self._distance_stale_invalidations += 1
            return None
        return distance

    @property
    def stereo_distance_diagnostics(self) -> StereoDistancePublicationDiagnostics:
        with self._distance_diagnostics_lock:
            rejected = sum(self._distance_rejections.values())
            return StereoDistancePublicationDiagnostics(
                candidates_received=self._distance_candidates_received,
                accepted=self._distance_candidates_accepted,
                rejected=rejected,
                rejected_mode_mismatch=self._distance_rejections[
                    StereoDistanceRejectionReason.MODE_MISMATCH
                ],
                rejected_intent_revision_changed=self._distance_rejections[
                    StereoDistanceRejectionReason.INTENT_REVISION_CHANGED
                ],
                rejected_target_changed=self._distance_rejections[
                    StereoDistanceRejectionReason.TARGET_CHANGED
                ],
                rejected_generation_invalid=self._distance_rejections[
                    StereoDistanceRejectionReason.GENERATION_INVALID
                ],
                rejected_main_camera_mismatch=self._distance_rejections[
                    StereoDistanceRejectionReason.MAIN_CAMERA_MISMATCH
                ],
                rejected_stale=self._distance_rejections[
                    StereoDistanceRejectionReason.STALE
                ],
                rejected_shutdown=self._distance_rejections[
                    StereoDistanceRejectionReason.SHUTDOWN
                ],
                rejected_invalid_candidate=self._distance_rejections[
                    StereoDistanceRejectionReason.INVALID_CANDIDATE
                ],
                stale_invalidations=self._distance_stale_invalidations,
            )

    def set_distance_mode(self, mode: DistanceMeasurementMode) -> None:
        if not isinstance(mode, DistanceMeasurementMode):
            raise TypeError("mode must be DistanceMeasurementMode")
        if self._distance_mode is mode:
            return
        self._distance_mode = mode
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._refresh_manual_distance()

    def set_manual_distance_m(self, distance_m: float) -> None:
        self._manual_distance_m = quantize_constant_distance(distance_m)
        if self._distance_mode is DistanceMeasurementMode.CONSTANT:
            self._refresh_manual_distance()

    def _refresh_manual_distance(self) -> None:
        self._current_distance = None
        target = self._selected_target
        if (
            self._distance_mode is DistanceMeasurementMode.CONSTANT
            and self._main_camera is CameraRole.STEREO_LEFT
            and target is not None
            and self._session_gate.accepts(target.camera, target.generation)
        ):
            # This is a runtime operator setting, not a frame measurement.
            self._current_distance = DistanceResult(
                target=target,
                distance_m=float(self._manual_distance_m),
                source=DistanceSource.MANUAL,
                source_frame_id=None,
                capture_id=None,
                measured_timestamp_ns=self._clock_ns(),
            )

    @property
    def turret_state(self) -> TurretState:
        return self._turret_state

    @property
    def pending_control_mode(self) -> TurretControlMode | None:
        return self._pending_control_mode

    def latest_vision_result(self, camera: CameraRole) -> VisionResult | None:
        return self._latest_results.get(camera)

    def apply_tracking_control_config_update(
        self,
        update: ConfigUpdate[TrackingControlConfig],
    ) -> bool:
        """Apply the newest runtime estimator policy without changing target state."""
        if not isinstance(update, ConfigUpdate):
            raise TypeError("update must be ConfigUpdate[TrackingControlConfig]")
        if not isinstance(update.config, TrackingControlConfig):
            raise TypeError("update.config must be TrackingControlConfig")
        if update.revision <= self._tracking_control_config_revision:
            return False
        self._tracking_estimator.replace_config(update.config)
        self._tracking_control_config_revision = update.revision
        return True

    def accept_camera_session(self, session: CameraSessionStarted) -> bool:
        """Accept a generation barrier before any data from that generation."""
        if not self._session_gate.accept(session):
            return False
        if session.camera is CameraRole.STEREO_LEFT:
            self._invalidate_distance_intent()
            if self._stereo_target_joiner is not None:
                self._stereo_target_joiner.start_generation(session.generation)
        if session.camera in self._latest_results:
            self._latest_results[session.camera] = None
            self._camera_clock_observations[session.camera] = None
            self._camera_clock_log_session[session.camera] = None
            self._camera_clock_log_state[session.camera] = None
            self._camera_clock_log_last_snapshot_receive_ns[session.camera] = None
        if self._target_intent_uses_camera(session.camera):
            # VisionPipeline.start() already reset the processor and discarded old
            # control intent. An old-generation CLEAR must not cross this barrier.
            self._invalidate_target_intent(stop_motion=True)
        return True

    def accept_target_lifecycle_event(self, event: TargetLifecycleEvent) -> bool:
        """Apply one lifecycle event only when it still matches current intent."""
        if isinstance(event, TargetAcquired):
            return self._accept_target_acquired(event)
        if isinstance(event, TargetAcquisitionFailed):
            return self._accept_target_acquisition_failed(event)
        if isinstance(event, TargetTrackingEnded):
            return self._accept_target_tracking_ended(event)
        raise TypeError("event must be a TargetLifecycleEvent")

    def accept_vision_result(self, result: VisionResult) -> bool:
        camera = result.frame.camera
        if camera not in self._latest_results:
            return False
        if not self._session_gate.accepts(camera, result.frame.generation):
            return False
        self._latest_results[camera] = result
        self._observe_camera_clock(result.frame)
        if camera is CameraRole.STEREO_LEFT and self._stereo_target_joiner is not None:
            self._stereo_target_joiner.offer_vision_result(result)
        if camera is self._main_camera and self._selected_target is not None:
            self._update_tracking_from_result(result)
        return True

    def accept_camera_status(self, status: CameraStatus) -> bool:
        """Accept current-generation status and enforce main-camera loss safety."""
        if not isinstance(status, CameraStatus):
            raise TypeError("status must be CameraStatus")
        if status.camera not in self._latest_results:
            return False
        accepted_generation = self._session_gate.accepted_generation(status.camera)
        if accepted_generation is not None and status.generation != accepted_generation:
            return False
        if (
            status.camera is self._main_camera
            and status.state in _DISCONNECTED_CAMERA_STATES
            and self._target_intent_uses_camera(status.camera)
        ):
            cameras = self._target_intent_cameras(default=status.camera)
            self._invalidate_target_intent(stop_motion=True)
            self._publish_clear(cameras)
        return True

    def accept_turret_state(self, state: TurretState) -> None:
        if not isinstance(state, TurretState):
            raise TypeError("state must be TurretState")
        previous = self._turret_state
        self._turret_state = state

        if self._pending_control_mode is state.control_mode:
            self._pending_control_mode = None

        lost_connection = (
            previous.connection_state is TurretConnectionState.READY
            and state.connection_state is not TurretConnectionState.READY
        )
        motors_became_off = (
            previous.motor_state is not MotorState.OFF
            and state.motor_state is MotorState.OFF
        )
        mode_became_relative = (
            previous.control_mode is not TurretControlMode.RELATIVE
            and state.control_mode is TurretControlMode.RELATIVE
        )
        if lost_connection:
            self._pending_control_mode = None
        if (
            (lost_connection or motors_became_off or mode_became_relative)
            and self._has_target_intent()
        ):
            cameras = self._target_intent_cameras()
            self._invalidate_target_intent(stop_motion=False)
            self._publish_clear(cameras)

    def _emit_operator_action(
        self,
        action: str,
        *,
        accepted: bool,
        reason: str | None = None,
        camera: CameraRole | None = None,
        generation: int | None = None,
        frame_id: int | None = None,
        image_x_px: float | None = None,
        image_y_px: float | None = None,
        requested: object | None = None,
        resulting_target_id: int | None = None,
        resulting_control_mode: TurretControlMode | None = None,
    ) -> None:
        action_id = self._next_operator_action_id
        self._next_operator_action_id += 1
        emit_event(
            _LOGGER,
            "OPERATOR_ACTION",
            source_instance="core:operator",
            action=action,
            action_id=action_id,
            camera=None if camera is None else camera.value,
            generation=generation,
            frame_id=frame_id,
            image_x_px=image_x_px,
            image_y_px=image_y_px,
            requested=requested,
            accepted=accepted,
            reason=reason,
            resulting_target_id=resulting_target_id,
            resulting_control_mode=(
                None if resulting_control_mode is None else resulting_control_mode.value
            ),
        )

    def handle_relative_click(
        self,
        vision_result: VisionResult,
        x_px: float,
        y_px: float,
    ) -> bool:
        frame = vision_result.frame
        reason: str | None = None
        if self._turret_state.control_mode is not TurretControlMode.RELATIVE:
            reason = "control_mode_not_relative"
        elif self._pending_control_mode is not None:
            reason = "control_mode_transition_pending"
        elif frame.camera is not self._main_camera:
            reason = "camera_not_main"
        elif frame.camera not in self._camera_motion_control_roles:
            reason = "camera_motion_control_blocked"
        else:
            model = self._session_gate.camera_model(frame.camera, frame.generation)
            if model is None:
                reason = "generation_not_accepted"
            elif not self._pixel_in_frame(frame, x_px, y_px):
                reason = "pixel_out_of_frame"
        if reason is not None:
            self._emit_operator_action(
                "move_relative",
                accepted=False,
                reason=reason,
                camera=frame.camera,
                generation=frame.generation,
                frame_id=frame.frame_id,
                image_x_px=float(x_px),
                image_y_px=float(y_px),
            )
            return False
        model = self._session_gate.camera_model(frame.camera, frame.generation)
        if model is None:
            raise RuntimeError("accepted relative click lost camera model")
        command = self._aiming.relative_command(
            camera_model=model,
            frame=frame,
            x_px=x_px,
            y_px=y_px,
        )
        if self._motion_calibration is not None:
            command = self._motion_calibration.correct(frame.camera, command)
        self._turret.submit_move_relative(command)
        self._emit_operator_action(
            "move_relative",
            accepted=True,
            camera=frame.camera,
            generation=frame.generation,
            frame_id=frame.frame_id,
            image_x_px=float(x_px),
            image_y_px=float(y_px),
            requested=command,
            resulting_control_mode=self._turret_state.control_mode,
        )
        return True

    def submit_calibration_move(
        self,
        camera: CameraRole,
        generation: int,
        command: MoveRelativeCommand,
    ) -> bool:
        """Submit one wizard-owned baseline move without active click correction."""
        if not isinstance(command, MoveRelativeCommand):
            raise TypeError("command must be MoveRelativeCommand")
        state = self._turret_state
        if (
            camera is not self._main_camera
            or camera not in self._camera_motion_control_roles
            or not self._session_gate.accepts(camera, generation)
            or state.connection_state is not TurretConnectionState.READY
            or state.motor_state is not MotorState.ON
            or state.control_mode is not TurretControlMode.RELATIVE
            or self._pending_control_mode is not None
        ):
            return False
        self._turret.submit_move_relative(command)
        return True

    def request_target_selection(
        self,
        vision_result: VisionResult,
        x_px: float,
        y_px: float,
        *,
        auto_select_candidate_on_first_click: bool = False,
        candidate_click_tolerance_px: int = 15,
    ) -> bool:
        """Publish a backend-neutral request for the exact displayed frame."""
        if not isinstance(auto_select_candidate_on_first_click, bool):
            raise TypeError("auto_select_candidate_on_first_click must be bool")
        if (
            isinstance(candidate_click_tolerance_px, bool)
            or not isinstance(candidate_click_tolerance_px, int)
        ):
            raise TypeError("candidate_click_tolerance_px must be int")
        if not 0 <= candidate_click_tolerance_px <= 50:
            raise ValueError("candidate_click_tolerance_px must be in [0, 50]")
        if not isinstance(vision_result, VisionResult):
            raise TypeError("vision_result must be VisionResult")
        frame = vision_result.frame
        reason: str | None = None
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            reason = "control_mode_not_tracking"
        elif self._pending_control_mode is not None:
            reason = "control_mode_transition_pending"
        elif self._turret_state.connection_state is not TurretConnectionState.READY:
            reason = "turret_not_ready"
        elif self._turret_state.motor_state is not MotorState.ON:
            reason = "motor_not_on"
        elif frame.camera is not self._main_camera:
            reason = "camera_not_main"
        elif not self._session_gate.accepts(frame.camera, frame.generation):
            reason = "generation_not_accepted"
        elif not self._vision_processing_enabled[frame.camera]:
            reason = "vision_processing_disabled"
        elif not self._pixel_in_frame(frame, x_px, y_px):
            reason = "pixel_out_of_frame"
        if reason is not None:
            self._emit_operator_action(
                "target_select",
                accepted=False,
                reason=reason,
                camera=frame.camera,
                generation=frame.generation,
                frame_id=frame.frame_id,
                image_x_px=float(x_px),
                image_y_px=float(y_px),
            )
            return False

        selection_id = self._next_selection_id
        self._next_selection_id += 1
        request = TargetSelectionRequest(
            selection_id=selection_id,
            camera=frame.camera,
            generation=frame.generation,
            frame_id=frame.frame_id,
            x_px=float(x_px),
            y_px=float(y_px),
            candidate_selection_enabled=True,
            auto_select_candidate_on_first_click=auto_select_candidate_on_first_click,
            candidate_click_tolerance_px=candidate_click_tolerance_px,
        )

        had_selected_target = self._selected_target is not None
        self._pending_selection = request
        self._selected_target = None
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._tracking_estimator.reset()
        self._tracking_estimator_reset_reason = "target_selection"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()
        if had_selected_target:
            self._turret.stop_motion()
        self._target_controls[frame.camera].publish(request)
        self._emit_operator_action(
            "target_select",
            accepted=True,
            camera=frame.camera,
            generation=frame.generation,
            frame_id=frame.frame_id,
            image_x_px=float(x_px),
            image_y_px=float(y_px),
            requested={
                "selection_id": selection_id,
                "candidate_selection_enabled": True,
                "auto_select_candidate_on_first_click": (
                    auto_select_candidate_on_first_click
                ),
                "candidate_click_tolerance_px": candidate_click_tolerance_px,
            },
            resulting_target_id=None,
            resulting_control_mode=self._turret_state.control_mode,
        )
        return True

    def request_target_candidate_selection(
        self,
        candidates: TargetCandidateSet,
        candidate_id: int,
        x_px: float,
        y_px: float,
    ) -> bool:
        """Confirm one proposal for the current pending target selection."""
        if not isinstance(candidates, TargetCandidateSet):
            raise TypeError("candidates must be TargetCandidateSet")
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise TypeError("candidate_id must be int")
        pending = self._pending_selection
        latest = self._latest_results.get(candidates.camera)
        reason: str | None = None
        if pending is None:
            reason = "no_pending_selection"
        elif candidates.selection_id != pending.selection_id:
            reason = "selection_id_mismatch"
        elif candidates.camera is not pending.camera:
            reason = "camera_mismatch"
        elif candidates.generation != pending.generation:
            reason = "generation_mismatch"
        elif candidates.camera is not self._main_camera:
            reason = "camera_not_main"
        elif not self._session_gate.accepts(candidates.camera, candidates.generation):
            reason = "generation_not_accepted"
        elif (
            latest is None
            or latest.frame.generation != candidates.generation
            or latest.frame.frame_id != candidates.frame_id
        ):
            reason = "candidate_frame_stale"
        elif not any(
            item.candidate_id == candidate_id for item in candidates.candidates
        ):
            reason = "candidate_not_present"
        if reason is not None:
            self._emit_operator_action(
                "target_candidate_select",
                accepted=False,
                reason=reason,
                camera=candidates.camera,
                generation=candidates.generation,
                frame_id=candidates.frame_id,
                image_x_px=float(x_px),
                image_y_px=float(y_px),
                requested={"candidate_id": candidate_id},
            )
            return False
        assert pending is not None
        command = TargetCandidateSelectionRequest(
            selection_id=pending.selection_id,
            camera=pending.camera,
            generation=pending.generation,
            frame_id=candidates.frame_id,
            candidate_id=candidate_id,
            x_px=float(x_px),
            y_px=float(y_px),
        )
        self._target_controls[pending.camera].publish(command)
        self._emit_operator_action(
            "target_candidate_select",
            accepted=True,
            camera=pending.camera,
            generation=pending.generation,
            frame_id=candidates.frame_id,
            image_x_px=float(x_px),
            image_y_px=float(y_px),
            requested={
                "selection_id": pending.selection_id,
                "candidate_id": candidate_id,
            },
            resulting_target_id=None,
            resulting_control_mode=self._turret_state.control_mode,
        )
        return True

    def deselect_target(self) -> bool:
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            self._emit_operator_action(
                "target_clear", accepted=False, reason="control_mode_not_tracking"
            )
            return False
        cameras = self._target_intent_cameras(default=self._main_camera)
        self._invalidate_target_intent(stop_motion=True)
        self._publish_clear(cameras)
        self._emit_operator_action(
            "target_clear",
            accepted=True,
            resulting_control_mode=self._turret_state.control_mode,
        )
        return True

    def swap_main_preview(self) -> CameraRole:
        old_main = self._main_camera
        self._invalidate_distance_intent()
        self._main_camera = (
            CameraRole.STEREO_LEFT
            if self._main_camera is CameraRole.OVERVIEW
            else CameraRole.OVERVIEW
        )
        self._refresh_stereo_join_target()
        if (
            self._turret_state.control_mode is TurretControlMode.TRACKING
            and self._has_target_intent()
        ):
            self._invalidate_target_intent(stop_motion=True)
            self._publish_clear((old_main, self._main_camera))
        self._refresh_manual_distance()
        self._emit_operator_action(
            "camera_swap",
            accepted=True,
            requested={"old_main_camera": old_main.value, "new_main_camera": self._main_camera.value},
            camera=self._main_camera,
            resulting_control_mode=self._turret_state.control_mode,
        )
        return self._main_camera

    def request_control_mode(self, mode: TurretControlMode) -> bool:
        if not isinstance(mode, TurretControlMode):
            raise TypeError("mode must be TurretControlMode")
        if not self._turret.set_control_mode(mode):
            self._emit_operator_action(
                "control_mode_request",
                accepted=False,
                reason="turret_ingress_rejected",
                requested={"mode": mode.value},
                resulting_control_mode=self._turret_state.control_mode,
            )
            return False
        if (
            self._turret_state.control_mode is TurretControlMode.TRACKING
            and mode is TurretControlMode.RELATIVE
        ):
            cameras = self._target_intent_cameras(default=self._main_camera)
            self._invalidate_target_intent(stop_motion=False)
            self._publish_clear(cameras)
        self._pending_control_mode = mode
        self._emit_operator_action(
            "control_mode_request",
            accepted=True,
            requested={"mode": mode.value},
            resulting_control_mode=self._turret_state.control_mode,
        )
        return True

    def stop_motion(self) -> bool:
        if self._turret_state.control_mode is TurretControlMode.TRACKING:
            cameras = self._target_intent_cameras(default=self._main_camera)
            self._invalidate_target_intent(stop_motion=False)
            self._publish_clear(cameras)
        accepted = self._turret.stop_motion()
        self._emit_operator_action(
            "stop_motion",
            accepted=accepted,
            reason=None if accepted else "turret_ingress_rejected",
            resulting_control_mode=self._turret_state.control_mode,
        )
        return accepted

    def emergency_stop(self) -> None:
        cameras = self._target_intent_cameras(default=self._main_camera)
        self._invalidate_target_intent(stop_motion=False)
        self._publish_clear(cameras)
        self._turret.request_emergency()
        self._emit_operator_action(
            "emergency",
            accepted=True,
            resulting_control_mode=self._turret_state.control_mode,
        )

    def motor_on(self) -> bool:
        accepted = self._turret.motor_on()
        self._emit_operator_action(
            "motor_on",
            accepted=accepted,
            reason=None if accepted else "turret_ingress_rejected",
            resulting_control_mode=self._turret_state.control_mode,
        )
        return accepted

    def motor_off(self) -> bool:
        accepted = self._turret.motor_off()
        if not accepted:
            self._emit_operator_action(
                "motor_off",
                accepted=False,
                reason="turret_ingress_rejected",
                resulting_control_mode=self._turret_state.control_mode,
            )
            return False
        cameras = self._target_intent_cameras(default=self._main_camera)
        self._invalidate_target_intent(stop_motion=False)
        self._publish_clear(cameras)
        self._emit_operator_action(
            "motor_off",
            accepted=True,
            resulting_control_mode=self._turret_state.control_mode,
        )
        return True

    def _accept_target_acquired(self, event: TargetAcquired) -> bool:
        pending = self._pending_selection
        target = event.target
        if (
            pending is None
            or target.camera is not self._main_camera
            or not self._session_gate.accepts(target.camera, target.generation)
            or self._turret_state.control_mode is not TurretControlMode.TRACKING
            or self._turret_state.connection_state
            is not TurretConnectionState.READY
            or self._turret_state.motor_state is not MotorState.ON
            or self._pending_control_mode is not None
            or event.selection_id != pending.selection_id
            or event.request_frame_id != pending.frame_id
            or target.camera is not pending.camera
            or target.generation != pending.generation
        ):
            return False

        self._pending_selection = None
        self._selected_target = target
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._refresh_manual_distance()
        self._tracking_estimator.reset()
        self._tracking_estimator_reset_reason = "target_acquired"
        self._tracking_control_visualization = None
        result = self._latest_results.get(target.camera)
        if result is not None:
            self._update_tracking_from_result(result)
        return True

    def _accept_target_acquisition_failed(
        self,
        event: TargetAcquisitionFailed,
    ) -> bool:
        pending = self._pending_selection
        if (
            pending is None
            or not self._session_gate.accepts(event.camera, event.generation)
            or event.selection_id != pending.selection_id
            or event.request_frame_id != pending.frame_id
            or event.camera is not pending.camera
            or event.generation != pending.generation
        ):
            return False
        self._pending_selection = None
        self._selected_target = None
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._tracking_estimator.reset()
        self._tracking_estimator_reset_reason = "acquisition_failed"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()
        return True

    def _accept_target_tracking_ended(self, event: TargetTrackingEnded) -> bool:
        if (
            event.target != self._selected_target
            or not self._session_gate.accepts(
                event.target.camera,
                event.target.generation,
            )
        ):
            return False
        self._selected_target = None
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._tracking_estimator.reset()
        self._tracking_estimator_reset_reason = "tracking_ended"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()
        self._turret.stop_motion()
        return True

    def _observe_camera_clock(self, frame: FramePacket) -> None:
        tracker = self._camera_clock_trackers.get(frame.camera)
        if tracker is None:
            return
        try:
            observation = tracker.observe_frame(frame)
        except CameraClockSessionError as exc:
            self._camera_clock_observations[frame.camera] = None
            _LOGGER.warning(
                "Camera clock observation rejected camera=%s generation=%d frame=%d: %s",
                frame.camera.value,
                frame.generation,
                frame.frame_id,
                exc,
            )
            return
        self._camera_clock_observations[frame.camera] = observation
        self._log_camera_clock_observation(observation)

    def _log_camera_clock_observation(
        self, observation: CameraClockObservation
    ) -> None:
        camera = observation.camera
        session = (observation.generation, observation.source)
        if self._camera_clock_log_session[camera] != session:
            self._camera_clock_log_session[camera] = session
            self._camera_clock_log_state[camera] = None
            self._camera_clock_log_last_snapshot_receive_ns[camera] = None
            _LOGGER.info(
                "camera-clock-session camera=%s generation=%d source=%s",
                camera.value,
                observation.generation,
                observation.source.value,
            )

        diagnostics = observation.diagnostics
        invalid_reason = (
            None
            if diagnostics.invalid_reason is None
            else diagnostics.invalid_reason.value
        )
        state_key = (diagnostics.state, invalid_reason)
        previous_state = self._camera_clock_log_state[camera]
        if previous_state != state_key:
            self._camera_clock_log_state[camera] = state_key
            _LOGGER.info(
                "camera-clock-state camera=%s generation=%d previous=%s current=%s "
                "reason=%s samples=%d span_ns=%d drift_ppm=%s rms_ns=%s",
                camera.value,
                observation.generation,
                "none" if previous_state is None else previous_state[0].value,
                diagnostics.state.value,
                invalid_reason,
                diagnostics.sample_count,
                diagnostics.device_span_ns,
                diagnostics.drift_ppm,
                diagnostics.fit_rms_ns,
            )

        if observation.sample_outcome in (
            ClockSampleOutcome.REJECTED_OUTLIER,
            ClockSampleOutcome.REJECTED_OUT_OF_ORDER,
        ):
            _LOGGER.warning(
                "camera-clock-sample-rejected camera=%s generation=%d outcome=%s "
                "source_timestamp_ns=%d receive_timestamp_ns=%d",
                camera.value,
                observation.generation,
                observation.sample_outcome.value,
                observation.source_timestamp_ns,
                observation.receive_timestamp_ns,
            )
        elif observation.sample_outcome is ClockSampleOutcome.RESET_ACCEPTED:
            _LOGGER.info(
                "camera-clock-mapper-reset camera=%s generation=%d "
                "source_timestamp_ns=%d receive_timestamp_ns=%d",
                camera.value,
                observation.generation,
                observation.source_timestamp_ns,
                observation.receive_timestamp_ns,
            )

        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        last_snapshot = self._camera_clock_log_last_snapshot_receive_ns[camera]
        receive_ns = observation.receive_timestamp_ns
        if (
            last_snapshot is not None
            and receive_ns >= last_snapshot
            and receive_ns - last_snapshot < _CAMERA_CLOCK_DEBUG_INTERVAL_NS
        ):
            return
        self._camera_clock_log_last_snapshot_receive_ns[camera] = receive_ns
        mapped = observation.receive_anchored_timestamp
        mapped_timestamp_ns = None if mapped is None else mapped.pc_timestamp_ns
        mapped_error_ns = None if mapped is None else mapped.estimated_error_ns
        mapped_minus_receive_ns = (
            None
            if mapped_timestamp_ns is None
            else mapped_timestamp_ns - receive_ns
        )
        _LOGGER.debug(
            "camera-clock-snapshot camera=%s generation=%d source=%s "
            "source_timestamp_ns=%d receive_timestamp_ns=%d mapped_timestamp_ns=%s "
            "mapped_minus_receive_ns=%s estimated_error_ns=%s state=%s reason=%s "
            "samples=%d submitted=%d rejected_outlier=%d rejected_out_of_order=%d "
            "resets=%d span_ns=%d age_ns=%s slope=%s offset_ns=%s drift_ppm=%s "
            "rms_ns=%s median_uncertainty_ns=%s",
            camera.value,
            observation.generation,
            observation.source.value,
            observation.source_timestamp_ns,
            receive_ns,
            mapped_timestamp_ns,
            mapped_minus_receive_ns,
            mapped_error_ns,
            diagnostics.state.value,
            invalid_reason,
            diagnostics.sample_count,
            diagnostics.submitted_sample_count,
            diagnostics.rejected_outlier_count,
            diagnostics.rejected_out_of_order_count,
            diagnostics.reset_count,
            diagnostics.device_span_ns,
            diagnostics.age_ns,
            diagnostics.slope,
            diagnostics.offset_ns,
            diagnostics.drift_ppm,
            diagnostics.fit_rms_ns,
            diagnostics.median_uncertainty_ns,
        )
        emit_event(
            _LOGGER,
            "CLOCK_SNAPSHOT",
            source_instance=f"clock:camera:{camera.value}",
            time_source=observation.source.value,
            camera=camera.value,
            generation=observation.generation,
            state=diagnostics.state.value,
            sample_count=diagnostics.sample_count,
            span_ns=diagnostics.device_span_ns,
            offset_ns=diagnostics.offset_ns,
            drift_ppm=diagnostics.drift_ppm,
            rms_ns=diagnostics.fit_rms_ns,
            valid_from_source_timestamp=None,
            valid_to_source_timestamp=None,
            reset_count=diagnostics.reset_count,
            wrap_count=0,
            uncertainty_ns=diagnostics.median_uncertainty_ns,
            reason=invalid_reason,
            submitted_sample_count=diagnostics.submitted_sample_count,
            rejected_outlier_count=diagnostics.rejected_outlier_count,
            rejected_out_of_order_count=diagnostics.rejected_out_of_order_count,
            source_timestamp_ns=observation.source_timestamp_ns,
            receive_timestamp_ns=receive_ns,
            mapped_timestamp_ns=mapped_timestamp_ns,
            mapped_minus_receive_ns=mapped_minus_receive_ns,
            estimated_error_ns=mapped_error_ns,
            age_ns=diagnostics.age_ns,
            slope=diagnostics.slope,
        )

    def _tracking_observation_timestamp(
        self, frame: FramePacket
    ) -> tuple[int, CameraPcTimeBasis, str | None]:
        if not self._use_camera_time_for_tracking:
            return frame.receive_timestamp_ns, CameraPcTimeBasis.RECEIVE_TIME, None
        observation = self._camera_clock_observations.get(frame.camera)
        if (
            observation is None
            or observation.generation != frame.generation
            or observation.receive_timestamp_ns != frame.receive_timestamp_ns
        ):
            return (
                frame.receive_timestamp_ns,
                CameraPcTimeBasis.RECEIVE_TIME,
                "clock-observation-unavailable",
            )
        if observation.source is FrameTimestampSource.RECEIVE_TIME:
            return (
                frame.receive_timestamp_ns,
                CameraPcTimeBasis.RECEIVE_TIME,
                "source-receive-time",
            )
        mapped = observation.receive_anchored_timestamp
        if (
            mapped is None
            or mapped.basis is not CameraPcTimeBasis.RECEIVE_ANCHORED_SOURCE_CLOCK
        ):
            reason = f"mapper-{observation.diagnostics.state.value}"
            if observation.diagnostics.invalid_reason is not None:
                reason += f":{observation.diagnostics.invalid_reason.value}"
            return frame.receive_timestamp_ns, CameraPcTimeBasis.RECEIVE_TIME, reason
        return mapped.pc_timestamp_ns, mapped.basis, None

    def _apply_tracking_time_basis(
        self,
        basis: CameraPcTimeBasis,
        *,
        frame: FramePacket,
        fallback_reason: str | None,
    ) -> None:
        previous = self._tracking_time_basis
        self._tracking_time_basis = basis
        if previous is not basis:
            _LOGGER.info(
                "tracking-time-basis camera=%s generation=%d frame_id=%d "
                "camera_time_enabled=%s previous=%s current=%s",
                frame.camera.value,
                frame.generation,
                frame.frame_id,
                self._use_camera_time_for_tracking,
                "none" if previous is None else previous.value,
                basis.value,
            )

        previous_fallback = self._tracking_time_fallback_reason
        self._tracking_time_fallback_reason = fallback_reason
        if self._use_camera_time_for_tracking and fallback_reason != previous_fallback:
            if fallback_reason is None:
                if previous_fallback is not None:
                    _LOGGER.info(
                        "tracking-time-fallback-cleared camera=%s generation=%d "
                        "frame_id=%d previous_reason=%s",
                        frame.camera.value,
                        frame.generation,
                        frame.frame_id,
                        previous_fallback,
                    )
            else:
                _LOGGER.warning(
                    "tracking-time-fallback camera=%s generation=%d frame_id=%d "
                    "reason=%s using=%s",
                    frame.camera.value,
                    frame.generation,
                    frame.frame_id,
                    fallback_reason,
                    CameraPcTimeBasis.RECEIVE_TIME.value,
                )

        if previous is None or previous is basis:
            return
        self._tracking_estimator.reset_history()
        self._tracking_estimator_reset_reason = "tracking_time_basis_changed"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()

    def _update_tracking_from_result(self, result: VisionResult) -> None:
        target = self._selected_target
        if target is None:
            return
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            return
        if self._pending_control_mode is not None:
            return
        if target.camera is not result.frame.camera:
            return
        if target.generation != result.frame.generation:
            return
        tracked = self._find_track(result, target.track_id)
        if tracked is None:
            self._tracking_estimator.reset_history()
            self._tracking_estimator_reset_reason = "track_missing"
            self._tracking_control_visualization = None
            self._turret.invalidate_tracking_control()
            return
        self._submit_tracking_control(result, tracked, target)

    def _submit_tracking_control(
        self,
        result: VisionResult,
        tracked: TrackedObject,
        target: TargetRef,
    ) -> None:
        model = self.tracking_camera_model(target.camera, target.generation)
        if model is None:
            self._tracking_estimator.reset_history()
            self._tracking_estimator_reset_reason = "camera_model_unavailable"
            self._tracking_control_visualization = None
            self._turret.invalidate_tracking_control()
            return
        (
            observation_timestamp_ns,
            time_basis,
            fallback_reason,
        ) = self._tracking_observation_timestamp(result.frame)
        self._apply_tracking_time_basis(
            time_basis,
            frame=result.frame,
            fallback_reason=fallback_reason,
        )
        error = self._aiming.tracking_error(
            camera_model=model,
            frame=result.frame,
            tracked=tracked,
            target=target,
            observation_timestamp_ns=observation_timestamp_ns,
        )
        aim_x_px, aim_y_px = self._aiming.aim_point(result.frame.camera, result.frame)
        target_x_px, target_y_px = self._aiming.tracked_center(tracked)
        projection = getattr(model, "projection_matrix", None)
        fx_px = None if projection is None else float(projection[0][0])
        fy_px = None if projection is None else float(projection[1][1])
        cx_px = None if projection is None else float(projection[0][2])
        cy_px = None if projection is None else float(projection[1][2])
        nominal_hfov = (
            self._tracking_stereo_left_nominal_hfov_deg
            if result.frame.camera is CameraRole.STEREO_LEFT
            else None
        )
        effective_hfov = (
            self._tracking_stereo_left_hfov_deg
            if result.frame.camera is CameraRole.STEREO_LEFT
            else None
        )
        emit_event(
            _LOGGER,
            "AIMING_SAMPLE",
            source_instance=f"core:aiming:{result.frame.camera.value}",
            camera=result.frame.camera.value,
            generation=result.frame.generation,
            frame_id=result.frame.frame_id,
            target_id=target.track_id,
            observation_monotonic_ns=error.timestamp_ns,
            observation_age_ms=max(0.0, (monotonic_ns() - error.timestamp_ns) / 1_000_000.0),
            camera_model_source=(
                "tracking_hfov_override"
                if nominal_hfov is not None
                and effective_hfov is not None
                and effective_hfov != nominal_hfov
                else "calibration"
            ),
            nominal_hfov_deg=nominal_hfov,
            effective_hfov_deg=effective_hfov,
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=cx_px,
            cy_px=cy_px,
            aim_x_px=aim_x_px,
            aim_y_px=aim_y_px,
            target_x_px=target_x_px,
            target_y_px=target_y_px,
            error_x_px=target_x_px - aim_x_px,
            error_y_px=target_y_px - aim_y_px,
            error_x_deg=error.error_x_deg,
            error_y_deg=error.error_y_deg,
            valid=True,
            invalid_reason=None,
        )
        previous_timestamp = self._tracking_estimator.diagnostics.observation_timestamp_ns
        if (
            previous_timestamp is not None
            and error.timestamp_ns > previous_timestamp
            and error.timestamp_ns - previous_timestamp
            > self._tracking_gap_timeout_ns
        ):
            self._tracking_estimator.reset_history()
            self._tracking_estimator_reset_reason = "observation_gap"
        platform_motion = self._turret.platform_motion_snapshot()
        diagnostics = self._tracking_estimator.observe(error, platform_motion)
        estimator_config = self._tracking_estimator.config.velocity_estimator
        emit_event(
            _LOGGER,
            "ESTIMATOR_SAMPLE",
            source_instance="core:tracking_estimator",
            camera=result.frame.camera.value,
            generation=result.frame.generation,
            frame_id=result.frame.frame_id,
            target_id=target.track_id,
            estimator_revision=self._tracking_control_config_revision,
            window_ms=estimator_config.window_ms,
            min_span_ms=estimator_config.min_span_ms,
            min_samples=estimator_config.min_samples,
            max_samples=estimator_config.max_samples,
            max_fit_rms_deg=estimator_config.max_fit_rms_deg,
            max_target_rate_x_deg_s=estimator_config.max_target_rate_x_deg_s,
            max_target_rate_y_deg_s=estimator_config.max_target_rate_y_deg_s,
            platform_motion_source=diagnostics.platform_source.value,
            history_reset_reason=self._tracking_estimator_reset_reason,
            x={
                "sample_count": diagnostics.x.sample_count,
                "actual_span_ms": diagnostics.x.actual_span_ms,
                "relative_rate_deg_s": diagnostics.x.relative_rate_deg_s,
                "platform_rate_deg_s": diagnostics.x.platform_rate_deg_s,
                "target_rate_deg_s": diagnostics.x.target_rate_deg_s,
                "fit_rms_deg": diagnostics.x.fit_rms_deg,
                "rate_valid": diagnostics.x.valid,
                "invalid_reason": None if diagnostics.x.invalid_reason is None else diagnostics.x.invalid_reason.value,
                "prediction_offset_deg": diagnostics.x.prediction_offset_deg,
            },
            y={
                "sample_count": diagnostics.y.sample_count,
                "actual_span_ms": diagnostics.y.actual_span_ms,
                "relative_rate_deg_s": diagnostics.y.relative_rate_deg_s,
                "platform_rate_deg_s": diagnostics.y.platform_rate_deg_s,
                "target_rate_deg_s": diagnostics.y.target_rate_deg_s,
                "fit_rms_deg": diagnostics.y.fit_rms_deg,
                "rate_valid": diagnostics.y.valid,
                "invalid_reason": None if diagnostics.y.invalid_reason is None else diagnostics.y.invalid_reason.value,
                "prediction_offset_deg": diagnostics.y.prediction_offset_deg,
            },
        )
        self._tracking_estimator_reset_reason = None
        platform_rate = (
            platform_motion.rate_at(error.timestamp_ns)
            if platform_motion.valid
            else None
        )
        last_anchor = platform_motion.anchors[-1] if platform_motion.anchors else None
        emit_event(
            _LOGGER,
            "PLATFORM_MODEL_SAMPLE",
            source_instance="core:platform_model",
            camera=result.frame.camera.value,
            generation=result.frame.generation,
            frame_id=result.frame.frame_id,
            target_id=target.track_id,
            logical_command_target_x_deg_s=None if last_anchor is None else last_anchor.commanded_rate_x_deg_s,
            logical_command_target_y_deg_s=None if last_anchor is None else last_anchor.commanded_rate_y_deg_s,
            logical_effective_velocity_x_deg_s=None if platform_rate is None else platform_rate.rate_x_deg_s,
            logical_effective_velocity_y_deg_s=None if platform_rate is None else platform_rate.rate_y_deg_s,
            configured_acceleration_x_deg_s2=None if last_anchor is None else last_anchor.acceleration_x_deg_s2,
            configured_acceleration_y_deg_s2=None if last_anchor is None else last_anchor.acceleration_y_deg_s2,
            anchor_timestamp_ns=None if last_anchor is None else last_anchor.timestamp_ns,
            valid=platform_motion.valid and platform_rate is not None,
            invalid_reason=None if platform_motion.valid and platform_rate is not None else "platform_rate_unavailable",
        )
        self._log_tracking_observation(
            result,
            error,
            diagnostics,
            time_basis=time_basis,
            fallback_reason=fallback_reason,
        )
        self._tracking_control_visualization = TrackingControlVisualization(
            target=target,
            camera=result.frame.camera,
            generation=result.frame.generation,
            frame_id=result.frame.frame_id,
            observation_timestamp_ns=error.timestamp_ns,
            observed_center_px=self._aiming.tracked_center(tracked),
            rate_valid_x=diagnostics.x.valid,
            rate_valid_y=diagnostics.y.valid,
            target_rate_x_deg_s=diagnostics.x.target_rate_deg_s,
            target_rate_y_deg_s=diagnostics.y.target_rate_deg_s,
            raw_prediction_offset_x_deg=diagnostics.x.raw_prediction_offset_deg,
            raw_prediction_offset_y_deg=diagnostics.y.raw_prediction_offset_deg,
            applied_prediction_offset_x_deg=(
                diagnostics.x.applied_prediction_offset_deg
            ),
            applied_prediction_offset_y_deg=(
                diagnostics.y.applied_prediction_offset_deg
            ),
        )
        if result.frame.camera not in self._camera_motion_control_roles:
            return
        self._turret.submit_tracking_control(
            TrackingControlSample(
                target=error.target,
                real_error_x_deg=error.error_x_deg,
                real_error_y_deg=error.error_y_deg,
                target_rate_x_deg_s=diagnostics.x.target_rate_deg_s,
                target_rate_y_deg_s=diagnostics.y.target_rate_deg_s,
                prediction_offset_x_deg=diagnostics.x.prediction_offset_deg,
                prediction_offset_y_deg=diagnostics.y.prediction_offset_deg,
                rate_valid_x=diagnostics.x.valid,
                rate_valid_y=diagnostics.y.valid,
                observation_timestamp_ns=error.timestamp_ns,
                frame_id=result.frame.frame_id,
            )
        )

    def _log_tracking_observation(
        self,
        result: VisionResult,
        error: TrackingError,
        diagnostics: TrackingEstimatorDiagnostics,
        *,
        time_basis: CameraPcTimeBasis,
        fallback_reason: str | None,
    ) -> None:
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        config = self._tracking_estimator.config
        prediction = config.prediction
        feed_forward = config.feed_forward
        x = diagnostics.x
        y = diagnostics.y
        prediction_clamped_x = (
            prediction.enabled
            and x.valid
            and x.raw_prediction_offset_deg != x.applied_prediction_offset_deg
        )
        prediction_clamped_y = (
            prediction.enabled
            and y.valid
            and y.raw_prediction_offset_deg != y.applied_prediction_offset_deg
        )
        clock_observation = self._camera_clock_observations.get(result.frame.camera)
        clock_observation_matches = (
            clock_observation is not None
            and clock_observation.generation == result.frame.generation
            and clock_observation.receive_timestamp_ns
            == result.frame.receive_timestamp_ns
        )
        source_timestamp_ns = (
            result.frame.receive_timestamp_ns
            if result.frame.source_timestamp is None
            else result.frame.source_timestamp.timestamp_ns
        )
        source_timestamp_type = (
            FrameTimestampSource.RECEIVE_TIME.value
            if result.frame.source_timestamp is None
            else result.frame.source_timestamp.source.value
        )
        mapped = (
            None
            if not clock_observation_matches or clock_observation is None
            else clock_observation.receive_anchored_timestamp
        )
        mapped_timestamp_ns = None if mapped is None else mapped.pc_timestamp_ns
        mapped_minus_receive_ns = (
            None
            if mapped_timestamp_ns is None
            else mapped_timestamp_ns - result.frame.receive_timestamp_ns
        )
        mapped_error_ns = None if mapped is None else mapped.estimated_error_ns
        mapper_state = (
            None
            if not clock_observation_matches or clock_observation is None
            else clock_observation.diagnostics.state.value
        )
        _LOGGER.debug(
            "tracking-observation "
            "target=%s camera=%s generation=%d frame_id=%d "
            "camera_time_enabled=%s time_basis=%s fallback_reason=%s "
            "source_timestamp_type=%s source_timestamp_ns=%d "
            "receive_timestamp_ns=%d mapped_timestamp_ns=%s "
            "mapped_minus_receive_ns=%s mapped_estimated_error_ns=%s "
            "mapper_state=%s observation_timestamp_ns=%d "
            "real_error_x_deg=%.6f real_error_y_deg=%.6f "
            "relative_rate_x_deg_s=%s relative_rate_y_deg_s=%s "
            "platform_rate_x_deg_s=%s platform_rate_y_deg_s=%s "
            "target_rate_x_deg_s=%s target_rate_y_deg_s=%s "
            "rate_valid_x=%s rate_valid_y=%s "
            "invalid_reason_x=%s invalid_reason_y=%s "
            "platform_video_latency_ms=%d "
            "prediction_enabled=%s prediction_horizon_ms=%d "
            "prediction_raw_x_deg=%.6f prediction_raw_y_deg=%.6f "
            "prediction_applied_x_deg=%.6f prediction_applied_y_deg=%.6f "
            "prediction_clamped_x=%s prediction_clamped_y=%s "
            "ff_enabled=%s ff_gain_x=%.6f ff_gain_y=%.6f",
            error.target,
            result.frame.camera.value,
            result.frame.generation,
            result.frame.frame_id,
            self._use_camera_time_for_tracking,
            time_basis.value,
            fallback_reason,
            source_timestamp_type,
            source_timestamp_ns,
            result.frame.receive_timestamp_ns,
            mapped_timestamp_ns,
            mapped_minus_receive_ns,
            mapped_error_ns,
            mapper_state,
            error.timestamp_ns,
            error.error_x_deg,
            error.error_y_deg,
            x.relative_rate_deg_s,
            y.relative_rate_deg_s,
            x.platform_rate_deg_s,
            y.platform_rate_deg_s,
            x.target_rate_deg_s,
            y.target_rate_deg_s,
            x.valid,
            y.valid,
            None if x.invalid_reason is None else x.invalid_reason.value,
            None if y.invalid_reason is None else y.invalid_reason.value,
            config.platform_motion.video_latency_ms,
            prediction.enabled,
            prediction.horizon_ms,
            x.raw_prediction_offset_deg,
            y.raw_prediction_offset_deg,
            x.applied_prediction_offset_deg,
            y.applied_prediction_offset_deg,
            prediction_clamped_x,
            prediction_clamped_y,
            feed_forward.enabled,
            feed_forward.gain_x,
            feed_forward.gain_y,
        )

    def _invalidate_target_intent(self, *, stop_motion: bool) -> None:
        self._pending_selection = None
        self._selected_target = None
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._tracking_estimator.reset()
        self._tracking_estimator_reset_reason = "target_intent_invalidated"
        self._tracking_control_visualization = None
        self._turret.invalidate_tracking_control()
        if stop_motion:
            self._turret.stop_motion()

    def _refresh_stereo_join_target(self) -> None:
        joiner = self._stereo_target_joiner
        if joiner is None:
            return
        target = self._selected_target
        if (
            self._distance_mode in (
                DistanceMeasurementMode.STEREO_EXACT,
                DistanceMeasurementMode.STEREO_TIMESTAMP,
            )
            and self._main_camera is CameraRole.STEREO_LEFT
            and target is not None
            and target.camera is CameraRole.STEREO_LEFT
            and self._session_gate.accepts(target.camera, target.generation)
        ):
            joiner.set_target(target, intent_revision=self._distance_intent_revision)
            return
        joiner.set_target(None, intent_revision=self._distance_intent_revision)

    def invalidate_distance_computation(self) -> None:
        """Invalidate stereo work and close target joining for shutdown."""
        self._distance_shutdown = True
        self._invalidate_distance_intent()
        joiner = self._stereo_target_joiner
        if joiner is not None:
            joiner.set_target(None, intent_revision=self._distance_intent_revision)

    def _consume_stereo_distance_candidate(self) -> None:
        results = self._stereo_distance_results
        if results is None:
            return
        snapshot = results.snapshot()
        if snapshot.revision <= self._stereo_distance_result_revision:
            return
        self._stereo_distance_result_revision = snapshot.revision
        candidate = snapshot.value
        if candidate is None:
            return
        with self._distance_diagnostics_lock:
            self._distance_candidates_received += 1
        rejection_reason = self._stereo_distance_rejection_reason(candidate)
        if rejection_reason is not None:
            with self._distance_diagnostics_lock:
                self._distance_rejections[rejection_reason] += 1
            return
        with self._distance_diagnostics_lock:
            self._distance_candidates_accepted += 1
        self._current_distance = DistanceResult(
            target=candidate.target,
            distance_m=candidate.distance_m,
            source=DistanceSource.STEREO,
            source_frame_id=candidate.source_frame_id,
            capture_id=candidate.capture_id,
            measured_timestamp_ns=candidate.measured_timestamp_ns,
        )

    def _accept_stereo_distance_candidate(
        self,
        candidate: StereoDistanceCandidatePort,
    ) -> bool:
        return self._stereo_distance_rejection_reason(candidate) is None

    def _stereo_distance_rejection_reason(
        self,
        candidate: StereoDistanceCandidatePort,
    ) -> StereoDistanceRejectionReason | None:
        now_ns = self._clock_ns()
        if self._distance_shutdown:
            return StereoDistanceRejectionReason.SHUTDOWN
        if candidate.intent_revision != self._distance_intent_revision:
            return StereoDistanceRejectionReason.INTENT_REVISION_CHANGED
        mode_matches = (
            self._distance_mode is DistanceMeasurementMode.STEREO_EXACT
            and candidate.pairing_kind is StereoPairingKind.EXACT
            and isinstance(candidate.capture_id, int)
            and not isinstance(candidate.capture_id, bool)
        ) or (
            self._distance_mode is DistanceMeasurementMode.STEREO_TIMESTAMP
            and candidate.pairing_kind is StereoPairingKind.TIMESTAMP
            and candidate.capture_id is None
        )
        if not mode_matches:
            return StereoDistanceRejectionReason.MODE_MISMATCH
        if self._main_camera is not CameraRole.STEREO_LEFT:
            return StereoDistanceRejectionReason.MAIN_CAMERA_MISMATCH
        if (
            self._selected_target != candidate.target
            or candidate.target.camera is not CameraRole.STEREO_LEFT
        ):
            return StereoDistanceRejectionReason.TARGET_CHANGED
        if not self._session_gate.accepts(
            candidate.target.camera, candidate.target.generation
        ):
            return StereoDistanceRejectionReason.GENERATION_INVALID
        if (
            candidate.source_frame_id < 0
            or not isfinite(candidate.distance_m)
            or candidate.distance_m <= 0.0
            or candidate.measured_timestamp_ns > now_ns
        ):
            return StereoDistanceRejectionReason.INVALID_CANDIDATE
        if (
            now_ns - candidate.measured_timestamp_ns
            > self._distance_stale_timeout_ns
        ):
            return StereoDistanceRejectionReason.STALE
        return None

    def _invalidate_distance_intent(self) -> None:
        self._distance_intent_revision += 1
        self._current_distance = None

    def _publish_clear(self, cameras: tuple[CameraRole, ...]) -> None:
        for camera in dict.fromkeys(cameras):
            generation = self._session_gate.accepted_generation(camera)
            if generation is None:
                continue
            self._target_controls[camera].publish(
                ClearTargetSelection(camera=camera, generation=generation)
            )

    def _has_target_intent(self) -> bool:
        return self._pending_selection is not None or self._selected_target is not None

    def _target_intent_uses_camera(self, camera: CameraRole) -> bool:
        pending = self._pending_selection
        target = self._selected_target
        return bool(
            (pending is not None and pending.camera is camera)
            or (target is not None and target.camera is camera)
        )

    def _target_intent_cameras(
        self,
        *,
        default: CameraRole | None = None,
    ) -> tuple[CameraRole, ...]:
        cameras: list[CameraRole] = []
        if self._pending_selection is not None:
            cameras.append(self._pending_selection.camera)
        if self._selected_target is not None:
            cameras.append(self._selected_target.camera)
        if not cameras and default is not None:
            cameras.append(default)
        return tuple(dict.fromkeys(cameras))

    @staticmethod
    def _find_track(result: VisionResult, track_id: int) -> TrackedObject | None:
        return next(
            (item for item in result.tracked_objects if item.track_id == track_id),
            None,
        )

    @staticmethod
    def _pixel_in_frame(result_frame: FramePacket, x_px: float, y_px: float) -> bool:
        height, width = result_frame.image.shape[:2]
        return 0.0 <= x_px < width and 0.0 <= y_px < height


__all__ = ["Mediator", "TurretPort"]
