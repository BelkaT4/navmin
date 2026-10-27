"""Core mediator for coordinated Vision target lifecycle and Turret intents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import Lock
from time import monotonic_ns
from typing import Protocol

from navmin.concurrency import LatestValue, TargetControlChannel
from navmin.config.models import (
    AimingConfig,
    AxesConfig,
    TrackingControlConfig,
    UiConfig,
)
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
    MotorState,
    MoveRelativeCommand,
    PlatformMotionSnapshot,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetLifecycleEvent,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TrackedObject,
    TrackingControlSample,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.motion_curve import interpolate_motion
from navmin.vision.stereo_target_join import StereoPairingKind

from .aiming import Aiming
from .distance import DistanceMeasurementMode, quantize_constant_distance
from .motion_calibration import MotionObservation, axis_angle
from .session_gate import CameraSessionGate
from .tracking_estimator import TrackingEstimator, TrackingEstimatorDiagnostics

_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)
_DISCONNECTED_CAMERA_STATES = (
    CameraState.RECONNECTING,
    CameraState.ERROR,
    CameraState.STOPPED,
)


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
        calibration_axes: AxesConfig | None = None,
        camera_stale_timeout_ms: int = 500,
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

        self._turret = turret
        self._target_controls = dict(target_controls)
        self._vision_processing_enabled = enabled
        self._session_gate = session_gate or CameraSessionGate()
        self._stereo_target_joiner = stereo_target_joiner
        self._stereo_distance_results = stereo_distance_results
        self._stereo_distance_result_revision = 0
        self._aiming = Aiming(aiming_config)
        self.calibration_axes = calibration_axes
        self.calibration_observations: list[MotionObservation] = []
        self._calibration_pending: tuple[FramePacket, str, float, float] | None = None
        self._calibration_previous: tuple[str, float] | None = None
        self._calibration_stale_ns = camera_stale_timeout_ms * 1_000_000
        self._tracking_estimator = TrackingEstimator(tracking_control_config)
        self._tracking_control_config_revision = 0
        self._tracking_gap_timeout_ns = tracking_gap_timeout_ms * 1_000_000
        self._main_camera = ui_config.default_camera
        self._pending_selection: TargetSelectionRequest | None = None
        self._selected_target: TargetRef | None = None
        self._distance_mode = DistanceMeasurementMode.CONSTANT
        self._manual_distance_m = quantize_constant_distance(initial_manual_distance_m)
        self._distance_stale_timeout_ns = distance_stale_timeout_ms * 1_000_000
        self._clock_ns = clock_ns
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
        self._latest_results: dict[CameraRole, VisionResult | None] = {
            camera: None for camera in _NORMAL_CAMERAS
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
    def tracking_estimator(self) -> TrackingEstimator:
        return self._tracking_estimator

    @property
    def tracking_diagnostics(self) -> TrackingEstimatorDiagnostics:
        return self._tracking_estimator.diagnostics

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
        if session.camera is self._main_camera:
            self.cancel_motion_calibration()
        if session.camera is CameraRole.STEREO_LEFT:
            self._invalidate_distance_intent()
            if self._stereo_target_joiner is not None:
                self._stereo_target_joiner.start_generation(session.generation)
        elif session.camera is CameraRole.STEREO_RIGHT:
            if self._stereo_target_joiner is not None:
                self._stereo_target_joiner.start_right_generation(
                    session.generation
                )
        if session.camera in self._latest_results:
            self._latest_results[session.camera] = None
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
        if camera is CameraRole.STEREO_LEFT and self._stereo_target_joiner is not None:
            self._stereo_target_joiner.offer_vision_result(result)
        if camera is self._main_camera and self._selected_target is not None:
            self._update_tracking_from_result(result)
        return True

    def accept_stereo_right_vision_result(self, result: VisionResult) -> bool:
        """Accept right detections for stereo correspondence without UI targeting."""
        if result.frame.camera is not CameraRole.STEREO_RIGHT:
            return False
        if not self._session_gate.accepts(
            CameraRole.STEREO_RIGHT, result.frame.generation
        ):
            return False
        if self._stereo_target_joiner is None:
            return False
        self._stereo_target_joiner.offer_right_vision_result(result)
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
        if status.camera is self._main_camera and status.state in _DISCONNECTED_CAMERA_STATES:
            self.cancel_motion_calibration()
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
        if (
            state.connection_state is not TurretConnectionState.READY
            or state.motor_state is not MotorState.ON
            or state.control_mode is not TurretControlMode.RELATIVE
        ):
            self.cancel_motion_calibration()

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

    def handle_relative_click(
        self,
        vision_result: VisionResult,
        x_px: float,
        y_px: float,
    ) -> bool:
        if self._calibration_pending is not None:
            return False
        self._calibration_previous = None
        if self._turret_state.control_mode is not TurretControlMode.RELATIVE:
            return False
        if self._pending_control_mode is not None:
            return False
        frame = vision_result.frame
        if frame.camera is not self._main_camera:
            return False
        model = self._session_gate.camera_model(frame.camera, frame.generation)
        if model is None or not self._pixel_in_frame(frame, x_px, y_px):
            return False
        command = self._aiming.relative_command(
            camera_model=model,
            frame=frame,
            x_px=x_px,
            y_px=y_px,
        )
        if self.calibration_axes is not None:
            for axis, angle in (("x", command.delta_x_deg), ("y", command.delta_y_deg)):
                mechanics = getattr(self.calibration_axes, axis)
                try:
                    corrected = interpolate_motion(mechanics.relative_motion_curve, angle)
                except ValueError:
                    return False
                if abs(corrected) > mechanics.max_relative_move_deg:
                    return False
        self._turret.submit_move_relative(command)
        return True

    @property
    def calibration_pending(self) -> bool:
        return self._calibration_pending is not None

    def cancel_motion_calibration(self) -> None:
        self._calibration_pending = None
        self._calibration_previous = None

    def _calibration_model(self, result: VisionResult, x: float, y: float) -> CameraModel:
        state = self._turret_state
        frame = result.frame
        if (
            state.control_mode is not TurretControlMode.RELATIVE
            or state.motor_state is not MotorState.ON
            or state.connection_state is not TurretConnectionState.READY
            or self._pending_control_mode is not None
            or frame.camera is not self._main_camera
            or not self._pixel_in_frame(frame, x, y)
            or not 0 <= self._clock_ns() - frame.receive_timestamp_ns < self._calibration_stale_ns
        ):
            raise ValueError("Нужны свежий кадр, RELATIVE, связь READY и моторы ON")
        model = self._session_gate.camera_model(frame.camera, frame.generation)
        if model is None:
            raise ValueError("Сессия камеры изменилась")
        return model

    def begin_motion_measurement(
        self, result: VisionResult, x: float, y: float, axis: str,
    ) -> float:
        if self.calibration_axes is None or axis not in ("x", "y"):
            raise ValueError("Нет настроек механики для калибровки")
        if getattr(self.calibration_axes, axis).relative_motion_curve:
            raise ValueError("Для новой серии удалите relative-motion-curve выбранной оси и перезапустите приложение")
        if self._calibration_pending is not None:
            raise ValueError("Сначала завершите или отмените текущий замер")
        model = self._calibration_model(result, x, y)
        aim = self._aiming.aim_point(result.frame.camera, result.frame)
        before = axis_angle(model, x, y, axis)
        command = before - axis_angle(model, *aim, axis)
        if not 0.1 <= abs(command) <= getattr(self.calibration_axes, axis).max_relative_move_deg:
            raise ValueError("Угол меньше 0.1° либо превышает предел relative move")
        self._turret.submit_move_relative(MoveRelativeCommand(
            delta_x_deg=command if axis == "x" else 0.0,
            delta_y_deg=command if axis == "y" else 0.0,
        ))
        self._calibration_pending = (result.frame, axis, command, before)
        return command

    def finish_motion_measurement(
        self, result: VisionResult, x: float, y: float,
    ) -> MotionObservation:
        pending = self._calibration_pending
        if pending is None:
            raise ValueError("Нет начатого замера")
        model = self._calibration_model(result, x, y)
        before_frame, axis, command, before = pending
        frame = result.frame
        if (
            frame.camera is not before_frame.camera
            or frame.generation != before_frame.generation
            or frame.frame_id <= before_frame.frame_id
            or frame.receive_timestamp_ns <= before_frame.receive_timestamp_ns
        ):
            raise ValueError("Нужен новый кадр той же камеры и сессии")
        measured = before - axis_angle(model, x, y, axis)
        previous = self._calibration_previous
        reversal = None if previous is None or previous[0] != axis else previous[1] * command < 0
        observation = MotionObservation(
            axis, frame.camera, frame.generation, before_frame.frame_id, frame.frame_id,
            command, measured, reversal,
        )
        self.calibration_observations.append(observation)
        self._calibration_pending = None
        self._calibration_previous = (
            (axis, command) if measured * command > 0 else None
        )
        return observation

    def request_target_selection(
        self,
        vision_result: VisionResult,
        x_px: float,
        y_px: float,
    ) -> bool:
        """Publish a backend-neutral request for the exact displayed frame."""
        if not isinstance(vision_result, VisionResult):
            raise TypeError("vision_result must be VisionResult")
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            return False
        if self._pending_control_mode is not None:
            return False
        if self._turret_state.connection_state is not TurretConnectionState.READY:
            return False
        if self._turret_state.motor_state is not MotorState.ON:
            return False
        frame = vision_result.frame
        if frame.camera is not self._main_camera:
            return False
        if not self._session_gate.accepts(frame.camera, frame.generation):
            return False
        if not self._vision_processing_enabled[frame.camera]:
            return False
        if not self._pixel_in_frame(frame, x_px, y_px):
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
        )

        # A new SELECT is itself the target-episode boundary. Do not publish a
        # mandatory CLEAR first: only the newest operator intent may reach worker.
        had_selected_target = self._selected_target is not None
        self._pending_selection = request
        self._selected_target = None
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._tracking_estimator.reset()
        self._turret.invalidate_tracking_control()
        if had_selected_target:
            self._turret.stop_motion()
        self._target_controls[frame.camera].publish(request)
        return True

    def deselect_target(self) -> bool:
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            return False
        cameras = self._target_intent_cameras(default=self._main_camera)
        self._invalidate_target_intent(stop_motion=True)
        self._publish_clear(cameras)
        return True

    def swap_main_preview(self) -> CameraRole:
        self.cancel_motion_calibration()
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
        return self._main_camera

    def request_control_mode(self, mode: TurretControlMode) -> bool:
        self.cancel_motion_calibration()
        if not isinstance(mode, TurretControlMode):
            raise TypeError("mode must be TurretControlMode")
        if not self._turret.set_control_mode(mode):
            return False
        if (
            self._turret_state.control_mode is TurretControlMode.TRACKING
            and mode is TurretControlMode.RELATIVE
        ):
            cameras = self._target_intent_cameras(default=self._main_camera)
            self._invalidate_target_intent(stop_motion=False)
            self._publish_clear(cameras)
        self._pending_control_mode = mode
        return True

    def stop_motion(self) -> bool:
        self.cancel_motion_calibration()
        if self._turret_state.control_mode is TurretControlMode.TRACKING:
            cameras = self._target_intent_cameras(default=self._main_camera)
            self._invalidate_target_intent(stop_motion=False)
            self._publish_clear(cameras)
        return self._turret.stop_motion()

    def emergency_stop(self) -> None:
        self.cancel_motion_calibration()
        cameras = self._target_intent_cameras(default=self._main_camera)
        self._invalidate_target_intent(stop_motion=False)
        self._publish_clear(cameras)
        self._turret.request_emergency()

    def motor_on(self) -> bool:
        # MOTOR_ON intentionally restores neither pending nor selected targets.
        return self._turret.motor_on()

    def motor_off(self) -> bool:
        self.cancel_motion_calibration()
        if not self._turret.motor_off():
            return False
        cameras = self._target_intent_cameras(default=self._main_camera)
        self._invalidate_target_intent(stop_motion=False)
        self._publish_clear(cameras)
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
        self._turret.invalidate_tracking_control()
        self._turret.stop_motion()
        return True

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
            self._turret.invalidate_tracking_control()
            return
        self._submit_tracking_control(result, tracked, target)

    def _submit_tracking_control(
        self,
        result: VisionResult,
        tracked: TrackedObject,
        target: TargetRef,
    ) -> None:
        model = self._session_gate.camera_model(target.camera, target.generation)
        if model is None:
            self._tracking_estimator.reset_history()
            self._turret.invalidate_tracking_control()
            return
        error = self._aiming.tracking_error(
            camera_model=model,
            frame=result.frame,
            tracked=tracked,
            target=target,
        )
        previous_timestamp = self._tracking_estimator.diagnostics.observation_timestamp_ns
        if (
            previous_timestamp is not None
            and error.timestamp_ns > previous_timestamp
            and error.timestamp_ns - previous_timestamp
            > self._tracking_gap_timeout_ns
        ):
            self._tracking_estimator.reset_history()
        platform_rate = self._turret.platform_motion_snapshot().rate_at(
            error.timestamp_ns
        )
        diagnostics = self._tracking_estimator.observe(error, platform_rate)
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
            )
        )

    def _invalidate_target_intent(self, *, stop_motion: bool) -> None:
        self._pending_selection = None
        self._selected_target = None
        self._invalidate_distance_intent()
        self._refresh_stereo_join_target()
        self._tracking_estimator.reset()
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
