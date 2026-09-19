"""Minimal Core mediator for the prototype RELATIVE and TRACKING paths."""

from __future__ import annotations

from typing import Protocol

from navmin.config.models import AimingConfig, UiConfig
from navmin.contracts import (
    CameraRole,
    CameraSessionStarted,
    FramePacket,
    MoveRelativeCommand,
    TargetRef,
    TrackedObject,
    TrackingError,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)

from .aiming import Aiming
from .session_gate import CameraSessionGate

_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)


class TurretPort(Protocol):
    """Owner-local typed subset of the existing TurretWorker boundary."""

    @property
    def current_state(self) -> TurretState: ...

    def submit_move_relative(self, command: MoveRelativeCommand) -> None: ...

    def submit_tracking_error(self, error: TrackingError) -> None: ...

    def invalidate_tracking_error(self) -> None: ...

    def set_control_mode(self, mode: TurretControlMode) -> None: ...

    def stop_motion(self) -> None: ...

    def motor_on(self) -> None: ...

    def motor_off(self) -> None: ...

    def request_emergency(self) -> None: ...


class Mediator:
    """Coordinate accepted Vision state, Aiming, and Turret intents."""

    def __init__(
        self,
        *,
        aiming_config: AimingConfig,
        ui_config: UiConfig,
        turret: TurretPort,
        session_gate: CameraSessionGate | None = None,
    ) -> None:
        if ui_config.default_camera not in _NORMAL_CAMERAS:
            raise ValueError("ui.default_camera must be Overview or Stereo Left")
        self._turret = turret
        self._session_gate = session_gate or CameraSessionGate()
        self._aiming = Aiming(aiming_config)
        self._main_camera = ui_config.default_camera
        self._selected_target: TargetRef | None = None
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
    def main_camera(self) -> CameraRole:
        return self._main_camera

    @property
    def selected_target(self) -> TargetRef | None:
        return self._selected_target

    @property
    def turret_state(self) -> TurretState:
        return self._turret_state

    @property
    def pending_control_mode(self) -> TurretControlMode | None:
        return self._pending_control_mode

    def latest_vision_result(self, camera: CameraRole) -> VisionResult | None:
        return self._latest_results.get(camera)

    def accept_camera_session(self, session: CameraSessionStarted) -> bool:
        if not self._session_gate.accept(session):
            return False
        if session.camera in self._latest_results:
            self._latest_results[session.camera] = None
        if (
            self._selected_target is not None
            and self._selected_target.camera is session.camera
        ):
            self._clear_tracking_intent(stop_motion=True)
        return True

    def accept_vision_result(self, result: VisionResult) -> bool:
        camera = result.frame.camera
        if camera not in self._latest_results:
            return False
        if not self._session_gate.accepts(camera, result.frame.generation):
            return False
        self._latest_results[camera] = result
        if camera is self._main_camera and self._selected_target is not None:
            self._update_tracking_from_result(result)
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
        if lost_connection:
            self._pending_control_mode = None
            self._clear_tracking_intent(stop_motion=False)
            return

        if (
            state.control_mode is TurretControlMode.RELATIVE
            and self._selected_target is not None
        ):
            self._clear_tracking_intent(stop_motion=False)

    def handle_relative_click(
        self,
        vision_result: VisionResult,
        x_px: float,
        y_px: float,
    ) -> bool:
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
        self._turret.submit_move_relative(command)
        return True

    def select_target(self, target: TargetRef) -> bool:
        if not isinstance(target, TargetRef):
            raise TypeError("target must be TargetRef")
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            return False
        if self._pending_control_mode is not None:
            return False
        if target.camera is not self._main_camera:
            return False
        if not self._session_gate.accepts(target.camera, target.generation):
            return False
        result = self._latest_results.get(self._main_camera)
        if result is None or result.frame.generation != target.generation:
            return False
        tracked = self._find_track(result, target.track_id)
        if tracked is None:
            return False

        self._selected_target = target
        self._submit_tracking_error(result, tracked, target)
        return True

    def deselect_target(self) -> bool:
        if self._turret_state.control_mode is not TurretControlMode.TRACKING:
            return False
        self._clear_tracking_intent(stop_motion=True)
        return True

    def swap_main_preview(self) -> CameraRole:
        self._main_camera = (
            CameraRole.STEREO_LEFT
            if self._main_camera is CameraRole.OVERVIEW
            else CameraRole.OVERVIEW
        )
        if self._turret_state.control_mode is TurretControlMode.TRACKING:
            self._clear_tracking_intent(stop_motion=True)
        return self._main_camera

    def request_control_mode(self, mode: TurretControlMode) -> None:
        if not isinstance(mode, TurretControlMode):
            raise TypeError("mode must be TurretControlMode")
        if (
            self._turret_state.control_mode is TurretControlMode.TRACKING
            and mode is TurretControlMode.RELATIVE
        ):
            self._clear_tracking_intent(stop_motion=False)
        self._pending_control_mode = mode
        self._turret.set_control_mode(mode)

    def stop_motion(self) -> None:
        if self._turret_state.control_mode is TurretControlMode.TRACKING:
            self._clear_tracking_intent(stop_motion=False)
        self._turret.stop_motion()

    def emergency_stop(self) -> None:
        self._clear_tracking_intent(stop_motion=False)
        self._turret.request_emergency()

    def motor_on(self) -> None:
        self._turret.motor_on()

    def motor_off(self) -> None:
        self._turret.motor_off()

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
            self._turret.invalidate_tracking_error()
            return
        self._submit_tracking_error(result, tracked, target)

    def _submit_tracking_error(
        self,
        result: VisionResult,
        tracked: TrackedObject,
        target: TargetRef,
    ) -> None:
        model = self._session_gate.camera_model(target.camera, target.generation)
        if model is None:
            self._turret.invalidate_tracking_error()
            return
        error = self._aiming.tracking_error(
            camera_model=model,
            frame=result.frame,
            tracked=tracked,
            target=target,
        )
        self._turret.submit_tracking_error(error)

    def _clear_tracking_intent(self, *, stop_motion: bool) -> None:
        self._selected_target = None
        self._turret.invalidate_tracking_error()
        if stop_motion:
            self._turret.stop_motion()

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
