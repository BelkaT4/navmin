from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from navmin.config.models import AimingConfig, AimPointConfig, AimPointsConfig, UiConfig
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    CameraSessionStarted,
    FramePacket,
    MotorState,
    MoveRelativeCommand,
    TargetRef,
    TrackedObject,
    TrackingError,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.core import CameraSessionGate, Mediator


class _CameraModel:
    def __init__(self, *, cx: float = 50.0, cy: float = 40.0) -> None:
        self.cx = cx
        self.cy = cy

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay((x_px - self.cx) / 50.0, (y_px - self.cy) / 40.0, 1.0)


class _FakeTurret:
    def __init__(self, state: TurretState | None = None) -> None:
        self._state = state or _turret_state()
        self.relative_commands: list[MoveRelativeCommand] = []
        self.tracking_errors: list[TrackingError] = []
        self.invalidations = 0
        self.mode_requests: list[TurretControlMode] = []
        self.stop_requests = 0
        self.motor_on_requests = 0
        self.motor_off_requests = 0
        self.emergency_requests = 0
        self.control_ingress_open = True

    @property
    def current_state(self) -> TurretState:
        return self._state

    def submit_move_relative(self, command: MoveRelativeCommand) -> None:
        self.relative_commands.append(command)

    def submit_tracking_error(self, error: TrackingError) -> None:
        self.tracking_errors.append(error)

    def invalidate_tracking_error(self) -> None:
        self.invalidations += 1

    def set_control_mode(self, mode: TurretControlMode) -> bool:
        if not self.control_ingress_open:
            return False
        self.mode_requests.append(mode)
        return True

    def stop_motion(self) -> bool:
        if not self.control_ingress_open:
            return False
        self.stop_requests += 1
        return True

    def motor_on(self) -> bool:
        if not self.control_ingress_open:
            return False
        self.motor_on_requests += 1
        return True

    def motor_off(self) -> bool:
        if not self.control_ingress_open:
            return False
        self.motor_off_requests += 1
        return True

    def request_emergency(self) -> None:
        self.emergency_requests += 1


def _turret_state(
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    connection: TurretConnectionState = TurretConnectionState.READY,
) -> TurretState:
    return TurretState(
        connection_state=connection,
        motor_state=MotorState.ON,
        control_mode=mode,
        max_speed_x_deg_s=50.0,
        max_speed_y_deg_s=50.0,
        acceleration_x_deg_s2=100.0,
        acceleration_y_deg_s2=100.0,
    )


def _aiming_config(*, lead_time_ms: int = 100) -> AimingConfig:
    return AimingConfig(
        lead_time_ms=lead_time_ms,
        target_lost_timeout_ms=500,
        aim_points=AimPointsConfig(
            overview=AimPointConfig(50, 40),
            stereo_left=AimPointConfig(50, 40),
        ),
    )


def _ui_config(camera: CameraRole = CameraRole.OVERVIEW) -> UiConfig:
    return UiConfig(camera, False, False)


def _session(
    camera: CameraRole,
    generation: int,
    model: _CameraModel | None = None,
) -> CameraSessionStarted:
    return CameraSessionStarted(camera, generation, model or _CameraModel(), 1)


def _result(
    camera: CameraRole,
    generation: int,
    *,
    track_ids: tuple[int, ...] = (7,),
    timestamp_ns: int = 1000,
    velocity_x: float = 0.0,
) -> VisionResult:
    image = np.zeros((81, 101, 3), dtype=np.uint8)
    image.flags.writeable = False
    frame = FramePacket(camera, generation, 4, None, timestamp_ns, image)
    tracked = tuple(
        TrackedObject(
            track_id,
            BBox(55 + index * 10, 36, 10, 8),
            velocity_x,
            0.0,
            3,
        )
        for index, track_id in enumerate(track_ids)
    )
    return VisionResult(frame, tracked, 5)


def _mediator(
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    main: CameraRole = CameraRole.OVERVIEW,
) -> tuple[Mediator, _FakeTurret]:
    turret = _FakeTurret(_turret_state(mode=mode))
    return (
        Mediator(
            aiming_config=_aiming_config(),
            ui_config=_ui_config(main),
            turret=turret,
        ),
        turret,
    )


def _accept_main(mediator: Mediator, *, generation: int = 1) -> VisionResult:
    camera = mediator.main_camera
    mediator.accept_camera_session(_session(camera, generation))
    result = _result(camera, generation)
    assert mediator.accept_vision_result(result)
    return result


def _confirm_mode(
    mediator: Mediator,
    mode: TurretControlMode,
    *,
    connection: TurretConnectionState = TurretConnectionState.READY,
) -> None:
    mediator.accept_turret_state(_turret_state(mode=mode, connection=connection))


def test_camera_session_gate_rejects_data_before_barrier_and_old_generation() -> None:
    gate = CameraSessionGate()
    model = _CameraModel()

    assert not gate.accepts(CameraRole.OVERVIEW, 1)
    assert gate.camera_model(CameraRole.OVERVIEW, 1) is None
    assert gate.accept(_session(CameraRole.OVERVIEW, 1, model))
    assert gate.accepts(CameraRole.OVERVIEW, 1)
    assert gate.camera_model(CameraRole.OVERVIEW, 1) is model
    assert not gate.accepts(CameraRole.OVERVIEW, 0)


def test_camera_session_gate_new_generation_replaces_model_without_rollback() -> None:
    gate = CameraSessionGate()
    first = _CameraModel(cx=10)
    second = _CameraModel(cx=20)

    assert gate.accept(_session(CameraRole.OVERVIEW, 1, first))
    assert gate.accept(_session(CameraRole.OVERVIEW, 2, second))
    assert not gate.accept(_session(CameraRole.OVERVIEW, 1, first))
    assert not gate.accept(_session(CameraRole.OVERVIEW, 2, first))
    assert gate.accepted_generation(CameraRole.OVERVIEW) == 2
    assert gate.camera_model(CameraRole.OVERVIEW, 2) is second


def test_camera_session_gate_camera_roles_are_independent() -> None:
    gate = CameraSessionGate()
    assert gate.accept(_session(CameraRole.OVERVIEW, 3))
    assert gate.accept(_session(CameraRole.STEREO_LEFT, 1))
    assert gate.accepted_generation(CameraRole.OVERVIEW) == 3
    assert gate.accepted_generation(CameraRole.STEREO_LEFT) == 1


def test_relative_click_rejects_wrong_camera_and_stale_generation() -> None:
    mediator, turret = _mediator()
    current = _accept_main(mediator)
    mediator.accept_camera_session(_session(CameraRole.STEREO_LEFT, 1))

    assert not mediator.handle_relative_click(
        _result(CameraRole.STEREO_LEFT, 1), 60, 40
    )
    mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 2))
    assert not mediator.handle_relative_click(current, 60, 40)
    assert turret.relative_commands == []


def test_relative_click_submits_once_and_later_generation_change_does_not_cancel_it(
) -> None:
    mediator, turret = _mediator()
    result = _accept_main(mediator)

    assert mediator.handle_relative_click(result, 60, 40)
    assert len(turret.relative_commands) == 1
    assert turret.relative_commands[0].delta_x_deg > 0

    mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 2))
    assert len(turret.relative_commands) == 1
    assert turret.stop_requests == 0


def test_tracking_request_is_not_optimistic_and_selection_is_rejected_in_relative(
) -> None:
    mediator, turret = _mediator()
    _accept_main(mediator)
    target = TargetRef(CameraRole.OVERVIEW, 1, 7)

    assert not mediator.select_target(target)
    mediator.request_control_mode(TurretControlMode.TRACKING)

    assert turret.mode_requests == [TurretControlMode.TRACKING]
    assert mediator.turret_state.control_mode is TurretControlMode.RELATIVE
    assert mediator.pending_control_mode is TurretControlMode.TRACKING
    assert not mediator.select_target(target)


def test_rejected_mode_request_does_not_create_pending_control_mode() -> None:
    mediator, turret = _mediator()
    turret.control_ingress_open = False

    assert not mediator.request_control_mode(TurretControlMode.TRACKING)
    assert turret.mode_requests == []
    assert mediator.pending_control_mode is None
    assert mediator.turret_state.control_mode is TurretControlMode.RELATIVE


def test_confirmed_tracking_validates_target_and_invalid_request_preserves_selection(
) -> None:
    mediator, _turret = _mediator()
    _accept_main(mediator)
    assert mediator.request_control_mode(TurretControlMode.TRACKING)
    _confirm_mode(mediator, TurretControlMode.TRACKING)
    assert mediator.pending_control_mode is None
    first = TargetRef(CameraRole.OVERVIEW, 1, 7)

    assert mediator.select_target(first)
    assert not mediator.select_target(TargetRef(CameraRole.STEREO_LEFT, 1, 7))
    assert not mediator.select_target(TargetRef(CameraRole.OVERVIEW, 0, 7))
    assert not mediator.select_target(TargetRef(CameraRole.OVERVIEW, 1, 99))
    assert mediator.selected_target == first


def test_selected_track_miss_invalidates_and_reappearance_resumes() -> None:
    mediator, turret = _mediator(mode=TurretControlMode.TRACKING)
    mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 1))
    initial = _result(CameraRole.OVERVIEW, 1, velocity_x=10.0, timestamp_ns=1000)
    mediator.accept_vision_result(initial)
    target = TargetRef(CameraRole.OVERVIEW, 1, 7)

    assert mediator.select_target(target)
    assert len(turret.tracking_errors) == 1
    first_error = turret.tracking_errors[-1]
    assert first_error.target == target
    assert first_error.timestamp_ns == 1000

    assert mediator.accept_vision_result(_result(CameraRole.OVERVIEW, 1, track_ids=()))
    assert mediator.selected_target == target
    assert turret.invalidations == 1
    assert len(turret.tracking_errors) == 1

    assert mediator.accept_vision_result(
        _result(CameraRole.OVERVIEW, 1, timestamp_ns=3000)
    )
    assert mediator.selected_target == target
    assert len(turret.tracking_errors) == 2
    assert turret.tracking_errors[-1].timestamp_ns == 3000


def test_deselect_and_tracking_to_relative_clear_target_before_forwarding() -> None:
    mediator, turret = _mediator(mode=TurretControlMode.TRACKING)
    _accept_main(mediator)
    target = TargetRef(CameraRole.OVERVIEW, 1, 7)
    assert mediator.select_target(target)

    assert mediator.deselect_target()
    assert mediator.selected_target is None
    assert turret.invalidations == 1
    assert turret.stop_requests == 1

    assert mediator.select_target(target)
    before_invalidations = turret.invalidations
    mediator.request_control_mode(TurretControlMode.RELATIVE)
    assert mediator.selected_target is None
    assert turret.invalidations == before_invalidations + 1
    assert turret.mode_requests[-1] is TurretControlMode.RELATIVE
    assert mediator.turret_state.control_mode is TurretControlMode.TRACKING


def test_generation_restart_and_tracking_swap_clear_target_and_stop() -> None:
    mediator, turret = _mediator(mode=TurretControlMode.TRACKING)
    _accept_main(mediator)
    assert mediator.select_target(TargetRef(CameraRole.OVERVIEW, 1, 7))

    assert mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 2))
    assert mediator.selected_target is None
    assert turret.stop_requests == 1
    assert turret.invalidations == 1
    assert mediator.latest_vision_result(CameraRole.OVERVIEW) is None

    mediator.accept_vision_result(_result(CameraRole.OVERVIEW, 2))
    assert mediator.select_target(TargetRef(CameraRole.OVERVIEW, 2, 7))
    assert mediator.swap_main_preview() is CameraRole.STEREO_LEFT
    assert mediator.selected_target is None
    assert turret.stop_requests == 2


def test_relative_swap_changes_main_only_and_does_not_cancel_submitted_move() -> None:
    mediator, turret = _mediator()
    result = _accept_main(mediator)
    assert mediator.handle_relative_click(result, 60, 40)

    assert mediator.swap_main_preview() is CameraRole.STEREO_LEFT
    assert len(turret.relative_commands) == 1
    assert turret.stop_requests == 0
    assert turret.invalidations == 0


def test_stop_emergency_motor_forwarding_and_connection_loss_boundaries() -> None:
    mediator, turret = _mediator(mode=TurretControlMode.TRACKING)
    _accept_main(mediator)
    target = TargetRef(CameraRole.OVERVIEW, 1, 7)
    assert mediator.select_target(target)

    assert mediator.stop_motion()
    assert mediator.selected_target is None
    assert turret.stop_requests == 1

    assert mediator.select_target(target)
    mediator.emergency_stop()
    assert mediator.selected_target is None
    assert turret.emergency_requests == 1

    assert mediator.motor_on()
    assert mediator.motor_off()
    assert turret.motor_on_requests == 1
    assert turret.motor_off_requests == 1

    assert mediator.select_target(target)
    before = turret.invalidations
    mediator.accept_turret_state(
        replace(
            _turret_state(mode=TurretControlMode.TRACKING),
            connection_state=TurretConnectionState.CONNECTING,
        )
    )
    assert mediator.selected_target is None
    assert turret.invalidations == before + 1
    assert turret.stop_requests == 1


def test_core_never_creates_velocity_setpoint_or_pid_output() -> None:
    mediator, turret = _mediator(mode=TurretControlMode.TRACKING)
    _accept_main(mediator)
    assert mediator.select_target(TargetRef(CameraRole.OVERVIEW, 1, 7))

    assert turret.tracking_errors
    assert all(isinstance(item, TrackingError) for item in turret.tracking_errors)


def test_vision_result_is_rejected_until_matching_session_barrier_is_accepted() -> None:
    mediator, _turret = _mediator()
    result = _result(CameraRole.OVERVIEW, 1)

    assert not mediator.accept_vision_result(result)
    assert mediator.latest_vision_result(CameraRole.OVERVIEW) is None
    assert mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 1))
    assert mediator.accept_vision_result(result)
    assert mediator.latest_vision_result(CameraRole.OVERVIEW) is result


def test_startup_main_camera_comes_from_ui_config_and_stereo_right_is_rejected(
) -> None:
    mediator, _turret = _mediator(main=CameraRole.STEREO_LEFT)
    assert mediator.main_camera is CameraRole.STEREO_LEFT

    turret = _FakeTurret()
    with pytest.raises(ValueError):
        Mediator(
            aiming_config=_aiming_config(),
            ui_config=_ui_config(CameraRole.STEREO_RIGHT),
            turret=turret,
        )


def test_relative_stop_only_forwards_without_tracking_invalidation() -> None:
    mediator, turret = _mediator()

    assert mediator.stop_motion()

    assert turret.stop_requests == 1
    assert turret.invalidations == 0
