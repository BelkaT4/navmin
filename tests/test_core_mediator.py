from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from navmin.concurrency import TargetControlChannel
from navmin.config import load_config
from navmin.config.models import AimingConfig, AimPointConfig, AimPointsConfig, UiConfig
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    CameraSessionStarted,
    CameraState,
    CameraStatus,
    ClearTargetSelection,
    FramePacket,
    MotorState,
    MoveRelativeCommand,
    PlatformMotionAnchor,
    PlatformMotionSnapshot,
    PlatformMotionSource,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TargetTrackingEndReason,
    TrackedObject,
    TrackingControlSample,
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
        self.tracking_controls: list[TrackingControlSample] = []
        self.invalidations = 0
        self.mode_requests: list[TurretControlMode] = []
        self.stop_requests = 0
        self.motor_on_requests = 0
        self.motor_off_requests = 0
        self.emergency_requests = 0
        self.control_ingress_open = True
        self.platform_snapshot = PlatformMotionSnapshot(
            source=PlatformMotionSource.COMMAND_MODEL,
            valid=True,
            anchors=(
                PlatformMotionAnchor(
                    timestamp_ns=0,
                    modeled_rate_x_deg_s=0.0,
                    modeled_rate_y_deg_s=0.0,
                    commanded_rate_x_deg_s=0.0,
                    commanded_rate_y_deg_s=0.0,
                    acceleration_x_deg_s2=100.0,
                    acceleration_y_deg_s2=100.0,
                ),
            ),
        )

    @property
    def current_state(self) -> TurretState:
        return self._state

    def submit_move_relative(self, command: MoveRelativeCommand) -> None:
        self.relative_commands.append(command)

    def submit_tracking_control(self, sample: TrackingControlSample) -> None:
        self.tracking_controls.append(sample)

    def invalidate_tracking_control(self) -> None:
        self.invalidations += 1

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot:
        return self.platform_snapshot

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
    motor: MotorState = MotorState.ON,
    connection: TurretConnectionState = TurretConnectionState.READY,
) -> TurretState:
    return TurretState(
        connection_state=connection,
        motor_state=motor,
        control_mode=mode,
        max_speed_x_deg_s=50.0,
        max_speed_y_deg_s=50.0,
        acceleration_x_deg_s2=100.0,
        acceleration_y_deg_s2=100.0,
    )


def _aiming_config() -> AimingConfig:
    return AimingConfig(
        target_lost_timeout_ms=500,
        aim_points=AimPointsConfig(
            overview=AimPointConfig(50, 40),
            stereo_left=AimPointConfig(50, 40),
        ),
    )


def _ui_config(camera: CameraRole = CameraRole.OVERVIEW) -> UiConfig:
    return UiConfig(camera, False, False)


def _tracking_control_config():
    defaults = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "navmin.json"
    )
    return load_config(defaults).tracking_control


def _session(camera: CameraRole, generation: int) -> CameraSessionStarted:
    return CameraSessionStarted(camera, generation, _CameraModel(), generation)


def _result(
    camera: CameraRole,
    generation: int,
    *,
    frame_id: int = 4,
    track_ids: tuple[int, ...] = (7,),
    timestamp_ns: int = 1000,
    velocity_x: float = 0.0,
) -> VisionResult:
    image = np.zeros((81, 101, 3), dtype=np.uint8)
    image.flags.writeable = False
    frame = FramePacket(camera, generation, frame_id, None, timestamp_ns, image)
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


def _controls() -> dict[CameraRole, TargetControlChannel]:
    return {camera: TargetControlChannel(camera) for camera in _NORMAL_CAMERAS}


_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)


def _mediator(
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    motor: MotorState = MotorState.ON,
    main: CameraRole = CameraRole.OVERVIEW,
    processing_enabled: dict[CameraRole, bool] | None = None,
) -> tuple[Mediator, _FakeTurret, dict[CameraRole, TargetControlChannel]]:
    turret = _FakeTurret(_turret_state(mode=mode, motor=motor))
    controls = _controls()
    mediator = Mediator(
        aiming_config=_aiming_config(),
        tracking_control_config=_tracking_control_config(),
        tracking_gap_timeout_ms=200,
        ui_config=_ui_config(main),
        turret=turret,
        target_controls=controls,
        vision_processing_enabled=processing_enabled,
    )
    return mediator, turret, controls


def _accept_main(mediator: Mediator, *, generation: int = 1) -> VisionResult:
    camera = mediator.main_camera
    assert mediator.accept_camera_session(_session(camera, generation))
    result = _result(camera, generation)
    assert mediator.accept_vision_result(result)
    return result


def _request(
    mediator: Mediator,
    result: VisionResult,
    *,
    x_px: float = 60.0,
    y_px: float = 40.0,
) -> TargetSelectionRequest:
    assert mediator.request_target_selection(result, x_px, y_px)
    request = mediator.pending_selection
    assert request is not None
    return request


def _acquired(request: TargetSelectionRequest, track_id: int = 7) -> TargetAcquired:
    return TargetAcquired(
        selection_id=request.selection_id,
        target=TargetRef(request.camera, request.generation, track_id),
        request_frame_id=request.frame_id,
        decision_frame_id=request.frame_id,
        decision_frame_timestamp_ns=1000,
    )


def _failed(request: TargetSelectionRequest) -> TargetAcquisitionFailed:
    return TargetAcquisitionFailed(
        selection_id=request.selection_id,
        camera=request.camera,
        generation=request.generation,
        request_frame_id=request.frame_id,
        reason=TargetAcquisitionFailureReason.NO_TARGET_AT_POINT,
        decision_frame_id=request.frame_id,
        decision_frame_timestamp_ns=1000,
    )


def test_camera_session_gate_rejects_data_before_barrier_and_old_generation() -> None:
    gate = CameraSessionGate()

    assert not gate.accepts(CameraRole.OVERVIEW, 1)
    assert gate.accept(_session(CameraRole.OVERVIEW, 1))
    assert gate.camera_model(CameraRole.OVERVIEW, 1) is not None
    assert gate.accept(_session(CameraRole.OVERVIEW, 2))
    assert not gate.accept(_session(CameraRole.OVERVIEW, 1))
    assert not gate.accept(_session(CameraRole.OVERVIEW, 2))
    assert gate.accepted_generation(CameraRole.OVERVIEW) == 2
    assert gate.accept(_session(CameraRole.STEREO_LEFT, 1))


def test_relative_click_uses_accepted_main_session_only() -> None:
    mediator, turret, _target_controls = _mediator()
    current = _accept_main(mediator)
    assert mediator.accept_camera_session(_session(CameraRole.STEREO_LEFT, 1))

    assert not mediator.handle_relative_click(
        _result(CameraRole.STEREO_LEFT, 1), 60, 40
    )
    assert mediator.handle_relative_click(current, 60, 40)
    assert len(turret.relative_commands) == 1
    assert turret.relative_commands[0].delta_x_deg > 0

    assert mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 2))
    assert not mediator.handle_relative_click(current, 60, 40)


def test_tracking_selection_rejected_in_relative_and_during_mode_transition() -> None:
    mediator, turret, controls = _mediator()
    result = _accept_main(mediator)

    assert not mediator.request_target_selection(result, 60, 40)
    assert mediator.request_control_mode(TurretControlMode.TRACKING)
    assert not mediator.request_target_selection(result, 60, 40)
    assert mediator.pending_selection is None
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) is None
    assert turret.mode_requests == [TurretControlMode.TRACKING]


def test_point_request_allocates_pending_and_acquired_accepts_target() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)

    request = _request(mediator, result)

    assert mediator.selected_target is None
    assert request.selection_id == 1
    assert request.frame_id == result.frame.frame_id
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) == request
    assert turret.invalidations == 1

    assert mediator.accept_target_lifecycle_event(_acquired(request))
    assert mediator.pending_selection is None
    assert mediator.selected_target == TargetRef(CameraRole.OVERVIEW, 1, 7)
    assert turret.tracking_controls[-1].target == mediator.selected_target
    assert all(
        isinstance(item, TrackingControlSample) for item in turret.tracking_controls
    )


def test_core_uses_platform_rate_at_observation_timestamp_not_latest_anchor() -> None:
    mediator, turret, _controls_value = _mediator(
        mode=TurretControlMode.TRACKING
    )
    initial = _accept_main(mediator)
    turret.platform_snapshot = PlatformMotionSnapshot(
        source=PlatformMotionSource.COMMAND_MODEL,
        valid=True,
        anchors=(
            PlatformMotionAnchor(0, 2.0, 0.0, 2.0, 0.0, 100.0, 100.0),
            PlatformMotionAnchor(
                200_000_000,
                30.0,
                0.0,
                30.0,
                0.0,
                100.0,
                100.0,
            ),
        ),
    )
    request = _request(mediator, initial)
    assert mediator.accept_target_lifecycle_event(_acquired(request))

    for timestamp_ns in (50_000_000, 100_000_000, 150_000_000):
        assert mediator.accept_vision_result(
            _result(
                CameraRole.OVERVIEW,
                1,
                timestamp_ns=timestamp_ns,
            )
        )

    sample = turret.tracking_controls[-1]
    assert sample.rate_valid_x
    assert sample.target_rate_x_deg_s == pytest.approx(2.0)
    assert turret.platform_snapshot.anchors[-1].modeled_rate_x_deg_s == 30.0


def test_uncovered_platform_history_disables_contributions_but_submits_pid_error() -> (
    None
):
    mediator, turret, _controls_value = _mediator(
        mode=TurretControlMode.TRACKING
    )
    initial = _accept_main(mediator)
    turret.platform_snapshot = PlatformMotionSnapshot(
        source=PlatformMotionSource.COMMAND_MODEL,
        valid=True,
        anchors=(
            PlatformMotionAnchor(
                1_000_000_000,
                0.0,
                0.0,
                0.0,
                0.0,
                100.0,
                100.0,
            ),
        ),
    )
    request = _request(mediator, initial)
    assert mediator.accept_target_lifecycle_event(_acquired(request))

    sample = turret.tracking_controls[-1]
    assert sample.real_error_x_deg != 0.0
    assert not sample.rate_valid_x and not sample.rate_valid_y
    assert sample.prediction_offset_x_deg == 0.0
    assert sample.prediction_offset_y_deg == 0.0


def test_new_target_requires_estimator_history_to_be_collected_again() -> None:
    mediator, turret, _controls_value = _mediator(
        mode=TurretControlMode.TRACKING
    )
    initial = _accept_main(mediator)
    first_request = _request(mediator, initial)
    assert mediator.accept_target_lifecycle_event(_acquired(first_request))
    for timestamp_ns in (50_000_000, 100_000_000, 150_000_000):
        assert mediator.accept_vision_result(
            _result(CameraRole.OVERVIEW, 1, timestamp_ns=timestamp_ns)
        )
    assert turret.tracking_controls[-1].rate_valid_x

    latest = _result(
        CameraRole.OVERVIEW,
        1,
        frame_id=5,
        track_ids=(7, 8),
        timestamp_ns=160_000_000,
    )
    assert mediator.accept_vision_result(latest)
    second_request = _request(mediator, latest, x_px=75.0)
    assert mediator.accept_target_lifecycle_event(_acquired(second_request, track_id=8))

    assert turret.tracking_controls[-1].target.track_id == 8
    assert not turret.tracking_controls[-1].rate_valid_x


def test_new_click_replaces_active_target_with_select_only_and_stops_motion() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    first = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(first))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)
    old_invalidations = turret.invalidations

    second = _request(mediator, result, x_px=61.0)

    assert second.selection_id == first.selection_id + 1
    assert mediator.selected_target is None
    assert mediator.pending_selection == second
    assert turret.invalidations == old_invalidations + 1
    assert turret.stop_requests == 1
    vision_control = controls[CameraRole.OVERVIEW].consume_latest(generation=1)
    assert vision_control == second
    assert isinstance(vision_control, TargetSelectionRequest)


def test_stale_acquired_and_old_failed_cannot_replace_newer_pending() -> None:
    mediator, turret, _controls_value = _mediator(
        mode=TurretControlMode.TRACKING
    )
    result = _accept_main(mediator)
    first = _request(mediator, result)
    second = _request(mediator, result, x_px=62.0)

    assert not mediator.accept_target_lifecycle_event(_acquired(first))
    assert not mediator.accept_target_lifecycle_event(_failed(first))
    assert mediator.pending_selection == second
    assert mediator.selected_target is None
    assert turret.stop_requests == 0

    assert mediator.accept_target_lifecycle_event(_failed(second))
    assert mediator.pending_selection is None
    assert mediator.selected_target is None


def test_tracking_ended_accepts_only_current_target_and_stops_motion() -> None:
    mediator, turret, _controls_value = _mediator(
        mode=TurretControlMode.TRACKING
    )
    result = _accept_main(mediator)
    request = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    current = mediator.selected_target
    assert current is not None

    old = replace(current, track_id=current.track_id + 1)
    stale_event = TargetTrackingEnded(
        old,
        TargetTrackingEndReason.LOST_UNRECOVERED,
        result.frame.frame_id,
        result.frame.receive_timestamp_ns,
    )
    assert not mediator.accept_target_lifecycle_event(stale_event)
    assert mediator.selected_target == current

    current_event = replace(stale_event, target=current)
    assert mediator.accept_target_lifecycle_event(current_event)
    assert mediator.selected_target is None
    assert turret.stop_requests == 1


def test_generation_barrier_invalidates_pending_and_old_data_and_control() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    old_result = _accept_main(mediator)
    old_request = _request(mediator, old_result)

    assert mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 2))

    assert mediator.pending_selection is None
    assert mediator.selected_target is None
    assert mediator.latest_vision_result(CameraRole.OVERVIEW) is None
    assert turret.stop_requests == 1
    assert not mediator.accept_target_lifecycle_event(_acquired(old_request))
    assert not mediator.accept_vision_result(old_result)
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=2) is None


def test_vision_result_is_rejected_until_matching_barrier_is_accepted() -> None:
    mediator, _turret, _controls_value = _mediator()
    result = _result(CameraRole.OVERVIEW, 1)

    assert not mediator.accept_vision_result(result)
    assert mediator.latest_vision_result(CameraRole.OVERVIEW) is None
    assert mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 1))
    assert mediator.accept_vision_result(result)
    assert mediator.latest_vision_result(CameraRole.OVERVIEW) is result


def test_deselect_and_tracking_to_relative_publish_clear() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    request = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)

    assert mediator.deselect_target()
    assert mediator.pending_selection is None
    assert mediator.selected_target is None
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) == (
        ClearTargetSelection(CameraRole.OVERVIEW, 1)
    )
    assert turret.stop_requests == 1

    request = _request(mediator, result)
    assert mediator.request_control_mode(TurretControlMode.RELATIVE)
    assert mediator.pending_selection is None
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) == (
        ClearTargetSelection(CameraRole.OVERVIEW, 1)
    )
    assert turret.mode_requests[-1] is TurretControlMode.RELATIVE
    assert request.selection_id == 2


def test_relative_to_tracking_confirms_without_automatic_selection() -> None:
    mediator, turret, _controls_value = _mediator()
    _accept_main(mediator)

    assert mediator.request_control_mode(TurretControlMode.TRACKING)
    assert mediator.pending_selection is None
    assert mediator.selected_target is None
    mediator.accept_turret_state(_turret_state(mode=TurretControlMode.TRACKING))

    assert mediator.pending_control_mode is None
    assert mediator.pending_selection is None
    assert mediator.selected_target is None
    assert turret.mode_requests == [TurretControlMode.TRACKING]


def test_tracking_swap_clears_both_processors_and_relative_swap_keeps_move() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    assert mediator.accept_camera_session(_session(CameraRole.STEREO_LEFT, 3))
    _request(mediator, result)

    assert mediator.swap_main_preview() is CameraRole.STEREO_LEFT
    assert mediator.pending_selection is None
    assert mediator.selected_target is None
    assert turret.stop_requests == 1
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) == (
        ClearTargetSelection(CameraRole.OVERVIEW, 1)
    )
    assert controls[CameraRole.STEREO_LEFT].consume_latest(generation=3) == (
        ClearTargetSelection(CameraRole.STEREO_LEFT, 3)
    )

    relative, relative_turret, _relative_controls = _mediator()
    relative_result = _accept_main(relative)
    assert relative.handle_relative_click(relative_result, 60, 40)
    assert relative.swap_main_preview() is CameraRole.STEREO_LEFT
    assert len(relative_turret.relative_commands) == 1
    assert relative_turret.stop_requests == 0


def test_main_disconnect_clears_target_but_preview_disconnect_does_not() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    assert mediator.accept_camera_session(_session(CameraRole.STEREO_LEFT, 5))
    request = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)

    preview_status = CameraStatus(
        CameraRole.STEREO_LEFT,
        CameraState.ERROR,
        5,
        1000,
    )
    assert mediator.accept_camera_status(preview_status)
    assert mediator.selected_target is not None
    assert turret.stop_requests == 0

    main_status = replace(
        preview_status,
        camera=CameraRole.OVERVIEW,
        generation=1,
    )
    assert mediator.accept_camera_status(main_status)
    assert mediator.selected_target is None
    assert turret.stop_requests == 1
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) == (
        ClearTargetSelection(CameraRole.OVERVIEW, 1)
    )


def test_emergency_and_accepted_motor_off_clear_and_motor_on_does_not_restore() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    first = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(first))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)

    mediator.emergency_stop()
    assert mediator.selected_target is None
    assert turret.emergency_requests == 1
    assert isinstance(
        controls[CameraRole.OVERVIEW].consume_latest(generation=1),
        ClearTargetSelection,
    )

    second = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(second))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)
    assert mediator.motor_off()
    assert mediator.selected_target is None
    assert mediator.pending_selection is None
    assert turret.motor_off_requests == 1
    assert isinstance(
        controls[CameraRole.OVERVIEW].consume_latest(generation=1),
        ClearTargetSelection,
    )

    assert mediator.motor_on()
    assert mediator.selected_target is None
    assert mediator.pending_selection is None
    assert turret.motor_on_requests == 1


def test_selection_is_rejected_before_vision_when_motors_or_processing_are_off() -> (
    None
):
    processing = {
        CameraRole.OVERVIEW: False,
        CameraRole.STEREO_LEFT: True,
    }
    disabled, disabled_turret, disabled_controls = _mediator(
        mode=TurretControlMode.TRACKING,
        processing_enabled=processing,
    )
    disabled_result = _accept_main(disabled)
    assert not disabled.request_target_selection(disabled_result, 60, 40)
    assert disabled.pending_selection is None
    assert disabled_turret.invalidations == 0
    assert disabled_controls[CameraRole.OVERVIEW].consume_latest(generation=1) is None

    motors_off, off_turret, off_controls = _mediator(
        mode=TurretControlMode.TRACKING,
        motor=MotorState.OFF,
    )
    off_result = _accept_main(motors_off)
    assert not motors_off.request_target_selection(off_result, 60, 40)
    assert motors_off.pending_selection is None
    assert off_turret.invalidations == 0
    assert off_controls[CameraRole.OVERVIEW].consume_latest(generation=1) is None


def test_temporary_track_miss_keeps_target_and_invalidates_tracking_error() -> None:
    mediator, turret, _controls_value = _mediator(
        mode=TurretControlMode.TRACKING
    )
    initial = _accept_main(mediator)
    request = _request(mediator, initial)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    selected = mediator.selected_target
    assert selected is not None
    before = turret.invalidations

    assert mediator.accept_vision_result(
        _result(CameraRole.OVERVIEW, 1, frame_id=5, track_ids=())
    )
    assert mediator.selected_target == selected
    assert turret.invalidations == before + 1

    assert mediator.accept_vision_result(
        _result(CameraRole.OVERVIEW, 1, frame_id=6, timestamp_ns=3000)
    )
    assert mediator.selected_target == selected
    assert turret.tracking_controls[-1].observation_timestamp_ns == 3000


def test_turret_connection_loss_clears_tracking_intent() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    request = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)

    mediator.accept_turret_state(
        replace(
            _turret_state(mode=TurretControlMode.TRACKING),
            connection_state=TurretConnectionState.CONNECTING,
        )
    )

    assert mediator.selected_target is None
    assert mediator.pending_selection is None
    assert isinstance(
        controls[CameraRole.OVERVIEW].consume_latest(generation=1),
        ClearTargetSelection,
    )
    assert turret.stop_requests == 0


def test_rejected_control_requests_preserve_current_target_intent() -> None:
    mediator, turret, controls = _mediator(mode=TurretControlMode.TRACKING)
    result = _accept_main(mediator)
    request = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    controls[CameraRole.OVERVIEW].consume_latest(generation=1)
    selected = mediator.selected_target
    turret.control_ingress_open = False

    assert not mediator.request_control_mode(TurretControlMode.RELATIVE)
    assert not mediator.motor_off()
    assert mediator.selected_target == selected
    assert controls[CameraRole.OVERVIEW].consume_latest(generation=1) is None


def test_relative_stop_only_forwards_without_target_invalidation() -> None:
    mediator, turret, _controls_value = _mediator()

    assert mediator.stop_motion()

    assert turret.stop_requests == 1
    assert turret.invalidations == 0


def test_startup_main_camera_and_target_channel_validation() -> None:
    mediator, _turret, _controls_value = _mediator(main=CameraRole.STEREO_LEFT)
    assert mediator.main_camera is CameraRole.STEREO_LEFT

    turret = _FakeTurret()
    with pytest.raises(ValueError):
        Mediator(
            aiming_config=_aiming_config(),
            tracking_control_config=_tracking_control_config(),
            tracking_gap_timeout_ms=200,
            ui_config=_ui_config(CameraRole.STEREO_RIGHT),
            turret=turret,
            target_controls=_controls(),
        )
    with pytest.raises(ValueError):
        Mediator(
            aiming_config=_aiming_config(),
            tracking_control_config=_tracking_control_config(),
            tracking_gap_timeout_ms=200,
            ui_config=_ui_config(),
            turret=turret,
            target_controls={CameraRole.OVERVIEW: TargetControlChannel(CameraRole.OVERVIEW)},
        )
