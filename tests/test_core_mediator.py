from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic_ns

import numpy as np
import pytest

from navmin.calibration import load_stereo_calibration
from navmin.concurrency import LatestValue, TargetControlChannel
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
    DistanceSource,
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
from navmin.core import CameraSessionGate, DistanceMeasurementMode, Mediator
from navmin.vision.stereo_distance import StereoDistanceEstimate, StereoDistanceMethod
from navmin.vision.stereo_distance_worker import (
    StereoDistanceCandidate,
    StereoDistanceWorker,
)
from navmin.vision.stereo_pairing import StereoFramePair, StereoPairIdentity
from navmin.vision.stereo_target_join import (
    StereoPairingKind,
    StereoTargetJoiner,
    StereoTargetObservation,
)
from navmin.vision.stereo_timestamp_pairing import (
    TimestampStereoFramePair,
    TimestampStereoPairIdentity,
)


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


class _FakeClock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


def _controls() -> dict[CameraRole, TargetControlChannel]:
    return {camera: TargetControlChannel(camera) for camera in _NORMAL_CAMERAS}


_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)


def _mediator(
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    motor: MotorState = MotorState.ON,
    main: CameraRole = CameraRole.OVERVIEW,
    processing_enabled: dict[CameraRole, bool] | None = None,
    stereo_target_joiner: StereoTargetJoiner | None = None,
    stereo_distance_results: LatestValue[StereoDistanceCandidate] | None = None,
    distance_stale_timeout_ms: int = 300,
    clock_ns=monotonic_ns,
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
        stereo_target_joiner=stereo_target_joiner,
        stereo_distance_results=stereo_distance_results,
        distance_stale_timeout_ms=distance_stale_timeout_ms,
        clock_ns=clock_ns,
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

@pytest.mark.parametrize(
    ("input_m", "expected"),
    [
        (0, 10),
        (9, 10),
        (14.9, 10),
        (15, 20),
        (104, 100),
        (105, 110),
        (499, 500),
        (999, 500),
    ],
)
def test_s4_constant_distance_grid(input_m: float, expected: int) -> None:
    from navmin.core import quantize_constant_distance

    assert quantize_constant_distance(input_m) == expected


def test_s4_distance_modes_and_target_lifecycle() -> None:
    from navmin.contracts import DistanceSource
    from navmin.core import DistanceMeasurementMode

    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT, mode=TurretControlMode.TRACKING
    )
    assert mediator.distance_mode is DistanceMeasurementMode.CONSTANT
    assert mediator.manual_distance_m == 100
    assert mediator.current_distance is None
    result = _accept_main(mediator)
    request = _request(mediator, result)
    assert mediator.current_distance is None
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    distance = mediator.current_distance
    assert distance is not None
    assert distance.target == mediator.selected_target
    assert distance.source is DistanceSource.MANUAL
    assert distance.distance_m == 100
    assert distance.source_frame_id is None
    assert distance.capture_id is None
    assert distance.measured_timestamp_ns > 0

    mediator.set_manual_distance_m(135)
    assert mediator.current_distance is not None
    assert mediator.current_distance.distance_m == 140
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    assert mediator.current_distance is None
    mediator.set_manual_distance_m(200)
    assert mediator.current_distance is None
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
    assert mediator.current_distance is None
    mediator.set_distance_mode(DistanceMeasurementMode.CONSTANT)
    assert mediator.current_distance is not None
    assert mediator.current_distance.distance_m == 200

    assert mediator.swap_main_preview() is CameraRole.OVERVIEW
    assert mediator.current_distance is None
    assert mediator.distance_mode is DistanceMeasurementMode.CONSTANT
    assert mediator.swap_main_preview() is CameraRole.STEREO_LEFT
    assert mediator.current_distance is None  # swap clears tracking intent
    assert mediator.manual_distance_m == 200


def test_s4_new_selection_invalidates_manual_distance() -> None:
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT, mode=TurretControlMode.TRACKING
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    assert mediator.current_distance is not None
    _request(mediator, frame)
    assert mediator.current_distance is None
    assert mediator.selected_target is None


def test_s4_current_distance_invalidated_on_current_camera_error() -> None:
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT, mode=TurretControlMode.TRACKING
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    assert mediator.current_distance is not None

    assert mediator.accept_camera_status(
        CameraStatus(
            camera=CameraRole.STEREO_LEFT,
            generation=1,
            state=CameraState.ERROR,
            last_receive_timestamp_ns=frame.frame.receive_timestamp_ns,
            error_code="test-loss",
            message="test loss",
        )
    )
    assert mediator.selected_target is None
    assert mediator.current_distance is None


def test_s4_invalid_constant_distance_does_not_modify_state() -> None:
    from navmin.core import quantize_constant_distance

    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            quantize_constant_distance(invalid)
    for invalid in (True, "100", None):
        with pytest.raises(TypeError):
            quantize_constant_distance(invalid)


def test_s4_exact_join_target_follows_mediator_mode_and_lifecycle() -> None:
    from navmin.core import DistanceMeasurementMode

    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
    )
    result = _accept_main(mediator)
    assert joiner.state.left_generation == 1
    request = _request(mediator, result)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    assert joiner.state.target is None

    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    assert joiner.state.target == mediator.selected_target

    next_result = _result(CameraRole.STEREO_LEFT, 1, frame_id=5)
    assert mediator.accept_vision_result(next_result)
    assert joiner.state.pending_result_frame_id == 5

    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
    assert joiner.state.target == mediator.selected_target
    assert joiner.state.pending_result_frame_id is None

    assert mediator.accept_camera_session(_session(CameraRole.STEREO_LEFT, 2))
    assert joiner.state.left_generation == 2
    assert joiner.state.target is None


def test_s4_exact_mode_rejects_overview_target_distance() -> None:
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.OVERVIEW,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    target = mediator.selected_target
    assert target is not None and target.camera is CameraRole.OVERVIEW

    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    assert joiner.state.target is None
    results.publish(
        StereoDistanceCandidate(
            target=target,
            intent_revision=joiner.state.intent_revision,
            distance_m=12.4,
            source_frame_id=frame.frame.frame_id,
            capture_id=77,
            measured_timestamp_ns=monotonic_ns(),
        )
    )
    assert mediator.current_distance is None


def _stereo_candidate(
    target: TargetRef,
    *,
    intent_revision: int,
    timestamp_ns: int | None = None,
) -> StereoDistanceCandidate:
    return StereoDistanceCandidate(
        target=target,
        intent_revision=intent_revision,
        distance_m=12.4,
        source_frame_id=9,
        capture_id=77,
        measured_timestamp_ns=monotonic_ns() if timestamp_ns is None else timestamp_ns,
    )


def test_s4_exact_candidate_becomes_public_distance_only_after_revalidation() -> None:
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    target = mediator.selected_target
    assert target is not None

    results.publish(
        _stereo_candidate(
            target,
            intent_revision=joiner.state.intent_revision,
        )
    )
    distance = mediator.current_distance

    assert distance is not None
    assert distance.target == target
    assert distance.source is DistanceSource.STEREO
    assert distance.distance_m == pytest.approx(12.4)
    assert distance.source_frame_id == 9
    assert distance.capture_id == 77


def test_s4_exact_join_worker_chain_publishes_target_bound_distance() -> None:
    clock = _FakeClock(10_000_000_000)
    worker = StereoDistanceWorker(
        calibration=load_stereo_calibration(
            Path(__file__).resolve().parents[1]
            / "config"
            / "defaults"
            / "calibration"
            / "stereo.json"
        ),
        compute=lambda _observation: StereoDistanceEstimate(
            distance_m=18.6,
            median_disparity_px=10.0,
            valid_pixel_count=5,
            valid_pixel_fraction=1.0,
            method=StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN,
        ),
    )
    joiner = StereoTargetJoiner(worker.submit_observation)
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=worker.results,
        clock_ns=clock,
    )
    initial = _accept_main(mediator)
    request = _request(mediator, initial)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    target = mediator.selected_target
    assert target is not None

    exact_result = _result(
        CameraRole.STEREO_LEFT,
        target.generation,
        frame_id=5,
        timestamp_ns=clock(),
    )
    exact_result = replace(
        exact_result,
        frame=replace(exact_result.frame, capture_id=77),
    )
    right = replace(
        exact_result.frame,
        camera=CameraRole.STEREO_RIGHT,
        frame_id=55,
    )
    pair = StereoFramePair(
        identity=StereoPairIdentity(target.generation, target.generation, 77),
        left=exact_result.frame,
        right=right,
    )

    worker.start()
    try:
        assert mediator.accept_vision_result(exact_result)
        observation = joiner.offer_pair(pair)
        assert observation is not None
        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None and snapshot.value is not None

        distance = mediator.current_distance
        assert distance is not None
        assert distance.target == target
        assert distance.source is DistanceSource.STEREO
        assert distance.distance_m == pytest.approx(18.6)
        assert distance.source_frame_id == 5
        assert distance.capture_id == 77
        assert distance.measured_timestamp_ns == clock()
    finally:
        worker.shutdown()


def test_s5_timestamp_join_worker_chain_publishes_target_bound_distance() -> None:
    clock = _FakeClock(20_000_000_000)
    worker = StereoDistanceWorker(
        calibration=load_stereo_calibration(
            Path(__file__).resolve().parents[1]
            / "config"
            / "defaults"
            / "calibration"
            / "stereo.json"
        ),
        compute=lambda _observation: StereoDistanceEstimate(
            distance_m=12.4,
            median_disparity_px=10.0,
            valid_pixel_count=5,
            valid_pixel_fraction=1.0,
            method=StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN,
        ),
    )
    joiner = StereoTargetJoiner(worker.submit_observation)
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=worker.results,
        clock_ns=clock,
    )
    initial = _accept_main(mediator)
    request = _request(mediator, initial)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
    target = mediator.selected_target
    assert target is not None

    left_result = _result(
        CameraRole.STEREO_LEFT,
        target.generation,
        frame_id=6,
        timestamp_ns=clock(),
    )
    left = replace(left_result.frame, capture_id=None)
    left_result = replace(left_result, frame=left)
    right = replace(
        left,
        camera=CameraRole.STEREO_RIGHT,
        frame_id=56,
        receive_timestamp_ns=clock() + 5_000_000,
    )
    pair = TimestampStereoFramePair(
        identity=TimestampStereoPairIdentity(
            target.generation,
            target.generation,
            left.frame_id,
            right.frame_id,
            left.receive_timestamp_ns,
            right.receive_timestamp_ns,
            5_000_000,
        ),
        left=left,
        right=right,
    )

    worker.start()
    try:
        assert mediator.accept_vision_result(left_result)
        observation = joiner.offer_pair(pair)
        assert observation is not None
        assert observation.pairing_kind is StereoPairingKind.TIMESTAMP
        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None and snapshot.value is not None

        distance = mediator.current_distance
        assert distance is not None
        assert distance.target == target
        assert distance.source is DistanceSource.STEREO
        assert distance.distance_m == pytest.approx(12.4)
        assert distance.source_frame_id == left.frame_id
        assert distance.capture_id is None
        assert distance.measured_timestamp_ns == left.receive_timestamp_ns
    finally:
        worker.shutdown()


def test_s5_timestamp_compute_finishing_after_mode_switch_is_rejected() -> None:
    started = Event()
    release = Event()

    def compute(_observation: StereoTargetObservation):
        started.set()
        assert release.wait(1.0)
        return StereoDistanceEstimate(
            distance_m=12.4,
            median_disparity_px=10.0,
            valid_pixel_count=5,
            valid_pixel_fraction=1.0,
            method=StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN,
        )

    worker = StereoDistanceWorker(
        calibration=load_stereo_calibration(
            Path(__file__).resolve().parents[1]
            / "config"
            / "defaults"
            / "calibration"
            / "stereo.json"
        ),
        compute=compute,
    )
    joiner = StereoTargetJoiner(worker.submit_observation)
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=worker.results,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
    target = mediator.selected_target
    assert target is not None

    left = replace(
        frame.frame,
        capture_id=None,
        receive_timestamp_ns=monotonic_ns(),
    )
    right = replace(
        left,
        camera=CameraRole.STEREO_RIGHT,
        frame_id=99,
        receive_timestamp_ns=left.receive_timestamp_ns + 1_000_000,
    )
    observation = StereoTargetObservation(
        target=target,
        intent_revision=joiner.state.intent_revision,
        pair=TimestampStereoFramePair(
            identity=TimestampStereoPairIdentity(
                target.generation,
                target.generation,
                left.frame_id,
                right.frame_id,
                left.receive_timestamp_ns,
                right.receive_timestamp_ns,
                1_000_000,
            ),
            left=left,
            right=right,
        ),
        roi=BBox(55, 36, 10, 8),
        pairing_kind=StereoPairingKind.TIMESTAMP,
    )

    worker.start()
    try:
        assert worker.submit_observation(observation)
        assert started.wait(1.0)
        mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
        release.set()
        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None and snapshot.value is not None
        assert snapshot.value.pairing_kind is StereoPairingKind.TIMESTAMP
        assert mediator.current_distance is None
    finally:
        worker.shutdown()


@pytest.mark.parametrize(
    ("mode", "kind", "capture_id"),
    [
        (DistanceMeasurementMode.STEREO_EXACT, StereoPairingKind.EXACT, 77),
        (DistanceMeasurementMode.STEREO_TIMESTAMP, StereoPairingKind.TIMESTAMP, None),
    ],
)
def test_s5_stereo_distance_expires_without_falling_back_to_constant(
    mode: DistanceMeasurementMode,
    kind: StereoPairingKind,
    capture_id: int | None,
) -> None:
    clock = _FakeClock(1_000_000_000)
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
        distance_stale_timeout_ms=10,
        clock_ns=clock,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(mode)
    target = mediator.selected_target
    assert target is not None
    results.publish(
        replace(
            _stereo_candidate(
                target,
                intent_revision=joiner.state.intent_revision,
                timestamp_ns=clock(),
            ),
            pairing_kind=kind,
            capture_id=capture_id,
        )
    )

    assert mediator.current_distance is not None
    clock.advance_ms(11)
    assert mediator.current_distance is None
    assert mediator._current_distance is None
    assert mediator.distance_mode is mode
    assert mediator.manual_distance_m == 100
    assert mediator.stereo_distance_diagnostics.accepted == 1
    assert mediator.stereo_distance_diagnostics.stale_invalidations == 1


def test_stereo_publication_diagnostics_separate_stale_lifecycle_and_shutdown() -> None:
    clock = _FakeClock(2_000_000_000)
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
        distance_stale_timeout_ms=10,
        clock_ns=clock,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    target = mediator.selected_target
    assert target is not None
    revision = joiner.state.intent_revision

    clock.advance_ms(20)
    results.publish(
        _stereo_candidate(target, intent_revision=revision, timestamp_ns=1_000_000_000)
    )
    assert mediator.current_distance is None

    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    results.publish(
        _stereo_candidate(target, intent_revision=revision, timestamp_ns=clock())
    )
    assert mediator.current_distance is None

    mediator.invalidate_distance_computation()
    results.publish(
        _stereo_candidate(
            target,
            intent_revision=joiner.state.intent_revision,
            timestamp_ns=clock(),
        )
    )
    assert mediator.current_distance is None

    diagnostics = mediator.stereo_distance_diagnostics
    assert diagnostics.candidates_received == 3
    assert diagnostics.accepted == 0
    assert diagnostics.rejected == 3
    assert diagnostics.rejected_stale == 1
    assert diagnostics.rejected_intent_revision_changed == 1
    assert diagnostics.rejected_shutdown == 1


def test_s4_constant_distance_never_time_stales() -> None:
    clock = _FakeClock(2_000_000_000)
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        distance_stale_timeout_ms=10,
        clock_ns=clock,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))

    first = mediator.current_distance
    assert first is not None and first.source is DistanceSource.MANUAL
    clock.advance_ms(60_000)
    assert mediator.current_distance == first


def test_s4_new_valid_stereo_result_replaces_current_result() -> None:
    clock = _FakeClock(3_000_000_000)
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
        clock_ns=clock,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    target = mediator.selected_target
    assert target is not None
    revision = joiner.state.intent_revision

    results.publish(
        replace(
            _stereo_candidate(target, intent_revision=revision, timestamp_ns=clock()),
            distance_m=12.4,
            source_frame_id=9,
            capture_id=77,
        )
    )
    first = mediator.current_distance
    assert first is not None and first.distance_m == pytest.approx(12.4)

    clock.advance_ms(1)
    results.publish(
        replace(
            _stereo_candidate(target, intent_revision=revision, timestamp_ns=clock()),
            distance_m=18.7,
            source_frame_id=10,
            capture_id=78,
        )
    )
    second = mediator.current_distance
    assert second is not None
    assert second.distance_m == pytest.approx(18.7)
    assert second.source_frame_id == 10
    assert second.capture_id == 78


@pytest.mark.parametrize(
    "boundary",
    [
        "mode-cycle",
        "target-replacement",
        "clear",
        "tracking-ended",
        "new-generation",
        "swap-overview",
        "shutdown",
    ],
)
def test_s4_running_compute_is_rejected_after_lifecycle_boundary(
    boundary: str,
) -> None:
    started = Event()
    release = Event()

    def compute(_observation: StereoTargetObservation):
        started.set()
        assert release.wait(1.0)
        return StereoDistanceEstimate(
            distance_m=18.0,
            median_disparity_px=10.0,
            valid_pixel_count=5,
            valid_pixel_fraction=1.0,
            method=StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN,
        )

    calibration_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "calibration"
        / "stereo.json"
    )
    worker = StereoDistanceWorker(
        calibration=load_stereo_calibration(calibration_path),
        compute=compute,
    )
    joiner = StereoTargetJoiner(worker.submit_observation)
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=worker.results,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    target = mediator.selected_target
    assert target is not None

    left = replace(
        frame.frame,
        capture_id=77,
        receive_timestamp_ns=monotonic_ns(),
    )
    right = replace(
        left,
        camera=CameraRole.STEREO_RIGHT,
        frame_id=99,
    )
    observation = StereoTargetObservation(
        target=target,
        intent_revision=joiner.state.intent_revision,
        pair=StereoFramePair(
            identity=StereoPairIdentity(target.generation, 1, 77),
            left=left,
            right=right,
        ),
        roi=BBox(55, 36, 10, 8),
    )

    worker.start()
    try:
        assert worker.submit_observation(observation)
        assert started.wait(1.0)
        if boundary == "mode-cycle":
            mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
            mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
        elif boundary == "target-replacement":
            replacement = _request(mediator, frame, x_px=62.0)
            assert mediator.accept_target_lifecycle_event(
                _acquired(replacement, track_id=8)
            )
            assert mediator.selected_target != target
        elif boundary == "clear":
            assert mediator.deselect_target()
        elif boundary == "tracking-ended":
            assert mediator.accept_target_lifecycle_event(
                TargetTrackingEnded(
                    target,
                    TargetTrackingEndReason.LOST_UNRECOVERED,
                    frame.frame.frame_id,
                    frame.frame.receive_timestamp_ns,
                )
            )
        elif boundary == "new-generation":
            assert mediator.accept_camera_session(_session(CameraRole.STEREO_LEFT, 2))
        elif boundary == "swap-overview":
            assert mediator.swap_main_preview() is CameraRole.OVERVIEW
        else:
            assert boundary == "shutdown"
            mediator.invalidate_distance_computation()
            assert joiner.state.target is None
        release.set()
        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None and snapshot.value is not None
        assert mediator.current_distance is None
    finally:
        worker.shutdown()


@pytest.mark.parametrize(
    ("mode", "kind", "capture_id", "accepted"),
    [
        (DistanceMeasurementMode.STEREO_EXACT, StereoPairingKind.EXACT, 77, True),
        (DistanceMeasurementMode.STEREO_EXACT, StereoPairingKind.TIMESTAMP, None, False),
        (DistanceMeasurementMode.STEREO_TIMESTAMP, StereoPairingKind.TIMESTAMP, None, True),
        (DistanceMeasurementMode.STEREO_TIMESTAMP, StereoPairingKind.EXACT, 77, False),
        (DistanceMeasurementMode.CONSTANT, StereoPairingKind.TIMESTAMP, None, False),
        (DistanceMeasurementMode.STEREO_EXACT, StereoPairingKind.EXACT, None, False),
        (DistanceMeasurementMode.STEREO_TIMESTAMP, StereoPairingKind.TIMESTAMP, 77, False),
    ],
)
def test_s5_3_candidate_mode_and_capture_identity_guard(
    mode: DistanceMeasurementMode,
    kind: StereoPairingKind,
    capture_id: int | None,
    accepted: bool,
) -> None:
    clock = _FakeClock(5_000_000_000)
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
        clock_ns=clock,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(mode)
    target = mediator.selected_target
    assert target is not None
    results.publish(
        replace(
            _stereo_candidate(target, intent_revision=joiner.state.intent_revision),
            measured_timestamp_ns=clock(),
            pairing_kind=kind,
            capture_id=capture_id,
        )
    )
    distance = mediator.current_distance
    assert (distance is not None and distance.source is DistanceSource.STEREO) == accepted
    if accepted:
        assert distance is not None and distance.capture_id == capture_id


@pytest.mark.parametrize(
    ("initial", "next_mode"),
    [
        (DistanceMeasurementMode.STEREO_EXACT, DistanceMeasurementMode.STEREO_TIMESTAMP),
        (DistanceMeasurementMode.STEREO_TIMESTAMP, DistanceMeasurementMode.STEREO_EXACT),
    ],
)
def test_s5_3_mode_switch_rejects_late_old_kind_and_clears_distance(
    initial: DistanceMeasurementMode,
    next_mode: DistanceMeasurementMode,
) -> None:
    clock = _FakeClock(5_000_000_000)
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    joiner = StereoTargetJoiner()
    mediator, _, _ = _mediator(
        main=CameraRole.STEREO_LEFT,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
        clock_ns=clock,
    )
    frame = _accept_main(mediator)
    request = _request(mediator, frame)
    assert mediator.accept_target_lifecycle_event(_acquired(request))
    mediator.set_distance_mode(initial)
    target = mediator.selected_target
    assert target is not None
    old_kind = (
        StereoPairingKind.EXACT
        if initial is DistanceMeasurementMode.STEREO_EXACT
        else StereoPairingKind.TIMESTAMP
    )
    old_capture_id = 77 if old_kind is StereoPairingKind.EXACT else None
    stale = replace(
        _stereo_candidate(target, intent_revision=joiner.state.intent_revision),
        pairing_kind=old_kind,
        capture_id=old_capture_id,
        measured_timestamp_ns=clock(),
    )
    results.publish(stale)
    assert mediator.current_distance is not None
    mediator.set_distance_mode(next_mode)
    assert mediator.current_distance is None
    assert joiner.state.target == target
    results.publish(replace(stale, intent_revision=joiner.state.intent_revision))
    assert mediator.current_distance is None
    new_kind = (
        StereoPairingKind.EXACT
        if next_mode is DistanceMeasurementMode.STEREO_EXACT
        else StereoPairingKind.TIMESTAMP
    )
    results.publish(replace(
        stale,
        intent_revision=joiner.state.intent_revision,
        pairing_kind=new_kind,
        capture_id=77 if new_kind is StereoPairingKind.EXACT else None,
    ))
    assert mediator.current_distance is not None
