from __future__ import annotations

from dataclasses import replace

import pytest

from navmin.config.models import (
    AxesConfig,
    AxisMechanicsConfig,
    PidControllerConfig,
    SerialConfig,
    Stm32Config,
    TurretConfig,
)
from navmin.contracts import (
    AxisVelocitySetpoint,
    CameraRole,
    ConfigUpdate,
    MotorState,
    MoveRelativeCommand,
    TargetRef,
    TrackingError,
    TurretControlMode,
)
from navmin.turret.controller import TurretController
from navmin.turret.hal import RelativeMoveLimitError, TurretHal
from navmin.turret.pid import AxisPid, PidGains
from navmin.turret.protocol import (
    CommandCode,
    MoveRelativePayload,
    ProtocolValueError,
    ResultCode,
    SetConfigPayload,
    SetVelocityPayload,
    decode_request,
)
from navmin.turret.session import TurretSession
from navmin.turret.simulator import FakeResponseSpec, FakeStm32Endpoint, FakeTransport


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _CallbackTransport(FakeTransport):
    def __init__(self, endpoint: FakeStm32Endpoint | None = None) -> None:
        super().__init__(endpoint)
        self.after_write = None

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        super().write_frame(frame, timeout_s)
        callback = self.after_write
        if callback is not None:
            self.after_write = None
            callback()


def _config(
    *,
    invert_x: bool = True,
    invert_y: bool = False,
    full_steps_x: int = 200,
    full_steps_y: int = 200,
    microstep_x: int = 16,
    microstep_y: int = 16,
    max_relative_x: float = 180.0,
    max_relative_y: float = 180.0,
    kp_x: float = 2.0,
    ki_x: float = 1.0,
    kd_x: float = 0.5,
    kp_y: float = 2.0,
    ki_y: float = 1.0,
    kd_y: float = 0.5,
    max_speed_x: float = 90.0,
    max_speed_y: float = 90.0,
    acceleration_x: float = 180.0,
    acceleration_y: float = 180.0,
    watchdog_ms: int = 100,
) -> TurretConfig:
    return TurretConfig(
        serial=SerialConfig("fake", 9600, 100, 2, 2),
        axes=AxesConfig(
            x=AxisMechanicsConfig(
                invert_x, full_steps_x, microstep_x, max_relative_x
            ),
            y=AxisMechanicsConfig(
                invert_y, full_steps_y, microstep_y, max_relative_y
            ),
        ),
        controller=PidControllerConfig(kp_x, ki_x, kd_x, kp_y, ki_y, kd_y),
        stm32=Stm32Config(
            max_speed_x,
            max_speed_y,
            acceleration_x,
            acceleration_y,
            watchdog_ms,
        ),
        emulate_stm32=True,
    )


def _stack(
    config: TurretConfig | None = None,
    *,
    clock: _Clock | None = None,
    transport: FakeTransport | None = None,
) -> tuple[TurretController, TurretHal, TurretSession, FakeTransport, _Clock]:
    cfg = config or _config()
    fake = transport or FakeTransport()
    fake.open()
    session = TurretSession(
        fake,
        response_timeout_s=0.1,
        max_retries=2,
        inter_request_delay_s=0.0,
    )
    hal = TurretHal(session, cfg)
    test_clock = clock or _Clock()
    controller = TurretController(hal, cfg, clock=test_clock)
    return controller, hal, session, fake, test_clock


def _target(track_id: int = 1) -> TargetRef:
    return TargetRef(CameraRole.OVERVIEW, generation=1, track_id=track_id)


def _tracking_error(
    target: TargetRef,
    x: float,
    y: float = 0.0,
    *,
    timestamp_ns: int = 1,
) -> TrackingError:
    return TrackingError(target, x, y, timestamp_ns)


def _enter_tracking_off(
    controller: TurretController, transport: FakeTransport
) -> None:
    controller.motor_off()
    controller.set_control_mode(TurretControlMode.TRACKING)
    transport.raw_write_history.clear()


def test_hal_signed_conversion_inverts_direction_but_not_unsigned_magnitudes() -> None:
    _, hal, _, _, _ = _stack()

    assert hal.effective_steps_per_revolution("x") == 3200
    assert hal.degrees_to_steps("x", 90.0) == -800
    assert hal.velocity_to_steps_s("x", 45.0) == -400
    assert hal.degrees_to_steps("y", -90.0) == -800
    assert hal.speed_magnitude_to_steps_s("x", 90.0) == 800
    assert hal.acceleration_magnitude_to_steps_s2("x", 180.0) == 1600


def test_hal_relative_limit_is_checked_before_any_transmission() -> None:
    _, hal, _, transport, _ = _stack(_config(max_relative_x=10.0))

    with pytest.raises(RelativeMoveLimitError):
        hal.move_relative(MoveRelativeCommand(10.1, 0.0))

    assert transport.raw_write_history == []


def test_hal_move_and_velocity_reject_int32_overflow_before_transmission() -> None:
    config = _config(full_steps_x=1 << 31, microstep_x=1, max_relative_x=400.0)
    _, hal, _, transport, _ = _stack(config)

    with pytest.raises(ProtocolValueError):
        hal.move_relative(MoveRelativeCommand(361.0, 0.0))
    with pytest.raises(ProtocolValueError):
        hal.set_velocity(AxisVelocitySetpoint(361.0, 0.0, 1))

    assert transport.raw_write_history == []


def test_hal_set_config_full_snapshot_uses_positive_unsigned_step_units() -> None:
    config = _config()
    _, hal, _, transport, _ = _stack(config)

    result = hal.sync_stm32_config()

    assert result is not None and result.response is not None
    request = decode_request(transport.raw_write_history[-1])
    assert request.command is CommandCode.SET_CONFIG
    assert request.payload == SetConfigPayload(800, 800, 1600, 1600, 100)
    assert hal.applied_stm32_config == config.stm32


def test_hal_set_config_rejects_uint32_overflow_before_transmission() -> None:
    config = _config(full_steps_x=1 << 32, microstep_x=1, max_speed_x=1000.0)
    _, hal, _, transport, _ = _stack(config)

    with pytest.raises(ProtocolValueError):
        hal.sync_stm32_config()

    assert transport.raw_write_history == []


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_speed_x_deg_s", 0.01),
        ("acceleration_x_deg_s2", 0.01),
    ],
)
def test_hal_positive_set_config_magnitude_quantized_to_zero_is_rejected(
    field_name: str,
    value: float,
) -> None:
    config = _config()
    config = replace(config, stm32=replace(config.stm32, **{field_name: value}))
    _, hal, _, transport, _ = _stack(config)

    with pytest.raises(ProtocolValueError):
        hal.sync_stm32_config()

    assert transport.raw_write_history == []


def test_hal_extreme_finite_conversion_raises_protocol_error_not_overflow() -> None:
    config = _config(max_speed_x=1e308)
    _, hal, _, transport, _ = _stack(config)

    with pytest.raises(ProtocolValueError):
        hal.sync_stm32_config()

    assert transport.raw_write_history == []


def test_hal_runtime_mechanics_change_is_restart_only() -> None:
    config = _config(invert_x=True, max_relative_x=10.0)
    _, hal, session, transport, _ = _stack(config)
    changed_axes = replace(
        config.axes,
        x=replace(
            config.axes.x,
            invert=False,
            full_steps_per_revolution=400,
            max_relative_move_deg=20.0,
        ),
    )
    updated_config = replace(config, axes=changed_axes)

    result = hal.apply_config_update(ConfigUpdate(1, updated_config))

    assert result is None
    assert hal.degrees_to_steps("x", 90.0) == -800
    with pytest.raises(RelativeMoveLimitError):
        hal.move_relative(MoveRelativeCommand(15.0, 0.0))
    assert transport.raw_write_history == []

    restarted_hal = TurretHal(session, updated_config)
    move_result = restarted_hal.move_relative(MoveRelativeCommand(15.0, 0.0))

    assert move_result.response is not None
    assert move_result.response.result is ResultCode.OK
    request = decode_request(transport.raw_write_history[-1])
    assert request.command is CommandCode.MOVE_RELATIVE


def test_hal_dynamic_stm32_config_is_latest_only_during_in_flight_operation() -> None:
    config = _config()
    transport = _CallbackTransport()
    _, hal, _, _, _ = _stack(config, transport=transport)
    update_1 = ConfigUpdate(
        1,
        replace(config, stm32=replace(config.stm32, max_speed_x_deg_s=45.0)),
    )
    update_2 = ConfigUpdate(
        2,
        replace(config, stm32=replace(config.stm32, max_speed_x_deg_s=30.0)),
    )

    def publish_two_updates() -> None:
        assert hal.apply_config_update(update_1) is None
        assert hal.apply_config_update(update_2) is None

    transport.after_write = publish_two_updates
    hal.move_relative(MoveRelativeCommand(1.0, 0.0))

    requests = [decode_request(raw) for raw in transport.raw_write_history]
    assert [request.command for request in requests] == [
        CommandCode.MOVE_RELATIVE,
        CommandCode.SET_CONFIG,
    ]
    assert isinstance(requests[-1].payload, SetConfigPayload)
    assert requests[-1].payload.max_speed_x_steps_s == 267
    assert hal.applied_config_revision == 2
    assert hal.pending_config_revision is None


def test_motor_state_changes_only_after_matching_ok() -> None:
    endpoint = FakeStm32Endpoint()
    controller, hal, _, _, _ = _stack(transport=FakeTransport(endpoint))
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_STATE))

    controller.motor_on()
    assert hal.motor_state is MotorState.UNKNOWN

    controller.motor_on()
    assert hal.motor_state is MotorState.ON
    controller.motor_off()
    assert hal.motor_state is MotorState.OFF


def test_axis_pid_first_sample_is_p_only_and_next_sample_uses_real_dt() -> None:
    pid = AxisPid(PidGains(1.0, 1.0, 1.0), output_limit=10.0)

    assert pid.step(1.0, now_s=2.0) == pytest.approx(1.0)
    assert pid.i_term == 0.0
    assert pid.step(2.0, now_s=2.5) == pytest.approx(5.0)
    assert pid.i_term == pytest.approx(1.0)


def test_axis_pid_conditional_anti_windup_and_integral_limit() -> None:
    pid = AxisPid(PidGains(2.0, 1.0, 0.0), output_limit=1.0)

    assert pid.step(1.0, now_s=0.0) == 1.0
    assert pid.step(1.0, now_s=1.0) == 1.0
    assert pid.i_term == 0.0

    assert pid.step(-0.2, now_s=2.0) == pytest.approx(-0.6)
    assert pid.i_term == pytest.approx(-0.2)


def test_axis_pid_limit_decrease_clamps_i_and_increase_preserves_it() -> None:
    pid = AxisPid(PidGains(0.0, 1.0, 0.0), output_limit=10.0)
    pid.step(2.0, now_s=0.0)
    pid.step(2.0, now_s=1.0)
    assert pid.i_term == pytest.approx(2.0)

    pid.set_output_limit(0.5)
    assert pid.i_term == pytest.approx(0.5)
    assert pid.has_sample

    pid.set_output_limit(20.0)
    assert pid.i_term == pytest.approx(0.5)
    assert pid.has_sample


def test_controller_relative_pending_motion_is_latest_only() -> None:
    controller, _, _, transport, _ = _stack()
    controller.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
    latest = MoveRelativeCommand(2.0, -1.0)
    controller.submit_move_relative(latest)

    controller.flush_pending_motion()

    request = decode_request(transport.raw_write_history[-1])
    assert request.command is CommandCode.MOVE_RELATIVE
    assert request.payload == MoveRelativePayload(-18, -9)
    assert controller.pending_motion is None


def test_controller_tracking_pending_is_latest_and_revision_processed_once() -> None:
    controller, _, _, transport, clock = _stack()
    _enter_tracking_off(controller, transport)
    target = _target()

    assert controller.submit_tracking_error(1, _tracking_error(target, 1.0))
    first_pending = controller.pending_motion
    clock.now = 0.05
    assert not controller.submit_tracking_error(1, _tracking_error(target, 10.0))
    assert controller.pending_motion == first_pending
    assert controller.submit_tracking_error(2, _tracking_error(target, 2.0))

    controller.flush_pending_motion()

    assert len(transport.raw_write_history) == 1
    assert decode_request(transport.raw_write_history[0]).command is CommandCode.SET_VELOCITY
    assert controller.last_processed_revision == 2


def test_target_change_resets_pid_before_new_tracking_sample() -> None:
    controller, _, _, transport, clock = _stack()
    _enter_tracking_off(controller, transport)

    controller.submit_tracking_error(1, _tracking_error(_target(1), 1.0))
    clock.now = 0.05
    controller.submit_tracking_error(2, _tracking_error(_target(1), 2.0))
    assert controller.pid_x.i_term != 0.0

    clock.now = 0.06
    controller.submit_tracking_error(3, _tracking_error(_target(2), 1.0))

    assert controller.pid_x.i_term == 0.0
    assert controller.pending_motion == AxisVelocitySetpoint(2.0, 0.0, 1)


def test_long_tracking_gap_resets_pid_before_next_sample() -> None:
    controller, _, _, transport, clock = _stack(_config(watchdog_ms=100))
    _enter_tracking_off(controller, transport)
    target = _target()

    controller.submit_tracking_error(1, _tracking_error(target, 1.0))
    clock.now = 0.05
    controller.submit_tracking_error(2, _tracking_error(target, 2.0))
    assert controller.pid_x.i_term != 0.0

    clock.now = 0.2
    controller.submit_tracking_error(3, _tracking_error(target, 1.0))

    assert controller.pid_x.i_term == 0.0
    assert controller.pending_motion == AxisVelocitySetpoint(2.0, 0.0, 1)


def test_runtime_gain_change_resets_only_changed_axis_in_tracking() -> None:
    config = _config(kp_x=0.0, kd_x=0.0, kp_y=0.0, kd_y=0.0)
    controller, _, _, transport, clock = _stack(config)
    _enter_tracking_off(controller, transport)
    target = _target()
    controller.submit_tracking_error(1, _tracking_error(target, 1.0, 1.0))
    clock.now = 0.05
    controller.submit_tracking_error(2, _tracking_error(target, 1.0, 1.0))
    assert controller.pid_x.i_term > 0
    assert controller.pid_y.i_term > 0

    changed_pid = replace(config.controller, pid_kp_x=3.0)
    controller.apply_config_update(ConfigUpdate(1, replace(config, controller=changed_pid)))

    assert not controller.pid_x.has_sample
    assert controller.pid_x.i_term == 0.0
    assert controller.pid_y.has_sample
    assert controller.pid_y.i_term > 0


def test_runtime_limit_decrease_clamps_i_increase_preserves_and_gain_reset_wins() -> None:
    config = _config(
        kp_x=0.0, ki_x=1.0, kd_x=0.0, max_speed_x=10.0, watchdog_ms=1000
    )
    controller, _, _, transport, clock = _stack(config)
    _enter_tracking_off(controller, transport)
    target = _target()
    controller.submit_tracking_error(1, _tracking_error(target, 4.0))
    clock.now = 0.5
    controller.submit_tracking_error(2, _tracking_error(target, 4.0))
    assert controller.pid_x.i_term == pytest.approx(2.0)

    lower = replace(config, stm32=replace(config.stm32, max_speed_x_deg_s=0.5))
    controller.apply_config_update(ConfigUpdate(1, lower))
    assert controller.pid_x.i_term == pytest.approx(0.5)
    assert controller.pid_x.has_sample

    higher = replace(lower, stm32=replace(lower.stm32, max_speed_x_deg_s=20.0))
    controller.apply_config_update(ConfigUpdate(2, higher))
    assert controller.pid_x.i_term == pytest.approx(0.5)
    assert controller.pid_x.has_sample

    gains_and_limit = replace(
        higher,
        controller=replace(higher.controller, pid_kp_x=1.0),
        stm32=replace(higher.stm32, max_speed_x_deg_s=0.25),
    )
    controller.apply_config_update(ConfigUpdate(3, gains_and_limit))
    assert controller.pid_x.i_term == 0.0
    assert not controller.pid_x.has_sample
    assert controller.pid_x.output_limit == 0.25


def test_config_update_itself_does_not_change_mode_or_create_motion() -> None:
    config = _config()
    controller, _, _, _, _ = _stack(config)
    original = MoveRelativeCommand(1.0, 0.0)
    controller.submit_move_relative(original)
    changed = replace(config, controller=replace(config.controller, pid_kp_x=5.0))

    controller.apply_config_update(ConfigUpdate(1, changed))

    assert controller.control_mode is TurretControlMode.RELATIVE
    assert controller.pending_motion == original


def test_mode_transition_zero_handshake_applies_only_after_ok() -> None:
    endpoint = FakeStm32Endpoint()
    controller, hal, _, transport, _ = _stack(transport=FakeTransport(endpoint))
    controller.motor_on()
    transport.raw_write_history.clear()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_STATE))

    failed = controller.set_control_mode(TurretControlMode.TRACKING)
    assert failed is not None and failed.response is not None
    assert failed.response.result is ResultCode.INVALID_STATE
    assert controller.control_mode is TurretControlMode.RELATIVE

    succeeded = controller.set_control_mode(TurretControlMode.TRACKING)
    assert succeeded is not None and succeeded.response is not None
    assert succeeded.response.result is ResultCode.OK
    assert controller.control_mode is TurretControlMode.TRACKING
    requests = [decode_request(raw) for raw in transport.raw_write_history]
    assert all(request.command is CommandCode.SET_VELOCITY for request in requests)
    assert all(request.payload == SetVelocityPayload(0, 0) for request in requests)
    assert hal.motor_state is MotorState.ON


@pytest.mark.parametrize("mode", [TurretControlMode.RELATIVE, TurretControlMode.TRACKING])
def test_stop_motion_invalidates_pending_keeps_mode_and_sends_zero_when_on(
    mode: TurretControlMode,
) -> None:
    controller, _, _, transport, _ = _stack()
    controller.motor_on()
    if mode is TurretControlMode.TRACKING:
        controller.set_control_mode(mode)
        controller.submit_tracking_error(1, _tracking_error(_target(), 1.0))
    else:
        controller.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
    transport.raw_write_history.clear()

    result = controller.stop_motion()

    assert result is not None and result.response is not None
    assert result.response.result is ResultCode.OK
    assert controller.control_mode is mode
    assert controller.pending_motion is None
    request = decode_request(transport.raw_write_history[-1])
    assert request.command is CommandCode.SET_VELOCITY
    assert request.payload == SetVelocityPayload(0, 0)


def test_motor_off_invalidates_motion_and_motor_on_does_not_replay_it() -> None:
    controller, hal, _, transport, _ = _stack()
    controller.submit_move_relative(MoveRelativeCommand(5.0, 0.0))

    controller.motor_off()
    assert controller.pending_motion is None
    assert hal.motor_state is MotorState.OFF

    transport.raw_write_history.clear()
    controller.submit_move_relative(MoveRelativeCommand(3.0, 0.0))
    controller.motor_on()

    assert hal.motor_state is MotorState.ON
    assert controller.pending_motion is None
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.MOTOR_ON
    ]


def test_zero_velocity_tracking_setpoint_is_ordinary_motion_not_hidden_stop() -> None:
    controller, _, _, transport, _ = _stack()
    _enter_tracking_off(controller, transport)

    controller.submit_tracking_error(1, _tracking_error(_target(), 0.0, 0.0))
    assert controller.pid_x.has_sample and controller.pid_y.has_sample
    controller.flush_pending_motion()

    request = decode_request(transport.raw_write_history[-1])
    assert request.command is CommandCode.SET_VELOCITY
    assert request.payload == SetVelocityPayload(0, 0)
    assert controller.control_mode is TurretControlMode.TRACKING
    assert controller.pid_x.has_sample and controller.pid_y.has_sample


def test_emergency_clears_pending_and_resets_pid_via_existing_session_path() -> None:
    controller, _, session, transport, _ = _stack()
    _enter_tracking_off(controller, transport)
    controller.submit_tracking_error(1, _tracking_error(_target(), 1.0))
    transport.raw_write_history.clear()

    response = controller.emergency_stop()

    assert response is not None and response.result is ResultCode.OK
    assert controller.pending_motion is None
    assert not controller.pid_x.has_sample and not controller.pid_y.has_sample
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.EMERGENCY_STOP
    ]
    assert session.next_request_id == 2


def test_tracking_invalidation_and_recovery_boundary_reset_controller_state() -> None:
    controller, hal, _, transport, _ = _stack()
    _enter_tracking_off(controller, transport)
    controller.submit_tracking_error(4, _tracking_error(_target(), 1.0))

    assert controller.invalidate_tracking_error(5)
    assert controller.pending_motion is None
    assert not controller.pid_x.has_sample
    assert controller.last_processed_revision == 5

    controller.submit_tracking_error(6, _tracking_error(_target(), 1.0))
    controller.reset_for_transport_recovery()
    assert controller.pending_motion is None
    assert not controller.pid_x.has_sample
    assert hal.motor_state is MotorState.UNKNOWN


def test_gain_change_outside_tracking_does_not_force_immediate_pid_reset() -> None:
    config = _config(kp_x=0.0, ki_x=1.0, kd_x=0.0, watchdog_ms=1000)
    controller, _, _, transport, clock = _stack(config)
    _enter_tracking_off(controller, transport)
    target = _target()
    controller.submit_tracking_error(1, _tracking_error(target, 2.0))
    clock.now = 0.5
    controller.submit_tracking_error(2, _tracking_error(target, 2.0))
    assert controller.pid_x.i_term == pytest.approx(1.0)

    controller.set_control_mode(TurretControlMode.RELATIVE)
    # The mode exit is itself a reset boundary; build state explicitly to prove
    # the config update does not add another hidden reset outside TRACKING.
    controller.pid_x.step(1.0, now_s=1.0)
    controller.pid_x.step(1.0, now_s=1.5)
    before = controller.pid_x.i_term
    changed = replace(config, controller=replace(config.controller, pid_kp_x=3.0))

    controller.apply_config_update(ConfigUpdate(1, changed))

    assert controller.pid_x.has_sample
    assert controller.pid_x.i_term == before
    assert controller.pid_x.gains.kp == 3.0


def test_tracking_to_relative_uses_zero_handshake_before_applying_mode() -> None:
    endpoint = FakeStm32Endpoint()
    controller, _, _, transport, _ = _stack(transport=FakeTransport(endpoint))
    controller.motor_on()
    controller.set_control_mode(TurretControlMode.TRACKING)
    transport.raw_write_history.clear()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_STATE))

    failed = controller.set_control_mode(TurretControlMode.RELATIVE)
    assert failed is not None and failed.response is not None
    assert failed.response.result is ResultCode.INVALID_STATE
    assert controller.control_mode is TurretControlMode.TRACKING

    succeeded = controller.set_control_mode(TurretControlMode.RELATIVE)
    assert succeeded is not None and succeeded.response is not None
    assert succeeded.response.result is ResultCode.OK
    assert controller.control_mode is TurretControlMode.RELATIVE
    assert [decode_request(raw).payload for raw in transport.raw_write_history] == [
        SetVelocityPayload(0, 0),
        SetVelocityPayload(0, 0),
    ]


def test_motor_off_invalidates_tracking_motion_and_resets_pid() -> None:
    controller, hal, _, transport, _ = _stack()
    controller.motor_on()
    controller.set_control_mode(TurretControlMode.TRACKING)
    controller.submit_tracking_error(1, _tracking_error(_target(), 1.0))
    assert controller.pid_x.has_sample
    assert controller.pending_motion is not None
    transport.raw_write_history.clear()

    result = controller.motor_off()

    assert result.response is not None and result.response.result is ResultCode.OK
    assert controller.pending_motion is None
    assert not controller.pid_x.has_sample and not controller.pid_y.has_sample
    assert hal.motor_state is MotorState.OFF
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.MOTOR_OFF
    ]
