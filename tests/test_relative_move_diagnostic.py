from __future__ import annotations

from dataclasses import replace

import pytest

from navmin.concurrency import LatestValue
from navmin.config import load_config
from navmin.contracts import (
    MotorState,
    MoveRelativeCommand,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
)
from navmin.diagnostics.relative_move import (
    RelativeMoveDiagnosticError,
    build_relative_move_diagnostic_plan,
    execute_relative_move_diagnostic,
    format_relative_move_plan,
)
from navmin.turret.protocol import MoveRelativePayload


def _config():
    config = load_config("config/defaults/navmin.json")
    axis = replace(
        config.turret.axes.x,
        invert=False,
        full_steps_per_revolution=4000,
        microstep_divider=1,
        max_relative_move_deg=45.0,
    )
    turret = replace(
        config.turret,
        axes=replace(config.turret.axes, x=axis, y=axis),
        emulate_stm32=False,
    )
    return config, turret


def _state(
    *,
    connection: TurretConnectionState = TurretConnectionState.READY,
    motor: MotorState = MotorState.OFF,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
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


class _FakeWorker:
    def __init__(self, *, startup_state: TurretState | None = None) -> None:
        self.state_updates: LatestValue[TurretState] = LatestValue()
        self.state_updates.publish(
            startup_state
            or _state(connection=TurretConnectionState.DISCONNECTED)
        )
        self.moves: list[MoveRelativeCommand] = []
        self.motor_on_calls = 0
        self.motor_off_calls = 0
        self.emergency_calls = 0
        self.shutdown_calls = 0
        self.on_submit = None

    @property
    def current_state(self) -> TurretState:
        state = self.state_updates.get()
        assert state is not None
        return state

    def start(self) -> None:
        current = self.current_state
        if current.connection_state is TurretConnectionState.DISCONNECTED:
            self.state_updates.publish(_state())

    def motor_on(self) -> bool:
        self.motor_on_calls += 1
        current = self.current_state
        self.state_updates.publish(replace(current, motor_state=MotorState.ON))
        return True

    def motor_off(self) -> bool:
        self.motor_off_calls += 1
        current = self.current_state
        if current.connection_state is not TurretConnectionState.READY:
            return False
        self.state_updates.publish(replace(current, motor_state=MotorState.OFF))
        return True

    def submit_move_relative(self, command: MoveRelativeCommand) -> None:
        callback = self.on_submit
        if callback is not None:
            callback()
        self.moves.append(command)
        self.state_updates.publish(self.current_state)

    def request_emergency(self) -> None:
        self.emergency_calls += 1
        self.state_updates.publish(self.current_state)

    def shutdown(self, timeout: float) -> None:
        assert timeout > 0
        self.shutdown_calls += 1


def test_plan_uses_exact_4000_step_payload_without_camera_or_calibration() -> None:
    _, turret = _config()

    plan = build_relative_move_diagnostic_plan(turret, axis="x", degrees=4.0)

    assert plan.command == MoveRelativeCommand(4.0, 0.0)
    assert plan.preview.payload == MoveRelativePayload(44, 0)
    assert plan.selected_axis_effective_steps_per_revolution == 4000
    assert plan.selected_axis_quantized_degrees == pytest.approx(3.96)
    assert "microstep-divider: 1" in format_relative_move_plan(plan)


def test_execute_requires_two_confirmations_and_cleans_up_motor_state() -> None:
    config, turret = _config()
    plan = build_relative_move_diagnostic_plan(turret, axis="x", degrees=4.0)
    worker = _FakeWorker()
    answers = iter(("MOTOR ON", "MOVE", ""))

    execute_relative_move_diagnostic(
        turret,
        config.tracking_control,
        plan,
        worker_factory=lambda _turret, _tracking: worker,
        prompt=lambda _message: next(answers),
        output=lambda _message: None,
        timeout_s=0.1,
    )

    assert worker.motor_on_calls == 1
    assert worker.moves == [MoveRelativeCommand(4.0, 0.0)]
    assert worker.motor_off_calls == 1
    assert worker.current_state.motor_state is MotorState.OFF
    assert worker.shutdown_calls == 1


def test_execute_blocks_move_if_mode_changes_before_second_confirmation() -> None:
    config, turret = _config()
    plan = build_relative_move_diagnostic_plan(turret, axis="x", degrees=4.0)
    worker = _FakeWorker()
    prompt_count = 0

    def prompt(_message: str) -> str:
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count == 1:
            return "MOTOR ON"
        if prompt_count == 2:
            worker.state_updates.publish(
                replace(worker.current_state, control_mode=TurretControlMode.TRACKING)
            )
            return "MOVE"
        return ""

    with pytest.raises(RelativeMoveDiagnosticError, match="RELATIVE"):
        execute_relative_move_diagnostic(
            turret,
            config.tracking_control,
            plan,
            worker_factory=lambda _turret, _tracking: worker,
            prompt=prompt,
            output=lambda _message: None,
            timeout_s=0.1,
        )

    assert worker.moves == []
    assert worker.motor_off_calls == 1
    assert worker.shutdown_calls == 1


def test_execute_blocks_move_if_motors_are_no_longer_on_before_send() -> None:
    config, turret = _config()
    plan = build_relative_move_diagnostic_plan(turret, axis="x", degrees=1.0)
    worker = _FakeWorker()
    prompt_count = 0

    def prompt(_message: str) -> str:
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count == 1:
            return "MOTOR ON"
        if prompt_count == 2:
            worker.state_updates.publish(
                replace(worker.current_state, motor_state=MotorState.OFF)
            )
            return "MOVE"
        return ""

    with pytest.raises(RelativeMoveDiagnosticError, match="not confirmed ON"):
        execute_relative_move_diagnostic(
            turret,
            config.tracking_control,
            plan,
            worker_factory=lambda _turret, _tracking: worker,
            prompt=prompt,
            output=lambda _message: None,
            timeout_s=0.1,
        )

    assert worker.moves == []
    assert worker.shutdown_calls == 1


def test_execute_blocks_move_if_connection_is_lost_before_send() -> None:
    config, turret = _config()
    plan = build_relative_move_diagnostic_plan(turret, axis="y", degrees=-2.0)
    worker = _FakeWorker()
    prompt_count = 0

    def prompt(_message: str) -> str:
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count == 1:
            return "MOTOR ON"
        if prompt_count == 2:
            worker.state_updates.publish(
                replace(
                    worker.current_state,
                    connection_state=TurretConnectionState.CONNECTING,
                )
            )
            return "MOVE"
        return ""

    with pytest.raises(RelativeMoveDiagnosticError, match="connection is not READY"):
        execute_relative_move_diagnostic(
            turret,
            config.tracking_control,
            plan,
            worker_factory=lambda _turret, _tracking: worker,
            prompt=prompt,
            output=lambda _message: None,
            timeout_s=0.1,
        )

    assert worker.moves == []
    assert worker.motor_off_calls == 1
    assert worker.emergency_calls == 1
    assert worker.shutdown_calls == 1


def test_keyboard_interrupt_requests_emergency_before_cleanup() -> None:
    config, turret = _config()
    plan = build_relative_move_diagnostic_plan(turret, axis="x", degrees=1.0)
    worker = _FakeWorker()
    answers = iter(("MOTOR ON",))

    def prompt(_message: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt

    with pytest.raises(RelativeMoveDiagnosticError, match="interrupted"):
        execute_relative_move_diagnostic(
            turret,
            config.tracking_control,
            plan,
            worker_factory=lambda _turret, _tracking: worker,
            prompt=prompt,
            output=lambda _message: None,
            timeout_s=0.1,
        )

    assert worker.moves == []
    assert worker.emergency_calls == 1
    assert worker.motor_off_calls == 1
    assert worker.shutdown_calls == 1
