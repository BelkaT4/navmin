from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from threading import Event
from time import monotonic

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
    CameraRole,
    ConfigUpdate,
    MotorState,
    MoveRelativeCommand,
    TargetRef,
    TrackingError,
    TurretConnectionState,
    TurretControlMode,
)
from navmin.turret.protocol import CommandCode, ResultCode, decode_request
from navmin.turret.simulator import (
    FakeReadFailure,
    FakeResponseSpec,
    FakeStm32Endpoint,
    FakeTransport,
)
from navmin.turret.transport import TransportDisconnectedError
from navmin.turret.worker import TurretWorker, WorkerShutdownError


def _config(
    *,
    baudrate: int = 9600,
    response_timeout_ms: int = 10,
    max_retries: int = 0,
    inter_request_delay_ms: int = 1,
    emulate_stm32: bool = True,
) -> TurretConfig:
    axis = AxisMechanicsConfig(False, 2000, 16, 45.0)
    return TurretConfig(
        serial=SerialConfig(
            "/dev/ttyUSB0",
            baudrate,
            response_timeout_ms,
            max_retries,
            inter_request_delay_ms,
        ),
        axes=AxesConfig(axis, axis),
        controller=PidControllerConfig(1.0, 0.1, 0.0, 1.0, 0.1, 0.0),
        stm32=Stm32Config(50.0, 60.0, 100.0, 120.0, 200),
        emulate_stm32=emulate_stm32,
    )


class _FailOpenTransport(FakeTransport):
    def open(self) -> None:
        raise TransportDisconnectedError("simulated missing serial device")


class _RecordingFactory:
    def __init__(self, endpoint: FakeStm32Endpoint | None = None) -> None:
        self.endpoint = endpoint or FakeStm32Endpoint()
        self.calls: list[tuple[str, int, bool]] = []
        self.transports: list[FakeTransport] = []
        self.fail_open_count = 0
        self.always_fail = False
        self.transport_type: type[FakeTransport] = FakeTransport

    def __call__(self, port: str, baudrate: int, emulate: bool) -> FakeTransport:
        self.calls.append((port, baudrate, emulate))
        if self.always_fail or self.fail_open_count > 0:
            if self.fail_open_count > 0:
                self.fail_open_count -= 1
            transport = _FailOpenTransport(self.endpoint, baudrate=baudrate)
        else:
            transport = self.transport_type(self.endpoint, baudrate=baudrate)
        self.transports.append(transport)
        return transport




class _CallbackFakeTransport(FakeTransport):
    def __init__(self, endpoint: FakeStm32Endpoint, *, baudrate: int) -> None:
        super().__init__(endpoint, baudrate=baudrate)
        self.after_command: dict[CommandCode, Callable[[], None]] = {}

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        super().write_frame(frame, timeout_s)
        command = decode_request(frame).command
        callback = self.after_command.pop(command, None)
        if callback is not None:
            callback()


class _BlockingFakeTransport(FakeTransport):
    def __init__(self, endpoint: FakeStm32Endpoint, *, baudrate: int) -> None:
        super().__init__(endpoint, baudrate=baudrate)
        self._blocked_command: CommandCode | None = None
        self.write_seen = Event()
        self.release_read = Event()
        self._block_current_read = False

    def block_next(self, command: CommandCode) -> None:
        self._blocked_command = command
        self.write_seen.clear()
        self.release_read.clear()

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        super().write_frame(frame, timeout_s)
        request = decode_request(frame)
        if request.command is self._blocked_command:
            self._blocked_command = None
            self._block_current_read = True
            self.write_seen.set()

    def read_frame(self, timeout_s: float) -> bytes:
        if self._block_current_read:
            self._block_current_read = False
            if not self.release_read.wait(timeout_s):
                return super().read_frame(timeout_s)
        return super().read_frame(timeout_s)


class _BlockingRecoveryFactory(_RecordingFactory):
    def __init__(self, endpoint: FakeStm32Endpoint | None = None) -> None:
        super().__init__(endpoint)
        self.transport_type = _BlockingFakeTransport
        self.block_next_recovery_set_config = False
        self.blocked_transport_ready = Event()
        self.blocked_transport: _BlockingFakeTransport | None = None

    def __call__(self, port: str, baudrate: int, emulate: bool) -> FakeTransport:
        transport = super().__call__(port, baudrate, emulate)
        if self.block_next_recovery_set_config:
            assert isinstance(transport, _BlockingFakeTransport)
            self.block_next_recovery_set_config = False
            transport.block_next(CommandCode.SET_CONFIG)
            self.blocked_transport = transport
            self.blocked_transport_ready.set()
        return transport


def _wait_state(
    worker: TurretWorker,
    expected: TurretConnectionState,
    *,
    timeout: float = 1.0,
):
    snapshot = worker.state_updates.snapshot()
    deadline = monotonic() + timeout
    while True:
        state = snapshot.value
        if state is not None and state.connection_state is expected:
            return state
        remaining = deadline - monotonic()
        assert remaining > 0, f"timed out waiting for {expected.value}: {state}"
        next_snapshot = worker.state_updates.wait_for_revision(
            snapshot.revision, remaining
        )
        assert next_snapshot is not None
        snapshot = next_snapshot


def _wait_factory_calls(factory: _RecordingFactory, count: int, timeout: float = 1.0) -> None:
    deadline = monotonic() + timeout
    while len(factory.calls) < count:
        assert monotonic() < deadline
        Event().wait(0.002)


def _shutdown(worker: TurretWorker) -> None:
    if worker.is_alive():
        worker.shutdown(1.0)


def test_startup_recovery_order_ready_state_and_no_automatic_motor_on() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(baudrate=115200), transport_factory=factory)
    assert worker.current_state.connection_state is TurretConnectionState.DISCONNECTED
    assert worker.current_state.max_speed_x_deg_s is None

    worker.start()
    try:
        state = _wait_state(worker, TurretConnectionState.READY)
        commands = [request.command for request in endpoint.request_history]
        assert commands == [
            CommandCode.EMERGENCY_STOP,
            CommandCode.MOTOR_OFF,
            CommandCode.SET_BAUDRATE,
            CommandCode.SET_CONFIG,
        ]
        assert [baud for _, baud, _ in factory.calls] == [115200, 9600]
        assert CommandCode.MOTOR_ON not in commands
        assert state.motor_state is MotorState.OFF
        assert state.control_mode is TurretControlMode.RELATIVE
        assert state.max_speed_x_deg_s == 50.0
        assert state.max_speed_y_deg_s == 60.0
        assert state.acceleration_x_deg_s2 == 100.0
        assert state.acceleration_y_deg_s2 == 120.0
    finally:
        _shutdown(worker)

    stopped = worker.current_state
    assert worker.ready is False
    assert stopped.connection_state is TurretConnectionState.DISCONNECTED
    assert stopped.motor_state is MotorState.UNKNOWN
    assert stopped.max_speed_x_deg_s is None
    assert stopped.acceleration_y_deg_s2 is None
    assert factory.transports[-1].is_open is False


def test_cross_thread_emergency_signal_never_writes_from_caller_thread() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.transport_type = _BlockingFakeTransport
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        transport = worker._transport
        assert isinstance(transport, _BlockingFakeTransport)
        transport.block_next(CommandCode.MOVE_RELATIVE)
        before = len(transport.raw_write_history)

        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        assert transport.write_seen.wait(0.5)
        assert len(transport.raw_write_history) == before + 1

        worker.request_emergency()
        # Caller-side request only signals; the worker cannot write Emergency
        # until the already-written physical attempt is allowed to finish.
        assert len(transport.raw_write_history) == before + 1
        transport.release_read.set()

        deadline = monotonic() + 1.0
        while endpoint.request_history[-1].command is not CommandCode.EMERGENCY_STOP:
            assert monotonic() < deadline
            Event().wait(0.002)
        assert [r.command for r in endpoint.request_history[-2:]] == [
            CommandCode.MOVE_RELATIVE,
            CommandCode.EMERGENCY_STOP,
        ]
    finally:
        _shutdown(worker)


def test_invalid_request_id_resyncs_with_emergency_without_reconnect() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        call_count = len(factory.calls)
        endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_REQUEST_ID))
        before = len(endpoint.request_history)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))

        deadline = monotonic() + 1.0
        while len(endpoint.request_history) < before + 2:
            assert monotonic() < deadline
            Event().wait(0.002)
        assert [r.command for r in endpoint.request_history[-2:]] == [
            CommandCode.MOVE_RELATIVE,
            CommandCode.EMERGENCY_STOP,
        ]
        assert len(factory.calls) == call_count
        assert worker.ready
    finally:
        _shutdown(worker)


def test_control_mailbox_preserves_set_mode_then_motor_on_order() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.transport_type = _BlockingFakeTransport
    worker = TurretWorker(
        _config(response_timeout_ms=1_000),
        transport_factory=factory,
    )
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        transport = factory.transports[-1]
        assert isinstance(transport, _BlockingFakeTransport)
        changed = replace(
            worker._desired_config,
            stm32=replace(worker._desired_config.stm32, max_speed_x_deg_s=71.0),
        )
        transport.block_next(CommandCode.SET_CONFIG)
        history_start = len(endpoint.request_history)
        worker.submit_config_update(ConfigUpdate(1, changed))
        assert transport.write_seen.wait(0.5)

        assert worker.set_control_mode(TurretControlMode.TRACKING)
        assert worker.motor_on()
        transport.release_read.set()

        deadline = monotonic() + 1.0
        while not (
            worker.current_state.control_mode is TurretControlMode.TRACKING
            and worker.current_state.motor_state is MotorState.ON
        ):
            assert monotonic() < deadline
            Event().wait(0.002)

        assert [
            request.command for request in endpoint.request_history[history_start:]
        ] == [
            CommandCode.SET_CONFIG,
            CommandCode.MOTOR_ON,
        ]
    finally:
        _shutdown(worker)


def test_control_mailbox_preserves_motor_off_then_set_mode_order() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.transport_type = _BlockingFakeTransport
    worker = TurretWorker(
        _config(response_timeout_ms=1_000),
        transport_factory=factory,
    )
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        assert worker.motor_on()
        deadline = monotonic() + 0.5
        while worker.current_state.motor_state is not MotorState.ON:
            assert monotonic() < deadline
            Event().wait(0.002)

        transport = factory.transports[-1]
        assert isinstance(transport, _BlockingFakeTransport)
        transport.block_next(CommandCode.MOVE_RELATIVE)
        history_start = len(endpoint.request_history)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        assert transport.write_seen.wait(0.5)

        assert worker.motor_off()
        assert worker.set_control_mode(TurretControlMode.TRACKING)
        transport.release_read.set()

        deadline = monotonic() + 1.0
        while not (
            worker.current_state.motor_state is MotorState.OFF
            and worker.current_state.control_mode is TurretControlMode.TRACKING
        ):
            assert monotonic() < deadline
            Event().wait(0.002)

        assert [
            request.command for request in endpoint.request_history[history_start:]
        ] == [
            CommandCode.MOVE_RELATIVE,
            CommandCode.MOTOR_OFF,
        ]
    finally:
        _shutdown(worker)


def test_control_mailbox_preserves_stop_motion_before_mode_request() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.transport_type = _BlockingFakeTransport
    worker = TurretWorker(
        _config(response_timeout_ms=1_000),
        transport_factory=factory,
    )
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        assert worker.motor_on()
        deadline = monotonic() + 0.5
        while worker.current_state.motor_state is not MotorState.ON:
            assert monotonic() < deadline
            Event().wait(0.002)

        transport = factory.transports[-1]
        assert isinstance(transport, _BlockingFakeTransport)
        transport.block_next(CommandCode.MOVE_RELATIVE)
        history_start = len(endpoint.request_history)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        assert transport.write_seen.wait(0.5)

        assert worker.stop_motion()
        assert worker.set_control_mode(TurretControlMode.TRACKING)
        transport.release_read.set()

        deadline = monotonic() + 1.0
        while worker.current_state.control_mode is not TurretControlMode.TRACKING:
            assert monotonic() < deadline
            Event().wait(0.002)

        assert [
            request.command for request in endpoint.request_history[history_start:]
        ] == [
            CommandCode.MOVE_RELATIVE,
            CommandCode.SET_VELOCITY,
            CommandCode.SET_VELOCITY,
        ]
        assert worker.current_state.motor_state is MotorState.ON
    finally:
        _shutdown(worker)


def test_accepted_control_mailbox_is_discarded_when_recovery_begins() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.transport_type = _BlockingFakeTransport
    worker = TurretWorker(
        _config(response_timeout_ms=1_000),
        transport_factory=factory,
    )
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        transport = factory.transports[-1]
        assert isinstance(transport, _BlockingFakeTransport)
        transport.block_next(CommandCode.MOVE_RELATIVE)
        history_start = len(endpoint.request_history)
        factory_calls = len(factory.calls)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        assert transport.write_seen.wait(0.5)

        assert worker.set_control_mode(TurretControlMode.TRACKING)
        assert worker.motor_on()
        transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        transport.release_read.set()

        _wait_factory_calls(factory, factory_calls + 1)
        state = _wait_state(worker, TurretConnectionState.READY)
        Event().wait(0.03)

        commands = [
            request.command for request in endpoint.request_history[history_start:]
        ]
        assert CommandCode.MOTOR_ON not in commands
        assert CommandCode.SET_VELOCITY not in commands
        assert state.motor_state is MotorState.OFF
        assert state.control_mode is TurretControlMode.RELATIVE
    finally:
        _shutdown(worker)


def test_invalid_request_id_failed_emergency_resync_enters_recovery() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        call_count = len(factory.calls)
        endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_REQUEST_ID))
        endpoint.queue_response(FakeResponseSpec(result=ResultCode.INTERNAL_ERROR))
        factory.always_fail = True
        history_start = len(endpoint.request_history)

        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))

        state = _wait_state(worker, TurretConnectionState.CONNECTING)
        assert not worker.ready
        assert state.connection_state is TurretConnectionState.CONNECTING
        assert [request.command for request in endpoint.request_history[history_start:]] == [
            CommandCode.MOVE_RELATIVE,
            CommandCode.EMERGENCY_STOP,
        ]
        _wait_factory_calls(factory, call_count + 1)
    finally:
        _shutdown(worker)


def test_matching_command_error_does_not_trigger_physical_reconnect() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        call_count = len(factory.calls)
        state_revision = worker.state_updates.snapshot().revision
        endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_ARGUMENT))
        worker.submit_move_relative(MoveRelativeCommand(2.0, 0.0))
        assert worker.state_updates.wait_for_revision(state_revision, 0.5) is not None
        assert len(factory.calls) == call_count
        assert worker.ready
    finally:
        _shutdown(worker)


def test_transport_loss_publishes_unknown_physical_state_and_preserves_mode() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        worker.set_control_mode(TurretControlMode.TRACKING)
        deadline = monotonic() + 0.5
        while worker.current_state.control_mode is not TurretControlMode.TRACKING:
            assert monotonic() < deadline
            Event().wait(0.002)

        worker.submit_tracking_error(
            TrackingError(TargetRef(CameraRole.OVERVIEW, 1, 1), 1.0, 0.0, 1)
        )
        deadline = monotonic() + 0.5
        while not worker._controller.pid_x.has_sample:
            assert monotonic() < deadline
            Event().wait(0.002)

        factory.always_fail = True
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        worker.submit_tracking_error(
            TrackingError(TargetRef(CameraRole.OVERVIEW, 1, 1), 2.0, 0.0, 2)
        )
        state = _wait_state(worker, TurretConnectionState.CONNECTING)
        assert state.motor_state is MotorState.UNKNOWN
        assert state.max_speed_x_deg_s is None
        assert state.acceleration_y_deg_s2 is None
        assert state.control_mode is TurretControlMode.TRACKING
        assert worker._controller.pending_motion is None
        assert worker._controller.pid_x.has_sample is False
    finally:
        _shutdown(worker)


def test_reconnect_candidates_are_last_known_desired_startup_and_deduplicated() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(baudrate=38400), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        initial_calls = len(factory.calls)
        worker._last_known_baud = 19200
        endpoint._baudrate = 9600
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        _wait_state(worker, TurretConnectionState.CONNECTING)
        _wait_state(worker, TurretConnectionState.READY)
        reconnect_bauds = [baud for _, baud, _ in factory.calls[initial_calls:]]
        assert reconnect_bauds[:3] == [19200, 38400, 9600]
    finally:
        _shutdown(worker)


def test_reconnect_has_no_finite_attempt_limit_backoff_caps_and_resets() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.fail_open_count = 5
    waits: list[float] = []

    def wait_without_sleep(stop_token, timeout_s: float) -> bool:
        waits.append(timeout_s)
        return stop_token.is_stop_requested()

    worker = TurretWorker(
        _config(), transport_factory=factory, cooperative_wait=wait_without_sleep
    )
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        assert waits[:5] == [0.25, 0.5, 1.0, 2.0, 2.0]
        assert len(factory.calls) >= 6

        waits_before = len(waits)
        factory.fail_open_count = 1
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        _wait_state(worker, TurretConnectionState.CONNECTING)
        _wait_state(worker, TurretConnectionState.READY)
        assert waits[waits_before] == 0.25
    finally:
        _shutdown(worker)


def test_stop_token_interrupts_reconnect_backoff_and_final_state_is_disconnected() -> None:
    factory = _RecordingFactory()
    factory.always_fail = True
    entered_backoff = Event()

    def cooperative_wait(stop_token, timeout_s: float) -> bool:
        entered_backoff.set()
        return stop_token.wait(timeout_s)

    worker = TurretWorker(
        _config(), transport_factory=factory, cooperative_wait=cooperative_wait
    )
    worker.start()
    assert entered_backoff.wait(0.5)
    started = monotonic()
    worker.request_stop()
    worker.join(0.2)
    elapsed = monotonic() - started
    assert not worker.is_alive()
    assert elapsed < 0.2
    assert worker.ready is False
    assert worker.current_state.connection_state is TurretConnectionState.DISCONNECTED


def test_bounded_shutdown_reports_alive_worker_and_logs_error(caplog) -> None:
    factory = _RecordingFactory()
    factory.always_fail = True
    entered_wait = Event()
    release = Event()

    def stuck_wait(stop_token, timeout_s: float) -> bool:
        del timeout_s
        entered_wait.set()
        release.wait()
        return stop_token.is_stop_requested()

    worker = TurretWorker(
        _config(), transport_factory=factory, cooperative_wait=stuck_wait
    )
    worker.start()
    assert entered_wait.wait(0.5)
    with (
        caplog.at_level("ERROR", logger="navmin.turret.worker"),
        pytest.raises(WorkerShutdownError),
    ):
        worker.shutdown(0.01)
    assert "failed to stop" in caplog.text.lower()
    release.set()
    worker.join(0.5)
    assert not worker.is_alive()


def test_dynamic_set_config_publishes_only_confirmed_applied_snapshot() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        updated = replace(
            worker._desired_config,
            stm32=replace(worker._desired_config.stm32, max_speed_x_deg_s=77.0),
        )
        worker.submit_config_update(ConfigUpdate(1, updated))
        deadline = monotonic() + 0.5
        while worker.current_state.max_speed_x_deg_s != 77.0:
            assert monotonic() < deadline
            Event().wait(0.002)
        assert endpoint.request_history[-1].command is CommandCode.SET_CONFIG

        endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_ARGUMENT))
        rejected = replace(
            updated,
            stm32=replace(updated.stm32, max_speed_x_deg_s=88.0),
        )
        worker.submit_config_update(ConfigUpdate(2, rejected))
        state_revision = worker.state_updates.snapshot().revision
        assert worker.state_updates.wait_for_revision(state_revision, 0.5) is not None
        assert worker.current_state.max_speed_x_deg_s == 77.0
    finally:
        _shutdown(worker)


def test_baud_update_is_latest_only_and_deferred_while_motors_on() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        worker.motor_on()
        deadline = monotonic() + 0.5
        while worker.current_state.motor_state is not MotorState.ON:
            assert monotonic() < deadline
            Event().wait(0.002)

        baseline_baud_commands = sum(
            request.command is CommandCode.SET_BAUDRATE
            for request in endpoint.request_history
        )
        cfg1 = replace(
            worker._desired_config,
            serial=replace(worker._desired_config.serial, baudrate=57600),
        )
        cfg2 = replace(cfg1, serial=replace(cfg1.serial, baudrate=115200))
        worker.submit_config_update(ConfigUpdate(1, cfg1))
        worker.submit_config_update(ConfigUpdate(2, cfg2))
        Event().wait(0.03)
        assert worker._transport.baudrate == 9600
        assert sum(
            request.command is CommandCode.SET_BAUDRATE
            for request in endpoint.request_history
        ) == baseline_baud_commands

        worker.motor_off()
        deadline = monotonic() + 0.5
        while worker._transport.baudrate != 115200:
            assert monotonic() < deadline
            Event().wait(0.002)
        assert worker.current_state.motor_state is MotorState.OFF
        assert endpoint.request_history[-2].command is CommandCode.MOTOR_OFF
        assert endpoint.request_history[-1].command is CommandCode.SET_BAUDRATE
    finally:
        _shutdown(worker)


def test_serial_timing_change_reconnects_and_runtime_emulation_switch_is_ignored() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(emulate_stm32=True), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        before = len(factory.calls)
        changed = replace(
            worker._desired_config,
            serial=replace(worker._desired_config.serial, response_timeout_ms=25),
            emulate_stm32=False,
        )
        worker.submit_config_update(ConfigUpdate(1, changed))
        _wait_factory_calls(factory, before + 1)
        deadline = monotonic() + 1.0
        while not worker.ready or worker._session._response_timeout_s != 0.025:
            assert monotonic() < deadline
            Event().wait(0.002)
        assert all(emulate is True for _, _, emulate in factory.calls[before:])
    finally:
        _shutdown(worker)


def test_normal_motion_submitted_during_recovery_is_not_replayed_after_ready() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.fail_open_count = 1
    entered_backoff = Event()
    release_backoff = Event()

    def gated_wait(stop_token, timeout_s: float) -> bool:
        del timeout_s
        entered_backoff.set()
        release_backoff.wait(0.5)
        return stop_token.is_stop_requested()

    worker = TurretWorker(
        _config(), transport_factory=factory, cooperative_wait=gated_wait
    )
    worker.start()
    try:
        assert entered_backoff.wait(0.5)
        worker.submit_move_relative(MoveRelativeCommand(10.0, 0.0))
        worker.motor_on()
        release_backoff.set()
        _wait_state(worker, TurretConnectionState.READY)
        commands = [request.command for request in endpoint.request_history]
        assert CommandCode.MOVE_RELATIVE not in commands
        assert CommandCode.MOTOR_ON not in commands
        assert worker.current_state.motor_state is MotorState.OFF
    finally:
        _shutdown(worker)


def test_ordinary_retry_exhaustion_enters_recovery() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(max_retries=0), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        before = len(factory.calls)
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        transport.queue_read_failure(FakeReadFailure.TIMEOUT)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        _wait_factory_calls(factory, before + 1)
        _wait_state(worker, TurretConnectionState.READY)
        assert len(factory.calls) > before
    finally:
        _shutdown(worker)


def test_emergency_retry_exhaustion_enters_recovery() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(max_retries=0), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        before = len(factory.calls)
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        transport.queue_read_failure(FakeReadFailure.TIMEOUT)
        worker.request_emergency()
        _wait_factory_calls(factory, before + 1)
        _wait_state(worker, TurretConnectionState.READY)
        assert len(factory.calls) > before
    finally:
        _shutdown(worker)


def test_failed_bounded_baud_recovery_enters_worker_recovery() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(max_retries=0), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        before = len(factory.calls)
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        # Initial SET_BAUDRATE response + all three uncertainty candidates fail.
        for _ in range(4):
            transport.queue_read_failure(FakeReadFailure.TIMEOUT)
        changed = replace(
            worker._desired_config,
            serial=replace(worker._desired_config.serial, baudrate=115200),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))
        _wait_factory_calls(factory, before + 1)
        _wait_state(worker, TurretConnectionState.READY)
        assert len(factory.calls) > before
    finally:
        _shutdown(worker)


def test_ready_is_not_published_until_set_config_is_confirmed() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.OK))
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.OK))
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_ARGUMENT))
    factory = _RecordingFactory(endpoint)
    entered_backoff = Event()
    release_backoff = Event()

    def gated_wait(stop_token, timeout_s: float) -> bool:
        del timeout_s
        entered_backoff.set()
        release_backoff.wait(0.5)
        return stop_token.is_stop_requested()

    worker = TurretWorker(
        _config(), transport_factory=factory, cooperative_wait=gated_wait
    )
    worker.start()
    try:
        assert entered_backoff.wait(0.5)
        assert worker.ready is False
        assert worker.current_state.connection_state is TurretConnectionState.CONNECTING
        assert worker.current_state.max_speed_x_deg_s is None
        release_backoff.set()
        _wait_state(worker, TurretConnectionState.READY)
    finally:
        _shutdown(worker)


def test_port_change_does_not_reuse_old_port_last_known_baud() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(baudrate=38400), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        worker._last_known_baud = 19200
        before = len(factory.calls)
        changed = replace(
            worker._desired_config,
            serial=replace(worker._desired_config.serial, port="/dev/ttyUSB9"),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))
        _wait_factory_calls(factory, before + 1)
        _wait_state(worker, TurretConnectionState.READY)
        new_calls = factory.calls[before:]
        assert new_calls[0][0] == "/dev/ttyUSB9"
        assert new_calls[0][1] == 38400
        assert all(baud != 19200 for _, baud, _ in new_calls)
    finally:
        _shutdown(worker)


def test_reconnect_does_not_apply_runtime_restart_only_axis_mechanics() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        original_steps = worker._hal.degrees_to_steps("x", 1.0)
        new_axis = replace(worker._desired_config.axes.x, invert=True)
        changed = replace(
            worker._desired_config,
            axes=replace(worker._desired_config.axes, x=new_axis),
            serial=replace(worker._desired_config.serial, response_timeout_ms=25),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))
        deadline = monotonic() + 1.0
        while not worker.ready or worker._session._response_timeout_s != 0.025:
            assert monotonic() < deadline
            Event().wait(0.002)
        assert worker._hal.degrees_to_steps("x", 1.0) == original_steps
    finally:
        _shutdown(worker)


def test_high_rate_tracking_path_does_not_emit_info_per_setpoint(caplog) -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        worker.set_control_mode(TurretControlMode.TRACKING)
        deadline = monotonic() + 0.5
        while worker.current_state.control_mode is not TurretControlMode.TRACKING:
            assert monotonic() < deadline
            Event().wait(0.002)
        caplog.clear()
        with caplog.at_level("INFO", logger="navmin.turret.worker"):
            for revision in range(1, 6):
                worker.submit_tracking_error(
                    TrackingError(
                        TargetRef(CameraRole.OVERVIEW, 1, 1),
                        0.1 * revision,
                        0.0,
                        revision,
                    )
                )
            deadline = monotonic() + 0.5
            while (
                worker._controller.last_processed_revision
                != worker._motion_inputs.snapshot().revision
            ):
                assert monotonic() < deadline
                Event().wait(0.002)
        assert "SET_VELOCITY" not in caplog.text
        assert "TrackingError" not in caplog.text
    finally:
        _shutdown(worker)


def test_recovery_set_config_uses_freshest_update_before_ready() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    factory.transport_type = _CallbackFakeTransport
    worker = TurretWorker(_config(), transport_factory=factory)
    updated = replace(
        worker._desired_config,
        stm32=replace(worker._desired_config.stm32, max_speed_x_deg_s=73.0),
    )

    # First/only candidate transport exists only after start, so arm callback
    # by wrapping the factory and installing it on construction.
    original_factory = worker._transport_factory

    def callback_factory(port: str, baudrate: int, emulate: bool):
        transport = original_factory(port, baudrate, emulate)
        assert isinstance(transport, _CallbackFakeTransport)
        transport.after_command[CommandCode.SET_CONFIG] = lambda: worker.submit_config_update(
            ConfigUpdate(1, updated)
        )
        return transport

    worker._transport_factory = callback_factory
    worker.start()
    try:
        state = _wait_state(worker, TurretConnectionState.READY)
        set_configs = [
            request
            for request in endpoint.request_history
            if request.command is CommandCode.SET_CONFIG
        ]
        assert len(set_configs) == 2
        assert state.max_speed_x_deg_s == 73.0
    finally:
        _shutdown(worker)


def test_unrecoverable_local_recovery_failure_publishes_error() -> None:
    def broken_factory(port: str, baudrate: int, emulate: bool):
        del port, baudrate, emulate
        raise RuntimeError("broken local factory invariant")

    worker = TurretWorker(_config(), transport_factory=broken_factory)
    worker.start()
    state = _wait_state(worker, TurretConnectionState.ERROR)
    worker.join(0.5)
    assert not worker.is_alive()
    assert state.motor_state is MotorState.UNKNOWN
    assert state.max_speed_x_deg_s is None


def test_recovery_defers_pending_config_until_after_motor_off() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    entered_backoff = Event()
    release_backoff = Event()

    def gated_wait(stop_token, timeout_s: float) -> bool:
        del timeout_s
        entered_backoff.set()
        release_backoff.wait(0.5)
        return stop_token.is_stop_requested()

    worker = TurretWorker(
        _config(), transport_factory=factory, cooperative_wait=gated_wait
    )
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)

        factory.fail_open_count = 1
        transport = worker._transport
        assert isinstance(transport, FakeTransport)
        transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        assert entered_backoff.wait(0.5)

        history_start = len(endpoint.request_history)
        changed = replace(
            worker._desired_config,
            stm32=replace(worker._desired_config.stm32, max_speed_x_deg_s=75.0),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))
        release_backoff.set()

        state = _wait_state(worker, TurretConnectionState.READY)
        commands = [
            request.command for request in endpoint.request_history[history_start:]
        ]
        assert commands == [
            CommandCode.EMERGENCY_STOP,
            CommandCode.MOTOR_OFF,
            CommandCode.SET_CONFIG,
        ]
        assert commands.count(CommandCode.SET_CONFIG) == 1
        assert state.max_speed_x_deg_s == 75.0
    finally:
        _shutdown(worker)


def test_normal_ingress_during_active_recovery_is_not_replayed_after_ready() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _BlockingRecoveryFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        history_start = len(endpoint.request_history)

        factory.block_next_recovery_set_config = True
        changed = replace(
            worker._desired_config,
            serial=replace(
                worker._desired_config.serial,
                response_timeout_ms=500,
            ),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))

        assert factory.blocked_transport_ready.wait(0.5)
        blocked = factory.blocked_transport
        assert blocked is not None
        assert blocked.write_seen.wait(0.5)
        assert worker.current_state.connection_state is TurretConnectionState.CONNECTING

        assert not worker.set_control_mode(TurretControlMode.TRACKING)
        assert not worker.motor_on()
        worker.submit_move_relative(MoveRelativeCommand(10.0, 0.0))

        blocked.release_read.set()
        _wait_state(worker, TurretConnectionState.READY)
        Event().wait(0.03)

        commands = [
            request.command for request in endpoint.request_history[history_start:]
        ]
        assert commands == [
            CommandCode.EMERGENCY_STOP,
            CommandCode.MOTOR_OFF,
            CommandCode.SET_CONFIG,
        ]
        assert CommandCode.MOTOR_ON not in commands
        assert CommandCode.MOVE_RELATIVE not in commands
        assert worker.current_state.motor_state is MotorState.OFF
        assert worker.current_state.control_mode is TurretControlMode.RELATIVE
    finally:
        _shutdown(worker)


def test_emergency_during_active_recovery_is_serviced_before_ready() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _BlockingRecoveryFactory(endpoint)
    worker = TurretWorker(_config(), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        history_start = len(endpoint.request_history)

        factory.block_next_recovery_set_config = True
        changed = replace(
            worker._desired_config,
            serial=replace(worker._desired_config.serial, response_timeout_ms=500),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))

        assert factory.blocked_transport_ready.wait(0.5)
        blocked = factory.blocked_transport
        assert blocked is not None
        assert blocked.write_seen.wait(0.5)
        assert worker.current_state.connection_state is TurretConnectionState.CONNECTING

        worker.request_emergency()
        blocked.release_read.set()
        _wait_state(worker, TurretConnectionState.READY)

        commands = [
            request.command for request in endpoint.request_history[history_start:]
        ]
        assert commands == [
            CommandCode.EMERGENCY_STOP,
            CommandCode.MOTOR_OFF,
            CommandCode.SET_CONFIG,
            CommandCode.EMERGENCY_STOP,
            CommandCode.MOTOR_OFF,
            CommandCode.SET_CONFIG,
        ]
        assert worker.current_state.motor_state is MotorState.OFF
    finally:
        _shutdown(worker)


def test_emergency_preempted_set_config_is_confirmed_before_later_motion() -> None:
    endpoint = FakeStm32Endpoint()
    factory = _RecordingFactory(endpoint)
    worker = TurretWorker(_config(inter_request_delay_ms=10), transport_factory=factory)
    worker.start()
    try:
        _wait_state(worker, TurretConnectionState.READY)
        session = worker._session
        assert session is not None
        entered_wait = Event()
        release_wait = Event()
        wait_used = False

        def wait_once(delay_s: float) -> None:
            nonlocal wait_used
            del delay_s
            if wait_used:
                return
            wait_used = True
            entered_wait.set()
            assert release_wait.wait(0.5)

        session._wait_hook = wait_once
        history_start = len(endpoint.request_history)
        changed = replace(
            worker._desired_config,
            stm32=replace(worker._desired_config.stm32, max_speed_x_deg_s=75.0),
        )
        worker.submit_config_update(ConfigUpdate(1, changed))
        assert entered_wait.wait(0.5)

        worker.request_emergency()
        release_wait.set()

        deadline = monotonic() + 1.0
        while worker.current_state.max_speed_x_deg_s != 75.0:
            assert monotonic() < deadline
            Event().wait(0.002)

        worker.submit_move_relative(MoveRelativeCommand(1.0, 0.0))
        deadline = monotonic() + 1.0
        while endpoint.request_history[-1].command is not CommandCode.MOVE_RELATIVE:
            assert monotonic() < deadline
            Event().wait(0.002)

        commands = [
            request.command for request in endpoint.request_history[history_start:]
        ]
        assert commands == [
            CommandCode.EMERGENCY_STOP,
            CommandCode.SET_CONFIG,
            CommandCode.MOVE_RELATIVE,
        ]
        assert worker.current_state.max_speed_x_deg_s == 75.0
    finally:
        _shutdown(worker)
