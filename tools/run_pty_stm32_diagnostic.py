"""Exercise production SerialTransport and TurretWorker through a Linux PTY."""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from navmin.config.models import (
    AxesConfig,
    AxisMechanicsConfig,
    PidControllerConfig,
    SerialConfig,
    Stm32Config,
    TurretConfig,
)
from navmin.contracts import (
    MotorState,
    MoveRelativeCommand,
    TurretConnectionState,
    TurretControlMode,
)
from navmin.diagnostics.pty_stm32 import (
    PtyDiagnosticError,
    PtyFault,
    PtyFaultKind,
    PtyRequestRecord,
    PtyStm32Emulator,
    check_pty_preflight,
)
from navmin.turret.protocol import CommandCode, ResultCode
from navmin.turret.session import TurretSession
from navmin.turret.transport import SerialTransport, TransportError
from navmin.turret.worker import TurretWorker, WorkerShutdownError


class DiagnosticFailure(RuntimeError):
    """A required diagnostic observation was not obtained."""


@dataclass(frozen=True)
class GateOutcome:
    gate: str
    status: str
    evidence: tuple[str, ...]


class _RecordingSerialFactory:
    """Record construction facts without wrapping production transport I/O."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []
        self.transports: list[SerialTransport] = []

    def __call__(self, port: str, baudrate: int, emulate: bool) -> SerialTransport:
        self.calls.append((port, baudrate, emulate))
        if emulate:
            raise DiagnosticFailure("worker unexpectedly requested FakeTransport mode")
        transport = SerialTransport(port, baudrate)
        self.transports.append(transport)
        return transport


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticFailure(message)


def _wait_until(predicate, *, timeout_s: float, failure: str) -> None:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.005)
    raise DiagnosticFailure(failure)


def _wait_worker_state(
    worker: TurretWorker,
    predicate,
    *,
    timeout_s: float,
    after_revision: int | None = None,
    failure: str,
):
    snapshot = worker.state_updates.snapshot()
    minimum_revision = snapshot.revision if after_revision is None else after_revision
    deadline = monotonic() + timeout_s
    while True:
        state = snapshot.value
        if snapshot.revision > minimum_revision and state is not None and predicate(state):
            return state
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise DiagnosticFailure(f"{failure}; last state={state}")
        changed = worker.state_updates.wait_for_revision(snapshot.revision, remaining)
        if changed is None:
            raise DiagnosticFailure(f"{failure}; last state={state}")
        snapshot = changed


def _wait_ready(worker: TurretWorker, *, timeout_s: float):
    state = worker.current_state
    if state.connection_state is TurretConnectionState.READY:
        return state
    return _wait_worker_state(
        worker,
        lambda current: current.connection_state is TurretConnectionState.READY,
        timeout_s=timeout_s,
        after_revision=0,
        failure="worker did not reach READY",
    )


def _wait_request_count(
    emulator: PtyStm32Emulator,
    count: int,
    *,
    timeout_s: float,
) -> None:
    _wait_until(
        lambda: emulator.stats().physical_request_count >= count,
        timeout_s=timeout_s,
        failure=f"PTY bridge did not observe {count} physical requests",
    )


def _commands(records: tuple[PtyRequestRecord, ...]) -> tuple[CommandCode, ...]:
    return tuple(
        record.request.command for record in records if record.request is not None
    )


def _command_summary(commands: tuple[CommandCode, ...]) -> str:
    return ", ".join(command.name for command in commands)


def _diagnostic_config(port: str) -> TurretConfig:
    axis = AxisMechanicsConfig(False, 2000, 16, 45.0)
    return TurretConfig(
        serial=SerialConfig(port, 9600, 150, 1, 2),
        axes=AxesConfig(axis, axis),
        controller=PidControllerConfig(1.0, 0.1, 0.0, 1.0, 0.1, 0.0),
        stm32=Stm32Config(50.0, 60.0, 100.0, 120.0, 200),
        emulate_stm32=False,
    )


def _gate_b(
    emulator: PtyStm32Emulator,
    session: TurretSession,
) -> GateOutcome:
    before = emulator.stats()
    result = session.transact(CommandCode.PING)
    response = result.response
    _require(response is not None, "PING returned no response")
    _require(response.result is ResultCode.OK, "PING did not return OK")
    _require(response.command is CommandCode.PING, "PING response command mismatch")
    _require(response.request_id == 0, "initial PING REQUEST_ID was not zero")
    after = emulator.stats()
    _require(
        after.physical_request_count == before.physical_request_count + 1,
        "basic PING did not cross exactly one physical PTY request",
    )
    return GateOutcome(
        "B",
        "GREEN",
        (
            "production SerialTransport + real pyserial opened the stable PTY slave",
            "PING response: OK",
            f"REQUEST_ID: {response.request_id}",
            f"physical requests/responses: {after.physical_request_count}/{after.response_count}",
            "FakeTransport: not used",
        ),
    )


def _run_single_stream_fault(
    emulator: PtyStm32Emulator,
    session: TurretSession,
    fault: PtyFault,
) -> tuple[int, int]:
    before_stats = emulator.stats()
    before_endpoint = emulator.endpoint_snapshot()
    emulator.queue_fault(fault)
    result = session.transact(CommandCode.PING)
    _require(
        result.response is not None and result.response.result is ResultCode.OK,
        f"{fault.kind.value} did not produce successful transaction",
    )
    after_stats = emulator.stats()
    after_endpoint = emulator.endpoint_snapshot()
    physical_delta = after_stats.physical_request_count - before_stats.physical_request_count
    executed_delta = (
        after_endpoint.executed_request_count - before_endpoint.executed_request_count
    )
    return physical_delta, executed_delta


def _gate_c(
    emulator: PtyStm32Emulator,
    session: TurretSession,
    *,
    response_timeout_s: float,
) -> GateOutcome:
    observations: list[str] = []
    for label, fault in (
        (
            "fragmented response",
            PtyFault(PtyFaultKind.FRAGMENTED_RESPONSE, delay_s=0.003),
        ),
        ("leading garbage", PtyFault(PtyFaultKind.LEADING_GARBAGE)),
        (
            "overlapping start prefix",
            PtyFault(PtyFaultKind.OVERLAPPING_START_PREFIX),
        ),
    ):
        physical_delta, executed_delta = _run_single_stream_fault(
            emulator, session, fault
        )
        _require(physical_delta == 1, f"{label} unexpectedly caused a retry")
        _require(executed_delta == 1, f"{label} endpoint execution mismatch")
        observations.append(
            f"{label}: PASS (physical requests={physical_delta}, executions={executed_delta})"
        )

    delay_s = min(0.03, response_timeout_s / 3.0)
    started = monotonic()
    physical_delta, executed_delta = _run_single_stream_fault(
        emulator,
        session,
        PtyFault(PtyFaultKind.DELAYED_RESPONSE, delay_s=delay_s),
    )
    elapsed = monotonic() - started
    _require(physical_delta == 1, "bounded response delay unexpectedly caused retry")
    _require(executed_delta == 1, "bounded delay endpoint execution mismatch")
    _require(elapsed >= delay_s, "configured response delay was not observed")
    observations.append(
        f"delayed response: PASS ({elapsed * 1000.0:.1f} ms < "
        f"{response_timeout_s * 1000.0:.0f} ms timeout, no retry)"
    )
    stats = emulator.stats()
    observations.append(
        f"cumulative physical requests/responses: "
        f"{stats.physical_request_count}/{stats.response_count}"
    )
    return GateOutcome("C", "GREEN", tuple(observations))


def _retry_observation(
    emulator: PtyStm32Emulator,
    session: TurretSession,
    fault_kind: PtyFaultKind,
) -> str:
    before_stats = emulator.stats()
    before_endpoint = emulator.endpoint_snapshot()
    emulator.queue_fault(PtyFault(fault_kind))
    result = session.transact(CommandCode.PING)
    _require(
        result.response is not None and result.response.result is ResultCode.OK,
        f"{fault_kind.value} retry did not complete successfully",
    )
    after_endpoint = emulator.endpoint_snapshot()
    records = emulator.requests_after(before_stats.physical_request_count)
    _require(len(records) == 2, f"{fault_kind.value} did not produce two requests")
    _require(
        records[0].raw_frame == records[1].raw_frame,
        f"{fault_kind.value} retry raw bytes differ",
    )
    _require(
        records[0].request is not None and records[1].request is not None,
        f"{fault_kind.value} retry request could not be decoded",
    )
    first_request = records[0].request
    second_request = records[1].request
    _require(
        first_request.request_id == second_request.request_id,
        f"{fault_kind.value} retry REQUEST_ID differs",
    )
    executed_delta = (
        after_endpoint.executed_request_count - before_endpoint.executed_request_count
    )
    _require(executed_delta == 1, f"{fault_kind.value} executed more than once")
    return (
        f"{fault_kind.value}: physical requests=2, raw equality=yes, "
        f"REQUEST_ID={first_request.request_id}, endpoint executions={executed_delta}"
    )


def _gate_d(
    emulator: PtyStm32Emulator,
    session: TurretSession,
) -> GateOutcome:
    dropped = _retry_observation(
        emulator, session, PtyFaultKind.DROP_RESPONSE
    )
    bad_crc = _retry_observation(
        emulator, session, PtyFaultKind.BAD_CRC_RESPONSE
    )
    return GateOutcome("D", "GREEN", (dropped, bad_crc))


def _gate_e(
    emulator: PtyStm32Emulator,
    session: TurretSession,
    transport: SerialTransport,
) -> GateOutcome:
    before = transport.baudrate
    response = session.set_baudrate(115200)
    _require(response.result is ResultCode.OK, "SET_BAUDRATE(115200) failed")
    backend = transport._serial
    _require(transport.baudrate == 115200, "SerialTransport baud did not change")
    _require(
        backend is not None and backend.baudrate == 115200,
        "underlying pyserial baud did not change",
    )
    _require(
        emulator.endpoint_snapshot().baudrate == 115200,
        "endpoint baud did not change to 115200",
    )
    ping_new = session.transact(CommandCode.PING).response
    _require(
        ping_new is not None and ping_new.result is ResultCode.OK,
        "post-115200 PING failed",
    )

    rollback = session.set_baudrate(9600)
    _require(rollback.result is ResultCode.OK, "SET_BAUDRATE(9600) rollback failed")
    _require(transport.baudrate == 9600, "SerialTransport rollback baud mismatch")
    _require(
        backend.baudrate == 9600,
        "underlying pyserial rollback baud mismatch",
    )
    _require(
        emulator.endpoint_snapshot().baudrate == 9600,
        "endpoint rollback baud mismatch",
    )
    ping_old = session.transact(CommandCode.PING).response
    _require(
        ping_old is not None and ping_old.result is ResultCode.OK,
        "post-rollback PING failed",
    )
    return GateOutcome(
        "E",
        "GREEN",
        (
            f"before baud: {before}",
            "SET_BAUDRATE 9600 -> 115200: OK",
            "SerialTransport / pyserial / endpoint after: 115200 / 115200 / 115200",
            f"post-transition PING REQUEST_ID: {ping_new.request_id}",
            "SET_BAUDRATE 115200 -> 9600: OK",
            "rollback PING: OK",
            "PTY validates logical/configuration transition, not physical UART bit timing",
        ),
    )


def _gate_f(
    emulator: PtyStm32Emulator,
    worker: TurretWorker,
    factory: _RecordingSerialFactory,
    *,
    timeout_s: float,
) -> GateOutcome:
    before = emulator.stats().physical_request_count
    worker.start()
    ready = _wait_ready(worker, timeout_s=timeout_s)
    startup_records = emulator.requests_after(before)
    startup_commands = _commands(startup_records)
    expected_startup = (
        CommandCode.EMERGENCY_STOP,
        CommandCode.MOTOR_OFF,
        CommandCode.SET_CONFIG,
    )
    _require(
        startup_commands[:3] == expected_startup,
        f"unexpected worker recovery commands: {_command_summary(startup_commands)}",
    )
    _require(ready.motor_state is MotorState.OFF, "recovery did not leave motors OFF")
    _require(
        ready.max_speed_x_deg_s is not None,
        "recovery did not publish applied STM32 limits",
    )
    _require(len(factory.transports) == 1, "startup did not create one transport")
    _require(factory.transports[0].is_open, "startup SerialTransport is not open")

    ordinary_before = emulator.stats().physical_request_count
    revision = worker.state_updates.snapshot().revision
    _require(worker.motor_on(), "MOTOR_ON ingress was rejected")
    _wait_worker_state(
        worker,
        lambda state: state.motor_state is MotorState.ON,
        timeout_s=timeout_s,
        after_revision=revision,
        failure="MOTOR_ON was not confirmed",
    )

    revision = worker.state_updates.snapshot().revision
    _require(
        worker.set_control_mode(TurretControlMode.TRACKING),
        "TRACKING mode ingress was rejected",
    )
    _wait_worker_state(
        worker,
        lambda state: state.control_mode is TurretControlMode.TRACKING,
        timeout_s=timeout_s,
        after_revision=revision,
        failure="TRACKING mode was not confirmed",
    )

    request_count = emulator.stats().physical_request_count
    _require(worker.stop_motion(), "STOP_MOTION ingress was rejected")
    _wait_request_count(emulator, request_count + 1, timeout_s=timeout_s)

    revision = worker.state_updates.snapshot().revision
    _require(
        worker.set_control_mode(TurretControlMode.RELATIVE),
        "RELATIVE mode ingress was rejected",
    )
    _wait_worker_state(
        worker,
        lambda state: state.control_mode is TurretControlMode.RELATIVE,
        timeout_s=timeout_s,
        after_revision=revision,
        failure="RELATIVE mode was not confirmed",
    )

    request_count = emulator.stats().physical_request_count
    worker.submit_move_relative(MoveRelativeCommand(1.0, -0.5))
    _wait_request_count(emulator, request_count + 1, timeout_s=timeout_s)

    revision = worker.state_updates.snapshot().revision
    _require(worker.motor_off(), "MOTOR_OFF ingress was rejected")
    final_state = _wait_worker_state(
        worker,
        lambda state: state.motor_state is MotorState.OFF,
        timeout_s=timeout_s,
        after_revision=revision,
        failure="MOTOR_OFF was not confirmed",
    )
    ordinary_commands = _commands(emulator.requests_after(ordinary_before))
    for required in (
        CommandCode.MOTOR_ON,
        CommandCode.SET_VELOCITY,
        CommandCode.MOVE_RELATIVE,
        CommandCode.MOTOR_OFF,
    ):
        _require(required in ordinary_commands, f"{required.name} was not observed")
    return GateOutcome(
        "F",
        "GREEN",
        (
            f"worker connection: {final_state.connection_state.name}",
            f"motor after recovery: {ready.motor_state.name}",
            f"real SerialTransport instances: {len(factory.transports)}",
            f"startup commands: {_command_summary(startup_commands[:3])}",
            f"ordinary commands: {_command_summary(ordinary_commands)}",
            "MOTOR_ON, TRACKING, STOP_MOTION, RELATIVE, MOVE_RELATIVE, MOTOR_OFF: confirmed",
        ),
    )


def _gate_g(
    emulator: PtyStm32Emulator,
    worker: TurretWorker,
    *,
    timeout_s: float,
) -> GateOutcome:
    revision = worker.state_updates.snapshot().revision
    _require(worker.motor_on(), "pre-Emergency MOTOR_ON ingress was rejected")
    _wait_worker_state(
        worker,
        lambda state: state.motor_state is MotorState.ON,
        timeout_s=timeout_s,
        after_revision=revision,
        failure="pre-Emergency MOTOR_ON was not confirmed",
    )
    before = emulator.stats().physical_request_count
    revision = worker.state_updates.snapshot().revision
    worker.request_emergency()
    _wait_request_count(emulator, before + 1, timeout_s=timeout_s)
    state = _wait_worker_state(
        worker,
        lambda current: (
            current.connection_state is TurretConnectionState.READY
            and current.motor_state is MotorState.ON
        ),
        timeout_s=timeout_s,
        after_revision=revision,
        failure="worker did not publish READY after Emergency",
    )
    commands = _commands(emulator.requests_after(before))
    _require(
        CommandCode.EMERGENCY_STOP in commands,
        "EMERGENCY_STOP did not cross the PTY boundary",
    )
    return GateOutcome(
        "G",
        "GREEN",
        (
            f"commands observed: {_command_summary(commands)}",
            f"worker connection after Emergency: {state.connection_state.name}",
            f"motor after Emergency: {state.motor_state.name} (Emergency != MOTOR_OFF)",
            "public request_emergency() remained signal-only; worker thread retained UART ownership",
        ),
    )


def _wait_disconnect_and_ready(
    worker: TurretWorker,
    *,
    after_revision: int,
    timeout_s: float,
):
    snapshot = worker.state_updates.snapshot()
    seen_non_ready = False
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        state = snapshot.value
        if snapshot.revision > after_revision and state is not None:
            if state.connection_state is not TurretConnectionState.READY:
                seen_non_ready = True
            elif seen_non_ready:
                return state
        remaining = deadline - monotonic()
        changed = worker.state_updates.wait_for_revision(
            snapshot.revision, max(0.0, remaining)
        )
        if changed is None:
            break
        snapshot = changed
    raise DiagnosticFailure(
        f"worker did not expose non-READY -> READY recovery; last={snapshot.value}"
    )


def _gate_h(
    emulator: PtyStm32Emulator,
    worker: TurretWorker,
    factory: _RecordingSerialFactory,
    *,
    timeout_s: float,
) -> GateOutcome:
    stable_path = emulator.stable_port_path
    old_slave = emulator.slave_path
    before = emulator.stats()
    before_factory_count = len(factory.transports)
    old_transport = factory.transports[-1]
    revision = worker.state_updates.snapshot().revision

    emulator.queue_fault(PtyFault(PtyFaultKind.HARD_DISCONNECT))
    emulator.queue_fault(PtyFault(PtyFaultKind.DELAYED_RESPONSE, delay_s=0.05))
    _require(worker.stop_motion(), "disconnect-trigger STOP_MOTION ingress rejected")
    _wait_until(
        lambda: emulator.stats().replacement_count > before.replacement_count,
        timeout_s=timeout_s,
        failure="hard disconnect did not replace the PTY pair",
    )
    new_slave = emulator.slave_path
    recovered = _wait_disconnect_and_ready(
        worker,
        after_revision=revision,
        timeout_s=timeout_s,
    )
    _require(old_slave != new_slave, "replacement reused the old PTY slave path")
    _require(
        emulator.stable_port_path == stable_path,
        "stable diagnostic port path changed across replacement",
    )
    _require(
        len(factory.transports) > before_factory_count,
        "worker did not create a new SerialTransport",
    )
    new_transport = factory.transports[-1]
    _require(new_transport is not old_transport, "worker reused old SerialTransport")
    _require(not old_transport.is_open, "old SerialTransport remained open")
    _require(new_transport.is_open, "new SerialTransport is not open")
    _require(
        all(call[0] == stable_path for call in factory.calls),
        "worker changed configured port string during reconnect",
    )
    _require(
        recovered.motor_state is MotorState.OFF,
        "hard reconnect did not return with motors OFF",
    )
    recovery_commands = _commands(
        emulator.requests_after(before.physical_request_count)
    )
    _require(
        CommandCode.EMERGENCY_STOP in recovery_commands,
        "recovery Emergency was not observed",
    )
    _require(
        CommandCode.SET_CONFIG in recovery_commands,
        "recovery SET_CONFIG was not observed",
    )

    revision = worker.state_updates.snapshot().revision
    _require(worker.motor_on(), "post-recovery MOTOR_ON ingress rejected")
    post_recovery = _wait_worker_state(
        worker,
        lambda state: state.motor_state is MotorState.ON,
        timeout_s=timeout_s,
        after_revision=revision,
        failure="post-recovery MOTOR_ON was not confirmed",
    )
    return GateOutcome(
        "H",
        "GREEN",
        (
            f"old PTY slave: {old_slave}",
            f"new PTY slave: {new_slave}",
            f"stable path unchanged: {stable_path}",
            "worker observed loss: READY -> CONNECTING -> READY",
            f"SerialTransport instances: {before_factory_count} -> {len(factory.transports)}",
            f"recovery commands: {_command_summary(recovery_commands)}",
            f"motor after recovery / post-recovery action: OFF / {post_recovery.motor_state.name}",
        ),
    )


def _print_outcomes(outcomes: list[GateOutcome], *, result: str) -> None:
    print()
    for outcome in outcomes:
        print(f"GATE {outcome.gate}: {outcome.status}")
        for evidence in outcome.evidence:
            print(f"  {evidence}")
    print()
    print(f"RESULT: {result}")


def run_diagnostic(*, timeout_seconds: float) -> bool:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    print("=== PTY STM32 / PRODUCTION SERIAL DIAGNOSTIC ===")
    print("NavMin: TurretWorker / Session -> SerialTransport -> pyserial -> PTY slave")
    print("STM32: PTY master byte bridge -> FakeStm32Endpoint")
    print("FakeTransport: not used")

    outcomes: list[GateOutcome] = []
    emulator: PtyStm32Emulator | None = None
    direct_transport: SerialTransport | None = None
    worker: TurretWorker | None = None
    stable_path: str | None = None
    temporary_directory: str | None = None
    baseline_non_daemon = {
        thread.ident for thread in threading.enumerate() if not thread.daemon
    }
    gates = tuple("ABCDEFGHI")
    active_gate = "A"
    failure: BaseException | None = None

    try:
        preflight = check_pty_preflight()
        _require(preflight.linux, "platform is not Linux")
        _require(preflight.openpty_available, "os.openpty() is unavailable")
        _require(preflight.pyserial_available, "pyserial is unavailable")
        emulator = PtyStm32Emulator()
        emulator.start()
        stable_path = emulator.stable_port_path
        temporary_directory = emulator.temporary_directory
        probe = SerialTransport(stable_path, 9600)
        probe.open()
        probe.close()
        outcomes.append(
            GateOutcome(
                "A",
                "GREEN",
                (
                    "Linux/openpty: OK",
                    "pyserial: OK",
                    f"PTY created: OK ({emulator.slave_path})",
                    f"stable diagnostic path: {stable_path}",
                    "SerialTransport.open(): OK",
                ),
            )
        )

        direct_transport = SerialTransport(stable_path, 9600)
        direct_transport.open()
        session = TurretSession(
            direct_transport,
            response_timeout_s=0.15,
            max_retries=1,
            inter_request_delay_s=0.002,
        )
        active_gate = "B"
        outcomes.append(_gate_b(emulator, session))
        active_gate = "C"
        outcomes.append(
            _gate_c(emulator, session, response_timeout_s=0.15)
        )
        active_gate = "D"
        outcomes.append(_gate_d(emulator, session))
        active_gate = "E"
        outcomes.append(_gate_e(emulator, session, direct_transport))
        direct_transport.close()
        direct_transport = None

        factory = _RecordingSerialFactory()
        worker = TurretWorker(
            _diagnostic_config(stable_path),
            transport_factory=factory,
        )
        active_gate = "F"
        outcomes.append(
            _gate_f(emulator, worker, factory, timeout_s=timeout_seconds)
        )
        active_gate = "G"
        outcomes.append(_gate_g(emulator, worker, timeout_s=timeout_seconds))
        active_gate = "H"
        outcomes.append(
            _gate_h(emulator, worker, factory, timeout_s=timeout_seconds)
        )
    except (DiagnosticFailure, PtyDiagnosticError, TransportError, RuntimeError) as exc:
        failure = exc
        completed = {outcome.gate for outcome in outcomes}
        if active_gate not in completed:
            outcomes.append(
                GateOutcome(
                    active_gate,
                    "RED",
                    (f"{type(exc).__name__}: {exc}",),
                )
            )
        for gate in gates[gates.index(active_gate) + 1 : -1]:
            if gate not in completed:
                outcomes.append(
                    GateOutcome(gate, "NOT RUN", (f"blocked by GATE {active_gate}",))
                )
    finally:
        cleanup_errors: list[str] = []
        if worker is not None and worker.is_alive():
            try:
                worker.shutdown(2.0)
            except WorkerShutdownError as exc:
                cleanup_errors.append(f"TurretWorker: {exc}")
        if direct_transport is not None:
            try:
                direct_transport.close()
            except TransportError as exc:
                cleanup_errors.append(f"direct SerialTransport: {exc}")
        emulator_thread_stopped = True
        fds_closed = True
        if emulator is not None:
            try:
                emulator.stop(2.0)
            except (OSError, PtyDiagnosticError) as exc:
                cleanup_errors.append(f"PTY emulator: {exc}")
            emulator_thread_stopped = not emulator.is_alive()
            fds_closed = emulator.open_fd_count == 0
            if emulator.service_error is not None:
                cleanup_errors.append(
                    f"PTY service error: {type(emulator.service_error).__name__}: "
                    f"{emulator.service_error}"
                )
        port_removed = stable_path is None or not Path(stable_path).exists()
        directory_removed = (
            temporary_directory is None or not Path(temporary_directory).exists()
        )
        leaked_threads = tuple(
            thread.name
            for thread in threading.enumerate()
            if not thread.daemon and thread.ident not in baseline_non_daemon
        )
        if not emulator_thread_stopped:
            cleanup_errors.append("PTY service thread remains alive")
        if not fds_closed:
            cleanup_errors.append("PTY owner still reports open file descriptors")
        if not port_removed or not directory_removed:
            cleanup_errors.append("temporary PTY path/directory remains")
        if leaked_threads:
            cleanup_errors.append(f"new non-daemon threads remain: {leaked_threads}")
        if cleanup_errors:
            outcomes.append(GateOutcome("I", "RED", tuple(cleanup_errors)))
            failure = failure or DiagnosticFailure("cleanup failed")
        else:
            outcomes.append(
                GateOutcome(
                    "I",
                    "GREEN",
                    (
                        "TurretWorker stopped",
                        "PTY service thread stopped",
                        "all owned PTY file descriptors closed",
                        "temporary symlink and directory removed",
                        "no new non-daemon threads",
                    ),
                )
            )

    passed = failure is None and all(outcome.status == "GREEN" for outcome in outcomes)
    _print_outcomes(outcomes, result="PASS" if passed else "FAIL")
    return passed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate production NavMin serial transport through a Linux PTY STM32 emulator."
        )
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="bounded wait for worker gates (default: 5)",
    )
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return 0 if run_diagnostic(timeout_seconds=args.timeout_seconds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
