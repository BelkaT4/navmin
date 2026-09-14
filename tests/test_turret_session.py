from __future__ import annotations

from collections.abc import Callable
from threading import Barrier, Event, Thread

import pytest

from navmin.turret.protocol import (
    CommandCode,
    PayloadFormatError,
    ProtocolResponse,
    ResultCode,
    SetBaudratePayload,
    SetVelocityPayload,
    decode_request,
    encode_response,
)
from navmin.turret.session import (
    BaudRecoveryError,
    EmergencyRetryExhaustedError,
    SessionBlockedError,
    SessionRetryExhaustedError,
    SessionTransportError,
    TurretSession,
)
from navmin.turret.simulator import (
    FakeReadFailure,
    FakeResponseSpec,
    FakeStm32Endpoint,
    FakeTransport,
)


class _AfterWriteTransport(FakeTransport):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.after_write: Callable[[], None] | None = None
        self._hook_used = False

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        super().write_frame(frame, timeout_s)
        if self.after_write is not None and not self._hook_used:
            self._hook_used = True
            self.after_write()


class _CompletionBoundaryEvent:
    """Event wrapper that pauses the second clear at Emergency completion."""

    def __init__(self) -> None:
        self._event = Event()
        self._clear_count = 0
        self.completion_boundary = Barrier(2)
        self.release_boundary = Barrier(2)

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._clear_count += 1
        if self._clear_count == 2:
            self.completion_boundary.wait(timeout=0.5)
            self.release_boundary.wait(timeout=0.5)
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()


def _session(
    transport: FakeTransport | None = None,
    *,
    max_retries: int = 2,
    initial_request_id: int = 0,
    waits: list[float] | None = None,
    wait_hook: Callable[[float], None] | None = None,
) -> tuple[TurretSession, FakeTransport]:
    transport = transport or FakeTransport()
    transport.open()
    wait_log = waits if waits is not None else []
    session = TurretSession(
        transport,
        response_timeout_s=0.1,
        max_retries=max_retries,
        inter_request_delay_s=0.002,
        initial_request_id=initial_request_id,
        wait_hook=wait_hook or wait_log.append,
    )
    return session, transport


def test_request_id_starts_at_zero_and_advances_for_new_transactions() -> None:
    session, transport = _session()

    session.transact(CommandCode.PING)
    session.transact(CommandCode.MOTOR_OFF)

    assert [request.request_id for request in transport.endpoint.request_history] == [0, 1]
    assert session.next_request_id == 2


def test_request_id_wraps_from_65535_to_zero() -> None:
    session, transport = _session(initial_request_id=0xFFFF)

    session.transact(CommandCode.PING)
    session.transact(CommandCode.MOTOR_OFF)

    assert [request.request_id for request in transport.endpoint.request_history] == [
        0xFFFF,
        0,
    ]
    assert session.next_request_id == 1


def test_local_request_construction_failure_does_not_consume_request_id() -> None:
    session, transport = _session(max_retries=0)

    with pytest.raises(PayloadFormatError):
        session.transact(CommandCode.PING, SetBaudratePayload(9600))

    assert session.next_request_id == 0
    session.transact(CommandCode.PING)
    assert transport.endpoint.request_history[-1].request_id == 0
    assert session.next_request_id == 1


def test_timeout_retry_reuses_exact_same_raw_request_and_id() -> None:
    session, transport = _session(max_retries=1)
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)

    result = session.transact(CommandCode.PING)

    assert result.response == ProtocolResponse(0, CommandCode.PING, ResultCode.OK)
    assert transport.raw_write_history[0] == transport.raw_write_history[1]
    assert [decode_request(raw).request_id for raw in transport.raw_write_history] == [0, 0]
    assert session.next_request_id == 1


def test_bad_crc_response_triggers_exact_retry() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(corrupt_crc=True))
    session, transport = _session(FakeTransport(endpoint), max_retries=1)

    result = session.transact(CommandCode.PING)

    assert result.response == ProtocolResponse(0, CommandCode.PING, ResultCode.OK)
    assert transport.raw_write_history[0] == transport.raw_write_history[1]


@pytest.mark.parametrize(
    "stale",
    [
        ProtocolResponse(99, CommandCode.PING, ResultCode.OK),
        ProtocolResponse(0, CommandCode.MOTOR_OFF, ResultCode.OK),
    ],
)
def test_stale_or_mismatched_response_is_ignored_until_matching_response(
    stale: ProtocolResponse,
) -> None:
    session, transport = _session()
    transport.queue_read_frame(encode_response(stale))

    result = session.transact(CommandCode.PING)

    assert result.response == ProtocolResponse(0, CommandCode.PING, ResultCode.OK)
    assert len(transport.raw_write_history) == 1
    assert transport.event_history == ["write", "read", "read"]


def test_matching_command_error_is_returned_not_classified_as_transport_loss() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.MOTORS_OFF))
    session, _ = _session(FakeTransport(endpoint))

    result = session.transact(
        CommandCode.SET_VELOCITY, SetVelocityPayload(10, -10)
    )

    assert result.response is not None
    assert result.response.result is ResultCode.MOTORS_OFF
    assert not session.normal_traffic_blocked


def test_retry_exhaustion_blocks_normal_traffic_with_explicit_failure() -> None:
    session, transport = _session(max_retries=2)
    for _ in range(3):
        transport.queue_read_failure(FakeReadFailure.TIMEOUT)

    with pytest.raises(SessionRetryExhaustedError) as error:
        session.transact(CommandCode.PING)

    assert error.value.attempts == 3
    assert session.normal_traffic_blocked
    assert len(transport.raw_write_history) == 3
    assert len(set(transport.raw_write_history)) == 1


@pytest.mark.parametrize(
    "failure",
    [FakeReadFailure.IO_ERROR, FakeReadFailure.DISCONNECT],
)
def test_physical_io_failure_is_explicit_session_failure_and_blocks_normal(
    failure: FakeReadFailure,
) -> None:
    session, transport = _session(max_retries=2)
    transport.queue_read_failure(failure)

    with pytest.raises(SessionTransportError):
        session.transact(CommandCode.PING)

    assert session.normal_traffic_blocked
    assert len(transport.raw_write_history) == 1


def test_invalid_request_id_blocks_normal_until_successful_emergency_resync() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_REQUEST_ID))
    session, transport = _session(FakeTransport(endpoint))

    result = session.transact(CommandCode.PING)

    assert result.response is not None
    assert result.response.result is ResultCode.INVALID_REQUEST_ID
    assert session.normal_traffic_blocked
    with pytest.raises(SessionBlockedError):
        session.transact(CommandCode.PING)

    emergency = session.request_emergency()

    assert emergency == ProtocolResponse(1, CommandCode.EMERGENCY_STOP, ResultCode.OK)
    assert not session.normal_traffic_blocked
    assert session.next_request_id == 2
    assert [request.command for request in transport.endpoint.request_history] == [
        CommandCode.PING,
        CommandCode.EMERGENCY_STOP,
    ]


def test_successful_emergency_resync_sets_next_id_to_emergency_id_plus_one() -> None:
    session, transport = _session(initial_request_id=0xFFFF)

    emergency = session.request_emergency()
    session.transact(CommandCode.PING)

    assert emergency == ProtocolResponse(
        0xFFFF, CommandCode.EMERGENCY_STOP, ResultCode.OK
    )
    assert [request.request_id for request in transport.endpoint.request_history] == [
        0xFFFF,
        0,
    ]
    assert session.next_request_id == 1


def test_emergency_retry_reuses_exact_same_raw_request() -> None:
    session, transport = _session(max_retries=1)
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)

    response = session.request_emergency()

    assert response == ProtocolResponse(0, CommandCode.EMERGENCY_STOP, ResultCode.OK)
    assert transport.raw_write_history[0] == transport.raw_write_history[1]


def test_matching_non_ok_emergency_response_is_returned_and_session_stays_blocked() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INTERNAL_ERROR))
    session, _ = _session(FakeTransport(endpoint))

    response = session.request_emergency()

    assert response == ProtocolResponse(
        0, CommandCode.EMERGENCY_STOP, ResultCode.INTERNAL_ERROR
    )
    assert session.normal_traffic_blocked
    assert session.next_request_id == 1


def test_emergency_retry_exhaustion_leaves_session_blocked() -> None:
    session, transport = _session(max_retries=1)
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)

    with pytest.raises(EmergencyRetryExhaustedError):
        session.request_emergency()

    assert session.normal_traffic_blocked
    assert len(transport.raw_write_history) == 2


def test_emergency_while_idle_is_sent_immediately() -> None:
    session, transport = _session()

    response = session.request_emergency()

    assert response is not None
    assert decode_request(transport.raw_write_history[0]).command is CommandCode.EMERGENCY_STOP
    assert transport.event_history[:2] == ["write", "read"]


def test_emergency_requested_during_attempt_waits_for_current_response_then_sends() -> None:
    transport = _AfterWriteTransport()
    session, _ = _session(transport, max_retries=2)
    transport.after_write = session.request_emergency

    result = session.transact(CommandCode.PING)

    assert result.preempted_by_emergency
    assert result.response == ProtocolResponse(0, CommandCode.PING, ResultCode.OK)
    assert result.emergency_response == ProtocolResponse(
        1, CommandCode.EMERGENCY_STOP, ResultCode.OK
    )
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.PING,
        CommandCode.EMERGENCY_STOP,
    ]
    assert transport.event_history == ["write", "read", "write", "read"]


def test_emergency_after_current_timeout_suppresses_ordinary_retries() -> None:
    transport = _AfterWriteTransport()
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)
    session, _ = _session(transport, max_retries=2)
    transport.after_write = session.request_emergency

    result = session.transact(CommandCode.PING)

    assert result.preempted_by_emergency
    assert result.response is None
    assert result.emergency_response == ProtocolResponse(
        1, CommandCode.EMERGENCY_STOP, ResultCode.OK
    )
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.PING,
        CommandCode.EMERGENCY_STOP,
    ]
    assert transport.event_history == ["write", "read", "write", "read"]


def test_emergency_during_inter_request_delay_preempts_unsent_ordinary_request() -> None:
    waits: list[float] = []
    session_ref: dict[str, TurretSession] = {}

    def request_emergency_during_wait(delay_s: float) -> None:
        waits.append(delay_s)
        session_ref["session"].request_emergency()

    session, transport = _session(wait_hook=request_emergency_during_wait)
    session_ref["session"] = session
    session.transact(CommandCode.PING)

    result = session.transact(CommandCode.MOTOR_OFF)

    assert result.preempted_by_emergency
    assert result.response is None
    assert result.emergency_response == ProtocolResponse(
        2, CommandCode.EMERGENCY_STOP, ResultCode.OK
    )
    assert waits == [0.002]
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.PING,
        CommandCode.EMERGENCY_STOP,
    ]
    assert [decode_request(raw).request_id for raw in transport.raw_write_history] == [
        0,
        2,
    ]
    assert session.next_request_id == 3


def test_repeated_emergency_during_active_emergency_is_coalesced_without_ghost() -> None:
    transport = _AfterWriteTransport()
    session, _ = _session(transport)
    transport.after_write = session.request_emergency

    response = session.request_emergency()
    result = session.transact(CommandCode.PING)

    assert response == ProtocolResponse(0, CommandCode.EMERGENCY_STOP, ResultCode.OK)
    assert not result.preempted_by_emergency
    assert result.response == ProtocolResponse(1, CommandCode.PING, ResultCode.OK)
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.EMERGENCY_STOP,
        CommandCode.PING,
    ]
    assert session.next_request_id == 2


def test_emergency_signal_linearized_after_completion_remains_pending() -> None:
    session, transport = _session(max_retries=0)
    completion_event = _CompletionBoundaryEvent()
    session._emergency_requested = completion_event
    responses: list[ProtocolResponse | None] = []

    emergency_thread = Thread(target=lambda: responses.append(session.request_emergency()))
    emergency_thread.start()
    completion_event.completion_boundary.wait(timeout=0.5)

    signal_thread = Thread(target=session.signal_emergency)
    signal_thread.start()
    completion_event.release_boundary.wait(timeout=0.5)
    emergency_thread.join(0.5)
    signal_thread.join(0.5)

    assert not emergency_thread.is_alive()
    assert not signal_thread.is_alive()
    assert responses == [ProtocolResponse(0, CommandCode.EMERGENCY_STOP, ResultCode.OK)]
    assert session.normal_traffic_blocked
    assert session.emergency_pending

    second = session.request_emergency()

    assert second == ProtocolResponse(1, CommandCode.EMERGENCY_STOP, ResultCode.OK)
    assert not session.normal_traffic_blocked
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.EMERGENCY_STOP,
        CommandCode.EMERGENCY_STOP,
    ]


def test_inter_request_delay_is_injected_not_real_sleep() -> None:
    waits: list[float] = []
    session, _ = _session(waits=waits)

    session.transact(CommandCode.PING)
    session.transact(CommandCode.MOTOR_OFF)

    assert waits == [0.002]


def test_set_baudrate_normal_success_switches_pc_after_old_baud_response() -> None:
    waits: list[float] = []
    endpoint = FakeStm32Endpoint(baudrate=9600)
    session, transport = _session(FakeTransport(endpoint, baudrate=9600), waits=waits)

    response = session.set_baudrate(115200)

    assert response == ProtocolResponse(0, CommandCode.SET_BAUDRATE, ResultCode.OK)
    assert endpoint.baudrate == 115200
    assert transport.baudrate == 115200
    assert transport.raw_write_baudrate_history == [9600]
    assert waits == [0.002]


def test_emergency_during_set_baudrate_runs_next_on_confirmed_new_baud() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    transport = _AfterWriteTransport(endpoint, baudrate=9600)
    session, _ = _session(transport)
    transport.after_write = session.request_emergency

    response = session.set_baudrate(115200)

    assert response == ProtocolResponse(0, CommandCode.SET_BAUDRATE, ResultCode.OK)
    assert endpoint.baudrate == 115200
    assert transport.baudrate == 115200
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.SET_BAUDRATE,
        CommandCode.EMERGENCY_STOP,
    ]
    assert [decode_request(raw).request_id for raw in transport.raw_write_history] == [
        0,
        1,
    ]
    assert transport.raw_write_baudrate_history == [9600, 115200]
    assert session.next_request_id == 2
    assert not session.normal_traffic_blocked


def test_emergency_survives_uncertain_set_baudrate_recovery_until_link_is_known() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    transport = _AfterWriteTransport(endpoint, baudrate=9600)
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)
    session, _ = _session(transport)
    transport.after_write = session.request_emergency

    response = session.set_baudrate(115200)

    assert response == ProtocolResponse(0, CommandCode.SET_BAUDRATE, ResultCode.OK)
    assert endpoint.baudrate == 115200
    assert transport.baudrate == 115200
    assert [decode_request(raw).command for raw in transport.raw_write_history] == [
        CommandCode.SET_BAUDRATE,
        CommandCode.SET_BAUDRATE,
        CommandCode.EMERGENCY_STOP,
    ]
    assert [decode_request(raw).request_id for raw in transport.raw_write_history] == [
        0,
        0,
        1,
    ]
    assert transport.raw_write_history[0] == transport.raw_write_history[1]
    assert transport.raw_write_baudrate_history == [9600, 115200, 115200]
    assert session.next_request_id == 2
    assert not session.normal_traffic_blocked


def test_lost_set_baudrate_response_recovers_when_stm32_already_on_new_baud() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    session, transport = _session(FakeTransport(endpoint, baudrate=9600))
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)

    response = session.set_baudrate(115200)

    assert response.result is ResultCode.OK
    assert endpoint.baudrate == 115200
    assert transport.baudrate == 115200
    assert transport.raw_write_baudrate_history == [9600, 115200]
    assert transport.raw_write_history[0] == transport.raw_write_history[1]
    assert [decode_request(raw).request_id for raw in transport.raw_write_history] == [0, 0]


def test_lost_set_baudrate_response_recovers_when_original_request_not_executed() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    session, transport = _session(FakeTransport(endpoint, baudrate=9600))
    transport.drop_next_request()

    response = session.set_baudrate(115200)

    assert response.result is ResultCode.OK
    assert transport.raw_write_baudrate_history == [9600, 115200, 9600]
    assert len(set(transport.raw_write_history)) == 1
    assert len(endpoint.request_history) == 1
    assert endpoint.request_history[0].request_id == 0
    assert endpoint.baudrate == 115200
    assert transport.baudrate == 115200


def test_uncertain_baud_candidates_follow_new_old_startup_order() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    session, transport = _session(FakeTransport(endpoint, baudrate=38400))

    response = session.set_baudrate(115200)

    assert response.result is ResultCode.OK
    assert transport.raw_write_baudrate_history == [
        38400,
        115200,
        38400,
        9600,
    ]
    assert transport.baudrate == 115200
    assert endpoint.baudrate == 115200


@pytest.mark.parametrize(
    ("old_baud", "new_baud", "expected_attempt_bauds"),
    [
        (9600, 115200, [9600, 115200, 9600]),
        (38400, 9600, [38400, 9600, 38400]),
        (9600, 9600, [9600, 9600]),
    ],
)
def test_uncertain_baud_candidate_order_is_deduplicated(
    old_baud: int, new_baud: int, expected_attempt_bauds: list[int]
) -> None:
    endpoint = FakeStm32Endpoint(baudrate=old_baud)
    transport = FakeTransport(endpoint, baudrate=old_baud)
    transport.drop_next_request()
    session, _ = _session(transport)

    session.set_baudrate(new_baud)

    assert transport.raw_write_baudrate_history == expected_attempt_bauds
    recovery_attempts = expected_attempt_bauds[1:]
    assert len(recovery_attempts) == len(set(recovery_attempts))


def test_same_set_baudrate_transaction_is_reused_across_all_uncertainty_attempts() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    session, transport = _session(FakeTransport(endpoint, baudrate=38400))

    session.set_baudrate(115200)

    assert len(transport.raw_write_history) == 4
    assert len(set(transport.raw_write_history)) == 1
    decoded = [decode_request(raw) for raw in transport.raw_write_history]
    assert all(request.request_id == 0 for request in decoded)
    assert all(request.payload == SetBaudratePayload(115200) for request in decoded)
    assert session.next_request_id == 1


def test_failed_bounded_baud_recovery_is_explicit_and_does_not_loop_forever() -> None:
    endpoint = FakeStm32Endpoint(baudrate=19200)
    session, transport = _session(FakeTransport(endpoint, baudrate=38400))

    with pytest.raises(BaudRecoveryError) as error:
        session.set_baudrate(115200)

    assert error.value.candidates == (115200, 38400, 9600)
    assert transport.raw_write_baudrate_history == [
        38400,
        115200,
        38400,
        9600,
    ]
    assert session.normal_traffic_blocked


def test_set_baudrate_matching_non_ok_result_is_returned_not_transport_loss() -> None:
    endpoint = FakeStm32Endpoint(baudrate=9600)
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_STATE))
    session, transport = _session(FakeTransport(endpoint, baudrate=9600))

    response = session.set_baudrate(115200)

    assert response.result is ResultCode.INVALID_STATE
    assert transport.baudrate == 9600
    assert endpoint.baudrate == 9600
    assert not session.normal_traffic_blocked
