"""PC-side Turret protocol session and transaction state machine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from threading import Event, Lock
from time import monotonic, monotonic_ns, sleep

from navmin.diagnostic_logging import emit_event

from .protocol import (
    SUPPORTED_BAUDRATES,
    UINT16_MAX,
    CommandCode,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    RequestPayload,
    ResultCode,
    SetBaudratePayload,
    decode_response,
    encode_request,
    require_uint16,
)
from .transport import (
    PhysicalTransport,
    TransportDisconnectedError,
    TransportError,
    TransportIOError,
    TransportTimeoutError,
)

_LOGGER = logging.getLogger(__name__)

class SessionError(RuntimeError):
    """Base class for Turret protocol-session failures."""


class SessionBlockedError(SessionError):
    """Normal traffic is blocked until a successful Emergency resync."""


class SessionBusyError(SessionError):
    """A second normal transaction was requested while one is active."""


class SessionTransportError(SessionError):
    """A physical disconnect/I/O failure made the current exchange unusable."""


class SessionRetryExhaustedError(SessionError):
    """An ordinary transaction exhausted its bounded exact retry plan."""

    def __init__(self, request: ProtocolRequest, attempts: int) -> None:
        self.request = request
        self.attempts = attempts
        super().__init__(
            f"{request.command.name} request {request.request_id} exhausted "
            f"{attempts} physical attempts"
        )


class EmergencyRetryExhaustedError(SessionError):
    """Emergency exact retries were exhausted without a matching response."""

    def __init__(self, request: ProtocolRequest, attempts: int) -> None:
        self.request = request
        self.attempts = attempts
        super().__init__(
            f"Emergency request {request.request_id} exhausted {attempts} attempts"
        )


class BaudRecoveryError(SessionError):
    """Bounded uncertain SET_BAUDRATE recovery could not confirm the link."""

    def __init__(self, candidates: tuple[int, ...]) -> None:
        self.candidates = candidates
        super().__init__(
            "SET_BAUDRATE recovery failed for candidates "
            + " -> ".join(str(candidate) for candidate in candidates)
        )


@dataclass(frozen=True)
class SessionResult:
    """Result of one ordinary call, optionally followed by queued Emergency."""

    response: ProtocolResponse | None
    emergency_response: ProtocolResponse | None = None
    preempted_by_emergency: bool = False


@dataclass(frozen=True)
class _Transaction:
    request: ProtocolRequest
    raw_request: bytes
    control_id: int | None = None
    logical_command: object | None = None


@dataclass(frozen=True)
class _AttemptOutcome:
    response: ProtocolResponse | None
    transmitted: bool


type Clock = Callable[[], float]
type WaitHook = Callable[[float], None]


def _require_positive_seconds(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    seconds = float(value)
    if seconds <= 0 or not isfinite(seconds):
        raise ValueError(f"{name} must be a positive finite number")
    return seconds


def _require_nonnegative_seconds(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    seconds = float(value)
    if seconds < 0 or not isfinite(seconds):
        raise ValueError(f"{name} must be a non-negative finite number")
    return seconds


def _require_max_retries(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_retries must be an integer")
    if value < 0:
        raise ValueError("max_retries must be non-negative")
    return value


def _deduplicate_baud_candidates(*values: int) -> tuple[int, ...]:
    candidates: list[int] = []
    for value in values:
        if value not in candidates:
            candidates.append(value)
    return tuple(candidates)


class TurretSession:
    """Single-owner request/response session over one physical transport.

    The class deliberately owns only protocol sequencing and bounded physical
    transactions. HAL/controller/reconnect policy belongs to later checkpoints.
    """

    def __init__(
        self,
        transport: PhysicalTransport,
        *,
        response_timeout_s: float,
        max_retries: int,
        inter_request_delay_s: float,
        initial_request_id: int = 0,
        connection_epoch: int = 0,
        clock: Clock = monotonic,
        wait_hook: WaitHook = sleep,
    ) -> None:
        if not isinstance(transport, PhysicalTransport):
            raise TypeError("transport must implement PhysicalTransport")
        self._transport = transport
        self._response_timeout_s = _require_positive_seconds(
            response_timeout_s, name="response_timeout_s"
        )
        self._max_retries = _require_max_retries(max_retries)
        self._inter_request_delay_s = _require_nonnegative_seconds(
            inter_request_delay_s, name="inter_request_delay_s"
        )
        self._next_request_id = require_uint16(
            initial_request_id, name="initial_request_id"
        )
        if isinstance(connection_epoch, bool) or not isinstance(connection_epoch, int):
            raise TypeError("connection_epoch must be an integer")
        if connection_epoch < 0:
            raise ValueError("connection_epoch must be >= 0")
        self._connection_epoch = connection_epoch
        self._clock = clock
        self._wait_hook = wait_hook
        self._transaction_active = False
        self._attempt_in_flight = False
        self._emergency_requested = Event()
        self._emergency_state_lock = Lock()
        self._emergency_active = False
        self._normal_traffic_blocked = False
        self._inter_request_delay_pending = False

    @property
    def next_request_id(self) -> int:
        return self._next_request_id

    @property
    def normal_traffic_blocked(self) -> bool:
        return self._normal_traffic_blocked

    @property
    def emergency_pending(self) -> bool:
        with self._emergency_state_lock:
            return self._emergency_requested.is_set()

    @property
    def physical_attempt_in_flight(self) -> bool:
        return self._attempt_in_flight

    def transact(
        self,
        command: CommandCode,
        payload: RequestPayload = None,
        *,
        control_id: int | None = None,
        logical_command: object | None = None,
    ) -> SessionResult:
        """Execute one ordinary transaction with bounded exact retries."""
        if command is CommandCode.EMERGENCY_STOP:
            raise ValueError("use request_emergency() for EMERGENCY_STOP")
        if command is CommandCode.SET_BAUDRATE:
            raise ValueError("use set_baudrate() for controlled SET_BAUDRATE")
        self._require_normal_available()
        transaction = self._new_transaction(
            command,
            payload,
            control_id=control_id,
            logical_command=logical_command,
        )

        self._transaction_active = True
        try:
            attempts = self._max_retries + 1
            for attempt in range(1, attempts + 1):
                outcome = self._physical_attempt(
                    transaction, attempt=attempt, allow_emergency_preemption=True
                )
                response = outcome.response
                if not outcome.transmitted and self._emergency_requested.is_set():
                    emergency = self._run_emergency_transaction()
                    return SessionResult(None, emergency, True)
                if response is not None:
                    if response.result is ResultCode.INVALID_REQUEST_ID:
                        self._normal_traffic_blocked = True
                    if self._emergency_requested.is_set():
                        emergency = self._run_emergency_transaction()
                        return SessionResult(response, emergency, True)
                    return SessionResult(response)

                if self._emergency_requested.is_set():
                    emergency = self._run_emergency_transaction()
                    return SessionResult(None, emergency, True)
                if attempt < attempts:
                    emit_event(
                        _LOGGER,
                        "UART_RETRY",
                        source_instance=f"uart:connection_epoch:{self._connection_epoch}",
                        connection_epoch=self._connection_epoch,
                        request_id=transaction.request.request_id,
                        control_id=transaction.control_id,
                        completed_attempt=attempt,
                        next_attempt=attempt + 1,
                        reason="no_matching_response",
                        backoff_ms=self._inter_request_delay_s * 1000.0,
                    )

            self._normal_traffic_blocked = True
            raise SessionRetryExhaustedError(transaction.request, attempts)
        finally:
            self._transaction_active = False

    def signal_emergency(self) -> bool:
        """Thread-safe signal-only Emergency preemption boundary.

        This method never performs transport I/O. The worker may call it from
        another thread to suppress ordinary retries; the worker-owned session
        operation services the Emergency at the next safe physical boundary.
        """
        with self._emergency_state_lock:
            if self._emergency_active:
                return False
            self._normal_traffic_blocked = True
            self._emergency_requested.set()
            return True

    def request_emergency(self) -> ProtocolResponse | None:
        """Request Emergency now, or queue it behind the current physical attempt."""
        with self._emergency_state_lock:
            self._normal_traffic_blocked = True
            if self._emergency_active:
                return None
            if self._transaction_active:
                self._emergency_requested.set()
                return None

        self._transaction_active = True
        try:
            return self._run_emergency_transaction()
        finally:
            self._transaction_active = False

    def set_baudrate(self, baudrate: int) -> ProtocolResponse:
        """Perform controlled SET_BAUDRATE with bounded old/new uncertainty recovery.

        Precondition: the caller has already confirmed that STM32 motors are OFF.
        """
        self._require_normal_available()
        if baudrate not in SUPPORTED_BAUDRATES:
            raise ValueError(
                f"baudrate must be one of {', '.join(map(str, SUPPORTED_BAUDRATES))}"
            )

        old_baudrate = self._transport.baudrate
        transaction = self._new_transaction(
            CommandCode.SET_BAUDRATE, SetBaudratePayload(baudrate)
        )
        self._transaction_active = True
        try:
            outcome = self._physical_attempt(
                transaction, attempt=1, allow_emergency_preemption=True
            )
            if not outcome.transmitted and self._emergency_requested.is_set():
                emergency = self._run_emergency_transaction()
                if emergency.result is not ResultCode.OK:
                    raise SessionBlockedError(
                        "SET_BAUDRATE was preempted by unsuccessful Emergency"
                    )
                old_baudrate = self._transport.baudrate
                transaction = self._new_transaction(
                    CommandCode.SET_BAUDRATE, SetBaudratePayload(baudrate)
                )
                outcome = self._physical_attempt(
                    transaction, attempt=1, allow_emergency_preemption=False
                )

            response = outcome.response
            if response is not None:
                return self._finish_baud_and_service_emergency(response, baudrate)

            candidates = _deduplicate_baud_candidates(
                baudrate, old_baudrate, 9600
            )
            for recovery_attempt, candidate in enumerate(candidates, start=2):
                emit_event(
                    _LOGGER,
                    "UART_RETRY",
                    source_instance=f"uart:connection_epoch:{self._connection_epoch}",
                    connection_epoch=self._connection_epoch,
                    request_id=transaction.request.request_id,
                    control_id=transaction.control_id,
                    completed_attempt=recovery_attempt - 1,
                    next_attempt=recovery_attempt,
                    reason="baud_recovery_probe",
                    backoff_ms=None,
                    candidate_baud=candidate,
                )
                try:
                    self._transport.set_baudrate(candidate)
                    response = self._physical_attempt(
                        transaction,
                        attempt=recovery_attempt,
                        allow_emergency_preemption=False,
                    ).response
                except (
                    SessionTransportError,
                    TransportDisconnectedError,
                    TransportIOError,
                ) as exc:
                    self._normal_traffic_blocked = True
                    self._emergency_requested.clear()
                    raise BaudRecoveryError(candidates) from exc

                if response is None:
                    continue
                return self._finish_baud_and_service_emergency(response, baudrate)

            self._normal_traffic_blocked = True
            self._emergency_requested.clear()
            raise BaudRecoveryError(candidates)
        finally:
            self._transaction_active = False

    def _finish_baud_and_service_emergency(
        self, response: ProtocolResponse, desired_baudrate: int
    ) -> ProtocolResponse:
        try:
            result = self._finish_baud_response(response, desired_baudrate)
        except BaudRecoveryError:
            self._emergency_requested.clear()
            raise

        if self._emergency_requested.is_set():
            self._run_emergency_transaction()
        return result

    def _finish_baud_response(
        self, response: ProtocolResponse, desired_baudrate: int
    ) -> ProtocolResponse:
        if response.result is ResultCode.INVALID_REQUEST_ID:
            self._normal_traffic_blocked = True
            return response
        if response.result is not ResultCode.OK:
            return response

        # STM32 switches only after transmitting OK on the old/current baud.
        self._consume_inter_request_delay()
        if self._transport.baudrate == desired_baudrate:
            return response
        try:
            self._transport.set_baudrate(desired_baudrate)
        except TransportError as exc:
            self._normal_traffic_blocked = True
            raise BaudRecoveryError((desired_baudrate,)) from exc
        return response

    def _run_emergency_transaction(self) -> ProtocolResponse:
        with self._emergency_state_lock:
            self._emergency_requested.clear()
            self._emergency_active = True
        try:
            transaction = self._new_transaction(CommandCode.EMERGENCY_STOP, None)
            attempts = self._max_retries + 1

            for attempt in range(1, attempts + 1):
                response = self._physical_attempt(
                    transaction,
                    attempt=attempt,
                    allow_emergency_preemption=False,
                ).response
                if response is None:
                    if attempt < attempts:
                        emit_event(
                            _LOGGER,
                            "UART_RETRY",
                            source_instance=(
                                f"uart:connection_epoch:{self._connection_epoch}"
                            ),
                            connection_epoch=self._connection_epoch,
                            request_id=transaction.request.request_id,
                            control_id=transaction.control_id,
                            completed_attempt=attempt,
                            next_attempt=attempt + 1,
                            reason="emergency_no_matching_response",
                            backoff_ms=self._inter_request_delay_s * 1000.0,
                        )
                    continue
                if response.result is ResultCode.OK:
                    self._next_request_id = (
                        transaction.request.request_id + 1
                    ) & UINT16_MAX
                    self._normal_traffic_blocked = False
                else:
                    self._normal_traffic_blocked = True
                return response

            self._normal_traffic_blocked = True
            raise EmergencyRetryExhaustedError(transaction.request, attempts)
        finally:
            with self._emergency_state_lock:
                # A signal linearized while Emergency is active coalesces with it.
                # A signal linearized after this completion boundary must remain
                # pending for a subsequent Emergency.
                self._emergency_requested.clear()
                self._emergency_active = False

    def _physical_attempt(
        self,
        transaction: _Transaction,
        *,
        attempt: int,
        allow_emergency_preemption: bool,
    ) -> _AttemptOutcome:
        self._consume_inter_request_delay()
        if allow_emergency_preemption and self._emergency_requested.is_set():
            return _AttemptOutcome(response=None, transmitted=False)

        source_instance = f"uart:connection_epoch:{self._connection_epoch}"
        request = transaction.request
        tx_ns = monotonic_ns()
        emit_event(
            _LOGGER,
            "UART_TX",
            source_instance=source_instance,
            event_monotonic_ns=tx_ns,
            connection_epoch=self._connection_epoch,
            request_id=request.request_id,
            attempt=attempt,
            control_id=transaction.control_id,
            command_type=request.command.name,
            logical_command=transaction.logical_command,
            protocol_payload=request.payload,
            payload_bytes=len(transaction.raw_request),
            tx_monotonic_ns=tx_ns,
        )
        self._attempt_in_flight = True
        try:
            try:
                self._transport.write_frame(
                    transaction.raw_request, self._response_timeout_s
                )
            except TransportTimeoutError:
                rx_ns = monotonic_ns()
                self._log_uart_terminal(
                    transaction,
                    attempt=attempt,
                    tx_ns=tx_ns,
                    rx_ns=rx_ns,
                    transaction_result="timeout",
                    error_reason="write_timeout",
                )
                return _AttemptOutcome(response=None, transmitted=True)
            except (TransportDisconnectedError, TransportIOError) as exc:
                rx_ns = monotonic_ns()
                self._log_uart_terminal(
                    transaction,
                    attempt=attempt,
                    tx_ns=tx_ns,
                    rx_ns=rx_ns,
                    transaction_result="io_error",
                    error_reason=type(exc).__name__,
                )
                self._normal_traffic_blocked = True
                raise SessionTransportError(str(exc)) from exc

            deadline = self._clock() + self._response_timeout_s
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    rx_ns = monotonic_ns()
                    self._log_uart_terminal(
                        transaction,
                        attempt=attempt,
                        tx_ns=tx_ns,
                        rx_ns=rx_ns,
                        transaction_result="timeout",
                        error_reason="response_timeout",
                    )
                    return _AttemptOutcome(response=None, transmitted=True)
                try:
                    raw_response = self._transport.read_frame(remaining)
                except TransportTimeoutError:
                    rx_ns = monotonic_ns()
                    self._log_uart_terminal(
                        transaction,
                        attempt=attempt,
                        tx_ns=tx_ns,
                        rx_ns=rx_ns,
                        transaction_result="timeout",
                        error_reason="response_timeout",
                    )
                    return _AttemptOutcome(response=None, transmitted=True)
                except (TransportDisconnectedError, TransportIOError) as exc:
                    rx_ns = monotonic_ns()
                    self._log_uart_terminal(
                        transaction,
                        attempt=attempt,
                        tx_ns=tx_ns,
                        rx_ns=rx_ns,
                        transaction_result="io_error",
                        error_reason=type(exc).__name__,
                    )
                    self._normal_traffic_blocked = True
                    raise SessionTransportError(str(exc)) from exc

                rx_ns = monotonic_ns()
                try:
                    response = decode_response(raw_response)
                except ProtocolError as exc:
                    self._log_uart_terminal(
                        transaction,
                        attempt=attempt,
                        tx_ns=tx_ns,
                        rx_ns=rx_ns,
                        transaction_result="protocol_error",
                        error_reason=type(exc).__name__,
                        crc_valid=False,
                    )
                    return _AttemptOutcome(response=None, transmitted=True)

                if (
                    response.request_id != request.request_id
                    or response.command is not request.command
                ):
                    self._log_uart_terminal(
                        transaction,
                        attempt=attempt,
                        tx_ns=tx_ns,
                        rx_ns=rx_ns,
                        transaction_result="request_mismatch",
                        error_reason="response_identity_mismatch",
                        response=response,
                    )
                    continue

                self._inter_request_delay_pending = True
                self._log_uart_terminal(
                    transaction,
                    attempt=attempt,
                    tx_ns=tx_ns,
                    rx_ns=rx_ns,
                    transaction_result="ok",
                    error_reason=None,
                    response=response,
                    crc_valid=True,
                )
                return _AttemptOutcome(response=response, transmitted=True)
        finally:
            self._attempt_in_flight = False

    def _log_uart_terminal(
        self,
        transaction: _Transaction,
        *,
        attempt: int,
        tx_ns: int,
        rx_ns: int,
        transaction_result: str,
        error_reason: str | None,
        response: ProtocolResponse | None = None,
        crc_valid: bool | None = None,
    ) -> None:
        emit_event(
            _LOGGER,
            "UART_RX",
            source_instance=f"uart:connection_epoch:{self._connection_epoch}",
            event_monotonic_ns=rx_ns,
            connection_epoch=self._connection_epoch,
            request_id=transaction.request.request_id,
            attempt=attempt,
            control_id=transaction.control_id,
            command_type=transaction.request.command.name,
            rx_monotonic_ns=rx_ns,
            response_code=None if response is None else response.result.name,
            response_payload=None,
            crc_valid=crc_valid,
            transaction_result=transaction_result,
            round_trip_ms=max(0.0, (rx_ns - tx_ns) / 1_000_000.0),
            error_reason=error_reason,
        )

    def _new_transaction(
        self,
        command: CommandCode,
        payload: RequestPayload,
        *,
        control_id: int | None = None,
        logical_command: object | None = None,
    ) -> _Transaction:
        request_id = self._next_request_id
        request = ProtocolRequest(request_id, command, payload)
        raw_request = encode_request(request)
        self._next_request_id = (request_id + 1) & UINT16_MAX
        return _Transaction(
            request=request,
            raw_request=raw_request,
            control_id=control_id,
            logical_command=logical_command,
        )

    def _consume_inter_request_delay(self) -> None:
        if not self._inter_request_delay_pending:
            return
        self._inter_request_delay_pending = False
        if self._inter_request_delay_s > 0:
            self._wait_hook(self._inter_request_delay_s)

    def _require_normal_available(self) -> None:
        if self._transaction_active:
            raise SessionBusyError("another session transaction is active")
        if self._normal_traffic_blocked:
            raise SessionBlockedError(
                "normal traffic is blocked until successful Emergency resync"
            )


__all__ = [
    "BaudRecoveryError",
    "EmergencyRetryExhaustedError",
    "SessionBlockedError",
    "SessionBusyError",
    "SessionError",
    "SessionResult",
    "SessionRetryExhaustedError",
    "SessionTransportError",
    "TurretSession",
]
