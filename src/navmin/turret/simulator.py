"""Deterministic frame-level fake STM32 transport used by Turret tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .protocol import (
    SUPPORTED_BAUDRATES,
    CommandCode,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    ResultCode,
    SetBaudratePayload,
    decode_request,
    encode_response,
)
from .transport import (
    PhysicalTransport,
    TransportDisconnectedError,
    TransportIOError,
    TransportTimeoutError,
)


@dataclass(frozen=True)
class FakeResponseSpec:
    """How the fake STM32 should encode the response to one valid request."""

    result: ResultCode = ResultCode.OK
    request_id: int | None = None
    command: CommandCode | None = None
    corrupt_crc: bool = False
    raw_response: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, ResultCode):
            raise TypeError("result must be ResultCode")
        if self.command is not None and not isinstance(self.command, CommandCode):
            raise TypeError("command must be CommandCode or None")


class FakeReadFailure(Enum):
    TIMEOUT = "timeout"
    IO_ERROR = "io_error"
    DISCONNECT = "disconnect"


class FakeStm32Endpoint:
    """Frame-level fake endpoint with firmware-like request sequencing/cache."""

    def __init__(
        self,
        *,
        baudrate: int = 9600,
        initial_expected_request_id: int = 0,
    ) -> None:
        if baudrate not in SUPPORTED_BAUDRATES:
            raise ValueError("baudrate must be a supported protocol baudrate")
        if (
            isinstance(initial_expected_request_id, bool)
            or not isinstance(initial_expected_request_id, int)
            or not 0 <= initial_expected_request_id <= 0xFFFF
        ):
            raise ValueError("initial_expected_request_id must fit uint16")
        self.raw_request_history: list[bytes] = []
        self.request_history: list[ProtocolRequest] = []
        self.executed_request_history: list[ProtocolRequest] = []
        self._response_specs: deque[FakeResponseSpec] = deque()
        self._baudrate = baudrate
        self._last_response_baudrate = baudrate
        self._expected_request_id = initial_expected_request_id
        self._last_signature: tuple[int, CommandCode, object | None] | None = None
        self._last_response: bytes | None = None

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @property
    def last_response_baudrate(self) -> int:
        return self._last_response_baudrate

    @property
    def expected_request_id(self) -> int:
        return self._expected_request_id

    def queue_response(self, spec: FakeResponseSpec) -> None:
        self._response_specs.append(spec)

    @staticmethod
    def _signature(request: ProtocolRequest) -> tuple[int, CommandCode, object | None]:
        return (request.request_id, request.command, request.payload)

    @staticmethod
    def _invalid_request_id_response(request: ProtocolRequest) -> bytes:
        return encode_response(
            ProtocolResponse(
                request_id=request.request_id,
                command=request.command,
                result=ResultCode.INVALID_REQUEST_ID,
            )
        )

    def handle_request(self, raw_frame: bytes) -> bytes | None:
        raw = bytes(raw_frame)
        self.raw_request_history.append(raw)
        try:
            request = decode_request(raw)
        except ProtocolError:
            return None

        self.request_history.append(request)
        self._last_response_baudrate = self._baudrate
        signature = self._signature(request)

        # Firmware exact retry is handled before Emergency/ordinary sequence logic.
        if signature == self._last_signature and self._last_response is not None:
            return self._last_response

        if request.command is not CommandCode.EMERGENCY_STOP:
            if (
                self._last_signature is not None
                and self._last_signature[0] == request.request_id
            ):
                return self._invalid_request_id_response(request)
            if request.request_id != self._expected_request_id:
                return self._invalid_request_id_response(request)

        # A new Emergency is a special sequence-resync boundary regardless of the
        # ordinary expected ID. A new ordinary request reaches this point only at
        # the currently expected ID.
        self.executed_request_history.append(request)
        spec = (
            self._response_specs.popleft()
            if self._response_specs
            else FakeResponseSpec()
        )

        response = ProtocolResponse(
            request_id=(
                request.request_id if spec.request_id is None else spec.request_id
            ),
            command=request.command if spec.command is None else spec.command,
            result=spec.result,
        )
        canonical_response = encode_response(response)

        self._expected_request_id = (request.request_id + 1) & 0xFFFF
        self._last_signature = signature
        self._last_response = canonical_response

        response_baudrate = self._baudrate
        self._last_response_baudrate = response_baudrate
        if request.command is CommandCode.SET_BAUDRATE:
            payload = request.payload
            if (
                response.request_id == request.request_id
                and response.command is request.command
                and response.result is ResultCode.OK
                and isinstance(payload, SetBaudratePayload)
                and payload.baudrate in SUPPORTED_BAUDRATES
            ):
                # The response is modeled as physically emitted at the old baud;
                # only then does the endpoint move to the requested baud.
                self._baudrate = payload.baudrate

        if spec.raw_response is not None:
            return bytes(spec.raw_response)
        raw_response = bytearray(canonical_response)
        if spec.corrupt_crc:
            raw_response[-1] ^= 0xFF
        return bytes(raw_response)


class FakeTransport(PhysicalTransport):
    """Deterministic physical transport with scriptable frame-level failures."""

    def __init__(
        self,
        endpoint: FakeStm32Endpoint | None = None,
        *,
        baudrate: int = 9600,
    ) -> None:
        self.endpoint = endpoint or FakeStm32Endpoint()
        self._baudrate = 9600
        self.baudrate_history: list[int] = []
        self.set_baudrate(baudrate)
        self._is_open = False
        self.raw_write_history: list[bytes] = []
        self.raw_write_baudrate_history: list[int] = []
        self.event_history: list[str] = []
        self._pending_responses: deque[tuple[bytes | None, int]] = deque()
        self._read_failures: deque[FakeReadFailure] = deque()
        self._injected_read_frames: deque[bytes] = deque()
        self._drop_request_count = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def baudrate(self) -> int:
        return self._baudrate

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False
        self._pending_responses.clear()
        self._injected_read_frames.clear()

    def set_baudrate(self, baudrate: int) -> None:
        if (
            isinstance(baudrate, bool)
            or not isinstance(baudrate, int)
            or baudrate <= 0
        ):
            raise ValueError("baudrate must be a positive integer")
        self._baudrate = baudrate
        self.baudrate_history.append(baudrate)

    def queue_read_failure(self, failure: FakeReadFailure) -> None:
        if not isinstance(failure, FakeReadFailure):
            raise TypeError("failure must be FakeReadFailure")
        self._read_failures.append(failure)

    def queue_read_frame(self, frame: bytes) -> None:
        self._injected_read_frames.append(bytes(frame))

    def drop_next_request(self) -> None:
        """Record the next physical write but do not deliver it to STM32."""
        self._drop_request_count += 1

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        del timeout_s
        self._require_open()
        raw = bytes(frame)
        self.event_history.append("write")
        self.raw_write_history.append(raw)
        self.raw_write_baudrate_history.append(self._baudrate)

        if self._drop_request_count:
            self._drop_request_count -= 1
            self._pending_responses.append((None, self._baudrate))
            return

        if self._baudrate != self.endpoint.baudrate:
            self._pending_responses.append((None, self._baudrate))
            return

        response = self.endpoint.handle_request(raw)
        self._pending_responses.append(
            (response, self.endpoint.last_response_baudrate)
        )

    def read_frame(self, timeout_s: float) -> bytes:
        del timeout_s
        self._require_open()
        self.event_history.append("read")

        if self._read_failures:
            failure = self._read_failures.popleft()
            if self._pending_responses:
                self._pending_responses.popleft()
            if failure is FakeReadFailure.TIMEOUT:
                raise TransportTimeoutError("simulated response timeout")
            if failure is FakeReadFailure.IO_ERROR:
                raise TransportIOError("simulated physical I/O failure")
            self._is_open = False
            raise TransportDisconnectedError("simulated disconnect")

        if self._injected_read_frames:
            return self._injected_read_frames.popleft()

        if not self._pending_responses:
            raise TransportTimeoutError("no simulated response is pending")
        response, response_baudrate = self._pending_responses.popleft()
        if response is None or response_baudrate != self._baudrate:
            raise TransportTimeoutError("simulated response timeout")
        return response

    def _require_open(self) -> None:
        if not self._is_open:
            raise TransportDisconnectedError("fake transport is not open")


__all__ = [
    "FakeReadFailure",
    "FakeResponseSpec",
    "FakeStm32Endpoint",
    "FakeTransport",
]
