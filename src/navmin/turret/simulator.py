"""Deterministic frame-level fake STM32 transport used by Turret tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .protocol import (
    CommandCode,
    ProtocolError,
    ProtocolRequest,
    ProtocolResponse,
    ResultCode,
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
    """Frame-level fake endpoint that parses requests using production codec."""

    def __init__(self) -> None:
        self.raw_request_history: list[bytes] = []
        self.request_history: list[ProtocolRequest] = []
        self._response_specs: deque[FakeResponseSpec] = deque()

    def queue_response(self, spec: FakeResponseSpec) -> None:
        self._response_specs.append(spec)

    def handle_request(self, raw_frame: bytes) -> bytes | None:
        raw = bytes(raw_frame)
        self.raw_request_history.append(raw)
        try:
            request = decode_request(raw)
        except ProtocolError:
            return None

        self.request_history.append(request)
        spec = (
            self._response_specs.popleft()
            if self._response_specs
            else FakeResponseSpec()
        )
        if spec.raw_response is not None:
            return bytes(spec.raw_response)

        response = ProtocolResponse(
            request_id=(
                request.request_id if spec.request_id is None else spec.request_id
            ),
            command=request.command if spec.command is None else spec.command,
            result=spec.result,
        )
        raw_response = bytearray(encode_response(response))
        if spec.corrupt_crc:
            raw_response[-1] ^= 0xFF
        return bytes(raw_response)


class FakeTransport(PhysicalTransport):
    """Deterministic physical transport with separately scriptable read failures."""

    def __init__(
        self,
        endpoint: FakeStm32Endpoint | None = None,
        *,
        baudrate: int = 9600,
    ) -> None:
        self.endpoint = endpoint or FakeStm32Endpoint()
        self._baudrate = 9600
        self.set_baudrate(baudrate)
        self._is_open = False
        self.raw_write_history: list[bytes] = []
        self._pending_responses: deque[bytes | None] = deque()
        self._read_failures: deque[FakeReadFailure] = deque()

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

    def set_baudrate(self, baudrate: int) -> None:
        if (
            isinstance(baudrate, bool)
            or not isinstance(baudrate, int)
            or baudrate <= 0
        ):
            raise ValueError("baudrate must be a positive integer")
        self._baudrate = baudrate

    def queue_read_failure(self, failure: FakeReadFailure) -> None:
        if not isinstance(failure, FakeReadFailure):
            raise TypeError("failure must be FakeReadFailure")
        self._read_failures.append(failure)

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        del timeout_s
        self._require_open()
        raw = bytes(frame)
        self.raw_write_history.append(raw)
        self._pending_responses.append(self.endpoint.handle_request(raw))

    def read_frame(self, timeout_s: float) -> bytes:
        del timeout_s
        self._require_open()
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

        if not self._pending_responses:
            raise TransportTimeoutError("no simulated response is pending")
        response = self._pending_responses.popleft()
        if response is None:
            raise TransportTimeoutError("simulated endpoint produced no response")
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
