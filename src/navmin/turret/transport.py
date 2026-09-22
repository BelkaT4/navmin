"""Physical transport boundary for the turret serial protocol."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from time import monotonic
from typing import Any, Protocol, runtime_checkable

from .protocol import MIN_RESPONSE_LENGTH, START_BYTES


class TransportError(OSError):
    """Base class for physical transport failures."""


class TransportTimeoutError(TransportError):
    """A bounded physical operation did not complete before its timeout."""


class TransportDisconnectedError(TransportError):
    """The physical transport is not connected/open."""


class TransportIOError(TransportError):
    """The physical transport reported an I/O failure."""


@runtime_checkable
class PhysicalTransport(Protocol):
    @property
    def is_open(self) -> bool: ...

    @property
    def baudrate(self) -> int: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def set_baudrate(self, baudrate: int) -> None: ...

    def write_frame(self, frame: bytes, timeout_s: float) -> None: ...

    def read_frame(self, timeout_s: float) -> bytes: ...


type SerialFactory = Callable[..., Any]


def _require_timeout(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        raise TypeError("timeout_s must be a number")
    timeout = float(timeout_s)
    if timeout <= 0 or not isfinite(timeout):
        raise ValueError("timeout_s must be a positive finite number")
    return timeout


def _require_baudrate(baudrate: int) -> int:
    if (
        isinstance(baudrate, bool)
        or not isinstance(baudrate, int)
        or baudrate <= 0
    ):
        raise ValueError("baudrate must be a positive integer")
    return baudrate


class SerialTransport:
    """Bounded raw-frame transport backed by pyserial."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        *,
        serial_factory: SerialFactory | None = None,
    ) -> None:
        if not isinstance(port, str) or not port:
            raise ValueError("port must be a non-empty string")
        self._port = port
        self._baudrate = _require_baudrate(baudrate)
        self._serial_factory = serial_factory
        self._serial: Any | None = None
        self._serial_timeout_error: type[Exception] | None = None
        self._serial_io_error: type[Exception] | None = None

    @property
    def is_open(self) -> bool:
        serial = self._serial
        return bool(serial is not None and getattr(serial, "is_open", False))

    @property
    def baudrate(self) -> int:
        return self._baudrate

    def open(self) -> None:
        if self.is_open:
            return

        factory = self._serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise TransportIOError("pyserial is not available") from exc
            factory = serial.Serial
            self._serial_timeout_error = serial.SerialTimeoutException
            self._serial_io_error = serial.SerialException

        try:
            self._serial = factory(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=0,
                write_timeout=0,
            )
        except Exception as exc:
            self._serial = None
            raise self._translate_error(exc, "opening serial port") from exc

    def close(self) -> None:
        serial = self._serial
        self._serial = None
        if serial is None:
            return
        try:
            serial.close()
        except Exception as exc:
            raise self._translate_error(exc, "closing serial port") from exc

    def set_baudrate(self, baudrate: int) -> None:
        value = _require_baudrate(baudrate)
        serial = self._serial
        if serial is not None and getattr(serial, "is_open", False):
            try:
                serial.baudrate = value
            except Exception as exc:
                raise self._translate_error(exc, "changing baudrate") from exc
        self._baudrate = value

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        timeout = _require_timeout(timeout_s)
        raw = bytes(frame)
        if not raw:
            raise ValueError("frame must not be empty")
        serial = self._require_open_serial()
        try:
            serial.write_timeout = timeout
            written = serial.write(raw)
        except Exception as exc:
            raise self._translate_error(exc, "writing serial frame") from exc
        if written != len(raw):
            raise TransportIOError(
                f"partial serial write: wrote {written} of {len(raw)} bytes"
            )

    def read_frame(self, timeout_s: float) -> bytes:
        timeout = _require_timeout(timeout_s)
        serial = self._require_open_serial()
        deadline = monotonic() + timeout

        start_prefix_length = 0
        while True:
            byte = self._read_exact(serial, 1, deadline)[0]

            if start_prefix_length == 0:
                if byte == START_BYTES[0]:
                    start_prefix_length = 1
                continue

            if byte == START_BYTES[1]:
                declared_length = self._read_exact(serial, 1, deadline)[0]
                if declared_length < MIN_RESPONSE_LENGTH:
                    start_prefix_length = 0
                    continue

                header = START_BYTES + bytes((declared_length,))
                remainder = self._read_exact(serial, declared_length - 3, deadline)
                return header + remainder

            start_prefix_length = 1 if byte == START_BYTES[0] else 0

    def _read_exact(self, serial: Any, size: int, deadline: float) -> bytes:
        data = bytearray()
        while len(data) < size:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TransportTimeoutError(
                    f"serial read timed out after {len(data)} of {size} bytes"
                )
            try:
                serial.timeout = remaining
                chunk = serial.read(size - len(data))
            except Exception as exc:
                raise self._translate_error(exc, "reading serial frame") from exc
            if not chunk:
                raise TransportTimeoutError(
                    f"serial read timed out after {len(data)} of {size} bytes"
                )
            data.extend(chunk)
        return bytes(data)

    def _require_open_serial(self) -> Any:
        serial = self._serial
        if serial is None or not getattr(serial, "is_open", False):
            raise TransportDisconnectedError("serial transport is not open")
        return serial

    def _translate_error(self, exc: Exception, operation: str) -> TransportError:
        if isinstance(exc, TransportError):
            return exc
        timeout_type = self._serial_timeout_error
        if timeout_type is not None and isinstance(exc, timeout_type):
            return TransportTimeoutError(f"timeout while {operation}: {exc}")
        io_type = self._serial_io_error
        if io_type is not None and isinstance(exc, io_type):
            return TransportIOError(f"serial I/O error while {operation}: {exc}")
        if isinstance(exc, TimeoutError):
            return TransportTimeoutError(f"timeout while {operation}: {exc}")
        if isinstance(exc, OSError):
            return TransportIOError(f"I/O error while {operation}: {exc}")
        return TransportIOError(f"serial failure while {operation}: {exc}")


__all__ = [
    "PhysicalTransport",
    "SerialFactory",
    "SerialTransport",
    "TransportDisconnectedError",
    "TransportError",
    "TransportIOError",
    "TransportTimeoutError",
]
