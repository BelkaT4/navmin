"""Linux PTY byte-stream bridge to the existing fake STM32 endpoint."""

from __future__ import annotations

import errno
import os
import platform
import select
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Self

from navmin.turret.protocol import (
    MAX_FRAME_LENGTH,
    MIN_REQUEST_LENGTH,
    START_BYTES,
    ProtocolError,
    ProtocolRequest,
    decode_request,
)
from navmin.turret.simulator import FakeResponseSpec, FakeStm32Endpoint

_MASTER_READ_SIZE = 7
_SERVICE_POLL_SECONDS = 0.02
_WRITE_TIMEOUT_SECONDS = 0.5
_RECENT_HISTORY_LIMIT = 128


class PtyDiagnosticError(RuntimeError):
    """The PTY diagnostic boundary could not start, run, or stop cleanly."""


class RequestFrameAssembler:
    """Incrementally assemble bounded NavMin request frames from arbitrary chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes | bytearray | memoryview) -> tuple[bytes, ...]:
        raw = bytes(chunk)
        frames: list[bytes] = []
        for offset in range(0, len(raw), MAX_FRAME_LENGTH):
            self._buffer.extend(raw[offset : offset + MAX_FRAME_LENGTH])
            frames.extend(self._drain())
            if len(self._buffer) > MAX_FRAME_LENGTH:
                # A valid uint8-length candidate can never require this much data.
                # Retain only a possible overlapping START prefix.
                self._buffer[:] = (
                    START_BYTES[:1] if self._buffer[-1:] == START_BYTES[:1] else b""
                )
        return tuple(frames)

    def _drain(self) -> list[bytes]:
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(START_BYTES)
            if start < 0:
                self._buffer[:] = (
                    START_BYTES[:1] if self._buffer[-1:] == START_BYTES[:1] else b""
                )
                return frames
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                return frames

            declared_length = self._buffer[2]
            if declared_length < MIN_REQUEST_LENGTH:
                del self._buffer[0]
                continue
            if len(self._buffer) < declared_length:
                return frames

            frames.append(bytes(self._buffer[:declared_length]))
            del self._buffer[:declared_length]


class PtyFaultKind(Enum):
    FRAGMENTED_RESPONSE = "fragmented-response"
    LEADING_GARBAGE = "leading-garbage"
    OVERLAPPING_START_PREFIX = "overlapping-start-prefix"
    DELAYED_RESPONSE = "delayed-response"
    DROP_RESPONSE = "drop-response"
    BAD_CRC_RESPONSE = "bad-crc-response"
    HARD_DISCONNECT = "hard-disconnect"


@dataclass(frozen=True)
class PtyFault:
    """One deterministic fault consumed by the next complete request boundary."""

    kind: PtyFaultKind
    delay_s: float = 0.01
    fragment_sizes: tuple[int, ...] = (1, 2)
    garbage: bytes = b"\x00garbage"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PtyFaultKind):
            raise TypeError("kind must be PtyFaultKind")
        if self.delay_s < 0:
            raise ValueError("delay_s must be non-negative")
        if not self.fragment_sizes or any(
            type(size) is not int or size <= 0 for size in self.fragment_sizes
        ):
            raise ValueError("fragment_sizes must contain positive integers")


class PtyFaultQueue:
    """Small FIFO of one-shot response-boundary faults."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._faults: deque[PtyFault] = deque()

    def put(self, fault: PtyFault) -> None:
        if not isinstance(fault, PtyFault):
            raise TypeError("fault must be PtyFault")
        with self._lock:
            self._faults.append(fault)

    def pop(self) -> PtyFault | None:
        with self._lock:
            return self._faults.popleft() if self._faults else None

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._faults)


class StableDiagnosticPort:
    """Own a temporary stable symlink that can be atomically repointed."""

    def __init__(self) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="navmin-pty-"))
        self._path = self._directory / "stm32"
        self._revision = 0
        self._cleaned = False

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def directory(self) -> str:
        return str(self._directory)

    @property
    def exists(self) -> bool:
        return self._path.is_symlink()

    def repoint(self, target: str) -> None:
        if self._cleaned:
            raise PtyDiagnosticError("stable diagnostic port was already cleaned")
        if not target:
            raise ValueError("target must not be empty")
        self._revision += 1
        replacement = self._directory / f".stm32-next-{self._revision}"
        try:
            replacement.symlink_to(target)
            os.replace(replacement, self._path)
        finally:
            replacement.unlink(missing_ok=True)

    def read_target(self) -> str:
        return os.readlink(self._path)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        shutil.rmtree(self._directory)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()


@dataclass(frozen=True)
class PtyRequestRecord:
    sequence: int
    raw_frame: bytes
    request: ProtocolRequest | None


@dataclass(frozen=True)
class PtyEmulatorStats:
    physical_request_count: int
    response_count: int
    replacement_count: int
    recent_requests: tuple[PtyRequestRecord, ...]
    recent_events: tuple[str, ...]


@dataclass(frozen=True)
class EndpointSnapshot:
    request_count: int
    executed_request_count: int
    baudrate: int
    expected_request_id: int


@dataclass(frozen=True)
class PtyReplacement:
    stable_path: str
    old_slave_path: str
    new_slave_path: str


@dataclass(frozen=True)
class PtyPreflightResult:
    linux: bool
    openpty_available: bool
    pyserial_available: bool

    @property
    def ok(self) -> bool:
        return self.linux and self.openpty_available and self.pyserial_available


def check_pty_preflight() -> PtyPreflightResult:
    """Check requirements that do not mutate production configuration."""
    try:
        import serial

        pyserial_available = callable(serial.Serial)
    except ImportError:
        pyserial_available = False
    return PtyPreflightResult(
        linux=platform.system() == "Linux",
        openpty_available=callable(getattr(os, "openpty", None)),
        pyserial_available=pyserial_available,
    )


class PtyStm32Emulator:
    """Own a replaceable Linux PTY and bridge frames to ``FakeStm32Endpoint``."""

    def __init__(self, endpoint: FakeStm32Endpoint | None = None) -> None:
        self.endpoint = endpoint or FakeStm32Endpoint()
        self._stable_port: StableDiagnosticPort | None = None
        self._master_fd: int | None = None
        self._slave_path: str | None = None
        self._pair_lock = Lock()
        self._endpoint_lock = Lock()
        self._stats_lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._assembler = RequestFrameAssembler()
        self._faults = PtyFaultQueue()
        self._request_count = 0
        self._response_count = 0
        self._replacement_count = 0
        self._recent_requests: deque[PtyRequestRecord] = deque(
            maxlen=_RECENT_HISTORY_LIMIT
        )
        self._recent_events: deque[str] = deque(maxlen=_RECENT_HISTORY_LIMIT)
        self._service_error: BaseException | None = None

    @property
    def stable_port_path(self) -> str:
        stable = self._stable_port
        if stable is None:
            raise PtyDiagnosticError("PTY emulator is not started")
        return stable.path

    @property
    def temporary_directory(self) -> str:
        stable = self._stable_port
        if stable is None:
            raise PtyDiagnosticError("PTY emulator is not started")
        return stable.directory

    @property
    def slave_path(self) -> str:
        with self._pair_lock:
            if self._slave_path is None:
                raise PtyDiagnosticError("PTY emulator is not started")
            return self._slave_path

    @property
    def service_error(self) -> BaseException | None:
        return self._service_error

    @property
    def open_fd_count(self) -> int:
        with self._pair_lock:
            return int(self._master_fd is not None)

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        if self._thread is not None or self._stable_port is not None:
            raise PtyDiagnosticError("PTY emulator instance is one-shot")
        preflight = check_pty_preflight()
        if not preflight.linux or not preflight.openpty_available:
            raise PtyDiagnosticError("Linux os.openpty() is required")

        stable = StableDiagnosticPort()
        master_fd: int | None = None
        try:
            master_fd, slave_path = self._create_pair()
            stable.repoint(slave_path)
        except (OSError, PtyDiagnosticError, ValueError):
            if master_fd is not None:
                self._close_fd(master_fd)
            stable.cleanup()
            raise
        self._stable_port = stable
        with self._pair_lock:
            self._master_fd = master_fd
            self._slave_path = slave_path
        self._thread = Thread(
            target=self._service_loop,
            name="pty-stm32-emulator",
            daemon=False,
        )
        try:
            self._thread.start()
        except RuntimeError:
            with self._pair_lock:
                self._master_fd = None
                self._slave_path = None
            self._close_fd(master_fd)
            stable.cleanup()
            self._stable_port = None
            raise
        self._record_event(f"started:{slave_path}")

    def stop(self, timeout_s: float = 2.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._stop.set()
        with self._pair_lock:
            master_fd = self._master_fd
            self._master_fd = None
            self._slave_path = None
        if master_fd is not None:
            self._close_fd(master_fd)
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        stable = self._stable_port
        self._stable_port = None
        if stable is not None:
            stable.cleanup()
        self._faults.clear()
        if thread is not None and thread.is_alive():
            raise PtyDiagnosticError("PTY service thread did not stop within deadline")

    def queue_fault(self, fault: PtyFault) -> None:
        self._faults.put(fault)

    def replace_pty(self) -> PtyReplacement:
        stable = self._stable_port
        if stable is None or self._stop.is_set():
            raise PtyDiagnosticError("PTY emulator is not running")
        new_master_fd, new_slave_path = self._create_pair()
        try:
            stable.repoint(new_slave_path)
        except (OSError, PtyDiagnosticError, ValueError):
            self._close_fd(new_master_fd)
            raise
        with self._pair_lock:
            old_master_fd = self._master_fd
            old_slave_path = self._slave_path
            self._master_fd = new_master_fd
            self._slave_path = new_slave_path
        if old_master_fd is not None:
            self._close_fd(old_master_fd)
        if old_slave_path is None:
            raise PtyDiagnosticError("old PTY pair was missing during replacement")
        with self._stats_lock:
            self._replacement_count += 1
        self._record_event(f"replaced:{old_slave_path}->{new_slave_path}")
        return PtyReplacement(stable.path, old_slave_path, new_slave_path)

    def stats(self) -> PtyEmulatorStats:
        with self._stats_lock:
            return PtyEmulatorStats(
                physical_request_count=self._request_count,
                response_count=self._response_count,
                replacement_count=self._replacement_count,
                recent_requests=tuple(self._recent_requests),
                recent_events=tuple(self._recent_events),
            )

    def endpoint_snapshot(self) -> EndpointSnapshot:
        with self._endpoint_lock:
            return EndpointSnapshot(
                request_count=len(self.endpoint.request_history),
                executed_request_count=len(self.endpoint.executed_request_history),
                baudrate=self.endpoint.baudrate,
                expected_request_id=self.endpoint.expected_request_id,
            )

    def requests_after(self, sequence: int) -> tuple[PtyRequestRecord, ...]:
        return tuple(
            record for record in self.stats().recent_requests if record.sequence > sequence
        )

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @staticmethod
    def _create_pair() -> tuple[int, str]:
        try:
            master_fd, slave_fd = os.openpty()
        except OSError as exc:
            raise PtyDiagnosticError(f"cannot create PTY pair: {exc}") from exc
        try:
            slave_path = os.ttyname(slave_fd)
            os.set_blocking(master_fd, False)
        except OSError:
            os.close(master_fd)
            raise
        finally:
            os.close(slave_fd)
        return master_fd, slave_path

    def _service_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._pair_lock:
                    master_fd = self._master_fd
                if master_fd is None:
                    self._stop.wait(_SERVICE_POLL_SECONDS)
                    continue
                try:
                    readable, _, _ = select.select(
                        (master_fd,), (), (), _SERVICE_POLL_SECONDS
                    )
                    if not readable:
                        continue
                    chunk = os.read(master_fd, _MASTER_READ_SIZE)
                except OSError as exc:
                    if exc.errno in {errno.EIO, errno.EBADF}:
                        self._stop.wait(_SERVICE_POLL_SECONDS)
                        continue
                    raise
                if not chunk:
                    self._stop.wait(_SERVICE_POLL_SECONDS)
                    continue
                for frame in self._assembler.feed(chunk):
                    self._handle_frame(master_fd, frame)
        except (OSError, PtyDiagnosticError) as exc:
            self._service_error = exc
            self._record_event(f"service-error:{type(exc).__name__}:{exc}")

    def _handle_frame(self, master_fd: int, raw_frame: bytes) -> None:
        try:
            request = decode_request(raw_frame)
        except ProtocolError:
            request = None
        with self._stats_lock:
            self._request_count += 1
            sequence = self._request_count
            self._recent_requests.append(
                PtyRequestRecord(sequence, bytes(raw_frame), request)
            )

        fault = self._faults.pop()
        with self._endpoint_lock:
            if fault is not None and fault.kind is PtyFaultKind.BAD_CRC_RESPONSE:
                self.endpoint.queue_response(FakeResponseSpec(corrupt_crc=True))
            response = self.endpoint.handle_request(raw_frame)
        if response is None:
            self._record_event(f"request-{sequence}:no-response")
            return
        if fault is not None and fault.kind is PtyFaultKind.DROP_RESPONSE:
            self._record_event(f"request-{sequence}:dropped")
            return
        if fault is not None and fault.kind is PtyFaultKind.HARD_DISCONNECT:
            replacement = self.replace_pty()
            self._record_event(
                f"request-{sequence}:hard-disconnect:{replacement.new_slave_path}"
            )
            return
        if (
            fault is not None
            and fault.kind is PtyFaultKind.DELAYED_RESPONSE
            and self._stop.wait(fault.delay_s)
        ):
            return

        prefix = b""
        if fault is not None and fault.kind is PtyFaultKind.LEADING_GARBAGE:
            prefix = fault.garbage
        elif fault is not None and fault.kind is PtyFaultKind.OVERLAPPING_START_PREFIX:
            prefix = START_BYTES[:1]

        if prefix and not self._write_all(master_fd, prefix):
            return
        if fault is not None and fault.kind is PtyFaultKind.FRAGMENTED_RESPONSE:
            offset = 0
            for size in fault.fragment_sizes:
                if offset >= len(response):
                    break
                end = min(len(response), offset + size)
                if not self._write_all(master_fd, response[offset:end]):
                    return
                offset = end
                if offset < len(response) and self._stop.wait(fault.delay_s):
                    return
            if offset < len(response) and not self._write_all(master_fd, response[offset:]):
                return
        elif not self._write_all(master_fd, response):
            return

        with self._stats_lock:
            self._response_count += 1
        label = "normal" if fault is None else fault.kind.value
        self._record_event(f"request-{sequence}:response:{label}")

    def _write_all(self, master_fd: int, payload: bytes) -> bool:
        view = memoryview(payload)
        deadline = monotonic() + _WRITE_TIMEOUT_SECONDS
        while view and not self._stop.is_set():
            try:
                written = os.write(master_fd, view)
                view = view[written:]
            except BlockingIOError:
                if monotonic() >= deadline:
                    self._record_event("response-write-timeout")
                    return False
                self._stop.wait(0.001)
            except OSError as exc:
                if exc.errno in {errno.EIO, errno.EBADF}:
                    self._record_event(f"response-write-lost:{exc.errno}")
                    return False
                raise
        return not view

    def _record_event(self, event: str) -> None:
        with self._stats_lock:
            self._recent_events.append(event)

    @staticmethod
    def _close_fd(fd: int) -> None:
        try:
            os.close(fd)
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise


__all__ = [
    "EndpointSnapshot",
    "PtyDiagnosticError",
    "PtyEmulatorStats",
    "PtyFault",
    "PtyFaultKind",
    "PtyFaultQueue",
    "PtyPreflightResult",
    "PtyReplacement",
    "PtyRequestRecord",
    "PtyStm32Emulator",
    "RequestFrameAssembler",
    "StableDiagnosticPort",
    "check_pty_preflight",
]
