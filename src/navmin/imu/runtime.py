"""Threaded passive GY-85 UDP runtime and immutable latest-state snapshots."""

from __future__ import annotations

import copy
import logging
import socket
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import Lock, Thread
from time import monotonic
from typing import Protocol

from navmin.concurrency import LatestValue
from navmin.lifecycle import StopToken

from .core import (
    MAX_DATAGRAM_BYTES,
    ImuCalibration,
    ImuSample,
    OrientationFilter,
    axis_matrix,
    euler_degrees,
    stationary_bias,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_STALE_TIMEOUT_S = 0.5
_DEFAULT_SOCKET_TIMEOUT_S = 0.05
_MAX_GYRO_BIAS_SAMPLES = 10_000


class GyroBiasState(StrEnum):
    """Runtime-only automatic gyro-bias acquisition state."""

    WAITING = "waiting"
    ACQUIRING = "acquiring"
    READY = "ready"


@dataclass(frozen=True)
class ImuSourceIdentity:
    """One UDP sender endpoint observed by the receiver."""

    host: str
    port: int


@dataclass(frozen=True)
class ImuRuntimeState:
    """Immutable latest-state view for UI and later hardware observability."""

    running: bool
    fresh: bool
    stale: bool
    source: ImuSourceIdentity | None
    sender_timestamp_s: float | None
    receive_timestamp_s: float | None
    accel_g: tuple[float, float, float] | None
    gyro_raw_dps: tuple[float, float, float] | None
    gyro_corrected_dps: tuple[float, float, float] | None
    mag_raw: tuple[float, float, float] | None
    mag_corrected: tuple[float, float, float] | None
    gyro_bias_state: GyroBiasState
    gyro_bias_dps: tuple[float, float, float]
    gyro_bias_sample_count: int
    accel_calibrated: bool
    mag_calibrated: bool
    orientation_quaternion: tuple[float, float, float, float]
    relative_quaternion: tuple[float, float, float, float]
    euler_degrees: tuple[float, float, float]
    orientation_valid: bool
    heading_valid: bool
    mounting_axes: tuple[int, int, int]
    relative_zero: tuple[float, float, float, float]
    invalid_packets: int
    duplicate_packets: int
    out_of_order_packets: int
    foreign_packets: int
    last_error: str | None


class _UdpSocket(Protocol):
    def bind(self, address: tuple[str, int]) -> None: ...

    def settimeout(self, timeout: float) -> None: ...

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]: ...

    def close(self) -> None: ...


SocketFactory = Callable[[int, int], _UdpSocket]
Clock = Callable[[], float]


class ImuStreamState:
    """Single-thread-owned protocol/order/stale/orientation state machine."""

    def __init__(
        self,
        *,
        calibration: ImuCalibration | None = None,
        expected_source: str | None = None,
        stale_timeout_s: float = _DEFAULT_STALE_TIMEOUT_S,
    ) -> None:
        if expected_source is not None and not expected_source.strip():
            raise ValueError("expected_source must be non-empty when configured")
        if stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be > 0")
        self._calibration = copy.deepcopy(calibration or ImuCalibration())
        self._calibration.validate()
        self._expected_source = (
            None if expected_source is None else expected_source.strip()
        )
        self._stale_timeout_s = float(stale_timeout_s)
        self._filter = OrientationFilter()
        self._source: ImuSourceIdentity | None = None
        self._last_sample: ImuSample | None = None
        self._last_receive_s: float | None = None
        self._last_sender_timestamp_s: float | None = None
        self._stale_announced = False
        self._gyro_bias_state = GyroBiasState.WAITING
        self._gyro_bias_samples: deque[ImuSample] = deque(maxlen=_MAX_GYRO_BIAS_SAMPLES)
        self._last_error: str | None = None
        self.invalid_packets = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.foreign_packets = 0

    @property
    def stale_timeout_s(self) -> float:
        return self._stale_timeout_s

    @property
    def calibration(self) -> ImuCalibration:
        """Return a defensive copy; callers cannot mutate live runtime state."""
        return copy.deepcopy(self._calibration)

    def is_stale(self, now_s: float) -> bool:
        return (
            self._last_receive_s is None
            or now_s - self._last_receive_s > self._stale_timeout_s
        )

    def refresh_stale(self, now_s: float) -> bool:
        """Return True only on a fresh->stale transition."""
        if self._last_receive_s is None or not self.is_stale(now_s):
            return False
        if self._stale_announced:
            return False
        self._stale_announced = True
        LOGGER.info("IMU stream stale")
        return True

    def accept_datagram(
        self,
        data: bytes,
        peer: tuple[str, int],
        *,
        received_at_s: float,
    ) -> bool:
        """Validate one datagram and update state without damaging the last valid sample."""
        if self._expected_source is not None and peer[0] != self._expected_source:
            self.foreign_packets += 1
            return False
        try:
            sample = ImuSample.decode(data)
        except (ValueError, TypeError, UnicodeError, OverflowError):
            self.invalid_packets += 1
            return False

        restarted = self._last_receive_s is None or self.is_stale(received_at_s)
        if not restarted:
            if self._source is not None and (peer[0], peer[1]) != (
                self._source.host,
                self._source.port,
            ):
                self.foreign_packets += 1
                return False
            assert self._last_sender_timestamp_s is not None
            if sample.timestamp == self._last_sender_timestamp_s:
                self.duplicate_packets += 1
                return False
            if sample.timestamp < self._last_sender_timestamp_s:
                self.out_of_order_packets += 1
                return False

        if restarted:
            self._begin_stream_session(peer)
        self._source = ImuSourceIdentity(peer[0], peer[1])
        self._last_sample = sample
        self._last_receive_s = received_at_s
        self._last_sender_timestamp_s = sample.timestamp
        self._stale_announced = False
        self._filter.update(sample, self._calibration)
        self._advance_gyro_bias(sample)
        return True

    def _begin_stream_session(self, peer: tuple[str, int]) -> None:
        recovered = self._last_receive_s is not None
        self._filter = OrientationFilter()
        self._gyro_bias_samples.clear()
        self._gyro_bias_state = GyroBiasState.ACQUIRING
        self._calibration = replace(self._calibration, gyro_ready=False)
        if not self._calibration.mag_ready:
            self._calibration = replace(
                self._calibration,
                zero=[1.0, 0.0, 0.0, 0.0],
            )
        self._last_sender_timestamp_s = None
        self._source = ImuSourceIdentity(peer[0], peer[1])
        if recovered:
            LOGGER.info("IMU stream recovered; gyro bias acquisition restarted")
        else:
            LOGGER.info("IMU source accepted; gyro bias acquisition started")

    def _advance_gyro_bias(self, sample: ImuSample) -> None:
        if self._gyro_bias_state is not GyroBiasState.ACQUIRING:
            return
        self._gyro_bias_samples.append(sample)
        if len(self._gyro_bias_samples) < 2:
            return
        elapsed = self._gyro_bias_samples[-1].timestamp - self._gyro_bias_samples[0].timestamp
        if elapsed < 5.0:
            return
        try:
            bias = stationary_bias(list(self._gyro_bias_samples))
        except ValueError as exc:
            self._last_error = f"gyro bias rejected: {exc}"
            self._gyro_bias_samples.clear()
            LOGGER.info("IMU gyro bias rejected; acquisition restarted: %s", exc)
            return
        self._calibration = replace(
            self._calibration,
            gyro_bias=bias,
            gyro_ready=True,
            zero=[1.0, 0.0, 0.0, 0.0],
        )
        self._gyro_bias_state = GyroBiasState.READY
        self._gyro_bias_samples.clear()
        self._filter = OrientationFilter()
        self._last_error = None
        LOGGER.info("IMU gyro bias acquired")

    def record_runtime_error(self, error: str) -> None:
        """Expose one expected receiver/runtime fault without discarding valid sensor state."""
        self._last_error = error

    def snapshot(self, *, running: bool, now_s: float) -> ImuRuntimeState:
        sample = self._last_sample
        fresh = running and sample is not None and not self.is_stale(now_s)
        stale = not fresh
        mapping = axis_matrix(self._calibration.axes)
        gyro_corrected: tuple[float, float, float] | None = None
        mag_corrected: tuple[float, float, float] | None = None
        accel: tuple[float, float, float] | None = None
        gyro: tuple[float, float, float] | None = None
        mag: tuple[float, float, float] | None = None
        if sample is not None:
            accel = tuple(float(value) for value in sample.accel)
            gyro = tuple(float(value) for value in sample.gyro)
            if self._calibration.gyro_ready:
                corrected_gyro = mapping @ (
                    sample.gyro - self._calibration.gyro_bias
                )
                gyro_corrected = tuple(float(value) for value in corrected_gyro)
            if sample.mag is not None:
                mag = tuple(float(value) for value in sample.mag)
                if self._calibration.mag_ready:
                    corrected_mag = mapping @ (
                        (sample.mag - self._calibration.mag_bias)
                        * self._calibration.mag_scale
                    )
                    mag_corrected = tuple(float(value) for value in corrected_mag)
        relative = self._filter.relative(self._calibration)
        quaternion = tuple(float(value) for value in self._filter.q)
        relative_quaternion = tuple(float(value) for value in relative)
        euler = tuple(float(value) for value in euler_degrees(relative))
        return ImuRuntimeState(
            running=running,
            fresh=fresh,
            stale=stale,
            source=self._source,
            sender_timestamp_s=None if sample is None else sample.timestamp,
            receive_timestamp_s=self._last_receive_s,
            accel_g=accel,
            gyro_raw_dps=gyro,
            gyro_corrected_dps=gyro_corrected,
            mag_raw=mag,
            mag_corrected=mag_corrected,
            gyro_bias_state=self._gyro_bias_state,
            gyro_bias_dps=tuple(float(value) for value in self._calibration.gyro_bias),
            gyro_bias_sample_count=len(self._gyro_bias_samples),
            accel_calibrated=self._calibration.accel_ready,
            mag_calibrated=self._calibration.mag_ready,
            orientation_quaternion=quaternion,
            relative_quaternion=relative_quaternion,
            euler_degrees=euler,
            orientation_valid=fresh and self._gyro_bias_state is GyroBiasState.READY,
            heading_valid=self._filter.heading_valid,
            mounting_axes=tuple(self._calibration.axes),
            relative_zero=tuple(float(value) for value in self._calibration.zero),
            invalid_packets=self.invalid_packets,
            duplicate_packets=self.duplicate_packets,
            out_of_order_packets=self.out_of_order_packets,
            foreign_packets=self.foreign_packets,
            last_error=self._last_error,
        )


class ImuRuntime:
    """Dedicated bounded UDP worker intended to be owned by ApplicationRuntime."""

    def __init__(
        self,
        *,
        bind_address: str,
        port: int,
        expected_source: str | None = None,
        calibration: ImuCalibration | None = None,
        stale_timeout_s: float = _DEFAULT_STALE_TIMEOUT_S,
        socket_timeout_s: float = _DEFAULT_SOCKET_TIMEOUT_S,
        socket_factory: SocketFactory = socket.socket,
        clock: Clock = monotonic,
        thread_name: str = "imu-gy85",
    ) -> None:
        if not bind_address:
            raise ValueError("bind_address must be non-empty")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be in range 0..65535")
        if socket_timeout_s <= 0:
            raise ValueError("socket_timeout_s must be > 0")
        self.bind_address = bind_address
        self.port = port
        self._socket_timeout_s = socket_timeout_s
        self._socket_factory = socket_factory
        self._clock = clock
        self._stream = ImuStreamState(
            calibration=calibration,
            expected_source=expected_source,
            stale_timeout_s=stale_timeout_s,
        )
        self.state_updates: LatestValue[ImuRuntimeState] = LatestValue()
        self._stop_token = StopToken()
        self._lifecycle_lock = Lock()
        self._socket: _UdpSocket | None = None
        self._thread = Thread(target=self._run, name=thread_name, daemon=False)
        self._started = False

    @property
    def calibration(self) -> ImuCalibration:
        return self._stream.calibration

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("IMU runtime already started")
            receiver = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                receiver.bind((self.bind_address, self.port))
                receiver.settimeout(self._socket_timeout_s)
            except OSError:
                receiver.close()
                raise
            self._socket = receiver
            self._started = True
            self.state_updates.publish(
                self._stream.snapshot(running=True, now_s=self._clock())
            )
            self._thread.start()
        LOGGER.info(
            "IMU receiver started bind=%s port=%d",
            self.bind_address,
            self.port,
        )

    def request_stop(self) -> None:
        self._stop_token.request_stop()

    def join(self, timeout: float | None = None) -> None:
        if self._started:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> bool:
        self.request_stop()
        self.join(timeout)
        return not self.is_alive()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the receiver within the shared ApplicationRuntime boundary."""
        if not self.stop(timeout):
            raise RuntimeError(
                f"IMU receiver did not stop within {timeout:.3f} seconds"
            )

    def _run(self) -> None:
        assert self._socket is not None
        receiver = self._socket
        try:
            while not self._stop_token.is_stop_requested():
                try:
                    data, peer = receiver.recvfrom(MAX_DATAGRAM_BYTES + 1)
                except TimeoutError:
                    now = self._clock()
                    if self._stream.refresh_stale(now):
                        self.state_updates.publish(
                            self._stream.snapshot(running=True, now_s=now)
                        )
                    continue
                except OSError as exc:
                    if self._stop_token.is_stop_requested():
                        break
                    message = f"UDP receive failed: {exc}"
                    self._stream.record_runtime_error(message)
                    self.state_updates.publish(
                        self._stream.snapshot(running=True, now_s=self._clock())
                    )
                    LOGGER.error("IMU %s", message)
                    break
                now = self._clock()
                self._stream.accept_datagram(data, peer, received_at_s=now)
                self.state_updates.publish(
                    self._stream.snapshot(running=True, now_s=now)
                )
        finally:
            receiver.close()
            with self._lifecycle_lock:
                self._socket = None
            self.state_updates.publish(
                self._stream.snapshot(running=False, now_s=self._clock())
            )
            LOGGER.info("IMU receiver stopped")


__all__ = [
    "GyroBiasState",
    "ImuRuntime",
    "ImuRuntimeState",
    "ImuSourceIdentity",
    "ImuStreamState",
]
