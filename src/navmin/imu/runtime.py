"""Threaded passive GY-85 UDP runtime and immutable latest-state snapshots."""

from __future__ import annotations

import copy
import logging
import socket
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Protocol

import numpy as np

from navmin.concurrency import LatestValue
from navmin.lifecycle import StopToken

from .core import (
    MAX_DATAGRAM_BYTES,
    ImuCalibration,
    ImuSample,
    OrientationFilter,
    accelerometer_calibration,
    axis_matrix,
    euler_degrees,
    magnetic_calibration,
    stationary_bias,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_STALE_TIMEOUT_S = 0.5
_DEFAULT_SOCKET_TIMEOUT_S = 0.05
_MAX_GYRO_BIAS_SAMPLES = 10_000
_MAX_RECENT_SAMPLES = 512
_MAX_MAG_CALIBRATION_SAMPLES = 20_000
_CALIBRATION_COMMAND_CAPACITY = 16
_ACCEL_FACE_MIN_SAMPLES = 12
_ACCEL_FACE_MIN_SPAN_S = 0.5
_ACCEL_FACE_WINDOW_S = 1.0
_ACCEL_FACE_MAX_GAP_S = 0.15
_ACCEL_STD_LIMIT_G = 0.025
_GYRO_STD_LIMIT_DPS = 1.5


class GyroBiasState(StrEnum):
    """Runtime-only automatic gyro-bias acquisition state."""

    WAITING = "waiting"
    ACQUIRING = "acquiring"
    READY = "ready"


class ImuCalibrationCommandType(StrEnum):
    """Operator calibration intents consumed only by the IMU worker thread."""

    RESTART_GYRO_BIAS = "restart-gyro-bias"
    CAPTURE_ACCEL_FACE = "capture-accel-face"
    START_MAGNETOMETER = "start-magnetometer"
    FINISH_MAGNETOMETER = "finish-magnetometer"
    SET_RELATIVE_ZERO = "set-relative-zero"


@dataclass(frozen=True)
class ImuCalibrationCommand:
    """One bounded calibration request from UI to the IMU owner thread."""

    kind: ImuCalibrationCommandType
    face: int | None = None


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
    accel_faces_captured: tuple[int, ...] = ()
    mag_calibration_active: bool = False
    mag_calibration_sample_count: int = 0
    mag_calibration_octants: int = 0
    calibration_message: str | None = None


class _UdpSocket(Protocol):
    def bind(self, address: tuple[str, int]) -> None: ...

    def settimeout(self, timeout: float) -> None: ...

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]: ...

    def close(self) -> None: ...


SocketFactory = Callable[[int, int], _UdpSocket]
Clock = Callable[[], float]
CalibrationWriter = Callable[[ImuCalibration], None]


class ImuCalibrationControls(Protocol):
    """Typed passive calibration ingress exposed to the operator UI."""

    def request_gyro_bias_calibration(self) -> bool: ...

    def request_accelerometer_face(self, face: int) -> bool: ...

    def request_magnetometer_start(self) -> bool: ...

    def request_magnetometer_finish(self) -> bool: ...

    def request_relative_zero(self) -> bool: ...


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
        self._recent_samples: deque[ImuSample] = deque(maxlen=_MAX_RECENT_SAMPLES)
        self._accel_faces: dict[int, np.ndarray] = {}
        self._mag_calibration_samples: deque[ImuSample] = deque(
            maxlen=_MAX_MAG_CALIBRATION_SAMPLES
        )
        self._mag_calibration_active = False
        self._calibration_revision = 0
        self._calibration_message: str | None = None
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

    @property
    def calibration_revision(self) -> int:
        return self._calibration_revision

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
        self._recent_samples.append(sample)
        if self._mag_calibration_active and sample.mag is not None:
            self._mag_calibration_samples.append(sample)
        self._filter.update(sample, self._calibration)
        self._advance_gyro_bias(sample)
        return True

    def _begin_stream_session(self, peer: tuple[str, int]) -> None:
        recovered = self._last_receive_s is not None
        interrupted_calibration = bool(self._accel_faces) or self._mag_calibration_active
        self._filter = OrientationFilter()
        self._gyro_bias_samples.clear()
        self._recent_samples.clear()
        self._accel_faces.clear()
        self._mag_calibration_samples.clear()
        self._mag_calibration_active = False
        self._gyro_bias_state = GyroBiasState.ACQUIRING
        self._calibration = replace(self._calibration, gyro_ready=False)
        if not self._calibration.mag_ready:
            self._calibration = replace(
                self._calibration,
                zero=[1.0, 0.0, 0.0, 0.0],
            )
        self._last_sender_timestamp_s = None
        self._source = ImuSourceIdentity(peer[0], peer[1])
        if interrupted_calibration:
            self._calibration_message = (
                "Поток IMU перезапущен; незавершённая калибровка отменена. "
                "Не двигайте датчик 5 секунд для гироскопа."
            )
        else:
            self._calibration_message = "Не двигайте датчик 5 секунд для гироскопа."
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
            self._calibration_message = f"Гироскоп: {exc}. Сбор начат заново."
            self._gyro_bias_samples.clear()
            LOGGER.info("IMU gyro bias rejected; acquisition restarted: %s", exc)
            return
        self._calibration = replace(
            self._calibration,
            gyro_bias=bias,
            gyro_ready=True,
            zero=[1.0, 0.0, 0.0, 0.0],
        )
        self._calibration_revision += 1
        self._gyro_bias_state = GyroBiasState.READY
        self._gyro_bias_samples.clear()
        self._filter = OrientationFilter()
        self._last_error = None
        self._calibration_message = "Гироскоп: смещение рассчитано."
        LOGGER.info("IMU gyro bias acquired")

    def restart_gyro_bias_calibration(self) -> None:
        """Restart the existing five-second stationary gyro-bias acquisition."""
        self._gyro_bias_samples.clear()
        self._recent_samples.clear()
        self._gyro_bias_state = GyroBiasState.ACQUIRING
        self._calibration = replace(self._calibration, gyro_ready=False)
        self._filter = OrientationFilter()
        self._last_error = None
        self._calibration_message = "Гироскоп: не двигайте датчик 5 секунд."
        LOGGER.info("IMU gyro bias acquisition restarted by operator")

    def capture_accelerometer_face(self, face: int) -> None:
        """Capture one stable raw-sensor +/-X/Y/Z face from the recent sample window."""
        if face not in {-3, -2, -1, 1, 2, 3}:
            raise ValueError("грань акселерометра должна быть одной из ±X, ±Y, ±Z")
        samples = self._stationary_face_samples()
        values = np.array([sample.accel for sample in samples])
        mean = np.mean(values, axis=0)
        axis = abs(face) - 1
        sign = 1.0 if face > 0 else -1.0
        other = np.delete(mean, axis)
        if sign * mean[axis] < 0.75 or np.max(np.abs(other)) > 0.35:
            raise ValueError("положение не соответствует выбранной грани датчика")
        if len(self._accel_faces) == 6:
            self._accel_faces.clear()
        self._accel_faces[face] = mean
        captured = len(self._accel_faces)
        if captured < 6:
            self._calibration_message = (
                f"Акселерометр: грань {self._face_label(face)} принята ({captured}/6)."
            )
            return
        bias, scale = accelerometer_calibration(self._accel_faces)
        self._calibration = replace(
            self._calibration,
            accel_bias=bias,
            accel_scale=scale,
            accel_ready=True,
        )
        self._calibration_revision += 1
        self._filter = OrientationFilter()
        self._calibration_message = "Акселерометр: шесть граней рассчитаны."
        LOGGER.info("IMU six-face accelerometer calibration acquired")

    def _stationary_face_samples(self) -> list[ImuSample]:
        if not self._recent_samples:
            raise ValueError("нет свежих данных для измерения грани")
        latest_timestamp = self._recent_samples[-1].timestamp
        samples = [
            sample
            for sample in self._recent_samples
            if latest_timestamp - sample.timestamp <= _ACCEL_FACE_WINDOW_S
        ]
        if len(samples) < _ACCEL_FACE_MIN_SAMPLES:
            raise ValueError("подержите датчик неподвижно не менее 0,5 секунды")
        span = samples[-1].timestamp - samples[0].timestamp
        if span < _ACCEL_FACE_MIN_SPAN_S:
            raise ValueError("подержите датчик неподвижно не менее 0,5 секунды")
        if np.max(np.diff([sample.timestamp for sample in samples])) > _ACCEL_FACE_MAX_GAP_S:
            raise ValueError("есть пропуски данных; повторите измерение грани")
        accel = np.array([sample.accel for sample in samples])
        gyro = np.array([sample.gyro for sample in samples])
        if np.max(np.std(accel, axis=0)) > _ACCEL_STD_LIMIT_G:
            raise ValueError("акселерометр двигался; повторите измерение грани")
        if np.max(np.std(gyro, axis=0)) > _GYRO_STD_LIMIT_DPS:
            raise ValueError("датчик вращался; повторите измерение грани")
        mean = np.mean(accel, axis=0)
        if not 0.8 < np.linalg.norm(mean) < 1.2:
            raise ValueError("акселерометр не показывает покой около 1 g")
        return samples

    def start_magnetometer_calibration(self, *, now_s: float) -> None:
        """Start a bounded raw magnetometer sweep without discarding old calibration."""
        if self.is_stale(now_s):
            raise ValueError("нет свежих данных IMU")
        if self._mag_calibration_active:
            raise ValueError("сбор магнитометра уже запущен")
        self._mag_calibration_samples.clear()
        self._mag_calibration_active = True
        self._calibration_message = (
            "Магнитометр: вращайте датчик во всех направлениях не менее 10 секунд."
        )
        LOGGER.info("IMU magnetometer calibration capture started")

    def finish_magnetometer_calibration(self) -> None:
        """Finish the current sweep and atomically adopt it only after validation."""
        if not self._mag_calibration_active:
            raise ValueError("сбор магнитометра не запущен")
        self._mag_calibration_active = False
        samples = list(self._mag_calibration_samples)
        self._mag_calibration_samples.clear()
        bias, scale, radius = magnetic_calibration(samples)
        self._calibration = replace(
            self._calibration,
            mag_bias=bias,
            mag_scale=scale,
            mag_radius=radius,
            mag_ready=True,
        )
        self._calibration_revision += 1
        self._filter = OrientationFilter()
        self._calibration_message = "Магнитометр: калибровка рассчитана."
        LOGGER.info("IMU magnetometer calibration acquired")

    def set_relative_zero(self, *, now_s: float) -> None:
        """Persist the current valid orientation as the operator relative zero."""
        if self.is_stale(now_s) or self._last_sample is None:
            raise ValueError("нет свежих данных IMU")
        if self._gyro_bias_state is not GyroBiasState.READY:
            raise ValueError("сначала дождитесь калибровки гироскопа")
        zero = np.asarray(self._filter.q, dtype=float)
        norm = float(np.linalg.norm(zero))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("текущая ориентация недействительна")
        zero = zero / norm
        self._calibration = replace(self._calibration, zero=zero.tolist())
        self._calibration_revision += 1
        self._calibration_message = "Текущая ориентация принята как относительный ноль."
        LOGGER.info("IMU relative zero updated")

    def record_calibration_error(self, error: str) -> None:
        self._calibration_message = f"Калибровка: {error}"

    def record_calibration_persisted(self) -> None:
        message = self._calibration_message or "Калибровка IMU обновлена."
        if not message.endswith(" "):
            message += " "
        self._calibration_message = message + "Файл калибровки сохранён."

    def record_calibration_persistence_unavailable(self) -> None:
        message = self._calibration_message or "Калибровка IMU обновлена."
        if not message.endswith(" "):
            message += " "
        self._calibration_message = message + "Сохранение файла не настроено."

    def record_calibration_persistence_error(self, error: str) -> None:
        message = self._calibration_message or "Калибровка IMU обновлена."
        if not message.endswith(" "):
            message += " "
        self._calibration_message = message + f"Ошибка сохранения: {error}"

    @staticmethod
    def _face_label(face: int) -> str:
        axis = "XYZ"[abs(face) - 1]
        sign = "+" if face > 0 else "−"
        return f"{axis}{sign}"

    def _mag_calibration_octant_count(self) -> int:
        values = [
            sample.mag
            for sample in self._mag_calibration_samples
            if sample.mag is not None
        ]
        if len(values) < 2:
            return 0
        array = np.asarray(values, dtype=float)
        center = (np.min(array, axis=0) + np.max(array, axis=0)) / 2
        return len({tuple(row > center) for row in array})

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
            accel_faces_captured=tuple(sorted(self._accel_faces)),
            mag_calibration_active=self._mag_calibration_active,
            mag_calibration_sample_count=len(self._mag_calibration_samples),
            mag_calibration_octants=(
                self._mag_calibration_octant_count()
                if self._mag_calibration_active
                else 0
            ),
            calibration_message=self._calibration_message,
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
        self._calibration_commands: Queue[ImuCalibrationCommand] = Queue(
            maxsize=_CALIBRATION_COMMAND_CAPACITY
        )
        self._calibration_writer: CalibrationWriter | None = None
        self._last_persist_attempt_revision = self._stream.calibration_revision
        self._stop_token = StopToken()
        self._lifecycle_lock = Lock()
        self._socket: _UdpSocket | None = None
        self._thread = Thread(target=self._run, name=thread_name, daemon=False)
        self._started = False

    @property
    def calibration(self) -> ImuCalibration:
        return self._stream.calibration

    def set_calibration_writer(self, writer: CalibrationWriter) -> None:
        """Install launcher-owned persistence before starting the worker."""
        if not callable(writer):
            raise TypeError("writer must be callable")
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("IMU calibration writer must be set before start")
            self._calibration_writer = writer

    def request_gyro_bias_calibration(self) -> bool:
        return self._submit_calibration_command(
            ImuCalibrationCommand(ImuCalibrationCommandType.RESTART_GYRO_BIAS)
        )

    def request_accelerometer_face(self, face: int) -> bool:
        if face not in {-3, -2, -1, 1, 2, 3}:
            raise ValueError("face must be one of ±1, ±2, ±3")
        return self._submit_calibration_command(
            ImuCalibrationCommand(ImuCalibrationCommandType.CAPTURE_ACCEL_FACE, face)
        )

    def request_magnetometer_start(self) -> bool:
        return self._submit_calibration_command(
            ImuCalibrationCommand(ImuCalibrationCommandType.START_MAGNETOMETER)
        )

    def request_magnetometer_finish(self) -> bool:
        return self._submit_calibration_command(
            ImuCalibrationCommand(ImuCalibrationCommandType.FINISH_MAGNETOMETER)
        )

    def request_relative_zero(self) -> bool:
        return self._submit_calibration_command(
            ImuCalibrationCommand(ImuCalibrationCommandType.SET_RELATIVE_ZERO)
        )

    def _submit_calibration_command(self, command: ImuCalibrationCommand) -> bool:
        try:
            self._calibration_commands.put_nowait(command)
        except Full:
            LOGGER.warning("IMU calibration command queue is full")
            return False
        return True

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

    def _process_calibration_commands(self, *, now_s: float) -> bool:
        processed = False
        while True:
            try:
                command = self._calibration_commands.get_nowait()
            except Empty:
                break
            processed = True
            try:
                if command.kind is ImuCalibrationCommandType.RESTART_GYRO_BIAS:
                    self._stream.restart_gyro_bias_calibration()
                elif command.kind is ImuCalibrationCommandType.CAPTURE_ACCEL_FACE:
                    if self._stream.is_stale(now_s):
                        raise ValueError("нет свежих данных IMU")
                    assert command.face is not None
                    self._stream.capture_accelerometer_face(command.face)
                elif command.kind is ImuCalibrationCommandType.START_MAGNETOMETER:
                    self._stream.start_magnetometer_calibration(now_s=now_s)
                elif command.kind is ImuCalibrationCommandType.FINISH_MAGNETOMETER:
                    self._stream.finish_magnetometer_calibration()
                elif command.kind is ImuCalibrationCommandType.SET_RELATIVE_ZERO:
                    self._stream.set_relative_zero(now_s=now_s)
                else:
                    raise AssertionError("unknown IMU calibration command")
            except ValueError as exc:
                self._stream.record_calibration_error(str(exc))
                LOGGER.info("IMU calibration command rejected: %s", exc)
            self._persist_calibration_if_changed()
        return processed

    def _persist_calibration_if_changed(self) -> None:
        revision = self._stream.calibration_revision
        if revision <= self._last_persist_attempt_revision:
            return
        self._last_persist_attempt_revision = revision
        writer = self._calibration_writer
        if writer is None:
            self._stream.record_calibration_persistence_unavailable()
            LOGGER.warning("IMU calibration changed but no persistence writer is configured")
            return
        try:
            writer(self._stream.calibration)
        except (OSError, ValueError, TypeError) as exc:
            self._stream.record_calibration_persistence_error(str(exc))
            LOGGER.error("IMU calibration persistence failed: %s", exc)
            return
        self._stream.record_calibration_persisted()
        LOGGER.info("IMU calibration persisted")

    def _run(self) -> None:
        assert self._socket is not None
        receiver = self._socket
        try:
            while not self._stop_token.is_stop_requested():
                now = self._clock()
                if self._process_calibration_commands(now_s=now):
                    self.state_updates.publish(
                        self._stream.snapshot(running=True, now_s=now)
                    )
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
                self._persist_calibration_if_changed()
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
    "ImuCalibrationCommand",
    "ImuCalibrationCommandType",
    "ImuCalibrationControls",
    "ImuRuntime",
    "ImuRuntimeState",
    "ImuSourceIdentity",
    "ImuStreamState",
]
