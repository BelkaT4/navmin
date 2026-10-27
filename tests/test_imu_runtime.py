from __future__ import annotations

import json
from queue import Empty, Queue
from threading import Lock
from time import monotonic, sleep

import numpy as np
import pytest

from navmin.imu import GyroBiasState, ImuCalibration, ImuRuntime, ImuStreamState


def packet(
    timestamp: float,
    *,
    accel: list[float] | None = None,
    gyro: list[float] | None = None,
    mag: list[float] | None = None,
) -> bytes:
    return json.dumps(
        {
            "type": "navmin-gy85",
            "version": 1,
            "t": timestamp,
            "accel_g": [0.0, 0.0, 1.0] if accel is None else accel,
            "gyro_dps": [1.0, -19.0, -18.0] if gyro is None else gyro,
            "mag_raw": mag,
        }
    ).encode()


def test_stream_rejects_duplicate_out_of_order_and_foreign_source() -> None:
    stream = ImuStreamState(expected_source="192.0.2.10")
    assert stream.accept_datagram(packet(1.0), ("192.0.2.10", 5000), received_at_s=10.0)
    before = stream.snapshot(running=True, now_s=10.0)

    assert not stream.accept_datagram(packet(1.0), ("192.0.2.10", 5000), received_at_s=10.1)
    assert not stream.accept_datagram(packet(0.9), ("192.0.2.10", 5000), received_at_s=10.2)
    assert not stream.accept_datagram(packet(2.0), ("192.0.2.11", 5000), received_at_s=10.3)
    assert not stream.accept_datagram(packet(2.0), ("192.0.2.10", 5001), received_at_s=10.3)

    after = stream.snapshot(running=True, now_s=10.3)
    assert after.sender_timestamp_s == before.sender_timestamp_s == 1.0
    assert after.duplicate_packets == 1
    assert after.out_of_order_packets == 1
    assert after.foreign_packets == 2


def test_stream_stale_recovery_accepts_new_sender_timestamp_and_restarts_bias() -> None:
    stream = ImuStreamState()
    for index in range(251):
        timestamp = index / 50
        assert stream.accept_datagram(
            packet(timestamp),
            ("192.0.2.10", 5000),
            received_at_s=20.0 + timestamp,
        )
    ready = stream.snapshot(running=True, now_s=25.0)
    assert ready.gyro_bias_state is GyroBiasState.READY
    assert ready.orientation_valid
    assert ready.gyro_bias_dps == pytest.approx((1.0, -19.0, -18.0))

    assert stream.refresh_stale(25.6)
    stale = stream.snapshot(running=True, now_s=25.6)
    assert not stale.fresh
    assert stale.stale
    assert stale.gyro_bias_state is GyroBiasState.READY

    assert stream.accept_datagram(
        packet(0.1),
        ("192.0.2.20", 6000),
        received_at_s=25.7,
    )
    recovered = stream.snapshot(running=True, now_s=25.7)
    assert recovered.fresh
    assert not recovered.stale
    assert not recovered.orientation_valid
    assert recovered.source is not None
    assert recovered.source.host == "192.0.2.20"
    assert recovered.sender_timestamp_s == 0.1
    assert recovered.gyro_bias_state is GyroBiasState.ACQUIRING
    assert recovered.gyro_bias_sample_count == 1
    assert not stream.calibration.gyro_ready


def test_unstable_automatic_bias_is_rejected_and_restarts_bounded_acquisition() -> None:
    stream = ImuStreamState()
    for index in range(251):
        timestamp = index / 50
        gyro = [1.0, -19.0 + (10.0 if index % 2 else 0.0), -18.0]
        assert stream.accept_datagram(
            packet(timestamp, gyro=gyro),
            ("192.0.2.10", 5000),
            received_at_s=30.0 + timestamp,
        )
    state = stream.snapshot(running=True, now_s=35.0)
    assert state.gyro_bias_state is GyroBiasState.ACQUIRING
    assert state.gyro_bias_sample_count == 0
    assert not stream.calibration.gyro_ready
    assert state.last_error is not None
    assert "gyro bias rejected" in state.last_error

    for index in range(1, 252):
        timestamp = 5.0 + index / 50
        assert stream.accept_datagram(
            packet(timestamp),
            ("192.0.2.10", 5000),
            received_at_s=35.0 + index / 50,
        )
    recovered = stream.snapshot(running=True, now_s=40.1)
    assert recovered.gyro_bias_state is GyroBiasState.READY
    assert recovered.gyro_bias_sample_count == 0
    assert recovered.last_error is None


def test_snapshot_is_immutable_and_exposes_only_valid_corrected_values() -> None:
    calibration = ImuCalibration(axes=[2, -1, 3])
    stream = ImuStreamState(calibration=calibration)
    for index in range(251):
        timestamp = index / 50
        assert stream.accept_datagram(
            packet(timestamp),
            ("192.0.2.10", 5000),
            received_at_s=1.0 + timestamp,
        )
    assert stream.accept_datagram(
        packet(5.02, gyro=[3.0, -17.0, -16.0]),
        ("192.0.2.10", 5000),
        received_at_s=6.02,
    )
    state = stream.snapshot(running=True, now_s=6.02)
    assert state.gyro_raw_dps == (3.0, -17.0, -16.0)
    assert state.gyro_corrected_dps == pytest.approx((2.0, -2.0, 2.0))
    assert isinstance(state.orientation_quaternion, tuple)
    assert isinstance(state.mounting_axes, tuple)

    copy_of_calibration = stream.calibration
    copy_of_calibration.gyro_bias[0] = 999
    assert stream.calibration.gyro_bias[0] != 999


class FakeSocket:
    def __init__(self) -> None:
        self.bound: tuple[str, int] | None = None
        self.timeout = 0.05
        self.closed = False
        self._queue: Queue[tuple[bytes, tuple[str, int]]] = Queue()

    def bind(self, address: tuple[str, int]) -> None:
        self.bound = address

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        try:
            return self._queue.get(timeout=self.timeout)
        except Empty as exc:
            raise TimeoutError from exc

    def close(self) -> None:
        self.closed = True

    def push(self, data: bytes, peer: tuple[str, int]) -> None:
        self._queue.put((data, peer))


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self._value = value
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value


def test_runtime_owns_dedicated_socket_thread_and_publishes_stale_transition() -> None:
    fake_socket = FakeSocket()
    clock = FakeClock()
    runtime = ImuRuntime(
        bind_address="127.0.0.1",
        port=8891,
        socket_factory=lambda _family, _kind: fake_socket,
        clock=clock,
        socket_timeout_s=0.01,
    )
    runtime.start()
    initial = runtime.state_updates.snapshot()
    assert initial.value is not None
    assert initial.value.running
    assert not initial.value.fresh
    assert initial.value.stale
    assert not initial.value.orientation_valid
    assert fake_socket.bound == ("127.0.0.1", 8891)
    assert runtime.is_alive()

    fake_socket.push(packet(1.0), ("192.0.2.10", 5000))
    accepted = runtime.state_updates.wait_for_revision(initial.revision, timeout=1.0)
    assert accepted is not None and accepted.value is not None
    assert accepted.value.fresh
    assert accepted.value.source is not None

    clock.set(100.6)
    stale = runtime.state_updates.wait_for_revision(accepted.revision, timeout=1.0)
    assert stale is not None and stale.value is not None
    assert not stale.value.fresh

    assert runtime.stop(timeout=1.0)
    stopped = runtime.state_updates.snapshot().value
    assert stopped is not None and not stopped.running
    assert stopped.stale and not stopped.fresh
    assert fake_socket.closed
    assert not runtime.is_alive()


def test_runtime_shutdown_matches_application_owned_worker_contract() -> None:
    fake_socket = FakeSocket()
    runtime = ImuRuntime(
        bind_address="127.0.0.1",
        port=8891,
        socket_factory=lambda _family, _kind: fake_socket,
        socket_timeout_s=0.01,
    )

    runtime.start()
    runtime.shutdown(timeout=1.0)

    assert fake_socket.closed
    assert not runtime.is_alive()
    state = runtime.state_updates.snapshot().value
    assert state is not None
    assert not state.running


def test_runtime_bind_failure_is_synchronous_and_does_not_start_thread() -> None:
    class BrokenSocket(FakeSocket):
        def bind(self, address: tuple[str, int]) -> None:
            raise OSError("address already in use")

    fake_socket = BrokenSocket()
    runtime = ImuRuntime(
        bind_address="0.0.0.0",
        port=8891,
        socket_factory=lambda _family, _kind: fake_socket,
    )
    with pytest.raises(OSError, match="address already in use"):
        runtime.start()
    assert fake_socket.closed
    assert not runtime.is_alive()
    assert runtime.state_updates.get() is None


def _feed_face(
    stream: ImuStreamState,
    *,
    start_timestamp: float,
    accel: list[float],
    samples: int = 61,
) -> float:
    for index in range(samples):
        timestamp = start_timestamp + index / 50
        assert stream.accept_datagram(
            packet(timestamp, accel=accel, gyro=[0.0, 0.0, 0.0]),
            ("192.0.2.10", 5000),
            received_at_s=100.0 + timestamp,
        )
    return start_timestamp + samples / 50


def test_operator_six_face_accelerometer_calibration_uses_stable_raw_faces() -> None:
    stream = ImuStreamState()
    bias = np.array([0.04, -0.03, 0.07])
    raw_amplitude = np.array([1.04, 0.98, 1.08])
    timestamp = 0.0

    for face in (1, -1, 2, -2, 3, -3):
        axis = abs(face) - 1
        direction = np.zeros(3)
        direction[axis] = 1.0 if face > 0 else -1.0
        raw = bias + direction * raw_amplitude
        timestamp = _feed_face(
            stream,
            start_timestamp=timestamp,
            accel=raw.tolist(),
        )
        stream.capture_accelerometer_face(face)

    calibration = stream.calibration
    state = stream.snapshot(running=True, now_s=100.0 + timestamp)
    assert calibration.accel_ready
    assert calibration.accel_bias == pytest.approx(bias)
    assert calibration.accel_scale == pytest.approx(1 / raw_amplitude)
    assert state.accel_faces_captured == (-3, -2, -1, 1, 2, 3)
    assert state.calibration_message is not None
    assert "шесть граней" in state.calibration_message


def test_accelerometer_face_rejects_wrong_orientation_without_losing_progress() -> None:
    stream = ImuStreamState()
    _feed_face(
        stream,
        start_timestamp=0.0,
        accel=[1.0, 0.0, 0.0],
    )
    stream.capture_accelerometer_face(1)

    with pytest.raises(ValueError, match="не соответствует"):
        stream.capture_accelerometer_face(-1)

    state = stream.snapshot(running=True, now_s=101.3)
    assert state.accel_faces_captured == (1,)
    assert not state.accel_calibrated


def _magnetic_vector(index: int, count: int = 600) -> list[float]:
    z = 1 - 2 * (index + 0.5) / count
    phi = index * np.pi * (3 - np.sqrt(5))
    direction = np.array(
        [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    )
    return (np.array([100.0, -200.0, 300.0]) + direction * [600, 900, 450]).tolist()


def test_operator_magnetometer_capture_adopts_only_valid_completed_sweep() -> None:
    previous = ImuCalibration(
        mag_bias=[11.0, 22.0, 33.0],
        mag_scale=[1.1, 0.9, 1.0],
        mag_radius=500.0,
        mag_ready=True,
    )
    stream = ImuStreamState(calibration=previous)
    assert stream.accept_datagram(
        packet(0.0, mag=_magnetic_vector(0)),
        ("192.0.2.10", 5000),
        received_at_s=50.0,
    )
    stream.start_magnetometer_calibration(now_s=50.0)
    assert stream.calibration.mag_bias == previous.mag_bias

    for index in range(1, 601):
        timestamp = index / 50
        assert stream.accept_datagram(
            packet(timestamp, mag=_magnetic_vector(index - 1)),
            ("192.0.2.10", 5000),
            received_at_s=50.0 + timestamp,
        )

    collecting = stream.snapshot(running=True, now_s=62.0)
    assert collecting.mag_calibration_active
    assert collecting.mag_calibration_sample_count == 600
    assert collecting.mag_calibration_octants == 8
    stream.finish_magnetometer_calibration()

    calibration = stream.calibration
    state = stream.snapshot(running=True, now_s=62.0)
    assert calibration.mag_ready
    assert calibration.mag_bias == pytest.approx([100, -200, 300], abs=4)
    assert calibration.mag_radius > 0
    assert not state.mag_calibration_active
    assert state.mag_calibration_sample_count == 0

    previous_valid = stream.calibration
    stream.start_magnetometer_calibration(now_s=62.0)
    for index in range(1, 40):
        timestamp = 12.0 + index / 50
        assert stream.accept_datagram(
            packet(timestamp, mag=[500.0, 0.0, 0.0]),
            ("192.0.2.10", 5000),
            received_at_s=50.0 + timestamp,
        )
    with pytest.raises(ValueError, match="10 секунд"):
        stream.finish_magnetometer_calibration()
    assert stream.calibration == previous_valid


def test_operator_relative_zero_uses_current_valid_orientation() -> None:
    stream = ImuStreamState()
    for index in range(251):
        timestamp = index / 50
        assert stream.accept_datagram(
            packet(timestamp, gyro=[0.0, 0.0, 0.0]),
            ("192.0.2.10", 5000),
            received_at_s=20.0 + timestamp,
        )
    assert stream.snapshot(running=True, now_s=25.0).orientation_valid

    timestamp = 5.02
    assert stream.accept_datagram(
        packet(
            timestamp,
            accel=[0.3, 0.2, 0.9327379053],
            gyro=[0.0, 0.0, 0.0],
        ),
        ("192.0.2.10", 5000),
        received_at_s=25.02,
    )
    before = stream.snapshot(running=True, now_s=25.02)
    assert max(abs(value) for value in before.euler_degrees[:2]) > 5.0

    stream.set_relative_zero(now_s=25.02)
    after = stream.snapshot(running=True, now_s=25.02)
    assert after.euler_degrees == pytest.approx((0.0, 0.0, 0.0), abs=1e-8)
    assert stream.calibration.zero == pytest.approx(before.orientation_quaternion)


def _wait_for_runtime_state(runtime: ImuRuntime, predicate, timeout: float = 2.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = runtime.state_updates.get()
        if state is not None and predicate(state):
            return state
        sleep(0.005)
    raise AssertionError("timed out waiting for IMU runtime state")


def test_runtime_owns_calibration_commands_and_persistence_callback() -> None:
    fake_socket = FakeSocket()
    persisted: list[ImuCalibration] = []
    runtime = ImuRuntime(
        bind_address="127.0.0.1",
        port=8891,
        socket_factory=lambda _family, _kind: fake_socket,
        socket_timeout_s=0.005,
    )
    runtime.set_calibration_writer(lambda calibration: persisted.append(calibration))
    runtime.start()
    try:
        for index in range(251):
            fake_socket.push(
                packet(index / 50, gyro=[0.0, 0.0, 0.0]),
                ("192.0.2.10", 5000),
            )
        ready = _wait_for_runtime_state(
            runtime,
            lambda state: state.gyro_bias_state is GyroBiasState.READY,
        )
        assert ready.orientation_valid
        assert persisted
        assert persisted[-1].gyro_ready

        before_count = len(persisted)
        assert runtime.request_relative_zero()
        zeroed = _wait_for_runtime_state(
            runtime,
            lambda state: (
                state.calibration_message is not None
                and "относительный ноль" in state.calibration_message
                and "сохранён" in state.calibration_message
            ),
        )
        assert zeroed.orientation_valid
        assert len(persisted) == before_count + 1
    finally:
        runtime.shutdown(timeout=1.0)
