from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from navmin.imu import (
    MAX_DATAGRAM_BYTES,
    ImuCalibration,
    ImuSample,
    OrientationFilter,
    accelerometer_calibration,
    axis_matrix,
    euler_degrees,
    magnetic_calibration,
    quaternion_from_euler,
    rotation_matrix,
    stationary_bias,
)


def packet(timestamp: float = 1.0, **updates: object) -> bytes:
    data = {
        "type": "navmin-gy85",
        "version": 1,
        "t": timestamp,
        "accel_g": [0, 0, 1],
        "gyro_dps": [0, 0, 0],
        "mag_raw": [500, 0, 0],
    }
    data.update(updates)
    return json.dumps(data).encode()


def test_protocol_v1_accepts_valid_packet_and_null_magnetometer() -> None:
    sample = ImuSample.decode(packet(12.5, mag_raw=None))
    assert sample.timestamp == 12.5
    assert sample.accel == pytest.approx([0, 0, 1])
    assert sample.gyro == pytest.approx([0, 0, 0])
    assert sample.mag is None


@pytest.mark.parametrize(
    "data",
    [
        b"not-json",
        packet(type="other"),
        packet(version=2),
        packet(version=True),
        json.dumps(
            {
                "type": "navmin-gy85",
                "version": 1,
                "t": 1.0,
                "accel_g": [0, 0, 1],
                "gyro_dps": [0, 0, 0],
            }
        ).encode(),
        packet(t=-1),
        packet(t=float("nan")),
        packet(accel_g=[0, 1]),
        packet(gyro_dps=[0, True, 0]),
        packet(gyro_dps=[0, "bad", 0]),
        packet(gyro_dps=[0, 2400, 0]),
        packet(mag_raw=[0, 0, float("inf")]),
        packet(mag_raw=[0, 0, 40000]),
    ],
)
def test_protocol_v1_rejects_invalid_data(data: bytes) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        ImuSample.decode(data)


def test_protocol_v1_rejects_oversized_datagram() -> None:
    data = packet() + b" " * (MAX_DATAGRAM_BYTES + 1)
    assert len(data) > MAX_DATAGRAM_BYTES
    with pytest.raises(ValueError, match="too large"):
        ImuSample.decode(data)


def test_gyro_calibration_rejects_instability_and_dropouts() -> None:
    samples = [
        ImuSample.decode(packet(i / 50, gyro_dps=[1, -19, -18])) for i in range(251)
    ]
    assert stationary_bias(samples) == pytest.approx([1, -19, -18])
    noisy = [
        replace(sample, gyro=sample.gyro + [0, 10 * (i % 2), 0])
        for i, sample in enumerate(samples)
    ]
    with pytest.raises(ValueError, match="нестабилен"):
        stationary_bias(noisy)
    with pytest.raises(ValueError, match="пропуски"):
        stationary_bias(samples[:80] + samples[100:])


def magnetic_samples() -> list[ImuSample]:
    result = []
    for i in range(1000):
        z = 1 - 2 * (i + 0.5) / 1000
        phi = i * np.pi * (3 - np.sqrt(5))
        direction = np.array(
            [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
        )
        result.append(
            ImuSample(
                i / 50,
                np.array([0.0, 0.0, 1.0]),
                np.zeros(3),
                np.array([100, -200, 300]) + direction * [600, 900, 450],
            )
        )
    return result


def test_magnetometer_corrects_bias_scale_and_rejects_flat_sweep() -> None:
    samples = magnetic_samples()
    bias, scale, radius = magnetic_calibration(samples)
    assert bias == pytest.approx([100, -200, 300], abs=3)
    norms = [np.linalg.norm((sample.mag - bias) * scale) for sample in samples]
    assert np.std(norms) / radius < 0.01
    with pytest.raises(ValueError):
        magnetic_calibration(
            [replace(sample, mag=sample.mag * [1, 1, 0]) for sample in samples]
        )


def test_six_face_accelerometer_calibration() -> None:
    bias = np.array([0.04, -0.03, 0.07])
    scales = np.array([1.04, 0.98, 1.08])
    faces = {}
    for sign in (-1, 1):
        for axis in range(3):
            direction = np.eye(3)[axis] * sign
            faces[sign * (axis + 1)] = bias + direction * scales
    actual_bias, actual_scale = accelerometer_calibration(faces)
    assert actual_bias == pytest.approx(bias)
    assert actual_scale == pytest.approx(1 / scales)
    with pytest.raises(ValueError):
        accelerometer_calibration({1: faces[1]})


@pytest.mark.parametrize("angles", [(30, -25, 80), (170, 20, -179), (-150, -40, 179)])
def test_tilt_compensated_heading_and_quaternion_zero(angles: tuple[int, int, int]) -> None:
    expected = quaternion_from_euler(*np.radians(angles))
    matrix = rotation_matrix(expected)
    sample = ImuSample(
        1,
        matrix.T @ [0.0, 0.0, 1.0],
        np.zeros(3),
        matrix.T @ [500.0, 0.0, 200.0],
    )
    calibration = ImuCalibration(
        gyro_ready=True,
        mag_ready=True,
        mag_radius=np.linalg.norm([500, 0, 200]),
    )
    fusion = OrientationFilter()
    fusion.update(sample, calibration)
    assert rotation_matrix(fusion.q) == pytest.approx(matrix)
    calibration.zero = fusion.q.tolist()
    assert euler_degrees(fusion.relative(calibration)) == pytest.approx(
        [0, 0, 0], abs=1e-10
    )


def test_gyro_integrates_across_wrap_and_rejects_magnetic_interference() -> None:
    fusion = OrientationFilter()
    calibration = ImuCalibration(gyro_ready=True)
    fusion.q = quaternion_from_euler(0, 0, np.radians(179))
    fusion.timestamp = 0.0
    fusion.update(
        ImuSample(
            0.02,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 100.0]),
            None,
        ),
        calibration,
    )
    assert euler_degrees(fusion.q)[2] == pytest.approx(-179, abs=0.01)
    calibration.mag_ready = True
    calibration.mag_radius = 500
    fusion.update(
        ImuSample(
            0.04,
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.array([3000.0, 0.0, 0.0]),
        ),
        calibration,
    )
    assert not fusion.heading_valid
    assert abs(euler_degrees(fusion.q)[2]) > 175


def test_orientation_operates_without_fresh_magnetometer() -> None:
    calibration = ImuCalibration(gyro_ready=True, mag_ready=True, mag_radius=500)
    fusion = OrientationFilter()
    fusion.update(
        ImuSample(1.0, np.array([0.0, 0.0, 1.0]), np.zeros(3), None),
        calibration,
    )
    assert not fusion.heading_valid
    assert euler_degrees(fusion.q) == pytest.approx([0, 0, 0])


@pytest.mark.parametrize(
    "axes, expected",
    [
        ([1, 2, 3], np.eye(3)),
        ([2, -1, 3], np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
    ],
)
def test_mounting_signed_axis_mapping_is_right_handed(
    axes: list[int], expected: np.ndarray
) -> None:
    assert axis_matrix(axes) == pytest.approx(expected)


@pytest.mark.parametrize("axes", ([1, 1, 3], [1, 2, -3], [True, 2, 3]))
def test_mounting_rejects_invalid_or_reflected_mapping(axes: list[int]) -> None:
    with pytest.raises(ValueError):
        axis_matrix(axes)


def test_calibration_roundtrip_and_corrupt_file_is_not_partially_applied(tmp_path) -> None:
    path = tmp_path / "imu.json"
    calibration = ImuCalibration(gyro_bias=[1, -19, -18], gyro_ready=True)
    calibration.save(path)
    assert ImuCalibration.load(path) == calibration
    assert not path.with_suffix(".json.tmp").exists()

    original = '{"version": 2}'
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        ImuCalibration.load(path)
    assert path.read_text(encoding="utf-8") == original


def test_calibration_strictly_rejects_unknown_or_incomplete_fields(tmp_path) -> None:
    path = tmp_path / "imu.json"
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        ImuCalibration.load(path)

    data = ImuCalibration().__dict__.copy()
    data["unexpected"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        ImuCalibration.load(path)
