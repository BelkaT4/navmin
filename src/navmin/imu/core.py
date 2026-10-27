"""Passive GY-85 protocol, calibration, mounting, and orientation algorithms."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MAX_DATAGRAM_BYTES = 2048
PROTOCOL_TYPE = "navmin-gy85"
PROTOCOL_VERSION = 1


def vector(value: object) -> np.ndarray:
    """Return one strict finite three-component sensor vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("expected three numeric components")
    if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value):
        raise ValueError("components must be numbers")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("components must be finite")
    return result


@dataclass(frozen=True)
class ImuSample:
    """One validated protocol-v1 sample in the Raspberry Pi monotonic time domain."""

    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray | None

    @classmethod
    def decode(cls, data: bytes) -> ImuSample:
        """Decode one protocol-v1 JSON datagram without accepting non-finite values."""
        if len(data) > MAX_DATAGRAM_BYTES:
            raise ValueError("IMU packet too large")
        payload = json.loads(data)
        required_fields = {"type", "version", "t", "accel_g", "gyro_dps", "mag_raw"}
        if not isinstance(payload, dict) or not required_fields.issubset(payload):
            raise ValueError("missing required IMU packet fields")
        if payload.get("type") != PROTOCOL_TYPE:
            raise ValueError("not a NavMin GY-85 packet")
        if type(payload.get("version")) is not int or payload["version"] != PROTOCOL_VERSION:
            raise ValueError("unsupported IMU packet version")
        timestamp = payload.get("t")
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError("invalid sender timestamp")
        accel = vector(payload.get("accel_g"))
        gyro = vector(payload.get("gyro_dps"))
        mag = None if payload.get("mag_raw") is None else vector(payload["mag_raw"])
        if np.max(np.abs(accel)) > 16 or np.max(np.abs(gyro)) > 2300:
            raise ValueError("sensor range exceeded")
        if mag is not None and np.max(np.abs(mag)) > 32768:
            raise ValueError("magnetometer range exceeded")
        return cls(float(timestamp), accel, gyro, mag)


@dataclass
class ImuCalibration:
    """Persistent site/sensor calibration compatible with the colleague implementation."""

    version: int = 1
    gyro_bias: list[float] = field(default_factory=lambda: [0.0] * 3)
    gyro_ready: bool = False
    accel_bias: list[float] = field(default_factory=lambda: [0.0] * 3)
    accel_scale: list[float] = field(default_factory=lambda: [1.0] * 3)
    accel_ready: bool = False
    mag_bias: list[float] = field(default_factory=lambda: [0.0] * 3)
    mag_scale: list[float] = field(default_factory=lambda: [1.0] * 3)
    mag_radius: float = 0.0
    mag_ready: bool = False
    # Body frame: X forward, Y left, Z up; rows select signed sensor axes.
    axes: list[int] = field(default_factory=lambda: [1, 2, 3])
    zero: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])

    def validate(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported IMU calibration version")
        for name in ("gyro_ready", "accel_ready", "mag_ready"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        for name in ("gyro_bias", "accel_bias", "mag_bias", "accel_scale", "mag_scale"):
            values = vector(getattr(self, name))
            if name.endswith("scale") and np.any(values <= 0):
                raise ValueError(f"{name} must be positive")
            if name.endswith("scale") and (np.min(values) < 0.1 or np.max(values) > 10):
                raise ValueError(f"{name} outside supported calibration range")
            if name.endswith("bias") and np.max(np.abs(values)) > 32768:
                raise ValueError(f"{name} outside sensor range")
        if type(self.mag_radius) not in (float, int) or not math.isfinite(self.mag_radius):
            raise ValueError("invalid magnetic radius")
        if not 0 <= self.mag_radius <= 65536 or (
            self.mag_ready and self.mag_radius <= 0
        ):
            raise ValueError("invalid calibrated magnetic radius")
        axis_matrix(self.axes)
        if not isinstance(self.zero, list) or len(self.zero) != 4:
            raise ValueError("zero must be a unit quaternion")
        if any(type(component) not in (int, float) for component in self.zero):
            raise ValueError("invalid zero quaternion")
        if not np.isfinite(self.zero).all() or abs(np.linalg.norm(self.zero) - 1) > 1e-5:
            raise ValueError("zero must be a unit quaternion")

    @classmethod
    def load(cls, path: Path) -> ImuCalibration:
        """Strictly load calibration; missing files yield an uncalibrated default."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("invalid IMU calibration fields")
        calibration = cls(**data)
        calibration.validate()
        return calibration

    def save(self, path: Path) -> None:
        """Atomically replace one calibration JSON after strict validation."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(asdict(self), indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def axis_matrix(axes: list[int]) -> np.ndarray:
    """Build the signed sensor-to-body axis mapping and reject reflections."""
    if not isinstance(axes, list) or len(axes) != 3:
        raise ValueError("select three sensor axes")
    if any(type(axis) is not int for axis in axes) or sorted(abs(axis) for axis in axes) != [
        1,
        2,
        3,
    ]:
        raise ValueError("sensor axes must be distinct")
    matrix = np.zeros((3, 3))
    for row, axis in enumerate(axes):
        matrix[row, abs(axis) - 1] = 1 if axis > 0 else -1
    if np.linalg.det(matrix) < 0.99:
        raise ValueError("axes must form a right-handed frame")
    return matrix


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_degrees(q: np.ndarray) -> tuple[float, float, float]:
    matrix = rotation_matrix(q)
    return tuple(
        np.degrees(
            [
                math.atan2(matrix[2, 1], matrix[2, 2]),
                math.asin(float(np.clip(-matrix[2, 0], -1, 1))),
                math.atan2(matrix[1, 0], matrix[0, 0]),
            ]
        )
    )


class OrientationFilter:
    """Quaternion gyro integration with gravity/tilt-compensated magnetic correction."""

    def __init__(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.timestamp: float | None = None
        self.heading_valid = False

    def update(self, sample: ImuSample, calibration: ImuCalibration) -> None:
        mapping = axis_matrix(calibration.axes)
        accel = mapping @ (
            (sample.accel - calibration.accel_bias) * calibration.accel_scale
        )
        gyro = mapping @ (sample.gyro - calibration.gyro_bias)
        dt = 0 if self.timestamp is None else sample.timestamp - self.timestamp
        initial = self.timestamp is None or dt > 0.2
        self.timestamp = sample.timestamp
        if not initial and calibration.gyro_ready:
            angular = np.radians(gyro) * dt
            angle = float(np.linalg.norm(angular))
            if angle > 1e-12:
                delta = np.r_[
                    math.cos(angle / 2), angular / angle * math.sin(angle / 2)
                ]
                self.q = quaternion_product(self.q, delta)
                self.q /= np.linalg.norm(self.q)
        norm = float(np.linalg.norm(accel))
        self.heading_valid = False
        if not 0.85 <= norm <= 1.15:
            return
        roll = math.atan2(accel[1], accel[2])
        pitch = math.atan2(-accel[0], math.hypot(accel[1], accel[2]))
        yaw = math.radians(euler_degrees(self.q)[2])
        if sample.mag is not None and calibration.mag_ready:
            mag = mapping @ (
                (sample.mag - calibration.mag_bias) * calibration.mag_scale
            )
            magnetic_norm = float(np.linalg.norm(mag))
            if (
                0.65 * calibration.mag_radius
                <= magnetic_norm
                <= 1.35 * calibration.mag_radius
            ):
                horizontal = rotation_matrix(quaternion_from_euler(roll, pitch, 0)) @ mag
                if math.hypot(horizontal[0], horizontal[1]) > 0.1 * magnetic_norm:
                    yaw = -math.atan2(horizontal[1], horizontal[0])
                    self.heading_valid = True
        measured = quaternion_from_euler(roll, pitch, yaw)
        if initial:
            self.q = measured
        else:
            # Align quaternion hemispheres to avoid a discontinuity at +/-180 degrees.
            if np.dot(self.q, measured) < 0:
                measured = -measured
            weight = dt / (0.8 + dt)
            self.q = (1 - weight) * self.q + weight * measured
            self.q /= np.linalg.norm(self.q)

    def relative(self, calibration: ImuCalibration) -> np.ndarray:
        inverse = np.asarray(calibration.zero) * np.array([1, -1, -1, -1])
        return quaternion_product(inverse, self.q)


def stationary_bias(samples: list[ImuSample]) -> list[float]:
    """Estimate gyro bias from at least five continuous stationary seconds."""
    if len(samples) < 150 or samples[-1].timestamp - samples[0].timestamp < 4.5:
        raise ValueError("нужно не менее 5 секунд непрерывных данных")
    if np.max(np.diff([sample.timestamp for sample in samples])) > 0.15:
        raise ValueError("пропуски данных: повторите калибровку")
    gyro = np.array([sample.gyro for sample in samples])
    accel = np.array([sample.accel for sample in samples])
    if np.max(np.std(gyro, axis=0)) > 1.5 or np.max(np.std(accel, axis=0)) > 0.025:
        raise ValueError("датчик двигался или гироскоп нестабилен; результат отклонён")
    if not 0.8 < np.linalg.norm(np.mean(accel, axis=0)) < 1.2:
        raise ValueError("акселерометр не показывает покой около 1 g")
    if np.max(np.abs(np.mean(gyro, axis=0))) > 100:
        raise ValueError("слишком большое смещение гироскопа; проверьте датчик")
    magnetic = np.array([sample.mag for sample in samples if sample.mag is not None])
    if len(magnetic) > 50:
        norm = np.linalg.norm(np.mean(magnetic, axis=0))
        if norm > 1 and np.max(np.std(magnetic, axis=0)) / norm > 0.025:
            raise ValueError(
                "магнитное поле менялось: вращение или помехи во время калибровки"
            )
    return np.mean(gyro, axis=0).tolist()


def magnetic_calibration(
    samples: list[ImuSample],
) -> tuple[list[float], list[float], float]:
    """Fit the colleague implementation's offset plus diagonal scale model."""
    valid = [sample for sample in samples if sample.mag is not None]
    if len(valid) < 300 or valid[-1].timestamp - valid[0].timestamp < 10:
        raise ValueError("нужно не менее 10 секунд вращения во всех направлениях")
    values = np.array([sample.mag for sample in valid])
    low, high = np.min(values, axis=0), np.max(values, axis=0)
    radii = (high - low) / 2
    if np.min(radii) < 100 or np.max(radii) / np.min(radii) > 5:
        raise ValueError("недостаточный охват осей или сильное искажение магнитного поля")
    bias = (high + low) / 2
    centered = values - bias
    octants = {tuple(row > 0) for row in centered}
    if len(octants) < 8:
        raise ValueError("повращайте датчик также в недостающих направлениях")
    radius = float(np.mean(radii))
    scale = radius / radii
    norms = np.linalg.norm(centered * scale, axis=1)
    if np.std(norms) / np.mean(norms) > 0.2:
        raise ValueError("магнитное поле нестабильно или нужна полная эллипсоидная калибровка")
    return bias.tolist(), scale.tolist(), radius


def accelerometer_calibration(
    faces: dict[int, np.ndarray],
) -> tuple[list[float], list[float]]:
    """Compute per-axis offset/scale from the six accepted stationary faces."""
    if set(faces) != {-3, -2, -1, 1, 2, 3}:
        raise ValueError("сначала измерьте все шесть положений")
    bias, scale = [], []
    for axis in range(3):
        low, high = faces[-axis - 1][axis], faces[axis + 1][axis]
        if not 1.5 < high - low < 2.5:
            raise ValueError("неверные положения акселерометра или неверный масштаб")
        bias.append(float((high + low) / 2))
        scale.append(float(2 / (high - low)))
    return bias, scale
