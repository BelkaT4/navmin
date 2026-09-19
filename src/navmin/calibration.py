"""Strict calibration loading and immutable working-frame camera models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from navmin.contracts import CameraModel, CameraRay

SUPPORTED_CALIBRATION_SCHEMA_VERSION = 1
OPENCV_PINHOLE_DISTORTION_LENGTHS = frozenset({4, 5, 8, 12, 14})

type Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
type Matrix3x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
type Matrix4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
type Vector = tuple[float, ...]


class CalibrationError(ValueError):
    """Base class for calibration loading/validation errors."""


class CalibrationFileNotFoundError(CalibrationError):
    pass


class CalibrationJsonError(CalibrationError):
    def __init__(self, message: str) -> None:
        self.path = "$"
        self.reason = message
        super().__init__(f"$: {message}")


class CalibrationValidationError(CalibrationError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class UnsupportedCalibrationSchemaError(CalibrationValidationError):
    def __init__(self, actual: int) -> None:
        self.actual = actual
        super().__init__(
            "schema_version",
            (
                f"unsupported schema version {actual}; "
                f"expected {SUPPORTED_CALIBRATION_SCHEMA_VERSION}"
            ),
        )


class CalibrationResolutionMismatchError(CalibrationValidationError):
    def __init__(
        self,
        expected: tuple[int, int],
        actual: tuple[int, int],
    ) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            "image_width/image_height",
            (
                f"source resolution {actual[0]}x{actual[1]} does not match "
                f"calibration {expected[0]}x{expected[1]}"
            ),
        )


def _validation(path: str, reason: str) -> NoReturn:
    raise CalibrationValidationError(path, reason)


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _validation(path, "expected object")
    return value


def _fields(value: Any, required: set[str]) -> dict[str, Any]:
    obj = _object(value, "$")
    unknown = [key for key in obj if key not in required]
    if unknown:
        _validation(str(unknown[0]), "unknown field")
    missing = sorted(required - set(obj))
    if missing:
        _validation(missing[0], "missing required field")
    return obj


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int:
        _validation(path, "expected integer")
    if value <= 0:
        _validation(path, "must be > 0")
    return value


def _schema_version(value: Any) -> int:
    if type(value) is not int:
        _validation("schema_version", "expected integer")
    if value != SUPPORTED_CALIBRATION_SCHEMA_VERSION:
        raise UnsupportedCalibrationSchemaError(value)
    return value


def _finite_number(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        _validation(path, "expected number")
    try:
        number = float(value)
    except OverflowError:
        _validation(path, "number must be finite")
    if not math.isfinite(number):
        _validation(path, "number must be finite")
    return number


def _matrix(
    value: Any,
    path: str,
    rows: int,
    columns: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        _validation(path, f"expected {rows}x{columns} matrix")
    result: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        row_path = f"{path}[{row_index}]"
        if not isinstance(row, list) or len(row) != columns:
            _validation(row_path, f"expected row with {columns} elements")
        result.append(
            tuple(
                _finite_number(item, f"{row_path}[{column_index}]")
                for column_index, item in enumerate(row)
            )
        )
    return tuple(result)


def _vector(value: Any, path: str) -> Vector:
    """Accept JSON flat, 1xN, or Nx1 vectors without resizing coefficients."""
    if not isinstance(value, list) or not value:
        _validation(path, "expected non-empty vector")

    if all(not isinstance(item, list) for item in value):
        return tuple(
            _finite_number(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )

    if len(value) == 1 and isinstance(value[0], list) and value[0]:
        return tuple(
            _finite_number(item, f"{path}[0][{index}]")
            for index, item in enumerate(value[0])
        )

    if all(isinstance(item, list) and len(item) == 1 for item in value):
        return tuple(
            _finite_number(item[0], f"{path}[{index}][0]")
            for index, item in enumerate(value)
        )

    _validation(path, "expected flat, 1xN, or Nx1 vector")


def _pinhole_distortion_vector(value: Any, path: str) -> Vector:
    vector = _vector(value, path)
    if len(vector) not in OPENCV_PINHOLE_DISTORTION_LENGTHS:
        allowed = ", ".join(
            str(length) for length in sorted(OPENCV_PINHOLE_DISTORTION_LENGTHS)
        )
        _validation(path, f"expected OpenCV pinhole distortion length in {{{allowed}}}")
    return vector


def _fisheye_distortion_vector(value: Any, path: str) -> Vector:
    vector = _vector(value, path)
    if len(vector) != 4:
        _validation(path, "expected OpenCV fisheye distortion vector length 4")
    return vector


def _fixed_vector(value: Any, path: str, length: int) -> Vector:
    vector = _vector(value, path)
    if len(vector) != length:
        _validation(path, f"expected vector length {length}")
    return vector


def _read_json(path: str | Path) -> Any:
    calibration_path = Path(path)
    try:
        text = calibration_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CalibrationFileNotFoundError(
            f"calibration file not found: {calibration_path}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise CalibrationJsonError(f"cannot read calibration JSON: {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CalibrationJsonError(
            f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _check_source_size(
    calibration_size: tuple[int, int],
    source_size: tuple[int, int] | None,
) -> None:
    if source_size is None:
        return
    if (
        not isinstance(source_size, tuple)
        or len(source_size) != 2
        or type(source_size[0]) is not int
        or type(source_size[1]) is not int
        or source_size[0] <= 0
        or source_size[1] <= 0
    ):
        raise ValueError("source_size must be a (positive width, positive height) tuple")
    if source_size != calibration_size:
        raise CalibrationResolutionMismatchError(calibration_size, source_size)


@dataclass(frozen=True)
class OverviewCalibration:
    schema_version: int
    image_width: int
    image_height: int
    K: Matrix3x3
    D: Vector
    new_camera_matrix: Matrix3x3


@dataclass(frozen=True)
class StereoCalibration:
    schema_version: int
    image_width: int
    image_height: int
    K_left: Matrix3x3
    D_left: Vector
    K_right: Matrix3x3
    D_right: Vector
    R: Matrix3x3
    T: Vector
    R1: Matrix3x3
    R2: Matrix3x3
    P1: Matrix3x4
    P2: Matrix3x4
    Q: Matrix4x4


@dataclass(frozen=True)
class WorkingFrameCameraModel:
    """Immutable pinhole unprojection model for an already corrected frame."""

    _inverse_projection: Matrix3x3

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        if type(x_px) not in (int, float) or type(y_px) not in (int, float):
            raise TypeError("pixel coordinates must be numbers")
        x = float(x_px)
        y = float(y_px)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("pixel coordinates must be finite")

        m = self._inverse_projection
        ray_x = m[0][0] * x + m[0][1] * y + m[0][2]
        ray_y = m[1][0] * x + m[1][1] * y + m[1][2]
        ray_z = m[2][0] * x + m[2][1] * y + m[2][2]
        norm = math.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("pixel cannot be converted to a finite camera ray")
        return CameraRay(ray_x / norm, ray_y / norm, ray_z / norm)


def load_overview_calibration(
    path: str | Path,
    *,
    source_size: tuple[int, int] | None = None,
) -> OverviewCalibration:
    obj = _fields(
        _read_json(path),
        {
            "schema_version",
            "image_width",
            "image_height",
            "K",
            "D",
            "new_camera_matrix",
        },
    )
    calibration = OverviewCalibration(
        schema_version=_schema_version(obj["schema_version"]),
        image_width=_positive_int(obj["image_width"], "image_width"),
        image_height=_positive_int(obj["image_height"], "image_height"),
        K=_matrix(obj["K"], "K", 3, 3),  # type: ignore[arg-type]
        D=_fisheye_distortion_vector(obj["D"], "D"),
        new_camera_matrix=_matrix(  # type: ignore[arg-type]
            obj["new_camera_matrix"], "new_camera_matrix", 3, 3
        ),
    )
    _check_source_size(
        (calibration.image_width, calibration.image_height), source_size
    )
    return calibration


def load_stereo_calibration(
    path: str | Path,
    *,
    source_size: tuple[int, int] | None = None,
) -> StereoCalibration:
    obj = _fields(
        _read_json(path),
        {
            "schema_version",
            "image_width",
            "image_height",
            "K_left",
            "D_left",
            "K_right",
            "D_right",
            "R",
            "T",
            "R1",
            "R2",
            "P1",
            "P2",
            "Q",
        },
    )
    calibration = StereoCalibration(
        schema_version=_schema_version(obj["schema_version"]),
        image_width=_positive_int(obj["image_width"], "image_width"),
        image_height=_positive_int(obj["image_height"], "image_height"),
        K_left=_matrix(obj["K_left"], "K_left", 3, 3),  # type: ignore[arg-type]
        D_left=_pinhole_distortion_vector(obj["D_left"], "D_left"),
        K_right=_matrix(obj["K_right"], "K_right", 3, 3),  # type: ignore[arg-type]
        D_right=_pinhole_distortion_vector(obj["D_right"], "D_right"),
        R=_matrix(obj["R"], "R", 3, 3),  # type: ignore[arg-type]
        T=_fixed_vector(obj["T"], "T", 3),
        R1=_matrix(obj["R1"], "R1", 3, 3),  # type: ignore[arg-type]
        R2=_matrix(obj["R2"], "R2", 3, 3),  # type: ignore[arg-type]
        P1=_matrix(obj["P1"], "P1", 3, 4),  # type: ignore[arg-type]
        P2=_matrix(obj["P2"], "P2", 3, 4),  # type: ignore[arg-type]
        Q=_matrix(obj["Q"], "Q", 4, 4),  # type: ignore[arg-type]
    )
    _check_source_size(
        (calibration.image_width, calibration.image_height), source_size
    )
    return calibration


def _camera_model_from_matrix(
    matrix: tuple[tuple[float, ...], ...],
    path: str,
) -> WorkingFrameCameraModel:
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise CalibrationValidationError(
            path, "working-frame projection matrix must be invertible"
        ) from exc
    if not np.isfinite(inverse).all():
        raise CalibrationValidationError(
            path, "working-frame inverse projection must be finite"
        )
    values = tuple(tuple(float(item) for item in row) for row in inverse.tolist())
    return WorkingFrameCameraModel(values)  # type: ignore[arg-type]


def overview_camera_model(calibration: OverviewCalibration) -> CameraModel:
    """Build the Overview model from corrected working-frame geometry."""
    return _camera_model_from_matrix(
        calibration.new_camera_matrix, "new_camera_matrix"
    )


def stereo_camera_models(
    calibration: StereoCalibration,
) -> tuple[CameraModel, CameraModel]:
    """Build rectified left/right models from P1/P2 working-frame geometry."""
    left_matrix = tuple(row[:3] for row in calibration.P1)
    right_matrix = tuple(row[:3] for row in calibration.P2)
    return (
        _camera_model_from_matrix(left_matrix, "P1"),
        _camera_model_from_matrix(right_matrix, "P2"),
    )


__all__ = [
    "OPENCV_PINHOLE_DISTORTION_LENGTHS",
    "CalibrationError",
    "CalibrationFileNotFoundError",
    "CalibrationJsonError",
    "CalibrationResolutionMismatchError",
    "CalibrationValidationError",
    "OverviewCalibration",
    "StereoCalibration",
    "UnsupportedCalibrationSchemaError",
    "WorkingFrameCameraModel",
    "load_overview_calibration",
    "load_stereo_calibration",
    "overview_camera_model",
    "stereo_camera_models",
]
