from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest

from navmin.calibration import (
    CalibrationJsonError,
    CalibrationResolutionMismatchError,
    CalibrationValidationError,
    UnsupportedCalibrationSchemaError,
    load_overview_calibration,
    load_stereo_calibration,
    overview_camera_model,
    stereo_camera_models,
)


def _overview_calibration() -> dict:
    return {
        "schema_version": 1,
        "image_width": 640,
        "image_height": 480,
        "K": [[1000.0, 0.0, 300.0], [0.0, 1000.0, 200.0], [0.0, 0.0, 1.0]],
        "D": [0.1, -0.2, 0.001, 0.002, 0.0],
        "new_camera_matrix": [
            [100.0, 0.0, 320.0],
            [0.0, 100.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _stereo_calibration() -> dict:
    return {
        "schema_version": 1,
        "image_width": 640,
        "image_height": 480,
        "K_left": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "D_left": [[0.1, -0.1, 0.0, 0.0, 0.01]],
        "K_right": [[510.0, 0.0, 318.0], [0.0, 510.0, 241.0], [0.0, 0.0, 1.0]],
        "D_right": [[0.1], [-0.1], [0.0], [0.0], [0.01]],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "T": [[-0.46], [0.0], [0.0]],
        "R1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "R2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "P1": [
            [200.0, 0.0, 320.0, 0.0],
            [0.0, 200.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "P2": [
            [200.0, 0.0, 320.0, -92.0],
            [0.0, 200.0, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "Q": [
            [1.0, 0.0, 0.0, -320.0],
            [0.0, 1.0, 0.0, -240.0],
            [0.0, 0.0, 0.0, 200.0],
            [0.0, 0.0, 1 / 0.46, 0.0],
        ],
    }


def _write(path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_overview_calibration_is_immutable(tmp_path) -> None:
    path = tmp_path / "overview.json"
    _write(path, _overview_calibration())

    calibration = load_overview_calibration(path, source_size=(640, 480))

    assert calibration.image_width == 640
    assert len(calibration.D) == 5
    with pytest.raises(FrozenInstanceError):
        calibration.image_width = 1  # type: ignore[misc]


def test_valid_stereo_calibration_normalizes_vector_shapes(tmp_path) -> None:
    path = tmp_path / "stereo.json"
    _write(path, _stereo_calibration())

    calibration = load_stereo_calibration(path, source_size=(640, 480))

    assert calibration.T == (-0.46, 0.0, 0.0)
    assert len(calibration.D_left) == 5
    assert len(calibration.D_right) == 5
    assert len(calibration.P1) == 3
    assert len(calibration.Q) == 4


@pytest.mark.parametrize("length", [4, 5, 8, 12, 14])
def test_opencv_pinhole_distortion_lengths_are_accepted(tmp_path, length) -> None:
    path = tmp_path / "overview.json"
    data = _overview_calibration()
    data["D"] = [0.0] * length
    _write(path, data)

    calibration = load_overview_calibration(path)

    assert len(calibration.D) == length


@pytest.mark.parametrize(
    ("kind", "mutate", "expected_path"),
    [
        (
            "overview",
            lambda d: d.__setitem__("K", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "K",
        ),
        (
            "overview",
            lambda d: d.__setitem__("D", [0.0] * 6),
            "D",
        ),
        (
            "stereo",
            lambda d: d.__setitem__("P1", [[1.0, 0.0, 0.0]] * 3),
            "P1[0]",
        ),
        (
            "stereo",
            lambda d: d.__setitem__("T", [1.0, 2.0]),
            "T",
        ),
    ],
)
def test_invalid_matrix_or_vector_shapes_are_rejected(
    tmp_path, kind, mutate, expected_path
) -> None:
    data = _overview_calibration() if kind == "overview" else _stereo_calibration()
    mutate(data)
    path = tmp_path / f"{kind}.json"
    _write(path, data)

    with pytest.raises(CalibrationValidationError) as exc_info:
        if kind == "overview":
            load_overview_calibration(path)
        else:
            load_stereo_calibration(path)

    assert exc_info.value.path == expected_path


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_calibration_values_are_rejected(tmp_path, invalid) -> None:
    path = tmp_path / "overview.json"
    data = _overview_calibration()
    data["K"][1][1] = invalid
    _write(path, data)

    with pytest.raises(CalibrationValidationError) as exc_info:
        load_overview_calibration(path)

    assert exc_info.value.path == "K[1][1]"
    assert "finite" in exc_info.value.reason


def test_exact_source_resolution_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "overview.json"
    _write(path, _overview_calibration())

    with pytest.raises(CalibrationResolutionMismatchError) as exc_info:
        load_overview_calibration(path, source_size=(641, 480))

    assert exc_info.value.expected == (640, 480)
    assert exc_info.value.actual == (641, 480)


def test_camera_model_returns_normalized_ray_using_working_frame_geometry(tmp_path) -> None:
    path = tmp_path / "overview.json"
    _write(path, _overview_calibration())
    calibration = load_overview_calibration(path)
    model = overview_camera_model(calibration)

    center = model.pixel_to_ray(320, 240)
    right = model.pixel_to_ray(420, 240)

    assert center.x == pytest.approx(0.0, abs=1e-12)
    assert center.y == pytest.approx(0.0, abs=1e-12)
    assert center.z == pytest.approx(1.0, abs=1e-12)
    assert math.sqrt(right.x**2 + right.y**2 + right.z**2) == pytest.approx(1.0)
    assert right.x == pytest.approx(1 / math.sqrt(2))
    assert right.z == pytest.approx(1 / math.sqrt(2))


def test_stereo_models_use_rectified_p1_p2_geometry(tmp_path) -> None:
    path = tmp_path / "stereo.json"
    _write(path, _stereo_calibration())
    calibration = load_stereo_calibration(path)
    left_model, right_model = stereo_camera_models(calibration)

    left = left_model.pixel_to_ray(520, 240)
    right = right_model.pixel_to_ray(520, 240)

    assert left.x == pytest.approx(1 / math.sqrt(2))
    assert right.x == pytest.approx(1 / math.sqrt(2))
    assert left.z == pytest.approx(1 / math.sqrt(2))
    assert right.z == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize(
    ("kind", "mutate", "error_type", "expected_path"),
    [
        (
            "overview",
            lambda d: d.__setitem__("extra", 1),
            CalibrationValidationError,
            "extra",
        ),
        (
            "stereo",
            lambda d: d.__setitem__("schema_version", 2),
            UnsupportedCalibrationSchemaError,
            "schema_version",
        ),
        (
            "overview",
            lambda d: d.__setitem__("image_width", True),
            CalibrationValidationError,
            "image_width",
        ),
    ],
)
def test_calibration_unknown_schema_and_type_errors_are_explicit(
    tmp_path, kind, mutate, error_type, expected_path
) -> None:
    data = _overview_calibration() if kind == "overview" else _stereo_calibration()
    mutate(data)
    path = tmp_path / f"{kind}.json"
    _write(path, data)

    with pytest.raises(error_type) as exc_info:
        if kind == "overview":
            load_overview_calibration(path)
        else:
            load_stereo_calibration(path)

    assert exc_info.value.path == expected_path


def test_malformed_calibration_json_is_explicit_error(tmp_path) -> None:
    path = tmp_path / "overview.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(CalibrationJsonError):
        load_overview_calibration(path)


def test_stereo_source_resolution_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "stereo.json"
    _write(path, _stereo_calibration())

    with pytest.raises(CalibrationResolutionMismatchError):
        load_stereo_calibration(path, source_size=(640, 481))


def test_missing_calibration_file_is_explicit_error(tmp_path) -> None:
    from navmin.calibration import CalibrationFileNotFoundError

    with pytest.raises(CalibrationFileNotFoundError):
        load_overview_calibration(tmp_path / "overview.json")


def test_missing_required_calibration_field_reports_path(tmp_path) -> None:
    path = tmp_path / "overview.json"
    data = _overview_calibration()
    del data["new_camera_matrix"]
    _write(path, data)

    with pytest.raises(CalibrationValidationError) as exc_info:
        load_overview_calibration(path)

    assert exc_info.value.path == "new_camera_matrix"
