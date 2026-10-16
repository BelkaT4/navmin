from __future__ import annotations

from dataclasses import replace
from threading import Event

import cv2
import numpy as np
import pytest

from navmin.calibration import StereoCalibration
from navmin.contracts import BBox, CameraRole, FramePacket, TargetRef
from navmin.vision import stereo_distance
from navmin.vision.stereo_distance import (
    StereoDistanceEstimate,
    StereoDistanceFailure,
    StereoDistanceFailureReason,
    StereoDistanceMethod,
    StereoPoint3D,
    StereoSgbmSettings,
    compute_sgbm_disparity,
    estimate_distance_from_disparity_roi,
    estimate_distance_from_stereo_pair,
    reproject_disparity_point,
    validate_stereo_pair_geometry,
)
from navmin.vision.stereo_distance_worker import StereoDistanceWorker
from navmin.vision.stereo_pairing import StereoFramePair, StereoPairIdentity
from navmin.vision.stereo_target_join import StereoPairingKind, StereoTargetObservation
from navmin.vision.stereo_timestamp_pairing import (
    TimestampStereoFramePair,
    TimestampStereoPairIdentity,
)


def calibration(
    *,
    width: int = 640,
    height: int = 480,
    focal: float = 200.0,
    baseline: float = 0.46,
) -> StereoCalibration:
    return StereoCalibration(
        schema_version=1,
        translation_unit="m",
        image_width=width,
        image_height=height,
        K_left=((focal, 0.0, width / 2), (0.0, focal, height / 2), (0.0, 0.0, 1.0)),
        D_left=(0.0, 0.0, 0.0, 0.0),
        K_right=((focal, 0.0, width / 2), (0.0, focal, height / 2), (0.0, 0.0, 1.0)),
        D_right=(0.0, 0.0, 0.0, 0.0),
        R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        T=(-baseline, 0.0, 0.0),
        R1=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        R2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        P1=(
            (focal, 0.0, width / 2, 0.0),
            (0.0, focal, height / 2, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        P2=(
            (focal, 0.0, width / 2, -focal * baseline),
            (0.0, focal, height / 2, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        Q=(
            (1.0, 0.0, 0.0, -width / 2),
            (0.0, 1.0, 0.0, -height / 2),
            (0.0, 0.0, 0.0, focal),
            (0.0, 0.0, 1.0 / baseline, 0.0),
        ),
    )


def pair(*, width: int = 640, height: int = 480) -> StereoFramePair:
    def frame(camera: CameraRole) -> FramePacket:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image.flags.writeable = False
        return FramePacket(
            camera=camera,
            generation=1,
            frame_id=1,
            capture_id=7,
            receive_timestamp_ns=1_000,
            image=image,
        )

    return StereoFramePair(
        identity=StereoPairIdentity(1, 1, 7),
        left=frame(CameraRole.STEREO_LEFT),
        right=frame(CameraRole.STEREO_RIGHT),
    )


def test_q_reprojection_at_principal_point_matches_known_depth() -> None:
    result = reproject_disparity_point(
        calibration(),
        x_px=320.0,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=1.0,
    )

    assert isinstance(result, StereoPoint3D)
    assert result.x_m == pytest.approx(0.0)
    assert result.y_m == pytest.approx(0.0)
    assert result.z_m == pytest.approx(4.6)
    assert result.distance_m == pytest.approx(4.6)


def test_q_reprojection_returns_euclidean_3d_distance() -> None:
    result = reproject_disparity_point(
        calibration(),
        x_px=420.0,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=1.0,
    )

    assert isinstance(result, StereoPoint3D)
    assert result.x_m == pytest.approx(2.3)
    assert result.y_m == pytest.approx(0.0)
    assert result.z_m == pytest.approx(4.6)
    assert result.distance_m == pytest.approx((2.3**2 + 4.6**2) ** 0.5)


def test_meter_and_millimeter_geometry_produce_same_distance() -> None:
    meters = reproject_disparity_point(
        calibration(baseline=0.46),
        x_px=320.0,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=1.0,
    )
    millimeters = reproject_disparity_point(
        calibration(baseline=460.0),
        x_px=320.0,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=0.001,
    )

    assert isinstance(meters, StereoPoint3D)
    assert isinstance(millimeters, StereoPoint3D)
    assert millimeters.distance_m == pytest.approx(meters.distance_m)
    assert millimeters.z_m == pytest.approx(meters.z_m)


@pytest.mark.parametrize(
    "scale",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_length_scale_is_typed_failure(scale: float) -> None:
    result = reproject_disparity_point(
        calibration(),
        x_px=320.0,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=scale,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INVALID_LENGTH_SCALE
    )


@pytest.mark.parametrize(
    "disparity",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_disparity_is_typed_failure(disparity: float) -> None:
    result = reproject_disparity_point(
        calibration(),
        x_px=320.0,
        y_px=240.0,
        disparity_px=disparity,
        length_scale_to_m=1.0,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INVALID_DISPARITY
    )


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pixel_coordinate_is_typed_failure(coordinate: float) -> None:
    result = reproject_disparity_point(
        calibration(),
        x_px=coordinate,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=1.0,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INVALID_PIXEL_COORDINATE
    )


def test_non_finite_or_zero_homogeneous_reprojection_is_rejected() -> None:
    base = calibration()
    zero_w = replace(
        base,
        Q=(
            base.Q[0],
            base.Q[1],
            base.Q[2],
            (0.0, 0.0, 0.0, 0.0),
        ),
    )

    result = reproject_disparity_point(
        zero_w,
        x_px=320.0,
        y_px=240.0,
        disparity_px=20.0,
        length_scale_to_m=1.0,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.NON_FINITE_REPROJECTION
    )


def test_pair_geometry_accepts_exact_working_frame_resolution() -> None:
    assert validate_stereo_pair_geometry(pair(), calibration()) is None


def test_pair_geometry_rejects_left_right_shape_mismatch() -> None:
    stereo_pair = pair()
    right_image = np.zeros((479, 640, 3), dtype=np.uint8)
    right_image.flags.writeable = False
    mismatched = replace(
        stereo_pair,
        right=replace(stereo_pair.right, image=right_image),
    )

    result = validate_stereo_pair_geometry(mismatched, calibration())

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.FRAME_SHAPE_MISMATCH
    )


def test_pair_geometry_rejects_calibration_resolution_mismatch() -> None:
    result = validate_stereo_pair_geometry(pair(), calibration(width=641, height=480))

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.CALIBRATION_RESOLUTION_MISMATCH
    )


def test_pair_geometry_rejects_wrong_image_shape() -> None:
    stereo_pair = pair()
    bad_image = np.zeros((480, 640), dtype=np.uint8)
    bad_image.flags.writeable = False
    malformed = replace(
        stereo_pair,
        left=replace(stereo_pair.left, image=bad_image),
    )

    result = validate_stereo_pair_geometry(malformed, calibration())

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INVALID_FRAME_SHAPE
    )


def depth_only_calibration(
    *,
    width: int = 8,
    height: int = 6,
    focal: float = 8.0,
    baseline: float = 0.5,
) -> StereoCalibration:
    base = calibration(
        width=width,
        height=height,
        focal=focal,
        baseline=baseline,
    )
    return replace(
        base,
        Q=(
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, focal),
            (0.0, 0.0, 1.0 / baseline, 0.0),
        ),
    )


def test_roi_aggregation_returns_typed_median_estimate() -> None:
    disparity = np.full((6, 8), 2.0, dtype=np.float32)

    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        disparity,
        BBox(x=2, y=1, width=4, height=3),
        length_scale_to_m=1.0,
        min_valid_pixel_count=12,
    )

    assert isinstance(result, StereoDistanceEstimate)
    assert result.distance_m == pytest.approx(2.0)
    assert result.median_disparity_px == pytest.approx(2.0)
    assert result.valid_pixel_count == 12
    assert result.valid_pixel_fraction == pytest.approx(1.0)
    assert result.method is StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN


def test_roi_meter_and_millimeter_geometry_produce_same_distance() -> None:
    disparity = np.full((6, 8), 2.0, dtype=np.float32)
    roi = BBox(x=1, y=1, width=3, height=3)

    meters = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(baseline=0.5),
        disparity,
        roi,
        length_scale_to_m=1.0,
        min_valid_pixel_count=9,
    )
    millimeters = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(baseline=500.0),
        disparity,
        roi,
        length_scale_to_m=0.001,
        min_valid_pixel_count=9,
    )

    assert isinstance(meters, StereoDistanceEstimate)
    assert isinstance(millimeters, StereoDistanceEstimate)
    assert millimeters.distance_m == pytest.approx(meters.distance_m)
    assert millimeters.median_disparity_px == pytest.approx(
        meters.median_disparity_px
    )


@pytest.mark.parametrize(
    "roi",
    [
        BBox(x=-1, y=0, width=2, height=2),
        BBox(x=0, y=-1, width=2, height=2),
        BBox(x=7, y=0, width=2, height=2),
        BBox(x=0, y=5, width=2, height=2),
    ],
)
def test_roi_outside_frame_is_rejected_without_clipping(roi: BBox) -> None:
    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        np.full((6, 8), 2.0, dtype=np.float32),
        roi,
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.ROI_OUT_OF_FRAME
    )


@pytest.mark.parametrize(
    "roi",
    [BBox(x=1, y=1, width=0, height=2), BBox(x=1, y=1, width=2, height=0)],
)
def test_empty_roi_is_rejected(roi: BBox) -> None:
    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        np.full((6, 8), 2.0, dtype=np.float32),
        roi,
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(StereoDistanceFailureReason.EMPTY_ROI)


def test_invalid_disparity_pixels_are_excluded_and_reported() -> None:
    disparity = np.zeros((6, 8), dtype=np.float64)
    disparity[1:3, 1:4] = np.array(
        [[2.0, 2.0, np.nan], [2.0, -1.0, np.inf]],
        dtype=np.float64,
    )

    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        disparity,
        BBox(x=1, y=1, width=3, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=3,
    )

    assert isinstance(result, StereoDistanceEstimate)
    assert result.distance_m == pytest.approx(2.0)
    assert result.median_disparity_px == pytest.approx(2.0)
    assert result.valid_pixel_count == 3
    assert result.valid_pixel_fraction == pytest.approx(0.5)


def test_insufficient_valid_roi_pixels_is_typed_failure() -> None:
    disparity = np.zeros((6, 8), dtype=np.float64)
    disparity[1:3, 1:4] = np.array(
        [[2.0, 2.0, np.nan], [0.0, -1.0, np.inf]],
        dtype=np.float64,
    )

    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        disparity,
        BBox(x=1, y=1, width=3, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=3,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INSUFFICIENT_VALID_PIXELS
    )


def test_outlier_disparity_does_not_dominate_roi_median() -> None:
    disparity = np.zeros((6, 8), dtype=np.float64)
    disparity[2, 1:6] = np.array([2.0, 2.0, 2.0, 2.0, 200.0])

    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        disparity,
        BBox(x=1, y=2, width=5, height=1),
        length_scale_to_m=1.0,
        min_valid_pixel_count=5,
    )

    assert isinstance(result, StereoDistanceEstimate)
    assert result.median_disparity_px == pytest.approx(2.0)
    assert result.distance_m == pytest.approx(2.0)


@pytest.mark.parametrize(
    "disparity",
    [
        np.zeros((5, 8), dtype=np.float32),
        np.zeros((6, 8, 1), dtype=np.float32),
        np.full((6, 8), "2", dtype="<U1"),
    ],
)
def test_invalid_disparity_map_is_rejected(disparity: np.ndarray) -> None:
    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        disparity,
        BBox(x=1, y=1, width=2, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INVALID_DISPARITY_MAP
    )


@pytest.mark.parametrize("min_valid_pixel_count", [0, -1, True, 1.5])
def test_invalid_min_valid_pixel_count_is_rejected(
    min_valid_pixel_count: object,
) -> None:
    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(),
        np.full((6, 8), 2.0, dtype=np.float32),
        BBox(x=1, y=1, width=2, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=min_valid_pixel_count,  # type: ignore[arg-type]
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INVALID_MIN_VALID_PIXELS
    )


def test_roi_non_finite_reprojection_is_typed_failure() -> None:
    base = depth_only_calibration()
    invalid_q = replace(
        base,
        Q=(base.Q[0], base.Q[1], base.Q[2], (0.0, 0.0, 0.0, 0.0)),
    )

    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        invalid_q,
        np.full((6, 8), 2.0, dtype=np.float32),
        BBox(x=1, y=1, width=2, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.NON_FINITE_REPROJECTION
    )


def test_roi_non_positive_distance_is_typed_failure() -> None:
    base = depth_only_calibration()
    zero_point_q = replace(
        base,
        Q=(
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
    )

    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        zero_point_q,
        np.full((6, 8), 2.0, dtype=np.float32),
        BBox(x=1, y=1, width=2, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.NON_POSITIVE_DISTANCE
    )


def test_roi_aggregation_propagates_calibration_resolution_mismatch() -> None:
    result = estimate_distance_from_disparity_roi(
        pair(width=8, height=6),
        depth_only_calibration(width=9, height=6),
        np.full((6, 8), 2.0, dtype=np.float32),
        BBox(x=1, y=1, width=2, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.CALIBRATION_RESOLUTION_MISMATCH
    )


class _FakeSgbmMatcher:
    def __init__(self, raw: np.ndarray) -> None:
        self._raw = raw

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        assert left.shape == right.shape
        return self._raw.copy()


def test_sgbm_converts_opencv_fixed_point_and_masks_invalid_disparity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.array(
        [
            [-16, 0, 16, 32],
            [48, 64, 80, 96],
            [112, 128, 144, 160],
            [176, 192, 208, 224],
        ],
        dtype=np.int16,
    )
    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> _FakeSgbmMatcher:
        captured.update(kwargs)
        return _FakeSgbmMatcher(raw)

    monkeypatch.setattr(stereo_distance.cv2, "StereoSGBM_create", fake_create)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    disparity = compute_sgbm_disparity(
        image,
        image,
        StereoSgbmSettings(
            num_disparities=17,
            block_size=4,
            clahe_enabled=False,
            processing_scale=1.0,
        ),
    )

    assert np.isnan(disparity[0, 0])
    assert np.isnan(disparity[0, 1])
    assert disparity[0, 2] == pytest.approx(1.0)
    assert disparity[0, 3] == pytest.approx(2.0)
    assert captured["numDisparities"] == 16
    assert captured["blockSize"] == 5


def _synthetic_rectified_pair(
    *,
    width: int = 640,
    height: int = 480,
    disparity_px: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    mono = rng.integers(0, 256, (height, width), dtype=np.uint8)
    left = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    right = np.zeros_like(left)
    right[:, : width - disparity_px] = left[:, disparity_px:]
    return left, right


def _median_valid_disparity(disparity: np.ndarray) -> float:
    roi = disparity[80:-80, 220:-80]
    values = roi[np.isfinite(roi)]
    assert values.size > 0
    return float(np.median(values))


def test_sgbm_recovers_known_rectified_disparity_from_old_navmin_path() -> None:
    left, right = _synthetic_rectified_pair()

    disparity = compute_sgbm_disparity(
        left,
        right,
        StereoSgbmSettings(
            num_disparities=160,
            block_size=5,
            uniqueness_ratio=5,
            speckle_window_size=0,
            processing_scale=1.0,
        ),
    )

    assert abs(_median_valid_disparity(disparity) - 96.0) < 0.5


def test_sgbm_processing_scale_restores_original_pixel_disparity() -> None:
    left, right = _synthetic_rectified_pair()
    base = StereoSgbmSettings(
        num_disparities=160,
        block_size=5,
        uniqueness_ratio=5,
        speckle_window_size=0,
        processing_scale=1.0,
    )

    full = compute_sgbm_disparity(left, right, base)
    half = compute_sgbm_disparity(
        left,
        right,
        replace(base, processing_scale=0.5),
    )

    full_median = _median_valid_disparity(full)
    half_median = _median_valid_disparity(half)
    assert abs(full_median - 96.0) < 0.5
    assert abs(half_median - 96.0) < 1.0
    assert abs(full_median - half_median) < 1.0


def test_sgbm_rejects_mismatched_image_shapes() -> None:
    left = np.zeros((48, 64, 3), dtype=np.uint8)
    right = np.zeros((47, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="identical shape"):
        compute_sgbm_disparity(left, right, StereoSgbmSettings())



def _pair_from_images(left: np.ndarray, right: np.ndarray) -> StereoFramePair:
    left_frame = left.copy()
    right_frame = right.copy()
    left_frame.flags.writeable = False
    right_frame.flags.writeable = False
    return StereoFramePair(
        identity=StereoPairIdentity(1, 1, 7),
        left=FramePacket(
            camera=CameraRole.STEREO_LEFT,
            generation=1,
            frame_id=1,
            capture_id=7,
            receive_timestamp_ns=1_000,
            image=left_frame,
        ),
        right=FramePacket(
            camera=CameraRole.STEREO_RIGHT,
            generation=1,
            frame_id=1,
            capture_id=7,
            receive_timestamp_ns=1_000,
            image=right_frame,
        ),
    )


def test_pair_to_distance_recovers_known_synthetic_depth() -> None:
    left, right = _synthetic_rectified_pair()
    stereo_pair = _pair_from_images(left, right)
    stereo_calibration = depth_only_calibration(
        width=640,
        height=480,
        focal=200.0,
        baseline=0.46,
    )

    result = estimate_distance_from_stereo_pair(
        stereo_pair,
        stereo_calibration,
        StereoSgbmSettings(
            num_disparities=160,
            block_size=5,
            uniqueness_ratio=5,
            speckle_window_size=0,
            processing_scale=1.0,
        ),
        roi=BBox(x=220, y=80, width=340, height=320),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1_000,
    )

    assert isinstance(result, StereoDistanceEstimate)
    assert result.median_disparity_px == pytest.approx(96.0, abs=0.5)
    assert result.distance_m == pytest.approx(200.0 * 0.46 / 96.0, abs=0.01)


def test_pair_to_distance_none_roi_uses_full_left_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stereo_pair = pair(width=8, height=6)
    disparity = np.full((6, 8), 2.0, dtype=np.float32)

    def fake_disparity(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        settings: StereoSgbmSettings,
    ) -> np.ndarray:
        assert left_bgr is stereo_pair.left.image
        assert right_bgr is stereo_pair.right.image
        return disparity

    monkeypatch.setattr(stereo_distance, "compute_sgbm_disparity", fake_disparity)
    result = estimate_distance_from_stereo_pair(
        stereo_pair,
        depth_only_calibration(),
        StereoSgbmSettings(),
        length_scale_to_m=1.0,
        min_valid_pixel_count=48,
    )

    assert isinstance(result, StereoDistanceEstimate)
    assert result.distance_m == pytest.approx(2.0)
    assert result.valid_pixel_count == 48
    assert result.valid_pixel_fraction == pytest.approx(1.0)


def test_pair_to_distance_explicit_roi_ignores_background_disparity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stereo_pair = pair(width=8, height=6)
    disparity = np.full((6, 8), 4.0, dtype=np.float32)
    disparity[2:4, 3:5] = 2.0

    monkeypatch.setattr(
        stereo_distance,
        "compute_sgbm_disparity",
        lambda left_bgr, right_bgr, settings: disparity,
    )
    result = estimate_distance_from_stereo_pair(
        stereo_pair,
        depth_only_calibration(),
        StereoSgbmSettings(),
        roi=BBox(x=3, y=2, width=2, height=2),
        length_scale_to_m=1.0,
        min_valid_pixel_count=4,
    )

    assert isinstance(result, StereoDistanceEstimate)
    assert result.median_disparity_px == pytest.approx(2.0)
    assert result.distance_m == pytest.approx(2.0)
    assert result.valid_pixel_count == 4


def test_pair_to_distance_textureless_pair_rejects_without_stale_fallback() -> None:
    stereo_pair = pair(width=64, height=48)

    result = estimate_distance_from_stereo_pair(
        stereo_pair,
        depth_only_calibration(width=64, height=48),
        StereoSgbmSettings(
            num_disparities=16,
            block_size=5,
            speckle_window_size=0,
            processing_scale=1.0,
        ),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.INSUFFICIENT_VALID_PIXELS
    )


def test_pair_to_distance_rejects_geometry_before_running_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stereo_pair = pair(width=8, height=6)

    def unexpected_matcher(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("SGBM must not run for invalid pair/calibration geometry")

    monkeypatch.setattr(
        stereo_distance,
        "compute_sgbm_disparity",
        unexpected_matcher,
    )
    result = estimate_distance_from_stereo_pair(
        stereo_pair,
        depth_only_calibration(width=9, height=6),
        StereoSgbmSettings(),
        length_scale_to_m=1.0,
        min_valid_pixel_count=1,
    )

    assert result == StereoDistanceFailure(
        StereoDistanceFailureReason.CALIBRATION_RESOLUTION_MISMATCH
    )


def test_pair_to_distance_processing_scale_preserves_distance() -> None:
    left, right = _synthetic_rectified_pair()
    stereo_pair = _pair_from_images(left, right)
    stereo_calibration = depth_only_calibration(
        width=640,
        height=480,
        focal=200.0,
        baseline=0.46,
    )
    base_settings = StereoSgbmSettings(
        num_disparities=160,
        block_size=5,
        uniqueness_ratio=5,
        speckle_window_size=0,
        processing_scale=1.0,
    )
    roi = BBox(x=220, y=80, width=340, height=320)

    full = estimate_distance_from_stereo_pair(
        stereo_pair,
        stereo_calibration,
        base_settings,
        roi=roi,
        length_scale_to_m=1.0,
        min_valid_pixel_count=1_000,
    )
    half = estimate_distance_from_stereo_pair(
        stereo_pair,
        stereo_calibration,
        replace(base_settings, processing_scale=0.5),
        roi=roi,
        length_scale_to_m=1.0,
        min_valid_pixel_count=1_000,
    )

    assert isinstance(full, StereoDistanceEstimate)
    assert isinstance(half, StereoDistanceEstimate)
    expected_distance_m = 200.0 * 0.46 / 96.0
    assert full.distance_m == pytest.approx(expected_distance_m, abs=0.01)
    assert half.distance_m == pytest.approx(expected_distance_m, abs=0.02)
    assert half.distance_m == pytest.approx(full.distance_m, abs=0.02)


def _worker_observation(
    frame_id: int,
    *,
    intent_revision: int = 1,
    timestamp_ns: int = 1_000,
) -> StereoTargetObservation:
    base = pair(width=8, height=6)
    left = replace(
        base.left,
        frame_id=frame_id,
        receive_timestamp_ns=timestamp_ns,
    )
    right = replace(base.right, frame_id=frame_id + 100)
    exact_pair = replace(base, left=left, right=right)
    return StereoTargetObservation(
        target=TargetRef(CameraRole.STEREO_LEFT, 1, 7),
        intent_revision=intent_revision,
        pair=exact_pair,
        roi=BBox(1, 1, 4, 3),
    )


def _worker_estimate(distance_m: float) -> StereoDistanceEstimate:
    return StereoDistanceEstimate(
        distance_m=distance_m,
        median_disparity_px=10.0,
        valid_pixel_count=12,
        valid_pixel_fraction=1.0,
        method=StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN,
    )


def test_distance_worker_keeps_only_latest_not_started_job() -> None:
    first_started = Event()
    release_first = Event()
    calls: list[int] = []

    def compute(observation: StereoTargetObservation):
        frame_id = observation.pair.left.frame_id
        calls.append(frame_id)
        if frame_id == 1:
            first_started.set()
            assert release_first.wait(1.0)
        return _worker_estimate(float(frame_id))

    worker = StereoDistanceWorker(
        calibration=calibration(width=8, height=6),
        compute=compute,
    )
    worker.start()
    try:
        assert worker.submit_observation(_worker_observation(1))
        assert first_started.wait(1.0)
        assert worker.submit_observation(_worker_observation(2))
        assert worker.submit_observation(_worker_observation(3))
        release_first.set()

        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None
        if snapshot.value is None or snapshot.value.source_frame_id != 3:
            snapshot = worker.results.wait_for_revision(snapshot.revision, 1.0)
            assert snapshot is not None
        assert snapshot.value is not None
        assert snapshot.value.source_frame_id == 3
        assert calls == [1, 3]
    finally:
        worker.shutdown()


def test_distance_worker_candidate_uses_exact_left_observation_identity() -> None:
    observation = _worker_observation(9, intent_revision=17, timestamp_ns=123_456)
    worker = StereoDistanceWorker(
        calibration=calibration(width=8, height=6),
        compute=lambda _observation: _worker_estimate(42.5),
    )
    worker.start()
    try:
        assert worker.submit_observation(observation)
        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None
        candidate = snapshot.value
        assert candidate is not None
        assert candidate.target == observation.target
        assert candidate.intent_revision == 17
        assert candidate.distance_m == 42.5
        assert candidate.source_frame_id == 9
        assert candidate.capture_id == observation.pair.identity.capture_id
        assert candidate.measured_timestamp_ns == 123_456
    finally:
        worker.shutdown()


def test_timestamp_distance_uses_single_s3_core_and_preserves_candidate_kind() -> None:
    base = pair(width=8, height=6)
    left = replace(base.left, capture_id=None, frame_id=19, receive_timestamp_ns=555_000)
    right = replace(base.right, capture_id=None, receive_timestamp_ns=556_000)
    timestamp_pair = TimestampStereoFramePair(
        identity=TimestampStereoPairIdentity(1, 1, 19, right.frame_id, 555_000, 556_000, 1_000),
        left=left,
        right=right,
    )
    assert validate_stereo_pair_geometry(timestamp_pair, calibration(width=8, height=6)) is None
    observation = StereoTargetObservation(
        target=TargetRef(CameraRole.STEREO_LEFT, 1, 7),
        intent_revision=5,
        pair=timestamp_pair,
        roi=BBox(1, 1, 4, 3),
        pairing_kind=StereoPairingKind.TIMESTAMP,
    )
    worker = StereoDistanceWorker(
        calibration=calibration(width=8, height=6),
        compute=lambda _observation: _worker_estimate(12.4),
    )
    worker.start()
    try:
        assert worker.submit_observation(observation)
        snapshot = worker.results.wait_for_revision(0, 1.0)
        assert snapshot is not None
        candidate = snapshot.value
        assert candidate is not None
        assert candidate.pairing_kind is StereoPairingKind.TIMESTAMP
        assert candidate.capture_id is None
        assert candidate.measured_timestamp_ns == 555_000
        assert candidate.source_frame_id == 19
        assert candidate.intent_revision == 5
    finally:
        worker.shutdown()


def test_observation_rejects_pair_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="pairing_kind"):
        StereoTargetObservation(
            target=TargetRef(CameraRole.STEREO_LEFT, 1, 7),
            intent_revision=2,
            pair=pair(width=8, height=6),
            roi=BBox(1, 1, 4, 3),
            pairing_kind=StereoPairingKind.TIMESTAMP,
        )
