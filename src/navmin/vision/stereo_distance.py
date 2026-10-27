"""Pure stereo geometry contracts used by the distance algorithm core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite, log, sqrt

import cv2
import numpy as np

from navmin.calibration import StereoCalibration
from navmin.contracts import BBox, TrackedObject
from navmin.vision.stereo_pairing import StereoFramePair
from navmin.vision.stereo_timestamp_pairing import TimestampStereoFramePair

type StereoDistancePair = StereoFramePair | TimestampStereoFramePair


class StereoDistanceFailureReason(Enum):
    """Machine-readable reasons a stereo distance operation was rejected."""

    INVALID_LENGTH_SCALE = "invalid_length_scale"
    INVALID_PIXEL_COORDINATE = "invalid_pixel_coordinate"
    INVALID_DISPARITY = "invalid_disparity"
    INVALID_DISPARITY_MAP = "invalid_disparity_map"
    INVALID_MIN_VALID_PIXELS = "invalid_min_valid_pixels"
    INVALID_FRAME_SHAPE = "invalid_frame_shape"
    FRAME_SHAPE_MISMATCH = "frame_shape_mismatch"
    CALIBRATION_RESOLUTION_MISMATCH = "calibration_resolution_mismatch"
    NON_FINITE_REPROJECTION = "non_finite_reprojection"
    NON_POSITIVE_DISTANCE = "non_positive_distance"
    ROI_OUT_OF_FRAME = "roi_out_of_frame"
    EMPTY_ROI = "empty_roi"
    INSUFFICIENT_VALID_PIXELS = "insufficient_valid_pixels"
    DETECTOR_NO_MATCH = "detector_no_match"
    DETECTOR_AMBIGUOUS = "detector_ambiguous"


class StereoDistanceMethod(Enum):
    """Pure source/method used to derive an internal stereo estimate."""

    DISPARITY_MAP_ROI_MEDIAN = "disparity_map_roi_median"
    DETECTOR_CENTER_DISPARITY = "detector_center_disparity"


@dataclass(frozen=True, slots=True)
class StereoSgbmSettings:
    """Narrow settings retained from the previous working NavMin SGBM core."""

    min_disparity: int = 0
    num_disparities: int = 512
    block_size: int = 7
    uniqueness_ratio: int = 8
    disp12_max_diff: int = 1
    speckle_window_size: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 31
    mode: str = "SGBM_3WAY"
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    processing_scale: float = 0.5


@dataclass(frozen=True, slots=True)
class StereoDistanceEstimate:
    """Successful robust distance estimate from one disparity-map ROI."""

    distance_m: float
    median_disparity_px: float
    valid_pixel_count: int
    valid_pixel_fraction: float
    method: StereoDistanceMethod


@dataclass(frozen=True, slots=True)
class StereoDistanceFailure:
    """Explicit fail-safe outcome for invalid stereo input or geometry."""

    reason: StereoDistanceFailureReason


@dataclass(frozen=True, slots=True)
class StereoPoint3D:
    """One finite Q-reprojected point expressed in metres."""

    x_m: float
    y_m: float
    z_m: float
    distance_m: float


StereoReprojectionOutcome = StereoPoint3D | StereoDistanceFailure
StereoDistanceEstimateOutcome = StereoDistanceEstimate | StereoDistanceFailure


def estimate_distance_from_stereo_pair(
    pair: StereoDistancePair,
    calibration: StereoCalibration,
    settings: StereoSgbmSettings,
    *,
    length_scale_to_m: float,
    min_valid_pixel_count: int,
    roi: BBox | None = None,
) -> StereoDistanceEstimateOutcome:
    """Compute full-frame SGBM once and aggregate distance in a Left ROI.

    Both supported stereo pair types contain corrected/rectified working frames from
    the accepted S1/S2 pipeline. No pairing, rectification, resize-to-calibration,
    or stale-distance fallback is performed here. ``roi=None`` means the full
    rectified Left working frame.
    """
    geometry_failure = validate_stereo_pair_geometry(pair, calibration)
    if geometry_failure is not None:
        return geometry_failure

    disparity_map_px = compute_sgbm_disparity(
        pair.left.image,
        pair.right.image,
        settings,
    )
    frame_height, frame_width = pair.left.image.shape[:2]
    target_roi = roi or BBox(x=0, y=0, width=frame_width, height=frame_height)
    return estimate_distance_from_disparity_roi(
        pair,
        calibration,
        disparity_map_px,
        target_roi,
        length_scale_to_m=length_scale_to_m,
        min_valid_pixel_count=min_valid_pixel_count,
    )


def estimate_distance_from_detector_pair(
    pair: StereoDistancePair,
    calibration: StereoCalibration,
    left_roi: BBox,
    right_detections: tuple[TrackedObject, ...],
    *,
    length_scale_to_m: float,
) -> StereoDistanceEstimateOutcome:
    """Match independent detections by rectified epipolar geometry."""
    geometry_failure = validate_stereo_pair_geometry(pair, calibration)
    if geometry_failure is not None:
        return geometry_failure
    if left_roi.width <= 0 or left_roi.height <= 0:
        return StereoDistanceFailure(StereoDistanceFailureReason.EMPTY_ROI)
    frame_height, frame_width = pair.left.image.shape[:2]
    if (
        left_roi.x < 0
        or left_roi.y < 0
        or left_roi.x + left_roi.width > frame_width
        or left_roi.y + left_roi.height > frame_height
    ):
        return StereoDistanceFailure(StereoDistanceFailureReason.ROI_OUT_OF_FRAME)
    left_x = left_roi.x + left_roi.width / 2.0
    left_y = left_roi.y + left_roi.height / 2.0
    candidates: list[tuple[float, float]] = []
    for detection in right_detections:
        bbox = detection.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        right_x = bbox.x + bbox.width / 2.0
        right_y = bbox.y + bbox.height / 2.0
        disparity = left_x - right_x
        vertical_error = abs(left_y - right_y)
        if disparity <= 0.0:
            continue
        if vertical_error > max(8.0, 0.3 * max(left_roi.height, bbox.height)):
            continue
        height_ratio = bbox.height / left_roi.height
        if not 0.5 <= height_ratio <= 2.0:
            continue
        score = vertical_error / max(left_roi.height, bbox.height) + abs(
            log(height_ratio)
        )
        candidates.append((score, disparity))
    if not candidates:
        return StereoDistanceFailure(StereoDistanceFailureReason.DETECTOR_NO_MATCH)
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.05:
        return StereoDistanceFailure(StereoDistanceFailureReason.DETECTOR_AMBIGUOUS)
    best_disparity = candidates[0][1]
    point = reproject_disparity_point(
        calibration,
        x_px=left_x,
        y_px=left_y,
        disparity_px=best_disparity,
        length_scale_to_m=length_scale_to_m,
    )
    if isinstance(point, StereoDistanceFailure):
        return point
    return StereoDistanceEstimate(
        distance_m=point.distance_m,
        median_disparity_px=best_disparity,
        valid_pixel_count=1,
        valid_pixel_fraction=1.0,
        method=StereoDistanceMethod.DETECTOR_CENTER_DISPARITY,
    )


def compute_sgbm_disparity(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    settings: StereoSgbmSettings,
) -> np.ndarray:
    """Compute full-frame left disparity in original-frame pixel units.

    This is the previous working NavMin SGBM path with only naming/scope
    adaptation. OpenCV fixed-point disparity is converted with ``/ 16``.
    Invalid non-positive disparities are represented as ``NaN``. When
    processing is downscaled, disparity is resized back to the source frame and
    divided by the scale so values remain expressed in original-frame pixels.
    """
    _validate_sgbm_image_pair(left_bgr, right_bgr)
    normalized = _normalized_sgbm_settings(settings)
    scale = normalized.processing_scale

    if scale < 0.999:
        left_work = cv2.resize(
            left_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        right_work = cv2.resize(
            right_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        left_work, right_work = left_bgr, right_bgr

    gray_left = _sgbm_gray(left_work, normalized)
    gray_right = _sgbm_gray(right_work, normalized)

    scaled_min_disparity = round(normalized.min_disparity * scale)
    scaled_num_disparities = max(
        16,
        ceil((normalized.num_disparities * scale) / 16.0) * 16,
    )
    max_usable = max(16, (gray_left.shape[1] // 16 - 1) * 16)
    scaled_num_disparities = min(scaled_num_disparities, max_usable)

    block_size = normalized.block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=scaled_min_disparity,
        numDisparities=scaled_num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=normalized.disp12_max_diff,
        preFilterCap=normalized.pre_filter_cap,
        uniquenessRatio=normalized.uniqueness_ratio,
        speckleWindowSize=normalized.speckle_window_size,
        speckleRange=normalized.speckle_range,
        mode=_sgbm_mode(normalized.mode),
    )
    raw = matcher.compute(gray_left, gray_right)
    disparity = raw.astype(np.float32) / 16.0

    invalid_threshold = max(0.0, float(scaled_min_disparity))
    disparity[~np.isfinite(disparity) | (disparity <= invalid_threshold)] = np.nan
    if scale < 0.999:
        disparity = cv2.resize(
            disparity,
            (left_bgr.shape[1], left_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        disparity = disparity / scale
    return disparity


def _normalized_sgbm_settings(settings: StereoSgbmSettings) -> StereoSgbmSettings:
    """Preserve the previous NavMin SGBM normalization semantics."""
    if not isinstance(settings, StereoSgbmSettings):
        raise TypeError("settings must be StereoSgbmSettings")
    return StereoSgbmSettings(
        min_disparity=int(settings.min_disparity),
        num_disparities=max(
            16,
            ceil(max(16, settings.num_disparities) / 16.0) * 16,
        ),
        block_size=max(3, int(settings.block_size) | 1),
        uniqueness_ratio=max(0, int(settings.uniqueness_ratio)),
        disp12_max_diff=int(settings.disp12_max_diff),
        speckle_window_size=max(0, int(settings.speckle_window_size)),
        speckle_range=max(0, int(settings.speckle_range)),
        pre_filter_cap=max(1, int(settings.pre_filter_cap)),
        mode=settings.mode,
        clahe_enabled=bool(settings.clahe_enabled),
        clahe_clip_limit=max(0.1, float(settings.clahe_clip_limit)),
        clahe_tile_grid_size=max(2, int(settings.clahe_tile_grid_size)),
        processing_scale=min(1.0, max(0.25, float(settings.processing_scale))),
    )


def _sgbm_gray(image: np.ndarray, settings: StereoSgbmSettings) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if settings.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=settings.clahe_clip_limit,
            tileGridSize=(
                settings.clahe_tile_grid_size,
                settings.clahe_tile_grid_size,
            ),
        )
        gray = clahe.apply(gray)
    return gray


def _sgbm_mode(name: str) -> int:
    if name == "SGBM":
        return cv2.STEREO_SGBM_MODE_SGBM
    if name == "HH":
        return cv2.STEREO_SGBM_MODE_HH
    return cv2.STEREO_SGBM_MODE_SGBM_3WAY


def _validate_sgbm_image_pair(left_bgr: np.ndarray, right_bgr: np.ndarray) -> None:
    for name, image in (("left_bgr", left_bgr), ("right_bgr", right_bgr)):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError(f"{name} must be a non-empty uint8 HxWx3 BGR image")
    if left_bgr.shape != right_bgr.shape:
        raise ValueError("left and right SGBM images must have identical shape")


def validate_stereo_pair_geometry(
    pair: StereoDistancePair,
    calibration: StereoCalibration,
) -> StereoDistanceFailure | None:
    """Validate corrected stereo frame shapes against each other and calibration."""
    left_shape = pair.left.image.shape
    right_shape = pair.right.image.shape

    if not _is_working_frame_shape(left_shape) or not _is_working_frame_shape(
        right_shape
    ):
        return StereoDistanceFailure(StereoDistanceFailureReason.INVALID_FRAME_SHAPE)
    if left_shape != right_shape:
        return StereoDistanceFailure(StereoDistanceFailureReason.FRAME_SHAPE_MISMATCH)

    expected_shape = (calibration.image_height, calibration.image_width, 3)
    if left_shape != expected_shape:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.CALIBRATION_RESOLUTION_MISMATCH
        )
    return None


def estimate_distance_from_disparity_roi(
    pair: StereoDistancePair,
    calibration: StereoCalibration,
    disparity_map_px: np.ndarray,
    roi: BBox,
    *,
    length_scale_to_m: float,
    min_valid_pixel_count: int,
) -> StereoDistanceEstimateOutcome:
    """Robustly aggregate a precomputed disparity map inside a strict Left ROI.

    The ROI is expressed in rectified Stereo Left working-frame coordinates and
    must be fully inside the frame; this function never silently clips it.
    Finite positive disparities are reprojected through ``Q`` and the median 3-D
    Euclidean distance is returned in metres. Invalid disparity pixels are
    excluded from the sample set and remain visible through ``valid_pixel_*``
    diagnostics.
    """
    geometry_failure = validate_stereo_pair_geometry(pair, calibration)
    if geometry_failure is not None:
        return geometry_failure

    scale = _finite_positive_float(length_scale_to_m)
    if scale is None:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.INVALID_LENGTH_SCALE
        )

    if (
        isinstance(min_valid_pixel_count, bool)
        or not isinstance(min_valid_pixel_count, int)
        or min_valid_pixel_count <= 0
    ):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.INVALID_MIN_VALID_PIXELS
        )

    frame_height, frame_width = pair.left.image.shape[:2]
    if roi.width <= 0 or roi.height <= 0:
        return StereoDistanceFailure(StereoDistanceFailureReason.EMPTY_ROI)
    if (
        roi.x < 0
        or roi.y < 0
        or roi.x + roi.width > frame_width
        or roi.y + roi.height > frame_height
    ):
        return StereoDistanceFailure(StereoDistanceFailureReason.ROI_OUT_OF_FRAME)

    if (
        not isinstance(disparity_map_px, np.ndarray)
        or disparity_map_px.ndim != 2
        or disparity_map_px.shape != (frame_height, frame_width)
        or not (
            np.issubdtype(disparity_map_px.dtype, np.integer)
            or np.issubdtype(disparity_map_px.dtype, np.floating)
        )
    ):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.INVALID_DISPARITY_MAP
        )

    roi_disparity = disparity_map_px[
        roi.y : roi.y + roi.height,
        roi.x : roi.x + roi.width,
    ].astype(np.float64, copy=False)
    valid_mask = np.isfinite(roi_disparity) & (roi_disparity > 0.0)
    valid_pixel_count = int(np.count_nonzero(valid_mask))
    roi_pixel_count = roi.width * roi.height
    valid_pixel_fraction = valid_pixel_count / roi_pixel_count
    if valid_pixel_count < min_valid_pixel_count:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.INSUFFICIENT_VALID_PIXELS
        )

    local_y, local_x = np.nonzero(valid_mask)
    disparities = roi_disparity[valid_mask]
    x_px = local_x.astype(np.float64) + float(roi.x)
    y_px = local_y.astype(np.float64) + float(roi.y)
    homogeneous = np.column_stack(
        (x_px, y_px, disparities, np.ones(valid_pixel_count, dtype=np.float64))
    )
    q = np.asarray(calibration.Q, dtype=np.float64)
    projected = homogeneous @ q.T
    if not np.all(np.isfinite(projected)):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )

    w = projected[:, 3]
    if np.any(w == 0.0):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )

    coordinates_m = (projected[:, :3] / w[:, None]) * scale
    distances_m = np.linalg.norm(coordinates_m, axis=1)
    if not np.all(np.isfinite(coordinates_m)) or not np.all(
        np.isfinite(distances_m)
    ):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )
    if np.any(distances_m <= 0.0):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_POSITIVE_DISTANCE
        )

    return StereoDistanceEstimate(
        distance_m=float(np.median(distances_m)),
        median_disparity_px=float(np.median(disparities)),
        valid_pixel_count=valid_pixel_count,
        valid_pixel_fraction=valid_pixel_fraction,
        method=StereoDistanceMethod.DISPARITY_MAP_ROI_MEDIAN,
    )


def reproject_disparity_point(
    calibration: StereoCalibration,
    *,
    x_px: float,
    y_px: float,
    disparity_px: float,
    length_scale_to_m: float,
) -> StereoReprojectionOutcome:
    """Reproject one rectified-left pixel/disparity sample through ``Q``.

    ``StereoCalibration`` schema 1 does not encode the physical length unit of
    its geometry. ``length_scale_to_m`` is therefore mandatory and explicitly
    converts Q's calibration-space coordinates to metres.
    """
    scale = _finite_positive_float(length_scale_to_m)
    if scale is None:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.INVALID_LENGTH_SCALE
        )

    x = _finite_float(x_px)
    y = _finite_float(y_px)
    if x is None or y is None:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.INVALID_PIXEL_COORDINATE
        )

    disparity = _finite_float(disparity_px)
    if disparity is None or disparity <= 0.0:
        return StereoDistanceFailure(StereoDistanceFailureReason.INVALID_DISPARITY)

    q = calibration.Q
    homogeneous = (x, y, disparity, 1.0)
    projected = tuple(
        sum(q[row][column] * homogeneous[column] for column in range(4))
        for row in range(4)
    )
    if not all(isfinite(value) for value in projected):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )

    w = projected[3]
    if w == 0.0:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )

    raw_x = projected[0] / w
    raw_y = projected[1] / w
    raw_z = projected[2] / w
    if not all(isfinite(value) for value in (raw_x, raw_y, raw_z)):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )

    x_m = raw_x * scale
    y_m = raw_y * scale
    z_m = raw_z * scale
    distance_m = sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
    if not all(isfinite(value) for value in (x_m, y_m, z_m, distance_m)):
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_FINITE_REPROJECTION
        )
    if distance_m <= 0.0:
        return StereoDistanceFailure(
            StereoDistanceFailureReason.NON_POSITIVE_DISTANCE
        )

    return StereoPoint3D(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        distance_m=distance_m,
    )


def _is_working_frame_shape(shape: tuple[int, ...]) -> bool:
    return len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] == 3


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _finite_positive_float(value: object) -> float | None:
    result = _finite_float(value)
    if result is None or result <= 0.0:
        return None
    return result
