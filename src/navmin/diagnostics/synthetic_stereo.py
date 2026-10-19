"""Deterministic rectified stereo scenes for distance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import cv2
import numpy as np

from navmin.calibration import StereoCalibration
from navmin.contracts import BBox
from navmin.vision.stereo_distance import StereoPoint3D, reproject_disparity_point

DEFAULT_SYNTHETIC_WIDTH = 1296
DEFAULT_SYNTHETIC_HEIGHT = 976
DEFAULT_SYNTHETIC_FPS = 20

_IDENTITY_3X3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_ZERO_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class SyntheticStereoTargetSpec:
    track_id: int
    distance_m: float
    speed_px_s: float
    center_y_px: int
    start_center_x_px: int


@dataclass(frozen=True, slots=True)
class SyntheticStereoTargetState:
    spec: SyntheticStereoTargetSpec
    left_bbox: BBox
    right_bbox: BBox
    disparity_px: int
    expected_distance_m: float


@dataclass(frozen=True, slots=True)
class SyntheticStereoFrame:
    frame_index: int
    left: np.ndarray
    right: np.ndarray
    targets: tuple[SyntheticStereoTargetState, ...]


DEFAULT_WORKING_RANGE_TARGETS = (
    SyntheticStereoTargetSpec(1, 100.0, 30.0, 190, 430),
    SyntheticStereoTargetSpec(2, 150.0, 70.0, 488, 470),
    SyntheticStereoTargetSpec(3, 200.0, 120.0, 786, 510),
)


def rectified_synthetic_calibration(reference: StereoCalibration) -> StereoCalibration:
    """Keep accepted Q/P geometry while making correction maps identity-like.

    Synthetic frames are already rectified. Using projection intrinsics as input
    intrinsics with zero distortion and identity rectification prevents a second
    physical-camera warp while preserving the real accepted stereo geometry.
    """
    left_k = tuple(tuple(row[:3]) for row in reference.P1)
    right_k = tuple(tuple(row[:3]) for row in reference.P2)
    return replace(
        reference,
        K_left=left_k,
        D_left=_ZERO_DISTORTION,
        K_right=right_k,
        D_right=_ZERO_DISTORTION,
        R=_IDENTITY_3X3,
        R1=_IDENTITY_3X3,
        R2=_IDENTITY_3X3,
    )


class SyntheticWorkingRangeStereoScene:
    """Three textured rectified targets at 100/150/200 m.

    The Left and Right targets share the same deterministic texture. Right is a
    horizontal shift of Left by the target disparity. In moving mode both views
    use the same pixel velocity, so disparity remains constant for each target.
    """

    _PATCH_WIDTH = 260
    _PATCH_HEIGHT = 160
    _ROI_MARGIN = 25

    def __init__(
        self,
        calibration: StereoCalibration,
        *,
        fps: int = DEFAULT_SYNTHETIC_FPS,
        moving: bool = False,
        targets: tuple[SyntheticStereoTargetSpec, ...] = DEFAULT_WORKING_RANGE_TARGETS,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if calibration.image_width != DEFAULT_SYNTHETIC_WIDTH or calibration.image_height != DEFAULT_SYNTHETIC_HEIGHT:
            raise ValueError("working-range scene requires 1296x976 calibration")
        self.calibration = calibration
        self.fps = fps
        self.moving = moving
        self.targets = targets
        self._textures = {spec.track_id: self._make_texture(spec) for spec in targets}
        self._disparities = {
            spec.track_id: self._disparity_for_distance(spec) for spec in targets
        }

    def render(self, frame_index: int) -> SyntheticStereoFrame:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        left = self._background()
        right = self._background()
        states: list[SyntheticStereoTargetState] = []
        elapsed_s = frame_index / self.fps

        for spec in self.targets:
            movement = spec.speed_px_s * elapsed_s if self.moving else 0.0
            center_x = round(spec.start_center_x_px + movement)
            # Keep long-running diagnostics bounded without changing speed sign in
            # the short acceptance window used by the tool/tests.
            center_x = self._wrap_center_x(center_x)
            left_x = center_x - self._PATCH_WIDTH // 2
            top = spec.center_y_px - self._PATCH_HEIGHT // 2
            disparity = self._disparities[spec.track_id]
            right_x = left_x - disparity
            self._paste(left, self._textures[spec.track_id], left_x, top)
            self._paste(right, self._textures[spec.track_id], right_x, top)

            left_bbox = BBox(
                x=left_x + self._ROI_MARGIN,
                y=top + self._ROI_MARGIN,
                width=self._PATCH_WIDTH - 2 * self._ROI_MARGIN,
                height=self._PATCH_HEIGHT - 2 * self._ROI_MARGIN,
            )
            right_bbox = BBox(
                x=right_x + self._ROI_MARGIN,
                y=left_bbox.y,
                width=left_bbox.width,
                height=left_bbox.height,
            )
            point = reproject_disparity_point(
                self.calibration,
                x_px=left_bbox.x + left_bbox.width / 2.0,
                y_px=left_bbox.y + left_bbox.height / 2.0,
                disparity_px=float(disparity),
                length_scale_to_m=self.calibration.length_scale_to_m,
            )
            if not isinstance(point, StereoPoint3D):
                raise TypeError("calibration cannot reproject synthetic target disparity")
            states.append(
                SyntheticStereoTargetState(
                    spec=spec,
                    left_bbox=left_bbox,
                    right_bbox=right_bbox,
                    disparity_px=disparity,
                    expected_distance_m=point.distance_m,
                )
            )
            self._annotate(left, spec, left_x, top, disparity)
            self._annotate(right, spec, right_x, top, disparity)

        left.flags.writeable = False
        right.flags.writeable = False
        return SyntheticStereoFrame(frame_index, left, right, tuple(states))

    def _disparity_for_distance(self, spec: SyntheticStereoTargetSpec) -> int:
        x = float(spec.start_center_x_px)
        y = float(spec.center_y_px)
        target = spec.distance_m
        low = 0.25
        high = 511.0
        for _ in range(64):
            mid = (low + high) / 2.0
            point = reproject_disparity_point(
                self.calibration,
                x_px=x,
                y_px=y,
                disparity_px=mid,
                length_scale_to_m=self.calibration.length_scale_to_m,
            )
            if not isinstance(point, StereoPoint3D) or not isfinite(point.distance_m):
                raise ValueError("calibration cannot represent synthetic target")
            if point.distance_m > target:
                low = mid
            else:
                high = mid
        disparity = max(1, round((low + high) / 2.0))
        return disparity

    def _wrap_center_x(self, center_x: int) -> int:
        minimum = 360
        maximum = self.calibration.image_width - 160
        span = maximum - minimum
        if center_x <= maximum:
            return max(minimum, center_x)
        return minimum + (center_x - minimum) % span

    def _background(self) -> np.ndarray:
        frame = np.full(
            (self.calibration.image_height, self.calibration.image_width, 3),
            24,
            dtype=np.uint8,
        )
        for x in range(0, self.calibration.image_width, 64):
            cv2.line(frame, (x, 0), (x, self.calibration.image_height - 1), (31, 31, 31), 1)
        for y in range(0, self.calibration.image_height, 64):
            cv2.line(frame, (0, y), (self.calibration.image_width - 1, y), (31, 31, 31), 1)
        return frame

    def _make_texture(self, spec: SyntheticStereoTargetSpec) -> np.ndarray:
        rng = np.random.default_rng(10_000 + spec.track_id)
        mono = rng.integers(
            0,
            256,
            (self._PATCH_HEIGHT, self._PATCH_WIDTH),
            dtype=np.uint8,
        )
        patch = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(
            patch,
            (0, 0),
            (self._PATCH_WIDTH - 1, self._PATCH_HEIGHT - 1),
            (255, 255, 255),
            3,
        )
        return patch

    @staticmethod
    def _paste(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
        height, width = patch.shape[:2]
        if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
            raise ValueError("synthetic target moved outside frame")
        frame[y : y + height, x : x + width] = patch

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        spec: SyntheticStereoTargetSpec,
        x: int,
        y: int,
        disparity: int,
    ) -> None:
        cv2.putText(
            frame,
            f"T{spec.track_id} {spec.distance_m:.0f}m {spec.speed_px_s:.0f}px/s d={disparity}px",
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


__all__ = [
    "DEFAULT_SYNTHETIC_FPS",
    "DEFAULT_SYNTHETIC_HEIGHT",
    "DEFAULT_SYNTHETIC_WIDTH",
    "DEFAULT_WORKING_RANGE_TARGETS",
    "SyntheticStereoFrame",
    "SyntheticStereoTargetSpec",
    "SyntheticStereoTargetState",
    "SyntheticWorkingRangeStereoScene",
    "rectified_synthetic_calibration",
]
