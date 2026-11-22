"""Canonical camera geometry for deterministic NavMin tracking simulation.

These calibrations describe ideal, known simulation cameras.  They are not
copied from site-specific hardware calibration and therefore remain stable
across Linux/Windows hosts and physical camera replacements.
"""

from __future__ import annotations

from dataclasses import replace
from math import radians, tan

from navmin.calibration import OverviewCalibration, StereoCalibration

SYNTHETIC_TRACKING_WIDTH = 1296
SYNTHETIC_TRACKING_HEIGHT = 976
SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG = 90.0
SYNTHETIC_TRACKING_STEREO_HFOV_DEG = 6.3
SYNTHETIC_TRACKING_STEREO_BASELINE_M = 0.46

_IDENTITY_3X3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_ZERO_PINHOLE_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)

# Overview uses the production OpenCV fisheye corrector.  Zero fisheye
# coefficients are not an identity map for a pinhole working frame, so the
# canonical in-process SIM uses the truncated tan(theta)/theta series already
# used by NavMin diagnostics to make raw->working correction identity-like.
# The *working* camera remains an ideal pinhole with the matrix below.
_FISHEYE_IDENTITY_DISTORTION = (
    1.0 / 3.0,
    2.0 / 15.0,
    17.0 / 315.0,
    62.0 / 2835.0,
)


def _focal_from_horizontal_fov(width: int, horizontal_fov_deg: float) -> float:
    return width / (2.0 * tan(radians(horizontal_fov_deg) / 2.0))


def _camera_matrix(*, width: int, height: int, focal_length_px: float):
    return (
        (focal_length_px, 0.0, (width - 1) / 2.0),
        (0.0, focal_length_px, (height - 1) / 2.0),
        (0.0, 0.0, 1.0),
    )


def synthetic_tracking_overview_calibration() -> OverviewCalibration:
    """Return the fixed ideal Overview working geometry used by normal SIM."""
    focal = _focal_from_horizontal_fov(
        SYNTHETIC_TRACKING_WIDTH,
        SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG,
    )
    matrix = _camera_matrix(
        width=SYNTHETIC_TRACKING_WIDTH,
        height=SYNTHETIC_TRACKING_HEIGHT,
        focal_length_px=focal,
    )
    return OverviewCalibration(
        schema_version=1,
        image_width=SYNTHETIC_TRACKING_WIDTH,
        image_height=SYNTHETIC_TRACKING_HEIGHT,
        K=matrix,
        D=_FISHEYE_IDENTITY_DISTORTION,
        new_camera_matrix=matrix,
    )


def synthetic_tracking_stereo_calibration() -> StereoCalibration:
    """Return fixed ideal rectified stereo geometry used by normal SIM."""
    focal = _focal_from_horizontal_fov(
        SYNTHETIC_TRACKING_WIDTH,
        SYNTHETIC_TRACKING_STEREO_HFOV_DEG,
    )
    matrix = _camera_matrix(
        width=SYNTHETIC_TRACKING_WIDTH,
        height=SYNTHETIC_TRACKING_HEIGHT,
        focal_length_px=focal,
    )
    cx = matrix[0][2]
    cy = matrix[1][2]
    baseline = SYNTHETIC_TRACKING_STEREO_BASELINE_M
    p1 = (
        (focal, 0.0, cx, 0.0),
        (0.0, focal, cy, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    p2 = (
        (focal, 0.0, cx, -focal * baseline),
        (0.0, focal, cy, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    q = (
        (1.0, 0.0, 0.0, -cx),
        (0.0, 1.0, 0.0, -cy),
        (0.0, 0.0, 0.0, focal),
        (0.0, 0.0, 1.0 / baseline, 0.0),
    )
    return StereoCalibration(
        schema_version=1,
        translation_unit="m",
        image_width=SYNTHETIC_TRACKING_WIDTH,
        image_height=SYNTHETIC_TRACKING_HEIGHT,
        K_left=matrix,
        D_left=_ZERO_PINHOLE_DISTORTION,
        K_right=matrix,
        D_right=_ZERO_PINHOLE_DISTORTION,
        R=_IDENTITY_3X3,
        T=(-baseline, 0.0, 0.0),
        R1=_IDENTITY_3X3,
        R2=_IDENTITY_3X3,
        P1=p1,
        P2=p2,
        Q=q,
    )


def mixed_stereo_calibration(
    physical: StereoCalibration,
    *,
    left_synthetic: bool,
    right_synthetic: bool,
) -> StereoCalibration:
    """Combine canonical SIM side geometry with one physical stereo side.

    Mixed ON/SIM stereo distance is disabled by launch policy.  This object only
    lets each active camera use its correct raw->working correction/model while
    retaining one existing ApplicationRuntime stereo-calibration boundary.
    """
    if not left_synthetic and not right_synthetic:
        return physical
    synthetic = synthetic_tracking_stereo_calibration()
    if left_synthetic and right_synthetic:
        return synthetic
    if (
        physical.image_width != synthetic.image_width
        or physical.image_height != synthetic.image_height
    ):
        raise ValueError(
            "mixed real/synthetic stereo requires physical calibration at "
            f"{synthetic.image_width}x{synthetic.image_height}"
        )

    changes: dict[str, object] = {}
    if left_synthetic:
        changes.update(
            K_left=synthetic.K_left,
            D_left=synthetic.D_left,
            R1=synthetic.R1,
            P1=synthetic.P1,
        )
    if right_synthetic:
        changes.update(
            K_right=synthetic.K_right,
            D_right=synthetic.D_right,
            R2=synthetic.R2,
            P2=synthetic.P2,
        )
    return replace(physical, **changes)


__all__ = [
    "SYNTHETIC_TRACKING_HEIGHT",
    "SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG",
    "SYNTHETIC_TRACKING_STEREO_BASELINE_M",
    "SYNTHETIC_TRACKING_STEREO_HFOV_DEG",
    "SYNTHETIC_TRACKING_WIDTH",
    "mixed_stereo_calibration",
    "synthetic_tracking_overview_calibration",
    "synthetic_tracking_stereo_calibration",
]
