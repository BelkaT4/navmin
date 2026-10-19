from __future__ import annotations

import numpy as np
import pytest

from navmin.calibration import load_stereo_calibration
from navmin.contracts import CameraRole, FramePacket
from navmin.diagnostics.synthetic_stereo import (
    DEFAULT_WORKING_RANGE_TARGETS,
    SyntheticWorkingRangeStereoScene,
    rectified_synthetic_calibration,
)
from navmin.vision.stereo_distance import (
    StereoDistanceEstimate,
    StereoSgbmSettings,
    compute_sgbm_disparity,
    estimate_distance_from_disparity_roi,
)
from navmin.vision.stereo_pairing import StereoFramePair, StereoPairIdentity


def _calibration():
    return rectified_synthetic_calibration(
        load_stereo_calibration("config/calibration/stereo.json")
    )


def test_working_range_scene_uses_expected_distances_speeds_and_resolution() -> None:
    scene = SyntheticWorkingRangeStereoScene(_calibration(), moving=True)
    first = scene.render(0)
    later = scene.render(20)

    assert first.left.shape == (976, 1296, 3)
    assert first.right.shape == (976, 1296, 3)
    assert tuple(target.spec.distance_m for target in first.targets) == (100.0, 150.0, 200.0)
    assert tuple(target.spec.speed_px_s for target in first.targets) == (30.0, 70.0, 120.0)
    assert tuple(target.disparity_px for target in first.targets) == (55, 37, 28)
    for before, after in zip(first.targets, later.targets, strict=True):
        assert after.left_bbox.x - before.left_bbox.x == round(before.spec.speed_px_s)
        assert after.right_bbox.x - before.right_bbox.x == round(before.spec.speed_px_s)
        assert before.left_bbox.x - before.right_bbox.x == before.disparity_px
        assert after.left_bbox.x - after.right_bbox.x == after.disparity_px


def test_static_scene_production_sgbm_recovers_all_three_ranges() -> None:
    calibration = _calibration()
    rendered = SyntheticWorkingRangeStereoScene(calibration, moving=False).render(0)
    disparity = compute_sgbm_disparity(
        rendered.left,
        rendered.right,
        StereoSgbmSettings(),
    )
    pair = StereoFramePair(
        identity=StereoPairIdentity(1, 1, 1),
        left=FramePacket(CameraRole.STEREO_LEFT, 1, 1, 1, 1_000, rendered.left),
        right=FramePacket(CameraRole.STEREO_RIGHT, 1, 1, 1, 1_000, rendered.right),
    )

    estimates = []
    for target in rendered.targets:
        result = estimate_distance_from_disparity_roi(
            pair,
            calibration,
            disparity,
            target.left_bbox,
            length_scale_to_m=calibration.length_scale_to_m,
            min_valid_pixel_count=500,
        )
        assert isinstance(result, StereoDistanceEstimate)
        estimates.append(result.distance_m)
        assert result.median_disparity_px == pytest.approx(target.disparity_px, abs=1.0)
        assert result.distance_m == pytest.approx(target.spec.distance_m, rel=0.06)

    assert estimates[0] < estimates[1] < estimates[2]
    assert np.isfinite(disparity).any()


def test_rectified_synthetic_calibration_preserves_real_distance_geometry() -> None:
    reference = load_stereo_calibration("config/calibration/stereo.json")
    synthetic = rectified_synthetic_calibration(reference)

    assert synthetic.Q == reference.Q
    assert synthetic.P1 == reference.P1
    assert synthetic.P2 == reference.P2
    assert synthetic.translation_unit == reference.translation_unit
    assert synthetic.D_left == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert synthetic.D_right == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert DEFAULT_WORKING_RANGE_TARGETS[0].distance_m == 100.0
