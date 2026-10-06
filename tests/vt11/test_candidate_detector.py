import numpy as np
import cv2
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, config_from_dict, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection


def _frame_with_blobs() -> np.ndarray:
    frame = np.full((160, 220, 3), 110, dtype=np.uint8)
    # Target-like bright object close to click.
    cv2.rectangle(frame, (96, 74), (104, 82), (220, 220, 220), -1)
    # Stronger but farther distractor, still inside ROI.
    cv2.rectangle(frame, (145, 65), (156, 76), (255, 255, 255), -1)
    return frame


def _cfg(**kwargs) -> CandidateDetectorConfig:
    base = dict(
        blur_sigma_px=3.0,
        threshold_sigma=1.5,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1000,
        morph_close_px=3,
        spatial_sigma_px=30.0,
        spatial_weight=0.85,
        detector_weight=0.15,
    )
    base.update(kwargs)
    return CandidateDetectorConfig(**base)


def test_detects_multiple_candidates_and_spatial_prior_prefers_near_click():
    frame = _frame_with_blobs()
    selection = make_operator_selection(0, (100, 78), 70, frame.shape[1], frame.shape[0])
    result = detect_local_candidates(frame, selection, top_k=5, config=_cfg())
    assert len(result.candidates) >= 2
    best = result.candidates[0]
    assert best.distance_to_click_px < 8
    assert best.spatial_prior > result.candidates[1].spatial_prior
    assert best.candidate_id == 1


def test_detects_dark_local_contrast_object():
    frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    cv2.circle(frame, (80, 60), 5, (30, 30, 30), -1)
    selection = make_operator_selection(0, (82, 61), 40, 160, 120)
    result = detect_local_candidates(frame, selection, top_k=3, config=_cfg())
    assert result.candidates
    assert result.candidates[0].distance_to_click_px < 8


def test_candidate_coordinates_are_in_full_frame_space():
    frame = np.full((180, 260, 3), 100, dtype=np.uint8)
    cv2.rectangle(frame, (205, 130), (214, 139), (240, 240, 240), -1)
    selection = make_operator_selection(0, (210, 135), 35, 260, 180)
    result = detect_local_candidates(frame, selection, top_k=3, config=_cfg())
    assert result.candidates
    c = result.candidates[0]
    assert 200 <= c.center_x <= 220
    assert 125 <= c.center_y <= 145
    assert c.bbox.x >= selection.roi.x
    assert c.bbox.y >= selection.roi.y


def test_top_k_limits_shortlist():
    frame = np.full((180, 240, 3), 100, dtype=np.uint8)
    for x in (80, 100, 120, 140, 160):
        cv2.rectangle(frame, (x, 85), (x + 5, 90), (230, 230, 230), -1)
    selection = make_operator_selection(0, (120, 88), 70, 240, 180)
    result = detect_local_candidates(frame, selection, top_k=3, config=_cfg(morph_close_px=0))
    assert len(result.candidates) == 3
    assert [c.candidate_id for c in result.candidates] == [1, 2, 3]


def test_uniform_roi_returns_no_candidates():
    frame = np.full((100, 120, 3), 127, dtype=np.uint8)
    selection = make_operator_selection(0, (60, 50), 30, 120, 100)
    result = detect_local_candidates(frame, selection, top_k=5, config=_cfg())
    assert result.candidates == ()
    assert result.diagnostics.raw_components == 0


def test_config_validation_and_mapping():
    cfg = config_from_dict({"threshold_sigma": 2.0, "min_area_px": 4, "max_area_px": 40})
    assert cfg.threshold_sigma == pytest.approx(2.0)
    assert cfg.min_area_px == 4
    with pytest.raises(ValueError):
        CandidateDetectorConfig(min_area_px=20, max_area_px=10).validate()
