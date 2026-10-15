import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import Candidate
from navmin.vision.processors.vt11.acquisition.roi import RoiRect
from navmin.vision.processors.vt11.identity.scorer import (
    AcquireScoringConfig,
    CandidateFeatureVector,
    extract_candidate_features,
    score_candidates,
    score_feature_vector,
)


def _candidate(cid: int, x: int, y: int, spatial: float, area: int = 64) -> Candidate:
    return Candidate(
        candidate_id=cid,
        bbox=RoiRect(x=x, y=y, width=8, height=8),
        center_x=x + 3.5,
        center_y=y + 3.5,
        area_px=area,
        mean_response=40.0,
        peak_response=90.0,
        distance_to_click_px=5.0 + cid,
        spatial_prior=spatial,
        detector_strength=0.7,
        rank_score=spatial,
    )


def test_feature_extraction_returns_bounded_scores():
    frame = np.full((100, 120, 3), 100, dtype=np.uint8)
    frame[40:48, 50:58] = 230
    c = _candidate(1, 50, 40, 0.9)
    f = extract_candidate_features(frame, c)
    for name in ("spatial", "appearance", "edge", "projection", "contrast", "geometry"):
        assert 0.0 <= getattr(f, name) <= 1.0
    assert f.appearance > 0.2
    assert f.contrast > 0.2


def test_nearest_mode_is_exact_spatial_prior():
    f = CandidateFeatureVector(1, 0.73, 0.1, 0.2, 0.3, 0.4, 0.5)
    assert score_feature_vector(f, "nearest") == pytest.approx(0.73)


def test_individual_modes_use_corresponding_feature():
    f = CandidateFeatureVector(1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    assert score_feature_vector(f, "appearance") == pytest.approx(0.2)
    assert score_feature_vector(f, "edge") == pytest.approx(0.3)
    assert score_feature_vector(f, "projection") == pytest.approx(0.4)
    assert score_feature_vector(f, "contrast") == pytest.approx(0.5)
    assert score_feature_vector(f, "geometry") == pytest.approx(0.6)


def test_combined_uses_spatial_plus_normalized_m2_core():
    f = CandidateFeatureVector(1, 0.8, 0.6, 0.4, 0.2, 1.0, 1.0)
    iw = {"appearance": 0.603654, "edge": 0.288078, "projection": 0.108268}
    core = 0.603654 * 0.6 + 0.288078 * 0.4 + 0.108268 * 0.2
    expected = 0.45 * 0.8 + 0.55 * core
    assert score_feature_vector(f, "combined", identity_weights=iw, combined_spatial_weight=0.45) == pytest.approx(expected)


def test_custom_weights_are_normalized_automatically():
    f = CandidateFeatureVector(1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1)
    weights = {"spatial": 2.0, "appearance": 1.0, "edge": 0, "projection": 0, "contrast": 0, "geometry": 0}
    expected = (2.0 * 0.9 + 1.0 * 0.2) / 3.0
    assert score_feature_vector(f, "custom", custom_weights=weights) == pytest.approx(expected)


def test_nearest_scoring_ranks_candidates_and_reports_high_gap():
    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    c1 = _candidate(1, 30, 30, 0.92)
    c2 = _candidate(2, 55, 30, 0.40)
    cfg = AcquireScoringConfig(mode="nearest", confidence_gap_low=0.05, confidence_gap_high=0.12)
    result = score_candidates(frame, [c2, c1], config=cfg)
    assert result.best.candidate.candidate_id == 1
    assert result.candidates[0].final_rank == 1
    assert result.confidence_gap == pytest.approx(0.52)
    assert result.confidence_state == "HIGH"


def test_small_gap_is_low_confidence():
    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    c1 = _candidate(1, 30, 30, 0.90)
    c2 = _candidate(2, 55, 30, 0.88)
    cfg = AcquireScoringConfig(mode="nearest", confidence_gap_low=0.05, confidence_gap_high=0.12)
    result = score_candidates(frame, [c1, c2], config=cfg)
    assert result.confidence_gap == pytest.approx(0.02)
    assert result.confidence_state == "LOW"


def test_no_candidates_is_explicit_state():
    frame = np.full((40, 50, 3), 100, dtype=np.uint8)
    result = score_candidates(frame, [], config=AcquireScoringConfig(mode="combined"))
    assert result.best is None
    assert result.confidence_gap == 0.0
    assert result.confidence_state == "NO_CANDIDATES"


def test_invalid_custom_weights_rejected():
    with pytest.raises(ValueError):
        AcquireScoringConfig(mode="custom", custom_weights={"spatial": 0.0}).validate()
