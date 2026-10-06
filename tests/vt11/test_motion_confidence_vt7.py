import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.tracking.motion_model import (
    MotionPoint,
    dynamic_search_radius,
    motion_uncertainty_timed,
    predict_position_timed,
    robust_velocity_timed,
)
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def test_timed_velocity_handles_skipped_frames():
    history = [
        MotionPoint(0, 10.0, 20.0),
        MotionPoint(2, 16.0, 24.0),
        MotionPoint(4, 22.0, 28.0),
    ]
    assert robust_velocity_timed(history, window_points=5) == (3.0, 2.0)
    est = predict_position_timed(history, target_frame_index=6, fallback=(0.0, 0.0), window_points=5)
    assert est.predicted_xy == (28.0, 32.0)
    assert est.speed_px_per_frame == pytest.approx((13.0) ** 0.5)
    assert est.uncertainty_px == pytest.approx(0.0)


def test_motion_uncertainty_reflects_velocity_variation():
    stable = [MotionPoint(i, 10 + 3 * i, 20 + 2 * i) for i in range(5)]
    varied = [
        MotionPoint(0, 10, 20),
        MotionPoint(1, 13, 22),
        MotionPoint(2, 19, 22),
        MotionPoint(3, 20, 27),
        MotionPoint(4, 26, 27),
    ]
    assert motion_uncertainty_timed(stable, window_points=5) == pytest.approx(0.0)
    assert motion_uncertainty_timed(varied, window_points=5) > 0.0


def test_dynamic_search_radius_grows_and_clamps():
    r0 = dynamic_search_radius(
        base_radius_px=70,
        min_radius_px=50,
        max_radius_px=150,
        speed_px_per_frame=4,
        uncertainty_px=2,
        miss_count=0,
        speed_gain=1.0,
        uncertainty_gain=2.0,
        miss_growth_px=20.0,
    )
    r1 = dynamic_search_radius(
        base_radius_px=70,
        min_radius_px=50,
        max_radius_px=150,
        speed_px_per_frame=4,
        uncertainty_px=2,
        miss_count=2,
        speed_gain=1.0,
        uncertainty_gain=2.0,
        miss_growth_px=20.0,
    )
    assert r1 > r0
    assert dynamic_search_radius(
        base_radius_px=70,
        min_radius_px=50,
        max_radius_px=90,
        speed_px_per_frame=100,
        uncertainty_px=100,
        miss_count=10,
        speed_gain=10,
        uncertainty_gain=10,
        miss_growth_px=100,
    ) == 90


def _frames(count=18, *, blank_after=None):
    frames = []
    for k in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if blank_after is None or k < blank_after:
            x = 38 + 3 * k
            cv2.rectangle(frame, (x, 53), (x + 9, 61), (225, 225, 225), -1)
            cv2.rectangle(frame, (x + 7, 55), (x + 12, 58), (150, 150, 150), -1)
        frames.append(frame)
    return frames


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1200,
        spatial_sigma_px=35.0,
    )


def _train(frames):
    sel = make_operator_selection(
        frame_index=0,
        click_xy=(43, 57),
        radius_px=65,
        frame_width=180,
        frame_height=120,
    )
    detected = detect_local_candidates(frames[0], sel, top_k=5, config=_detector_cfg())
    assert detected.candidates

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=0,
        initial_candidate=detected.candidates[0],
        roi_radius_px=65,
        detector_config=_detector_cfg(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "CONFIRMED"
    return result, reader


def test_vt7_tracking_records_prediction_confidence_and_dynamic_gate():
    frames = _frames(18)
    train, reader = _train(frames)
    accepted_train = [r for r in train.frames if r.selected is not None]
    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in accepted_train],
        initial_history_frames=[r.frame_index for r in accepted_train],
        detector_config=_detector_cfg(),
        config=TrackingConfig(
            search_radius_px=60,
            max_search_radius_px=120,
            top_k=5,
            min_confidence_gap=0.0,
        ),
    )
    assert result.status == "COMPLETED"
    assert result.frames
    assert all(f.search_radius_px >= 60 for f in result.frames)
    assert all(0.0 <= f.confidence <= 1.0 for f in result.frames)
    accepted = [f for f in result.frames if f.selected is not None]
    assert accepted
    assert all(f.prediction_error_px is not None for f in accepted)
    assert all(f.diagnostic_reason.startswith("ACCEPTED") for f in accepted)
    assert result.mean_confidence > 0.0
    assert result.median_prediction_error_px >= 0.0
    assert result.max_search_radius_used_px >= 60


def test_vt7_no_candidates_has_explicit_reason_and_gate_expands_after_miss():
    frames = _frames(14, blank_after=8)
    train, reader = _train(frames)
    accepted_train = [r for r in train.frames if r.selected is not None]
    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in accepted_train],
        initial_history_frames=[r.frame_index for r in accepted_train],
        detector_config=_detector_cfg(),
        config=TrackingConfig(
            search_radius_px=55,
            max_search_radius_px=140,
            top_k=5,
            min_confidence_gap=0.0,
            lost_after_misses=3,
            miss_radius_growth_px=20,
        ),
    )
    missing = [f for f in result.frames if f.diagnostic_reason == "NO_CANDIDATES"]
    assert len(missing) >= 2
    assert missing[1].search_radius_px > missing[0].search_radius_px
    assert missing[-1].status == "LOST"


def test_vt7_gate_rejection_reports_reason():
    frames = _frames(10)
    for k in range(7, 10):
        frames[k][:] = 28
        cx = 44 + 3 * k
        cv2.circle(frames[k], (cx, 57), 7, (225, 225, 225), -1)
    train, reader = _train(frames)
    accepted_train = [r for r in train.frames if r.selected is not None]
    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in accepted_train],
        initial_history_frames=[r.frame_index for r in accepted_train],
        detector_config=_detector_cfg(),
        config=TrackingConfig(
            search_radius_px=70,
            max_search_radius_px=120,
            top_k=5,
            min_track_score=0.0,
            min_identity_similarity=0.99,
            min_confidence_gap=0.0,
            lost_after_misses=1,
        ),
    )
    assert result.status == "LOST"
    assert "IDENTITY_BELOW_THRESHOLD" in result.frames[0].diagnostic_reason


def test_vt7_tracking_config_validates_dynamic_gate():
    with pytest.raises(ValueError):
        TrackingConfig(search_radius_px=90, max_search_radius_px=80).validate()
    with pytest.raises(ValueError):
        TrackingConfig(speed_radius_gain=-1).validate()
