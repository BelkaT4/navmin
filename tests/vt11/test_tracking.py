import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.tracking.motion_model import predict_position, robust_velocity
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def _frames(count=18, *, blank_after=None):
    frames = []
    for k in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if blank_after is None or k < blank_after:
            x = 38 + 3 * k
            # Asymmetric target gives appearance/edge/projection a stable signature.
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


def _initial_candidate(frame):
    sel = make_operator_selection(
        frame_index=0,
        click_xy=(43, 57),
        radius_px=65,
        frame_width=180,
        frame_height=120,
    )
    result = detect_local_candidates(frame, sel, top_k=5, config=_detector_cfg())
    assert result.candidates
    return result.candidates[0]


def _train(frames):
    first = _initial_candidate(frames[0])

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=0,
        initial_candidate=first,
        roi_radius_px=65,
        detector_config=_detector_cfg(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "CONFIRMED"
    return result, reader


def test_robust_velocity_and_multistep_prediction():
    history = [(10.0, 20.0), (13.0, 22.0), (16.0, 24.0), (19.0, 26.0)]
    assert robust_velocity(history, window_frames=5) == (3.0, 2.0)
    est = predict_position(history, fallback=(0.0, 0.0), window_frames=5, steps_ahead=2)
    assert est.predicted_xy == (25.0, 30.0)


def test_tracking_follows_consistent_target_to_eof():
    frames = _frames(18)
    train, reader = _train(frames)
    history = [r.selected.center for r in train.frames if r.selected is not None]
    bank_size_before = len(train.reference_bank)

    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=train.frames[-1].frame_index + 1,
        reference_bank=train.reference_bank,
        initial_history=history,
        detector_config=_detector_cfg(),
        config=TrackingConfig(search_radius_px=70, top_k=5, min_confidence_gap=0.0),
    )
    assert result.status == "COMPLETED"
    assert result.tracked_frames == len(frames) - 7
    assert result.attempted_frames == len(frames) - 7
    assert result.lost_frame is None
    assert result.last_center is not None
    assert result.last_center[0] > history[-1][0]
    assert len(result.trajectory) == result.tracked_frames
    assert all(f.processing_time_ms > 0 for f in result.frames)
    # VT-6 is deliberately non-adaptive.
    assert len(train.reference_bank) == bank_size_before


def test_tracking_enters_lost_after_consecutive_blank_frames():
    frames = _frames(16, blank_after=9)
    train, reader = _train(frames)
    history = [r.selected.center for r in train.frames if r.selected is not None]

    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=history,
        detector_config=_detector_cfg(),
        config=TrackingConfig(search_radius_px=70, top_k=5, min_confidence_gap=0.0, lost_after_misses=2),
    )
    assert result.status == "LOST"
    assert result.lost_frame == 10
    assert result.frames[-2].status == "LOW_CONFIDENCE"
    assert result.frames[-1].status == "LOST"


def test_tracking_gate_can_reject_even_when_candidate_exists():
    frames = _frames(10)
    # Replace post-training target with a distinctly different circular blob at
    # the predicted location: the detector still has a candidate, but identity
    # gating should reject it.
    for k in range(7, 10):
        frames[k][:] = 28
        cx = 44 + 3 * k
        cv2.circle(frames[k], (cx, 57), 7, (225, 225, 225), -1)
    train, reader = _train(frames)
    history = [r.selected.center for r in train.frames if r.selected is not None]

    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=history,
        detector_config=_detector_cfg(),
        config=TrackingConfig(
            search_radius_px=70,
            top_k=5,
            min_track_score=0.0,
            min_identity_similarity=0.99,
            min_confidence_gap=0.0,
            lost_after_misses=1,
        ),
    )
    assert result.status == "LOST"
    assert result.frames[0].candidate_count >= 1
    assert result.frames[0].selected is None


def test_tracking_requires_reference_bank_and_history():
    frames = _frames(8)
    train, reader = _train(frames)
    with pytest.raises(ValueError):
        run_tracking(
            frame_reader=reader,
            frame_count=len(frames),
            frame_width=180,
            frame_height=120,
            start_frame=7,
            reference_bank=train.reference_bank,
            initial_history=[],
            detector_config=_detector_cfg(),
            config=TrackingConfig(),
        )


def test_tracking_config_validation():
    with pytest.raises(ValueError):
        TrackingConfig(top_k=1).validate()
    with pytest.raises(ValueError):
        TrackingConfig(lost_after_misses=0).validate()
