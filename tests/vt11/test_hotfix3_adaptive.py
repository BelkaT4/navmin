import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig, run_reacquisition
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def _detector(**kwargs):
    base = dict(
        threshold_sigma=1.5,
        min_response=2.0,
        min_area_px=2,
        max_area_px=2500,
        spatial_sigma_px=35.0,
        adaptive_area_enabled=True,
        adaptive_min_area_ratio=0.15,
        adaptive_max_area_ratio=4.0,
        expected_area_weight=0.5,
    )
    base.update(kwargs)
    return CandidateDetectorConfig(**base)


def test_hotfix3_expected_area_removes_tiny_noise_before_topk():
    frame = np.full((180, 240, 3), 40, np.uint8)
    # Tiny distractor exactly at the prediction.
    cv2.rectangle(frame, (98, 88), (99, 89), (255, 255, 255), -1)
    # Plausible target-sized component farther from the prediction.
    cv2.rectangle(frame, (140, 75), (165, 100), (220, 220, 220), -1)
    sel = make_operator_selection(0, (99, 89), 80, 240, 180)
    cfg = _detector(blur_sigma_px=1.5, morph_close_px=0)

    no_size = detect_local_candidates(frame, sel, top_k=5, config=cfg, expected_area_px=None)
    assert no_size.candidates
    assert any(c.area_px < 100 for c in no_size.candidates)

    sized = detect_local_candidates(frame, sel, top_k=5, config=cfg, expected_area_px=650.0)
    assert sized.candidates
    assert all(c.area_px >= sized.diagnostics.effective_min_area_px for c in sized.candidates)
    assert sized.diagnostics.effective_min_area_px >= 90
    assert sized.candidates[0].area_px > 100
    assert sized.diagnostics.valid_components < no_size.diagnostics.valid_components


def test_hotfix3_expected_area_allows_growth_beyond_old_2500_cap():
    frame = np.full((220, 260, 3), 35, np.uint8)
    # Filled circle gives a connected response component above the old 2500 px cap.
    cv2.circle(frame, (130, 110), 35, (230, 230, 230), -1)
    sel = make_operator_selection(0, (130, 110), 90, 260, 220)
    result = detect_local_candidates(frame, sel, top_k=5, config=_detector(), expected_area_px=1500.0)
    assert result.diagnostics.effective_max_area_px == 6000
    assert result.candidates
    assert result.candidates[0].area_px > 2500


def _draw_target(frame, x):
    cv2.rectangle(frame, (x, 52), (x + 12, 62), (225, 225, 225), -1)
    cv2.rectangle(frame, (x + 8, 55), (x + 16, 59), (155, 155, 155), -1)


def _train_for_reacq(frames):
    sel = make_operator_selection(0, (44, 57), 65, 180, 120)
    det = detect_local_candidates(frames[0], sel, top_k=5, config=_detector(max_area_px=1200))
    assert det.candidates

    def reader(i):
        return i, frames[i].copy()

    train = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=0,
        initial_candidate=det.candidates[0],
        roi_radius_px=65,
        detector_config=_detector(max_area_px=1200),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert train.decision == "CONFIRMED"
    return train, reader


def test_hotfix3_reacq_confirms_three_of_five_with_one_bad_frame_between():
    # TRAIN 0..6. Target is then absent; after LOST it appears on 11,12,14,
    # with frame 13 blank. Old consecutive-streak logic would reset at 13;
    # HOTFIX-3 must confirm on 14 because 3 of the last 5 are valid.
    frames = []
    visible = set(range(0, 7)) | {11, 12, 14, 15, 16}
    for k in range(20):
        f = np.full((120, 180, 3), 28, np.uint8)
        if k in visible:
            _draw_target(f, 38 + 2 * k)
        frames.append(f)
    train, reader = _train_for_reacq(frames)
    history = [
        MotionPoint(r.frame_index, r.selected.center_x, r.selected.center_y)
        for r in train.frames if r.selected is not None
    ]
    result = run_reacquisition(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=9,
        reference_bank=train.reference_bank,
        motion_history=history,
        detector_config=_detector(max_area_px=1200),
        config=ReacquisitionConfig(
            initial_radius_px=90,
            max_radius_px=170,
            radius_growth_px_per_frame=10,
            confirm_frames=3,
            confirm_window_frames=5,
            max_frames=9,
            min_confidence_gap=0.0,
            max_confirmation_error_px=90,
        ),
    )
    assert result.status == "CONFIRMED"
    assert result.confirmed_frame == 14
    assert result.t_reacq_frames == 5
    assert len(result.confirmation_points) >= 3
    assert any(f.status == "SEARCHING" for f in result.frames)
    assert result.frames[-1].confirmation_streak >= 3


def test_hotfix3_tracking_hysteresis_survives_two_bad_frames():
    # Simple moving target with two blank frames and then return. With the new
    # default 3-miss hysteresis TRACK must not declare LOST on the second miss.
    frames = []
    for k in range(14):
        f = np.full((120, 180, 3), 28, np.uint8)
        if k not in {8, 9}:
            _draw_target(f, 38 + 2 * k)
        frames.append(f)
    train, reader = _train_for_reacq(frames)
    accepted = [r for r in train.frames if r.selected is not None]
    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in accepted],
        initial_history_frames=[r.frame_index for r in accepted],
        detector_config=_detector(max_area_px=1200),
        config=TrackingConfig(search_radius_px=70, max_search_radius_px=140, min_confidence_gap=0.0),
    )
    assert result.status == "COMPLETED"
    miss_frames = [f for f in result.frames if f.frame_index in {8, 9}]
    assert [f.status for f in miss_frames] == ["LOW_CONFIDENCE", "LOW_CONFIDENCE"]
    assert any(f.frame_index == 10 and f.status == "TRACK" for f in result.frames)


def test_hotfix3_default_ema_gates_are_reachable_after_warmup():
    # Use a tiny synthetic bank/descriptor through the existing train helper.
    frames = []
    for k in range(10):
        f = np.full((120, 180, 3), 28, np.uint8)
        _draw_target(f, 38 + k)
        frames.append(f)
    train, _reader = _train_for_reacq(frames)
    bank = train.reference_bank.clone(include_adaptive=False)
    cfg = EMAConfig(
        mode="ema", alpha=0.06, min_confidence=0.70,
        min_identity_similarity=0.52, min_confidence_gap=0.15,
        max_prediction_error_px=10.0, warmup_accepted_frames=2,
        update_interval_frames=1,
    )
    adapter = EMAReferenceAdapter(bank, cfg)
    descriptor = bank.descriptors[-1]
    e1 = adapter.consider(frame_index=20, descriptor=descriptor, confidence=.73, identity_similarity=.56, confidence_gap=.20, prediction_error_px=2.0)
    e2 = adapter.consider(frame_index=21, descriptor=descriptor, confidence=.73, identity_similarity=.56, confidence_gap=.20, prediction_error_px=2.0)
    e3 = adapter.consider(frame_index=22, descriptor=descriptor, confidence=.73, identity_similarity=.56, confidence_gap=.20, prediction_error_px=2.0)
    assert not e1.updated and not e2.updated
    assert e3.updated
    assert bank.adaptive_descriptor is not None


def test_hotfix3_config_validation_for_confirmation_window_and_area_ratios():
    with pytest.raises(ValueError):
        ReacquisitionConfig(confirm_frames=3, confirm_window_frames=2).validate()
    with pytest.raises(ValueError):
        CandidateDetectorConfig(adaptive_min_area_ratio=2.0, adaptive_max_area_ratio=1.0).validate()
    with pytest.raises(ValueError):
        TrackingConfig(expected_area_update_alpha=0.0).validate()
