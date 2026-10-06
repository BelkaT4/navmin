import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.tracking.tracklets import TrackletConfig, TrackletEvidence, TrackletManager
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank


def _thermal_micro_frame(x=80, y=55, *, w=180, h=120):
    f = np.full((h, w, 3), 35, np.uint8)
    cv2.rectangle(f, (x, y), (x + 3, y + 3), (235, 235, 235), -1)
    return f


def test_rev2_micro_channel_keeps_tiny_thermal_target_below_normal_area_gate():
    frame = _thermal_micro_frame()
    sel = make_operator_selection(frame_index=0, click_xy=(82, 57), radius_px=55, frame_width=180, frame_height=120)
    cfg = CandidateDetectorConfig(
        method="thermal_hot_blob", threshold_sigma=1.4, min_response=2.0,
        min_area_px=2, max_area_px=2500, adaptive_area_enabled=True,
        adaptive_min_area_ratio=0.15, adaptive_max_area_ratio=4.0,
        micro_enabled=True, micro_min_area_px=3, micro_max_area_px=80,
        micro_response_multiplier=1.0, micro_top_k=4,
    )
    normal = detect_local_candidates(frame, sel, top_k=5, config=cfg, expected_area_px=500, include_micro=False)
    micro = detect_local_candidates(frame, sel, top_k=5, config=cfg, expected_area_px=500, include_micro=True)
    assert not normal.candidates
    assert any(c.detection_channel == "MICRO" for c in micro.candidates)
    assert micro.diagnostics.micro_valid_components >= 1


def _ev(frame, x, *, identity, gap=0.12, far=False, micro=False, bg=(0.0, 0.0, False)):
    return TrackletEvidence(
        frame, x, 40.0, identity, 0.80, 0.5, gap, object(),
        anchor_distance_px=140.0 if far else 20.0, far_jump=far, is_micro=micro,
        background_dx_px=bg[0], background_dy_px=bg[1], background_valid=bg[2],
    )


def test_rev2_far_tracklet_needs_stronger_identity_and_more_hits():
    cfg = TrackletConfig(
        max_tracklets=2, window_frames=6, confirm_hits=3, max_link_distance_px=30,
        min_mean_identity=0.45, min_mean_score=0.45, min_mean_gap=0.01,
        far_jump_confirm_hits=5, far_jump_min_mean_identity=0.88, far_jump_min_mean_gap=0.10,
        background_attachment_enabled=False,
    )
    weak = TrackletManager(cfg)
    for f in range(5): weak.update(f, [_ev(f, 120 + 2*f, identity=0.80, far=True)])
    assert weak.best_confirmed(4) is None

    strong = TrackletManager(cfg)
    for f in range(5): strong.update(f, [_ev(f, 120 + 2*f, identity=0.94, far=True, gap=0.16)])
    assert strong.best_confirmed(4) is not None


def test_rev2_background_attached_tracklet_stays_tentative():
    cfg = TrackletConfig(
        max_tracklets=1, window_frames=5, confirm_hits=3, max_link_distance_px=30,
        min_mean_identity=0.45, min_mean_score=0.45, min_mean_gap=0.01,
        background_attachment_enabled=True, background_motion_min_px=2.0,
        background_attachment_tolerance_px=1.0, background_attachment_max_fraction=0.8,
    )
    mgr = TrackletManager(cfg)
    for f in range(4):
        # Candidate moves exactly with +4 px/frame background translation.
        mgr.update(f, [_ev(f, 80 + 4*f, identity=0.92, far=True, bg=(4.0, 0.0, f > 0))])
    assert mgr.best_confirmed(3) is None


def _warmup_target(x, *, w=220, h=120):
    f = np.full((h, w, 3), 28, np.uint8)
    cv2.rectangle(f, (x, 51), (x+12, 62), (225,225,225), -1)
    return f


def test_rev2_safe_warmup_rejects_unexplained_large_jump():
    frame = _warmup_target(75)
    sel = make_operator_selection(frame_index=0, click_xy=(81,56), radius_px=40, frame_width=220, frame_height=120)
    dcfg = CandidateDetectorConfig(threshold_sigma=1.5, min_response=2, min_area_px=2, max_area_px=1200, adaptive_area_enabled=False)
    det = detect_local_candidates(frame, sel, top_k=5, config=dcfg)
    bank = ReferenceBank(max_size=2, descriptors=[descriptor_from_candidate(frame, det.candidates[0])])
    frames=[frame.copy(), frame.copy()]
    result=run_tracking(
        frame_reader=lambda i:(i,frames[i].copy()), frame_count=2, frame_width=220, frame_height=120,
        start_frame=0, reference_bank=bank, initial_history=[(150.0,56.0)], initial_history_frames=[-1],
        detector_config=dcfg,
        config=TrackingConfig(
            search_radius_px=100, min_search_radius_px=50, max_search_radius_px=150,
            top_k=5, min_track_score=0.2, min_identity_similarity=0.2, min_confidence_gap=0.0,
            lost_after_misses=1, track_warmup_frames=1, warmup_search_radius_px=150,
            warmup_min_track_score=0.2, warmup_min_identity_similarity=0.2, warmup_min_confidence_gap=0.0,
            identity_first_safe_warmup_enabled=True, warmup_max_unexplained_jump_px=60.0,
            global_shift_enabled=False,
        ),
    )
    assert result.frames[0].selected is None
    assert "WARMUP_LARGE_JUMP_UNEXPLAINED" in result.frames[0].diagnostic_reason


def test_rev2_default_config_is_identity_first_and_micro_enabled():
    from navmin.vision.processors.vt11.config.loader import load_config
    cfg=load_config()
    assert cfg["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert cfg["project"]["config_version"] == 31
    assert cfg["reacquisition"]["identity_first_enabled"] is True
    assert cfg["reacquisition"]["micro_detector_enabled"] is True
    assert cfg["reacquisition"]["far_jump_min_identity"] == pytest.approx(0.88)
    assert cfg["reacquisition"]["global_emergency_after_frames"] == 15
    assert cfg["tracking"]["identity_first_safe_warmup_enabled"] is True
