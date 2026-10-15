import hashlib
from types import SimpleNamespace

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG, load_config
from navmin.vision.processors.vt11.features.thermal import thermal_feature_metrics
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.profiles import effective_config
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile, preprocess_frame, runtime_detector_config
from navmin.vision.processors.vt11.tracking.tracklets import TrackletConfig, TrackletEvidence, TrackletManager
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.6, min_response=3.0, min_area_px=2, max_area_px=1400,
        spatial_sigma_px=150.0, adaptive_area_enabled=False, thermal_polarity="bright",
    )


def _draw_target(im, x, *, exact=True):
    if exact:
        cv2.rectangle(im, (x, 61), (x + 12, 70), (225, 225, 225), -1)
        cv2.rectangle(im, (x + 8, 63), (x + 17, 67), (150, 150, 150), -1)
    else:
        cv2.rectangle(im, (x, 61), (x + 11, 69), (205, 205, 205), -1)
        cv2.rectangle(im, (x + 7, 63), (x + 15, 67), (135, 135, 135), -1)


def _frame(x, *, w=240, h=140):
    im = np.full((h, w, 3), 28, np.uint8)
    _draw_target(im, x, exact=True)
    return im


def _thermal_profile(w=240, h=140):
    return ThermalProfile("thermal", "off", np.zeros((h, w), np.uint8), ())


def _thermal_reference_bank(tp):
    base = _frame(42)
    analysis = preprocess_frame(base, tp)
    sel = make_operator_selection(
        frame_index=0, click_xy=(49, 66), radius_px=55, frame_width=240, frame_height=140,
    )
    det = detect_local_candidates(
        analysis, sel, top_k=5, config=runtime_detector_config(_detector_cfg(), tp),
    )
    assert det.candidates
    return ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(analysis, det.candidates[0])])


def test_hotfix2_checkpoint_thermal_only_and_day_bytes_frozen():
    cfg = load_config()
    assert cfg["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert cfg["project"]["config_version"] == 31
    assert hashlib.sha256(DEFAULT_CONFIG.with_name("day_profile.yaml").read_bytes()).hexdigest() == (
        "902b15e22d2b1b2abb4da80ddd5ae1360577cdc16af1e7b2514a812703b994d4"
    )
    day = effective_config(cfg, "day")
    thermal = effective_config(cfg, "thermal")
    assert day["tracking"]["thermal_jump_quarantine_enabled"] is False
    assert thermal["tracking"]["thermal_jump_quarantine_enabled"] is True
    assert thermal["tracking"]["jump_quarantine_distance_px"] == 55.0
    assert thermal["tracking"]["jump_extreme_distance_px"] == 80.0
    assert thermal["tracking"]["size_update_max_prediction_error_px"] == 30.0
    assert thermal["tracking"]["scale_robust_history_frames"] == 12


def test_hotfix2_extreme_lone_jump_is_quarantined_before_confirmation():
    tp = _thermal_profile()
    bank = _thermal_reference_bank(tp)
    frames = [_frame(170), _frame(173), _frame(176), _frame(179), _frame(182)]

    def reader(i):
        return i, frames[i].copy()

    result = run_tracking(
        frame_reader=reader, frame_count=len(frames), frame_width=240, frame_height=140,
        start_frame=0, reference_bank=bank,
        initial_history=[(45.0, 66.0), (47.0, 66.0), (49.0, 66.0)],
        initial_history_frames=[-3, -2, -1], detector_config=_detector_cfg(),
        thermal_profile=tp,
        config=TrackingConfig(
            search_radius_px=190, max_search_radius_px=220, min_search_radius_px=50,
            identity_weight=0.70, motion_weight=0.18, size_weight=0.12,
            min_track_score=0.15, min_identity_similarity=0.15, min_confidence_gap=0.0,
            lost_after_misses=6, motion_break_enabled=True, motion_break_error_px=60.0,
            motion_break_visual_identity_min=0.20, motion_break_visual_gap_min=0.0,
            motion_break_visual_size_min=0.01, motion_relock_frames=3, global_shift_enabled=False,
            thermal_jump_quarantine_enabled=True, jump_quarantine_distance_px=55.0,
            jump_extreme_distance_px=80.0, jump_confirm_hits=2, jump_extreme_confirm_hits=3,
            jump_single_candidate_extra_hits=1, jump_link_distance_px=38.0,
            jump_min_identity=0.20, jump_confirm_min_identity=0.20, jump_min_gap=0.0,
            jump_min_size_score=0.01, jump_clutter_penalty_weight=0.14,
            reliable_anchor_update_max_prediction_error_px=30.0,
        ),
    )
    assert len(result.frames) >= 4
    for item in result.frames[:3]:
        assert item.selected is None
        assert item.jump_quarantine_active
        assert "JUMP_QUARANTINE" in item.diagnostic_reason
        assert item.anchor_distance_px > 100.0
        assert item.anchor_distance_norm > 3.0
    accepted = result.frames[3]
    assert accepted.selected is not None
    assert accepted.jump_quarantine_confirmed
    assert accepted.diagnostic_reason == "ACCEPTED_JUMP_QUARANTINE_CONFIRMED"


def test_hotfix2_reliable_anchor_prefers_plausible_local_target_over_far_exact_blob():
    tp = _thermal_profile()
    bank = _thermal_reference_bank(tp)
    frames = []
    for local_x, far_x in ((52, 170), (54, 173), (56, 176)):
        im = np.full((140, 240, 3), 28, np.uint8)
        _draw_target(im, local_x, exact=False)  # changed projection/thermal appearance
        _draw_target(im, far_x, exact=True)     # deceptively closer to old descriptor
        frames.append(im)

    def reader(i):
        return i, frames[i].copy()

    result = run_tracking(
        frame_reader=reader, frame_count=len(frames), frame_width=240, frame_height=140,
        start_frame=0, reference_bank=bank,
        initial_history=[(45.0, 66.0), (47.0, 66.0), (49.0, 66.0)],
        initial_history_frames=[-3, -2, -1], detector_config=_detector_cfg(), thermal_profile=tp,
        config=TrackingConfig(
            search_radius_px=195, max_search_radius_px=220, min_search_radius_px=50,
            identity_weight=1.0, motion_weight=0.0, size_weight=0.0,
            min_track_score=0.50, min_identity_similarity=0.50, min_confidence_gap=0.0,
            lost_after_misses=3, motion_break_enabled=True, motion_break_error_px=60.0,
            motion_break_visual_identity_min=0.60, motion_break_visual_gap_min=0.0,
            motion_break_visual_size_min=0.01, global_shift_enabled=False,
            thermal_jump_quarantine_enabled=True, jump_quarantine_distance_px=55.0,
            jump_extreme_distance_px=80.0, jump_min_identity=0.60,
            jump_confirm_min_identity=0.70, jump_min_gap=0.0, jump_min_size_score=0.01,
            jump_local_preference_max_identity_deficit=0.35,
        ),
    )
    first = result.frames[0]
    assert first.selected is not None
    assert first.selected.center_x < 100.0
    assert first.anchor_continuity_override
    assert first.diagnostic_reason == "ACCEPTED_ANCHOR_CONTINUITY"


def test_hotfix2_background_clutter_penalizes_tree_like_ring_but_not_uniform_ring():
    obj = np.full((15, 15), 80, np.uint8)
    cv2.circle(obj, (7, 7), 3, 220, -1)
    uniform = np.full((35, 35), 80, np.uint8)
    rng = np.random.default_rng(1234)
    clutter = rng.integers(45, 145, size=(35, 35), dtype=np.uint8)
    fu = thermal_feature_metrics(obj, uniform)
    fc = thermal_feature_metrics(obj, clutter)
    assert fu.background_clutter < 0.05
    assert fc.background_clutter > fu.background_clutter + 0.35


def test_hotfix2_tracklet_missing_runner_up_is_neutral_not_fake_large_gap():
    cfg = TrackletConfig(
        max_tracklets=1, window_frames=4, confirm_hits=3, max_link_distance_px=20.0,
        min_mean_identity=0.60, min_mean_score=0.60, min_mean_gap=0.20,
    )
    mgr = TrackletManager(cfg)
    for f in range(3):
        ev = TrackletEvidence(
            frame_index=f, x=100.0 + 2*f, y=80.0, identity=0.90, total_score=0.85,
            size_score=0.8, gap=0.0, candidate=SimpleNamespace(area_px=40), has_competitor=False,
        )
        mgr.update(f, [ev])
    assert mgr.best_confirmed(2) is not None
