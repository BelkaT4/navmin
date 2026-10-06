import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile, hud_evidence, hud_point_membership, runtime_detector_config
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig, run_reacquisition
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, _rank_candidates, run_tracking


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.5, min_response=2.0, min_area_px=2, max_area_px=900,
        adaptive_area_enabled=False, spatial_sigma_px=120.0, thermal_polarity="bright",
    )


def _blob_frame(x, y=55, *, w=140, h=100):
    im = np.full((h, w, 3), 28, np.uint8)
    cv2.rectangle(im, (x - 5, y - 4), (x + 6, y + 4), (225, 225, 225), -1)
    cv2.rectangle(im, (x + 2, y - 2), (x + 9, y + 2), (150, 150, 150), -1)
    return im


def _profile_with_hud(cx, cy=55, *, w=140, h=100, hard_half=8, soft_half=8):
    reticle = np.zeros((h, w), np.uint8)  # keep synthetic blob visible to detector
    hard = np.zeros_like(reticle)
    soft = np.zeros_like(reticle)
    hard[max(0, cy-hard_half):min(h, cy+hard_half+1), max(0, cx-hard_half):min(w, cx+hard_half+1)] = 255
    soft[max(0, cy-soft_half):min(h, cy+soft_half+1), max(0, cx-soft_half):min(w, cx+soft_half+1)] = 255
    return ThermalProfile(
        "thermal", "auto", reticle, (), hud_negative_enabled=True,
        hud_hard_mask=hard, hud_soft_mask=soft, hud_negative_penalty_weight=0.42,
        hud_hard_overlap_ratio=0.08, hud_soft_overlap_ratio=0.10,
    )


def _bank_from_target(profile, x=30, y=55):
    frame = _blob_frame(x, y)
    sel = make_operator_selection(0, (x, y), 30, frame.shape[1], frame.shape[0])
    det = detect_local_candidates(frame, sel, top_k=5, config=runtime_detector_config(_detector_cfg(), profile))
    assert det.candidates
    return ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(frame, det.candidates[0])])


def test_hotfix3_hud_evidence_marks_known_overlay_and_not_clear_background():
    profile = _profile_with_hud(70)
    frame = _blob_frame(70)
    sel = make_operator_selection(0, (70, 55), 25, 140, 100)
    det = detect_local_candidates(frame, sel, top_k=5, config=runtime_detector_config(_detector_cfg(), profile))
    assert det.candidates
    ev = hud_evidence(det.candidates[0], profile)
    assert ev.probability >= 0.9
    assert ev.hard_reject
    assert hud_point_membership((70, 55), profile) == (True, True)
    assert hud_point_membership((20, 20), profile) == (False, False)


def test_hotfix3_negative_penalty_survives_perfect_motion_self_consistency():
    profile = _profile_with_hud(70)
    bank = _bank_from_target(profile, x=70)
    frame = _blob_frame(70)
    sel = make_operator_selection(0, (70, 55), 25, 140, 100)
    det = detect_local_candidates(frame, sel, top_k=5, config=runtime_detector_config(_detector_cfg(), profile))
    ranked = _rank_candidates(
        frame, det.candidates, bank, (70.0, 55.0), cfg=TrackingConfig(min_confidence_gap=0.0),
        identity_weights=None, thermal_profile=profile, expected_area_px=bank.expected_size_px(),
    )
    assert ranked
    best = ranked[0]
    assert best.motion_score > 0.99
    assert best.hud_probability >= 0.9
    assert best.hud_negative_penalty > 0.35
    assert best.hud_hard_reject


def test_hotfix3_track_rejects_reticle_even_when_it_looks_like_target():
    profile = _profile_with_hud(80)
    bank = _bank_from_target(profile, x=30)
    frames = [_blob_frame(80), _blob_frame(80), _blob_frame(80)]

    def reader(i):
        return i, frames[i].copy()

    result = run_tracking(
        frame_reader=reader, frame_count=len(frames), frame_width=140, frame_height=100,
        start_frame=0, reference_bank=bank,
        initial_history=[(26.0,55.0),(28.0,55.0),(30.0,55.0)], initial_history_frames=[-3,-2,-1],
        detector_config=_detector_cfg(), thermal_profile=profile,
        config=TrackingConfig(
            search_radius_px=100, max_search_radius_px=120, min_search_radius_px=30,
            min_track_score=0.05, min_identity_similarity=0.05, min_confidence_gap=0.0,
            lost_after_misses=2, dynamic_search_enabled=False, global_shift_enabled=False,
            motion_break_enabled=False, thermal_jump_quarantine_enabled=False,
            thermal_hud_negative_enabled=True,
        ),
    )
    assert result.frames
    assert result.frames[0].selected is None
    assert result.frames[0].hud_hard_reject
    assert "HUD_HARD_REJECT" in result.frames[0].diagnostic_reason
    assert result.status == "LOST"


def test_hotfix3_true_motion_through_reticle_becomes_occlusion_then_temporal_exit():
    # Initial velocity is +4 px/frame, so frame 0 prediction lands in HUD at x=50.
    profile = _profile_with_hud(50, hard_half=6, soft_half=6)
    bank = _bank_from_target(profile, x=38)
    frames = [_blob_frame(50), _blob_frame(65), _blob_frame(69), _blob_frame(73)]

    def reader(i):
        return i, frames[i].copy()

    result = run_tracking(
        frame_reader=reader, frame_count=len(frames), frame_width=140, frame_height=100,
        start_frame=0, reference_bank=bank,
        initial_history=[(38.0,55.0),(42.0,55.0),(46.0,55.0)], initial_history_frames=[-3,-2,-1],
        detector_config=_detector_cfg(), thermal_profile=profile,
        config=TrackingConfig(
            search_radius_px=45, max_search_radius_px=70, min_search_radius_px=25,
            min_track_score=0.05, min_identity_similarity=0.05, min_confidence_gap=0.0,
            lost_after_misses=3, dynamic_search_enabled=False, global_shift_enabled=False,
            motion_break_enabled=False, thermal_jump_quarantine_enabled=False,
            thermal_hud_negative_enabled=True, hud_occlusion_max_frames=6,
            hud_occlusion_prediction_tolerance_px=30.0, hud_exit_confirm_hits=2,
            hud_exit_min_identity=0.20, hud_exit_max_probability=0.30,
        ),
    )
    assert result.frames[0].selected is None
    assert result.frames[0].diagnostic_reason.startswith("RETICLE_OCCLUSION")
    assert result.frames[0].miss_count == 0
    # First clean exit frame is only tentative evidence.
    assert result.frames[1].selected is None
    assert result.frames[1].miss_count == 0
    assert "HUD_EXIT_CONFIRM_1_OF_2" in result.frames[1].diagnostic_reason
    # Second coherent clean frame restores TRACK, rather than accepting HUD itself.
    assert result.frames[2].selected is not None
    assert result.frames[2].diagnostic_reason == "ACCEPTED_HUD_EXIT_REACQ"


def test_hotfix3_reacq_tracklets_cannot_confirm_hud_candidate():
    profile = _profile_with_hud(80)
    bank = _bank_from_target(profile, x=30)
    frames = [_blob_frame(30)] + [_blob_frame(80) for _ in range(7)]

    def reader(i):
        return i, frames[i].copy()

    result = run_reacquisition(
        frame_reader=reader, frame_count=len(frames), frame_width=140, frame_height=100,
        lost_frame=0, reference_bank=bank,
        motion_history=[MotionPoint(-2, 26.0,55.0), MotionPoint(-1,28.0,55.0), MotionPoint(0,30.0,55.0)],
        detector_config=_detector_cfg(), thermal_profile=profile, motion_broken=True,
        config=ReacquisitionConfig(
            initial_radius_px=110, max_radius_px=130, max_frames=7,
            min_score=0.05, min_identity_similarity=0.05, min_confidence_gap=0.0,
            robust_motion_break_enabled=True, tracklets_enabled=True,
            broken_min_score=0.05, broken_min_identity_similarity=0.05, broken_min_confidence_gap=0.0,
            tracklet_admission_min_identity=0.05, tracklet_admission_min_score=0.01,
            tracklet_confirm_hits=3, tracklet_window_frames=5, full_frame_after_frames=20,
            global_shift_enabled=False,
        ),
    )
    assert result.status != "CONFIRMED"
    assert not any(f.status == "CONFIRMED" for f in result.frames)
    assert any("TRACKLETS_NO_VISUAL_MATCH" in f.diagnostic_reason for f in result.frames)
