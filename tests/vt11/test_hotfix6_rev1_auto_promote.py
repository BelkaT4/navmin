import numpy as np
import cv2
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig, run_reacquisition
from navmin.vision.processors.vt11.tracking.background_motion import BackgroundMotionConfig, estimate_background_motion


def _target_frame(x, *, w=260, h=150):
    f = np.full((h, w, 3), 28, np.uint8)
    cv2.rectangle(f, (x, 66), (x + 13, 76), (225, 225, 225), -1)
    cv2.rectangle(f, (x + 8, 68), (x + 19, 72), (155, 155, 155), -1)
    return f


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.6, min_response=3.0, min_area_px=2, max_area_px=1800,
        spatial_sigma_px=35.0, adaptive_area_enabled=False,
    )


def _reference_bank():
    first = _target_frame(42)
    sel = make_operator_selection(frame_index=0, click_xy=(50, 71), radius_px=55,
                                  frame_width=260, frame_height=150)
    det = detect_local_candidates(first, sel, top_k=5, config=_detector_cfg())
    assert det.candidates
    return ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(first, det.candidates[0])])


def test_rev1_normal_reacq_auto_promotes_after_two_strong_visual_motion_disagreements():
    bank = _reference_bank()
    # Old motion history predicts near x~55, but after the LOST frame the same
    # target appears around x~185 and moves self-consistently from there.
    frames = [_target_frame(42 + 2*k) for k in range(8)]
    frames.append(np.full_like(frames[0], 28))  # LOST at 8
    for k in range(9, 16):
        frames.append(_target_frame(178 + 3*(k-9)))

    def reader(i):
        return i, frames[i].copy()

    history = [MotionPoint(k, 42 + 2*k + 7.0, 71.0) for k in range(5, 8)]
    result = run_reacquisition(
        frame_reader=reader, frame_count=len(frames), frame_width=260, frame_height=150,
        lost_frame=8, reference_bank=bank, motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=230, max_radius_px=250, radius_growth_px_per_frame=0,
            max_frames=7, confirm_frames=3, confirm_window_frames=5,
            max_confirmation_error_px=55, min_confidence_gap=0.0,
            robust_motion_break_enabled=True,
            auto_promote_enabled=True, auto_promote_window_frames=3, auto_promote_hits_required=2,
            auto_promote_min_identity=0.55, auto_promote_min_gap=0.0,
            auto_promote_min_size_score=0.01, auto_promote_max_motion_score=0.60,
            broken_min_score=0.25, broken_min_identity_similarity=0.25,
            broken_min_confidence_gap=0.0,
            tracklets_enabled=True, tracklet_top_k=3, tracklet_confirm_hits=3,
            tracklet_window_frames=5, tracklet_max_link_distance_px=40,
            global_shift_enabled=False,
        ),
        motion_broken=False,
    )
    assert result.status == "CONFIRMED"
    assert result.auto_promoted is True
    assert result.promotion_frame is not None
    assert result.motion_broken_mode is True
    assert result.motion_relocked is True
    assert any(f.auto_promoted_motion_break for f in result.frames)
    assert any("AUTO_PROMOTE" in f.promotion_reason for f in result.frames)
    assert any(f.tracklet_id > 0 for f in result.frames)
    assert result.confirmed_center[0] > 175


def test_rev1_auto_promote_needs_repeated_evidence_not_single_frame():
    bank = _reference_bank()
    frames = [_target_frame(42 + 2*k) for k in range(8)]
    frames.append(np.full_like(frames[0], 28))
    frames.append(_target_frame(180))  # one strong far candidate only
    frames.append(np.full_like(frames[0], 28))
    frames.append(np.full_like(frames[0], 28))

    def reader(i): return i, frames[i].copy()
    history = [MotionPoint(k, 42 + 2*k + 7.0, 71.0) for k in range(5, 8)]
    result = run_reacquisition(
        frame_reader=reader, frame_count=len(frames), frame_width=260, frame_height=150,
        lost_frame=8, reference_bank=bank, motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=230, max_radius_px=250, radius_growth_px_per_frame=0,
            max_frames=3, max_confirmation_error_px=55, min_confidence_gap=0.0,
            robust_motion_break_enabled=True,
            auto_promote_enabled=True, auto_promote_window_frames=3, auto_promote_hits_required=2,
            auto_promote_min_identity=0.55, auto_promote_min_gap=0.0,
            auto_promote_min_size_score=0.01, auto_promote_max_motion_score=0.60,
            tracklets_enabled=True, global_shift_enabled=False,
        ), motion_broken=False,
    )
    assert result.auto_promoted is False
    assert result.motion_broken_mode is False


def test_rev1_large_phase_shift_requires_feature_confirmation(monkeypatch):
    import navmin.vision.processors.vt11.tracking.background_motion as bm
    prev = np.zeros((120, 200, 3), np.uint8)
    curr = prev.copy()

    monkeypatch.setattr(bm, "_phase_estimate", lambda p, c, scale: (95.0, 0.0, 0.95))
    # phase 95px > 0.30*200=60, so it must not be directly accepted.
    confirmed = np.array([[1.0, 0.0, 32.0], [0.0, 1.0, -8.0]], dtype=np.float64)
    monkeypatch.setattr(bm, "_feature_affine", lambda p, c, mask, scale, cfg: (confirmed.copy(), 30))
    est = estimate_background_motion(
        prev, curr,
        config=BackgroundMotionConfig(
            downscale=0.5, max_dimension_px=160, phase_min_response=0.05,
            phase_direct_max_fraction=0.30, use_feature_fallback=True,
        ),
    )
    assert est.valid
    assert est.method == "FEATURE_AFFINE"
    assert est.dx_px == pytest.approx(32.0)
    assert est.dy_px == pytest.approx(-8.0)


def test_rev1_discontinuous_phase_shift_requires_feature_confirmation(monkeypatch):
    import navmin.vision.processors.vt11.tracking.background_motion as bm
    prev = np.zeros((120, 200, 3), np.uint8)
    curr = prev.copy()
    monkeypatch.setattr(bm, "_phase_estimate", lambda p, c, scale: (42.0, 4.0, 0.95))
    confirmed = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 3.0]], dtype=np.float64)
    monkeypatch.setattr(bm, "_feature_affine", lambda p, c, mask, scale, cfg: (confirmed.copy(), 28))
    est = estimate_background_motion(
        prev, curr,
        config=BackgroundMotionConfig(
            downscale=0.5, max_dimension_px=160, phase_min_response=0.05,
            phase_direct_max_fraction=1.0, phase_max_change_from_previous_px=20.0,
            use_feature_fallback=True,
        ),
        previous_shift_xy=(0.0, 0.0),
    )
    assert est.valid
    assert est.method == "FEATURE_AFFINE"
    assert est.dx_px == pytest.approx(12.0)


def test_rev1_default_config_enables_auto_promote_and_phase_plausibility():
    from navmin.vision.processors.vt11.config.loader import load_config
    cfg = load_config()
    assert cfg["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    r = cfg["reacquisition"]
    assert r["auto_promote_enabled"] is True
    assert r["auto_promote_hits_required"] == 2
    assert r["auto_promote_window_frames"] == 3
    assert r["auto_promote_min_identity"] == pytest.approx(0.70)
    assert r["global_shift_phase_direct_max_fraction"] == pytest.approx(0.30)
