import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.tracking.background_motion import BackgroundMotionConfig, estimate_background_motion, transform_point
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig, run_reacquisition
from navmin.vision.processors.vt11.tracking.tracklets import TrackletConfig, TrackletEvidence, TrackletManager


def _texture(h=180, w=260):
    rng = np.random.default_rng(123)
    img = rng.normal(90, 18, (h, w)).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for x, y in [(35, 40), (120, 35), (200, 90), (70, 140), (180, 150)]:
        cv2.rectangle(img, (x-5, y-4), (x+6, y+5), int(150 + (x % 70)), -1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_hotfix6_phase_correlation_recovers_large_camera_shift():
    prev = _texture()
    dx, dy = 31.0, -18.0
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    curr = cv2.warpAffine(prev, mat, (prev.shape[1], prev.shape[0]), borderMode=cv2.BORDER_REFLECT)
    est = estimate_background_motion(
        prev, curr,
        config=BackgroundMotionConfig(
            downscale=0.5, max_dimension_px=220,
            phase_min_response=0.05, use_feature_fallback=True,
        ),
        target_xy=(130, 90), target_radius_px=20,
    )
    assert est.valid
    assert est.method in {"PHASE", "FEATURE_AFFINE"}
    assert est.dx_px == pytest.approx(dx, abs=3.0)
    assert est.dy_px == pytest.approx(dy, abs=3.0)
    px = transform_point((100.0, 80.0), est)
    assert px[0] == pytest.approx(131.0, abs=4.0)
    assert px[1] == pytest.approx(62.0, abs=4.0)
    assert est.total_time_ms >= 0.0


def test_hotfix6_tracklet_confirms_new_self_consistent_trajectory():
    cfg = TrackletConfig(max_tracklets=3, window_frames=5, confirm_hits=3, max_link_distance_px=25,
                         min_mean_identity=0.5, min_mean_score=0.5, min_mean_gap=0.02)
    mgr = TrackletManager(cfg)
    for f in range(3):
        # One consistent target hypothesis plus a distractor that jumps around.
        evs = [
            TrackletEvidence(f, 150 + 4*f, 80 + 2*f, 0.92, 0.86, 0.8, 0.15, object()),
            TrackletEvidence(f, 30 + 70*(f % 2), 25 + 50*((f+1) % 2), 0.70, 0.65, 0.7, 0.05, object()),
        ]
        mgr.update(f, evs)
    confirmed = mgr.best_confirmed(2)
    assert confirmed is not None
    assert len(confirmed.recent(5, 2)) >= 3
    assert confirmed.last.x == pytest.approx(158.0)


def _target_frame(x, *, w=240, h=140):
    f = np.full((h, w, 3), 28, np.uint8)
    cv2.rectangle(f, (x, 61), (x + 12, 70), (225, 225, 225), -1)
    cv2.rectangle(f, (x + 8, 63), (x + 17, 67), (150, 150, 150), -1)
    return f


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.6, min_response=3.0, min_area_px=2, max_area_px=1400,
        spatial_sigma_px=35.0, adaptive_area_enabled=False,
    )


def test_hotfix6_broken_reacq_uses_tracklet_not_stale_motion_gate():
    # Reference object from the left side; after a synthetic camera jump the
    # same target is ~125 px away from the old motion prediction.
    first = _target_frame(42)
    sel = make_operator_selection(frame_index=0, click_xy=(49, 66), radius_px=55,
                                  frame_width=240, frame_height=140)
    det = detect_local_candidates(first, sel, top_k=5, config=_detector_cfg())
    assert det.candidates
    bank = ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(first, det.candidates[0])])

    frames = [_target_frame(42 + 2*k) for k in range(8)]
    frames.append(np.full_like(first, 28))  # LOST frame 8
    for k in range(9, 15):
        frames.append(_target_frame(170 + 2*(k-9)))

    def reader(i):
        return i, frames[i].copy()

    history = [MotionPoint(k, 42 + 2*k + 7.0, 66.0) for k in range(5, 8)]
    normal = run_reacquisition(
        frame_reader=reader, frame_count=len(frames), frame_width=240, frame_height=140,
        lost_frame=8, reference_bank=bank, motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=220, max_radius_px=240, radius_growth_px_per_frame=0,
            max_frames=6, confirm_frames=3, confirm_window_frames=5,
            max_confirmation_error_px=60, min_confidence_gap=0.0,
        ),
        motion_broken=False,
    )
    assert normal.status != "CONFIRMED"

    robust = run_reacquisition(
        frame_reader=reader, frame_count=len(frames), frame_width=240, frame_height=140,
        lost_frame=8, reference_bank=bank, motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=220, max_radius_px=240, radius_growth_px_per_frame=0,
            max_frames=6, confirm_frames=3, confirm_window_frames=5,
            max_confirmation_error_px=60, min_confidence_gap=0.0,
            robust_motion_break_enabled=True,
            broken_min_score=0.25, broken_min_identity_similarity=0.25, broken_min_confidence_gap=0.0,
            tracklets_enabled=True, tracklet_top_k=3, tracklet_confirm_hits=3,
            tracklet_window_frames=5, tracklet_max_link_distance_px=35,
            global_shift_enabled=False,
        ),
        motion_broken=True,
    )
    assert robust.status == "CONFIRMED"
    assert robust.motion_broken_mode
    assert robust.motion_relocked
    assert robust.confirmed_center[0] > 165
    assert any(f.tracklet_id > 0 for f in robust.frames)


def test_hotfix6_config_has_cpu_budget_and_event_driven_motion():
    from navmin.vision.processors.vt11.config.loader import load_config
    cfg = load_config()
    assert cfg["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert cfg["tracking"]["motion_break_enabled"] is True
    assert cfg["tracking"]["global_shift_enabled"] is True
    assert cfg["tracking"]["performance_target_fps"] == 25.0
    assert cfg["reacquisition"]["tracklets_enabled"] is True
    assert cfg["reacquisition"]["full_frame_after_frames"] == 15


def test_hotfix6_tracking_accepts_strong_visual_jump_and_resets_motion():
    from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking
    base = _target_frame(42)
    sel = make_operator_selection(frame_index=0, click_xy=(49, 66), radius_px=55,
                                  frame_width=240, frame_height=140)
    det = detect_local_candidates(base, sel, top_k=5, config=_detector_cfg())
    bank = ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(base, det.candidates[0])])
    frames = [_target_frame(170), _target_frame(173), _target_frame(176), _target_frame(179)]
    def reader(i): return i, frames[i].copy()
    history = [(45.0, 66.0), (47.0, 66.0), (49.0, 66.0)]
    result = run_tracking(
        frame_reader=reader, frame_count=len(frames), frame_width=240, frame_height=140,
        start_frame=0, reference_bank=bank, initial_history=history,
        initial_history_frames=[-3, -2, -1], detector_config=_detector_cfg(),
        config=TrackingConfig(
            search_radius_px=190, max_search_radius_px=220, min_search_radius_px=50,
            identity_weight=0.60, motion_weight=0.30, size_weight=0.10,
            min_track_score=0.80, min_identity_similarity=0.40, min_confidence_gap=0.0,
            lost_after_misses=1, motion_break_enabled=True, motion_break_error_px=60,
            motion_break_visual_identity_min=0.60, motion_break_visual_gap_min=0.0,
            motion_break_visual_size_min=0.01, motion_relock_frames=3,
            global_shift_enabled=False,
        ),
    )
    assert result.frames[0].selected is not None
    assert result.frames[0].diagnostic_reason == "ACCEPTED_MOTION_BREAK"
    assert result.frames[0].motion_break_triggered
    assert result.motion_break_events >= 1
    assert any(f.diagnostic_reason == "ACCEPTED_MOTION_RELOCK" for f in result.frames[1:])


def test_hotfix6_diagnostic_columns_include_runtime_budget(tmp_path):
    import csv
    from pathlib import Path
    from navmin.vision.processors.vt11.evaluation.diagnostic_log import DiagnosticLogConfig, DiagnosticRunLogger
    from navmin.vision.processors.vt11.tracking.live_events import LiveFrameEvent
    from navmin.vision.processors.vt11.tracking.tracker import TrackFrameResult
    logger = DiagnosticRunLogger(DiagnosticLogConfig(output_root=str(tmp_path), save_incident_frames=False, make_zip_bundle=False), metadata={"video":{"fps":25.0}})
    item = TrackFrameResult(
        frame_index=1, predicted_xy=(10,10), selected=None, selected_score=0.0,
        confidence_gap=0.0, candidate_count=0, identity_similarity=0.0, motion_score=0.0,
        size_score=0.0, status="LOW_CONFIDENCE", miss_count=1,
        motion_break_active=True, global_shift_dx_px=12.0, global_shift_dy_px=-7.0,
        global_shift_response=0.8, global_shift_method="PHASE", global_shift_time_ms=1.5,
        detector_time_ms=3.0, identity_time_ms=2.0, motion_time_ms=0.1,
        performance_budget_ms=40.0, over_budget=False, processing_time_ms=10.0,
    )
    logger.log_event(LiveFrameEvent("TRACK", 1, np.zeros((20,20,3),np.uint8), item))
    class R:
        status="STOPPED"; lost_events=0; reacq_successes=0; t_reacq_frames=(); mean_t_reacq_frames=0; max_t_reacq_frames=0
        adaptation_mode="adaptive_memory"; adaptation_updates=0; adaptation_slow_updates=0; adaptation_fast_updates=0
        adaptation_recent_updates=0; adaptation_recent_count=0; mean_track_confidence=0; track_accepted_frames=0; final_frame=1
        motion_break_events=1; robust_reacq_count=0; over_budget_frames=0
    logger.finish(R())
    with open(Path(logger.path)/"frames.csv", encoding="utf-8-sig", newline="") as fh:
        row=next(csv.DictReader(fh))
    assert row["motion_break_active"] == "1"
    assert row["global_shift_method"] == "PHASE"
    assert float(row["performance_budget_ms"]) == 40.0
    assert int(row["over_budget"]) == 0


def test_hotfix6_feature_affine_fallback_handles_small_rotation():
    prev = _texture(180, 260)
    center = (130.0, 90.0)
    mat = cv2.getRotationMatrix2D(center, 3.0, 1.0)
    mat[:, 2] += np.array([10.0, -6.0])
    curr = cv2.warpAffine(prev, mat, (260, 180), borderMode=cv2.BORDER_REFLECT)
    est = estimate_background_motion(
        prev, curr,
        config=BackgroundMotionConfig(
            downscale=0.6, max_dimension_px=260,
            phase_min_response=0.99,  # force affine fallback for this test
            use_feature_fallback=True, min_feature_inliers=6,
        ),
    )
    assert est.valid
    assert est.method == "FEATURE_AFFINE"
    p = (80.0, 70.0)
    expected = mat @ np.array([p[0], p[1], 1.0])
    got = transform_point(p, est)
    assert got[0] == pytest.approx(float(expected[0]), abs=4.0)
    assert got[1] == pytest.approx(float(expected[1]), abs=4.0)
