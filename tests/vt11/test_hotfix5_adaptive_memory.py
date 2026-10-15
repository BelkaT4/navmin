import csv
from pathlib import Path

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter
from navmin.vision.processors.vt11.evaluation.diagnostic_log import DiagnosticLogConfig, DiagnosticRunLogger
from navmin.vision.processors.vt11.identity.descriptor import TargetDescriptor, descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.tracking.live_events import LiveFrameEvent
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def _descriptor(value: float, *, size: float = 100.0) -> TargetDescriptor:
    a = np.full((16, 16), value, np.float32)
    p = np.full(16, value, np.float32)
    g = np.asarray([value, value], np.float32)
    return TargetDescriptor(a, p, p.copy(), a.copy(), g, size, (1, 2, 10, 10), value, min(1.0, value))


def test_hotfix5_adaptive_memory_fast_and_recent_do_not_modify_base():
    base = _descriptor(0.0, size=100)
    bank = ReferenceBank(max_size=3, descriptors=[base])
    before = bank.descriptors[0].appearance.copy()
    cfg = EMAConfig(
        mode="adaptive_memory",
        alpha=0.05,
        min_confidence=0.95,  # deliberately block slow EMA
        min_identity_similarity=0.95,
        min_confidence_gap=0.2,
        max_prediction_error_px=5,
        fast_alpha=0.25,
        recent_bank_size=5,
        fast_min_confidence=0.50,
        fast_min_identity_similarity=0.25,
        fast_min_confidence_gap=0.05,
        fast_min_motion_score=0.85,
        fast_max_prediction_error_px=20,
        fast_min_size_score=0.01,
        drift_score_trigger=0.20,
        drift_similarity_trigger=0.90,
        drift_size_ratio_trigger=1.2,
    )
    adapter = EMAReferenceAdapter(bank, cfg, identity_weights={"appearance": 1.0})
    event = adapter.consider(
        frame_index=10,
        descriptor=_descriptor(0.8, size=170),
        confidence=0.65,
        identity_similarity=0.40,
        confidence_gap=0.12,
        prediction_error_px=4.0,
        motion_score=0.97,
        size_score=0.30,
        force_fast=True,
    )
    assert event.updated
    assert "FAST" in event.layer
    assert "RECENT" in event.layer
    assert bank.fast_descriptor is not None
    assert bank.recent_count == 1
    assert bank.adaptive_descriptor is None  # slow gate was intentionally blocked
    assert np.array_equal(bank.descriptors[0].appearance, before)


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.6,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1000,
        spatial_sigma_px=35.0,
        adaptive_area_enabled=False,
    )


def _frame(x: int):
    f = np.full((120, 220, 3), 25, np.uint8)
    cv2.rectangle(f, (x, 52), (x + 12, 61), (225, 225, 225), -1)
    cv2.rectangle(f, (x + 8, 54), (x + 16, 58), (145, 145, 145), -1)
    return f


def test_hotfix5_track_warmup_recovers_from_bad_train_velocity():
    frames = [_frame(72 + 2 * k) for k in range(4)]
    # Build a position-independent descriptor from the actual target.
    sel = make_operator_selection(frame_index=0, click_xy=(78, 57), radius_px=45, frame_width=220, frame_height=120)
    det = detect_local_candidates(frames[0], sel, top_k=5, config=_detector_cfg())
    assert det.candidates
    bank = ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(frames[0], det.candidates[0])])

    def reader(i):
        return i, frames[i].copy()

    # Last TRAIN point is near x=150 and apparent TRAIN velocity is huge. The
    # real first TRACK target is near x=78. Warm-up must search around the last
    # confirmed point with a large ROI and ignore most of the bad motion prior.
    history = [(50.0, 57.0), (100.0, 57.0), (150.0, 57.0)]
    history_frames = [-3, -2, -1]
    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=220,
        frame_height=120,
        start_frame=0,
        reference_bank=bank,
        initial_history=history,
        initial_history_frames=history_frames,
        detector_config=_detector_cfg(),
        config=TrackingConfig(
            search_radius_px=30,
            min_search_radius_px=20,
            max_search_radius_px=70,
            top_k=5,
            min_confidence_gap=0.0,
            lost_after_misses=1,
            track_warmup_frames=3,
            warmup_search_radius_px=110,
            warmup_motion_weight=0.02,
            warmup_identity_weight=0.88,
            warmup_size_weight=0.10,
            warmup_motion_sigma_px=120,
            warmup_min_track_score=0.2,
            warmup_min_identity_similarity=0.2,
            warmup_min_confidence_gap=0.0,
        ),
    )
    assert result.frames[0].selected is not None
    assert result.frames[0].tracking_phase == "TRACK_WARMUP"
    assert result.frames[0].diagnostic_reason == "ACCEPTED_TRACK_WARMUP"
    assert result.status == "COMPLETED"


def test_hotfix5_diagnostic_log_contains_memory_and_phase_columns(tmp_path):
    cfg = DiagnosticLogConfig(output_root=str(tmp_path), save_incident_frames=False, make_zip_bundle=False)
    logger = DiagnosticRunLogger(cfg, metadata={"video": {"fps": 20.0}})
    from navmin.vision.processors.vt11.tracking.tracker import TrackFrameResult
    item = TrackFrameResult(
        frame_index=1, predicted_xy=(10, 10), selected=None, selected_score=0.4,
        confidence_gap=0.1, candidate_count=1, identity_similarity=0.4,
        motion_score=0.95, size_score=0.8, status="LOW_CONFIDENCE", miss_count=1,
        tracking_phase="TRACK_WARMUP", adaptation_layer="FAST+RECENT",
        adaptation_drift_score=0.42, adaptation_drift_reason="SIZE",
        adaptation_recent_count=3, adaptation_fast_remaining=4,
    )
    logger.log_event(LiveFrameEvent("TRACK", 1, np.zeros((20, 20, 3), np.uint8), item))
    class R:
        status="STOPPED"; lost_events=0; reacq_successes=0; t_reacq_frames=(); mean_t_reacq_frames=0; max_t_reacq_frames=0
        adaptation_mode="adaptive_memory"; adaptation_updates=1; adaptation_slow_updates=0; adaptation_fast_updates=1
        adaptation_recent_updates=1; adaptation_recent_count=3; mean_track_confidence=0.4; track_accepted_frames=0; final_frame=1
    logger.finish(R())
    with open(Path(logger.path) / "frames.csv", encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["tracking_phase"] == "TRACK_WARMUP"
    assert row["adaptation_layer"] == "FAST+RECENT"
    assert float(row["adaptation_drift_score"]) == 0.42
    assert int(row["adaptation_recent_count"]) == 3


def test_hotfix5_drift_rescue_requires_strong_spatial_evidence():
    from navmin.vision.processors.vt11.acquisition.candidate_detector import Candidate
    from navmin.vision.processors.vt11.acquisition.roi import RoiRect
    from navmin.vision.processors.vt11.identity.descriptor import DescriptorSimilarity
    from navmin.vision.processors.vt11.tracking.tracker import TrackCandidateScore, _drift_rescue_allowed
    cand = Candidate(1, RoiRect(90, 90, 12, 10), 96.0, 95.0, 90, 10.0, 20.0, 4.0, 0.9, 0.9, 0.9)
    sim = DescriptorSimilarity(0.4, 0.4, 0.8, 0.7, 0.5, 0.36, 0.4, 0.4)
    score = TrackCandidateScore(cand, _descriptor(0.5), sim, 0.97, 0.5, 0.44)
    cfg = TrackingConfig(
        drift_rescue_enabled=True,
        drift_rescue_min_motion_score=0.92,
        drift_rescue_min_identity_similarity=0.30,
        drift_rescue_min_confidence_gap=0.08,
        drift_rescue_min_size_score=0.04,
        drift_rescue_max_prediction_error_px=18.0,
    )
    assert _drift_rescue_allowed(
        score, 0.12, (94.0, 95.0), cfg, consecutive_rescues=0,
        reasons=["SCORE_BELOW_THRESHOLD", "IDENTITY_BELOW_THRESHOLD"],
    )
    assert not _drift_rescue_allowed(
        score, 0.02, (94.0, 95.0), cfg, consecutive_rescues=0,
        reasons=["SCORE_BELOW_THRESHOLD", "IDENTITY_BELOW_THRESHOLD"],
    )
