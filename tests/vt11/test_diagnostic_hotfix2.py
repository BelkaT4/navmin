from pathlib import Path
from types import SimpleNamespace
import csv
import json

import numpy as np
import pytest

from navmin.vision.processors.vt11.evaluation.diagnostic_log import DiagnosticLogConfig, DiagnosticRunLogger
from navmin.vision.processors.vt11.tracking.live_events import LiveFrameEvent


def _result(*, status="TRACK", reason="ACCEPTED", adapt_reason="ADAPT_OFF", updated=False):
    return SimpleNamespace(
        status=status,
        diagnostic_reason=reason,
        predicted_xy=(40.0, 50.0),
        selected=None,
        best_candidate_center=(42.0, 51.0),
        best_candidate_bbox=(37, 47, 10, 8),
        best_candidate_area_px=43,
        best_candidate_mean_response=12.5,
        best_candidate_peak_response=31.0,
        candidate_count=5,
        selected_score=0.61,
        second_score=0.55,
        identity_similarity=0.57,
        appearance_similarity=0.62,
        edge_similarity=0.51,
        projection_similarity=0.48,
        geometry_similarity=0.73,
        motion_score=0.90,
        size_score=0.88,
        confidence_gap=0.06,
        confidence=0.64,
        prediction_error_px=2.2,
        search_radius_px=90,
        speed_px_per_frame=3.0,
        motion_uncertainty_px=1.4,
        miss_count=0 if status == "TRACK" else 2,
        confirmation_streak=0,
        detector_threshold=7.2,
        detector_median_response=1.3,
        detector_robust_sigma=2.1,
        detector_raw_components=11,
        detector_valid_components=6,
        adaptation_updated=updated,
        adaptation_reason=adapt_reason,
        adaptation_update_count=2 if updated else 1,
        processing_time_ms=8.4,
    )


def _event(frame_index, *, mode="TRACK", **kwargs):
    frame = np.full((80, 100, 3), 40 + frame_index, dtype=np.uint8)
    return LiveFrameEvent(mode=mode, frame_index=frame_index, frame_bgr=frame, result=_result(**kwargs))


def _session(status="LOST_UNRECOVERED"):
    return SimpleNamespace(
        status=status,
        lost_events=1,
        reacq_successes=0,
        t_reacq_frames=(),
        mean_t_reacq_frames=0.0,
        max_t_reacq_frames=0,
        adaptation_mode="ema",
        adaptation_updates=2,
        mean_track_confidence=0.61,
        track_accepted_frames=12,
        final_frame=20,
    )


def test_hotfix2_logger_writes_full_frame_csv_and_summary(tmp_path):
    cfg = DiagnosticLogConfig(output_root=str(tmp_path), save_incident_frames=False)
    logger = DiagnosticRunLogger(cfg, metadata={"checkpoint": "VT-10-HOTFIX-2", "video": {"fps": 25.0}, "adaptation_mode": "ema"})
    logger.log_event(_event(4, adapt_reason="UPDATED", updated=True))
    logger.finish(_session("COMPLETED"))

    run = Path(logger.path)
    assert (run / "frames.csv").exists()
    assert (run / "events.log").exists()
    assert (run / "run_config.json").exists()
    assert (run / "summary.json").exists()
    assert (run / "quick_diagnosis.txt").exists()
    assert (run / "diagnostic_bundle.zip").exists()

    with (run / "frames.csv").open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["frame"] == "5"
    assert row["appearance_similarity"].startswith("0.620")
    assert row["edge_similarity"].startswith("0.510")
    assert row["projection_similarity"].startswith("0.480")
    assert row["best_x"] == "42.0"
    assert row["adaptation_reason"] == "UPDATED"
    assert row["adaptation_updated"] == "1"

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETED"
    assert summary["adaptation_mode"] == "ema"
    assert summary["adaptation_updates"] == 2
    assert summary["adaptation_reason_counts"]["UPDATED"] == 1


def test_hotfix2_logger_saves_frames_around_lost(tmp_path):
    cfg = DiagnosticLogConfig(
        output_root=str(tmp_path), save_incident_frames=True,
        pre_lost_frames=2, post_lost_frames=2,
    )
    logger = DiagnosticRunLogger(cfg, metadata={"video": {"fps": 20.0}})
    logger.log_event(_event(1))
    logger.log_event(_event(2, status="LOW_CONFIDENCE", reason="IDENTITY_BELOW_THRESHOLD"))
    logger.log_event(_event(3, status="LOST", reason="IDENTITY_BELOW_THRESHOLD"))
    logger.log_event(_event(4, mode="REACQ", status="SEARCHING", reason="SCORE_BELOW_THRESHOLD"))
    logger.log_event(_event(5, mode="REACQ", status="SEARCHING", reason="GAP_BELOW_THRESHOLD"))
    logger.finish(_session())

    incidents = list((Path(logger.path) / "incidents").glob("lost_*"))
    assert len(incidents) == 1
    pngs = list(incidents[0].glob("*.png"))
    # 3 buffered frames (2 pre + LOST) + 2 post; raw + annotated for each.
    assert len(pngs) == 10
    assert any("annotated" in p.name for p in pngs)


def test_hotfix2_log_config_validation():
    with pytest.raises(ValueError):
        DiagnosticLogConfig(output_root="").validate()
    with pytest.raises(ValueError):
        DiagnosticLogConfig(pre_lost_frames=-1).validate()
    DiagnosticLogConfig(pre_lost_frames=0, post_lost_frames=0).validate()


def test_hotfix2_rev1_unicode_incident_path(tmp_path):
    unicode_root = tmp_path / "Диагностика_Дмитрий"
    cfg = DiagnosticLogConfig(
        output_root=str(unicode_root), save_incident_frames=True,
        pre_lost_frames=1, post_lost_frames=1,
    )
    logger = DiagnosticRunLogger(cfg, metadata={"video": {"fps": 20.0}})
    logger.log_event(_event(10))
    logger.log_event(_event(11, status="LOST", reason="IDENTITY_BELOW_THRESHOLD"))
    logger.log_event(_event(12, mode="REACQ", status="SEARCHING", reason="SCORE_BELOW_THRESHOLD"))
    logger.finish(_session())

    incidents = list((Path(logger.path) / "incidents").glob("lost_*"))
    assert len(incidents) == 1
    pngs = list(incidents[0].glob("*.png"))
    assert len(pngs) == 6
    assert all(p.stat().st_size > 0 for p in pngs)
