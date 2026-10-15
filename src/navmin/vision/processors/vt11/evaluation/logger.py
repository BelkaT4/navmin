"""VT-10 CSV exporters for TRACK/REACQ/evaluation results."""
from __future__ import annotations

from pathlib import Path
import csv


def session_rows(session):
    rows = []
    for item in session.all_track_frames:
        rows.append((item.frame_index, "TRACK", item))
    for rr in session.reacquisitions:
        for item in rr.frames:
            rows.append((item.frame_index, "REACQ", item))
    rows.sort(key=lambda x: (x[0], 0 if x[1] == "TRACK" else 1))
    return rows


def export_session_csv(session, path: str | Path, *, fps: float | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "frame", "timestamp_s", "mode", "status", "x", "y", "pred_x", "pred_y",
        "prediction_error_px", "search_radius_px", "selected_score", "second_score", "identity_similarity",
        "appearance_similarity", "edge_similarity", "projection_similarity", "geometry_similarity",
        "motion_score", "size_score", "confidence_gap", "confidence", "candidate_count",
        "diagnostic_reason", "adaptation_reason", "adaptation_update_count", "processing_time_ms",
    ]
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for frame_index, mode, item in session_rows(session):
            sel = item.selected
            w.writerow({
                "frame": frame_index,
                "timestamp_s": "" if not fps or fps <= 0 else f"{frame_index / fps:.9f}",
                "mode": mode,
                "status": item.status,
                "x": "" if sel is None else f"{sel.center_x:.6f}",
                "y": "" if sel is None else f"{sel.center_y:.6f}",
                "pred_x": f"{item.predicted_xy[0]:.6f}",
                "pred_y": f"{item.predicted_xy[1]:.6f}",
                "prediction_error_px": "" if item.prediction_error_px is None else f"{item.prediction_error_px:.6f}",
                "search_radius_px": item.search_radius_px,
                "selected_score": f"{item.selected_score:.9f}",
                "second_score": f"{getattr(item, 'second_score', 0.0):.9f}",
                "identity_similarity": f"{item.identity_similarity:.9f}",
                "appearance_similarity": f"{getattr(item, 'appearance_similarity', 0.0):.9f}",
                "edge_similarity": f"{getattr(item, 'edge_similarity', 0.0):.9f}",
                "projection_similarity": f"{getattr(item, 'projection_similarity', 0.0):.9f}",
                "geometry_similarity": f"{getattr(item, 'geometry_similarity', 0.0):.9f}",
                "motion_score": f"{item.motion_score:.9f}",
                "size_score": f"{item.size_score:.9f}",
                "confidence_gap": f"{item.confidence_gap:.9f}",
                "confidence": f"{item.confidence:.9f}",
                "candidate_count": item.candidate_count,
                "diagnostic_reason": item.diagnostic_reason,
                "adaptation_reason": getattr(item, "adaptation_reason", "BLOCKED_REACQ" if mode == "REACQ" else ""),
                "adaptation_update_count": getattr(item, "adaptation_update_count", ""),
                "processing_time_ms": f"{getattr(item, 'processing_time_ms', 0.0):.6f}",
            })
    return p


def export_evaluation_csv(summary, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "outcome", "pred_x", "pred_y", "gt_x", "gt_y", "matched_id", "center_error_px"])
        for row in summary.frames:
            px, py = ("", "") if row.predicted_xy is None else row.predicted_xy
            gx, gy = ("", "") if row.target_xy is None else row.target_xy
            w.writerow([row.frame_index, row.outcome, px, py, gx, gy, row.matched_id or "", "" if row.center_error_px is None else row.center_error_px])
    return p
