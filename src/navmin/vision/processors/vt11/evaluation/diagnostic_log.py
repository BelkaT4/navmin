"""VT-10 HOTFIX-2 persistent diagnostics for LIVE/FAST tracking.

The GUI is intentionally not the source of truth for debugging.  This module
writes a crash-tolerant, UTF-8 diagnostic bundle while TRACK/REACQ is running:

- ``frames.csv``: one row per processed frame;
- ``events.log``: compact human-readable event stream;
- ``run_config.json``: exact parameters used for the run;
- ``summary.json`` / ``summary.txt``: terminal status and aggregate counters;
- ``incidents/``: raw + annotated frames around each TRACK LOST event.

The logger is Qt independent and can be used from LIVE worker, FAST mode, tests,
or batch experiments.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
import zipfile
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class DiagnosticLogConfig:
    enabled: bool = True
    output_root: str = "logs/live_diagnostics"
    save_incident_frames: bool = True
    pre_lost_frames: int = 8
    post_lost_frames: int = 12
    flush_every_frame: bool = True
    make_zip_bundle: bool = True

    def validate(self) -> None:
        if self.pre_lost_frames < 0:
            raise ValueError("pre_lost_frames must be >= 0")
        if self.post_lost_frames < 0:
            raise ValueError("post_lost_frames must be >= 0")
        if not str(self.output_root).strip():
            raise ValueError("output_root must not be empty")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)) if value.size else None,
            "max": float(np.max(value)) if value.size else None,
            "mean": float(np.mean(value)) if value.size else None,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _candidate_info(item: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    selected = getattr(item, "selected", None)
    best_center = getattr(item, "best_candidate_center", None)
    best_bbox = getattr(item, "best_candidate_bbox", None)
    best_area = getattr(item, "best_candidate_area_px", 0)
    best_mean_response = getattr(item, "best_candidate_mean_response", 0.0)
    best_peak_response = getattr(item, "best_candidate_peak_response", 0.0)
    if best_center is None and selected is not None:
        best_center = (selected.center_x, selected.center_y)
    if best_bbox is None and selected is not None:
        b = selected.bbox
        best_bbox = (b.x, b.y, b.width, b.height)
    if not best_area and selected is not None:
        best_area = selected.area_px
    if not best_mean_response and selected is not None:
        best_mean_response = selected.mean_response
    if not best_peak_response and selected is not None:
        best_peak_response = selected.peak_response
    return selected, best_center, best_bbox, best_area, best_mean_response, best_peak_response


class DiagnosticRunLogger:
    """Persistent one-run logger.

    Rows are flushed on every frame by default, so a useful log survives even
    if the GUI or decoder later fails.  A small ring buffer keeps only the last
    ``pre_lost_frames`` images in memory; full videos are never retained.
    """

    COLUMNS = [
        "frame", "frame_index", "timestamp_s", "mode", "status", "diagnostic_reason",
        "pred_x", "pred_y", "accepted_x", "accepted_y", "best_x", "best_y",
        "best_bbox_x", "best_bbox_y", "best_bbox_w", "best_bbox_h", "best_area_px",
        "best_mean_response", "best_peak_response", "candidate_count",
        "selected_score", "second_score", "identity_similarity", "appearance_similarity",
        "edge_similarity", "projection_similarity", "geometry_similarity", "color_similarity",
        "thermal_contrast_similarity", "hot_area_similarity", "peak_prominence_similarity",
        "hot_core_compactness_similarity", "radial_hotspot_profile_similarity", "energy_concentration_similarity",
        "thermal_2d_ratio_similarity", "thermal_background_clutter",
        "static_overlay_penalty", "hud_probability", "hud_hard_overlap", "hud_soft_overlap",
        "hud_hard_reject", "hud_negative_penalty", "reticle_occlusion_active", "reticle_occlusion_frames", "reticle_exit_hits",
        "motion_score",
        "size_score", "confidence_gap", "confidence", "prediction_error_px",
        "search_radius_px", "speed_px_per_frame", "motion_uncertainty_px",
        "miss_count", "confirmation_streak", "confirmation_window_size",
        "expected_area_px", "scale_state", "scale_base_area_px", "scale_current_area_px", "scale_pending_count", "scale_update_reason",
        "reliable_area_anchor_px", "scale_log_distance", "scale_allowed_log_distance", "scale_excess_log_distance",
        "scale_nonlinear_gate", "scale_anchor_age_frames", "scale_quarantine_active", "scale_quarantine_hits",
        "scale_quarantine_required_hits", "scale_quarantine_confirmed",
        "detector_effective_min_area_px", "detector_effective_max_area_px",
        "detector_threshold", "detector_median_response",
        "detector_robust_sigma", "detector_raw_components", "detector_valid_components",
        "adaptation_updated", "adaptation_reason", "adaptation_update_count",
        "tracking_phase", "adaptation_layer", "adaptation_drift_score", "adaptation_drift_reason",
        "adaptation_recent_count", "adaptation_fast_remaining",
        "motion_break_active", "motion_break_triggered", "motion_break_reason", "motion_broken_mode",
        "auto_promoted_motion_break", "promotion_evidence_hits", "promotion_reason",
        "tracklet_id", "active_tracklets", "selected_tracklet_x", "selected_tracklet_y", "selected_tracklet_area_px",
        "selected_tracklet_identity", "selected_tracklet_score", "selected_tracklet_gap", "selected_tracklet_thermal_profile",
        "reacq_mode", "candidate_channel",
        "reliable_anchor_x", "reliable_anchor_y", "anchor_distance_px", "anchor_distance_norm",
        "anchor_age_frames", "anchor_allowed_distance_px",
        "frame_reliability_score", "frame_reliability_residual", "frame_reliability_phase_response",
        "frame_degraded_active", "frame_degraded_frames",
        "jump_quarantine_active", "jump_quarantine_hits", "jump_quarantine_required_hits",
        "jump_quarantine_confirmed", "jump_global_motion_veto", "anchor_continuity_override",
        "far_jump_candidate", "micro_candidate_count",
        "global_shift_dx_px", "global_shift_dy_px", "global_shift_response", "global_shift_method",
        "global_shift_time_ms", "detector_time_ms", "identity_time_ms", "motion_time_ms",
        "performance_budget_ms", "over_budget", "processing_time_ms",
    ]

    def __init__(
        self,
        config: DiagnosticLogConfig | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or DiagnosticLogConfig()
        self.config.validate()
        self.metadata = dict(metadata or {})
        self.enabled = bool(self.config.enabled)
        self.run_dir: Path | None = None
        self._csv_fh = None
        self._csv_writer = None
        self._log_fh = None
        self._pre_buffer: deque[tuple[int, str, str, np.ndarray, Any]] = deque(
            maxlen=max(1, self.config.pre_lost_frames + 1)
        )
        self._active_incident: dict[str, Any] | None = None
        self._incident_count = 0
        self._event_count = 0
        self._status_counts: Counter[str] = Counter()
        self._diagnostic_reason_counts: Counter[str] = Counter()
        self._adaptation_reason_counts: Counter[str] = Counter()
        self._first_lost_frame: int | None = None
        self._last_logged_frame: int | None = None
        self._closed = False
        if self.enabled:
            self._open()

    @property
    def path(self) -> str:
        return "" if self.run_dir is None else str(self.run_dir)

    def _open(self) -> None:
        root = Path(self.config.output_root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "incidents").mkdir(exist_ok=True)

        with (self.run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "created_local": datetime.now().isoformat(timespec="seconds"),
                    "logger": _jsonable(self.config),
                    "run": _jsonable(self.metadata),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

        self._csv_fh = (self.run_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="")
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self.COLUMNS)
        self._csv_writer.writeheader()
        self._csv_fh.flush()
        self._log_fh = (self.run_dir / "events.log").open("w", encoding="utf-8")
        self._write_text("DIAGNOSTIC START")
        if self.metadata:
            adapt_mode = self.metadata.get("adaptation_mode", "?")
            video = self.metadata.get("video", {})
            self._write_text(
                f"video={video.get('path', video.get('name', '?'))}; adaptation_mode={adapt_mode}; "
                f"checkpoint={self.metadata.get('checkpoint', '?')}"
            )


    def save_aux_image(self, name: str, image: np.ndarray) -> str:
        """Save an auxiliary diagnostic image (e.g. learned reticle mask)."""
        if not self.enabled or self.run_dir is None or image is None:
            return ""
        target = self.run_dir / str(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(image)
        ok, encoded = cv2.imencode(target.suffix or ".png", arr)
        if not ok:
            return ""
        target.write_bytes(encoded.tobytes())
        return str(target)

    def _write_text(self, message: str) -> None:
        if self._log_fh is None:
            return
        self._log_fh.write(f"[{datetime.now().isoformat(timespec='milliseconds')}] {message}\n")
        self._log_fh.flush()

    def _row(self, event: Any) -> dict[str, Any]:
        item = event.result
        selected, best_center, best_bbox, best_area, best_mean, best_peak = _candidate_info(item)
        pred = getattr(item, "predicted_xy", (None, None))
        accepted_x = "" if selected is None else float(selected.center_x)
        accepted_y = "" if selected is None else float(selected.center_y)
        best_x = "" if best_center is None else float(best_center[0])
        best_y = "" if best_center is None else float(best_center[1])
        if best_bbox is None:
            bx = by = bw = bh = ""
        else:
            bx, by, bw, bh = best_bbox
        fps = float(self.metadata.get("video", {}).get("fps", 0.0) or 0.0)
        return {
            "frame": int(event.frame_index) + 1,
            "frame_index": int(event.frame_index),
            "timestamp_s": "" if fps <= 0 else f"{event.frame_index / fps:.9f}",
            "mode": event.mode,
            "status": getattr(item, "status", ""),
            "diagnostic_reason": getattr(item, "diagnostic_reason", ""),
            "pred_x": "" if pred[0] is None else f"{float(pred[0]):.6f}",
            "pred_y": "" if pred[1] is None else f"{float(pred[1]):.6f}",
            "accepted_x": accepted_x,
            "accepted_y": accepted_y,
            "best_x": best_x,
            "best_y": best_y,
            "best_bbox_x": bx,
            "best_bbox_y": by,
            "best_bbox_w": bw,
            "best_bbox_h": bh,
            "best_area_px": int(best_area or 0),
            "best_mean_response": f"{float(best_mean or 0.0):.6f}",
            "best_peak_response": f"{float(best_peak or 0.0):.6f}",
            "candidate_count": int(getattr(item, "candidate_count", 0)),
            "selected_score": f"{float(getattr(item, 'selected_score', 0.0)):.9f}",
            "second_score": f"{float(getattr(item, 'second_score', 0.0)):.9f}",
            "identity_similarity": f"{float(getattr(item, 'identity_similarity', 0.0)):.9f}",
            "appearance_similarity": f"{float(getattr(item, 'appearance_similarity', 0.0)):.9f}",
            "edge_similarity": f"{float(getattr(item, 'edge_similarity', 0.0)):.9f}",
            "projection_similarity": f"{float(getattr(item, 'projection_similarity', 0.0)):.9f}",
            "geometry_similarity": f"{float(getattr(item, 'geometry_similarity', 0.0)):.9f}",
            "color_similarity": f"{float(getattr(item, 'color_similarity', 0.0)):.9f}",
            "thermal_contrast_similarity": f"{float(getattr(item, 'thermal_contrast_similarity', 0.0)):.9f}",
            "hot_area_similarity": f"{float(getattr(item, 'hot_area_similarity', 0.0)):.9f}",
            "peak_prominence_similarity": f"{float(getattr(item, 'peak_prominence_similarity', 0.0)):.9f}",
            "hot_core_compactness_similarity": f"{float(getattr(item, 'hot_core_compactness_similarity', 0.0)):.9f}",
            "radial_hotspot_profile_similarity": f"{float(getattr(item, 'radial_hotspot_profile_similarity', 0.0)):.9f}",
            "energy_concentration_similarity": f"{float(getattr(item, 'energy_concentration_similarity', 0.0)):.9f}",
            "thermal_2d_ratio_similarity": f"{float(getattr(item, 'thermal_2d_ratio_similarity', 0.0)):.9f}",
            "thermal_background_clutter": f"{float(getattr(item, 'thermal_background_clutter', 0.0)):.9f}",
            "static_overlay_penalty": f"{float(getattr(item, 'static_overlay_penalty', 0.0)):.9f}",
            "hud_probability": f"{float(getattr(item, 'hud_probability', 0.0)):.9f}",
            "hud_hard_overlap": f"{float(getattr(item, 'hud_hard_overlap', 0.0)):.9f}",
            "hud_soft_overlap": f"{float(getattr(item, 'hud_soft_overlap', 0.0)):.9f}",
            "hud_hard_reject": int(bool(getattr(item, 'hud_hard_reject', False))),
            "hud_negative_penalty": f"{float(getattr(item, 'hud_negative_penalty', 0.0)):.9f}",
            "reticle_occlusion_active": int(bool(getattr(item, 'reticle_occlusion_active', False))),
            "reticle_occlusion_frames": int(getattr(item, 'reticle_occlusion_frames', 0)),
            "reticle_exit_hits": int(getattr(item, 'reticle_exit_hits', 0)),
            "motion_score": f"{float(getattr(item, 'motion_score', 0.0)):.9f}",
            "size_score": f"{float(getattr(item, 'size_score', 0.0)):.9f}",
            "confidence_gap": f"{float(getattr(item, 'confidence_gap', 0.0)):.9f}",
            "confidence": f"{float(getattr(item, 'confidence', 0.0)):.9f}",
            "prediction_error_px": "" if getattr(item, "prediction_error_px", None) is None else f"{float(item.prediction_error_px):.6f}",
            "search_radius_px": int(getattr(item, "search_radius_px", 0)),
            "speed_px_per_frame": f"{float(getattr(item, 'speed_px_per_frame', 0.0)):.6f}",
            "motion_uncertainty_px": f"{float(getattr(item, 'motion_uncertainty_px', 0.0)):.6f}",
            "miss_count": int(getattr(item, "miss_count", 0)),
            "confirmation_streak": int(getattr(item, "confirmation_streak", 0)),
            "confirmation_window_size": int(getattr(item, "confirmation_window_size", 0)),
            "expected_area_px": f"{float(getattr(item, 'expected_area_px', 0.0)):.6f}",
            "scale_state": getattr(item, "scale_state", ""),
            "scale_base_area_px": f"{float(getattr(item, 'scale_base_area_px', 0.0)):.6f}",
            "scale_current_area_px": f"{float(getattr(item, 'scale_current_area_px', 0.0)):.6f}",
            "scale_pending_count": int(getattr(item, "scale_pending_count", 0)),
            "scale_update_reason": getattr(item, "scale_update_reason", ""),
            "reliable_area_anchor_px": f"{float(getattr(item, 'reliable_area_anchor_px', 0.0)):.6f}",
            "scale_log_distance": f"{float(getattr(item, 'scale_log_distance', 0.0)):.6f}",
            "scale_allowed_log_distance": f"{float(getattr(item, 'scale_allowed_log_distance', 0.0)):.6f}",
            "scale_excess_log_distance": f"{float(getattr(item, 'scale_excess_log_distance', 0.0)):.6f}",
            "scale_nonlinear_gate": f"{float(getattr(item, 'scale_nonlinear_gate', 1.0)):.9f}",
            "scale_anchor_age_frames": int(getattr(item, "scale_anchor_age_frames", 0)),
            "scale_quarantine_active": int(bool(getattr(item, "scale_quarantine_active", False))),
            "scale_quarantine_hits": int(getattr(item, "scale_quarantine_hits", 0)),
            "scale_quarantine_required_hits": int(getattr(item, "scale_quarantine_required_hits", 0)),
            "scale_quarantine_confirmed": int(bool(getattr(item, "scale_quarantine_confirmed", False))),
            "detector_effective_min_area_px": int(getattr(item, "detector_effective_min_area_px", 0)),
            "detector_effective_max_area_px": int(getattr(item, "detector_effective_max_area_px", 0)),
            "detector_threshold": f"{float(getattr(item, 'detector_threshold', 0.0)):.6f}",
            "detector_median_response": f"{float(getattr(item, 'detector_median_response', 0.0)):.6f}",
            "detector_robust_sigma": f"{float(getattr(item, 'detector_robust_sigma', 0.0)):.6f}",
            "detector_raw_components": int(getattr(item, "detector_raw_components", 0)),
            "detector_valid_components": int(getattr(item, "detector_valid_components", 0)),
            "adaptation_updated": int(bool(getattr(item, "adaptation_updated", False))),
            "adaptation_reason": getattr(item, "adaptation_reason", "BLOCKED_REACQ" if event.mode == "REACQ" else ""),
            "adaptation_update_count": int(getattr(item, "adaptation_update_count", 0)),
            "tracking_phase": getattr(item, "tracking_phase", "TRACK" if event.mode == "TRACK" else event.mode),
            "adaptation_layer": getattr(item, "adaptation_layer", "NONE"),
            "adaptation_drift_score": f"{float(getattr(item, 'adaptation_drift_score', 0.0)):.9f}",
            "adaptation_drift_reason": getattr(item, "adaptation_drift_reason", ""),
            "adaptation_recent_count": int(getattr(item, "adaptation_recent_count", 0)),
            "adaptation_fast_remaining": int(getattr(item, "adaptation_fast_remaining", 0)),
            "motion_break_active": int(bool(getattr(item, "motion_break_active", False))),
            "motion_break_triggered": int(bool(getattr(item, "motion_break_triggered", False))),
            "motion_break_reason": getattr(item, "motion_break_reason", ""),
            "motion_broken_mode": int(bool(getattr(item, "motion_broken_mode", False))),
            "auto_promoted_motion_break": int(bool(getattr(item, "auto_promoted_motion_break", False))),
            "promotion_evidence_hits": int(getattr(item, "promotion_evidence_hits", 0)),
            "promotion_reason": getattr(item, "promotion_reason", ""),
            "tracklet_id": int(getattr(item, "tracklet_id", 0)),
            "active_tracklets": int(getattr(item, "active_tracklets", 0)),
            "selected_tracklet_x": "" if getattr(item, "selected_tracklet_x", None) is None else f"{float(item.selected_tracklet_x):.6f}",
            "selected_tracklet_y": "" if getattr(item, "selected_tracklet_y", None) is None else f"{float(item.selected_tracklet_y):.6f}",
            "selected_tracklet_area_px": int(getattr(item, "selected_tracklet_area_px", 0)),
            "selected_tracklet_identity": f"{float(getattr(item, 'selected_tracklet_identity', 0.0)):.9f}",
            "selected_tracklet_score": f"{float(getattr(item, 'selected_tracklet_score', 0.0)):.9f}",
            "selected_tracklet_gap": f"{float(getattr(item, 'selected_tracklet_gap', 0.0)):.9f}",
            "selected_tracklet_thermal_profile": f"{float(getattr(item, 'selected_tracklet_thermal_profile', 0.0)):.9f}",
            "reacq_mode": getattr(item, "reacq_mode", ""),
            "candidate_channel": getattr(item, "candidate_channel", getattr(getattr(item, "selected", None), "detection_channel", "NONE")),
            "reliable_anchor_x": "" if getattr(item, "reliable_anchor_xy", None) is None else f"{float(item.reliable_anchor_xy[0]):.6f}",
            "reliable_anchor_y": "" if getattr(item, "reliable_anchor_xy", None) is None else f"{float(item.reliable_anchor_xy[1]):.6f}",
            "anchor_distance_px": f"{float(getattr(item, 'anchor_distance_px', 0.0)):.6f}",
            "anchor_distance_norm": f"{float(getattr(item, 'anchor_distance_norm', 0.0)):.6f}",
            "anchor_age_frames": int(getattr(item, "anchor_age_frames", 0)),
            "anchor_allowed_distance_px": f"{float(getattr(item, 'anchor_allowed_distance_px', 0.0)):.6f}",
            "frame_reliability_score": f"{float(getattr(item, 'frame_reliability_score', 1.0)):.9f}",
            "frame_reliability_residual": f"{float(getattr(item, 'frame_reliability_residual', 0.0)):.9f}",
            "frame_reliability_phase_response": f"{float(getattr(item, 'frame_reliability_phase_response', 1.0)):.9f}",
            "frame_degraded_active": int(bool(getattr(item, "frame_degraded_active", False))),
            "frame_degraded_frames": int(getattr(item, "frame_degraded_frames", 0)),
            "jump_quarantine_active": int(bool(getattr(item, "jump_quarantine_active", False))),
            "jump_quarantine_hits": int(getattr(item, "jump_quarantine_hits", 0)),
            "jump_quarantine_required_hits": int(getattr(item, "jump_quarantine_required_hits", 0)),
            "jump_quarantine_confirmed": int(bool(getattr(item, "jump_quarantine_confirmed", False))),
            "jump_global_motion_veto": int(bool(getattr(item, "jump_global_motion_veto", False))),
            "anchor_continuity_override": int(bool(getattr(item, "anchor_continuity_override", False))),
            "far_jump_candidate": int(bool(getattr(item, "far_jump_candidate", False))),
            "micro_candidate_count": int(getattr(item, "micro_candidate_count", 0)),
            "global_shift_dx_px": f"{float(getattr(item, 'global_shift_dx_px', 0.0)):.6f}",
            "global_shift_dy_px": f"{float(getattr(item, 'global_shift_dy_px', 0.0)):.6f}",
            "global_shift_response": f"{float(getattr(item, 'global_shift_response', 0.0)):.9f}",
            "global_shift_method": getattr(item, "global_shift_method", "OFF"),
            "global_shift_time_ms": f"{float(getattr(item, 'global_shift_time_ms', 0.0)):.6f}",
            "detector_time_ms": f"{float(getattr(item, 'detector_time_ms', 0.0)):.6f}",
            "identity_time_ms": f"{float(getattr(item, 'identity_time_ms', 0.0)):.6f}",
            "motion_time_ms": f"{float(getattr(item, 'motion_time_ms', 0.0)):.6f}",
            "performance_budget_ms": f"{float(getattr(item, 'performance_budget_ms', 0.0)):.6f}",
            "over_budget": int(bool(getattr(item, "over_budget", False))),
            "processing_time_ms": f"{float(getattr(item, 'processing_time_ms', 0.0)):.6f}",
        }

    @staticmethod
    def _annotated(event: Any) -> np.ndarray:
        frame = event.frame_bgr.copy()
        item = event.result
        pred = getattr(item, "predicted_xy", None)
        radius = int(getattr(item, "search_radius_px", 0))
        if pred is not None:
            px, py = int(round(pred[0])), int(round(pred[1]))
            cv2.drawMarker(frame, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
            if radius > 0:
                cv2.circle(frame, (px, py), radius, (255, 0, 255), 1)

        selected, best_center, best_bbox, *_ = _candidate_info(item)
        if best_bbox is not None:
            x, y, w, h = map(int, best_bbox)
            color = (0, 255, 0) if selected is not None else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if best_center is not None:
            cx, cy = int(round(best_center[0])), int(round(best_center[1]))
            color = (0, 255, 0) if selected is not None else (0, 165, 255)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)

        label = (
            f"F={event.frame_index + 1} {event.mode}/{getattr(item, 'status', '')} "
            f"S={getattr(item, 'selected_score', 0.0):.3f} "
            f"I={getattr(item, 'identity_similarity', 0.0):.3f} "
            f"G={getattr(item, 'confidence_gap', 0.0):.3f}"
        )
        cv2.rectangle(frame, (5, 5), (min(frame.shape[1]-5, 680), 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def _write_png_unicode(path: Path, image: np.ndarray) -> None:
        """Write PNG without passing a Unicode filesystem path to OpenCV.

        ``cv2.imwrite`` can silently fail on some Windows/OpenCV builds when the
        path contains non-ASCII characters (for example a Cyrillic Windows user
        profile in pytest's temporary directory).  Encoding in memory and letting
        Python/Pathlib perform the filesystem write is Unicode-safe on Windows.
        """
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise IOError(f"OpenCV failed to encode PNG for {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())

    def _save_incident_frame(self, folder: Path, seq: int, frame_index: int, mode: str, status: str, frame: np.ndarray, event: Any) -> None:
        stem = f"{seq:03d}_frame_{frame_index + 1:06d}_{mode}_{status}"
        self._write_png_unicode(folder / f"{stem}_raw.png", frame)
        self._write_png_unicode(folder / f"{stem}_annotated.png", self._annotated(event))

    def _start_incident(self, event: Any) -> None:
        if not self.config.save_incident_frames or self.run_dir is None:
            return
        self._incident_count += 1
        folder = self.run_dir / "incidents" / f"lost_{self._incident_count:02d}_frame_{event.frame_index + 1:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        buffered = list(self._pre_buffer)
        for seq, (fi, mode, status, frame, ev) in enumerate(buffered):
            self._save_incident_frame(folder, seq, fi, mode, status, frame, ev)
        self._active_incident = {
            "folder": folder,
            "post_remaining": int(self.config.post_lost_frames),
            "next_seq": len(buffered),
        }
        self._write_text(f"INCIDENT LOST frame={event.frame_index + 1}; folder={folder}")

    def _continue_incident(self, event: Any) -> None:
        inc = self._active_incident
        if inc is None or inc["post_remaining"] <= 0:
            return
        self._save_incident_frame(
            inc["folder"], inc["next_seq"], event.frame_index, event.mode,
            getattr(event.result, "status", ""), event.frame_bgr, event,
        )
        inc["next_seq"] += 1
        inc["post_remaining"] -= 1
        if inc["post_remaining"] <= 0:
            self._active_incident = None

    def log_event(self, event: Any) -> None:
        if not self.enabled or self._closed:
            return
        row = self._row(event)
        self._csv_writer.writerow(row)
        if self.config.flush_every_frame:
            self._csv_fh.flush()
        self._event_count += 1
        self._last_logged_frame = int(event.frame_index)
        self._status_counts[f"{row['mode']}/{row['status']}"] += 1
        if row["diagnostic_reason"]:
            self._diagnostic_reason_counts[str(row["diagnostic_reason"])] += 1
        if row["adaptation_reason"]:
            self._adaptation_reason_counts[str(row["adaptation_reason"])] += 1
        if event.mode == "TRACK" and row["status"] == "LOST" and self._first_lost_frame is None:
            self._first_lost_frame = int(event.frame_index)

        interesting = (
            row["status"] in {"LOW_CONFIDENCE", "LOST", "SEARCHING", "TENTATIVE", "CONFIRMED"}
            or bool(row["adaptation_updated"])
            or row["diagnostic_reason"] != "ACCEPTED"
        )
        if interesting:
            self._write_text(
                f"F={row['frame']} {row['mode']}/{row['status']} reason={row['diagnostic_reason']} "
                f"Score={row['selected_score']} Identity={row['identity_similarity']} "
                f"App={row['appearance_similarity']} Edge={row['edge_similarity']} Proj={row['projection_similarity']} Color={row.get('color_similarity','0')} "
                f"Gap={row['confidence_gap']} ROI={row['search_radius_px']} cand={row['candidate_count']} "
                f"Area~{row['expected_area_px']}[{row['detector_effective_min_area_px']}..{row['detector_effective_max_area_px']}] "
                f"votes={row['confirmation_streak']}/{row['confirmation_window_size']} "
                f"phase={row['tracking_phase']} adapt={row['adaptation_reason']} layer={row['adaptation_layer']} "
                f"drift={row['adaptation_drift_score']} recent={row['adaptation_recent_count']} "
                f"fast_left={row['adaptation_fast_remaining']} upd={row['adaptation_update_count']} "
                f"motion_break={row['motion_break_active'] or row['motion_broken_mode']} "
                f"auto_promote={row.get('auto_promoted_motion_break', 0)} "
                f"reacq_mode={row.get('reacq_mode','')} channel={row.get('candidate_channel','')} "
                f"anchor_d={row.get('anchor_distance_px','')} Dn={row.get('anchor_distance_norm','')} age={row.get('anchor_age_frames',0)} allow={row.get('anchor_allowed_distance_px','')} "
                f"scaleA={row.get('reliable_area_anchor_px','')} Ds={row.get('scale_log_distance','')}/{row.get('scale_allowed_log_distance','')} "
                f"scaleGate={row.get('scale_nonlinear_gate','')} scaleQ={row.get('scale_quarantine_hits',0)}/{row.get('scale_quarantine_required_hits',0)} "
                f"Q={row.get('frame_reliability_score','')} degraded={row.get('frame_degraded_active',0)}:{row.get('frame_degraded_frames',0)} K2D={row.get('thermal_2d_ratio_similarity','')} "
                f"jumpQ={row.get('jump_quarantine_hits',0)}/{row.get('jump_quarantine_required_hits',0)} "
                f"veto={row.get('jump_global_motion_veto',0)} anchor_pref={row.get('anchor_continuity_override',0)} clutter={row.get('thermal_background_clutter','')} "
                f"hudP={row.get('hud_probability','')} hudHard={row.get('hud_hard_reject',0)} "
                f"occ={row.get('reticle_occlusion_active',0)}:{row.get('reticle_occlusion_frames',0)} exit={row.get('reticle_exit_hits',0)} "
                f"far={row.get('far_jump_candidate',0)} micro={row.get('micro_candidate_count',0)} "
                f"shift={row['global_shift_method']}({row['global_shift_dx_px']},{row['global_shift_dy_px']}) "
                f"t={row['processing_time_ms']}ms budget={row['performance_budget_ms']}ms over={row['over_budget']}"
            )

        # Preserve pre-LOST frames before starting the incident.  The LOST frame
        # itself is already in this buffer and will therefore be saved once.
        self._pre_buffer.append(
            (int(event.frame_index), str(event.mode), str(row["status"]), event.frame_bgr.copy(), event)
        )
        if event.mode == "TRACK" and row["status"] == "LOST":
            self._start_incident(event)
        elif self._active_incident is not None:
            self._continue_incident(event)

    def finish(self, result: Any) -> None:
        if not self.enabled or self._closed:
            return
        summary = {
            "status": getattr(result, "status", ""),
            "events_logged": self._event_count,
            "lost_events": int(getattr(result, "lost_events", 0)),
            "reacq_successes": int(getattr(result, "reacq_successes", 0)),
            "t_reacq_frames": list(getattr(result, "t_reacq_frames", ()) or ()),
            "mean_t_reacq_frames": float(getattr(result, "mean_t_reacq_frames", 0.0)),
            "max_t_reacq_frames": int(getattr(result, "max_t_reacq_frames", 0)),
            "adaptation_mode": getattr(result, "adaptation_mode", ""),
            "adaptation_updates": int(getattr(result, "adaptation_updates", 0)),
            "adaptation_slow_updates": int(getattr(result, "adaptation_slow_updates", 0)),
            "adaptation_fast_updates": int(getattr(result, "adaptation_fast_updates", 0)),
            "adaptation_recent_updates": int(getattr(result, "adaptation_recent_updates", 0)),
            "adaptation_recent_count": int(getattr(result, "adaptation_recent_count", 0)),
            "mean_track_confidence": float(getattr(result, "mean_track_confidence", 0.0)),
            "track_accepted_frames": int(getattr(result, "track_accepted_frames", 0)),
            "motion_break_events": int(getattr(result, "motion_break_events", 0)),
            "robust_reacq_count": int(getattr(result, "robust_reacq_count", 0)),
            "auto_promoted_reacq_count": int(getattr(result, "auto_promoted_reacq_count", 0)),
            "over_budget_frames": int(getattr(result, "over_budget_frames", 0)),
            "final_frame": getattr(result, "final_frame", None),
            "incident_count": self._incident_count,
            "first_lost_frame": self._first_lost_frame,
            "first_lost_frame_display": None if self._first_lost_frame is None else self._first_lost_frame + 1,
            "last_logged_frame": self._last_logged_frame,
            "last_logged_frame_display": None if self._last_logged_frame is None else self._last_logged_frame + 1,
            "status_counts": dict(self._status_counts),
            "diagnostic_reason_counts": dict(self._diagnostic_reason_counts),
            "adaptation_reason_counts": dict(self._adaptation_reason_counts),
        }
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        with (self.run_dir / "summary.txt").open("w", encoding="utf-8") as fh:
            for k, v in summary.items():
                fh.write(f"{k}: {v}\n")

        dominant_diag = self._diagnostic_reason_counts.most_common(3)
        dominant_adapt = self._adaptation_reason_counts.most_common(3)
        with (self.run_dir / "quick_diagnosis.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"status: {summary['status']}\n")
            fh.write(f"first_lost_frame: {summary['first_lost_frame_display']}\n")
            fh.write(f"adaptation_mode: {summary['adaptation_mode']}\n")
            fh.write(f"adaptation_updates: {summary['adaptation_updates']}\n")
            if summary['adaptation_mode'] == 'off':
                fh.write("adaptation_note: Fixed ReferenceBank selected; absence of adaptation updates is expected.\n")
            elif summary['adaptation_updates'] == 0 and summary.get('adaptation_recent_updates', 0) == 0:
                fh.write("adaptation_note: adaptation selected but no memory layer updated; inspect adaptation_reason_counts.\n")
            else:
                fh.write(
                    f"adaptation_note: Slow={summary.get('adaptation_slow_updates',0)}, "
                    f"Fast={summary.get('adaptation_fast_updates',0)}, "
                    f"Recent={summary.get('adaptation_recent_updates',0)}; inspect frames.csv around drift/LOST.\n"
                )
            fh.write(f"motion_break_events: {summary.get('motion_break_events',0)}\n")
            fh.write(f"robust_reacq_count: {summary.get('robust_reacq_count',0)}\n")
            fh.write(f"auto_promoted_reacq_count: {summary.get('auto_promoted_reacq_count',0)}\n")
            fh.write(f"over_budget_frames: {summary.get('over_budget_frames',0)}\n")
            fh.write(f"dominant_diagnostic_reasons: {dominant_diag}\n")
            fh.write(f"dominant_adaptation_reasons: {dominant_adapt}\n")
            fh.write("For detailed analysis upload diagnostic_bundle.zip from this directory.\n")
        self._write_text(
            f"DIAGNOSTIC FINISH status={summary['status']}; LOST={summary['lost_events']}; "
            f"REACQ={summary['reacq_successes']}; adapt={summary['adaptation_mode']}; "
            f"updates={summary['adaptation_updates']}"
        )
        self.close()
        if self.config.make_zip_bundle:
            self._make_zip_bundle()

    def _make_zip_bundle(self) -> Path | None:
        if self.run_dir is None:
            return None
        bundle = self.run_dir / "diagnostic_bundle.zip"
        files = [p for p in self.run_dir.rglob("*") if p.is_file() and p != bundle]
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, arcname=str(path.relative_to(self.run_dir)))
        return bundle

    def fail(self, message: str) -> None:
        if not self.enabled or self._closed:
            return
        with (self.run_dir / "error.txt").open("w", encoding="utf-8") as fh:
            fh.write(str(message) + "\n")
        self._write_text(f"DIAGNOSTIC ERROR: {message}")
        self.close()
        if self.config.make_zip_bundle:
            self._make_zip_bundle()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._csv_fh is not None:
            self._csv_fh.flush()
            self._csv_fh.close()
        if self._log_fh is not None:
            self._log_fh.flush()
            self._log_fh.close()
