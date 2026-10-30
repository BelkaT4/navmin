"""Pure adapter from accepted NavMin observations to recording schema v1."""

from __future__ import annotations

from navmin.contracts import TargetRef, VisionResult
from navmin.recording import (
    RecordingBestCandidate,
    RecordingControlPrediction,
    RecordingControlState,
    RecordingFramePacket,
    RecordingOverlaySnapshot,
    RecordingPoint,
    RecordingSearchRegion,
    RecordingSelectedTarget,
    RecordingSelection,
    RecordingTrackingState,
    SearchRegionKind,
)
from navmin.vision.processor import VisionTrackingDiagnostics


def assemble_recording_frame_packet(
    *,
    result: VisionResult,
    session_started_monotonic_ns: int,
    diagnostics: VisionTrackingDiagnostics | None,
    selected_target: TargetRef | None,
    selection: RecordingSelection | None,
    control_prediction: RecordingControlPrediction | None,
    control_state: RecordingControlState | None,
) -> RecordingFramePacket:
    """Build one source-coordinate recording packet for one exact frame.

    Stale/mismatched Vision diagnostics are deliberately converted to ``None``
    instead of being carried forward. Selection and control geometry are
    supplied by the integration owner after exact-frame checks on those paths.
    """
    frame = result.frame
    if session_started_monotonic_ns < 0:
        raise ValueError("session_started_monotonic_ns must be non-negative")
    if diagnostics is not None and not _diagnostics_match_result(diagnostics, result):
        diagnostics = None

    selected = _selected_target(result, selected_target)
    best_candidate = (
        None
        if diagnostics is None or diagnostics.best_candidate_bbox is None
        else RecordingBestCandidate(diagnostics.best_candidate_bbox)
    )
    search_region = _search_region(diagnostics)
    tracking = (
        None
        if diagnostics is None
        else RecordingTrackingState(
            phase=diagnostics.phase,
            status=diagnostics.status,
            target_ref=_target_ref_text(selected_target),
            diagnostic_reason=diagnostics.diagnostic_reason,
            candidate_count=diagnostics.candidate_count,
        )
    )
    return RecordingFramePacket(
        frame=frame,
        relative_time_ns=max(
            0, frame.receive_timestamp_ns - session_started_monotonic_ns
        ),
        capture_timestamp_ns=None,
        tracking=tracking,
        selection=selection,
        overlay=RecordingOverlaySnapshot(
            selected_target=selected,
            best_candidate=best_candidate,
            search_region=search_region,
            control_prediction=control_prediction,
        ),
        control=control_state,
    )


def _diagnostics_match_result(
    diagnostics: VisionTrackingDiagnostics,
    result: VisionResult,
) -> bool:
    return bool(
        diagnostics.camera is result.frame.camera
        and diagnostics.generation == result.frame.generation
        and diagnostics.frame_id == result.frame.frame_id
    )


def _selected_target(
    result: VisionResult,
    target: TargetRef | None,
) -> RecordingSelectedTarget | None:
    if (
        target is None
        or target.camera is not result.frame.camera
        or target.generation != result.frame.generation
    ):
        return None
    tracked = next(
        (item for item in result.tracked_objects if item.track_id == target.track_id),
        None,
    )
    if tracked is None:
        return None
    bbox = tracked.bbox
    return RecordingSelectedTarget(
        bbox=bbox,
        center=RecordingPoint(
            bbox.x + bbox.width / 2.0,
            bbox.y + bbox.height / 2.0,
        ),
    )


def _search_region(
    diagnostics: VisionTrackingDiagnostics | None,
) -> RecordingSearchRegion | None:
    if diagnostics is None:
        return None
    if diagnostics.full_frame_search:
        return RecordingSearchRegion(SearchRegionKind.FULL_FRAME)
    return RecordingSearchRegion(
        SearchRegionKind.CIRCLE,
        center=RecordingPoint(*diagnostics.predicted_xy),
        radius_px=float(diagnostics.search_radius_px),
    )


def _target_ref_text(target: TargetRef | None) -> str | None:
    if target is None:
        return None
    return f"{target.camera.value}:{target.generation}:{target.track_id}"


__all__ = ["assemble_recording_frame_packet"]
