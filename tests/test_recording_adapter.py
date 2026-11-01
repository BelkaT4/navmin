from __future__ import annotations

import numpy as np

from navmin.contracts import (
    BBox,
    CameraRole,
    FramePacket,
    TargetRef,
    TrackedObject,
    VisionResult,
)
from navmin.recording import (
    RecordingControlPrediction,
    RecordingControlState,
    RecordingPoint,
    RecordingSelection,
)
from navmin.recording_adapter import assemble_recording_frame_packet
from navmin.vision.processor import VisionTrackingDiagnostics


def _result(*, frame_id: int = 7) -> VisionResult:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image.flags.writeable = False
    return VisionResult(
        frame=FramePacket(
            CameraRole.STEREO_LEFT,
            3,
            frame_id,
            None,
            1_500_000_000,
            image,
        ),
        tracked_objects=(TrackedObject(9, BBox(10, 6, 8, 6), 0.0, 0.0, 4),),
        processing_time_ns=12,
    )


def _diagnostics(*, frame_id: int = 7, full_frame: bool = False) -> VisionTrackingDiagnostics:
    return VisionTrackingDiagnostics(
        camera=CameraRole.STEREO_LEFT,
        generation=3,
        frame_id=frame_id,
        phase="REACQ" if full_frame else "TRACK",
        status="SEARCHING" if full_frame else "TRACK",
        predicted_xy=(14.0, 9.0),
        search_radius_px=12,
        full_frame_search=full_frame,
        search_mode="GLOBAL_EMERGENCY_SCAN" if full_frame else "TRACK",
        candidate_count=2,
        best_candidate_center=(15.0, 10.0),
        best_candidate_bbox=BBox(11, 7, 7, 5),
        confidence=0.7,
        prediction_error_px=1.5,
        diagnostic_reason="TEST",
    )


def test_adapter_builds_source_coordinate_packet_for_exact_frame() -> None:
    result = _result()
    target = TargetRef(CameraRole.STEREO_LEFT, 3, 9)
    packet = assemble_recording_frame_packet(
        result=result,
        session_started_monotonic_ns=1_400_000_000,
        diagnostics=_diagnostics(),
        selected_target=target,
        selection=RecordingSelection(RecordingPoint(12.5, 8.0)),
        control_prediction=RecordingControlPrediction(
            RecordingPoint(14.0, 9.0), RecordingPoint(18.0, 9.5)
        ),
        control_state=RecordingControlState(True, True, False),
    )

    assert packet.relative_time_ns == 100_000_000
    assert packet.tracking is not None
    assert packet.tracking.phase == "TRACK"
    assert packet.tracking.target_ref == "stereo_left:3:9"
    assert packet.overlay.selected_target is not None
    assert packet.overlay.selected_target.bbox == BBox(10, 6, 8, 6)
    assert packet.overlay.selected_target.center == RecordingPoint(14.0, 9.0)
    assert packet.overlay.best_candidate is not None
    assert packet.overlay.best_candidate.bbox == BBox(11, 7, 7, 5)
    assert packet.overlay.search_region is not None
    assert packet.overlay.search_region.center == RecordingPoint(14.0, 9.0)
    assert packet.selection == RecordingSelection(RecordingPoint(12.5, 8.0))
    assert packet.control == RecordingControlState(True, True, False)


def test_adapter_rejects_stale_diagnostics_instead_of_reusing_them() -> None:
    packet = assemble_recording_frame_packet(
        result=_result(frame_id=8),
        session_started_monotonic_ns=1_400_000_000,
        diagnostics=_diagnostics(frame_id=7),
        selected_target=TargetRef(CameraRole.STEREO_LEFT, 3, 9),
        selection=None,
        control_prediction=None,
        control_state=None,
    )

    assert packet.tracking is None
    assert packet.overlay.best_candidate is None
    assert packet.overlay.search_region is None
    assert packet.overlay.selected_target is not None


def test_adapter_serializes_global_search_semantics() -> None:
    packet = assemble_recording_frame_packet(
        result=_result(),
        session_started_monotonic_ns=1_400_000_000,
        diagnostics=_diagnostics(full_frame=True),
        selected_target=None,
        selection=None,
        control_prediction=None,
        control_state=None,
    )

    assert packet.overlay.search_region is not None
    assert packet.overlay.search_region.kind.value == "full-frame"
    assert packet.overlay.search_region.center is None
    assert packet.overlay.search_region.radius_px is None


def test_adapter_clamps_negative_relative_time_to_zero() -> None:
    packet = assemble_recording_frame_packet(
        result=_result(),
        session_started_monotonic_ns=2_000_000_000,
        diagnostics=None,
        selected_target=None,
        selection=None,
        control_prediction=None,
        control_state=None,
    )

    assert packet.relative_time_ns == 0
