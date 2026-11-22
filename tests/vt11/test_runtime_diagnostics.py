from __future__ import annotations

from navmin.vision.processor import VisionRuntimeSettings
from navmin.vision.processors.vt11.processor import Vt11VisionProcessor
from navmin.vision.processors.vt11.tracking.reacquisition import ReacqFrameResult
from navmin.vision.processors.vt11.tracking.tracker import TrackFrameResult

from .test_candidate_selection_runtime import _packet


def test_track_diagnostics_publish_rev5_provenance_and_block_reason() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    processor.apply_runtime_settings(VisionRuntimeSettings(True))
    frame = _packet(9)
    result = TrackFrameResult(
        frame_index=9,
        predicted_xy=(80.0, 54.0),
        selected=None,
        selected_score=0.51,
        confidence_gap=0.09,
        candidate_count=1,
        identity_similarity=0.55,
        motion_score=0.7,
        size_score=0.8,
        status="LOW_CONFIDENCE",
        miss_count=0,
        search_radius_px=180,
        diagnostic_reason="LINE_RECOVERY_PENDING_OR_INCONSISTENT",
        candidate_channel="LINE_RECOVERY",
        candidate_separated_from_line=True,
        blur_variant="5-diagonal",
        maneuver_hypothesis_xy=(101.0, 61.0),
        maneuver_hits=1,
        line_recovery_hits=1,
        adaptation_reason="BLOCKED_LINE_OVERLAP",
    )

    diagnostics = processor._tracking_diagnostics(frame, result, phase="TRACK")

    assert diagnostics.detection_channel == "LINE_RECOVERY"
    assert diagnostics.separated_from_line
    assert diagnostics.blur_robust_enabled
    assert diagnostics.blur_variant == "5-diagonal"
    assert diagnostics.maneuver_state == "PENDING"
    assert diagnostics.maneuver_hits == 1
    assert diagnostics.line_recovery_hits == 1
    assert diagnostics.diagnostic_reason == "LINE_RECOVERY_PENDING_OR_INCONSISTENT"
    assert diagnostics.adaptation_block_reason == "BLOCKED_LINE_OVERLAP"


def test_reacq_diagnostics_keep_line_recovery_provenance_without_maneuver_state() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    processor.apply_runtime_settings(VisionRuntimeSettings(True))
    frame = _packet(12)
    result = ReacqFrameResult(
        frame_index=12,
        predicted_xy=(84.0, 55.0),
        search_radius_px=180,
        selected=None,
        selected_score=0.53,
        identity_similarity=0.56,
        motion_score=0.6,
        size_score=0.75,
        confidence_gap=0.1,
        confidence=0.5,
        candidate_count=1,
        confirmation_streak=2,
        confirmation_window_size=3,
        status="TENTATIVE",
        diagnostic_reason="TENTATIVE_OK_2_OF_3",
        prediction_error_px=12.0,
        candidate_channel="LINE_RECOVERY",
        candidate_separated_from_line=True,
        line_recovery_hits=2,
        blur_variant="ORIGINAL",
        adaptation_reason="BLOCKED_REACQ",
    )

    diagnostics = processor._tracking_diagnostics(frame, result, phase="REACQ")

    assert diagnostics.detection_channel == "LINE_RECOVERY"
    assert diagnostics.separated_from_line
    assert diagnostics.blur_robust_enabled
    assert diagnostics.blur_variant == "ORIGINAL"
    assert diagnostics.maneuver_state == "NOT_APPLICABLE"
    assert diagnostics.maneuver_hits == 0
    assert diagnostics.line_recovery_hits == 2
    assert diagnostics.adaptation_block_reason == "BLOCKED_REACQ"
