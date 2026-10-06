import inspect

import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectorConfig,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.initial_train import (
    InitialTrainConfig,
    run_initial_train,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import (
    ReacquisitionConfig,
    ReacquisitionRuntime,
    run_reacquisition,
)
from navmin.vision.processors.vt11.tracking.tracker import (
    TrackingConfig,
    TrackingRuntime,
    run_tracking,
)


def _detector_config() -> CandidateDetectorConfig:
    return CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1200,
        spatial_sigma_px=35.0,
    )


def _frames(
    count: int,
    *,
    invisible: set[int] | None = None,
    velocity: int = 3,
) -> list[np.ndarray]:
    invisible = invisible or set()
    frames: list[np.ndarray] = []
    for frame_index in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if frame_index not in invisible:
            x = 38 + velocity * frame_index
            cv2.rectangle(frame, (x, 53), (x + 9, 61), (225, 225, 225), -1)
            cv2.rectangle(frame, (x + 7, 55), (x + 12, 58), (150, 150, 150), -1)
        frames.append(frame)
    return frames


def _train(frames: list[np.ndarray]):
    selection = make_operator_selection(0, (43, 57), 65, 180, 120)
    detected = detect_local_candidates(
        frames[0],
        selection,
        top_k=5,
        config=_detector_config(),
    )
    assert detected.candidates

    def reader(index: int) -> tuple[int, np.ndarray]:
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=0,
        initial_candidate=detected.candidates[0],
        roi_radius_px=65,
        detector_config=_detector_config(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "CONFIRMED"
    accepted = [item for item in result.frames if item.selected is not None]
    history = [
        MotionPoint(item.frame_index, item.selected.center_x, item.selected.center_y)
        for item in accepted
        if item.selected is not None
    ]
    return result.reference_bank, history, reader


def _track_signature(item) -> tuple:
    bbox = None
    if item.selected is not None:
        bbox = (
            item.selected.bbox.x,
            item.selected.bbox.y,
            item.selected.bbox.width,
            item.selected.bbox.height,
        )
    return (
        item.frame_index,
        item.status,
        bbox,
        item.miss_count,
        item.candidate_count,
        item.diagnostic_reason,
        item.selected_score,
        item.confidence_gap,
        item.identity_similarity,
        item.motion_score,
        item.size_score,
    )


def _reacq_signature(item) -> tuple:
    bbox = None
    if item.selected is not None:
        bbox = (
            item.selected.bbox.x,
            item.selected.bbox.y,
            item.selected.bbox.width,
            item.selected.bbox.height,
        )
    return (
        item.frame_index,
        item.status,
        bbox,
        item.confirmation_streak,
        item.candidate_count,
        item.diagnostic_reason,
        item.selected_score,
        item.confidence_gap,
        item.identity_similarity,
        item.motion_score,
        item.size_score,
    )


def test_tracking_batch_wrapper_matches_incremental_steps() -> None:
    frames = _frames(18)
    bank, history, reader = _train(frames)
    config = TrackingConfig(
        search_radius_px=70,
        top_k=5,
        min_confidence_gap=0.0,
    )
    batch = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=bank,
        initial_history=[point.xy for point in history],
        initial_history_frames=[point.frame_index for point in history],
        detector_config=_detector_config(),
        config=config,
    )
    runtime = TrackingRuntime(
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=bank,
        initial_history=[point.xy for point in history],
        initial_history_frames=[point.frame_index for point in history],
        detector_config=_detector_config(),
        config=config,
    )
    incremental = []
    while not runtime.done:
        incremental.append(runtime.step(frames[runtime.next_frame_index].copy()))

    for actual, expected in zip(incremental, batch.frames, strict=True):
        actual_signature = _track_signature(actual)
        expected_signature = _track_signature(expected)
        assert actual_signature[:6] == expected_signature[:6]
        assert actual_signature[6:] == pytest.approx(expected_signature[6:], abs=1e-7)
    runtime.close()


def test_tracking_runtime_collections_remain_bounded_on_long_run() -> None:
    frames = _frames(180, velocity=0)
    bank, history, _reader = _train(frames)
    config = TrackingConfig(
        search_radius_px=70,
        top_k=5,
        min_confidence_gap=0.0,
        velocity_window_frames=5,
    )
    adapter = EMAReferenceAdapter(
        bank,
        EMAConfig(
            mode="ema",
            warmup_accepted_frames=0,
            update_interval_frames=1,
            min_confidence=0.0,
            min_identity_similarity=0.0,
            min_confidence_gap=0.0,
            max_prediction_error_px=100.0,
        ),
    )
    runtime = TrackingRuntime(
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=bank,
        initial_history=[point.xy for point in history] * 40,
        initial_history_frames=list(range(-273, 7)),
        detector_config=_detector_config(),
        config=config,
        adaptation_adapter=adapter,
    )
    for frame in frames[7:]:
        result = runtime.step(frame.copy())
        assert result.status == "TRACK"
        sizes = runtime.collection_state_sizes()
        assert sizes["timed_history"] <= max(config.velocity_window_frames + 3, 10)
        assert sizes["warmup_points"] <= max(config.velocity_window_frames + 2, 4)
        assert sizes["recent_prediction_errors"] <= max(2, config.motion_break_window_frames)
        assert sizes["relock_points"] <= config.motion_relock_frames
        assert sizes["jump_evidence"] <= max(
            6,
            config.jump_extreme_confirm_hits + config.jump_single_candidate_extra_hits + 2,
        )
        assert sizes["scale_evidence"] <= max(8, config.scale_quarantine_extreme_confirm_hits + 3)
        assert sizes["reticle_exit_evidence"] <= max(3, config.hud_exit_confirm_hits + 1)
        assert sizes["adaptation_events"] <= runtime.DEFAULT_ADAPTATION_EVENT_HISTORY
        assert not {
            "results",
            "trajectory",
            "accepted_scores",
            "confidences",
            "prediction_errors",
            "global_shift_times",
        } & runtime.persistent_state.keys()
    runtime.close()


def test_reacquisition_batch_wrapper_matches_incremental_steps() -> None:
    frames = _frames(24, invisible=set(range(9, 13)))
    bank, history, reader = _train(frames)
    config = ReacquisitionConfig(
        initial_radius_px=90,
        max_radius_px=170,
        radius_growth_px_per_frame=15,
        confirm_frames=3,
        max_frames=12,
        min_confidence_gap=0.0,
        max_confirmation_error_px=80,
    )
    batch = run_reacquisition(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=10,
        reference_bank=bank,
        motion_history=history,
        detector_config=_detector_config(),
        config=config,
    )
    runtime = ReacquisitionRuntime(
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=10,
        previous_frame=frames[10].copy(),
        quality_reference_frame=frames[history[-1].frame_index].copy(),
        reference_bank=bank,
        motion_history=history,
        detector_config=_detector_config(),
        config=config,
    )
    incremental = []
    while not runtime.done:
        incremental.append(runtime.step(frames[runtime.next_frame_index].copy()))

    assert incremental[0].frame_index == 11
    for actual, expected in zip(incremental, batch.frames, strict=True):
        actual_signature = _reacq_signature(actual)
        expected_signature = _reacq_signature(expected)
        assert actual_signature[:6] == expected_signature[:6]
        assert actual_signature[6:] == pytest.approx(expected_signature[6:], abs=1e-7)
    runtime.close()


def test_reacquisition_runtime_collections_remain_bounded_to_attempt_budget() -> None:
    frames = _frames(110, invisible=set(range(7, 110)), velocity=0)
    bank, history, _reader = _train(frames)
    long_history = [
        MotionPoint(frame_index, history[-1].x, history[-1].y)
        for frame_index in range(-493, 7)
    ]
    config = ReacquisitionConfig(max_frames=90)
    runtime = ReacquisitionRuntime(
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=7,
        previous_frame=frames[7].copy(),
        quality_reference_frame=frames[6].copy(),
        reference_bank=bank,
        motion_history=long_history,
        detector_config=_detector_config(),
        config=config,
    )
    while not runtime.done:
        runtime.step(frames[runtime.next_frame_index].copy())
        sizes = runtime.collection_state_sizes()
        assert sizes["base_history"] <= max(2, config.velocity_window_frames)
        assert sizes["promotion_window"] <= max(2, config.auto_promote_window_frames)
        assert sizes["confirmation_window"] <= config.confirm_window_frames
        assert sizes["confirmation_area_window"] <= config.confirm_window_frames
        assert sizes.get("tracklets", 0) <= config.tracklet_top_k
        assert sizes.get("tracklet_evidence_max", 0) <= config.tracklet_window_frames + 3
        assert not {"results", "confidences", "global_times"} & runtime.persistent_state.keys()

    assert runtime.attempted_frames == config.max_frames
    runtime.close()


def test_compatibility_wrappers_contain_no_algorithm_copy() -> None:
    tracking_wrapper = inspect.getsource(run_tracking)
    reacquisition_wrapper = inspect.getsource(run_reacquisition)
    assert "TrackingRuntime(" in tracking_wrapper
    assert "runtime.step(frame)" in tracking_wrapper
    assert "_rank_candidates(" not in tracking_wrapper
    assert "ReacquisitionRuntime(" in reacquisition_wrapper
    assert "runtime.step(frame)" in reacquisition_wrapper
    assert "_rank_candidates(" not in reacquisition_wrapper
