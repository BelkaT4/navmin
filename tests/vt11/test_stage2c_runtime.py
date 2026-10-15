from dataclasses import replace
import inspect

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.acquire import Vt11AcquisitionAttempt
from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectorConfig,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.initial_train import (
    InitialTrainConfig,
    run_initial_train,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.config.loader import load_profile_config
from navmin.vision.processors.vt11.runtime_config import (
    Vt11ResolvedConfig,
    Vt11RuntimeSettings,
)
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig
from navmin.vision.processors.vt11.tracking.session import (
    TrackingSessionConfig,
    Vt11TargetSession,
    run_tracking_session,
)
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig


def _detector_config() -> CandidateDetectorConfig:
    return CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1200,
        spatial_sigma_px=35.0,
    )


def _frames(count: int, *, invisible: set[int] | None = None) -> list[np.ndarray]:
    invisible = invisible or set()
    frames = []
    for frame_index in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if frame_index not in invisible:
            x = 38 + 3 * frame_index
            cv2.rectangle(frame, (x, 53), (x + 9, 61), (225, 225, 225), -1)
            cv2.rectangle(frame, (x + 7, 55), (x + 12, 58), (150, 150, 150), -1)
        frames.append(frame)
    return frames


def _stationary_frames(
    count: int, *, invisible: set[int] | None = None
) -> list[np.ndarray]:
    invisible = invisible or set()
    frames = []
    for frame_index in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if frame_index not in invisible:
            cv2.rectangle(frame, (43, 53), (52, 61), (225, 225, 225), -1)
            cv2.rectangle(frame, (50, 55), (55, 58), (150, 150, 150), -1)
        frames.append(frame)
    return frames


def _train(frames: list[np.ndarray]):
    detector = _detector_config()
    selection = make_operator_selection(0, (43, 57), 65, 180, 120)
    candidate = detect_local_candidates(
        frames[0], selection, top_k=5, config=detector
    ).candidates[0]

    def reader(index: int) -> tuple[int, np.ndarray]:
        return index, frames[index].copy()

    train = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=0,
        initial_candidate=candidate,
        roi_radius_px=65,
        detector_config=detector,
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert train.decision == "CONFIRMED"
    selected = [item for item in train.frames if item.selected is not None]
    return detector, train, selected, reader


def test_resolved_config_keeps_original_algorithm_budgets() -> None:
    day = Vt11ResolvedConfig.from_effective_config(
        "day", load_profile_config("day")
    )
    thermal = Vt11ResolvedConfig.from_effective_config(
        "thermal", load_profile_config("thermal")
    )
    assert day.initial_train.initial_frames == 7
    assert not day.initial_train.temporal_click_acquire_enabled
    assert thermal.initial_train.temporal_click_acquire_enabled
    assert thermal.initial_train.temporal_click_bootstrap_frames == 5
    assert day.reacquisition.max_frames == thermal.reacquisition.max_frames == 90
    assert day.finite_session.max_reacq_cycles == 5


def test_acquisition_attempt_is_bounded_and_creates_session_only_on_confirmed() -> None:
    frames = _frames(20)
    resolved = Vt11ResolvedConfig.from_effective_config(
        "day", load_profile_config("day")
    )
    attempt = Vt11AcquisitionAttempt(
        frame_index=0,
        frame_bgr=frames[0],
        click_xy=(43, 57),
        frame_width=180,
        frame_height=120,
        resolved_config=resolved,
    )
    assert attempt.required_window_frames == resolved.initial_train.initial_frames
    for frame_index in range(1, attempt.required_window_frames):
        assert attempt.collection_state_sizes()["raw_frames"] <= attempt.required_window_frames
        attempt.step(frame_index, frames[frame_index])
    assert attempt.result is not None
    assert attempt.result.decision == "CONFIRMED"
    assert attempt.result.target_session is not None
    assert attempt.result.target_session.next_frame_index == 7
    assert attempt.result.retrospective_replay_frames == 0
    assert attempt.collection_state_sizes()["raw_frames"] == 0


def test_acquisition_attempt_does_not_extend_ambiguous_train() -> None:
    frames = _frames(7, invisible=set(range(1, 7)))
    resolved = Vt11ResolvedConfig.from_effective_config(
        "day", load_profile_config("day")
    )
    attempt = Vt11AcquisitionAttempt(
        frame_index=0,
        frame_bgr=frames[0],
        click_xy=(43, 57),
        frame_width=180,
        frame_height=120,
        resolved_config=resolved,
    )
    for frame_index in range(1, attempt.required_window_frames):
        attempt.step(frame_index, frames[frame_index])
    assert attempt.result is not None
    assert attempt.result.decision == "AMBIGUOUS"
    assert attempt.result.target_session is None
    assert len(attempt.result.train_result.frames) == 7

    observe_config = replace(
        resolved,
        initial_train=replace(
            resolved.initial_train,
            confirm_min_valid_fraction=1.0,
            confirm_min_median_score=1.0,
            confirm_min_median_gap=1.0,
            observe_min_valid_fraction=0.0,
            observe_min_median_score=0.0,
        ),
    )
    visible = _frames(7, invisible={3})
    observe = Vt11AcquisitionAttempt(
        frame_index=0,
        frame_bgr=visible[0],
        click_xy=(43, 57),
        frame_width=180,
        frame_height=120,
        resolved_config=observe_config,
    )
    for frame_index in range(1, observe.required_window_frames):
        observe.step(frame_index, visible[frame_index])
    assert observe.result is not None
    assert observe.result.decision == "OBSERVE_MORE"
    assert observe.result.target_session is None
    assert len(observe.result.train_result.frames) == 7


def test_retrospective_catch_up_is_hidden_and_enforces_integration_limit() -> None:
    frames = _frames(20)
    resolved = Vt11ResolvedConfig.from_effective_config(
        "day", load_profile_config("day")
    )
    temporal_train = replace(
        resolved.initial_train,
        temporal_click_acquire_enabled=True,
        temporal_click_bootstrap_frames=5,
    )
    resolved = replace(resolved, initial_train=temporal_train)
    attempt = Vt11AcquisitionAttempt(
        frame_index=0,
        frame_bgr=frames[0],
        click_xy=(43, 57),
        frame_width=180,
        frame_height=120,
        resolved_config=resolved,
    )
    for frame_index in range(1, attempt.required_window_frames):
        attempt.step(frame_index, frames[frame_index])
    assert attempt.result is not None
    assert attempt.result.decision == "CONFIRMED"
    assert attempt.result.retrospective_replay_frames == 4
    assert attempt.result.target_session is not None
    assert attempt.result.target_session.next_frame_index == 11

    rejected = Vt11AcquisitionAttempt(
        frame_index=0,
        frame_bgr=frames[0],
        click_xy=(43, 57),
        frame_width=180,
        frame_height=120,
        resolved_config=resolved,
        runtime_settings=Vt11RuntimeSettings(max_retrospective_replay_frames=3),
    )
    for frame_index in range(1, rejected.required_window_frames):
        rejected.step(frame_index, frames[frame_index])
    assert rejected.result is not None
    assert rejected.result.target_session is None
    assert rejected.result.continuation_error is not None
    assert rejected.result.continuation_error.startswith(
        "RETROSPECTIVE_REPLAY_LIMIT_EXCEEDED"
    )


def test_target_session_uses_explicit_read_only_transition_api() -> None:
    frames = _frames(12)
    detector, train, selected, _reader = _train(frames)
    session = Vt11TargetSession(
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[item.selected.center for item in selected],
        initial_history_frames=[item.frame_index for item in selected],
        detector_config=detector,
        tracking_config=TrackingConfig(min_confidence_gap=0.0),
        reacquisition_config=ReacquisitionConfig(),
    )
    session.step(7, frames[7])
    snapshot = session.tracking_continuity_snapshot
    assert snapshot is not None
    assert snapshot.motion_history
    assert snapshot.previous_frame is not None
    assert not snapshot.previous_frame.flags.writeable
    source = inspect.getsource(Vt11TargetSession)
    assert "gi_frame" not in source
    assert "f_locals" not in source
    session.close()


def test_live_session_has_no_five_cycle_limit_but_batch_keeps_it() -> None:
    frame_count = 75
    invisible: set[int] = set()
    for lost_start in range(9, frame_count, 7):
        invisible.update((lost_start, lost_start + 1))
    frames = _stationary_frames(frame_count, invisible=invisible)
    detector, train, selected, reader = _train(frames)
    tracking = TrackingConfig(
        search_radius_px=65,
        max_search_radius_px=130,
        min_confidence_gap=0.0,
        lost_after_misses=2,
        track_warmup_frames=0,
    )
    reacquisition = ReacquisitionConfig(
        initial_radius_px=90,
        max_radius_px=170,
        radius_growth_px_per_frame=15,
        confirm_frames=3,
        confirm_window_frames=5,
        max_frames=8,
        min_confidence_gap=0.0,
        max_confirmation_error_px=80,
    )
    batch = run_tracking_session(
        frame_reader=reader,
        frame_count=frame_count,
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[item.selected.center for item in selected],
        initial_history_frames=[item.frame_index for item in selected],
        detector_config=detector,
        tracking_config=tracking,
        reacquisition_config=reacquisition,
        session_config=TrackingSessionConfig(max_reacq_cycles=5),
    )
    assert batch.status == "REACQ_LIMIT"
    assert batch.reacq_successes == 5

    live = Vt11TargetSession(
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[item.selected.center for item in selected],
        initial_history_frames=[item.frame_index for item in selected],
        detector_config=detector,
        tracking_config=tracking,
        reacquisition_config=reacquisition,
    )
    while live.next_frame_index < frame_count and live.terminal_status is None:
        frame_index = live.next_frame_index
        live.step(frame_index, frames[frame_index])
        sizes = live.collection_state_sizes()
        assert sizes["motion_history"] <= 40
        assert sizes.get("adaptation_events", 0) <= 32
    assert live.terminal_status is None
    assert live.reacq_successes > 5
    assert live.reacq_attempts > 5
    live.close()
