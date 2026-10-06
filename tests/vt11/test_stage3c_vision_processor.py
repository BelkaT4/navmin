from __future__ import annotations

import numpy as np
import pytest

from navmin.contracts import (
    BBox,
    CameraRole,
    ClearTargetSelection,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetSelectionRequest,
)
from navmin.vision.processors.vt11.processor import (
    Vt11ProcessorState,
    Vt11VisionProcessor,
)

from .test_stage3b_vision_processor import (
    _AcquisitionPlan,
    _FakeSession,
    _activate_target,
    _candidate,
    _install_acquisition_plans,
    _packet,
    _retain,
    _select,
    _track_step,
)


class _FixedDetector:
    def __init__(self, *boxes: BBox) -> None:
        self.boxes = tuple(boxes)
        self.detect_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def detect(self, image: np.ndarray) -> tuple[BBox, ...]:
        del image
        self.detect_calls += 1
        return self.boxes


def _clear(generation: int = 1) -> ClearTargetSelection:
    return ClearTargetSelection(CameraRole.OVERVIEW, generation)


def _select_generation(
    frame_id: int,
    *,
    generation: int,
    selection_id: int,
) -> TargetSelectionRequest:
    return TargetSelectionRequest(
        selection_id=selection_id,
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=frame_id,
        x_px=43.0,
        y_px=57.0,
    )


def test_clear_in_discovery_preserves_warmed_legacy14_state() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    detector = _FixedDetector(BBox(20, 18, 16, 12))
    processor._legacy._detector = detector

    processor.process(_packet(0))
    before = processor.process(_packet(1)).result.tracked_objects
    reset_calls = detector.reset_calls

    assert processor.apply_target_control(_clear()) == ()
    after = processor.process(_packet(2)).result.tracked_objects

    assert [item.track_id for item in before] == [1]
    assert [item.track_id for item in after] == [1]
    assert detector.reset_calls == reset_calls
    assert processor.state is Vt11ProcessorState.DISCOVERY


def test_clear_destroys_acquiring_episode_without_resetting_generation_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_type = _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(decision="CONFIRMED", required_frames=20),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    detector = _FixedDetector()
    processor._legacy._detector = detector
    _retain(processor, range(5))
    assert processor.apply_target_control(_select(0)) == ()
    assert processor.state is Vt11ProcessorState.ACQUIRING
    before_clear_frames = tuple(attempt_type.instances[0].seen_frames)
    reset_calls = detector.reset_calls

    assert processor.apply_target_control(_clear()) == ()

    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._acquisition is None
    assert processor._target_session is None
    assert processor._current_target is None
    assert processor._camera is CameraRole.OVERVIEW
    assert processor._generation == 1
    assert processor._public_track_ids.next_id == 1
    assert len(processor._frame_history) == 5
    assert len(processor._timing_history) == 5
    assert detector.reset_calls == reset_calls + 1

    output = processor.process(_packet(5), processing_enabled=False)
    assert output.result.tracked_objects == ()
    assert output.target_events == ()
    assert tuple(attempt_type.instances[0].seen_frames) == before_clear_frames


def test_clear_closes_target_but_preserves_history_and_monotonic_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_session = _FakeSession(next_frame_index=2)
    new_session = _FakeSession(next_frame_index=5)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=2,
            session=old_session,
        ),
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=1,
            session=new_session,
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(5))
    (first,) = processor.apply_target_control(_select(0, selection_id=1))

    assert processor.apply_target_control(_clear()) == ()

    assert isinstance(first, TargetAcquired)
    assert first.target.track_id == 1
    assert old_session.closed
    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._public_track_ids.next_id == 2
    assert len(processor._frame_history) == 5
    assert len(processor._timing_history) == 5
    assert processor.apply_target_control(_clear()) == ()

    (second,) = processor.apply_target_control(_select(4, selection_id=2))
    assert isinstance(second, TargetAcquired)
    assert second.target.track_id == 2


def test_hard_reset_destroys_acquisition_and_all_generation_local_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_type = _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(decision="CONFIRMED", required_frames=20),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(4))
    assert processor.apply_target_control(_select(0)) == ()
    seen_before_reset = tuple(attempt_type.instances[0].seen_frames)

    assert processor.reset() is None

    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._camera is None
    assert processor._generation is None
    assert processor._acquisition is None
    assert processor._target_session is None
    assert processor._current_target is None
    assert processor._current_public_target_id is None
    assert processor._initial_acquisition_decision_frame_id is None
    assert processor._public_track_ids.next_id == 1
    assert len(processor._frame_history) == 0
    assert len(processor._timing_history) == 0

    output = processor.process(
        _packet(0, generation=2),
        processing_enabled=False,
    )
    assert output.result.tracked_objects == ()
    assert output.target_events == ()
    assert tuple(attempt_type.instances[0].seen_frames) == seen_before_reset


def test_generation_change_closes_old_target_and_starts_clean_id_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_session = _FakeSession(next_frame_index=1)
    new_session = _FakeSession(next_frame_index=1)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=1,
            session=old_session,
        ),
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=1,
            session=new_session,
        ),
    )
    processor = Vt11VisionProcessor(
        profile_name="day",
        history_clock_ns=lambda: 0,
    )
    _retain(processor, (0,))
    (old_acquired,) = processor.apply_target_control(_select(0, selection_id=1))
    assert isinstance(old_acquired, TargetAcquired)
    assert old_acquired.target.track_id == 1

    new_frame = _packet(0, generation=2, timestamp_ns=2_000_000_000)
    output = processor.process(new_frame, processing_enabled=False)

    assert output.result.frame is new_frame
    assert output.result.tracked_objects == ()
    assert output.target_events == ()
    assert old_session.closed
    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._camera is CameraRole.OVERVIEW
    assert processor._generation == 2
    assert processor._public_track_ids.next_id == 1
    assert len(processor._frame_history) == 1
    assert len(processor._timing_history) == 1

    (stale,) = processor.apply_target_control(_select(0, selection_id=2))
    assert isinstance(stale, TargetAcquisitionFailed)
    assert stale.reason is TargetAcquisitionFailureReason.STALE_SELECTION
    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._public_track_ids.next_id == 1

    (current,) = processor.apply_target_control(
        _select_generation(0, generation=2, selection_id=3)
    )
    assert isinstance(current, TargetAcquired)
    assert current.target.generation == 2
    assert current.target.track_id == 1


def test_nonuniform_processed_timing_maps_velocity_and_preserves_exact_frame() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    timestamps = (0, 100_000_000, 300_000_000, 400_000_000)
    for frame_id, timestamp_ns in enumerate(timestamps):
        processor.process(
            _packet(frame_id, timestamp_ns=timestamp_ns),
            processing_enabled=False,
        )
    candidate = _candidate()
    session = _FakeSession(
        4,
        (
            _track_step(
                4,
                status="TRACK",
                candidate=candidate,
                velocity=(1.5, -0.5),
            ),
        ),
    )
    _activate_target(processor, session, decision_frame_id=3, track_id=7)
    frame = _packet(4, timestamp_ns=500_000_000)

    output = processor.process(frame)

    assert output.result.frame is frame
    (tracked,) = output.result.tracked_objects
    assert tracked.track_id == 7
    assert tracked.velocity_x_px_s == pytest.approx(15.0)
    assert tracked.velocity_y_px_s == pytest.approx(-5.0)
    assert tracked.age_frames == 2


def test_repeated_select_clear_keeps_histories_and_sessions_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_count = 40
    attempt_type = _install_acquisition_plans(
        monkeypatch,
        *(
            _AcquisitionPlan(decision="CONFIRMED", required_frames=1)
            for _ in range(episode_count)
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")

    for frame_id in range(episode_count):
        processor.process(_packet(frame_id), processing_enabled=False)
        (event,) = processor.apply_target_control(
            _select(frame_id, selection_id=frame_id + 1)
        )
        assert isinstance(event, TargetAcquired)
        assert event.target.track_id == frame_id + 1
        assert processor.apply_target_control(_clear()) == ()

    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._acquisition is None
    assert processor._target_session is None
    assert processor._current_target is None
    assert processor._public_track_ids.next_id == episode_count + 1
    assert len(processor._frame_history) == 12
    assert len(processor._timing_history) == 12
    assert len(attempt_type.instances) == episode_count
    assert all(
        instance.result.target_session.closed
        for instance in attempt_type.instances
    )
