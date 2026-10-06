from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest

from navmin.contracts import (
    CameraRole,
    FramePacket,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TargetTrackingEndReason,
)
from navmin.vision.processors.vt11 import processor as processor_module
from navmin.vision.processors.vt11.acquisition.candidate_detector import Candidate
from navmin.vision.processors.vt11.acquisition.roi import RoiRect
from navmin.vision.processors.vt11.processor import (
    Vt11ProcessorState,
    Vt11VisionProcessor,
)
from navmin.vision.processors.vt11.tracking.reacquisition import ReacqFrameResult
from navmin.vision.processors.vt11.tracking.session import TargetSessionStep
from navmin.vision.processors.vt11.tracking.tracker import TrackFrameResult


@dataclass
class _AcquisitionPlan:
    decision: str
    required_frames: int = 3
    no_candidate: bool = False
    retrospective_replay_frames: int = 0
    continuation_error: str | None = None
    session: _FakeSession | None = None


class _FakeSession:
    def __init__(
        self,
        next_frame_index: int,
        steps: tuple[TargetSessionStep, ...] = (),
    ) -> None:
        self.next_frame_index = next_frame_index
        self._steps = deque(steps)
        self.seen_frames: list[int] = []
        self.terminal_status: str | None = None
        self.closed = False

    def step(self, frame_index: int, frame: np.ndarray) -> TargetSessionStep:
        del frame
        assert frame_index == self.next_frame_index
        self.seen_frames.append(frame_index)
        self.next_frame_index += 1
        if self._steps:
            step = self._steps.popleft()
        else:
            step = _track_step(frame_index, status="LOW_CONFIDENCE")
        self.terminal_status = step.terminal_status
        return step

    def close(self) -> None:
        self.closed = True


def _install_acquisition_plans(
    monkeypatch: pytest.MonkeyPatch,
    *plans: _AcquisitionPlan,
) -> type:
    pending = deque(plans)

    class _PlannedAttempt:
        instances: ClassVar[list[_PlannedAttempt]] = []

        def __init__(self, *, frame_index: int, **kwargs) -> None:
            del kwargs
            self.plan = pending.popleft()
            self.start_frame = frame_index
            self.next_frame_index = frame_index + 1
            self.seen_frames = [frame_index]
            self.done = False
            self.result = None
            self.selection = SimpleNamespace(frame_index=frame_index)
            self.instances.append(self)
            if self.plan.required_frames == 1:
                self._finish()

        def _finish(self) -> None:
            decision_frame = self.start_frame + self.plan.required_frames - 1
            train = None
            if not self.plan.no_candidate:
                train = SimpleNamespace(
                    frames=(SimpleNamespace(frame_index=decision_frame),)
                )
            session = self.plan.session
            if self.plan.decision == "CONFIRMED" and session is None:
                session = _FakeSession(decision_frame + 1)
            self.result = SimpleNamespace(
                decision=self.plan.decision,
                scoring=SimpleNamespace(
                    best=None if self.plan.no_candidate else object()
                ),
                train_result=train,
                target_session=session,
                retrospective_replay_frames=(
                    self.plan.retrospective_replay_frames
                ),
                continuation_error=self.plan.continuation_error,
            )
            self.done = True

        def step(self, frame_index: int, frame: np.ndarray):
            del frame
            assert frame_index == self.next_frame_index
            self.seen_frames.append(frame_index)
            self.next_frame_index += 1
            if len(self.seen_frames) == self.plan.required_frames:
                self._finish()
            return self.result

    monkeypatch.setattr(
        processor_module,
        "Vt11AcquisitionAttempt",
        _PlannedAttempt,
    )
    return _PlannedAttempt


def _packet(
    frame_id: int,
    *,
    timestamp_ns: int | None = None,
    generation: int = 1,
    image: np.ndarray | None = None,
) -> FramePacket:
    if image is None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
    image.flags.writeable = False
    return FramePacket(
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=(
            frame_id * 100_000_000 if timestamp_ns is None else timestamp_ns
        ),
        image=image,
    )


def _select(frame_id: int, *, selection_id: int = 1) -> TargetSelectionRequest:
    return TargetSelectionRequest(
        selection_id=selection_id,
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=frame_id,
        x_px=43.0,
        y_px=57.0,
    )


def _retain(
    processor: Vt11VisionProcessor,
    frame_ids: range | tuple[int, ...],
) -> None:
    for frame_id in frame_ids:
        processor.process(_packet(frame_id), processing_enabled=False)


def _candidate(x: int = 30, y: int = 20) -> Candidate:
    return Candidate(
        candidate_id=1,
        bbox=RoiRect(x, y, 12, 8),
        center_x=x + 5.5,
        center_y=y + 3.5,
        area_px=96,
        mean_response=10.0,
        peak_response=20.0,
        distance_to_click_px=0.0,
        spatial_prior=1.0,
        detector_strength=1.0,
        rank_score=1.0,
    )


def _track_step(
    frame_id: int,
    *,
    status: str,
    candidate: Candidate | None = None,
    velocity: tuple[float, float] = (0.0, 0.0),
    terminal_status: str | None = None,
) -> TargetSessionStep:
    result = TrackFrameResult(
        frame_index=frame_id,
        predicted_xy=(0.0, 0.0),
        selected=candidate,
        selected_score=1.0,
        confidence_gap=1.0,
        candidate_count=0 if candidate is None else 1,
        identity_similarity=1.0,
        motion_score=1.0,
        size_score=1.0,
        status=status,
        miss_count=0,
        velocity_xy=velocity,
    )
    return TargetSessionStep(
        phase="TRACK",
        result=result,
        phase_after="REACQ" if status == "LOST" else "TRACK",
        segment_ended=status == "LOST",
        terminal_status=terminal_status,
    )


def _reacq_step(
    frame_id: int,
    *,
    status: str,
    candidate: Candidate | None = None,
    terminal_status: str | None = None,
) -> TargetSessionStep:
    result = ReacqFrameResult(
        frame_index=frame_id,
        predicted_xy=(0.0, 0.0),
        search_radius_px=90,
        selected=candidate,
        selected_score=1.0,
        identity_similarity=1.0,
        motion_score=1.0,
        size_score=1.0,
        confidence_gap=1.0,
        confidence=1.0,
        candidate_count=0 if candidate is None else 1,
        confirmation_streak=0,
        confirmation_window_size=0,
        status=status,
        diagnostic_reason=status,
        prediction_error_px=None,
    )
    return TargetSessionStep(
        phase="REACQ",
        result=result,
        phase_after="TRACK" if status == "CONFIRMED" else "REACQ",
        segment_ended=status == "CONFIRMED" or terminal_status is not None,
        terminal_status=terminal_status,
    )


def _activate_target(
    processor: Vt11VisionProcessor,
    session: _FakeSession,
    *,
    decision_frame_id: int = 10,
    track_id: int = 7,
) -> None:
    processor._state = Vt11ProcessorState.TARGET
    processor._target_session = session
    processor._current_target = TargetRef(
        CameraRole.OVERVIEW,
        generation=1,
        track_id=track_id,
    )
    processor._current_public_target_id = track_id
    processor._initial_acquisition_decision_frame_id = decision_frame_id


def test_historical_select_replays_train_then_only_post_train_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(next_frame_index=107)
    attempt_type = _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=7,
            session=session,
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(100, 109))

    events = processor.apply_target_control(_select(100, selection_id=51))

    assert attempt_type.instances[0].seen_frames == list(range(100, 107))
    assert session.seen_frames == [107, 108]
    assert events == (
        TargetAcquired(
            selection_id=51,
            target=TargetRef(CameraRole.OVERVIEW, generation=1, track_id=1),
            request_frame_id=100,
            decision_frame_id=106,
            decision_frame_timestamp_ns=10_600_000_000,
        ),
    )
    assert processor.state is Vt11ProcessorState.TARGET


def test_live_acquisition_consumes_future_frames_and_keeps_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(next_frame_index=3)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=3,
            session=session,
        ),
    )
    now_ns = [0]
    processor = Vt11VisionProcessor(
        profile_name="day",
        history_clock_ns=lambda: now_ns[0],
    )
    _retain(processor, (0,))
    assert processor.apply_target_control(_select(0)) == ()
    assert processor.state is Vt11ProcessorState.ACQUIRING
    assert processor._public_track_ids.next_id == 1

    # Once accepted, acquisition owns its bounded timestamp context; expiry of
    # the original click in the general history must not turn it stale.
    now_ns[0] = 500_000_001
    first = _packet(1)
    second = _packet(2)
    assert processor.process(first).target_events == ()
    output = processor.process(second)

    assert isinstance(output.target_events[0], TargetAcquired)
    assert output.target_events[0].request_frame_id == 0
    assert output.target_events[0].decision_frame_id == 2
    assert output.target_events[0].decision_frame_timestamp_ns == 200_000_000
    assert output.result.tracked_objects == ()
    assert processor._public_track_ids.next_id == 2
    assert processor._frame_history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=1,
        now_ns=now_ns[0],
    ) is first
    assert processor._frame_history.find_exact(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=2,
        now_ns=now_ns[0],
    ) is second


@pytest.mark.parametrize(
    ("plan", "expected_reason"),
    [
        (
            _AcquisitionPlan(
                decision="AMBIGUOUS",
                required_frames=1,
                no_candidate=True,
            ),
            TargetAcquisitionFailureReason.NO_TARGET_AT_POINT,
        ),
        (
            _AcquisitionPlan(decision="AMBIGUOUS"),
            TargetAcquisitionFailureReason.AMBIGUOUS,
        ),
        (
            _AcquisitionPlan(decision="OBSERVE_MORE"),
            TargetAcquisitionFailureReason.INSUFFICIENT_EVIDENCE,
        ),
    ],
)
def test_acquisition_failure_mapping_does_not_allocate_public_id(
    monkeypatch: pytest.MonkeyPatch,
    plan: _AcquisitionPlan,
    expected_reason: TargetAcquisitionFailureReason,
) -> None:
    _install_acquisition_plans(monkeypatch, plan)
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(3))

    (event,) = processor.apply_target_control(_select(0))

    assert isinstance(event, TargetAcquisitionFailed)
    assert event.reason is expected_reason
    assert processor._public_track_ids.next_id == 1
    assert processor.state is Vt11ProcessorState.DISCOVERY


def test_post_train_replay_above_six_fails_without_allocating_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(next_frame_index=2)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=2,
            session=session,
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(10))

    (event,) = processor.apply_target_control(_select(0))

    assert isinstance(event, TargetAcquisitionFailed)
    assert event.reason is TargetAcquisitionFailureReason.ACQUISITION_FAILED
    assert event.decision_frame_id == 9
    assert session.seen_frames == []
    assert session.closed
    assert processor._public_track_ids.next_id == 1


def test_post_train_replay_at_six_frame_boundary_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(next_frame_index=2)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=2,
            session=session,
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(8))

    (event,) = processor.apply_target_control(_select(0))

    assert isinstance(event, TargetAcquired)
    assert session.seen_frames == list(range(2, 8))
    assert processor._public_track_ids.next_id == 2


def test_new_select_supersedes_incomplete_acquiring_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new_session = _FakeSession(next_frame_index=5)
    attempt_type = _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(decision="CONFIRMED", required_frames=10),
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=3,
            session=new_session,
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(5))
    assert processor.apply_target_control(_select(0, selection_id=1)) == ()
    assert processor.state is Vt11ProcessorState.ACQUIRING

    (event,) = processor.apply_target_control(_select(2, selection_id=2))

    assert isinstance(event, TargetAcquired)
    assert event.selection_id == 2
    assert attempt_type.instances[0].seen_frames == list(range(5))
    assert attempt_type.instances[1].seen_frames == [2, 3, 4]
    assert processor.state is Vt11ProcessorState.TARGET


def test_new_select_retires_old_target_and_failed_attempt_does_not_restore_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_session = _FakeSession(next_frame_index=2)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan(
            decision="CONFIRMED",
            required_frames=2,
            session=old_session,
        ),
        _AcquisitionPlan(
            decision="AMBIGUOUS",
            required_frames=1,
            no_candidate=True,
        ),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(5))
    (acquired,) = processor.apply_target_control(_select(0, selection_id=1))
    assert isinstance(acquired, TargetAcquired)

    (failed,) = processor.apply_target_control(_select(2, selection_id=2))

    assert old_session.closed
    assert isinstance(failed, TargetAcquisitionFailed)
    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._current_target is None
    assert processor._target_session is None
    assert processor._public_track_ids.next_id == 2


def test_stale_select_retires_active_target_without_current_frame_fallback() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, (10,))
    old_session = _FakeSession(next_frame_index=11)
    _activate_target(processor, old_session, track_id=3)

    (event,) = processor.apply_target_control(_select(9, selection_id=8))

    assert isinstance(event, TargetAcquisitionFailed)
    assert event.reason is TargetAcquisitionFailureReason.STALE_SELECTION
    assert event.request_frame_id == 9
    assert event.decision_frame_id is None
    assert old_session.closed
    assert processor.state is Vt11ProcessorState.DISCOVERY
    assert processor._current_target is None


def test_new_select_same_object_creates_new_episode_and_new_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_session = _FakeSession(next_frame_index=2)
    new_session = _FakeSession(next_frame_index=4)
    _install_acquisition_plans(
        monkeypatch,
        _AcquisitionPlan("CONFIRMED", required_frames=2, session=old_session),
        _AcquisitionPlan("CONFIRMED", required_frames=2, session=new_session),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, range(5))

    (first,) = processor.apply_target_control(_select(0, selection_id=1))
    (second,) = processor.apply_target_control(_select(2, selection_id=2))

    assert isinstance(first, TargetAcquired)
    assert isinstance(second, TargetAcquired)
    assert first.target.track_id == 1
    assert second.target.track_id == 2
    assert old_session.closed


def test_track_and_reacq_status_mapping_preserves_one_public_id() -> None:
    candidate = _candidate()
    steps = (
        _track_step(11, status="TRACK", candidate=candidate, velocity=(2.0, -1.0)),
        _track_step(12, status="LOW_CONFIDENCE", candidate=candidate),
        _track_step(13, status="LOST"),
        _reacq_step(14, status="SEARCHING", candidate=candidate),
        _reacq_step(15, status="TENTATIVE", candidate=candidate),
        _reacq_step(16, status="CONFIRMED", candidate=candidate),
        _track_step(17, status="TRACK", candidate=candidate, velocity=(1.0, 1.0)),
    )
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, (9, 10))
    _activate_target(processor, _FakeSession(11, steps))

    outputs = [processor.process(_packet(frame_id)) for frame_id in range(11, 18)]

    first = outputs[0].result.tracked_objects[0]
    assert first.track_id == 7
    assert first.velocity_x_px_s == pytest.approx(20.0)
    assert first.velocity_y_px_s == pytest.approx(-10.0)
    assert first.age_frames == 2
    assert outputs[1].result.tracked_objects == ()
    assert outputs[2].result.tracked_objects == ()
    assert outputs[3].result.tracked_objects == ()
    assert outputs[4].result.tracked_objects == ()
    reacquired = outputs[5].result.tracked_objects[0]
    assert reacquired.track_id == 7
    assert reacquired.velocity_x_px_s == 0.0
    assert reacquired.velocity_y_px_s == 0.0
    assert reacquired.age_frames == 7
    assert all(output.target_events == () for output in outputs)
    assert outputs[6].result.tracked_objects[0].track_id == 7


def test_terminal_reacq_loss_emits_tracking_ended_and_returns_to_discovery() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    _retain(processor, (20,))
    session = _FakeSession(
        21,
        (
            _reacq_step(
                21,
                status="SEARCHING",
                terminal_status="LOST_UNRECOVERED",
            ),
        ),
    )
    _activate_target(processor, session, decision_frame_id=18, track_id=4)

    output = processor.process(_packet(21, timestamp_ns=2_345_000_000))

    assert output.result.tracked_objects == ()
    assert output.target_events == (
        TargetTrackingEnded(
            target=TargetRef(CameraRole.OVERVIEW, generation=1, track_id=4),
            reason=TargetTrackingEndReason.LOST_UNRECOVERED,
            decision_frame_id=21,
            decision_frame_timestamp_ns=2_345_000_000,
        ),
    )
    assert session.closed
    assert processor.state is Vt11ProcessorState.DISCOVERY


def test_timing_history_is_bounded_and_uses_robust_recent_intervals() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    timestamps = [index * 100_000_000 for index in range(20)]
    timestamps[-2] += 700_000_000
    for frame_id, timestamp_ns in enumerate(timestamps):
        processor._append_timing(
            _packet(frame_id, timestamp_ns=timestamp_ns)
        )

    assert len(processor._timing_history) == 12
    assert processor._frames_per_second() == pytest.approx(10.0)


def _moving_frames(count: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_index in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        x = 38 + 3 * frame_index
        cv2.rectangle(frame, (x, 53), (x + 9, 61), (225, 225, 225), -1)
        cv2.rectangle(frame, (x + 7, 55), (x + 12, 58), (150, 150, 150), -1)
        frames.append(frame)
    return frames


def test_real_vt11_acquires_from_exact_historical_frame_and_catches_up() -> None:
    processor = Vt11VisionProcessor(profile_name="day", history_clock_ns=lambda: 0)
    frames = _moving_frames(9)
    for frame_id, image in enumerate(frames):
        processor.process(
            _packet(frame_id, image=image),
            processing_enabled=False,
        )

    events = processor.apply_target_control(_select(0, selection_id=99))

    assert len(events) == 1
    assert isinstance(events[0], TargetAcquired)
    assert events[0].request_frame_id == 0
    assert events[0].decision_frame_id == 6
    assert events[0].decision_frame_timestamp_ns == 600_000_000
    assert processor.state is Vt11ProcessorState.TARGET
