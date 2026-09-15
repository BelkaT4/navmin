from __future__ import annotations

from threading import Event, Lock, Thread, current_thread
from typing import Self

import numpy as np
import pytest

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.concurrency import InvalidatableLatest, LatestValue
from navmin.contracts import (
    CameraRole,
    CameraState,
    CameraStatus,
    FramePacket,
    VisionResult,
)
from navmin.vision.pipeline import (
    InMemoryFrameSource,
    MissingCalibrationError,
    VisionPipeline,
    WorkingFrameError,
    build_working_frame_corrector,
    overview_corrector,
    stereo_left_corrector,
)

WIDTH = 32
HEIGHT = 24
K = ((20.0, 0.0, 15.0), (0.0, 20.0, 11.0), (0.0, 0.0, 1.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
P_LEFT = ((20.0, 0.0, 13.0, 0.0), (0.0, 20.0, 11.0, 0.0), (0.0, 0.0, 1.0, 0.0))
Q = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def overview_calibration() -> OverviewCalibration:
    return OverviewCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
        K=K,
        D=(0.2, 0.0, 0.0, 0.0, 0.0),
        new_camera_matrix=K,
    )


def stereo_calibration() -> StereoCalibration:
    return StereoCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
        K_left=K,
        D_left=(0.0, 0.0, 0.0, 0.0, 0.0),
        K_right=K,
        D_right=(0.0, 0.0, 0.0, 0.0, 0.0),
        R=IDENTITY,
        T=(-0.46, 0.0, 0.0),
        R1=IDENTITY,
        R2=IDENTITY,
        P1=P_LEFT,
        P2=P_LEFT,
        Q=Q,
    )


def source_frame(value: int = 0) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = (np.arange(WIDTH, dtype=np.uint8) + value) % 255
    frame[:, :, 1] = np.arange(HEIGHT, dtype=np.uint8)[:, None]
    frame[:, :, 2] = value
    return frame


class _RecordingProcessor:
    def __init__(self) -> None:
        self.frames: list[FramePacket] = []
        self.processing_flags: list[bool] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionResult:
        self.frames.append(frame)
        self.processing_flags.append(processing_enabled)
        return VisionResult(frame=frame, tracked_objects=(), processing_time_ns=0)


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 100
        return self.value


class _ObservedStateLock:
    def __init__(self) -> None:
        self._lock = Lock()
        self.restart_attempted = Event()
        self.restart_observed_locked: bool | None = None

    def __enter__(self) -> Self:
        if current_thread().name == "vision-restart":
            self.restart_observed_locked = self._lock.locked()
            self.restart_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


class _BlockingResultLatest(InvalidatableLatest[VisionResult]):
    def __init__(self) -> None:
        super().__init__()
        self.publish_entered = Event()
        self.release_publish = Event()

    def publish(self, value: VisionResult) -> None:
        self.publish_entered.set()
        if not self.release_publish.wait(timeout=2.0):
            raise TimeoutError("test did not release blocked VisionResult publication")
        super().publish(value)


class _BlockingErrorStatus(LatestValue[CameraStatus]):
    def __init__(self) -> None:
        super().__init__()
        self.error_publish_entered = Event()
        self.release_error_publish = Event()

    def publish(self, value: CameraStatus) -> None:
        if value.state is CameraState.ERROR:
            self.error_publish_entered.set()
            if not self.release_error_publish.wait(timeout=2.0):
                raise TimeoutError("test did not release blocked ERROR publication")
        super().publish(value)


@pytest.mark.parametrize(
    ("camera", "corrector"),
    [
        (CameraRole.OVERVIEW, lambda: overview_corrector(overview_calibration())),
        (CameraRole.STEREO_LEFT, lambda: stereo_left_corrector(stereo_calibration())),
    ],
)
def test_working_frame_correction_preserves_size_and_publishes_read_only_image(
    camera,
    corrector,
) -> None:
    raw = source_frame(7)
    fixed = corrector().correct(raw)

    assert fixed.shape == raw.shape == (HEIGHT, WIDTH, 3)
    assert fixed.dtype == np.uint8
    assert not fixed.flags.writeable
    assert fixed is not raw
    assert not np.array_equal(fixed, raw)


def test_missing_calibration_is_explicit_and_has_no_raw_fallback() -> None:
    with pytest.raises(MissingCalibrationError):
        build_working_frame_corrector(CameraRole.OVERVIEW)
    with pytest.raises(MissingCalibrationError):
        build_working_frame_corrector(CameraRole.STEREO_LEFT)


def test_resolution_mismatch_sets_error_and_never_publishes_raw_frame() -> None:
    processor = _RecordingProcessor()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processor=processor,
        timestamp_clock_ns=_Clock(),
    )
    pipeline.start()
    pipeline.submit_decoded_frame(np.zeros((HEIGHT, WIDTH + 1, 3), dtype=np.uint8))

    with pytest.raises(WorkingFrameError):
        pipeline.process_latest()

    assert pipeline.latest_result.get() is None
    assert processor.frames == []
    status = pipeline.status.get()
    assert status is not None
    assert status.state is CameraState.ERROR


@pytest.mark.parametrize(
    ("camera", "corrector", "source_capture_id", "expected_capture_id"),
    [
        (CameraRole.OVERVIEW, lambda: overview_corrector(overview_calibration()), 55, None),
        (
            CameraRole.STEREO_LEFT,
            lambda: stereo_left_corrector(stereo_calibration()),
            55,
            55,
        ),
    ],
)
def test_pipeline_stamps_frame_contract_and_capture_id(
    camera,
    corrector,
    source_capture_id,
    expected_capture_id,
) -> None:
    processor = _RecordingProcessor()
    clock = _Clock()
    pipeline = VisionPipeline(
        camera=camera,
        corrector=corrector(),
        processor=processor,
        timestamp_clock_ns=clock,
    )
    session = pipeline.start()
    pipeline.submit_decoded_frame(source_frame(4), capture_id=source_capture_id)
    result = pipeline.process_latest()

    assert result is not None
    assert result.frame.camera is camera
    assert result.frame.generation == session.generation == 1
    assert result.frame.frame_id == 0
    assert result.frame.capture_id == expected_capture_id
    assert result.frame.receive_timestamp_ns == 200
    assert result.frame.image.shape == (HEIGHT, WIDTH, 3)
    assert not result.frame.image.flags.writeable
    status = pipeline.status.get()
    assert status is not None
    assert status.state is CameraState.ONLINE
    assert status.generation == session.generation


def test_camera_session_barrier_precedes_generation_data_and_restart_clears_old_state() -> None:
    processor = _RecordingProcessor()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processor=processor,
        timestamp_clock_ns=_Clock(),
    )

    first_session = pipeline.start()
    observed_first = pipeline.session_barriers.receive_nowait()
    assert observed_first == first_session
    assert pipeline.latest_result.get() is None

    pipeline.submit_decoded_frame(source_frame(1))
    first_result = pipeline.process_latest()
    assert first_result is not None
    assert first_result.frame.generation == 1

    pipeline.submit_decoded_frame(source_frame(2))
    second_session = pipeline.start()
    assert second_session.generation == 2
    assert pipeline.latest_result.get() is None
    assert pipeline.process_latest() is None
    observed_second = pipeline.session_barriers.receive_nowait()
    assert observed_second == second_session

    pipeline.submit_decoded_frame(source_frame(3))
    second_result = pipeline.process_latest()
    assert second_result is not None
    assert second_result.frame.generation == 2
    assert second_result.frame.frame_id == 0
    assert pipeline.latest_result.get() is second_result
    assert processor.reset_calls == 2


def test_late_old_generation_failure_does_not_replace_new_generation_status() -> None:
    base_corrector = overview_corrector(overview_calibration())

    class RestartThenFailCorrector:
        camera_model = base_corrector.camera_model

        def __init__(self) -> None:
            self.pipeline: VisionPipeline | None = None

        def correct(self, image: np.ndarray) -> np.ndarray:
            assert self.pipeline is not None
            self.pipeline.start()
            raise WorkingFrameError("old generation failed after restart")

    corrector = RestartThenFailCorrector()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=corrector,
        processor=_RecordingProcessor(),
        timestamp_clock_ns=_Clock(),
    )
    corrector.pipeline = pipeline
    pipeline.start()
    pipeline.submit_decoded_frame(source_frame(9))

    with pytest.raises(WorkingFrameError):
        pipeline.process_latest()

    status = pipeline.status.get()
    assert status is not None
    assert status.generation == 2
    assert status.state is CameraState.STARTING
    assert pipeline.latest_result.get() is None


def test_latest_only_pending_slot_processes_freshest_frame_without_fifo_backlog() -> None:
    processor = _RecordingProcessor()
    clock = _Clock()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processor=processor,
        timestamp_clock_ns=clock,
    )
    pipeline.start()

    first = source_frame(10)
    pipeline.submit_decoded_frame(first)
    first[:, :, :] = 255
    pipeline.submit_decoded_frame(source_frame(20))
    pipeline.submit_decoded_frame(source_frame(30))

    result = pipeline.process_latest()

    assert result is not None
    assert len(processor.frames) == 1
    assert result.frame.receive_timestamp_ns == 400
    assert result.frame.frame_id == 0
    assert pipeline.process_latest() is None


def test_in_memory_source_is_latest_only() -> None:
    source = InMemoryFrameSource()
    source.push(source_frame(1), capture_id=1)
    source.push(source_frame(2), capture_id=2)
    source.push(source_frame(3), capture_id=3)

    latest = source.read()

    assert latest is not None
    assert latest.capture_id == 3
    assert np.array_equal(latest.image, source_frame(3))
    assert source.read() is None


def test_in_memory_source_and_processing_disabled_use_same_pipeline_path() -> None:
    processor = _RecordingProcessor()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processor=processor,
        timestamp_clock_ns=_Clock(),
        processing_enabled=False,
    )
    source = InMemoryFrameSource()
    source.push(source_frame(8))
    pipeline.start()

    assert pipeline.submit_from_source(source)
    result = pipeline.process_latest()

    assert result is not None
    assert result.tracked_objects == ()
    assert processor.processing_flags == [False]
    assert not pipeline.submit_from_source(source)


def test_restart_cannot_interleave_after_success_generation_check_before_publication() -> None:
    latest_result = _BlockingResultLatest()
    state_lock = _ObservedStateLock()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processor=_RecordingProcessor(),
        latest_result=latest_result,
        timestamp_clock_ns=_Clock(),
    )
    pipeline._state_lock = state_lock
    pipeline.start()
    pipeline.submit_decoded_frame(source_frame(12))

    processing_result: list[VisionResult | None] = []
    processing_thread = Thread(
        target=lambda: processing_result.append(pipeline.process_latest()),
        name="vision-processing",
    )
    processing_thread.start()
    assert latest_result.publish_entered.wait(timeout=2.0)

    restart_thread = Thread(target=pipeline.start, name="vision-restart")
    restart_thread.start()
    assert state_lock.restart_attempted.wait(timeout=2.0)
    restart_observed_locked = state_lock.restart_observed_locked

    latest_result.release_publish.set()
    processing_thread.join(timeout=2.0)
    restart_thread.join(timeout=2.0)
    assert not processing_thread.is_alive()
    assert not restart_thread.is_alive()
    assert len(processing_result) == 1
    assert restart_observed_locked is True

    assert pipeline.generation == 2
    assert pipeline.latest_result.get() is None
    status = pipeline.status.get()
    assert status is not None
    assert status.generation == 2
    assert status.state is CameraState.STARTING


def test_restart_cannot_interleave_after_failure_generation_check_before_error_publication() -> None:
    base_corrector = overview_corrector(overview_calibration())

    class FailingCorrector:
        camera_model = base_corrector.camera_model

        def correct(self, image: np.ndarray) -> np.ndarray:
            del image
            raise WorkingFrameError("deterministic old-generation failure")

    status_slot = _BlockingErrorStatus()
    state_lock = _ObservedStateLock()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=FailingCorrector(),
        processor=_RecordingProcessor(),
        status=status_slot,
        timestamp_clock_ns=_Clock(),
    )
    pipeline._state_lock = state_lock
    pipeline.start()
    pipeline.submit_decoded_frame(source_frame(13))

    processing_errors: list[BaseException] = []

    def process_failure() -> None:
        try:
            pipeline.process_latest()
        except WorkingFrameError as exc:
            processing_errors.append(exc)

    processing_thread = Thread(target=process_failure, name="vision-processing")
    processing_thread.start()
    assert status_slot.error_publish_entered.wait(timeout=2.0)

    restart_thread = Thread(target=pipeline.start, name="vision-restart")
    restart_thread.start()
    assert state_lock.restart_attempted.wait(timeout=2.0)
    restart_observed_locked = state_lock.restart_observed_locked

    status_slot.release_error_publish.set()
    processing_thread.join(timeout=2.0)
    restart_thread.join(timeout=2.0)
    assert not processing_thread.is_alive()
    assert not restart_thread.is_alive()
    assert len(processing_errors) == 1
    assert isinstance(processing_errors[0], WorkingFrameError)
    assert restart_observed_locked is True

    assert pipeline.generation == 2
    assert pipeline.latest_result.get() is None
    status = pipeline.status.get()
    assert status is not None
    assert status.generation == 2
    assert status.state is CameraState.STARTING
