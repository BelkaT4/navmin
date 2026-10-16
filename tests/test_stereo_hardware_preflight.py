from __future__ import annotations

from threading import Event

import numpy as np

from navmin.contracts import CameraRole, FramePacket, VisionResult
from navmin.diagnostics.stereo_preflight import (
    StereoHardwarePreflight,
    StereoPreflightComponents,
    StereoPreflightPairingSink,
)
from navmin.diagnostics.stereo_runtime import StereoRuntimeDiagnostics
from navmin.vision.camera_worker import CameraWorker
from navmin.vision.pipeline import (
    InMemoryFrameSource,
    VisionPipeline,
    WorkingFrameError,
)
from navmin.vision.processor import VisionProcessorOutput
from navmin.vision.stereo_pairing import StereoPairer
from navmin.vision.stereo_timestamp_pairing import TimestampStereoPairer


class _CameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float):
        raise AssertionError("preflight does not use camera geometry")


class _IdentityCorrector:
    camera_model = _CameraModel()

    def correct(self, image: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(image)
        result.flags.writeable = False
        return result


class _RejectingCorrector(_IdentityCorrector):
    def correct(self, image: np.ndarray) -> np.ndarray:
        raise WorkingFrameError("source resolution 8x6 does not match calibration 9x6")


class _NoopProcessor:
    def reset(self) -> None:
        return None

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        assert not processing_enabled
        return VisionProcessorOutput(VisionResult(frame, (), 0))


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def wait(self, seconds: float) -> None:
        Event().wait(0.005)  # Yield to the real CameraWorker owner threads.
        self.now += seconds


def _components(
    *,
    left_frame: bool,
    right_frame: bool,
    reject_left: bool = False,
) -> tuple[StereoPreflightComponents, _FakeTime]:
    exact = StereoPairer(right_frame_buffer_size=2, pair_timeout_ms=100)
    timestamp = TimestampStereoPairer(
        right_frame_buffer_size=2,
        pair_timeout_ms=100,
    )
    for camera in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
        exact.start_generation(camera, 1)
        timestamp.start_generation(camera, 1)
    sink = StereoPreflightPairingSink(exact, timestamp)
    sources = {
        CameraRole.STEREO_LEFT: InMemoryFrameSource(),
        CameraRole.STEREO_RIGHT: InMemoryFrameSource(),
    }
    workers = {}
    for camera in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
        pipeline = VisionPipeline(
            camera=camera,
            corrector=(
                _RejectingCorrector()
                if reject_left and camera is CameraRole.STEREO_LEFT
                else _IdentityCorrector()
            ),
            processor=_NoopProcessor(),
            corrected_frame_sink=sink,
            processing_enabled=False,
        )
        workers[camera] = CameraWorker(
            source=sources[camera],
            pipeline=pipeline,
            idle_wait_s=0.001,
        )

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    if left_frame:
        sources[CameraRole.STEREO_LEFT].push(
            image,
            capture_id=None,
            receive_timestamp_ns=1_000_000,
        )
    if right_frame:
        sources[CameraRole.STEREO_RIGHT].push(
            image,
            capture_id=None,
            receive_timestamp_ns=1_001_000,
        )
    diagnostics = StereoRuntimeDiagnostics(
        left_pipeline=workers[CameraRole.STEREO_LEFT].pipeline,
        right_pipeline=workers[CameraRole.STEREO_RIGHT].pipeline,
        exact_pairer=exact,
        timestamp_pairer=timestamp,
    )
    fake_time = _FakeTime()
    return (
        StereoPreflightComponents(
            left_worker=workers[CameraRole.STEREO_LEFT],
            right_worker=workers[CameraRole.STEREO_RIGHT],
            exact_pairer=exact,
            timestamp_pairer=timestamp,
            diagnostics=diagnostics,
        ),
        fake_time,
    )


def _run(*, left_frame: bool, right_frame: bool, reject_left: bool = False):
    components, fake_time = _components(
        left_frame=left_frame,
        right_frame=right_frame,
        reject_left=reject_left,
    )
    outcome = StereoHardwarePreflight(
        components,
        clock=fake_time,
        wait=fake_time.wait,
    ).run(0.2)
    return components, outcome


def test_preflight_capture_id_none_pairs_by_timestamp_and_stops_cleanly() -> None:
    components, outcome = _run(left_frame=True, right_frame=True)

    assert outcome.passed
    assert outcome.snapshot.left_camera.frames_received == 1
    assert outcome.snapshot.right_camera.frames_received == 1
    assert outcome.snapshot.left_camera.decoded_width == 8
    assert outcome.snapshot.right_camera.decoded_height == 6
    assert outcome.snapshot.exact_pairing.exact_pairs_emitted == 0
    assert outcome.snapshot.exact_pairing.capture_id_missing == 2
    assert outcome.snapshot.timestamp_pairing.timestamp_pairs_emitted == 1
    assert not components.left_worker.is_alive()
    assert not components.right_worker.is_alive()
    assert outcome.to_dict()["workers_stopped"] is True


def test_preflight_no_frame_left_is_hard_failure() -> None:
    _components_value, outcome = _run(left_frame=False, right_frame=True)

    assert not outcome.passed
    assert "stereo_left: no decoded frames" in outcome.hard_failures
    assert not any("capture_id" in failure for failure in outcome.hard_failures)


def test_preflight_no_frame_right_is_hard_failure() -> None:
    _components_value, outcome = _run(left_frame=True, right_frame=False)

    assert not outcome.passed
    assert "stereo_right: no decoded frames" in outcome.hard_failures


def test_preflight_reports_actual_resolution_before_pipeline_rejection() -> None:
    _components_value, outcome = _run(
        left_frame=True,
        right_frame=True,
        reject_left=True,
    )

    assert not outcome.passed
    assert outcome.snapshot.left_camera.decoded_width == 8
    assert outcome.snapshot.left_camera.decoded_height == 6
    assert any("worker/source failed" in failure for failure in outcome.hard_failures)
