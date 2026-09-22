from __future__ import annotations

import time
from threading import Event, Thread

import numpy as np
import pytest

from navmin.calibration import OverviewCalibration
from navmin.config.models import CameraConfig
from navmin.contracts import CameraRole, CameraState
from navmin.vision import gstreamer_source
from navmin.vision.camera_worker import CameraWorker, build_camera_worker
from navmin.vision.gstreamer_source import (
    GStreamerRtpJpegSource,
    UnsupportedCameraTransportError,
    build_rtp_jpeg_pipeline_description,
    initialize_gstreamer_runtime,
)
from navmin.vision.pipeline import (
    InMemoryFrameSource,
    VisionPipeline,
    overview_corrector,
)

WIDTH = 32
HEIGHT = 24
K = ((20.0, 0.0, 15.0), (0.0, 20.0, 11.0), (0.0, 0.0, 1.0))


def camera_config(**overrides) -> CameraConfig:
    values = {
        "enabled": True,
        "address": "0.0.0.0",
        "port": 8888,
        "rtp_enabled": True,
        "buffer_size": 1,
        "processing_enabled": True,
        "vision_processor_class": "Legacy14VisionProcessor",
    }
    values.update(overrides)
    return CameraConfig(**values)


def overview_calibration() -> OverviewCalibration:
    return OverviewCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
        K=K,
        D=(0.02, -0.003, 0.0005, 0.0),
        new_camera_matrix=K,
    )


def frame(value: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


class _FakeBackend:
    def __init__(self, on_frame, on_failure) -> None:
        self.on_frame = on_frame
        self.on_failure = on_failure
        self.started = False
        self.stopped = False
        self.polled_failure: BaseException | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def poll_failure(self) -> BaseException | None:
        return self.polled_failure


class _BackendFactory:
    def __init__(self) -> None:
        self.description: str | None = None
        self.backend: _FakeBackend | None = None

    def __call__(self, description, on_frame, on_failure):
        self.description = description
        self.backend = _FakeBackend(on_frame, on_failure)
        return self.backend


def test_gstreamer_runtime_initialization_is_serialized_and_cached(monkeypatch) -> None:
    calls: list[int] = []

    def fake_import() -> tuple[object, object]:
        calls.append(1)
        time.sleep(0.02)
        return object(), object()

    monkeypatch.setattr(gstreamer_source, "_GSTREAMER_MODULES", [])
    monkeypatch.setattr(gstreamer_source, "_import_gstreamer_modules", fake_import)

    threads = [
        Thread(target=initialize_gstreamer_runtime),
        Thread(target=initialize_gstreamer_runtime),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(calls) == 1


def test_rtp_jpeg_pipeline_uses_camera_config_and_low_latency_appsink() -> None:
    description = build_rtp_jpeg_pipeline_description(
        camera_config(address="127.0.0.1", port=8889, buffer_size=1)
    )

    assert 'udpsrc address="127.0.0.1" port=8889' in description
    assert "application/x-rtp,media=video,encoding-name=JPEG,payload=26,clock-rate=90000" in description
    assert "! rtpjpegdepay ! jpegdec ! videoconvert ! video/x-raw,format=BGR" in description
    assert "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false" in description


def test_non_rtp_camera_config_is_explicitly_unsupported() -> None:
    with pytest.raises(UnsupportedCameraTransportError):
        build_rtp_jpeg_pipeline_description(camera_config(rtp_enabled=False))


def test_gstreamer_source_owns_frame_memory_and_replaces_latest_without_fifo() -> None:
    factory = _BackendFactory()
    timestamps = iter((101, 202, 303))
    source = GStreamerRtpJpegSource(
        camera_config(),
        timestamp_clock_ns=lambda: next(timestamps),
        backend_factory=factory,
    )
    source.start()
    assert source.started
    assert factory.backend is not None and factory.backend.started

    first = frame(10)
    factory.backend.on_frame(first)
    first[:, :, :] = 99
    factory.backend.on_frame(frame(20))
    factory.backend.on_frame(frame(30))

    latest = source.read()
    assert latest is not None
    assert latest.capture_id is None
    assert latest.receive_timestamp_ns == 303
    assert np.array_equal(latest.image, frame(30))
    assert source.read() is None

    source.stop()
    assert not source.started
    assert factory.backend.stopped


def test_gstreamer_source_surfaces_backend_failure() -> None:
    factory = _BackendFactory()
    source = GStreamerRtpJpegSource(camera_config(), backend_factory=factory)
    source.start()
    assert factory.backend is not None
    error = RuntimeError("decoder failed")
    factory.backend.polled_failure = error

    assert source.failure is error
    source.stop()


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_build_camera_worker_binds_existing_camera_config_without_parallel_schema() -> None:
    config = camera_config(
        address="0.0.0.0",
        port=8889,
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )
    worker = build_camera_worker(
        camera=CameraRole.STEREO_LEFT,
        config=config,
        corrector=overview_corrector(overview_calibration()),
    )

    assert isinstance(worker.source, GStreamerRtpJpegSource)
    assert 'port=8889' in worker.source.pipeline_description
    assert worker.pipeline.camera is CameraRole.STEREO_LEFT


def test_camera_worker_uses_existing_source_to_pipeline_path_and_stops_bounded() -> None:
    source = InMemoryFrameSource()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processing_enabled=False,
    )
    worker = CameraWorker(source=source, pipeline=pipeline, idle_wait_s=0.01)
    worker.start()
    _wait_until(lambda: pipeline.generation == 1)

    source.push(frame(7), receive_timestamp_ns=123456)
    _wait_until(lambda: pipeline.latest_result.get() is not None)
    result = pipeline.latest_result.get()
    assert result is not None
    assert result.frame.receive_timestamp_ns == 123456
    assert result.frame.capture_id is None
    assert result.tracked_objects == ()

    assert worker.stop(timeout=1.0)
    assert not worker.is_alive()
    status = pipeline.status.get()
    assert status is not None
    assert status.state is CameraState.STOPPED


class _FailingSource:
    def __init__(self) -> None:
        self.started = Event()
        self.stopped = False
        self.error = RuntimeError("source runtime failure")

    def start(self) -> None:
        self.started.set()

    def stop(self) -> None:
        self.stopped = True

    def read(self):
        return None

    @property
    def failure(self) -> BaseException | None:
        return self.error if self.started.is_set() else None


def test_camera_worker_publishes_source_failure_and_exits_predictably() -> None:
    source = _FailingSource()
    pipeline = VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processing_enabled=False,
    )
    worker = CameraWorker(source=source, pipeline=pipeline, idle_wait_s=0.01)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert source.stopped
    status = pipeline.status.get()
    assert status is not None
    assert status.state is CameraState.ERROR
    assert status.error_code == "RuntimeError"
    assert "source runtime failure" in (status.message or "")
