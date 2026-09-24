from __future__ import annotations

import time
from threading import Event, Lock, Thread

import numpy as np
import pytest

from navmin.calibration import OverviewCalibration
from navmin.config.models import (
    CameraConfig,
    RtpJpegSourceConfig,
    RtspDecoderMode,
    RtspProtocol,
    RtspSourceConfig,
)
from navmin.contracts import CameraRole, CameraState
from navmin.vision import gstreamer_source
from navmin.vision.camera_source import create_camera_source
from navmin.vision.camera_worker import CameraWorker, build_camera_worker
from navmin.vision.gstreamer_source import (
    CameraSourceError,
    GStreamerRtpJpegSource,
    GStreamerRtspSource,
    build_rtp_jpeg_pipeline_description,
    build_rtsp_pipeline_description,
    find_missing_gstreamer_elements,
    initialize_gstreamer_runtime,
    rtsp_uri_for_diagnostics,
)
from navmin.vision.pipeline import (
    DecodedFrame,
    InMemoryFrameSource,
    VisionPipeline,
    overview_corrector,
)

WIDTH = 32
HEIGHT = 24
K = ((20.0, 0.0, 15.0), (0.0, 20.0, 11.0), (0.0, 0.0, 1.0))


def rtp_source_config(
    *,
    bind_address: str = "0.0.0.0",
    port: int = 8888,
    buffer_size: int = 1,
) -> RtpJpegSourceConfig:
    return RtpJpegSourceConfig(
        bind_address=bind_address,
        port=port,
        buffer_size=buffer_size,
    )


def rtsp_source_config(
    *,
    uri: str = "rtsp://camera.local:8554/stream",
    protocol: RtspProtocol = RtspProtocol.TCP,
    latency_ms: int = 100,
    drop_on_latency: bool = True,
    buffer_size: int = 1,
) -> RtspSourceConfig:
    return RtspSourceConfig(
        uri=uri,
        protocol=protocol,
        decoder_mode=RtspDecoderMode.SOFTWARE,
        latency_ms=latency_ms,
        drop_on_latency=drop_on_latency,
        buffer_size=buffer_size,
    )


def camera_config(
    *,
    source: RtpJpegSourceConfig | RtspSourceConfig | None = None,
    processing_enabled: bool = True,
    vision_processor_class: str = "Legacy14VisionProcessor",
) -> CameraConfig:
    return CameraConfig(
        enabled=True,
        source=source or rtp_source_config(),
        processing_enabled=processing_enabled,
        vision_processor_class=vision_processor_class,
    )


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




def test_gstreamer_element_probe_reuses_cached_runtime(monkeypatch) -> None:
    calls: list[str] = []

    class ElementFactory:
        @staticmethod
        def find(name):
            calls.append(name)
            return None if name == "jpegdec" else object()

    fake_gst = type("FakeGst", (), {"ElementFactory": ElementFactory})
    monkeypatch.setattr(gstreamer_source, "_GSTREAMER_MODULES", [(object(), fake_gst)])

    assert find_missing_gstreamer_elements(("udpsrc", "jpegdec", "appsink")) == (
        "jpegdec",
    )
    assert calls == ["udpsrc", "jpegdec", "appsink"]

def test_rtp_jpeg_pipeline_uses_typed_source_config_and_low_latency_appsink() -> None:
    description = build_rtp_jpeg_pipeline_description(
        rtp_source_config(bind_address="127.0.0.1", port=8889, buffer_size=1)
    )

    assert 'udpsrc address="127.0.0.1" port=8889' in description
    assert "application/x-rtp,media=video,encoding-name=JPEG,payload=26,clock-rate=90000" in description
    assert "! rtpjpegdepay ! jpegdec ! videoconvert ! video/x-raw,format=BGR" in description
    assert "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false" in description


def test_rtsp_diagnostic_uri_redacts_credentials_and_query() -> None:
    assert (
        rtsp_uri_for_diagnostics(
            "rtsp://operator:secret@camera.local:8554/stream?token=hidden"
        )
        == "rtsp://camera.local:8554/stream"
    )


def test_rtsp_source_redacts_credentials_from_surfaced_backend_failure() -> None:
    uri = "rtsp://operator:secret@camera.local:8554/stream?token=hidden"
    factory = _BackendFactory()
    source = GStreamerRtspSource(
        rtsp_source_config(uri=uri),
        backend_factory=factory,
    )
    source.start()
    assert factory.backend is not None
    factory.backend.polled_failure = CameraSourceError(
        f"failed to connect to {uri}; token=hidden; user=operator"
    )

    failure = source.failure
    assert failure is not None
    detail = str(failure)
    assert "rtsp://camera.local:8554/stream" in detail
    assert "secret" not in detail
    assert "hidden" not in detail
    assert "operator" not in detail
    source.stop()


@pytest.mark.parametrize(
    ("protocol", "drop_on_latency", "expected_protocol", "expected_drop"),
    [
        (RtspProtocol.TCP, True, "protocols=tcp", "drop-on-latency=true"),
        (RtspProtocol.UDP, False, "protocols=udp", "drop-on-latency=false"),
    ],
)
def test_rtsp_pipeline_is_h264_software_low_latency(
    protocol: RtspProtocol,
    drop_on_latency: bool,
    expected_protocol: str,
    expected_drop: str,
) -> None:
    description = build_rtsp_pipeline_description(
        rtsp_source_config(
            protocol=protocol,
            latency_ms=75,
            drop_on_latency=drop_on_latency,
        )
    )

    assert 'rtspsrc location="rtsp://camera.local:8554/stream"' in description
    assert expected_protocol in description
    assert "latency=75" in description
    assert expected_drop in description
    assert "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert" in description
    assert "video/x-raw,format=BGR" in description
    assert "max-buffers=1 drop=true sync=false" in description


def test_gstreamer_source_owns_frame_memory_and_replaces_latest_without_fifo() -> None:
    factory = _BackendFactory()
    timestamps = iter((101, 202, 303))
    source = GStreamerRtpJpegSource(
        rtp_source_config(),
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


def test_rtsp_source_uses_same_decoded_frame_contract_and_cleanup() -> None:
    factory = _BackendFactory()
    source = GStreamerRtspSource(
        rtsp_source_config(protocol=RtspProtocol.TCP),
        timestamp_clock_ns=lambda: 404,
        backend_factory=factory,
    )

    source.start()
    assert factory.backend is not None
    factory.backend.on_frame(frame(42))

    decoded = source.read()
    assert decoded is not None
    assert decoded.image.dtype == np.uint8
    assert decoded.image.shape == (HEIGHT, WIDTH, 3)
    assert decoded.receive_timestamp_ns == 404
    assert decoded.capture_id is None
    assert source.read() is None

    source.stop()
    assert factory.backend.stopped


def test_gstreamer_source_surfaces_backend_failure() -> None:
    factory = _BackendFactory()
    source = GStreamerRtpJpegSource(rtp_source_config(), backend_factory=factory)
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


def test_source_factory_selects_transport_specific_implementation() -> None:
    rtp = create_camera_source(camera_config(source=rtp_source_config()))
    rtsp = create_camera_source(camera_config(source=rtsp_source_config()))

    assert isinstance(rtp, GStreamerRtpJpegSource)
    assert isinstance(rtsp, GStreamerRtspSource)


def test_build_camera_worker_binds_existing_camera_config_without_parallel_schema() -> None:
    config = camera_config(
        source=rtp_source_config(bind_address="0.0.0.0", port=8889),
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


class _BackendHistoryFactory:
    def __init__(self) -> None:
        self.backends: list[_FakeBackend] = []

    def __call__(self, description, on_frame, on_failure):
        del description
        backend = _FakeBackend(on_frame, on_failure)
        self.backends.append(backend)
        return backend


class _FailFirstBackendFactory(_BackendHistoryFactory):
    def __call__(self, description, on_frame, on_failure):
        backend = super().__call__(description, on_frame, on_failure)
        if len(self.backends) == 1:
            def fail_start() -> None:
                raise CameraSourceError("initial backend start failed")

            backend.start = fail_start
        return backend


def test_gstreamer_rtsp_source_can_restart_after_backend_start_failure() -> None:
    factory = _FailFirstBackendFactory()
    source = GStreamerRtspSource(
        rtsp_source_config(),
        timestamp_clock_ns=lambda: 515,
        backend_factory=factory,
    )

    with pytest.raises(CameraSourceError, match="initial backend start failed"):
        source.start()
    assert not source.started
    assert factory.backends[0].stopped

    source.start()
    assert source.started
    second_backend = factory.backends[-1]
    second_backend.on_frame(frame(15))
    decoded = source.read()
    assert decoded is not None
    assert np.array_equal(decoded.image, frame(15))
    source.stop()


def test_gstreamer_reconnect_session_ignores_callbacks_from_old_backend() -> None:
    factory = _BackendHistoryFactory()
    source = GStreamerRtspSource(
        rtsp_source_config(),
        timestamp_clock_ns=iter((101, 202, 303)).__next__,
        backend_factory=factory,
    )

    source.start()
    first_backend = factory.backends[-1]
    first_backend.on_frame(frame(10))
    first = source.read()
    assert first is not None and np.array_equal(first.image, frame(10))

    first_backend.polled_failure = CameraSourceError("session one failed")
    assert source.failure is first_backend.polled_failure
    source.stop()
    source.start()
    second_backend = factory.backends[-1]

    first_backend.on_frame(frame(99))
    first_backend.on_failure(CameraSourceError("late old failure"))
    second_backend.on_frame(frame(20))

    assert source.failure is None
    recovered = source.read()
    assert recovered is not None
    assert np.array_equal(recovered.image, frame(20))
    assert source.read() is None
    source.stop()


def test_transport_sources_declare_reconnect_capability() -> None:
    rtp = GStreamerRtpJpegSource(rtp_source_config(), backend_factory=_BackendFactory())
    rtsp = GStreamerRtspSource(rtsp_source_config(), backend_factory=_BackendFactory())

    assert not rtp.supports_reconnect
    assert rtsp.supports_reconnect


class _RecoverableSource:
    supports_reconnect = True

    def __init__(self, *, start_failures: int = 0) -> None:
        self._lock = Lock()
        self._start_failures = start_failures
        self._failure: BaseException | None = None
        self._frame: DecodedFrame | None = None
        self.start_calls = 0
        self.stop_calls = 0
        self.started = False
        self.on_start = None

    def start(self) -> None:
        with self._lock:
            self.start_calls += 1
            call = self.start_calls
            self._failure = None
        if call <= self._start_failures:
            raise CameraSourceError(f"start failed {call}")
        callback = self.on_start
        if callback is not None:
            callback(call)
        with self._lock:
            self.started = True

    def stop(self) -> None:
        with self._lock:
            self.stop_calls += 1
            self.started = False
            self._frame = None

    def read(self) -> DecodedFrame | None:
        with self._lock:
            value = self._frame
            self._frame = None
            return value

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._failure = error

    def push(self, value: int, *, timestamp_ns: int) -> None:
        with self._lock:
            self._frame = DecodedFrame(
                image=frame(value),
                capture_id=None,
                receive_timestamp_ns=timestamp_ns,
            )


def _pipeline() -> VisionPipeline:
    return VisionPipeline(
        camera=CameraRole.OVERVIEW,
        corrector=overview_corrector(overview_calibration()),
        processing_enabled=False,
    )


def test_recoverable_worker_initial_failures_do_not_create_fake_generations() -> None:
    source = _RecoverableSource(start_failures=1)
    pipeline = _pipeline()
    worker = CameraWorker(
        source=source,
        pipeline=pipeline,
        reconnect_delays_s=(10.0,),
    )
    worker.start()
    _wait_until(
        lambda: (
            pipeline.status.get() is not None
            and pipeline.status.get().state is CameraState.RECONNECTING
        )
    )

    status = pipeline.status.get()
    assert status is not None
    assert status.generation is None
    assert pipeline.generation == 0

    assert worker.stop(timeout=1.0)
    assert source.start_calls == 1
    assert pipeline.generation == 0


def test_recoverable_worker_uses_bounded_backoff_sequence(monkeypatch) -> None:
    source = _RecoverableSource(start_failures=6)
    pipeline = _pipeline()
    worker = CameraWorker(source=source, pipeline=pipeline)
    waits: list[float] = []
    real_wait = worker._stop_token.wait

    def record_wait(timeout: float | None = None) -> bool:
        if timeout in (0.25, 0.5, 1.0, 2.0):
            waits.append(timeout)
            return False
        return real_wait(timeout)

    monkeypatch.setattr(worker._stop_token, "wait", record_wait)
    worker.start()
    _wait_until(lambda: pipeline.generation == 1)

    assert waits[:6] == [0.25, 0.5, 1.0, 2.0, 2.0, 2.0]
    assert source.start_calls == 7

    source.push(7, timestamp_ns=700)
    _wait_until(lambda: pipeline.latest_result.get() is not None)
    source.fail(CameraSourceError("lost after recovery"))
    _wait_until(lambda: pipeline.generation == 2)
    assert waits[6] == 0.25

    assert worker.stop(timeout=1.0)


def test_recoverable_worker_reconnects_with_new_generation_and_clears_old_result() -> None:
    source = _RecoverableSource()
    pipeline = _pipeline()
    second_start_gate = Event()
    allow_second_start = Event()

    def on_start(call: int) -> None:
        if call == 2:
            second_start_gate.set()
            allow_second_start.wait(1.0)

    source.on_start = on_start
    worker = CameraWorker(
        source=source,
        pipeline=pipeline,
        reconnect_delays_s=(0.001,),
    )
    worker.start()
    _wait_until(lambda: pipeline.generation == 1)
    first_session = pipeline.session_barriers.receive(timeout=1.0)
    assert first_session.generation == 1

    source.push(10, timestamp_ns=100)
    _wait_until(lambda: pipeline.latest_result.get() is not None)
    first_result = pipeline.latest_result.get()
    assert first_result is not None and first_result.frame.generation == 1

    source.fail(CameraSourceError("RTSP transport lost"))
    assert second_start_gate.wait(1.0)
    status = pipeline.status.get()
    assert status is not None
    assert status.state is CameraState.RECONNECTING
    assert status.generation == 1
    assert status.last_receive_timestamp_ns == 100
    assert status.error_code == "CameraSourceError"
    assert "RTSP transport lost" in (status.message or "")
    assert pipeline.latest_result.get() is first_result

    allow_second_start.set()
    _wait_until(lambda: pipeline.generation == 2)
    second_session = pipeline.session_barriers.receive(timeout=1.0)
    assert second_session.generation == 2
    assert pipeline.latest_result.get() is None

    source.push(20, timestamp_ns=200)
    _wait_until(lambda: pipeline.latest_result.get() is not None)
    recovered = pipeline.latest_result.get()
    assert recovered is not None
    assert recovered.frame.generation == 2
    assert recovered.frame.receive_timestamp_ns == 200
    assert worker.stop(timeout=1.0)


def test_recoverable_worker_shutdown_interrupts_reconnect_backoff() -> None:
    source = _RecoverableSource(start_failures=100)
    pipeline = _pipeline()
    worker = CameraWorker(
        source=source,
        pipeline=pipeline,
        reconnect_delays_s=(10.0,),
    )
    worker.start()
    _wait_until(
        lambda: (
            pipeline.status.get() is not None
            and pipeline.status.get().state is CameraState.RECONNECTING
        )
    )
    calls_before_stop = source.start_calls

    started_at = time.monotonic()
    assert worker.stop(timeout=1.0)
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert source.start_calls == calls_before_stop
    status = pipeline.status.get()
    assert status is not None and status.state is CameraState.STOPPED


def test_stop_during_successful_reconnect_attempt_does_not_publish_new_session() -> None:
    source = _RecoverableSource()
    pipeline = _pipeline()
    worker = CameraWorker(source=source, pipeline=pipeline)

    def stop_from_start(call: int) -> None:
        assert call == 1
        worker.request_stop()

    source.on_start = stop_from_start
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert pipeline.generation == 0
    assert pipeline.latest_result.get() is None


def test_reconnecting_camera_does_not_block_independent_camera_worker() -> None:
    reconnecting_source = _RecoverableSource(start_failures=100)
    reconnecting_pipeline = _pipeline()
    reconnecting_worker = CameraWorker(
        source=reconnecting_source,
        pipeline=reconnecting_pipeline,
        reconnect_delays_s=(10.0,),
    )
    online_source = InMemoryFrameSource()
    online_pipeline = VisionPipeline(
        camera=CameraRole.STEREO_LEFT,
        corrector=overview_corrector(overview_calibration()),
        processing_enabled=False,
    )
    online_worker = CameraWorker(
        source=online_source,
        pipeline=online_pipeline,
        idle_wait_s=0.01,
    )

    reconnecting_worker.start()
    online_worker.start()
    _wait_until(
        lambda: (
            reconnecting_pipeline.status.get() is not None
            and reconnecting_pipeline.status.get().state is CameraState.RECONNECTING
        )
    )
    _wait_until(lambda: online_pipeline.generation == 1)

    online_source.push(frame(7), receive_timestamp_ns=707)
    _wait_until(lambda: online_pipeline.latest_result.get() is not None)
    result = online_pipeline.latest_result.get()
    assert result is not None
    assert result.frame.receive_timestamp_ns == 707

    assert reconnecting_worker.stop(timeout=1.0)
    assert online_worker.stop(timeout=1.0)
