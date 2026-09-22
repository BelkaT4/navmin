"""Low-latency RTP/JPEG-over-UDP decoded-frame source for Vision."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock
from time import monotonic_ns
from typing import Protocol

import numpy as np

from navmin.concurrency import LatestValue
from navmin.config.models import CameraConfig

from .pipeline import DecodedFrame

LOGGER = logging.getLogger(__name__)

RTP_JPEG_CAPS = (
    "application/x-rtp,media=video,encoding-name=JPEG,payload=26,clock-rate=90000"
)

_GSTREAMER_RUNTIME_LOCK = Lock()
_GSTREAMER_MODULES: list[tuple[object, object]] = []


class CameraSourceError(RuntimeError):
    """Base error for the production decoded-frame source."""


class UnsupportedCameraTransportError(CameraSourceError):
    """The configured transport is not implemented by this source."""


class GStreamerUnavailableError(CameraSourceError):
    """PyGObject/GStreamer runtime is unavailable or incomplete."""


def _import_gstreamer_modules() -> tuple[object, object]:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import GLib, Gst, GstApp

        Gst.init(None)
        _ = GstApp.AppSink
    except (AttributeError, ImportError, ValueError) as exc:
        raise GStreamerUnavailableError(
            "PyGObject with Gst 1.0 and GstApp 1.0 is required"
        ) from exc
    return GLib, Gst


def _load_gstreamer_modules() -> tuple[object, object]:
    with _GSTREAMER_RUNTIME_LOCK:
        if _GSTREAMER_MODULES:
            return _GSTREAMER_MODULES[0]
        modules = _import_gstreamer_modules()
        _GSTREAMER_MODULES.append(modules)
        return modules


def initialize_gstreamer_runtime() -> None:
    """Initialize PyGObject/GStreamer once before camera worker threads start."""
    _load_gstreamer_modules()


class _SourceBackend(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def poll_failure(self) -> BaseException | None: ...


BackendFactory = Callable[
    [str, Callable[[np.ndarray], None], Callable[[BaseException], None]],
    _SourceBackend,
]


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_rtp_jpeg_pipeline_description(config: CameraConfig) -> str:
    """Build the v1 receiver pipeline from the existing CameraConfig."""
    if not config.rtp_enabled:
        raise UnsupportedCameraTransportError(
            "CameraConfig.rtp_enabled must be true for the RTP/JPEG source"
        )
    return (
        f"udpsrc address={_gst_quote(config.address)} port={config.port} "
        f"caps={_gst_quote(RTP_JPEG_CAPS)} "
        "! rtpjpegdepay "
        "! jpegdec "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink name=sink emit-signals=true "
        f"max-buffers={config.buffer_size} drop=true sync=false"
    )


class GStreamerRtpJpegSource:
    """Own one Gst pipeline and expose only the freshest decoded BGR frame."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        timestamp_clock_ns: Callable[[], int] = monotonic_ns,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self._config = config
        self.pipeline_description = build_rtp_jpeg_pipeline_description(config)
        self._timestamp_clock_ns = timestamp_clock_ns
        self._backend_factory = backend_factory or _default_backend_factory
        self._latest: LatestValue[DecodedFrame] = LatestValue()
        self._last_read_revision = 0
        self._backend: _SourceBackend | None = None
        self._failure: BaseException | None = None
        self._started = False
        self._state_lock = Lock()

    @property
    def failure(self) -> BaseException | None:
        self._poll_backend_failure()
        with self._state_lock:
            return self._failure

    @property
    def started(self) -> bool:
        with self._state_lock:
            return self._started

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._failure = None
            backend = self._backend_factory(
                self.pipeline_description,
                self._publish_frame,
                self._record_failure,
            )
            self._backend = backend
        try:
            backend.start()
        except CameraSourceError as exc:
            self._record_failure(exc)
            try:
                backend.stop()
            finally:
                with self._state_lock:
                    self._backend = None
            raise
        with self._state_lock:
            self._started = True
        LOGGER.info(
            "GStreamer camera source started bind=%s port=%d",
            self._config.address,
            self._config.port,
        )

    def stop(self) -> None:
        with self._state_lock:
            backend = self._backend
            was_started = self._started
            self._started = False
            self._backend = None
        if backend is not None:
            backend.stop()
        if was_started:
            LOGGER.info(
                "GStreamer camera source stopped bind=%s port=%d",
                self._config.address,
                self._config.port,
            )

    def read(self) -> DecodedFrame | None:
        self._poll_backend_failure()
        snapshot = self._latest.snapshot()
        if snapshot.value is None or snapshot.revision <= self._last_read_revision:
            return None
        self._last_read_revision = snapshot.revision
        return snapshot.value

    def _publish_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            self._record_failure(CameraSourceError("decoded frame must be numpy.ndarray"))
            return
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            self._record_failure(
                CameraSourceError(
                    f"decoded frame must be BGR uint8 HxWx3, got {frame.dtype} {frame.shape!r}"
                )
            )
            return
        owned = np.array(frame, copy=True, order="C")
        self._latest.publish(
            DecodedFrame(
                image=owned,
                capture_id=None,
                receive_timestamp_ns=self._timestamp_clock_ns(),
            )
        )

    def _record_failure(self, error: BaseException) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = error

    def _poll_backend_failure(self) -> None:
        with self._state_lock:
            backend = self._backend
        if backend is None:
            return
        error = backend.poll_failure()
        if error is not None:
            self._record_failure(error)


class _PyGObjectGstBackend:
    """Thin owner-local PyGObject adapter; no GLib application loop is required."""

    def __init__(
        self,
        pipeline_description: str,
        on_frame: Callable[[np.ndarray], None],
        on_failure: Callable[[BaseException], None],
    ) -> None:
        GLib, Gst = _load_gstreamer_modules()
        self._GLib = GLib
        self._Gst = Gst
        self._on_frame = on_frame
        self._on_failure = on_failure
        try:
            self._pipeline = Gst.parse_launch(pipeline_description)
        except GLib.Error as exc:
            raise CameraSourceError(f"cannot create GStreamer pipeline: {exc}") from exc
        self._sink = self._pipeline.get_by_name("sink")
        if self._sink is None:
            raise CameraSourceError("GStreamer pipeline does not expose appsink 'sink'")
        self._sink.connect("new-sample", self._on_new_sample)
        self._bus = self._pipeline.get_bus()

    def start(self) -> None:
        result = self._pipeline.set_state(self._Gst.State.PLAYING)
        if result == self._Gst.StateChangeReturn.FAILURE:
            raise CameraSourceError("GStreamer pipeline failed to enter PLAYING")

    def stop(self) -> None:
        try:
            result = self._pipeline.set_state(self._Gst.State.NULL)
        except self._GLib.Error as exc:
            raise CameraSourceError(f"cannot stop GStreamer pipeline: {exc}") from exc
        if result == self._Gst.StateChangeReturn.FAILURE:
            raise CameraSourceError("GStreamer pipeline failed to enter NULL")

    def poll_failure(self) -> BaseException | None:
        message = self._bus.timed_pop_filtered(
            0,
            self._Gst.MessageType.ERROR | self._Gst.MessageType.EOS,
        )
        if message is None:
            return None
        if message.type == self._Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = f": {debug}" if debug else ""
            return CameraSourceError(f"GStreamer error: {error}{detail}")
        return CameraSourceError("GStreamer stream reached EOS")

    def _on_new_sample(self, sink):
        Gst = self._Gst
        try:
            frame = self._copy_sample_frame(sink)
        except CameraSourceError as exc:
            self._on_failure(exc)
            return Gst.FlowReturn.ERROR
        self._on_frame(frame)
        return Gst.FlowReturn.OK

    def _copy_sample_frame(self, sink) -> np.ndarray:
        Gst = self._Gst
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                raise CameraSourceError("GStreamer appsink returned no sample")
            caps = sample.get_caps()
            if caps is None or caps.get_size() < 1:
                raise CameraSourceError("decoded GStreamer sample has no caps")
            structure = caps.get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            if width <= 0 or height <= 0:
                raise CameraSourceError(
                    f"decoded GStreamer sample has invalid size {width}x{height}"
                )
            buffer = sample.get_buffer()
            if buffer is None:
                raise CameraSourceError("decoded GStreamer sample has no buffer")
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                raise CameraSourceError("cannot map decoded GStreamer buffer")
            try:
                expected_bytes = width * height * 3
                raw = np.frombuffer(map_info.data, dtype=np.uint8)
                if raw.size != expected_bytes:
                    raise CameraSourceError(
                        "decoded BGR buffer size does not match caps: "
                        f"{raw.size} bytes for {width}x{height}"
                    )
                # Gst owns map_info.data. The independent copy must happen before unmap.
                return raw.reshape((height, width, 3)).copy()
            finally:
                buffer.unmap(map_info)
        except CameraSourceError:
            raise
        except (
            self._GLib.Error,
            BufferError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise CameraSourceError(f"cannot decode GStreamer sample: {exc}") from exc


def _default_backend_factory(
    pipeline_description: str,
    on_frame: Callable[[np.ndarray], None],
    on_failure: Callable[[BaseException], None],
) -> _SourceBackend:
    return _PyGObjectGstBackend(pipeline_description, on_frame, on_failure)


__all__ = [
    "RTP_JPEG_CAPS",
    "CameraSourceError",
    "GStreamerRtpJpegSource",
    "GStreamerUnavailableError",
    "UnsupportedCameraTransportError",
    "build_rtp_jpeg_pipeline_description",
    "initialize_gstreamer_runtime",
]
