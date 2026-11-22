"""Reusable PyGObject appsrc JPEG -> RTP/JPEG -> UDP sender boundary."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import cv2
import numpy as np

from navmin.vision.gstreamer_source import (
    CameraSourceError,
    GStreamerUnavailableError,
    find_missing_gstreamer_elements,
    initialize_gstreamer_runtime,
)

APPSRC_RTP_JPEG_SENDER_ELEMENTS = (
    "appsrc",
    "jpegparse",
    "rtpjpegpay",
    "udpsink",
)


@dataclass(frozen=True, slots=True)
class AppsrcRtpJpegFrame:
    """One BGR frame and source timing supplied to the RTP/JPEG sender."""

    image: np.ndarray
    pts_ns: int
    capture_id: int | None = None


@dataclass(frozen=True, slots=True)
class AppsrcRtpJpegSenderConfig:
    """One localhost-capable appsrc RTP/JPEG sender endpoint."""

    host: str
    port: int
    fps: int
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ValueError("port must be an integer in range 1..65535")
        if type(self.fps) is not int or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in range 1..100")


def build_appsrc_rtp_jpeg_sender_pipeline_description(
    *,
    host: str,
    port: int,
    fps: int,
) -> str:
    """Build the established appsrc JPEG -> RTP/JPEG sender pipeline."""
    config = AppsrcRtpJpegSenderConfig(host=host, port=port, fps=fps)
    return (
        "appsrc name=source is-live=true format=time do-timestamp=false "
        f"caps=image/jpeg,framerate={config.fps}/1 "
        "! jpegparse "
        "! rtpjpegpay pt=26 "
        f"! udpsink host={config.host} port={config.port} sync=false async=false"
    )


def check_appsrc_rtp_jpeg_sender_runtime() -> str | None:
    """Return a static sender-runtime error without opening sockets."""
    try:
        initialize_gstreamer_runtime()
        missing = find_missing_gstreamer_elements(APPSRC_RTP_JPEG_SENDER_ELEMENTS)
    except GStreamerUnavailableError as exc:
        return str(exc)
    if missing:
        return "missing GStreamer elements: " + ", ".join(missing)
    return None


def _load_gst() -> Any:
    try:
        initialize_gstreamer_runtime()
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp

        _ = GstApp.AppSrc
    except (AttributeError, ImportError, ValueError, GStreamerUnavailableError) as exc:
        raise CameraSourceError(
            "PyGObject with Gst 1.0 and GstApp 1.0 is required for synthetic RTP/JPEG"
        ) from exc
    return Gst


class AppsrcRtpJpegSender:
    """Push supplied BGR frames through the established GStreamer appsrc path."""

    def __init__(
        self,
        config: AppsrcRtpJpegSenderConfig,
        frame_supplier: Callable[[], AppsrcRtpJpegFrame],
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not isinstance(config, AppsrcRtpJpegSenderConfig):
            raise TypeError("config must be AppsrcRtpJpegSenderConfig")
        if not callable(frame_supplier):
            raise TypeError("frame_supplier must be callable")
        self.config = config
        self.pipeline_description = build_appsrc_rtp_jpeg_sender_pipeline_description(
            host=config.host,
            port=config.port,
            fps=config.fps,
        )
        self._frame_supplier = frame_supplier
        self._clock = clock
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._pipeline: Any | None = None
        self._appsrc: Any | None = None
        self._bus: Any | None = None
        self._gst: Any | None = None
        self._failure: BaseException | None = None
        self._started = False

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
        Gst = _load_gst()
        missing = find_missing_gstreamer_elements(APPSRC_RTP_JPEG_SENDER_ELEMENTS)
        if missing:
            raise CameraSourceError(
                "missing GStreamer sender elements: " + ", ".join(missing)
            )
        try:
            pipeline = Gst.parse_launch(self.pipeline_description)
        except Exception as exc:
            raise CameraSourceError(
                f"cannot create synthetic RTP/JPEG sender pipeline: {exc}"
            ) from exc
        appsrc = pipeline.get_by_name("source")
        if appsrc is None:
            pipeline.set_state(Gst.State.NULL)
            raise CameraSourceError(
                "synthetic RTP/JPEG sender pipeline does not expose appsrc 'source'"
            )
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise CameraSourceError(
                "synthetic RTP/JPEG sender failed to enter PLAYING"
            )

        self._stop_event.clear()
        with self._lock:
            self._gst = Gst
            self._pipeline = pipeline
            self._appsrc = appsrc
            self._bus = pipeline.get_bus()
            self._failure = None
            self._started = True
            self._thread = Thread(
                target=self._run,
                name=f"synthetic-rtp-jpeg-sender-{self.config.port}",
                daemon=False,
            )
            thread = self._thread
        thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if timeout <= 0.0:
            raise ValueError("timeout must be > 0")
        with self._lock:
            thread = self._thread
            pipeline = self._pipeline
            appsrc = self._appsrc
            Gst = self._gst
            was_started = self._started
            self._started = False
        if not was_started and pipeline is None:
            return
        self._stop_event.set()
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                self._record_failure(
                    CameraSourceError(
                        f"synthetic RTP/JPEG sender did not stop within {timeout:.3f} s"
                    )
                )
        if appsrc is not None:
            # EOS is best-effort during teardown; setting the pipeline to NULL
            # below is the authoritative shutdown boundary.
            with suppress(Exception):
                appsrc.emit("end-of-stream")
        if pipeline is not None and Gst is not None:
            pipeline.set_state(Gst.State.NULL)
        with self._lock:
            self._thread = None
            self._pipeline = None
            self._appsrc = None
            self._bus = None
            self._gst = None

    def _run(self) -> None:
        interval_s = 1.0 / self.config.fps
        interval_ns = round(1_000_000_000 / self.config.fps)
        deadline_s = self._clock()
        last_capture_id: int | None = None
        while not self._stop_event.is_set():
            now_s = self._clock()
            remaining_s = deadline_s - now_s
            if remaining_s > 0.0 and self._stop_event.wait(remaining_s):
                return
            deadline_s += interval_s
            try:
                self._raise_bus_failure_if_any()
                frame = self._frame_supplier()
                if not isinstance(frame, AppsrcRtpJpegFrame):
                    raise CameraSourceError(
                        "synthetic RTP/JPEG frame supplier returned invalid frame"
                    )
                if frame.capture_id is not None and frame.capture_id == last_capture_id:
                    continue
                image = frame.image
                if (
                    not isinstance(image, np.ndarray)
                    or image.dtype != np.uint8
                    or image.ndim != 3
                    or image.shape[2] != 3
                ):
                    raise CameraSourceError(
                        "synthetic RTP/JPEG sender requires BGR uint8 HxWx3 frames"
                    )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )
                if not ok:
                    raise CameraSourceError("OpenCV failed to encode synthetic JPEG")
                payload = encoded.tobytes()
                with self._lock:
                    Gst = self._gst
                    appsrc = self._appsrc
                    if not self._started or Gst is None or appsrc is None:
                        return
                buffer = Gst.Buffer.new_allocate(None, len(payload), None)
                buffer.fill(0, payload)
                buffer.pts = frame.pts_ns
                buffer.dts = Gst.CLOCK_TIME_NONE
                buffer.duration = interval_ns
                flow = appsrc.emit("push-buffer", buffer)
                if flow != Gst.FlowReturn.OK:
                    raise CameraSourceError(
                        f"synthetic RTP/JPEG appsrc push failed: {flow}"
                    )
                last_capture_id = frame.capture_id
            except Exception as exc:  # noqa: BLE001
                # Thread boundary: surface any supplier/GStreamer failure through
                # the camera-source failure channel instead of losing the thread.
                self._record_failure(exc)
                return

    def _raise_bus_failure_if_any(self) -> None:
        with self._lock:
            bus = self._bus
            Gst = self._gst
        if bus is None or Gst is None:
            return
        message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message is None:
            return
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = str(error)
            if debug:
                detail = f"{detail}; {debug}"
            raise CameraSourceError(f"synthetic RTP/JPEG sender failed: {detail}")
        if not self._stop_event.is_set():
            raise CameraSourceError("synthetic RTP/JPEG sender reached unexpected EOS")

    def _record_failure(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error


__all__ = [
    "APPSRC_RTP_JPEG_SENDER_ELEMENTS",
    "AppsrcRtpJpegFrame",
    "AppsrcRtpJpegSender",
    "AppsrcRtpJpegSenderConfig",
    "build_appsrc_rtp_jpeg_sender_pipeline_description",
    "check_appsrc_rtp_jpeg_sender_runtime",
]
