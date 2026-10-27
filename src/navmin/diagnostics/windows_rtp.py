"""Windows synthetic RTP/JPEG sender using in-process GStreamer appsrc.

No fdsrc, inherited file descriptor, subprocess, or physical hardware needed.
"""

from __future__ import annotations

from threading import Event, Thread
from time import monotonic

import cv2

from navmin.diagnostics.localhost_rtp import (
    PreflightCheck,
    PreflightResult,
    RtpJpegSenderConfig,
    SenderProcessError,
    SyntheticDiagnosticFrameGenerator,
)


def _gst():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def check_windows_sender() -> PreflightResult:
    try:
        gst = _gst()
        missing = [
            name for name in ("appsrc", "jpegparse", "rtpjpegpay", "udpsink")
            if gst.ElementFactory.find(name) is None
        ]
        return PreflightResult((PreflightCheck(
            "in-process GStreamer appsrc sender",
            not missing,
            "missing: " + ", ".join(missing) if missing else "available",
        ),))
    except (ImportError, ValueError, RuntimeError) as exc:
        return PreflightResult((PreflightCheck("in-process sender", False, str(exc)),))


class WindowsRtpJpegSender:
    """Push bounded-rate diagnostic JPEGs into GStreamer without fdsrc."""

    def __init__(self, config: RtpJpegSenderConfig, *, frame_generator=None) -> None:
        self.config = config
        self._generator = frame_generator or SyntheticDiagnosticFrameGenerator(
            camera=config.camera, width=config.width, height=config.height,
        )
        self._pipeline = None
        self._thread: Thread | None = None
        self._stop = Event()
        self._error: Exception | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._error is None

    @property
    def frames_generated(self) -> int:
        return self._generator.frames_generated

    def start(self) -> None:
        if self._pipeline is not None:
            raise SenderProcessError("Windows sender already started")
        try:
            gst = _gst()
            c = self.config
            desc = (
                'appsrc name=diagnostic_src is-live=true format=time block=true '
                f'caps="image/jpeg,framerate={c.fps}/1,width={c.width},height={c.height}" '
                '! jpegparse ! rtpjpegpay pt=26 '
                f'! udpsink host={c.host} port={c.port} sync=false async=false'
            )
            pipeline = gst.parse_launch(desc)
            src = pipeline.get_by_name("diagnostic_src")
            if src is None or pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
                pipeline.set_state(gst.State.NULL)
                raise SenderProcessError("GStreamer appsrc pipeline refused PLAYING")
            self._pipeline = pipeline
            self._stop.clear()
            self._error = None
            self._thread = Thread(target=self._write, args=(gst, src),
                                  name=f"windows-rtp-sender-{c.port}", daemon=False)
            self._thread.start()
        except Exception as exc:
            self.stop()
            raise SenderProcessError(f"Windows sender startup failed: {exc}") from exc

    def _write(self, gst, source) -> None:
        period = 1.0 / self.config.fps
        deadline = monotonic()
        try:
            while not self._stop.is_set():
                ok, encoded = cv2.imencode(".jpg", self._generator.next_frame(),
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise SenderProcessError("JPEG encoding failed")
                data = encoded.tobytes()
                buffer = gst.Buffer.new_allocate(None, len(data), None)
                buffer.fill(0, data)
                frame_index = self._generator.frames_generated - 1
                buffer.pts = frame_index * gst.SECOND // self.config.fps
                buffer.duration = gst.SECOND // self.config.fps
                result = source.emit("push-buffer", buffer)
                if result != gst.FlowReturn.OK:
                    if not self._stop.is_set():
                        raise SenderProcessError(f"GStreamer push-buffer returned {result}")
                    break
                deadline += period
                self._stop.wait(max(0.0, deadline - monotonic()))
        except (SenderProcessError, cv2.error, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._error = exc

    def stop(self) -> None:
        self._stop.set()
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                gst = _gst()
                pipeline.set_state(gst.State.NULL)
            finally:
                if self._thread is not None:
                    self._thread.join(timeout=3.0)
                    if self._thread.is_alive():
                        raise SenderProcessError("Windows sender thread did not stop")
                self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise SenderProcessError(f"Windows sender failed: {error}")
