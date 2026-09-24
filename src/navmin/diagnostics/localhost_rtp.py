"""Diagnostics-only localhost RTP/JPEG sender and runtime preflight."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from tempfile import TemporaryFile
from threading import Event, Thread
from time import monotonic, sleep
from typing import IO, Any, Self

import cv2
import numpy as np

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config.models import CameraConfig, RtpJpegSourceConfig
from navmin.contracts import CameraRole
from navmin.vision.gstreamer_source import (
    GStreamerUnavailableError,
    initialize_gstreamer_runtime,
)

GST_RECEIVER_ELEMENTS = (
    "udpsrc",
    "rtpjpegdepay",
    "jpegdec",
    "videoconvert",
    "appsink",
)
GST_SENDER_ELEMENTS = (
    "fdsrc",
    "jpegparse",
    "rtpjpegpay",
    "udpsink",
)

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 20

_TARGET_RADIUS_PX = 6
_TARGET_SPEED_PX_PER_FRAME = 2
_OVERVIEW_BACKGROUND_BGR = (32, 48, 64)
_STEREO_LEFT_BACKGROUND_BGR = (70, 44, 30)
_TARGET_BGR = (245, 245, 245)
_TEXT_BGR = (245, 245, 245)
_FOOTER_TOP_RATIO = 0.85

_IDENTITY_3X3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_IDENTITY_4X4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@contextmanager
def _temporary_stderr_file() -> Iterator[IO[str]]:
    with TemporaryFile(mode="w+t", encoding="utf-8") as stderr:
        yield stderr


class SenderProcessError(RuntimeError):
    """The synthetic GStreamer sender could not be started or stopped."""


@dataclass(frozen=True)
class RtpJpegSenderConfig:
    """One deterministic synthetic RTP/JPEG sender endpoint."""

    port: int
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    host: str = "127.0.0.1"
    camera: CameraRole = CameraRole.OVERVIEW

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ValueError("port must be an integer in range 1..65535")
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("fps", self.fps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.host != "127.0.0.1":
            raise ValueError("localhost diagnostic sender host must be 127.0.0.1")
        if self.camera not in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT):
            raise ValueError("camera must be Overview or Stereo Left")


def _ping_pong_coordinate(frame_index: int, start: int, end: int) -> int:
    span = end - start
    phase = (frame_index * _TARGET_SPEED_PX_PER_FRAME) % (2 * span)
    return start + (phase if phase <= span else 2 * span - phase)


def _source_timestamp(value: datetime) -> str:
    milliseconds = value.microsecond // 1000
    return f"SOURCE {value:%H:%M:%S}.{milliseconds:03d}"


class SyntheticDiagnosticFrameGenerator:
    """Render one detector-friendly source frame before transport encoding."""

    def __init__(
        self,
        *,
        camera: CameraRole,
        width: int,
        height: int,
        wall_clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        if camera not in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT):
            raise ValueError("camera must be Overview or Stereo Left")
        for name, value in (("width", width), ("height", height)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.camera = camera
        self.width = width
        self.height = height
        self._wall_clock = wall_clock
        self._frame_counter = 0

    @property
    def frames_generated(self) -> int:
        return self._frame_counter

    def target_center(self, frame_counter: int) -> tuple[int, int]:
        if self.camera is CameraRole.OVERVIEW:
            x_base = _ping_pong_coordinate(frame_counter, 190, 280)
            y_base = 82
        else:
            x_base = _ping_pong_coordinate(frame_counter, 40, 130)
            y_base = 170
        return (
            round(x_base * self.width / DEFAULT_WIDTH),
            round(y_base * self.height / DEFAULT_HEIGHT),
        )

    def next_frame(self) -> np.ndarray:
        frame_counter = self._frame_counter
        timestamp = self._wall_clock()
        background = (
            _OVERVIEW_BACKGROUND_BGR
            if self.camera is CameraRole.OVERVIEW
            else _STEREO_LEFT_BACKGROUND_BGR
        )
        frame = np.full((self.height, self.width, 3), background, dtype=np.uint8)
        scale = min(self.width / DEFAULT_WIDTH, self.height / DEFAULT_HEIGHT)
        radius = max(4, min(15, round(_TARGET_RADIUS_PX * scale)))
        cv2.circle(
            frame,
            self.target_center(frame_counter),
            radius,
            _TARGET_BGR,
            -1,
        )

        footer_top = round(self.height * _FOOTER_TOP_RATIO)
        footer_color = tuple(max(0, component // 3) for component in background)
        cv2.rectangle(
            frame,
            (0, footer_top),
            (self.width - 1, self.height - 1),
            footer_color,
            -1,
        )
        font_scale = max(0.35, min(1.2, scale * 0.38))
        thickness = max(1, round(scale))
        first_baseline = round(self.height * 0.91)
        second_baseline = round(self.height * 0.975)
        camera_name = (
            "OVERVIEW"
            if self.camera is CameraRole.OVERVIEW
            else "STEREO LEFT"
        )
        cv2.putText(
            frame,
            _source_timestamp(timestamp),
            (max(2, round(4 * scale)), first_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _TEXT_BGR,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FRAME {frame_counter}  {camera_name}",
            (max(2, round(4 * scale)), second_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _TEXT_BGR,
            thickness,
            cv2.LINE_AA,
        )
        self._frame_counter += 1
        return frame


def build_sender_command(
    config: RtpJpegSenderConfig,
    *,
    gst_launch: str = "gst-launch-1.0",
) -> tuple[str, ...]:
    """Build a shell-free JPEG stdin -> RTP/JPEG -> UDP command."""
    if not gst_launch:
        raise ValueError("gst_launch must not be empty")
    return (
        gst_launch,
        "-q",
        "fdsrc",
        "fd=0",
        "do-timestamp=true",
        "!",
        f"image/jpeg,framerate={config.fps}/1",
        "!",
        "jpegparse",
        "!",
        (
            "image/jpeg,"
            f"width={config.width},height={config.height},framerate={config.fps}/1"
        ),
        "!",
        "rtpjpegpay",
        "pt=26",
        "!",
        "udpsink",
        f"host={config.host}",
        f"port={config.port}",
        "sync=false",
        "async=false",
    )


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if not check.ok)


def _check_python_gstreamer() -> str | None:
    try:
        initialize_gstreamer_runtime()
    except GStreamerUnavailableError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _check_gstreamer_elements(
    elements: tuple[str, ...],
    *,
    inspect: str | None,
    runner: Callable[..., Any],
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    for element in elements:
        if inspect is None:
            checks.append(
                PreflightCheck(
                    name=f"GStreamer element {element}",
                    ok=False,
                    detail="gst-inspect-1.0 unavailable",
                )
            )
            continue
        try:
            completed = runner(
                [inspect, element],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(
                PreflightCheck(
                    name=f"GStreamer element {element}",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        ok = completed.returncode == 0
        detail = "available"
        if not ok:
            output = (completed.stderr or completed.stdout or "not available").strip()
            detail = output.splitlines()[-1] if output else "not available"
        checks.append(
            PreflightCheck(
                name=f"GStreamer element {element}",
                ok=ok,
                detail=detail,
            )
        )
    return checks


def check_gstreamer_sender_runtime(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> PreflightResult:
    """Check only diagnostics sender executables/elements, not receiver bindings."""
    checks: list[PreflightCheck] = []
    executable_paths: dict[str, str | None] = {}
    for executable in ("gst-launch-1.0", "gst-inspect-1.0"):
        path = which(executable)
        executable_paths[executable] = path
        checks.append(
            PreflightCheck(
                name=executable,
                ok=path is not None,
                detail=path or "not found in PATH",
            )
        )
    checks.extend(
        _check_gstreamer_elements(
            GST_SENDER_ELEMENTS,
            inspect=executable_paths["gst-inspect-1.0"],
            runner=runner,
        )
    )
    return PreflightResult(tuple(checks))


def check_gstreamer_runtime(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    python_binding_check: Callable[[], str | None] = _check_python_gstreamer,
) -> PreflightResult:
    """Check executables, Python bindings, and every required Gst element."""
    checks: list[PreflightCheck] = []
    executable_paths: dict[str, str | None] = {}
    for executable in ("gst-launch-1.0", "gst-inspect-1.0"):
        path = which(executable)
        executable_paths[executable] = path
        checks.append(
            PreflightCheck(
                name=executable,
                ok=path is not None,
                detail=path or "not found in PATH",
            )
        )

    binding_error = python_binding_check()
    checks.append(
        PreflightCheck(
            name="Python gi / Gst 1.0 / GstApp 1.0",
            ok=binding_error is None,
            detail="available" if binding_error is None else binding_error,
        )
    )

    checks.extend(
        _check_gstreamer_elements(
            (*GST_RECEIVER_ELEMENTS, *GST_SENDER_ELEMENTS),
            inspect=executable_paths["gst-inspect-1.0"],
            runner=runner,
        )
    )
    return PreflightResult(tuple(checks))


class LocalhostRtpJpegSender:
    """Own one bounded-lifecycle ``gst-launch-1.0`` sender subprocess."""

    def __init__(
        self,
        config: RtpJpegSenderConfig,
        *,
        gst_launch: str = "gst-launch-1.0",
        process_factory: Callable[..., Any] = subprocess.Popen,
        startup_probe_seconds: float = 0.25,
        terminate_timeout_seconds: float = 2.0,
        monotonic_clock: Callable[[], float] = monotonic,
        sleep_fn: Callable[[float], None] = sleep,
        wall_clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        if startup_probe_seconds < 0.0:
            raise ValueError("startup_probe_seconds must be >= 0")
        if terminate_timeout_seconds <= 0.0:
            raise ValueError("terminate_timeout_seconds must be > 0")
        self.config = config
        self.command = build_sender_command(config, gst_launch=gst_launch)
        self._process_factory = process_factory
        self._startup_probe_seconds = startup_probe_seconds
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep_fn
        self._frame_generator = SyntheticDiagnosticFrameGenerator(
            camera=config.camera,
            width=config.width,
            height=config.height,
            wall_clock=wall_clock,
        )
        self._process: Any | None = None
        self._stderr_context: AbstractContextManager[IO[str]] | None = None
        self._stderr: IO[str] | None = None
        self._writer_stop = Event()
        self._writer_thread: Thread | None = None
        self._writer_error: OSError | RuntimeError | None = None

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._writer_thread is not None
            and self._writer_thread.is_alive()
            and self._writer_error is None
        )

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.poll()

    @property
    def frames_generated(self) -> int:
        return self._frame_generator.frames_generated

    def start(self) -> None:
        if self._process is not None:
            raise SenderProcessError("sender instance has already been started")
        stderr_context = _temporary_stderr_file()
        try:
            self._stderr = stderr_context.__enter__()
        except OSError as exc:
            raise SenderProcessError(f"cannot create sender stderr capture: {exc}") from exc
        self._stderr_context = stderr_context
        try:
            self._process = self._process_factory(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                text=False,
                close_fds=True,
            )
        except OSError as exc:
            self._close_stderr()
            raise SenderProcessError(f"cannot start GStreamer sender: {exc}") from exc

        if self._process.stdin is None:
            self._close_stderr()
            raise SenderProcessError("GStreamer sender stdin pipe was not created")
        self._writer_thread = Thread(
            target=self._write_frames,
            name=f"localhost-rtp-sender-{self.config.port}",
            daemon=False,
        )
        self._writer_thread.start()

        deadline = self._monotonic_clock() + self._startup_probe_seconds
        while (
            self._monotonic_clock() < deadline
            and self._process.poll() is None
            and self._writer_error is None
        ):
            self._sleep(min(0.02, max(0.0, deadline - self._monotonic_clock())))
        returncode = self._process.poll()
        if returncode is not None or self._writer_error is not None:
            detail = self._read_stderr()
            writer_detail = (
                ""
                if self._writer_error is None
                else f"; frame writer: {self._writer_error}"
            )
            self._stop_writer()
            self._close_stderr()
            raise SenderProcessError(
                f"GStreamer sender failed during startup with code {returncode}"
                + (f": {detail}" if detail else "")
                + writer_detail
            )

    def stop(self) -> None:
        process = self._process
        if process is None:
            self._stop_writer()
            self._close_stderr()
            return
        error: SenderProcessError | None = None
        self._writer_stop.set()
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self._terminate_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=self._terminate_timeout_seconds)
                    except subprocess.TimeoutExpired as exc:
                        error = SenderProcessError(
                            "GStreamer sender did not exit after terminate and kill"
                        )
                        error.__cause__ = exc
        finally:
            writer_error = self._stop_writer()
            self._close_stderr()
        if error is None and writer_error is not None:
            error = writer_error
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def _read_stderr(self) -> str:
        if self._stderr is None:
            return ""
        self._stderr.flush()
        self._stderr.seek(0)
        return self._stderr.read().strip()

    def _write_frames(self) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        period_s = 1.0 / self.config.fps
        next_deadline = self._monotonic_clock()
        while not self._writer_stop.is_set():
            frame = self._frame_generator.next_frame()
            try:
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    (cv2.IMWRITE_JPEG_QUALITY, 85),
                )
                if not encoded:
                    raise RuntimeError("OpenCV JPEG encoder rejected synthetic frame")
                view = memoryview(jpeg).cast("B")
                while view and not self._writer_stop.is_set():
                    written = process.stdin.write(view)
                    if written is None or written <= 0:
                        raise RuntimeError("GStreamer sender stdin accepted no bytes")
                    view = view[written:]
                process.stdin.flush()
            except (cv2.error, BrokenPipeError, OSError, RuntimeError, ValueError) as exc:
                if not self._writer_stop.is_set():
                    self._writer_error = exc
                return
            next_deadline += period_s
            now = self._monotonic_clock()
            next_deadline = max(next_deadline, now)
            if self._writer_stop.wait(max(0.0, next_deadline - now)):
                return

    def _stop_writer(self) -> SenderProcessError | None:
        self._writer_stop.set()
        writer = self._writer_thread
        if writer is not None:
            writer.join(self._terminate_timeout_seconds)
        process = self._process
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if writer is not None and writer.is_alive():
            return SenderProcessError("sender frame writer did not stop bounded")
        if self._writer_error is not None:
            return SenderProcessError(f"sender frame writer failed: {self._writer_error}")
        return None

    def _close_stderr(self) -> None:
        stderr_context = self._stderr_context
        self._stderr_context = None
        self._stderr = None
        if stderr_context is not None:
            stderr_context.__exit__(None, None, None)


def diagnostic_camera_config(port: int) -> CameraConfig:
    """Build an owner-local CameraConfig for the production receiver."""
    RtpJpegSenderConfig(port=port)
    return CameraConfig(
        enabled=True,
        source=RtpJpegSourceConfig(
            bind_address="127.0.0.1",
            port=port,
            buffer_size=1,
        ),
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _camera_matrix(
    width: int,
    height: int,
    *,
    focal_length: float,
) -> tuple[tuple[float, ...], ...]:
    return (
        (focal_length, 0.0, (width - 1) / 2.0),
        (0.0, focal_length, (height - 1) / 2.0),
        (0.0, 0.0, 1.0),
    )


def diagnostic_overview_calibration(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> OverviewCalibration:
    """Build exact-size zero-distortion fisheye calibration for diagnostics."""
    RtpJpegSenderConfig(port=1, width=width, height=height)
    matrix = _camera_matrix(
        width,
        height,
        focal_length=1000.0 * width / DEFAULT_WIDTH,
    )
    return OverviewCalibration(
        schema_version=1,
        image_width=width,
        image_height=height,
        K=matrix,
        D=(0.0, 0.0, 0.0, 0.0),
        new_camera_matrix=matrix,
    )


def diagnostic_stereo_calibration(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> StereoCalibration:
    """Build exact-size zero-distortion Stereo Left calibration for diagnostics."""
    RtpJpegSenderConfig(port=1, width=width, height=height)
    matrix = _camera_matrix(
        width,
        height,
        focal_length=250.0 * width / DEFAULT_WIDTH,
    )
    projection = (
        (matrix[0][0], 0.0, matrix[0][2], 0.0),
        (0.0, matrix[1][1], matrix[1][2], 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    return StereoCalibration(
        schema_version=1,
        image_width=width,
        image_height=height,
        K_left=matrix,
        D_left=(0.0, 0.0, 0.0, 0.0, 0.0),
        K_right=matrix,
        D_right=(0.0, 0.0, 0.0, 0.0, 0.0),
        R=_IDENTITY_3X3,
        T=(-0.46, 0.0, 0.0),
        R1=_IDENTITY_3X3,
        R2=_IDENTITY_3X3,
        P1=projection,
        P2=projection,
        Q=_IDENTITY_4X4,
    )


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "GST_RECEIVER_ELEMENTS",
    "GST_SENDER_ELEMENTS",
    "LocalhostRtpJpegSender",
    "PreflightCheck",
    "PreflightResult",
    "RtpJpegSenderConfig",
    "SenderProcessError",
    "SyntheticDiagnosticFrameGenerator",
    "build_sender_command",
    "check_gstreamer_runtime",
    "check_gstreamer_sender_runtime",
    "diagnostic_camera_config",
    "diagnostic_overview_calibration",
    "diagnostic_stereo_calibration",
]
