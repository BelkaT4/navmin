"""Diagnostics-only localhost RTP/JPEG sender and runtime preflight."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from tempfile import TemporaryFile
from time import monotonic, sleep
from typing import IO, Any, Self

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config.models import CameraConfig

GST_RECEIVER_ELEMENTS = (
    "udpsrc",
    "rtpjpegdepay",
    "jpegdec",
    "videoconvert",
    "appsink",
)
GST_SENDER_ELEMENTS = (
    "videotestsrc",
    "jpegenc",
    "jpegparse",
    "rtpjpegpay",
    "udpsink",
)

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 20

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
    pattern: str = "ball"

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
        if not self.pattern or any(character.isspace() for character in self.pattern):
            raise ValueError("pattern must be one non-empty GStreamer token")


def build_sender_command(
    config: RtpJpegSenderConfig,
    *,
    gst_launch: str = "gst-launch-1.0",
) -> tuple[str, ...]:
    """Build a shell-free videotestsrc -> RTP/JPEG -> UDP command."""
    if not gst_launch:
        raise ValueError("gst_launch must not be empty")
    return (
        gst_launch,
        "-q",
        "videotestsrc",
        "is-live=true",
        f"pattern={config.pattern}",
        "!",
        "videoconvert",
        "!",
        (
            "video/x-raw,format=I420,"
            f"width={config.width},height={config.height},framerate={config.fps}/1"
        ),
        "!",
        "jpegenc",
        "!",
        "jpegparse",
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
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp

        Gst.init(None)
        _ = GstApp.AppSink
    except (ImportError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


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

    inspect = executable_paths["gst-inspect-1.0"]
    for element in (*GST_RECEIVER_ELEMENTS, *GST_SENDER_ELEMENTS):
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
        self._process: Any | None = None
        self._stderr_context: AbstractContextManager[IO[str]] | None = None
        self._stderr: IO[str] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.poll()

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
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                text=True,
                close_fds=True,
            )
        except OSError as exc:
            self._close_stderr()
            raise SenderProcessError(f"cannot start GStreamer sender: {exc}") from exc

        deadline = self._monotonic_clock() + self._startup_probe_seconds
        while self._monotonic_clock() < deadline and self._process.poll() is None:
            self._sleep(min(0.02, max(0.0, deadline - self._monotonic_clock())))
        returncode = self._process.poll()
        if returncode is not None:
            detail = self._read_stderr()
            self._close_stderr()
            raise SenderProcessError(
                f"GStreamer sender exited during startup with code {returncode}"
                + (f": {detail}" if detail else "")
            )

    def stop(self) -> None:
        process = self._process
        if process is None:
            self._close_stderr()
            return
        error: SenderProcessError | None = None
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
            self._close_stderr()
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
        address="127.0.0.1",
        port=port,
        rtp_enabled=True,
        buffer_size=1,
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _camera_matrix(width: int, height: int) -> tuple[tuple[float, ...], ...]:
    focal_length = float(max(width, height))
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
    matrix = _camera_matrix(width, height)
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
    matrix = _camera_matrix(width, height)
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
    "build_sender_command",
    "check_gstreamer_runtime",
    "diagnostic_camera_config",
    "diagnostic_overview_calibration",
    "diagnostic_stereo_calibration",
]
