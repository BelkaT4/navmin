"""Typed launcher-side startup capability checks for selected NavMin backends."""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from navmin.config.models import (
    RtpJpegSourceConfig,
    RtspDecoderMode,
    RtspProtocol,
    RtspSourceConfig,
)
from navmin.diagnostics.localhost_rtp import (
    PreflightResult,
    check_gstreamer_sender_runtime,
)
from navmin.diagnostics.pty_stm32 import PtyPreflightResult, check_pty_preflight
from navmin.launcher import LoadedApplicationInputs
from navmin.vision.gstreamer_source import (
    RTP_JPEG_GSTREAMER_ELEMENTS,
    RTSP_H264_GSTREAMER_ELEMENTS,
    GStreamerUnavailableError,
    find_missing_gstreamer_elements,
    initialize_gstreamer_runtime,
)

_PREFLIGHT_SCHEMA_VERSION = 1
_CAMERA_BACKENDS = frozenset({"real", "localhost"})
_TURRET_BACKENDS = frozenset({"real", "pty"})


class PreflightStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class StartupPreflightCheck:
    name: str
    status: PreflightStatus
    detail: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class StartupPreflightReport:
    checks: tuple[StartupPreflightCheck, ...]

    @property
    def overall_status(self) -> PreflightStatus:
        statuses = {check.status for check in self.checks}
        if PreflightStatus.FAIL in statuses:
            return PreflightStatus.FAIL
        if PreflightStatus.WARN in statuses:
            return PreflightStatus.WARN
        return PreflightStatus.PASS

    @property
    def ok(self) -> bool:
        return self.overall_status is not PreflightStatus.FAIL

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": _PREFLIGHT_SCHEMA_VERSION,
            "overall_status": self.overall_status.value,
            "checks": [check.to_mapping() for check in self.checks],
        }


@dataclass(frozen=True)
class StartupBackendSelection:
    overview: str
    stereo_left: str
    turret: str

    def __post_init__(self) -> None:
        if self.overview not in _CAMERA_BACKENDS:
            raise ValueError(f"unsupported Overview backend: {self.overview}")
        if self.stereo_left not in _CAMERA_BACKENDS:
            raise ValueError(f"unsupported Stereo Left backend: {self.stereo_left}")
        if self.turret not in _TURRET_BACKENDS:
            raise ValueError(f"unsupported Turret backend: {self.turret}")


StringCheck = Callable[[], str | None]
ReceiverCheck = Callable[[tuple[str, ...]], str | None]
UdpCheck = Callable[[str, int], str | None]
SerialPathCheck = Callable[[str], str | None]
PtyCheck = Callable[[], PtyPreflightResult]
SenderCheck = Callable[[], PreflightResult]


def run_startup_preflight(
    inputs: LoadedApplicationInputs,
    *,
    selection: StartupBackendSelection,
    session_dir: Path,
    session_directory_check: Callable[[Path], str | None] | None = None,
    qt_check: StringCheck | None = None,
    receiver_check: ReceiverCheck | None = None,
    sender_check: SenderCheck | None = None,
    pyserial_check: StringCheck | None = None,
    pty_check: PtyCheck | None = None,
    udp_check: UdpCheck | None = None,
    serial_path_check: SerialPathCheck | None = None,
) -> StartupPreflightReport:
    """Run static checks required by exactly the selected startup backends."""
    session_directory_check = session_directory_check or _check_session_directory
    qt_check = qt_check or _check_pyqt6
    receiver_check = receiver_check or _check_gstreamer_receiver
    sender_check = sender_check or check_gstreamer_sender_runtime
    pyserial_check = pyserial_check or _check_pyserial
    pty_check = pty_check or check_pty_preflight
    udp_check = udp_check or _check_udp_bind
    serial_path_check = serial_path_check or _check_serial_device

    checks: list[StartupPreflightCheck] = []
    checks.append(_result_check("Session directory", session_directory_check(session_dir)))
    checks.append(_result_check("PyQt6 QtWidgets", qt_check()))

    camera_specs = (
        (
            "Overview",
            selection.overview,
            inputs.config.vision.cameras.overview,
            inputs.overview_calibration.image_width,
            inputs.overview_calibration.image_height,
        ),
        (
            "Stereo Left",
            selection.stereo_left,
            inputs.config.vision.cameras.stereo_left,
            inputs.stereo_calibration.image_width,
            inputs.stereo_calibration.image_height,
        ),
    )

    endpoints: list[tuple[str, str, int]] = []
    receiver_elements: list[str] = []
    localhost_selected = False
    real_camera_selected = False
    for label, backend, camera, width, height in camera_specs:
        localhost_selected = localhost_selected or backend == "localhost"
        real_camera_selected = real_camera_selected or backend == "real"
        checks.append(
            StartupPreflightCheck(
                name=f"{label} calibration",
                status=PreflightStatus.PASS,
                detail=f"typed calibration loaded ({width}x{height})",
            )
        )

        source = camera.source
        if isinstance(source, RtpJpegSourceConfig):
            receiver_elements.extend(RTP_JPEG_GSTREAMER_ELEMENTS)
            source_error = _rtp_jpeg_config_error(source)
            checks.append(
                _result_check(
                    f"{label} RTP/JPEG configuration",
                    source_error,
                    success_detail=(
                        f"bind={source.bind_address}:{source.port}; "
                        f"buffer-size={source.buffer_size}"
                    ),
                )
            )
            endpoints.append((label, source.bind_address, source.port))
        elif isinstance(source, RtspSourceConfig):
            receiver_elements.extend(RTSP_H264_GSTREAMER_ELEMENTS)
            source_error = _rtsp_config_error(source)
            checks.append(
                _result_check(
                    f"{label} RTSP configuration",
                    source_error,
                    success_detail=(
                        f"uri={source.uri}; H.264 {source.protocol.value}; "
                        f"decoder={source.decoder_mode.value}; "
                        f"latency={source.latency_ms} ms; "
                        f"drop-on-latency={source.drop_on_latency}"
                    ),
                )
            )
            if backend == "localhost":
                checks.append(
                    StartupPreflightCheck(
                        name=f"{label} localhost transport",
                        status=PreflightStatus.FAIL,
                        detail="localhost diagnostic sender requires rtp-jpeg",
                    )
                )
        else:
            checks.append(
                StartupPreflightCheck(
                    name=f"{label} camera source",
                    status=PreflightStatus.FAIL,
                    detail=f"unsupported source config: {type(source).__name__}",
                )
            )

    unique_receiver_elements = tuple(dict.fromkeys(receiver_elements))
    checks.append(
        _result_check(
            "GStreamer receiver",
            receiver_check(unique_receiver_elements),
            success_detail=(
                "required elements available: " + ", ".join(unique_receiver_elements)
            ),
        )
    )

    duplicate_detail = _duplicate_endpoint_detail(endpoints)
    if duplicate_detail is not None:
        checks.append(
            StartupPreflightCheck(
                name="Camera UDP endpoints",
                status=PreflightStatus.FAIL,
                detail=duplicate_detail,
            )
        )

    for label, address, port in endpoints:
        checks.append(
            _result_check(
                f"{label} UDP endpoint",
                udp_check(address, port),
                success_detail=f"bind available at {address}:{port}",
            )
        )

    if localhost_selected:
        sender_result = sender_check()
        if sender_result.ok:
            checks.append(
                StartupPreflightCheck(
                    name="GStreamer localhost sender",
                    status=PreflightStatus.PASS,
                    detail="gst-launch/gst-inspect and sender elements available",
                )
            )
        else:
            detail = "; ".join(
                f"{failure.name}: {failure.detail}"
                for failure in sender_result.failures
            ) or "sender runtime unavailable"
            checks.append(
                StartupPreflightCheck(
                    name="GStreamer localhost sender",
                    status=PreflightStatus.FAIL,
                    detail=detail,
                )
            )

    if real_camera_selected:
        checks.append(
            StartupPreflightCheck(
                name="Real camera reachability",
                status=PreflightStatus.WARN,
                detail=(
                    "static preflight does not probe RTP packets, RTSP endpoints, "
                    "or camera ONLINE"
                ),
            )
        )

    checks.append(_result_check("pyserial", pyserial_check()))
    if selection.turret == "real":
        checks.append(
            _result_check(
                "Serial device",
                serial_path_check(inputs.config.turret.serial.port),
                success_detail=f"accessible: {inputs.config.turret.serial.port}",
            )
        )
        checks.append(
            StartupPreflightCheck(
                name="STM32 protocol reachability",
                status=PreflightStatus.WARN,
                detail="static preflight does not open the serial device or query STM32",
            )
        )
    else:
        pty = pty_check()
        missing: list[str] = []
        if not pty.linux:
            missing.append("Linux required")
        if not pty.openpty_available:
            missing.append("os.openpty unavailable")
        if not pty.pyserial_available:
            missing.append("pyserial unavailable")
        checks.append(
            StartupPreflightCheck(
                name="PTY capability",
                status=PreflightStatus.PASS if not missing else PreflightStatus.FAIL,
                detail="Linux os.openpty available" if not missing else "; ".join(missing),
            )
        )

    return StartupPreflightReport(tuple(checks))


def format_preflight_summary(report: StartupPreflightReport) -> str:
    """Render a compact operator summary without dumping machine-readable JSON."""
    lines = [f"PREFLIGHT {report.overall_status.value.upper()}", ""]
    width = max((len(check.name) for check in report.checks), default=0)
    for check in report.checks:
        suffix = ""
        if check.status is not PreflightStatus.PASS:
            suffix = f" — {check.detail}"
        lines.append(f"  {check.name:<{width}}  {check.status.value.upper()}{suffix}")
    return "\n".join(lines)


def _result_check(
    name: str,
    error: str | None,
    *,
    success_detail: str = "available",
) -> StartupPreflightCheck:
    return StartupPreflightCheck(
        name=name,
        status=PreflightStatus.PASS if error is None else PreflightStatus.FAIL,
        detail=success_detail if error is None else error,
    )



def _rtp_jpeg_config_error(source: RtpJpegSourceConfig) -> str | None:
    if not source.bind_address:
        return "bind-address must be non-empty"
    if type(source.port) is not int or not 1 <= source.port <= 65_535:
        return "port must be an integer in range 1..65535"
    if type(source.buffer_size) is not int or source.buffer_size < 1:
        return "buffer-size must be an integer >= 1"
    return None


def _rtsp_config_error(source: RtspSourceConfig) -> str | None:
    try:
        parsed = urlsplit(source.uri)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        return f"invalid RTSP URI: {exc}"
    if parsed.scheme.lower() != "rtsp" or parsed.hostname is None:
        return "URI must use rtsp:// and include a host"
    if port is not None and not 1 <= port <= 65_535:
        return "RTSP URI port must be in range 1..65535"
    if source.protocol not in (RtspProtocol.TCP, RtspProtocol.UDP):
        return "protocol must be tcp or udp"
    if source.decoder_mode is not RtspDecoderMode.SOFTWARE:
        return "decoder-mode must be software"
    if type(source.latency_ms) is not int or source.latency_ms < 0:
        return "latency-ms must be an integer >= 0"
    if type(source.drop_on_latency) is not bool:
        return "drop-on-latency must be boolean"
    if type(source.buffer_size) is not int or source.buffer_size < 1:
        return "buffer-size must be an integer >= 1"
    return None

def _check_session_directory(path: Path) -> str | None:
    directory = Path(path)
    if not directory.is_dir():
        return f"not a directory: {directory}"
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=".preflight-write-",
            delete=False,
        ) as probe:
            probe.write(b"navmin-preflight")
            probe_path = Path(probe.name)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
    return None


def _check_pyqt6() -> str | None:
    try:
        from PyQt6 import QtWidgets

        _ = QtWidgets.QWidget
    except (AttributeError, ImportError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _check_gstreamer_receiver(element_names: tuple[str, ...]) -> str | None:
    try:
        initialize_gstreamer_runtime()
        missing = find_missing_gstreamer_elements(element_names)
    except GStreamerUnavailableError as exc:
        return str(exc)
    if missing:
        return "missing elements: " + ", ".join(missing)
    return None


def _check_pyserial() -> str | None:
    try:
        import serial

        if not callable(serial.Serial):
            return "serial.Serial is unavailable"
    except (AttributeError, ImportError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _check_udp_bind(address: str, port: int) -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((address, port))
    except OSError as exc:
        return f"cannot bind {address}:{port}: {type(exc).__name__}: {exc}"
    finally:
        sock.close()
    return None


def _duplicate_endpoint_detail(endpoints: list[tuple[str, str, int]]) -> str | None:
    seen: dict[tuple[str, int], str] = {}
    for label, address, port in endpoints:
        key = (address, port)
        previous = seen.get(key)
        if previous is not None:
            return f"{previous} and {label} both use {address}:{port}"
        seen[key] = label
    return None


def _check_serial_device(port: str) -> str | None:
    path = Path(port)
    if not path.exists():
        return f"{port} does not exist"
    if not os.access(path, os.R_OK):
        return f"{port} is not readable"
    if not os.access(path, os.W_OK):
        return f"{port} is not writable"
    return None


__all__ = [
    "PreflightStatus",
    "StartupBackendSelection",
    "StartupPreflightCheck",
    "StartupPreflightReport",
    "format_preflight_summary",
    "run_startup_preflight",
]
