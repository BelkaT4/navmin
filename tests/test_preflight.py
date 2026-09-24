from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

import navmin.preflight as preflight_module
from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config.models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    AppConfig,
    AxesConfig,
    AxisMechanicsConfig,
    CameraConfig,
    CamerasConfig,
    PidControllerConfig,
    ProcessingScope,
    RtpJpegSourceConfig,
    RtspDecoderMode,
    RtspProtocol,
    RtspSourceConfig,
    SerialConfig,
    StereoDistanceConfig,
    Stm32Config,
    TurretConfig,
    UiConfig,
    VisionConfig,
    VisionDistanceConfig,
)
from navmin.contracts import CameraRole, DistanceSource
from navmin.diagnostics.localhost_rtp import PreflightCheck, PreflightResult
from navmin.diagnostics.pty_stm32 import PtyPreflightResult
from navmin.launcher import LoadedApplicationInputs
from navmin.preflight import (
    PreflightStatus,
    StartupBackendSelection,
    StartupPreflightCheck,
    StartupPreflightReport,
    format_preflight_summary,
    run_startup_preflight,
)

K = ((50.0, 0.0, 31.5), (0.0, 50.0, 23.5), (0.0, 0.0, 1.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _camera(port: int, *, address: str = "127.0.0.1") -> CameraConfig:
    return CameraConfig(
        enabled=True,
        source=RtpJpegSourceConfig(
            bind_address=address,
            port=port,
            buffer_size=1,
        ),
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _rtsp_camera(
    uri: str = "rtsp://camera.local/stream",
    *,
    protocol: RtspProtocol = RtspProtocol.TCP,
) -> CameraConfig:
    return CameraConfig(
        enabled=True,
        source=RtspSourceConfig(
            uri=uri,
            protocol=protocol,
            decoder_mode=RtspDecoderMode.SOFTWARE,
            latency_ms=100,
            drop_on_latency=True,
            buffer_size=1,
        ),
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _inputs(
    overview_port: int,
    stereo_port: int,
    *,
    serial_port: str = "/definitely/missing/ttyUSB0",
) -> LoadedApplicationInputs:
    center = AimPointConfig(x_px=None, y_px=None)
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    config = AppConfig(
        schema_version=1,
        vision=VisionConfig(
            processing_scope=ProcessingScope.MAIN_AND_PREVIEW,
            cameras=CamerasConfig(
                overview=_camera(overview_port),
                stereo_left=_camera(stereo_port),
                stereo_right=_camera(stereo_port + 1),
            ),
            distance=VisionDistanceConfig(
                source=DistanceSource.MANUAL,
                manual_distance_m=100.0,
                distance_stale_timeout_ms=300,
                stereo=StereoDistanceConfig(
                    stereo_enabled=False,
                    right_frame_buffer_size=4,
                    pair_timeout_ms=100,
                ),
            ),
            camera_stale_timeout_ms=500,
            simulation_mode=False,
        ),
        aiming=AimingConfig(
            lead_time_ms=100,
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(overview=center, stereo_left=center),
        ),
        turret=TurretConfig(
            serial=SerialConfig(
                port=serial_port,
                baudrate=9600,
                response_timeout_ms=50,
                max_retries=1,
                inter_request_delay_ms=1,
            ),
            axes=AxesConfig(x=axis, y=axis),
            controller=PidControllerConfig(
                pid_kp_x=1.0,
                pid_ki_x=0.0,
                pid_kd_x=0.0,
                pid_kp_y=1.0,
                pid_ki_y=0.0,
                pid_kd_y=0.0,
            ),
            stm32=Stm32Config(
                max_speed_x_deg_s=50.0,
                max_speed_y_deg_s=50.0,
                acceleration_x_deg_s2=100.0,
                acceleration_y_deg_s2=100.0,
                velocity_watchdog_timeout_ms=200,
            ),
            emulate_stm32=False,
        ),
        ui=UiConfig(
            default_camera=CameraRole.OVERVIEW,
            show_fps=False,
            show_stereo_right_diagnostics=False,
        ),
    )
    overview = OverviewCalibration(
        schema_version=1,
        image_width=64,
        image_height=48,
        K=K,
        D=(0.0, 0.0, 0.0, 0.0),
        new_camera_matrix=K,
    )
    projection = (
        (50.0, 0.0, 31.5, 0.0),
        (0.0, 50.0, 23.5, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    stereo = StereoCalibration(
        schema_version=1,
        image_width=64,
        image_height=48,
        K_left=K,
        D_left=(0.0, 0.0, 0.0, 0.0, 0.0),
        K_right=K,
        D_right=(0.0, 0.0, 0.0, 0.0, 0.0),
        R=IDENTITY,
        T=(-0.46, 0.0, 0.0),
        R1=IDENTITY,
        R2=IDENTITY,
        P1=projection,
        P2=projection,
        Q=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    return LoadedApplicationInputs(config, overview, stereo)


def _sender_result(ok: bool = True) -> PreflightResult:
    return PreflightResult(
        (
            PreflightCheck(
                name="sender",
                ok=ok,
                detail="available" if ok else "missing",
            ),
        )
    )


def _run(
    tmp_path: Path,
    inputs: LoadedApplicationInputs,
    selection: StartupBackendSelection,
    **overrides,
) -> StartupPreflightReport:
    defaults = {
        "session_directory_check": lambda _path: None,
        "qt_check": lambda: None,
        "receiver_check": lambda _elements: None,
        "sender_check": lambda: _sender_result(),
        "pyserial_check": lambda: None,
        "pty_check": lambda: PtyPreflightResult(True, True, True),
        "udp_check": lambda _address, _port: None,
        "serial_path_check": lambda _port: None,
    }
    defaults.update(overrides)
    return run_startup_preflight(
        inputs,
        selection=selection,
        session_dir=tmp_path,
        **defaults,
    )


@pytest.mark.parametrize(
    ("statuses", "expected", "ok"),
    [
        ([PreflightStatus.PASS], PreflightStatus.PASS, True),
        ([PreflightStatus.PASS, PreflightStatus.WARN], PreflightStatus.WARN, True),
        ([PreflightStatus.WARN, PreflightStatus.FAIL], PreflightStatus.FAIL, False),
    ],
)
def test_report_aggregation(statuses, expected, ok) -> None:
    report = StartupPreflightReport(
        tuple(
            StartupPreflightCheck(f"check-{index}", status, "detail")
            for index, status in enumerate(statuses)
        )
    )

    assert report.overall_status is expected
    assert report.ok is ok


def test_report_json_serialization_and_console_summary() -> None:
    report = StartupPreflightReport(
        (
            StartupPreflightCheck("Qt", PreflightStatus.PASS, "available"),
            StartupPreflightCheck("serial", PreflightStatus.FAIL, "missing"),
        )
    )

    encoded = json.dumps(report.to_mapping())
    decoded = json.loads(encoded)
    assert decoded == {
        "schema_version": 1,
        "overall_status": "fail",
        "checks": [
            {"name": "Qt", "status": "pass", "detail": "available"},
            {"name": "serial", "status": "fail", "detail": "missing"},
        ],
    }
    summary = format_preflight_summary(report)
    assert summary.startswith("PREFLIGHT FAIL")
    assert "serial" in summary
    assert "missing" in summary


def test_real_cameras_require_receiver_but_not_sender_tools(tmp_path) -> None:
    sender_calls = 0

    def sender_check():
        nonlocal sender_calls
        sender_calls += 1
        return _sender_result(False)

    report = _run(
        tmp_path,
        _inputs(18_881, 18_882),
        StartupBackendSelection("real", "real", "pty"),
        sender_check=sender_check,
    )

    assert report.ok
    assert sender_calls == 0
    assert any(check.name == "GStreamer receiver" for check in report.checks)
    assert all(check.name != "GStreamer localhost sender" for check in report.checks)


def test_localhost_camera_requires_receiver_and_sender_tools(tmp_path) -> None:
    calls: list[str] = []

    report = _run(
        tmp_path,
        _inputs(18_883, 18_884),
        StartupBackendSelection("localhost", "real", "pty"),
        receiver_check=lambda _elements: calls.append("receiver") or None,
        sender_check=lambda: calls.append("sender") or _sender_result(),
    )

    assert report.ok
    assert calls == ["receiver", "sender"]
    assert any(check.name == "GStreamer localhost sender" for check in report.checks)


def test_real_turret_checks_physical_path_without_pty(tmp_path) -> None:
    calls: list[str] = []

    def pty_must_not_run():
        raise AssertionError("PTY capability must not be checked for real turret")

    report = _run(
        tmp_path,
        _inputs(18_885, 18_886, serial_port="/dev/ttyTEST"),
        StartupBackendSelection("localhost", "localhost", "real"),
        pty_check=pty_must_not_run,
        serial_path_check=lambda port: calls.append(port) or None,
    )

    assert report.ok
    assert calls == ["/dev/ttyTEST"]
    assert any(check.name == "Serial device" for check in report.checks)
    assert all(check.name != "PTY capability" for check in report.checks)


def test_pty_turret_ignores_missing_physical_serial_path(tmp_path) -> None:
    def serial_path_must_not_run(_port):
        raise AssertionError("physical serial path must not be checked for PTY")

    report = _run(
        tmp_path,
        _inputs(18_887, 18_888, serial_port="/missing/physical/device"),
        StartupBackendSelection("localhost", "localhost", "pty"),
        serial_path_check=serial_path_must_not_run,
    )

    assert report.overall_status is PreflightStatus.PASS
    assert any(check.name == "PTY capability" for check in report.checks)
    assert all(check.name != "Serial device" for check in report.checks)


@pytest.mark.parametrize(
    ("overview", "stereo", "turret", "sender_expected", "pty_expected", "serial_expected"),
    [
        ("localhost", "localhost", "pty", True, True, False),
        ("real", "real", "pty", False, True, False),
        ("localhost", "localhost", "real", True, False, True),
        ("real", "real", "real", False, False, True),
        ("localhost", "real", "pty", True, True, False),
        ("real", "localhost", "pty", True, True, False),
    ],
)
def test_backend_matrix_has_no_cross_contamination(
    tmp_path,
    overview,
    stereo,
    turret,
    sender_expected,
    pty_expected,
    serial_expected,
) -> None:
    calls = {"sender": 0, "pty": 0, "serial": 0}

    def sender_check():
        calls["sender"] += 1
        return _sender_result()

    def pty_check():
        calls["pty"] += 1
        return PtyPreflightResult(True, True, True)

    def serial_check(_port):
        calls["serial"] += 1

    report = _run(
        tmp_path,
        _inputs(18_890, 18_891),
        StartupBackendSelection(overview, stereo, turret),
        sender_check=sender_check,
        pty_check=pty_check,
        serial_path_check=serial_check,
    )

    assert report.ok
    assert bool(calls["sender"]) is sender_expected
    assert bool(calls["pty"]) is pty_expected
    assert bool(calls["serial"]) is serial_expected


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def test_udp_available_ports_pass_and_probe_socket_is_closed(tmp_path) -> None:
    overview_port = _free_udp_port()
    stereo_port = _free_udp_port()
    while stereo_port == overview_port:
        stereo_port = _free_udp_port()
    inputs = _inputs(overview_port, stereo_port)

    report = _run(
        tmp_path,
        inputs,
        StartupBackendSelection("localhost", "localhost", "pty"),
        udp_check=preflight_module._check_udp_bind,
    )

    assert report.ok
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", overview_port))
    finally:
        probe.close()


def test_udp_occupied_port_fails(tmp_path) -> None:
    occupied = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupied.bind(("127.0.0.1", 0))
    occupied_port = int(occupied.getsockname()[1])
    stereo_port = _free_udp_port()
    try:
        report = _run(
            tmp_path,
            _inputs(occupied_port, stereo_port),
            StartupBackendSelection("localhost", "localhost", "pty"),
            udp_check=preflight_module._check_udp_bind,
        )
    finally:
        occupied.close()

    assert not report.ok
    failure = next(
        check for check in report.checks if check.name == "Overview UDP endpoint"
    )
    assert failure.status is PreflightStatus.FAIL
    assert "cannot bind" in failure.detail


def test_duplicate_camera_endpoint_fails_before_runtime(tmp_path) -> None:
    port = _free_udp_port()
    inputs = _inputs(port, port)

    report = _run(
        tmp_path,
        inputs,
        StartupBackendSelection("localhost", "localhost", "pty"),
    )

    assert not report.ok
    failure = next(
        check for check in report.checks if check.name == "Camera UDP endpoints"
    )
    assert "Overview and Stereo Left" in failure.detail
    assert f"127.0.0.1:{port}" in failure.detail


def test_mixed_rtsp_and_rtp_preflight_checks_only_relevant_dependencies(
    tmp_path,
) -> None:
    inputs = _inputs(18_892, 18_893)
    cameras = inputs.config.vision.cameras
    config = replace(
        inputs.config,
        vision=replace(
            inputs.config.vision,
            cameras=replace(
                cameras,
                overview=_rtsp_camera(protocol=RtspProtocol.UDP),
            ),
        ),
    )
    element_calls: list[tuple[str, ...]] = []
    udp_calls: list[tuple[str, int]] = []

    report = _run(
        tmp_path,
        replace(inputs, config=config),
        StartupBackendSelection("real", "real", "pty"),
        receiver_check=lambda elements: element_calls.append(elements) or None,
        udp_check=lambda address, port: udp_calls.append((address, port)) or None,
    )

    assert report.ok
    assert len(element_calls) == 1
    elements = element_calls[0]
    assert {
        "rtspsrc",
        "rtph264depay",
        "h264parse",
        "avdec_h264",
        "videoconvert",
        "appsink",
        "udpsrc",
        "rtpjpegdepay",
        "jpegdec",
    }.issubset(elements)
    assert udp_calls == [("127.0.0.1", 18_893)]
    assert any(check.name == "Overview RTSP configuration" for check in report.checks)
    assert any(
        check.name == "Stereo Left RTP/JPEG configuration" for check in report.checks
    )


def test_receiver_capability_probe_uses_production_initializer_and_factory_probe(
    monkeypatch,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(
        preflight_module,
        "initialize_gstreamer_runtime",
        lambda: events.append("initialize"),
    )
    monkeypatch.setattr(
        preflight_module,
        "find_missing_gstreamer_elements",
        lambda elements: events.append(elements) or (),
    )

    elements = ("rtspsrc", "rtph264depay", "avdec_h264")
    assert preflight_module._check_gstreamer_receiver(elements) is None
    assert events == ["initialize", elements]


def test_serial_device_check_requires_exists_readable_and_writable(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "ttyUSB0"
    path.write_bytes(b"")

    monkeypatch.setattr(
        preflight_module.os,
        "access",
        lambda _path, mode: mode != preflight_module.os.R_OK,
    )
    assert preflight_module._check_serial_device(str(path)) == f"{path} is not readable"

    monkeypatch.setattr(
        preflight_module.os,
        "access",
        lambda _path, mode: mode != preflight_module.os.W_OK,
    )
    assert preflight_module._check_serial_device(str(path)) == f"{path} is not writable"

    monkeypatch.setattr(preflight_module.os, "access", lambda _path, _mode: True)
    assert preflight_module._check_serial_device(str(path)) is None
    assert preflight_module._check_serial_device(str(tmp_path / "missing")) == (
        f"{tmp_path / 'missing'} does not exist"
    )


def test_session_directory_probe_is_removed(tmp_path) -> None:
    before = set(tmp_path.iterdir())
    assert preflight_module._check_session_directory(tmp_path) is None
    assert set(tmp_path.iterdir()) == before
