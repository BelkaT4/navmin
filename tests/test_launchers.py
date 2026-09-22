from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from navmin.__main__ import main as normal_main
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
    SerialConfig,
    StereoDistanceConfig,
    Stm32Config,
    TurretConfig,
    UiConfig,
    VisionConfig,
    VisionDistanceConfig,
)
from navmin.contracts import CameraRole, DistanceSource
from navmin.diagnostic_launcher import (
    CameraEndpoint,
    DiagnosticEndpoints,
    DiagnosticSelection,
    TurretEndpoint,
)
from navmin.diagnostic_launcher import main as diagnostic_main
from navmin.launcher import (
    LoadedApplicationInputs,
    run_loaded_application,
    session_file_logging,
    validate_normal_hardware_config,
)

WIDTH = 64
HEIGHT = 48
K = ((50.0, 0.0, 31.5), (0.0, 50.0, 23.5), (0.0, 0.0, 1.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _camera(port: int) -> CameraConfig:
    return CameraConfig(
        enabled=True,
        address="0.0.0.0",
        port=port,
        rtp_enabled=True,
        buffer_size=1,
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _config(*, emulate_stm32: bool = False) -> AppConfig:
    center = AimPointConfig(x_px=None, y_px=None)
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    return AppConfig(
        schema_version=1,
        vision=VisionConfig(
            processing_scope=ProcessingScope.MAIN_AND_PREVIEW,
            cameras=CamerasConfig(
                overview=_camera(8888),
                stereo_left=_camera(8889),
                stereo_right=_camera(8890),
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
                port="/dev/ttyUSB0",
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
            emulate_stm32=emulate_stm32,
        ),
        ui=UiConfig(
            default_camera=CameraRole.OVERVIEW,
            show_fps=False,
            show_stereo_right_diagnostics=False,
        ),
    )


def _overview() -> OverviewCalibration:
    return OverviewCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
        K=K,
        D=(0.0, 0.0, 0.0, 0.0),
        new_camera_matrix=K,
    )


def _stereo() -> StereoCalibration:
    projection = (
        (50.0, 0.0, 31.5, 0.0),
        (0.0, 50.0, 23.5, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    return StereoCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
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


def _inputs(*, emulate_stm32: bool = False) -> LoadedApplicationInputs:
    return LoadedApplicationInputs(_config(emulate_stm32=emulate_stm32), _overview(), _stereo())


def test_run_loaded_application_uses_shared_runtime_ui_surface_and_shutdown() -> None:
    events: list[str] = []
    inputs = _inputs()
    ui_dependencies = SimpleNamespace(
        mediator=object(),
        camera_bindings={CameraRole.OVERVIEW: object()},
        turret_states=object(),
        camera_stale_timeout_ms=321,
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.ui_dependencies = ui_dependencies

        def start(self) -> None:
            events.append("start")

        def shutdown(self) -> None:
            events.append("shutdown")

    def builder(**kwargs) -> FakeRuntime:
        assert kwargs == {
            "config": inputs.config,
            "overview_calibration": inputs.overview_calibration,
            "stereo_calibration": inputs.stereo_calibration,
        }
        return FakeRuntime()

    def ui_runner(**kwargs) -> int:
        events.append("ui")
        assert kwargs["mediator"] is ui_dependencies.mediator
        assert kwargs["camera_bindings"] is ui_dependencies.camera_bindings
        assert kwargs["turret_states"] is ui_dependencies.turret_states
        assert kwargs["camera_stale_timeout_ms"] == 321
        assert kwargs["argv"] == ("--qt-test",)
        return 7

    assert (
        run_loaded_application(
            inputs,
            qt_argv=("--qt-test",),
            runtime_builder=builder,
            ui_runner=ui_runner,
        )
        == 7
    )
    assert events == ["start", "ui", "shutdown"]


def test_normal_launcher_rejects_legacy_fake_turret_configuration() -> None:
    validate_normal_hardware_config(_config())
    with pytest.raises(ValueError, match="emulate-stm32=false"):
        validate_normal_hardware_config(_config(emulate_stm32=True))


def test_diagnostic_selection_overrides_only_requested_external_endpoints(monkeypatch) -> None:
    sender_configs = []
    sender_events: list[str] = []
    pty_events: list[str] = []

    class FakeSender:
        def __init__(self, config) -> None:
            sender_configs.append(config)

        def start(self) -> None:
            sender_events.append("start")

        def stop(self) -> None:
            sender_events.append("stop")

    class FakePty:
        stable_port_path = "/tmp/navmin-test-pty/stm32"

        def start(self) -> None:
            pty_events.append("start")

        def stop(self) -> None:
            pty_events.append("stop")

    monkeypatch.setattr("navmin.diagnostic_launcher.LocalhostRtpJpegSender", FakeSender)
    monkeypatch.setattr("navmin.diagnostic_launcher.PtyStm32Emulator", FakePty)

    endpoints = DiagnosticEndpoints(
        selection=DiagnosticSelection(
            overview=CameraEndpoint.LOCALHOST,
            stereo_left=CameraEndpoint.REAL,
            turret=TurretEndpoint.PTY,
        ),
        inputs=_inputs(emulate_stm32=True),
    )
    effective = endpoints.start()
    try:
        assert effective.config.vision.cameras.overview.address == "127.0.0.1"
        assert effective.config.vision.cameras.overview.port == 8888
        assert effective.config.vision.cameras.stereo_left.address == "0.0.0.0"
        assert effective.config.turret.serial.port == "/tmp/navmin-test-pty/stm32"
        assert effective.config.turret.emulate_stm32 is False
        assert len(sender_configs) == 1
        assert sender_configs[0].port == 8888
        assert sender_configs[0].width == WIDTH
        assert sender_configs[0].height == HEIGHT
        assert pty_events == ["start"]
    finally:
        endpoints.stop()

    assert sender_events == ["start", "stop"]
    assert pty_events == ["start", "stop"]


def test_diagnostic_real_selection_still_disables_fake_transport() -> None:
    inputs = _inputs(emulate_stm32=True)
    endpoints = DiagnosticEndpoints(
        selection=DiagnosticSelection(
            overview=CameraEndpoint.REAL,
            stereo_left=CameraEndpoint.REAL,
            turret=TurretEndpoint.REAL,
        ),
        inputs=inputs,
    )
    effective = endpoints.start()
    try:
        assert effective.config.turret.emulate_stm32 is False
        assert effective.config.turret.serial.port == inputs.config.turret.serial.port
        assert effective.config.vision.cameras == inputs.config.vision.cameras
    finally:
        endpoints.stop()


def test_session_file_logging_writes_diagnostic_detail(tmp_path) -> None:
    path = tmp_path / "logs" / "diagnostic.log"
    logger = logging.getLogger("navmin.launcher-test")

    with session_file_logging(path, level=logging.DEBUG):
        logger.debug("diagnostic-detail")

    assert "diagnostic-detail" in path.read_text(encoding="utf-8")


def test_diagnostic_synthetic_inputs_run_without_config_files(
    monkeypatch,
    tmp_path,
) -> None:
    captured_inputs: list[LoadedApplicationInputs] = []

    def fail_file_load(_paths):
        raise AssertionError("file-backed inputs must not be loaded")

    class FakeEndpoints:
        def __init__(self, *, selection, inputs) -> None:
            assert selection == DiagnosticSelection(
                overview=CameraEndpoint.LOCALHOST,
                stereo_left=CameraEndpoint.LOCALHOST,
                turret=TurretEndpoint.PTY,
            )
            captured_inputs.append(inputs)

        def start(self):
            base = captured_inputs[-1]
            effective = replace(
                base,
                config=replace(
                    base.config,
                    turret=replace(
                        base.config.turret,
                        serial=replace(
                            base.config.turret.serial,
                            port="/tmp/navmin-test-pty/stm32",
                        ),
                        emulate_stm32=False,
                    ),
                ),
            )
            return effective

        def stop(self) -> None:
            return None

    def fake_run(inputs) -> int:
        captured_inputs.append(inputs)
        return 0

    monkeypatch.setattr(
        "navmin.diagnostic_launcher.load_application_inputs",
        fail_file_load,
    )
    monkeypatch.setattr(
        "navmin.diagnostic_launcher.DiagnosticEndpoints",
        FakeEndpoints,
    )
    monkeypatch.setattr(
        "navmin.diagnostic_launcher.run_loaded_application",
        fake_run,
    )

    assert diagnostic_main(
        [
            "--overview",
            "localhost",
            "--stereo-left",
            "localhost",
            "--turret",
            "pty",
            "--synthetic-inputs",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    ) == 0

    inputs = captured_inputs[0]
    assert inputs.overview_calibration.image_width == 320
    assert inputs.overview_calibration.image_height == 240
    assert inputs.stereo_calibration.image_width == 320
    assert inputs.stereo_calibration.image_height == 240
    assert inputs.config.turret.emulate_stm32 is False
    assert inputs.config.vision.cameras.overview.port == 8888
    assert inputs.config.vision.cameras.stereo_left.port == 8889

    session = _single_session(tmp_path / "logs")
    effective_config = json.loads(
        (session / "inputs" / "effective-config.json").read_text(encoding="utf-8")
    )
    assert effective_config["turret"]["serial"]["port"] == "/tmp/navmin-test-pty/stm32"
    assert effective_config["turret"]["emulate-stm32"] is False
    assert (session / "inputs" / "overview-calibration.json").is_file()
    assert (session / "inputs" / "stereo-calibration.json").is_file()
    source_hashes = json.loads(
        (session / "inputs" / "source-hashes.json").read_text(encoding="utf-8")
    )
    assert all(item["source"] == "synthetic" for item in source_hashes.values())
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["overview_backend"] == "localhost"
    assert manifest["stereo_left_backend"] == "localhost"
    assert manifest["turret_backend"] == "pty"
    assert manifest["input_mode"] == "synthetic"


def test_diagnostic_synthetic_inputs_reject_mixed_real_endpoints(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        diagnostic_main(
            [
                "--overview",
                "real",
                "--stereo-left",
                "localhost",
                "--turret",
                "pty",
                "--synthetic-inputs",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
    assert exc_info.value.code == 2


def _single_session(log_dir: Path) -> Path:
    sessions = [path for path in log_dir.iterdir() if path.is_dir()]
    assert len(sessions) == 1
    return sessions[0]


def _write_source_placeholders(tmp_path: Path) -> tuple[Path, Path, Path]:
    config_path = tmp_path / "config.json"
    overview_path = tmp_path / "overview.json"
    stereo_path = tmp_path / "stereo.json"
    config_path.write_text("{}", encoding="utf-8")
    overview_path.write_text("{}", encoding="utf-8")
    stereo_path.write_text("{}", encoding="utf-8")
    return config_path, overview_path, stereo_path


def test_normal_missing_config_keeps_input_failure_session_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs: pytest.fail("runtime must not start after input failure"),
    )
    log_dir = tmp_path / "logs"
    exit_code = normal_main(
        [
            "--config",
            str(tmp_path / "missing-config.json"),
            "--overview-calibration",
            str(tmp_path / "missing-overview.json"),
            "--stereo-calibration",
            str(tmp_path / "missing-stereo.json"),
            "--log-dir",
            str(log_dir),
        ]
    )

    assert exit_code == 2
    session = _single_session(log_dir)
    assert (session / "runtime.log").is_file()
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "input-failed"
    assert manifest["exit_code"] == 2
    assert not (session / "inputs" / "effective-config.json").exists()


def test_normal_runtime_failure_and_clean_run_finalize_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    config_path, overview_path, stereo_path = _write_source_placeholders(tmp_path)
    inputs = _inputs()
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: inputs)

    def run_failure(_inputs) -> int:
        raise RuntimeError("runtime-boom")

    monkeypatch.setattr("navmin.__main__.run_loaded_application", run_failure)
    failed_logs = tmp_path / "failed-logs"
    assert normal_main(
        [
            "--config",
            str(config_path),
            "--overview-calibration",
            str(overview_path),
            "--stereo-calibration",
            str(stereo_path),
            "--log-dir",
            str(failed_logs),
        ]
    ) == 2
    failed_session = _single_session(failed_logs)
    failed_manifest = json.loads(
        (failed_session / "manifest.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["status"] == "runtime-failed"
    assert failed_manifest["failure_message"] == "runtime-boom"
    runtime_log = (failed_session / "runtime.log").read_text(encoding="utf-8")
    assert "runtime-boom" in runtime_log
    assert "Traceback (most recent call last)" in runtime_log

    monkeypatch.setattr("navmin.__main__.run_loaded_application", lambda _inputs: 0)
    clean_logs = tmp_path / "clean-logs"
    assert normal_main(
        [
            "--config",
            str(config_path),
            "--overview-calibration",
            str(overview_path),
            "--stereo-calibration",
            str(stereo_path),
            "--log-dir",
            str(clean_logs),
        ]
    ) == 0
    clean_manifest = json.loads(
        (_single_session(clean_logs) / "manifest.json").read_text(encoding="utf-8")
    )
    assert clean_manifest["status"] == "completed"
    assert clean_manifest["exit_code"] == 0


def test_diagnostic_cleanup_failure_preserves_primary_runtime_failure(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeEndpoints:
        def __init__(self, *, selection, inputs) -> None:
            self.inputs = inputs

        def start(self):
            return self.inputs

        def stop(self) -> None:
            raise RuntimeError("cleanup-boom")

    monkeypatch.setattr("navmin.diagnostic_launcher.DiagnosticEndpoints", FakeEndpoints)

    def fail_runtime(_inputs) -> int:
        raise RuntimeError("runtime-boom")

    monkeypatch.setattr("navmin.diagnostic_launcher.run_loaded_application", fail_runtime)
    log_dir = tmp_path / "logs"
    assert diagnostic_main(
        [
            "--overview",
            "localhost",
            "--stereo-left",
            "localhost",
            "--turret",
            "pty",
            "--synthetic-inputs",
            "--log-dir",
            str(log_dir),
        ]
    ) == 2

    manifest = json.loads(
        (_single_session(log_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "cleanup-failed"
    assert manifest["failure_message"] == "runtime-boom"
    assert manifest["cleanup_failure_message"] == "cleanup-boom"
