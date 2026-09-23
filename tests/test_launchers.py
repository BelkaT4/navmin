from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from navmin.__main__ import main as normal_main
from navmin.calibration import (
    OverviewCalibration,
    StereoCalibration,
    load_overview_calibration,
    load_stereo_calibration,
)
from navmin.config import load_config, save_config
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
from navmin.input_recovery import InputRecoveryError, recover_default_inputs
from navmin.launcher import (
    ApplicationInputError,
    LauncherPaths,
    LoadedApplicationInputs,
    run_loaded_application,
    session_file_logging,
    validate_normal_hardware_config,
)
from navmin.preflight import (
    PreflightStatus,
    StartupPreflightCheck,
    StartupPreflightReport,
)

WIDTH = 64
HEIGHT = 48
K = ((50.0, 0.0, 31.5), (0.0, 50.0, 23.5), (0.0, 0.0, 1.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
TEST_LOCAL_TZ = timezone(timedelta(hours=7))
RECOVERY_TEST_NOW = datetime(2026, 9, 23, 10, 15, 30, tzinfo=TEST_LOCAL_TZ)


def _preflight_report(status: PreflightStatus = PreflightStatus.PASS) -> StartupPreflightReport:
    return StartupPreflightReport(
        (StartupPreflightCheck("test preflight", status, "deterministic test result"),)
    )


@pytest.fixture(autouse=True)
def _deterministic_launcher_preflight(monkeypatch) -> None:
    check = lambda *args, **kwargs: _preflight_report()
    monkeypatch.setattr("navmin.__main__.run_startup_preflight", check)
    monkeypatch.setattr("navmin.diagnostic_launcher.run_startup_preflight", check)
    monkeypatch.setattr("navmin.__main__._prompt_input_recovery", lambda _error: False)


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
        assert kwargs["show_diagnostic_clock"] is False
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
        assert sender_configs[0].camera is CameraRole.OVERVIEW
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

    def fake_run(inputs, *, show_diagnostic_clock: bool) -> int:
        assert show_diagnostic_clock
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


def _recovery_paths(tmp_path: Path) -> LauncherPaths:
    return LauncherPaths(
        config=tmp_path / "config.json",
        overview_calibration=tmp_path / "calibration" / "overview.json",
        stereo_calibration=tmp_path / "calibration" / "stereo.json",
        log_dir=tmp_path / "logs",
    )


def test_load_application_inputs_identifies_exact_failing_file(tmp_path) -> None:
    paths = _recovery_paths(tmp_path)
    paths.overview_calibration.parent.mkdir(parents=True)
    save_config(paths.config, _config())
    paths.overview_calibration.write_text("{broken", encoding="utf-8")

    from navmin.launcher import load_application_inputs

    with pytest.raises(ApplicationInputError) as exc_info:
        load_application_inputs(paths)

    assert exc_info.value.label == "overview calibration"
    assert exc_info.value.path == paths.overview_calibration
    assert str(paths.overview_calibration) in str(exc_info.value)
    assert "malformed JSON" in str(exc_info.value)


def test_input_recovery_creates_valid_safe_defaults_without_existing_files(
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    result = recover_default_inputs(paths, now=RECOVERY_TEST_NOW)

    assert result.timestamp == "20260923-101530"
    assert result.backups == ()
    assert result.restored_paths == (
        paths.config,
        paths.overview_calibration,
        paths.stereo_calibration,
    )

    config = load_config(paths.config)
    overview = load_overview_calibration(paths.overview_calibration)
    stereo = load_stereo_calibration(paths.stereo_calibration)
    assert config.turret.serial.port == "/dev/navmin-configure-serial-port"
    assert config.turret.controller.pid_kp_x == 0.0
    assert config.turret.controller.pid_kp_y == 0.0
    assert config.turret.axes.x.max_relative_move_deg == 1.0
    assert config.turret.axes.y.max_relative_move_deg == 1.0
    assert overview.image_width == 320
    assert overview.image_height == 240
    assert stereo.image_width == 320
    assert stereo.image_height == 240
    assert stereo.T == (0.0, 0.0, 0.0)


def test_input_recovery_uses_common_local_timestamp_and_collision_suffix(
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    paths.overview_calibration.parent.mkdir(parents=True)
    originals = {
        paths.config: b"broken-config",
        paths.overview_calibration: b"broken-overview",
        paths.stereo_calibration: b"broken-stereo",
    }
    for path, content in originals.items():
        path.write_bytes(content)

    existing_backup = tmp_path / "config.json-20260923-101530.bak"
    existing_backup.write_bytes(b"older-backup")

    result = recover_default_inputs(paths, now=RECOVERY_TEST_NOW)

    assert result.timestamp == "20260923-101530-01"
    assert existing_backup.read_bytes() == b"older-backup"
    assert {backup.name for _, backup in result.backups} == {
        "config.json-20260923-101530-01.bak",
        "overview.json-20260923-101530-01.bak",
        "stereo.json-20260923-101530-01.bak",
    }
    for source, backup in result.backups:
        assert backup.read_bytes() == originals[source]


def test_input_recovery_backup_failure_keeps_original_set_untouched(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    paths.overview_calibration.parent.mkdir(parents=True)
    originals = {
        paths.config: b"config-original",
        paths.overview_calibration: b"overview-original",
        paths.stereo_calibration: b"stereo-original",
    }
    for path, content in originals.items():
        path.write_bytes(content)

    from navmin import input_recovery

    real_backup = input_recovery._create_backup
    calls = 0

    def fail_second_backup(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InputRecoveryError("backup-failed")
        real_backup(source, destination)

    monkeypatch.setattr(input_recovery, "_create_backup", fail_second_backup)

    with pytest.raises(InputRecoveryError, match="backup-failed"):
        recover_default_inputs(paths, now=RECOVERY_TEST_NOW)

    for path, content in originals.items():
        assert path.read_bytes() == content
    assert not list(tmp_path.rglob("*.bak"))
    assert not list(tmp_path.rglob("*.recovery-*.tmp"))


def test_input_recovery_install_failure_rolls_back_original_set(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    paths.overview_calibration.parent.mkdir(parents=True)
    originals = {
        paths.config: b"config-original",
        paths.overview_calibration: b"overview-original",
        paths.stereo_calibration: b"stereo-original",
    }
    for path, content in originals.items():
        path.write_bytes(content)

    from navmin import input_recovery

    real_replace = input_recovery.os.replace

    def fail_overview_install(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == paths.overview_calibration
            and ".recovery-" in source_path.name
        ):
            raise OSError("install-failed")
        real_replace(source, destination)

    monkeypatch.setattr(input_recovery.os, "replace", fail_overview_install)

    with pytest.raises(InputRecoveryError, match="install-failed"):
        recover_default_inputs(paths, now=RECOVERY_TEST_NOW)

    for path, content in originals.items():
        assert path.read_bytes() == content
    assert not list(tmp_path.rglob("*.bak"))
    assert not list(tmp_path.rglob("*.recovery-*.tmp"))
    assert not list(tmp_path.rglob("*.rollback.tmp"))


def test_normal_input_failure_can_restore_defaults_but_never_continues_startup(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    paths = _recovery_paths(tmp_path)
    shown_results = []
    monkeypatch.setattr("navmin.__main__._prompt_input_recovery", lambda _error: True)
    monkeypatch.setattr(
        "navmin.__main__._show_input_recovery_result", shown_results.append
    )
    monkeypatch.setattr(
        "navmin.__main__.run_startup_preflight",
        lambda *args, **kwargs: pytest.fail("preflight must wait for a new launch"),
    )
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs: pytest.fail("runtime must wait for a new launch"),
    )

    exit_code = normal_main(
        [
            "--config",
            str(paths.config),
            "--overview-calibration",
            str(paths.overview_calibration),
            "--stereo-calibration",
            str(paths.stereo_calibration),
            "--log-dir",
            str(paths.log_dir),
        ]
    )

    assert exit_code == 2
    assert len(shown_results) == 1
    assert paths.config.is_file()
    assert paths.overview_calibration.is_file()
    assert paths.stereo_calibration.is_file()
    assert "INPUT ERROR: config" in capsys.readouterr().err
    manifest = json.loads(
        (_single_session(paths.log_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "input-failed"


def test_normal_preflight_only_input_failure_never_opens_recovery_dialog(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    monkeypatch.setattr(
        "navmin.__main__._prompt_input_recovery",
        lambda _error: pytest.fail("--preflight-only must remain headless"),
    )

    assert normal_main(
        [
            "--config",
            str(paths.config),
            "--overview-calibration",
            str(paths.overview_calibration),
            "--stereo-calibration",
            str(paths.stereo_calibration),
            "--preflight-only",
            "--log-dir",
            str(paths.log_dir),
        ]
    ) == 2

    assert not paths.config.exists()
    assert not paths.overview_calibration.exists()
    assert not paths.stereo_calibration.exists()


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
    assert not (session / "preflight.json").exists()


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

    def fail_runtime(_inputs, *, show_diagnostic_clock: bool) -> int:
        assert show_diagnostic_clock
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


def test_normal_preflight_only_pass_skips_runtime_and_keeps_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    config_path, overview_path, stereo_path = _write_source_placeholders(tmp_path)
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs: pytest.fail("runtime must not start in --preflight-only"),
    )

    log_dir = tmp_path / "logs"
    assert normal_main(
        [
            "--config",
            str(config_path),
            "--overview-calibration",
            str(overview_path),
            "--stereo-calibration",
            str(stereo_path),
            "--preflight-only",
            "--log-dir",
            str(log_dir),
        ]
    ) == 0

    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((session / "preflight.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["exit_code"] == 0
    assert preflight["overall_status"] == "pass"
    assert (session / "runtime.log").is_file()
    assert not (session / "inputs" / "effective-config.json").exists()


def test_diagnostic_preflight_only_uses_planned_localhost_endpoints_without_starting(
    monkeypatch,
    tmp_path,
) -> None:
    captured: list[LoadedApplicationInputs] = []

    def preflight(inputs, **kwargs):
        del kwargs
        captured.append(inputs)
        return _preflight_report()

    class MustNotConstructEndpoints:
        def __init__(self, **kwargs) -> None:
            raise AssertionError(kwargs)

    monkeypatch.setattr("navmin.diagnostic_launcher.run_startup_preflight", preflight)
    monkeypatch.setattr(
        "navmin.diagnostic_launcher.DiagnosticEndpoints",
        MustNotConstructEndpoints,
    )
    monkeypatch.setattr(
        "navmin.diagnostic_launcher.run_loaded_application",
        lambda _inputs: pytest.fail("runtime must not start in --preflight-only"),
    )

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
            "--preflight-only",
            "--log-dir",
            str(log_dir),
        ]
    ) == 0

    assert len(captured) == 1
    planned = captured[0]
    assert planned.config.vision.cameras.overview.address == "127.0.0.1"
    assert planned.config.vision.cameras.stereo_left.address == "127.0.0.1"
    assert planned.config.turret.serial.port == "diagnostic-pty"

    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert (session / "preflight.json").is_file()
    assert not (session / "inputs" / "effective-config.json").exists()


@pytest.mark.parametrize("diagnostic", [False, True])
def test_preflight_failure_blocks_endpoints_runtime_and_records_status(
    monkeypatch,
    tmp_path,
    diagnostic,
) -> None:
    failed = _preflight_report(PreflightStatus.FAIL)
    if diagnostic:
        monkeypatch.setattr(
            "navmin.diagnostic_launcher.run_startup_preflight",
            lambda *args, **kwargs: failed,
        )

        class MustNotConstructEndpoints:
            def __init__(self, **kwargs) -> None:
                raise AssertionError(kwargs)

        monkeypatch.setattr(
            "navmin.diagnostic_launcher.DiagnosticEndpoints",
            MustNotConstructEndpoints,
        )
        monkeypatch.setattr(
            "navmin.diagnostic_launcher.run_loaded_application",
            lambda _inputs: pytest.fail("runtime must not start after preflight FAIL"),
        )
        log_dir = tmp_path / "diagnostic-logs"
        exit_code = diagnostic_main(
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
        )
    else:
        config_path, overview_path, stereo_path = _write_source_placeholders(tmp_path)
        monkeypatch.setattr(
            "navmin.__main__.load_application_inputs",
            lambda _paths: _inputs(),
        )
        monkeypatch.setattr(
            "navmin.__main__.run_startup_preflight",
            lambda *args, **kwargs: failed,
        )
        monkeypatch.setattr(
            "navmin.__main__.run_loaded_application",
            lambda _inputs: pytest.fail("runtime must not start after preflight FAIL"),
        )
        log_dir = tmp_path / "normal-logs"
        exit_code = normal_main(
            [
                "--config",
                str(config_path),
                "--overview-calibration",
                str(overview_path),
                "--stereo-calibration",
                str(stereo_path),
                "--log-dir",
                str(log_dir),
            ]
        )

    assert exit_code == 2
    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((session / "preflight.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "preflight-failed"
    assert manifest["exit_code"] == 2
    assert preflight["overall_status"] == "fail"
    assert (session / "runtime.log").is_file()
    assert not (session / "inputs" / "effective-config.json").exists()


def test_diagnostic_full_launch_orders_preflight_before_endpoints_and_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []

    def preflight(inputs, **kwargs):
        del inputs, kwargs
        events.append("preflight")
        return _preflight_report()

    class FakeEndpoints:
        def __init__(self, *, selection, inputs) -> None:
            del selection
            self.inputs = inputs
            events.append("construct-endpoints")

        def start(self):
            events.append("start-endpoints")
            return self.inputs

        def stop(self) -> None:
            events.append("stop-endpoints")

    def run(_inputs, *, show_diagnostic_clock: bool) -> int:
        assert show_diagnostic_clock
        events.append("runtime")
        return 0

    monkeypatch.setattr("navmin.diagnostic_launcher.run_startup_preflight", preflight)
    monkeypatch.setattr("navmin.diagnostic_launcher.DiagnosticEndpoints", FakeEndpoints)
    monkeypatch.setattr("navmin.diagnostic_launcher.run_loaded_application", run)

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

    assert events == [
        "preflight",
        "construct-endpoints",
        "start-endpoints",
        "runtime",
        "stop-endpoints",
    ]


def test_normal_full_launch_orders_preflight_before_runtime(monkeypatch, tmp_path) -> None:
    config_path, overview_path, stereo_path = _write_source_placeholders(tmp_path)
    events: list[str] = []
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())

    def preflight(inputs, **kwargs):
        del inputs, kwargs
        events.append("preflight")
        return _preflight_report()

    def run(_inputs) -> int:
        events.append("runtime")
        return 0

    monkeypatch.setattr("navmin.__main__.run_startup_preflight", preflight)
    monkeypatch.setattr("navmin.__main__.run_loaded_application", run)

    assert normal_main(
        [
            "--config",
            str(config_path),
            "--overview-calibration",
            str(overview_path),
            "--stereo-calibration",
            str(stereo_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    ) == 0
    assert events == ["preflight", "runtime"]


def test_preflight_only_warn_is_success(monkeypatch, tmp_path) -> None:
    config_path, overview_path, stereo_path = _write_source_placeholders(tmp_path)
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr(
        "navmin.__main__.run_startup_preflight",
        lambda *args, **kwargs: _preflight_report(PreflightStatus.WARN),
    )
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs: pytest.fail("runtime must not start in --preflight-only"),
    )

    log_dir = tmp_path / "logs"
    assert normal_main(
        [
            "--config",
            str(config_path),
            "--overview-calibration",
            str(overview_path),
            "--stereo-calibration",
            str(stereo_path),
            "--preflight-only",
            "--log-dir",
            str(log_dir),
        ]
    ) == 0
    preflight = json.loads(
        (_single_session(log_dir) / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["overall_status"] == "warn"


def test_unexpected_preflight_exception_is_runtime_failure_before_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    config_path, overview_path, stereo_path = _write_source_placeholders(tmp_path)
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())

    def explode(*args, **kwargs):
        del args, kwargs
        raise KeyError("preflight-bug")

    monkeypatch.setattr("navmin.__main__.run_startup_preflight", explode)
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs: pytest.fail("runtime must not start after preflight exception"),
    )

    log_dir = tmp_path / "logs"
    assert normal_main(
        [
            "--config",
            str(config_path),
            "--overview-calibration",
            str(overview_path),
            "--stereo-calibration",
            str(stereo_path),
            "--log-dir",
            str(log_dir),
        ]
    ) == 2

    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "runtime-failed"
    assert manifest["failure_type"] == "KeyError"
    assert "preflight-bug" in manifest["failure_message"]
    assert not (session / "preflight.json").exists()
    runtime_log = (session / "runtime.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in runtime_log
    assert "preflight-bug" in runtime_log
