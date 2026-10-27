from __future__ import annotations

import json
import logging
import shutil
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from navmin.__main__ import (
    _interactive_options,
    _run_interactive,
)
from navmin.__main__ import (
    _parser as normal_parser,
)
from navmin.__main__ import (
    main as normal_main,
)
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
    ImuConfig,
    OverviewCameraConfig,
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
from navmin.diagnostic_launcher import (
    CameraEndpoint,
    DiagnosticEndpoints,
    DiagnosticSelection,
    TurretEndpoint,
    _planned_preflight_inputs,
)
from navmin.diagnostic_launcher import (
    _parser as diagnostic_parser,
)
from navmin.diagnostic_launcher import main as diagnostic_main
from navmin.diagnostics.synthetic_sources import SyntheticFrameSource
from navmin.host_environment import WindowsHostEnvironment
from navmin.imu import ImuCalibration
from navmin.input_recovery import InputRecoveryError, recover_default_inputs
from navmin.launch_options import (
    CameraSourceMode,
    LaunchOptions,
    LaunchSelectionError,
    StereoRightMode,
    Stm32Mode,
    apply_launch_options,
    default_launch_options,
    resolve_launch_options,
)
from navmin.launch_runtime import (
    LaunchRuntimeBindingError,
    build_launch_runtime_bindings,
)
from navmin.launcher import (
    ApplicationInputError,
    LauncherPaths,
    LoadedApplicationInputs,
    load_application_inputs,
    resolve_imu_calibration_path,
    run_loaded_application,
    session_file_logging,
    validate_normal_hardware_config,
)
from navmin.preflight import (
    PreflightStatus,
    StartupPreflightCheck,
    StartupPreflightReport,
)
from navmin.session_artifacts import SessionArtifacts, SourceInputPaths
from navmin.vision.gstreamer_source import GStreamerRtpJpegSource

WIDTH = 64
HEIGHT = 48
K = ((50.0, 0.0, 31.5), (0.0, 50.0, 23.5), (0.0, 0.0, 1.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
TEST_LOCAL_TZ = timezone(timedelta(hours=7))
RECOVERY_TEST_NOW = datetime(2026, 9, 23, 10, 15, 30, tzinfo=TEST_LOCAL_TZ)
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    monkeypatch.setattr(
        "navmin.__main__.create_host_environment",
        lambda _root: WindowsHostEnvironment(),
    )


def _camera(port: int) -> CameraConfig:
    return CameraConfig(
        enabled=True,
        source=RtpJpegSourceConfig(
            bind_address="0.0.0.0",
            port=port,
            buffer_size=1,
        ),
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _overview_camera(port: int) -> OverviewCameraConfig:
    camera = _camera(port)
    return OverviewCameraConfig(
        enabled=camera.enabled,
        source=camera.source,
        processing_enabled=camera.processing_enabled,
        vision_processor_class=camera.vision_processor_class,
        calibration_file="calibration/overview.json",
    )


def _default_tracking_control_config():
    defaults = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "navmin.json"
    )
    return load_config(defaults).tracking_control


def _config(*, emulate_stm32: bool = False) -> AppConfig:
    center = AimPointConfig(x_px=None, y_px=None)
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    return AppConfig(
        schema_version=4,
        vision=VisionConfig(
            processing_scope=ProcessingScope.MAIN_AND_PREVIEW,
            cameras=CamerasConfig(
                overview=_overview_camera(8888),
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
                    calibration_file="calibration/stereo.json",
                ),
            ),
            camera_stale_timeout_ms=500,
            simulation_mode=False,
        ),
        aiming=AimingConfig(
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(overview=center, stereo_left=center),
        ),
        tracking_control=_default_tracking_control_config(),
        imu=ImuConfig(
            enabled=False,
            bind_address="0.0.0.0",
            port=8891,
            expected_source=None,
            calibration_file="calibration/imu.json",
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


def test_default_launch_options_follow_persistent_source_defaults() -> None:
    config = _config(emulate_stm32=True)
    config = replace(
        config,
        vision=replace(
            config.vision,
            cameras=replace(
                config.vision.cameras,
                stereo_right=replace(
                    config.vision.cameras.stereo_right,
                    enabled=False,
                ),
            ),
        ),
    )

    options = default_launch_options(
        config,
        config_path=Path("config/site-a/navmin.json"),
    )

    assert options == LaunchOptions(
        overview=CameraSourceMode.ON,
        stereo_left=CameraSourceMode.ON,
        stereo_right=StereoRightMode.OFF,
        stm32=Stm32Mode.SIM,
        config_path=Path("config/site-a/navmin.json"),
    )


@pytest.mark.parametrize("camera_name", ["overview", "stereo_left"])
def test_default_launch_options_reject_disabled_mandatory_camera(
    camera_name: str,
) -> None:
    config = _config()
    cameras = config.vision.cameras
    camera = getattr(cameras, camera_name)
    config = replace(
        config,
        vision=replace(
            config.vision,
            cameras=replace(cameras, **{camera_name: replace(camera, enabled=False)}),
        ),
    )

    with pytest.raises(LaunchSelectionError, match="current runtime requires it"):
        default_launch_options(config)


def test_apply_launch_options_is_immutable_and_resolves_stm32_sim() -> None:
    config = _config()
    options = LaunchOptions(
        overview=CameraSourceMode.SIM,
        stereo_left=CameraSourceMode.SIM,
        stereo_right=StereoRightMode.SIM,
        stm32=Stm32Mode.SIM,
    )

    effective = apply_launch_options(config, options)

    assert effective.config is not config
    assert config.turret.emulate_stm32 is False
    assert effective.config.turret.emulate_stm32 is True
    assert effective.config.vision.cameras.overview.source is config.vision.cameras.overview.source
    assert (
        effective.config.vision.cameras.stereo_left.source
        is config.vision.cameras.stereo_left.source
    )
    assert effective.config.vision.distance.stereo.stereo_enabled is False
    assert effective.warnings == ()
    assert effective.real_camera_roles == ()
    assert effective.simulated_camera_roles == (
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
        CameraRole.STEREO_RIGHT,
    )
    assert effective.requires_camera_network is False


@pytest.mark.parametrize(
    ("left_mode", "right_mode"),
    [
        (CameraSourceMode.ON, StereoRightMode.ON),
        (CameraSourceMode.SIM, StereoRightMode.SIM),
    ],
)
def test_apply_launch_options_keeps_compatible_stereo_enabled(
    left_mode: CameraSourceMode,
    right_mode: StereoRightMode,
) -> None:
    config = _config()
    config = replace(
        config,
        vision=replace(
            config.vision,
            distance=replace(
                config.vision.distance,
                stereo=replace(
                    config.vision.distance.stereo,
                    stereo_enabled=True,
                ),
            ),
        ),
    )
    options = LaunchOptions(
        overview=CameraSourceMode.ON,
        stereo_left=left_mode,
        stereo_right=right_mode,
        stm32=Stm32Mode.ON,
    )

    effective = apply_launch_options(config, options)

    assert effective.config.vision.distance.stereo.stereo_enabled is True
    assert effective.warnings == ()
    expected_real_roles = [CameraRole.OVERVIEW]
    if left_mode is CameraSourceMode.ON:
        expected_real_roles.extend((CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT))
    assert effective.real_camera_roles == tuple(expected_real_roles)
    assert effective.requires_camera_network is True


@pytest.mark.parametrize(
    ("left_mode", "right_mode"),
    [
        (CameraSourceMode.ON, StereoRightMode.SIM),
        (CameraSourceMode.SIM, StereoRightMode.ON),
    ],
)
def test_apply_launch_options_disables_mixed_stereo_pairing(
    left_mode: CameraSourceMode,
    right_mode: StereoRightMode,
) -> None:
    config = _config()
    config = replace(
        config,
        vision=replace(
            config.vision,
            distance=replace(
                config.vision.distance,
                stereo=replace(
                    config.vision.distance.stereo,
                    stereo_enabled=True,
                ),
            ),
        ),
    )
    options = LaunchOptions(
        overview=CameraSourceMode.ON,
        stereo_left=left_mode,
        stereo_right=right_mode,
        stm32=Stm32Mode.ON,
    )

    effective = apply_launch_options(config, options)

    assert effective.config.vision.cameras.stereo_left.enabled is True
    assert effective.config.vision.cameras.stereo_right.enabled is True
    assert effective.config.vision.distance.stereo.stereo_enabled is False
    assert len(effective.warnings) == 1
    assert "mixed real/sim" in effective.warnings[0]


def test_apply_launch_options_right_off_disables_only_optional_stereo_feature() -> None:
    config = _config()
    config = replace(
        config,
        vision=replace(
            config.vision,
            distance=replace(
                config.vision.distance,
                stereo=replace(
                    config.vision.distance.stereo,
                    stereo_enabled=True,
                ),
            ),
        ),
    )
    options = LaunchOptions(
        overview=CameraSourceMode.ON,
        stereo_left=CameraSourceMode.ON,
        stereo_right=StereoRightMode.OFF,
        stm32=Stm32Mode.ON,
    )

    effective = apply_launch_options(config, options)

    assert effective.config.vision.cameras.overview.enabled is True
    assert effective.config.vision.cameras.stereo_left.enabled is True
    assert effective.config.vision.cameras.stereo_right.enabled is False
    assert effective.config.vision.distance.stereo.stereo_enabled is False
    assert len(effective.warnings) == 1
    assert "Stereo Right is OFF" in effective.warnings[0]
    assert effective.real_camera_roles == (
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    )


def _inputs_with_working_range_stereo() -> LoadedApplicationInputs:
    stereo = load_stereo_calibration(
        REPO_ROOT / "config" / "defaults" / "calibration" / "stereo.json"
    )
    return replace(_inputs(), stereo_calibration=stereo)


def test_launch_runtime_bindings_all_sim_are_in_process_and_show_clock() -> None:
    inputs = _inputs_with_working_range_stereo()
    options = LaunchOptions(
        overview=CameraSourceMode.SIM,
        stereo_left=CameraSourceMode.SIM,
        stereo_right=StereoRightMode.SIM,
        stm32=Stm32Mode.SIM,
    )

    bindings = build_launch_runtime_bindings(inputs, options)

    assert bindings.show_diagnostic_clock is True
    assert bindings.inputs.config.turret.emulate_stm32 is True
    assert bindings.inputs.overview_calibration is not inputs.overview_calibration
    assert bindings.inputs.stereo_calibration is not inputs.stereo_calibration
    assert inputs.config.turret.emulate_stm32 is False

    cameras = bindings.inputs.config.vision.cameras
    overview_source = bindings.factories.overview_source_factory(cameras.overview)
    left_source = bindings.factories.stereo_left_source_factory(cameras.stereo_left)
    right_source = bindings.factories.stereo_right_source_factory(cameras.stereo_right)
    for source in (overview_source, left_source, right_source):
        assert isinstance(source, SyntheticFrameSource)
        source.start()
        frame = source.read()
        assert frame is not None
        assert frame.capture_id is None
        assert frame.receive_timestamp_ns is not None
        source.stop()

    assert overview_source.read() is None
    assert left_source.read() is None
    assert right_source.read() is None


def test_launch_runtime_bindings_shared_stereo_scene_is_used_by_both_factories(
    monkeypatch,
) -> None:
    inputs = _inputs_with_working_range_stereo()
    instances = []

    class FakeSharedStereo:
        def __init__(self, calibration) -> None:
            self.calibration = calibration
            self.calls = []
            instances.append(self)

        def next_frame(self, camera: CameraRole) -> np.ndarray:
            self.calls.append(camera)
            return np.zeros(
                (self.calibration.image_height, self.calibration.image_width, 3),
                dtype=np.uint8,
            )

    monkeypatch.setattr(
        "navmin.launch_runtime.SynchronizedSyntheticStereoSource",
        FakeSharedStereo,
    )
    bindings = build_launch_runtime_bindings(
        inputs,
        LaunchOptions(
            overview=CameraSourceMode.ON,
            stereo_left=CameraSourceMode.SIM,
            stereo_right=StereoRightMode.SIM,
            stm32=Stm32Mode.ON,
        ),
    )
    left = bindings.factories.stereo_left_source_factory(
        bindings.inputs.config.vision.cameras.stereo_left
    )
    right = bindings.factories.stereo_right_source_factory(
        bindings.inputs.config.vision.cameras.stereo_right
    )
    left.start()
    right.start()
    assert left.read() is not None
    assert right.read() is not None

    assert len(instances) == 1
    assert instances[0].calls == [CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT]


@pytest.mark.parametrize(
    ("left_mode", "right_mode", "synthetic_side"),
    [
        (CameraSourceMode.SIM, StereoRightMode.ON, "left"),
        (CameraSourceMode.ON, StereoRightMode.SIM, "right"),
    ],
)
def test_launch_runtime_bindings_mixed_stereo_keep_real_side_calibration(
    left_mode: CameraSourceMode,
    right_mode: StereoRightMode,
    synthetic_side: str,
) -> None:
    inputs = _inputs_with_working_range_stereo()
    reference = inputs.stereo_calibration
    bindings = build_launch_runtime_bindings(
        inputs,
        LaunchOptions(
            overview=CameraSourceMode.ON,
            stereo_left=left_mode,
            stereo_right=right_mode,
            stm32=Stm32Mode.ON,
        ),
    )
    effective = bindings.inputs.stereo_calibration

    assert bindings.inputs.config.vision.distance.stereo.stereo_enabled is False
    assert len(bindings.selection.warnings) == 1
    if synthetic_side == "left":
        assert effective.K_left == tuple(tuple(row[:3]) for row in reference.P1)
        assert effective.D_left == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert effective.K_right == reference.K_right
        assert effective.D_right == reference.D_right
        assert effective.R2 == reference.R2
    else:
        assert effective.K_right == tuple(tuple(row[:3]) for row in reference.P2)
        assert effective.D_right == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert effective.K_left == reference.K_left
        assert effective.D_left == reference.D_left
        assert effective.R1 == reference.R1


def test_launch_runtime_bindings_real_sources_preserve_production_factories() -> None:
    inputs = _inputs_with_working_range_stereo()
    bindings = build_launch_runtime_bindings(
        inputs,
        LaunchOptions(
            overview=CameraSourceMode.ON,
            stereo_left=CameraSourceMode.ON,
            stereo_right=StereoRightMode.ON,
            stm32=Stm32Mode.ON,
        ),
    )

    assert bindings.show_diagnostic_clock is False
    cameras = bindings.inputs.config.vision.cameras
    assert isinstance(
        bindings.factories.overview_source_factory(cameras.overview),
        GStreamerRtpJpegSource,
    )
    assert bindings.inputs.overview_calibration is inputs.overview_calibration
    assert bindings.inputs.stereo_calibration is inputs.stereo_calibration


def test_launch_runtime_bindings_reject_unsupported_synthetic_stereo_size() -> None:
    with pytest.raises(LaunchRuntimeBindingError, match="1296x976"):
        build_launch_runtime_bindings(
            _inputs(),
            LaunchOptions(
                overview=CameraSourceMode.ON,
                stereo_left=CameraSourceMode.SIM,
                stereo_right=StereoRightMode.OFF,
                stm32=Stm32Mode.ON,
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
        translation_unit="m",
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


def _config_with_calibration_refs(
    overview_ref: str,
    stereo_ref: str,
    imu_ref: str | None = None,
) -> AppConfig:
    config = _config()
    cameras = replace(
        config.vision.cameras,
        overview=replace(
            config.vision.cameras.overview,
            calibration_file=overview_ref,
        ),
    )
    distance = replace(
        config.vision.distance,
        stereo=replace(
            config.vision.distance.stereo,
            calibration_file=stereo_ref,
        ),
    )
    result = replace(
        config,
        vision=replace(config.vision, cameras=cameras, distance=distance),
    )
    if imu_ref is not None:
        result = replace(result, imu=replace(result.imu, calibration_file=imu_ref))
    return result


def _copy_default_calibrations(target_root: Path) -> tuple[Path, Path]:
    target_dir = target_root / "calibration"
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir = REPO_ROOT / "config" / "defaults" / "calibration"
    overview = target_dir / "overview.json"
    stereo = target_dir / "stereo.json"
    shutil.copyfile(source_dir / "overview.json", overview)
    shutil.copyfile(source_dir / "stereo.json", stereo)
    return overview, stereo


def test_public_and_diagnostic_parsers_keep_separate_contracts() -> None:
    normal_args = normal_parser().parse_args([])
    assert normal_args.config == Path("config/navmin.json")
    assert normal_args.overview is None
    assert normal_args.stereo_left is None
    assert normal_args.stereo_right is None
    assert normal_args.stm32 is None
    assert normal_args.preflight_only is False
    assert normal_args.verbose is False
    assert not hasattr(normal_args, "overview_calibration")
    assert not hasattr(normal_args, "stereo_calibration")
    assert not hasattr(normal_args, "log_dir")

    diagnostic_args = diagnostic_parser().parse_args(
        ["--overview", "real", "--stereo-left", "real", "--turret", "real"]
    )
    assert diagnostic_args.config == Path("config/navmin.json")
    assert diagnostic_args.overview_calibration is None
    assert diagnostic_args.stereo_calibration is None
    assert diagnostic_args.stereo_right is None

def test_config_owned_relative_calibration_paths_follow_selected_config_directory(
    monkeypatch,
    tmp_path,
) -> None:
    cwd = tmp_path / "cwd"
    selected_root = tmp_path / "sets" / "test-a"
    cwd.mkdir()
    selected_root.mkdir(parents=True)
    selected_config = selected_root / "navmin.json"
    save_config(selected_config, _config())
    overview, stereo = _copy_default_calibrations(selected_root)

    # A decoy relative to CWD must never be used for config-owned references.
    decoy_dir = cwd / "calibration"
    decoy_dir.mkdir()
    (decoy_dir / "overview.json").write_text("{broken", encoding="utf-8")
    (decoy_dir / "stereo.json").write_text("{broken", encoding="utf-8")
    monkeypatch.chdir(cwd)

    inputs = load_application_inputs(
        LauncherPaths(config=Path("../sets/test-a/navmin.json"))
    )

    assert inputs.source_paths is not None
    assert inputs.source_paths.config == selected_config.resolve()
    assert inputs.source_paths.overview_calibration == overview.resolve()
    assert inputs.source_paths.stereo_calibration == stereo.resolve()


def test_absolute_config_owned_calibration_paths_are_used_unchanged(tmp_path) -> None:
    selected_root = tmp_path / "set"
    selected_root.mkdir()
    overview, stereo = _copy_default_calibrations(tmp_path / "absolute")
    config_path = selected_root / "navmin.json"
    save_config(
        config_path,
        _config_with_calibration_refs(str(overview), str(stereo)),
    )

    inputs = load_application_inputs(LauncherPaths(config=config_path))

    assert inputs.source_paths is not None
    assert inputs.source_paths.overview_calibration == overview.resolve()
    assert inputs.source_paths.stereo_calibration == stereo.resolve()


def test_relative_cli_calibration_overrides_win_and_use_cwd_semantics(
    monkeypatch,
    tmp_path,
) -> None:
    selected_root = tmp_path / "selected"
    cwd = tmp_path / "cwd"
    selected_root.mkdir()
    cwd.mkdir()
    config_path = selected_root / "navmin.json"
    save_config(config_path, _config())
    _copy_default_calibrations(selected_root)

    override_overview, override_stereo = _copy_default_calibrations(cwd / "override")
    monkeypatch.chdir(cwd)
    inputs = load_application_inputs(
        LauncherPaths(
            config=config_path,
            overview_calibration=Path("override/calibration/overview.json"),
            stereo_calibration=Path("override/calibration/stereo.json"),
        )
    )

    assert inputs.source_paths is not None
    assert inputs.source_paths.overview_calibration == override_overview.resolve()
    assert inputs.source_paths.stereo_calibration == override_stereo.resolve()


def test_cli_override_paths_are_the_paths_recorded_in_session_source_hashes(
    tmp_path,
) -> None:
    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    config_path = selected_root / "navmin.json"
    save_config(config_path, _config())
    _copy_default_calibrations(selected_root)
    override_overview, override_stereo = _copy_default_calibrations(tmp_path / "override")
    inputs = load_application_inputs(
        LauncherPaths(
            config=config_path,
            overview_calibration=override_overview,
            stereo_calibration=override_stereo,
        )
    )
    assert inputs.source_paths is not None

    artifacts = SessionArtifacts.create(
        tmp_path / "logs",
        mode="normal",
        overview_backend="real",
        stereo_left_backend="real",
        turret_backend="real",
        input_mode="files",
        git_root=tmp_path / "no-git",
    )
    artifacts.write_effective_inputs(
        config=inputs.config,
        overview_calibration=inputs.overview_calibration,
        stereo_calibration=inputs.stereo_calibration,
        source_paths=inputs.source_paths,
    )
    sources = json.loads(
        (artifacts.inputs_dir / "source-hashes.json").read_text(encoding="utf-8")
    )

    assert sources["config"]["path"] == str(config_path.resolve())
    assert sources["overview_calibration"]["path"] == str(override_overview.resolve())
    assert sources["stereo_calibration"]["path"] == str(override_stereo.resolve())


def test_run_loaded_application_uses_shared_runtime_ui_surface_and_shutdown() -> None:
    events: list[str] = []
    inputs = _inputs()
    pid_config = inputs.config.turret.controller
    pid_updates: list[PidControllerConfig] = []
    apply_pid_tuning = pid_updates.append
    ui_dependencies = SimpleNamespace(
        mediator=object(),
        camera_bindings={CameraRole.OVERVIEW: object()},
        turret_states=object(),
        imu_states=object(),
        imu_controls=object(),
        camera_stale_timeout_ms=321,
        pid_controller_config=pid_config,
        apply_pid_tuning=apply_pid_tuning,
        tracking_control_config=inputs.config.tracking_control,
        apply_tracking_control_tuning=lambda _config: None,
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.imu_runtime = None
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
            "imu_calibration": inputs.imu_calibration,
        }
        return FakeRuntime()

    def ui_runner(**kwargs) -> int:
        events.append("ui")
        assert kwargs["mediator"] is ui_dependencies.mediator
        assert kwargs["camera_bindings"] is ui_dependencies.camera_bindings
        assert kwargs["turret_states"] is ui_dependencies.turret_states
        assert kwargs["imu_states"] is ui_dependencies.imu_states
        assert kwargs["imu_controls"] is ui_dependencies.imu_controls
        assert kwargs["camera_stale_timeout_ms"] == 321
        assert kwargs["pid_controller_config"] is pid_config
        assert kwargs["apply_pid_tuning"] is apply_pid_tuning
        assert kwargs["tracking_control_config"] is inputs.config.tracking_control
        assert (
            kwargs["apply_tracking_control_tuning"]
            is ui_dependencies.apply_tracking_control_tuning
        )
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


def test_run_loaded_application_installs_launcher_owned_imu_calibration_writer(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config" / "navmin.json"
    expected_imu_path = tmp_path / "config" / "calibration" / "imu.json"
    base = _inputs()
    config = replace(base.config, imu=replace(base.config.imu, enabled=True))
    inputs = LoadedApplicationInputs(
        config=config,
        overview_calibration=base.overview_calibration,
        stereo_calibration=base.stereo_calibration,
        imu_calibration=ImuCalibration(),
        source_paths=SourceInputPaths(
            config=config_path,
            overview_calibration=tmp_path / "overview.json",
            stereo_calibration=tmp_path / "stereo.json",
        ),
    )

    class FakeImuRuntime:
        def __init__(self) -> None:
            self.writer = None

        def set_calibration_writer(self, writer) -> None:
            self.writer = writer

    class FakeRuntime:
        def __init__(self) -> None:
            self.imu_runtime = FakeImuRuntime()
            self.ui_dependencies = SimpleNamespace(
                mediator=object(),
                camera_bindings={},
                turret_states=object(),
                imu_states=object(),
                imu_controls=object(),
                camera_stale_timeout_ms=500,
                pid_controller_config=config.turret.controller,
                apply_pid_tuning=lambda _config: None,
                tracking_control_config=config.tracking_control,
                apply_tracking_control_tuning=lambda _config: None,
            )

        def start(self) -> None:
            assert self.imu_runtime.writer is not None
            self.imu_runtime.writer(
                ImuCalibration(gyro_bias=[1.0, 2.0, 3.0], gyro_ready=True)
            )

        def shutdown(self) -> None:
            pass

    runtime = FakeRuntime()
    result = run_loaded_application(
        inputs,
        runtime_builder=lambda **_kwargs: runtime,
        ui_runner=lambda **_kwargs: 0,
    )

    assert result == 0
    saved = ImuCalibration.load(expected_imu_path)
    assert saved.gyro_ready
    assert saved.gyro_bias == [1.0, 2.0, 3.0]


def test_normal_launcher_rejects_legacy_fake_turret_configuration() -> None:
    validate_normal_hardware_config(_config())
    with pytest.raises(ValueError, match="emulate-stm32=false"):
        validate_normal_hardware_config(_config(emulate_stm32=True))


def test_diagnostic_selection_overrides_only_requested_external_endpoints(monkeypatch) -> None:
    sender_configs = []
    sender_events: list[str] = []
    pty_events: list[str] = []

    class FakeSender:
        def __init__(self, config, **_kwargs) -> None:
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
        assert effective.config.vision.cameras.overview.source.bind_address == "127.0.0.1"
        assert effective.config.vision.cameras.overview.source.port == 8888
        assert effective.config.vision.cameras.stereo_left.source.bind_address == "0.0.0.0"
        assert effective.config.vision.cameras.stereo_right.enabled is False
        assert effective.stereo_calibration == endpoints._base_inputs.stereo_calibration
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
        assert (
            effective.config.vision.cameras.overview
            == inputs.config.vision.cameras.overview
        )
        assert (
            effective.config.vision.cameras.stereo_left
            == inputs.config.vision.cameras.stereo_left
        )
        assert effective.config.vision.cameras.stereo_right == replace(
            inputs.config.vision.cameras.stereo_right,
            enabled=False,
        )
    finally:
        endpoints.stop()


def test_diagnostic_explicit_real_stereo_right_enables_configured_source() -> None:
    inputs = _inputs()
    right = replace(inputs.config.vision.cameras.stereo_right, enabled=False)
    config = replace(
        inputs.config,
        vision=replace(
            inputs.config.vision,
            cameras=replace(inputs.config.vision.cameras, stereo_right=right),
        ),
    )
    endpoints = DiagnosticEndpoints(
        selection=DiagnosticSelection(
            overview=CameraEndpoint.REAL,
            stereo_left=CameraEndpoint.REAL,
            turret=TurretEndpoint.REAL,
            stereo_right=CameraEndpoint.REAL,
        ),
        inputs=replace(inputs, config=config),
    )

    effective = endpoints.start()
    try:
        assert effective.config.vision.cameras.stereo_right.enabled is True
        assert (
            effective.config.vision.cameras.stereo_right.source
            == inputs.config.vision.cameras.stereo_right.source
        )
        assert effective.overview_calibration == inputs.overview_calibration
        assert effective.stereo_calibration == inputs.stereo_calibration
    finally:
        endpoints.stop()


def test_diagnostic_localhost_stereo_right_uses_stereo_calibration(
    monkeypatch,
) -> None:
    sender_configs = []
    sender_events: list[str] = []

    class FakeSender:
        def __init__(self, config, **_kwargs) -> None:
            sender_configs.append(config)

        def start(self) -> None:
            sender_events.append("start")

        def stop(self) -> None:
            sender_events.append("stop")

    monkeypatch.setattr("navmin.diagnostic_launcher.LocalhostRtpJpegSender", FakeSender)
    inputs = _inputs()
    right = replace(inputs.config.vision.cameras.stereo_right, enabled=False)
    config = replace(
        inputs.config,
        vision=replace(
            inputs.config.vision,
            cameras=replace(inputs.config.vision.cameras, stereo_right=right),
        ),
    )
    endpoints = DiagnosticEndpoints(
        selection=DiagnosticSelection(
            overview=CameraEndpoint.REAL,
            stereo_left=CameraEndpoint.REAL,
            turret=TurretEndpoint.REAL,
            stereo_right=CameraEndpoint.LOCALHOST,
        ),
        inputs=replace(inputs, config=config),
    )

    effective = endpoints.start()
    try:
        source = effective.config.vision.cameras.stereo_right.source
        assert effective.config.vision.cameras.stereo_right.enabled is True
        assert isinstance(source, RtpJpegSourceConfig)
        assert source.bind_address == "127.0.0.1"
        assert source.port == 8890
        assert len(sender_configs) == 1
        assert sender_configs[0].camera is CameraRole.STEREO_RIGHT
        assert sender_configs[0].width == WIDTH
        assert sender_configs[0].height == HEIGHT
        assert effective.stereo_calibration == inputs.stereo_calibration
    finally:
        endpoints.stop()

    assert sender_events == ["start", "stop"]


def test_all_localhost_profile_uses_image_identity_calibrations(monkeypatch) -> None:
    class FakeSender:
        def __init__(self, _config, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr("navmin.diagnostic_launcher.LocalhostRtpJpegSender", FakeSender)

    inputs = _inputs()
    distorted_stereo = replace(
        inputs.stereo_calibration,
        K_left=((70.0, 0.0, 31.5), (0.0, 70.0, 23.5), (0.0, 0.0, 1.0)),
        D_left=(0.2, -0.1, 0.01, 0.0, 0.0),
        K_right=((72.0, 0.0, 31.5), (0.0, 72.0, 23.5), (0.0, 0.0, 1.0)),
        D_right=(-0.1, 0.05, 0.0, 0.0, 0.0),
    )
    inputs = replace(inputs, stereo_calibration=distorted_stereo)
    selection = DiagnosticSelection(
        overview=CameraEndpoint.LOCALHOST,
        stereo_left=CameraEndpoint.LOCALHOST,
        stereo_right=CameraEndpoint.LOCALHOST,
        turret=TurretEndpoint.REAL,
    )

    endpoints = DiagnosticEndpoints(selection=selection, inputs=inputs)
    effective = endpoints.start()
    try:
        assert effective.overview_calibration.D != inputs.overview_calibration.D
        assert effective.stereo_calibration.P1 == inputs.stereo_calibration.P1
        assert effective.stereo_calibration.P2 == inputs.stereo_calibration.P2
        assert effective.stereo_calibration.Q == inputs.stereo_calibration.Q
        assert effective.stereo_calibration.K_left == tuple(
            tuple(row[:3]) for row in inputs.stereo_calibration.P1
        )
        assert effective.stereo_calibration.K_right == tuple(
            tuple(row[:3]) for row in inputs.stereo_calibration.P2
        )
        assert effective.stereo_calibration.D_left == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert effective.stereo_calibration.D_right == (0.0, 0.0, 0.0, 0.0, 0.0)
    finally:
        endpoints.stop()

    planned = _planned_preflight_inputs(selection, inputs)
    assert planned.overview_calibration == effective.overview_calibration
    assert planned.stereo_calibration == effective.stereo_calibration


def test_all_localhost_working_range_profile_shares_stereo_frame_source(
    monkeypatch,
) -> None:
    senders = []

    class FakeSender:
        def __init__(self, config, **kwargs) -> None:
            senders.append((config, kwargs))

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr("navmin.diagnostic_launcher.LocalhostRtpJpegSender", FakeSender)
    inputs = _inputs()
    stereo = load_stereo_calibration(REPO_ROOT / "config/calibration/stereo.json")
    inputs = replace(inputs, stereo_calibration=stereo)
    selection = DiagnosticSelection(
        overview=CameraEndpoint.LOCALHOST,
        stereo_left=CameraEndpoint.LOCALHOST,
        stereo_right=CameraEndpoint.LOCALHOST,
        turret=TurretEndpoint.REAL,
    )

    endpoints = DiagnosticEndpoints(selection=selection, inputs=inputs)
    effective = endpoints.start()
    try:
        by_camera = {config.camera: kwargs for config, kwargs in senders}
        left_supplier = by_camera[CameraRole.STEREO_LEFT]["frame_supplier"]
        right_supplier = by_camera[CameraRole.STEREO_RIGHT]["frame_supplier"]
        assert left_supplier is not None
        assert right_supplier is not None
        left = left_supplier()
        right = right_supplier()
        assert left.shape == right.shape == (976, 1296, 3)
        assert not np.array_equal(left, right)
        assert effective.stereo_calibration.image_width == 1296
        assert effective.stereo_calibration.image_height == 976
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
    assert inputs.config.vision.cameras.overview.source.port == 8888
    assert inputs.config.vision.cameras.stereo_left.source.port == 8889

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


def test_disabled_imu_does_not_require_calibration_file(tmp_path) -> None:
    selected_root = tmp_path / "disabled-imu"
    selected_root.mkdir()
    config_path = selected_root / "navmin.json"
    save_config(config_path, _config())
    _copy_default_calibrations(selected_root)
    imu_path = selected_root / "calibration" / "imu.json"
    imu_path.write_text("{broken-but-disabled", encoding="utf-8")

    inputs = load_application_inputs(LauncherPaths(config=config_path))

    assert inputs.imu_calibration is None
    assert imu_path.read_text(encoding="utf-8") == "{broken-but-disabled"


def test_enabled_imu_loads_strict_config_owned_calibration(tmp_path) -> None:
    selected_root = tmp_path / "enabled-imu"
    selected_root.mkdir()
    config_path = selected_root / "navmin.json"
    base = _config()
    config = replace(base, imu=replace(base.imu, enabled=True))
    save_config(config_path, config)
    _copy_default_calibrations(selected_root)
    expected = ImuCalibration(
        accel_bias=[0.1, 0.2, 0.3],
        accel_ready=True,
    )
    imu_path = selected_root / "calibration" / "imu.json"
    expected.save(imu_path)

    inputs = load_application_inputs(LauncherPaths(config=config_path))

    assert inputs.imu_calibration == expected


def test_enabled_imu_invalid_calibration_is_recoverable_input_error(tmp_path) -> None:
    selected_root = tmp_path / "invalid-imu"
    selected_root.mkdir()
    config_path = selected_root / "navmin.json"
    base = _config()
    config = replace(base, imu=replace(base.imu, enabled=True))
    save_config(config_path, config)
    _copy_default_calibrations(selected_root)
    imu_path = selected_root / "calibration" / "imu.json"
    imu_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ApplicationInputError) as exc_info:
        load_application_inputs(LauncherPaths(config=config_path))

    assert exc_info.value.label == "IMU calibration"
    assert exc_info.value.path == imu_path.resolve()
    assert exc_info.value.recoverable


def test_imu_calibration_path_is_owned_by_selected_root_config(tmp_path, monkeypatch) -> None:
    config = _config()
    relative = replace(
        config,
        imu=replace(config.imu, calibration_file="calibration/imu-a.json"),
    )
    selected = tmp_path / "site-a" / "navmin.json"
    assert resolve_imu_calibration_path(selected, relative) == (
        selected.parent / "calibration" / "imu-a.json"
    ).resolve()

    absolute_path = tmp_path / "external" / "imu.json"
    absolute = replace(
        config,
        imu=replace(config.imu, calibration_file=str(absolute_path)),
    )
    assert resolve_imu_calibration_path(selected, absolute) == absolute_path.resolve()


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

    with pytest.raises(ApplicationInputError) as exc_info:
        load_application_inputs(paths)

    assert exc_info.value.label == "overview calibration"
    assert exc_info.value.path == paths.overview_calibration
    assert str(paths.overview_calibration) in str(exc_info.value)
    assert "malformed JSON" in str(exc_info.value)


@pytest.mark.parametrize(
    "flag",
    ["--overview-calibration", "--stereo-calibration", "--log-dir"],
)
def test_public_parser_rejects_removed_operator_overrides(flag: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        normal_parser().parse_args([flag, "unused"])

    assert exc_info.value.code == 2

def test_recovery_uses_canonical_targets_and_leaves_custom_calibration_files_untouched(
    tmp_path,
) -> None:
    selected_root = tmp_path / "test-a"
    selected_root.mkdir()
    custom_dir = tmp_path / "custom-calibrations"
    custom_dir.mkdir()
    custom_overview = custom_dir / "overview-a.json"
    custom_stereo = custom_dir / "stereo-a.json"
    custom_imu = custom_dir / "imu-a.json"
    custom_overview.write_bytes(b"operator-overview-custom")
    custom_stereo.write_bytes(b"operator-stereo-custom")
    custom_imu.write_bytes(b"operator-imu-custom")

    config_path = selected_root / "navmin.json"
    save_config(
        config_path,
        _config_with_calibration_refs(
            str(custom_overview),
            str(custom_stereo),
            str(custom_imu),
        ),
    )
    custom_before = (
        custom_overview.read_bytes(),
        custom_stereo.read_bytes(),
        custom_imu.read_bytes(),
    )

    result = recover_default_inputs(LauncherPaths(config=config_path), now=RECOVERY_TEST_NOW)

    canonical_overview = selected_root / "calibration" / "overview.json"
    canonical_stereo = selected_root / "calibration" / "stereo.json"
    assert result.restored_paths == (
        config_path.resolve(),
        canonical_overview,
        canonical_stereo,
        selected_root / "calibration" / "imu.json",
    )
    assert (
        ImuCalibration.load(selected_root / "calibration" / "imu.json")
        == ImuCalibration()
    )
    assert custom_overview.read_bytes() == custom_before[0]
    assert custom_stereo.read_bytes() == custom_before[1]
    assert custom_imu.read_bytes() == custom_before[2]
    recovered = load_config(config_path)
    assert recovered.vision.cameras.overview.calibration_file == "calibration/overview.json"
    assert recovered.vision.distance.stereo.calibration_file == "calibration/stereo.json"
    load_overview_calibration(canonical_overview)
    load_stereo_calibration(canonical_stereo)


def test_input_recovery_creates_valid_safe_defaults_without_existing_files(
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    result = recover_default_inputs(paths, now=RECOVERY_TEST_NOW)

    assert result.timestamp == "20260923-101530"
    assert result.backups == ()
    imu_path = paths.config.parent / "calibration" / "imu.json"
    assert result.restored_paths == (
        paths.config,
        paths.overview_calibration,
        paths.stereo_calibration,
        imu_path,
    )

    config = load_config(paths.config)
    overview = load_overview_calibration(paths.overview_calibration)
    stereo = load_stereo_calibration(paths.stereo_calibration)
    imu = ImuCalibration.load(imu_path)
    assert config.turret.serial.port == "/dev/navmin-configure-serial-port"
    assert config.schema_version == 4
    assert config.vision.cameras.overview.calibration_file == "calibration/overview.json"
    assert config.vision.distance.stereo.calibration_file == "calibration/stereo.json"
    assert config.imu.calibration_file == "calibration/imu.json"
    assert not config.imu.enabled
    assert imu == ImuCalibration()
    assert isinstance(config.vision.cameras.overview.source, RtpJpegSourceConfig)
    assert config.vision.cameras.overview.source.bind_address == "0.0.0.0"
    assert config.vision.cameras.overview.source.port == 8888
    assert isinstance(config.vision.cameras.stereo_left.source, RtpJpegSourceConfig)
    assert config.vision.cameras.stereo_left.source.port == 8889
    assert config.turret.controller.pid_kp_x == 0.0
    assert config.turret.controller.pid_kp_y == 0.0
    assert config.turret.axes.x.max_relative_move_deg == 1.0
    assert config.turret.axes.y.max_relative_move_deg == 1.0
    assert overview.image_width == 320
    assert overview.image_height == 240
    assert stereo.image_width == 1296
    assert stereo.image_height == 976
    assert stereo.translation_unit == "mm"
    assert stereo.length_scale_to_m == 0.001
    assert stereo.T == (
        -464.858897078744,
        -0.8951708439316364,
        45.641669363174564,
    )


def test_input_recovery_replaces_malformed_root_and_invalid_canonical_calibrations(
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    paths.overview_calibration.parent.mkdir(parents=True)
    paths.config.write_text("{malformed-root", encoding="utf-8")
    paths.overview_calibration.write_text("{invalid-overview", encoding="utf-8")
    paths.stereo_calibration.write_text("{invalid-stereo", encoding="utf-8")

    result = recover_default_inputs(paths, now=RECOVERY_TEST_NOW)

    assert len(result.backups) == 3
    recovered = load_config(paths.config)
    assert recovered.schema_version == 4
    assert recovered.vision.cameras.overview.calibration_file == "calibration/overview.json"
    assert recovered.vision.distance.stereo.calibration_file == "calibration/stereo.json"
    assert load_overview_calibration(paths.overview_calibration).schema_version == 1
    assert load_stereo_calibration(paths.stereo_calibration).schema_version == 1


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


def test_zero_argument_interactive_input_failure_can_restore_defaults(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _recovery_paths(tmp_path)
    shown_results = []
    monkeypatch.setattr("navmin.__main__._DEFAULT_CONFIG", paths.config)
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", paths.log_dir)
    monkeypatch.setattr("navmin.__main__._prompt_input_recovery", lambda _error: True)
    monkeypatch.setattr(
        "navmin.__main__._show_input_recovery_result", shown_results.append
    )
    monkeypatch.setattr(
        "navmin.__main__.run_startup_preflight",
        lambda *args, **kwargs: pytest.fail("preflight must wait for a new launch"),
    )

    assert normal_main([]) == 2
    assert len(shown_results) == 1
    assert paths.config.is_file()
    assert paths.overview_calibration.is_file()
    assert paths.stereo_calibration.is_file()

def test_noninteractive_input_failure_never_opens_recovery_dialog(
    monkeypatch,
    tmp_path,
) -> None:
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", log_dir)
    monkeypatch.setattr(
        "navmin.__main__._prompt_input_recovery",
        lambda _error: pytest.fail("non-interactive launch must not ask questions"),
    )

    assert normal_main(["--config", str(tmp_path / "missing.json"), "-p"]) == 2
    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "input-failed"

def test_noninteractive_missing_config_keeps_input_failure_session_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", log_dir)
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: pytest.fail("runtime must not start"),
    )

    assert normal_main(["--config", str(tmp_path / "missing-config.json")]) == 2
    session = _single_session(log_dir)
    assert (session / "runtime.log").is_file()
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "input-failed"

def test_normal_runtime_failure_and_clean_run_finalize_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
    inputs = _inputs()
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: inputs)

    def run_failure(_inputs, **_kwargs) -> int:
        raise RuntimeError("runtime-boom")

    failed_logs = tmp_path / "failed-logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", failed_logs)
    monkeypatch.setattr("navmin.__main__.run_loaded_application", run_failure)
    assert normal_main(["--config", str(config_path)]) == 2
    failed_session = _single_session(failed_logs)
    failed_manifest = json.loads(
        (failed_session / "manifest.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["status"] == "runtime-failed"
    assert failed_manifest["failure_message"] == "runtime-boom"
    runtime_log = (failed_session / "runtime.log").read_text(encoding="utf-8")
    assert "runtime-boom" in runtime_log
    assert "Traceback (most recent call last)" in runtime_log

    clean_logs = tmp_path / "clean-logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", clean_logs)
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: 0,
    )
    assert normal_main(["--config", str(config_path)]) == 0
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
    config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: pytest.fail(
            "runtime must not start in --preflight-only"
        ),
    )
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", log_dir)

    assert normal_main(["--config", str(config_path), "--preflight-only"]) == 0
    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((session / "preflight.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["exit_code"] == 0
    assert preflight["overall_status"] == "pass"
    assert (session / "runtime.log").is_file()
    assert not (session / "inputs" / "effective-config.json").exists()

def test_planned_localhost_backend_replaces_rtsp_source_with_rtp_jpeg() -> None:
    inputs = _inputs()
    overview = replace(
        inputs.config.vision.cameras.overview,
        source=RtspSourceConfig(
            uri="rtsp://camera.local/stream",
            protocol=RtspProtocol.TCP,
            decoder_mode=RtspDecoderMode.SOFTWARE,
            latency_ms=100,
            drop_on_latency=True,
            buffer_size=3,
        ),
    )
    cameras = replace(inputs.config.vision.cameras, overview=overview)
    config = replace(inputs.config, vision=replace(inputs.config.vision, cameras=cameras))

    planned = _planned_preflight_inputs(
        DiagnosticSelection(
            overview=CameraEndpoint.LOCALHOST,
            stereo_left=CameraEndpoint.REAL,
            turret=TurretEndpoint.PTY,
        ),
        replace(inputs, config=config),
    )

    source = planned.config.vision.cameras.overview.source
    assert isinstance(source, RtpJpegSourceConfig)
    assert source.bind_address == "127.0.0.1"
    assert source.port == 8888
    assert source.buffer_size == 3
    assert planned.config.vision.cameras.stereo_right.enabled is False


def test_diagnostic_preflight_passes_explicit_stereo_right_backend(
    monkeypatch,
    tmp_path,
) -> None:
    inputs = _inputs()
    right = replace(inputs.config.vision.cameras.stereo_right, enabled=False)
    inputs = replace(
        inputs,
        config=replace(
            inputs.config,
            vision=replace(
                inputs.config.vision,
                cameras=replace(inputs.config.vision.cameras, stereo_right=right),
            ),
        ),
    )
    captured: list[tuple[LoadedApplicationInputs, object]] = []

    monkeypatch.setattr(
        "navmin.diagnostic_launcher.load_application_inputs",
        lambda _paths: inputs,
    )

    def preflight(planned, *, selection, session_dir):
        del session_dir
        captured.append((planned, selection))
        return _preflight_report()

    monkeypatch.setattr("navmin.diagnostic_launcher.run_startup_preflight", preflight)

    assert diagnostic_main(
        [
            "--overview",
            "real",
            "--stereo-left",
            "real",
            "--stereo-right",
            "real",
            "--turret",
            "pty",
            "--preflight-only",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    ) == 0

    assert len(captured) == 1
    planned, selection = captured[0]
    assert planned.config.vision.cameras.stereo_right.enabled is True
    assert selection.stereo_right == "real"


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
    assert planned.config.vision.cameras.overview.source.bind_address == "127.0.0.1"
    assert planned.config.vision.cameras.stereo_left.source.bind_address == "127.0.0.1"
    assert planned.config.vision.cameras.stereo_right.enabled is False
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
        config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
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
            lambda _inputs, **_kwargs: pytest.fail(
                "runtime must not start after preflight FAIL"
            ),
        )
        log_dir = tmp_path / "normal-logs"
        monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", log_dir)
        exit_code = normal_main(["--config", str(config_path)])

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
        assert inputs.config.vision.cameras.stereo_right.enabled is False
        assert kwargs["selection"].stereo_right is None
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
    config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
    events: list[str] = []
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "logs")

    def preflight(inputs, **kwargs):
        assert inputs.config.vision.cameras.stereo_right.enabled is True
        assert kwargs["selection"].stereo_right == "real"
        events.append("preflight")
        return _preflight_report()

    def run(_inputs, **kwargs) -> int:
        assert kwargs["overview_source_factory"] is not None
        assert kwargs["stereo_left_source_factory"] is not None
        assert kwargs["stereo_right_source_factory"] is not None
        events.append("runtime")
        return 0

    monkeypatch.setattr("navmin.__main__.run_startup_preflight", preflight)
    monkeypatch.setattr("navmin.__main__.run_loaded_application", run)

    assert normal_main(["--config", str(config_path)]) == 0
    assert events == ["preflight", "runtime"]

def test_normal_launcher_omits_disabled_stereo_right_from_preflight(
    monkeypatch,
    tmp_path,
) -> None:
    config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
    inputs = _inputs()
    cameras = replace(
        inputs.config.vision.cameras,
        stereo_right=replace(inputs.config.vision.cameras.stereo_right, enabled=False),
    )
    disabled_inputs = replace(
        inputs,
        config=replace(
            inputs.config,
            vision=replace(inputs.config.vision, cameras=cameras),
        ),
    )
    monkeypatch.setattr(
        "navmin.__main__.load_application_inputs",
        lambda _paths: disabled_inputs,
    )
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "logs")

    def preflight(effective_inputs, **kwargs):
        assert effective_inputs.config.vision.cameras.stereo_right.enabled is False
        assert kwargs["selection"].stereo_right is None
        return _preflight_report()

    monkeypatch.setattr("navmin.__main__.run_startup_preflight", preflight)
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: 0,
    )

    assert normal_main(["--config", str(config_path)]) == 0

def test_preflight_only_warn_is_success(monkeypatch, tmp_path) -> None:
    config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr(
        "navmin.__main__.run_startup_preflight",
        lambda *args, **kwargs: _preflight_report(PreflightStatus.WARN),
    )
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: pytest.fail(
            "runtime must not start in --preflight-only"
        ),
    )
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", log_dir)

    assert normal_main(["--config", str(config_path), "-p"]) == 0
    preflight = json.loads(
        (_single_session(log_dir) / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["overall_status"] == "warn"

def test_unexpected_preflight_exception_is_runtime_failure_before_runtime(
    monkeypatch,
    tmp_path,
) -> None:
    config_path, _overview_path, _stereo_path = _write_source_placeholders(tmp_path)
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())

    def explode(*args, **kwargs):
        del args, kwargs
        raise KeyError("preflight-bug")

    monkeypatch.setattr("navmin.__main__.run_startup_preflight", explode)
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: pytest.fail(
            "runtime must not start after preflight exception"
        ),
    )
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", log_dir)

    assert normal_main(["--config", str(config_path)]) == 2
    session = _single_session(log_dir)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "runtime-failed"
    assert manifest["failure_type"] == "KeyError"
    assert "preflight-bug" in manifest["failure_message"]
    assert not (session / "preflight.json").exists()
    runtime_log = (session / "runtime.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in runtime_log
    assert "preflight-bug" in runtime_log



@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--overview", "on"),
        ("--overview", "sim"),
        ("--stereo-left", "on"),
        ("--stereo-left", "sim"),
        ("--stereo-right", "on"),
        ("--stereo-right", "sim"),
        ("--stereo-right", "off"),
        ("--stm32", "on"),
        ("--stm32", "sim"),
    ],
)
def test_public_cli_accepts_source_contract(flag: str, value: str) -> None:
    args = normal_parser().parse_args([flag, value])
    attribute = flag.removeprefix("--").replace("-", "_")
    assert getattr(args, attribute) == value


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--overview", "off"),
        ("--stereo-left", "off"),
        ("--stereo-right", "invalid"),
        ("--stm32", "off"),
    ],
)
def test_public_cli_rejects_invalid_source_contract(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        normal_parser().parse_args([flag, value])

    assert exc_info.value.code == 2


def test_help_exits_before_launcher_side_effects(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "navmin.__main__._run_noninteractive",
        lambda _args: pytest.fail("help must not enter non-interactive execution"),
    )
    monkeypatch.setattr(
        "navmin.__main__._run_interactive",
        lambda: pytest.fail("help must not enter interactive execution"),
    )
    monkeypatch.setattr(
        "navmin.__main__.create_host_environment",
        lambda _root: pytest.fail("help must not prepare host networking"),
    )

    with pytest.raises(SystemExit) as exc_info:
        normal_main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--overview {on,sim}" in output
    assert "--stereo-left {on,sim}" in output
    assert "--stereo-right {on,sim,off}" in output
    assert "--stm32 {on,sim}" in output
    assert "--preflight-only" in output
    assert "--verbose" in output
    assert "--config" in output


def _scripted_input(*answers: str):
    iterator = iter(answers)

    def read(_prompt: str) -> str:
        return next(iterator)

    return read


def test_zero_arguments_use_interactive_path(monkeypatch) -> None:
    monkeypatch.setattr("navmin.__main__._run_interactive", lambda: 23)
    monkeypatch.setattr(
        "navmin.__main__._run_noninteractive",
        lambda _args: pytest.fail("zero arguments must not use non-interactive path"),
    )

    assert normal_main([]) == 23


def test_interactive_defaults_come_from_config_and_summary_is_complete() -> None:
    config = _config(emulate_stm32=True)
    config = replace(
        config,
        vision=replace(
            config.vision,
            cameras=replace(
                config.vision.cameras,
                stereo_right=replace(
                    config.vision.cameras.stereo_right,
                    enabled=False,
                ),
            ),
        ),
    )
    output: list[str] = []

    options = _interactive_options(
        config,
        config_path=Path("config/navmin.json"),
        input_fn=_scripted_input("", "", "", "", "", "", ""),
        output_fn=output.append,
    )

    assert options == LaunchOptions(
        overview=CameraSourceMode.ON,
        stereo_left=CameraSourceMode.ON,
        stereo_right=StereoRightMode.OFF,
        stm32=Stm32Mode.SIM,
        preflight_only=False,
        verbose=False,
        config_path=Path("config/navmin.json"),
    )
    text = "\n".join(output)
    assert "Overview      ON" in text
    assert "Stereo Left   ON" in text
    assert "Stereo Right  OFF" in text
    assert "STM32         SIM" in text
    assert "Preflight     OFF" in text
    assert "Logs          NORMAL" in text


def test_interactive_accepts_explicit_source_preflight_and_logging_choices() -> None:
    output: list[str] = []

    options = _interactive_options(
        _config(),
        config_path=Path("config/site.json"),
        input_fn=_scripted_input("sim", "2", "sim", "sim", "on", "verbose", "Y"),
        output_fn=output.append,
    )

    assert options == LaunchOptions(
        overview=CameraSourceMode.SIM,
        stereo_left=CameraSourceMode.SIM,
        stereo_right=StereoRightMode.SIM,
        stm32=Stm32Mode.SIM,
        preflight_only=True,
        verbose=True,
        config_path=Path("config/site.json"),
    )


def test_interactive_invalid_choice_and_confirmation_reprompt() -> None:
    output: list[str] = []

    options = _interactive_options(
        _config(),
        config_path=Path("config/navmin.json"),
        input_fn=_scripted_input(
            "invalid",
            "on",
            "",
            "off",
            "",
            "",
            "",
            "maybe",
            "y",
        ),
        output_fn=output.append,
    )

    assert options is not None
    assert options.stereo_right is StereoRightMode.OFF
    assert "Некорректный выбор. Повторите ввод." in output
    assert "Введите Y, N или нажмите Enter." in output


def test_interactive_n_cancels_without_launch(monkeypatch) -> None:
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr(
        "navmin.__main__._run_noninteractive",
        lambda *_args, **_kwargs: pytest.fail("cancel must not launch"),
    )
    output: list[str] = []

    result = _run_interactive(
        input_fn=_scripted_input("", "", "", "", "", "", "n"),
        output_fn=output.append,
    )

    assert result == 0
    assert output[-1] == "Запуск отменён."


def test_interactive_ctrl_c_cancels_cleanly(monkeypatch) -> None:
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    monkeypatch.setattr(
        "navmin.__main__._run_noninteractive",
        lambda *_args, **_kwargs: pytest.fail("Ctrl+C must not launch"),
    )
    output: list[str] = []

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    assert _run_interactive(input_fn=interrupt, output_fn=output.append) == 0
    assert output[-1] == "Запуск отменён."


def test_interactive_confirmation_launches_through_unified_execution(monkeypatch) -> None:
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: _inputs())
    calls = []

    def run(args, *, origin="non-interactive") -> int:
        calls.append((args, origin))
        return 29

    monkeypatch.setattr("navmin.__main__._run_noninteractive", run)

    result = _run_interactive(
        input_fn=_scripted_input("sim", "sim", "off", "sim", "on", "verbose", ""),
        output_fn=lambda _line: None,
    )

    assert result == 29
    assert len(calls) == 1
    args, origin = calls[0]
    assert origin == "interactive"
    assert args.overview == "sim"
    assert args.stereo_left == "sim"
    assert args.stereo_right == "off"
    assert args.stm32 == "sim"
    assert args.preflight_only is True
    assert args.verbose is True
    assert args.config == Path("config/navmin.json")


def test_any_cli_argument_uses_noninteractive_path(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "navmin.__main__._run_interactive",
        lambda: pytest.fail("CLI arguments must not use interactive path"),
    )

    def run(args) -> int:
        calls.append(args)
        return 17

    monkeypatch.setattr("navmin.__main__._run_noninteractive", run)
    assert normal_main(["-v"]) == 17
    assert len(calls) == 1
    assert calls[0].verbose is True


def test_resolve_launch_options_allows_explicit_mandatory_source_override() -> None:
    config = _config()
    disabled = replace(
        config,
        vision=replace(
            config.vision,
            cameras=replace(
                config.vision.cameras,
                overview=replace(config.vision.cameras.overview, enabled=False),
                stereo_left=replace(config.vision.cameras.stereo_left, enabled=False),
            ),
        ),
    )

    options = resolve_launch_options(
        disabled,
        overview=CameraSourceMode.SIM,
        stereo_left=CameraSourceMode.ON,
    )

    assert options.overview is CameraSourceMode.SIM
    assert options.stereo_left is CameraSourceMode.ON
    with pytest.raises(LaunchSelectionError, match="Overview is disabled"):
        resolve_launch_options(disabled)


def test_noninteractive_source_selection_drives_effective_preflight(
    monkeypatch,
    tmp_path,
) -> None:
    inputs = _inputs()
    captured = []
    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: inputs)
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "logs")

    def preflight(effective_inputs, **kwargs):
        captured.append((effective_inputs, kwargs))
        return _preflight_report(PreflightStatus.WARN)

    monkeypatch.setattr("navmin.__main__.run_startup_preflight", preflight)
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: pytest.fail("preflight-only must not run UI"),
    )

    assert normal_main(
        [
            "--config",
            "config/site.json",
            "--overview",
            "sim",
            "--stereo-right",
            "off",
            "--stm32",
            "sim",
            "-p",
        ]
    ) == 0
    effective_inputs, kwargs = captured[0]
    selection = kwargs["selection"]
    assert selection.overview == "sim"
    assert selection.stereo_left == "real"
    assert selection.stereo_right is None
    assert selection.turret == "sim"
    assert effective_inputs.config.vision.cameras.stereo_right.enabled is False
    assert effective_inputs.config.vision.distance.stereo.stereo_enabled is False
    assert effective_inputs.config.turret.emulate_stm32 is True
    assert kwargs["selection_warnings"]


def test_host_activation_follows_effective_real_camera_selection(
    monkeypatch,
    tmp_path,
) -> None:
    inputs = load_application_inputs(
        LauncherPaths(config=REPO_ROOT / "config" / "defaults" / "navmin.json")
    )
    requirements: list[bool] = []

    class RecordingHost:
        @contextmanager
        def activation(self, *, camera_network_required: bool):
            requirements.append(camera_network_required)
            yield

    monkeypatch.setattr("navmin.__main__.load_application_inputs", lambda _paths: inputs)
    monkeypatch.setattr(
        "navmin.__main__.create_host_environment", lambda _root: RecordingHost()
    )
    monkeypatch.setattr(
        "navmin.__main__.run_loaded_application",
        lambda _inputs, **_kwargs: pytest.fail("preflight-only must not run UI"),
    )

    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "sim-logs")
    assert normal_main(
        [
            "--overview",
            "sim",
            "--stereo-left",
            "sim",
            "--stereo-right",
            "off",
            "--stm32",
            "sim",
            "-p",
        ]
    ) == 0

    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "mixed-logs")
    assert normal_main(
        [
            "--overview",
            "on",
            "--stereo-left",
            "sim",
            "--stereo-right",
            "off",
            "--stm32",
            "sim",
            "-p",
        ]
    ) == 0

    assert requirements == [False, True]


def test_verbose_flag_selects_debug_session_file_logging(monkeypatch, tmp_path) -> None:
    levels = []

    @contextmanager
    def capture_logging(_path, *, level):
        levels.append(level)
        yield _path

    monkeypatch.setattr("navmin.__main__.session_file_logging", capture_logging)
    monkeypatch.setattr(
        "navmin.__main__.load_application_inputs",
        lambda _paths: (_ for _ in ()).throw(ValueError("stop after logging setup")),
    )

    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "normal")
    assert normal_main(["--config", "missing-a.json"]) == 2
    monkeypatch.setattr("navmin.__main__._DEFAULT_LOG_DIR", tmp_path / "verbose")
    assert normal_main(["--config", "missing-b.json", "-v"]) == 2

    assert levels == [logging.INFO, logging.DEBUG]
