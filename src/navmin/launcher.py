"""Thin file/CLI launch support around the shared application composition."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from navmin.calibration import (
    OverviewCalibration,
    StereoCalibration,
    load_overview_calibration,
    load_stereo_calibration,
    stereo_left_horizontal_fov_deg,
)
from navmin.config import AppConfig, ConfigManager, config_to_mapping, load_config
from navmin.contracts import CameraRole
from navmin.diagnostic_logging import diagnostic_events_enabled, emit_event
from navmin.imu import ImuCalibration
from navmin.logging_setup import session_file_logging
from navmin.session_artifacts import SourceInputPaths
from navmin.turret.worker import TransportFactory
from navmin.vision.camera_worker import SourceFactory

if TYPE_CHECKING:
    from navmin.application import ApplicationRuntime
    from navmin.diagnostics.live_video_latency_calibration import (
        LiveVideoLatencyCalibrationResult,
    )


@dataclass(frozen=True)
class LauncherPaths:
    """Filesystem selections owned by launchers rather than ApplicationRuntime.

    Calibration paths are explicit CLI overrides. ``None`` means that the
    corresponding path must be read from the selected root config.
    """

    config: Path
    overview_calibration: Path | None = None
    stereo_calibration: Path | None = None
    log_dir: Path = Path("logs")


@dataclass(frozen=True)
class LoadedApplicationInputs:
    config: AppConfig
    overview_calibration: OverviewCalibration
    stereo_calibration: StereoCalibration
    imu_calibration: ImuCalibration | None = None
    source_paths: SourceInputPaths | None = None


RuntimeBuilder = Callable[..., "ApplicationRuntime"]
UiRunner = Callable[..., int]




def _sha256_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _redacted_serial_port(port: object) -> str | None:
    if not isinstance(port, str):
        return None
    normalized = port.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1] or "serial"


def emit_config_snapshot(
    inputs: LoadedApplicationInputs,
    *,
    overview_backend: str | None,
    stereo_left_backend: str | None,
    stereo_right_backend: str | None,
    turret_backend: str | None,
) -> None:
    """Emit one redacted effective runtime configuration snapshot in VERBOSE."""
    if not diagnostic_events_enabled():
        return
    mapping = config_to_mapping(inputs.config)
    vision = mapping["vision"]
    cameras = vision["cameras"]
    source_paths = inputs.source_paths
    stereo = inputs.stereo_calibration
    nominal_hfov = stereo_left_horizontal_fov_deg(stereo)
    camera_backends = {
        "overview": overview_backend,
        "stereo-left": stereo_left_backend,
        "stereo-right": stereo_right_backend,
    }
    camera_summary = {}
    for role, camera in cameras.items():
        source = camera.get("source", {})
        camera_summary[role] = {
            "backend": camera_backends.get(role),
            "working_resolution": (
                [
                    inputs.overview_calibration.image_width,
                    inputs.overview_calibration.image_height,
                ]
                if role == "overview"
                else [stereo.image_width, stereo.image_height]
            ),
            "processing_enabled": camera.get("processing-enabled"),
            "processor_class": camera.get("vision-processor-class"),
            "processor_profile": camera.get("vision-processor-profile"),
            "endpoint_identity": {
                "type": source.get("type"),
                "port": source.get("port"),
            },
        }
    tracking = mapping["tracking-control"]
    turret = mapping["turret"]
    source_config = None if source_paths is None else source_paths.config
    source_overview = (
        None if source_paths is None else source_paths.overview_calibration
    )
    source_stereo = None if source_paths is None else source_paths.stereo_calibration
    emit_event(
        logging.getLogger(__name__),
        "CONFIG_SNAPSHOT",
        source_instance="launcher:config",
        config_schema_version=mapping["schema-version"],
        config_sha256=_sha256_or_none(source_config),
        overview_calibration_sha256=_sha256_or_none(source_overview),
        stereo_calibration_sha256=_sha256_or_none(source_stereo),
        cameras=camera_summary,
        tracking={
            "pid": turret["controller"],
            "feed_forward": tracking["feed-forward"],
            "prediction": tracking["prediction"],
            "velocity_estimator": tracking["velocity-estimator"],
            "platform_motion": tracking["platform-motion"],
            "stereo_left_nominal_hfov_deg": nominal_hfov,
            "stereo_left_effective_hfov_deg": nominal_hfov,
            "stereo_left_fx_px": float(stereo.P1[0][0]),
            "stereo_left_fy_px": float(stereo.P1[1][1]),
        },
        turret={
            "backend": turret_backend,
            "stm32": turret["stm32"],
            "axes": turret["axes"],
        },
        serial={
            **{key: value for key, value in turret["serial"].items() if key != "port"},
            "port_redacted": _redacted_serial_port(turret["serial"].get("port")),
        },
        imu=mapping.get("imu"),
    )


class ApplicationInputError(ValueError):
    """Identify which local startup input failed while preserving its cause."""

    def __init__(
        self,
        label: str,
        path: Path,
        cause: BaseException,
        *,
        recoverable: bool,
    ) -> None:
        self.label = label
        self.path = path
        self.cause = cause
        self.recoverable = recoverable
        super().__init__(f"{label} ({path}): {cause}")


def resolve_selected_config_path(path: str | Path) -> Path:
    """Resolve a CLI-selected root config path using normal CWD semantics."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (Path.cwd() / candidate).resolve(strict=False)


def _cwd_resolved(path: str | Path) -> Path:
    return resolve_selected_config_path(path)


def _config_owned_path(config_path: Path, reference: str) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (config_path.parent / candidate).resolve(strict=False)


def resolve_imu_calibration_path(
    config_path: str | Path,
    config: AppConfig,
) -> Path:
    """Resolve config-owned IMU calibration using the selected root config."""
    resolved_config = _cwd_resolved(config_path)
    return _config_owned_path(resolved_config, config.imu.calibration_file)


def resolve_application_source_paths(
    paths: LauncherPaths,
    config: AppConfig,
) -> SourceInputPaths:
    """Resolve the one authoritative file set used by loading and evidence."""
    config_path = _cwd_resolved(paths.config)

    overview_ref = config.vision.cameras.overview.calibration_file
    stereo_ref = config.vision.distance.stereo.calibration_file

    overview_path = (
        _cwd_resolved(paths.overview_calibration)
        if paths.overview_calibration is not None
        else _config_owned_path(config_path, overview_ref)
    )
    stereo_path = (
        _cwd_resolved(paths.stereo_calibration)
        if paths.stereo_calibration is not None
        else _config_owned_path(config_path, stereo_ref)
    )
    return SourceInputPaths(
        config=config_path,
        overview_calibration=overview_path,
        stereo_calibration=stereo_path,
    )


def validate_normal_hardware_config(config: AppConfig) -> None:
    """Reject legacy fake-transport configuration in the normal launcher."""
    if config.turret.emulate_stm32:
        raise ValueError(
            "normal launcher requires turret.emulate-stm32=false; "
            "use the diagnostic launcher for PTY operation"
        )


def load_application_inputs(paths: LauncherPaths) -> LoadedApplicationInputs:
    """Load strict typed inputs from one authoritative resolved source-path set."""
    config_path = _cwd_resolved(paths.config)
    try:
        config = load_config(config_path)
    except (OSError, ValueError) as exc:
        raise ApplicationInputError(
            "config", config_path, exc, recoverable=True
        ) from exc

    try:
        source_paths = resolve_application_source_paths(paths, config)
    except ValueError as exc:
        raise ApplicationInputError(
            "config", config_path, exc, recoverable=True
        ) from exc

    try:
        overview = load_overview_calibration(source_paths.overview_calibration)
    except (OSError, ValueError) as exc:
        raise ApplicationInputError(
            "overview calibration",
            source_paths.overview_calibration,
            exc,
            recoverable=paths.overview_calibration is None,
        ) from exc
    try:
        stereo = load_stereo_calibration(source_paths.stereo_calibration)
    except (OSError, ValueError) as exc:
        raise ApplicationInputError(
            "stereo calibration",
            source_paths.stereo_calibration,
            exc,
            recoverable=paths.stereo_calibration is None,
        ) from exc

    imu_calibration = None
    if config.imu.enabled:
        imu_path = resolve_imu_calibration_path(config_path, config)
        try:
            imu_calibration = ImuCalibration.load(imu_path)
        except (OSError, ValueError) as exc:
            raise ApplicationInputError(
                "IMU calibration",
                imu_path,
                exc,
                recoverable=True,
            ) from exc
    return LoadedApplicationInputs(
        config,
        overview,
        stereo,
        imu_calibration=imu_calibration,
        source_paths=source_paths,
    )


def run_loaded_application(
    inputs: LoadedApplicationInputs,
    *,
    qt_argv: Sequence[str] = (),
    show_diagnostic_clock: bool = False,
    runtime_builder: RuntimeBuilder | None = None,
    ui_runner: UiRunner | None = None,
    overview_source_factory: SourceFactory | None = None,
    stereo_left_source_factory: SourceFactory | None = None,
    stereo_right_source_factory: SourceFactory | None = None,
    turret_transport_factory: TransportFactory | None = None,
    video_latency_calibration_result_writer: (
        Callable[[LiveVideoLatencyCalibrationResult], None] | None
    ) = None,
) -> int:
    """Build/start one shared runtime, run Qt, then always perform safe shutdown."""
    using_default_runtime_builder = runtime_builder is None
    if runtime_builder is None:
        from navmin.application import build_application_runtime

        runtime_builder = build_application_runtime
    if ui_runner is None:
        from navmin.ui import run_ui

        ui_runner = run_ui

    runtime_kwargs = {
        "config": inputs.config,
        "overview_calibration": inputs.overview_calibration,
        "stereo_calibration": inputs.stereo_calibration,
        "imu_calibration": inputs.imu_calibration,
    }
    if using_default_runtime_builder and inputs.source_paths is not None:
        runtime_kwargs["motion_calibration_path"] = (
            inputs.source_paths.config.parent / "calibration" / "motion.json"
        )
    source_factories = (
        overview_source_factory,
        stereo_left_source_factory,
        stereo_right_source_factory,
    )
    if any(factory is not None for factory in source_factories):
        if any(factory is None for factory in source_factories):
            raise ValueError("all three camera source factories must be provided together")
        from navmin.application import ApplicationFactories

        runtime_kwargs["factories"] = ApplicationFactories(
            overview_source_factory=overview_source_factory,
            stereo_left_source_factory=stereo_left_source_factory,
            stereo_right_source_factory=stereo_right_source_factory,
            turret_transport_factory=turret_transport_factory,
        )
    elif turret_transport_factory is not None:
        from navmin.application import ApplicationFactories

        runtime_kwargs["factories"] = ApplicationFactories(
            turret_transport_factory=turret_transport_factory,
        )
    runtime = runtime_builder(**runtime_kwargs)
    if video_latency_calibration_result_writer is not None:
        setter = getattr(runtime, "set_video_latency_calibration_result_writer", None)
        if setter is not None:
            setter(video_latency_calibration_result_writer)
    imu_runtime = runtime.imu_runtime
    if imu_runtime is not None and inputs.source_paths is not None:
        imu_calibration_path = resolve_imu_calibration_path(
            inputs.source_paths.config,
            inputs.config,
        )
        imu_runtime.set_calibration_writer(
            lambda calibration, path=imu_calibration_path: calibration.save(path)
        )
    persist_tracking_visualization = None
    persist_ui_config = None
    runtime.start()
    try:
        ui = runtime.ui_dependencies
        apply_aiming_tuning = ui.apply_aiming_tuning
        apply_pid_tuning = ui.apply_pid_tuning
        apply_motion_limits_tuning = ui.apply_motion_limits_tuning
        apply_tracking_control_tuning = ui.apply_tracking_control_tuning
        apply_vision_processor_selection = ui.apply_vision_processor_selection
        if inputs.source_paths is not None:
            config_path = inputs.source_paths.config

            def persist_candidate(transform):
                config_manager = ConfigManager.load(config_path)
                candidate = transform(config_manager.config)
                config_manager.apply_runtime_update(
                    config_to_mapping(candidate),
                    persist=True,
                )

            def persist_tracking_visualization(config):
                persist_candidate(
                    lambda current: replace(
                        current, tracking_visualization=config
                    )
                )

            def persist_ui_config(config):
                persist_candidate(lambda current: replace(current, ui=config))

            def persist_and_apply_aiming(config):
                persist_candidate(lambda current: replace(current, aiming=config))
                ui.apply_aiming_tuning(config)

            def persist_and_apply_pid(config):
                persist_candidate(
                    lambda current: replace(
                        current,
                        turret=replace(current.turret, controller=config),
                    )
                )
                ui.apply_pid_tuning(config)

            def persist_and_apply_motion_limits(config):
                persist_candidate(
                    lambda current: replace(
                        current,
                        turret=replace(current.turret, stm32=config),
                    )
                )
                ui.apply_motion_limits_tuning(config)

            def persist_and_apply_tracking_control(config):
                persist_candidate(
                    lambda current: replace(current, tracking_control=config)
                )
                ui.apply_tracking_control_tuning(config)

            def persist_and_apply_vision_processor(camera, selection):
                def transform(current):
                    cameras = current.vision.cameras
                    if camera is CameraRole.OVERVIEW:
                        cameras = replace(
                            cameras,
                            overview=replace(
                                cameras.overview,
                                vision_processor_class=selection.class_name,
                                vision_processor_profile=selection.profile_name,
                            ),
                        )
                    elif camera is CameraRole.STEREO_LEFT:
                        cameras = replace(
                            cameras,
                            stereo_left=replace(
                                cameras.stereo_left,
                                vision_processor_class=selection.class_name,
                                vision_processor_profile=selection.profile_name,
                            ),
                        )
                    elif camera is CameraRole.STEREO_RIGHT:
                        cameras = replace(
                            cameras,
                            stereo_right=replace(
                                cameras.stereo_right,
                                vision_processor_class=selection.class_name,
                                vision_processor_profile=selection.profile_name,
                            ),
                        )
                    else:
                        raise ValueError(f"unsupported camera role: {camera}")
                    return replace(
                        current,
                        vision=replace(current.vision, cameras=cameras),
                    )

                persist_candidate(transform)
                if ui.apply_vision_processor_selection is None:
                    raise RuntimeError("Vision processor selection is unavailable")
                ui.apply_vision_processor_selection(camera, selection)

            apply_aiming_tuning = persist_and_apply_aiming
            apply_pid_tuning = persist_and_apply_pid
            apply_motion_limits_tuning = persist_and_apply_motion_limits
            apply_tracking_control_tuning = persist_and_apply_tracking_control
            apply_vision_processor_selection = persist_and_apply_vision_processor

        return ui_runner(
            mediator=ui.mediator,
            camera_bindings=ui.camera_bindings,
            turret_states=ui.turret_states,
            imu_states=ui.imu_states,
            imu_controls=ui.imu_controls,
            camera_stale_timeout_ms=ui.camera_stale_timeout_ms,
            aiming_config=ui.aiming_config,
            apply_aiming_tuning=apply_aiming_tuning,
            pid_controller_config=ui.pid_controller_config,
            apply_pid_tuning=apply_pid_tuning,
            stm32_config=ui.stm32_config,
            apply_motion_limits_tuning=apply_motion_limits_tuning,
            tracking_control_config=ui.tracking_control_config,
            apply_tracking_control_tuning=apply_tracking_control_tuning,
            tracking_stereo_left_hfov_deg=getattr(
                ui, "tracking_stereo_left_hfov_deg", None
            ),
            apply_tracking_stereo_left_hfov_deg=getattr(
                ui, "apply_tracking_stereo_left_hfov_deg", None
            ),
            tracking_visualization_config=ui.tracking_visualization_config,
            ui_config=ui.ui_config,
            persist_ui_config=persist_ui_config,
            vision_processor_selections=ui.vision_processor_selections,
            apply_vision_processor_selection=apply_vision_processor_selection,
            recording_runtime=getattr(ui, "recording_runtime", None),
            start_recording=getattr(ui, "start_recording", None),
            motion_calibration_runtime=getattr(
                ui, "motion_calibration_runtime", None
            ),
            begin_motion_calibration_profile=getattr(
                ui, "begin_motion_calibration_profile", None
            ),
            restore_motion_calibration_profile=getattr(
                ui, "restore_motion_calibration_profile", None
            ),
            video_latency_calibration_states=getattr(
                ui, "video_latency_calibration_states", None
            ),
            start_video_latency_calibration=getattr(
                ui, "start_video_latency_calibration", None
            ),
            cancel_video_latency_calibration=getattr(
                ui, "cancel_video_latency_calibration", None
            ),
            persist_tracking_visualization=persist_tracking_visualization,
            show_diagnostic_clock=show_diagnostic_clock,
            argv=qt_argv,
        )
    finally:
        runtime.shutdown()


__all__ = [
    "ApplicationInputError",
    "LauncherPaths",
    "LoadedApplicationInputs",
    "emit_config_snapshot",
    "load_application_inputs",
    "resolve_application_source_paths",
    "resolve_imu_calibration_path",
    "resolve_selected_config_path",
    "run_loaded_application",
    "session_file_logging",
    "validate_normal_hardware_config",
]
