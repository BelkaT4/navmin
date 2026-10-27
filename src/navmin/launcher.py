"""Thin file/CLI launch support around the shared application composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from navmin.calibration import (
    OverviewCalibration,
    StereoCalibration,
    load_overview_calibration,
    load_stereo_calibration,
)
from navmin.config import AppConfig, load_config
from navmin.logging_setup import session_file_logging
from navmin.session_artifacts import SourceInputPaths

if TYPE_CHECKING:
    from navmin.application import ApplicationRuntime


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
    source_paths: SourceInputPaths | None = None


RuntimeBuilder = Callable[..., "ApplicationRuntime"]
UiRunner = Callable[..., int]


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
    return LoadedApplicationInputs(
        config,
        overview,
        stereo,
        source_paths=source_paths,
    )


def run_loaded_application(
    inputs: LoadedApplicationInputs,
    *,
    qt_argv: Sequence[str] = (),
    show_diagnostic_clock: bool = False,
    runtime_builder: RuntimeBuilder | None = None,
    ui_runner: UiRunner | None = None,
    calibration_config_path: Path | None = None,
) -> int:
    """Build/start one shared runtime, run Qt, then always perform safe shutdown."""
    if runtime_builder is None:
        from navmin.application import build_application_runtime

        runtime_builder = build_application_runtime
    if ui_runner is None:
        from navmin.ui import run_ui

        ui_runner = run_ui

    runtime = runtime_builder(
        config=inputs.config,
        overview_calibration=inputs.overview_calibration,
        stereo_calibration=inputs.stereo_calibration,
    )
    runtime.start()
    try:
        ui = runtime.ui_dependencies
        destination = calibration_config_path or (
            inputs.source_paths.config if inputs.source_paths is not None else None
        )

        def save_motion_calibration(axis, estimate):
            from navmin.config import load_config, save_config
            from navmin.core.motion_calibration import corrected_config

            if destination is None:
                raise ValueError("Для этого запуска не задан файл сохранения калибровки")
            if ui.mediator.calibration_pending:
                raise ValueError("Сначала завершите замер")
            baseline = ui.mediator.calibration_axes
            if baseline is None:
                raise ValueError("Нет настроек механики")
            source = load_config(destination) if destination.exists() else inputs.config
            corrected = corrected_config(source, axis, getattr(baseline, axis), estimate)
            # Persist before publishing: a write failure must not change live motion.
            save_config(destination, corrected)
            from dataclasses import replace
            axes = replace(baseline, **{axis: getattr(corrected.turret.axes, axis)})
            runtime.apply_motion_calibration(axes)
            return str(destination)

        return ui_runner(
            mediator=ui.mediator,
            camera_bindings=ui.camera_bindings,
            stereo_right_binding=ui.stereo_right_binding,
            turret_states=ui.turret_states,
            camera_stale_timeout_ms=ui.camera_stale_timeout_ms,
            pid_controller_config=ui.pid_controller_config,
            apply_pid_tuning=ui.apply_pid_tuning,
            tracking_control_config=ui.tracking_control_config,
            apply_tracking_control_tuning=ui.apply_tracking_control_tuning,
            show_diagnostic_clock=show_diagnostic_clock,
            argv=qt_argv,
            save_motion_calibration=save_motion_calibration,
        )
    finally:
        runtime.shutdown()


__all__ = [
    "ApplicationInputError",
    "LauncherPaths",
    "LoadedApplicationInputs",
    "load_application_inputs",
    "resolve_application_source_paths",
    "resolve_selected_config_path",
    "run_loaded_application",
    "session_file_logging",
    "validate_normal_hardware_config",
]
