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

if TYPE_CHECKING:
    from navmin.application import ApplicationRuntime


@dataclass(frozen=True)
class LauncherPaths:
    """Filesystem inputs owned by launchers rather than ApplicationRuntime."""

    config: Path
    overview_calibration: Path
    stereo_calibration: Path
    log_dir: Path


@dataclass(frozen=True)
class LoadedApplicationInputs:
    config: AppConfig
    overview_calibration: OverviewCalibration
    stereo_calibration: StereoCalibration


RuntimeBuilder = Callable[..., "ApplicationRuntime"]
UiRunner = Callable[..., int]


def validate_normal_hardware_config(config: AppConfig) -> None:
    """Reject legacy fake-transport configuration in the normal launcher."""
    if config.turret.emulate_stm32:
        raise ValueError(
            "normal launcher requires turret.emulate-stm32=false; "
            "use the diagnostic launcher for PTY operation"
        )


def load_application_inputs(paths: LauncherPaths) -> LoadedApplicationInputs:
    """Load the strict typed inputs consumed by ``build_application_runtime``."""
    config = load_config(paths.config)
    overview = load_overview_calibration(paths.overview_calibration)
    stereo = load_stereo_calibration(paths.stereo_calibration)
    return LoadedApplicationInputs(config, overview, stereo)


def run_loaded_application(
    inputs: LoadedApplicationInputs,
    *,
    qt_argv: Sequence[str] = (),
    runtime_builder: RuntimeBuilder | None = None,
    ui_runner: UiRunner | None = None,
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
        return ui_runner(
            mediator=ui.mediator,
            camera_bindings=ui.camera_bindings,
            turret_states=ui.turret_states,
            camera_stale_timeout_ms=ui.camera_stale_timeout_ms,
            argv=qt_argv,
        )
    finally:
        runtime.shutdown()


__all__ = [
    "LauncherPaths",
    "LoadedApplicationInputs",
    "load_application_inputs",
    "run_loaded_application",
    "session_file_logging",
    "validate_normal_hardware_config",
]
