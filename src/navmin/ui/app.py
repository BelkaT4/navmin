"""Reusable launch boundary for externally composed NavMin dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PyQt6.QtWidgets import QApplication

from navmin.concurrency import LatestValue
from navmin.config.models import (
    PidControllerConfig,
    TrackingControlConfig,
    TrackingVisualizationConfig,
)
from navmin.contracts import CameraRole, TurretState
from navmin.core import Mediator
from navmin.imu import ImuCalibrationControls, ImuRuntimeState
from navmin.recording import RecordingRuntime, RecordingSessionInfo

from .bridge import CameraUiBinding
from .main_window import MainWindow


def run_ui(
    *,
    mediator: Mediator,
    camera_bindings: Mapping[CameraRole, CameraUiBinding],
    turret_states: LatestValue[TurretState],
    camera_stale_timeout_ms: int,
    pid_controller_config: PidControllerConfig,
    apply_pid_tuning: Callable[[PidControllerConfig], None],
    tracking_control_config: TrackingControlConfig,
    apply_tracking_control_tuning: Callable[[TrackingControlConfig], None],
    tracking_visualization_config: TrackingVisualizationConfig,
    recording_runtime: RecordingRuntime | None = None,
    start_recording: Callable[[CameraRole, int, int], RecordingSessionInfo] | None = None,
    persist_tracking_visualization: (
        Callable[[TrackingVisualizationConfig], None] | None
    ) = None,
    imu_states: LatestValue[ImuRuntimeState] | None = None,
    imu_controls: ImuCalibrationControls | None = None,
    show_diagnostic_clock: bool = False,
    argv: Sequence[str] = (),
) -> int:
    """Run only the Qt UI; application composition owns all workers."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(list(argv))
    window = MainWindow(
        mediator=mediator,
        camera_bindings=camera_bindings,
        turret_states=turret_states,
        camera_stale_timeout_ms=camera_stale_timeout_ms,
        pid_controller_config=pid_controller_config,
        apply_pid_tuning=apply_pid_tuning,
        tracking_control_config=tracking_control_config,
        apply_tracking_control_tuning=apply_tracking_control_tuning,
        tracking_visualization_config=tracking_visualization_config,
        recording_runtime=recording_runtime,
        start_recording=start_recording,
        persist_tracking_visualization=persist_tracking_visualization,
        imu_states=imu_states,
        imu_controls=imu_controls,
        show_diagnostic_clock=show_diagnostic_clock,
    )
    window.showFullScreen()
    return application.exec()


__all__ = ["run_ui"]
