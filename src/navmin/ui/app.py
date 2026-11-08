"""Reusable launch boundary for externally composed NavMin dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PyQt6.QtWidgets import QApplication

from navmin.concurrency import LatestValue
from navmin.config.models import (
    AimingConfig,
    PidControllerConfig,
    Stm32Config,
    TrackingControlConfig,
    TrackingVisualizationConfig,
    UiConfig,
)
from navmin.contracts import CameraRole, TurretState
from navmin.core import Mediator
from navmin.imu import ImuCalibrationControls, ImuRuntimeState
from navmin.motion_calibration import MotionCalibrationRuntime, MotionProfile
from navmin.recording import RecordingRuntime, RecordingSessionInfo
from navmin.vision.processor import VisionProcessorSelection

from .bridge import CameraUiBinding
from .main_window import MainWindow


def run_ui(
    *,
    mediator: Mediator,
    camera_bindings: Mapping[CameraRole, CameraUiBinding],
    turret_states: LatestValue[TurretState],
    camera_stale_timeout_ms: int,
    aiming_config: AimingConfig,
    apply_aiming_tuning: Callable[[AimingConfig], None],
    pid_controller_config: PidControllerConfig,
    apply_pid_tuning: Callable[[PidControllerConfig], None],
    stm32_config: Stm32Config,
    apply_motion_limits_tuning: Callable[[Stm32Config], None],
    tracking_control_config: TrackingControlConfig,
    apply_tracking_control_tuning: Callable[[TrackingControlConfig], None],
    tracking_visualization_config: TrackingVisualizationConfig,
    ui_config: UiConfig,
    persist_ui_config: Callable[[UiConfig], None] | None = None,
    vision_processor_selections: Mapping[CameraRole, VisionProcessorSelection] | None = None,
    apply_vision_processor_selection: (
        Callable[[CameraRole, VisionProcessorSelection], None] | None
    ) = None,
    recording_runtime: RecordingRuntime | None = None,
    start_recording: Callable[[CameraRole, int, int], RecordingSessionInfo] | None = None,
    motion_calibration_runtime: MotionCalibrationRuntime | None = None,
    begin_motion_calibration_profile: Callable[[MotionProfile], None] | None = None,
    restore_motion_calibration_profile: Callable[[], None] | None = None,
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
        aiming_config=aiming_config,
        apply_aiming_tuning=apply_aiming_tuning,
        pid_controller_config=pid_controller_config,
        apply_pid_tuning=apply_pid_tuning,
        stm32_config=stm32_config,
        apply_motion_limits_tuning=apply_motion_limits_tuning,
        tracking_control_config=tracking_control_config,
        apply_tracking_control_tuning=apply_tracking_control_tuning,
        tracking_visualization_config=tracking_visualization_config,
        ui_config=ui_config,
        persist_ui_config=persist_ui_config,
        vision_processor_selections=vision_processor_selections,
        apply_vision_processor_selection=apply_vision_processor_selection,
        recording_runtime=recording_runtime,
        start_recording=start_recording,
        motion_calibration_runtime=motion_calibration_runtime,
        begin_motion_calibration_profile=begin_motion_calibration_profile,
        restore_motion_calibration_profile=restore_motion_calibration_profile,
        persist_tracking_visualization=persist_tracking_visualization,
        imu_states=imu_states,
        imu_controls=imu_controls,
        show_diagnostic_clock=show_diagnostic_clock,
    )
    window.showFullScreen()
    return application.exec()


__all__ = ["run_ui"]
