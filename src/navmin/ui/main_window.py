"""Prototype-first NavMin main window."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import monotonic_ns

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QCloseEvent, QKeyEvent, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from navmin.concurrency import LatestValue
from navmin.config.models import (
    PidControllerConfig,
    TrackingControlConfig,
    TrackingVisualizationConfig,
)
from navmin.contracts import (
    CameraRole,
    CameraSessionStarted,
    CameraStatus,
    DistanceSource,
    MotorState,
    TargetRef,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.core import (
    DistanceMeasurementMode,
    Mediator,
    TrackingControlVisualization,
)
from navmin.imu import ImuCalibrationControls, ImuRuntimeState
from navmin.recording import (
    RecordingControlPrediction,
    RecordingControlState,
    RecordingPoint,
    RecordingRuntime,
    RecordingSelection,
    RecordingSessionInfo,
    RecordingSessionStatus,
    RecordingState,
)
from navmin.recording_adapter import assemble_recording_frame_packet
from navmin.vision.processor import VisionTrackingDiagnostics

from .bridge import CameraUiBinding, UiStatePump
from .imu_window import ImuWindow
from .operator_window import OperatorWindow
from .video_view import (
    PreparedVisionFrame,
    VideoView,
    prepare_vision_frame,
    project_control_prediction,
)

LOGGER = logging.getLogger(__name__)

_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)
_DISTANCE_MODE_OPTIONS = (
    (DistanceMeasurementMode.CONSTANT, "Константа"),
    (DistanceMeasurementMode.STEREO_EXACT, "Стерео — точная пара"),
    (DistanceMeasurementMode.STEREO_TIMESTAMP, "Стерео — по времени"),
)
_DISTANCE_MODE_INDEX = {
    mode: index for index, (mode, _label) in enumerate(_DISTANCE_MODE_OPTIONS)
}


class _DistanceSlider(QSlider):
    """Keep one standard wheel event equal to one 10 m operator step."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        step = self.singleStep() if delta > 0 else -self.singleStep()
        self.setValue(self.value() + step)
        event.accept()


class _VideoArea(QWidget):
    def __init__(self, main_view: VideoView, preview_view: VideoView) -> None:
        super().__init__()
        self._main_view = main_view
        self._preview_view = preview_view
        main_view.setParent(self)
        preview_view.setParent(self)
        preview_view.raise_()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._main_view.setGeometry(self.rect())
        preview_width = min(360, max(220, self.width() // 4))
        preview_height = min(240, max(150, int(preview_width * 3 / 4)))
        margin = 18
        self._preview_view.setGeometry(
            max(0, self.width() - preview_width - margin),
            max(0, self.height() - preview_height - margin),
            min(preview_width, self.width()),
            min(preview_height, self.height()),
        )
        self._preview_view.raise_()
        super().resizeEvent(event)


@dataclass
class _PendingRecordingFrame:
    result: VisionResult
    diagnostics: VisionTrackingDiagnostics | None
    selected_target: TargetRef | None
    control_prediction: RecordingControlPrediction | None
    control_state: RecordingControlState | None
    selection: RecordingSelection | None = None


class MainWindow(QMainWindow):
    """Minimal operator UI backed exclusively by Mediator-owned state."""

    def __init__(
        self,
        *,
        mediator: Mediator,
        camera_bindings: Mapping[CameraRole, CameraUiBinding],
        turret_states: LatestValue[TurretState],
        camera_stale_timeout_ms: int,
        pid_controller_config: PidControllerConfig,
        apply_pid_tuning: Callable[[PidControllerConfig], None],
        tracking_control_config: TrackingControlConfig,
        apply_tracking_control_tuning: Callable[[TrackingControlConfig], None],
        tracking_visualization_config: TrackingVisualizationConfig | None = None,
        recording_runtime: RecordingRuntime | None = None,
        start_recording: Callable[[CameraRole, int, int], RecordingSessionInfo] | None = None,
        persist_tracking_visualization: (
            Callable[[TrackingVisualizationConfig], None] | None
        ) = None,
        clock_ns: Callable[[], int] = monotonic_ns,
        wall_clock: Callable[[], datetime] = datetime.now,
        show_diagnostic_clock: bool = False,
        imu_states: LatestValue[ImuRuntimeState] | None = None,
        imu_controls: ImuCalibrationControls | None = None,
        frame_preparer: Callable[[VisionResult], PreparedVisionFrame] = prepare_vision_frame,
        start_timer: bool = True,
        start_fullscreen: bool = True,
    ) -> None:
        super().__init__()
        if set(camera_bindings) != set(_NORMAL_CAMERAS):
            raise ValueError("UI requires exactly Overview and Stereo Left bindings")
        if any(camera is not binding.camera for camera, binding in camera_bindings.items()):
            raise ValueError("camera binding keys must match binding.camera")
        self._mediator = mediator
        self._camera_bindings = dict(camera_bindings)
        self._clock_ns = clock_ns
        self._wall_clock = wall_clock
        self._show_diagnostic_clock = show_diagnostic_clock
        self._frame_preparer = frame_preparer
        if tracking_visualization_config is None:
            tracking_visualization_config = TrackingVisualizationConfig(
                show_search_region=False,
                show_best_candidate=False,
                show_control_prediction=False,
            )
        self._tracking_control_config = tracking_control_config
        self._on_apply_tracking_control_tuning = apply_tracking_control_tuning
        self._tracking_visualization_config = tracking_visualization_config
        self._persist_tracking_visualization = persist_tracking_visualization
        self._recording_runtime = recording_runtime or RecordingRuntime(Path("recordings"))
        self._start_recording = start_recording or self._unavailable_recording_start
        self._recording_camera: CameraRole | None = None
        self._recording_pending: _PendingRecordingFrame | None = None
        self._recording_ui_started_ns: int | None = None
        self._recording_failure_reported: BaseException | None = None
        self._prepared_frames: dict[CameraRole, PreparedVisionFrame] = {}
        self._camera_statuses: dict[CameraRole, CameraStatus] = {}
        self._tracking_diagnostics: dict[CameraRole, VisionTrackingDiagnostics] = {}
        self._shown_main_camera = mediator.main_camera
        self._shown_selection = mediator.selected_target
        stale_timeout_ns = camera_stale_timeout_ms * 1_000_000

        preview_camera = self._other_camera(mediator.main_camera)
        self.main_view = VideoView(
            camera=mediator.main_camera,
            preview=False,
            stale_timeout_ns=stale_timeout_ns,
            on_main_click=self._handle_main_click,
        )
        self.preview_view = VideoView(
            camera=preview_camera,
            preview=True,
            stale_timeout_ns=stale_timeout_ns,
            on_swap=self.swap_main_preview,
        )
        self.video_area = _VideoArea(self.main_view, self.preview_view)
        self.main_view.set_tracking_visualization(
            show_search_region=tracking_visualization_config.show_search_region,
            show_best_candidate=tracking_visualization_config.show_best_candidate,
            show_control_prediction=(
                tracking_visualization_config.show_control_prediction
            ),
        )
        self.preview_view.set_tracking_visualization(
            show_search_region=False,
            show_best_candidate=False,
            show_control_prediction=False,
        )

        self.mode_button = QPushButton()
        self.mode_button.setFixedSize(170, 40)
        self.mode_button.clicked.connect(self._request_other_mode)
        self.motor_button = QPushButton()
        self.motor_button.setFixedSize(145, 40)
        self.motor_button.clicked.connect(self._toggle_motor)
        self.connection_label = QLabel()
        self.connection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_label.setFixedSize(170, 40)

        self.distance_panel = QWidget()
        self.distance_panel.setFixedSize(325, 48)
        distance_layout = QVBoxLayout(self.distance_panel)
        distance_layout.setContentsMargins(0, 0, 0, 0)
        distance_layout.setSpacing(0)
        self.distance_title_label = QLabel("ДАЛЬНОСТЬ:")
        self.distance_title_label.setFixedHeight(15)
        distance_layout.addWidget(self.distance_title_label)
        distance_row = QHBoxLayout()
        distance_row.setContentsMargins(0, 0, 0, 0)
        distance_row.setSpacing(5)
        self.distance_mode_combo = QComboBox()
        self.distance_mode_combo.setFixedSize(160, 32)
        for _mode, label in _DISTANCE_MODE_OPTIONS:
            self.distance_mode_combo.addItem(label)
        self.distance_mode_combo.currentIndexChanged.connect(
            self._distance_mode_changed
        )
        self.distance_slider = _DistanceSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(10, 500)
        self.distance_slider.setSingleStep(10)
        self.distance_slider.setPageStep(10)
        self.distance_slider.setFixedSize(100, 32)
        self.distance_slider.valueChanged.connect(self._manual_distance_changed)
        self.distance_value_label = QLabel("—")
        self.distance_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distance_value_label.setFixedSize(50, 32)
        distance_row.addWidget(self.distance_mode_combo)
        distance_row.addWidget(self.distance_slider)
        distance_row.addWidget(self.distance_value_label)
        distance_layout.addLayout(distance_row)
        self._syncing_distance_controls = False

        self.horizon_panel = QWidget()
        self.horizon_panel.setFixedSize(260, 48)
        horizon_layout = QVBoxLayout(self.horizon_panel)
        horizon_layout.setContentsMargins(0, 0, 0, 0)
        horizon_layout.setSpacing(0)
        self.horizon_title_label = QLabel("Горизонт прогноза")
        self.horizon_title_label.setFixedHeight(15)
        horizon_layout.addWidget(self.horizon_title_label)
        horizon_row = QHBoxLayout()
        horizon_row.setContentsMargins(0, 0, 0, 0)
        horizon_row.setSpacing(5)
        self.horizon_slider = QSlider(Qt.Orientation.Horizontal)
        self.horizon_slider.setRange(0, 20)
        self.horizon_slider.setSingleStep(1)
        self.horizon_slider.setPageStep(1)
        self.horizon_slider.setFixedSize(190, 32)
        initial_horizon_step = min(
            20, max(0, round(tracking_control_config.prediction.horizon_ms / 50))
        )
        self.horizon_slider.setValue(initial_horizon_step)
        self.horizon_value_label = QLabel(
            self._format_horizon_seconds(initial_horizon_step * 50)
        )
        self.horizon_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.horizon_value_label.setFixedSize(60, 32)
        horizon_row.addWidget(self.horizon_slider)
        horizon_row.addWidget(self.horizon_value_label)
        horizon_layout.addLayout(horizon_row)
        self.horizon_slider.valueChanged.connect(self._horizon_changed)

        self.recording_panel = QWidget()
        self.recording_panel.setFixedSize(145, 48)
        recording_layout = QHBoxLayout(self.recording_panel)
        recording_layout.setContentsMargins(0, 6, 0, 6)
        recording_layout.setSpacing(5)
        self.recording_button = QPushButton("REC")
        self.recording_button.setFixedSize(55, 36)
        self.recording_button.clicked.connect(self._toggle_recording)
        self.recording_time_label = QLabel("00:00:00")
        self.recording_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_time_label.setFixedSize(82, 36)
        recording_layout.addWidget(self.recording_button)
        recording_layout.addWidget(self.recording_time_label)

        self.now_label = QLabel()
        self.now_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.now_label.setFixedSize(175, 40)
        self.now_label.setVisible(show_diagnostic_clock)
        self.emergency_button = QPushButton("EMERGENCY")
        self.emergency_button.setFixedSize(190, 48)
        self.emergency_button.setStyleSheet(
            "QPushButton { background: #a51616; color: white; font-size: 18px; font-weight: bold; }"
            "QPushButton:pressed { background: #710d0d; }"
        )
        self.emergency_button.clicked.connect(self._mediator.emergency_stop)

        self.operational_bar = QWidget()
        self.operational_bar.setFixedHeight(62)
        bar_layout = QHBoxLayout(self.operational_bar)
        bar_layout.setContentsMargins(10, 7, 10, 7)
        bar_layout.setSpacing(9)
        bar_layout.addWidget(self.mode_button)
        bar_layout.addWidget(self.motor_button)
        bar_layout.addWidget(self.connection_label)
        bar_layout.addWidget(self.now_label)
        bar_layout.addWidget(self.distance_panel)
        bar_layout.addWidget(self.horizon_panel)
        bar_layout.addWidget(self.recording_panel)
        bar_layout.addStretch(1)
        bar_layout.addWidget(self.emergency_button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video_area, 1)
        layout.addWidget(self.operational_bar, 0)
        self.setCentralWidget(central)
        self.setWindowTitle("BelkaT4 / NavMin")
        self.resize(1280, 800)

        self.imu_window = ImuWindow(
            owner=self,
            states=imu_states,
            controls=imu_controls,
            on_toggle_operator=self.toggle_operator_window,
        )
        self.operator_window = OperatorWindow(
            owner=self,
            on_toggle_fullscreen=self.toggle_fullscreen,
            on_exit=self.request_exit,
            on_show_imu=self.show_imu_window,
            pid_controller_config=pid_controller_config,
            on_apply_pid_tuning=apply_pid_tuning,
            tracking_control_config=tracking_control_config,
            on_apply_tracking_control_tuning=self._apply_tracking_control_tuning,
            tracking_visualization_config=tracking_visualization_config,
            on_tracking_visualization_changed=self._set_tracking_visualization,
        )

        self.state_pump = UiStatePump(
            mediator=mediator,
            camera_bindings=self._camera_bindings,
            turret_states=turret_states,
            on_camera_session=self._camera_session_accepted,
            on_vision_result=self._vision_result_accepted,
            on_camera_status=self._camera_status_changed,
            on_turret_state=self._turret_state_changed,
            on_presentation_tick=self.refresh_presentation,
            parent=self,
        )
        self._refresh_turret_controls()
        self._refresh_view_roles()
        self.refresh_presentation()
        if start_timer:
            self.state_pump.start()
        if start_fullscreen:
            self.showFullScreen()
        self.operator_window.set_fullscreen_state(self.isFullScreen())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.toggle_operator_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() is QEvent.Type.WindowStateChange:
            self.operator_window.set_fullscreen_state(self.isFullScreen())

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_recording()
        self.imu_window.close()
        self.operator_window.close()
        super().closeEvent(event)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self.operator_window.set_fullscreen_state(self.isFullScreen())

    def toggle_operator_window(self) -> None:
        if self.operator_window.isVisible():
            self.operator_window.hide()
            return
        self.operator_window.show()
        self.operator_window.raise_()
        self.operator_window.activateWindow()

    def show_imu_window(self) -> None:
        self.imu_window.show()
        self.imu_window.raise_()
        self.imu_window.activateWindow()

    def request_exit(self) -> None:
        if self._confirm_exit():
            self.close()

    def _confirm_exit(self) -> bool:
        dialog = QMessageBox(self.operator_window)
        dialog.setWindowTitle("Выход")
        dialog.setText("Выйти из NavMin?")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes
        )
        cancel_button = dialog.button(QMessageBox.StandardButton.Cancel)
        exit_button = dialog.button(QMessageBox.StandardButton.Yes)
        cancel_button.setText("Отмена")
        exit_button.setText("Выйти")
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return dialog.exec() == QMessageBox.StandardButton.Yes

    def swap_main_preview(self) -> None:
        self._mediator.swap_main_preview()
        self.refresh_presentation()

    def refresh_presentation(self) -> None:
        self.operator_window.set_tracking_diagnostics(
            self._mediator.tracking_diagnostics
        )
        if self._mediator.main_camera is not self._shown_main_camera:
            self._shown_main_camera = self._mediator.main_camera
            self._refresh_view_roles()
        if self._mediator.selected_target != self._shown_selection:
            self._shown_selection = self._mediator.selected_target
            self.main_view.set_selected_target(self._shown_selection)
            self.preview_view.set_selected_target(self._shown_selection)
        self._refresh_distance_presentation()
        self._refresh_aim_point()
        now_ns = self._clock_ns()
        self.main_view.refresh_freshness(now_ns)
        self.preview_view.refresh_freshness(now_ns)
        self._refresh_recording_presentation(now_ns)
        if self._show_diagnostic_clock:
            now = self._wall_clock()
            milliseconds = now.microsecond // 1000
            self.now_label.setText(f"NOW {now:%H:%M:%S}.{milliseconds:03d}")

    @staticmethod
    def _format_horizon_seconds(horizon_ms: int) -> str:
        return f"{horizon_ms / 1000:.2f} с"

    def _apply_tracking_control_tuning(
        self, config: TrackingControlConfig
    ) -> None:
        self._on_apply_tracking_control_tuning(config)
        self._tracking_control_config = config
        if hasattr(self, "operator_window"):
            self.operator_window.set_tracking_control_config(config)
        horizon_step = min(20, max(0, round(config.prediction.horizon_ms / 50)))
        if self.horizon_slider.value() != horizon_step:
            self.horizon_slider.blockSignals(True)
            try:
                self.horizon_slider.setValue(horizon_step)
            finally:
                self.horizon_slider.blockSignals(False)
        self.horizon_value_label.setText(
            self._format_horizon_seconds(config.prediction.horizon_ms)
        )

    def _horizon_changed(self, step: int) -> None:
        horizon_ms = step * 50
        current = self._tracking_control_config
        if current.prediction.horizon_ms == horizon_ms:
            self.horizon_value_label.setText(
                self._format_horizon_seconds(horizon_ms)
            )
            return
        config = replace(
            current,
            prediction=replace(current.prediction, horizon_ms=horizon_ms),
        )

        try:
            self._on_apply_tracking_control_tuning(config)
            self._tracking_control_config = config
            self.operator_window.set_tracking_control_horizon_ms(horizon_ms)
            self.horizon_value_label.setText(
                self._format_horizon_seconds(horizon_ms)
            )
        except (TypeError, ValueError):
            old_step = min(
                20, max(0, round(current.prediction.horizon_ms / 50))
            )
            self.horizon_slider.blockSignals(True)
            try:
                self.horizon_slider.setValue(old_step)
            finally:
                self.horizon_slider.blockSignals(False)
            self.horizon_value_label.setText(
                self._format_horizon_seconds(current.prediction.horizon_ms)
            )
            return

    def _set_tracking_visualization(
        self, config: TrackingVisualizationConfig
    ) -> None:
        if self._persist_tracking_visualization is not None:
            self._persist_tracking_visualization(config)
        self._tracking_visualization_config = config
        self.main_view.set_tracking_visualization(
            show_search_region=config.show_search_region,
            show_best_candidate=config.show_best_candidate,
            show_control_prediction=config.show_control_prediction,
        )
        self.preview_view.set_tracking_visualization(
            show_search_region=False,
            show_best_candidate=False,
            show_control_prediction=False,
        )

    def _distance_mode_changed(self, index: int) -> None:
        if self._syncing_distance_controls:
            return
        if not 0 <= index < len(_DISTANCE_MODE_OPTIONS):
            return
        mode, _label = _DISTANCE_MODE_OPTIONS[index]
        self._mediator.set_distance_mode(mode)
        self._refresh_distance_presentation()

    def _manual_distance_changed(self, value_m: int) -> None:
        if self._syncing_distance_controls:
            return
        self._mediator.set_manual_distance_m(value_m)
        self._refresh_distance_presentation()

    def _refresh_distance_presentation(self) -> None:
        mode = self._mediator.distance_mode
        manual_distance_m = self._mediator.manual_distance_m
        stereo_left_main = self._mediator.main_camera is CameraRole.STEREO_LEFT

        distance = self._mediator.current_distance
        expected_source = (
            DistanceSource.MANUAL
            if mode is DistanceMeasurementMode.CONSTANT
            else DistanceSource.STEREO
        )
        display_distance = (
            distance
            if stereo_left_main
            and distance is not None
            and distance.source is expected_source
            else None
        )

        self._syncing_distance_controls = True
        try:
            self.distance_mode_combo.setCurrentIndex(_DISTANCE_MODE_INDEX[mode])
            self.distance_mode_combo.setEnabled(stereo_left_main)
            self.distance_slider.setValue(manual_distance_m)
            show_slider = stereo_left_main and mode is DistanceMeasurementMode.CONSTANT
            self.distance_slider.setVisible(show_slider)
            self.distance_slider.setEnabled(show_slider)

            if not stereo_left_main:
                self.distance_value_label.setText("—")
            elif mode is DistanceMeasurementMode.CONSTANT:
                self.distance_value_label.setText(f"{manual_distance_m} м")
            else:
                self.distance_value_label.setText(
                    "—"
                    if display_distance is None
                    else f"{display_distance.distance_m:.1f} м"
                )
        finally:
            self._syncing_distance_controls = False

        self.main_view.set_target_distance(display_distance)
        self.preview_view.set_target_distance(None)

    def _refresh_aim_point(self) -> None:
        result = self.main_view.displayed_result
        point = (
            None
            if result is None
            else self._mediator.aiming.aim_point(result.frame.camera, result.frame)
        )
        self.main_view.set_aim_point(point)
        self.preview_view.set_aim_point(None)

    def _camera_session_accepted(self, session: CameraSessionStarted) -> None:
        self._prepared_frames.pop(session.camera, None)
        self._tracking_diagnostics.pop(session.camera, None)
        self._apply_camera_to_views(session.camera)
        self.refresh_presentation()

    def _vision_result_accepted(self, result: VisionResult) -> None:
        prepared = self._frame_preparer(result)
        camera = result.frame.camera
        self._prepared_frames[camera] = prepared
        diagnostics = self._camera_bindings[camera].tracking_diagnostics.get()
        if self._diagnostics_match_result(diagnostics, result):
            assert diagnostics is not None
            self._tracking_diagnostics[camera] = diagnostics
        else:
            self._tracking_diagnostics.pop(camera, None)
        self._apply_camera_to_views(camera)
        self._refresh_recording_presentation(self._clock_ns())
        recording_diagnostics = (
            diagnostics if self._diagnostics_match_result(diagnostics, result) else None
        )
        self._recording_accept_result(result, recording_diagnostics)

    @staticmethod
    def _diagnostics_match_result(
        diagnostics: VisionTrackingDiagnostics | None,
        result: VisionResult,
    ) -> bool:
        return bool(
            diagnostics is not None
            and diagnostics.camera is result.frame.camera
            and diagnostics.generation == result.frame.generation
            and diagnostics.frame_id == result.frame.frame_id
        )

    def _camera_status_changed(self, status: CameraStatus) -> None:
        self._camera_statuses[status.camera] = status
        self._apply_camera_to_views(status.camera)
        self.operator_window.set_camera_status(status.camera, status)

    def _turret_state_changed(self, _state: TurretState) -> None:
        self._refresh_turret_controls()
        self.refresh_presentation()

    def _refresh_view_roles(self) -> None:
        main_camera = self._mediator.main_camera
        preview_camera = self._other_camera(main_camera)
        self.main_view.set_camera(main_camera)
        self.preview_view.set_camera(preview_camera)
        self.main_view.set_prepared_frame(self._prepared_frames.get(main_camera))
        self.main_view.set_selected_target(self._mediator.selected_target)
        self.main_view.set_tracking_diagnostics(
            self._tracking_diagnostics.get(main_camera)
        )
        main_result = self.main_view.displayed_result
        main_control = self._control_visualization_for_result(main_result)
        main_model = (
            None
            if main_result is None
            else self._mediator.session_gate.camera_model(
                main_result.frame.camera,
                main_result.frame.generation,
            )
        )
        self.main_view.set_tracking_control_visualization(main_control, main_model)
        self.preview_view.set_prepared_frame(self._prepared_frames.get(preview_camera))
        self.preview_view.set_selected_target(self._mediator.selected_target)
        self.preview_view.set_tracking_diagnostics(None)
        self.preview_view.set_tracking_control_visualization(None, None)
        self.main_view.set_camera_status(self._camera_statuses.get(main_camera))
        self.preview_view.set_camera_status(self._camera_statuses.get(preview_camera))

    def _apply_camera_to_views(self, camera: CameraRole) -> None:
        for view in (self.main_view, self.preview_view):
            if view.camera is camera:
                view.set_prepared_frame(self._prepared_frames.get(camera))
                view.set_selected_target(self._mediator.selected_target)
                view.set_tracking_diagnostics(
                    self._tracking_diagnostics.get(camera)
                    if view is self.main_view
                    else None
                )
                result = view.displayed_result
                control = (
                    self._control_visualization_for_result(result)
                    if view is self.main_view
                    else None
                )
                model = (
                    None
                    if result is None or view is self.preview_view
                    else self._mediator.session_gate.camera_model(
                        result.frame.camera,
                        result.frame.generation,
                    )
                )
                view.set_tracking_control_visualization(control, model)
                view.set_camera_status(self._camera_statuses.get(camera))

    def _control_visualization_for_result(
        self,
        result: VisionResult | None,
    ) -> TrackingControlVisualization | None:
        visualization = self._mediator.tracking_control_visualization
        selected = self._mediator.selected_target
        if (
            result is None
            or visualization is None
            or selected is None
            or visualization.target != selected
            or visualization.camera is not result.frame.camera
            or visualization.generation != result.frame.generation
            or visualization.frame_id != result.frame.frame_id
        ):
            return None
        return visualization

    def _handle_main_click(
        self,
        displayed_result: VisionResult,
        x_px: float,
        y_px: float,
    ) -> None:
        if displayed_result is not self.main_view.displayed_result:
            return
        if not self.main_view.interaction_allowed(
            self._mediator.session_gate,
            self._clock_ns(),
        ):
            return
        mode = self._mediator.turret_state.control_mode
        if mode is TurretControlMode.RELATIVE:
            self._mediator.handle_relative_click(displayed_result, x_px, y_px)
        elif mode is TurretControlMode.TRACKING:
            self._record_selection_for_pending(displayed_result, x_px, y_px)
            self._mediator.request_target_selection(
                displayed_result,
                x_px,
                y_px,
            )
        self.refresh_presentation()

    @staticmethod
    def _unavailable_recording_start(
        _camera: CameraRole,
        _width: int,
        _height: int,
    ) -> RecordingSessionInfo:
        raise RuntimeError("recording is not connected to ApplicationRuntime")

    def _toggle_recording(self) -> None:
        state = self._recording_runtime.state
        if state is RecordingState.RECORDING:
            self._stop_recording()
            return
        if state is RecordingState.FAILED:
            self._stop_recording()
        result = self.main_view.displayed_result
        if result is None:
            LOGGER.warning("Recording start ignored because main camera has no frame")
            return
        camera = result.frame.camera
        height, width = result.frame.image.shape[:2]
        try:
            self._start_recording(camera, width, height)
        except (OSError, RuntimeError, ValueError):
            LOGGER.exception("Recording start failed")
            self._recording_camera = None
            self._recording_pending = None
            self._recording_ui_started_ns = None
            return
        self._recording_camera = camera
        self._recording_pending = None
        self._recording_ui_started_ns = self._clock_ns()
        self._recording_failure_reported = None
        LOGGER.info("Recording started camera=%s", camera.value)
        self._refresh_recording_presentation(self._clock_ns())

    def _stop_recording(self) -> None:
        state = self._recording_runtime.state
        if state is RecordingState.IDLE:
            self._recording_camera = None
            self._recording_pending = None
            self._recording_ui_started_ns = None
            return
        if state is RecordingState.RECORDING:
            self._flush_recording_pending()
        try:
            status = self._recording_runtime.stop()
        except RuntimeError:
            LOGGER.exception("Recording stop failed")
            status = RecordingSessionStatus.FAILED
        if status is RecordingSessionStatus.FAILED:
            failure = self._recording_runtime.poll_failure()
            LOGGER.error("Recording session failed: %s", failure or "unknown failure")
        else:
            LOGGER.info("Recording stopped")
        self._recording_camera = None
        self._recording_pending = None
        self._recording_ui_started_ns = None
        self._refresh_recording_presentation(self._clock_ns())

    def _refresh_recording_presentation(self, now_ns: int) -> None:
        state = self._recording_runtime.state
        if state is RecordingState.FAILED:
            failure = self._recording_runtime.poll_failure()
            if failure is not self._recording_failure_reported:
                self._recording_failure_reported = failure
                LOGGER.error("Recording writer failed: %s", failure or "unknown failure")
            self._stop_recording()
            state = self._recording_runtime.state
        if state is RecordingState.RECORDING:
            self.recording_button.setText("STOP")
            self.recording_button.setEnabled(True)
            started_ns = self._recording_ui_started_ns
            elapsed_ns = 0 if started_ns is None else max(0, now_ns - started_ns)
            total_seconds = elapsed_ns // 1_000_000_000
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.recording_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            return
        self.recording_button.setText("REC")
        self.recording_button.setEnabled(self.main_view.displayed_result is not None)
        self.recording_time_label.setText("00:00:00")

    def _recording_accept_result(
        self,
        result: VisionResult,
        diagnostics: VisionTrackingDiagnostics | None,
    ) -> None:
        if (
            self._recording_runtime.state is not RecordingState.RECORDING
            or self._recording_camera is not result.frame.camera
        ):
            return
        self._flush_recording_pending()
        control = self._control_visualization_for_result(result)
        control_prediction = None
        control_state = None
        if control is not None:
            model = self._mediator.session_gate.camera_model(
                result.frame.camera, result.frame.generation
            )
            projected = (
                None if model is None else project_control_prediction(control, model)
            )
            if projected is not None:
                control_prediction = RecordingControlPrediction(
                    from_point=RecordingPoint(*projected.observed_px),
                    to_point=RecordingPoint(*projected.predicted_px),
                )
            control_state = RecordingControlState(
                prediction_enabled=self._tracking_control_config.prediction.enabled,
                rate_valid_x=control.rate_valid_x,
                rate_valid_y=control.rate_valid_y,
            )
        self._recording_pending = _PendingRecordingFrame(
            result=result,
            diagnostics=diagnostics,
            selected_target=self._mediator.selected_target,
            control_prediction=control_prediction,
            control_state=control_state,
        )

    def _record_selection_for_pending(
        self,
        result: VisionResult,
        x_px: float,
        y_px: float,
    ) -> None:
        pending = self._recording_pending
        if (
            pending is None
            or self._recording_runtime.state is not RecordingState.RECORDING
            or pending.result is not result
        ):
            return
        pending.selection = RecordingSelection(RecordingPoint(float(x_px), float(y_px)))

    def _flush_recording_pending(self) -> None:
        pending = self._recording_pending
        self._recording_pending = None
        session = self._recording_runtime.session
        if pending is None or session is None:
            return
        packet = assemble_recording_frame_packet(
            result=pending.result,
            session_started_monotonic_ns=session.started_monotonic_ns,
            diagnostics=pending.diagnostics,
            selected_target=pending.selected_target,
            selection=pending.selection,
            control_prediction=pending.control_prediction,
            control_state=pending.control_state,
        )
        try:
            self._recording_runtime.submit(packet)
        except (TypeError, ValueError) as error:
            LOGGER.error("Recording packet rejected: %s", error)

    def _request_other_mode(self) -> None:
        current = self._mediator.turret_state.control_mode
        requested = (
            TurretControlMode.TRACKING
            if current is TurretControlMode.RELATIVE
            else TurretControlMode.RELATIVE
        )
        self._mediator.request_control_mode(requested)
        self._refresh_turret_controls()
        self.refresh_presentation()

    def _toggle_motor(self) -> None:
        motor = self._mediator.turret_state.motor_state
        if motor is MotorState.ON:
            self._mediator.motor_off()
        elif motor is MotorState.OFF:
            self._mediator.motor_on()

    def _refresh_turret_controls(self) -> None:
        state = self._mediator.turret_state
        pending = self._mediator.pending_control_mode
        text = f"РЕЖИМ: {state.control_mode.value.upper()}"
        if pending is not None:
            text += f" → {pending.value.upper()}"
        self.mode_button.setText(text)
        self.mode_button.setEnabled(pending is None)
        self.motor_button.setText(f"МОТОР: {state.motor_state.value.upper()}")
        self.motor_button.setEnabled(state.motor_state is not MotorState.UNKNOWN)
        self.connection_label.setText(
            f"СВЯЗЬ: {state.connection_state.value.upper()}"
        )
        self.operator_window.set_turret_state(state, pending)

    @staticmethod
    def _other_camera(camera: CameraRole) -> CameraRole:
        return (
            CameraRole.STEREO_LEFT
            if camera is CameraRole.OVERVIEW
            else CameraRole.OVERVIEW
        )


__all__ = ["MainWindow"]
