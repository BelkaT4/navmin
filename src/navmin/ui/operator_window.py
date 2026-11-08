"""Modeless runtime settings/status window owned by the NavMin main window."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from navmin.config.models import (
    PidControllerConfig,
    Stm32Config,
    TrackingControlConfig,
    TrackingVisualizationConfig,
)
from navmin.contracts import CameraRole, CameraStatus, TurretControlMode, TurretState
from navmin.core import AxisTrackingEstimate, TrackingEstimatorDiagnostics
from navmin.vision.processor import (
    DEFAULT_VISION_PROCESSOR_CLASS,
    VT11_VISION_PROCESSOR_CLASS,
    VisionProcessorSelection,
    vision_processor_profiles,
)

_CAMERA_NAMES = {
    CameraRole.OVERVIEW: "Обзорная",
    CameraRole.STEREO_LEFT: "Левая",
    CameraRole.STEREO_RIGHT: "Правая",
}
_CAMERA_ORDER = (
    CameraRole.OVERVIEW,
    CameraRole.STEREO_LEFT,
    CameraRole.STEREO_RIGHT,
)
_PROCESSOR_LABELS = {
    DEFAULT_VISION_PROCESSOR_CLASS: "Legacy14",
    VT11_VISION_PROCESSOR_CLASS: "VT11",
}


class OperatorWindow(QWidget):
    """Modeless settings/status window with MainWindow as transient owner."""

    def __init__(
        self,
        *,
        owner: QWidget,
        on_toggle_fullscreen: Callable[[], None],
        on_exit: Callable[[], None],
        on_show_imu: Callable[[], None],
        on_show_aim_point: Callable[[], None],
        on_reset_aim_point: Callable[[CameraRole], None],
        show_stereo_right_camera: bool,
        on_show_stereo_right_camera_changed: Callable[[bool], None],
        pid_controller_config: PidControllerConfig,
        on_apply_pid_tuning: Callable[[PidControllerConfig], None],
        stm32_config: Stm32Config,
        on_apply_motion_limits_tuning: Callable[[Stm32Config], None],
        tracking_control_config: TrackingControlConfig,
        on_apply_tracking_control_tuning: Callable[[TrackingControlConfig], None],
        tracking_visualization_config: TrackingVisualizationConfig,
        on_tracking_visualization_changed: Callable[[TrackingVisualizationConfig], None],
        use_camera_time_for_tracking: bool,
        on_use_camera_time_for_tracking_changed: Callable[[bool], None],
        vision_processor_selections: Mapping[CameraRole, VisionProcessorSelection] | None = None,
        on_apply_vision_processor_selection: (
            Callable[[CameraRole, VisionProcessorSelection], None] | None
        ) = None,
        available_cameras: tuple[CameraRole, ...] = (
            CameraRole.OVERVIEW,
            CameraRole.STEREO_LEFT,
        ),
        on_show_motion_calibration: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._on_toggle_fullscreen = on_toggle_fullscreen
        self._on_exit = on_exit
        self._on_show_imu = on_show_imu
        self._on_show_aim_point = on_show_aim_point
        self._on_reset_aim_point = on_reset_aim_point
        self._on_show_stereo_right_camera_changed = on_show_stereo_right_camera_changed
        self._syncing_show_right_camera = False
        self._on_apply_pid_tuning = on_apply_pid_tuning
        self._stm32_config = stm32_config
        self._on_apply_motion_limits_tuning = on_apply_motion_limits_tuning
        self._tracking_control_config = tracking_control_config
        self._on_apply_tracking_control_tuning = on_apply_tracking_control_tuning
        self._tracking_visualization_config = tracking_visualization_config
        self._on_tracking_visualization_changed = on_tracking_visualization_changed
        self._on_use_camera_time_for_tracking_changed = (
            on_use_camera_time_for_tracking_changed
        )
        self._vision_processor_selections = dict(vision_processor_selections or {})
        self._on_apply_vision_processor_selection = on_apply_vision_processor_selection
        self._available_cameras = set(available_cameras)
        self._on_show_motion_calibration = on_show_motion_calibration
        self._syncing_tracking_visualization = False

        self.setWindowTitle("NavMin — настройки")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.camera_labels = {
            camera: QLabel(
                f"{_CAMERA_NAMES[camera]}: "
                f"{'NO STATUS' if camera in self._available_cameras else 'ОТКЛЮЧЕНА'}"
            )
            for camera in _CAMERA_ORDER
        }
        self.controller_label = QLabel("Контроллер: DISCONNECTED")
        self.mode_label = QLabel("Режим: RELATIVE")
        self.motor_label = QLabel("Мотор: UNKNOWN")
        self.show_right_camera_checkbox = QCheckBox("Показывать правую камеру")
        self.show_right_camera_checkbox.setChecked(show_stereo_right_camera)
        self.show_right_camera_checkbox.setEnabled(
            CameraRole.STEREO_RIGHT in self._available_cameras
        )
        self.show_right_camera_checkbox.stateChanged.connect(
            self._apply_show_right_camera
        )
        self.camera_error_label = QLabel("")
        self.camera_error_label.setWordWrap(True)
        self.use_camera_time_for_tracking_checkbox = QCheckBox(
            "Использовать время камеры в управлении сопровождением"
        )
        self.use_camera_time_for_tracking_checkbox.setChecked(
            use_camera_time_for_tracking
        )
        self.use_camera_time_for_tracking_checkbox.setToolTip(
            "OFF: управление использует прежнее PC receive time. "
            "ON: после прогрева ClockMapper управление использует временную шкалу "
            "камеры, привязанную к receive time; абсолютная задержка камеры пока "
            "не компенсируется."
        )
        self.use_camera_time_for_tracking_checkbox.stateChanged.connect(
            self._apply_camera_time_for_tracking
        )
        self.camera_time_status_label = QLabel(
            "Время камеры: используется только для диагностики"
        )
        self.camera_time_status_label.setWordWrap(True)

        self._build_pid_controls(pid_controller_config)
        self._build_feed_forward_controls(tracking_control_config)
        self._build_motion_limits_controls(stm32_config)
        self._build_tracking_controls(tracking_control_config)
        self._build_visualization_controls(tracking_visualization_config)
        self._build_detector_controls()
        self._build_diagnostics_controls()
        self._build_action_controls()
        self._build_layout()

        self.setMinimumSize(720, 560)
        self.resize(880, 720)

    def _build_pid_controls(self, config: PidControllerConfig) -> None:
        self.pid_inputs = {
            "pid_kp_x": self._non_negative_spinbox(config.pid_kp_x, step=0.01),
            "pid_ki_x": self._non_negative_spinbox(config.pid_ki_x, step=0.01),
            "pid_kd_x": self._non_negative_spinbox(config.pid_kd_x, step=0.01),
            "pid_kp_y": self._non_negative_spinbox(config.pid_kp_y, step=0.01),
            "pid_ki_y": self._non_negative_spinbox(config.pid_ki_y, step=0.01),
            "pid_kd_y": self._non_negative_spinbox(config.pid_kd_y, step=0.01),
        }
        self.pid_apply_button = QPushButton("Применить PID")
        self.pid_apply_button.clicked.connect(self._apply_pid_tuning)
        self.pid_status_label = QLabel("Статус PID: —")
        self.pid_status_label.setWordWrap(True)
        self.pid_error_label = QLabel("")
        self.pid_error_label.setWordWrap(True)

    def _build_feed_forward_controls(self, config: TrackingControlConfig) -> None:
        self.ff_enabled_checkbox = QCheckBox("Использовать Feed Forward")
        self.ff_gain_x = self._tracking_gain_spinbox(config.feed_forward.gain_x)
        self.ff_gain_y = self._tracking_gain_spinbox(config.feed_forward.gain_y)
        self.ff_apply_button = QPushButton("Применить FF")
        self.ff_apply_button.clicked.connect(self._apply_feed_forward_tuning)
        self.ff_status_label = QLabel("Статус FF: —")
        self.ff_status_label.setWordWrap(True)
        self.ff_error_label = QLabel("")
        self.ff_error_label.setWordWrap(True)

    def _build_motion_limits_controls(self, config: Stm32Config) -> None:
        self.motion_limit_inputs = {
            "max_speed_x_deg_s": self._positive_spinbox(
                config.max_speed_x_deg_s, step=5.0, suffix=" °/с"
            ),
            "max_speed_y_deg_s": self._positive_spinbox(
                config.max_speed_y_deg_s, step=5.0, suffix=" °/с"
            ),
            "acceleration_x_deg_s2": self._positive_spinbox(
                config.acceleration_x_deg_s2, step=5.0, suffix=" °/с²"
            ),
            "acceleration_y_deg_s2": self._positive_spinbox(
                config.acceleration_y_deg_s2, step=5.0, suffix=" °/с²"
            ),
        }
        self.motion_limits_apply_button = QPushButton("Применить ограничения")
        self.motion_limits_apply_button.clicked.connect(self._apply_motion_limits_tuning)
        self.motion_limits_status_label = QLabel("Статус ограничений: —")
        self.motion_limits_status_label.setWordWrap(True)
        self.motion_limits_error_label = QLabel("")
        self.motion_limits_error_label.setWordWrap(True)

    def _build_tracking_controls(self, config: TrackingControlConfig) -> None:
        self.prediction_enabled_checkbox = QCheckBox(
            "Использовать прогноз для управления"
        )
        self.tracking_inputs = {
            "window_ms": QSpinBox(),
        }
        window = self.tracking_inputs["window_ms"]
        window.setRange(1, 10_000)
        window.setSingleStep(10)
        window.setSuffix(" мс")
        window.setValue(config.velocity_estimator.window_ms)
        window.setMaximumWidth(150)
        window.setToolTip(
            "Интервал истории наблюдений для оценки скорости цели. "
            "Меньшее значение быстрее реагирует, большее лучше сглаживает шум."
        )
        self.tracking_apply_button = QPushButton("Применить настройки отслеживания")
        self.tracking_apply_button.clicked.connect(self._apply_tracking_tuning)
        self.tracking_status_label = QLabel("Статус отслеживания: —")
        self.tracking_status_label.setWordWrap(True)
        self.tracking_error_label = QLabel("")
        self.tracking_error_label.setWordWrap(True)
        self._sync_tracking_controls(config)

    def _build_visualization_controls(
        self, config: TrackingVisualizationConfig
    ) -> None:
        self.show_search_region_checkbox = QCheckBox("Показывать область поиска")
        self.show_best_candidate_checkbox = QCheckBox("Показывать лучший кандидат")
        self.show_control_prediction_checkbox = QCheckBox("Показывать прогноз")
        self._sync_tracking_visualization_controls(config)
        self.show_search_region_checkbox.stateChanged.connect(
            self._apply_tracking_visualization
        )
        self.show_best_candidate_checkbox.stateChanged.connect(
            self._apply_tracking_visualization
        )
        self.show_control_prediction_checkbox.stateChanged.connect(
            self._apply_tracking_visualization
        )

    def _build_detector_controls(self) -> None:
        self.detector_controls: dict[CameraRole, dict[str, object]] = {}
        for camera in _CAMERA_ORDER:
            selection = self._vision_processor_selections.get(camera)
            detector = QComboBox()
            detector.addItem("Legacy14", DEFAULT_VISION_PROCESSOR_CLASS)
            detector.addItem("VT11", VT11_VISION_PROCESSOR_CLASS)
            if selection is not None:
                index = detector.findData(selection.class_name)
                if index >= 0:
                    detector.setCurrentIndex(index)

            profile = QComboBox()
            self._populate_profile_combo(
                profile,
                str(detector.currentData()),
                None if selection is None else selection.profile_name,
            )
            apply_button = QPushButton("Применить")
            status = QLabel("—")
            status.setWordWrap(True)
            available = (
                camera in self._available_cameras
                and self._on_apply_vision_processor_selection is not None
                and selection is not None
            )
            detector.setEnabled(available)
            profile.setEnabled(available)
            apply_button.setEnabled(available)
            detector.currentIndexChanged.connect(
                lambda _index, cam=camera: self._detector_changed(cam)
            )
            apply_button.clicked.connect(
                lambda _checked=False, cam=camera: self._apply_detector_selection(cam)
            )
            if not available:
                status.setText(
                    "Камера отключена"
                    if camera not in self._available_cameras
                    else "Недоступно"
                )
            self.detector_controls[camera] = {
                "detector": detector,
                "profile": profile,
                "apply": apply_button,
                "status": status,
            }

    @staticmethod
    def _populate_profile_combo(
        profile: QComboBox,
        class_name: str,
        preferred: str | None,
    ) -> None:
        profile.blockSignals(True)
        try:
            profile.clear()
            profiles = vision_processor_profiles(class_name)
            for profile_name in profiles:
                profile.addItem(profile_name, profile_name)
            if preferred is not None:
                index = profile.findData(preferred)
                if index >= 0:
                    profile.setCurrentIndex(index)
        finally:
            profile.blockSignals(False)

    def _build_diagnostics_controls(self) -> None:
        self.platform_source_label = QLabel("Platform source: —")
        self.tracking_diagnostic_labels = {
            "relative_rate_x": QLabel("—"),
            "relative_rate_y": QLabel("—"),
            "platform_rate_x": QLabel("—"),
            "platform_rate_y": QLabel("—"),
            "target_rate_x": QLabel("—"),
            "target_rate_y": QLabel("—"),
            "fit_rms_x": QLabel("—"),
            "fit_rms_y": QLabel("—"),
            "sample_count_x": QLabel("0"),
            "sample_count_y": QLabel("0"),
            "actual_span_x": QLabel("0 ms"),
            "actual_span_y": QLabel("0 ms"),
            "validity_x": QLabel("INVALID"),
            "validity_y": QLabel("INVALID"),
            "prediction_offset_x": QLabel("0 deg"),
            "prediction_offset_y": QLabel("0 deg"),
        }

    def _build_action_controls(self) -> None:
        self.aim_point_button = QPushButton("ПРИЦЕЛЬНАЯ ТОЧКА")
        self.aim_point_button.clicked.connect(self._on_show_aim_point)
        self.aim_point_reset_buttons = {
            CameraRole.OVERVIEW: QPushButton("Сброс обзорной"),
            CameraRole.STEREO_LEFT: QPushButton("Сброс левой"),
            CameraRole.STEREO_RIGHT: QPushButton("Сброс правой"),
        }
        for camera, button in self.aim_point_reset_buttons.items():
            button.clicked.connect(
                lambda _checked=False, cam=camera: self._on_reset_aim_point(cam)
            )
        self.position_button = QPushButton("GY-85 — ПОЛОЖЕНИЕ И КАЛИБРОВКА")
        self.position_button.clicked.connect(self._on_show_imu)
        self.motion_calibration_button = QPushButton("КАЛИБРОВКА ДВИЖЕНИЯ")
        self.motion_calibration_button.setEnabled(
            self._on_show_motion_calibration is not None
        )
        if self._on_show_motion_calibration is not None:
            self.motion_calibration_button.clicked.connect(
                self._on_show_motion_calibration
            )
        self.fullscreen_button = QPushButton()
        self.fullscreen_button.clicked.connect(self._on_toggle_fullscreen)
        self.exit_button = QPushButton("Выход")
        self.exit_button.clicked.connect(self._on_exit)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        status_group = QGroupBox("Состояние системы")
        status_layout = QGridLayout(status_group)
        for column, camera in enumerate(_CAMERA_ORDER):
            status_layout.addWidget(self.camera_labels[camera], 0, column)
        status_layout.addWidget(self.controller_label, 1, 0)
        status_layout.addWidget(self.mode_label, 1, 1)
        status_layout.addWidget(self.motor_label, 1, 2)
        root.addWidget(status_group)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        self.navigation = QListWidget()
        self.navigation.setFixedWidth(210)
        self.navigation.setWordWrap(False)
        self.navigation.addItems(
            [
                "Камеры",
                "Турель",
                "Отслеживание",
                "Калибровка",
                "Дополнительно",
            ]
        )
        self.pages = QStackedWidget()
        for page in (
            self._camera_page(),
            self._turret_page(),
            self._observation_page(),
            self._calibration_page(),
            self._additional_page(),
        ):
            self.pages.addWidget(self._scroll_page(page))
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        body.addWidget(self.navigation, 0)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)

    @staticmethod
    def _scroll_page(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _camera_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.show_right_camera_checkbox)
        if CameraRole.STEREO_RIGHT not in self._available_cameras:
            disabled_note = QLabel(
                "Правая камера отключена в конфигурации запуска; настройка отображения "
                "станет доступна после её включения."
            )
            disabled_note.setWordWrap(True)
            layout.addWidget(disabled_note)
        layout.addWidget(self.camera_error_label)
        display_note = QLabel(
            "Основная отображаемая камера выбирается кликом по одному из превью. "
            "Отдельный переключатель здесь не дублируется."
        )
        display_note.setWordWrap(True)
        layout.addWidget(display_note)
        right_note = QLabel(
            "Правая камера может отображаться и иметь собственную прицельную точку, "
            "но пока не является управляющей камерой для выбора цели и наведения."
        )
        right_note.setWordWrap(True)
        layout.addWidget(right_note)
        layout.addStretch(1)
        return page

    def _turret_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        pid_group = QGroupBox("PID")
        pid_layout = QGridLayout(pid_group)
        pid_layout.addWidget(QLabel(""), 0, 0)
        pid_layout.addWidget(QLabel("Kp"), 0, 1)
        pid_layout.addWidget(QLabel("Ki"), 0, 2)
        pid_layout.addWidget(QLabel("Kd"), 0, 3)
        pid_layout.addWidget(QLabel("Ось X"), 1, 0)
        pid_layout.addWidget(self.pid_inputs["pid_kp_x"], 1, 1)
        pid_layout.addWidget(self.pid_inputs["pid_ki_x"], 1, 2)
        pid_layout.addWidget(self.pid_inputs["pid_kd_x"], 1, 3)
        pid_layout.addWidget(QLabel("Ось Y"), 2, 0)
        pid_layout.addWidget(self.pid_inputs["pid_kp_y"], 2, 1)
        pid_layout.addWidget(self.pid_inputs["pid_ki_y"], 2, 2)
        pid_layout.addWidget(self.pid_inputs["pid_kd_y"], 2, 3)
        pid_layout.addWidget(self.pid_apply_button, 3, 0, 1, 4)
        pid_layout.addWidget(self.pid_status_label, 4, 0, 1, 4)
        pid_layout.addWidget(self.pid_error_label, 5, 0, 1, 4)
        layout.addWidget(pid_group)

        ff_group = QGroupBox("Feed Forward")
        ff_layout = QGridLayout(ff_group)
        ff_layout.addWidget(self.ff_enabled_checkbox, 0, 0, 1, 2)
        ff_layout.addWidget(QLabel("Gain X"), 1, 0)
        ff_layout.addWidget(self.ff_gain_x, 1, 1)
        ff_layout.addWidget(QLabel("Gain Y"), 2, 0)
        ff_layout.addWidget(self.ff_gain_y, 2, 1)
        ff_layout.addWidget(self.ff_apply_button, 3, 0, 1, 2)
        ff_layout.addWidget(self.ff_status_label, 4, 0, 1, 2)
        ff_layout.addWidget(self.ff_error_label, 5, 0, 1, 2)
        layout.addWidget(ff_group)

        limits_group = QGroupBox("Ограничения движения")
        limits_layout = QGridLayout(limits_group)
        rows = (
            ("Макс. скорость X", "max_speed_x_deg_s"),
            ("Макс. скорость Y", "max_speed_y_deg_s"),
            ("Ускорение X", "acceleration_x_deg_s2"),
            ("Ускорение Y", "acceleration_y_deg_s2"),
        )
        for row, (title, key) in enumerate(rows):
            limits_layout.addWidget(QLabel(title), row, 0)
            limits_layout.addWidget(self.motion_limit_inputs[key], row, 1)
        limits_layout.addWidget(self.motion_limits_apply_button, len(rows), 0, 1, 2)
        limits_layout.addWidget(
            self.motion_limits_status_label, len(rows) + 1, 0, 1, 2
        )
        limits_layout.addWidget(
            self.motion_limits_error_label, len(rows) + 2, 0, 1, 2
        )
        layout.addWidget(limits_group)
        layout.addStretch(1)
        return page

    def _observation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        detector_group = QGroupBox("Детектор и профиль")
        detector_layout = QGridLayout(detector_group)
        detector_layout.addWidget(QLabel("Камера"), 0, 0)
        detector_layout.addWidget(QLabel("Детектор"), 0, 1)
        detector_layout.addWidget(QLabel("Профиль"), 0, 2)
        detector_layout.addWidget(QLabel("Статус"), 0, 4)
        for row, camera in enumerate(_CAMERA_ORDER, start=1):
            controls = self.detector_controls[camera]
            detector_layout.addWidget(QLabel(_CAMERA_NAMES[camera]), row, 0)
            detector_layout.addWidget(controls["detector"], row, 1)
            detector_layout.addWidget(controls["profile"], row, 2)
            detector_layout.addWidget(controls["apply"], row, 3)
            detector_layout.addWidget(controls["status"], row, 4)
        layout.addWidget(detector_group)

        visualization_group = QGroupBox("Визуализация")
        visualization_layout = QVBoxLayout(visualization_group)
        visualization_layout.addWidget(self.show_search_region_checkbox)
        visualization_layout.addWidget(self.show_best_candidate_checkbox)
        visualization_layout.addWidget(self.show_control_prediction_checkbox)
        layout.addWidget(visualization_group)

        tracking_group = QGroupBox("Сопровождение")
        tracking_layout = QGridLayout(tracking_group)
        tracking_layout.addWidget(self.prediction_enabled_checkbox, 0, 0, 1, 2)
        tracking_layout.addWidget(QLabel("Окно оценки скорости цели"), 1, 0)
        tracking_layout.addWidget(self.tracking_inputs["window_ms"], 1, 1)
        tracking_layout.addWidget(self.tracking_apply_button, 2, 0, 1, 2)
        tracking_layout.addWidget(self.tracking_status_label, 3, 0, 1, 2)
        tracking_layout.addWidget(self.tracking_error_label, 4, 0, 1, 2)
        layout.addWidget(tracking_group)
        layout.addStretch(1)
        return page

    def _calibration_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.aim_point_button)
        reset_row = QWidget()
        reset_layout = QHBoxLayout(reset_row)
        reset_layout.setContentsMargins(0, 0, 0, 0)
        reset_layout.setSpacing(6)
        for camera in _CAMERA_ORDER:
            reset_layout.addWidget(self.aim_point_reset_buttons[camera])
        layout.addWidget(reset_row)
        layout.addWidget(self.motion_calibration_button)
        layout.addWidget(self.position_button)
        layout.addStretch(1)
        return page

    def _additional_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        diagnostics_group = QGroupBox("Оценка движения цели — диагностика")
        diagnostics_layout = QGridLayout(diagnostics_group)
        diagnostics_layout.addWidget(self.platform_source_label, 0, 0, 1, 3)
        diagnostics_layout.addWidget(QLabel(""), 1, 0)
        diagnostics_layout.addWidget(QLabel("X"), 1, 1)
        diagnostics_layout.addWidget(QLabel("Y"), 1, 2)
        rows = (
            ("Relative rate, deg/s", "relative_rate"),
            ("Platform rate, deg/s", "platform_rate"),
            ("Target rate, deg/s", "target_rate"),
            ("Fit RMS, deg", "fit_rms"),
            ("Samples", "sample_count"),
            ("Span", "actual_span"),
            ("Validity / reason", "validity"),
            ("Prediction offset", "prediction_offset"),
        )
        for row, (title, key) in enumerate(rows, start=2):
            diagnostics_layout.addWidget(QLabel(title), row, 0)
            diagnostics_layout.addWidget(
                self.tracking_diagnostic_labels[f"{key}_x"], row, 1
            )
            diagnostics_layout.addWidget(
                self.tracking_diagnostic_labels[f"{key}_y"], row, 2
            )
        layout.addWidget(diagnostics_group)
        camera_time_group = QGroupBox("Временные метки камеры")
        camera_time_layout = QVBoxLayout(camera_time_group)
        camera_time_layout.addWidget(self.use_camera_time_for_tracking_checkbox)
        camera_time_layout.addWidget(self.camera_time_status_label)
        layout.addWidget(camera_time_group)
        layout.addWidget(self.fullscreen_button)
        layout.addWidget(self.exit_button)
        layout.addStretch(1)
        return page

    @staticmethod
    def _non_negative_spinbox(value: float, *, step: float) -> QDoubleSpinBox:
        editor = QDoubleSpinBox()
        editor.setRange(0.0, 1_000_000.0)
        editor.setSingleStep(step)
        editor.setDecimals(2)
        editor.setMaximumWidth(150)
        editor.setValue(float(value))
        return editor

    @staticmethod
    def _positive_spinbox(value: float, *, step: float, suffix: str) -> QDoubleSpinBox:
        editor = QDoubleSpinBox()
        editor.setRange(0.01, 1_000_000.0)
        editor.setSingleStep(step)
        editor.setDecimals(2)
        editor.setMaximumWidth(170)
        editor.setSuffix(suffix)
        editor.setValue(float(value))
        return editor

    @staticmethod
    def _tracking_gain_spinbox(value: float) -> QDoubleSpinBox:
        editor = QDoubleSpinBox()
        editor.setRange(0.0, 1.5)
        editor.setSingleStep(0.05)
        editor.setDecimals(2)
        editor.setMaximumWidth(150)
        editor.setValue(value)
        return editor

    def _apply_pid_tuning(self) -> None:
        config = PidControllerConfig(
            pid_kp_x=self.pid_inputs["pid_kp_x"].value(),
            pid_ki_x=self.pid_inputs["pid_ki_x"].value(),
            pid_kd_x=self.pid_inputs["pid_kd_x"].value(),
            pid_kp_y=self.pid_inputs["pid_kp_y"].value(),
            pid_ki_y=self.pid_inputs["pid_ki_y"].value(),
            pid_kd_y=self.pid_inputs["pid_kd_y"].value(),
        )
        try:
            self._on_apply_pid_tuning(config)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.pid_error_label.setText(f"Ошибка PID: {exc}")
            return
        self.pid_error_label.clear()
        self.pid_status_label.setText(
            "Отправлено: "
            f"X P={config.pid_kp_x:g} I={config.pid_ki_x:g} D={config.pid_kd_x:g}; "
            f"Y P={config.pid_kp_y:g} I={config.pid_ki_y:g} D={config.pid_kd_y:g}"
        )

    def _apply_feed_forward_tuning(self) -> None:
        config = replace(
            self._tracking_control_config,
            feed_forward=replace(
                self._tracking_control_config.feed_forward,
                enabled=self.ff_enabled_checkbox.isChecked(),
                gain_x=self.ff_gain_x.value(),
                gain_y=self.ff_gain_y.value(),
            ),
        )
        try:
            self._on_apply_tracking_control_tuning(config)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.ff_error_label.setText(f"Ошибка FF: {exc}")
            return
        self._tracking_control_config = config
        self.ff_error_label.clear()
        self.ff_status_label.setText(
            "Отправлено: "
            f"{'ON' if config.feed_forward.enabled else 'OFF'}, "
            f"gain=({config.feed_forward.gain_x:g}, {config.feed_forward.gain_y:g})"
        )

    def _apply_motion_limits_tuning(self) -> None:
        config = replace(
            self._stm32_config,
            max_speed_x_deg_s=self.motion_limit_inputs["max_speed_x_deg_s"].value(),
            max_speed_y_deg_s=self.motion_limit_inputs["max_speed_y_deg_s"].value(),
            acceleration_x_deg_s2=self.motion_limit_inputs[
                "acceleration_x_deg_s2"
            ].value(),
            acceleration_y_deg_s2=self.motion_limit_inputs[
                "acceleration_y_deg_s2"
            ].value(),
        )
        try:
            self._on_apply_motion_limits_tuning(config)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.motion_limits_error_label.setText(f"Ошибка ограничений: {exc}")
            return
        self._stm32_config = config
        self.motion_limits_error_label.clear()
        self.motion_limits_status_label.setText(
            "Отправлено: "
            f"скорость=({config.max_speed_x_deg_s:g}, {config.max_speed_y_deg_s:g}) °/с; "
            f"ускорение=({config.acceleration_x_deg_s2:g}, "
            f"{config.acceleration_y_deg_s2:g}) °/с²"
        )

    def _sync_tracking_controls(self, config: TrackingControlConfig) -> None:
        self._tracking_control_config = config
        self.ff_enabled_checkbox.setChecked(config.feed_forward.enabled)
        self.prediction_enabled_checkbox.setChecked(config.prediction.enabled)
        self.ff_gain_x.setValue(config.feed_forward.gain_x)
        self.ff_gain_y.setValue(config.feed_forward.gain_y)
        self.tracking_inputs["window_ms"].setValue(config.velocity_estimator.window_ms)

    def set_tracking_control_config(self, config: TrackingControlConfig) -> None:
        self._sync_tracking_controls(config)

    def set_tracking_control_horizon_ms(self, horizon_ms: int) -> None:
        self._tracking_control_config = replace(
            self._tracking_control_config,
            prediction=replace(
                self._tracking_control_config.prediction,
                horizon_ms=horizon_ms,
            ),
        )

    def _sync_tracking_visualization_controls(
        self, config: TrackingVisualizationConfig
    ) -> None:
        self._syncing_tracking_visualization = True
        try:
            self._tracking_visualization_config = config
            self.show_search_region_checkbox.setChecked(config.show_search_region)
            self.show_best_candidate_checkbox.setChecked(config.show_best_candidate)
            self.show_control_prediction_checkbox.setChecked(
                config.show_control_prediction
            )
        finally:
            self._syncing_tracking_visualization = False

    def _apply_tracking_visualization(self, _state: int = 0) -> None:
        if self._syncing_tracking_visualization:
            return
        config = TrackingVisualizationConfig(
            show_search_region=self.show_search_region_checkbox.isChecked(),
            show_best_candidate=self.show_best_candidate_checkbox.isChecked(),
            show_control_prediction=self.show_control_prediction_checkbox.isChecked(),
        )
        try:
            self._on_tracking_visualization_changed(config)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.tracking_error_label.setText(f"Ошибка визуализации: {exc}")
            self._sync_tracking_visualization_controls(
                self._tracking_visualization_config
            )
            return
        self._tracking_visualization_config = config
        self.tracking_error_label.clear()

    def _apply_tracking_tuning(self) -> None:
        config = replace(
            self._tracking_control_config,
            prediction=replace(
                self._tracking_control_config.prediction,
                enabled=self.prediction_enabled_checkbox.isChecked(),
            ),
            velocity_estimator=replace(
                self._tracking_control_config.velocity_estimator,
                window_ms=self.tracking_inputs["window_ms"].value(),
            ),
        )
        try:
            self._on_apply_tracking_control_tuning(config)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.tracking_error_label.setText(f"Ошибка отслеживания: {exc}")
            return
        self._tracking_control_config = config
        self.tracking_error_label.clear()
        self.tracking_status_label.setText(
            "Отправлено: "
            f"Prediction={'ON' if config.prediction.enabled else 'OFF'}; "
            f"окно={config.velocity_estimator.window_ms} мс"
        )

    def _detector_changed(self, camera: CameraRole) -> None:
        controls = self.detector_controls[camera]
        detector = controls["detector"]
        profile = controls["profile"]
        assert isinstance(detector, QComboBox)
        assert isinstance(profile, QComboBox)
        self._populate_profile_combo(
            profile,
            str(detector.currentData()),
            None,
        )
        profile.setEnabled(detector.isEnabled())

    def _apply_detector_selection(self, camera: CameraRole) -> None:
        controls = self.detector_controls[camera]
        detector = controls["detector"]
        profile = controls["profile"]
        status = controls["status"]
        assert isinstance(detector, QComboBox)
        assert isinstance(profile, QComboBox)
        assert isinstance(status, QLabel)
        callback = self._on_apply_vision_processor_selection
        if callback is None:
            return
        class_name = str(detector.currentData())
        profile_name = str(profile.currentData())
        try:
            selection = VisionProcessorSelection(class_name, profile_name)
            callback(camera, selection)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            status.setText(f"Ошибка: {exc}")
            return
        self._vision_processor_selections[camera] = selection
        status.setText(
            "Отправлено: "
            f"{_PROCESSOR_LABELS[selection.class_name]} / {selection.profile_name}"
        )

    def _apply_camera_time_for_tracking(self, _state: int = 0) -> None:
        enabled = self.use_camera_time_for_tracking_checkbox.isChecked()
        try:
            self._on_use_camera_time_for_tracking_changed(enabled)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.use_camera_time_for_tracking_checkbox.blockSignals(True)
            try:
                self.use_camera_time_for_tracking_checkbox.setChecked(not enabled)
            finally:
                self.use_camera_time_for_tracking_checkbox.blockSignals(False)
            self.camera_time_status_label.setText(
                f"Ошибка переключения временной шкалы: {exc}"
            )
            return
        self.camera_time_status_label.setText(
            "Время камеры: включено для управления после готовности ClockMapper"
            if enabled
            else "Время камеры: используется только для диагностики"
        )

    def _apply_show_right_camera(self, _state: int = 0) -> None:
        if self._syncing_show_right_camera:
            return
        enabled = self.show_right_camera_checkbox.isChecked()
        try:
            self._on_show_stereo_right_camera_changed(enabled)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.camera_error_label.setText(f"Ошибка отображения Правой камеры: {exc}")
            self.set_show_stereo_right_camera(not enabled)
            return
        self.camera_error_label.clear()

    def set_show_stereo_right_camera(self, enabled: bool) -> None:
        self._syncing_show_right_camera = True
        try:
            self.show_right_camera_checkbox.setChecked(bool(enabled))
        finally:
            self._syncing_show_right_camera = False

    def set_tracking_diagnostics(
        self,
        diagnostics: TrackingEstimatorDiagnostics,
    ) -> None:
        self.platform_source_label.setText(
            f"Platform source: {diagnostics.platform_source.name}"
        )
        self._set_axis_diagnostics("x", diagnostics.x)
        self._set_axis_diagnostics("y", diagnostics.y)

    def _set_axis_diagnostics(
        self,
        axis: str,
        diagnostics: AxisTrackingEstimate,
    ) -> None:
        labels = self.tracking_diagnostic_labels
        labels[f"relative_rate_{axis}"].setText(
            self._format_optional(diagnostics.relative_rate_deg_s, "deg/s")
        )
        labels[f"platform_rate_{axis}"].setText(
            self._format_optional(diagnostics.platform_rate_deg_s, "deg/s")
        )
        labels[f"target_rate_{axis}"].setText(
            self._format_optional(diagnostics.target_rate_deg_s, "deg/s")
        )
        labels[f"fit_rms_{axis}"].setText(
            self._format_optional(diagnostics.fit_rms_deg, "deg")
        )
        labels[f"sample_count_{axis}"].setText(str(diagnostics.sample_count))
        labels[f"actual_span_{axis}"].setText(f"{diagnostics.actual_span_ms:g} ms")
        validity = "VALID"
        if not diagnostics.valid:
            reason = diagnostics.invalid_reason
            validity = f"INVALID ({'—' if reason is None else reason.value})"
        labels[f"validity_{axis}"].setText(validity)
        labels[f"prediction_offset_{axis}"].setText(
            f"{diagnostics.prediction_offset_deg:g} deg"
        )

    @staticmethod
    def _format_optional(value: float | None, unit: str) -> str:
        return "—" if value is None else f"{value:g} {unit}"

    def set_camera_status(
        self,
        camera: CameraRole,
        status: CameraStatus | None,
    ) -> None:
        label = self.camera_labels.get(camera)
        if label is None:
            return
        if camera not in self._available_cameras:
            label.setText(f"{_CAMERA_NAMES[camera]}: ОТКЛЮЧЕНА")
            return
        state = "NO STATUS" if status is None else status.state.value.upper()
        label.setText(f"{_CAMERA_NAMES[camera]}: {state}")

    def set_turret_state(
        self,
        state: TurretState,
        pending_mode: TurretControlMode | None,
    ) -> None:
        self.controller_label.setText(
            f"Контроллер: {state.connection_state.value.upper()}"
        )
        mode = state.control_mode.value.upper()
        if pending_mode is not None:
            mode += f" → {pending_mode.value.upper()}"
        self.mode_label.setText(f"Режим: {mode}")
        self.motor_label.setText(f"Мотор: {state.motor_state.value.upper()}")

    def set_aim_point_selection_state(self, active: bool) -> None:
        self.aim_point_button.setText(
            "ПРИЦЕЛЬНАЯ ТОЧКА — ОЖИДАНИЕ КЛИКА" if active else "ПРИЦЕЛЬНАЯ ТОЧКА"
        )

    def set_fullscreen_state(self, fullscreen: bool) -> None:
        state = "ВКЛ" if fullscreen else "ВЫКЛ"
        self.fullscreen_button.setText(f"Полноэкранный режим: {state}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F11:
            self._on_toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = ["OperatorWindow"]
