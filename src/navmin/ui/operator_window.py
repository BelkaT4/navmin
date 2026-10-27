"""Small modeless runtime panel owned by the NavMin main window."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import isfinite

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from navmin.config.models import PidControllerConfig, TrackingControlConfig
from navmin.contracts import CameraRole, CameraStatus, TurretControlMode, TurretState
from navmin.core import AxisTrackingEstimate, TrackingEstimatorDiagnostics

_CAMERA_NAMES = {
    CameraRole.OVERVIEW: "Overview",
    CameraRole.STEREO_LEFT: "Stereo Left",
}


class OperatorWindow(QWidget):
    """Modeless status/admin window with MainWindow as its transient owner."""

    def __init__(
        self,
        *,
        owner: QWidget,
        on_toggle_fullscreen: Callable[[], None],
        on_exit: Callable[[], None],
        on_show_imu: Callable[[], None],
        pid_controller_config: PidControllerConfig,
        on_apply_pid_tuning: Callable[[PidControllerConfig], None],
        tracking_control_config: TrackingControlConfig,
        on_apply_tracking_control_tuning: Callable[[TrackingControlConfig], None],
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._on_toggle_fullscreen = on_toggle_fullscreen
        self._on_exit = on_exit
        self._on_show_imu = on_show_imu
        self._on_apply_pid_tuning = on_apply_pid_tuning
        self._tracking_control_config = tracking_control_config
        self._on_apply_tracking_control_tuning = on_apply_tracking_control_tuning
        self.setWindowTitle("Operator Window")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.camera_labels = {
            camera: QLabel(f"{name}: NO STATUS")
            for camera, name in _CAMERA_NAMES.items()
        }
        self.controller_label = QLabel("Контроллер: DISCONNECTED")
        self.mode_label = QLabel("Режим: RELATIVE")
        self.motor_label = QLabel("Мотор: UNKNOWN")
        self.pid_inputs = {
            "pid_kp_x": QLineEdit(self._format_gain(pid_controller_config.pid_kp_x)),
            "pid_ki_x": QLineEdit(self._format_gain(pid_controller_config.pid_ki_x)),
            "pid_kd_x": QLineEdit(self._format_gain(pid_controller_config.pid_kd_x)),
            "pid_kp_y": QLineEdit(self._format_gain(pid_controller_config.pid_kp_y)),
            "pid_ki_y": QLineEdit(self._format_gain(pid_controller_config.pid_ki_y)),
            "pid_kd_y": QLineEdit(self._format_gain(pid_controller_config.pid_kd_y)),
        }
        for editor in self.pid_inputs.values():
            editor.setMaximumWidth(110)
        self.pid_apply_button = QPushButton("Применить PID")
        self.pid_apply_button.clicked.connect(self._apply_pid_tuning)
        self.pid_status_label = QLabel("Статус PID: —")
        self.pid_status_label.setWordWrap(True)
        self.pid_error_label = QLabel("")
        self.pid_error_label.setWordWrap(True)
        self.ff_enabled_checkbox = QCheckBox("Feed-forward enabled")
        self.ff_enabled_checkbox.setChecked(tracking_control_config.feed_forward.enabled)
        self.prediction_enabled_checkbox = QCheckBox("Prediction enabled")
        self.prediction_enabled_checkbox.setChecked(
            tracking_control_config.prediction.enabled
        )
        self.tracking_inputs = {
            "gain_x": QLineEdit(
                self._format_gain(tracking_control_config.feed_forward.gain_x)
            ),
            "gain_y": QLineEdit(
                self._format_gain(tracking_control_config.feed_forward.gain_y)
            ),
            "horizon_ms": QLineEdit(
                str(tracking_control_config.prediction.horizon_ms)
            ),
            "window_ms": QLineEdit(
                str(tracking_control_config.velocity_estimator.window_ms)
            ),
        }
        for editor in self.tracking_inputs.values():
            editor.setMaximumWidth(110)
        self.tracking_apply_button = QPushButton("Применить tracking")
        self.tracking_apply_button.clicked.connect(self._apply_tracking_tuning)
        self.tracking_status_label = QLabel("Статус tracking: —")
        self.tracking_status_label.setWordWrap(True)
        self.tracking_error_label = QLabel("")
        self.tracking_error_label.setWordWrap(True)
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
        self.position_button = QPushButton("ПОЛОЖЕНИЕ")
        self.position_button.clicked.connect(self._on_show_imu)
        self.fullscreen_button = QPushButton()
        self.fullscreen_button.clicked.connect(self._on_toggle_fullscreen)
        self.exit_button = QPushButton("Выход")
        self.exit_button.clicked.connect(self._on_exit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)
        for camera in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT):
            layout.addWidget(self.camera_labels[camera])
        layout.addSpacing(4)
        layout.addWidget(self.controller_label)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.motor_label)
        layout.addSpacing(6)

        pid_group = QGroupBox("PID — временная настройка")
        pid_layout = QGridLayout(pid_group)
        pid_layout.addWidget(QLabel(""), 0, 0)
        pid_layout.addWidget(QLabel("Kp"), 0, 1)
        pid_layout.addWidget(QLabel("Ki"), 0, 2)
        pid_layout.addWidget(QLabel("Kd"), 0, 3)
        pid_layout.addWidget(QLabel("PID X"), 1, 0)
        pid_layout.addWidget(self.pid_inputs["pid_kp_x"], 1, 1)
        pid_layout.addWidget(self.pid_inputs["pid_ki_x"], 1, 2)
        pid_layout.addWidget(self.pid_inputs["pid_kd_x"], 1, 3)
        pid_layout.addWidget(QLabel("PID Y"), 2, 0)
        pid_layout.addWidget(self.pid_inputs["pid_kp_y"], 2, 1)
        pid_layout.addWidget(self.pid_inputs["pid_ki_y"], 2, 2)
        pid_layout.addWidget(self.pid_inputs["pid_kd_y"], 2, 3)
        pid_layout.addWidget(self.pid_apply_button, 3, 0, 1, 4)
        pid_layout.addWidget(self.pid_status_label, 4, 0, 1, 4)
        pid_layout.addWidget(self.pid_error_label, 5, 0, 1, 4)
        layout.addWidget(pid_group)
        tracking_group = QGroupBox("Predictive tracking — временная настройка")
        tracking_layout = QGridLayout(tracking_group)
        tracking_layout.addWidget(self.ff_enabled_checkbox, 0, 0, 1, 2)
        tracking_layout.addWidget(QLabel("FF gain X"), 1, 0)
        tracking_layout.addWidget(self.tracking_inputs["gain_x"], 1, 1)
        tracking_layout.addWidget(QLabel("FF gain Y"), 2, 0)
        tracking_layout.addWidget(self.tracking_inputs["gain_y"], 2, 1)
        tracking_layout.addWidget(self.prediction_enabled_checkbox, 3, 0, 1, 2)
        tracking_layout.addWidget(QLabel("Prediction horizon, ms"), 4, 0)
        tracking_layout.addWidget(self.tracking_inputs["horizon_ms"], 4, 1)
        tracking_layout.addWidget(QLabel("Estimator window, ms"), 5, 0)
        tracking_layout.addWidget(self.tracking_inputs["window_ms"], 5, 1)
        tracking_layout.addWidget(self.tracking_apply_button, 6, 0, 1, 2)
        tracking_layout.addWidget(self.tracking_status_label, 7, 0, 1, 2)
        tracking_layout.addWidget(self.tracking_error_label, 8, 0, 1, 2)
        layout.addWidget(tracking_group)

        diagnostics_group = QGroupBox("Tracking estimator — диагностика")
        diagnostics_layout = QGridLayout(diagnostics_group)
        diagnostics_layout.addWidget(self.platform_source_label, 0, 0, 1, 3)
        diagnostics_layout.addWidget(QLabel(""), 1, 0)
        diagnostics_layout.addWidget(QLabel("X"), 1, 1)
        diagnostics_layout.addWidget(QLabel("Y"), 1, 2)
        diagnostic_rows = (
            ("Relative rate, deg/s", "relative_rate"),
            ("Platform rate, deg/s", "platform_rate"),
            ("Target rate, deg/s", "target_rate"),
            ("Fit RMS, deg", "fit_rms"),
            ("Samples", "sample_count"),
            ("Span", "actual_span"),
            ("Validity / reason", "validity"),
            ("Prediction offset", "prediction_offset"),
        )
        for row, (title, key) in enumerate(diagnostic_rows, start=2):
            diagnostics_layout.addWidget(QLabel(title), row, 0)
            diagnostics_layout.addWidget(
                self.tracking_diagnostic_labels[f"{key}_x"], row, 1
            )
            diagnostics_layout.addWidget(
                self.tracking_diagnostic_labels[f"{key}_y"], row, 2
            )
        layout.addWidget(diagnostics_group)
        layout.addSpacing(6)
        layout.addWidget(self.position_button)
        layout.addWidget(self.fullscreen_button)
        layout.addWidget(self.exit_button)
        self.setMinimumWidth(420)
        self.adjustSize()

    def _apply_pid_tuning(self) -> None:
        try:
            config = PidControllerConfig(
                pid_kp_x=self._read_gain("pid_kp_x", "Kp X"),
                pid_ki_x=self._read_gain("pid_ki_x", "Ki X"),
                pid_kd_x=self._read_gain("pid_kd_x", "Kd X"),
                pid_kp_y=self._read_gain("pid_kp_y", "Kp Y"),
                pid_ki_y=self._read_gain("pid_ki_y", "Ki Y"),
                pid_kd_y=self._read_gain("pid_kd_y", "Kd Y"),
            )
            self._on_apply_pid_tuning(config)
        except (TypeError, ValueError) as exc:
            self.pid_error_label.setText(f"Ошибка PID: {exc}")
            return

        self.pid_error_label.clear()
        self.pid_status_label.setText(
            "Отправлено:\n"
            f"X P={config.pid_kp_x:g} I={config.pid_ki_x:g} D={config.pid_kd_x:g}\n"
            f"Y P={config.pid_kp_y:g} I={config.pid_ki_y:g} D={config.pid_kd_y:g}"
        )

    def _read_gain(self, field: str, label: str) -> float:
        text = self.pid_inputs[field].text().strip()
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label}: введите число >= 0") from exc
        if not isfinite(value):
            raise ValueError(f"{label}: значение должно быть конечным")
        if value < 0:
            raise ValueError(f"{label}: значение должно быть >= 0")
        return value

    def _apply_tracking_tuning(self) -> None:
        try:
            config = replace(
                self._tracking_control_config,
                feed_forward=replace(
                    self._tracking_control_config.feed_forward,
                    enabled=self.ff_enabled_checkbox.isChecked(),
                    gain_x=self._read_tracking_number("gain_x", "FF gain X"),
                    gain_y=self._read_tracking_number("gain_y", "FF gain Y"),
                ),
                prediction=replace(
                    self._tracking_control_config.prediction,
                    enabled=self.prediction_enabled_checkbox.isChecked(),
                    horizon_ms=self._read_tracking_integer(
                        "horizon_ms", "Prediction horizon", minimum=0
                    ),
                ),
            )
            window_ms = self._read_tracking_integer(
                "window_ms", "Estimator window", minimum=1
            )
            config = replace(
                config,
                velocity_estimator=replace(
                    config.velocity_estimator,
                    window_ms=window_ms,
                ),
            )
            self._on_apply_tracking_control_tuning(config)
        except (TypeError, ValueError) as exc:
            self.tracking_error_label.setText(f"Ошибка tracking: {exc}")
            return

        self._tracking_control_config = config
        self.tracking_error_label.clear()
        self.tracking_status_label.setText(
            "Отправлено: "
            f"FF={'ON' if config.feed_forward.enabled else 'OFF'} "
            f"gain=({config.feed_forward.gain_x:g}, "
            f"{config.feed_forward.gain_y:g}); "
            f"Prediction={'ON' if config.prediction.enabled else 'OFF'} "
            f"horizon={config.prediction.horizon_ms} ms; "
            f"window={config.velocity_estimator.window_ms} ms"
        )

    def _read_tracking_number(self, field: str, label: str) -> float:
        text = self.tracking_inputs[field].text().strip()
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label}: введите число >= 0") from exc
        if not isfinite(value):
            raise ValueError(f"{label}: значение должно быть конечным")
        if value < 0:
            raise ValueError(f"{label}: значение должно быть >= 0")
        return value

    def _read_tracking_integer(
        self,
        field: str,
        label: str,
        *,
        minimum: int,
    ) -> int:
        text = self.tracking_inputs[field].text().strip()
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{label}: введите целое >= {minimum}") from exc
        if value < minimum:
            raise ValueError(f"{label}: значение должно быть >= {minimum}")
        return value

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
        labels[f"actual_span_{axis}"].setText(
            f"{diagnostics.actual_span_ms:g} ms"
        )
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

    @staticmethod
    def _format_gain(value: float) -> str:
        return f"{float(value):g}"

    def set_camera_status(
        self,
        camera: CameraRole,
        status: CameraStatus | None,
    ) -> None:
        label = self.camera_labels.get(camera)
        if label is None:
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
