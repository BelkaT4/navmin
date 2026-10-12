"""Small modeless runtime panel owned by the NavMin main window."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from navmin.config.models import PidControllerConfig
from navmin.contracts import CameraRole, CameraStatus, TurretControlMode, TurretState

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
        pid_controller_config: PidControllerConfig,
        on_apply_pid_tuning: Callable[[PidControllerConfig], None],
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._on_toggle_fullscreen = on_toggle_fullscreen
        self._on_exit = on_exit
        self._on_apply_pid_tuning = on_apply_pid_tuning
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
        layout.addSpacing(6)
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
