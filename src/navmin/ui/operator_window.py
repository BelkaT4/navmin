"""Small modeless runtime panel owned by the NavMin main window."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

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
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._on_toggle_fullscreen = on_toggle_fullscreen
        self._on_exit = on_exit
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
        layout.addWidget(self.fullscreen_button)
        layout.addWidget(self.exit_button)
        self.setMinimumWidth(300)
        self.adjustSize()

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
