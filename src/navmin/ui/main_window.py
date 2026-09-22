"""Prototype-first NavMin main window."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from time import monotonic_ns

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QCloseEvent, QKeyEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from navmin.concurrency import LatestValue
from navmin.contracts import (
    CameraRole,
    CameraSessionStarted,
    CameraStatus,
    MotorState,
    TargetRef,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.core import Mediator

from .bridge import CameraUiBinding, UiStatePump
from .operator_window import OperatorWindow
from .video_view import PreparedVisionFrame, VideoView, prepare_vision_frame

_NORMAL_CAMERAS = (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)


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


class MainWindow(QMainWindow):
    """Minimal operator UI backed exclusively by Mediator-owned state."""

    def __init__(
        self,
        *,
        mediator: Mediator,
        camera_bindings: Mapping[CameraRole, CameraUiBinding],
        turret_states: LatestValue[TurretState],
        camera_stale_timeout_ms: int,
        clock_ns: Callable[[], int] = monotonic_ns,
        wall_clock: Callable[[], datetime] = datetime.now,
        show_diagnostic_clock: bool = False,
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
        self._prepared_frames: dict[CameraRole, PreparedVisionFrame] = {}
        self._camera_statuses: dict[CameraRole, CameraStatus] = {}
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

        self.mode_button = QPushButton()
        self.mode_button.setFixedSize(170, 40)
        self.mode_button.clicked.connect(self._request_other_mode)
        self.motor_button = QPushButton()
        self.motor_button.setFixedSize(145, 40)
        self.motor_button.clicked.connect(self._toggle_motor)
        self.connection_label = QLabel()
        self.connection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.connection_label.setFixedSize(170, 40)
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

        self.operator_window = OperatorWindow(
            owner=self,
            on_toggle_fullscreen=self.toggle_fullscreen,
            on_exit=self.request_exit,
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
        if self._mediator.main_camera is not self._shown_main_camera:
            self._shown_main_camera = self._mediator.main_camera
            self._refresh_view_roles()
        if self._mediator.selected_target != self._shown_selection:
            self._shown_selection = self._mediator.selected_target
            self.main_view.set_selected_target(self._shown_selection)
            self.preview_view.set_selected_target(self._shown_selection)
        now_ns = self._clock_ns()
        self.main_view.refresh_freshness(now_ns)
        self.preview_view.refresh_freshness(now_ns)
        if self._show_diagnostic_clock:
            now = self._wall_clock()
            milliseconds = now.microsecond // 1000
            self.now_label.setText(f"NOW {now:%H:%M:%S}.{milliseconds:03d}")

    def _camera_session_accepted(self, session: CameraSessionStarted) -> None:
        self._prepared_frames.pop(session.camera, None)
        self._apply_camera_to_views(session.camera)
        self.refresh_presentation()

    def _vision_result_accepted(self, result: VisionResult) -> None:
        prepared = self._frame_preparer(result)
        self._prepared_frames[result.frame.camera] = prepared
        self._apply_camera_to_views(result.frame.camera)

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
        self.preview_view.set_prepared_frame(self._prepared_frames.get(preview_camera))
        self.main_view.set_camera_status(self._camera_statuses.get(main_camera))
        self.preview_view.set_camera_status(self._camera_statuses.get(preview_camera))
        self.main_view.set_selected_target(self._mediator.selected_target)
        self.preview_view.set_selected_target(self._mediator.selected_target)

    def _apply_camera_to_views(self, camera: CameraRole) -> None:
        for view in (self.main_view, self.preview_view):
            if view.camera is camera:
                view.set_prepared_frame(self._prepared_frames.get(camera))
                view.set_camera_status(self._camera_statuses.get(camera))

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
            tracked = self._hit_test(displayed_result, x_px, y_px)
            if tracked is None:
                self._mediator.deselect_target()
            else:
                self._mediator.select_target(
                    TargetRef(
                        camera=displayed_result.frame.camera,
                        generation=displayed_result.frame.generation,
                        track_id=tracked,
                    )
                )
        self.refresh_presentation()

    @staticmethod
    def _hit_test(result: VisionResult, x_px: float, y_px: float) -> int | None:
        candidates: list[tuple[float, int]] = []
        for tracked in result.tracked_objects:
            bbox = tracked.bbox
            if not (
                bbox.x <= x_px < bbox.x + bbox.width
                and bbox.y <= y_px < bbox.y + bbox.height
            ):
                continue
            center_x = bbox.x + bbox.width / 2.0
            center_y = bbox.y + bbox.height / 2.0
            distance_squared = (center_x - x_px) ** 2 + (center_y - y_px) ** 2
            candidates.append((distance_squared, tracked.track_id))
        return min(candidates)[1] if candidates else None

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
