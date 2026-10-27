"""Modeless display and passive calibration controls for the GY-85 runtime."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QKeyEvent,
    QPainter,
    QPen,
    QPolygonF,
    QShowEvent,
)
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from navmin.concurrency import LatestValue
from navmin.imu import (
    GyroBiasState,
    ImuCalibrationControls,
    ImuRuntimeState,
    rotation_matrix,
)

_FACE_LABELS = {
    1: "X+",
    -1: "X−",
    2: "Y+",
    -2: "Y−",
    3: "Z+",
    -3: "Z−",
}


class AttitudeView(QWidget):
    """Software-rendered body orientation copied from the accepted old IMU UI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.fresh = False

    def set_orientation(
        self,
        quaternion: tuple[float, float, float, float],
        *,
        fresh: bool,
    ) -> None:
        self.quaternion = np.asarray(quaternion, dtype=float)
        self.fresh = fresh
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101c2a"))
        scale = min(self.width() / 4.5, self.height() / 4)

        def project(point: np.ndarray) -> QPointF:
            x, y, z = point
            return QPointF(
                self.width() * 0.5 + scale * (0.8 * x - 0.6 * y),
                self.height() * 0.58 + scale * (0.35 * x + 0.45 * y - z),
            )

        painter.setPen(QPen(QColor("#23384b"), 1))
        for coordinate in np.linspace(-1.8, 1.8, 13):
            painter.drawLine(
                project(np.array([coordinate, -1.8, -0.45])),
                project(np.array([coordinate, 1.8, -0.45])),
            )
            painter.drawLine(
                project(np.array([-1.8, coordinate, -0.45])),
                project(np.array([1.8, coordinate, -0.45])),
            )

        rotation = rotation_matrix(self.quaternion)
        vertices = (
            np.array(
                [
                    [-0.45, -0.28, -0.2],
                    [0.45, -0.28, -0.2],
                    [0.45, 0.28, -0.2],
                    [-0.45, 0.28, -0.2],
                    [-0.45, -0.28, 0.2],
                    [0.45, -0.28, 0.2],
                    [0.45, 0.28, 0.2],
                    [-0.45, 0.28, 0.2],
                ]
            )
            @ rotation.T
        )
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        faces.sort(
            key=lambda face: float(
                np.mean(vertices[list(face)], axis=0) @ [0.6, 0.8, 0.57]
            )
        )
        for face in faces:
            painter.setBrush(QColor("#284354" if self.fresh else "#343c42"))
            painter.setPen(QPen(QColor("#80c7df" if self.fresh else "#67717a"), 2))
            painter.drawPolygon(QPolygonF([project(vertices[index]) for index in face]))

        origin = project(np.zeros(3))
        for axis, color, label in (
            ([1.45, 0, 0], "#ffbd59", "Вперёд X"),
            ([0, 0.9, 0], "#5ddd99", "Y"),
            ([0, 0, 0.9], "#65baff", "Z"),
        ):
            endpoint = project(rotation @ np.array(axis))
            painter.setPen(QPen(QColor(color), 5 if label.startswith("Вперёд") else 2))
            painter.drawLine(origin, endpoint)
            painter.drawEllipse(endpoint, 4, 4)
            painter.drawText(endpoint + QPointF(8, -8), label)

        painter.setPen(QColor("#e6f1fa"))
        painter.drawText(18, 28, "Ориентация относительно заданного нуля")
        if not self.fresh:
            painter.setPen(QColor("#ffbd59"))
            painter.drawText(18, self.height() - 18, "НЕТ СВЕЖИХ ДАННЫХ")
        painter.end()


class ImuWindow(QWidget):
    """Modeless view over one ApplicationRuntime-owned passive IMU channel."""

    def __init__(
        self,
        *,
        owner: QWidget,
        states: LatestValue[ImuRuntimeState] | None,
        controls: ImuCalibrationControls | None,
        on_toggle_operator: Callable[[], None],
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._states = states
        self._controls = controls
        self._on_toggle_operator = on_toggle_operator
        self._local_control_message = ""
        self.setWindowTitle("NavMin — положение / GY-85")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(900, 820)

        self.attitude_view = AttitudeView(self)
        self.status_label = QLabel("STALE — состояние IMU ещё не получено")
        self.status_label.setWordWrap(True)
        self.angles_label = QLabel("Крен: —    Подъём: —    Азимут: —")
        self.quality_label = QLabel("Калибровка: —")
        self.quality_label.setWordWrap(True)
        self.sensor_label = QLabel("Показания датчика ещё не получены")
        self.sensor_label.setWordWrap(True)
        self.source_label = QLabel("Источник: —")
        self.source_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(self.attitude_view, 1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.angles_label)

        details = QGroupBox("Состояние IMU")
        details_layout = QGridLayout(details)
        details_layout.addWidget(QLabel("Качество"), 0, 0)
        details_layout.addWidget(self.quality_label, 0, 1)
        details_layout.addWidget(QLabel("Датчики"), 1, 0)
        details_layout.addWidget(self.sensor_label, 1, 1)
        details_layout.addWidget(QLabel("Источник"), 2, 0)
        details_layout.addWidget(self.source_label, 2, 1)
        details_layout.addWidget(QLabel("Диагностика"), 3, 0)
        details_layout.addWidget(self.error_label, 3, 1)
        details_layout.setColumnStretch(1, 1)
        layout.addWidget(details)

        calibration = QGroupBox("Калибровка GY-85")
        calibration_layout = QVBoxLayout(calibration)
        self.calibration_status_label = QLabel(
            "Команды калибровки доступны только при работающем IMU."
        )
        self.calibration_status_label.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_status_label)

        self.gyro_calibration_button = QPushButton("Гироскоп: повторить 5 с")
        self.gyro_calibration_button.clicked.connect(self._restart_gyro_bias)
        calibration_layout.addWidget(self.gyro_calibration_button)

        accel_note = QLabel(
            "Акселерометр: положите датчик неподвижно выбранной сырой осью вверх "
            "или вниз, затем нажмите соответствующую грань."
        )
        accel_note.setWordWrap(True)
        calibration_layout.addWidget(accel_note)
        accel_row = QHBoxLayout()
        self.accel_face_buttons: dict[int, QPushButton] = {}
        for face in (1, -1, 2, -2, 3, -3):
            button = QPushButton(_FACE_LABELS[face])
            button.clicked.connect(
                lambda _checked=False, selected_face=face: self._capture_accel_face(
                    selected_face
                )
            )
            self.accel_face_buttons[face] = button
            accel_row.addWidget(button)
        calibration_layout.addLayout(accel_row)

        mag_note = QLabel(
            "Магнитометр: начните сбор, затем плавно поворачивайте датчик во всех "
            "направлениях не менее 10 секунд и завершите сбор."
        )
        mag_note.setWordWrap(True)
        calibration_layout.addWidget(mag_note)
        mag_row = QHBoxLayout()
        self.mag_start_button = QPushButton("Магнитометр: начать")
        self.mag_finish_button = QPushButton("Магнитометр: завершить")
        self.mag_start_button.clicked.connect(self._start_magnetometer)
        self.mag_finish_button.clicked.connect(self._finish_magnetometer)
        mag_row.addWidget(self.mag_start_button)
        mag_row.addWidget(self.mag_finish_button)
        calibration_layout.addLayout(mag_row)

        self.relative_zero_button = QPushButton("Принять текущую ориентацию как ноль")
        self.relative_zero_button.clicked.connect(self._set_relative_zero)
        calibration_layout.addWidget(self.relative_zero_button)
        layout.addWidget(calibration)

        note = QLabel(
            "GY-85 работает только как измерительный слой. Калибровка изменяет только "
            "данные IMU и его calibration-файл; она не управляет двигателями, режимом "
            "наведения или сопровождением цели."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.refresh_once)
        self.refresh_once()

    @property
    def states(self) -> LatestValue[ImuRuntimeState] | None:
        return self._states

    def refresh_once(self) -> None:
        state = None if self._states is None else self._states.get()
        if state is None:
            self._render_missing_state()
            return
        self._render_state(state)

    def _render_missing_state(self) -> None:
        if self._states is None:
            self.status_label.setText("ERROR — IMU отключён в конфигурации")
        else:
            self.status_label.setText("STALE — ожидание состояния IMU")
        self.angles_label.setText("Крен: —    Подъём: —    Азимут: —")
        self.quality_label.setText("Калибровка: —")
        self.sensor_label.setText("Показания датчика ещё не получены")
        self.source_label.setText("Источник: —")
        self.error_label.setText("")
        self.calibration_status_label.setText(
            "Команды калибровки недоступны без работающего IMU."
        )
        self._set_calibration_controls_enabled(None)
        self.attitude_view.set_orientation((1.0, 0.0, 0.0, 0.0), fresh=False)

    def _render_state(self, state: ImuRuntimeState) -> None:
        if not state.running:
            status = "ERROR — IMU receiver остановлен"
        elif state.stale or not state.fresh:
            status = "STALE — нет свежих данных IMU"
        elif state.gyro_bias_state is not GyroBiasState.READY:
            status = "CALIBRATING — калибровка смещения гироскопа"
        elif state.mag_calibration_active:
            status = "CALIBRATING — сбор данных магнитометра"
        elif state.last_error:
            status = "ERROR — ошибка IMU runtime"
        else:
            status = "LIVE — данные IMU поступают"
        self.status_label.setText(status)

        if state.fresh and state.orientation_valid:
            roll, pitch, yaw = state.euler_degrees
            self.angles_label.setText(
                f"Крен: {roll:+.1f}°    Подъём: {-pitch:+.1f}°    Азимут: {yaw:+.1f}°"
            )
        else:
            self.angles_label.setText("Крен: —    Подъём: —    Азимут: —")
        accel_status = "калиброван" if state.accel_calibrated else "не калиброван"
        mag_status = "калиброван" if state.mag_calibrated else "не калиброван"
        heading_status = "VALID" if state.heading_valid else "UNAVAILABLE"
        faces = ", ".join(_FACE_LABELS[face] for face in state.accel_faces_captured)
        faces = faces or "нет"
        mag_capture = (
            f"; сбор MAG={state.mag_calibration_sample_count}; "
            f"охват≈{state.mag_calibration_octants}/8"
            if state.mag_calibration_active
            else ""
        )
        self.quality_label.setText(
            f"Гироскоп: {state.gyro_bias_state.value}; "
            f"акселерометр: {accel_status}; грани: {faces}; "
            f"магнитометр: {mag_status}{mag_capture}; "
            f"магнитная коррекция: {heading_status}; "
            f"монтаж: {state.mounting_axes}"
        )
        self.sensor_label.setText(
            f"ACC, g: {self._format_vector(state.accel_g)}\n"
            f"GYRO raw, °/с: {self._format_vector(state.gyro_raw_dps)}\n"
            f"GYRO corrected, °/с: {self._format_vector(state.gyro_corrected_dps)}\n"
            f"MAG raw: {self._format_vector(state.mag_raw)}\n"
            f"MAG corrected: {self._format_vector(state.mag_corrected)}\n"
            f"Смещение гироскопа, °/с: {self._format_vector(state.gyro_bias_dps)}"
        )
        source = (
            "—"
            if state.source is None
            else f"{state.source.host}:{state.source.port}"
        )
        sender_timestamp = self._format_timestamp(state.sender_timestamp_s)
        receive_timestamp = self._format_timestamp(state.receive_timestamp_s)
        self.source_label.setText(
            f"Источник: {source}; Pi t={sender_timestamp}; "
            f"PC receive={receive_timestamp}; "
            f"bias samples={state.gyro_bias_sample_count}"
        )
        diagnostics = (
            f"invalid={state.invalid_packets}; duplicate={state.duplicate_packets}; "
            f"out-of-order={state.out_of_order_packets}; "
            f"foreign={state.foreign_packets}"
        )
        if state.last_error:
            diagnostics += f"; last error: {state.last_error}"
        self.error_label.setText(diagnostics)
        message = state.calibration_message or self._local_control_message
        if not message:
            message = "Калибровка готова к действиям оператора."
        self.calibration_status_label.setText(message)
        self._set_calibration_controls_enabled(state)
        for face, button in self.accel_face_buttons.items():
            mark = " ✓" if face in state.accel_faces_captured else ""
            button.setText(_FACE_LABELS[face] + mark)
        self.attitude_view.set_orientation(
            state.relative_quaternion,
            fresh=state.fresh and state.orientation_valid,
        )

    def _set_calibration_controls_enabled(
        self,
        state: ImuRuntimeState | None,
    ) -> None:
        available = self._controls is not None and state is not None and state.running
        fresh = bool(available and state is not None and state.fresh)
        mag_active = bool(state is not None and state.mag_calibration_active)
        self.gyro_calibration_button.setEnabled(fresh and not mag_active)
        for button in self.accel_face_buttons.values():
            button.setEnabled(fresh and not mag_active)
        self.mag_start_button.setEnabled(fresh and not mag_active)
        self.mag_finish_button.setEnabled(available and mag_active)
        self.relative_zero_button.setEnabled(
            fresh and not mag_active and bool(state and state.orientation_valid)
        )

    def _restart_gyro_bias(self) -> None:
        self._request_control(
            "request_gyro_bias_calibration",
            queued="Гироскоп: команда отправлена; не двигайте датчик 5 секунд.",
        )

    def _capture_accel_face(self, face: int) -> None:
        self._request_control(
            "request_accelerometer_face",
            face,
            queued=f"Акселерометр: измерение {_FACE_LABELS[face]} отправлено.",
        )

    def _start_magnetometer(self) -> None:
        self._request_control(
            "request_magnetometer_start",
            queued="Магнитометр: начало сбора отправлено.",
        )

    def _finish_magnetometer(self) -> None:
        self._request_control(
            "request_magnetometer_finish",
            queued="Магнитометр: завершение сбора отправлено.",
        )

    def _set_relative_zero(self) -> None:
        self._request_control(
            "request_relative_zero",
            queued="Установка относительного нуля отправлена.",
        )

    def _request_control(self, method_name: str, *args: object, queued: str) -> None:
        if self._controls is None:
            self._local_control_message = "Команды калибровки недоступны."
            self.calibration_status_label.setText(self._local_control_message)
            return
        method = getattr(self._controls, method_name, None)
        if method is None:
            self._local_control_message = "Команда калибровки не поддерживается runtime."
            self.calibration_status_label.setText(self._local_control_message)
            return
        try:
            accepted = bool(method(*args))
        except (TypeError, ValueError, RuntimeError) as exc:
            self._local_control_message = f"Команда отклонена: {exc}"
        else:
            self._local_control_message = (
                queued if accepted else "Очередь команд IMU занята; повторите действие."
            )
        self.calibration_status_label.setText(self._local_control_message)

    @staticmethod
    def _format_vector(values: tuple[float, ...] | None) -> str:
        if values is None:
            return "—"
        return "[" + ", ".join(f"{value:.3g}" for value in values) + "]"

    @staticmethod
    def _format_timestamp(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._on_toggle_operator()
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        self.refresh_once()
        self.timer.start()
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        super().closeEvent(event)


__all__ = ["AttitudeView", "ImuWindow"]
