"""Read-only modeless display for the passive GY-85 runtime state."""

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
    QLabel,
    QVBoxLayout,
    QWidget,
)

from navmin.concurrency import LatestValue
from navmin.imu import GyroBiasState, ImuRuntimeState, rotation_matrix


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
    """Modeless read-only view over one ApplicationRuntime-owned IMU state slot."""

    def __init__(
        self,
        *,
        owner: QWidget,
        states: LatestValue[ImuRuntimeState] | None,
        on_toggle_operator: Callable[[], None],
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._states = states
        self._on_toggle_operator = on_toggle_operator
        self.setWindowTitle("NavMin — положение / GY-85")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(900, 680)

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

        note = QLabel(
            "GY-85 работает только как измерительный слой. Это окно не управляет "
            "двигателями, режимом наведения или сопровождением цели."
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
        self.attitude_view.set_orientation((1.0, 0.0, 0.0, 0.0), fresh=False)

    def _render_state(self, state: ImuRuntimeState) -> None:
        if not state.running:
            status = "ERROR — IMU receiver остановлен"
        elif state.stale or not state.fresh:
            status = "STALE — нет свежих данных IMU"
        elif (
            state.gyro_bias_state is not GyroBiasState.READY
            or not state.orientation_valid
        ):
            status = "CALIBRATING — калибровка смещения гироскопа"
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
        self.quality_label.setText(
            f"Гироскоп: {state.gyro_bias_state.value}; "
            f"акселерометр: {accel_status}; "
            f"магнитометр: {mag_status}; "
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
        self.attitude_view.set_orientation(
            state.relative_quaternion,
            fresh=state.fresh and state.orientation_valid,
        )

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
