"""Operator-driven measurement controls; geometry remains in Core."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from navmin.contracts import VisionResult
from navmin.core import Mediator
from navmin.core.motion_calibration import MotionObservation, estimate_motion


class MotionErrorMap(QWidget):
    """Plot measured movement against the command with error-coloured points."""

    def __init__(self, owner: QWidget) -> None:
        super().__init__(owner)
        self._samples: tuple[MotionObservation, ...] = ()
        self.setMinimumHeight(210)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def maximum_abs_error_deg(self) -> float:
        return max((abs(sample.error_deg) for sample in self._samples), default=0.0)

    def set_observations(self, observations: list[MotionObservation], axis: str) -> None:
        self._samples = tuple(sample for sample in observations if sample.axis == axis)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(54, 24, -18, -42)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(QPen(self.palette().mid().color()))
        painter.drawRect(area)
        if not self._samples:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Нет замеров для выбранной оси")
            return

        extent = max(
            1.0,
            *(abs(value) for sample in self._samples for value in (sample.commanded_deg, sample.measured_deg)),
        )

        def x_pos(angle: float) -> float:
            return area.left() + (angle + extent) * area.width() / (2 * extent)

        def y_pos(angle: float) -> float:
            return area.bottom() - (angle + extent) * area.height() / (2 * extent)

        painter.setPen(QPen(self.palette().mid().color(), 1, Qt.PenStyle.DashLine))
        painter.drawLine(area.left(), int(y_pos(0.0)), area.right(), int(y_pos(0.0)))
        painter.drawLine(int(x_pos(0.0)), area.top(), int(x_pos(0.0)), area.bottom())
        painter.setPen(QPen(QColor("#7f8c8d"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(x_pos(-extent)), int(y_pos(-extent)), int(x_pos(extent)), int(y_pos(extent)))

        maximum_error = self.maximum_abs_error_deg
        for sample in self._samples:
            ratio = 0.0 if maximum_error == 0 else abs(sample.error_deg) / maximum_error
            color = QColor.fromHsvF((1.0 - ratio) * 0.33, 0.85, 0.9)
            painter.setPen(QPen(color.darker(130), 1))
            painter.setBrush(color)
            painter.drawEllipse(int(x_pos(sample.commanded_deg)) - 5, int(y_pos(sample.measured_deg)) - 5, 10, 10)

        painter.setPen(self.palette().text().color())
        painter.drawText(4, area.top() + 8, "измерено, °")
        painter.drawText(area.left(), self.height() - 12, "команда, °")
        painter.drawText(
            area.right() - 245,
            self.height() - 12,
            f"зелёный — малая ошибка; красный — большая, max {maximum_error:.3f}°",
        )


class MotionCalibrationWindow(QDialog):
    def __init__(self, owner: QWidget, mediator: Mediator, save_motion_calibration=None) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._mediator = mediator
        self._save_motion_calibration = save_motion_calibration
        self._mark_after = False
        self._measuring = False
        self.setWindowTitle("Калибровка движения камер")
        self.resize(620, 700)
        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Камеры вращаются вместе с платформой. Выберите ось и кликните на "
            "неподвижную точку сцены в основном видео: будет выполнен поворот только "
            "по этой оси. После полной остановки нажмите кнопку ниже и отметьте "
            "ТУ ЖЕ точку сцены в новом кадре (не лазерное пятно).\n"
            "Проверяйте неподвижность сцены и отсутствие других движений. "
            "Для масштаба: предварительное движение, затем три разных угла подряд в плюс и в минус (разброс ≥2×). "
            "Для люфта: дополнительно минимум два реверса в каждую сторону. "
            "Используйте разные углы; движение должно заметно превышать люфт."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.axis = QComboBox()
        self.axis.addItems(["x", "y"])
        layout.addWidget(self.axis)
        self.status = QLabel("Кликните на неподвижную точку сцены")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.settled = QPushButton("Движение завершено — отметить точку повторно")
        self.settled.clicked.connect(self._arm_after)
        layout.addWidget(self.settled)
        self.cancel = QPushButton("Отменить замер / остановить движение")
        self.cancel.clicked.connect(self._cancel)
        layout.addWidget(self.cancel)
        reset = QPushButton("Удалить замеры и начать новую серию")
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)
        layout.addWidget(QLabel("Карта ошибки: команда по горизонтали, измерение по вертикали"))
        self.error_map = MotionErrorMap(self)
        layout.addWidget(self.error_map)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results)
        report = QPushButton("Сохранить отчёт измерений…")
        report.clicked.connect(self._save_report)
        layout.addWidget(report)
        config = QPushButton("Сохранить и применить калибровку выбранной оси")
        config.clicked.connect(self._save_config)
        layout.addWidget(config)
        self.axis.currentTextChanged.connect(self._refresh_error_map)
        self.refresh()

    def _refresh_error_map(self) -> None:
        self.error_map.set_observations(
            self._mediator.calibration_observations, self.axis.currentText()
        )

    def refresh(self) -> None:
        pending = self._mediator.calibration_pending
        self.axis.setEnabled(not pending)
        self.settled.setEnabled(pending and not self._mark_after)
        if not pending:
            self._mark_after = False
            if self._measuring:
                self.status.setText(
                    "Замер прерван. Нажмите «Отменить замер» перед новым движением."
                )

    def _arm_after(self) -> None:
        if self._mediator.calibration_pending:
            self._mark_after = True
            self.status.setText("Отметьте ту же неподвижную точку в новом кадре")
        self.refresh()

    def handle_click(self, result: VisionResult, x: float, y: float) -> None:
        try:
            if not self._mediator.calibration_pending:
                if self._measuring:
                    raise ValueError("Замер прерван. Сначала нажмите «Отменить замер».")
                angle = self._mediator.begin_motion_measurement(
                    result, x, y, self.axis.currentText()
                )
                self._measuring = True
                self.status.setText(
                    f"Команда {angle:+.3f}°. Дождитесь полной остановки."
                )
            elif self._mark_after:
                sample = self._mediator.finish_motion_measurement(result, x, y)
                self._measuring = False
                self.results.append(
                    f"{sample.axis}: команда {sample.commanded_deg:+.3f}°, "
                    f"измерено {sample.measured_deg:+.3f}°, ошибка {sample.error_deg:+.3f}°, "
                    f"реверс: {sample.reversal}"
                )
                self.status.setText("Замер записан. Выберите следующую точку сцены.")
                self._show_estimate()
                self._refresh_error_map()
        except ValueError as exc:
            self.status.setText(str(exc))
        self.refresh()

    def _show_estimate(self) -> None:
        try:
            estimate = estimate_motion(
                self._mediator.calibration_observations, self.axis.currentText()
            )
        except ValueError as exc:
            self.results.append(str(exc))
            return
        self.results.append(
            f"Диагностический линейный масштаб: {estimate.gain:.6f}; "
            f"RMS линейной модели: {estimate.rms_deg:.3f}°. "
            f"Кривая: {len(estimate.curve)} узлов, диапазон "
            f"{estimate.curve[0][0]:+.3f}…{estimate.curve[-1][0]:+.3f}°. "
            f"Оценка люфта (+ / −): {estimate.backlash_positive_deg} / "
            f"{estimate.backlash_negative_deg}° (None = мало замеров)."
        )

    def _cancel(self) -> None:
        self._mediator.stop_motion()
        self._measuring = False
        self.status.setText(
            "Замер отменён. Следующее движение — с неизвестным преднатягом."
        )
        self.refresh()

    def _reset(self) -> None:
        self._cancel()
        self._mediator.calibration_observations.clear()
        self.results.clear()
        self._refresh_error_map()
        self.status.setText("Новая серия. Кликните на неподвижную точку сцены.")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._measuring:
            self._cancel()
        else:
            self._mediator.cancel_motion_calibration()
        super().closeEvent(event)

    def reject(self) -> None:
        if self._measuring:
            self._cancel()
        else:
            self._mediator.cancel_motion_calibration()
        super().reject()

    def _save_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Отчёт", "motion-calibration.json", "JSON (*.json)"
        )
        if not path:
            return
        samples = [
            dict(asdict(s), camera=s.camera.value)
            for s in self._mediator.calibration_observations
        ]
        try:
            Path(path).write_text(
                json.dumps(
                    {
                        "schema-version": 1,
                        "completion": "operator-confirmed",
                        "axes": asdict(self._mediator.calibration_axes),
                        "observations": samples,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self.status.setText("Отчёт сохранён")
        except (OSError, TypeError, ValueError) as exc:
            self.status.setText(str(exc))

    def _save_config(self) -> None:
        try:
            if self._mediator.calibration_pending:
                raise ValueError("Сначала завершите замер")
            axis = self.axis.currentText()
            estimate = estimate_motion(self._mediator.calibration_observations, axis)
            if self._save_motion_calibration is None:
                raise ValueError("Сохранение калибровки недоступно в этом запуске")
            destination = self._save_motion_calibration(axis, estimate)
            self.status.setText(
                f"Сохранено: {destination}. Поправка передана для следующих движений RELATIVE, "
                "перезапуск не нужен. Люфт автоматически не компенсируется."
            )
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
