"""Modeless UI for in-runtime Overview video-latency calibration."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from navmin.concurrency import LatestValue
from navmin.diagnostics.live_video_latency_calibration import (
    LiveCalibrationPhase,
    LiveVideoLatencyCalibrationError,
    LiveVideoLatencyCalibrationState,
)


class VideoLatencyCalibrationWindow(QWidget):
    """Show progress/result while production workers remain running."""

    def __init__(
        self,
        *,
        owner: QWidget,
        states: LatestValue[LiveVideoLatencyCalibrationState],
        start_calibration: Callable[[], bool],
        cancel_calibration: Callable[[], None],
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._states = states
        self._start_calibration = start_calibration
        self._cancel_calibration = cancel_calibration
        self._last_revision = 0

        self.setWindowTitle("Калибровка задержки видео")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumWidth(520)

        description = QLabel(
            "Используется текущий production-поток Overview без перезапуска камеры. "
            "Турель выполняет небольшой симметричный профиль только по оси Y; "
            "обычные команды движения на время измерения блокируются."
        )
        description.setWordWrap(True)
        self.status_label = QLabel("Готово к запуску.")
        self.status_label.setWordWrap(True)
        self.progress_label = QLabel("Ось: Y · движение: ±6 °/с")
        self.progress_label.setWordWrap(True)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.start_button = QPushButton("Начать калибровку")
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.hide)

        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.close_button)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    def closeEvent(self, event: QCloseEvent) -> None:
        state = self._states.get()
        if state is not None and state.phase in (
            LiveCalibrationPhase.WARMUP,
            LiveCalibrationPhase.RUNNING,
        ):
            self._cancel_calibration()
        super().closeEvent(event)

    def _start(self) -> None:
        answer = QMessageBox.question(
            self,
            "Калибровка задержки видео",
            "Убедитесь, что у турели есть свободный ход по оси Y, Overview смотрит "
            "на неподвижную детализированную сцену, а двигатели включены. Начать?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is not QMessageBox.StandardButton.Yes:
            return
        try:
            started = self._start_calibration()
        except (LiveVideoLatencyCalibrationError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, "Калибровка", str(exc))
            return
        if not started:
            QMessageBox.information(self, "Калибровка", "Калибровка уже выполняется.")

    def _cancel(self) -> None:
        self._cancel_calibration()
        self.cancel_button.setEnabled(False)

    def _poll(self) -> None:
        snapshot = self._states.snapshot()
        if snapshot.revision <= self._last_revision:
            return
        self._last_revision = snapshot.revision
        state = snapshot.value
        if state is None:
            return
        self.status_label.setText(state.message)
        active = state.phase in (
            LiveCalibrationPhase.WARMUP, LiveCalibrationPhase.RUNNING
        )
        self.start_button.setEnabled(not active)
        self.cancel_button.setEnabled(active)
        if state.phase is LiveCalibrationPhase.RUNNING:
            self.progress_label.setText(
                f"Прогон {state.run_index}/{state.run_count} · "
                f"Y={state.current_rate_y_deg_s:+.1f} °/с · "
                f"flow samples={state.flow_observations}"
            )
        elif state.phase is LiveCalibrationPhase.COMPLETE and state.result is not None:
            aggregate = state.result.aggregate
            self.progress_label.setText("Калибровочный профиль завершён, Y=0 °/с.")
            self.result_label.setText(
                f"Рекомендованное video-latency-ms: "
                f"{state.result.recommended_video_latency_ms}\n"
                f"median: {aggregate.median_latency_ms} мс\n"
                f"MAD: {aggregate.mad_latency_ms} мс\n"
                f"valid runs: {aggregate.valid_run_count}/{state.result.run_count}\n"
                "Значение автоматически в конфигурацию не записывается.\n"
                "Для использования результата укажите его вручную в "
                "tracking-control → platform-motion → video-latency-ms "
                "в config/navmin.json."
            )
        elif state.phase in (
            LiveCalibrationPhase.FAILED, LiveCalibrationPhase.CANCELLED
        ):
            self.progress_label.setText("Y=0 °/с; обычное управление восстановлено.")
            self.result_label.setText("")


__all__ = ["VideoLatencyCalibrationWindow"]
