"""Standalone, gyro-gated backlash calibration window."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from time import monotonic

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from navmin.backlash_calibration import (
    AxisBacklashDelay,
    BacklashCalibrationRuntime,
)
from navmin.concurrency import LatestValue
from navmin.contracts import (
    MotorState,
    MoveRelativeCommand,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
)
from navmin.core import Mediator
from navmin.imu import (
    GyroBiasState,
    ImuCalibrationControls,
    ImuRuntimeState,
)

_STEPS: tuple[tuple[str, int, str], ...] = (
    ("x", 1, "prime"),
    ("x", 1, "baseline+"),
    ("x", -1, "reverse+to-"),
    ("x", -1, "baseline-"),
    ("x", 1, "reverse-to+"),
    ("x", -1, "reverse+to-"),
    ("y", 1, "prime"),
    ("y", 1, "baseline+"),
    ("y", -1, "reverse+to-"),
    ("y", -1, "baseline-"),
    ("y", 1, "reverse-to+"),
    ("y", -1, "reverse+to-"),
    ("x", -1, "prime"),
    ("x", -1, "baseline-"),
    ("x", 1, "reverse-to+"),
    ("x", 1, "baseline+"),
    ("x", -1, "reverse+to-"),
    ("x", 1, "reverse-to+"),
    ("y", -1, "prime"),
    ("y", -1, "baseline-"),
    ("y", 1, "reverse-to+"),
    ("y", 1, "baseline+"),
    ("y", -1, "reverse+to-"),
    ("y", 1, "reverse-to+"),
)


class BacklashCalibrationWindow(QWidget):
    """Calibrate directional reversal dead time independently from camera motion."""

    def __init__(
        self,
        *,
        owner: QWidget,
        mediator: Mediator,
        imu_states: LatestValue[ImuRuntimeState] | None,
        imu_controls: ImuCalibrationControls | None,
        turret_states: LatestValue[TurretState],
        runtime: BacklashCalibrationRuntime,
        motion_calibration_active: Callable[[], bool],
        max_relative_move_x_deg: float,
        max_relative_move_y_deg: float,
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._mediator = mediator
        self._imu_states = imu_states
        self._imu_controls = imu_controls
        self._turret_states = turret_states
        self._runtime = runtime
        self._motion_calibration_active = motion_calibration_active
        self._max_angles = {"x": max_relative_move_x_deg, "y": max_relative_move_y_deg}
        self._running = False
        self._waiting_for_gyro_calibration = False
        self._gyro_bias_reacquisition_seen = False
        self._step_index = 0
        self._waiting_for_start = False
        self._waiting_for_redirect = False
        self._command_time = 0.0
        self._step_deadline = 0.0
        self._step_redirect_baseline: float | None = None
        self._step_onset_delay_s: float | None = None
        self._measurements: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._pending_profile: tuple[AxisBacklashDelay, AxisBacklashDelay] | None = None

        self.setWindowTitle("NavMin — отдельная калибровка люфта")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(520, 390)

        note = QLabel(
            "Калибровка не использует камеры. При запуске мастер заново измерит "
            "неподвижное смещение гироскопа (около 5 секунд), затем выполнит короткие "
            "движения по X/Y и сохранит задержку фактического старта после реверса."
        )
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(note)

        settings = QGroupBox("Параметры измерения")
        form = QFormLayout(settings)
        self.angle_input = QDoubleSpinBox()
        max_test_angle = min(
            10.0, max_relative_move_x_deg / 4.0, max_relative_move_y_deg / 4.0
        )
        max_test_angle = max(0.01, max_test_angle)
        self.angle_input.setRange(min(0.05, max_test_angle), max_test_angle)
        self.angle_input.setSingleStep(min(0.05, max_test_angle))
        self.angle_input.setValue(min(5.0, max_test_angle))
        self.angle_input.setDecimals(1)
        self.angle_input.setSuffix("° на шаг")
        form.addRow("Тестовый ход", self.angle_input)
        layout.addWidget(settings)

        self.status_label = QLabel("Ожидание. Турель должна быть неподвижна.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.result_label = QLabel(self._format_current_profile())
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        self.start_button = QPushButton("Начать калибровку люфта")
        self.start_button.clicked.connect(self.start)
        self.save_button = QPushButton("Сохранить профиль")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        self.cancel_button = QPushButton("Остановить / закрыть")
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.start_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.cancel_button)

        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._advance)

    def start(self) -> None:
        if self._running:
            return
        if self._motion_calibration_active():
            QMessageBox.warning(
                self,
                "Калибровка недоступна",
                "Сначала завершите калибровку движения камеры.",
            )
            return
        imu = self._imu_state()
        turret = self._turret_state()
        reason = self._readiness_error(imu, turret, require_bias_ready=False)
        if reason is None and imu is not None and imu.motion_detected:
            reason = "Гироскоп видит движение. Остановите турель и повторите."
        if reason is None and self._imu_controls is None:
            reason = "Управление калибровкой гироскопа недоступно."
        if reason:
            QMessageBox.warning(self, "Калибровка недоступна", reason)
            return
        if not self._mediator.begin_backlash_calibration():
            QMessageBox.warning(
                self,
                "Калибровка недоступна",
                "Не удалось зарезервировать управление относительным движением.",
            )
            return
        assert self._imu_controls is not None
        if not self._imu_controls.request_gyro_bias_calibration():
            self._mediator.end_backlash_calibration()
            QMessageBox.warning(
                self,
                "Калибровка недоступна",
                "IMU не принял запрос на стационарную калибровку гироскопа.",
            )
            return
        self._measurements.clear()
        self._step_index = 0
        self._running = True
        self._waiting_for_gyro_calibration = True
        self._gyro_bias_reacquisition_seen = False
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self._step_deadline = monotonic() + 10.0
        self.status_label.setText(
            "Калибрую смещение гироскопа: не двигайте турель примерно 5 секунд."
        )
        self._timer.start()

    def _issue_step(self) -> None:
        axis, direction, _kind = _STEPS[self._step_index]
        angle = self.angle_input.value() * direction
        command = (
            MoveRelativeCommand(angle, 0.0)
            if axis == "x"
            else MoveRelativeCommand(0.0, angle)
        )
        self._command_time = monotonic()
        imu = self._imu_state()
        self._step_redirect_baseline = (
            None if imu is None else imu.motion_redirected_timestamp_s
        )
        self._step_onset_delay_s = None
        if not self._mediator.submit_backlash_calibration_move(command):
            self._abort("Турель перестала быть готова к MOVE_RELATIVE.")
            return
        self._waiting_for_start = True
        self._waiting_for_redirect = False
        self._step_deadline = self._command_time + 5.0
        self.status_label.setText(
            f"Шаг {self._step_index + 1}/{len(_STEPS)}: "
            f"{axis.upper()} {'+' if direction > 0 else '−'}"
            f"{self.angle_input.value():.1f}°. "
            "Жду фактическое движение по гироскопу."
        )

    def _advance(self) -> None:
        imu = self._imu_state()
        turret = self._turret_state()
        if self._waiting_for_gyro_calibration:
            reason = self._readiness_error(imu, turret, require_bias_ready=False)
            if reason:
                self._abort(reason)
                return
            assert imu is not None
            if imu.motion_detected:
                self._abort("Турель сдвинулась во время стационарной калибровки IMU.")
                return
            if imu.gyro_bias_state is not GyroBiasState.READY:
                self._gyro_bias_reacquisition_seen = True
            if (
                self._gyro_bias_reacquisition_seen
                and imu.gyro_bias_state is GyroBiasState.READY
                and imu.orientation_valid
            ):
                self._waiting_for_gyro_calibration = False
                self._issue_step()
            elif monotonic() >= self._step_deadline:
                self._abort(
                    "Гироскоп не завершил stationary bias calibration за 10 секунд."
                )
            return
        reason = self._readiness_error(imu, turret)
        if reason:
            self._abort(reason)
            return
        assert imu is not None
        now = monotonic()
        if self._waiting_for_redirect:
            if (
                imu.motion_redirected_timestamp_s is not None
                and imu.motion_redirected_timestamp_s
                != self._step_redirect_baseline
            ):
                axis, _direction, kind = _STEPS[self._step_index]
                if kind != "prime" and self._step_onset_delay_s is not None:
                    self._measurements[(axis, kind)].append(
                        self._step_onset_delay_s
                    )
                self._waiting_for_redirect = False
                self._waiting_for_start = False
                self.status_label.setText(
                    f"Шаг {self._step_index + 1}/{len(_STEPS)}: "
                    "гироскоп подтвердил реверс; жду остановку."
                )
            elif now >= self._step_deadline:
                self._abort("Гироскоп не подтвердил фактический реверс за 1 секунду.")
            return
        if self._waiting_for_start:
            if imu.motion_detected:
                axis, _direction, kind = _STEPS[self._step_index]
                motion_started_at = _host_motion_start_time(imu, now)
                self._step_onset_delay_s = max(
                    0.0, motion_started_at - self._command_time
                )
                reverse_confirmation_s = 1.0 if kind.startswith("reverse") else 10.0
                self._step_deadline = now + reverse_confirmation_s
                if kind.startswith("reverse") and (
                    imu.motion_redirected_timestamp_s
                    == self._step_redirect_baseline
                ):
                    self._waiting_for_redirect = True
                    self.status_label.setText(
                        f"Шаг {self._step_index + 1}/{len(_STEPS)}: "
                        "ожидаю подтверждения смены направления гироскопом."
                    )
                    return
                if kind != "prime" and self._step_onset_delay_s is not None:
                    self._measurements[(axis, kind)].append(self._step_onset_delay_s)
                self._waiting_for_start = False
                self.status_label.setText(
                    f"Шаг {self._step_index + 1}/{len(_STEPS)}: "
                    "гироскоп подтвердил движение; жду остановку."
                )
                return
            if now >= self._step_deadline:
                self._abort("Гироскоп не обнаружил движение за 5 секунд.")
            return
        if not imu.motion_detected:
            self._step_index += 1
            if self._step_index >= len(_STEPS):
                self._finish()
            else:
                self._issue_step()
        elif now >= self._step_deadline:
            self._abort("Движение не остановилось за 10 секунд.")

    def _finish(self) -> None:
        try:
            x = self._calculate_axis("x")
            y = self._calculate_axis("y")
        except ValueError as exc:
            self._abort(str(exc))
            return
        self._pending_profile = (x, y)
        self._running = False
        self._mediator.end_backlash_calibration()
        self._timer.stop()
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.status_label.setText(
            "Измерения завершены. Проверьте задержки и сохраните профиль."
        )
        self.result_label.setText(
            "Измеренная задержка люфта (с):\n"
            f"X: +→− {x.positive_to_negative_s:.3f}; "
            f"−→+ {x.negative_to_positive_s:.3f}\n"
            f"Y: +→− {y.positive_to_negative_s:.3f}; "
            f"−→+ {y.negative_to_positive_s:.3f}"
        )

    def _calculate_axis(self, axis: str) -> AxisBacklashDelay:
        values = self._measurements
        base_positive = values[(axis, "baseline+")]
        base_negative = values[(axis, "baseline-")]
        reverse_positive_to_negative = values[(axis, "reverse+to-")]
        reverse_negative_to_positive = values[(axis, "reverse-to+")]
        if not all(
            (
                base_positive,
                base_negative,
                reverse_positive_to_negative,
                reverse_negative_to_positive,
            )
        ):
            raise ValueError(
                f"Недостаточно подтверждённых измерений для оси {axis.upper()}."
            )
        baseline_p = sum(base_positive) / len(base_positive)
        baseline_n = sum(base_negative) / len(base_negative)
        reverse_pn = sum(reverse_positive_to_negative) / len(
            reverse_positive_to_negative
        )
        reverse_np = sum(reverse_negative_to_positive) / len(
            reverse_negative_to_positive
        )
        return AxisBacklashDelay(
            positive_to_negative_s=max(0.0, reverse_pn - baseline_n),
            negative_to_positive_s=max(0.0, reverse_np - baseline_p),
        )

    def save(self) -> None:
        profile = getattr(self, "_pending_profile", None)
        if profile is None:
            return
        try:
            self._runtime.save(*profile)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return
        self.result_label.setText(self._format_current_profile())
        self.status_label.setText(f"Профиль люфта сохранён: {self._runtime.path}")
        self.save_button.setEnabled(False)

    def cancel(self) -> None:
        if self._running:
            self._request_stop()
            self._abort("Калибровка остановлена оператором.", stop=False)
        self.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._running:
            self._abort("Окно закрыто оператором.")
        event.accept()

    def _abort(self, reason: str, *, stop: bool = True) -> None:
        if stop and self._running:
            self._request_stop()
        self._running = False
        self._waiting_for_gyro_calibration = False
        self._mediator.end_backlash_calibration()
        self._timer.stop()
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.status_label.setText(f"Калибровка прервана: {reason}")

    def _request_stop(self) -> None:
        if not self._mediator.stop_motion():
            self._mediator.emergency_stop()

    def _imu_state(self) -> ImuRuntimeState | None:
        return None if self._imu_states is None else self._imu_states.get()

    def _turret_state(self) -> TurretState | None:
        return self._turret_states.get()

    @staticmethod
    def _readiness_error(
        imu: ImuRuntimeState | None,
        turret: TurretState | None,
        *,
        require_bias_ready: bool = True,
    ) -> str | None:
        if imu is None or not imu.running or not imu.fresh or imu.stale:
            return "Нет свежих данных гироскопа."
        if require_bias_ready and (
            imu.gyro_bias_state is not GyroBiasState.READY
            or not imu.orientation_valid
        ):
            return "Сначала неподвижно откалибруйте гироскоп и дождитесь READY."
        if turret is None or turret.connection_state is not TurretConnectionState.READY:
            return "Контроллер турели не в состоянии READY."
        if turret.motor_state is not MotorState.ON:
            return "Включите моторы перед калибровкой."
        if turret.control_mode is not TurretControlMode.RELATIVE:
            return "Переключите турель в режим RELATIVE."
        return None

    def _format_current_profile(self) -> str:
        profile = self._runtime.profile
        if profile is None:
            return "Сохранённого профиля люфта для текущей механики нет."
        return (
            "Текущий профиль (задержка, с):\n"
            f"X: +→− {profile.x.positive_to_negative_s:.3f}; "
            f"−→+ {profile.x.negative_to_positive_s:.3f}\n"
            f"Y: +→− {profile.y.positive_to_negative_s:.3f}; "
            f"−→+ {profile.y.negative_to_positive_s:.3f}"
        )


def _host_motion_start_time(state: ImuRuntimeState, fallback_s: float) -> float:
    """Map the IMU sender event timestamp onto the receiver monotonic clock."""
    if (
        state.receive_timestamp_s is None
        or state.sender_timestamp_s is None
        or state.motion_started_timestamp_s is None
    ):
        return fallback_s
    sample_age_s = state.sender_timestamp_s - state.motion_started_timestamp_s
    if sample_age_s < 0.0:
        return fallback_s
    return state.receive_timestamp_s - sample_age_s


__all__ = ["BacklashCalibrationWindow"]
