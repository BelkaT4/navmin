"""Operator-facing automated motion calibration wizard."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from time import monotonic_ns

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from navmin.contracts import (
    CameraRole,
    CameraSessionStarted,
    CameraState,
    CameraStatus,
    MotorState,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.core import Mediator
from navmin.motion_calibration import (
    CameraMotionCalibration,
    MotionCalibrationError,
    MotionCalibrationRuntime,
    MotionCalibrationWizard,
    MotionProfile,
    WizardPhase,
    WizardSettings,
)

_CAMERAS = (
    (CameraRole.OVERVIEW, "Overview"),
    (CameraRole.STEREO_LEFT, "Stereo Left"),
)
LOGGER = logging.getLogger(__name__)
_PROFILE_APPLY_TIMEOUT_NS = 5_000_000_000


class MotionCalibrationWindow(QWidget):
    """One modeless wizard; image clicks still come from the main video view."""

    def __init__(
        self,
        *,
        owner: QWidget,
        mediator: Mediator,
        runtime: MotionCalibrationRuntime,
        begin_motion_profile: Callable[[MotionProfile], None],
        restore_motion_profile: Callable[[], None],
        stale_timeout_ms: int,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        super().__init__(owner, Qt.WindowType.Tool)
        self._mediator = mediator
        self._runtime = runtime
        self._begin_motion_profile = begin_motion_profile
        self._restore_motion_profile = restore_motion_profile
        self._stale_timeout_ms = stale_timeout_ms
        self._clock_ns = clock_ns
        self._wizard: MotionCalibrationWizard | None = None
        self._profile: CameraMotionCalibration | None = None
        self._session_generation: int | None = None
        self._temporary_profile_active = False
        self._awaiting_profile_confirmation = False
        self._profile_confirmation_deadline_ns: int | None = None

        self.setWindowTitle("Калибровка движения")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self.camera_combo = QComboBox()
        for _camera, label in _CAMERAS:
            self.camera_combo.addItem(label)
        self.angle_inputs: dict[str, QDoubleSpinBox] = {}
        angles_group = QGroupBox("Командные углы, градусы")
        angles_layout = QGridLayout(angles_group)
        angles_layout.addWidget(QLabel("Ось"), 0, 0)
        for column, title in enumerate(("A1", "A2", "A3"), start=1):
            angles_layout.addWidget(QLabel(title), 0, column)
        defaults = {"x": (4.0, 8.0, 14.0), "y": (3.0, 6.0, 10.0)}
        for row, axis in enumerate(("x", "y"), start=1):
            angles_layout.addWidget(QLabel(axis.upper()), row, 0)
            for column, value in enumerate(defaults[axis], start=1):
                editor = QDoubleSpinBox()
                editor.setRange(0.1, 360.0)
                editor.setDecimals(2)
                editor.setValue(value)
                self.angle_inputs[f"{axis}{column}"] = editor
                angles_layout.addWidget(editor, row, column)

        state = mediator.turret_state
        max_speed = min(
            value for value in (state.max_speed_x_deg_s, state.max_speed_y_deg_s) if value
        ) if state.max_speed_x_deg_s and state.max_speed_y_deg_s else 10.0
        acceleration = min(
            value
            for value in (state.acceleration_x_deg_s2, state.acceleration_y_deg_s2)
            if value
        ) if state.acceleration_x_deg_s2 and state.acceleration_y_deg_s2 else 20.0
        self.extrapolation_input = QDoubleSpinBox()
        self.extrapolation_input.setRange(0.0, 100.0)
        self.extrapolation_input.setValue(20.0)
        self.extrapolation_input.setSuffix(" %")
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.1, max_speed)
        self.speed_input.setValue(min(5.0, max_speed))
        self.speed_input.setSuffix(" °/с")
        self.acceleration_input = QDoubleSpinBox()
        self.acceleration_input.setRange(0.1, acceleration)
        self.acceleration_input.setValue(min(10.0, acceleration))
        self.acceleration_input.setSuffix(" °/с²")
        self.settle_input = QSpinBox()
        self.settle_input.setRange(0, 10_000)
        self.settle_input.setValue(750)
        self.settle_input.setSuffix(" мс")

        settings = QGroupBox("Параметры")
        settings_layout = QFormLayout(settings)
        settings_layout.addRow("Камера", self.camera_combo)
        settings_layout.addRow("Внешняя зона", self.extrapolation_input)
        settings_layout.addRow("Макс. скорость", self.speed_input)
        settings_layout.addRow("Ускорение", self.acceleration_input)
        settings_layout.addRow("Запас стабилизации", self.settle_input)

        self.progress_label = QLabel("Задайте параметры и нажмите «Начать».")
        self.progress_label.setWordWrap(True)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.start_button = QPushButton("Начать")
        self.start_button.clicked.connect(self._start)
        self.repeat_button = QPushButton("Повторить проблемный шаг")
        self.repeat_button.clicked.connect(self._repeat)
        self.repeat_button.setVisible(False)
        self.accept_button = QPushButton("Принять расхождение")
        self.accept_button.clicked.connect(self._accept_disagreement)
        self.accept_button.setVisible(False)
        self.save_button = QPushButton("Сохранить и применить")
        self.save_button.clicked.connect(self._save)
        self.save_button.setEnabled(False)
        self.cancel_button = QPushButton("Отмена / закрыть")
        self.cancel_button.clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(settings)
        layout.addWidget(angles_group)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.repeat_button)
        layout.addWidget(self.accept_button)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.cancel_button)
        self.setMinimumWidth(520)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @property
    def calibration_active(self) -> bool:
        return bool(
            self._wizard is not None
            and self._wizard.phase
            not in (WizardPhase.COMPLETE, WizardPhase.CANCELLED)
        )

    def handle_click(self, result: VisionResult, x_px: float, y_px: float) -> bool:
        wizard = self._wizard
        if not self.calibration_active or wizard is None:
            return False
        model = self._mediator.session_gate.camera_model(
            result.frame.camera, result.frame.generation
        )
        if model is None:
            self.abort("camera model unavailable")
            return True
        try:
            wizard.handle_click(
                frame=result.frame,
                camera_model=model,
                x_px=x_px,
                y_px=y_px,
            )
        except MotionCalibrationError as exc:
            if wizard.phase == WizardPhase.CANCELLED:
                self.abort(str(exc))
            else:
                self.progress_label.setText(f"Измерение отклонено: {exc}")
            return True
        except Exception as exc:
            LOGGER.exception("Unexpected motion calibration measurement failure")
            self.abort(f"непредвиденная ошибка измерения: {exc}")
            return True
        self._refresh_progress()
        self._finish_if_complete()
        return True

    def accept_camera_session(self, session: CameraSessionStarted) -> None:
        wizard = self._wizard
        if (
            self.calibration_active
            and wizard is not None
            and session.camera is wizard.settings.camera
            and session.generation != self._session_generation
        ):
            self.abort("camera generation changed")

    def accept_camera_status(self, status: CameraStatus) -> None:
        wizard = self._wizard
        if (
            self.calibration_active
            and wizard is not None
            and status.camera is wizard.settings.camera
            and status.state is not CameraState.ONLINE
        ):
            self.abort(f"camera state changed to {status.state.value}")

    def accept_turret_state(self, state: TurretState) -> None:
        if not self.calibration_active:
            self._update_motion_limits(state)
            return
        if state.connection_state is not TurretConnectionState.READY:
            self.abort("turret disconnected")
        elif state.motor_state is not MotorState.ON:
            self.abort("motor is not ON")
        elif state.control_mode is not TurretControlMode.RELATIVE:
            self.abort("control mode changed")
        elif self._awaiting_profile_confirmation:
            self._start_wizard_if_profile_confirmed(state)

    def _update_motion_limits(self, state: TurretState) -> None:
        speeds = (state.max_speed_x_deg_s, state.max_speed_y_deg_s)
        accelerations = (
            state.acceleration_x_deg_s2,
            state.acceleration_y_deg_s2,
        )
        if all(value is not None and value > 0.0 for value in speeds):
            self.speed_input.setMaximum(min(float(value) for value in speeds if value))
        if all(value is not None and value > 0.0 for value in accelerations):
            self.acceleration_input.setMaximum(
                min(float(value) for value in accelerations if value)
            )

    def abort(self, reason: str) -> None:
        wizard = self._wizard
        if wizard is not None:
            wizard.cancel(reason)
        self._restore_profile_once()
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self._profile = None
        self._awaiting_profile_confirmation = False
        self._profile_confirmation_deadline_ns = None
        self._set_inputs_enabled(True)
        self.repeat_button.setVisible(False)
        self.accept_button.setVisible(False)
        self.progress_label.setText(
            f"Калибровка прервана: {reason}. Старый профиль не изменён."
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.calibration_active:
            self.abort("окно калибровки закрыто")
        elif self._profile is not None and self.save_button.isEnabled():
            self._discard_unsaved_result()
        self._restore_profile_once()
        super().closeEvent(event)

    def _start(self) -> None:
        if self.calibration_active:
            return
        camera = _CAMERAS[self.camera_combo.currentIndex()][0]
        state = self._mediator.turret_state
        if camera is not self._mediator.main_camera:
            self._show_error("Сначала сделайте выбранную камеру основной.")
            return
        if (
            state.connection_state is not TurretConnectionState.READY
            or state.motor_state is not MotorState.ON
            or state.control_mode is not TurretControlMode.RELATIVE
            or self._mediator.pending_control_mode is not None
        ):
            self._show_error(
                "Нужны READY, MOTOR ON и подтверждённый режим RELATIVE."
            )
            return
        generation = self._mediator.session_gate.accepted_generation(camera)
        if generation is None:
            self._show_error("У выбранной камеры нет активной generation.")
            return
        profile = MotionProfile(
            max_speed_deg_s=self.speed_input.value(),
            acceleration_deg_s2=self.acceleration_input.value(),
            settle_margin_ms=self.settle_input.value(),
        )
        settings = WizardSettings(
            camera=camera,
            x_angles=tuple(
                self.angle_inputs[f"x{index}"].value() for index in range(1, 4)
            ),
            y_angles=tuple(
                self.angle_inputs[f"y{index}"].value() for index in range(1, 4)
            ),
            extrapolation_percent=self.extrapolation_input.value(),
            motion_profile=profile,
        )
        try:
            wizard = MotionCalibrationWizard(
                settings=settings,
                mechanics=self._runtime.axes,
                camera_calibration_identity=self._runtime.camera_identities[camera],
                stale_timeout_ms=self._stale_timeout_ms,
                submit_move=self._mediator.submit_calibration_move,
                clock_ns=self._clock_ns,
            )
            self._begin_motion_profile(profile)
            self._temporary_profile_active = True
        except (KeyError, MotionCalibrationError, RuntimeError, TypeError, ValueError) as exc:
            self._restore_profile_once()
            self._show_error(str(exc))
            return
        self._wizard = wizard
        self._profile = None
        self._session_generation = generation
        self._awaiting_profile_confirmation = True
        self._profile_confirmation_deadline_ns = (
            self._clock_ns() + _PROFILE_APPLY_TIMEOUT_NS
        )
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.summary_label.clear()
        self._set_inputs_enabled(False)
        self.progress_label.setText(
            "Применяется безопасный профиль скорости и ускорения калибровки…"
        )
        self._start_wizard_if_profile_confirmed(state)

    def _tick(self) -> None:
        wizard = self._wizard
        if not self.calibration_active or wizard is None:
            return
        if self._awaiting_profile_confirmation:
            deadline = self._profile_confirmation_deadline_ns
            if deadline is not None and self._clock_ns() >= deadline:
                self.abort(
                    "временный профиль скорости/ускорения не подтверждён STM32"
                )
            return
        wizard.tick()
        self._refresh_progress()
        self._finish_if_complete()

    def _start_wizard_if_profile_confirmed(self, state: TurretState) -> None:
        if not self._awaiting_profile_confirmation:
            return
        wizard = self._wizard
        profile = wizard.settings.motion_profile if wizard is not None else None
        generation = self._session_generation
        if wizard is None or profile is None or generation is None:
            self.abort("не удалось подготовить запуск калибровки")
            return
        values = (
            state.max_speed_x_deg_s,
            state.max_speed_y_deg_s,
            state.acceleration_x_deg_s2,
            state.acceleration_y_deg_s2,
        )
        if any(value is None for value in values):
            return
        speed_x, speed_y, accel_x, accel_y = values
        if not (
            math.isclose(float(speed_x), profile.max_speed_deg_s, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(float(speed_y), profile.max_speed_deg_s, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(
                float(accel_x), profile.acceleration_deg_s2, rel_tol=1e-9, abs_tol=1e-9
            )
            and math.isclose(
                float(accel_y), profile.acceleration_deg_s2, rel_tol=1e-9, abs_tol=1e-9
            )
        ):
            return
        self._awaiting_profile_confirmation = False
        self._profile_confirmation_deadline_ns = None
        try:
            wizard.start(generation=generation)
        except MotionCalibrationError as exc:
            self.abort(f"не удалось начать калибровку: {exc}")
            return
        self._refresh_progress()

    def _discard_unsaved_result(self) -> None:
        self._wizard = None
        self._profile = None
        self._session_generation = None
        self.save_button.setEnabled(False)
        self.summary_label.clear()
        self.progress_label.setText("Несохранённый результат калибровки отброшен.")

    def _finish_if_complete(self) -> None:
        wizard = self._wizard
        if wizard is None or wizard.phase != WizardPhase.COMPLETE:
            return
        self._restore_profile_once()
        try:
            self._profile = wizard.build_profile()
        except MotionCalibrationError as exc:
            self.abort(f"результат не прошёл validation: {exc}")
            return
        self.progress_label.setText(
            "Серия завершена. Проверьте итог и выберите «Сохранить и применить» "
            "или закройте окно без сохранения."
        )
        self.summary_label.setText(self._format_summary(self._profile))
        self.save_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self._set_inputs_enabled(True)

    def _save(self) -> None:
        profile = self._profile
        wizard = self._wizard
        if profile is None or wizard is None or wizard.phase != WizardPhase.COMPLETE:
            return
        try:
            self._runtime.save_and_apply(
                profile,
                extrapolation_percent=wizard.settings.extrapolation_percent,
            )
        except OSError as exc:
            self.progress_label.setText(
                f"Файл не сохранён: {exc}. Старый runtime-профиль остался активным."
            )
            return
        self.save_button.setEnabled(False)
        self._profile = None
        self.progress_label.setText("Калибровка сохранена и применена.")

    def _repeat(self) -> None:
        wizard = self._wizard
        if wizard is None:
            return
        try:
            wizard.repeat_problematic_step()
        except MotionCalibrationError as exc:
            self._show_error(str(exc))
            return
        self._refresh_progress()

    def _accept_disagreement(self) -> None:
        wizard = self._wizard
        if wizard is None:
            return
        try:
            wizard.accept_disagreement()
        except MotionCalibrationError as exc:
            self._show_error(str(exc))
            return
        self._refresh_progress()
        self._finish_if_complete()

    def _refresh_progress(self) -> None:
        wizard = self._wizard
        if wizard is None:
            return
        step = wizard.current_step
        total = len(wizard.steps)
        number = min(wizard.step_index + 1, total)
        prefix = f"Camera: {wizard.settings.camera.value}; шаг {number}/{total}. "
        if wizard.phase == WizardPhase.MOVING:
            text = "Турель выполняет движение. Не выбирайте точку до стабилизации."
        elif wizard.phase == WizardPhase.WAIT_FIRST_CLICK:
            text = "Выберите неподвижную точку."
        elif wizard.phase == WizardPhase.WAIT_SECOND_CLICK:
            text = "Отметьте ту же неподвижную точку."
        elif wizard.phase == WizardPhase.REVIEW_REPEAT:
            text = "Повторные измерения заметно расходятся. Повторите шаг или примите расхождение."
        elif wizard.phase == WizardPhase.COMPLETE:
            text = "Серия завершена."
        else:
            text = wizard.phase
        if step is not None:
            text += f" Ось {step.axis.upper()}, команда {step.command_deg:+g}°."
        self.progress_label.setText(prefix + text)
        review = wizard.phase == WizardPhase.REVIEW_REPEAT
        self.repeat_button.setVisible(review)
        self.accept_button.setVisible(review)

    def _restore_profile_once(self) -> None:
        if not self._temporary_profile_active:
            return
        try:
            self._restore_motion_profile()
        finally:
            self._temporary_profile_active = False

    def _set_inputs_enabled(self, enabled: bool) -> None:
        self.camera_combo.setEnabled(enabled)
        self.extrapolation_input.setEnabled(enabled)
        self.speed_input.setEnabled(enabled)
        self.acceleration_input.setEnabled(enabled)
        self.settle_input.setEnabled(enabled)
        for editor in self.angle_inputs.values():
            editor.setEnabled(enabled)

    def _format_summary(self, profile: CameraMotionCalibration) -> str:
        lines = [
            f"Camera: {profile.camera.value}",
            (
                "Motion profile: "
                f"{profile.motion_profile.max_speed_deg_s:g} °/с, "
                f"{profile.motion_profile.acceleration_deg_s2:g} °/с², "
                f"settle {profile.motion_profile.settle_margin_ms} мс"
            ),
        ]
        for axis in ("x", "y"):
            value = profile.axes.get(axis)
            if value is None:
                lines.append(f"{axis.upper()}: недоступно")
                continue
            mechanics = self._runtime.axes.x if axis == "x" else self._runtime.axes.y
            percent = self.extrapolation_input.value()
            calibration_limit = (
                profile.calibration_limit_x_deg
                if axis == "x"
                else profile.calibration_limit_y_deg
            )
            effective_limit = min(mechanics.max_relative_move_deg, calibration_limit)
            positive_extended = min(
                effective_limit,
                value.positive.trusted_boundary_deg * (1.0 + percent / 100.0),
            )
            negative_extended = min(
                effective_limit,
                value.negative.trusted_boundary_deg * (1.0 + percent / 100.0),
            )
            loss = value.reversal_loss
            lines.append(
                f"{axis.upper()}: trusted -{value.negative.trusted_boundary_deg:.3f}° "
                f"… +{value.positive.trusted_boundary_deg:.3f}°; "
                f"extended -{negative_extended:.3f}° … +{positive_extended:.3f}°; "
                f"repeat spread {value.quality_max_repeat_spread_deg:.3f}°; "
                f"reversal +→- {_optional(loss.positive_to_negative_deg)}; "
                f"-→+ {_optional(loss.negative_to_positive_deg)}"
            )
        if profile.warnings:
            lines.append("Предупреждения: " + "; ".join(profile.warnings))
        return "\n".join(lines)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Калибровка движения", message)


def _optional(value: float | None) -> str:
    return "недоступно" if value is None else f"{value:.3f}°"


__all__ = ["MotionCalibrationWindow"]
