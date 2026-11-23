"""Turret Controller: applied mode, PID ownership, and latest motion intent."""

from __future__ import annotations

import logging
from math import isfinite
from time import monotonic, monotonic_ns
from typing import Protocol

from navmin.config.models import (
    PidControllerConfig,
    TrackingControlConfig,
    TurretConfig,
)
from navmin.contracts import (
    AxisVelocitySetpoint,
    ConfigUpdate,
    MotorState,
    MoveRelativeCommand,
    TargetRef,
    TrackingControlSample,
    TurretControlMode,
)
from navmin.diagnostic_logging import emit_event

from .hal import TurretHal
from .pid import AxisPid, PidGains
from .protocol import ProtocolResponse, ResultCode
from .session import SessionResult

_LOGGER = logging.getLogger(__name__)


class Clock(Protocol):
    def __call__(self) -> float: ...


class ControllerError(RuntimeError):
    """Base class for Controller state/operation errors."""


class ControllerModeError(ControllerError):
    """A motion intent is incompatible with the authoritative applied mode."""


class ControllerStateError(ControllerError):
    """A control boundary cannot safely proceed from current confirmed state."""


type PendingMotion = MoveRelativeCommand | AxisVelocitySetpoint | None


class TurretController:
    """Single owner of applied control mode, PID, and latest pending motion."""

    def __init__(
        self,
        hal: TurretHal,
        config: TurretConfig,
        tracking_control_config: TrackingControlConfig,
        *,
        clock: Clock = monotonic,
    ) -> None:
        if not isinstance(hal, TurretHal):
            raise TypeError("hal must be TurretHal")
        if not isinstance(config, TurretConfig):
            raise TypeError("config must be TurretConfig")
        if not isinstance(tracking_control_config, TrackingControlConfig):
            raise TypeError("tracking_control_config must be TrackingControlConfig")
        self._hal = hal
        self._config = config
        self._tracking_control_config = tracking_control_config
        self._config_revision = 0
        self._clock = clock
        self._control_mode = TurretControlMode.RELATIVE
        self._pending_motion: PendingMotion = None
        self._last_processed_revision: int | None = None
        self._target: TargetRef | None = None
        self._last_tracking_time_s: float | None = None
        self._last_applied_tracking_velocity: AxisVelocitySetpoint | None = None
        self._next_control_id = 1
        self._pid_x = AxisPid(
            self._x_gains(config.controller),
            output_limit=config.stm32.max_speed_x_deg_s,
            integral_limit=self._x_integral_limit(config),
        )
        self._pid_y = AxisPid(
            self._y_gains(config.controller),
            output_limit=config.stm32.max_speed_y_deg_s,
            integral_limit=self._y_integral_limit(config),
        )

    @property
    def control_mode(self) -> TurretControlMode:
        return self._control_mode

    @property
    def pending_motion(self) -> PendingMotion:
        return self._pending_motion

    @property
    def last_processed_revision(self) -> int | None:
        return self._last_processed_revision

    @property
    def pid_x(self) -> AxisPid:
        return self._pid_x

    @property
    def pid_y(self) -> AxisPid:
        return self._pid_y

    def submit_move_relative(self, command: MoveRelativeCommand) -> None:
        if self._control_mode is not TurretControlMode.RELATIVE:
            raise ControllerModeError("MoveRelative is only valid in RELATIVE mode")
        if not isinstance(command, MoveRelativeCommand):
            raise TypeError("command must be MoveRelativeCommand")
        self._pending_motion = command

    def submit_tracking_control(
        self, revision: int, sample: TrackingControlSample
    ) -> bool:
        if self._control_mode is not TurretControlMode.TRACKING:
            raise ControllerModeError(
                "TrackingControlSample is only valid in TRACKING mode"
            )
        self._require_revision(revision)
        if not isinstance(sample, TrackingControlSample):
            raise TypeError("sample must be TrackingControlSample")
        if (
            self._last_processed_revision is not None
            and revision <= self._last_processed_revision
        ):
            return False
        if self._hal.platform_motion_zero_settling:
            self._pending_motion = None
            return False

        now = self._require_finite_time(self._clock())
        target_changed = self._target is not None and sample.target != self._target
        gap_exceeded = (
            self._last_tracking_time_s is not None
            and now - self._last_tracking_time_s
            > self._config.stm32.velocity_watchdog_timeout_ms / 1000.0
        )
        reset_reason = None
        if target_changed:
            reset_reason = "target_changed"
        elif gap_exceeded:
            reset_reason = "tracking_gap_exceeded"
        if reset_reason is not None:
            self._last_applied_tracking_velocity = None
            self._reset_pid()

        prediction = self._tracking_control_config.prediction
        feed_forward = self._tracking_control_config.feed_forward
        x_usable = sample.rate_valid_x and sample.target_rate_x_deg_s is not None
        y_usable = sample.rate_valid_y and sample.target_rate_y_deg_s is not None
        velocity_x = self._pid_x.step(
            sample.real_error_x_deg,
            now_s=now,
            proportional_offset=(
                sample.prediction_offset_x_deg
                if x_usable and prediction.enabled
                else 0.0
            ),
            feed_forward=(
                feed_forward.gain_x * sample.target_rate_x_deg_s
                if x_usable and feed_forward.enabled
                else 0.0
            ),
        )
        velocity_y = self._pid_y.step(
            sample.real_error_y_deg,
            now_s=now,
            proportional_offset=(
                sample.prediction_offset_y_deg
                if y_usable and prediction.enabled
                else 0.0
            ),
            feed_forward=(
                feed_forward.gain_y * sample.target_rate_y_deg_s
                if y_usable and feed_forward.enabled
                else 0.0
            ),
        )
        control_id = self._next_control_id
        self._next_control_id += 1
        self._log_tracking_control(
            control_id,
            sample,
            now_s=now,
            x_usable=x_usable,
            y_usable=y_usable,
            integrator_reset_reason=reset_reason,
        )
        previous_pending = self._pending_motion
        self._pending_motion = AxisVelocitySetpoint(
            velocity_x_deg_s=velocity_x,
            velocity_y_deg_s=velocity_y,
            timestamp_ns=sample.observation_timestamp_ns,
            control_id=control_id,
        )
        emit_event(
            _LOGGER,
            "CONTROL_DECISION",
            source_instance="turret:controller",
            control_id=control_id,
            decision=(
                "replace"
                if isinstance(previous_pending, AxisVelocitySetpoint)
                else "send"
            ),
            command_type="SET_VELOCITY",
            logical_velocity_x_deg_s=velocity_x,
            logical_velocity_y_deg_s=velocity_y,
            replaced_control_id=(
                previous_pending.control_id
                if isinstance(previous_pending, AxisVelocitySetpoint)
                else None
            ),
            reason=None,
            pending_revision=revision,
        )
        self._target = sample.target
        self._last_tracking_time_s = now
        self._last_processed_revision = revision
        return True

    def _log_tracking_control(
        self,
        control_id: int,
        sample: TrackingControlSample,
        *,
        now_s: float,
        x_usable: bool,
        y_usable: bool,
        integrator_reset_reason: str | None,
    ) -> None:
        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        x_step = self._pid_x.last_step
        y_step = self._pid_y.last_step
        if x_step is None or y_step is None:
            return
        feed_forward = self._tracking_control_config.feed_forward
        dt_ms = (
            None
            if self._last_tracking_time_s is None
            else max(0.0, (now_s - self._last_tracking_time_s) * 1000.0)
        )
        observation_age_ms = max(
            0.0, (monotonic_ns() - sample.observation_timestamp_ns) / 1_000_000.0
        )
        emit_event(
            _LOGGER,
            "TRACK_CONTROL",
            source_instance="turret:controller",
            control_id=control_id,
            camera=sample.target.camera.value,
            generation=sample.target.generation,
            frame_id=sample.frame_id,
            target_id=sample.target.track_id,
            control_timestamp_ns=monotonic_ns(),
            dt_ms=dt_ms,
            observation_age_ms=observation_age_ms,
            rate_valid_x=sample.rate_valid_x,
            rate_valid_y=sample.rate_valid_y,
            x={
                "real_error_deg": sample.real_error_x_deg,
                "prediction_offset_deg": (
                    x_step.proportional_offset if x_usable else 0.0
                ),
                "p_error_deg": (
                    sample.real_error_x_deg
                    + (x_step.proportional_offset if x_usable else 0.0)
                ),
                "p_term_deg_s": x_step.p_term,
                "i_term_deg_s": x_step.i_term,
                "d_term_deg_s": x_step.d_term,
                "ff_term_deg_s": x_step.feed_forward,
                "ff_gain": feed_forward.gain_x,
                "target_rate_deg_s": sample.target_rate_x_deg_s,
                "i_state_deg_s": self._pid_x.i_term,
                "integrator_updated": x_step.integrator_updated,
                "integrator_reset": integrator_reset_reason is not None,
                "integrator_reset_reason": integrator_reset_reason,
            },
            y={
                "real_error_deg": sample.real_error_y_deg,
                "prediction_offset_deg": (
                    y_step.proportional_offset if y_usable else 0.0
                ),
                "p_error_deg": (
                    sample.real_error_y_deg
                    + (y_step.proportional_offset if y_usable else 0.0)
                ),
                "p_term_deg_s": y_step.p_term,
                "i_term_deg_s": y_step.i_term,
                "d_term_deg_s": y_step.d_term,
                "ff_term_deg_s": y_step.feed_forward,
                "ff_gain": feed_forward.gain_y,
                "target_rate_deg_s": sample.target_rate_y_deg_s,
                "i_state_deg_s": self._pid_y.i_term,
                "integrator_updated": y_step.integrator_updated,
                "integrator_reset": integrator_reset_reason is not None,
                "integrator_reset_reason": integrator_reset_reason,
            },
            raw_velocity_x_deg_s=x_step.unclamped_output,
            raw_velocity_y_deg_s=y_step.unclamped_output,
            limited_velocity_x_deg_s=x_step.output,
            limited_velocity_y_deg_s=y_step.output,
            clamp_x_active=x_step.clamped,
            clamp_y_active=y_step.clamped,
            anti_windup_x_active=x_step.anti_windup_active,
            anti_windup_y_active=y_step.anti_windup_active,
            max_velocity_x_deg_s=self._pid_x.output_limit,
            max_velocity_y_deg_s=self._pid_y.output_limit,
        )

    def apply_tracking_control_config(self, config: TrackingControlConfig) -> None:
        """Replace FF/prediction policy without disturbing PID state."""
        if not isinstance(config, TrackingControlConfig):
            raise TypeError("config must be TrackingControlConfig")
        self._tracking_control_config = config

    def continue_tracking_motion(self, revision: int) -> bool:
        """Repeat the last confirmed tracking velocity without recomputing PID."""
        self._require_revision(revision)
        if (
            self._last_processed_revision is not None
            and revision <= self._last_processed_revision
        ):
            return False
        self._last_processed_revision = revision
        if self._control_mode is not TurretControlMode.TRACKING:
            return False
        previous = self._last_applied_tracking_velocity
        if previous is None or self._hal.platform_motion_zero_settling:
            return False

        control_id = self._next_control_id
        self._next_control_id += 1
        self._pending_motion = AxisVelocitySetpoint(
            velocity_x_deg_s=previous.velocity_x_deg_s,
            velocity_y_deg_s=previous.velocity_y_deg_s,
            timestamp_ns=previous.timestamp_ns,
            control_id=control_id,
        )
        emit_event(
            _LOGGER,
            "CONTROL_DECISION",
            source_instance="turret:controller",
            control_id=control_id,
            decision="repeat",
            command_type="SET_VELOCITY",
            logical_velocity_x_deg_s=previous.velocity_x_deg_s,
            logical_velocity_y_deg_s=previous.velocity_y_deg_s,
            replaced_control_id=previous.control_id,
            reason="target_temporarily_missing",
            pending_revision=revision,
        )
        return True

    def invalidate_tracking_control(self, revision: int) -> bool:
        self._require_revision(revision)
        if (
            self._last_processed_revision is not None
            and revision <= self._last_processed_revision
        ):
            return False
        pending = self._pending_motion
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        if isinstance(pending, AxisVelocitySetpoint) and pending.control_id is not None:
            emit_event(
                _LOGGER,
                "CONTROL_DECISION",
                source_instance="turret:controller",
                control_id=pending.control_id,
                decision="cancel",
                command_type="SET_VELOCITY",
                logical_velocity_x_deg_s=pending.velocity_x_deg_s,
                logical_velocity_y_deg_s=pending.velocity_y_deg_s,
                replaced_control_id=None,
                reason="tracking_control_invalidated",
                pending_revision=revision,
            )
        self._reset_pid()
        self._last_processed_revision = revision
        return True

    def flush_pending_motion(self) -> SessionResult | None:
        motion = self._pending_motion
        self._pending_motion = None
        if motion is None:
            return None
        if isinstance(motion, MoveRelativeCommand):
            if self._control_mode is not TurretControlMode.RELATIVE:
                raise ControllerModeError("pending relative motion in non-RELATIVE mode")
            return self._hal.move_relative(motion)
        if self._control_mode is not TurretControlMode.TRACKING:
            raise ControllerModeError("pending velocity motion in non-TRACKING mode")
        result = self._hal.set_velocity(motion)
        if self._session_result_ok(result):
            self._last_applied_tracking_velocity = motion
        return result

    def set_control_mode(self, mode: TurretControlMode) -> SessionResult | None:
        if not isinstance(mode, TurretControlMode):
            raise TypeError("mode must be TurretControlMode")
        if mode is self._control_mode:
            return None
        if self._hal.motor_state is MotorState.UNKNOWN:
            raise ControllerStateError("mode transition requires confirmed motor state")

        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        old_mode = self._control_mode

        if old_mode is TurretControlMode.TRACKING:
            self._reset_pid()

        result: SessionResult | None = None
        if self._hal.motor_state is MotorState.ON:
            result = self._hal.set_velocity(
                self._zero_velocity(),
                recover_command_model_after_zero=(
                    old_mode is TurretControlMode.RELATIVE
                    and mode is TurretControlMode.TRACKING
                ),
            )
            if not self._session_result_ok(result):
                return result

        if old_mode is TurretControlMode.RELATIVE and mode is TurretControlMode.TRACKING:
            self._reset_pid()

        self._control_mode = mode
        if mode is not TurretControlMode.TRACKING:
            self._target = None
            self._last_tracking_time_s = None
        return result

    def stop_motion(self) -> SessionResult | None:
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        self._reset_pid()
        if self._hal.motor_state is MotorState.ON:
            return self._hal.set_velocity(self._zero_velocity())
        return None

    def set_calibration_velocity(self, setpoint: AxisVelocitySetpoint) -> SessionResult:
        """Apply one calibration-owned velocity without changing control mode."""
        if not isinstance(setpoint, AxisVelocitySetpoint):
            raise TypeError("setpoint must be AxisVelocitySetpoint")
        if self._hal.motor_state is not MotorState.ON:
            raise ControllerStateError("calibration velocity requires MOTOR ON")
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        self._reset_pid()
        self._target = None
        self._last_tracking_time_s = None
        return self._hal.set_velocity(setpoint)

    def motor_off(self) -> SessionResult:
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        self._reset_pid()
        return self._hal.motor_off()

    def motor_on(self) -> SessionResult:
        # Never retain/replay a motion intent across a MOTOR_ON boundary.
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        return self._hal.motor_on()

    def emergency_stop(self) -> ProtocolResponse | None:
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        self._reset_pid()
        return self._hal.emergency_stop()

    def reset_for_transport_recovery(self) -> None:
        """Controller-side reset boundary; Stage 3D owns actual reconnect."""
        self._pending_motion = None
        self._last_applied_tracking_velocity = None
        self._reset_pid()
        self._hal.mark_transport_lost()

    def apply_config_update(
        self,
        update: ConfigUpdate[TurretConfig],
        *,
        defer_stm32: bool = False,
    ) -> SessionResult | None:
        if not isinstance(update, ConfigUpdate):
            raise TypeError("update must be ConfigUpdate[TurretConfig]")
        if not isinstance(update.config, TurretConfig):
            raise TypeError("update.config must be TurretConfig")
        if update.revision <= self._config_revision:
            return None

        old = self._config
        new = update.config
        self._apply_axis_runtime_config(
            self._pid_x,
            old_gains=self._x_gains(old.controller),
            new_gains=self._x_gains(new.controller),
            old_limit=old.stm32.max_speed_x_deg_s,
            new_limit=new.stm32.max_speed_x_deg_s,
            old_integral_limit=self._x_integral_limit(old),
            new_integral_limit=self._x_integral_limit(new),
        )
        self._apply_axis_runtime_config(
            self._pid_y,
            old_gains=self._y_gains(old.controller),
            new_gains=self._y_gains(new.controller),
            old_limit=old.stm32.max_speed_y_deg_s,
            new_limit=new.stm32.max_speed_y_deg_s,
            old_integral_limit=self._y_integral_limit(old),
            new_integral_limit=self._y_integral_limit(new),
        )
        self._config = new
        self._config_revision = update.revision
        if defer_stm32:
            self._hal.stage_config_update(update)
            return None
        return self._hal.apply_config_update(update)

    def _apply_axis_runtime_config(
        self,
        pid: AxisPid,
        *,
        old_gains: PidGains,
        new_gains: PidGains,
        old_limit: float,
        new_limit: float,
        old_integral_limit: float,
        new_integral_limit: float,
    ) -> None:
        gains_changed = new_gains != old_gains
        limit_changed = new_limit != old_limit
        integral_limit_changed = new_integral_limit != old_integral_limit
        if gains_changed:
            pid.set_gains(new_gains)
        if limit_changed:
            pid.set_output_limit(new_limit)
        if integral_limit_changed:
            pid.set_integral_limit(new_integral_limit)
        if gains_changed and self._control_mode is TurretControlMode.TRACKING:
            # Gain reset has priority over any same-revision output-limit clamp.
            pid.reset()

    def _reset_pid(self) -> None:
        self._pid_x.reset()
        self._pid_y.reset()
        self._target = None
        self._last_tracking_time_s = None

    @staticmethod
    def _session_result_ok(result: SessionResult) -> bool:
        return result.response is not None and result.response.result is ResultCode.OK

    @staticmethod
    def _zero_velocity() -> AxisVelocitySetpoint:
        return AxisVelocitySetpoint(0.0, 0.0, 0)

    @staticmethod
    def _x_integral_limit(config: TurretConfig) -> float:
        value = config.controller.pid_integral_limit_x_deg_s
        return config.stm32.max_speed_x_deg_s if value is None else value

    @staticmethod
    def _y_integral_limit(config: TurretConfig) -> float:
        value = config.controller.pid_integral_limit_y_deg_s
        return config.stm32.max_speed_y_deg_s if value is None else value

    @staticmethod
    def _x_gains(config: PidControllerConfig) -> PidGains:
        return PidGains(config.pid_kp_x, config.pid_ki_x, config.pid_kd_x)

    @staticmethod
    def _y_gains(config: PidControllerConfig) -> PidGains:
        return PidGains(config.pid_kp_y, config.pid_ki_y, config.pid_kd_y)

    @staticmethod
    def _require_revision(revision: int) -> None:
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise TypeError("revision must be an integer")
        if revision < 0:
            raise ValueError("revision must be non-negative")

    @staticmethod
    def _require_finite_time(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("clock must return a number")
        result = float(value)
        if not isfinite(result):
            raise ValueError("clock must return a finite value")
        return result


__all__ = [
    "ControllerError",
    "ControllerModeError",
    "ControllerStateError",
    "PendingMotion",
    "TurretController",
]
