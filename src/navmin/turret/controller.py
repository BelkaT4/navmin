"""Turret Controller: applied mode, PID ownership, and latest motion intent."""

from __future__ import annotations

from math import isfinite
from time import monotonic
from typing import Protocol

from navmin.config.models import PidControllerConfig, TurretConfig
from navmin.contracts import (
    AxisVelocitySetpoint,
    ConfigUpdate,
    MotorState,
    MoveRelativeCommand,
    TargetRef,
    TrackingError,
    TurretControlMode,
)

from .hal import TurretHal
from .pid import AxisPid, PidGains
from .protocol import ProtocolResponse, ResultCode
from .session import SessionResult


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
        *,
        clock: Clock = monotonic,
    ) -> None:
        if not isinstance(hal, TurretHal):
            raise TypeError("hal must be TurretHal")
        if not isinstance(config, TurretConfig):
            raise TypeError("config must be TurretConfig")
        self._hal = hal
        self._config = config
        self._config_revision = 0
        self._clock = clock
        self._control_mode = TurretControlMode.RELATIVE
        self._pending_motion: PendingMotion = None
        self._last_processed_revision: int | None = None
        self._target: TargetRef | None = None
        self._last_tracking_time_s: float | None = None
        self._pid_x = AxisPid(
            self._x_gains(config.controller),
            output_limit=config.stm32.max_speed_x_deg_s,
        )
        self._pid_y = AxisPid(
            self._y_gains(config.controller),
            output_limit=config.stm32.max_speed_y_deg_s,
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

    def submit_tracking_error(self, revision: int, error: TrackingError) -> bool:
        if self._control_mode is not TurretControlMode.TRACKING:
            raise ControllerModeError("TrackingError is only valid in TRACKING mode")
        self._require_revision(revision)
        if not isinstance(error, TrackingError):
            raise TypeError("error must be TrackingError")
        if (
            self._last_processed_revision is not None
            and revision <= self._last_processed_revision
        ):
            return False

        now = self._require_finite_time(self._clock())
        target_changed = self._target is not None and error.target != self._target
        gap_exceeded = (
            self._last_tracking_time_s is not None
            and now - self._last_tracking_time_s
            > self._config.stm32.velocity_watchdog_timeout_ms / 1000.0
        )
        if target_changed or gap_exceeded:
            self._reset_pid()

        velocity_x = self._pid_x.step(error.error_x_deg, now_s=now)
        velocity_y = self._pid_y.step(error.error_y_deg, now_s=now)
        self._pending_motion = AxisVelocitySetpoint(
            velocity_x_deg_s=velocity_x,
            velocity_y_deg_s=velocity_y,
            timestamp_ns=error.timestamp_ns,
        )
        self._target = error.target
        self._last_tracking_time_s = now
        self._last_processed_revision = revision
        return True

    def invalidate_tracking_error(self, revision: int) -> bool:
        self._require_revision(revision)
        if (
            self._last_processed_revision is not None
            and revision <= self._last_processed_revision
        ):
            return False
        self._pending_motion = None
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
        return self._hal.set_velocity(motion)

    def set_control_mode(self, mode: TurretControlMode) -> SessionResult | None:
        if not isinstance(mode, TurretControlMode):
            raise TypeError("mode must be TurretControlMode")
        if mode is self._control_mode:
            return None
        if self._hal.motor_state is MotorState.UNKNOWN:
            raise ControllerStateError("mode transition requires confirmed motor state")

        self._pending_motion = None
        old_mode = self._control_mode

        if old_mode is TurretControlMode.TRACKING:
            self._reset_pid()

        result: SessionResult | None = None
        if self._hal.motor_state is MotorState.ON:
            result = self._hal.set_velocity(self._zero_velocity())
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
        self._reset_pid()
        if self._hal.motor_state is MotorState.ON:
            return self._hal.set_velocity(self._zero_velocity())
        return None

    def motor_off(self) -> SessionResult:
        self._pending_motion = None
        self._reset_pid()
        return self._hal.motor_off()

    def motor_on(self) -> SessionResult:
        # Never retain/replay a motion intent across a MOTOR_ON boundary.
        self._pending_motion = None
        return self._hal.motor_on()

    def emergency_stop(self) -> ProtocolResponse | None:
        self._pending_motion = None
        self._reset_pid()
        return self._hal.emergency_stop()

    def reset_for_transport_recovery(self) -> None:
        """Controller-side reset boundary; Stage 3D owns actual reconnect."""
        self._pending_motion = None
        self._reset_pid()
        self._hal.mark_transport_lost()

    def apply_config_update(
        self, update: ConfigUpdate[TurretConfig]
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
        )
        self._apply_axis_runtime_config(
            self._pid_y,
            old_gains=self._y_gains(old.controller),
            new_gains=self._y_gains(new.controller),
            old_limit=old.stm32.max_speed_y_deg_s,
            new_limit=new.stm32.max_speed_y_deg_s,
        )
        self._config = new
        self._config_revision = update.revision
        return self._hal.apply_config_update(update)

    def _apply_axis_runtime_config(
        self,
        pid: AxisPid,
        *,
        old_gains: PidGains,
        new_gains: PidGains,
        old_limit: float,
        new_limit: float,
    ) -> None:
        gains_changed = new_gains != old_gains
        limit_changed = new_limit != old_limit
        if gains_changed:
            pid.set_gains(new_gains)
        if limit_changed:
            pid.set_output_limit(new_limit)
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
