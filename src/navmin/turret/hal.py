"""PC-side Turret HAL: units, motor confirmation, and STM32 config sync."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from navmin.config.models import AxisMechanicsConfig, Stm32Config, TurretConfig
from navmin.contracts import (
    AxisVelocitySetpoint,
    ConfigUpdate,
    MotorState,
    MoveRelativeCommand,
)

from .protocol import (
    CommandCode,
    MoveRelativePayload,
    ProtocolResponse,
    ResultCode,
    SetConfigPayload,
    SetVelocityPayload,
    require_int32,
    require_uint32,
)
from .session import SessionResult, TurretSession

type AxisName = Literal["x", "y"]


class HalError(RuntimeError):
    """Base class for HAL-side validation failures."""


class RelativeMoveLimitError(HalError):
    """A requested relative move exceeds the configured PC-side limit."""


@dataclass(frozen=True)
class _PendingConfig:
    revision: int
    config: Stm32Config
    payload: SetConfigPayload


class TurretHal:
    """Application-unit HAL over the accepted Stage 3B ``TurretSession``."""

    def __init__(self, session: TurretSession, config: TurretConfig) -> None:
        if not isinstance(session, TurretSession):
            raise TypeError("session must be TurretSession")
        if not isinstance(config, TurretConfig):
            raise TypeError("config must be TurretConfig")
        self._session = session
        # Mechanical conversion is restart-only in v1: keep this snapshot forever.
        self._mechanics_x = config.axes.x
        self._mechanics_y = config.axes.y
        self._desired_stm32_config = config.stm32
        self._latest_config_revision = 0
        self._pending_config: _PendingConfig | None = None
        self._applied_stm32_config: Stm32Config | None = None
        self._applied_config_revision: int | None = None
        self._motor_state = MotorState.UNKNOWN
        self._operation_active = False
        self._config_drain_defer_depth = 0

    @property
    def motor_state(self) -> MotorState:
        return self._motor_state

    @property
    def applied_stm32_config(self) -> Stm32Config | None:
        return self._applied_stm32_config

    @property
    def applied_config_revision(self) -> int | None:
        return self._applied_config_revision

    @property
    def pending_config_revision(self) -> int | None:
        return None if self._pending_config is None else self._pending_config.revision

    def rebind_session(self, session: TurretSession) -> None:
        """Bind a recovered session without changing restart-only mechanics."""
        if not isinstance(session, TurretSession):
            raise TypeError("session must be TurretSession")
        if self._operation_active:
            raise HalError("cannot rebind session during an active HAL operation")
        self._session = session

    @contextmanager
    def defer_pending_config_drain(self):
        """Suppress implicit SET_CONFIG drain across an explicit safety boundary."""
        self._config_drain_defer_depth += 1
        try:
            yield
        finally:
            self._config_drain_defer_depth -= 1

    def effective_steps_per_revolution(self, axis: AxisName) -> int:
        mechanics = self._mechanics(axis)
        return mechanics.full_steps_per_revolution * mechanics.microstep_divider

    def degrees_to_steps(self, axis: AxisName, value_deg: float) -> int:
        value = self._require_finite(value_deg, name="value_deg")
        steps = self._round_steps(axis, value)
        if self._mechanics(axis).invert:
            steps = -steps
        return require_int32(steps, name=f"{axis}_steps")

    def velocity_to_steps_s(self, axis: AxisName, value_deg_s: float) -> int:
        value = self._require_finite(value_deg_s, name="value_deg_s")
        steps = self._round_steps(axis, value)
        if self._mechanics(axis).invert:
            steps = -steps
        return require_int32(steps, name=f"{axis}_steps_s")

    def speed_magnitude_to_steps_s(self, axis: AxisName, value_deg_s: float) -> int:
        value = self._require_nonnegative_finite(value_deg_s, name="value_deg_s")
        # Axis inversion must never affect unsigned magnitudes.
        return require_uint32(
            self._round_steps(axis, value), name=f"{axis}_max_speed_steps_s"
        )

    def acceleration_magnitude_to_steps_s2(
        self, axis: AxisName, value_deg_s2: float
    ) -> int:
        value = self._require_nonnegative_finite(value_deg_s2, name="value_deg_s2")
        # Axis inversion must never affect unsigned magnitudes.
        return require_uint32(
            self._round_steps(axis, value), name=f"{axis}_acceleration_steps_s2"
        )

    def make_set_config_payload(self, config: Stm32Config) -> SetConfigPayload:
        if not isinstance(config, Stm32Config):
            raise TypeError("config must be Stm32Config")
        return SetConfigPayload(
            max_speed_x_steps_s=self.speed_magnitude_to_steps_s(
                "x", config.max_speed_x_deg_s
            ),
            max_speed_y_steps_s=self.speed_magnitude_to_steps_s(
                "y", config.max_speed_y_deg_s
            ),
            acceleration_x_steps_s2=self.acceleration_magnitude_to_steps_s2(
                "x", config.acceleration_x_deg_s2
            ),
            acceleration_y_steps_s2=self.acceleration_magnitude_to_steps_s2(
                "y", config.acceleration_y_deg_s2
            ),
            velocity_watchdog_timeout_ms=require_uint32(
                config.velocity_watchdog_timeout_ms,
                name="velocity_watchdog_timeout_ms",
            ),
        )

    def move_relative(self, command: MoveRelativeCommand) -> SessionResult:
        if not isinstance(command, MoveRelativeCommand):
            raise TypeError("command must be MoveRelativeCommand")
        self._validate_relative_limit("x", command.delta_x_deg)
        self._validate_relative_limit("y", command.delta_y_deg)
        payload = MoveRelativePayload(
            delta_x_steps=self.degrees_to_steps("x", command.delta_x_deg),
            delta_y_steps=self.degrees_to_steps("y", command.delta_y_deg),
        )
        return self._ordinary(CommandCode.MOVE_RELATIVE, payload)

    def set_velocity(self, setpoint: AxisVelocitySetpoint) -> SessionResult:
        if not isinstance(setpoint, AxisVelocitySetpoint):
            raise TypeError("setpoint must be AxisVelocitySetpoint")
        payload = SetVelocityPayload(
            velocity_x_steps_s=self.velocity_to_steps_s(
                "x", setpoint.velocity_x_deg_s
            ),
            velocity_y_steps_s=self.velocity_to_steps_s(
                "y", setpoint.velocity_y_deg_s
            ),
        )
        return self._ordinary(CommandCode.SET_VELOCITY, payload)

    def motor_on(self) -> SessionResult:
        result = self._ordinary(CommandCode.MOTOR_ON)
        if self._response_ok(result):
            self._motor_state = MotorState.ON
        return result

    def motor_off(self) -> SessionResult:
        result = self._ordinary(CommandCode.MOTOR_OFF)
        if self._response_ok(result):
            self._motor_state = MotorState.OFF
        return result

    def emergency_stop(self) -> ProtocolResponse | None:
        response = self._session.request_emergency()
        self._drain_pending_config_if_idle()
        return response

    def stage_config_update(self, update: ConfigUpdate[TurretConfig]) -> bool:
        """Accept latest config without performing transport I/O.

        Stage 3D uses this during recovery so the final SET_CONFIG is emitted
        exactly once, after Emergency/MOTOR_OFF/baud recovery.
        """
        if not isinstance(update, ConfigUpdate):
            raise TypeError("update must be ConfigUpdate[TurretConfig]")
        if not isinstance(update.config, TurretConfig):
            raise TypeError("update.config must be TurretConfig")
        if update.revision <= self._latest_config_revision:
            return False
        self._latest_config_revision = update.revision

        if update.config.stm32 != self._desired_stm32_config:
            self._desired_stm32_config = update.config.stm32
            self._pending_config = _PendingConfig(
                revision=update.revision,
                config=update.config.stm32,
                payload=self.make_set_config_payload(update.config.stm32),
            )
        elif self._pending_config is not None:
            # A newer global config revision may carry the same STM32 snapshot.
            # Keep latest-only revision ownership without creating a queue.
            self._pending_config = _PendingConfig(
                revision=update.revision,
                config=self._pending_config.config,
                payload=self._pending_config.payload,
            )
        return True

    def apply_config_update(
        self, update: ConfigUpdate[TurretConfig]
    ) -> SessionResult | None:
        if not self.stage_config_update(update):
            return None
        if self._operation_active:
            return None
        return self.flush_pending_config()

    def sync_stm32_config(self) -> SessionResult | None:
        """Queue and transmit the full current desired STM32 snapshot."""
        self._pending_config = _PendingConfig(
            revision=self._latest_config_revision,
            config=self._desired_stm32_config,
            payload=self.make_set_config_payload(self._desired_stm32_config),
        )
        if self._operation_active:
            return None
        return self.flush_pending_config()

    def flush_pending_config(self) -> SessionResult | None:
        if self._operation_active:
            return None
        last_result: SessionResult | None = None
        while self._pending_config is not None:
            pending = self._pending_config
            self._operation_active = True
            try:
                result = self._session.transact(CommandCode.SET_CONFIG, pending.payload)
            finally:
                self._operation_active = False
            last_result = result
            if not self._response_ok(result):
                return result

            self._applied_stm32_config = pending.config
            self._applied_config_revision = pending.revision
            if (
                self._pending_config is not None
                and self._pending_config.revision == pending.revision
            ):
                self._pending_config = None
        return last_result

    def mark_transport_lost(self) -> None:
        """Invalidate only confirmed physical state; reconnect is owned by Stage 3D."""
        self._motor_state = MotorState.UNKNOWN
        self._applied_stm32_config = None
        self._applied_config_revision = None

    def _ordinary(self, command: CommandCode, payload=None) -> SessionResult:
        self._operation_active = True
        try:
            result = self._session.transact(command, payload)
        finally:
            self._operation_active = False
        self._drain_pending_config_if_idle()
        return result

    def _drain_pending_config_if_idle(self) -> None:
        if self._config_drain_defer_depth > 0:
            return
        if self._pending_config is not None and not self._operation_active:
            self.flush_pending_config()

    def _round_steps(self, axis: AxisName, value: float) -> int:
        return round(value * self.effective_steps_per_revolution(axis) / 360.0)

    def _mechanics(self, axis: AxisName) -> AxisMechanicsConfig:
        if axis == "x":
            return self._mechanics_x
        if axis == "y":
            return self._mechanics_y
        raise ValueError("axis must be 'x' or 'y'")

    def _validate_relative_limit(self, axis: AxisName, value_deg: float) -> None:
        value = self._require_finite(value_deg, name=f"delta_{axis}_deg")
        limit = self._mechanics(axis).max_relative_move_deg
        if abs(value) > limit:
            raise RelativeMoveLimitError(
                f"delta_{axis}_deg {value} exceeds ±{limit} degree limit"
            )

    @staticmethod
    def _response_ok(result: SessionResult) -> bool:
        return result.response is not None and result.response.result is ResultCode.OK

    @staticmethod
    def _require_finite(value: float, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _require_nonnegative_finite(cls, value: float, *, name: str) -> float:
        result = cls._require_finite(value, name=name)
        if result < 0:
            raise ValueError(f"{name} must be non-negative")
        return result


__all__ = ["HalError", "RelativeMoveLimitError", "TurretHal"]
