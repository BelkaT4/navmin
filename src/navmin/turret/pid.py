"""Controller-owned PID primitives for Turret tracking."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_nonnegative_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if result < 0 or not isfinite(result):
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _require_positive_finite(value: float, *, name: str) -> float:
    result = _require_nonnegative_finite(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class PidStepDiagnostics:
    real_error: float
    proportional_offset: float
    p_term: float
    i_term: float
    d_term: float
    feed_forward: float
    unclamped_output: float
    output: float
    clamped: bool
    integrator_updated: bool
    anti_windup_active: bool


@dataclass(frozen=True)
class PidGains:
    kp: float
    ki: float
    kd: float

    def __post_init__(self) -> None:
        _require_nonnegative_finite(self.kp, name="kp")
        _require_nonnegative_finite(self.ki, name="ki")
        _require_nonnegative_finite(self.kd, name="kd")


class AxisPid:
    """One-axis PID in application units (degrees / degrees per second)."""

    def __init__(
        self,
        gains: PidGains,
        *,
        output_limit: float,
        integral_limit: float | None = None,
    ) -> None:
        if not isinstance(gains, PidGains):
            raise TypeError("gains must be PidGains")
        self._gains = gains
        self._output_limit = _require_positive_finite(
            output_limit, name="output_limit"
        )
        self._integral_limit = (
            self._output_limit
            if integral_limit is None
            else _require_nonnegative_finite(integral_limit, name="integral_limit")
        )
        self._i_term = 0.0
        self._previous_error = 0.0
        self._previous_time_s = 0.0
        self._has_sample = False
        self._last_step: PidStepDiagnostics | None = None

    @property
    def gains(self) -> PidGains:
        return self._gains

    @property
    def output_limit(self) -> float:
        return self._output_limit

    @property
    def integral_limit(self) -> float:
        return self._integral_limit

    @property
    def i_term(self) -> float:
        return self._i_term

    @property
    def has_sample(self) -> bool:
        return self._has_sample

    @property
    def last_step(self) -> PidStepDiagnostics | None:
        return self._last_step

    def reset(self) -> None:
        self._i_term = 0.0
        self._previous_error = 0.0
        self._previous_time_s = 0.0
        self._has_sample = False
        self._last_step = None

    def set_gains(self, gains: PidGains) -> None:
        if not isinstance(gains, PidGains):
            raise TypeError("gains must be PidGains")
        self._gains = gains

    def set_output_limit(self, output_limit: float) -> None:
        self._output_limit = _require_positive_finite(
            output_limit, name="output_limit"
        )
        self._i_term = _clamp(
            self._i_term, min(self._integral_limit, self._output_limit)
        )

    def set_integral_limit(self, integral_limit: float) -> None:
        self._integral_limit = _require_nonnegative_finite(
            integral_limit, name="integral_limit"
        )
        self._i_term = _clamp(
            self._i_term, min(self._integral_limit, self._output_limit)
        )

    def step(
        self,
        real_error: float,
        *,
        now_s: float,
        proportional_offset: float = 0.0,
        feed_forward: float = 0.0,
    ) -> float:
        error_value = _require_nonnegative_or_signed_finite(
            real_error, name="real_error"
        )
        now = _require_nonnegative_or_signed_finite(now_s, name="now_s")
        offset = _require_nonnegative_or_signed_finite(
            proportional_offset, name="proportional_offset"
        )
        ff_term = _require_nonnegative_or_signed_finite(
            feed_forward, name="feed_forward"
        )
        p_term = self._gains.kp * (error_value + offset)

        if not self._has_sample:
            self._previous_error = error_value
            self._previous_time_s = now
            self._has_sample = True
            unclamped_output = p_term + ff_term
            output = _clamp(unclamped_output, self._output_limit)
            self._last_step = PidStepDiagnostics(
                real_error=error_value,
                proportional_offset=offset,
                p_term=p_term,
                i_term=0.0,
                d_term=0.0,
                feed_forward=ff_term,
                unclamped_output=unclamped_output,
                output=output,
                clamped=output != unclamped_output,
                integrator_updated=False,
                anti_windup_active=False,
            )
            return output

        dt = now - self._previous_time_s
        if dt <= 0:
            d_term = 0.0
            candidate_i = self._i_term
        else:
            d_term = self._gains.kd * (error_value - self._previous_error) / dt
            candidate_i = _clamp(
                self._i_term + self._gains.ki * error_value * dt,
                min(self._integral_limit, self._output_limit),
            )

        candidate_output = p_term + candidate_i + d_term + ff_term
        drives_positive_saturation = (
            candidate_output > self._output_limit and error_value > 0
        )
        drives_negative_saturation = (
            candidate_output < -self._output_limit and error_value < 0
        )
        anti_windup_active = drives_positive_saturation or drives_negative_saturation
        integrator_updated = not anti_windup_active
        if integrator_updated:
            self._i_term = candidate_i

        unclamped_output = p_term + self._i_term + d_term + ff_term
        output = _clamp(unclamped_output, self._output_limit)
        self._last_step = PidStepDiagnostics(
            real_error=error_value,
            proportional_offset=offset,
            p_term=p_term,
            i_term=self._i_term,
            d_term=d_term,
            feed_forward=ff_term,
            unclamped_output=unclamped_output,
            output=output,
            clamped=output != unclamped_output,
            integrator_updated=integrator_updated,
            anti_windup_active=anti_windup_active,
        )
        self._previous_error = error_value
        self._previous_time_s = now
        return output


def _require_nonnegative_or_signed_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = ["AxisPid", "PidGains", "PidStepDiagnostics"]
