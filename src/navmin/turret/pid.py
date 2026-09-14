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

    def __init__(self, gains: PidGains, *, output_limit: float) -> None:
        if not isinstance(gains, PidGains):
            raise TypeError("gains must be PidGains")
        self._gains = gains
        self._output_limit = _require_positive_finite(
            output_limit, name="output_limit"
        )
        self._i_term = 0.0
        self._previous_error = 0.0
        self._previous_time_s = 0.0
        self._has_sample = False

    @property
    def gains(self) -> PidGains:
        return self._gains

    @property
    def output_limit(self) -> float:
        return self._output_limit

    @property
    def i_term(self) -> float:
        return self._i_term

    @property
    def has_sample(self) -> bool:
        return self._has_sample

    def reset(self) -> None:
        self._i_term = 0.0
        self._previous_error = 0.0
        self._previous_time_s = 0.0
        self._has_sample = False

    def set_gains(self, gains: PidGains) -> None:
        if not isinstance(gains, PidGains):
            raise TypeError("gains must be PidGains")
        self._gains = gains

    def set_output_limit(self, output_limit: float) -> None:
        self._output_limit = _require_positive_finite(
            output_limit, name="output_limit"
        )
        self._i_term = _clamp(self._i_term, self._output_limit)

    def step(self, error: float, *, now_s: float) -> float:
        error_value = _require_nonnegative_or_signed_finite(error, name="error")
        now = _require_nonnegative_or_signed_finite(now_s, name="now_s")
        p_term = self._gains.kp * error_value

        if not self._has_sample:
            self._previous_error = error_value
            self._previous_time_s = now
            self._has_sample = True
            return _clamp(p_term, self._output_limit)

        dt = now - self._previous_time_s
        if dt <= 0:
            d_term = 0.0
            candidate_i = self._i_term
        else:
            d_term = self._gains.kd * (error_value - self._previous_error) / dt
            candidate_i = _clamp(
                self._i_term + self._gains.ki * error_value * dt,
                self._output_limit,
            )

        candidate_output = p_term + candidate_i + d_term
        drives_positive_saturation = (
            candidate_output > self._output_limit and error_value > 0
        )
        drives_negative_saturation = (
            candidate_output < -self._output_limit and error_value < 0
        )
        if not drives_positive_saturation and not drives_negative_saturation:
            self._i_term = candidate_i

        output = _clamp(p_term + self._i_term + d_term, self._output_limit)
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


__all__ = ["AxisPid", "PidGains"]
