"""Timestamped angular-rate estimation for predictive tracking control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from navmin.config.models import TrackingControlConfig, VelocityEstimatorConfig
from navmin.contracts import (
    PlatformMotionRate,
    PlatformMotionSource,
    TargetRef,
    TrackingError,
)


class TrackingEstimateInvalidReason(Enum):
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    INSUFFICIENT_SPAN = "insufficient_span"
    FIT_RMS = "fit_rms"
    PLATFORM_UNAVAILABLE = "platform_unavailable"
    TARGET_RATE_LIMIT = "target_rate_limit"
    TIMESTAMP_INVALID = "timestamp_invalid"
    NO_ACTIVE_TARGET = "no_active_target"


@dataclass(frozen=True)
class AxisTrackingEstimate:
    relative_rate_deg_s: float | None
    platform_rate_deg_s: float | None
    target_rate_deg_s: float | None
    fit_rms_deg: float | None
    sample_count: int
    actual_span_ms: float
    valid: bool
    invalid_reason: TrackingEstimateInvalidReason | None
    prediction_offset_deg: float


@dataclass(frozen=True)
class TrackingEstimatorDiagnostics:
    target: TargetRef | None
    observation_timestamp_ns: int | None
    platform_source: PlatformMotionSource
    x: AxisTrackingEstimate
    y: AxisTrackingEstimate


@dataclass(frozen=True)
class _Observation:
    timestamp_ns: int
    error_x_deg: float
    error_y_deg: float


@dataclass(frozen=True)
class _Fit:
    slope_deg_s: float
    rms_deg: float


class TrackingEstimator:
    """Estimate target angular rate from current-error observations.

    The estimator owns one bounded observation history for both axes and fits
    each axis independently. It never emits an actuator command; unusable data
    becomes typed invalid diagnostics with a zero prediction contribution.
    """

    def __init__(self, config: TrackingControlConfig) -> None:
        if not isinstance(config, TrackingControlConfig):
            raise TypeError("config must be TrackingControlConfig")
        self._config = config
        self._target: TargetRef | None = None
        self._history: list[_Observation] = []
        self._diagnostics = self._empty_diagnostics(
            TrackingEstimateInvalidReason.NO_ACTIVE_TARGET
        )

    @property
    def config(self) -> TrackingControlConfig:
        return self._config

    @property
    def diagnostics(self) -> TrackingEstimatorDiagnostics:
        return self._diagnostics

    @property
    def sample_count(self) -> int:
        return len(self._history)

    def replace_config(self, config: TrackingControlConfig) -> None:
        """Apply one immutable snapshot with estimator-specific reset semantics."""
        if not isinstance(config, TrackingControlConfig):
            raise TypeError("config must be TrackingControlConfig")
        estimator_changed = config.velocity_estimator != self._config.velocity_estimator
        source_changed = config.platform_motion != self._config.platform_motion
        self._config = config
        if estimator_changed or source_changed:
            self.reset_history()

    def reset(self) -> None:
        """Clear target identity and all observation continuity."""
        self._target = None
        self._history.clear()
        self._diagnostics = self._empty_diagnostics(
            TrackingEstimateInvalidReason.NO_ACTIVE_TARGET
        )

    def reset_history(self) -> None:
        """Clear observations while retaining the current target identity."""
        self._history.clear()
        reason = (
            TrackingEstimateInvalidReason.INSUFFICIENT_SAMPLES
            if self._target is not None
            else TrackingEstimateInvalidReason.NO_ACTIVE_TARGET
        )
        self._diagnostics = self._empty_diagnostics(reason)

    def observe(
        self,
        error: TrackingError,
        platform_motion: PlatformMotionRate | None,
    ) -> TrackingEstimatorDiagnostics:
        if not isinstance(error, TrackingError):
            raise TypeError("error must be TrackingError")
        if platform_motion is not None and not isinstance(
            platform_motion, PlatformMotionRate
        ):
            raise TypeError("platform_motion must be PlatformMotionRate or None")
        if not math.isfinite(error.error_x_deg) or not math.isfinite(error.error_y_deg):
            raise ValueError("tracking error must be finite")

        if error.target != self._target:
            self._target = error.target
            self._history.clear()

        previous_timestamp = self._history[-1].timestamp_ns if self._history else None
        if previous_timestamp is not None and error.timestamp_ns <= previous_timestamp:
            self._diagnostics = self._build_diagnostics(
                observation_timestamp_ns=error.timestamp_ns,
                platform_motion=platform_motion,
                forced_reason=TrackingEstimateInvalidReason.TIMESTAMP_INVALID,
            )
            return self._diagnostics

        self._history.append(
            _Observation(
                timestamp_ns=error.timestamp_ns,
                error_x_deg=error.error_x_deg,
                error_y_deg=error.error_y_deg,
            )
        )
        self._evict(error.timestamp_ns, self._config.velocity_estimator)
        self._diagnostics = self._build_diagnostics(
            observation_timestamp_ns=error.timestamp_ns,
            platform_motion=platform_motion,
        )
        return self._diagnostics

    def _evict(self, timestamp_ns: int, config: VelocityEstimatorConfig) -> None:
        cutoff_ns = timestamp_ns - config.window_ms * 1_000_000
        while self._history and self._history[0].timestamp_ns < cutoff_ns:
            self._history.pop(0)
        overflow = len(self._history) - config.max_samples
        if overflow > 0:
            del self._history[:overflow]

    def _build_diagnostics(
        self,
        *,
        observation_timestamp_ns: int,
        platform_motion: PlatformMotionRate | None,
        forced_reason: TrackingEstimateInvalidReason | None = None,
    ) -> TrackingEstimatorDiagnostics:
        sample_count = len(self._history)
        actual_span_ms = self._actual_span_ms()
        x_fit = self._fit("x")
        y_fit = self._fit("y")
        platform_x, platform_y = self._platform_rates(
            platform_motion, observation_timestamp_ns
        )
        estimator_config = self._config.velocity_estimator

        return TrackingEstimatorDiagnostics(
            target=self._target,
            observation_timestamp_ns=observation_timestamp_ns,
            platform_source=self._config.platform_motion.source,
            x=self._axis_diagnostics(
                fit=x_fit,
                platform_rate=platform_x,
                sample_count=sample_count,
                actual_span_ms=actual_span_ms,
                max_target_rate=estimator_config.max_target_rate_x_deg_s,
                max_prediction_angle=self._config.prediction.max_angle_x_deg,
                forced_reason=forced_reason,
            ),
            y=self._axis_diagnostics(
                fit=y_fit,
                platform_rate=platform_y,
                sample_count=sample_count,
                actual_span_ms=actual_span_ms,
                max_target_rate=estimator_config.max_target_rate_y_deg_s,
                max_prediction_angle=self._config.prediction.max_angle_y_deg,
                forced_reason=forced_reason,
            ),
        )

    def _axis_diagnostics(
        self,
        *,
        fit: _Fit | None,
        platform_rate: float | None,
        sample_count: int,
        actual_span_ms: float,
        max_target_rate: float,
        max_prediction_angle: float,
        forced_reason: TrackingEstimateInvalidReason | None,
    ) -> AxisTrackingEstimate:
        estimator_config = self._config.velocity_estimator
        relative_rate = None if fit is None else fit.slope_deg_s
        fit_rms = None if fit is None else fit.rms_deg
        target_rate = (
            None
            if relative_rate is None or platform_rate is None
            else relative_rate + platform_rate
        )

        reason = forced_reason
        if reason is None and sample_count < estimator_config.min_samples:
            reason = TrackingEstimateInvalidReason.INSUFFICIENT_SAMPLES
        if reason is None and actual_span_ms < estimator_config.min_span_ms:
            reason = TrackingEstimateInvalidReason.INSUFFICIENT_SPAN
        if (
            reason is None
            and fit_rms is not None
            and fit_rms > estimator_config.max_fit_rms_deg
        ):
            reason = TrackingEstimateInvalidReason.FIT_RMS
        if reason is None and platform_rate is None:
            reason = TrackingEstimateInvalidReason.PLATFORM_UNAVAILABLE
        if (
            reason is None
            and target_rate is not None
            and abs(target_rate) > max_target_rate
        ):
            reason = TrackingEstimateInvalidReason.TARGET_RATE_LIMIT

        valid = reason is None
        prediction_offset = 0.0
        if valid and self._config.prediction.enabled and target_rate is not None:
            raw_offset = target_rate * self._config.prediction.horizon_ms / 1000.0
            prediction_offset = max(
                -max_prediction_angle, min(max_prediction_angle, raw_offset)
            )

        return AxisTrackingEstimate(
            relative_rate_deg_s=relative_rate,
            platform_rate_deg_s=platform_rate,
            target_rate_deg_s=target_rate,
            fit_rms_deg=fit_rms,
            sample_count=sample_count,
            actual_span_ms=actual_span_ms,
            valid=valid,
            invalid_reason=reason,
            prediction_offset_deg=prediction_offset,
        )

    def _fit(self, axis: str) -> _Fit | None:
        if len(self._history) < 2:
            return None
        first_timestamp = self._history[0].timestamp_ns
        times_s = [
            (sample.timestamp_ns - first_timestamp) / 1_000_000_000.0
            for sample in self._history
        ]
        values = [
            sample.error_x_deg if axis == "x" else sample.error_y_deg
            for sample in self._history
        ]
        mean_time = sum(times_s) / len(times_s)
        mean_value = sum(values) / len(values)
        denominator = sum((time_s - mean_time) ** 2 for time_s in times_s)
        if denominator <= 0.0:
            return None
        slope = (
            sum(
                (time_s - mean_time) * (value - mean_value)
                for time_s, value in zip(times_s, values, strict=True)
            )
            / denominator
        )
        intercept = mean_value - slope * mean_time
        rms = math.sqrt(
            sum(
                (value - (intercept + slope * time_s)) ** 2
                for time_s, value in zip(times_s, values, strict=True)
            )
            / len(values)
        )
        return _Fit(slope_deg_s=slope, rms_deg=rms)

    def _actual_span_ms(self) -> float:
        if len(self._history) < 2:
            return 0.0
        return (
            self._history[-1].timestamp_ns - self._history[0].timestamp_ns
        ) / 1_000_000.0

    @staticmethod
    def _platform_rates(
        platform_motion: PlatformMotionRate | None,
        observation_timestamp_ns: int,
    ) -> tuple[float | None, float | None]:
        if (
            platform_motion is None
            or platform_motion.timestamp_ns != observation_timestamp_ns
        ):
            return None, None

        def finite_or_none(value: float | None) -> float | None:
            if value is None or not math.isfinite(value):
                return None
            return value

        return (
            finite_or_none(platform_motion.rate_x_deg_s),
            finite_or_none(platform_motion.rate_y_deg_s),
        )

    def _empty_diagnostics(
        self, reason: TrackingEstimateInvalidReason
    ) -> TrackingEstimatorDiagnostics:
        axis = AxisTrackingEstimate(
            relative_rate_deg_s=None,
            platform_rate_deg_s=None,
            target_rate_deg_s=None,
            fit_rms_deg=None,
            sample_count=0,
            actual_span_ms=0.0,
            valid=False,
            invalid_reason=reason,
            prediction_offset_deg=0.0,
        )
        return TrackingEstimatorDiagnostics(
            target=self._target,
            observation_timestamp_ns=None,
            platform_source=self._config.platform_motion.source,
            x=axis,
            y=axis,
        )


__all__ = [
    "AxisTrackingEstimate",
    "TrackingEstimateInvalidReason",
    "TrackingEstimator",
    "TrackingEstimatorDiagnostics",
]
