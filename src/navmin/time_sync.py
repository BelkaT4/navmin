"""Robust affine mapping from a device clock into PC monotonic time."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class ClockMappingState(Enum):
    UNAVAILABLE = "unavailable"
    WARMING = "warming"
    VALID = "valid"
    INVALID = "invalid"
    STALE = "stale"


class ClockMappingInvalidReason(Enum):
    NO_SAMPLES = "no_samples"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    INSUFFICIENT_SPAN = "insufficient_span"
    FIT_RMS = "fit_rms"
    DRIFT_LIMIT = "drift_limit"
    STALE = "stale"


class ClockSampleOutcome(Enum):
    ACCEPTED = "accepted"
    REJECTED_OUTLIER = "rejected_outlier"
    REJECTED_OUT_OF_ORDER = "rejected_out_of_order"
    RESET_ACCEPTED = "reset_accepted"


@dataclass(frozen=True, slots=True)
class ClockMapperConfig:
    capacity: int = 32
    min_samples: int = 4
    min_span_ns: int = 1_000_000_000
    max_fit_rms_ns: int = 5_000_000
    max_abs_drift_ppm: float = 2000.0
    max_mapping_age_ns: int = 5_000_000_000
    reset_backstep_ns: int = 1_000_000_000
    outlier_floor_ns: int = 10_000_000
    outlier_sigma: float = 6.0

    def __post_init__(self) -> None:
        _positive_int("capacity", self.capacity)
        _positive_int("min_samples", self.min_samples)
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        if self.capacity < self.min_samples:
            raise ValueError("capacity must be >= min_samples")
        _positive_int("min_span_ns", self.min_span_ns)
        _positive_int("max_fit_rms_ns", self.max_fit_rms_ns)
        if not isinstance(self.max_abs_drift_ppm, int | float) or isinstance(
            self.max_abs_drift_ppm, bool
        ):
            raise TypeError("max_abs_drift_ppm must be a number")
        if not math.isfinite(self.max_abs_drift_ppm) or self.max_abs_drift_ppm <= 0:
            raise ValueError("max_abs_drift_ppm must be finite and > 0")
        _positive_int("max_mapping_age_ns", self.max_mapping_age_ns)
        _positive_int("reset_backstep_ns", self.reset_backstep_ns)
        _positive_int("outlier_floor_ns", self.outlier_floor_ns)
        if not isinstance(self.outlier_sigma, int | float) or isinstance(
            self.outlier_sigma, bool
        ):
            raise TypeError("outlier_sigma must be a number")
        if not math.isfinite(self.outlier_sigma) or self.outlier_sigma <= 0:
            raise ValueError("outlier_sigma must be finite and > 0")


@dataclass(frozen=True, slots=True)
class ClockSyncSample:
    device_timestamp_ns: int
    pc_timestamp_ns: int
    observed_at_pc_ns: int
    uncertainty_ns: int = 0

    def __post_init__(self) -> None:
        _nonnegative_int("device_timestamp_ns", self.device_timestamp_ns)
        _nonnegative_int("pc_timestamp_ns", self.pc_timestamp_ns)
        _nonnegative_int("observed_at_pc_ns", self.observed_at_pc_ns)
        _nonnegative_int("uncertainty_ns", self.uncertainty_ns)
        if self.observed_at_pc_ns < self.pc_timestamp_ns:
            raise ValueError("observed_at_pc_ns must be >= pc_timestamp_ns")


@dataclass(frozen=True, slots=True)
class MappedTimestamp:
    device_timestamp_ns: int
    pc_timestamp_ns: int
    estimated_error_ns: int


@dataclass(frozen=True, slots=True)
class ClockMappingDiagnostics:
    state: ClockMappingState
    invalid_reason: ClockMappingInvalidReason | None
    sample_count: int
    submitted_sample_count: int
    rejected_outlier_count: int
    rejected_out_of_order_count: int
    reset_count: int
    device_span_ns: int
    last_device_timestamp_ns: int | None
    last_observed_pc_ns: int | None
    age_ns: int | None
    slope: float | None
    offset_ns: float | None
    drift_ppm: float | None
    fit_rms_ns: float | None
    median_uncertainty_ns: float | None


@dataclass(frozen=True, slots=True)
class _ClockFit:
    slope: float
    offset_ns: float
    rms_ns: float
    median_uncertainty_ns: float


class ClockMapper:
    """Map one monotonic device clock into PC monotonic time.

    The mapper deliberately owns no wall clock and no scheduler. Callers submit
    timestamp pairs and provide ``now_pc_ns`` when asking whether the mapping is
    fresh enough to use.
    """

    def __init__(self, config: ClockMapperConfig | None = None) -> None:
        self._config = ClockMapperConfig() if config is None else config
        if not isinstance(self._config, ClockMapperConfig):
            raise TypeError("config must be ClockMapperConfig or None")
        self._samples: deque[ClockSyncSample] = deque(maxlen=self._config.capacity)
        self._fit: _ClockFit | None = None
        self._submitted_sample_count = 0
        self._rejected_outlier_count = 0
        self._rejected_out_of_order_count = 0
        self._reset_count = 0

    @property
    def config(self) -> ClockMapperConfig:
        return self._config

    def reset(self) -> None:
        """Clear current clock-session continuity while retaining lifetime counters."""
        self._samples.clear()
        self._fit = None
        self._reset_count += 1

    def observe(self, sample: ClockSyncSample) -> ClockSampleOutcome:
        if not isinstance(sample, ClockSyncSample):
            raise TypeError("sample must be ClockSyncSample")
        self._submitted_sample_count += 1

        if self._samples:
            previous = self._samples[-1].device_timestamp_ns
            if sample.device_timestamp_ns <= previous:
                backstep_ns = previous - sample.device_timestamp_ns
                if backstep_ns >= self._config.reset_backstep_ns:
                    self.reset()
                    self._accept(sample)
                    return ClockSampleOutcome.RESET_ACCEPTED
                self._rejected_out_of_order_count += 1
                return ClockSampleOutcome.REJECTED_OUT_OF_ORDER

        candidate = (*self._samples, sample)
        if self._newest_is_rejectable_outlier(candidate):
            self._rejected_outlier_count += 1
            return ClockSampleOutcome.REJECTED_OUTLIER

        self._accept(sample)
        return ClockSampleOutcome.ACCEPTED

    def diagnostics(self, *, now_pc_ns: int) -> ClockMappingDiagnostics:
        _nonnegative_int("now_pc_ns", now_pc_ns)
        sample_count = len(self._samples)
        last_sample = self._samples[-1] if self._samples else None
        device_span_ns = (
            self._samples[-1].device_timestamp_ns
            - self._samples[0].device_timestamp_ns
            if sample_count >= 2
            else 0
        )
        age_ns = (
            max(0, now_pc_ns - last_sample.observed_at_pc_ns)
            if last_sample is not None
            else None
        )
        state, invalid_reason = self._state(
            sample_count=sample_count,
            device_span_ns=device_span_ns,
            age_ns=age_ns,
        )
        fit = self._fit
        return ClockMappingDiagnostics(
            state=state,
            invalid_reason=invalid_reason,
            sample_count=sample_count,
            submitted_sample_count=self._submitted_sample_count,
            rejected_outlier_count=self._rejected_outlier_count,
            rejected_out_of_order_count=self._rejected_out_of_order_count,
            reset_count=self._reset_count,
            device_span_ns=device_span_ns,
            last_device_timestamp_ns=(
                None if last_sample is None else last_sample.device_timestamp_ns
            ),
            last_observed_pc_ns=(
                None if last_sample is None else last_sample.observed_at_pc_ns
            ),
            age_ns=age_ns,
            slope=None if fit is None else fit.slope,
            offset_ns=None if fit is None else fit.offset_ns,
            drift_ppm=None if fit is None else (fit.slope - 1.0) * 1_000_000.0,
            fit_rms_ns=None if fit is None else fit.rms_ns,
            median_uncertainty_ns=(
                None if fit is None else fit.median_uncertainty_ns
            ),
        )

    def map_timestamp(
        self,
        device_timestamp_ns: int,
        *,
        now_pc_ns: int,
    ) -> MappedTimestamp | None:
        _nonnegative_int("device_timestamp_ns", device_timestamp_ns)
        diagnostics = self.diagnostics(now_pc_ns=now_pc_ns)
        fit = self._fit
        if diagnostics.state is not ClockMappingState.VALID or fit is None:
            return None

        mapped = fit.slope * device_timestamp_ns + fit.offset_ns
        if not math.isfinite(mapped) or mapped < 0:
            return None
        estimated_error = max(fit.rms_ns, fit.median_uncertainty_ns)
        return MappedTimestamp(
            device_timestamp_ns=device_timestamp_ns,
            pc_timestamp_ns=round(mapped),
            estimated_error_ns=math.ceil(estimated_error),
        )

    def _accept(self, sample: ClockSyncSample) -> None:
        self._samples.append(sample)
        self._fit = self._fit_samples(tuple(self._samples))

    def _newest_is_rejectable_outlier(
        self, samples: tuple[ClockSyncSample, ...]
    ) -> bool:
        if len(samples) < self._config.min_samples + 1:
            return False
        inliers = _robust_inliers(samples, self._config)
        if inliers is None or len(inliers) == len(samples):
            return False
        newest = samples[-1]
        if newest in inliers:
            return False
        if len(inliers) < self._config.min_samples:
            return False
        span_ns = inliers[-1].device_timestamp_ns - inliers[0].device_timestamp_ns
        return span_ns >= self._config.min_span_ns

    def _fit_samples(self, samples: tuple[ClockSyncSample, ...]) -> _ClockFit | None:
        if len(samples) < 2:
            return None
        selected = _robust_inliers(samples, self._config)
        if selected is None or (
            len(selected) < self._config.min_samples
            or selected[-1].device_timestamp_ns - selected[0].device_timestamp_ns
            < self._config.min_span_ns
        ):
            selected = samples
        return _ols_fit(selected)

    def _state(
        self,
        *,
        sample_count: int,
        device_span_ns: int,
        age_ns: int | None,
    ) -> tuple[ClockMappingState, ClockMappingInvalidReason | None]:
        if sample_count == 0:
            return ClockMappingState.UNAVAILABLE, ClockMappingInvalidReason.NO_SAMPLES
        if sample_count < self._config.min_samples:
            return (
                ClockMappingState.WARMING,
                ClockMappingInvalidReason.INSUFFICIENT_SAMPLES,
            )
        if device_span_ns < self._config.min_span_ns:
            return ClockMappingState.WARMING, ClockMappingInvalidReason.INSUFFICIENT_SPAN
        fit = self._fit
        if fit is None:
            return ClockMappingState.INVALID, ClockMappingInvalidReason.FIT_RMS
        if fit.rms_ns > self._config.max_fit_rms_ns:
            return ClockMappingState.INVALID, ClockMappingInvalidReason.FIT_RMS
        drift_ppm = abs((fit.slope - 1.0) * 1_000_000.0)
        if drift_ppm > self._config.max_abs_drift_ppm:
            return ClockMappingState.INVALID, ClockMappingInvalidReason.DRIFT_LIMIT
        if age_ns is not None and age_ns > self._config.max_mapping_age_ns:
            return ClockMappingState.STALE, ClockMappingInvalidReason.STALE
        return ClockMappingState.VALID, None


def _robust_inliers(
    samples: tuple[ClockSyncSample, ...], config: ClockMapperConfig
) -> tuple[ClockSyncSample, ...] | None:
    if len(samples) < 3:
        return None
    slopes = [
        (right.pc_timestamp_ns - left.pc_timestamp_ns)
        / (right.device_timestamp_ns - left.device_timestamp_ns)
        for index, left in enumerate(samples[:-1])
        for right in samples[index + 1 :]
        if right.device_timestamp_ns > left.device_timestamp_ns
    ]
    if not slopes:
        return None
    slope = float(median(slopes))
    intercept = float(
        median(
            sample.pc_timestamp_ns - slope * sample.device_timestamp_ns
            for sample in samples
        )
    )
    residuals = tuple(
        sample.pc_timestamp_ns - (slope * sample.device_timestamp_ns + intercept)
        for sample in samples
    )
    residual_center = float(median(residuals))
    mad = float(median(abs(value - residual_center) for value in residuals))
    robust_scale = 1.4826 * mad
    median_uncertainty = float(median(sample.uncertainty_ns for sample in samples))
    threshold = max(
        float(config.outlier_floor_ns),
        config.outlier_sigma * max(robust_scale, median_uncertainty, 1.0),
    )
    return tuple(
        sample
        for sample, residual in zip(samples, residuals, strict=True)
        if abs(residual - residual_center) <= threshold
    )


def _ols_fit(samples: tuple[ClockSyncSample, ...]) -> _ClockFit:
    mean_x = sum(sample.device_timestamp_ns for sample in samples) / len(samples)
    mean_y = sum(sample.pc_timestamp_ns for sample in samples) / len(samples)
    centered_x = [sample.device_timestamp_ns - mean_x for sample in samples]
    centered_y = [sample.pc_timestamp_ns - mean_y for sample in samples]
    denominator = sum(value * value for value in centered_x)
    slope = (
        1.0
        if denominator == 0
        else sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
        / denominator
    )
    offset = mean_y - slope * mean_x
    residuals = [
        sample.pc_timestamp_ns - (slope * sample.device_timestamp_ns + offset)
        for sample in samples
    ]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return _ClockFit(
        slope=float(slope),
        offset_ns=float(offset),
        rms_ns=float(rms),
        median_uncertainty_ns=float(
            median(sample.uncertainty_ns for sample in samples)
        ),
    )


def _nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _positive_int(name: str, value: int) -> None:
    _nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be > 0")


__all__ = [
    "ClockMapper",
    "ClockMapperConfig",
    "ClockMappingDiagnostics",
    "ClockMappingInvalidReason",
    "ClockMappingState",
    "ClockSampleOutcome",
    "ClockSyncSample",
    "MappedTimestamp",
]
