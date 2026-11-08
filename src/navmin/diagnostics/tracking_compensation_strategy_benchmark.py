"""Stage E synthetic comparison of tracking platform-compensation strategies.

This module is diagnostic-only.  It does not change production tracking or
control behaviour.  The benchmark holds the synthetic observation stream fixed
and compares four ways to combine the production trailing OLS relative-rate
estimate with command-model platform motion:

``current``
    Add instantaneous platform rate at PC observation/receive time.
``latency-corrected``
    Add instantaneous platform rate at the estimated image-capture time.
``window-aligned``
    Add the OLS platform-angle slope over the estimator's receive-time window.
``latency-corrected-window-aligned``
    Add the OLS platform-angle slope over the same window shifted by calibrated
    effective video latency.

The last three are diagnostic oracles/candidates only; none is wired into the
runtime by this stage.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from navmin.config import load_config
from navmin.config.models import AppConfig, TrackingControlConfig
from navmin.contracts import (
    CameraRole,
    PlatformMotionRate,
    PlatformMotionSnapshot,
    TargetRef,
    TrackingError,
)
from navmin.core.tracking_estimator import TrackingEstimator
from navmin.turret.platform_motion import CommandMotionModel

_DEFAULT_WINDOWS_MS = (200, 300, 500)
_DEFAULT_TRUE_LATENCIES_MS = (0, 100, 150, 200, 250)
_DEFAULT_CALIBRATION_ERRORS_MS = (-50, -25, -10, 0, 10, 25, 50)
_DEFAULT_FPS = 20
_BASE_NS = 20_000_000_000
_TIMELINE_END_MS = 2_700
_OBSERVATION_START_MS = 300
_OBSERVATION_END_MS = 2_500
_PROBE_PREDICTION_HORIZON_MS = 200
_PROBE_PREDICTION_MAX_ANGLE_DEG = 5.0
_PROBE_FF_GAIN = 0.25
_TARGET = TargetRef(CameraRole.OVERVIEW, generation=1, track_id=1)

CURRENT = "current"
LATENCY_CORRECTED = "latency-corrected"
WINDOW_ALIGNED = "window-aligned"
COMBINED = "latency-corrected-window-aligned"
_STRATEGIES = (CURRENT, LATENCY_CORRECTED, WINDOW_ALIGNED, COMBINED)


@dataclass(frozen=True, slots=True)
class CompensationScenario:
    name: str
    target_rate_deg_s: float
    velocity_commands: tuple[tuple[int, float], ...]
    jitter_drop: bool = False


@dataclass(frozen=True, slots=True)
class CompensationPeakSample:
    receive_time_ms: int
    true_capture_time_ms: int
    estimated_capture_time_ms: int
    true_target_rate_deg_s: float
    relative_rate_deg_s: float
    platform_compensation_deg_s: float
    estimated_target_rate_deg_s: float
    target_rate_error_deg_s: float
    probe_prediction_error_deg: float
    probe_rate_effect_command_error_deg_s: float
    sample_count: int
    actual_span_ms: float


@dataclass(frozen=True, slots=True)
class CompensationStrategyResult:
    strategy: str
    valid_frame_count: int
    peak_abs_target_rate_error_deg_s: float | None
    rms_target_rate_error_deg_s: float | None
    peak_abs_probe_prediction_error_deg: float | None
    peak_abs_probe_rate_effect_command_error_deg_s: float | None
    wrong_direction_frame_count: int
    peak_sample: CompensationPeakSample | None


@dataclass(frozen=True, slots=True)
class CompensationCaseResult:
    scenario: str
    window_ms: int
    true_video_latency_ms: int
    calibration_error_ms: int
    estimated_video_latency_ms: int
    frame_count: int
    valid_frame_count: int
    invalid_reasons: dict[str, int]
    strategies: tuple[CompensationStrategyResult, ...]


@dataclass(frozen=True, slots=True)
class TrackingCompensationFindings:
    representative_window_ms: int
    representative_true_latency_ms: int
    representative_current_peak_error_deg_s: float
    representative_latency_corrected_peak_error_deg_s: float
    representative_window_aligned_peak_error_deg_s: float
    representative_combined_peak_error_deg_s: float
    representative_combined_reduction_percent: float
    perfect_calibration_combined_max_error_deg_s: float
    combined_error_at_minus_50ms_deg_s: float
    combined_error_at_minus_25ms_deg_s: float
    combined_error_at_minus_10ms_deg_s: float
    combined_error_at_plus_10ms_deg_s: float
    combined_error_at_plus_25ms_deg_s: float
    combined_error_at_plus_50ms_deg_s: float
    current_accel_error_sign: int
    current_brake_error_sign: int
    combined_accel_error_sign: int
    combined_brake_error_sign: int
    current_reverses_error_direction: bool
    combined_removes_reversal_with_perfect_calibration: bool
    latency_correction_alone_leaves_window_bias: bool
    window_alignment_alone_leaves_latency_bias: bool
    combined_outperforms_partial_strategies: bool
    combined_with_25ms_error_beats_current: bool
    recommended_candidate: str


@dataclass(frozen=True, slots=True)
class TrackingCompensationBenchmarkResult:
    status: str
    config_path: str
    fps: int
    acceleration_x_deg_s2: float
    windows_ms: tuple[int, ...]
    true_latencies_ms: tuple[int, ...]
    calibration_errors_ms: tuple[int, ...]
    scenarios: tuple[str, ...]
    strategies: tuple[str, ...]
    probe_prediction_horizon_ms: int
    probe_prediction_max_angle_deg: float
    probe_ff_gain: float
    pid_kp_x: float
    cases: tuple[CompensationCaseResult, ...]
    findings: TrackingCompensationFindings

    @property
    def completed(self) -> bool:
        return self.status == "COMPLETE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Timeline:
    platform_angle_deg: tuple[float, ...]
    platform_motion: PlatformMotionSnapshot
    scenario: CompensationScenario

    def platform_rate(self, time_ms: int) -> float:
        if time_ms < 0:
            raise ValueError("time_ms must be non-negative")
        rate = self.platform_motion.rate_at(_BASE_NS + time_ms * 1_000_000)
        if rate is None or rate.rate_x_deg_s is None:
            raise RuntimeError(f"platform rate unavailable at {time_ms} ms")
        return rate.rate_x_deg_s

    def platform_angle(self, time_ms: int) -> float:
        if time_ms < 0 or time_ms >= len(self.platform_angle_deg):
            raise ValueError(f"time outside synthetic timeline: {time_ms} ms")
        return self.platform_angle_deg[time_ms]

    def target_angle(self, time_ms: int) -> float:
        return 5.0 + self.scenario.target_rate_deg_s * time_ms / 1000.0

    def error(self, time_ms: int) -> float:
        return self.target_angle(time_ms) - self.platform_angle(time_ms)


@dataclass(frozen=True, slots=True)
class _HistorySample:
    receive_time_ms: int
    true_capture_time_ms: int


_SCENARIOS = (
    CompensationScenario(
        name="stationary_target_accelerate",
        target_rate_deg_s=0.0,
        velocity_commands=((800, 40.0),),
    ),
    CompensationScenario(
        name="moving_target_accelerate",
        target_rate_deg_s=10.0,
        velocity_commands=((800, 40.0),),
    ),
    CompensationScenario(
        name="stationary_target_brake",
        target_rate_deg_s=0.0,
        velocity_commands=((200, 40.0), (900, 0.0)),
    ),
    CompensationScenario(
        name="stationary_target_reverse",
        target_rate_deg_s=0.0,
        velocity_commands=((200, 40.0), (900, -40.0)),
    ),
    CompensationScenario(
        name="stationary_target_accelerate_jitter_drop",
        target_rate_deg_s=0.0,
        velocity_commands=((800, 40.0),),
        jitter_drop=True,
    ),
)


def _build_timeline(config: AppConfig, scenario: CompensationScenario) -> _Timeline:
    model = CommandMotionModel(
        acceleration_x_deg_s2=config.turret.stm32.acceleration_x_deg_s2,
        acceleration_y_deg_s2=config.turret.stm32.acceleration_y_deg_s2,
    )
    model.mark_known_zero(timestamp_ns=_BASE_NS)
    for time_ms, velocity_deg_s in scenario.velocity_commands:
        model.record_velocity_command(
            timestamp_ns=_BASE_NS + time_ms * 1_000_000,
            commanded_rate_x_deg_s=velocity_deg_s,
            commanded_rate_y_deg_s=0.0,
        )
    snapshot = model.snapshot()
    angles = [0.0]
    for time_ms in range(1, _TIMELINE_END_MS + 1):
        before = snapshot.rate_at(_BASE_NS + (time_ms - 1) * 1_000_000)
        current = snapshot.rate_at(_BASE_NS + time_ms * 1_000_000)
        if (
            before is None
            or current is None
            or before.rate_x_deg_s is None
            or current.rate_x_deg_s is None
        ):
            raise RuntimeError("synthetic command model became unavailable")
        angles.append(
            angles[-1]
            + (before.rate_x_deg_s + current.rate_x_deg_s) * 0.5 / 1000.0
        )
    return _Timeline(
        platform_angle_deg=tuple(angles),
        platform_motion=snapshot,
        scenario=scenario,
    )


def _observation_times_ms(*, fps: int, jitter_drop: bool) -> tuple[int, ...]:
    period_ms = round(1000 / fps)
    nominal = list(range(_OBSERVATION_START_MS, _OBSERVATION_END_MS + 1, period_ms))
    if not jitter_drop:
        return tuple(nominal)
    jitter_pattern = (-8, 5, 12, -6, 0, 7, -10)
    observed: list[int] = []
    for index, value in enumerate(nominal):
        if index > 0 and index % 7 == 0:
            continue
        adjusted = value + jitter_pattern[index % len(jitter_pattern)]
        if observed and adjusted <= observed[-1]:
            adjusted = observed[-1] + 1
        observed.append(adjusted)
    return tuple(observed)


def _tracking_config_for_window(
    base: TrackingControlConfig,
    window_ms: int,
) -> TrackingControlConfig:
    return replace(
        base,
        velocity_estimator=replace(base.velocity_estimator, window_ms=window_ms),
    )


def _evict_history(
    history: list[_HistorySample],
    *,
    receive_time_ms: int,
    config: TrackingControlConfig,
) -> None:
    cutoff_ms = receive_time_ms - config.velocity_estimator.window_ms
    while history and history[0].receive_time_ms < cutoff_ms:
        history.pop(0)
    overflow = len(history) - config.velocity_estimator.max_samples
    if overflow > 0:
        del history[:overflow]


def _ols_slope(times_ms: list[int], values: list[float]) -> float | None:
    if len(times_ms) < 2:
        return None
    first = times_ms[0]
    times_s = [(value - first) / 1000.0 for value in times_ms]
    mean_time = sum(times_s) / len(times_s)
    mean_value = sum(values) / len(values)
    denominator = sum((value - mean_time) ** 2 for value in times_s)
    if denominator <= 0.0:
        return None
    return sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(times_s, values, strict=True)
    ) / denominator


def _probe_prediction(target_rate_deg_s: float) -> float:
    raw = target_rate_deg_s * _PROBE_PREDICTION_HORIZON_MS / 1000.0
    return max(
        -_PROBE_PREDICTION_MAX_ANGLE_DEG,
        min(_PROBE_PREDICTION_MAX_ANGLE_DEG, raw),
    )


def _sign(value: float, *, epsilon: float = 1e-9) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _strategy_result(
    strategy: str,
    samples: list[CompensationPeakSample],
) -> CompensationStrategyResult:
    if not samples:
        return CompensationStrategyResult(
            strategy=strategy,
            valid_frame_count=0,
            peak_abs_target_rate_error_deg_s=None,
            rms_target_rate_error_deg_s=None,
            peak_abs_probe_prediction_error_deg=None,
            peak_abs_probe_rate_effect_command_error_deg_s=None,
            wrong_direction_frame_count=0,
            peak_sample=None,
        )
    peak = max(samples, key=lambda item: abs(item.target_rate_error_deg_s))
    return CompensationStrategyResult(
        strategy=strategy,
        valid_frame_count=len(samples),
        peak_abs_target_rate_error_deg_s=max(
            abs(item.target_rate_error_deg_s) for item in samples
        ),
        rms_target_rate_error_deg_s=math.sqrt(
            sum(item.target_rate_error_deg_s**2 for item in samples) / len(samples)
        ),
        peak_abs_probe_prediction_error_deg=max(
            abs(item.probe_prediction_error_deg) for item in samples
        ),
        peak_abs_probe_rate_effect_command_error_deg_s=max(
            abs(item.probe_rate_effect_command_error_deg_s) for item in samples
        ),
        wrong_direction_frame_count=sum(
            1
            for item in samples
            if abs(item.true_target_rate_deg_s) > 1e-9
            and item.estimated_target_rate_deg_s * item.true_target_rate_deg_s < 0.0
        ),
        peak_sample=peak,
    )


def _run_case(
    config: AppConfig,
    timeline: _Timeline,
    scenario: CompensationScenario,
    *,
    window_ms: int,
    true_video_latency_ms: int,
    calibration_error_ms: int,
    fps: int,
) -> CompensationCaseResult:
    estimated_latency_ms = true_video_latency_ms + calibration_error_ms
    if estimated_latency_ms < 0:
        raise ValueError("estimated latency must not be negative")
    tracking_config = _tracking_config_for_window(config.tracking_control, window_ms)
    estimator = TrackingEstimator(tracking_config)
    history: list[_HistorySample] = []
    invalid_reasons: Counter[str] = Counter()
    samples_by_strategy: dict[str, list[CompensationPeakSample]] = {
        strategy: [] for strategy in _STRATEGIES
    }
    frame_count = 0

    for receive_time_ms in _observation_times_ms(
        fps=fps,
        jitter_drop=scenario.jitter_drop,
    ):
        true_capture_time_ms = receive_time_ms - true_video_latency_ms
        estimated_capture_time_ms = receive_time_ms - estimated_latency_ms
        if true_capture_time_ms < 0 or estimated_capture_time_ms < 0:
            continue
        frame_count += 1
        real_error = timeline.error(true_capture_time_ms)
        observation_timestamp_ns = _BASE_NS + receive_time_ms * 1_000_000
        platform_rate_receive = timeline.platform_rate(receive_time_ms)
        history.append(
            _HistorySample(
                receive_time_ms=receive_time_ms,
                true_capture_time_ms=true_capture_time_ms,
            )
        )
        _evict_history(
            history,
            receive_time_ms=receive_time_ms,
            config=tracking_config,
        )
        diagnostics = estimator.observe(
            TrackingError(
                target=_TARGET,
                error_x_deg=real_error,
                error_y_deg=0.0,
                timestamp_ns=observation_timestamp_ns,
            ),
            PlatformMotionRate(
                timestamp_ns=observation_timestamp_ns,
                rate_x_deg_s=platform_rate_receive,
                rate_y_deg_s=0.0,
            ),
        )
        axis = diagnostics.x
        if axis.sample_count != len(history):
            raise RuntimeError(
                "benchmark history diverged from TrackingEstimator history"
            )
        if not axis.valid or axis.relative_rate_deg_s is None:
            reason = (
                "unknown"
                if axis.invalid_reason is None
                else axis.invalid_reason.value
            )
            invalid_reasons[reason] += 1
            continue

        receive_times = [item.receive_time_ms for item in history]
        receive_window_platform_slope = _ols_slope(
            receive_times,
            [timeline.platform_angle(item.receive_time_ms) for item in history],
        )
        estimated_capture_window_platform_slope = _ols_slope(
            receive_times,
            [
                timeline.platform_angle(item.receive_time_ms - estimated_latency_ms)
                for item in history
            ],
        )
        if (
            receive_window_platform_slope is None
            or estimated_capture_window_platform_slope is None
        ):
            raise RuntimeError("valid estimator sample lacks platform window slope")

        compensation_by_strategy = {
            CURRENT: platform_rate_receive,
            LATENCY_CORRECTED: timeline.platform_rate(estimated_capture_time_ms),
            WINDOW_ALIGNED: receive_window_platform_slope,
            COMBINED: estimated_capture_window_platform_slope,
        }
        true_prediction = _probe_prediction(scenario.target_rate_deg_s)
        for strategy, platform_compensation in compensation_by_strategy.items():
            estimated_rate = axis.relative_rate_deg_s + platform_compensation
            rate_error = estimated_rate - scenario.target_rate_deg_s
            prediction_error = _probe_prediction(estimated_rate) - true_prediction
            rate_effect_command_error = (
                config.turret.controller.pid_kp_x * prediction_error
                + _PROBE_FF_GAIN * rate_error
            )
            samples_by_strategy[strategy].append(
                CompensationPeakSample(
                    receive_time_ms=receive_time_ms,
                    true_capture_time_ms=true_capture_time_ms,
                    estimated_capture_time_ms=estimated_capture_time_ms,
                    true_target_rate_deg_s=scenario.target_rate_deg_s,
                    relative_rate_deg_s=axis.relative_rate_deg_s,
                    platform_compensation_deg_s=platform_compensation,
                    estimated_target_rate_deg_s=estimated_rate,
                    target_rate_error_deg_s=rate_error,
                    probe_prediction_error_deg=prediction_error,
                    probe_rate_effect_command_error_deg_s=rate_effect_command_error,
                    sample_count=axis.sample_count,
                    actual_span_ms=axis.actual_span_ms,
                )
            )

    strategy_results = tuple(
        _strategy_result(strategy, samples_by_strategy[strategy])
        for strategy in _STRATEGIES
    )
    valid_frame_count = max(
        (result.valid_frame_count for result in strategy_results),
        default=0,
    )
    return CompensationCaseResult(
        scenario=scenario.name,
        window_ms=window_ms,
        true_video_latency_ms=true_video_latency_ms,
        calibration_error_ms=calibration_error_ms,
        estimated_video_latency_ms=estimated_latency_ms,
        frame_count=frame_count,
        valid_frame_count=valid_frame_count,
        invalid_reasons=dict(sorted(invalid_reasons.items())),
        strategies=strategy_results,
    )


def _find_case(
    cases: tuple[CompensationCaseResult, ...],
    *,
    scenario: str,
    window_ms: int,
    true_latency_ms: int,
    calibration_error_ms: int,
) -> CompensationCaseResult:
    for case in cases:
        if (
            case.scenario == scenario
            and case.window_ms == window_ms
            and case.true_video_latency_ms == true_latency_ms
            and case.calibration_error_ms == calibration_error_ms
        ):
            return case
    raise RuntimeError(
        "missing compensation case "
        f"{scenario} window={window_ms} latency={true_latency_ms} "
        f"calibration_error={calibration_error_ms}"
    )


def _find_strategy(
    case: CompensationCaseResult,
    strategy: str,
) -> CompensationStrategyResult:
    for result in case.strategies:
        if result.strategy == strategy:
            return result
    raise RuntimeError(f"missing strategy {strategy}")


def _require_peak_error(
    cases: tuple[CompensationCaseResult, ...],
    *,
    scenario: str,
    window_ms: int,
    true_latency_ms: int,
    calibration_error_ms: int,
    strategy: str,
) -> float:
    case = _find_case(
        cases,
        scenario=scenario,
        window_ms=window_ms,
        true_latency_ms=true_latency_ms,
        calibration_error_ms=calibration_error_ms,
    )
    result = _find_strategy(case, strategy)
    if result.peak_abs_target_rate_error_deg_s is None:
        raise RuntimeError("representative compensation case has no valid samples")
    return result.peak_abs_target_rate_error_deg_s


def _require_peak_signed_error(
    cases: tuple[CompensationCaseResult, ...],
    *,
    scenario: str,
    window_ms: int,
    true_latency_ms: int,
    calibration_error_ms: int,
    strategy: str,
) -> float:
    case = _find_case(
        cases,
        scenario=scenario,
        window_ms=window_ms,
        true_latency_ms=true_latency_ms,
        calibration_error_ms=calibration_error_ms,
    )
    result = _find_strategy(case, strategy)
    if result.peak_sample is None:
        raise RuntimeError("representative compensation case has no peak sample")
    return result.peak_sample.target_rate_error_deg_s


def _build_findings(
    cases: tuple[CompensationCaseResult, ...],
) -> TrackingCompensationFindings:
    window_ms = 200
    latency_ms = 150
    scenario = "stationary_target_accelerate"
    current = _require_peak_error(
        cases,
        scenario=scenario,
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=CURRENT,
    )
    latency_corrected = _require_peak_error(
        cases,
        scenario=scenario,
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=LATENCY_CORRECTED,
    )
    window_aligned = _require_peak_error(
        cases,
        scenario=scenario,
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=WINDOW_ALIGNED,
    )
    combined = _require_peak_error(
        cases,
        scenario=scenario,
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=COMBINED,
    )
    sensitivity = {
        offset: _require_peak_error(
            cases,
            scenario=scenario,
            window_ms=window_ms,
            true_latency_ms=latency_ms,
            calibration_error_ms=offset,
            strategy=COMBINED,
        )
        for offset in (-50, -25, -10, 10, 25, 50)
    }
    perfect_combined_errors = [
        result.peak_abs_target_rate_error_deg_s
        for case in cases
        if case.calibration_error_ms == 0
        for result in case.strategies
        if result.strategy == COMBINED
        and result.peak_abs_target_rate_error_deg_s is not None
    ]
    accel_current_signed = _require_peak_signed_error(
        cases,
        scenario="stationary_target_accelerate",
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=CURRENT,
    )
    brake_current_signed = _require_peak_signed_error(
        cases,
        scenario="stationary_target_brake",
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=CURRENT,
    )
    accel_combined_signed = _require_peak_signed_error(
        cases,
        scenario="stationary_target_accelerate",
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=COMBINED,
    )
    brake_combined_signed = _require_peak_signed_error(
        cases,
        scenario="stationary_target_brake",
        window_ms=window_ms,
        true_latency_ms=latency_ms,
        calibration_error_ms=0,
        strategy=COMBINED,
    )
    current_accel_sign = _sign(accel_current_signed)
    current_brake_sign = _sign(brake_current_signed)
    combined_accel_sign = _sign(accel_combined_signed, epsilon=0.05)
    combined_brake_sign = _sign(brake_combined_signed, epsilon=0.05)
    reduction = 0.0 if current <= 0.0 else 100.0 * (1.0 - combined / current)
    return TrackingCompensationFindings(
        representative_window_ms=window_ms,
        representative_true_latency_ms=latency_ms,
        representative_current_peak_error_deg_s=current,
        representative_latency_corrected_peak_error_deg_s=latency_corrected,
        representative_window_aligned_peak_error_deg_s=window_aligned,
        representative_combined_peak_error_deg_s=combined,
        representative_combined_reduction_percent=reduction,
        perfect_calibration_combined_max_error_deg_s=max(perfect_combined_errors),
        combined_error_at_minus_50ms_deg_s=sensitivity[-50],
        combined_error_at_minus_25ms_deg_s=sensitivity[-25],
        combined_error_at_minus_10ms_deg_s=sensitivity[-10],
        combined_error_at_plus_10ms_deg_s=sensitivity[10],
        combined_error_at_plus_25ms_deg_s=sensitivity[25],
        combined_error_at_plus_50ms_deg_s=sensitivity[50],
        current_accel_error_sign=current_accel_sign,
        current_brake_error_sign=current_brake_sign,
        combined_accel_error_sign=combined_accel_sign,
        combined_brake_error_sign=combined_brake_sign,
        current_reverses_error_direction=(
            current_accel_sign != 0
            and current_brake_sign != 0
            and current_accel_sign == -current_brake_sign
        ),
        combined_removes_reversal_with_perfect_calibration=(
            combined_accel_sign == 0 and combined_brake_sign == 0
        ),
        latency_correction_alone_leaves_window_bias=(
            latency_corrected > max(0.5, combined * 10.0)
        ),
        window_alignment_alone_leaves_latency_bias=(
            window_aligned > max(0.5, combined * 10.0)
        ),
        combined_outperforms_partial_strategies=(
            combined < latency_corrected and combined < window_aligned
        ),
        combined_with_25ms_error_beats_current=max(
            sensitivity[-25], sensitivity[25]
        )
        < current,
        recommended_candidate=COMBINED,
    )


def run_tracking_compensation_strategy_benchmark(
    config_path: str | Path = "config/navmin.json",
    *,
    windows_ms: tuple[int, ...] = _DEFAULT_WINDOWS_MS,
    true_latencies_ms: tuple[int, ...] = _DEFAULT_TRUE_LATENCIES_MS,
    calibration_errors_ms: tuple[int, ...] = _DEFAULT_CALIBRATION_ERRORS_MS,
    fps: int = _DEFAULT_FPS,
) -> TrackingCompensationBenchmarkResult:
    """Compare four diagnostic platform-compensation strategies."""

    if fps <= 0:
        raise ValueError("fps must be > 0")
    if not windows_ms or any(value <= 0 for value in windows_ms):
        raise ValueError("windows_ms must contain positive values")
    if not true_latencies_ms or any(value < 0 for value in true_latencies_ms):
        raise ValueError("true_latencies_ms must contain non-negative values")
    if not calibration_errors_ms:
        raise ValueError("calibration_errors_ms must not be empty")
    if 0 not in calibration_errors_ms:
        raise ValueError("calibration_errors_ms must include 0")

    path = Path(config_path)
    config = load_config(path)
    timelines = {
        scenario.name: _build_timeline(config, scenario) for scenario in _SCENARIOS
    }
    cases = tuple(
        _run_case(
            config,
            timelines[scenario.name],
            scenario,
            window_ms=window_ms,
            true_video_latency_ms=true_latency_ms,
            calibration_error_ms=calibration_error_ms,
            fps=fps,
        )
        for scenario in _SCENARIOS
        for window_ms in windows_ms
        for true_latency_ms in true_latencies_ms
        for calibration_error_ms in calibration_errors_ms
        if true_latency_ms + calibration_error_ms >= 0
    )
    findings = _build_findings(cases)
    return TrackingCompensationBenchmarkResult(
        status="COMPLETE",
        config_path=str(path),
        fps=fps,
        acceleration_x_deg_s2=config.turret.stm32.acceleration_x_deg_s2,
        windows_ms=windows_ms,
        true_latencies_ms=true_latencies_ms,
        calibration_errors_ms=calibration_errors_ms,
        scenarios=tuple(scenario.name for scenario in _SCENARIOS),
        strategies=_STRATEGIES,
        probe_prediction_horizon_ms=_PROBE_PREDICTION_HORIZON_MS,
        probe_prediction_max_angle_deg=_PROBE_PREDICTION_MAX_ANGLE_DEG,
        probe_ff_gain=_PROBE_FF_GAIN,
        pid_kp_x=config.turret.controller.pid_kp_x,
        cases=cases,
        findings=findings,
    )


def write_tracking_compensation_strategy_json(
    result: TrackingCompensationBenchmarkResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "COMBINED",
    "CURRENT",
    "LATENCY_CORRECTED",
    "WINDOW_ALIGNED",
    "CompensationCaseResult",
    "CompensationPeakSample",
    "CompensationScenario",
    "CompensationStrategyResult",
    "TrackingCompensationBenchmarkResult",
    "TrackingCompensationFindings",
    "run_tracking_compensation_strategy_benchmark",
    "write_tracking_compensation_strategy_json",
]
