"""Synthetic benchmark for tracking-rate bias during commanded turret motion.

The benchmark deliberately exercises the production ``TrackingEstimator``,
``CommandMotionModel`` and ``AxisPid`` without changing control behaviour.  It
models a camera observation as an angular error captured some fixed time before
its PC observation timestamp.  This makes it possible to separate two effects:

* source/receive latency: platform rate at receipt differs from the rate when the
  image was formed;
* estimator-window mismatch: the fitted relative rate is a trailing-window
  quantity while current production compensation adds an instantaneous platform
  rate.

The window-matched values in this module are diagnostic oracles only.  They are
not a proposed production control path.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from navmin.config import load_config
from navmin.config.models import AppConfig, TrackingControlConfig
from navmin.contracts import (
    CameraRole,
    PlatformMotionRate,
    TargetRef,
    TrackingError,
)
from navmin.core.tracking_estimator import TrackingEstimator
from navmin.turret.pid import AxisPid, PidGains
from navmin.turret.platform_motion import CommandMotionModel

_DEFAULT_WINDOWS_MS = (100, 200, 300, 500)
_DEFAULT_LATENCIES_MS = (0, 25, 50, 75, 100, 125, 150, 175, 200, 250)
_DEFAULT_FPS = 20
_BASE_NS = 10_000_000_000
_TIMELINE_END_MS = 2_600
_TIMELINE_STEP_MS = 1
_OBSERVATION_START_MS = 300
_OBSERVATION_END_MS = 2_500
_TARGET = TargetRef(CameraRole.OVERVIEW, generation=1, track_id=1)


@dataclass(frozen=True, slots=True)
class TrackingLatencyScenario:
    name: str
    target_rate_deg_s: float
    velocity_commands: tuple[tuple[int, float], ...]
    jitter_drop: bool = False


@dataclass(frozen=True, slots=True)
class TrackingLatencyPeakSample:
    receive_time_ms: int
    capture_time_ms: int
    real_error_deg: float
    true_target_rate_deg_s: float
    relative_rate_deg_s: float
    platform_rate_receive_deg_s: float
    platform_rate_capture_deg_s: float
    platform_rate_window_matched_deg_s: float
    estimated_target_rate_deg_s: float
    capture_aligned_target_rate_deg_s: float
    window_matched_target_rate_deg_s: float
    target_rate_error_deg_s: float
    latency_component_deg_s: float
    window_component_deg_s: float
    prediction_offset_deg: float
    true_prediction_offset_deg: float
    prediction_error_deg: float
    p_term_deg_s: float
    i_term_deg_s: float
    d_term_deg_s: float
    ff_term_deg_s: float
    command_deg_s: float
    ideal_command_deg_s: float
    command_error_deg_s: float
    fit_rms_deg: float
    sample_count: int
    actual_span_ms: float


@dataclass(frozen=True, slots=True)
class TrackingLatencyCaseResult:
    scenario: str
    window_ms: int
    video_latency_ms: int
    frame_count: int
    valid_frame_count: int
    invalid_reasons: dict[str, int]
    peak_abs_target_rate_error_deg_s: float | None
    rms_target_rate_error_deg_s: float | None
    peak_abs_prediction_error_deg: float | None
    peak_abs_command_error_deg_s: float | None
    peak_abs_latency_component_deg_s: float | None
    peak_abs_window_component_deg_s: float | None
    peak_abs_window_matched_target_rate_error_deg_s: float | None
    wrong_direction_frame_count: int
    validity_transition_count: int
    valid_to_invalid_count: int
    invalid_to_valid_count: int
    prediction_saturation_frame_count: int
    peak_sample: TrackingLatencyPeakSample | None

    @property
    def has_valid_estimate(self) -> bool:
        return self.valid_frame_count > 0


@dataclass(frozen=True, slots=True)
class TrackingLatencyFindings:
    hypothesis_reproduced: bool
    window_only_bias_reproduced: bool
    latency_amplifies_bias: bool
    accel_rate_error_sign: int
    brake_rate_error_sign: int
    reversal_of_rate_error_direction_reproduced: bool
    prediction_effect_active: bool
    accel_prediction_sign: int
    brake_prediction_sign: int
    reversal_of_prediction_direction_reproduced: bool
    regular_20fps_window_100ms_never_valid: bool
    representative_window_ms: int
    representative_latency_ms: int
    representative_accel_peak_rate_error_deg_s: float
    representative_accel_peak_prediction_error_deg: float
    representative_brake_peak_rate_error_deg_s: float
    representative_brake_peak_prediction_error_deg: float


@dataclass(frozen=True, slots=True)
class TrackingLatencyBenchmarkResult:
    status: str
    config_path: str
    fps: int
    prediction_horizon_ms: int
    prediction_enabled: bool
    feed_forward_enabled: bool
    ff_gain_x: float
    pid_kp_x: float
    pid_ki_x: float
    pid_kd_x: float
    max_speed_x_deg_s: float
    acceleration_x_deg_s2: float
    windows_ms: tuple[int, ...]
    latencies_ms: tuple[int, ...]
    scenarios: tuple[str, ...]
    cases: tuple[TrackingLatencyCaseResult, ...]
    findings: TrackingLatencyFindings

    @property
    def completed(self) -> bool:
        return self.status == "COMPLETE"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Timeline:
    model: CommandMotionModel
    platform_angle_deg: tuple[float, ...]
    scenario: TrackingLatencyScenario

    def platform_rate(self, time_ms: int) -> float:
        rate = self.model.snapshot().rate_at(_BASE_NS + time_ms * 1_000_000)
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
    capture_time_ms: int


_SCENARIOS = (
    TrackingLatencyScenario(
        name="stationary_target_accelerate",
        target_rate_deg_s=0.0,
        velocity_commands=((800, 40.0),),
    ),
    TrackingLatencyScenario(
        name="moving_target_accelerate",
        target_rate_deg_s=10.0,
        velocity_commands=((800, 40.0),),
    ),
    TrackingLatencyScenario(
        name="stationary_target_brake",
        target_rate_deg_s=0.0,
        velocity_commands=((200, 40.0), (900, 0.0)),
    ),
    TrackingLatencyScenario(
        name="stationary_target_reverse",
        target_rate_deg_s=0.0,
        velocity_commands=((200, 40.0), (900, -40.0)),
    ),
    TrackingLatencyScenario(
        name="stationary_target_accelerate_jitter_drop",
        target_rate_deg_s=0.0,
        velocity_commands=((800, 40.0),),
        jitter_drop=True,
    ),
)


def _build_timeline(config: AppConfig, scenario: TrackingLatencyScenario) -> _Timeline:
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
    for time_ms in range(_TIMELINE_STEP_MS, _TIMELINE_END_MS + 1, _TIMELINE_STEP_MS):
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
            + (before.rate_x_deg_s + current.rate_x_deg_s)
            * 0.5
            * _TIMELINE_STEP_MS
            / 1000.0
        )
    return _Timeline(model=model, platform_angle_deg=tuple(angles), scenario=scenario)


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
    base: TrackingControlConfig, window_ms: int
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
    cutoff = receive_time_ms - config.velocity_estimator.window_ms
    while history and history[0].receive_time_ms < cutoff:
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


def _clamped_prediction(config: TrackingControlConfig, target_rate: float) -> float:
    if not config.prediction.enabled:
        return 0.0
    raw = target_rate * config.prediction.horizon_ms / 1000.0
    return max(
        -config.prediction.max_angle_x_deg,
        min(config.prediction.max_angle_x_deg, raw),
    )


def _pid(config: AppConfig) -> AxisPid:
    gains = config.turret.controller
    return AxisPid(
        PidGains(gains.pid_kp_x, gains.pid_ki_x, gains.pid_kd_x),
        output_limit=config.turret.stm32.max_speed_x_deg_s,
    )


def _sign(value: float, *, epsilon: float = 1e-9) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _run_case(
    config: AppConfig,
    scenario: TrackingLatencyScenario,
    *,
    window_ms: int,
    video_latency_ms: int,
    fps: int,
) -> TrackingLatencyCaseResult:
    tracking_config = _tracking_config_for_window(config.tracking_control, window_ms)
    timeline = _build_timeline(config, scenario)
    estimator = TrackingEstimator(tracking_config)
    actual_pid = _pid(config)
    ideal_pid = _pid(config)
    history: list[_HistorySample] = []
    invalid_reasons: Counter[str] = Counter()
    valid_samples: list[TrackingLatencyPeakSample] = []
    wrong_direction_frames = 0
    validity_transition_count = 0
    valid_to_invalid_count = 0
    invalid_to_valid_count = 0
    prediction_saturation_frame_count = 0
    previous_usable: bool | None = None

    observation_times = _observation_times_ms(
        fps=fps,
        jitter_drop=scenario.jitter_drop,
    )
    for receive_time_ms in observation_times:
        capture_time_ms = receive_time_ms - video_latency_ms
        if capture_time_ms < 0:
            continue
        real_error = timeline.error(capture_time_ms)
        observation_timestamp_ns = _BASE_NS + receive_time_ms * 1_000_000
        platform_rate_receive = timeline.platform_rate(receive_time_ms)
        platform_rate_capture = timeline.platform_rate(capture_time_ms)

        history.append(
            _HistorySample(
                receive_time_ms=receive_time_ms,
                capture_time_ms=capture_time_ms,
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

        usable = axis.valid and axis.target_rate_deg_s is not None
        if previous_usable is not None and usable != previous_usable:
            validity_transition_count += 1
            if previous_usable:
                valid_to_invalid_count += 1
            else:
                invalid_to_valid_count += 1
        previous_usable = usable
        estimated_rate = axis.target_rate_deg_s if usable else None
        prediction = axis.prediction_offset_deg if usable else 0.0
        ff = (
            tracking_config.feed_forward.gain_x * estimated_rate
            if (
                usable
                and tracking_config.feed_forward.enabled
                and estimated_rate is not None
            )
            else 0.0
        )
        command = actual_pid.step(
            real_error,
            now_s=receive_time_ms / 1000.0,
            proportional_offset=prediction,
            feed_forward=ff,
        )

        true_prediction = _clamped_prediction(
            tracking_config, scenario.target_rate_deg_s
        )
        true_ff = (
            tracking_config.feed_forward.gain_x * scenario.target_rate_deg_s
            if tracking_config.feed_forward.enabled
            else 0.0
        )
        ideal_command = ideal_pid.step(
            real_error,
            now_s=receive_time_ms / 1000.0,
            proportional_offset=true_prediction,
            feed_forward=true_ff,
        )

        if not usable:
            reason = (
                "unknown"
                if axis.invalid_reason is None
                else axis.invalid_reason.value
            )
            invalid_reasons[reason] += 1
            continue

        if (
            tracking_config.prediction.enabled
            and axis.raw_prediction_offset_deg != axis.prediction_offset_deg
        ):
            prediction_saturation_frame_count += 1

        platform_window_slope = _ols_slope(
            [sample.receive_time_ms for sample in history],
            [timeline.platform_angle(sample.capture_time_ms) for sample in history],
        )
        if platform_window_slope is None or axis.relative_rate_deg_s is None:
            raise RuntimeError("valid estimator sample lacks a window-matched slope")

        capture_aligned_target_rate = axis.relative_rate_deg_s + platform_rate_capture
        window_matched_target_rate = axis.relative_rate_deg_s + platform_window_slope
        target_rate_error = estimated_rate - scenario.target_rate_deg_s
        prediction_error = prediction - true_prediction
        latency_component = platform_rate_receive - platform_rate_capture
        window_component = platform_rate_capture - platform_window_slope
        if (
            abs(scenario.target_rate_deg_s) > 1e-9
            and estimated_rate * scenario.target_rate_deg_s < 0.0
        ):
            wrong_direction_frames += 1

        step = actual_pid.last_step
        if step is None:
            raise RuntimeError("AxisPid omitted diagnostics after step")
        valid_samples.append(
            TrackingLatencyPeakSample(
                receive_time_ms=receive_time_ms,
                capture_time_ms=capture_time_ms,
                real_error_deg=real_error,
                true_target_rate_deg_s=scenario.target_rate_deg_s,
                relative_rate_deg_s=axis.relative_rate_deg_s,
                platform_rate_receive_deg_s=platform_rate_receive,
                platform_rate_capture_deg_s=platform_rate_capture,
                platform_rate_window_matched_deg_s=platform_window_slope,
                estimated_target_rate_deg_s=estimated_rate,
                capture_aligned_target_rate_deg_s=capture_aligned_target_rate,
                window_matched_target_rate_deg_s=window_matched_target_rate,
                target_rate_error_deg_s=target_rate_error,
                latency_component_deg_s=latency_component,
                window_component_deg_s=window_component,
                prediction_offset_deg=prediction,
                true_prediction_offset_deg=true_prediction,
                prediction_error_deg=prediction_error,
                p_term_deg_s=step.p_term,
                i_term_deg_s=step.i_term,
                d_term_deg_s=step.d_term,
                ff_term_deg_s=step.feed_forward,
                command_deg_s=command,
                ideal_command_deg_s=ideal_command,
                command_error_deg_s=command - ideal_command,
                fit_rms_deg=0.0 if axis.fit_rms_deg is None else axis.fit_rms_deg,
                sample_count=axis.sample_count,
                actual_span_ms=axis.actual_span_ms,
            )
        )

    if not valid_samples:
        return TrackingLatencyCaseResult(
            scenario=scenario.name,
            window_ms=window_ms,
            video_latency_ms=video_latency_ms,
            frame_count=sum(invalid_reasons.values()),
            valid_frame_count=0,
            invalid_reasons=dict(sorted(invalid_reasons.items())),
            peak_abs_target_rate_error_deg_s=None,
            rms_target_rate_error_deg_s=None,
            peak_abs_prediction_error_deg=None,
            peak_abs_command_error_deg_s=None,
            peak_abs_latency_component_deg_s=None,
            peak_abs_window_component_deg_s=None,
            peak_abs_window_matched_target_rate_error_deg_s=None,
            wrong_direction_frame_count=0,
            validity_transition_count=validity_transition_count,
            valid_to_invalid_count=valid_to_invalid_count,
            invalid_to_valid_count=invalid_to_valid_count,
            prediction_saturation_frame_count=0,
            peak_sample=None,
        )

    peak_sample = max(valid_samples, key=lambda item: abs(item.target_rate_error_deg_s))
    return TrackingLatencyCaseResult(
        scenario=scenario.name,
        window_ms=window_ms,
        video_latency_ms=video_latency_ms,
        frame_count=len(valid_samples) + sum(invalid_reasons.values()),
        valid_frame_count=len(valid_samples),
        invalid_reasons=dict(sorted(invalid_reasons.items())),
        peak_abs_target_rate_error_deg_s=max(
            abs(item.target_rate_error_deg_s) for item in valid_samples
        ),
        rms_target_rate_error_deg_s=math.sqrt(
            sum(item.target_rate_error_deg_s**2 for item in valid_samples)
            / len(valid_samples)
        ),
        peak_abs_prediction_error_deg=max(
            abs(item.prediction_error_deg) for item in valid_samples
        ),
        peak_abs_command_error_deg_s=max(
            abs(item.command_error_deg_s) for item in valid_samples
        ),
        peak_abs_latency_component_deg_s=max(
            abs(item.latency_component_deg_s) for item in valid_samples
        ),
        peak_abs_window_component_deg_s=max(
            abs(item.window_component_deg_s) for item in valid_samples
        ),
        peak_abs_window_matched_target_rate_error_deg_s=max(
            abs(item.window_matched_target_rate_deg_s - scenario.target_rate_deg_s)
            for item in valid_samples
        ),
        wrong_direction_frame_count=wrong_direction_frames,
        validity_transition_count=validity_transition_count,
        valid_to_invalid_count=valid_to_invalid_count,
        invalid_to_valid_count=invalid_to_valid_count,
        prediction_saturation_frame_count=prediction_saturation_frame_count,
        peak_sample=peak_sample,
    )


def _find_case(
    cases: tuple[TrackingLatencyCaseResult, ...],
    scenario: str,
    window_ms: int,
    latency_ms: int,
) -> TrackingLatencyCaseResult:
    for case in cases:
        if (
            case.scenario == scenario
            and case.window_ms == window_ms
            and case.video_latency_ms == latency_ms
        ):
            return case
    raise RuntimeError(
        f"missing benchmark case {scenario} window={window_ms} latency={latency_ms}"
    )


def _require_peak(case: TrackingLatencyCaseResult) -> TrackingLatencyPeakSample:
    if case.peak_sample is None:
        raise RuntimeError(
            f"benchmark case has no valid samples: {case.scenario} "
            f"window={case.window_ms} latency={case.video_latency_ms}"
        )
    return case.peak_sample


def _build_findings(
    cases: tuple[TrackingLatencyCaseResult, ...],
) -> TrackingLatencyFindings:
    representative_window = 200
    representative_latency = 150
    accel = _find_case(
        cases,
        "stationary_target_accelerate",
        representative_window,
        representative_latency,
    )
    accel_zero_latency = _find_case(
        cases,
        "stationary_target_accelerate",
        representative_window,
        0,
    )
    brake = _find_case(
        cases,
        "stationary_target_brake",
        representative_window,
        representative_latency,
    )
    accel_peak = _require_peak(accel)
    accel_zero_peak = _require_peak(accel_zero_latency)
    brake_peak = _require_peak(brake)
    window_100_cases = tuple(
        case
        for case in cases
        if case.scenario == "stationary_target_accelerate" and case.window_ms == 100
    )

    accel_rate_sign = _sign(accel_peak.target_rate_error_deg_s)
    brake_rate_sign = _sign(brake_peak.target_rate_error_deg_s)
    prediction_effect_active = any(
        abs(value) > 1e-9
        for value in (
            accel_peak.prediction_error_deg,
            brake_peak.prediction_error_deg,
        )
    )
    accel_prediction_sign = _sign(accel_peak.prediction_error_deg)
    brake_prediction_sign = _sign(brake_peak.prediction_error_deg)
    return TrackingLatencyFindings(
        hypothesis_reproduced=abs(accel_peak.target_rate_error_deg_s) >= 5.0,
        window_only_bias_reproduced=(
            abs(accel_zero_peak.target_rate_error_deg_s) >= 5.0
            and abs(accel_zero_peak.latency_component_deg_s) < 1e-9
        ),
        latency_amplifies_bias=(
            abs(accel_peak.target_rate_error_deg_s)
            > abs(accel_zero_peak.target_rate_error_deg_s)
        ),
        accel_rate_error_sign=accel_rate_sign,
        brake_rate_error_sign=brake_rate_sign,
        reversal_of_rate_error_direction_reproduced=(
            accel_rate_sign != 0
            and brake_rate_sign != 0
            and accel_rate_sign == -brake_rate_sign
        ),
        prediction_effect_active=prediction_effect_active,
        accel_prediction_sign=accel_prediction_sign,
        brake_prediction_sign=brake_prediction_sign,
        reversal_of_prediction_direction_reproduced=(
            prediction_effect_active
            and accel_prediction_sign != 0
            and brake_prediction_sign != 0
            and accel_prediction_sign == -brake_prediction_sign
        ),
        regular_20fps_window_100ms_never_valid=bool(window_100_cases)
        and all(case.valid_frame_count == 0 for case in window_100_cases),
        representative_window_ms=representative_window,
        representative_latency_ms=representative_latency,
        representative_accel_peak_rate_error_deg_s=accel_peak.target_rate_error_deg_s,
        representative_accel_peak_prediction_error_deg=accel_peak.prediction_error_deg,
        representative_brake_peak_rate_error_deg_s=brake_peak.target_rate_error_deg_s,
        representative_brake_peak_prediction_error_deg=brake_peak.prediction_error_deg,
    )


def run_tracking_latency_benchmark(
    config_path: str | Path = "config/navmin.json",
    *,
    windows_ms: tuple[int, ...] = _DEFAULT_WINDOWS_MS,
    latencies_ms: tuple[int, ...] = _DEFAULT_LATENCIES_MS,
    fps: int = _DEFAULT_FPS,
) -> TrackingLatencyBenchmarkResult:
    """Run the complete Stage-B diagnostic matrix using production primitives."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if not windows_ms or any(value <= 0 for value in windows_ms):
        raise ValueError("windows_ms must contain positive values")
    if not latencies_ms or any(value < 0 for value in latencies_ms):
        raise ValueError("latencies_ms must contain non-negative values")

    path = Path(config_path)
    config = load_config(path)
    cases = tuple(
        _run_case(
            config,
            scenario,
            window_ms=window_ms,
            video_latency_ms=latency_ms,
            fps=fps,
        )
        for scenario in _SCENARIOS
        for window_ms in windows_ms
        for latency_ms in latencies_ms
    )
    findings = _build_findings(cases)
    turret = config.turret
    tracking = config.tracking_control
    return TrackingLatencyBenchmarkResult(
        status="COMPLETE",
        config_path=str(path),
        fps=fps,
        prediction_horizon_ms=tracking.prediction.horizon_ms,
        prediction_enabled=tracking.prediction.enabled,
        feed_forward_enabled=tracking.feed_forward.enabled,
        ff_gain_x=tracking.feed_forward.gain_x,
        pid_kp_x=turret.controller.pid_kp_x,
        pid_ki_x=turret.controller.pid_ki_x,
        pid_kd_x=turret.controller.pid_kd_x,
        max_speed_x_deg_s=turret.stm32.max_speed_x_deg_s,
        acceleration_x_deg_s2=turret.stm32.acceleration_x_deg_s2,
        windows_ms=windows_ms,
        latencies_ms=latencies_ms,
        scenarios=tuple(scenario.name for scenario in _SCENARIOS),
        cases=cases,
        findings=findings,
    )


def write_tracking_latency_benchmark_json(
    result: TrackingLatencyBenchmarkResult, path: str | Path
) -> Path:
    output = Path(path)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "TrackingLatencyBenchmarkResult",
    "TrackingLatencyCaseResult",
    "TrackingLatencyFindings",
    "TrackingLatencyPeakSample",
    "TrackingLatencyScenario",
    "run_tracking_latency_benchmark",
    "write_tracking_latency_benchmark_json",
]
