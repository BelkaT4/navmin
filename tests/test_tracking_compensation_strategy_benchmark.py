from __future__ import annotations

from pathlib import Path

import pytest

from navmin.diagnostics.tracking_compensation_strategy_benchmark import (
    COMBINED,
    CURRENT,
    LATENCY_CORRECTED,
    WINDOW_ALIGNED,
    run_tracking_compensation_strategy_benchmark,
    write_tracking_compensation_strategy_json,
)


@pytest.fixture(scope="module")
def benchmark():
    return run_tracking_compensation_strategy_benchmark()


def _case(
    benchmark,
    scenario: str,
    calibration_error_ms: int = 0,
    *,
    window_ms: int = 200,
    true_latency_ms: int = 150,
):
    return next(
        case
        for case in benchmark.cases
        if case.scenario == scenario
        and case.window_ms == window_ms
        and case.true_video_latency_ms == true_latency_ms
        and case.calibration_error_ms == calibration_error_ms
    )


def _strategy(case, name: str):
    return next(result for result in case.strategies if result.strategy == name)


def _error(case, name: str) -> float:
    value = _strategy(case, name).peak_abs_target_rate_error_deg_s
    assert value is not None
    return value


def test_combined_alignment_removes_both_dynamic_biases(benchmark) -> None:
    case = _case(benchmark, "stationary_target_accelerate")
    current = _error(case, CURRENT)
    latency_only = _error(case, LATENCY_CORRECTED)
    window_only = _error(case, WINDOW_ALIGNED)
    combined = _error(case, COMBINED)

    assert current > 0.10 * benchmark.acceleration_x_deg_s2
    assert latency_only > 1.0
    assert window_only > 1.0
    assert combined < 0.05
    assert combined < latency_only
    assert combined < window_only
    assert benchmark.findings.combined_outperforms_partial_strategies


def test_combined_alignment_recovers_moving_target_rate(benchmark) -> None:
    case = _case(benchmark, "moving_target_accelerate")
    combined = _strategy(case, COMBINED)

    assert combined.valid_frame_count > 0
    assert combined.peak_abs_target_rate_error_deg_s is not None
    assert combined.peak_abs_target_rate_error_deg_s < 0.05
    assert combined.wrong_direction_frame_count == 0


def test_combined_removes_accel_brake_false_sign_reversal(benchmark) -> None:
    accelerate = _case(benchmark, "stationary_target_accelerate")
    brake = _case(benchmark, "stationary_target_brake")
    current_accel = _strategy(accelerate, CURRENT)
    current_brake = _strategy(brake, CURRENT)
    combined_accel = _strategy(accelerate, COMBINED)
    combined_brake = _strategy(brake, COMBINED)

    assert current_accel.peak_sample is not None
    assert current_brake.peak_sample is not None
    assert current_accel.peak_sample.target_rate_error_deg_s > 0.0
    assert current_brake.peak_sample.target_rate_error_deg_s < 0.0
    assert combined_accel.peak_abs_target_rate_error_deg_s is not None
    assert combined_brake.peak_abs_target_rate_error_deg_s is not None
    assert combined_accel.peak_abs_target_rate_error_deg_s < 0.05
    assert combined_brake.peak_abs_target_rate_error_deg_s < 0.05
    assert benchmark.findings.current_reverses_error_direction
    assert benchmark.findings.combined_removes_reversal_with_perfect_calibration


def test_combined_calibration_error_degrades_gradually(benchmark) -> None:
    minus_10 = _error(_case(benchmark, "stationary_target_accelerate", -10), COMBINED)
    minus_25 = _error(_case(benchmark, "stationary_target_accelerate", -25), COMBINED)
    minus_50 = _error(_case(benchmark, "stationary_target_accelerate", -50), COMBINED)
    plus_10 = _error(_case(benchmark, "stationary_target_accelerate", 10), COMBINED)
    plus_25 = _error(_case(benchmark, "stationary_target_accelerate", 25), COMBINED)
    plus_50 = _error(_case(benchmark, "stationary_target_accelerate", 50), COMBINED)
    current = _error(_case(benchmark, "stationary_target_accelerate"), CURRENT)

    assert minus_10 < minus_25 < minus_50
    assert plus_10 < plus_25 < plus_50
    assert max(minus_25, plus_25) < current
    assert benchmark.findings.combined_with_25ms_error_beats_current


def test_combined_alignment_survives_jitter_and_drops(benchmark) -> None:
    case = _case(benchmark, "stationary_target_accelerate_jitter_drop")
    combined = _strategy(case, COMBINED)

    assert combined.valid_frame_count > 0
    assert combined.peak_abs_target_rate_error_deg_s is not None
    assert combined.peak_abs_target_rate_error_deg_s < 0.05


def test_strategy_probe_exposes_prediction_and_ff_consequence(benchmark) -> None:
    case = _case(benchmark, "stationary_target_accelerate")
    current = _strategy(case, CURRENT)
    combined = _strategy(case, COMBINED)

    assert current.peak_abs_probe_prediction_error_deg is not None
    assert current.peak_abs_probe_rate_effect_command_error_deg_s is not None
    assert combined.peak_abs_probe_prediction_error_deg is not None
    assert combined.peak_abs_probe_rate_effect_command_error_deg_s is not None
    assert current.peak_abs_probe_prediction_error_deg > 1.0
    assert current.peak_abs_probe_rate_effect_command_error_deg_s > 1.0
    assert combined.peak_abs_probe_prediction_error_deg < 0.05
    assert combined.peak_abs_probe_rate_effect_command_error_deg_s < 0.05


def test_result_json_is_machine_readable(benchmark, tmp_path: Path) -> None:
    output = tmp_path / "tracking-compensation.json"

    write_tracking_compensation_strategy_json(benchmark, output)

    text = output.read_text(encoding="utf-8")
    assert '"status": "COMPLETE"' in text
    assert '"recommended_candidate": "latency-corrected-window-aligned"' in text
    assert '"combined_outperforms_partial_strategies": true' in text
