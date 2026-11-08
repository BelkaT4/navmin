from __future__ import annotations

from pathlib import Path

import pytest

from navmin.diagnostics.tracking_latency_benchmark import (
    run_tracking_latency_benchmark,
    write_tracking_latency_benchmark_json,
)


def _case(result, scenario: str, window_ms: int, latency_ms: int):
    return next(
        case
        for case in result.cases
        if case.scenario == scenario
        and case.window_ms == window_ms
        and case.video_latency_ms == latency_ms
    )


@pytest.fixture(scope="module")
def benchmark():
    return run_tracking_latency_benchmark()


def test_stationary_target_reproduces_window_and_latency_bias(benchmark) -> None:
    zero = _case(benchmark, "stationary_target_accelerate", 200, 0)
    delayed = _case(benchmark, "stationary_target_accelerate", 200, 150)

    assert zero.peak_sample is not None
    assert delayed.peak_sample is not None
    assert zero.peak_abs_target_rate_error_deg_s is not None
    assert zero.peak_abs_target_rate_error_deg_s > 0.05 * benchmark.acceleration_x_deg_s2
    assert zero.peak_abs_latency_component_deg_s == pytest.approx(0.0, abs=1e-9)
    assert zero.peak_abs_window_component_deg_s is not None
    assert zero.peak_abs_window_component_deg_s > 10.0

    assert delayed.peak_abs_target_rate_error_deg_s is not None
    assert delayed.peak_abs_target_rate_error_deg_s > 30.0
    assert (
        delayed.peak_abs_target_rate_error_deg_s
        > zero.peak_abs_target_rate_error_deg_s
    )
    assert delayed.peak_abs_latency_component_deg_s is not None
    assert delayed.peak_abs_latency_component_deg_s > 0.10 * benchmark.acceleration_x_deg_s2
    if benchmark.findings.prediction_effect_active:
        assert delayed.peak_sample.prediction_error_deg > 0.0
    else:
        assert delayed.peak_sample.prediction_error_deg == pytest.approx(0.0)


def test_braking_reverses_false_rate_direction(benchmark) -> None:
    accelerate = _case(benchmark, "stationary_target_accelerate", 200, 150)
    brake = _case(benchmark, "stationary_target_brake", 200, 150)

    assert accelerate.peak_sample is not None
    assert brake.peak_sample is not None
    assert accelerate.peak_sample.target_rate_error_deg_s > 0.0
    assert brake.peak_sample.target_rate_error_deg_s < 0.0
    assert benchmark.findings.reversal_of_rate_error_direction_reproduced

    if benchmark.findings.prediction_effect_active:
        assert accelerate.peak_sample.prediction_error_deg > 0.0
        assert brake.peak_sample.prediction_error_deg < 0.0
        assert benchmark.findings.reversal_of_prediction_direction_reproduced
    else:
        assert accelerate.peak_sample.prediction_error_deg == pytest.approx(0.0)
        assert brake.peak_sample.prediction_error_deg == pytest.approx(0.0)
        assert not benchmark.findings.reversal_of_prediction_direction_reproduced


def test_window_matched_oracle_removes_dynamic_compensation_bias(benchmark) -> None:
    for scenario in ("stationary_target_accelerate", "moving_target_accelerate"):
        case = _case(benchmark, scenario, 200, 150)
        assert case.peak_abs_window_matched_target_rate_error_deg_s is not None
        assert case.peak_abs_window_matched_target_rate_error_deg_s < 0.02


def test_regular_100ms_window_cannot_meet_four_samples_at_20fps(benchmark) -> None:
    cases = [
        case
        for case in benchmark.cases
        if case.scenario == "stationary_target_accelerate" and case.window_ms == 100
    ]

    assert cases
    assert all(case.valid_frame_count == 0 for case in cases)
    assert all(
        case.invalid_reasons.get("insufficient_samples", 0) > 0 for case in cases
    )
    assert benchmark.findings.regular_20fps_window_100ms_never_valid


def test_jitter_and_drops_still_reproduce_spurious_rate(benchmark) -> None:
    case = _case(benchmark, "stationary_target_accelerate_jitter_drop", 200, 150)

    assert case.valid_frame_count > 0
    assert case.peak_abs_target_rate_error_deg_s is not None
    assert case.peak_abs_target_rate_error_deg_s > 20.0
    assert case.validity_transition_count > 2
    assert case.valid_to_invalid_count > 0
    assert case.invalid_to_valid_count > 0


def test_result_json_is_machine_readable(benchmark, tmp_path: Path) -> None:
    output = tmp_path / "tracking-latency.json"

    write_tracking_latency_benchmark_json(benchmark, output)

    text = output.read_text(encoding="utf-8")
    assert '"status": "COMPLETE"' in text
    assert '"hypothesis_reproduced": true' in text
    assert '"window_only_bias_reproduced": true' in text
    assert '"stationary_target_reverse"' in text
