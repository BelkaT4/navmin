from __future__ import annotations

from pathlib import Path

import pytest

from navmin.diagnostics.tracking_selection_benchmark import (
    run_tracking_selection_benchmark,
    write_tracking_selection_benchmark_json,
)


@pytest.fixture(scope="module")
def benchmark():
    return run_tracking_selection_benchmark()


def _scenario(benchmark, policy: str, scenario: str):
    return next(
        item
        for item in benchmark.scenarios
        if item.policy == policy and item.scenario == scenario
    )


def test_first_selection_starts_without_explicit_stop_and_with_rate_invalid(benchmark) -> None:
    first = _scenario(benchmark, "runtime", "first-selection")
    tracking = [event for event in first.events if event.kind == "tracking"]

    assert first.stop_count == 0
    assert first.invalidation_count == 1
    assert tracking[0].rate_valid_x is False
    assert tracking[0].proportional_offset_x_deg == pytest.approx(0.0)
    assert tracking[0].ff_term_x_deg_s == pytest.approx(0.0)
    assert first.acquisition_jump_deg_s > 0.0
    assert benchmark.findings.first_selection_has_no_stop_boundary


def test_replacement_inserts_zero_between_old_and_new_target_commands(benchmark) -> None:
    replacement = _scenario(benchmark, "runtime", "replacement-selection")
    events = replacement.events
    stop_index = next(index for index, event in enumerate(events) if event.kind == "stop")
    new_tracking = [
        event
        for event in events[stop_index + 1 :]
        if event.kind == "tracking" and event.target_track_id == 2
    ]

    assert replacement.pre_selection_command_x_deg_s != pytest.approx(0.0)
    assert replacement.stop_count == 1
    assert replacement.post_stop_command_x_deg_s == pytest.approx(0.0)
    assert new_tracking
    assert new_tracking[0].rate_valid_x is False
    assert new_tracking[0].command_x_deg_s != pytest.approx(0.0)
    assert replacement.stop_jump_deg_s > 0.0
    assert replacement.acquisition_jump_deg_s > 0.0
    assert benchmark.findings.replacement_has_explicit_zero_stop_boundary
    assert benchmark.findings.replacement_stop_interrupts_nonzero_command
    assert benchmark.findings.replacement_restarts_with_rate_invalid_p_only


def test_replacement_requires_estimator_history_again(benchmark) -> None:
    replacement = _scenario(benchmark, "runtime", "replacement-selection")

    assert replacement.first_valid_elapsed_ms >= 100
    assert replacement.first_valid_elapsed_ms <= 200
    assert benchmark.findings.rate_validity_rewarms_after_replacement


def test_active_effects_probe_enters_prediction_and_ff_at_first_valid_boundary(benchmark) -> None:
    effects_off = _scenario(benchmark, "effects-off-probe", "replacement-selection")
    probe = _scenario(benchmark, "active-effects-probe", "replacement-selection")
    probe_tracking = [
        event
        for event in probe.events
        if event.kind == "tracking" and event.target_track_id == 2
    ]
    first_valid = next(event for event in probe_tracking if event.rate_valid_x)

    assert first_valid.proportional_offset_x_deg is not None
    assert first_valid.proportional_offset_x_deg < -0.1
    assert first_valid.ff_term_x_deg_s is not None
    assert first_valid.ff_term_x_deg_s < -0.1
    assert probe.first_valid_rate_effect_x_deg_s < -0.1
    assert (
        probe.first_valid_adjacent_jump_deg_s
        > effects_off.first_valid_adjacent_jump_deg_s + 0.5
    )
    assert benchmark.findings.active_effects_enter_at_valid_boundary
    assert benchmark.findings.active_effects_increase_first_valid_jump


def test_result_json_is_machine_readable(benchmark, tmp_path: Path) -> None:
    output = tmp_path / "tracking-selection.json"

    write_tracking_selection_benchmark_json(benchmark, output)

    text = output.read_text(encoding="utf-8")
    assert '"status": "COMPLETE"' in text
    assert '"replacement_has_explicit_zero_stop_boundary": true' in text
    assert '"active_effects_enter_at_valid_boundary": true' in text
    assert '"effects-off-probe"' in text
    assert '"active-effects-probe"' in text
