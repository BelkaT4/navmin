#!/usr/bin/env python3
"""Run Stage C synthetic target-selection command-discontinuity benchmark."""

from __future__ import annotations

import argparse

from navmin.diagnostics.tracking_selection_benchmark import (
    run_tracking_selection_benchmark,
    write_tracking_selection_benchmark_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/navmin.json")
    parser.add_argument("--output-json", default="tracking_selection_stageC.json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_tracking_selection_benchmark(args.config)
    findings = result.findings
    print("TRACKING SELECTION BENCHMARK: COMPLETE")
    print(
        "  runtime config: "
        f"prediction={result.runtime_prediction_enabled} "
        f"horizon={result.runtime_prediction_horizon_ms} ms "
        f"FF={result.runtime_feed_forward_enabled} "
        f"gain_x={result.runtime_ff_gain_x:.3f}"
    )
    print(
        "  active-effects probe: "
        f"horizon={result.probe_prediction_horizon_ms} ms "
        f"FF gain_x={result.probe_ff_gain_x:.3f}"
    )
    print("FINDINGS")
    print(
        "  first selection has no STOP boundary: "
        f"{findings.first_selection_has_no_stop_boundary}"
    )
    print(
        "  replacement inserts explicit zero STOP: "
        f"{findings.replacement_has_explicit_zero_stop_boundary}"
    )
    print(
        "  replacement STOP interrupts nonzero command: "
        f"{findings.replacement_stop_interrupts_nonzero_command}"
    )
    print(
        "  new target restarts rate-invalid P-only: "
        f"{findings.replacement_restarts_with_rate_invalid_p_only}"
    )
    print(
        "  estimator rewarms after replacement: "
        f"{findings.rate_validity_rewarms_after_replacement}"
    )
    print(
        "  active prediction/FF enter at first VALID boundary: "
        f"{findings.active_effects_enter_at_valid_boundary}"
    )
    print(
        "  active effects increase first-VALID command jump: "
        f"{findings.active_effects_increase_first_valid_jump}"
    )
    print("COMMAND JUMPS")
    print(
        "  runtime first-selection initial: "
        f"{findings.runtime_first_selection_initial_jump_deg_s:.3f} deg/s"
    )
    print(
        "  runtime replacement old->STOP: "
        f"{findings.runtime_replacement_stop_jump_deg_s:.3f} deg/s"
    )
    print(
        "  runtime replacement STOP->new P-only: "
        f"{findings.runtime_replacement_acquisition_jump_deg_s:.3f} deg/s"
    )
    print(
        "  runtime replacement first VALID adjacent jump: "
        f"{findings.runtime_replacement_first_valid_jump_deg_s:.3f} deg/s"
    )
    print(
        "  effects-off probe first VALID adjacent jump: "
        f"{findings.probe_off_replacement_first_valid_jump_deg_s:.3f} deg/s"
    )
    print(
        "  active-effects probe first VALID adjacent jump: "
        f"{findings.probe_replacement_first_valid_jump_deg_s:.3f} deg/s"
    )
    print(
        "  probe rate-effect contribution at first VALID: "
        f"{findings.probe_replacement_rate_effect_deg_s:+.3f} deg/s"
    )
    output = write_tracking_selection_benchmark_json(result, args.output_json)
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
