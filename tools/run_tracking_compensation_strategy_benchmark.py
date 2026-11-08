#!/usr/bin/env python3
"""Run Stage E synthetic tracking compensation strategy benchmark."""

from __future__ import annotations

import argparse

from navmin.diagnostics.tracking_compensation_strategy_benchmark import (
    run_tracking_compensation_strategy_benchmark,
    write_tracking_compensation_strategy_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/navmin.json")
    parser.add_argument("--output-json", default="tracking_compensation_stageE.json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_tracking_compensation_strategy_benchmark(args.config)
    findings = result.findings
    print("TRACKING COMPENSATION STRATEGY BENCHMARK: COMPLETE")
    print(
        "  runtime basis: "
        f"accel_x={result.acceleration_x_deg_s2:.1f} deg/s^2 "
        f"Kp_x={result.pid_kp_x:.3f} fps={result.fps}"
    )
    print(f"  windows: {list(result.windows_ms)} ms")
    print(f"  true video latency: {list(result.true_latencies_ms)} ms")
    print(f"  calibration errors: {list(result.calibration_errors_ms)} ms")
    print(f"  strategies: {list(result.strategies)}")
    print(
        "  diagnostic effects probe: "
        f"prediction={result.probe_prediction_horizon_ms} ms "
        f"max={result.probe_prediction_max_angle_deg:.1f} deg "
        f"FF gain={result.probe_ff_gain:.2f}"
    )
    print(
        "REPRESENTATIVE stationary target / acceleration / "
        "200 ms window / 150 ms latency"
    )
    print(
        "  current: "
        f"{findings.representative_current_peak_error_deg_s:.3f} deg/s peak error"
    )
    print(
        "  latency-corrected only: "
        f"{findings.representative_latency_corrected_peak_error_deg_s:.3f} deg/s"
    )
    print(
        "  window-aligned only: "
        f"{findings.representative_window_aligned_peak_error_deg_s:.3f} deg/s"
    )
    print(
        "  latency-corrected + window-aligned: "
        f"{findings.representative_combined_peak_error_deg_s:.3f} deg/s"
    )
    print(
        "  combined reduction vs current: "
        f"{findings.representative_combined_reduction_percent:.2f}%"
    )
    print("CALIBRATION SENSITIVITY — combined strategy, same representative case")
    print(
        "  -50/-25/-10 ms: "
        f"{findings.combined_error_at_minus_50ms_deg_s:.3f} / "
        f"{findings.combined_error_at_minus_25ms_deg_s:.3f} / "
        f"{findings.combined_error_at_minus_10ms_deg_s:.3f} deg/s"
    )
    print(
        "  +10/+25/+50 ms: "
        f"{findings.combined_error_at_plus_10ms_deg_s:.3f} / "
        f"{findings.combined_error_at_plus_25ms_deg_s:.3f} / "
        f"{findings.combined_error_at_plus_50ms_deg_s:.3f} deg/s"
    )
    print("FINDINGS")
    print(
        "  latency correction alone leaves window bias: "
        f"{findings.latency_correction_alone_leaves_window_bias}"
    )
    print(
        "  window alignment alone leaves latency bias: "
        f"{findings.window_alignment_alone_leaves_latency_bias}"
    )
    print(
        "  combined outperforms both partial strategies: "
        f"{findings.combined_outperforms_partial_strategies}"
    )
    print(
        "  combined removes accel/brake false-sign reversal with perfect calibration: "
        f"{findings.combined_removes_reversal_with_perfect_calibration}"
    )
    print(
        "  combined with ±25 ms calibration error still beats current: "
        f"{findings.combined_with_25ms_error_beats_current}"
    )
    print(f"  candidate for production evaluation: {findings.recommended_candidate}")
    output = write_tracking_compensation_strategy_json(result, args.output_json)
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
