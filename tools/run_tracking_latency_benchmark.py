#!/usr/bin/env python3
"""Run Stage B synthetic tracking/platform-motion latency benchmark."""

from __future__ import annotations

import argparse

from navmin.diagnostics.tracking_latency_benchmark import (
    run_tracking_latency_benchmark,
    write_tracking_latency_benchmark_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/navmin.json")
    parser.add_argument("--output-json", default="tracking_latency_stageB.json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_tracking_latency_benchmark(args.config)
    findings = result.findings
    print("TRACKING LATENCY BENCHMARK: COMPLETE")
    print(
        "  runtime config: "
        f"accel_x={result.acceleration_x_deg_s2:.1f} deg/s^2 "
        f"prediction={result.prediction_enabled} "
        f"horizon={result.prediction_horizon_ms} ms "
        f"FF={result.feed_forward_enabled} gain_x={result.ff_gain_x:.3f} "
        f"Kp_x={result.pid_kp_x:.3f}"
    )
    print(f"  windows: {list(result.windows_ms)} ms")
    print(f"  video latency: {list(result.latencies_ms)} ms")
    print(f"  scenarios: {list(result.scenarios)}")
    print("FINDINGS")
    print(f"  hypothesis reproduced: {findings.hypothesis_reproduced}")
    print(f"  window-only bias at zero latency: {findings.window_only_bias_reproduced}")
    print(f"  latency amplifies bias: {findings.latency_amplifies_bias}")
    print(
        "  accel representative: "
        "target-rate error="
        f"{findings.representative_accel_peak_rate_error_deg_s:+.3f} deg/s, "
        "prediction error="
        f"{findings.representative_accel_peak_prediction_error_deg:+.3f} deg"
    )
    print(
        "  brake representative: "
        "target-rate error="
        f"{findings.representative_brake_peak_rate_error_deg_s:+.3f} deg/s, "
        "prediction error="
        f"{findings.representative_brake_peak_prediction_error_deg:+.3f} deg"
    )
    print(
        "  target-rate error direction reversal accel/brake: "
        f"{findings.reversal_of_rate_error_direction_reproduced}"
    )
    print(f"  prediction effect active in this config: {findings.prediction_effect_active}")
    print(
        "  prediction direction reversal accel/brake: "
        f"{findings.reversal_of_prediction_direction_reproduced}"
    )
    print(
        "  100 ms window at regular 20 fps never reaches valid: "
        f"{findings.regular_20fps_window_100ms_never_valid}"
    )
    output = write_tracking_latency_benchmark_json(result, args.output_json)
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
