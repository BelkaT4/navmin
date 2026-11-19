#!/usr/bin/env python3
"""Run Stage F production tracking-time compensation proof."""

from __future__ import annotations

import argparse

from navmin.diagnostics.tracking_time_compensation_proof import (
    run_production_tracking_time_compensation_proof,
    write_production_tracking_time_compensation_proof_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--acceleration-x-deg-s2", type=float, default=150.0)
    parser.add_argument("--true-video-latency-ms", type=int, default=150)
    parser.add_argument("--window-ms", type=int, default=200)
    args = parser.parse_args()

    result = run_production_tracking_time_compensation_proof(
        acceleration_x_deg_s2=args.acceleration_x_deg_s2,
        true_video_latency_ms=args.true_video_latency_ms,
        window_ms=args.window_ms,
    )
    print(f"PRODUCTION TRACKING TIME COMPENSATION PROOF: {result.status}")
    print(
        "  basis: "
        f"accel_x={result.acceleration_x_deg_s2:.1f} deg/s^2 "
        f"window={result.window_ms} ms fps={result.fps}"
    )
    print("PERFECT CALIBRATION")
    for case in result.cases:
        print(
            f"  {case.scenario}: "
            f"peak_error={case.peak_abs_target_rate_error_deg_s:.9f} deg/s "
            f"valid_frames={case.valid_frame_count}"
        )
    print("CALIBRATION ROBUSTNESS")
    print(f"  -25 ms: {result.minus_25ms_peak_error_deg_s:.3f} deg/s")
    print(f"  +25 ms: {result.plus_25ms_peak_error_deg_s:.3f} deg/s")
    print(
        "  zero-latency window alignment: "
        f"{result.zero_latency_window_aligned_peak_error_deg_s:.9f} deg/s"
    )
    print(f"  invalid platform safe: {result.invalid_platform_is_safe}")
    if args.output_json:
        path = write_production_tracking_time_compensation_proof_json(
            result, args.output_json
        )
        print(f"JSON: {path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
