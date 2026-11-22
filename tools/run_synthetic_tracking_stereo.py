#!/usr/bin/env python3
"""Run the deterministic synthetic tracking stereo proof."""

from __future__ import annotations

import argparse

from navmin.diagnostics.synthetic_tracking_stereo import (
    run_synthetic_tracking_stereo_diagnostic,
    write_synthetic_tracking_stereo_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-pair + VT11 join + SGBM over the synthetic tracking world."
    )
    parser.add_argument(
        "--stereo-calibration",
        default=None,
        help="Optional Stereo calibration JSON; default uses canonical SIM geometry.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Frame count; default is one complete synthetic world cycle.",
    )
    parser.add_argument("--selection-frame", type=int, default=8)
    parser.add_argument(
        "--distance-stride",
        type=int,
        default=100,
        help="Run production SGBM every Nth joined target observation.",
    )
    parser.add_argument("--json", default=None, help="Optional JSON result path.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_synthetic_tracking_stereo_diagnostic(
        stereo_calibration_path=args.stereo_calibration,
        frame_count=args.frames,
        selection_frame=args.selection_frame,
        distance_stride=args.distance_stride,
    )
    print(
        "SYNTHETIC TRACKING STEREO "
        f"status={result.status} frames={result.frames} "
        f"acquired={result.target_acquired_frame} ended={result.target_end_frame} "
        f"exact_pairs={result.exact_pairs} observations={result.target_observations} "
        f"distance={result.distance_successes}/{result.distance_attempts} "
        f"mean_abs_error_m={result.mean_abs_distance_error_m}"
    )
    for sample in result.samples:
        print(
            f"frame={sample.frame_index:4d} phase={sample.phase:22s} "
            f"distance={sample.measured_distance_m:.3f}m "
            f"error={sample.distance_error_m:+.3f}m "
            f"disp={sample.measured_disparity_px:.2f}px"
        )
    if args.json is not None:
        write_synthetic_tracking_stereo_json(result, args.json)
        print(f"JSON: {args.json}")
    return 0 if result.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
