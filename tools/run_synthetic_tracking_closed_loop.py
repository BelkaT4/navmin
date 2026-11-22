#!/usr/bin/env python3
"""Run the headless synthetic closed-loop tracking diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from navmin.diagnostics.synthetic_tracking_closed_loop import (
    run_synthetic_tracking_closed_loop,
    write_synthetic_tracking_closed_loop_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/navmin.json")
    parser.add_argument("--stereo-calibration", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--selection-frame", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    result = run_synthetic_tracking_closed_loop(
        config_path=args.config,
        stereo_calibration_path=args.stereo_calibration,
        frame_count=args.frames,
        selection_frame=args.selection_frame,
        sample_stride=args.sample_stride,
    )
    print(
        "SYNTHETIC CLOSED LOOP "
        f"status={result.status} frames={result.frames} "
        f"acquired={result.target_acquired_frame} ended={result.target_end_frame} "
        f"missing={result.missing_tracking_frames} "
        f"rate_valid_x={result.rate_valid_x_frames}"
    )
    for phase in result.phase_summaries:
        print(
            f"{phase.phase:22s} frames={phase.frames:4d} "
            f"tracked={phase.tracked_frames:4d} "
            f"valid_x={phase.rate_valid_x_frames:4d} "
            f"mean|err_x|={phase.mean_abs_error_yaw_deg:.3f}deg "
            f"p95|err_x|={phase.p95_abs_error_yaw_deg:.3f}deg "
            f"mean|rate_err_x|={phase.mean_abs_target_rate_error_x_deg_s}"
        )
    if args.json is not None:
        write_synthetic_tracking_closed_loop_json(result, args.json)
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
