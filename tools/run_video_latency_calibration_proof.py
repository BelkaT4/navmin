#!/usr/bin/env python3
"""Run Stage D synthetic optical-flow video-latency calibration proof."""

from __future__ import annotations

import argparse
from pathlib import Path

from navmin.diagnostics.video_latency_calibration import (
    run_video_latency_calibration_proof,
    write_video_latency_calibration_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional JSON result path",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_video_latency_calibration_proof()
    print(f"VIDEO LATENCY CALIBRATION PROOF: {result.status}")
    print(f"  fps: {result.fps}")
    print(
        "  search: "
        f"{result.search_range_ms[0]}..{result.search_range_ms[1]} ms "
        f"step={result.search_step_ms} ms"
    )
    print("RUNS")
    for run in result.runs:
        print(
            "  "
            f"known={run.known_latency_ms:3d} ms "
            f"estimated={run.estimated_latency_ms!s:>3} ms "
            f"abs_error={run.absolute_error_ms!s:>2} ms "
            f"corr={run.best_correlation!s} "
            f"rmse={run.best_rmse_deg_s!s} "
            f"flow_samples={run.flow_observations}"
        )
    aggregate = result.aggregate
    print("AGGREGATE")
    print(f"  valid runs: {aggregate.valid_run_count}")
    print(f"  mean latency: {aggregate.mean_latency_ms} ms")
    print(f"  median latency: {aggregate.median_latency_ms} ms")
    print(f"  stddev: {aggregate.stddev_latency_ms} ms")
    print(f"  MAD: {aggregate.mad_latency_ms} ms")
    print(
        "  insufficient excitation rejected: "
        f"{result.insufficient_excitation_rejected}"
    )
    if args.output_json is not None:
        path = write_video_latency_calibration_json(result, args.output_json)
        print(f"JSON: {path}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
