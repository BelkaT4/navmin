#!/usr/bin/env python3
"""Run Stage A integrated synthetic camera-time tracking proof."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from navmin.diagnostics.tracking_time_integration import (
    TrackingTimeIntegrationProofError,
    run_tracking_time_integration_proof,
    write_tracking_time_integration_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise synthetic sensor RTP PTS through the production RTP/JPEG source "
            "contract, ClockMapper ownership, tracking-time OFF/ON selection, and "
            "generation restart semantics."
        )
    )
    parser.add_argument("--port", type=int, default=18_948)
    parser.add_argument("--skip-gstreamer", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.port <= 65_535:
        print("TRACKING TIME INTEGRATION PROOF: FAIL — invalid UDP port", file=sys.stderr)
        return 2
    try:
        result = run_tracking_time_integration_proof(
            include_gstreamer=not args.skip_gstreamer,
            port=args.port,
        )
    except (TrackingTimeIntegrationProofError, ValueError) as exc:
        print(f"TRACKING TIME INTEGRATION PROOF: FAIL — {exc}", file=sys.stderr)
        return 1

    deterministic = result.deterministic
    print(f"DETERMINISTIC PRODUCTION-PATH PROOF: {deterministic.status}")
    print(f"  source timestamp: {deterministic.source_timestamp_type}")
    print(f"  generation 1 states: {list(deterministic.first_generation.mapper_states)}")
    print(f"  generation 2 states: {list(deterministic.second_generation.mapper_states)}")
    print(
        "  OFF uses receive time: "
        f"{deterministic.off_observation_timestamp_ns == deterministic.off_receive_timestamp_ns}"
    )
    print(
        "  ON uses mapped time: "
        f"{deterministic.on_observation_timestamp_ns == deterministic.on_mapped_timestamp_ns}"
    )
    print(
        "  restart accepts reset PTS: "
        f"{deterministic.source_restart_accepted_reset_pts}"
    )

    if args.skip_gstreamer:
        print("REAL GSTREAMER LOCALHOST PROOF: SKIPPED")
    elif result.gstreamer is None:
        print(f"REAL GSTREAMER LOCALHOST PROOF: BLOCKED — {result.gstreamer_blocked_reason}")
    else:
        print(f"REAL GSTREAMER LOCALHOST PROOF: {result.gstreamer.status}")
        print(f"  mapper states: {list(result.gstreamer.mapper_states)}")

    if args.output_json is not None:
        write_tracking_time_integration_json(result, args.output_json)
        print(f"JSON: {args.output_json}")

    if not deterministic.passed:
        return 1
    if not args.skip_gstreamer and result.gstreamer is not None and not result.gstreamer.passed:
        return 1
    # Missing multimedia runtime is an explicit environmental BLOCKED result, not
    # a false functional failure of the deterministic Stage A proof.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
