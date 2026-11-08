#!/usr/bin/env python3
"""Run TSP-3R: PC receiver-pipeline recovery with an unchanged sender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from navmin.diagnostics.timestamp_transport import (
    TimestampTransportProofError,
    run_timestamp_receiver_recovery_proof,
    write_timestamp_receiver_recovery_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the second RTP/JPEG sender alive, replace only the PC receiver pipeline, "
            "and verify that source-relative PTS timing becomes coherent again."
        )
    )
    parser.add_argument("--port", type=int, default=18_938, help="localhost UDP port")
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run_timestamp_receiver_recovery_proof(port=args.port)
    except (TimestampTransportProofError, ValueError) as exc:
        print(f"TIMESTAMP RECEIVER RECOVERY PROOF: BLOCKED — {exc}", file=sys.stderr)
        return 2

    print(f"TIMESTAMP RECEIVER RECOVERY PROOF: {result.status}")
    print(f"GStreamer: {result.gstreamer_version}")
    print(f"Tolerance ns: {result.tolerance_ns}")
    print("SESSION 1")
    print(f"  source PTS ns: {list(result.first_source_pts_ns)}")
    print(f"  output PTS ns: {list(result.first_output_pts_ns)}")
    print("SESSION 2 THROUGH OLD RECEIVER")
    print(f"  source PTS ns: {list(result.second_source_before_reset_pts_ns)}")
    print(f"  output PTS ns: {list(result.second_output_before_reset_pts_ns)}")
    print(f"  received all: {result.second_before_reset_received_all_frames}")
    print(
        "  relative timing preserved: "
        f"{result.second_before_reset_relative_timing_preserved}"
    )
    print("PC RECEIVER RECREATED; SAME SENDER SESSION 2 KEPT RUNNING")
    print(f"  recovery source PTS ns: {list(result.recovery_source_pts_ns)}")
    print(f"  recovery output PTS ns: {list(result.recovery_output_pts_ns)}")
    print(f"  received all: {result.recovery_received_all_frames}")
    print(f"  relative timing preserved: {result.recovery_relative_timing_preserved}")
    print(f"  same sender SSRC: {result.same_second_sender_ssrc}")
    print(
        "  sender identity before reset: "
        f"{result.second_sender_identity_before_reset}"
    )
    print(
        "  sender identity after reset:  "
        f"{result.second_sender_identity_after_reset}"
    )
    print(f"Conclusion: {result.conclusion}")
    if args.output_json is not None:
        write_timestamp_receiver_recovery_json(result, args.output_json)
        print(f"JSON: {args.output_json}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
