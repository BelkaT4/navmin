#!/usr/bin/env python3
"""Run the TSP-2R sender-restart RTP identity/continuity proof."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from navmin.diagnostics.timestamp_transport import (
    DEFAULT_TOLERANCE_NS,
    TimestampTransportProofError,
    run_timestamp_sender_restart_proof,
    write_timestamp_sender_restart_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random sender restart with controlled RTP SSRC/sequence/timestamp "
            "continuity while keeping the candidate receiver alive."
        )
    )
    parser.add_argument("--port", type=int, default=18_918, help="first localhost UDP port")
    parser.add_argument(
        "--tolerance-us",
        type=int,
        default=DEFAULT_TOLERANCE_NS // 1_000,
        help="maximum allowed relative-timing error in microseconds",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.tolerance_us < 0:
        print("TIMESTAMP SENDER RESTART PROOF: FAIL — tolerance must be >= 0", file=sys.stderr)
        return 2
    try:
        result = run_timestamp_sender_restart_proof(
            base_port=args.port,
            tolerance_ns=args.tolerance_us * 1_000,
        )
    except (TimestampTransportProofError, ValueError) as exc:
        print(f"TIMESTAMP SENDER RESTART PROOF: BLOCKED — {exc}", file=sys.stderr)
        return 2

    print(f"TIMESTAMP SENDER RESTART PROOF: {result.status}")
    print(f"GStreamer: {result.gstreamer_version}")
    for variant in result.variants:
        print(f"VARIANT {variant.name}")
        print(f"  controls: {variant.second_sender_controls}")
        print(f"  first identity: {variant.first_sender_identity}")
        print(f"  second identity: {variant.second_sender_identity}")
        print(f"  first output PTS ns: {list(variant.first_output_pts_ns)}")
        print(f"  second output PTS ns: {list(variant.second_output_pts_ns)}")
        print(f"  second received all: {variant.second_received_all_frames}")
        print(
            "  second relative timing preserved: "
            f"{variant.second_relative_timing_preserved}"
        )
        print(f"  jitter before: {variant.jitter_stats_before_restart}")
        print(f"  jitter after:  {variant.jitter_stats_after_restart}")
    print(f"Conclusion: {result.conclusion}")
    if args.output_json is not None:
        write_timestamp_sender_restart_json(result, args.output_json)
        print(f"JSON: {args.output_json}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
