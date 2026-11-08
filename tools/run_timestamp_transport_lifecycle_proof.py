#!/usr/bin/env python3
"""Run the synthetic TSP-2 RTP/JPEG timestamp lifecycle proof."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from navmin.diagnostics.timestamp_transport import (
    DEFAULT_TOLERANCE_NS,
    TimestampTransportProofError,
    run_timestamp_transport_lifecycle_proof,
    write_timestamp_transport_lifecycle_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe RTP/JPEG timestamp behavior across gaps, 90 kHz wrap, large PTS, "
            "duplicate/backward PTS, and sender/receiver restarts."
        )
    )
    parser.add_argument("--port", type=int, default=18_908, help="first localhost UDP port")
    parser.add_argument(
        "--tolerance-us",
        type=int,
        default=DEFAULT_TOLERANCE_NS // 1_000,
        help="maximum allowed delta/offset variation in microseconds",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.tolerance_us < 0:
        print("TIMESTAMP LIFECYCLE PROOF: FAIL — tolerance must be >= 0", file=sys.stderr)
        return 2
    try:
        result = run_timestamp_transport_lifecycle_proof(
            base_port=args.port,
            tolerance_ns=args.tolerance_us * 1_000,
        )
    except (TimestampTransportProofError, ValueError) as exc:
        print(f"TIMESTAMP LIFECYCLE PROOF: BLOCKED — {exc}", file=sys.stderr)
        return 2

    print(f"TIMESTAMP LIFECYCLE PROOF: {result.status}")
    print(f"GStreamer: {result.gstreamer_version}")
    print(f"Tolerance: {result.tolerance_ns / 1_000:.0f} us")
    for case in result.deterministic_cases:
        assessment = case.assessment
        print(f"CASE {case.name}: {'PASS' if case.passed else 'FAIL'}")
        print(f"  input PTS ns:  {list(assessment.input_pts_ns)}")
        print(f"  output PTS ns: {list(assessment.output_pts_ns)}")
        print(f"  delta errors ns: {list(assessment.delta_errors_ns)}")
        print(f"  offset span ns: {assessment.offset_span_ns}")
        print(f"  origin preserved: {assessment.origin_preserved}")
    for case in result.observation_cases:
        print(f"OBSERVATION {case.name}")
        print(f"  input PTS ns:  {list(case.input_pts_ns)}")
        print(f"  output PTS ns: {list(case.output_pts_ns)}")
        print(f"  received all frames: {case.received_all_frames}")
        print(f"  output monotonic: {case.output_monotonic}")
        print(f"  note: {case.note}")
    for case in result.restart_cases:
        print(f"RESTART {case.name}: {'PASS' if case.passed else 'FAIL'}")
        print(f"  first output PTS ns: {list(case.first_assessment.output_pts_ns)}")
        print(f"  second output PTS ns: {list(case.second_assessment.output_pts_ns)}")
        print(
            "  second session origin preserved: "
            f"{case.second_session_origin_preserved}"
        )
        print(f"  note: {case.note}")
    print("Limitations:")
    for limitation in result.limitations:
        print(f"  - {limitation}")
    print(f"Conclusion: {result.conclusion}")

    if args.output_json is not None:
        write_timestamp_transport_lifecycle_json(result, args.output_json)
        print(f"JSON: {args.output_json}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
