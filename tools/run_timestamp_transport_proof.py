#!/usr/bin/env python3
"""Run the synthetic TSP-1 RTP/JPEG timestamp transport proof."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from navmin.diagnostics.timestamp_transport import (
    DEFAULT_TOLERANCE_NS,
    TimestampTransportProofError,
    run_timestamp_transport_proof,
    write_timestamp_transport_proof_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prove whether known synthetic Gst.Buffer PTS deltas survive the "
            "candidate RTP/JPEG + jitterbuffer + decode chain."
        )
    )
    parser.add_argument("--port", type=int, default=18_898, help="first localhost UDP port")
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
        print("TIMESTAMP TRANSPORT PROOF: FAIL — tolerance must be >= 0", file=sys.stderr)
        return 2
    try:
        result = run_timestamp_transport_proof(
            base_port=args.port,
            tolerance_ns=args.tolerance_us * 1_000,
        )
    except (TimestampTransportProofError, ValueError) as exc:
        print(f"TIMESTAMP TRANSPORT PROOF: BLOCKED — {exc}", file=sys.stderr)
        return 2

    print(f"TIMESTAMP TRANSPORT PROOF: {result.status}")
    print(f"GStreamer: {result.gstreamer_version}")
    print(f"Tolerance: {result.tolerance_ns / 1_000:.0f} us")
    for case in result.cases:
        assessment = case.assessment
        print(f"CASE {case.name}: {'PASS' if case.passed else 'FAIL'}")
        print(f"  input PTS ns:  {list(assessment.input_pts_ns)}")
        print(f"  output PTS ns: {list(assessment.output_pts_ns)}")
        print(f"  delta errors ns: {list(assessment.delta_errors_ns)}")
        print(f"  offsets ns: {list(assessment.per_frame_offsets_ns)}")
        print(f"  max |delta error| ns: {assessment.max_abs_delta_error_ns}")
        print(f"  offset span ns: {assessment.offset_span_ns}")
        print(f"  origin preserved: {assessment.origin_preserved}")
    print(f"Conclusion: {result.conclusion}")

    if args.output_json is not None:
        write_timestamp_transport_proof_json(result, args.output_json)
        print(f"JSON: {args.output_json}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
