"""Bounded runtime smoke for the production H.264 RTSP decoded-frame source."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import numpy as np

from navmin.config.models import RtspDecoderMode, RtspProtocol, RtspSourceConfig
from navmin.vision.gstreamer_source import (
    RTSP_H264_GSTREAMER_ELEMENTS,
    CameraSourceError,
    GStreamerRtspSource,
    GStreamerUnavailableError,
    find_missing_gstreamer_elements,
    initialize_gstreamer_runtime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive and decode an existing H.264 RTSP stream through the production "
            "NavMin GStreamerRtspSource."
        )
    )
    parser.add_argument("--uri", required=True, help="RTSP URI, for example rtsp://host/stream")
    parser.add_argument(
        "--protocol",
        choices=tuple(item.value for item in RtspProtocol),
        default=RtspProtocol.TCP.value,
    )
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument(
        "--drop-on-latency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.latency_ms < 0:
        raise ValueError("--latency-ms must be >= 0")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if (args.expected_width is None) != (args.expected_height is None):
        raise ValueError("--expected-width and --expected-height must be provided together")
    if args.expected_width is not None and args.expected_width <= 0:
        raise ValueError("--expected-width must be > 0")
    if args.expected_height is not None and args.expected_height <= 0:
        raise ValueError("--expected-height must be > 0")


def _check_runtime() -> None:
    initialize_gstreamer_runtime()
    missing = find_missing_gstreamer_elements(RTSP_H264_GSTREAMER_ELEMENTS)
    if missing:
        raise GStreamerUnavailableError(
            "missing required GStreamer elements: " + ", ".join(missing)
        )


def _run(args: argparse.Namespace) -> tuple[int, tuple[int, int] | None]:
    config = RtspSourceConfig(
        uri=args.uri,
        protocol=RtspProtocol(args.protocol),
        decoder_mode=RtspDecoderMode.SOFTWARE,
        latency_ms=args.latency_ms,
        drop_on_latency=args.drop_on_latency,
        buffer_size=1,
    )
    source = GStreamerRtspSource(config)
    frame_count = 0
    observed_size: tuple[int, int] | None = None
    deadline = time.monotonic() + args.duration_seconds
    source.start()
    try:
        while time.monotonic() < deadline:
            failure = source.failure
            if failure is not None:
                raise CameraSourceError(f"RTSP source failed: {failure}") from failure
            decoded = source.read()
            if decoded is None:
                time.sleep(0.005)
                continue
            image = decoded.image
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise CameraSourceError(
                    f"unexpected decoded frame contract: {image.dtype} {image.shape!r}"
                )
            size = (int(image.shape[1]), int(image.shape[0]))
            if observed_size is None:
                observed_size = size
            elif size != observed_size:
                raise CameraSourceError(
                    f"decoded frame size changed during smoke: {observed_size} -> {size}"
                )
            if args.expected_width is not None:
                expected = (args.expected_width, args.expected_height)
                if size != expected:
                    raise CameraSourceError(
                        f"decoded frame size {size[0]}x{size[1]} != expected "
                        f"{expected[0]}x{expected[1]}"
                    )
            frame_count += 1
    finally:
        source.stop()
    if frame_count == 0:
        raise CameraSourceError("no decoded RTSP frames received before timeout")
    return frame_count, observed_size


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        _check_runtime()
        frame_count, size = _run(args)
    except (CameraSourceError, GStreamerUnavailableError, ValueError) as exc:
        print(f"RTSP DIAGNOSTIC: FAIL — {exc}", file=sys.stderr)
        return 1

    assert size is not None
    print(
        "RTSP DIAGNOSTIC: PASS — "
        f"frames={frame_count} decoded_size={size[0]}x{size[1]} "
        f"protocol={args.protocol} decoder=software"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
