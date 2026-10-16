"""Run the production stereo camera/pairing preflight without Turret or SGBM."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from navmin.diagnostics.stereo_preflight import (
    StereoHardwarePreflight,
    StereoHardwarePreflightError,
    build_stereo_preflight_components,
    write_preflight_json,
)
from navmin.launcher import (
    ApplicationInputError,
    LauncherPaths,
    load_application_inputs,
)
from navmin.logging_setup import configure_logging
from navmin.vision.gstreamer_source import CameraSourceError, GStreamerUnavailableError

DEFAULT_PREFLIGHT_DURATION_S = 30.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Observe production Stereo Left/Right acquisition and exact/timestamp "
            "pairing without starting Turret, target tracking, or SGBM."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/navmin.json"))
    parser.add_argument("--stereo-calibration", type=Path, default=None)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_PREFLIGHT_DURATION_S,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(logging.INFO)
    if args.duration_seconds <= 0.0:
        print("STEREO PREFLIGHT: FAIL — --duration-seconds must be > 0", file=sys.stderr)
        return 2

    try:
        inputs = load_application_inputs(
            LauncherPaths(
                config=args.config,
                stereo_calibration=args.stereo_calibration,
            )
        )
        components = build_stereo_preflight_components(inputs)
        outcome = StereoHardwarePreflight(components).run(args.duration_seconds)
        if args.output_json is not None:
            write_preflight_json(outcome, args.output_json)
    except (
        ApplicationInputError,
        CameraSourceError,
        GStreamerUnavailableError,
        StereoHardwarePreflightError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"STEREO PREFLIGHT: FAIL — {exc}", file=sys.stderr)
        return 1

    payload = outcome.to_dict()
    for camera_name, camera in payload["cameras"].items():
        resolution = camera["decoded_resolution"]
        resolution_text = (
            "none"
            if resolution is None
            else f"{resolution['width']}x{resolution['height']}"
        )
        fps = camera["received_fps"]
        fps_text = "na" if fps is None else f"{fps:.3f}"
        print(
            "STEREO_CAMERA "
            f"camera={camera_name} enabled={str(camera['enabled']).lower()} "
            f"generation={camera['generation']} frames={camera['frame_count']} "
            f"decoded_resolution={resolution_text} received_fps={fps_text} "
            f"capture_id_present={camera['capture_id_present_count']} "
            f"capture_id_missing={camera['capture_id_missing_count']}"
        )

    exact = payload["exact_pairing"]
    timestamp = payload["timestamp_pairing"]
    skew = timestamp["skew_ms"]
    print(
        "STEREO_EXACT "
        f"pairs={exact['exact_pairs_emitted']} "
        f"capture_id_missing={exact['capture_id_missing']} "
        f"unmatched_discarded={exact['unmatched_discarded']}"
    )
    print(
        "STEREO_TIMESTAMP "
        f"pairs={timestamp['timestamp_pairs_emitted']} "
        f"pair_rate_hz={timestamp['pair_rate_hz']:.3f} "
        f"max_timestamp_skew_ms={timestamp['max_timestamp_skew_ms']:.3f} "
        f"skew_min_ms={_value(skew['minimum'])} "
        f"skew_p50_ms={_value(skew['p50'])} "
        f"skew_p95_ms={_value(skew['p95'])} "
        f"skew_max_ms={_value(skew['maximum'])} "
        f"pending_left_superseded={timestamp['pending_left_superseded']} "
        f"pair_attempts_expired={timestamp['pair_attempts_expired']} "
        f"pending_left_frame_id={timestamp['pending_left_frame_id']} "
        f"right_buffer_size={timestamp['right_buffer_size']} "
        f"right_capacity_evictions={timestamp['right_frames_evicted_by_capacity']} "
        f"right_pruned_too_old={timestamp['right_frames_pruned_too_old']} "
        f"right_timed_out={timestamp['right_frames_timed_out']} "
        f"skew_rejected_candidates={timestamp['skew_rejected_candidates']}"
    )
    if args.output_json is not None:
        print(f"STEREO_JSON path={args.output_json}")
    if outcome.passed:
        print("STEREO PREFLIGHT: PASS — workers_stopped=true")
        return 0
    for failure in outcome.hard_failures:
        print(f"STEREO_PREFLIGHT_FAILURE reason={failure}", file=sys.stderr)
    print(
        f"STEREO PREFLIGHT: FAIL — workers_stopped={str(outcome.clean_shutdown).lower()}",
        file=sys.stderr,
    )
    return 1


def _value(value: float | None) -> str:
    return "na" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
