#!/usr/bin/env python3
"""Preview the in-runtime Overview/Y video-latency calibration profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from navmin.diagnostics.live_video_latency_calibration import (
    LiveVideoLatencyCalibrationConfig,
    build_motion_profile,
    motion_profile_duration_s,
)
from navmin.launcher import (
    ApplicationInputError,
    LauncherPaths,
    load_application_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/navmin.json"))
    parser.add_argument("--rate-deg-s", type=float, default=6.0)
    parser.add_argument("--segment-seconds", type=float, default=0.8)
    parser.add_argument("--runs", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_application_inputs(LauncherPaths(config=args.config))
        config = LiveVideoLatencyCalibrationConfig(
            command_rate_deg_s=args.rate_deg_s,
            segment_duration_s=args.segment_seconds,
            run_count=args.runs,
        )
        config.validate()
    except (ApplicationInputError, OSError, TypeError, ValueError) as exc:
        print(f"VIDEO LATENCY CALIBRATION PREVIEW: FAIL — {exc}")
        return 2

    print("NAVMIN IN-RUNTIME VIDEO LATENCY CALIBRATION PREVIEW")
    print("  camera pipeline: existing production Overview (no restart)")
    print("  turret transport: existing TurretWorker (no second serial owner)")
    print("  calibration axis: Y only")
    print(f"  command rate: ±{config.command_rate_deg_s:.3f} deg/s")
    print(f"  runs: {config.run_count}")
    print(f"  per-run duration: {motion_profile_duration_s(config):.3f} s")
    print(
        "  configured video-latency-ms: "
        f"{inputs.config.tracking_control.platform_motion.video_latency_ms}"
    )
    print("  profile:")
    for step in build_motion_profile(config):
        print(f"    t={step.offset_s:.3f}s -> vy={step.rate_y_deg_s:+.3f} deg/s")
    print()
    print("Hardware execution is intentionally available only from the running UI:")
    print(
        "  Esc -> Калибровка -> "
        "КАЛИБРОВКА ЗАДЕРЖКИ ВИДЕО"
    )
    print("This preview opens no camera, serial device, or motor control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
