"""Normal real-hardware NavMin launcher."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from navmin.launcher import (
    LauncherPaths,
    load_application_inputs,
    make_session_log_path,
    run_loaded_application,
    session_file_logging,
    validate_normal_hardware_config,
)

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navmin",
        description="Run NavMin against real RTP cameras and a real STM32 serial device.",
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--overview-calibration",
        type=Path,
        default=Path("calibration/overview.json"),
    )
    parser.add_argument(
        "--stereo-calibration",
        type=Path,
        default=Path("calibration/stereo.json"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = LauncherPaths(
        config=args.config,
        overview_calibration=args.overview_calibration,
        stereo_calibration=args.stereo_calibration,
        log_dir=args.log_dir,
    )
    log_path = make_session_log_path(paths.log_dir, mode="normal")
    with session_file_logging(log_path, level=logging.INFO):
        LOGGER.info(
            "Normal launcher selected real RTP cameras + real STM32; "
            "config=%s overview_calibration=%s stereo_calibration=%s log=%s",
            paths.config,
            paths.overview_calibration,
            paths.stereo_calibration,
            log_path,
        )
        try:
            inputs = load_application_inputs(paths)
            validate_normal_hardware_config(inputs.config)
            return run_loaded_application(inputs)
        except (RuntimeError, ValueError) as exc:
            LOGGER.error("NavMin startup/runtime failed: %s", exc)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
