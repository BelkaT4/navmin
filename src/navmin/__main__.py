"""Normal real-hardware NavMin launcher."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from navmin.launcher import (
    LauncherPaths,
    load_application_inputs,
    run_loaded_application,
    session_file_logging,
    validate_normal_hardware_config,
)
from navmin.session_artifacts import (
    SessionArtifacts,
    SessionStatus,
    SourceInputPaths,
)

LOGGER = logging.getLogger(__name__)
_FAILURE_EXIT_CODE = 2
_INTERRUPTED_EXIT_CODE = 130


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
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Parent directory for per-session NavMin artifact directories.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = LauncherPaths(
        config=args.config,
        overview_calibration=args.overview_calibration,
        stereo_calibration=args.stereo_calibration,
        log_dir=args.log_dir,
    )
    artifacts = SessionArtifacts.create(
        paths.log_dir,
        mode="normal",
        overview_backend="real",
        stereo_left_backend="real",
        turret_backend="real",
        input_mode="files",
    )
    source_paths = SourceInputPaths(
        config=paths.config,
        overview_calibration=paths.overview_calibration,
        stereo_calibration=paths.stereo_calibration,
    )

    with session_file_logging(artifacts.runtime_log_path, level=logging.INFO):
        LOGGER.info(
            "Normal launcher selected real RTP cameras + real STM32; "
            "config=%s overview_calibration=%s stereo_calibration=%s session=%s",
            paths.config,
            paths.overview_calibration,
            paths.stereo_calibration,
            artifacts.session_dir,
        )
        try:
            inputs = load_application_inputs(paths)
            validate_normal_hardware_config(inputs.config)
        except (OSError, ValueError) as exc:
            LOGGER.exception("NavMin input loading failed")
            artifacts.finalize(
                status=SessionStatus.INPUT_FAILED,
                exit_code=_FAILURE_EXIT_CODE,
                failure=exc,
            )
            return _FAILURE_EXIT_CODE

        try:
            artifacts.write_effective_inputs(
                config=inputs.config,
                overview_calibration=inputs.overview_calibration,
                stereo_calibration=inputs.stereo_calibration,
                source_paths=source_paths,
            )
            exit_code = run_loaded_application(inputs)
        except KeyboardInterrupt as exc:
            LOGGER.warning("NavMin interrupted by operator")
            artifacts.finalize(
                status=SessionStatus.INTERRUPTED,
                exit_code=_INTERRUPTED_EXIT_CODE,
                failure=exc,
            )
            return _INTERRUPTED_EXIT_CODE
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("NavMin startup/runtime failed")
            artifacts.finalize(
                status=SessionStatus.RUNTIME_FAILED,
                exit_code=_FAILURE_EXIT_CODE,
                failure=exc,
            )
            return _FAILURE_EXIT_CODE

        if exit_code != 0:
            failure = RuntimeError(f"application returned nonzero exit code {exit_code}")
            LOGGER.error("%s", failure)
            artifacts.finalize(
                status=SessionStatus.RUNTIME_FAILED,
                exit_code=exit_code,
                failure=failure,
            )
            return exit_code

        artifacts.finalize(status=SessionStatus.COMPLETED, exit_code=0)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
