"""Normal real-hardware NavMin launcher."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from navmin.input_recovery import (
    InputRecoveryError,
    InputRecoveryResult,
    recover_default_inputs,
)
from navmin.launcher import (
    LauncherPaths,
    load_application_inputs,
    run_loaded_application,
    session_file_logging,
    validate_normal_hardware_config,
)
from navmin.preflight import (
    StartupBackendSelection,
    format_preflight_summary,
    run_startup_preflight,
)
from navmin.session_artifacts import (
    SessionArtifacts,
    SessionStatus,
    SourceInputPaths,
)

LOGGER = logging.getLogger(__name__)


def _prompt_input_recovery(error_text: str) -> bool:
    from navmin.input_recovery_dialog import prompt_input_recovery

    return prompt_input_recovery(error_text)


def _show_input_recovery_result(result: InputRecoveryResult) -> None:
    from navmin.input_recovery_dialog import show_input_recovery_result

    show_input_recovery_result(result)


def _show_input_recovery_failure(message: str) -> None:
    from navmin.input_recovery_dialog import show_input_recovery_failure

    show_input_recovery_failure(message)


_FAILURE_EXIT_CODE = 2
_INTERRUPTED_EXIT_CODE = 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navmin",
        description="Run NavMin against configured real camera sources and a real STM32 serial device.",
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
        "--preflight-only",
        action="store_true",
        help="Run static startup preflight, write session evidence, and exit.",
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
            "Normal launcher selected configured real cameras + real STM32; "
            "config=%s overview_calibration=%s stereo_calibration=%s session=%s",
            paths.config,
            paths.overview_calibration,
            paths.stereo_calibration,
            artifacts.session_dir,
        )
        try:
            inputs = load_application_inputs(paths)
        except (OSError, ValueError) as exc:
            LOGGER.exception("NavMin input loading failed")
            print(f"INPUT ERROR: {exc}", file=sys.stderr)
            should_recover = False
            if not args.preflight_only:
                try:
                    should_recover = _prompt_input_recovery(str(exc))
                except (ImportError, OSError, RuntimeError) as dialog_exc:
                    LOGGER.exception("NavMin input recovery dialog unavailable")
                    print(f"RECOVERY UI ERROR: {dialog_exc}", file=sys.stderr)
            if should_recover:
                try:
                    result = recover_default_inputs(paths)
                except InputRecoveryError as recovery_exc:
                    LOGGER.exception("NavMin input recovery failed")
                    print(f"RECOVERY ERROR: {recovery_exc}", file=sys.stderr)
                    try:
                        _show_input_recovery_failure(str(recovery_exc))
                    except (ImportError, OSError, RuntimeError):
                        LOGGER.exception("NavMin recovery failure dialog unavailable")
                else:
                    LOGGER.warning(
                        "NavMin local inputs restored; normal startup intentionally "
                        "stops until operator reviews hardware-specific values"
                    )
                    for source, backup in result.backups:
                        LOGGER.warning("Input backup: %s -> %s", source, backup)
                    try:
                        _show_input_recovery_result(result)
                    except (ImportError, OSError, RuntimeError):
                        LOGGER.exception("NavMin recovery result dialog unavailable")
            artifacts.finalize(
                status=SessionStatus.INPUT_FAILED,
                exit_code=_FAILURE_EXIT_CODE,
                failure=exc,
            )
            return _FAILURE_EXIT_CODE

        try:
            validate_normal_hardware_config(inputs.config)
        except ValueError as exc:
            LOGGER.exception("NavMin normal hardware config rejected")
            print(f"INPUT ERROR: {exc}", file=sys.stderr)
            artifacts.finalize(
                status=SessionStatus.INPUT_FAILED,
                exit_code=_FAILURE_EXIT_CODE,
                failure=exc,
            )
            return _FAILURE_EXIT_CODE

        try:
            report = run_startup_preflight(
                inputs,
                selection=StartupBackendSelection(
                    overview="real",
                    stereo_left="real",
                    turret="real",
                ),
                session_dir=artifacts.session_dir,
            )
            artifacts.write_preflight(report)
        except Exception as exc:
            LOGGER.exception("NavMin preflight implementation failed")
            artifacts.finalize(
                status=SessionStatus.RUNTIME_FAILED,
                exit_code=_FAILURE_EXIT_CODE,
                failure=exc,
            )
            return _FAILURE_EXIT_CODE

        summary = format_preflight_summary(report)
        print(summary)
        LOGGER.info("%s", summary.replace("\n", " | "))
        if not report.ok:
            artifacts.finalize(
                status=SessionStatus.PREFLIGHT_FAILED,
                exit_code=_FAILURE_EXIT_CODE,
            )
            return _FAILURE_EXIT_CODE
        if args.preflight_only:
            artifacts.finalize(status=SessionStatus.COMPLETED, exit_code=0)
            return 0

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
