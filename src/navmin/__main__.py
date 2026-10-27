"""Unified NavMin operator launcher frontend."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from navmin.host_environment import (
    HostConfigurationCreated,
    HostEnvironmentError,
    create_host_environment,
)
from navmin.input_recovery import (
    InputRecoveryError,
    InputRecoveryResult,
    recover_default_inputs,
)
from navmin.launch_options import (
    CameraSourceMode,
    LaunchOptions,
    StereoRightMode,
    Stm32Mode,
    default_launch_options,
    resolve_launch_options,
)
from navmin.launch_runtime import build_launch_runtime_bindings
from navmin.launcher import (
    LauncherPaths,
    load_application_inputs,
    run_loaded_application,
    session_file_logging,
)
from navmin.preflight import (
    StartupBackendSelection,
    format_preflight_summary,
    run_startup_preflight,
)
from navmin.session_artifacts import SessionArtifacts, SessionStatus

LOGGER = logging.getLogger(__name__)
_FAILURE_EXIT_CODE = 2
_INTERRUPTED_EXIT_CODE = 130
_DEFAULT_CONFIG = Path("config/navmin.json")
_DEFAULT_LOG_DIR = Path("logs")


def _prompt_input_recovery(error_text: str) -> bool:
    from navmin.input_recovery_dialog import prompt_input_recovery

    return prompt_input_recovery(error_text)


def _show_input_recovery_result(result: InputRecoveryResult) -> None:
    from navmin.input_recovery_dialog import show_input_recovery_result

    show_input_recovery_result(result)


def _show_input_recovery_failure(message: str) -> None:
    from navmin.input_recovery_dialog import show_input_recovery_failure

    show_input_recovery_failure(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navmin",
        description=(
            "Run NavMin with independent real/sim source selection. "
            "Running without arguments opens the interactive launch menu."
        ),
    )
    parser.add_argument(
        "--overview",
        choices=tuple(mode.value for mode in CameraSourceMode),
        default=None,
        help="Overview source: configured real source (on) or synthetic (sim).",
    )
    parser.add_argument(
        "--stereo-left",
        choices=tuple(mode.value for mode in CameraSourceMode),
        default=None,
        help="Stereo Left source: configured real source (on) or synthetic (sim).",
    )
    parser.add_argument(
        "--stereo-right",
        choices=tuple(mode.value for mode in StereoRightMode),
        default=None,
        help="Stereo Right source: configured real source, synthetic, or off.",
    )
    parser.add_argument(
        "--stm32",
        choices=tuple(mode.value for mode in Stm32Mode),
        default=None,
        help="STM32 source: configured real serial device (on) or in-process sim.",
    )
    parser.add_argument(
        "-p",
        "--preflight-only",
        action="store_true",
        help="Run startup preflight for the effective source selection and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug session-file logging.",
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    return parser


def _preflight_selection(options: LaunchOptions) -> StartupBackendSelection:
    return StartupBackendSelection(
        overview=("real" if options.overview is CameraSourceMode.ON else "sim"),
        stereo_left=(
            "real" if options.stereo_left is CameraSourceMode.ON else "sim"
        ),
        turret=("real" if options.stm32 is Stm32Mode.ON else "sim"),
        stereo_right=(
            None
            if options.stereo_right is StereoRightMode.OFF
            else "real"
            if options.stereo_right is StereoRightMode.ON
            else "sim"
        ),
    )


def _artifact_backend(value: str | None) -> str:
    return "config-default" if value is None else value


def _create_noninteractive_artifacts(args: argparse.Namespace) -> SessionArtifacts:
    return SessionArtifacts.create(
        _DEFAULT_LOG_DIR,
        mode="normal",
        overview_backend=_artifact_backend(args.overview),
        stereo_left_backend=_artifact_backend(args.stereo_left),
        turret_backend=_artifact_backend(args.stm32),
        input_mode="files",
    )


def _finalize_failure(
    artifacts: SessionArtifacts,
    *,
    status: SessionStatus,
    error: BaseException,
    exit_code: int = _FAILURE_EXIT_CODE,
) -> int:
    artifacts.finalize(status=status, exit_code=exit_code, failure=error)
    return exit_code


def _run_noninteractive(
    args: argparse.Namespace,
    *,
    origin: str = "non-interactive",
) -> int:
    paths = LauncherPaths(config=args.config, log_dir=_DEFAULT_LOG_DIR)
    artifacts = _create_noninteractive_artifacts(args)
    log_level = logging.DEBUG if args.verbose else logging.INFO

    with session_file_logging(artifacts.runtime_log_path, level=log_level):
        LOGGER.info(
            "Unified %s launcher: config=%s overview=%s "
            "stereo_left=%s stereo_right=%s stm32=%s preflight_only=%s "
            "verbose=%s session=%s",
            origin,
            args.config,
            args.overview,
            args.stereo_left,
            args.stereo_right,
            args.stm32,
            args.preflight_only,
            args.verbose,
            artifacts.session_dir,
        )
        try:
            inputs = load_application_inputs(paths)
            options = resolve_launch_options(
                inputs.config,
                config_path=args.config,
                overview=(
                    None if args.overview is None else CameraSourceMode(args.overview)
                ),
                stereo_left=(
                    None
                    if args.stereo_left is None
                    else CameraSourceMode(args.stereo_left)
                ),
                stereo_right=(
                    None
                    if args.stereo_right is None
                    else StereoRightMode(args.stereo_right)
                ),
                stm32=None if args.stm32 is None else Stm32Mode(args.stm32),
                preflight_only=args.preflight_only,
                verbose=args.verbose,
            )
            bindings = build_launch_runtime_bindings(inputs, options)
        except (OSError, ValueError) as exc:
            LOGGER.exception("NavMin non-interactive input/source resolution failed")
            print(f"INPUT ERROR: {exc}", file=sys.stderr)
            return _finalize_failure(
                artifacts,
                status=SessionStatus.INPUT_FAILED,
                error=exc,
            )

        effective_inputs = bindings.inputs
        selection = bindings.selection
        for warning in selection.warnings:
            LOGGER.warning("%s", warning)

        if effective_inputs.source_paths is not None:
            LOGGER.info(
                "Resolved input paths: config=%s overview_calibration=%s "
                "stereo_calibration=%s",
                effective_inputs.source_paths.config,
                effective_inputs.source_paths.overview_calibration,
                effective_inputs.source_paths.stereo_calibration,
            )

        try:
            host = create_host_environment(Path.cwd())
            with host.activation(
                camera_network_required=selection.requires_camera_network
            ):
                report = run_startup_preflight(
                    effective_inputs,
                    selection=_preflight_selection(options),
                    selection_warnings=selection.warnings,
                    session_dir=artifacts.session_dir,
                )
                artifacts.write_preflight(report)
                summary = format_preflight_summary(report)
                print(summary)
                LOGGER.info("%s", summary.replace("\n", " | "))
                if not report.ok:
                    artifacts.finalize(
                        status=SessionStatus.PREFLIGHT_FAILED,
                        exit_code=_FAILURE_EXIT_CODE,
                    )
                    return _FAILURE_EXIT_CODE
                if options.preflight_only:
                    artifacts.finalize(status=SessionStatus.COMPLETED, exit_code=0)
                    return 0

                artifacts.write_effective_inputs(
                    config=effective_inputs.config,
                    overview_calibration=effective_inputs.overview_calibration,
                    stereo_calibration=effective_inputs.stereo_calibration,
                    source_paths=effective_inputs.source_paths,
                )
                exit_code = run_loaded_application(
                    effective_inputs,
                    show_diagnostic_clock=bindings.show_diagnostic_clock,
                    overview_source_factory=(
                        bindings.factories.overview_source_factory
                    ),
                    stereo_left_source_factory=(
                        bindings.factories.stereo_left_source_factory
                    ),
                    stereo_right_source_factory=(
                        bindings.factories.stereo_right_source_factory
                    ),
                )
        except KeyboardInterrupt as exc:
            LOGGER.warning("NavMin interrupted by operator")
            artifacts.finalize(
                status=SessionStatus.INTERRUPTED,
                exit_code=_INTERRUPTED_EXIT_CODE,
                failure=exc,
            )
            return _INTERRUPTED_EXIT_CODE
        except HostConfigurationCreated as exc:
            LOGGER.warning("%s", exc)
            print(f"HOST CONFIG: {exc}", file=sys.stderr)
            return _finalize_failure(
                artifacts,
                status=SessionStatus.INPUT_FAILED,
                error=exc,
            )
        except HostEnvironmentError as exc:
            LOGGER.exception("NavMin host preparation/restore failed")
            print(f"HOST ERROR: {exc}", file=sys.stderr)
            return _finalize_failure(
                artifacts,
                status=SessionStatus.RUNTIME_FAILED,
                error=exc,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("NavMin startup/runtime failed")
            return _finalize_failure(
                artifacts,
                status=SessionStatus.RUNTIME_FAILED,
                error=exc,
            )
        except Exception as exc:
            LOGGER.exception("NavMin unexpected startup/preflight failure")
            return _finalize_failure(
                artifacts,
                status=SessionStatus.RUNTIME_FAILED,
                error=exc,
            )

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


def _interactive_failure_artifacts() -> SessionArtifacts:
    return SessionArtifacts.create(
        _DEFAULT_LOG_DIR,
        mode="normal",
        overview_backend="config-default",
        stereo_left_backend="config-default",
        turret_backend="config-default",
        input_mode="files",
    )


def _handle_interactive_input_failure(
    paths: LauncherPaths,
    error: BaseException,
) -> int:
    artifacts = _interactive_failure_artifacts()
    with session_file_logging(artifacts.runtime_log_path, level=logging.INFO):
        LOGGER.error("NavMin interactive input loading failed: %s", error)
        print(f"INPUT ERROR: {error}", file=sys.stderr)
        should_recover = False
        recoverable = bool(getattr(error, "recoverable", False))
        if recoverable:
            try:
                should_recover = _prompt_input_recovery(str(error))
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
                    "NavMin local inputs restored; startup intentionally stops "
                    "until operator reviews hardware-specific values"
                )
                for source, backup in result.backups:
                    LOGGER.warning("Input backup: %s -> %s", source, backup)
                try:
                    _show_input_recovery_result(result)
                except (ImportError, OSError, RuntimeError):
                    LOGGER.exception("NavMin recovery result dialog unavailable")
        return _finalize_failure(
            artifacts,
            status=SessionStatus.INPUT_FAILED,
            error=error,
        )


def _prompt_choice(
    label: str,
    choices: tuple[str, ...],
    default: str,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    output_fn(f"{label}:")
    for index, choice in enumerate(choices, start=1):
        suffix = " (default)" if choice == default else ""
        output_fn(f"  {index}) {choice.upper()}{suffix}")
    while True:
        answer = input_fn(f"Выбор [Enter={default.upper()}]: ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        output_fn("Некорректный выбор. Повторите ввод.")


def _interactive_options(
    config,
    *,
    config_path: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> LaunchOptions | None:
    defaults = default_launch_options(config, config_path=config_path)
    output_fn("NavMin — параметры запуска")
    output_fn("")

    overview = CameraSourceMode(
        _prompt_choice(
            "Overview",
            tuple(mode.value for mode in CameraSourceMode),
            defaults.overview.value,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    )
    stereo_left = CameraSourceMode(
        _prompt_choice(
            "Stereo Left",
            tuple(mode.value for mode in CameraSourceMode),
            defaults.stereo_left.value,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    )
    stereo_right = StereoRightMode(
        _prompt_choice(
            "Stereo Right",
            tuple(mode.value for mode in StereoRightMode),
            defaults.stereo_right.value,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    )
    stm32 = Stm32Mode(
        _prompt_choice(
            "STM32",
            tuple(mode.value for mode in Stm32Mode),
            defaults.stm32.value,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    )
    preflight_choice = _prompt_choice(
        "Preflight",
        ("off", "on"),
        "off",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    logging_choice = _prompt_choice(
        "Logging",
        ("normal", "verbose"),
        "normal",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    options = LaunchOptions(
        overview=overview,
        stereo_left=stereo_left,
        stereo_right=stereo_right,
        stm32=stm32,
        preflight_only=preflight_choice == "on",
        verbose=logging_choice == "verbose",
        config_path=config_path,
    )

    output_fn("")
    output_fn("NavMin — параметры запуска")
    output_fn("")
    output_fn(f"Overview      {options.overview.value.upper()}")
    output_fn(f"Stereo Left   {options.stereo_left.value.upper()}")
    output_fn(f"Stereo Right  {options.stereo_right.value.upper()}")
    output_fn(f"STM32         {options.stm32.value.upper()}")
    output_fn(f"Preflight     {'ON' if options.preflight_only else 'OFF'}")
    output_fn(f"Logs          {'VERBOSE' if options.verbose else 'NORMAL'}")
    output_fn("")

    while True:
        answer = input_fn("Запустить? [Y/n]: ").strip().lower()
        if answer in ("", "y"):
            return options
        if answer == "n":
            output_fn("Запуск отменён.")
            return None
        output_fn("Введите Y, N или нажмите Enter.")


def _interactive_namespace(options: LaunchOptions) -> argparse.Namespace:
    return argparse.Namespace(
        overview=options.overview.value,
        stereo_left=options.stereo_left.value,
        stereo_right=options.stereo_right.value,
        stm32=options.stm32.value,
        preflight_only=options.preflight_only,
        verbose=options.verbose,
        config=options.config_path,
    )


def _run_interactive(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    paths = LauncherPaths(config=_DEFAULT_CONFIG, log_dir=_DEFAULT_LOG_DIR)
    try:
        inputs = load_application_inputs(paths)
        options = _interactive_options(
            inputs.config,
            config_path=_DEFAULT_CONFIG,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    except KeyboardInterrupt:
        output_fn("")
        output_fn("Запуск отменён.")
        return 0
    except (OSError, ValueError) as exc:
        return _handle_interactive_input_failure(paths, exc)

    if options is None:
        return 0
    return _run_noninteractive(_interactive_namespace(options), origin="interactive")

def main(argv: Sequence[str] | None = None) -> int:
    user_argv = tuple(sys.argv[1:] if argv is None else argv)
    if not user_argv:
        return _run_interactive()
    args = _parser().parse_args(user_argv)
    return _run_noninteractive(args)


if __name__ == "__main__":
    raise SystemExit(main())
