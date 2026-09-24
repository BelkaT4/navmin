"""Diagnostic launcher using production receivers/transports with external endpoints."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from navmin.config import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    AppConfig,
    AxesConfig,
    AxisMechanicsConfig,
    CamerasConfig,
    PidControllerConfig,
    ProcessingScope,
    RtpJpegSourceConfig,
    SerialConfig,
    StereoDistanceConfig,
    Stm32Config,
    TurretConfig,
    UiConfig,
    VisionConfig,
    VisionDistanceConfig,
)
from navmin.contracts import CameraRole, DistanceSource
from navmin.diagnostics.localhost_rtp import (
    LocalhostRtpJpegSender,
    RtpJpegSenderConfig,
    SenderProcessError,
    diagnostic_camera_config,
    diagnostic_overview_calibration,
    diagnostic_stereo_calibration,
)
from navmin.diagnostics.pty_stm32 import PtyDiagnosticError, PtyStm32Emulator
from navmin.launcher import (
    LauncherPaths,
    LoadedApplicationInputs,
    load_application_inputs,
    run_loaded_application,
    session_file_logging,
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


class CameraEndpoint(Enum):
    REAL = "real"
    LOCALHOST = "localhost"


class TurretEndpoint(Enum):
    REAL = "real"
    PTY = "pty"


@dataclass(frozen=True)
class DiagnosticSelection:
    overview: CameraEndpoint
    stereo_left: CameraEndpoint
    turret: TurretEndpoint


class DiagnosticEndpoints:
    """Own only external localhost/PTY endpoints around shared ApplicationRuntime."""

    def __init__(
        self,
        *,
        selection: DiagnosticSelection,
        inputs: LoadedApplicationInputs,
    ) -> None:
        self.selection = selection
        self._base_inputs = inputs
        self._senders: list[LocalhostRtpJpegSender] = []
        self._pty: PtyStm32Emulator | None = None
        self._started = False

    def start(self) -> LoadedApplicationInputs:
        if self._started:
            raise RuntimeError("DiagnosticEndpoints is one-shot and already started")
        self._started = True

        config = replace(
            self._base_inputs.config,
            turret=replace(self._base_inputs.config.turret, emulate_stm32=False),
        )
        try:
            if self.selection.overview is CameraEndpoint.LOCALHOST:
                overview = self._base_inputs.overview_calibration
                config = _replace_camera_endpoint(
                    config,
                    camera_name="overview",
                    fallback_port=8888,
                )
                source = config.vision.cameras.overview.source
                assert isinstance(source, RtpJpegSourceConfig)
                self._start_sender(
                    camera_role=CameraRole.OVERVIEW,
                    port=source.port,
                    width=overview.image_width,
                    height=overview.image_height,
                )

            if self.selection.stereo_left is CameraEndpoint.LOCALHOST:
                stereo = self._base_inputs.stereo_calibration
                config = _replace_camera_endpoint(
                    config,
                    camera_name="stereo_left",
                    fallback_port=8889,
                )
                source = config.vision.cameras.stereo_left.source
                assert isinstance(source, RtpJpegSourceConfig)
                self._start_sender(
                    camera_role=CameraRole.STEREO_LEFT,
                    port=source.port,
                    width=stereo.image_width,
                    height=stereo.image_height,
                )

            if self.selection.turret is TurretEndpoint.PTY:
                pty = PtyStm32Emulator()
                pty.start()
                self._pty = pty
                serial = replace(config.turret.serial, port=pty.stable_port_path)
                config = replace(
                    config,
                    turret=replace(
                        config.turret,
                        serial=serial,
                        emulate_stm32=False,
                    ),
                )
        except (OSError, RuntimeError):
            self.stop()
            raise

        return replace(self._base_inputs, config=config)

    def stop(self) -> None:
        errors: list[RuntimeError] = []
        pty = self._pty
        self._pty = None
        if pty is not None:
            try:
                pty.stop()
            except PtyDiagnosticError as exc:
                errors.append(exc)
        while self._senders:
            sender = self._senders.pop()
            try:
                sender.stop()
            except SenderProcessError as exc:
                errors.append(exc)
        if errors:
            detail = "; ".join(str(error) for error in errors)
            raise RuntimeError(f"diagnostic endpoint cleanup failed: {detail}")

    def _start_sender(
        self,
        *,
        camera_role: CameraRole,
        port: int,
        width: int,
        height: int,
    ) -> None:
        sender = LocalhostRtpJpegSender(
            RtpJpegSenderConfig(
                port=port,
                width=width,
                height=height,
                camera=camera_role,
            )
        )
        sender.start()
        self._senders.append(sender)


def _synthetic_diagnostic_inputs() -> LoadedApplicationInputs:
    """Build explicit in-memory config/calibration inputs for the all-software profile."""
    center = AimPointConfig(x_px=None, y_px=None)
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    overview = replace(diagnostic_camera_config(8888), processing_enabled=True)
    stereo_left = replace(diagnostic_camera_config(8889), processing_enabled=True)
    stereo_right = diagnostic_camera_config(8890)
    config = AppConfig(
        schema_version=1,
        vision=VisionConfig(
            processing_scope=ProcessingScope.MAIN_AND_PREVIEW,
            cameras=CamerasConfig(
                overview=overview,
                stereo_left=stereo_left,
                stereo_right=stereo_right,
            ),
            distance=VisionDistanceConfig(
                source=DistanceSource.MANUAL,
                manual_distance_m=100.0,
                distance_stale_timeout_ms=300,
                stereo=StereoDistanceConfig(
                    stereo_enabled=False,
                    right_frame_buffer_size=4,
                    pair_timeout_ms=100,
                ),
            ),
            camera_stale_timeout_ms=500,
            simulation_mode=True,
        ),
        aiming=AimingConfig(
            lead_time_ms=100,
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(overview=center, stereo_left=center),
        ),
        turret=TurretConfig(
            serial=SerialConfig(
                port="diagnostic-pty",
                baudrate=9600,
                response_timeout_ms=50,
                max_retries=1,
                inter_request_delay_ms=1,
            ),
            axes=AxesConfig(x=axis, y=axis),
            controller=PidControllerConfig(
                pid_kp_x=1.0,
                pid_ki_x=0.0,
                pid_kd_x=0.0,
                pid_kp_y=1.0,
                pid_ki_y=0.0,
                pid_kd_y=0.0,
            ),
            stm32=Stm32Config(
                max_speed_x_deg_s=50.0,
                max_speed_y_deg_s=50.0,
                acceleration_x_deg_s2=100.0,
                acceleration_y_deg_s2=100.0,
                velocity_watchdog_timeout_ms=200,
            ),
            emulate_stm32=False,
        ),
        ui=UiConfig(
            default_camera=CameraRole.OVERVIEW,
            show_fps=False,
            show_stereo_right_diagnostics=False,
        ),
    )
    return LoadedApplicationInputs(
        config=config,
        overview_calibration=diagnostic_overview_calibration(),
        stereo_calibration=diagnostic_stereo_calibration(),
    )


def _replace_camera_endpoint(
    config: AppConfig,
    *,
    camera_name: str,
    fallback_port: int,
) -> AppConfig:
    cameras = config.vision.cameras
    camera = getattr(cameras, camera_name)
    source = camera.source
    port = source.port if isinstance(source, RtpJpegSourceConfig) else fallback_port
    local = replace(
        camera,
        source=RtpJpegSourceConfig(
            bind_address="127.0.0.1",
            port=port,
            buffer_size=source.buffer_size,
        ),
    )
    updated_cameras = replace(cameras, **{camera_name: local})
    return replace(config, vision=replace(config.vision, cameras=updated_cameras))


def _planned_preflight_inputs(
    selection: DiagnosticSelection,
    inputs: LoadedApplicationInputs,
) -> LoadedApplicationInputs:
    """Apply only pure endpoint overrides needed to preflight the planned profile."""
    config = replace(inputs.config, turret=replace(inputs.config.turret, emulate_stm32=False))
    if selection.overview is CameraEndpoint.LOCALHOST:
        config = _replace_camera_endpoint(
            config, camera_name="overview", fallback_port=8888
        )
    if selection.stereo_left is CameraEndpoint.LOCALHOST:
        config = _replace_camera_endpoint(
            config, camera_name="stereo_left", fallback_port=8889
        )
    return replace(inputs, config=config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_diagnostic_app.py",
        description=(
            "Run the shared NavMin application with explicit real/localhost camera "
            "endpoints and a real/PTY controller endpoint."
        ),
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run static startup preflight, write session evidence, and exit.",
    )
    parser.add_argument(
        "--synthetic-inputs",
        action="store_true",
        help=(
            "Use built-in 320x240 diagnostic config/calibrations. "
            "Allowed only with localhost/localhost/pty."
        ),
    )
    parser.add_argument(
        "--overview",
        choices=tuple(item.value for item in CameraEndpoint),
        required=True,
    )
    parser.add_argument(
        "--stereo-left",
        choices=tuple(item.value for item in CameraEndpoint),
        required=True,
    )
    parser.add_argument(
        "--turret",
        choices=tuple(item.value for item in TurretEndpoint),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    paths = LauncherPaths(
        config=args.config,
        overview_calibration=args.overview_calibration,
        stereo_calibration=args.stereo_calibration,
        log_dir=args.log_dir,
    )
    selection = DiagnosticSelection(
        overview=CameraEndpoint(args.overview),
        stereo_left=CameraEndpoint(args.stereo_left),
        turret=TurretEndpoint(args.turret),
    )
    fully_synthetic = selection == DiagnosticSelection(
        overview=CameraEndpoint.LOCALHOST,
        stereo_left=CameraEndpoint.LOCALHOST,
        turret=TurretEndpoint.PTY,
    )
    if args.synthetic_inputs and not fully_synthetic:
        parser.error(
            "--synthetic-inputs requires --overview localhost "
            "--stereo-left localhost --turret pty"
        )

    artifacts = SessionArtifacts.create(
        paths.log_dir,
        mode="diagnostic",
        overview_backend=selection.overview.value,
        stereo_left_backend=selection.stereo_left.value,
        turret_backend=selection.turret.value,
        input_mode="synthetic" if args.synthetic_inputs else "files",
    )
    source_paths = None
    if not args.synthetic_inputs:
        source_paths = SourceInputPaths(
            config=paths.config,
            overview_calibration=paths.overview_calibration,
            stereo_calibration=paths.stereo_calibration,
        )

    with session_file_logging(artifacts.runtime_log_path, level=logging.DEBUG):
        LOGGER.info(
            "Diagnostic launcher selected overview=%s stereo_left=%s turret=%s; "
            "inputs=%s config=%s overview_calibration=%s "
            "stereo_calibration=%s session=%s",
            selection.overview.value,
            selection.stereo_left.value,
            selection.turret.value,
            "synthetic" if args.synthetic_inputs else "files",
            paths.config,
            paths.overview_calibration,
            paths.stereo_calibration,
            artifacts.session_dir,
        )

        try:
            inputs = (
                _synthetic_diagnostic_inputs()
                if args.synthetic_inputs
                else load_application_inputs(paths)
            )
        except (OSError, ValueError) as exc:
            LOGGER.exception("NavMin diagnostic input loading failed")
            print(f"INPUT ERROR: {exc}", file=sys.stderr)
            artifacts.finalize(
                status=SessionStatus.INPUT_FAILED,
                exit_code=2,
                failure=exc,
            )
            return 2

        try:
            planned_inputs = _planned_preflight_inputs(selection, inputs)
            report = run_startup_preflight(
                planned_inputs,
                selection=StartupBackendSelection(
                    overview=selection.overview.value,
                    stereo_left=selection.stereo_left.value,
                    turret=selection.turret.value,
                ),
                session_dir=artifacts.session_dir,
            )
            artifacts.write_preflight(report)
        except Exception as exc:
            LOGGER.exception("NavMin diagnostic preflight implementation failed")
            artifacts.finalize(
                status=SessionStatus.RUNTIME_FAILED,
                exit_code=2,
                failure=exc,
            )
            return 2

        summary = format_preflight_summary(report)
        print(summary)
        LOGGER.info("%s", summary.replace("\n", " | "))
        if not report.ok:
            artifacts.finalize(
                status=SessionStatus.PREFLIGHT_FAILED,
                exit_code=2,
            )
            return 2
        if args.preflight_only:
            artifacts.finalize(status=SessionStatus.COMPLETED, exit_code=0)
            return 0

        endpoints: DiagnosticEndpoints | None = None
        primary_failure: BaseException | None = None
        cleanup_failure: RuntimeError | None = None
        status = SessionStatus.RUNTIME_FAILED
        exit_code = 2
        try:
            endpoints = DiagnosticEndpoints(selection=selection, inputs=inputs)
            effective_inputs = endpoints.start()
            artifacts.write_effective_inputs(
                config=effective_inputs.config,
                overview_calibration=effective_inputs.overview_calibration,
                stereo_calibration=effective_inputs.stereo_calibration,
                source_paths=source_paths,
            )
            exit_code = run_loaded_application(
                effective_inputs,
                show_diagnostic_clock=(
                    CameraEndpoint.LOCALHOST
                    in (selection.overview, selection.stereo_left)
                ),
            )
            if exit_code == 0:
                status = SessionStatus.COMPLETED
            else:
                primary_failure = RuntimeError(
                    f"application returned nonzero exit code {exit_code}"
                )
                LOGGER.error("%s", primary_failure)
        except KeyboardInterrupt as exc:
            LOGGER.warning("NavMin diagnostic run interrupted by operator")
            primary_failure = exc
            status = SessionStatus.INTERRUPTED
            exit_code = 130
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception("NavMin diagnostic startup/runtime failed")
            primary_failure = exc
            status = SessionStatus.RUNTIME_FAILED
            exit_code = 2
        finally:
            if endpoints is not None:
                try:
                    endpoints.stop()
                except RuntimeError as exc:
                    LOGGER.exception("Diagnostic endpoint cleanup failed")
                    cleanup_failure = exc
                    if exit_code == 0:
                        exit_code = 2

            final_status = (
                SessionStatus.CLEANUP_FAILED
                if cleanup_failure is not None
                else status
            )
            failure = primary_failure
            cleanup_evidence = cleanup_failure
            if failure is None and cleanup_failure is not None:
                failure = cleanup_failure
                cleanup_evidence = None
            artifacts.finalize(
                status=final_status,
                exit_code=exit_code,
                failure=failure,
                cleanup_failure=cleanup_evidence,
            )
        return exit_code


__all__ = [
    "CameraEndpoint",
    "DiagnosticEndpoints",
    "DiagnosticSelection",
    "TurretEndpoint",
    "main",
]
