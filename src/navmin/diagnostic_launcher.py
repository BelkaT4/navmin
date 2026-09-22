"""Diagnostic launcher using production receivers/transports with external endpoints."""

from __future__ import annotations

import argparse
import logging
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
    make_session_log_path,
    run_loaded_application,
    session_file_logging,
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
                camera = config.vision.cameras.overview
                self._start_sender(
                    port=camera.port,
                    width=overview.image_width,
                    height=overview.image_height,
                    pattern="ball",
                )
                config = _replace_camera_endpoint(
                    config,
                    camera_name="overview",
                )

            if self.selection.stereo_left is CameraEndpoint.LOCALHOST:
                stereo = self._base_inputs.stereo_calibration
                camera = config.vision.cameras.stereo_left
                self._start_sender(
                    port=camera.port,
                    width=stereo.image_width,
                    height=stereo.image_height,
                    pattern="smpte",
                )
                config = _replace_camera_endpoint(
                    config,
                    camera_name="stereo_left",
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
        port: int,
        width: int,
        height: int,
        pattern: str,
    ) -> None:
        sender = LocalhostRtpJpegSender(
            RtpJpegSenderConfig(
                port=port,
                width=width,
                height=height,
                pattern=pattern,
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


def _replace_camera_endpoint(config: AppConfig, *, camera_name: str) -> AppConfig:
    cameras = config.vision.cameras
    camera = getattr(cameras, camera_name)
    local = replace(camera, address="127.0.0.1", rtp_enabled=True)
    updated_cameras = replace(cameras, **{camera_name: local})
    return replace(config, vision=replace(config.vision, cameras=updated_cameras))


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
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
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
    log_path = make_session_log_path(paths.log_dir, mode="diagnostic")

    with session_file_logging(log_path, level=logging.DEBUG):
        LOGGER.info(
            "Diagnostic launcher selected overview=%s stereo_left=%s turret=%s; "
            "inputs=%s config=%s overview_calibration=%s "
            "stereo_calibration=%s log=%s",
            selection.overview.value,
            selection.stereo_left.value,
            selection.turret.value,
            "synthetic" if args.synthetic_inputs else "files",
            paths.config,
            paths.overview_calibration,
            paths.stereo_calibration,
            log_path,
        )
        endpoints: DiagnosticEndpoints | None = None
        exit_code = 2
        try:
            inputs = (
                _synthetic_diagnostic_inputs()
                if args.synthetic_inputs
                else load_application_inputs(paths)
            )
            endpoints = DiagnosticEndpoints(selection=selection, inputs=inputs)
            effective_inputs = endpoints.start()
            exit_code = run_loaded_application(effective_inputs)
        except (RuntimeError, ValueError) as exc:
            LOGGER.error("NavMin diagnostic startup/runtime failed: %s", exc)
        finally:
            if endpoints is not None:
                try:
                    endpoints.stop()
                except RuntimeError as exc:
                    LOGGER.error("Diagnostic endpoint cleanup failed: %s", exc)
                    exit_code = 2
        return exit_code


__all__ = [
    "CameraEndpoint",
    "DiagnosticEndpoints",
    "DiagnosticSelection",
    "TurretEndpoint",
    "main",
]
