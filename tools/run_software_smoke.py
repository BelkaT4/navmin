"""Hardware-free software smoke support for the NavMin production boundaries."""

from __future__ import annotations

import argparse
import logging
from threading import Event, Lock, Thread
from time import monotonic

import cv2
import numpy as np

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config.models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    AxesConfig,
    AxisMechanicsConfig,
    PidControllerConfig,
    SerialConfig,
    Stm32Config,
    TurretConfig,
    UiConfig,
)
from navmin.contracts import CameraRole
from navmin.core import Mediator
from navmin.lifecycle import StopToken
from navmin.logging_setup import configure_logging
from navmin.turret.worker import TransportFactory, TurretWorker, WorkerShutdownError
from navmin.vision.camera_worker import CameraWorker
from navmin.vision.pipeline import (
    InMemoryFrameSource,
    VisionPipeline,
    overview_corrector,
    stereo_left_corrector,
)

LOGGER = logging.getLogger("navmin.software_smoke")

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
_OVERVIEW_FPS = 20.0
_STEREO_LEFT_FPS = 15.0
_TARGET_RADIUS_PX = 6
_TARGET_SPEED_PX_PER_FRAME = 2

_IDENTITY_3X3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_Q_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def synthetic_overview_calibration() -> OverviewCalibration:
    """Return deterministic tool-local calibration for the 320x240 source."""
    camera_matrix = (
        (1000.0, 0.0, (FRAME_WIDTH - 1) / 2.0),
        (0.0, 1000.0, (FRAME_HEIGHT - 1) / 2.0),
        (0.0, 0.0, 1.0),
    )
    return OverviewCalibration(
        schema_version=1,
        image_width=FRAME_WIDTH,
        image_height=FRAME_HEIGHT,
        K=camera_matrix,
        D=(0.0, 0.0, 0.0, 0.0),
        new_camera_matrix=camera_matrix,
    )


def synthetic_stereo_calibration() -> StereoCalibration:
    """Return deterministic tool-local stereo calibration for the smoke source."""
    camera_matrix = (
        (250.0, 0.0, (FRAME_WIDTH - 1) / 2.0),
        (0.0, 250.0, (FRAME_HEIGHT - 1) / 2.0),
        (0.0, 0.0, 1.0),
    )
    projection = (
        (camera_matrix[0][0], 0.0, camera_matrix[0][2], 0.0),
        (0.0, camera_matrix[1][1], camera_matrix[1][2], 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    return StereoCalibration(
        schema_version=1,
        image_width=FRAME_WIDTH,
        image_height=FRAME_HEIGHT,
        K_left=camera_matrix,
        D_left=(0.0, 0.0, 0.0, 0.0, 0.0),
        K_right=camera_matrix,
        D_right=(0.0, 0.0, 0.0, 0.0, 0.0),
        R=_IDENTITY_3X3,
        T=(-0.46, 0.0, 0.0),
        R1=_IDENTITY_3X3,
        R2=_IDENTITY_3X3,
        P1=projection,
        P2=projection,
        Q=_Q_IDENTITY,
    )


def _ping_pong_coordinate(frame_index: int, start: int, end: int) -> int:
    span = end - start
    phase = (frame_index * _TARGET_SPEED_PX_PER_FRAME) % (2 * span)
    return start + (phase if phase <= span else 2 * span - phase)


def _synthetic_target_center(
    camera: CameraRole,
    frame_index: int,
) -> tuple[int, int]:
    if camera is CameraRole.OVERVIEW:
        return _ping_pong_coordinate(frame_index, 190, 280), 82
    if camera is CameraRole.STEREO_LEFT:
        return _ping_pong_coordinate(frame_index, 40, 130), 170
    raise ValueError("software smoke only produces Overview and Stereo Left")


def _overview_frame(frame_index: int) -> np.ndarray:
    frame = np.full(
        (FRAME_HEIGHT, FRAME_WIDTH, 3),
        (32, 48, 64),
        dtype=np.uint8,
    )
    cv2.circle(
        frame,
        _synthetic_target_center(CameraRole.OVERVIEW, frame_index),
        _TARGET_RADIUS_PX,
        (245, 245, 245),
        -1,
    )
    return frame


def _stereo_left_frame(frame_index: int) -> np.ndarray:
    frame = np.full(
        (FRAME_HEIGHT, FRAME_WIDTH, 3),
        (70, 44, 30),
        dtype=np.uint8,
    )
    cv2.circle(
        frame,
        _synthetic_target_center(CameraRole.STEREO_LEFT, frame_index),
        _TARGET_RADIUS_PX,
        (245, 245, 245),
        -1,
    )
    return frame


class SyntheticFrameProducer:
    """Small smoke-local producer for one hardware-free camera boundary."""

    def __init__(
        self,
        *,
        camera: CameraRole,
        source: InMemoryFrameSource,
    ) -> None:
        if camera is CameraRole.OVERVIEW:
            fps = _OVERVIEW_FPS
        elif camera is CameraRole.STEREO_LEFT:
            fps = _STEREO_LEFT_FPS
        else:
            raise ValueError("software smoke only produces Overview and Stereo Left")
        self.camera = camera
        self.source = source
        self.fps = fps
        self._stop_token = StopToken()
        self._thread = Thread(
            target=self._run,
            name=f"smoke-producer-{camera.value}",
            daemon=False,
        )
        self._counter_lock = Lock()
        self._frames_produced = 0
        self._pause_event = Event()

    @property
    def frames_produced(self) -> int:
        with self._counter_lock:
            return self._frames_produced

    def start(self) -> None:
        self._thread.start()

    def pause(self) -> None:
        """Temporarily stop publishing frames without stopping the producer thread."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume latest-only frame publication after :meth:`pause`."""
        self._pause_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def request_stop(self) -> None:
        self._stop_token.request_stop()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> bool:
        self.request_stop()
        self.join(timeout)
        return not self.is_alive()

    def _run(self) -> None:
        period_s = 1.0 / self.fps
        next_deadline = monotonic()
        frame_index = 0
        while not self._stop_token.is_stop_requested():
            if self._pause_event.is_set():
                # Keep the producer thread alive while allowing bounded shutdown.
                next_deadline = monotonic()
                if self._stop_token.wait(min(period_s, 0.02)):
                    break
                continue
            frame = (
                _overview_frame(frame_index)
                if self.camera is CameraRole.OVERVIEW
                else _stereo_left_frame(frame_index)
            )
            self.source.push(frame)
            with self._counter_lock:
                self._frames_produced += 1
            frame_index += 1
            next_deadline += period_s
            wait_s = max(0.0, next_deadline - monotonic())
            if self._stop_token.wait(wait_s):
                break


def _smoke_aiming_config() -> AimingConfig:
    center = AimPointConfig(x_px=None, y_px=None)
    return AimingConfig(
        lead_time_ms=100,
        target_lost_timeout_ms=500,
        aim_points=AimPointsConfig(overview=center, stereo_left=center),
    )


def _smoke_ui_config() -> UiConfig:
    return UiConfig(
        default_camera=CameraRole.OVERVIEW,
        show_fps=False,
        show_stereo_right_diagnostics=False,
    )


def _smoke_turret_config() -> TurretConfig:
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    return TurretConfig(
        serial=SerialConfig(
            port="software-smoke",
            baudrate=9600,
            response_timeout_ms=25,
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
        emulate_stm32=True,
    )


class SoftwareSmokeRuntime:
    """Smoke-only composition of existing production workers and boundaries."""

    def __init__(
        self,
        *,
        turret_transport_factory: TransportFactory | None = None,
    ) -> None:
        self.overview_source = InMemoryFrameSource()
        self.stereo_left_source = InMemoryFrameSource()

        self.overview_pipeline = VisionPipeline(
            camera=CameraRole.OVERVIEW,
            corrector=overview_corrector(synthetic_overview_calibration()),
            processing_enabled=True,
        )
        self.stereo_left_pipeline = VisionPipeline(
            camera=CameraRole.STEREO_LEFT,
            corrector=stereo_left_corrector(synthetic_stereo_calibration()),
            processing_enabled=True,
        )
        self.overview_worker = CameraWorker(
            source=self.overview_source,
            pipeline=self.overview_pipeline,
            idle_wait_s=0.005,
        )
        self.stereo_left_worker = CameraWorker(
            source=self.stereo_left_source,
            pipeline=self.stereo_left_pipeline,
            idle_wait_s=0.005,
        )
        self.overview_producer = SyntheticFrameProducer(
            camera=CameraRole.OVERVIEW,
            source=self.overview_source,
        )
        self.stereo_left_producer = SyntheticFrameProducer(
            camera=CameraRole.STEREO_LEFT,
            source=self.stereo_left_source,
        )

        self.turret_worker = TurretWorker(
            _smoke_turret_config(),
            transport_factory=turret_transport_factory,
        )
        self.mediator = Mediator(
            aiming_config=_smoke_aiming_config(),
            ui_config=_smoke_ui_config(),
            turret=self.turret_worker,
        )

        self._started_camera_workers: list[CameraWorker] = []
        self._started_producers: list[SyntheticFrameProducer] = []
        self._turret_started = False
        self._shutdown_complete = False

    def start(self) -> None:
        """Start Turret, then camera workers, then synthetic hardware producers."""
        if self._turret_started or self._started_camera_workers or self._started_producers:
            raise RuntimeError("software smoke runtime is one-shot")

        self.turret_worker.start()
        self._turret_started = True

        for worker in (self.overview_worker, self.stereo_left_worker):
            worker.start()
            self._started_camera_workers.append(worker)

        self._wait_for_camera_sessions(timeout=2.0)

        for producer in (self.overview_producer, self.stereo_left_producer):
            producer.start()
            self._started_producers.append(producer)

        LOGGER.info("Overview started")
        LOGGER.info("Stereo Left started")

    def camera_bindings(self):
        """Build the real UI binding map lazily so non-UI smoke stays headless."""
        from navmin.ui.bridge import CameraUiBinding

        return {
            CameraRole.OVERVIEW: CameraUiBinding(
                camera=CameraRole.OVERVIEW,
                session_barriers=self.overview_pipeline.session_barriers,
                latest_result=self.overview_pipeline.latest_result,
                status=self.overview_pipeline.status,
            ),
            CameraRole.STEREO_LEFT: CameraUiBinding(
                camera=CameraRole.STEREO_LEFT,
                session_barriers=self.stereo_left_pipeline.session_barriers,
                latest_result=self.stereo_left_pipeline.latest_result,
                status=self.stereo_left_pipeline.status,
            ),
        }

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop only started components in smoke ownership order, bounded."""
        if self._shutdown_complete:
            return
        LOGGER.info("software smoke shutdown starting")
        failures: list[str] = []

        for producer in reversed(self._started_producers):
            if not producer.stop(timeout=timeout):
                failures.append(f"{producer.camera.value} synthetic producer")
        self._started_producers.clear()

        for worker in reversed(self._started_camera_workers):
            if not worker.stop(timeout=timeout):
                failures.append(f"{worker.pipeline.camera.value} CameraWorker")
        self._started_camera_workers.clear()

        if self._turret_started:
            try:
                self.turret_worker.shutdown(timeout=timeout)
            except WorkerShutdownError as exc:
                failures.append(f"TurretWorker: {exc}")
            self._turret_started = False

        self._shutdown_complete = True
        LOGGER.info(
            "software smoke frames produced Overview=%d Stereo Left=%d",
            self.overview_producer.frames_produced,
            self.stereo_left_producer.frames_produced,
        )
        if failures:
            raise RuntimeError("software smoke shutdown failed: " + "; ".join(failures))
        LOGGER.info("software smoke shutdown complete")

    def _wait_for_camera_sessions(self, timeout: float) -> None:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if (
                self.overview_pipeline.generation >= 1
                and self.stereo_left_pipeline.generation >= 1
            ):
                return
            if not self.overview_worker.is_alive() or not self.stereo_left_worker.is_alive():
                break
            self.overview_pipeline.status.wait_for_revision(
                self.overview_pipeline.status.snapshot().revision,
                min(0.01, max(0.0, deadline - monotonic())),
            )
        raise RuntimeError("camera workers did not establish smoke sessions in time")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hardware-free NavMin software smoke composition."
    )
    parser.add_argument(
        "--auto-close-seconds",
        type=float,
        default=None,
        help="quit the Qt event loop automatically after a positive number of seconds",
    )
    args = parser.parse_args(argv)
    if args.auto_close_seconds is not None and args.auto_close_seconds <= 0.0:
        parser.error("--auto-close-seconds must be > 0")
    return args


def _run_qt_smoke(runtime: SoftwareSmokeRuntime, auto_close_seconds: float | None) -> int:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from navmin.ui.app import run_ui

    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    bindings = runtime.camera_bindings()
    if auto_close_seconds is not None:
        QTimer.singleShot(round(auto_close_seconds * 1000.0), application.quit)

    LOGGER.info("UI starting")
    return run_ui(
        mediator=runtime.mediator,
        camera_bindings=bindings,
        turret_states=runtime.turret_worker.state_updates,
        camera_stale_timeout_ms=500,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(logging.INFO)
    LOGGER.info("software smoke starting")
    runtime = SoftwareSmokeRuntime()
    exit_code = 1
    try:
        runtime.start()
        exit_code = _run_qt_smoke(runtime, args.auto_close_seconds)
    except Exception:
        LOGGER.exception("software smoke failed")
    finally:
        try:
            runtime.shutdown()
        except Exception:
            LOGGER.exception("software smoke shutdown failed")
            exit_code = 1
    return exit_code


__all__ = [
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "SoftwareSmokeRuntime",
    "SyntheticFrameProducer",
    "main",
    "synthetic_overview_calibration",
    "synthetic_stereo_calibration",
]


if __name__ == "__main__":
    raise SystemExit(main())
