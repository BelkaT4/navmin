"""Shared production application composition and owned worker lifecycle."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from time import monotonic
from types import MappingProxyType

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.concurrency import LatestValue
from navmin.config.models import (
    AppConfig,
    PidControllerConfig,
    TrackingControlConfig,
    TurretConfig,
)
from navmin.contracts import (
    CameraRole,
    ConfigUpdate,
    MotorState,
    TurretConnectionState,
    TurretState,
)
from navmin.core import Mediator
from navmin.turret.worker import TransportFactory, TurretWorker
from navmin.ui.bridge import CameraUiBinding
from navmin.vision.camera_source import create_camera_source
from navmin.vision.camera_worker import CameraWorker, SourceFactory, build_camera_worker
from navmin.vision.gstreamer_source import (
    GStreamerRtpJpegSource,
    GStreamerRtspSource,
    initialize_gstreamer_runtime,
)
from navmin.vision.pipeline import (
    overview_corrector,
    stereo_left_corrector,
    stereo_right_corrector,
)
from navmin.vision.stereo_pairing import StereoPairer

LOGGER = logging.getLogger(__name__)

_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 2.0
_OVERVIEW_COMPONENT = "Overview CameraWorker"
_STEREO_LEFT_COMPONENT = "Stereo Left CameraWorker"
_STEREO_RIGHT_COMPONENT = "Stereo Right CameraWorker"
_TURRET_COMPONENT = "TurretWorker"
_TURRET_SAFETY_COMPONENT = "Turret safety shutdown"


@dataclass(frozen=True)
class ApplicationFactories:
    """Narrow hardware-boundary seams; defaults retain production transports."""

    overview_source_factory: SourceFactory = create_camera_source
    stereo_left_source_factory: SourceFactory = create_camera_source
    stereo_right_source_factory: SourceFactory = create_camera_source
    turret_transport_factory: TransportFactory | None = None


@dataclass(frozen=True)
class UiRuntimeDependencies:
    """Complete dependency surface consumed by :func:`navmin.ui.run_ui`."""

    mediator: Mediator
    camera_bindings: Mapping[CameraRole, CameraUiBinding]
    turret_states: LatestValue[TurretState]
    camera_stale_timeout_ms: int
    pid_controller_config: PidControllerConfig
    apply_pid_tuning: Callable[[PidControllerConfig], None]
    tracking_control_config: TrackingControlConfig
    apply_tracking_control_tuning: Callable[[TrackingControlConfig], None]


@dataclass(frozen=True)
class ApplicationShutdownFailure:
    component: str
    error: Exception


class ApplicationShutdownError(RuntimeError):
    """One or more owned workers failed their bounded shutdown boundary."""

    def __init__(self, failures: tuple[ApplicationShutdownFailure, ...]) -> None:
        if not failures:
            raise ValueError("failures must not be empty")
        self.failures = failures
        details = "; ".join(
            f"{failure.component}: {failure.error}" for failure in failures
        )
        super().__init__(f"application shutdown failed: {details}")


class ApplicationRuntime:
    """Own normal cameras, optional Stereo Right, Turret, and Mediator."""

    def __init__(
        self,
        *,
        overview_worker: CameraWorker,
        stereo_left_worker: CameraWorker,
        stereo_right_worker: CameraWorker | None,
        stereo_pairer: StereoPairer | None,
        turret_worker: TurretWorker,
        mediator: Mediator,
        camera_stale_timeout_ms: int,
        turret_config: TurretConfig,
        tracking_control_config: TrackingControlConfig,
    ) -> None:
        if not isinstance(turret_config, TurretConfig):
            raise TypeError("turret_config must be TurretConfig")
        if not isinstance(tracking_control_config, TrackingControlConfig):
            raise TypeError("tracking_control_config must be TrackingControlConfig")
        self.overview_worker = overview_worker
        self.stereo_left_worker = stereo_left_worker
        self.stereo_right_worker = stereo_right_worker
        self.stereo_pairer = stereo_pairer
        self.turret_worker = turret_worker
        self.mediator = mediator
        self.camera_stale_timeout_ms = camera_stale_timeout_ms
        self._runtime_turret_config = turret_config
        self._pid_tuning_revision = 0
        self._runtime_tracking_control_config = tracking_control_config
        self._tracking_tuning_revision = 0

        camera_workers = {
            CameraRole.OVERVIEW: overview_worker,
            CameraRole.STEREO_LEFT: stereo_left_worker,
        }
        if stereo_right_worker is not None:
            camera_workers[CameraRole.STEREO_RIGHT] = stereo_right_worker
        self._camera_workers = MappingProxyType(camera_workers)
        self._camera_bindings = MappingProxyType(
            {
                CameraRole.OVERVIEW: self._binding(overview_worker),
                CameraRole.STEREO_LEFT: self._binding(stereo_left_worker),
            }
        )
        self._ui_dependencies = UiRuntimeDependencies(
            mediator=mediator,
            camera_bindings=self._camera_bindings,
            turret_states=turret_worker.state_updates,
            camera_stale_timeout_ms=camera_stale_timeout_ms,
            pid_controller_config=turret_config.controller,
            apply_pid_tuning=self.apply_pid_tuning,
            tracking_control_config=tracking_control_config,
            apply_tracking_control_tuning=self.apply_tracking_control_tuning,
        )
        self._active_components: set[str] = set()
        self._start_attempted = False
        self._shutdown_complete = False

    @property
    def camera_workers(self) -> Mapping[CameraRole, CameraWorker]:
        return self._camera_workers

    @property
    def camera_bindings(self) -> Mapping[CameraRole, CameraUiBinding]:
        return self._camera_bindings

    @property
    def turret_states(self) -> LatestValue[TurretState]:
        return self.turret_worker.state_updates

    @property
    def ui_dependencies(self) -> UiRuntimeDependencies:
        return self._ui_dependencies

    def apply_pid_tuning(self, config: PidControllerConfig) -> None:
        """Queue a temporary PID-only runtime config update for Turret."""
        self._validate_pid_tuning(config)
        next_revision = self._pid_tuning_revision + 1
        next_turret_config = replace(self._runtime_turret_config, controller=config)
        self.turret_worker.submit_config_update(
            ConfigUpdate(revision=next_revision, config=next_turret_config)
        )
        self._runtime_turret_config = next_turret_config
        self._pid_tuning_revision = next_revision
        LOGGER.info(
            "PID tuning update queued revision=%d "
            "x=(kp=%g, ki=%g, kd=%g) y=(kp=%g, ki=%g, kd=%g)",
            next_revision,
            config.pid_kp_x,
            config.pid_ki_x,
            config.pid_kd_x,
            config.pid_kp_y,
            config.pid_ki_y,
            config.pid_kd_y,
        )

    def apply_tracking_control_tuning(
        self,
        config: TrackingControlConfig,
    ) -> None:
        """Apply one temporary tracking-control snapshot to both runtime owners."""
        self._validate_tracking_control_tuning(config)
        next_revision = self._tracking_tuning_revision + 1
        update = ConfigUpdate(revision=next_revision, config=config)
        self.mediator.apply_tracking_control_config_update(update)
        self.turret_worker.submit_tracking_control_config_update(update)
        self._runtime_tracking_control_config = config
        self._tracking_tuning_revision = next_revision
        LOGGER.info(
            "Tracking tuning update queued revision=%d "
            "FF enabled=%s gain=(%g,%g) Prediction enabled=%s "
            "horizon=%d window=%d",
            next_revision,
            config.feed_forward.enabled,
            config.feed_forward.gain_x,
            config.feed_forward.gain_y,
            config.prediction.enabled,
            config.prediction.horizon_ms,
            config.velocity_estimator.window_ms,
        )

    @staticmethod
    def _validate_tracking_control_tuning(config: TrackingControlConfig) -> None:
        if not isinstance(config, TrackingControlConfig):
            raise TypeError("config must be TrackingControlConfig")
        estimator = config.velocity_estimator
        prediction = config.prediction
        feed_forward = config.feed_forward
        integer_minima = (
            ("window_ms", estimator.window_ms, 1),
            ("min_span_ms", estimator.min_span_ms, 1),
            ("min_samples", estimator.min_samples, 2),
            ("max_samples", estimator.max_samples, 2),
            ("horizon_ms", prediction.horizon_ms, 0),
        )
        for name, value, minimum in integer_minima:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if estimator.min_span_ms > estimator.window_ms:
            raise ValueError("min_span_ms must be <= window_ms")
        if estimator.max_samples < estimator.min_samples:
            raise ValueError("max_samples must be >= min_samples")
        for name, value, strictly_positive in (
            ("max_fit_rms_deg", estimator.max_fit_rms_deg, True),
            ("max_target_rate_x_deg_s", estimator.max_target_rate_x_deg_s, True),
            ("max_target_rate_y_deg_s", estimator.max_target_rate_y_deg_s, True),
            ("gain_x", feed_forward.gain_x, False),
            ("gain_y", feed_forward.gain_y, False),
            ("max_angle_x_deg", prediction.max_angle_x_deg, True),
            ("max_angle_y_deg", prediction.max_angle_y_deg, True),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if strictly_positive and numeric <= 0:
                raise ValueError(f"{name} must be > 0")
            if not strictly_positive and numeric < 0:
                raise ValueError(f"{name} must be >= 0")
        if type(feed_forward.enabled) is not bool:
            raise TypeError("feed_forward.enabled must be bool")
        if type(prediction.enabled) is not bool:
            raise TypeError("prediction.enabled must be bool")

    @staticmethod
    def _validate_pid_tuning(config: PidControllerConfig) -> None:
        if not isinstance(config, PidControllerConfig):
            raise TypeError("config must be PidControllerConfig")
        for name in (
            "pid_kp_x",
            "pid_ki_x",
            "pid_kd_x",
            "pid_kp_y",
            "pid_ki_y",
            "pid_kd_y",
        ):
            value = getattr(config, name)
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a number")
            try:
                numeric = float(value)
            except OverflowError as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if numeric < 0:
                raise ValueError(f"{name} must be >= 0")

    def start(self) -> None:
        """Start owned workers in deterministic order with automatic rollback."""
        if self._start_attempted:
            raise RuntimeError("ApplicationRuntime is one-shot and start was attempted")
        if self._shutdown_complete:
            raise RuntimeError("ApplicationRuntime has already been shut down")
        self._start_attempted = True

        try:
            if self._uses_production_gstreamer():
                initialize_gstreamer_runtime()
            self.turret_worker.start()
            self._active_components.add(_TURRET_COMPONENT)
            self.overview_worker.start()
            self._active_components.add(_OVERVIEW_COMPONENT)
            self.stereo_left_worker.start()
            self._active_components.add(_STEREO_LEFT_COMPONENT)
            if self.stereo_right_worker is not None:
                self.stereo_right_worker.start()
                self._active_components.add(_STEREO_RIGHT_COMPONENT)
        except RuntimeError as startup_error:
            failures = self._stop_active(_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS)
            if failures:
                startup_error.add_note(
                    "partial-start rollback failures: "
                    + "; ".join(
                        f"{failure.component}: {failure.error}"
                        for failure in failures
                    )
                )
            self._shutdown_complete = not self._active_components
            if self.stereo_pairer is not None:
                self.stereo_pairer.reset()
            raise

        LOGGER.info("Application runtime workers started")

    def shutdown(self, timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """Stop every owned worker, attempting all components before reporting."""
        timeout = self._validate_timeout(timeout)
        if self._shutdown_complete:
            return
        if not self._start_attempted:
            self._start_attempted = True

        failures: list[ApplicationShutdownFailure] = []
        if _TURRET_COMPONENT in self._active_components:
            safety_failure = self._prepare_turret_for_shutdown(timeout)
            if safety_failure is not None:
                failures.append(safety_failure)
        failures.extend(self._stop_active(timeout))
        if self.stereo_pairer is not None:
            self.stereo_pairer.reset()
        self._shutdown_complete = not self._active_components
        if failures:
            raise ApplicationShutdownError(tuple(failures))
        LOGGER.info("Application runtime workers stopped")

    def _prepare_turret_for_shutdown(
        self,
        timeout: float,
    ) -> ApplicationShutdownFailure | None:
        state = self.turret_worker.current_state
        if state.connection_state is not TurretConnectionState.READY:
            LOGGER.warning(
                "Skipping Turret safety exchange because connection is %s",
                state.connection_state.value,
            )
            return None
        if state.motor_state is not MotorState.ON:
            return None

        if not self.turret_worker.stop_motion():
            return ApplicationShutdownFailure(
                _TURRET_SAFETY_COMPONENT,
                RuntimeError("STOP_MOTION was rejected before shutdown"),
            )
        if not self.turret_worker.motor_off():
            return ApplicationShutdownFailure(
                _TURRET_SAFETY_COMPONENT,
                RuntimeError("MOTOR_OFF was rejected before shutdown"),
            )

        deadline = monotonic() + timeout
        snapshot = self.turret_worker.state_updates.snapshot()
        while True:
            current = snapshot.value
            if current is not None and current.motor_state is MotorState.OFF:
                return None
            if (
                current is not None
                and current.connection_state is not TurretConnectionState.READY
            ):
                return ApplicationShutdownFailure(
                    _TURRET_SAFETY_COMPONENT,
                    RuntimeError(
                        "Turret left READY before MOTOR_OFF was confirmed"
                    ),
                )
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                return ApplicationShutdownFailure(
                    _TURRET_SAFETY_COMPONENT,
                    RuntimeError(
                        f"MOTOR_OFF was not confirmed within {timeout:.3f} seconds"
                    ),
                )
            changed = self.turret_worker.state_updates.wait_for_revision(
                snapshot.revision,
                remaining,
            )
            if changed is None:
                return ApplicationShutdownFailure(
                    _TURRET_SAFETY_COMPONENT,
                    RuntimeError(
                        f"MOTOR_OFF was not confirmed within {timeout:.3f} seconds"
                    ),
                )
            snapshot = changed

    def _stop_active(
        self,
        timeout: float,
    ) -> tuple[ApplicationShutdownFailure, ...]:
        failures: list[ApplicationShutdownFailure] = []
        ordered = (
            (_STEREO_RIGHT_COMPONENT, self.stereo_right_worker),
            (_STEREO_LEFT_COMPONENT, self.stereo_left_worker),
            (_OVERVIEW_COMPONENT, self.overview_worker),
            (_TURRET_COMPONENT, self.turret_worker),
        )
        for component, worker in ordered:
            if worker is None or component not in self._active_components:
                continue
            try:
                if isinstance(worker, CameraWorker):
                    if not worker.stop(timeout=timeout):
                        raise RuntimeError(
                            f"worker did not stop within {timeout:.3f} seconds"
                        )
                else:
                    worker.shutdown(timeout=timeout)
            except RuntimeError as error:
                failures.append(ApplicationShutdownFailure(component, error))
            else:
                self._active_components.remove(component)
        return tuple(failures)

    def _uses_production_gstreamer(self) -> bool:
        return any(
            isinstance(worker.source, (GStreamerRtpJpegSource, GStreamerRtspSource))
            for worker in self._camera_workers.values()
        )

    @staticmethod
    def _binding(worker: CameraWorker) -> CameraUiBinding:
        pipeline = worker.pipeline
        return CameraUiBinding(
            camera=pipeline.camera,
            session_barriers=pipeline.session_barriers,
            target_events=pipeline.target_events,
            latest_result=pipeline.latest_result,
            status=pipeline.status,
        )

    @staticmethod
    def _validate_timeout(timeout: float) -> float:
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise TypeError("timeout must be a number")
        value = float(timeout)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError("timeout must be a positive finite number")
        return value


def build_application_runtime(
    *,
    config: AppConfig,
    overview_calibration: OverviewCalibration,
    stereo_calibration: StereoCalibration,
    factories: ApplicationFactories | None = None,
) -> ApplicationRuntime:
    """Compose the production NavMin runtime from already-loaded typed inputs."""
    if not isinstance(config, AppConfig):
        raise TypeError("config must be AppConfig")
    selected_factories = factories or ApplicationFactories()

    stereo_config = config.vision.distance.stereo
    stereo_pairer = None
    if (
        config.vision.cameras.stereo_left.enabled
        and config.vision.cameras.stereo_right.enabled
        and stereo_config.stereo_enabled
    ):
        stereo_pairer = StereoPairer(
            right_frame_buffer_size=stereo_config.right_frame_buffer_size,
            pair_timeout_ms=stereo_config.pair_timeout_ms,
        )

    overview_worker = build_camera_worker(
        camera=CameraRole.OVERVIEW,
        config=config.vision.cameras.overview,
        corrector=overview_corrector(overview_calibration),
        source_factory=selected_factories.overview_source_factory,
    )
    stereo_left_worker = build_camera_worker(
        camera=CameraRole.STEREO_LEFT,
        config=config.vision.cameras.stereo_left,
        corrector=stereo_left_corrector(stereo_calibration),
        source_factory=selected_factories.stereo_left_source_factory,
        corrected_frame_sink=stereo_pairer,
    )
    stereo_right_worker = None
    if config.vision.cameras.stereo_right.enabled:
        stereo_right_worker = build_camera_worker(
            camera=CameraRole.STEREO_RIGHT,
            config=config.vision.cameras.stereo_right,
            corrector=stereo_right_corrector(stereo_calibration),
            source_factory=selected_factories.stereo_right_source_factory,
            corrected_frame_sink=stereo_pairer,
        )
    turret_worker = TurretWorker(
        config.turret,
        config.tracking_control,
        transport_factory=selected_factories.turret_transport_factory,
    )
    mediator = Mediator(
        aiming_config=config.aiming,
        tracking_control_config=config.tracking_control,
        tracking_gap_timeout_ms=config.turret.stm32.velocity_watchdog_timeout_ms,
        ui_config=config.ui,
        turret=turret_worker,
        target_controls={
            CameraRole.OVERVIEW: overview_worker.pipeline.target_control,
            CameraRole.STEREO_LEFT: stereo_left_worker.pipeline.target_control,
        },
        vision_processing_enabled={
            CameraRole.OVERVIEW: config.vision.cameras.overview.processing_enabled,
            CameraRole.STEREO_LEFT: (
                config.vision.cameras.stereo_left.processing_enabled
            ),
        },
    )
    return ApplicationRuntime(
        overview_worker=overview_worker,
        stereo_left_worker=stereo_left_worker,
        stereo_right_worker=stereo_right_worker,
        stereo_pairer=stereo_pairer,
        turret_worker=turret_worker,
        mediator=mediator,
        camera_stale_timeout_ms=config.vision.camera_stale_timeout_ms,
        turret_config=config.turret,
        tracking_control_config=config.tracking_control,
    )


__all__ = [
    "ApplicationFactories",
    "ApplicationRuntime",
    "ApplicationShutdownError",
    "ApplicationShutdownFailure",
    "UiRuntimeDependencies",
    "build_application_runtime",
]
