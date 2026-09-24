"""Shared production application composition and owned worker lifecycle."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from types import MappingProxyType

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.concurrency import LatestValue
from navmin.config.models import AppConfig
from navmin.contracts import (
    CameraRole,
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
from navmin.vision.pipeline import overview_corrector, stereo_left_corrector

LOGGER = logging.getLogger(__name__)

_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 2.0
_OVERVIEW_COMPONENT = "Overview CameraWorker"
_STEREO_LEFT_COMPONENT = "Stereo Left CameraWorker"
_TURRET_COMPONENT = "TurretWorker"
_TURRET_SAFETY_COMPONENT = "Turret safety shutdown"


@dataclass(frozen=True)
class ApplicationFactories:
    """Narrow hardware-boundary seams; defaults retain production transports."""

    overview_source_factory: SourceFactory = create_camera_source
    stereo_left_source_factory: SourceFactory = create_camera_source
    turret_transport_factory: TransportFactory | None = None


@dataclass(frozen=True)
class UiRuntimeDependencies:
    """Complete dependency surface consumed by :func:`navmin.ui.run_ui`."""

    mediator: Mediator
    camera_bindings: Mapping[CameraRole, CameraUiBinding]
    turret_states: LatestValue[TurretState]
    camera_stale_timeout_ms: int


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
    """One-shot owner of the two normal cameras, Turret, and Mediator."""

    def __init__(
        self,
        *,
        overview_worker: CameraWorker,
        stereo_left_worker: CameraWorker,
        turret_worker: TurretWorker,
        mediator: Mediator,
        camera_stale_timeout_ms: int,
    ) -> None:
        self.overview_worker = overview_worker
        self.stereo_left_worker = stereo_left_worker
        self.turret_worker = turret_worker
        self.mediator = mediator
        self.camera_stale_timeout_ms = camera_stale_timeout_ms

        self._camera_workers = MappingProxyType(
            {
                CameraRole.OVERVIEW: overview_worker,
                CameraRole.STEREO_LEFT: stereo_left_worker,
            }
        )
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

    def start(self) -> None:
        """Start Turret, Overview, and Stereo Left with automatic rollback."""
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
            (_STEREO_LEFT_COMPONENT, self.stereo_left_worker),
            (_OVERVIEW_COMPONENT, self.overview_worker),
            (_TURRET_COMPONENT, self.turret_worker),
        )
        for component, worker in ordered:
            if component not in self._active_components:
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
            for worker in (self.overview_worker, self.stereo_left_worker)
        )

    @staticmethod
    def _binding(worker: CameraWorker) -> CameraUiBinding:
        pipeline = worker.pipeline
        return CameraUiBinding(
            camera=pipeline.camera,
            session_barriers=pipeline.session_barriers,
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
    )
    turret_worker = TurretWorker(
        config.turret,
        transport_factory=selected_factories.turret_transport_factory,
    )
    mediator = Mediator(
        aiming_config=config.aiming,
        ui_config=config.ui,
        turret=turret_worker,
    )
    return ApplicationRuntime(
        overview_worker=overview_worker,
        stereo_left_worker=stereo_left_worker,
        turret_worker=turret_worker,
        mediator=mediator,
        camera_stale_timeout_ms=config.vision.camera_stale_timeout_ms,
    )


__all__ = [
    "ApplicationFactories",
    "ApplicationRuntime",
    "ApplicationShutdownError",
    "ApplicationShutdownFailure",
    "UiRuntimeDependencies",
    "build_application_runtime",
]
