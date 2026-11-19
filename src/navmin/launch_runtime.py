"""Runtime bindings derived from immutable unified-launcher source selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.contracts import CameraRole
from navmin.diagnostics.localhost_rtp import (
    DEFAULT_FPS,
    SynchronizedSyntheticStereoSource,
    SyntheticDiagnosticFrameGenerator,
    diagnostic_overview_calibration,
)
from navmin.diagnostics.synthetic_sources import SyntheticFrameSource
from navmin.diagnostics.synthetic_stereo import (
    DEFAULT_SYNTHETIC_FPS,
    DEFAULT_SYNTHETIC_HEIGHT,
    DEFAULT_SYNTHETIC_WIDTH,
    rectified_synthetic_calibration,
    rectified_synthetic_side_calibration,
)
from navmin.diagnostics.synthetic_tracking_turret import (
    RealtimeSyntheticTrackingRig,
    SyntheticTrackingFrameSource,
    SyntheticTrackingStm32Endpoint,
    SyntheticTurretPlant,
)
from navmin.diagnostics.synthetic_tracking_world import SyntheticTrackingWorld
from navmin.launch_options import (
    CameraSourceMode,
    EffectiveLaunchSelection,
    LaunchOptions,
    StereoRightMode,
    Stm32Mode,
    apply_launch_options,
)
from navmin.launcher import LoadedApplicationInputs
from navmin.turret.simulator import FakeTransport
from navmin.turret.worker import TransportFactory
from navmin.vision.camera_source import create_camera_source
from navmin.vision.camera_worker import SourceFactory


class LaunchRuntimeBindingError(ValueError):
    """Resolved source selection cannot be represented by current runtime seams."""


@dataclass(frozen=True)
class LaunchRuntimeFactories:
    """Camera/turret factories selected for one application launch."""

    overview_source_factory: SourceFactory
    stereo_left_source_factory: SourceFactory
    stereo_right_source_factory: SourceFactory
    turret_transport_factory: TransportFactory | None = None


@dataclass(frozen=True)
class LaunchRuntimeBindings:
    """One launch's effective typed inputs and source factories."""

    selection: EffectiveLaunchSelection
    inputs: LoadedApplicationInputs
    factories: LaunchRuntimeFactories
    show_diagnostic_clock: bool


def _synthetic_overview_factory(
    inputs: LoadedApplicationInputs,
) -> tuple[SourceFactory, OverviewCalibration]:
    reference = inputs.overview_calibration
    calibration = diagnostic_overview_calibration(
        width=reference.image_width,
        height=reference.image_height,
    )

    def factory(_config):
        generator = SyntheticDiagnosticFrameGenerator(
            camera=CameraRole.OVERVIEW,
            width=calibration.image_width,
            height=calibration.image_height,
        )
        return SyntheticFrameSource(generator.next_frame, fps=DEFAULT_FPS)

    return factory, calibration


def _synthetic_stereo_factory(
    shared: SynchronizedSyntheticStereoSource,
    camera: CameraRole,
) -> SourceFactory:
    def factory(_config):
        return SyntheticFrameSource(
            lambda: shared.next_frame(camera),
            fps=DEFAULT_SYNTHETIC_FPS,
        )

    return factory


def _full_tracking_sim_factories(
    inputs: LoadedApplicationInputs,
    selection: EffectiveLaunchSelection,
) -> tuple[LaunchRuntimeFactories, OverviewCalibration, StereoCalibration]:
    reference = inputs.stereo_calibration
    if (
        reference.image_width != DEFAULT_SYNTHETIC_WIDTH
        or reference.image_height != DEFAULT_SYNTHETIC_HEIGHT
    ):
        raise LaunchRuntimeBindingError(
            "synthetic tracking world requires a 1296x976 reference stereo calibration"
        )
    world = SyntheticTrackingWorld(reference)
    plant = SyntheticTurretPlant(selection.config.turret)
    rig = RealtimeSyntheticTrackingRig(world, plant)
    endpoint = SyntheticTrackingStm32Endpoint(
        plant,
        request_sink=rig.apply_executed_request,
    )

    def source_factory(camera: CameraRole) -> SourceFactory:
        def factory(_config):
            return SyntheticTrackingFrameSource(rig, camera)

        return factory

    def turret_transport_factory(_port: str, baudrate: int, emulate_stm32: bool):
        if not emulate_stm32:
            raise RuntimeError("synthetic tracking transport requires STM32 SIM mode")
        return FakeTransport(endpoint, baudrate=baudrate)

    overview_calibration = diagnostic_overview_calibration(
        width=DEFAULT_SYNTHETIC_WIDTH,
        height=DEFAULT_SYNTHETIC_HEIGHT,
    )
    stereo_calibration = rectified_synthetic_calibration(reference)
    return (
        LaunchRuntimeFactories(
            overview_source_factory=source_factory(CameraRole.OVERVIEW),
            stereo_left_source_factory=source_factory(CameraRole.STEREO_LEFT),
            stereo_right_source_factory=source_factory(CameraRole.STEREO_RIGHT),
            turret_transport_factory=turret_transport_factory,
        ),
        overview_calibration,
        stereo_calibration,
    )


def build_launch_runtime_bindings(
    inputs: LoadedApplicationInputs,
    options: LaunchOptions,
) -> LaunchRuntimeBindings:
    """Resolve in-process synthetic bindings without mutating loaded inputs."""
    selection = apply_launch_options(inputs.config, options)
    overview_calibration = inputs.overview_calibration
    stereo_calibration = inputs.stereo_calibration

    overview_factory = create_camera_source
    left_factory = create_camera_source
    right_factory = create_camera_source
    turret_transport_factory: TransportFactory | None = None

    full_tracking_sim = (
        options.overview is CameraSourceMode.SIM
        and options.stereo_left is CameraSourceMode.SIM
        and options.stereo_right is StereoRightMode.SIM
        and options.stm32 is Stm32Mode.SIM
    )
    if full_tracking_sim:
        tracking_factories, overview_calibration, stereo_calibration = (
            _full_tracking_sim_factories(inputs, selection)
        )
        overview_factory = tracking_factories.overview_source_factory
        left_factory = tracking_factories.stereo_left_source_factory
        right_factory = tracking_factories.stereo_right_source_factory
        turret_transport_factory = tracking_factories.turret_transport_factory
    elif options.overview is CameraSourceMode.SIM:
        overview_factory, overview_calibration = _synthetic_overview_factory(inputs)

    left_sim = options.stereo_left is CameraSourceMode.SIM
    right_sim = options.stereo_right is StereoRightMode.SIM
    if (left_sim or right_sim) and not full_tracking_sim:
        reference = inputs.stereo_calibration
        if (
            reference.image_width != DEFAULT_SYNTHETIC_WIDTH
            or reference.image_height != DEFAULT_SYNTHETIC_HEIGHT
        ):
            raise LaunchRuntimeBindingError(
                "synthetic stereo requires a 1296x976 reference stereo calibration"
            )
        scene_calibration = rectified_synthetic_calibration(reference)
        shared = SynchronizedSyntheticStereoSource(scene_calibration)
        stereo_calibration = rectified_synthetic_side_calibration(
            reference,
            left_synthetic=left_sim,
            right_synthetic=right_sim,
        )
        if left_sim:
            left_factory = _synthetic_stereo_factory(shared, CameraRole.STEREO_LEFT)
        if right_sim:
            right_factory = _synthetic_stereo_factory(shared, CameraRole.STEREO_RIGHT)

    effective_inputs = replace(
        inputs,
        config=selection.config,
        overview_calibration=overview_calibration,
        stereo_calibration=stereo_calibration,
    )
    return LaunchRuntimeBindings(
        selection=selection,
        inputs=effective_inputs,
        factories=LaunchRuntimeFactories(
            overview_source_factory=overview_factory,
            stereo_left_source_factory=left_factory,
            stereo_right_source_factory=right_factory,
            turret_transport_factory=turret_transport_factory,
        ),
        show_diagnostic_clock=bool(selection.simulated_camera_roles),
    )


__all__ = [
    "LaunchRuntimeBindingError",
    "LaunchRuntimeBindings",
    "LaunchRuntimeFactories",
    "build_launch_runtime_bindings",
]
