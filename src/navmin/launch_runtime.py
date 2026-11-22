"""Runtime bindings derived from immutable unified-launcher source selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

from navmin.contracts import CameraRole
from navmin.diagnostics.synthetic_tracking_calibration import (
    mixed_stereo_calibration,
    synthetic_tracking_overview_calibration,
    synthetic_tracking_stereo_calibration,
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
    simulation: SimulationContext | None = None


@dataclass(frozen=True)
class SimulationContext:
    """Shared synthetic world/plant for every SIM component in one launch."""

    plant: SyntheticTurretPlant
    world: SyntheticTrackingWorld | None
    rig: RealtimeSyntheticTrackingRig | None
    stm32_endpoint: SyntheticTrackingStm32Endpoint | None


def _build_simulation_context(
    selection: EffectiveLaunchSelection,
) -> SimulationContext | None:
    simulated_cameras = selection.simulated_camera_roles
    stm32_sim = selection.options.stm32 is Stm32Mode.SIM
    if not simulated_cameras and not stm32_sim:
        return None

    plant = SyntheticTurretPlant(selection.config.turret)
    world = None
    rig = None
    if simulated_cameras:
        world = SyntheticTrackingWorld()
        rig = RealtimeSyntheticTrackingRig(
            world,
            plant,
            synchronize_stereo_pair=(
                selection.options.stereo_left is CameraSourceMode.SIM
                and selection.options.stereo_right is StereoRightMode.SIM
            ),
        )

    endpoint = None
    if stm32_sim:
        endpoint = SyntheticTrackingStm32Endpoint(
            plant,
            request_sink=(
                plant.apply_executed_request
                if rig is None
                else rig.apply_executed_request
            ),
        )
    return SimulationContext(
        plant=plant,
        world=world,
        rig=rig,
        stm32_endpoint=endpoint,
    )


def _synthetic_camera_factory(
    context: SimulationContext,
    camera: CameraRole,
) -> SourceFactory:
    rig = context.rig
    if rig is None:
        raise LaunchRuntimeBindingError(
            "synthetic camera source requires a shared synthetic tracking rig"
        )

    def factory(_config):
        return SyntheticTrackingFrameSource(rig, camera)

    return factory


def _synthetic_turret_factory(context: SimulationContext) -> TransportFactory:
    endpoint = context.stm32_endpoint
    if endpoint is None:
        raise LaunchRuntimeBindingError(
            "synthetic turret transport requires STM32 SIM mode"
        )

    def factory(_port: str, baudrate: int, emulate_stm32: bool):
        if not emulate_stm32:
            raise RuntimeError("synthetic tracking transport requires STM32 SIM mode")
        return FakeTransport(endpoint, baudrate=baudrate)

    return factory


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
    simulation = _build_simulation_context(selection)

    if options.overview is CameraSourceMode.SIM:
        assert simulation is not None
        overview_factory = _synthetic_camera_factory(
            simulation, CameraRole.OVERVIEW
        )
        overview_calibration = synthetic_tracking_overview_calibration()

    left_sim = options.stereo_left is CameraSourceMode.SIM
    right_sim = options.stereo_right is StereoRightMode.SIM
    if left_sim or right_sim:
        assert simulation is not None
        any_real_stereo = (
            options.stereo_left is CameraSourceMode.ON
            or options.stereo_right is StereoRightMode.ON
        )
        stereo_calibration = (
            mixed_stereo_calibration(
                inputs.stereo_calibration,
                left_synthetic=left_sim,
                right_synthetic=right_sim,
            )
            if any_real_stereo
            else synthetic_tracking_stereo_calibration()
        )
        if left_sim:
            left_factory = _synthetic_camera_factory(
                simulation, CameraRole.STEREO_LEFT
            )
        if right_sim:
            right_factory = _synthetic_camera_factory(
                simulation, CameraRole.STEREO_RIGHT
            )

    if options.stm32 is Stm32Mode.SIM:
        assert simulation is not None
        turret_transport_factory = _synthetic_turret_factory(simulation)

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
        simulation=simulation,
    )


__all__ = [
    "LaunchRuntimeBindingError",
    "LaunchRuntimeBindings",
    "LaunchRuntimeFactories",
    "SimulationContext",
    "build_launch_runtime_bindings",
]
