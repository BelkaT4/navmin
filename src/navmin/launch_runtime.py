"""Runtime bindings derived from immutable unified-launcher source selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

from navmin.calibration import StereoCalibration
from navmin.contracts import CameraRole
from navmin.diagnostics.localhost_rtp import (
    diagnostic_overview_calibration,
    diagnostic_stereo_calibration,
)
from navmin.diagnostics.synthetic_stereo import (
    DEFAULT_SYNTHETIC_HEIGHT,
    DEFAULT_SYNTHETIC_WIDTH,
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
    simulation: SimulationContext | None = None


@dataclass(frozen=True)
class SimulationContext:
    """Shared synthetic world/plant for every SIM component in one launch."""

    plant: SyntheticTurretPlant
    world: SyntheticTrackingWorld | None
    rig: RealtimeSyntheticTrackingRig | None
    stm32_endpoint: SyntheticTrackingStm32Endpoint | None


def _fallback_tracking_world_calibration() -> StereoCalibration:
    """Build internal world geometry when only Overview is synthetic.

    This is not exposed as the launch's effective stereo calibration.  It only
    gives ``SyntheticTrackingWorld`` a valid non-zero stereo baseline while the
    real/disabled stereo side keeps its loaded calibration unchanged.
    """
    calibration = diagnostic_stereo_calibration(
        width=DEFAULT_SYNTHETIC_WIDTH,
        height=DEFAULT_SYNTHETIC_HEIGHT,
    )
    fx = float(calibration.P1[0][0])
    baseline_native = abs(float(calibration.T[0]))
    p2 = (
        (
            float(calibration.P2[0][0]),
            float(calibration.P2[0][1]),
            float(calibration.P2[0][2]),
            -fx * baseline_native,
        ),
        calibration.P2[1],
        calibration.P2[2],
    )
    return replace(calibration, P2=p2)


def _build_simulation_context(
    inputs: LoadedApplicationInputs,
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
        reference = inputs.stereo_calibration
        stereo_simulated = (
            selection.options.stereo_left is CameraSourceMode.SIM
            or selection.options.stereo_right is StereoRightMode.SIM
        )
        reference_matches_world = (
            reference.image_width == DEFAULT_SYNTHETIC_WIDTH
            and reference.image_height == DEFAULT_SYNTHETIC_HEIGHT
        )
        if stereo_simulated and not reference_matches_world:
            raise LaunchRuntimeBindingError(
                "synthetic stereo requires a 1296x976 reference stereo calibration"
            )
        world_reference = (
            reference
            if reference_matches_world
            else _fallback_tracking_world_calibration()
        )
        world = SyntheticTrackingWorld(world_reference)
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
    simulation = _build_simulation_context(inputs, selection)

    if options.overview is CameraSourceMode.SIM:
        assert simulation is not None
        overview_factory = _synthetic_camera_factory(
            simulation, CameraRole.OVERVIEW
        )
        overview_calibration = diagnostic_overview_calibration(
            width=DEFAULT_SYNTHETIC_WIDTH,
            height=DEFAULT_SYNTHETIC_HEIGHT,
        )

    left_sim = options.stereo_left is CameraSourceMode.SIM
    right_sim = options.stereo_right is StereoRightMode.SIM
    if left_sim or right_sim:
        assert simulation is not None
        reference = inputs.stereo_calibration
        stereo_calibration = rectified_synthetic_side_calibration(
            reference,
            left_synthetic=left_sim,
            right_synthetic=right_sim,
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
