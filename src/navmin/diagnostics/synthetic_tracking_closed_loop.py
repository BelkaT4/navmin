"""Headless closed-loop diagnostic over the synthetic tracking world.

The diagnostic deliberately reuses the production VT11, Aiming,
TrackingEstimator, TurretController, TurretHal and protocol session.  Only the
camera source and physical turret are synthetic.  Ground truth never enters the
production control path; it is retained only for diagnostics and assertions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import atan2, cos, degrees, hypot, radians, sin
from pathlib import Path
from statistics import fmean

from navmin.calibration import load_stereo_calibration
from navmin.config import load_config
from navmin.contracts import (
    AxisVelocitySetpoint,
    CameraRay,
    CameraRole,
    FramePacket,
    TargetAcquired,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TrackingControlSample,
    TurretControlMode,
)
from navmin.core.aiming import Aiming
from navmin.core.tracking_estimator import TrackingEstimator
from navmin.turret.controller import TurretController
from navmin.turret.hal import TurretHal
from navmin.turret.session import TurretSession
from navmin.turret.simulator import FakeTransport
from navmin.vision.processors.vt11.processor import Vt11VisionProcessor

from .synthetic_tracking_turret import (
    SyntheticTrackingSimulation,
    SyntheticTrackingStm32Endpoint,
    SyntheticTurretPlant,
)
from .synthetic_tracking_world import (
    SyntheticTrackingCameraGeometry,
    SyntheticTrackingWorld,
    stereo_fov_deg,
)


class _SimulationClock:
    def __init__(self) -> None:
        self._seconds = 0.0

    def seconds(self) -> float:
        return self._seconds

    def nanoseconds(self) -> int:
        return round(self._seconds * 1_000_000_000.0)

    def advance(self, dt_s: float) -> None:
        self._seconds += dt_s


class SyntheticAngularCameraModel:
    """CameraModel matching the world's intentionally linear deg-to-pixel map."""

    def __init__(self, geometry: SyntheticTrackingCameraGeometry) -> None:
        if not isinstance(geometry, SyntheticTrackingCameraGeometry):
            raise TypeError("geometry must be SyntheticTrackingCameraGeometry")
        self._center_x_px = geometry.center_x_px + geometry.stereo_offset_px
        self._center_y_px = geometry.center_y_px
        self._pixels_per_degree_x = geometry.pixels_per_degree_x
        self._pixels_per_degree_y = geometry.pixels_per_degree_y

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        yaw_rad = radians(
            (float(x_px) - self._center_x_px) / self._pixels_per_degree_x
        )
        pitch_rad = radians(
            -(float(y_px) - self._center_y_px) / self._pixels_per_degree_y
        )
        cos_pitch = cos(pitch_rad)
        return CameraRay(
            x=sin(yaw_rad) * cos_pitch,
            y=-sin(pitch_rad),
            z=cos(yaw_rad) * cos_pitch,
        )

    def ray_to_pixel(self, ray: CameraRay) -> tuple[float, float]:
        if not isinstance(ray, CameraRay):
            raise TypeError("ray must be CameraRay")
        yaw_deg = degrees(atan2(ray.x, ray.z))
        pitch_deg = -degrees(atan2(ray.y, hypot(ray.x, ray.z)))
        return (
            self._center_x_px + yaw_deg * self._pixels_per_degree_x,
            self._center_y_px - pitch_deg * self._pixels_per_degree_y,
        )


@dataclass(frozen=True, slots=True)
class SyntheticClosedLoopFrame:
    frame_index: int
    time_s: float
    phase: str
    target_visible: bool
    target_acquired: bool
    tracked: bool
    rate_valid_x: bool
    rate_valid_y: bool
    target_yaw_deg: float
    target_pitch_deg: float
    target_yaw_rate_deg_s: float
    target_pitch_rate_deg_s: float
    turret_yaw_deg: float
    turret_pitch_deg: float
    turret_yaw_rate_deg_s: float
    turret_pitch_rate_deg_s: float
    true_error_yaw_deg: float
    true_error_pitch_deg: float
    observed_error_x_deg: float | None
    observed_error_y_deg: float | None
    estimated_target_rate_x_deg_s: float | None
    estimated_target_rate_y_deg_s: float | None
    commanded_velocity_x_deg_s: float | None
    commanded_velocity_y_deg_s: float | None


@dataclass(frozen=True, slots=True)
class SyntheticClosedLoopPhaseSummary:
    phase: str
    frames: int
    acquired_frames: int
    tracked_frames: int
    rate_valid_x_frames: int
    rate_valid_y_frames: int
    mean_abs_error_yaw_deg: float
    p95_abs_error_yaw_deg: float
    mean_abs_error_pitch_deg: float
    p95_abs_error_pitch_deg: float
    mean_abs_target_rate_error_x_deg_s: float | None
    mean_abs_target_rate_error_y_deg_s: float | None


@dataclass(frozen=True, slots=True)
class SyntheticClosedLoopResult:
    status: str
    fps: int
    true_stereo_hfov_deg: float
    tracking_stereo_hfov_deg: float
    frames: int
    duration_s: float
    target_acquired_frame: int | None
    target_end_frame: int | None
    missing_tracking_frames: int
    rate_valid_x_frames: int
    rate_valid_y_frames: int
    phase_summaries: tuple[SyntheticClosedLoopPhaseSummary, ...]
    samples: tuple[SyntheticClosedLoopFrame, ...]


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def _phase_summary(
    phase: str,
    frames: list[SyntheticClosedLoopFrame],
) -> SyntheticClosedLoopPhaseSummary:
    acquired = [item for item in frames if item.target_acquired]
    tracked = [item for item in frames if item.tracked]
    yaw_errors = [abs(item.true_error_yaw_deg) for item in acquired]
    pitch_errors = [abs(item.true_error_pitch_deg) for item in acquired]
    rate_x_errors = [
        abs(item.estimated_target_rate_x_deg_s - item.target_yaw_rate_deg_s)
        for item in tracked
        if item.rate_valid_x and item.estimated_target_rate_x_deg_s is not None
    ]
    rate_y_errors = [
        abs(item.estimated_target_rate_y_deg_s - item.target_pitch_rate_deg_s)
        for item in tracked
        if item.rate_valid_y and item.estimated_target_rate_y_deg_s is not None
    ]
    return SyntheticClosedLoopPhaseSummary(
        phase=phase,
        frames=len(frames),
        acquired_frames=len(acquired),
        tracked_frames=len(tracked),
        rate_valid_x_frames=sum(item.rate_valid_x for item in tracked),
        rate_valid_y_frames=sum(item.rate_valid_y for item in tracked),
        mean_abs_error_yaw_deg=0.0 if not yaw_errors else fmean(yaw_errors),
        p95_abs_error_yaw_deg=_percentile95(yaw_errors),
        mean_abs_error_pitch_deg=0.0 if not pitch_errors else fmean(pitch_errors),
        p95_abs_error_pitch_deg=_percentile95(pitch_errors),
        mean_abs_target_rate_error_x_deg_s=(
            None if not rate_x_errors else fmean(rate_x_errors)
        ),
        mean_abs_target_rate_error_y_deg_s=(
            None if not rate_y_errors else fmean(rate_y_errors)
        ),
    )


def run_synthetic_tracking_closed_loop(
    *,
    config_path: str | Path = "config/navmin.json",
    stereo_calibration_path: str | Path = "config/calibration/stereo.json",
    frame_count: int | None = None,
    selection_frame: int = 8,
    sample_stride: int = 1,
    true_stereo_hfov_deg: float | None = None,
    tracking_stereo_hfov_deg: float | None = None,
) -> SyntheticClosedLoopResult:
    """Run one deterministic headless closed-loop tracking diagnostic."""
    config = load_config(config_path)
    stereo_calibration = load_stereo_calibration(stereo_calibration_path)
    world = SyntheticTrackingWorld(
        stereo_calibration,
        stereo_true_hfov_deg=true_stereo_hfov_deg,
    )
    total_frames = (
        round(world.cycle_duration_s * world.fps)
        if frame_count is None
        else frame_count
    )
    if total_frames <= 0:
        raise ValueError("frame_count must be > 0")
    if not 0 <= selection_frame < total_frames:
        raise ValueError("selection_frame must address a generated frame")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be > 0")

    clock = _SimulationClock()
    plant = SyntheticTurretPlant(config.turret)
    endpoint = SyntheticTrackingStm32Endpoint(plant)
    transport = FakeTransport(endpoint)
    transport.open()
    session = TurretSession(
        transport,
        response_timeout_s=0.01,
        max_retries=0,
        inter_request_delay_s=0.0,
        clock=clock.seconds,
        wait_hook=lambda _seconds: None,
    )
    hal = TurretHal(session, config.turret, command_clock_ns=clock.nanoseconds)
    controller = TurretController(
        hal,
        config.turret,
        config.tracking_control,
        clock=clock.seconds,
    )

    # Mirror the safe startup state without involving ApplicationRuntime/Qt.
    hal.motor_off()
    hal.sync_stm32_config()
    controller.set_control_mode(TurretControlMode.TRACKING)
    controller.motor_on()

    simulation = SyntheticTrackingSimulation(world, plant)
    processor = Vt11VisionProcessor(
        profile_name="day",
        duration_clock_ns=clock.nanoseconds,
        history_clock_ns=clock.nanoseconds,
    )
    aiming = Aiming(config.aiming)
    estimator = TrackingEstimator(config.tracking_control)
    camera = CameraRole.STEREO_LEFT
    true_geometry = world.geometries[camera]
    calibration_hfov, calibration_vfov = stereo_fov_deg(stereo_calibration)
    if tracking_stereo_hfov_deg is None:
        tracking_hfov = calibration_hfov
        tracking_vfov = calibration_vfov
    else:
        tracking_hfov = float(tracking_stereo_hfov_deg)
        if not 0.0 < tracking_hfov < 180.0:
            raise ValueError("tracking_stereo_hfov_deg must be in (0, 180)")
        tracking_vfov = tracking_hfov * true_geometry.height / true_geometry.width
    tracking_geometry = replace(
        true_geometry,
        horizontal_fov_deg=tracking_hfov,
        vertical_fov_deg=tracking_vfov,
    )
    camera_model = SyntheticAngularCameraModel(tracking_geometry)

    selected_target: TargetRef | None = None
    acquired_frame: int | None = None
    end_frame: int | None = None
    revision = 0
    missing_tracking_frames = 0
    rate_valid_x_frames = 0
    rate_valid_y_frames = 0
    samples: list[SyntheticClosedLoopFrame] = []

    for frame_index in range(total_frames):
        snapshot = simulation.capture()
        frame_set = snapshot.frame_set
        truth = frame_set.truth
        projection = frame_set.projection(camera)
        packet = FramePacket(
            camera=camera,
            generation=1,
            frame_id=frame_index,
            capture_id=frame_set.capture_id,
            receive_timestamp_ns=clock.nanoseconds(),
            image=frame_set.image(camera),
        )
        output = processor.process(packet)

        if frame_index == selection_frame:
            selection_events = processor.apply_target_control(
                TargetSelectionRequest(
                    selection_id=1,
                    camera=camera,
                    generation=1,
                    frame_id=frame_index,
                    x_px=projection.center_x_px,
                    y_px=projection.center_y_px,
                )
            )
            for event in selection_events:
                if isinstance(event, TargetAcquired):
                    selected_target = event.target
                    acquired_frame = frame_index

        for event in output.target_events:
            if isinstance(event, TargetAcquired):
                selected_target = event.target
                acquired_frame = frame_index
            elif isinstance(event, TargetTrackingEnded):
                if selected_target == event.target:
                    selected_target = None
                    end_frame = frame_index
                    estimator.reset()
                    revision += 1
                    controller.invalidate_tracking_control(revision)

        tracked = None
        observed_error_x = None
        observed_error_y = None
        estimated_rate_x = None
        estimated_rate_y = None
        rate_valid_x = False
        rate_valid_y = False
        command_x = None
        command_y = None

        if selected_target is not None:
            tracked = next(
                (
                    item
                    for item in output.result.tracked_objects
                    if item.track_id == selected_target.track_id
                ),
                None,
            )
            revision += 1
            if tracked is None:
                missing_tracking_frames += 1
                estimator.reset_history()
                controller.invalidate_tracking_control(revision)
            else:
                error = aiming.tracking_error(
                    camera_model=camera_model,
                    frame=packet,
                    tracked=tracked,
                    target=selected_target,
                )
                diagnostics = estimator.observe(error, hal.platform_motion_snapshot)
                observed_error_x = error.error_x_deg
                observed_error_y = error.error_y_deg
                estimated_rate_x = diagnostics.x.target_rate_deg_s
                estimated_rate_y = diagnostics.y.target_rate_deg_s
                rate_valid_x = diagnostics.x.valid
                rate_valid_y = diagnostics.y.valid
                rate_valid_x_frames += int(rate_valid_x)
                rate_valid_y_frames += int(rate_valid_y)
                accepted = controller.submit_tracking_control(
                    revision,
                    TrackingControlSample(
                        target=selected_target,
                        real_error_x_deg=error.error_x_deg,
                        real_error_y_deg=error.error_y_deg,
                        target_rate_x_deg_s=diagnostics.x.target_rate_deg_s,
                        target_rate_y_deg_s=diagnostics.y.target_rate_deg_s,
                        prediction_offset_x_deg=diagnostics.x.prediction_offset_deg,
                        prediction_offset_y_deg=diagnostics.y.prediction_offset_deg,
                        rate_valid_x=diagnostics.x.valid,
                        rate_valid_y=diagnostics.y.valid,
                        observation_timestamp_ns=error.timestamp_ns,
                    ),
                )
                if accepted:
                    pending = controller.pending_motion
                    if isinstance(pending, AxisVelocitySetpoint):
                        command_x = pending.velocity_x_deg_s
                        command_y = pending.velocity_y_deg_s
                    controller.flush_pending_motion()

        if frame_index % sample_stride == 0:
            turret = plant.state
            samples.append(
                SyntheticClosedLoopFrame(
                    frame_index=frame_index,
                    time_s=clock.seconds(),
                    phase=truth.phase,
                    target_visible=projection.visible,
                    target_acquired=selected_target is not None,
                    tracked=tracked is not None,
                    rate_valid_x=rate_valid_x,
                    rate_valid_y=rate_valid_y,
                    target_yaw_deg=truth.target_yaw_deg,
                    target_pitch_deg=truth.target_pitch_deg,
                    target_yaw_rate_deg_s=truth.target_yaw_rate_deg_s,
                    target_pitch_rate_deg_s=truth.target_pitch_rate_deg_s,
                    turret_yaw_deg=turret.yaw_deg,
                    turret_pitch_deg=turret.pitch_deg,
                    turret_yaw_rate_deg_s=turret.yaw_rate_deg_s,
                    turret_pitch_rate_deg_s=turret.pitch_rate_deg_s,
                    true_error_yaw_deg=truth.true_error_yaw_deg,
                    true_error_pitch_deg=truth.true_error_pitch_deg,
                    observed_error_x_deg=observed_error_x,
                    observed_error_y_deg=observed_error_y,
                    estimated_target_rate_x_deg_s=estimated_rate_x,
                    estimated_target_rate_y_deg_s=estimated_rate_y,
                    commanded_velocity_x_deg_s=command_x,
                    commanded_velocity_y_deg_s=command_y,
                )
            )

        simulation.advance()
        clock.advance(1.0 / world.fps)

    phases: dict[str, list[SyntheticClosedLoopFrame]] = {}
    for sample in samples:
        phases.setdefault(sample.phase, []).append(sample)
    phase_summaries = tuple(
        _phase_summary(phase, phase_frames)
        for phase, phase_frames in phases.items()
    )
    return SyntheticClosedLoopResult(
        status="COMPLETE",
        fps=world.fps,
        true_stereo_hfov_deg=true_geometry.horizontal_fov_deg,
        tracking_stereo_hfov_deg=tracking_geometry.horizontal_fov_deg,
        frames=total_frames,
        duration_s=total_frames / world.fps,
        target_acquired_frame=acquired_frame,
        target_end_frame=end_frame,
        missing_tracking_frames=missing_tracking_frames,
        rate_valid_x_frames=rate_valid_x_frames,
        rate_valid_y_frames=rate_valid_y_frames,
        phase_summaries=phase_summaries,
        samples=tuple(samples),
    )


def write_synthetic_tracking_closed_loop_json(
    result: SyntheticClosedLoopResult,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "SyntheticAngularCameraModel",
    "SyntheticClosedLoopFrame",
    "SyntheticClosedLoopPhaseSummary",
    "SyntheticClosedLoopResult",
    "run_synthetic_tracking_closed_loop",
    "write_synthetic_tracking_closed_loop_json",
]
