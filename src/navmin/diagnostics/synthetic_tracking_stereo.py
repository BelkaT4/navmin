"""Production stereo-path proof over the synthetic tracking world.

The diagnostic keeps the synthetic target centered with a perfect virtual camera
pose so one full world trajectory can exercise the exact stereo pairer, VT11
Stereo Left tracking, target join, and production SGBM distance path without
mixing in closed-loop pointing error.  The Stage 3 closed-loop diagnostic remains
the owner of control-loop behaviour.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from navmin.calibration import load_stereo_calibration
from navmin.contracts import (
    CameraRole,
    FramePacket,
    TargetAcquired,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
)
from navmin.vision.processors.vt11.processor import Vt11VisionProcessor
from navmin.vision.stereo_distance import (
    StereoDistanceEstimate,
    StereoSgbmSettings,
    estimate_distance_from_stereo_pair,
)
from navmin.vision.stereo_pairing import StereoPairer
from navmin.vision.stereo_target_join import StereoTargetJoiner

from .synthetic_stereo import rectified_synthetic_calibration
from .synthetic_tracking_world import SyntheticTrackingWorld


class _StereoSimulationClock:
    def __init__(self) -> None:
        self._nanoseconds = 0

    def nanoseconds(self) -> int:
        return self._nanoseconds

    def set_frame(self, frame_index: int, fps: int) -> None:
        self._nanoseconds = round(frame_index * 1_000_000_000.0 / fps)


@dataclass(frozen=True, slots=True)
class SyntheticTrackingStereoDistanceSample:
    frame_index: int
    phase: str
    capture_id: int
    expected_distance_m: float
    measured_distance_m: float
    distance_error_m: float
    expected_disparity_px: float
    measured_disparity_px: float
    valid_pixel_count: int
    valid_pixel_fraction: float


@dataclass(frozen=True, slots=True)
class SyntheticTrackingStereoResult:
    status: str
    fps: int
    frames: int
    duration_s: float
    target_acquired_frame: int | None
    target_end_frame: int | None
    missing_tracking_frames: int
    exact_pairs: int
    target_observations: int
    distance_attempts: int
    distance_successes: int
    mean_abs_distance_error_m: float | None
    max_abs_distance_error_m: float | None
    samples: tuple[SyntheticTrackingStereoDistanceSample, ...]


def run_synthetic_tracking_stereo_diagnostic(
    *,
    stereo_calibration_path: str | Path = "config/calibration/stereo.json",
    frame_count: int | None = None,
    selection_frame: int = 8,
    distance_stride: int = 100,
) -> SyntheticTrackingStereoResult:
    """Exercise exact pairing, target association, and distance end-to-end.

    The virtual camera pose follows the ground-truth target exactly.  This keeps
    the narrow Stereo Left/Right target in view while the full ±30° world
    trajectory moves the background.  Ground truth is used only to position the
    synthetic camera; it is never passed to VT11, pairing, joining, or SGBM.
    """
    reference = load_stereo_calibration(stereo_calibration_path)
    calibration = rectified_synthetic_calibration(reference)
    world = SyntheticTrackingWorld(reference)
    total_frames = (
        round(world.cycle_duration_s * world.fps)
        if frame_count is None
        else frame_count
    )
    if total_frames <= 0:
        raise ValueError("frame_count must be > 0")
    if not 0 <= selection_frame < total_frames:
        raise ValueError("selection_frame must address a generated frame")
    if distance_stride <= 0:
        raise ValueError("distance_stride must be > 0")

    clock = _StereoSimulationClock()
    processor = Vt11VisionProcessor(
        profile_name="day",
        duration_clock_ns=clock.nanoseconds,
        history_clock_ns=clock.nanoseconds,
    )
    pairer = StereoPairer(
        right_frame_buffer_size=4,
        pair_timeout_ms=100,
        clock_ns=clock.nanoseconds,
    )
    pairer.start_generation(CameraRole.STEREO_LEFT, 1)
    pairer.start_generation(CameraRole.STEREO_RIGHT, 1)
    joiner = StereoTargetJoiner()
    joiner.start_generation(1)

    selected_target: TargetRef | None = None
    acquired_frame: int | None = None
    end_frame: int | None = None
    missing_tracking_frames = 0
    distance_attempts = 0
    samples: list[SyntheticTrackingStereoDistanceSample] = []
    settings = StereoSgbmSettings()

    for frame_index in range(total_frames):
        clock.set_frame(frame_index, world.fps)
        truth = world.truth(frame_index)
        frame_set = world.render(
            frame_index,
            turret_yaw_deg=truth.target_yaw_deg,
            turret_pitch_deg=truth.target_pitch_deg,
        )
        timestamp_ns = clock.nanoseconds()
        left_packet = FramePacket(
            camera=CameraRole.STEREO_LEFT,
            generation=1,
            frame_id=frame_index,
            capture_id=frame_set.capture_id,
            receive_timestamp_ns=timestamp_ns,
            image=frame_set.stereo_left,
        )
        right_packet = FramePacket(
            camera=CameraRole.STEREO_RIGHT,
            generation=1,
            frame_id=frame_index,
            capture_id=frame_set.capture_id,
            receive_timestamp_ns=timestamp_ns,
            image=frame_set.stereo_right,
        )
        output = processor.process(left_packet)

        if frame_index == selection_frame:
            projection = frame_set.projection(CameraRole.STEREO_LEFT)
            events = processor.apply_target_control(
                TargetSelectionRequest(
                    selection_id=1,
                    camera=CameraRole.STEREO_LEFT,
                    generation=1,
                    frame_id=frame_index,
                    x_px=projection.center_x_px,
                    y_px=projection.center_y_px,
                )
            )
            for event in events:
                if isinstance(event, TargetAcquired):
                    selected_target = event.target
                    acquired_frame = frame_index
                    joiner.set_target(selected_target)

        for event in output.target_events:
            if isinstance(event, TargetAcquired):
                selected_target = event.target
                acquired_frame = frame_index
                joiner.set_target(selected_target)
            elif isinstance(event, TargetTrackingEnded) and selected_target == event.target:
                selected_target = None
                end_frame = frame_index
                joiner.set_target(None)

        if selected_target is not None and not any(
            item.track_id == selected_target.track_id
            for item in output.result.tracked_objects
        ):
            missing_tracking_frames += 1

        # Offer Right first to prove the production exact pairer handles the
        # common asynchronous arrival order. Both frames share one capture_id.
        pairer.offer_frame(right_packet)
        pair = pairer.offer_frame(left_packet)
        observation = None
        if pair is not None:
            observation = joiner.offer_pair(pair)
        joined_from_result = joiner.offer_vision_result(output.result)
        if joined_from_result is not None:
            observation = joined_from_result

        if observation is None or frame_index % distance_stride != 0:
            continue
        distance_attempts += 1
        outcome = estimate_distance_from_stereo_pair(
            observation.pair,
            calibration,
            settings,
            length_scale_to_m=calibration.length_scale_to_m,
            min_valid_pixel_count=5,
            roi=observation.roi,
        )
        if not isinstance(outcome, StereoDistanceEstimate):
            continue
        error_m = outcome.distance_m - world.target_distance_m
        samples.append(
            SyntheticTrackingStereoDistanceSample(
                frame_index=frame_index,
                phase=truth.phase,
                capture_id=frame_set.capture_id,
                expected_distance_m=world.target_distance_m,
                measured_distance_m=outcome.distance_m,
                distance_error_m=error_m,
                expected_disparity_px=world.disparity_px,
                measured_disparity_px=outcome.median_disparity_px,
                valid_pixel_count=outcome.valid_pixel_count,
                valid_pixel_fraction=outcome.valid_pixel_fraction,
            )
        )

    errors = [abs(item.distance_error_m) for item in samples]
    pair_diagnostics = pairer.diagnostics
    join_diagnostics = joiner.diagnostics
    status = "PASS"
    if (
        acquired_frame is None
        or end_frame is not None
        or pair_diagnostics.exact_pairs_emitted != total_frames
        or not samples
        or len(samples) != distance_attempts
    ):
        status = "FAIL"

    return SyntheticTrackingStereoResult(
        status=status,
        fps=world.fps,
        frames=total_frames,
        duration_s=total_frames / world.fps,
        target_acquired_frame=acquired_frame,
        target_end_frame=end_frame,
        missing_tracking_frames=missing_tracking_frames,
        exact_pairs=pair_diagnostics.exact_pairs_emitted,
        target_observations=join_diagnostics.target_observations_emitted,
        distance_attempts=distance_attempts,
        distance_successes=len(samples),
        mean_abs_distance_error_m=None if not errors else fmean(errors),
        max_abs_distance_error_m=None if not errors else max(errors),
        samples=tuple(samples),
    )


def write_synthetic_tracking_stereo_json(
    result: SyntheticTrackingStereoResult,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "SyntheticTrackingStereoDistanceSample",
    "SyntheticTrackingStereoResult",
    "run_synthetic_tracking_stereo_diagnostic",
    "write_synthetic_tracking_stereo_json",
]
