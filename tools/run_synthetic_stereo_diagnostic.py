"""Run deterministic 100/150/200 m stereo diagnostics without hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from navmin.calibration import load_stereo_calibration
from navmin.contracts import (
    CameraRole,
    FramePacket,
    TargetRef,
    TrackedObject,
    VisionResult,
)
from navmin.diagnostics.synthetic_stereo import (
    SyntheticWorkingRangeStereoScene,
    rectified_synthetic_calibration,
)
from navmin.vision.stereo_distance import (
    StereoDistanceEstimate,
    StereoSgbmSettings,
    compute_sgbm_disparity,
    estimate_distance_from_disparity_roi,
)
from navmin.vision.stereo_target_join import StereoTargetJoiner
from navmin.vision.stereo_timestamp_pairing import TimestampStereoPairer


class DiagnosticFailure(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check stereo pairing, target binding and SGBM at 100/150/200 m.",
    )
    parser.add_argument(
        "--stereo-calibration",
        type=Path,
        default=Path("config/calibration/stereo.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/synthetic-stereo"),
    )
    return parser


def _run_mode(*, calibration, moving: bool, output_dir: Path) -> None:
    scene = SyntheticWorkingRangeStereoScene(calibration, moving=moving)
    settings = StereoSgbmSettings()
    mode = "moving" if moving else "static"
    frame_indices = (0, 20, 40) if moving else (0,)
    pairer = TimestampStereoPairer(
        right_frame_buffer_size=4,
        pair_timeout_ms=100,
        max_timestamp_skew_ms=30,
        clock_ns=lambda: 0,
    )
    pairer.start_generation(CameraRole.STEREO_LEFT, 1)
    pairer.start_generation(CameraRole.STEREO_RIGHT, 1)

    print(f"[{mode.upper()}]")
    for frame_index in frame_indices:
        rendered = scene.render(frame_index)
        timestamp_ns = 1_000_000_000 + frame_index * 50_000_000
        left = FramePacket(
            camera=CameraRole.STEREO_LEFT,
            generation=1,
            frame_id=frame_index + 1,
            capture_id=None,
            receive_timestamp_ns=timestamp_ns,
            image=rendered.left,
        )
        right = FramePacket(
            camera=CameraRole.STEREO_RIGHT,
            generation=1,
            frame_id=frame_index + 1,
            capture_id=None,
            receive_timestamp_ns=timestamp_ns + 5_000_000,
            image=rendered.right,
        )
        assert pairer.offer_frame(left) is None
        pair = pairer.offer_frame(right)
        if pair is None:
            raise DiagnosticFailure("timestamp pairer did not emit a pair")

        disparity = compute_sgbm_disparity(pair.left.image, pair.right.image, settings)
        tracked = tuple(
            TrackedObject(
                track_id=target.spec.track_id,
                bbox=target.left_bbox,
                velocity_x_px_s=target.spec.speed_px_s if moving else 0.0,
                velocity_y_px_s=0.0,
                age_frames=frame_index + 1,
            )
            for target in rendered.targets
        )
        result = VisionResult(frame=left, tracked_objects=tracked, processing_time_ns=1)

        for target in rendered.targets:
            joiner = StereoTargetJoiner()
            joiner.start_generation(1)
            joiner.set_target(
                TargetRef(CameraRole.STEREO_LEFT, 1, target.spec.track_id),
                intent_revision=1,
            )
            joiner.offer_pair(pair)
            observation = joiner.offer_vision_result(result)
            if observation is None:
                raise DiagnosticFailure(
                    f"target binding failed for track {target.spec.track_id}"
                )
            estimate = estimate_distance_from_disparity_roi(
                observation.pair,
                calibration,
                disparity,
                observation.roi,
                length_scale_to_m=calibration.length_scale_to_m,
                min_valid_pixel_count=500,
            )
            if not isinstance(estimate, StereoDistanceEstimate):
                raise DiagnosticFailure(
                    f"distance failed for target {target.spec.track_id}: {estimate.reason.value}"
                )
            error_m = estimate.distance_m - target.spec.distance_m
            print(
                f"frame={frame_index:02d} target=T{target.spec.track_id} "
                f"nominal={target.spec.distance_m:.1f}m "
                f"speed={target.spec.speed_px_s if moving else 0.0:.1f}px/s "
                f"rendered_d={target.disparity_px}px "
                f"sgbm_d={estimate.median_disparity_px:.2f}px "
                f"distance={estimate.distance_m:.2f}m error={error_m:+.2f}m"
            )
            tolerance_m = max(5.0, target.spec.distance_m * 0.06)
            if abs(error_m) > tolerance_m:
                raise DiagnosticFailure(
                    f"target T{target.spec.track_id} error {error_m:+.2f} m exceeds "
                    f"{tolerance_m:.2f} m"
                )

        if frame_index == frame_indices[-1]:
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / f"{mode}_left.png"), rendered.left)
            cv2.imwrite(str(output_dir / f"{mode}_right.png"), rendered.right)

    diag = pairer.diagnostics
    print(
        f"pairs={diag.timestamp_pairs_emitted} "
        f"skew_p50_ms={diag.skew.p50_ns / 1_000_000:.3f} "
        f"skew_p95_ms={diag.skew.p95_ns / 1_000_000:.3f}"
    )


def main() -> int:
    args = _parser().parse_args()
    reference = load_stereo_calibration(args.stereo_calibration)
    calibration = rectified_synthetic_calibration(reference)
    _run_mode(calibration=calibration, moving=False, output_dir=args.output_dir)
    _run_mode(calibration=calibration, moving=True, output_dir=args.output_dir)
    print("SYNTHETIC STEREO DIAGNOSTIC: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
