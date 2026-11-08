"""Read-only optical-flow calibration of effective video latency.

The calibration intentionally estimates *effective* latency relative to the
existing command-model platform motion.  It does not claim to separate camera,
encode, transport, decode, mechanical-response, or command-model errors.
Nothing in this module changes tracking/control behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

import cv2
import numpy as np

from navmin.contracts import BBox, CameraRay, PlatformMotionSnapshot
from navmin.turret.platform_motion import CommandMotionModel

_DEFAULT_FPS = 20
_DEFAULT_WIDTH = 240
_DEFAULT_HEIGHT = 180
_DEFAULT_FOCAL_LENGTH_PX = 700.0
_DEFAULT_KNOWN_LATENCIES_MS = (140, 145, 150, 155, 160)
_BASE_TIMESTAMP_NS = 1_000_000_000


class VideoLatencyCalibrationError(RuntimeError):
    """A required calibration observation cannot be produced."""


@dataclass(frozen=True, slots=True)
class OpticalFlowLatencyConfig:
    search_min_ms: int = 0
    search_max_ms: int = 300
    search_step_ms: int = 5
    max_corners: int = 220
    quality_level: float = 0.01
    min_distance_px: float = 5.0
    min_inliers: int = 20
    ransac_threshold_px: float = 2.0
    exclusion_margin_px: int = 8
    min_observations: int = 30
    min_excitation_std_deg_s: float = 1.0
    min_correlation: float = 0.90
    max_normalized_rmse: float = 0.35

    def validate(self) -> None:
        if self.search_min_ms < 0:
            raise ValueError("search_min_ms must be >= 0")
        if self.search_max_ms <= self.search_min_ms:
            raise ValueError("search_max_ms must be > search_min_ms")
        if self.search_step_ms <= 0:
            raise ValueError("search_step_ms must be > 0")
        if self.max_corners < 20:
            raise ValueError("max_corners must be >= 20")
        if not 0.0 < self.quality_level < 1.0:
            raise ValueError("quality_level must be in (0, 1)")
        if self.min_distance_px <= 0.0:
            raise ValueError("min_distance_px must be > 0")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be >= 4")
        if self.ransac_threshold_px <= 0.0:
            raise ValueError("ransac_threshold_px must be > 0")
        if self.exclusion_margin_px < 0:
            raise ValueError("exclusion_margin_px must be >= 0")
        if self.min_observations < 4:
            raise ValueError("min_observations must be >= 4")
        if self.min_excitation_std_deg_s <= 0.0:
            raise ValueError("min_excitation_std_deg_s must be > 0")
        if not 0.0 <= self.min_correlation <= 1.0:
            raise ValueError("min_correlation must be in [0, 1]")
        if self.max_normalized_rmse <= 0.0:
            raise ValueError("max_normalized_rmse must be > 0")


@dataclass(frozen=True, slots=True)
class OpticalFlowRateObservation:
    receive_midpoint_timestamp_ns: int
    source_interval_ns: int
    background_rate_x_deg_s: float
    translation_x_px: float
    inliers: int
    tracked_points: int


@dataclass(frozen=True, slots=True)
class LatencyCandidateScore:
    latency_ms: int
    observation_count: int
    rmse_deg_s: float
    normalized_rmse: float
    correlation: float
    platform_rate_std_deg_s: float


@dataclass(frozen=True, slots=True)
class VideoLatencyEstimate:
    valid: bool
    reason: str | None
    estimated_latency_ms: int | None
    observation_count: int
    best_score: LatencyCandidateScore | None
    candidates: tuple[LatencyCandidateScore, ...]


@dataclass(frozen=True, slots=True)
class SyntheticLatencyRunResult:
    known_latency_ms: int
    estimated_latency_ms: int | None
    absolute_error_ms: int | None
    valid: bool
    reason: str | None
    flow_observations: int
    median_flow_inliers: float
    best_rmse_deg_s: float | None
    best_correlation: float | None


@dataclass(frozen=True, slots=True)
class VideoLatencyAggregate:
    valid_run_count: int
    mean_latency_ms: float | None
    median_latency_ms: float | None
    stddev_latency_ms: float | None
    mad_latency_ms: float | None
    min_latency_ms: int | None
    max_latency_ms: int | None


@dataclass(frozen=True, slots=True)
class VideoLatencyCalibrationProofResult:
    status: str
    fps: int
    search_range_ms: tuple[int, int]
    search_step_ms: int
    runs: tuple[SyntheticLatencyRunResult, ...]
    aggregate: VideoLatencyAggregate
    known_latency_mean_ms: float
    estimated_latency_mean_error_ms: float | None
    max_absolute_estimation_error_ms: int | None
    insufficient_excitation_rejected: bool
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _SyntheticCameraModel:
    width: int
    height: int
    focal_length_px: float

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        x = (float(x_px) - cx) / self.focal_length_px
        y = (float(y_px) - cy) / self.focal_length_px
        norm = hypot(hypot(x, y), 1.0)
        return CameraRay(x / norm, y / norm, 1.0 / norm)

    def ray_to_pixel(self, ray: CameraRay) -> tuple[float, float]:
        if ray.z == 0.0:
            raise ValueError("ray.z must be non-zero")
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        return (
            cx + self.focal_length_px * ray.x / ray.z,
            cy + self.focal_length_px * ray.y / ray.z,
        )


def _yaw_deg(ray: CameraRay) -> float:
    return degrees(atan2(ray.x, ray.z))


def _analysis_mask(
    shape: tuple[int, int],
    exclusions: tuple[BBox, ...],
    margin_px: int,
) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    border = max(2, round(0.02 * min(height, width)))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    for bbox in exclusions:
        x0 = max(0, bbox.x - margin_px)
        y0 = max(0, bbox.y - margin_px)
        x1 = min(width, bbox.x + bbox.width + margin_px)
        y1 = min(height, bbox.y + bbox.height + margin_px)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    return mask


def estimate_optical_flow_rate_x(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    previous_receive_timestamp_ns: int,
    current_receive_timestamp_ns: int,
    source_interval_ns: int,
    camera_model: Any,
    exclusions: tuple[BBox, ...] = (),
    config: OpticalFlowLatencyConfig | None = None,
) -> OpticalFlowRateObservation:
    """Estimate background angular velocity between two frames using LK + RANSAC.

    ``source_interval_ns`` must describe the camera/source frame interval.  The
    receive timestamps are used only to place the observation on the PC timeline;
    receive jitter must not distort the visual angular-rate denominator.
    """

    cfg = config or OpticalFlowLatencyConfig()
    cfg.validate()
    if previous_frame.shape != current_frame.shape:
        raise VideoLatencyCalibrationError("calibration frames must have equal shape")
    if previous_frame.ndim not in (2, 3):
        raise VideoLatencyCalibrationError("calibration frame must be gray or BGR")
    if type(source_interval_ns) is not int or source_interval_ns <= 0:
        raise ValueError("source_interval_ns must be a positive integer")
    if (
        type(previous_receive_timestamp_ns) is not int
        or type(current_receive_timestamp_ns) is not int
        or previous_receive_timestamp_ns < 0
        or current_receive_timestamp_ns <= previous_receive_timestamp_ns
    ):
        raise ValueError("receive timestamps must be non-negative and increasing")

    previous_gray = (
        cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        if previous_frame.ndim == 3
        else previous_frame
    )
    current_gray = (
        cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if current_frame.ndim == 3
        else current_frame
    )
    mask = _analysis_mask(previous_gray.shape, exclusions, cfg.exclusion_margin_px)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance_px,
        mask=mask,
        blockSize=5,
        useHarrisDetector=False,
    )
    if points is None or len(points) < cfg.min_inliers:
        raise VideoLatencyCalibrationError("insufficient background features")
    next_points, status, _errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        raise VideoLatencyCalibrationError("optical flow did not return tracked points")
    good = status.reshape(-1) > 0
    source_points = points.reshape(-1, 2)[good]
    destination_points = next_points.reshape(-1, 2)[good]
    if len(source_points) < cfg.min_inliers:
        raise VideoLatencyCalibrationError("insufficient tracked background features")

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_threshold_px,
        maxIters=500,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None or inlier_mask is None:
        raise VideoLatencyCalibrationError("RANSAC background transform failed")
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count < cfg.min_inliers:
        raise VideoLatencyCalibrationError("insufficient RANSAC background inliers")

    dx_px = float(transform[0, 2])
    height, width = previous_gray.shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    center_yaw = _yaw_deg(camera_model.pixel_to_ray(center_x, center_y))
    shifted_yaw = _yaw_deg(camera_model.pixel_to_ray(center_x + dx_px, center_y))
    angular_delta_deg = shifted_yaw - center_yaw
    interval_s = source_interval_ns / 1_000_000_000.0
    background_rate = angular_delta_deg / interval_s
    if not isfinite(background_rate):
        raise VideoLatencyCalibrationError("optical-flow angular rate is not finite")

    return OpticalFlowRateObservation(
        receive_midpoint_timestamp_ns=(
            previous_receive_timestamp_ns + current_receive_timestamp_ns
        )
        // 2,
        source_interval_ns=source_interval_ns,
        background_rate_x_deg_s=background_rate,
        translation_x_px=dx_px,
        inliers=inlier_count,
        tracked_points=len(source_points),
    )


def _candidate_score(
    observations: tuple[OpticalFlowRateObservation, ...],
    platform_motion: PlatformMotionSnapshot,
    latency_ms: int,
) -> LatencyCandidateScore | None:
    latency_ns = latency_ms * 1_000_000
    visual_world_rates: list[float] = []
    platform_rates: list[float] = []
    for observation in observations:
        query_timestamp_ns = observation.receive_midpoint_timestamp_ns - latency_ns
        if query_timestamp_ns < 0:
            continue
        motion = platform_motion.rate_at(query_timestamp_ns)
        if motion is None or motion.rate_x_deg_s is None:
            continue
        # A stationary background moves opposite to the camera/platform.
        visual_world_rates.append(-observation.background_rate_x_deg_s)
        platform_rates.append(float(motion.rate_x_deg_s))
    if len(platform_rates) < 2:
        return None

    visual = np.asarray(visual_world_rates, dtype=np.float64)
    platform = np.asarray(platform_rates, dtype=np.float64)
    residual = visual - platform
    rmse = float(np.sqrt(np.mean(residual * residual)))
    platform_std = float(np.std(platform))
    rms_signal = float(np.sqrt(np.mean(platform * platform)))
    normalized_rmse = rmse / max(rms_signal, 1e-12)
    if float(np.std(visual)) <= 1e-12 or platform_std <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(visual, platform)[0, 1])
    return LatencyCandidateScore(
        latency_ms=latency_ms,
        observation_count=len(platform_rates),
        rmse_deg_s=rmse,
        normalized_rmse=normalized_rmse,
        correlation=correlation,
        platform_rate_std_deg_s=platform_std,
    )


def estimate_video_latency(
    observations: tuple[OpticalFlowRateObservation, ...],
    platform_motion: PlatformMotionSnapshot,
    *,
    config: OpticalFlowLatencyConfig | None = None,
) -> VideoLatencyEstimate:
    """Estimate effective video latency relative to command-model motion."""

    cfg = config or OpticalFlowLatencyConfig()
    cfg.validate()
    if not platform_motion.valid:
        return VideoLatencyEstimate(False, "platform-motion-invalid", None, 0, None, ())
    if len(observations) < cfg.min_observations:
        return VideoLatencyEstimate(
            False,
            "insufficient-observations",
            None,
            len(observations),
            None,
            (),
        )

    candidates = tuple(
        score
        for latency_ms in range(
            cfg.search_min_ms,
            cfg.search_max_ms + 1,
            cfg.search_step_ms,
        )
        if (
            score := _candidate_score(observations, platform_motion, latency_ms)
        )
        is not None
    )
    if not candidates:
        return VideoLatencyEstimate(
            False,
            "platform-motion-unavailable",
            None,
            len(observations),
            None,
            (),
        )
    best = min(candidates, key=lambda item: (item.rmse_deg_s, -item.correlation))
    if best.observation_count < cfg.min_observations:
        reason = "insufficient-observations"
    elif best.platform_rate_std_deg_s < cfg.min_excitation_std_deg_s:
        reason = "insufficient-excitation"
    elif best.correlation < cfg.min_correlation:
        reason = "low-correlation"
    elif best.normalized_rmse > cfg.max_normalized_rmse:
        reason = "high-fit-error"
    else:
        reason = None
    return VideoLatencyEstimate(
        valid=reason is None,
        reason=reason,
        estimated_latency_ms=best.latency_ms if reason is None else None,
        observation_count=best.observation_count,
        best_score=best,
        candidates=candidates,
    )


def aggregate_latency_estimates(
    estimates: tuple[VideoLatencyEstimate, ...],
) -> VideoLatencyAggregate:
    values = [
        estimate.estimated_latency_ms
        for estimate in estimates
        if estimate.valid and estimate.estimated_latency_ms is not None
    ]
    if not values:
        return VideoLatencyAggregate(0, None, None, None, None, None, None)
    value_array = np.asarray(values, dtype=np.float64)
    median_value = float(median(values))
    mad = float(median(abs(value - median_value) for value in values))
    return VideoLatencyAggregate(
        valid_run_count=len(values),
        mean_latency_ms=float(fmean(values)),
        median_latency_ms=median_value,
        stddev_latency_ms=float(pstdev(values)) if len(values) > 1 else 0.0,
        mad_latency_ms=mad,
        min_latency_ms=int(np.min(value_array)),
        max_latency_ms=int(np.max(value_array)),
    )


def _synthetic_platform_motion() -> PlatformMotionSnapshot:
    model = CommandMotionModel(
        acceleration_x_deg_s2=40.0,
        acceleration_y_deg_s2=40.0,
    )
    model.mark_known_zero(timestamp_ns=_BASE_TIMESTAMP_NS)
    for time_s, rate_deg_s in (
        (0.50, 8.0),
        (1.30, -8.0),
        (2.10, 8.0),
        (2.90, -8.0),
        (3.70, 0.0),
    ):
        model.record_velocity_command(
            timestamp_ns=_BASE_TIMESTAMP_NS + round(time_s * 1_000_000_000),
            commanded_rate_x_deg_s=rate_deg_s,
            commanded_rate_y_deg_s=0.0,
        )
    return model.snapshot()


def _platform_rate_x(
    platform_motion: PlatformMotionSnapshot,
    time_s: float,
) -> float:
    rate = platform_motion.rate_at(
        _BASE_TIMESTAMP_NS + round(time_s * 1_000_000_000)
    )
    return 0.0 if rate is None or rate.rate_x_deg_s is None else rate.rate_x_deg_s


def _integrated_platform_angles(
    platform_motion: PlatformMotionSnapshot,
    *,
    duration_s: float,
    integration_step_s: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, duration_s + integration_step_s, integration_step_s)
    rates = np.asarray(
        [_platform_rate_x(platform_motion, float(time_s)) for time_s in times],
        dtype=np.float64,
    )
    angles = np.zeros_like(times)
    angles[1:] = np.cumsum(
        (rates[:-1] + rates[1:]) * (integration_step_s / 2.0)
    )
    return times, angles


def _synthetic_texture(*, seed: int, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = np.clip(rng.normal(127.0, 48.0, (height, width)), 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    for _ in range(100):
        x = int(rng.integers(4, width - 4))
        y = int(rng.integers(4, height - 4))
        radius = int(rng.integers(1, 3))
        value = int(rng.integers(16, 240))
        cv2.circle(gray, (x, y), radius, value, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _synthetic_frame(
    texture: np.ndarray,
    *,
    camera_angle_deg: float,
    focal_length_px: float,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    height, width = texture.shape[:2]
    translation_x_px = -focal_length_px * np.deg2rad(camera_angle_deg)
    transform = np.asarray(
        ((1.0, 0.0, translation_x_px), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    frame = cv2.warpAffine(
        texture,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    noise = noise_rng.normal(0.0, 1.2, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def run_synthetic_latency_calibration(
    *,
    known_latency_ms: int,
    seed: int,
    fps: int = _DEFAULT_FPS,
    config: OpticalFlowLatencyConfig | None = None,
) -> tuple[SyntheticLatencyRunResult, VideoLatencyEstimate]:
    """Run one deterministic synthetic optical-flow latency calibration."""

    if known_latency_ms < 0:
        raise ValueError("known_latency_ms must be >= 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    cfg = config or OpticalFlowLatencyConfig()
    cfg.validate()
    platform_motion = _synthetic_platform_motion()
    duration_s = 4.55
    integration_times, integrated_angles = _integrated_platform_angles(
        platform_motion,
        duration_s=duration_s + 0.1,
    )
    camera_model = _SyntheticCameraModel(
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
        focal_length_px=_DEFAULT_FOCAL_LENGTH_PX,
    )
    texture = _synthetic_texture(
        seed=12345,
        width=_DEFAULT_WIDTH,
        height=_DEFAULT_HEIGHT,
    )
    rng = np.random.default_rng(seed)
    source_interval_s = 1.0 / fps
    source_times_s = np.arange(0.0, duration_s, source_interval_s)
    # Receive jitter is intentionally independent of the source cadence.
    receive_jitter_s = np.clip(
        rng.normal(0.0, 0.003, len(source_times_s)),
        -0.007,
        0.007,
    )
    receive_times_s = (
        source_times_s + known_latency_ms / 1000.0 + receive_jitter_s
    )
    for index in range(1, len(receive_times_s)):
        receive_times_s[index] = max(
            receive_times_s[index],
            receive_times_s[index - 1] + 0.001,
        )

    frames = tuple(
        _synthetic_frame(
            texture,
            camera_angle_deg=float(
                np.interp(source_time_s, integration_times, integrated_angles)
            ),
            focal_length_px=_DEFAULT_FOCAL_LENGTH_PX,
            noise_rng=rng,
        )
        for source_time_s in source_times_s
    )
    observations: list[OpticalFlowRateObservation] = []
    source_interval_ns = round(source_interval_s * 1_000_000_000)
    for index in range(1, len(frames)):
        try:
            observation = estimate_optical_flow_rate_x(
                frames[index - 1],
                frames[index],
                previous_receive_timestamp_ns=(
                    _BASE_TIMESTAMP_NS
                    + round(receive_times_s[index - 1] * 1_000_000_000)
                ),
                current_receive_timestamp_ns=(
                    _BASE_TIMESTAMP_NS
                    + round(receive_times_s[index] * 1_000_000_000)
                ),
                source_interval_ns=source_interval_ns,
                camera_model=camera_model,
                config=cfg,
            )
        except VideoLatencyCalibrationError:
            continue
        observations.append(observation)

    estimate = estimate_video_latency(
        tuple(observations),
        platform_motion,
        config=cfg,
    )
    estimated = estimate.estimated_latency_ms
    result = SyntheticLatencyRunResult(
        known_latency_ms=known_latency_ms,
        estimated_latency_ms=estimated,
        absolute_error_ms=(
            None if estimated is None else abs(estimated - known_latency_ms)
        ),
        valid=estimate.valid,
        reason=estimate.reason,
        flow_observations=len(observations),
        median_flow_inliers=(
            float(median(item.inliers for item in observations))
            if observations
            else 0.0
        ),
        best_rmse_deg_s=(
            None if estimate.best_score is None else estimate.best_score.rmse_deg_s
        ),
        best_correlation=(
            None if estimate.best_score is None else estimate.best_score.correlation
        ),
    )
    return result, estimate


def _insufficient_excitation_is_rejected(
    config: OpticalFlowLatencyConfig,
) -> bool:
    model = CommandMotionModel(
        acceleration_x_deg_s2=40.0,
        acceleration_y_deg_s2=40.0,
    )
    model.mark_known_zero(timestamp_ns=_BASE_TIMESTAMP_NS)
    observations = tuple(
        OpticalFlowRateObservation(
            receive_midpoint_timestamp_ns=_BASE_TIMESTAMP_NS + index * 50_000_000,
            source_interval_ns=50_000_000,
            background_rate_x_deg_s=0.0,
            translation_x_px=0.0,
            inliers=80,
            tracked_points=100,
        )
        for index in range(1, 70)
    )
    estimate = estimate_video_latency(observations, model.snapshot(), config=config)
    return not estimate.valid and estimate.reason == "insufficient-excitation"


def run_video_latency_calibration_proof(
    *,
    known_latencies_ms: tuple[int, ...] = _DEFAULT_KNOWN_LATENCIES_MS,
    fps: int = _DEFAULT_FPS,
    config: OpticalFlowLatencyConfig | None = None,
) -> VideoLatencyCalibrationProofResult:
    """Prove one-shot calibration + repeated-run spread on synthetic video."""

    cfg = config or OpticalFlowLatencyConfig()
    cfg.validate()
    if not known_latencies_ms:
        raise ValueError("known_latencies_ms must not be empty")

    paired_results = tuple(
        run_synthetic_latency_calibration(
            known_latency_ms=known_latency_ms,
            seed=700 + index,
            fps=fps,
            config=cfg,
        )
        for index, known_latency_ms in enumerate(known_latencies_ms)
    )
    runs = tuple(item[0] for item in paired_results)
    estimates = tuple(item[1] for item in paired_results)
    aggregate = aggregate_latency_estimates(estimates)
    errors = [
        run.absolute_error_ms
        for run in runs
        if run.valid and run.absolute_error_ms is not None
    ]
    mean_known = float(fmean(known_latencies_ms))
    estimated_mean_error = (
        None
        if aggregate.mean_latency_ms is None
        else aggregate.mean_latency_ms - mean_known
    )
    insufficient_excitation_rejected = _insufficient_excitation_is_rejected(cfg)
    all_runs_valid = all(run.valid for run in runs)
    max_error = max(errors) if errors else None
    passed = (
        all_runs_valid
        and max_error is not None
        and max_error <= cfg.search_step_ms
        and aggregate.valid_run_count == len(runs)
        and insufficient_excitation_rejected
    )
    return VideoLatencyCalibrationProofResult(
        status="PASS" if passed else "FAIL",
        fps=fps,
        search_range_ms=(cfg.search_min_ms, cfg.search_max_ms),
        search_step_ms=cfg.search_step_ms,
        runs=runs,
        aggregate=aggregate,
        known_latency_mean_ms=mean_known,
        estimated_latency_mean_error_ms=estimated_mean_error,
        max_absolute_estimation_error_ms=max_error,
        insufficient_excitation_rejected=insufficient_excitation_rejected,
        conclusion=(
            "Sparse LK optical flow can recover effective command-model/video latency "
            "and repeated calibration passes provide a directly observable spread."
            if passed
            else (
                "Synthetic optical-flow latency calibration did not meet "
                "acceptance criteria."
            )
        ),
    )


def write_video_latency_calibration_json(
    result: VideoLatencyCalibrationProofResult,
    path: str | Path,
) -> Path:
    import json

    destination = Path(path)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "LatencyCandidateScore",
    "OpticalFlowLatencyConfig",
    "OpticalFlowRateObservation",
    "SyntheticLatencyRunResult",
    "VideoLatencyAggregate",
    "VideoLatencyCalibrationError",
    "VideoLatencyCalibrationProofResult",
    "VideoLatencyEstimate",
    "aggregate_latency_estimates",
    "estimate_optical_flow_rate_x",
    "estimate_video_latency",
    "run_synthetic_latency_calibration",
    "run_video_latency_calibration_proof",
    "write_video_latency_calibration_json",
]
