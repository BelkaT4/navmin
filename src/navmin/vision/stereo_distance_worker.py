"""Bounded asynchronous execution boundary for stereo distance jobs."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from statistics import median
from threading import Condition, Lock, Thread
from time import monotonic_ns

from navmin.calibration import StereoCalibration
from navmin.concurrency import InvalidatableLatest
from navmin.config.models import StereoDistanceMethod as ConfigStereoDistanceMethod
from navmin.contracts import TargetRef

from .stereo_diagnostics import BoundedTimingSnapshot, BoundedTimingStats
from .stereo_distance import (
    StereoDistanceEstimate,
    StereoDistanceEstimateOutcome,
    StereoDistanceFailure,
    StereoDistanceFailureReason,
    StereoSgbmSettings,
    estimate_distance_from_detector_pair,
    estimate_distance_from_stereo_pair,
)
from .stereo_pairing import StereoFramePair
from .stereo_target_join import StereoPairingKind, StereoTargetObservation
from .stereo_timestamp_pairing import TimestampStereoFramePair

LOGGER = logging.getLogger(__name__)
_DEFAULT_MIN_VALID_PIXEL_COUNT = 5


@dataclass(frozen=True, slots=True)
class StereoDistanceCandidate:
    """Internal worker output awaiting Core lifecycle revalidation."""

    target: TargetRef
    intent_revision: int
    distance_m: float
    source_frame_id: int
    capture_id: int | None
    measured_timestamp_ns: int
    pairing_kind: StereoPairingKind = StereoPairingKind.EXACT
    sample_count: int = 1


type StereoDistanceCompute = Callable[[StereoTargetObservation], StereoDistanceEstimateOutcome]


class StereoDistanceFailureClass(Enum):
    """Stable diagnostic grouping of existing typed S3 failure reasons."""

    GEOMETRY_RESOLUTION = "geometry_resolution"
    INVALID_OR_NO_DISPARITY = "invalid_or_no_disparity"
    INSUFFICIENT_VALID_ROI_PIXELS = "insufficient_valid_roi_pixels"
    INVALID_ROI = "invalid_roi"
    OTHER_EXPECTED = "other_expected"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


@dataclass(frozen=True, slots=True)
class StereoDistanceSuccessDiagnostics:
    distance_m: float
    sample_count: int
    median_disparity_px: float
    valid_pixel_count: int
    valid_pixel_fraction: float
    left_generation: int
    right_generation: int
    left_frame_id: int
    right_frame_id: int
    left_receive_timestamp_ns: int
    right_receive_timestamp_ns: int
    capture_id: int | None
    pairing_kind: StereoPairingKind
    timestamp_skew_ns: int | None


@dataclass(frozen=True, slots=True)
class StereoDistanceWorkerDiagnostics:
    jobs_submitted: int
    pending_replacements: int
    computations_started: int
    computations_succeeded: int
    computations_failed: int
    shutdown_discarded: int
    failure_geometry_resolution: int
    failure_invalid_or_no_disparity: int
    failure_insufficient_valid_roi_pixels: int
    failure_invalid_roi: int
    failure_other_expected: int
    failure_unexpected_exception: int
    latest_failure_reason: StereoDistanceFailureReason | None
    latest_failure_class: StereoDistanceFailureClass | None
    compute_duration: BoundedTimingSnapshot
    latest_success: StereoDistanceSuccessDiagnostics | None


class StereoDistanceWorker:
    """Single worker with one latest-wins pending target observation.

    One observation may be executing while at most one newer observation waits.
    Replacing the pending observation never grows a queue. Public ``DistanceResult``
    ownership remains in Core; this worker publishes only internal candidates.
    """

    def __init__(
        self,
        *,
        calibration: StereoCalibration,
        settings: StereoSgbmSettings | None = None,
        method: ConfigStereoDistanceMethod = ConfigStereoDistanceMethod.ROI,
        aggregation_frame_count: int = 1,
        max_sample_gap_ms: int = 1000,
        min_valid_pixel_count: int = _DEFAULT_MIN_VALID_PIXEL_COUNT,
        compute: StereoDistanceCompute | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not isinstance(calibration, StereoCalibration):
            raise TypeError("calibration must be StereoCalibration")
        if not isinstance(method, ConfigStereoDistanceMethod):
            raise TypeError("method must be StereoDistanceMethod")
        for name, value, minimum in (
            ("aggregation_frame_count", aggregation_frame_count, 1),
            ("max_sample_gap_ms", max_sample_gap_ms, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            if value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(min_valid_pixel_count, bool)
            or not isinstance(min_valid_pixel_count, int)
            or min_valid_pixel_count <= 0
        ):
            raise ValueError("min_valid_pixel_count must be a positive integer")
        selected_settings = settings or StereoSgbmSettings()
        if not isinstance(selected_settings, StereoSgbmSettings):
            raise TypeError("settings must be StereoSgbmSettings")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")

        self._calibration = calibration
        self._settings = selected_settings
        self._method = method
        self._aggregation_frame_count = aggregation_frame_count
        self._max_sample_gap_ns = max_sample_gap_ms * 1_000_000
        self._distance_samples: deque[float] = deque(maxlen=aggregation_frame_count)
        self._sample_key: tuple[object, ...] | None = None
        self._last_sample_timestamp_ns: int | None = None
        self._min_valid_pixel_count = min_valid_pixel_count
        self._compute = compute or self._compute_default
        self._clock_ns = clock_ns
        self._condition = Condition(Lock())
        self._pending: StereoTargetObservation | None = None
        self._stop_requested = False
        self._thread: Thread | None = None
        self.results: InvalidatableLatest[StereoDistanceCandidate] = InvalidatableLatest()
        self._jobs_submitted = 0
        self._pending_replacements = 0
        self._computations_started = 0
        self._computations_succeeded = 0
        self._computations_failed = 0
        self._shutdown_discarded = 0
        self._failure_counts = {
            failure_class: 0 for failure_class in StereoDistanceFailureClass
        }
        self._latest_failure_reason: StereoDistanceFailureReason | None = None
        self._latest_failure_class: StereoDistanceFailureClass | None = None
        self._compute_duration = BoundedTimingStats()
        self._latest_success: StereoDistanceSuccessDiagnostics | None = None

    @property
    def diagnostics(self) -> StereoDistanceWorkerDiagnostics:
        with self._condition:
            return StereoDistanceWorkerDiagnostics(
                jobs_submitted=self._jobs_submitted,
                pending_replacements=self._pending_replacements,
                computations_started=self._computations_started,
                computations_succeeded=self._computations_succeeded,
                computations_failed=self._computations_failed,
                shutdown_discarded=self._shutdown_discarded,
                failure_geometry_resolution=self._failure_counts[
                    StereoDistanceFailureClass.GEOMETRY_RESOLUTION
                ],
                failure_invalid_or_no_disparity=self._failure_counts[
                    StereoDistanceFailureClass.INVALID_OR_NO_DISPARITY
                ],
                failure_insufficient_valid_roi_pixels=self._failure_counts[
                    StereoDistanceFailureClass.INSUFFICIENT_VALID_ROI_PIXELS
                ],
                failure_invalid_roi=self._failure_counts[
                    StereoDistanceFailureClass.INVALID_ROI
                ],
                failure_other_expected=self._failure_counts[
                    StereoDistanceFailureClass.OTHER_EXPECTED
                ],
                failure_unexpected_exception=self._failure_counts[
                    StereoDistanceFailureClass.UNEXPECTED_EXCEPTION
                ],
                latest_failure_reason=self._latest_failure_reason,
                latest_failure_class=self._latest_failure_class,
                compute_duration=self._compute_duration.snapshot(),
                latest_success=self._latest_success,
            )

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("StereoDistanceWorker is one-shot")
            self._thread = Thread(
                target=self._run,
                name="stereo-distance-worker",
                daemon=False,
            )
            thread = self._thread
        thread.start()

    def submit_observation(self, observation: StereoTargetObservation) -> bool:
        """Replace any not-yet-started job with the newest observation."""
        if not isinstance(observation, StereoTargetObservation):
            raise TypeError("observation must be StereoTargetObservation")
        with self._condition:
            if self._stop_requested:
                return False
            self._jobs_submitted += 1
            if self._pending is not None:
                self._pending_replacements += 1
            self._pending = observation
            self._condition.notify()
            return True

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            if self._pending is not None:
                self._shutdown_discarded += 1
            self._pending = None
            self._condition.notify_all()
        self.results.invalidate()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def stop(self, timeout: float = 2.0) -> bool:
        self.request_stop()
        self.join(timeout)
        return not self.is_alive()

    def shutdown(self, timeout: float = 2.0) -> None:
        if not self.stop(timeout):
            raise RuntimeError(
                f"StereoDistanceWorker did not stop within {timeout:.3f} seconds"
            )

    def _run(self) -> None:
        LOGGER.info("Stereo distance worker starting")
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_requested or self._pending is not None
                    )
                    if self._stop_requested:
                        return
                    observation = self._pending
                    self._pending = None
                if observation is None:
                    continue

                with self._condition:
                    self._computations_started += 1
                started_ns = self._clock_ns()
                try:
                    outcome = self._compute(observation)
                except Exception:  # pragma: no cover - defensive worker boundary
                    duration_ns = max(0, self._clock_ns() - started_ns)
                    with self._condition:
                        self._record_unexpected_failure_locked(duration_ns)
                    LOGGER.exception("Stereo distance computation failed unexpectedly")
                    continue
                duration_ns = max(0, self._clock_ns() - started_ns)
                if isinstance(outcome, StereoDistanceFailure):
                    with self._condition:
                        self._clear_samples_locked()
                        self._record_expected_failure_locked(outcome, duration_ns)
                    continue
                if not isinstance(outcome, StereoDistanceEstimate):
                    with self._condition:
                        self._clear_samples_locked()
                        self._record_unexpected_failure_locked(duration_ns)
                    LOGGER.error("Stereo distance compute returned unsupported outcome")
                    continue

                with self._condition:
                    aggregate_distance = self._append_sample_locked(
                        observation, outcome.distance_m
                    )
                    sample_count = len(self._distance_samples)
                    self._record_success_locked(
                        observation,
                        outcome,
                        duration_ns,
                        distance_m=(
                            aggregate_distance
                            if aggregate_distance is not None
                            else outcome.distance_m
                        ),
                        sample_count=sample_count,
                    )

                if aggregate_distance is None:
                    continue

                candidate = StereoDistanceCandidate(
                    target=observation.target,
                    intent_revision=observation.intent_revision,
                    distance_m=aggregate_distance,
                    source_frame_id=observation.pair.left.frame_id,
                    capture_id=(
                        observation.pair.identity.capture_id
                        if isinstance(observation.pair, StereoFramePair)
                        else None
                    ),
                    measured_timestamp_ns=observation.pair.left.receive_timestamp_ns,
                    pairing_kind=observation.pairing_kind,
                    sample_count=sample_count,
                )
                with self._condition:
                    if self._stop_requested:
                        self._shutdown_discarded += 1
                        return
                self.results.publish(candidate)
        finally:
            LOGGER.info("Stereo distance worker stopped")

    def _compute_default(
        self,
        observation: StereoTargetObservation,
    ) -> StereoDistanceEstimateOutcome:
        if self._method is ConfigStereoDistanceMethod.DETECTOR:
            return estimate_distance_from_detector_pair(
                observation.pair,
                self._calibration,
                observation.roi,
                observation.right_detections,
                length_scale_to_m=self._calibration.length_scale_to_m,
            )
        return estimate_distance_from_stereo_pair(
            observation.pair,
            self._calibration,
            self._settings,
            length_scale_to_m=self._calibration.length_scale_to_m,
            min_valid_pixel_count=self._min_valid_pixel_count,
            roi=observation.roi,
        )

    def _append_sample_locked(
        self,
        observation: StereoTargetObservation,
        distance_m: float,
    ) -> float | None:
        pair = observation.pair
        timestamp_ns = pair.left.receive_timestamp_ns
        key = (
            observation.target,
            observation.intent_revision,
            observation.pairing_kind,
            pair.left.generation,
            pair.right.generation,
        )
        if (
            key != self._sample_key
            or (
                self._last_sample_timestamp_ns is not None
                and timestamp_ns - self._last_sample_timestamp_ns
                > self._max_sample_gap_ns
            )
        ):
            self._clear_samples_locked()
        self._sample_key = key
        self._last_sample_timestamp_ns = timestamp_ns
        self._distance_samples.append(distance_m)
        if len(self._distance_samples) < self._aggregation_frame_count:
            return None
        return float(median(self._distance_samples))

    def _clear_samples_locked(self) -> None:
        self._distance_samples.clear()
        self._sample_key = None
        self._last_sample_timestamp_ns = None

    def _record_expected_failure_locked(
        self,
        failure: StereoDistanceFailure,
        duration_ns: int,
    ) -> None:
        failure_class = _classify_failure(failure.reason)
        self._computations_failed += 1
        self._failure_counts[failure_class] += 1
        self._latest_failure_reason = failure.reason
        self._latest_failure_class = failure_class
        self._compute_duration.add(duration_ns)

    def _record_unexpected_failure_locked(self, duration_ns: int) -> None:
        failure_class = StereoDistanceFailureClass.UNEXPECTED_EXCEPTION
        self._computations_failed += 1
        self._failure_counts[failure_class] += 1
        self._latest_failure_reason = None
        self._latest_failure_class = failure_class
        self._compute_duration.add(duration_ns)

    def _record_success_locked(
        self,
        observation: StereoTargetObservation,
        estimate: StereoDistanceEstimate,
        duration_ns: int,
        *,
        distance_m: float,
        sample_count: int,
    ) -> None:
        pair = observation.pair
        self._computations_succeeded += 1
        self._compute_duration.add(duration_ns)
        self._latest_success = StereoDistanceSuccessDiagnostics(
            distance_m=distance_m,
            sample_count=sample_count,
            median_disparity_px=estimate.median_disparity_px,
            valid_pixel_count=estimate.valid_pixel_count,
            valid_pixel_fraction=estimate.valid_pixel_fraction,
            left_generation=pair.left.generation,
            right_generation=pair.right.generation,
            left_frame_id=pair.left.frame_id,
            right_frame_id=pair.right.frame_id,
            left_receive_timestamp_ns=pair.left.receive_timestamp_ns,
            right_receive_timestamp_ns=pair.right.receive_timestamp_ns,
            capture_id=(
                pair.identity.capture_id if isinstance(pair, StereoFramePair) else None
            ),
            pairing_kind=observation.pairing_kind,
            timestamp_skew_ns=(
                pair.identity.timestamp_skew_ns
                if isinstance(pair, TimestampStereoFramePair)
                else None
            ),
        )


def _classify_failure(
    reason: StereoDistanceFailureReason,
) -> StereoDistanceFailureClass:
    if reason in {
        StereoDistanceFailureReason.INVALID_FRAME_SHAPE,
        StereoDistanceFailureReason.FRAME_SHAPE_MISMATCH,
        StereoDistanceFailureReason.CALIBRATION_RESOLUTION_MISMATCH,
    }:
        return StereoDistanceFailureClass.GEOMETRY_RESOLUTION
    if reason in {
        StereoDistanceFailureReason.INVALID_DISPARITY,
        StereoDistanceFailureReason.INVALID_DISPARITY_MAP,
    }:
        return StereoDistanceFailureClass.INVALID_OR_NO_DISPARITY
    if reason is StereoDistanceFailureReason.INSUFFICIENT_VALID_PIXELS:
        return StereoDistanceFailureClass.INSUFFICIENT_VALID_ROI_PIXELS
    if reason in {
        StereoDistanceFailureReason.ROI_OUT_OF_FRAME,
        StereoDistanceFailureReason.EMPTY_ROI,
    }:
        return StereoDistanceFailureClass.INVALID_ROI
    return StereoDistanceFailureClass.OTHER_EXPECTED


__all__ = [
    "StereoDistanceCandidate",
    "StereoDistanceFailureClass",
    "StereoDistanceSuccessDiagnostics",
    "StereoDistanceWorker",
    "StereoDistanceWorkerDiagnostics",
]
