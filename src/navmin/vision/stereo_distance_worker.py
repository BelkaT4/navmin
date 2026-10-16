"""Bounded asynchronous execution boundary for stereo distance jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Lock, Thread

from navmin.calibration import StereoCalibration
from navmin.concurrency import InvalidatableLatest
from navmin.contracts import TargetRef

from .stereo_distance import (
    StereoDistanceEstimate,
    StereoDistanceEstimateOutcome,
    StereoDistanceFailure,
    StereoSgbmSettings,
    estimate_distance_from_stereo_pair,
)
from .stereo_pairing import StereoFramePair
from .stereo_target_join import StereoPairingKind, StereoTargetObservation

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


type StereoDistanceCompute = Callable[[StereoTargetObservation], StereoDistanceEstimateOutcome]


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
        min_valid_pixel_count: int = _DEFAULT_MIN_VALID_PIXEL_COUNT,
        compute: StereoDistanceCompute | None = None,
    ) -> None:
        if not isinstance(calibration, StereoCalibration):
            raise TypeError("calibration must be StereoCalibration")
        if (
            isinstance(min_valid_pixel_count, bool)
            or not isinstance(min_valid_pixel_count, int)
            or min_valid_pixel_count <= 0
        ):
            raise ValueError("min_valid_pixel_count must be a positive integer")
        selected_settings = settings or StereoSgbmSettings()
        if not isinstance(selected_settings, StereoSgbmSettings):
            raise TypeError("settings must be StereoSgbmSettings")

        self._calibration = calibration
        self._settings = selected_settings
        self._min_valid_pixel_count = min_valid_pixel_count
        self._compute = compute or self._compute_default
        self._condition = Condition(Lock())
        self._pending: StereoTargetObservation | None = None
        self._stop_requested = False
        self._thread: Thread | None = None
        self.results: InvalidatableLatest[StereoDistanceCandidate] = InvalidatableLatest()

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
            self._pending = observation
            self._condition.notify()
            return True

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
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

                try:
                    outcome = self._compute(observation)
                except Exception:  # pragma: no cover - defensive worker boundary
                    LOGGER.exception("Stereo distance computation failed unexpectedly")
                    continue
                if isinstance(outcome, StereoDistanceFailure):
                    continue
                if not isinstance(outcome, StereoDistanceEstimate):
                    LOGGER.error("Stereo distance compute returned unsupported outcome")
                    continue

                candidate = StereoDistanceCandidate(
                    target=observation.target,
                    intent_revision=observation.intent_revision,
                    distance_m=outcome.distance_m,
                    source_frame_id=observation.pair.left.frame_id,
                    capture_id=(
                        observation.pair.identity.capture_id
                        if isinstance(observation.pair, StereoFramePair)
                        else None
                    ),
                    measured_timestamp_ns=observation.pair.left.receive_timestamp_ns,
                    pairing_kind=observation.pairing_kind,
                )
                with self._condition:
                    if self._stop_requested:
                        return
                self.results.publish(candidate)
        finally:
            LOGGER.info("Stereo distance worker stopped")

    def _compute_default(
        self,
        observation: StereoTargetObservation,
    ) -> StereoDistanceEstimateOutcome:
        return estimate_distance_from_stereo_pair(
            observation.pair,
            self._calibration,
            self._settings,
            length_scale_to_m=self._calibration.length_scale_to_m,
            min_valid_pixel_count=self._min_valid_pixel_count,
            roi=observation.roi,
        )


__all__ = ["StereoDistanceCandidate", "StereoDistanceWorker"]
