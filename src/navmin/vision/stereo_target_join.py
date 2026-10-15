"""Bounded exact join between one stereo pair and one selected-target observation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from navmin.concurrency import InvalidatableLatest
from navmin.contracts import (
    BBox,
    CameraRole,
    CameraSessionStarted,
    FramePacket,
    TargetRef,
    VisionResult,
)

from .stereo_pairing import StereoFramePair, StereoPairer


@dataclass(frozen=True, slots=True)
class StereoTargetObservation:
    """Exact target ROI bound to one distance-intent revision and stereo pair."""

    target: TargetRef
    intent_revision: int
    pair: StereoFramePair
    roi: BBox


@dataclass(frozen=True, slots=True)
class StereoTargetJoinerState:
    """Read-only bounded-state snapshot for deterministic tests and diagnostics."""

    left_generation: int | None
    target: TargetRef | None
    intent_revision: int
    pending_pair_frame_id: int | None
    pending_result_frame_id: int | None
    latest_observation_frame_id: int | None


class StereoTargetJoiner:
    """Join exact pair/result identity for only the active Stereo Left target.

    The joiner intentionally keeps one latest-useful pair and one latest-useful
    ``VisionResult``. It never searches nearest frames, timestamps, or target
    candidates. Target/generation boundaries clear pending and published state.
    """

    def __init__(
        self,
        observation_sink: Callable[[StereoTargetObservation], object] | None = None,
    ) -> None:
        self._lock = Lock()
        self._left_generation: int | None = None
        self._target: TargetRef | None = None
        self._intent_revision = 0
        self._observation_sink = observation_sink
        self._pending_pair: StereoFramePair | None = None
        self._pending_result: VisionResult | None = None
        self.observations: InvalidatableLatest[StereoTargetObservation] = (
            InvalidatableLatest()
        )

    @property
    def state(self) -> StereoTargetJoinerState:
        with self._lock:
            observation = self.observations.get()
            return StereoTargetJoinerState(
                left_generation=self._left_generation,
                target=self._target,
                intent_revision=self._intent_revision,
                pending_pair_frame_id=(
                    None if self._pending_pair is None else self._pending_pair.left.frame_id
                ),
                pending_result_frame_id=(
                    None
                    if self._pending_result is None
                    else self._pending_result.frame.frame_id
                ),
                latest_observation_frame_id=(
                    None if observation is None else observation.pair.left.frame_id
                ),
            )

    def start_generation(self, generation: int) -> None:
        """Accept the authoritative Stereo Left generation barrier."""
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise TypeError("generation must be an integer")
        if generation < 1:
            raise ValueError("generation must be >= 1")
        with self._lock:
            if self._left_generation == generation:
                return
            self._left_generation = generation
            self._target = None
            self._intent_revision = 0
            self._clear_inputs_locked()
            self.observations.invalidate()

    def set_target(
        self,
        target: TargetRef | None,
        *,
        intent_revision: int = 0,
    ) -> None:
        """Activate one exact Stereo Left target for one distance-intent revision."""
        if target is not None and target.camera is not CameraRole.STEREO_LEFT:
            raise ValueError("stereo target join requires Stereo Left target")
        if (
            isinstance(intent_revision, bool)
            or not isinstance(intent_revision, int)
            or intent_revision < 0
        ):
            raise ValueError("intent_revision must be a non-negative integer")
        with self._lock:
            if target == self._target and intent_revision == self._intent_revision:
                return
            self._target = target
            self._intent_revision = intent_revision
            self._clear_inputs_locked()
            self.observations.invalidate()

    def offer_pair(self, pair: StereoFramePair) -> StereoTargetObservation | None:
        """Offer one exact pair; old pending pair is replaced latest-wins."""
        if not isinstance(pair, StereoFramePair):
            raise TypeError("pair must be StereoFramePair")
        with self._lock:
            target = self._target
            if (
                target is None
                or self._left_generation is None
                or target.generation != self._left_generation
                or pair.left.generation != self._left_generation
            ):
                return None
            self._pending_pair = pair
            observation = self._try_join_locked()
        self._notify_observation(observation)
        return observation

    def offer_vision_result(
        self,
        result: VisionResult,
    ) -> StereoTargetObservation | None:
        """Offer one Stereo Left result; old pending result is replaced latest-wins."""
        if not isinstance(result, VisionResult):
            raise TypeError("result must be VisionResult")
        frame = result.frame
        if frame.camera is not CameraRole.STEREO_LEFT:
            return None
        with self._lock:
            target = self._target
            if (
                target is None
                or self._left_generation is None
                or target.generation != self._left_generation
                or frame.generation != self._left_generation
            ):
                return None
            self._pending_result = result
            observation = self._try_join_locked()
        self._notify_observation(observation)
        return observation

    def reset(self) -> None:
        """Clear generation, target, pending inputs, and published observation."""
        with self._lock:
            self._left_generation = None
            self._target = None
            self._intent_revision = 0
            self._clear_inputs_locked()
            self.observations.invalidate()

    def _try_join_locked(self) -> StereoTargetObservation | None:
        pair = self._pending_pair
        result = self._pending_result
        target = self._target
        if pair is None or result is None or target is None:
            return None

        left = pair.left
        frame = result.frame
        if frame.generation != left.generation:
            self._discard_older_frame_locked(left.frame_id, frame.frame_id)
            return None
        if frame.frame_id != left.frame_id:
            self._discard_older_frame_locked(left.frame_id, frame.frame_id)
            return None
        if frame.capture_id != left.capture_id:
            self._clear_inputs_locked()
            self.observations.invalidate()
            return None

        tracked = next(
            (item for item in result.tracked_objects if item.track_id == target.track_id),
            None,
        )
        self._clear_inputs_locked()
        if tracked is None:
            self.observations.invalidate()
            return None

        observation = StereoTargetObservation(
            target=target,
            intent_revision=self._intent_revision,
            pair=pair,
            roi=tracked.bbox,
        )
        self.observations.publish(observation)
        return observation

    def _discard_older_frame_locked(self, pair_frame_id: int, result_frame_id: int) -> None:
        if pair_frame_id < result_frame_id:
            self._pending_pair = None
        elif result_frame_id < pair_frame_id:
            self._pending_result = None
        else:
            self._clear_inputs_locked()

    def _notify_observation(
        self,
        observation: StereoTargetObservation | None,
    ) -> None:
        if observation is not None and self._observation_sink is not None:
            self._observation_sink(observation)

    def _clear_inputs_locked(self) -> None:
        self._pending_pair = None
        self._pending_result = None


class StereoPairingTargetJoinSink:
    """Corrected-frame sink that preserves S2 pairing and forwards new pairs."""

    def __init__(self, pairer: StereoPairer, joiner: StereoTargetJoiner) -> None:
        self._pairer = pairer
        self._joiner = joiner

    def on_camera_session_started(self, session: CameraSessionStarted) -> None:
        self._pairer.on_camera_session_started(session)

    def on_corrected_frame(self, frame: FramePacket) -> None:
        pair = self._pairer.offer_frame(frame)
        if pair is not None:
            self._joiner.offer_pair(pair)


__all__ = [
    "StereoPairingTargetJoinSink",
    "StereoTargetJoiner",
    "StereoTargetJoinerState",
    "StereoTargetObservation",
]
