"""Bounded exact-left-frame join for exact and timestamp stereo pairs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
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
from .stereo_timestamp_pairing import TimestampStereoFramePair, TimestampStereoPairer


class StereoPairingKind(Enum):
    """Internal pairing origin, independent of public distance source."""

    EXACT = "exact"
    TIMESTAMP = "timestamp"


type StereoTargetPair = StereoFramePair | TimestampStereoFramePair


@dataclass(frozen=True, slots=True)
class StereoTargetObservation:
    """Exact target ROI bound to one distance-intent revision and stereo pair."""

    target: TargetRef
    intent_revision: int
    pair: StereoTargetPair
    roi: BBox
    pairing_kind: StereoPairingKind = StereoPairingKind.EXACT

    def __post_init__(self) -> None:
        expected = (
            StereoPairingKind.EXACT
            if isinstance(self.pair, StereoFramePair)
            else StereoPairingKind.TIMESTAMP
            if isinstance(self.pair, TimestampStereoFramePair)
            else None
        )
        if expected is None or self.pairing_kind is not expected:
            raise ValueError("pairing_kind must agree with stereo pair type")


@dataclass(frozen=True, slots=True)
class StereoTargetJoinerState:
    """Read-only bounded-state snapshot for deterministic tests and diagnostics."""

    left_generation: int | None
    target: TargetRef | None
    intent_revision: int
    pending_pair_frame_id: int | None
    pending_result_frame_id: int | None
    latest_observation_frame_id: int | None


@dataclass(frozen=True, slots=True)
class StereoTargetJoinerDiagnostics:
    pairs_received: int
    vision_results_received: int
    target_observations_emitted: int
    pair_superseded_unmatched: int
    vision_result_superseded_unmatched: int
    rejected_inactive_target: int
    rejected_wrong_generation: int
    rejected_wrong_frame_id: int
    rejected_left_identity: int
    rejected_target_not_tracked: int
    generation_resets: int


class StereoTargetJoiner:
    """Join pair/result identity for only the active Stereo Left target.

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
        self._pending_pair: StereoTargetPair | None = None
        self._pending_result: VisionResult | None = None
        self.observations: InvalidatableLatest[StereoTargetObservation] = (
            InvalidatableLatest()
        )
        self._pairs_received = 0
        self._vision_results_received = 0
        self._target_observations_emitted = 0
        self._pair_superseded_unmatched = 0
        self._vision_result_superseded_unmatched = 0
        self._rejected_inactive_target = 0
        self._rejected_wrong_generation = 0
        self._rejected_wrong_frame_id = 0
        self._rejected_left_identity = 0
        self._rejected_target_not_tracked = 0
        self._generation_resets = 0

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

    @property
    def diagnostics(self) -> StereoTargetJoinerDiagnostics:
        with self._lock:
            return StereoTargetJoinerDiagnostics(
                pairs_received=self._pairs_received,
                vision_results_received=self._vision_results_received,
                target_observations_emitted=self._target_observations_emitted,
                pair_superseded_unmatched=self._pair_superseded_unmatched,
                vision_result_superseded_unmatched=(
                    self._vision_result_superseded_unmatched
                ),
                rejected_inactive_target=self._rejected_inactive_target,
                rejected_wrong_generation=self._rejected_wrong_generation,
                rejected_wrong_frame_id=self._rejected_wrong_frame_id,
                rejected_left_identity=self._rejected_left_identity,
                rejected_target_not_tracked=self._rejected_target_not_tracked,
                generation_resets=self._generation_resets,
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
            self._generation_resets += 1
            self._left_generation = generation
            self._target = None
            self._intent_revision = 0
            self._count_pending_discards_locked()
            self._clear_inputs_locked()
            self.observations.invalidate()

    def set_target(
        self,
        target: TargetRef | None,
        *,
        intent_revision: int = 0,
    ) -> None:
        """Activate one Stereo Left target for one distance-intent revision."""
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
            self._count_pending_discards_locked()
            self._clear_inputs_locked()
            self.observations.invalidate()

    def offer_pair(self, pair: StereoTargetPair) -> StereoTargetObservation | None:
        """Offer one stereo pair; old pending pair is replaced latest-wins."""
        if not isinstance(pair, (StereoFramePair, TimestampStereoFramePair)):
            raise TypeError("pair must be an exact or timestamp stereo pair")
        with self._lock:
            self._pairs_received += 1
            target = self._target
            if (
                target is None
                or self._left_generation is None
                or target.generation != self._left_generation
                or pair.left.generation != self._left_generation
            ):
                if target is None:
                    self._rejected_inactive_target += 1
                else:
                    self._rejected_wrong_generation += 1
                return None
            if self._pending_pair is not None:
                self._pair_superseded_unmatched += 1
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
            with self._lock:
                self._vision_results_received += 1
            return None
        with self._lock:
            self._vision_results_received += 1
            target = self._target
            if (
                target is None
                or self._left_generation is None
                or target.generation != self._left_generation
                or frame.generation != self._left_generation
            ):
                if target is None:
                    self._rejected_inactive_target += 1
                else:
                    self._rejected_wrong_generation += 1
                return None
            if self._pending_result is not None:
                self._vision_result_superseded_unmatched += 1
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
            self._count_pending_discards_locked()
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
            self._rejected_wrong_generation += 1
            self._discard_older_frame_locked(left.frame_id, frame.frame_id)
            return None
        if frame.frame_id != left.frame_id:
            self._rejected_wrong_frame_id += 1
            self._discard_older_frame_locked(left.frame_id, frame.frame_id)
            return None
        if (
            frame.capture_id != left.capture_id
            or frame.receive_timestamp_ns != left.receive_timestamp_ns
        ):
            self._rejected_left_identity += 1
            self._pair_superseded_unmatched += 1
            self._vision_result_superseded_unmatched += 1
            self._clear_inputs_locked()
            self.observations.invalidate()
            return None

        tracked = next(
            (item for item in result.tracked_objects if item.track_id == target.track_id),
            None,
        )
        self._clear_inputs_locked()
        if tracked is None:
            self._rejected_target_not_tracked += 1
            self.observations.invalidate()
            return None

        observation = StereoTargetObservation(
            target=target,
            intent_revision=self._intent_revision,
            pair=pair,
            roi=tracked.bbox,
            pairing_kind=(
                StereoPairingKind.EXACT
                if isinstance(pair, StereoFramePair)
                else StereoPairingKind.TIMESTAMP
            ),
        )
        self.observations.publish(observation)
        self._target_observations_emitted += 1
        return observation

    def _discard_older_frame_locked(self, pair_frame_id: int, result_frame_id: int) -> None:
        if pair_frame_id < result_frame_id:
            self._pending_pair = None
            self._pair_superseded_unmatched += 1
        elif result_frame_id < pair_frame_id:
            self._pending_result = None
            self._vision_result_superseded_unmatched += 1
        else:
            self._pair_superseded_unmatched += 1
            self._vision_result_superseded_unmatched += 1
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

    def _count_pending_discards_locked(self) -> None:
        if self._pending_pair is not None:
            self._pair_superseded_unmatched += 1
        if self._pending_result is not None:
            self._vision_result_superseded_unmatched += 1


class StereoPairingTargetJoinSink:
    """Route corrected frames into independent exact and compatibility pairers.

    Only the operator-selected pairing kind reaches the target joiner. Both
    pairers retain their independent generation barriers and state bounds.
    """

    def __init__(
        self,
        pairer: StereoPairer,
        joiner: StereoTargetJoiner,
        *,
        timestamp_pairer: TimestampStereoPairer | None = None,
        active_pairing_kind: Callable[[], StereoPairingKind | None] | None = None,
    ) -> None:
        self._pairer = pairer
        self._joiner = joiner
        self._timestamp_pairer = timestamp_pairer
        self._active_pairing_kind = active_pairing_kind
        self._routing_lock = Lock()
        self._last_kind: StereoPairingKind | None = None
        self._generations: dict[CameraRole, int] = {}

    def set_active_pairing_kind_provider(
        self, provider: Callable[[], StereoPairingKind | None]
    ) -> None:
        self._active_pairing_kind = provider

    def on_camera_session_started(self, session: CameraSessionStarted) -> None:
        with self._routing_lock:
            self._generations[session.camera] = session.generation
            self._pairer.on_camera_session_started(session)
            if self._timestamp_pairer is not None:
                self._timestamp_pairer.on_camera_session_started(session)

    def on_corrected_frame(self, frame: FramePacket) -> None:
        # An absent provider preserves the previous exact-only sink contract.
        kind = (
            StereoPairingKind.EXACT
            if self._active_pairing_kind is None
            else self._active_pairing_kind()
        )
        with self._routing_lock:
            if kind is not self._last_kind:
                if self._last_kind is not None:
                    # Mode changes discard unmatched input, not camera sessions.
                    self._pairer.reset()
                    if self._timestamp_pairer is not None:
                        self._timestamp_pairer.reset()
                    for camera, generation in self._generations.items():
                        self._pairer.start_generation(camera, generation)
                        if self._timestamp_pairer is not None:
                            self._timestamp_pairer.start_generation(camera, generation)
                self._last_kind = kind
            if kind is StereoPairingKind.EXACT:
                pair = self._pairer.offer_frame(frame)
            elif kind is StereoPairingKind.TIMESTAMP and self._timestamp_pairer is not None:
                pair = self._timestamp_pairer.offer_frame(frame)
            else:
                return
        if pair is not None:
            self._joiner.offer_pair(pair)


__all__ = [
    "StereoPairingKind",
    "StereoPairingTargetJoinSink",
    "StereoTargetJoiner",
    "StereoTargetJoinerDiagnostics",
    "StereoTargetJoinerState",
    "StereoTargetObservation",
    "StereoTargetPair",
]
