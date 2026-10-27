"""Bounded receive-time stereo compatibility pairing, separate from exact IDs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic_ns

from navmin.contracts import CameraRole, CameraSessionStarted, FramePacket
from navmin.vision.stereo_diagnostics import (
    STEREO_DIAGNOSTIC_WINDOW_SIZE,
    BoundedTimingSnapshot,
    BoundedTimingStats,
)
from navmin.vision.stereo_pairing import StereoPairContractError

DEFAULT_TIMESTAMP_PAIR_MAX_SKEW_MS = 30


@dataclass(frozen=True, slots=True)
class TimestampStereoPairIdentity:
    """Identity based on two host receive timestamps, never a capture ID."""

    left_generation: int
    right_generation: int
    left_frame_id: int
    right_frame_id: int
    left_receive_timestamp_ns: int
    right_receive_timestamp_ns: int
    timestamp_skew_ns: int  # Absolute, nonnegative skew.


@dataclass(frozen=True, slots=True)
class TimestampStereoFramePair:
    identity: TimestampStereoPairIdentity
    left: FramePacket
    right: FramePacket

    def __post_init__(self) -> None:
        if self.left.camera is not CameraRole.STEREO_LEFT:
            raise StereoPairContractError("left frame must belong to Stereo Left")
        if self.right.camera is not CameraRole.STEREO_RIGHT:
            raise StereoPairContractError("right frame must belong to Stereo Right")
        expected = TimestampStereoPairIdentity(
            self.left.generation,
            self.right.generation,
            self.left.frame_id,
            self.right.frame_id,
            self.left.receive_timestamp_ns,
            self.right.receive_timestamp_ns,
            abs(self.left.receive_timestamp_ns - self.right.receive_timestamp_ns),
        )
        if self.identity != expected:
            raise StereoPairContractError("timestamp pair identity does not match frames")


@dataclass(frozen=True, slots=True)
class TimestampStereoPairerState:
    left_generation: int | None
    right_generation: int | None
    right_buffer_size: int
    pending_left_frame_id: int | None
    latest_pair_identity: TimestampStereoPairIdentity | None
    latest_skew_ns: int | None


@dataclass(frozen=True, slots=True)
class TimestampStereoPairerDiagnostics:
    left_frames_seen: int
    right_frames_seen: int
    timestamp_pairs_emitted: int
    pending_left_superseded: int
    pair_attempts_expired: int
    right_frames_evicted_by_capacity: int
    right_frames_pruned_too_old: int
    right_frames_timed_out: int
    skew_rejected_candidates: int
    generation_resets: int
    max_timestamp_skew_ns: int
    skew: BoundedTimingSnapshot
    pending_left_frame_id: int | None = None
    right_buffer_size: int = 0


@dataclass(frozen=True, slots=True)
class _TimedFrame:
    frame: FramePacket
    arrival_ns: int


class TimestampStereoPairer:
    """Left-led, latest-useful, nonblocking PC receive-time compatibility pairer."""

    def __init__(
        self,
        *,
        right_frame_buffer_size: int,
        pair_timeout_ms: int,
        max_timestamp_skew_ms: int = DEFAULT_TIMESTAMP_PAIR_MAX_SKEW_MS,
        clock_ns: Callable[[], int] = monotonic_ns,
        diagnostic_window_size: int = STEREO_DIAGNOSTIC_WINDOW_SIZE,
    ) -> None:
        if right_frame_buffer_size < 1:
            raise ValueError("right_frame_buffer_size must be >= 1")
        if pair_timeout_ms < 1:
            raise ValueError("pair_timeout_ms must be >= 1")
        if max_timestamp_skew_ms < 0:
            raise ValueError("max_timestamp_skew_ms must be >= 0")
        self._capacity = right_frame_buffer_size
        self._timeout_ns = pair_timeout_ms * 1_000_000
        self._max_skew_ns = max_timestamp_skew_ms * 1_000_000
        self._clock_ns = clock_ns
        self._lock = Lock()
        self._left_generation: int | None = None
        self._right_generation: int | None = None
        self._right_buffer: deque[_TimedFrame] = deque()
        self._pending_left: _TimedFrame | None = None
        self._latest_pair: TimestampStereoFramePair | None = None
        self._last_left_frame_id: int | None = None
        self._last_right_frame_id: int | None = None
        self._left_frames_seen = 0
        self._right_frames_seen = 0
        self._timestamp_pairs_emitted = 0
        self._pending_left_superseded = 0
        self._pair_attempts_expired = 0
        self._right_frames_evicted_by_capacity = 0
        self._right_frames_pruned_too_old = 0
        self._right_frames_timed_out = 0
        self._skew_rejected_candidates = 0
        self._generation_resets = 0
        self._skew_stats = BoundedTimingStats(window_size=diagnostic_window_size)

    def on_camera_session_started(self, session: CameraSessionStarted) -> None:
        self.start_generation(session.camera, session.generation)

    def on_corrected_frame(self, frame: FramePacket) -> None:
        self.offer_frame(frame)

    @property
    def latest_pair(self) -> TimestampStereoFramePair | None:
        with self._lock:
            return self._latest_pair

    @property
    def state(self) -> TimestampStereoPairerState:
        with self._lock:
            self._expire_locked(self._clock_ns())
            identity = self._latest_pair.identity if self._latest_pair else None
            return TimestampStereoPairerState(
                left_generation=self._left_generation,
                right_generation=self._right_generation,
                right_buffer_size=len(self._right_buffer),
                pending_left_frame_id=(
                    self._pending_left.frame.frame_id if self._pending_left else None
                ),
                latest_pair_identity=identity,
                latest_skew_ns=identity.timestamp_skew_ns if identity else None,
            )

    @property
    def diagnostics(self) -> TimestampStereoPairerDiagnostics:
        with self._lock:
            self._expire_locked(self._clock_ns())
            return TimestampStereoPairerDiagnostics(
                left_frames_seen=self._left_frames_seen,
                right_frames_seen=self._right_frames_seen,
                timestamp_pairs_emitted=self._timestamp_pairs_emitted,
                pending_left_superseded=self._pending_left_superseded,
                pair_attempts_expired=self._pair_attempts_expired,
                right_frames_evicted_by_capacity=(
                    self._right_frames_evicted_by_capacity
                ),
                right_frames_pruned_too_old=self._right_frames_pruned_too_old,
                right_frames_timed_out=self._right_frames_timed_out,
                skew_rejected_candidates=self._skew_rejected_candidates,
                generation_resets=self._generation_resets,
                max_timestamp_skew_ns=self._max_skew_ns,
                skew=self._skew_stats.snapshot(),
                pending_left_frame_id=(
                    None
                    if self._pending_left is None
                    else self._pending_left.frame.frame_id
                ),
                right_buffer_size=len(self._right_buffer),
            )

    def start_generation(self, camera: CameraRole, generation: int) -> None:
        if camera not in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
            raise StereoPairContractError("stereo pairer accepts only stereo cameras")
        with self._lock:
            if camera is CameraRole.STEREO_LEFT:
                if generation == self._left_generation:
                    return
                self._left_generation = generation
            else:
                if generation == self._right_generation:
                    return
                self._right_generation = generation
            self._generation_resets += 1
            self._clear_locked()

    def offer_frame(self, frame: FramePacket) -> TimestampStereoFramePair | None:
        if frame.camera not in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
            raise StereoPairContractError("stereo pairer accepts only stereo cameras")
        now = self._clock_ns()
        with self._lock:
            if frame.camera is CameraRole.STEREO_LEFT:
                self._left_frames_seen += 1
            else:
                self._right_frames_seen += 1
            self._expire_locked(now)
            if frame.camera is CameraRole.STEREO_LEFT:
                if (
                    self._left_generation is None
                    or self._right_generation is None
                    or frame.generation != self._left_generation
                    or (
                        self._last_left_frame_id is not None
                        and frame.frame_id <= self._last_left_frame_id
                    )
                ):
                    return None
                self._last_left_frame_id = frame.frame_id
                if self._pending_left is not None:
                    self._pending_left_superseded += 1
                self._pending_left = None  # The new Left supersedes the old attempt.
                self._prune_right_older_than_left_locked(frame)
                index = self._nearest_right_index(frame)
                if index is not None:
                    right = self._right_buffer[index].frame
                    del self._right_buffer[index]
                    return self._publish_locked(frame, right)
                self._pending_left = _TimedFrame(frame, now)
                return None

            if (
                self._left_generation is None
                or self._right_generation is None
                or frame.generation != self._right_generation
                or (
                    self._last_right_frame_id is not None
                    and frame.frame_id <= self._last_right_frame_id
                )
            ):
                return None
            self._last_right_frame_id = frame.frame_id
            if self._pending_left is not None:
                left = self._pending_left.frame
                skew_ns = abs(
                    left.receive_timestamp_ns - frame.receive_timestamp_ns
                )
                if skew_ns <= self._max_skew_ns:
                    self._pending_left = None
                    return self._publish_locked(left, frame)
                self._skew_rejected_candidates += 1
            if len(self._right_buffer) == self._capacity:
                self._right_buffer.popleft()
                self._right_frames_evicted_by_capacity += 1
            self._right_buffer.append(_TimedFrame(frame, now))
            return None

    def expire(self) -> None:
        with self._lock:
            self._expire_locked(self._clock_ns())

    def reset(self) -> None:
        with self._lock:
            self._left_generation = None
            self._right_generation = None
            self._clear_locked()

    def _prune_right_older_than_left_locked(self, left: FramePacket) -> None:
        oldest_useful_timestamp_ns = left.receive_timestamp_ns - self._max_skew_ns
        previous_size = len(self._right_buffer)
        self._right_buffer = deque(
            item
            for item in self._right_buffer
            if item.frame.receive_timestamp_ns >= oldest_useful_timestamp_ns
        )
        self._right_frames_pruned_too_old += previous_size - len(self._right_buffer)

    def _nearest_right_index(self, left: FramePacket) -> int | None:
        candidates = [
            (
                abs(item.frame.receive_timestamp_ns - left.receive_timestamp_ns),
                item.frame.receive_timestamp_ns,
                item.frame.frame_id,
                index,
            )
            for index, item in enumerate(self._right_buffer)
        ]
        eligible = [
            candidate for candidate in candidates if candidate[0] <= self._max_skew_ns
        ]
        self._skew_rejected_candidates += len(candidates) - len(eligible)
        return min(eligible)[3] if eligible else None

    def _publish_locked(
        self,
        left: FramePacket,
        right: FramePacket,
    ) -> TimestampStereoFramePair:
        self._latest_pair = TimestampStereoFramePair(
            TimestampStereoPairIdentity(
                left.generation,
                right.generation,
                left.frame_id,
                right.frame_id,
                left.receive_timestamp_ns,
                right.receive_timestamp_ns,
                abs(left.receive_timestamp_ns - right.receive_timestamp_ns),
            ),
            left,
            right,
        )
        self._timestamp_pairs_emitted += 1
        self._skew_stats.add(self._latest_pair.identity.timestamp_skew_ns)
        return self._latest_pair

    def _expire_locked(self, now: int) -> None:
        if (
            self._pending_left
            and now - self._pending_left.arrival_ns >= self._timeout_ns
        ):
            self._pending_left = None
            self._pair_attempts_expired += 1
        previous_size = len(self._right_buffer)
        self._right_buffer = deque(
            item
            for item in self._right_buffer
            if now - item.arrival_ns < self._timeout_ns
        )
        self._right_frames_timed_out += previous_size - len(self._right_buffer)

    def _clear_locked(self) -> None:
        self._pending_left = None
        self._right_buffer.clear()
        self._latest_pair = None
        self._last_left_frame_id = None
        self._last_right_frame_id = None


__all__ = [
    "DEFAULT_TIMESTAMP_PAIR_MAX_SKEW_MS",
    "TimestampStereoFramePair",
    "TimestampStereoPairIdentity",
    "TimestampStereoPairer",
    "TimestampStereoPairerDiagnostics",
    "TimestampStereoPairerState",
]
