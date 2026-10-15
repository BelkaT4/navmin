"""Exact stereo-pair contracts and bounded non-blocking pairing state."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic_ns

from navmin.contracts import CameraRole, CameraSessionStarted, FramePacket


class StereoPairContractError(ValueError):
    """A stereo pair or pairer input violates the exact stereo contract."""


@dataclass(frozen=True, slots=True)
class StereoPairIdentity:
    """Exact identity of one stereo pair across both camera sessions."""

    left_generation: int
    right_generation: int
    capture_id: int


@dataclass(frozen=True, slots=True)
class StereoFramePair:
    """One exact pair of corrected Stereo Left and Stereo Right frames."""

    identity: StereoPairIdentity
    left: FramePacket
    right: FramePacket

    def __post_init__(self) -> None:
        _validate_camera_roles(self.left, self.right)
        left_capture_id = self.left.capture_id
        right_capture_id = self.right.capture_id
        if left_capture_id is None or right_capture_id is None:
            raise StereoPairContractError("exact stereo pair requires non-None capture_id")
        if left_capture_id != right_capture_id:
            raise StereoPairContractError("exact stereo pair requires matching capture_id")
        if self.identity.capture_id != left_capture_id:
            raise StereoPairContractError("pair identity capture_id does not match frames")
        if self.identity.left_generation != self.left.generation:
            raise StereoPairContractError(
                "pair identity left_generation does not match left frame"
            )
        if self.identity.right_generation != self.right.generation:
            raise StereoPairContractError(
                "pair identity right_generation does not match right frame"
            )


@dataclass(frozen=True, slots=True)
class StereoPairerState:
    """Small read-only snapshot for deterministic diagnostics and tests."""

    left_generation: int | None
    right_generation: int | None
    right_buffer_size: int
    pending_left_capture_id: int | None
    latest_pair_identity: StereoPairIdentity | None


@dataclass(frozen=True, slots=True)
class _TimedFrame:
    frame: FramePacket
    arrival_ns: int


class StereoPairer:
    """Thread-safe exact pairer with bounded realtime/latest-useful state.

    Stereo Left is the leader. Right frames may arrive first and are retained in a
    small arrival-ordered buffer. If Left arrives first, at most one pending Left
    attempt is retained; a different newer Left supersedes it. Matching uses only
    current camera generations and equal non-None ``capture_id`` values.
    """

    def __init__(
        self,
        *,
        right_frame_buffer_size: int,
        pair_timeout_ms: int,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if right_frame_buffer_size < 1:
            raise ValueError("right_frame_buffer_size must be >= 1")
        if pair_timeout_ms < 1:
            raise ValueError("pair_timeout_ms must be >= 1")
        self._right_frame_buffer_size = right_frame_buffer_size
        self._pair_timeout_ns = pair_timeout_ms * 1_000_000
        self._clock_ns = clock_ns
        self._lock = Lock()
        self._left_generation: int | None = None
        self._right_generation: int | None = None
        self._right_buffer: deque[_TimedFrame] = deque()
        self._pending_left: _TimedFrame | None = None
        self._latest_pair: StereoFramePair | None = None

    def on_camera_session_started(self, session: CameraSessionStarted) -> None:
        """Accept a Vision camera-session boundary through the corrected-frame seam."""
        self.start_generation(session.camera, session.generation)

    def on_corrected_frame(self, frame: FramePacket) -> None:
        """Accept one corrected frame through the Vision pipeline seam."""
        self.offer_frame(frame)

    @property
    def latest_pair(self) -> StereoFramePair | None:
        """Return the latest valid pair without consuming it."""
        with self._lock:
            return self._latest_pair

    @property
    def state(self) -> StereoPairerState:
        """Return a bounded state snapshot after deterministic timeout cleanup."""
        with self._lock:
            self._expire_locked(self._clock_ns())
            return StereoPairerState(
                left_generation=self._left_generation,
                right_generation=self._right_generation,
                right_buffer_size=len(self._right_buffer),
                pending_left_capture_id=(
                    None
                    if self._pending_left is None
                    else self._pending_left.frame.capture_id
                ),
                latest_pair_identity=(
                    None if self._latest_pair is None else self._latest_pair.identity
                ),
            )

    def start_generation(self, camera: CameraRole, generation: int) -> None:
        """Apply a camera-session generation barrier.

        A real generation transition invalidates all unmatched pairing state and
        the latest pair. Re-announcing the same current generation is idempotent.
        """
        if camera not in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
            raise StereoPairContractError("stereo pairer accepts only stereo cameras")
        with self._lock:
            if camera is CameraRole.STEREO_LEFT:
                if self._left_generation == generation:
                    return
                self._left_generation = generation
            else:
                if self._right_generation == generation:
                    return
                self._right_generation = generation
            self._clear_pairing_state_locked()

    def offer_frame(self, frame: FramePacket) -> StereoFramePair | None:
        """Offer one corrected frame without waiting for the opposite camera."""
        if frame.camera not in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
            raise StereoPairContractError("stereo pairer accepts only stereo cameras")
        now_ns = self._clock_ns()
        with self._lock:
            self._expire_locked(now_ns)
            if frame.camera is CameraRole.STEREO_LEFT:
                return self._offer_left_locked(frame, now_ns)
            return self._offer_right_locked(frame, now_ns)

    def expire(self) -> None:
        """Discard unmatched state whose bounded pairing lifetime has elapsed."""
        with self._lock:
            self._expire_locked(self._clock_ns())

    def reset(self) -> None:
        """Clear generations, unmatched state, and the latest pair."""
        with self._lock:
            self._left_generation = None
            self._right_generation = None
            self._clear_pairing_state_locked()

    def _offer_left_locked(
        self,
        frame: FramePacket,
        now_ns: int,
    ) -> StereoFramePair | None:
        if (
            self._left_generation is None
            or self._right_generation is None
            or frame.generation != self._left_generation
        ):
            return None
        capture_id = frame.capture_id
        if capture_id is None:
            return None
        if self._is_latest_identity_locked(capture_id):
            return None

        matched_index = self._find_right_index_locked(capture_id)
        if matched_index is not None:
            right = self._right_buffer[matched_index].frame
            del self._right_buffer[matched_index]
            self._pending_left = None
            return self._publish_pair_locked(frame, right)

        if (
            self._pending_left is not None
            and self._pending_left.frame.capture_id == capture_id
        ):
            return None

        self._pending_left = _TimedFrame(frame=frame, arrival_ns=now_ns)
        return None

    def _offer_right_locked(
        self,
        frame: FramePacket,
        now_ns: int,
    ) -> StereoFramePair | None:
        if (
            self._left_generation is None
            or self._right_generation is None
            or frame.generation != self._right_generation
        ):
            return None
        capture_id = frame.capture_id
        if capture_id is None:
            return None
        if self._is_latest_identity_locked(capture_id):
            return None

        pending_left = self._pending_left
        if pending_left is not None and pending_left.frame.capture_id == capture_id:
            self._pending_left = None
            return self._publish_pair_locked(pending_left.frame, frame)

        if self._find_right_index_locked(capture_id) is not None:
            return None

        if len(self._right_buffer) == self._right_frame_buffer_size:
            self._right_buffer.popleft()
        self._right_buffer.append(_TimedFrame(frame=frame, arrival_ns=now_ns))
        return None

    def _publish_pair_locked(
        self,
        left: FramePacket,
        right: FramePacket,
    ) -> StereoFramePair:
        pair = make_exact_stereo_pair(left, right)
        if pair is None:  # Defensive: callers reach here only after exact-ID match.
            raise AssertionError("exact stereo pair unexpectedly unavailable")
        self._latest_pair = pair
        return pair

    def _find_right_index_locked(self, capture_id: int) -> int | None:
        for index, timed in enumerate(self._right_buffer):
            if timed.frame.capture_id == capture_id:
                return index
        return None

    def _is_latest_identity_locked(self, capture_id: int) -> bool:
        latest = self._latest_pair
        return (
            latest is not None
            and latest.identity.left_generation == self._left_generation
            and latest.identity.right_generation == self._right_generation
            and latest.identity.capture_id == capture_id
        )

    def _expire_locked(self, now_ns: int) -> None:
        if (
            self._pending_left is not None
            and now_ns - self._pending_left.arrival_ns >= self._pair_timeout_ns
        ):
            self._pending_left = None
        if self._right_buffer:
            self._right_buffer = deque(
                timed
                for timed in self._right_buffer
                if now_ns - timed.arrival_ns < self._pair_timeout_ns
            )

    def _clear_pairing_state_locked(self) -> None:
        self._right_buffer.clear()
        self._pending_left = None
        self._latest_pair = None


def make_exact_stereo_pair(
    left: FramePacket,
    right: FramePacket,
) -> StereoFramePair | None:
    """Return an exact capture-ID pair, or ``None`` when exact identity is absent."""
    _validate_camera_roles(left, right)
    left_capture_id = left.capture_id
    right_capture_id = right.capture_id
    if left_capture_id is None or right_capture_id is None:
        return None
    if left_capture_id != right_capture_id:
        return None
    return StereoFramePair(
        identity=StereoPairIdentity(
            left_generation=left.generation,
            right_generation=right.generation,
            capture_id=left_capture_id,
        ),
        left=left,
        right=right,
    )


def _validate_camera_roles(left: FramePacket, right: FramePacket) -> None:
    if left.camera is not CameraRole.STEREO_LEFT:
        raise StereoPairContractError("left frame must belong to Stereo Left")
    if right.camera is not CameraRole.STEREO_RIGHT:
        raise StereoPairContractError("right frame must belong to Stereo Right")


__all__ = [
    "StereoFramePair",
    "StereoPairContractError",
    "StereoPairIdentity",
    "StereoPairer",
    "StereoPairerState",
    "make_exact_stereo_pair",
]
