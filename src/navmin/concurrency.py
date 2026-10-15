"""Small thread-safe primitives matching NavMin's latest-state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Full, Queue
from threading import Condition, Lock

from .contracts import (
    CameraRole,
    CameraSessionStarted,
    ClearTargetSelection,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetLifecycleEvent,
    TargetSelectionRequest,
    TargetTrackingEnded,
    VisionTargetControlCommand,
)

_TARGET_LIFECYCLE_EVENT_CAPACITY = 64


@dataclass(frozen=True)
class LatestSnapshot[T]:
    """Atomic view of one latest-state slot and its monotonic revision."""

    revision: int
    value: T | None


class LatestValue[T]:
    """Thread-safe one-slot state: new publications replace older values."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._revision = 0
        self._value: T | None = None

    def publish(self, value: T) -> None:
        with self._condition:
            self._revision += 1
            self._value = value
            self._condition.notify_all()

    def get(self) -> T | None:
        with self._condition:
            return self._value

    def snapshot(self) -> LatestSnapshot[T]:
        with self._condition:
            return LatestSnapshot(self._revision, self._value)

    def wait_for_revision(
        self,
        after_revision: int,
        timeout: float | None = None,
    ) -> LatestSnapshot[T] | None:
        """Wait until the slot revision becomes newer than ``after_revision``."""
        with self._condition:
            changed = self._condition.wait_for(
                lambda: self._revision > after_revision,
                timeout=timeout,
            )
            if not changed:
                return None
            return LatestSnapshot(self._revision, self._value)

    def _invalidate(self) -> None:
        with self._condition:
            self._revision += 1
            self._value = None
            self._condition.notify_all()


class InvalidatableLatest[T](LatestValue[T]):
    """Latest-state slot where invalidation is itself an observable revision."""

    def invalidate(self) -> None:
        self._invalidate()

    clear = invalidate


class CameraSessionBarrierChannel:
    """Lossless FIFO channel only for ordered CameraSessionStarted barriers."""

    def __init__(self) -> None:
        self._queue: Queue[CameraSessionStarted] = Queue()

    def publish(self, session: CameraSessionStarted) -> None:
        self._queue.put_nowait(session)

    def receive(self, timeout: float | None = None) -> CameraSessionStarted:
        return self._queue.get(timeout=timeout)

    def receive_nowait(self) -> CameraSessionStarted:
        return self._queue.get_nowait()


class TargetControlChannel:
    """Per-camera latest-wins channel for target-control intent."""

    def __init__(self, camera: CameraRole) -> None:
        self.camera = camera
        self._lock = Lock()
        self._revision = 0
        self._consumed_revision = 0
        self._command: VisionTargetControlCommand | None = None

    def publish(self, command: VisionTargetControlCommand) -> None:
        if not isinstance(command, (TargetSelectionRequest, ClearTargetSelection)):
            raise TypeError("command must be a VisionTargetControlCommand")
        if command.camera is not self.camera:
            raise ValueError("target-control command belongs to another camera")
        with self._lock:
            self._revision += 1
            self._command = command

    def consume_latest(
        self,
        *,
        generation: int,
    ) -> VisionTargetControlCommand | None:
        """Consume the newest unpublished intent if it belongs to the session."""
        with self._lock:
            if self._revision <= self._consumed_revision:
                return None
            self._consumed_revision = self._revision
            command = self._command
        if command is None or command.generation != generation:
            return None
        return command

    def discard_pending(self) -> None:
        """Make all currently published intent ineligible for later consumption."""
        with self._lock:
            self._consumed_revision = self._revision


class TargetLifecycleEventChannelOverflowError(RuntimeError):
    """The bounded lifecycle FIFO cannot accept another event."""


class TargetLifecycleEventChannel:
    """Bounded lossless FIFO for Vision target lifecycle decisions."""

    def __init__(self, capacity: int = _TARGET_LIFECYCLE_EVENT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._queue: Queue[TargetLifecycleEvent] = Queue(maxsize=capacity)

    def publish(self, event: TargetLifecycleEvent) -> None:
        if not isinstance(
            event,
            (TargetAcquired, TargetAcquisitionFailed, TargetTrackingEnded),
        ):
            raise TypeError("event must be a TargetLifecycleEvent")
        try:
            self._queue.put_nowait(event)
        except Full as exc:
            raise TargetLifecycleEventChannelOverflowError(
                "target lifecycle event channel overflow"
            ) from exc

    def receive(self, timeout: float | None = None) -> TargetLifecycleEvent:
        return self._queue.get(timeout=timeout)

    def receive_nowait(self) -> TargetLifecycleEvent:
        return self._queue.get_nowait()


__all__ = [
    "CameraSessionBarrierChannel",
    "InvalidatableLatest",
    "LatestSnapshot",
    "LatestValue",
    "TargetControlChannel",
    "TargetLifecycleEventChannel",
    "TargetLifecycleEventChannelOverflowError",
]
