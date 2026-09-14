"""Small thread-safe primitives matching NavMin's latest-state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Condition, Lock

from .contracts import CameraSessionStarted


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


__all__ = [
    "CameraSessionBarrierChannel",
    "InvalidatableLatest",
    "LatestSnapshot",
    "LatestValue",
]
