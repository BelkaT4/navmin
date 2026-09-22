"""Minimal cooperative worker lifecycle boundary."""

from __future__ import annotations

from threading import Event
from typing import Protocol


class StopToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = Event()

    def request_stop(self) -> None:
        self._event.set()

    def is_stop_requested(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation; return True when stop was requested."""
        return self._event.wait(timeout)


class WorkerHandle(Protocol):
    """Common boundary needed to request stop and wait for worker exit."""

    def request_stop(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...


def stop_and_join(worker: WorkerHandle, timeout: float | None = None) -> bool:
    """Request cooperative stop, join, and report whether the worker exited."""
    worker.request_stop()
    worker.join(timeout)
    return not worker.is_alive()
