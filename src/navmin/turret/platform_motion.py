"""Turret-owned acceleration-limited command-motion model."""

from __future__ import annotations

from collections import deque
from math import isfinite
from threading import Lock

from navmin.contracts import (
    PlatformMotionAnchor,
    PlatformMotionSnapshot,
    PlatformMotionSource,
)

_DEFAULT_HISTORY_CAPACITY = 256


class CommandMotionModel:
    """Bounded model of accepted velocity targets in the monotonic clock domain."""

    def __init__(
        self,
        *,
        acceleration_x_deg_s2: float,
        acceleration_y_deg_s2: float,
        capacity: int = _DEFAULT_HISTORY_CAPACITY,
    ) -> None:
        self._acceleration_x = self._positive_finite(
            acceleration_x_deg_s2, name="acceleration_x_deg_s2"
        )
        self._acceleration_y = self._positive_finite(
            acceleration_y_deg_s2, name="acceleration_y_deg_s2"
        )
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self._anchors: deque[PlatformMotionAnchor] = deque(maxlen=capacity)
        self._valid = False
        self._lock = Lock()

    def snapshot(self) -> PlatformMotionSnapshot:
        with self._lock:
            return PlatformMotionSnapshot(
                source=PlatformMotionSource.COMMAND_MODEL,
                valid=self._valid,
                anchors=tuple(self._anchors),
            )

    def record_velocity_command(
        self,
        *,
        timestamp_ns: int,
        commanded_rate_x_deg_s: float,
        commanded_rate_y_deg_s: float,
    ) -> None:
        timestamp = self._timestamp(timestamp_ns)
        command_x = self._finite(
            commanded_rate_x_deg_s, name="commanded_rate_x_deg_s"
        )
        command_y = self._finite(
            commanded_rate_y_deg_s, name="commanded_rate_y_deg_s"
        )
        with self._lock:
            if not self._valid:
                return
            rates = self._rate_at_locked(timestamp)
            if rates is None:
                self._invalidate_locked()
                return
            self._append_locked(
                PlatformMotionAnchor(
                    timestamp_ns=timestamp,
                    modeled_rate_x_deg_s=rates[0],
                    modeled_rate_y_deg_s=rates[1],
                    commanded_rate_x_deg_s=command_x,
                    commanded_rate_y_deg_s=command_y,
                    acceleration_x_deg_s2=self._acceleration_x,
                    acceleration_y_deg_s2=self._acceleration_y,
                )
            )

    def record_acceleration_limits(
        self,
        *,
        timestamp_ns: int,
        acceleration_x_deg_s2: float,
        acceleration_y_deg_s2: float,
    ) -> None:
        timestamp = self._timestamp(timestamp_ns)
        acceleration_x = self._positive_finite(
            acceleration_x_deg_s2, name="acceleration_x_deg_s2"
        )
        acceleration_y = self._positive_finite(
            acceleration_y_deg_s2, name="acceleration_y_deg_s2"
        )
        with self._lock:
            if self._valid:
                rates = self._rate_at_locked(timestamp)
                if rates is None:
                    self._invalidate_locked()
                else:
                    previous = self._anchors[-1]
                    self._append_locked(
                        PlatformMotionAnchor(
                            timestamp_ns=timestamp,
                            modeled_rate_x_deg_s=rates[0],
                            modeled_rate_y_deg_s=rates[1],
                            commanded_rate_x_deg_s=(
                                previous.commanded_rate_x_deg_s
                            ),
                            commanded_rate_y_deg_s=(
                                previous.commanded_rate_y_deg_s
                            ),
                            acceleration_x_deg_s2=acceleration_x,
                            acceleration_y_deg_s2=acceleration_y,
                        )
                    )
            self._acceleration_x = acceleration_x
            self._acceleration_y = acceleration_y

    def mark_known_zero(self, *, timestamp_ns: int) -> None:
        timestamp = self._timestamp(timestamp_ns)
        with self._lock:
            if self._anchors and timestamp < self._anchors[-1].timestamp_ns:
                self._invalidate_locked()
                return
            if not self._valid:
                self._anchors.clear()
            self._valid = True
            self._append_locked(
                PlatformMotionAnchor(
                    timestamp_ns=timestamp,
                    modeled_rate_x_deg_s=0.0,
                    modeled_rate_y_deg_s=0.0,
                    commanded_rate_x_deg_s=0.0,
                    commanded_rate_y_deg_s=0.0,
                    acceleration_x_deg_s2=self._acceleration_x,
                    acceleration_y_deg_s2=self._acceleration_y,
                )
            )

    def invalidate(self) -> None:
        with self._lock:
            self._invalidate_locked()

    def _rate_at_locked(self, timestamp_ns: int) -> tuple[float, float] | None:
        snapshot = PlatformMotionSnapshot(
            source=PlatformMotionSource.COMMAND_MODEL,
            valid=self._valid,
            anchors=tuple(self._anchors),
        )
        rate = snapshot.rate_at(timestamp_ns)
        if rate is None or rate.rate_x_deg_s is None or rate.rate_y_deg_s is None:
            return None
        return rate.rate_x_deg_s, rate.rate_y_deg_s

    def _append_locked(self, anchor: PlatformMotionAnchor) -> None:
        if self._anchors and anchor.timestamp_ns == self._anchors[-1].timestamp_ns:
            self._anchors[-1] = anchor
        else:
            self._anchors.append(anchor)

    def _invalidate_locked(self) -> None:
        self._valid = False
        self._anchors.clear()

    @staticmethod
    def _timestamp(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("timestamp_ns must be an integer")
        if value < 0:
            raise ValueError("timestamp_ns must be non-negative")
        return value

    @staticmethod
    def _finite(value: float, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _positive_finite(cls, value: float, *, name: str) -> float:
        result = cls._finite(value, name=name)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result


__all__ = ["CommandMotionModel"]
