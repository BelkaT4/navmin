"""Aggregated, bounded stereo runtime observability."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from threading import Event, Lock, Thread
from time import monotonic, monotonic_ns
from typing import Any

from navmin.core.mediator import Mediator, StereoDistancePublicationDiagnostics
from navmin.vision.pipeline import CameraRuntimeDiagnostics, VisionPipeline
from navmin.vision.stereo_distance_worker import (
    StereoDistanceWorker,
    StereoDistanceWorkerDiagnostics,
)
from navmin.vision.stereo_pairing import StereoPairer, StereoPairerDiagnostics
from navmin.vision.stereo_target_join import (
    StereoTargetJoiner,
    StereoTargetJoinerDiagnostics,
)
from navmin.vision.stereo_timestamp_pairing import (
    TimestampStereoPairer,
    TimestampStereoPairerDiagnostics,
)

LOGGER = logging.getLogger(__name__)

STEREO_DIAGNOSTIC_LOG_INTERVAL_S = 5.0


@dataclass(frozen=True, slots=True)
class StereoRuntimeDiagnosticSnapshot:
    """Structured evidence sampled from each state owner."""

    captured_monotonic_ns: int
    left_camera: CameraRuntimeDiagnostics
    right_camera: CameraRuntimeDiagnostics
    exact_pairing: StereoPairerDiagnostics
    timestamp_pairing: TimestampStereoPairerDiagnostics
    target_join: StereoTargetJoinerDiagnostics | None
    distance_worker: StereoDistanceWorkerDiagnostics | None
    publication: StereoDistancePublicationDiagnostics | None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


class StereoRuntimeDiagnostics:
    """Read-only composition of diagnostics owned by production components."""

    def __init__(
        self,
        *,
        left_pipeline: VisionPipeline,
        right_pipeline: VisionPipeline,
        exact_pairer: StereoPairer,
        timestamp_pairer: TimestampStereoPairer,
        target_joiner: StereoTargetJoiner | None = None,
        distance_worker: StereoDistanceWorker | None = None,
        mediator: Mediator | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self._left_pipeline = left_pipeline
        self._right_pipeline = right_pipeline
        self._exact_pairer = exact_pairer
        self._timestamp_pairer = timestamp_pairer
        self._target_joiner = target_joiner
        self._distance_worker = distance_worker
        self._mediator = mediator
        self._clock_ns = clock_ns

    def snapshot(self) -> StereoRuntimeDiagnosticSnapshot:
        return StereoRuntimeDiagnosticSnapshot(
            captured_monotonic_ns=self._clock_ns(),
            left_camera=self._left_pipeline.diagnostics,
            right_camera=self._right_pipeline.diagnostics,
            exact_pairing=self._exact_pairer.diagnostics,
            timestamp_pairing=self._timestamp_pairer.diagnostics,
            target_join=(
                None if self._target_joiner is None else self._target_joiner.diagnostics
            ),
            distance_worker=(
                None if self._distance_worker is None else self._distance_worker.diagnostics
            ),
            publication=(
                None
                if self._mediator is None
                else self._mediator.stereo_distance_diagnostics
            ),
        )


class StereoDiagnosticsReporter:
    """Periodic one-line INFO reporter with a deterministic direct-tick seam."""

    def __init__(
        self,
        diagnostics: StereoRuntimeDiagnostics,
        *,
        interval_s: float = STEREO_DIAGNOSTIC_LOG_INTERVAL_S,
        rate_clock: Callable[[], float] = monotonic,
        logger: logging.Logger = LOGGER,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be > 0")
        self.diagnostics = diagnostics
        self._interval_s = interval_s
        self._rate_clock = rate_clock
        self._logger = logger
        self._stop_event = Event()
        self._state_lock = Lock()
        self._thread: Thread | None = None
        self._last_rate_time: float | None = None
        self._last_pair_count = 0

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("StereoDiagnosticsReporter is one-shot")
            self._thread = Thread(
                target=self._run,
                name="stereo-diagnostics",
                daemon=False,
            )
            thread = self._thread
        thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        with self._state_lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def is_alive(self) -> bool:
        with self._state_lock:
            thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def stop(self, timeout: float = 2.0) -> bool:
        self.request_stop()
        self.join(timeout)
        return not self.is_alive()

    def shutdown(self, timeout: float = 2.0) -> None:
        if not self.stop(timeout):
            raise RuntimeError(
                f"StereoDiagnosticsReporter did not stop within {timeout:.3f} seconds"
            )

    def emit_once(self) -> StereoRuntimeDiagnosticSnapshot:
        snapshot = self.diagnostics.snapshot()
        pair_rate_hz = self._pair_rate(snapshot)
        self._logger.info(_format_summary(snapshot, pair_rate_hz=pair_rate_hz))
        return snapshot

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self.emit_once()

    def _pair_rate(self, snapshot: StereoRuntimeDiagnosticSnapshot) -> float:
        now = self._rate_clock()
        pair_count = (
            snapshot.exact_pairing.exact_pairs_emitted
            + snapshot.timestamp_pairing.timestamp_pairs_emitted
        )
        with self._state_lock:
            previous_time = self._last_rate_time
            previous_count = self._last_pair_count
            self._last_rate_time = now
            self._last_pair_count = pair_count
        if previous_time is None or now <= previous_time:
            return 0.0
        return max(0, pair_count - previous_count) / (now - previous_time)


def _format_summary(
    snapshot: StereoRuntimeDiagnosticSnapshot,
    *,
    pair_rate_hz: float,
) -> str:
    timestamp = snapshot.timestamp_pairing
    join = snapshot.target_join
    worker = snapshot.distance_worker
    publication = snapshot.publication
    fields: tuple[tuple[str, object], ...] = (
        ("left_generation", snapshot.left_camera.generation),
        ("left_frames", snapshot.left_camera.frames_received),
        ("left_fps", snapshot.left_camera.received_fps),
        ("right_generation", snapshot.right_camera.generation),
        ("right_frames", snapshot.right_camera.frames_received),
        ("right_fps", snapshot.right_camera.received_fps),
        ("exact_pairs", snapshot.exact_pairing.exact_pairs_emitted),
        ("timestamp_pairs", timestamp.timestamp_pairs_emitted),
        ("pair_rate_hz", pair_rate_hz),
        ("skew_p50_ms", _ns_to_ms(timestamp.skew.p50_ns)),
        ("skew_p95_ms", _ns_to_ms(timestamp.skew.p95_ns)),
        ("skew_max_ms", _ns_to_ms(timestamp.skew.maximum_ns)),
        ("join_ok", None if join is None else join.target_observations_emitted),
        ("sgbm_ok", None if worker is None else worker.computations_succeeded),
        ("sgbm_fail", None if worker is None else worker.computations_failed),
        (
            "sgbm_p95_ms",
            None if worker is None else _ns_to_ms(worker.compute_duration.p95_ns),
        ),
        ("distance_accept", None if publication is None else publication.accepted),
        (
            "distance_stale",
            None
            if publication is None
            else publication.rejected_stale + publication.stale_invalidations,
        ),
    )
    return "STEREO_DIAG " + " ".join(
        f"{key}={_format_value(value)}" for key, value in fields
    )


def _format_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _ns_to_ms(value_ns: int | None) -> float | None:
    return None if value_ns is None else value_ns / 1_000_000.0


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "STEREO_DIAGNOSTIC_LOG_INTERVAL_S",
    "StereoDiagnosticsReporter",
    "StereoRuntimeDiagnosticSnapshot",
    "StereoRuntimeDiagnostics",
]
