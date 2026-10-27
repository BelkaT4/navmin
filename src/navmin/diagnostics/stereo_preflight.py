"""Stereo camera/pairing hardware preflight without target or distance compute."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

from navmin.contracts import CameraRole, CameraSessionStarted, CameraState, FramePacket
from navmin.launcher import LoadedApplicationInputs
from navmin.vision.camera_source import create_camera_source
from navmin.vision.camera_worker import CameraWorker, SourceFactory, build_camera_worker
from navmin.vision.gstreamer_source import (
    GStreamerRtpJpegSource,
    GStreamerRtspSource,
    initialize_gstreamer_runtime,
)
from navmin.vision.pipeline import stereo_left_corrector, stereo_right_corrector
from navmin.vision.stereo_pairing import StereoPairer
from navmin.vision.stereo_timestamp_pairing import TimestampStereoPairer

from .stereo_runtime import StereoRuntimeDiagnostics, StereoRuntimeDiagnosticSnapshot

_POLL_INTERVAL_S = 0.05
_SHUTDOWN_TIMEOUT_S = 2.0


class StereoHardwarePreflightError(RuntimeError):
    """The preflight could not be constructed or completed safely."""


class StereoPreflightPairingSink:
    """Feed corrected frames to both production pairers for characterization."""

    def __init__(
        self,
        exact_pairer: StereoPairer,
        timestamp_pairer: TimestampStereoPairer,
    ) -> None:
        self._exact_pairer = exact_pairer
        self._timestamp_pairer = timestamp_pairer

    def on_camera_session_started(self, session: CameraSessionStarted) -> None:
        self._exact_pairer.on_camera_session_started(session)
        self._timestamp_pairer.on_camera_session_started(session)

    def on_corrected_frame(self, frame: FramePacket) -> None:
        self._exact_pairer.on_corrected_frame(frame)
        self._timestamp_pairer.on_corrected_frame(frame)


@dataclass(frozen=True, slots=True)
class StereoPreflightComponents:
    left_worker: CameraWorker
    right_worker: CameraWorker
    exact_pairer: StereoPairer
    timestamp_pairer: TimestampStereoPairer
    diagnostics: StereoRuntimeDiagnostics


@dataclass(frozen=True, slots=True)
class StereoHardwarePreflightOutcome:
    duration_seconds: float
    snapshot: StereoRuntimeDiagnosticSnapshot
    hard_failures: tuple[str, ...]
    clean_shutdown: bool

    @property
    def passed(self) -> bool:
        return not self.hard_failures and self.clean_shutdown

    def to_dict(self) -> dict[str, Any]:
        timestamp = self.snapshot.timestamp_pairing
        return {
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "captured_monotonic_ns": self.snapshot.captured_monotonic_ns,
            "cameras": {
                "stereo_left": _camera_payload(self.snapshot.left_camera),
                "stereo_right": _camera_payload(self.snapshot.right_camera),
            },
            "exact_pairing": self.snapshot.to_dict()["exact_pairing"],
            "timestamp_pairing": {
                **self.snapshot.to_dict()["timestamp_pairing"],
                "max_timestamp_skew_ms": timestamp.max_timestamp_skew_ns
                / 1_000_000.0,
                "pair_rate_hz": (
                    timestamp.timestamp_pairs_emitted / self.duration_seconds
                    if self.duration_seconds > 0.0
                    else 0.0
                ),
                "skew_ms": {
                    "latest": _ns_to_ms(timestamp.skew.latest_ns),
                    "minimum": _ns_to_ms(timestamp.skew.minimum_ns),
                    "maximum": _ns_to_ms(timestamp.skew.maximum_ns),
                    "p50": _ns_to_ms(timestamp.skew.p50_ns),
                    "p95": _ns_to_ms(timestamp.skew.p95_ns),
                },
            },
            "workers_stopped": self.clean_shutdown,
            "hard_failures": list(self.hard_failures),
        }


class StereoHardwarePreflight:
    """Own a bounded two-camera production acquisition/pairing run."""

    def __init__(
        self,
        components: StereoPreflightComponents,
        *,
        clock: Callable[[], float] = monotonic,
        wait: Callable[[float], object] | None = None,
    ) -> None:
        self._components = components
        self._clock = clock
        self._wait = wait or Event().wait
        self._run_attempted = False

    def run(self, duration_seconds: float) -> StereoHardwarePreflightOutcome:
        if duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be > 0")
        if self._run_attempted:
            raise RuntimeError("StereoHardwarePreflight is one-shot")
        self._run_attempted = True

        components = self._components
        workers = (components.left_worker, components.right_worker)
        if any(
            isinstance(worker.source, (GStreamerRtpJpegSource, GStreamerRtspSource))
            for worker in workers
        ):
            initialize_gstreamer_runtime()

        started_at = self._clock()
        deadline = started_at + duration_seconds
        stopped_early: list[CameraRole] = []
        left_started = False
        right_started = False
        snapshot: StereoRuntimeDiagnosticSnapshot | None = None
        left_stopped = True
        right_stopped = True
        try:
            components.left_worker.start()
            left_started = True
            components.right_worker.start()
            right_started = True
            while self._clock() < deadline:
                for worker in workers:
                    if not worker.is_alive() and worker.pipeline.camera not in stopped_early:
                        stopped_early.append(worker.pipeline.camera)
                if len(stopped_early) == len(workers):
                    break
                remaining = deadline - self._clock()
                self._wait(min(_POLL_INTERVAL_S, max(0.0, remaining)))
            snapshot = components.diagnostics.snapshot()
        finally:
            if right_started:
                right_stopped = components.right_worker.stop(_SHUTDOWN_TIMEOUT_S)
            if left_started:
                left_stopped = components.left_worker.stop(_SHUTDOWN_TIMEOUT_S)

        if snapshot is None:
            snapshot = components.diagnostics.snapshot()

        ended_at = self._clock()
        actual_duration = max(0.0, min(duration_seconds, ended_at - started_at))
        clean_shutdown = left_stopped and right_stopped
        failures = _hard_failures(
            components,
            snapshot,
            stopped_early=tuple(stopped_early),
            clean_shutdown=clean_shutdown,
        )
        return StereoHardwarePreflightOutcome(
            duration_seconds=actual_duration,
            snapshot=snapshot,
            hard_failures=failures,
            clean_shutdown=clean_shutdown,
        )


def build_stereo_preflight_components(
    inputs: LoadedApplicationInputs,
    *,
    left_source_factory: SourceFactory = create_camera_source,
    right_source_factory: SourceFactory = create_camera_source,
) -> StereoPreflightComponents:
    config = inputs.config
    left_config = config.vision.cameras.stereo_left
    right_config = config.vision.cameras.stereo_right
    if not left_config.enabled or not right_config.enabled:
        raise StereoHardwarePreflightError(
            "Stereo Left and Stereo Right must both be enabled in the selected config"
        )

    stereo = config.vision.distance.stereo
    exact_pairer = StereoPairer(
        right_frame_buffer_size=stereo.right_frame_buffer_size,
        pair_timeout_ms=stereo.pair_timeout_ms,
    )
    timestamp_pairer = TimestampStereoPairer(
        right_frame_buffer_size=stereo.right_frame_buffer_size,
        pair_timeout_ms=stereo.pair_timeout_ms,
    )
    sink = StereoPreflightPairingSink(exact_pairer, timestamp_pairer)
    left_worker = build_camera_worker(
        camera=CameraRole.STEREO_LEFT,
        config=replace(left_config, processing_enabled=False),
        corrector=stereo_left_corrector(inputs.stereo_calibration),
        source_factory=left_source_factory,
        corrected_frame_sink=sink,
    )
    right_worker = build_camera_worker(
        camera=CameraRole.STEREO_RIGHT,
        config=replace(right_config, processing_enabled=False),
        corrector=stereo_right_corrector(inputs.stereo_calibration),
        source_factory=right_source_factory,
        corrected_frame_sink=sink,
    )
    diagnostics = StereoRuntimeDiagnostics(
        left_pipeline=left_worker.pipeline,
        right_pipeline=right_worker.pipeline,
        exact_pairer=exact_pairer,
        timestamp_pairer=timestamp_pairer,
    )
    return StereoPreflightComponents(
        left_worker=left_worker,
        right_worker=right_worker,
        exact_pairer=exact_pairer,
        timestamp_pairer=timestamp_pairer,
        diagnostics=diagnostics,
    )


def write_preflight_json(
    outcome: StereoHardwarePreflightOutcome,
    path: Path,
) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(outcome.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hard_failures(
    components: StereoPreflightComponents,
    snapshot: StereoRuntimeDiagnosticSnapshot,
    *,
    stopped_early: tuple[CameraRole, ...],
    clean_shutdown: bool,
) -> tuple[str, ...]:
    failures: list[str] = []
    for role, camera, worker in (
        (CameraRole.STEREO_LEFT, snapshot.left_camera, components.left_worker),
        (CameraRole.STEREO_RIGHT, snapshot.right_camera, components.right_worker),
    ):
        if camera.frames_received == 0:
            failures.append(f"{role.value}: no decoded frames")
        status = worker.pipeline.status.get()
        if status is not None and status.state is CameraState.ERROR:
            details = status.message or status.error_code or "unknown camera error"
            failures.append(f"{role.value}: worker/source failed: {details}")
        elif role in stopped_early:
            failures.append(f"{role.value}: worker stopped before preflight deadline")
    if not clean_shutdown:
        failures.append("camera workers did not stop cleanly")
    return tuple(dict.fromkeys(failures))


def _camera_payload(camera: Any) -> dict[str, Any]:
    return {
        "enabled": True,
        "generation": camera.generation,
        "frame_count": camera.frames_received,
        "decoded_resolution": (
            None
            if camera.decoded_width is None or camera.decoded_height is None
            else {
                "width": camera.decoded_width,
                "height": camera.decoded_height,
            }
        ),
        "received_fps": camera.received_fps,
        "capture_id_present_count": camera.capture_id_present_count,
        "capture_id_missing_count": camera.capture_id_missing_count,
        "first_frame_timestamp_ns": camera.first_frame_timestamp_ns,
        "latest_frame_timestamp_ns": camera.latest_frame_timestamp_ns,
    }


def _ns_to_ms(value_ns: int | None) -> float | None:
    return None if value_ns is None else value_ns / 1_000_000.0


__all__ = [
    "StereoHardwarePreflight",
    "StereoHardwarePreflightError",
    "StereoHardwarePreflightOutcome",
    "StereoPreflightComponents",
    "StereoPreflightPairingSink",
    "build_stereo_preflight_components",
    "write_preflight_json",
]
