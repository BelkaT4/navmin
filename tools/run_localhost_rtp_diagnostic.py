"""Exercise the production RTP/JPEG camera path with localhost senders."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
from dataclasses import dataclass, replace
from itertools import pairwise
from queue import Empty
from time import monotonic, monotonic_ns, sleep

import numpy as np

from navmin.contracts import CameraRole, CameraState
from navmin.diagnostics.localhost_rtp import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    LocalhostRtpJpegSender,
    PreflightResult,
    RtpJpegSenderConfig,
    SenderProcessError,
    check_gstreamer_runtime,
    diagnostic_camera_config,
    diagnostic_overview_calibration,
    diagnostic_stereo_calibration,
)
from navmin.vision.camera_worker import CameraWorker, build_camera_worker
from navmin.vision.gstreamer_source import CameraSourceError, GStreamerRtpJpegSource
from navmin.vision.pipeline import overview_corrector, stereo_left_corrector

_WAIT_TIMEOUT_SECONDS = 5.0
_STALE_WAIT_SECONDS = 0.65
_WRONG_WIDTH = 352


class DiagnosticFailure(RuntimeError):
    """One required diagnostic observation was not obtained."""


@dataclass(frozen=True)
class GateOutcome:
    gate: str
    status: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class CameraObservation:
    camera: CameraRole
    port: int
    generation: int
    result_count: int
    observed_rate_hz: float
    first_frame_id: int
    last_frame_id: int
    first_receive_timestamp_ns: int
    last_receive_timestamp_ns: int


class _ResourceTracker:
    def __init__(self) -> None:
        self.senders: list[LocalhostRtpJpegSender] = []
        self.workers: list[CameraWorker] = []
        self.sources: list[GStreamerRtpJpegSource] = []

    def new_sender(self, config: RtpJpegSenderConfig) -> LocalhostRtpJpegSender:
        sender = LocalhostRtpJpegSender(config)
        self.senders.append(sender)
        return sender

    def remember_worker(self, worker: CameraWorker) -> None:
        self.workers.append(worker)

    def remember_source(self, source: GStreamerRtpJpegSource) -> None:
        self.sources.append(source)

    def cleanup(self) -> tuple[str, ...]:
        errors: list[str] = []
        for sender in reversed(self.senders):
            try:
                sender.stop()
            except SenderProcessError as exc:
                errors.append(f"sender port {sender.config.port}: {exc}")
        for worker in reversed(self.workers):
            if worker.is_alive() and not worker.stop(timeout=2.0):
                errors.append(f"CameraWorker {worker.pipeline.camera.value} did not stop")
        for source in reversed(self.sources):
            try:
                source.stop()
            except CameraSourceError as exc:
                errors.append(f"receiver port {source.pipeline_description}: {exc}")
        return tuple(errors)


def _wait_until(predicate, *, timeout: float, failure: str) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise DiagnosticFailure(failure)


def _raise_source_failure(source: GStreamerRtpJpegSource) -> None:
    failure = source.failure
    if failure is not None:
        raise DiagnosticFailure(str(failure))


def _collect_decoded_frames(
    source: GStreamerRtpJpegSource,
    *,
    minimum: int = 4,
    timeout: float = _WAIT_TIMEOUT_SECONDS,
) -> list:
    frames = []
    deadline = monotonic() + timeout
    while monotonic() < deadline and len(frames) < minimum:
        _raise_source_failure(source)
        frame = source.read()
        if frame is not None:
            frames.append(frame)
        else:
            sleep(0.01)
    if len(frames) < minimum:
        raise DiagnosticFailure(
            f"only {len(frames)} decoded frames arrived; expected at least {minimum}"
        )
    return frames


def _validate_decoded_frames(frames: list) -> None:
    if any(frame.image.dtype != np.uint8 for frame in frames):
        raise DiagnosticFailure("decoded sample dtype is not uint8")
    if any(frame.image.shape != (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3) for frame in frames):
        raise DiagnosticFailure("decoded sample shape does not match 320x240 BGR")
    if any(frame.capture_id is not None for frame in frames):
        raise DiagnosticFailure("RTP/JPEG source unexpectedly published capture_id")
    timestamps = [frame.receive_timestamp_ns for frame in frames]
    if any(timestamp is None for timestamp in timestamps):
        raise DiagnosticFailure("decoded sample has no receive timestamp")
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise DiagnosticFailure("receive timestamps did not progress")
    if np.array_equal(frames[0].image, frames[-1].image):
        raise DiagnosticFailure("synthetic image did not change across decoded frames")


def _run_single_transport_ordering(
    *,
    port: int,
    receiver_first: bool,
    resources: _ResourceTracker,
) -> tuple[int, int]:
    source = GStreamerRtpJpegSource(diagnostic_camera_config(port))
    sender = resources.new_sender(RtpJpegSenderConfig(port=port))
    try:
        if receiver_first:
            source.start()
            resources.remember_source(source)
            sender.start()
        else:
            sender.start()
            sleep(0.3)
            source.start()
            resources.remember_source(source)
        frames = _collect_decoded_frames(source)
        _validate_decoded_frames(frames)
        if not source.started:
            raise DiagnosticFailure("production receiver stopped while frames arrived")

        # Do not consume for several sender periods, then prove there is no replay queue.
        sleep(0.35)
        freshest = _collect_decoded_frames(source, minimum=1)[0]
        drained = 1
        for _ in range(16):
            if source.read() is not None:
                drained += 1
        if drained > 2:
            raise DiagnosticFailure(
                f"consumer pause exposed a replay backlog of {drained} frames"
            )
        return len(frames) + 1, freshest.receive_timestamp_ns
    finally:
        had_primary_failure = sys.exception() is not None
        cleanup_errors: list[str] = []
        try:
            sender.stop()
        except SenderProcessError as exc:
            cleanup_errors.append(f"sender cleanup: {exc}")
        try:
            source.stop()
        except CameraSourceError as exc:
            cleanup_errors.append(f"receiver cleanup: {exc}")
        if cleanup_errors and not had_primary_failure:
            raise DiagnosticFailure("; ".join(cleanup_errors))


def _gate_b(port: int, resources: _ResourceTracker) -> GateOutcome:
    receiver_first_count, first_timestamp = _run_single_transport_ordering(
        port=port,
        receiver_first=True,
        resources=resources,
    )
    sender_first_count, second_timestamp = _run_single_transport_ordering(
        port=port,
        receiver_first=False,
        resources=resources,
    )
    return GateOutcome(
        "B",
        "GREEN",
        (
            f"receiver-first decoded frames: {receiver_first_count}",
            f"sender-first decoded frames: {sender_first_count}",
            "samples: uint8 240x320x3 BGR caps, capture_id=None",
            f"receive timestamps observed: {first_timestamp}, {second_timestamp}",
            "consumer pause: freshest-only, no replay backlog",
        ),
    )


def _worker_for(camera: CameraRole, port: int) -> CameraWorker:
    corrector = (
        overview_corrector(diagnostic_overview_calibration())
        if camera is CameraRole.OVERVIEW
        else stereo_left_corrector(diagnostic_stereo_calibration())
    )
    return build_camera_worker(
        camera=camera,
        config=replace(diagnostic_camera_config(port), processing_enabled=True),
        corrector=corrector,
        idle_wait_s=0.005,
    )


def _status_or_failure(worker: CameraWorker):
    status = worker.pipeline.status.get()
    if status is not None and status.state is CameraState.ERROR:
        raise DiagnosticFailure(
            f"{worker.pipeline.camera.value} entered ERROR: "
            f"{status.error_code}: {status.message}"
        )
    if not worker.is_alive() and (
        status is None or status.state is not CameraState.STOPPED
    ):
        raise DiagnosticFailure(f"{worker.pipeline.camera.value} worker exited early")
    return status


def _wait_for_worker_progress(
    worker: CameraWorker,
    *,
    after_revision: int = 0,
    minimum_revision_delta: int = 3,
) -> None:
    def progressed() -> bool:
        status = _status_or_failure(worker)
        snapshot = worker.pipeline.latest_result.snapshot()
        return (
            status is not None
            and status.state is CameraState.ONLINE
            and snapshot.value is not None
            and snapshot.revision >= after_revision + minimum_revision_delta
        )

    _wait_until(
        progressed,
        timeout=_WAIT_TIMEOUT_SECONDS,
        failure=f"{worker.pipeline.camera.value} did not become ONLINE and progress",
    )


def _wait_for_stable_moving_track(
    worker: CameraWorker,
) -> tuple[int, tuple[float, float], tuple[float, float]]:
    deadline = monotonic() + _WAIT_TIMEOUT_SECONDS
    last_revision = 0
    observations: list[tuple[int, int, tuple[float, float]]] = []
    while monotonic() < deadline:
        _status_or_failure(worker)
        snapshot = worker.pipeline.latest_result.snapshot()
        if snapshot.revision <= last_revision or snapshot.value is None:
            sleep(0.01)
            continue
        last_revision = snapshot.revision
        tracked = snapshot.value.tracked_objects
        if len(tracked) != 1:
            sleep(0.01)
            continue
        item = tracked[0]
        center = (
            item.bbox.x + item.bbox.width / 2.0,
            item.bbox.y + item.bbox.height / 2.0,
        )
        if observations and item.track_id != observations[-1][0]:
            observations.clear()
        observations.append((item.track_id, item.age_frames, center))
        if len(observations) >= 4:
            recent = observations[-4:]
            if (
                len({entry[0] for entry in recent}) == 1
                and all(
                    current[1] > previous[1]
                    for previous, current in pairwise(recent)
                )
                and recent[0][2] != recent[-1][2]
            ):
                return recent[-1][0], recent[0][2], recent[-1][2]
        sleep(0.01)
    raise DiagnosticFailure(
        f"{worker.pipeline.camera.value} did not produce one stable moving Legacy14 track"
    )


def _assert_result_contract(worker: CameraWorker, *, port: int) -> tuple[int, int]:
    pipeline = worker.pipeline
    result = pipeline.latest_result.get()
    status = pipeline.status.get()
    if result is None or status is None:
        raise DiagnosticFailure("worker did not publish result/status")
    if pipeline.generation != 1 or result.frame.generation != 1:
        raise DiagnosticFailure("first worker session generation is not 1")
    if status.state is not CameraState.ONLINE or status.generation != 1:
        raise DiagnosticFailure("worker CameraStatus is not ONLINE generation 1")
    if result.frame.image.shape != (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3):
        raise DiagnosticFailure("corrected working frame dimensions are wrong")
    if result.frame.image.flags.writeable:
        raise DiagnosticFailure("working FramePacket.image is writeable")
    if result.frame.capture_id is not None:
        raise DiagnosticFailure("localhost RTP frame unexpectedly has capture_id")
    if status.last_receive_timestamp_ns != result.frame.receive_timestamp_ns:
        raise DiagnosticFailure("CameraStatus receive timestamp does not match result")
    if not isinstance(worker.source, GStreamerRtpJpegSource):
        raise DiagnosticFailure("worker did not use production GStreamerRtpJpegSource")
    if f"port={port}" not in worker.source.pipeline_description:
        raise DiagnosticFailure("worker receiver is bound to the wrong port")
    return result.frame.frame_id, result.frame.receive_timestamp_ns


def _gate_c(port: int, resources: _ResourceTracker) -> GateOutcome:
    worker = _worker_for(CameraRole.OVERVIEW, port)
    resources.remember_worker(worker)
    sender = resources.new_sender(RtpJpegSenderConfig(port=port))
    try:
        worker.start()
        _wait_until(
            lambda: worker.pipeline.generation == 1,
            timeout=2.0,
            failure="CameraWorker did not publish generation 1",
        )
        session = worker.pipeline.session_barriers.receive(timeout=1.0)
        sender.start()
        _wait_for_worker_progress(worker)
        track_id, first_center, last_center = _wait_for_stable_moving_track(worker)
        frame_id, receive_timestamp_ns = _assert_result_contract(worker, port=port)
        if session.generation != 1 or session.camera is not CameraRole.OVERVIEW:
            raise DiagnosticFailure("CameraSessionStarted does not match worker session")
        return GateOutcome(
            "C",
            "GREEN",
            (
                "path: GStreamerRtpJpegSource -> CameraWorker -> VisionPipeline",
                "CameraSessionStarted: overview generation=1",
                f"latest frame_id: {frame_id}",
                f"last_receive_timestamp_ns: {receive_timestamp_ns}",
                "working frame: 320x240, uint8, read-only, capture_id=None",
                (
                    f"Legacy14 track_id={track_id} moved "
                    f"{first_center} -> {last_center}"
                ),
            ),
        )
    finally:
        had_primary_failure = sys.exception() is not None
        cleanup_errors: list[str] = []
        try:
            sender.stop()
        except SenderProcessError as exc:
            cleanup_errors.append(f"sender cleanup: {exc}")
        if worker.is_alive() and not worker.stop(timeout=2.0):
            cleanup_errors.append("Gate C CameraWorker did not stop within 2 seconds")
        if cleanup_errors and not had_primary_failure:
            raise DiagnosticFailure("; ".join(cleanup_errors))


def _observe_camera(
    worker: CameraWorker,
    *,
    port: int,
    start_revision: int,
    start_frame_id: int,
    start_timestamp_ns: int,
    elapsed: float,
) -> CameraObservation:
    snapshot = worker.pipeline.latest_result.snapshot()
    result = snapshot.value
    if result is None:
        raise DiagnosticFailure(f"{worker.pipeline.camera.value} lost latest result")
    result_count = snapshot.revision - start_revision
    if result_count <= 0 or result.frame.frame_id <= start_frame_id:
        raise DiagnosticFailure(f"{worker.pipeline.camera.value} did not progress")
    if result.frame.receive_timestamp_ns <= start_timestamp_ns:
        raise DiagnosticFailure(
            f"{worker.pipeline.camera.value} receive timestamp did not progress"
        )
    return CameraObservation(
        camera=worker.pipeline.camera,
        port=port,
        generation=worker.pipeline.generation,
        result_count=result_count,
        observed_rate_hz=result_count / elapsed,
        first_frame_id=start_frame_id,
        last_frame_id=result.frame.frame_id,
        first_receive_timestamp_ns=start_timestamp_ns,
        last_receive_timestamp_ns=result.frame.receive_timestamp_ns,
    )


def _gate_d_and_e(
    *,
    overview_port: int,
    stereo_left_port: int,
    duration_seconds: float,
    resources: _ResourceTracker,
) -> tuple[GateOutcome, GateOutcome]:
    overview = _worker_for(CameraRole.OVERVIEW, overview_port)
    stereo = _worker_for(CameraRole.STEREO_LEFT, stereo_left_port)
    resources.remember_worker(overview)
    resources.remember_worker(stereo)
    overview_sender = resources.new_sender(
        RtpJpegSenderConfig(port=overview_port, camera=CameraRole.OVERVIEW)
    )
    stereo_sender = resources.new_sender(
        RtpJpegSenderConfig(
            port=stereo_left_port,
            camera=CameraRole.STEREO_LEFT,
        )
    )
    resumed_sender: LocalhostRtpJpegSender | None = None
    try:
        overview.start()
        stereo.start()
        _wait_until(
            lambda: overview.pipeline.generation == stereo.pipeline.generation == 1,
            timeout=2.0,
            failure="dual CameraWorkers did not start generation 1",
        )
        overview_session = overview.pipeline.session_barriers.receive(timeout=1.0)
        stereo_session = stereo.pipeline.session_barriers.receive(timeout=1.0)
        overview_sender.start()
        stereo_sender.start()
        _wait_for_worker_progress(overview)
        _wait_for_worker_progress(stereo)
        overview_track = _wait_for_stable_moving_track(overview)
        stereo_track = _wait_for_stable_moving_track(stereo)
        overview_frame_id, overview_timestamp = _assert_result_contract(
            overview, port=overview_port
        )
        stereo_frame_id, stereo_timestamp = _assert_result_contract(
            stereo, port=stereo_left_port
        )
        overview_start_revision = overview.pipeline.latest_result.snapshot().revision
        stereo_start_revision = stereo.pipeline.latest_result.snapshot().revision

        started_at = monotonic()
        deadline = started_at + duration_seconds
        while monotonic() < deadline:
            _status_or_failure(overview)
            _status_or_failure(stereo)
            sleep(min(0.05, max(0.0, deadline - monotonic())))
        elapsed = max(monotonic() - started_at, 1e-9)
        overview_observation = _observe_camera(
            overview,
            port=overview_port,
            start_revision=overview_start_revision,
            start_frame_id=overview_frame_id,
            start_timestamp_ns=overview_timestamp,
            elapsed=elapsed,
        )
        stereo_observation = _observe_camera(
            stereo,
            port=stereo_left_port,
            start_revision=stereo_start_revision,
            start_frame_id=stereo_frame_id,
            start_timestamp_ns=stereo_timestamp,
            elapsed=elapsed,
        )
        if overview_session.camera is stereo_session.camera:
            raise DiagnosticFailure("dual streams published the same camera role")

        gate_d = GateOutcome(
            "D",
            "GREEN",
            (
                _format_camera_observation(overview_observation),
                _format_camera_observation(stereo_observation),
                "both streams ONLINE, generation=1, independent ports and roles",
                (
                    "Legacy14 stable moving tracks: "
                    f"overview={overview_track[0]}, stereo-left={stereo_track[0]}"
                ),
            ),
        )

        overview_sender.stop()
        sleep(0.2)
        overview_stopped = overview.pipeline.latest_result.snapshot()
        stereo_during_stop = stereo.pipeline.latest_result.snapshot().revision
        sleep(_STALE_WAIT_SECONDS)
        overview_after_stale = overview.pipeline.latest_result.snapshot()
        stereo_after_stale = stereo.pipeline.latest_result.snapshot()
        overview_status = overview.pipeline.status.get()
        if overview_after_stale.revision != overview_stopped.revision:
            raise DiagnosticFailure("Overview revisions continued after sender stop")
        if stereo_after_stale.revision <= stereo_during_stop:
            raise DiagnosticFailure("Stereo Left stopped progressing with Overview sender")
        if overview.pipeline.generation != 1 or not overview.is_alive():
            raise DiagnosticFailure("Overview worker/generation changed during UDP silence")
        if overview_status is None or overview_status.last_receive_timestamp_ns is None:
            raise DiagnosticFailure("Overview lost its last receive timestamp")
        stale_age_ms = (
            monotonic_ns() - overview_status.last_receive_timestamp_ns
        ) / 1_000_000.0
        if stale_age_ms <= 500.0:
            raise DiagnosticFailure(
                f"Overview timestamp age only reached {stale_age_ms:.1f} ms"
            )

        resumed_sender = resources.new_sender(RtpJpegSenderConfig(port=overview_port))
        resumed_sender.start()

        def overview_resumed() -> bool:
            _status_or_failure(overview)
            snapshot = overview.pipeline.latest_result.snapshot()
            return (
                snapshot.revision > overview_after_stale.revision
                and snapshot.value is not None
                and snapshot.value.frame.receive_timestamp_ns
                > overview_status.last_receive_timestamp_ns
            )

        _wait_until(
            overview_resumed,
            timeout=_WAIT_TIMEOUT_SECONDS,
            failure="Overview did not resume on the same running receiver",
        )
        resumed_status = overview.pipeline.status.get()
        if (
            overview.pipeline.generation != 1
            or resumed_status is None
            or resumed_status.state is not CameraState.ONLINE
        ):
            raise DiagnosticFailure("Overview resume changed generation/status")
        gate_e = GateOutcome(
            "E",
            "GREEN",
            (
                f"Overview revision frozen at {overview_after_stale.revision}",
                f"Overview last-frame age after silence: {stale_age_ms:.1f} ms",
                (
                    "Stereo Left revision progressed "
                    f"{stereo_during_stop} -> {stereo_after_stale.revision}"
                ),
                "Overview resumed on port reuse with same worker and generation=1",
                "CameraStatus after resume: ONLINE",
            ),
        )
        return gate_d, gate_e
    finally:
        had_primary_failure = sys.exception() is not None
        cleanup_errors: list[str] = []
        for sender in (resumed_sender, overview_sender, stereo_sender):
            if sender is None:
                continue
            try:
                sender.stop()
            except SenderProcessError as exc:
                cleanup_errors.append(
                    f"sender port {sender.config.port} cleanup: {exc}"
                )
        for worker in (overview, stereo):
            if worker.is_alive() and not worker.stop(timeout=2.0):
                cleanup_errors.append(
                    f"{worker.pipeline.camera.value} did not stop within 2 seconds"
                )
        if cleanup_errors and not had_primary_failure:
            raise DiagnosticFailure("; ".join(cleanup_errors))


def _format_camera_observation(observation: CameraObservation) -> str:
    return (
        f"{observation.camera.value}: port={observation.port}, "
        f"results={observation.result_count}, "
        f"rate={observation.observed_rate_hz:.1f} Hz, "
        f"frame_id={observation.first_frame_id}->{observation.last_frame_id}, "
        f"generation={observation.generation}, dimensions=320x240"
    )


def _gate_f(port: int, resources: _ResourceTracker) -> GateOutcome:
    worker = _worker_for(CameraRole.OVERVIEW, port)
    resources.remember_worker(worker)
    sender = resources.new_sender(
        RtpJpegSenderConfig(port=port, width=_WRONG_WIDTH, height=DEFAULT_HEIGHT)
    )
    try:
        worker.start()
        sender.start()

        def failed_exactly() -> bool:
            status = worker.pipeline.status.get()
            return status is not None and status.state is CameraState.ERROR

        _wait_until(
            failed_exactly,
            timeout=_WAIT_TIMEOUT_SECONDS,
            failure="wrong-resolution stream did not reach CameraStatus.ERROR",
        )
        worker.join(timeout=2.0)
        status = worker.pipeline.status.get()
        if worker.is_alive():
            raise DiagnosticFailure("wrong-resolution CameraWorker did not exit")
        if status is None or status.message is None:
            raise DiagnosticFailure("wrong-resolution error has no diagnostic message")
        expected = "source resolution 352x240 does not match calibration 320x240"
        if expected not in status.message:
            raise DiagnosticFailure(
                f"wrong-resolution error did not identify mismatch: {status.message}"
            )
        if worker.pipeline.latest_result.get() is not None:
            raise DiagnosticFailure("wrong-resolution path published a raw fallback")
        return GateOutcome(
            "F",
            "GREEN",
            (
                "sender dimensions: 352x240; calibration: 320x240",
                f"CameraStatus: ERROR ({status.error_code})",
                f"error: {status.message}",
                "latest_result: None; raw/unscaled fallback: absent",
                "worker exit: bounded",
            ),
        )
    finally:
        had_primary_failure = sys.exception() is not None
        cleanup_errors: list[str] = []
        try:
            sender.stop()
        except SenderProcessError as exc:
            cleanup_errors.append(f"sender cleanup: {exc}")
        if worker.is_alive() and not worker.stop(timeout=2.0):
            cleanup_errors.append("Gate F CameraWorker did not stop within 2 seconds")
        if cleanup_errors and not had_primary_failure:
            raise DiagnosticFailure("; ".join(cleanup_errors))


def _assert_ports_bindable(ports: tuple[int, ...]) -> None:
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.bind(("127.0.0.1", port))
            sockets.append(probe)
    except OSError as exc:
        raise DiagnosticFailure(f"UDP port conflict or incomplete cleanup: {exc}") from exc
    finally:
        for probe in sockets:
            probe.close()


def _gate_g(
    *,
    resources: _ResourceTracker,
    ports: tuple[int, int],
    baseline_non_daemon_thread_ids: set[int | None],
) -> GateOutcome:
    cleanup_errors = resources.cleanup()
    if cleanup_errors:
        raise DiagnosticFailure("; ".join(cleanup_errors))
    if any(sender.is_running for sender in resources.senders):
        raise DiagnosticFailure("a sender subprocess is still running")
    if any(worker.is_alive() for worker in resources.workers):
        raise DiagnosticFailure("a CameraWorker is still alive")
    if any(source.started for source in resources.sources):
        raise DiagnosticFailure("a direct GStreamer receiver is still started")
    for worker in resources.workers:
        source = worker.source
        if isinstance(source, GStreamerRtpJpegSource) and source.started:
            raise DiagnosticFailure("a worker-owned Gst pipeline did not reach NULL")
    _assert_ports_bindable(ports)
    sleep(0.1)
    leaked_threads = [
        thread.name
        for thread in threading.enumerate()
        if not thread.daemon
        and thread.ident not in baseline_non_daemon_thread_ids
        and thread is not threading.current_thread()
    ]
    if leaked_threads:
        raise DiagnosticFailure(
            "new non-daemon threads remain after cleanup: " + ", ".join(leaked_threads)
        )
    return GateOutcome(
        "G",
        "GREEN",
        (
            f"sender subprocesses stopped: {len(resources.senders)}",
            f"CameraWorkers stopped: {len(resources.workers)}",
            "all production receiver pipelines stopped (Gst NULL)",
            f"ports reusable: {ports[0]}, {ports[1]}",
            "new non-daemon worker threads: 0",
        ),
    )


def _print_preflight(preflight: PreflightResult) -> None:
    print("Preflight:")
    for check in preflight.checks:
        marker = "OK" if check.ok else "MISSING"
        print(f"  {check.name}: {marker} — {check.detail}")


def _print_outcomes(outcomes: list[GateOutcome], *, result: str) -> None:
    print()
    for outcome in outcomes:
        print(f"GATE {outcome.gate}: {outcome.status}")
        for evidence in outcome.evidence:
            print(f"  {evidence}")
    print()
    print(f"RESULT: {result}")


def run_diagnostic(
    *,
    duration_seconds: float,
    overview_port: int,
    stereo_left_port: int,
) -> bool:
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be > 0")
    if overview_port == stereo_left_port:
        raise ValueError("Overview and Stereo Left ports must be different")
    diagnostic_camera_config(overview_port)
    diagnostic_camera_config(stereo_left_port)

    print("=== LOCALHOST RTP/JPEG DIAGNOSTIC ===")
    print(
        f"Synthetic format: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS} FPS"
    )
    print(f"Overview: 127.0.0.1:{overview_port}")
    print(f"Stereo Left: 127.0.0.1:{stereo_left_port}")
    print()

    preflight = check_gstreamer_runtime()
    _print_preflight(preflight)
    if not preflight.ok:
        failures = tuple(f"{check.name}: {check.detail}" for check in preflight.failures)
        outcomes = [GateOutcome("A", "RED", failures)]
        outcomes.extend(
            GateOutcome(gate, "NOT RUN", ("blocked by GATE A preflight",))
            for gate in ("B", "C", "D", "E", "F", "G")
        )
        _print_outcomes(outcomes, result="FAIL")
        return False

    try:
        _assert_ports_bindable((overview_port, stereo_left_port))
    except DiagnosticFailure as exc:
        outcomes = [GateOutcome("A", "RED", (str(exc),))]
        outcomes.extend(
            GateOutcome(gate, "NOT RUN", ("blocked by GATE A preflight",))
            for gate in ("B", "C", "D", "E", "F", "G")
        )
        _print_outcomes(outcomes, result="FAIL")
        return False
    print("  UDP ports: OK — requested ports are bindable")

    outcomes = [GateOutcome("A", "GREEN", ("all required runtime checks passed",))]
    resources = _ResourceTracker()
    baseline_thread_ids = {
        thread.ident for thread in threading.enumerate() if not thread.daemon
    }
    failure: tuple[str, str] | None = None
    gates = ("B", "C", "D", "E", "F")
    try:
        outcomes.append(_gate_b(overview_port, resources))
        outcomes.append(_gate_c(overview_port, resources))
        gate_d, gate_e = _gate_d_and_e(
            overview_port=overview_port,
            stereo_left_port=stereo_left_port,
            duration_seconds=duration_seconds,
            resources=resources,
        )
        outcomes.extend((gate_d, gate_e))
        outcomes.append(_gate_f(overview_port, resources))
    except (
        DiagnosticFailure,
        SenderProcessError,
        CameraSourceError,
        Empty,
        OSError,
        ValueError,
    ) as exc:
        completed = {outcome.gate for outcome in outcomes}
        failed_gate = next(gate for gate in gates if gate not in completed)
        failure = failed_gate, f"{type(exc).__name__}: {exc}"
        outcomes.append(GateOutcome(failed_gate, "RED", (failure[1],)))
        for gate in gates[gates.index(failed_gate) + 1 :]:
            outcomes.append(
                GateOutcome(gate, "NOT RUN", (f"blocked by GATE {failed_gate}",))
            )
    try:
        outcomes.append(
            _gate_g(
                resources=resources,
                ports=(overview_port, stereo_left_port),
                baseline_non_daemon_thread_ids=baseline_thread_ids,
            )
        )
    except (
        DiagnosticFailure,
        SenderProcessError,
        CameraSourceError,
        Empty,
        OSError,
        ValueError,
    ) as exc:
        outcomes.append(
            GateOutcome("G", "RED", (f"{type(exc).__name__}: {exc}",))
        )
        failure = failure or ("G", str(exc))

    passed = failure is None and all(outcome.status == "GREEN" for outcome in outcomes)
    _print_outcomes(outcomes, result="PASS" if passed else "FAIL")
    return passed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate localhost RTP/JPEG through the production NavMin camera path."
        )
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=30.0,
        help="dual-stream observation duration (default: 30)",
    )
    parser.add_argument("--overview-port", type=int, default=8888)
    parser.add_argument("--stereo-left-port", type=int, default=8889)
    args = parser.parse_args(argv)
    if args.duration_seconds <= 0.0:
        parser.error("--duration-seconds must be > 0")
    if not 1 <= args.overview_port <= 65_535:
        parser.error("--overview-port must be in range 1..65535")
    if not 1 <= args.stereo_left_port <= 65_535:
        parser.error("--stereo-left-port must be in range 1..65535")
    if args.overview_port == args.stereo_left_port:
        parser.error("camera ports must be different")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return 0 if run_diagnostic(
        duration_seconds=args.duration_seconds,
        overview_port=args.overview_port,
        stereo_left_port=args.stereo_left_port,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
