"""Parallel-safe diagnostic recording core.

This module intentionally owns only recording mechanics and the stable on-disk
recording schema.  Application/UI/config integration is a later stage so this
core can be developed in parallel with tracking-observability work.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import monotonic_ns
from typing import Protocol, TextIO

import cv2
import numpy as np

from navmin.contracts import BBox, CameraRole, FramePacket

LOGGER = logging.getLogger(__name__)

RECORDING_SCHEMA_VERSION = 1
DEFAULT_RECORDING_QUEUE_CAPACITY = 32
DEFAULT_STOP_TIMEOUT_S = 5.0
_JSONL_FLUSH_INTERVAL_FRAMES = 30
_VIDEO_FILE_NAME = "video.mkv"
_FRAMES_FILE_NAME = "frames.jsonl"
_METADATA_FILE_NAME = "metadata.json"


class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPING = "stopping"
    FAILED = "failed"


class RecordingSessionStatus(Enum):
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchRegionKind(Enum):
    CIRCLE = "circle"
    FULL_FRAME = "full-frame"


@dataclass(frozen=True)
class RecordingPoint:
    x: float
    y: float


@dataclass(frozen=True)
class RecordingSelectedTarget:
    bbox: BBox
    center: RecordingPoint


@dataclass(frozen=True)
class RecordingBestCandidate:
    bbox: BBox


@dataclass(frozen=True)
class RecordingSearchRegion:
    kind: SearchRegionKind
    center: RecordingPoint | None = None
    radius_px: float | None = None

    def __post_init__(self) -> None:
        if self.kind is SearchRegionKind.CIRCLE:
            if self.center is None or self.radius_px is None or self.radius_px < 0:
                raise ValueError("circle search region requires center and non-negative radius")
            return
        if self.center is not None or self.radius_px is not None:
            raise ValueError("full-frame search region cannot carry circle geometry")


@dataclass(frozen=True)
class RecordingControlPrediction:
    from_point: RecordingPoint
    to_point: RecordingPoint


@dataclass(frozen=True)
class RecordingOverlaySnapshot:
    selected_target: RecordingSelectedTarget | None = None
    best_candidate: RecordingBestCandidate | None = None
    search_region: RecordingSearchRegion | None = None
    control_prediction: RecordingControlPrediction | None = None


@dataclass(frozen=True)
class RecordingTrackingState:
    phase: str | None = None
    status: str | None = None
    target_ref: str | None = None
    diagnostic_reason: str | None = None
    candidate_count: int | None = None


@dataclass(frozen=True)
class RecordingSelection:
    point: RecordingPoint


@dataclass(frozen=True)
class RecordingControlState:
    prediction_enabled: bool
    rate_valid_x: bool
    rate_valid_y: bool


@dataclass(frozen=True)
class RecordingFramePacket:
    """One clean working frame plus same-frame recording annotations.

    The integration layer is responsible for exact-frame joins before creating
    this packet.  The recording core never looks up "latest" diagnostics.
    """

    frame: FramePacket
    relative_time_ns: int
    capture_timestamp_ns: int | None = None
    tracking: RecordingTrackingState | None = None
    selection: RecordingSelection | None = None
    overlay: RecordingOverlaySnapshot = field(default_factory=RecordingOverlaySnapshot)
    control: RecordingControlState | None = None

    def __post_init__(self) -> None:
        if self.relative_time_ns < 0:
            raise ValueError("relative_time_ns must be non-negative")


@dataclass(frozen=True)
class RecordingSessionPaths:
    session_id: str
    session_dir: Path
    video_path: Path
    frames_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class RecordingSessionInfo:
    paths: RecordingSessionPaths
    camera: CameraRole
    width: int
    height: int
    nominal_fps: float
    vision_backend: str
    started_at: datetime
    started_monotonic_ns: int
    codec: str
    video_backend: str


class VideoFrameWriter(Protocol):
    codec: str
    backend: str

    def write(self, image: np.ndarray) -> None: ...

    def close(self) -> None: ...


VideoWriterFactory = type["OpenCvMkvWriter"]


class OpenCvMkvWriter:
    """Streaming Matroska writer with a lossless-first codec policy.

    FFV1 is preferred because recordings are diagnostic evidence and small
    edges/blur must not be altered by a second lossy encode.  MJPG is retained
    as a broadly available intra-frame fallback when FFV1 is unavailable.

    OpenCV's VideoWriter does not expose per-frame PTS.  Exact monotonic timing
    is therefore authoritative in frames.jsonl; container timing uses the
    supplied nominal FPS until a later integration stage can provide a tested
    timestamp-capable GStreamer writer.
    """

    CODEC_CANDIDATES = ("FFV1", "MJPG")

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        nominal_fps: float,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("video dimensions must be positive")
        if nominal_fps <= 0:
            raise ValueError("nominal_fps must be positive")

        self._path = Path(path)
        self._width = width
        self._height = height
        self._writer: cv2.VideoWriter | None = None
        self.codec = ""
        self.backend = "opencv"

        failures: list[str] = []
        for codec in self.CODEC_CANDIDATES:
            writer = cv2.VideoWriter(
                str(self._path),
                cv2.VideoWriter_fourcc(*codec),
                float(nominal_fps),
                (width, height),
            )
            if writer.isOpened():
                self._writer = writer
                self.codec = codec
                try:
                    self.backend = writer.getBackendName()
                except cv2.error:
                    self.backend = "opencv"
                break
            writer.release()
            failures.append(codec)

        if self._writer is None:
            joined = ", ".join(failures) or "none"
            raise RuntimeError(
                f"unable to open Matroska writer; attempted codecs: {joined}"
            )

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("video writer is closed")
        if image.dtype != np.uint8:
            raise ValueError("recording frame must be uint8")
        if image.shape != (self._height, self._width, 3):
            raise ValueError(
                "recording frame shape mismatch: "
                f"expected {(self._height, self._width, 3)}, got {image.shape}"
            )
        self._writer.write(image)

    def close(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.release()


class RecordingRuntime:
    """Bounded asynchronous writer for one diagnostic recording session."""

    def __init__(
        self,
        recording_directory: Path,
        *,
        queue_capacity: int = DEFAULT_RECORDING_QUEUE_CAPACITY,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        monotonic_clock_ns=monotonic_ns,
        wall_clock=datetime.now,
        video_writer_factory=OpenCvMkvWriter,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if stop_timeout_s <= 0:
            raise ValueError("stop_timeout_s must be positive")
        self._recording_directory = Path(recording_directory)
        self._queue_capacity = queue_capacity
        self._stop_timeout_s = stop_timeout_s
        self._monotonic_clock_ns = monotonic_clock_ns
        self._wall_clock = wall_clock
        self._video_writer_factory = video_writer_factory

        self._lock = threading.Lock()
        self._state = RecordingState.IDLE
        self._session: RecordingSessionInfo | None = None
        self._queue: queue.Queue[RecordingFramePacket] | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._video_writer: VideoFrameWriter | None = None
        self._jsonl_handle: TextIO | None = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._failure: BaseException | None = None

    @property
    def state(self) -> RecordingState:
        with self._lock:
            return self._state

    @property
    def session(self) -> RecordingSessionInfo | None:
        with self._lock:
            return self._session

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames_written

    @property
    def frames_dropped(self) -> int:
        with self._lock:
            return self._frames_dropped

    def poll_failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def start(
        self,
        *,
        camera: CameraRole,
        width: int,
        height: int,
        nominal_fps: float,
        vision_backend: str,
    ) -> RecordingSessionInfo:
        with self._lock:
            if self._state is not RecordingState.IDLE:
                raise RuntimeError(f"recording cannot start from state {self._state.value}")

        started_at = self._wall_clock().astimezone()
        started_monotonic_ns = self._monotonic_clock_ns()
        paths = create_recording_session_paths(self._recording_directory, started_at)

        video_writer: VideoFrameWriter | None = None
        jsonl_handle: TextIO | None = None
        try:
            video_writer = self._video_writer_factory(
                paths.video_path,
                width=width,
                height=height,
                nominal_fps=nominal_fps,
            )
            jsonl_handle = paths.frames_path.open("x", encoding="utf-8", buffering=1)
            session = RecordingSessionInfo(
                paths=paths,
                camera=camera,
                width=width,
                height=height,
                nominal_fps=nominal_fps,
                vision_backend=vision_backend,
                started_at=started_at,
                started_monotonic_ns=started_monotonic_ns,
                codec=video_writer.codec,
                video_backend=video_writer.backend,
            )
            write_recording_metadata(
                session,
                status=RecordingSessionStatus.RECORDING,
                ended_at=None,
                duration_ns=None,
                frames_written=0,
                frames_dropped=0,
                failure=None,
            )
        except BaseException as exc:
            if jsonl_handle is not None:
                jsonl_handle.close()
            if video_writer is not None:
                video_writer.close()
            _write_failed_start_metadata(
                paths,
                camera=camera,
                width=width,
                height=height,
                vision_backend=vision_backend,
                started_at=started_at,
                failure=exc,
            )
            raise

        packet_queue: queue.Queue[RecordingFramePacket] = queue.Queue(
            maxsize=self._queue_capacity
        )
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._writer_loop,
            name="navmin-recording-writer",
            daemon=True,
        )

        with self._lock:
            self._state = RecordingState.RECORDING
            self._session = session
            self._queue = packet_queue
            self._stop_event = stop_event
            self._thread = thread
            self._video_writer = video_writer
            self._jsonl_handle = jsonl_handle
            self._frames_written = 0
            self._frames_dropped = 0
            self._failure = None

        thread.start()
        return session

    def submit(self, packet: RecordingFramePacket) -> bool:
        """Try to queue one packet without blocking the caller."""
        with self._lock:
            if self._state is not RecordingState.RECORDING:
                return False
            session = self._session
            packet_queue = self._queue
        assert session is not None
        assert packet_queue is not None

        self._validate_packet_for_session(packet, session)
        try:
            packet_queue.put_nowait(packet)
        except queue.Full:
            with self._lock:
                self._frames_dropped += 1
            return False
        return True

    def stop(self) -> RecordingSessionStatus:
        with self._lock:
            if self._state is RecordingState.IDLE:
                return RecordingSessionStatus.COMPLETED
            if self._state is RecordingState.FAILED:
                stop_event = self._stop_event
                thread = self._thread
            elif self._state is RecordingState.RECORDING:
                self._state = RecordingState.STOPPING
                stop_event = self._stop_event
                thread = self._thread
            else:
                raise RuntimeError(f"recording cannot stop from state {self._state.value}")

        assert stop_event is not None
        assert thread is not None
        stop_event.set()
        thread.join(timeout=self._stop_timeout_s)
        if thread.is_alive():
            failure = TimeoutError("recording writer did not stop before timeout")
            with self._lock:
                self._failure = self._failure or failure
                self._state = RecordingState.FAILED
            LOGGER.error("Recording writer stop timed out")
            return RecordingSessionStatus.FAILED

        with self._lock:
            failed = self._failure is not None
        self._finalize_after_thread()
        return RecordingSessionStatus.FAILED if failed else RecordingSessionStatus.COMPLETED

    def _writer_loop(self) -> None:
        try:
            while True:
                with self._lock:
                    packet_queue = self._queue
                    stop_event = self._stop_event
                assert packet_queue is not None
                assert stop_event is not None
                if stop_event.is_set() and packet_queue.empty():
                    break
                try:
                    packet = packet_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._write_packet(packet)
                finally:
                    packet_queue.task_done()
        except BaseException as exc:
            LOGGER.exception("Recording writer failed")
            with self._lock:
                self._failure = exc
                self._state = RecordingState.FAILED
            self._drop_remaining_packets()
        finally:
            self._close_open_outputs()

    def _write_packet(self, packet: RecordingFramePacket) -> None:
        with self._lock:
            session = self._session
            writer = self._video_writer
            handle = self._jsonl_handle
            record_index = self._frames_written
        assert session is not None
        assert writer is not None
        assert handle is not None

        line = serialize_recording_frame(packet, record_index=record_index)
        writer.write(packet.frame.image)
        handle.write(line)
        handle.write("\n")
        if (record_index + 1) % _JSONL_FLUSH_INTERVAL_FRAMES == 0:
            handle.flush()
        with self._lock:
            self._frames_written += 1

    def _drop_remaining_packets(self) -> None:
        with self._lock:
            packet_queue = self._queue
        if packet_queue is None:
            return
        dropped = 0
        while True:
            try:
                packet_queue.get_nowait()
            except queue.Empty:
                break
            else:
                packet_queue.task_done()
                dropped += 1
        if dropped:
            with self._lock:
                self._frames_dropped += dropped

    def _close_open_outputs(self) -> None:
        with self._lock:
            writer = self._video_writer
            handle = self._jsonl_handle
            self._video_writer = None
            self._jsonl_handle = None
        if handle is not None:
            handle.flush()
            handle.close()
        if writer is not None:
            writer.close()

    def _finalize_after_thread(self) -> None:
        with self._lock:
            session = self._session
            frames_written = self._frames_written
            frames_dropped = self._frames_dropped
            failure = self._failure
            self._state = RecordingState.IDLE
            self._session = None
            self._queue = None
            self._stop_event = None
            self._thread = None
            self._video_writer = None
            self._jsonl_handle = None
        if session is None:
            return

        ended_at = self._wall_clock().astimezone()
        duration_ns = max(0, self._monotonic_clock_ns() - session.started_monotonic_ns)
        write_recording_metadata(
            session,
            status=(
                RecordingSessionStatus.FAILED
                if failure is not None
                else RecordingSessionStatus.COMPLETED
            ),
            ended_at=ended_at,
            duration_ns=duration_ns,
            frames_written=frames_written,
            frames_dropped=frames_dropped,
            failure=failure,
        )

    @staticmethod
    def _validate_packet_for_session(
        packet: RecordingFramePacket, session: RecordingSessionInfo
    ) -> None:
        frame = packet.frame
        if frame.camera is not session.camera:
            raise ValueError(
                f"recording camera mismatch: expected {session.camera.value}, "
                f"got {frame.camera.value}"
            )
        expected_shape = (session.height, session.width, 3)
        if frame.image.shape != expected_shape:
            raise ValueError(
                f"recording frame shape mismatch: expected {expected_shape}, "
                f"got {frame.image.shape}"
            )
        if frame.image.dtype != np.uint8:
            raise ValueError("recording frame must be uint8 BGR")


def create_recording_session_paths(
    parent_directory: Path, started_at: datetime
) -> RecordingSessionPaths:
    """Atomically reserve a timestamp-named session directory."""
    parent = Path(parent_directory)
    parent.mkdir(parents=True, exist_ok=True)
    stem = started_at.strftime("%Y%m%d-%H%M%S")
    for suffix in range(1000):
        session_id = stem if suffix == 0 else f"{stem}-{suffix:02d}"
        session_dir = parent / session_id
        try:
            session_dir.mkdir()
        except FileExistsError:
            continue
        return RecordingSessionPaths(
            session_id=session_id,
            session_dir=session_dir,
            video_path=session_dir / _VIDEO_FILE_NAME,
            frames_path=session_dir / _FRAMES_FILE_NAME,
            metadata_path=session_dir / _METADATA_FILE_NAME,
        )
    raise RuntimeError(f"unable to allocate unique recording session for {stem}")


def serialize_recording_frame(packet: RecordingFramePacket, *, record_index: int) -> str:
    if record_index < 0:
        raise ValueError("record_index must be non-negative")
    frame = packet.frame
    payload = {
        "schema_version": RECORDING_SCHEMA_VERSION,
        "record_index": record_index,
        "frame": {
            "camera": frame.camera.value.replace("_", "-"),
            "generation": frame.generation,
            "frame_id": frame.frame_id,
            "width": int(frame.image.shape[1]),
            "height": int(frame.image.shape[0]),
            "capture_timestamp_ns": packet.capture_timestamp_ns,
            "receive_timestamp_ns": frame.receive_timestamp_ns,
            "relative_time_ns": packet.relative_time_ns,
        },
        "tracking": _tracking_payload(packet.tracking),
        "selection": _selection_payload(packet.selection),
        "overlay": _overlay_payload(packet.overlay),
        "control": _control_payload(packet.control),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_recording_metadata(
    session: RecordingSessionInfo,
    *,
    status: RecordingSessionStatus,
    ended_at: datetime | None,
    duration_ns: int | None,
    frames_written: int,
    frames_dropped: int,
    failure: BaseException | None,
) -> None:
    payload = {
        "schema_version": RECORDING_SCHEMA_VERSION,
        "session_id": session.paths.session_id,
        "camera": session.camera.value.replace("_", "-"),
        "working_frame": {
            "width": session.width,
            "height": session.height,
            "pixel_format": "BGR8",
        },
        "vision_backend": session.vision_backend,
        "started_at": session.started_at.isoformat(),
        "ended_at": ended_at.isoformat() if ended_at is not None else None,
        "duration_ns": duration_ns,
        "video": {
            "file": _VIDEO_FILE_NAME,
            "container": "matroska",
            "codec": session.codec,
            "backend": session.video_backend,
            "nominal_fps": session.nominal_fps,
            "timing": "jsonl-relative-time-authoritative",
        },
        "frames_written": frames_written,
        "frames_dropped": frames_dropped,
        "status": status.value,
        "failure_type": type(failure).__name__ if failure is not None else None,
        "failure_message": str(failure) if failure is not None else None,
    }
    _atomic_write_json(session.paths.metadata_path, payload)


def _write_failed_start_metadata(
    paths: RecordingSessionPaths,
    *,
    camera: CameraRole,
    width: int,
    height: int,
    vision_backend: str,
    started_at: datetime,
    failure: BaseException,
) -> None:
    payload = {
        "schema_version": RECORDING_SCHEMA_VERSION,
        "session_id": paths.session_id,
        "camera": camera.value.replace("_", "-"),
        "working_frame": {
            "width": width,
            "height": height,
            "pixel_format": "BGR8",
        },
        "vision_backend": vision_backend,
        "started_at": started_at.isoformat(),
        "ended_at": started_at.isoformat(),
        "duration_ns": 0,
        "video": {
            "file": _VIDEO_FILE_NAME,
            "container": "matroska",
            "codec": None,
            "backend": None,
            "nominal_fps": None,
            "timing": "jsonl-relative-time-authoritative",
        },
        "frames_written": 0,
        "frames_dropped": 0,
        "status": RecordingSessionStatus.FAILED.value,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure),
    }
    _atomic_write_json(paths.metadata_path, payload)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _point_payload(point: RecordingPoint) -> list[float]:
    return [point.x, point.y]


def _bbox_payload(bbox: BBox) -> list[int]:
    return [bbox.x, bbox.y, bbox.width, bbox.height]


def _tracking_payload(state: RecordingTrackingState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "phase": state.phase,
        "status": state.status,
        "target_ref": state.target_ref,
        "diagnostic_reason": state.diagnostic_reason,
        "candidate_count": state.candidate_count,
    }


def _selection_payload(selection: RecordingSelection | None) -> dict[str, object] | None:
    if selection is None:
        return None
    return {"point": _point_payload(selection.point)}


def _overlay_payload(overlay: RecordingOverlaySnapshot) -> dict[str, object]:
    selected = overlay.selected_target
    candidate = overlay.best_candidate
    region = overlay.search_region
    prediction = overlay.control_prediction
    return {
        "selected_target": (
            {
                "bbox": _bbox_payload(selected.bbox),
                "center": _point_payload(selected.center),
            }
            if selected is not None
            else None
        ),
        "best_candidate": (
            {"bbox": _bbox_payload(candidate.bbox)}
            if candidate is not None
            else None
        ),
        "search_region": _search_region_payload(region),
        "control_prediction": (
            {
                "from": _point_payload(prediction.from_point),
                "to": _point_payload(prediction.to_point),
            }
            if prediction is not None
            else None
        ),
    }


def _search_region_payload(region: RecordingSearchRegion | None) -> dict[str, object] | None:
    if region is None:
        return None
    if region.kind is SearchRegionKind.FULL_FRAME:
        return {"kind": SearchRegionKind.FULL_FRAME.value}
    assert region.center is not None
    assert region.radius_px is not None
    return {
        "kind": SearchRegionKind.CIRCLE.value,
        "center": _point_payload(region.center),
        "radius_px": region.radius_px,
    }


def _control_payload(state: RecordingControlState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "prediction_enabled": state.prediction_enabled,
        "rate_valid_x": state.rate_valid_x,
        "rate_valid_y": state.rate_valid_y,
    }


__all__ = [
    "DEFAULT_RECORDING_QUEUE_CAPACITY",
    "RECORDING_SCHEMA_VERSION",
    "OpenCvMkvWriter",
    "RecordingBestCandidate",
    "RecordingControlPrediction",
    "RecordingControlState",
    "RecordingFramePacket",
    "RecordingOverlaySnapshot",
    "RecordingPoint",
    "RecordingRuntime",
    "RecordingSearchRegion",
    "RecordingSelectedTarget",
    "RecordingSelection",
    "RecordingSessionInfo",
    "RecordingSessionPaths",
    "RecordingSessionStatus",
    "RecordingState",
    "RecordingTrackingState",
    "SearchRegionKind",
    "create_recording_session_paths",
    "serialize_recording_frame",
]
