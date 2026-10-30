from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from navmin.contracts import BBox, CameraRole, FramePacket
from navmin.recording import (
    OpenCvMkvWriter,
    RecordingBestCandidate,
    RecordingControlPrediction,
    RecordingControlState,
    RecordingFramePacket,
    RecordingOverlaySnapshot,
    RecordingPoint,
    RecordingRuntime,
    RecordingSearchRegion,
    RecordingSelectedTarget,
    RecordingSelection,
    RecordingSessionStatus,
    RecordingState,
    RecordingTrackingState,
    SearchRegionKind,
    create_recording_session_paths,
    serialize_recording_frame,
)


def _frame(
    frame_id: int,
    *,
    camera: CameraRole = CameraRole.STEREO_LEFT,
    generation: int = 4,
    width: int = 64,
    height: int = 48,
) -> FramePacket:
    image = np.full((height, width, 3), frame_id % 255, dtype=np.uint8)
    image.flags.writeable = False
    return FramePacket(
        camera=camera,
        generation=generation,
        frame_id=frame_id,
        capture_id=frame_id,
        receive_timestamp_ns=1_000_000_000 + frame_id * 50_000_000,
        image=image,
    )


class _FakeVideoWriter:
    instances: ClassVar[list[_FakeVideoWriter]] = []

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        nominal_fps: float,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.nominal_fps = nominal_fps
        self.codec = "FAKE"
        self.backend = "test"
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.path.touch()
        self.__class__.instances.append(self)

    def write(self, image: np.ndarray) -> None:
        self.frames.append(image.copy())

    def close(self) -> None:
        self.closed = True


class _BlockingVideoWriter(_FakeVideoWriter):
    release = threading.Event()
    entered = threading.Event()

    def write(self, image: np.ndarray) -> None:
        self.entered.set()
        self.release.wait(timeout=2.0)
        super().write(image)


class _FailingVideoWriter(_FakeVideoWriter):
    def write(self, image: np.ndarray) -> None:
        raise OSError("disk writer failed")


@pytest.fixture(autouse=True)
def _reset_fake_writers() -> None:
    _FakeVideoWriter.instances.clear()
    _BlockingVideoWriter.instances.clear()
    _BlockingVideoWriter.release.clear()
    _BlockingVideoWriter.entered.clear()
    _FailingVideoWriter.instances.clear()


def test_session_directory_uses_timestamp_and_collision_suffix(tmp_path) -> None:
    now = datetime(2026, 10, 27, 15, 5, 12, tzinfo=UTC)
    first = create_recording_session_paths(tmp_path / "recordings", now)
    second = create_recording_session_paths(tmp_path / "recordings", now)

    assert first.session_id == "20261027-150512"
    assert second.session_id == "20261027-150512-01"
    assert first.session_dir.is_dir()
    assert second.session_dir.is_dir()
    assert first.video_path.name == "video.mkv"
    assert first.frames_path.name == "frames.jsonl"
    assert first.metadata_path.name == "metadata.json"


def test_jsonl_serialization_uses_source_coordinates_and_nullable_fields() -> None:
    packet = RecordingFramePacket(
        frame=_frame(7),
        relative_time_ns=350_000_000,
        tracking=RecordingTrackingState(
            phase="TRACK",
            status="TRACKING",
            target_ref="stereo-left:4:22",
            diagnostic_reason="ACCEPTED_TRACK",
            candidate_count=3,
        ),
        selection=RecordingSelection(point=RecordingPoint(20.5, 18.0)),
        overlay=RecordingOverlaySnapshot(
            selected_target=RecordingSelectedTarget(
                bbox=BBox(10, 11, 20, 15),
                center=RecordingPoint(20.0, 18.5),
            ),
            best_candidate=RecordingBestCandidate(bbox=BBox(9, 10, 21, 16)),
            search_region=RecordingSearchRegion(
                kind=SearchRegionKind.CIRCLE,
                center=RecordingPoint(22.0, 19.0),
                radius_px=42.5,
            ),
            control_prediction=RecordingControlPrediction(
                from_point=RecordingPoint(20.0, 18.5),
                to_point=RecordingPoint(24.0, 18.0),
            ),
        ),
        control=RecordingControlState(
            prediction_enabled=False,
            rate_valid_x=True,
            rate_valid_y=False,
        ),
    )

    payload = json.loads(serialize_recording_frame(packet, record_index=5))

    assert payload["schema_version"] == 1
    assert payload["record_index"] == 5
    assert payload["frame"] == {
        "camera": "stereo-left",
        "generation": 4,
        "frame_id": 7,
        "width": 64,
        "height": 48,
        "capture_timestamp_ns": None,
        "receive_timestamp_ns": 1_350_000_000,
        "relative_time_ns": 350_000_000,
    }
    assert payload["overlay"]["selected_target"]["bbox"] == [10, 11, 20, 15]
    assert payload["overlay"]["search_region"] == {
        "kind": "circle",
        "center": [22.0, 19.0],
        "radius_px": 42.5,
    }
    assert payload["control"] == {
        "prediction_enabled": False,
        "rate_valid_x": True,
        "rate_valid_y": False,
    }

    empty = json.loads(
        serialize_recording_frame(
            RecordingFramePacket(frame=_frame(8), relative_time_ns=400_000_000),
            record_index=6,
        )
    )
    assert empty["tracking"] is None
    assert empty["selection"] is None
    assert empty["overlay"] == {
        "selected_target": None,
        "best_candidate": None,
        "search_region": None,
        "control_prediction": None,
    }
    assert empty["control"] is None


def test_search_region_rejects_inconsistent_geometry() -> None:
    with pytest.raises(ValueError, match="requires center"):
        RecordingSearchRegion(kind=SearchRegionKind.CIRCLE)
    with pytest.raises(ValueError, match="cannot carry"):
        RecordingSearchRegion(
            kind=SearchRegionKind.FULL_FRAME,
            center=RecordingPoint(1.0, 2.0),
        )


def test_runtime_writes_video_and_jsonl_as_one_ordered_sequence(tmp_path) -> None:
    wall_times = iter(
        [
            datetime(2026, 10, 27, 15, 5, 12, tzinfo=UTC),
            datetime(2026, 10, 27, 15, 5, 13, tzinfo=UTC),
        ]
    )
    monotonic_times = iter([1_000_000_000, 2_000_000_000])
    runtime = RecordingRuntime(
        tmp_path / "recordings",
        video_writer_factory=_FakeVideoWriter,
        wall_clock=lambda: next(wall_times),
        monotonic_clock_ns=lambda: next(monotonic_times),
    )

    session = runtime.start(
        camera=CameraRole.STEREO_LEFT,
        width=64,
        height=48,
        nominal_fps=20.0,
        vision_backend="VT11",
    )
    for index in range(3):
        assert runtime.submit(
            RecordingFramePacket(
                frame=_frame(index),
                relative_time_ns=index * 50_000_000,
            )
        )
    assert runtime.stop() is RecordingSessionStatus.COMPLETED
    assert runtime.state is RecordingState.IDLE

    writer = _FakeVideoWriter.instances[-1]
    assert writer.closed is True
    assert len(writer.frames) == 3
    rows = [
        json.loads(line)
        for line in session.paths.frames_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_index"] for row in rows] == [0, 1, 2]
    assert [row["frame"]["frame_id"] for row in rows] == [0, 1, 2]

    metadata = json.loads(session.paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["frames_written"] == 3
    assert metadata["frames_dropped"] == 0
    assert metadata["video"]["codec"] == "FAKE"
    assert metadata["video"]["timing"] == "jsonl-relative-time-authoritative"


def test_runtime_queue_is_bounded_and_drops_whole_packets(tmp_path) -> None:
    runtime = RecordingRuntime(
        tmp_path / "recordings",
        queue_capacity=1,
        video_writer_factory=_BlockingVideoWriter,
    )
    session = runtime.start(
        camera=CameraRole.STEREO_LEFT,
        width=64,
        height=48,
        nominal_fps=20.0,
        vision_backend="VT11",
    )

    assert runtime.submit(RecordingFramePacket(frame=_frame(0), relative_time_ns=0))
    assert _BlockingVideoWriter.entered.wait(timeout=1.0)
    assert runtime.submit(
        RecordingFramePacket(frame=_frame(1), relative_time_ns=50_000_000)
    )
    assert not runtime.submit(
        RecordingFramePacket(frame=_frame(2), relative_time_ns=100_000_000)
    )
    _BlockingVideoWriter.release.set()
    assert runtime.stop() is RecordingSessionStatus.COMPLETED

    rows = session.paths.frames_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert runtime.frames_written == 2
    assert runtime.frames_dropped == 1
    metadata = json.loads(session.paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["frames_written"] == 2
    assert metadata["frames_dropped"] == 1


def test_writer_failure_isolated_and_marks_session_failed(tmp_path) -> None:
    runtime = RecordingRuntime(
        tmp_path / "recordings",
        video_writer_factory=_FailingVideoWriter,
    )
    session = runtime.start(
        camera=CameraRole.STEREO_LEFT,
        width=64,
        height=48,
        nominal_fps=20.0,
        vision_backend="VT11",
    )
    assert runtime.submit(RecordingFramePacket(frame=_frame(0), relative_time_ns=0))

    for _ in range(100):
        if runtime.poll_failure() is not None:
            break
        threading.Event().wait(0.01)

    assert isinstance(runtime.poll_failure(), OSError)
    assert runtime.stop() is RecordingSessionStatus.FAILED
    metadata = json.loads(session.paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["failure_type"] == "OSError"
    assert "disk writer failed" in metadata["failure_message"]


def test_runtime_rejects_wrong_camera_and_shape_without_queueing(tmp_path) -> None:
    runtime = RecordingRuntime(
        tmp_path / "recordings",
        video_writer_factory=_FakeVideoWriter,
    )
    runtime.start(
        camera=CameraRole.STEREO_LEFT,
        width=64,
        height=48,
        nominal_fps=20.0,
        vision_backend="VT11",
    )

    with pytest.raises(ValueError, match="camera mismatch"):
        runtime.submit(
            RecordingFramePacket(
                frame=_frame(0, camera=CameraRole.OVERVIEW),
                relative_time_ns=0,
            )
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        runtime.submit(
            RecordingFramePacket(
                frame=_frame(0, width=32, height=24),
                relative_time_ns=0,
            )
        )
    assert runtime.stop() is RecordingSessionStatus.COMPLETED


def test_failed_start_persists_metadata_and_releases_writer(tmp_path) -> None:
    class _CannotOpenWriter:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("no mkv codec")

    runtime = RecordingRuntime(
        tmp_path / "recordings",
        video_writer_factory=_CannotOpenWriter,
        wall_clock=lambda: datetime(2026, 10, 27, 15, 5, 12, tzinfo=UTC),
    )
    with pytest.raises(RuntimeError, match="no mkv codec"):
        runtime.start(
            camera=CameraRole.STEREO_LEFT,
            width=64,
            height=48,
            nominal_fps=20.0,
            vision_backend="VT11",
        )

    session_dir = next((tmp_path / "recordings").iterdir())
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["failure_type"] == "RuntimeError"


def test_default_opencv_writer_produces_readable_mkv_when_codec_available(tmp_path) -> None:
    path = tmp_path / "video.mkv"
    try:
        writer = OpenCvMkvWriter(path, width=64, height=48, nominal_fps=20.0)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    for index in range(4):
        writer.write(_frame(index).image)
    codec = writer.codec
    writer.close()

    assert path.is_file()
    assert path.stat().st_size > 0
    capture = cv2.VideoCapture(str(path))
    try:
        count = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape == (48, 64, 3)
            count += 1
    finally:
        capture.release()
    assert count == 4
    assert codec in {"FFV1", "MJPG"}
