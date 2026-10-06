from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QSize, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from navmin.contracts import (
    BBox,
    CameraRole,
    ClearTargetSelection,
    FramePacket,
    TargetAcquired,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TargetTrackingEndReason,
    TrackedObject,
    VisionResult,
)
from navmin.diagnostics.vision_playback import (
    PlaybackBackend,
    PlaybackOptions,
    VisionPlaybackSession,
)
from navmin.diagnostics.vision_playback_ui import (
    PlaybackDisplayGeometry,
    PlaybackVideoWidget,
    VisionPlaybackWindow,
)
from navmin.vision.processor import VisionProcessorOutput


class _Clock:
    def __init__(self, step_ns: int = 1_000_000) -> None:
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.value += self.step_ns
        return self.value


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, fps: float = 25.0) -> None:
        self._frames = [frame.copy() for frame in frames]
        self._fps = fps
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def get(self, prop_id: int) -> float:
        assert prop_id == cv2.CAP_PROP_FPS
        return self._fps

    def release(self) -> None:
        self.released = True


class _UiProcessor:
    def __init__(
        self,
        *,
        objects_by_frame: dict[int, tuple[TrackedObject, ...]] | None = None,
        end_target_on_frame: int | None = None,
    ) -> None:
        self.commands: list[object] = []
        self.frames: list[FramePacket] = []
        self.objects_by_frame = objects_by_frame or {}
        self.end_target_on_frame = end_target_on_frame
        self.active_target: TargetRef | None = None

    def reset(self) -> None:
        self.active_target = None

    def apply_target_control(self, command) -> tuple:
        self.commands.append(command)
        if isinstance(command, ClearTargetSelection):
            self.active_target = None
            return ()
        assert isinstance(command, TargetSelectionRequest)
        target = TargetRef(command.camera, command.generation, 42)
        self.active_target = target
        frame = self.frames[command.frame_id]
        return (
            TargetAcquired(
                selection_id=command.selection_id,
                target=target,
                request_frame_id=command.frame_id,
                decision_frame_id=command.frame_id,
                decision_frame_timestamp_ns=frame.receive_timestamp_ns,
            ),
        )

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        assert processing_enabled
        self.frames.append(frame)
        events = ()
        if frame.frame_id == self.end_target_on_frame and self.active_target is not None:
            events = (
                TargetTrackingEnded(
                    target=self.active_target,
                    reason=TargetTrackingEndReason.LOST_UNRECOVERED,
                    decision_frame_id=frame.frame_id,
                    decision_frame_timestamp_ns=frame.receive_timestamp_ns,
                ),
            )
            self.active_target = None
        return VisionProcessorOutput(
            result=VisionResult(
                frame=frame,
                tracked_objects=self.objects_by_frame.get(frame.frame_id, ()),
                processing_time_ns=0,
            ),
            target_events=events,
        )


@pytest.fixture(scope="session")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _frame(value: int) -> np.ndarray:
    image = np.full((50, 100, 3), value, dtype=np.uint8)
    image[:, :, 1] = 100 + value
    return image


def _tracked(track_id: int, x: int = 10, y: int = 10) -> TrackedObject:
    return TrackedObject(track_id, BBox(x, y, 20, 15), 0.0, 0.0, 1)


def _session(
    tmp_path: Path,
    processor: _UiProcessor,
    *,
    frame_count: int = 2,
) -> tuple[VisionPlaybackSession, _FakeCapture]:
    video_path = tmp_path / "ui.avi"
    video_path.touch()
    capture = _FakeCapture([_frame(index) for index in range(frame_count)])
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=video_path,
            backend=PlaybackBackend.LEGACY14,
            camera_role=CameraRole.OVERVIEW,
            fps_override=25.0,
        ),
        capture_factory=lambda _: capture,
        processor_factory=lambda _: processor,
        clock_ns=_Clock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )
    return session, capture


def test_display_geometry_covers_scaling_letterbox_edges_bbox_and_roundtrip() -> None:
    source = QSize(100, 50)

    one_to_one = PlaybackDisplayGeometry.for_sizes(QSize(100, 50), source)
    assert (
        one_to_one.display_rect.left(),
        one_to_one.display_rect.top(),
        one_to_one.display_rect.width(),
        one_to_one.display_rect.height(),
    ) == (0.0, 0.0, 100.0, 50.0)

    scaled = PlaybackDisplayGeometry.for_sizes(QSize(200, 100), source)
    assert (
        scaled.display_rect.left(),
        scaled.display_rect.top(),
        scaled.display_rect.width(),
        scaled.display_rect.height(),
    ) == (0.0, 0.0, 200.0, 100.0)

    horizontal_bars = PlaybackDisplayGeometry.for_sizes(QSize(200, 200), source)
    assert (
        horizontal_bars.display_rect.left(),
        horizontal_bars.display_rect.top(),
        horizontal_bars.display_rect.width(),
        horizontal_bars.display_rect.height(),
    ) == (0.0, 50.0, 200.0, 100.0)
    assert horizontal_bars.widget_to_source(QPointF(100, 49)) is None
    assert horizontal_bars.widget_to_source(QPointF(0, 50)) == (0.0, 0.0)
    assert horizontal_bars.widget_to_source(QPointF(199, 149)) == pytest.approx(
        (99.0, 49.0)
    )

    square_source = QSize(100, 100)
    vertical_bars = PlaybackDisplayGeometry.for_sizes(QSize(300, 100), square_source)
    assert (
        vertical_bars.display_rect.left(),
        vertical_bars.display_rect.top(),
        vertical_bars.display_rect.width(),
        vertical_bars.display_rect.height(),
    ) == (100.0, 0.0, 100.0, 100.0)
    assert vertical_bars.widget_to_source(QPointF(99, 50)) is None
    assert vertical_bars.widget_to_source(QPointF(100, 0)) == (0.0, 0.0)

    mapped_bbox = horizontal_bars.source_bbox_to_widget(BBox(10, 5, 20, 10))
    assert mapped_bbox is not None
    assert (
        mapped_bbox.left(),
        mapped_bbox.top(),
        mapped_bbox.width(),
        mapped_bbox.height(),
    ) == pytest.approx((20.0, 60.0, 40.0, 20.0))

    source_point = (49.5, 24.5)
    widget_point = horizontal_bars.source_point_to_widget(source_point)
    assert widget_point is not None
    round_trip = horizontal_bars.widget_to_source(widget_point)
    assert round_trip is not None
    assert round_trip == pytest.approx(source_point, abs=0.5)


def test_left_click_uses_displayed_frame_right_click_clears_and_black_bar_is_ignored(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    processor = _UiProcessor(objects_by_frame={0: (_tracked(3),)})
    session, _capture = _session(tmp_path, processor)
    window = VisionPlaybackWindow(session, auto_start=False)
    window.video_view.setFixedSize(400, 400)
    window.show()
    qt_application.processEvents()
    window.advance_once()
    qt_application.processEvents()

    geometry = window.video_view.geometry_model()
    assert geometry is not None
    assert geometry.display_rect.top() > 0.0
    inside = geometry.source_point_to_widget((20.0, 15.0))
    assert inside is not None

    QTest.mouseClick(
        window.video_view,
        Qt.MouseButton.LeftButton,
        pos=inside.toPoint(),
    )
    QTest.mouseClick(
        window.video_view,
        Qt.MouseButton.LeftButton,
        pos=inside.toPoint(),
    )
    qt_application.processEvents()

    assert len(processor.commands) == 2
    first, second = processor.commands
    assert isinstance(first, TargetSelectionRequest)
    assert isinstance(second, TargetSelectionRequest)
    assert (first.selection_id, second.selection_id) == (1, 2)
    assert (first.frame_id, second.frame_id) == (0, 0)
    assert not any(isinstance(command, ClearTargetSelection) for command in processor.commands)

    QTest.mouseClick(
        window.video_view,
        Qt.MouseButton.LeftButton,
        pos=QPointF(200, 20).toPoint(),
    )
    qt_application.processEvents()
    assert len(processor.commands) == 2

    QTest.mouseClick(
        window.video_view,
        Qt.MouseButton.RightButton,
        pos=QPointF(200, 20).toPoint(),
    )
    qt_application.processEvents()
    assert len(processor.commands) == 3
    assert isinstance(processor.commands[-1], ClearTargetSelection)

    window.close()
    qt_application.processEvents()


def test_overlay_model_uses_only_current_result_and_exposes_track_ids(
    qt_application: QApplication,
) -> None:
    selected: list[tuple[float, float]] = []
    clears: list[bool] = []
    widget = PlaybackVideoWidget(
        on_select=lambda x, y: selected.append((x, y)),
        on_clear=lambda: clears.append(True),
    )
    widget.resize(400, 300)
    widget.show()

    image = _frame(0)
    image.flags.writeable = False
    packet = FramePacket(CameraRole.OVERVIEW, 1, 0, None, 1, image)
    discovery = VisionResult(
        packet,
        (_tracked(3), _tracked(7, 45, 20)),
        0,
    )
    widget.set_result(discovery)
    qt_application.processEvents()
    assert widget.displayed_track_ids == (3, 7)

    confirmed_target = VisionResult(
        packet,
        (_tracked(42, 30, 15),),
        0,
    )
    widget.set_result(confirmed_target)
    qt_application.processEvents()
    assert widget.displayed_track_ids == (42,)

    no_current_bbox = VisionResult(packet, (), 0)
    widget.set_result(no_current_bbox)
    qt_application.processEvents()
    assert widget.displayed_track_ids == ()

    widget.close()


def test_lifecycle_status_tracks_acquired_then_tracking_ended(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    processor = _UiProcessor(
        objects_by_frame={0: (_tracked(42),), 1: ()},
        end_target_on_frame=1,
    )
    session, _capture = _session(tmp_path, processor)
    window = VisionPlaybackWindow(session, auto_start=False)
    window.video_view.setFixedSize(400, 300)
    window.show()
    qt_application.processEvents()

    window.advance_once()
    qt_application.processEvents()
    geometry = window.video_view.geometry_model()
    assert geometry is not None
    inside = geometry.source_point_to_widget((20.0, 15.0))
    assert inside is not None
    QTest.mouseClick(
        window.video_view,
        Qt.MouseButton.LeftButton,
        pos=inside.toPoint(),
    )
    qt_application.processEvents()

    assert window.current_target_track_id == 42
    assert window.last_lifecycle_text == "TargetAcquired: ID 42"

    window.advance_once()
    qt_application.processEvents()
    assert window.video_view.displayed_track_ids == ()
    assert window.current_target_track_id is None
    assert window.last_lifecycle_text == "Target ended: LOST_UNRECOVERED"

    window.close()
    qt_application.processEvents()


def test_eof_keeps_last_frame_visible_disables_controls_and_logs_finished_once(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    processor = _UiProcessor(objects_by_frame={0: (_tracked(5),)})
    session, capture = _session(tmp_path, processor, frame_count=1)
    window = VisionPlaybackWindow(session, auto_start=False)
    window.video_view.setFixedSize(400, 300)
    window.show()
    qt_application.processEvents()

    window.advance_once()
    assert window.displayed_frame_id == 0
    window.advance_once()
    qt_application.processEvents()

    assert window.playback_finished
    assert session.finished
    assert not session.closed
    assert window.displayed_frame_id == 0
    assert window.video_view.displayed_track_ids == (5,)
    assert not window.video_view.interaction_enabled
    assert window.isVisible()

    geometry = window.video_view.geometry_model()
    assert geometry is not None
    inside = geometry.source_point_to_widget((20.0, 15.0))
    assert inside is not None
    QTest.mouseClick(
        window.video_view,
        Qt.MouseButton.LeftButton,
        pos=inside.toPoint(),
    )
    qt_application.processEvents()
    assert processor.commands == []

    assert session.run_dir is not None
    events = [
        json.loads(line)
        for line in (session.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["type"] == "PLAYBACK_FINISHED" for event in events) == 1
    assert not any(event["type"] == "TargetTrackingEnded" for event in events)

    window.close()
    qt_application.processEvents()
    assert session.closed
    assert capture.released
