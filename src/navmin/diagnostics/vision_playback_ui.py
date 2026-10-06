"""Interactive PyQt6 frontend for the standalone Vision playback diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from navmin.contracts import (
    BBox,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetLifecycleEvent,
    TargetTrackingEnded,
    VisionResult,
)
from navmin.ui.video_view import (
    PreparedVisionFrame,
    map_source_to_widget,
    map_widget_to_source,
    prepare_vision_frame,
    rendered_image_rect,
)

from .vision_playback import (
    PlaybackDiagnosticError,
    PlaybackTerminalState,
    VisionPlaybackSession,
)


@dataclass(frozen=True)
class PlaybackDisplayGeometry:
    """One aspect-ratio geometry model shared by clicks and bbox overlays."""

    source_size: QSize
    display_rect: QRectF

    @classmethod
    def for_sizes(
        cls,
        widget_size: QSize,
        source_size: QSize,
    ) -> PlaybackDisplayGeometry:
        return cls(
            source_size=source_size,
            display_rect=rendered_image_rect(widget_size, source_size),
        )

    def widget_to_source(self, point: QPointF) -> tuple[float, float] | None:
        return map_widget_to_source(point, self.display_rect, self.source_size)

    def source_point_to_widget(self, point: tuple[float, float]) -> QPointF | None:
        return map_source_to_widget(point, self.display_rect, self.source_size)

    def source_bbox_to_widget(self, bbox: BBox) -> QRectF | None:
        if self.display_rect.isEmpty() or self.source_size.isEmpty():
            return None
        if (
            bbox.x < 0
            or bbox.y < 0
            or bbox.width <= 0
            or bbox.height <= 0
            or bbox.x + bbox.width > self.source_size.width()
            or bbox.y + bbox.height > self.source_size.height()
        ):
            return None
        scale_x = self.display_rect.width() / self.source_size.width()
        scale_y = self.display_rect.height() / self.source_size.height()
        return QRectF(
            self.display_rect.left() + bbox.x * scale_x,
            self.display_rect.top() + bbox.y * scale_y,
            bbox.width * scale_x,
            bbox.height * scale_y,
        )


class PlaybackVideoWidget(QWidget):
    """Render one exact processed VisionResult and accept diagnostic controls."""

    def __init__(
        self,
        *,
        on_select,
        on_clear,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_select = on_select
        self._on_clear = on_clear
        self._prepared: PreparedVisionFrame | None = None
        self._interaction_enabled = True
        self._selected_track_id: int | None = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    @property
    def displayed_result(self) -> VisionResult | None:
        return None if self._prepared is None else self._prepared.result

    @property
    def displayed_track_ids(self) -> tuple[int, ...]:
        result = self.displayed_result
        if result is None:
            return ()
        return tuple(item.track_id for item in result.tracked_objects)

    @property
    def interaction_enabled(self) -> bool:
        return self._interaction_enabled

    def set_result(self, result: VisionResult) -> None:
        self._prepared = prepare_vision_frame(result)
        self.update()

    def set_interaction_enabled(self, enabled: bool) -> None:
        if enabled == self._interaction_enabled:
            return
        self._interaction_enabled = enabled
        self.update()

    def set_selected_track_id(self, track_id: int | None) -> None:
        if track_id == self._selected_track_id:
            return
        self._selected_track_id = track_id
        self.update()

    def geometry_model(self) -> PlaybackDisplayGeometry | None:
        result = self.displayed_result
        if result is None:
            return None
        height, width = result.frame.image.shape[:2]
        return PlaybackDisplayGeometry.for_sizes(
            self.size(),
            QSize(width, height),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._interaction_enabled or self.displayed_result is None:
            event.accept()
            return
        if event.button() is Qt.MouseButton.RightButton:
            self._on_clear()
            event.accept()
            return
        if event.button() is not Qt.MouseButton.LeftButton:
            event.accept()
            return
        geometry = self.geometry_model()
        if geometry is None:
            event.accept()
            return
        mapped = geometry.widget_to_source(event.position())
        if mapped is not None:
            self._on_select(*mapped)
        event.accept()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        prepared = self._prepared
        if prepared is None:
            self._draw_center_message(painter, "ОЖИДАНИЕ КАДРА")
            return

        geometry = self.geometry_model()
        if geometry is None:
            return
        painter.drawPixmap(
            geometry.display_rect,
            prepared.pixmap,
            QRectF(prepared.pixmap.rect()),
        )
        self._draw_objects(painter, geometry, prepared.result)
        if not self._interaction_enabled:
            painter.fillRect(geometry.display_rect, QColor(0, 0, 0, 55))

    def _draw_objects(
        self,
        painter: QPainter,
        geometry: PlaybackDisplayGeometry,
        result: VisionResult,
    ) -> None:
        painter.setFont(QFont("Sans Serif", 10, QFont.Weight.Bold))
        for tracked in result.tracked_objects:
            box = geometry.source_bbox_to_widget(tracked.bbox)
            if box is None:
                continue
            selected = tracked.track_id == self._selected_track_id
            painter.setPen(
                QPen(
                    QColor(255, 210, 30) if selected else QColor(50, 230, 90),
                    3 if selected else 2,
                )
            )
            painter.drawRect(box)

            label = f"ID {tracked.track_id}"
            metrics = painter.fontMetrics()
            label_rect = metrics.boundingRect(label).adjusted(-4, -2, 4, 2)
            label_rect.moveBottomLeft(box.topLeft().toPoint())
            painter.fillRect(label_rect, QColor(0, 0, 0, 170))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_center_message(self, painter: QPainter, text: str) -> None:
        painter.setFont(QFont("Sans Serif", 18, QFont.Weight.Bold))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


class VisionPlaybackWindow(QMainWindow):
    """Single-window, single-owner interactive playback UI."""

    def __init__(
        self,
        session: VisionPlaybackSession,
        *,
        auto_start: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.playback_finished = False
        self.had_error = False
        self.last_lifecycle_text = "—"
        self.current_target_track_id: int | None = None
        self._runtime_status = "Воспроизведение"
        self._auto_start = auto_start

        self.setWindowTitle("NavMin — Vision Playback Diagnostic")
        self.resize(1200, 850)

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.video_view = PlaybackVideoWidget(
            on_select=self._select_from_display,
            on_clear=self._clear_target,
            parent=root,
        )
        layout.addWidget(self.video_view, 1)

        status_widget = QWidget(root)
        status_layout = QGridLayout(status_widget)
        status_layout.setContentsMargins(4, 2, 4, 2)
        status_layout.setHorizontalSpacing(18)
        status_layout.setVerticalSpacing(3)
        self._status_values: dict[str, QLabel] = {}
        status_fields = (
            ("backend", "Backend"),
            ("frame", "Frame ID"),
            ("fps", "Playback FPS"),
            ("camera", "Camera role"),
            ("processor", "Processor state"),
            ("target", "Current target ID"),
            ("lifecycle", "Last lifecycle event"),
            ("status", "Status"),
        )
        for row, (key, title) in enumerate(status_fields):
            status_layout.addWidget(QLabel(f"{title}:"), row, 0)
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            status_layout.addWidget(value, row, 1)
            self._status_values[key] = value
        status_layout.setColumnStretch(1, 1)
        layout.addWidget(status_widget, 0)

        instructions = QLabel(
            "ЛКМ внутри изображения — SELECT • ПКМ в области видео — CLEAR • "
            "pause/seek/rewind/frame-step отсутствуют"
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions, 0)
        self.setCentralWidget(root)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.advance_once)

        self.session.start()
        self._refresh_status()
        if auto_start:
            self._timer.start(0)

    @property
    def displayed_frame_id(self) -> int | None:
        result = self.video_view.displayed_result
        return None if result is None else result.frame.frame_id

    def processor_state_text(self) -> str:
        state = getattr(self.session.processor, "state", None)
        if isinstance(state, Enum):
            return str(state.value).upper()
        if state is None:
            return "—"
        return str(state)

    def advance_once(self) -> None:
        if self.playback_finished or self.had_error or self.session.closed:
            return
        try:
            step = self.session.process_next()
        except (RuntimeError, ValueError, OSError, cv2.error) as exc:
            self._set_runtime_error(exc)
            return

        if step.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED:
            self.playback_finished = True
            self._runtime_status = "Воспроизведение окончено"
            self.video_view.set_interaction_enabled(False)
            self._refresh_status()
            return

        output = step.output
        if output is None or step.frame_started_ns is None:
            self._set_runtime_error(PlaybackDiagnosticError("invalid playback step"))
            return

        self.video_view.set_result(output.result)
        self._apply_lifecycle_events(output.target_events)
        self._refresh_status()

        if self._auto_start:
            delay_ms = self.session.remaining_delay_ms_after(step.frame_started_ns)
            self._timer.start(delay_ms)

    def _select_from_display(self, x_px: float, y_px: float) -> None:
        if self.playback_finished or self.had_error:
            return
        self.current_target_track_id = None
        self.video_view.set_selected_track_id(None)
        try:
            events = self.session.select_target(x_px, y_px)
        except (RuntimeError, ValueError, OSError, cv2.error) as exc:
            self._set_runtime_error(exc)
            return
        self._apply_lifecycle_events(events)
        self._refresh_status()

    def _clear_target(self) -> None:
        if self.playback_finished or self.had_error:
            return
        self.current_target_track_id = None
        self.video_view.set_selected_track_id(None)
        try:
            events = self.session.clear_target()
        except (RuntimeError, ValueError, OSError, cv2.error) as exc:
            self._set_runtime_error(exc)
            return
        self._apply_lifecycle_events(events)
        self._refresh_status()

    def _apply_lifecycle_events(
        self,
        events: tuple[TargetLifecycleEvent, ...],
    ) -> None:
        for event in events:
            if isinstance(event, TargetAcquired):
                self.current_target_track_id = event.target.track_id
                self.last_lifecycle_text = f"TargetAcquired: ID {event.target.track_id}"
            elif isinstance(event, TargetAcquisitionFailed):
                self.current_target_track_id = None
                self.last_lifecycle_text = (
                    f"Acquisition failed: {event.reason.value.upper()}"
                )
            elif isinstance(event, TargetTrackingEnded):
                if self.current_target_track_id == event.target.track_id:
                    self.current_target_track_id = None
                self.last_lifecycle_text = (
                    f"Target ended: {event.reason.value.upper()}"
                )
        self.video_view.set_selected_track_id(self.current_target_track_id)

    def _set_runtime_error(self, exc: Exception) -> None:
        self.had_error = True
        self._timer.stop()
        self._runtime_status = f"Ошибка: {exc}"
        self.video_view.set_interaction_enabled(False)
        self.session.close()
        self._refresh_status()

    def _refresh_status(self) -> None:
        options = self.session.options
        backend = options.backend.value
        if options.profile is not None:
            backend = f"{backend} ({options.profile})"
        self._status_values["backend"].setText(backend)
        self._status_values["frame"].setText(
            "—" if self.displayed_frame_id is None else str(self.displayed_frame_id)
        )
        self._status_values["fps"].setText(f"{self.session.probe.effective_fps:g}")
        self._status_values["camera"].setText(options.camera_role.value)
        self._status_values["processor"].setText(self.processor_state_text())
        self._status_values["target"].setText(
            "—"
            if self.current_target_track_id is None
            else str(self.current_target_track_id)
        )
        self._status_values["lifecycle"].setText(self.last_lifecycle_text)
        self._status_values["status"].setText(self._runtime_status)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._timer.stop()
        if not self.session.closed:
            self.session.close(
                log_closed=not self.session.finished and not self.had_error
            )
        super().closeEvent(event)


def run_interactive_session(session: VisionPlaybackSession) -> int:
    """Run one already-probed playback session in a single PyQt6 window."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    try:
        window = VisionPlaybackWindow(session)
    except Exception:
        session.close()
        raise
    window.show()
    exit_code = application.exec()

    run_dir = session.run_dir
    terminal = (
        PlaybackTerminalState.PLAYBACK_FINISHED.value
        if session.finished
        else "PLAYBACK_CLOSED"
    )
    print(
        f"{terminal} "
        f"frames={session.frames_processed} "
        f"backend={session.options.backend.value} "
        f"camera_role={session.options.camera_role.value} "
        f"fps={session.probe.effective_fps:g} "
        f"run_dir={run_dir}"
    )
    if window.had_error:
        return 1
    return int(exit_code)


__all__ = [
    "PlaybackDisplayGeometry",
    "PlaybackVideoWidget",
    "VisionPlaybackWindow",
    "run_interactive_session",
]
