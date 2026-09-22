"""Aspect-ratio-correct QWidget renderer for accepted Vision results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor,
    QEnterEvent,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PyQt6.QtWidgets import QPushButton, QWidget

from navmin.contracts import (
    CameraRole,
    CameraState,
    CameraStatus,
    TargetRef,
    VisionResult,
)
from navmin.core import CameraSessionGate

STALE_MESSAGE = "НЕТ НОВЫХ КАДРОВ"

_CAMERA_NAMES = {
    CameraRole.OVERVIEW: "Overview",
    CameraRole.STEREO_LEFT: "Stereo Left",
}


@dataclass(frozen=True)
class PreparedVisionFrame:
    """One UI-thread conversion of an accepted VisionResult."""

    result: VisionResult
    pixmap: QPixmap


def prepare_vision_frame(result: VisionResult) -> PreparedVisionFrame:
    image = result.frame.image
    height, width = image.shape[:2]
    qimage = QImage(
        image.data,
        width,
        height,
        int(image.strides[0]),
        QImage.Format.Format_BGR888,
    ).copy()
    return PreparedVisionFrame(result=result, pixmap=QPixmap.fromImage(qimage))


def rendered_image_rect(widget_size: QSize, source_size: QSize) -> QRectF:
    """Return the centered KeepAspectRatio destination rectangle."""
    if (
        widget_size.width() <= 0
        or widget_size.height() <= 0
        or source_size.width() <= 0
        or source_size.height() <= 0
    ):
        return QRectF()
    scale = min(
        widget_size.width() / source_size.width(),
        widget_size.height() / source_size.height(),
    )
    width = source_size.width() * scale
    height = source_size.height() * scale
    return QRectF(
        (widget_size.width() - width) / 2.0,
        (widget_size.height() - height) / 2.0,
        width,
        height,
    )


def map_widget_to_source(
    point: QPointF,
    display_rect: QRectF,
    source_size: QSize,
) -> tuple[float, float] | None:
    """Map a point through the rendered rectangle; reject letterbox areas."""
    if display_rect.isEmpty() or source_size.isEmpty():
        return None
    if not (
        display_rect.left() <= point.x() < display_rect.left() + display_rect.width()
        and display_rect.top() <= point.y() < display_rect.top() + display_rect.height()
    ):
        return None
    x_px = (point.x() - display_rect.left()) * source_size.width() / display_rect.width()
    y_px = (point.y() - display_rect.top()) * source_size.height() / display_rect.height()
    return (
        min(max(x_px, 0.0), source_size.width() - 1.0),
        min(max(y_px, 0.0), source_size.height() - 1.0),
    )


def is_camera_stale(
    status: CameraStatus | None,
    now_ns: int,
    *,
    threshold_ns: int,
) -> bool:
    if status is None or status.last_receive_timestamp_ns is None:
        return False
    return now_ns - status.last_receive_timestamp_ns >= threshold_ns


class VideoView(QWidget):
    """Paint one camera role and retain the exact displayed VisionResult."""

    def __init__(
        self,
        *,
        camera: CameraRole,
        preview: bool,
        stale_timeout_ns: int,
        on_main_click: Callable[[VisionResult, float, float], None] | None = None,
        on_swap: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self._preview = preview
        self._stale_timeout_ns = stale_timeout_ns
        self._on_main_click = on_main_click
        self._on_swap = on_swap
        self._prepared: PreparedVisionFrame | None = None
        self._status: CameraStatus | None = None
        self._selected_target: TargetRef | None = None
        self._stale = False
        self.setMinimumSize(1, 1)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self._preview_hovered = False
        self.swap_button: QPushButton | None = None
        if preview:
            button = QPushButton("⇄", self)
            button.setFixedSize(30, 26)
            button.setToolTip("Поменять main и preview")
            if on_swap is not None:
                button.clicked.connect(on_swap)
            self.swap_button = button
            self._apply_swap_button_style()

    @property
    def camera(self) -> CameraRole:
        return self._camera

    @property
    def displayed_result(self) -> VisionResult | None:
        return None if self._prepared is None else self._prepared.result

    @property
    def camera_status(self) -> CameraStatus | None:
        return self._status

    @property
    def is_stale(self) -> bool:
        return self._stale

    @property
    def stale_overlay_text(self) -> str | None:
        return STALE_MESSAGE if self._prepared is not None and self._stale else None

    def set_camera(self, camera: CameraRole) -> None:
        if camera is self._camera:
            return
        self._camera = camera
        self.update()

    def set_prepared_frame(self, prepared: PreparedVisionFrame | None) -> None:
        if prepared is self._prepared:
            return
        self._prepared = prepared
        self.update()

    def set_camera_status(self, status: CameraStatus | None) -> None:
        if status == self._status:
            return
        self._status = status
        self.update()

    def set_selected_target(self, target: TargetRef | None) -> None:
        if target == self._selected_target:
            return
        self._selected_target = target
        self.update()

    def refresh_freshness(self, now_ns: int) -> None:
        stale = self._prepared is not None and is_camera_stale(
            self._status,
            now_ns,
            threshold_ns=self._stale_timeout_ns,
        )
        if stale == self._stale:
            return
        self._stale = stale
        self.update()

    def rendered_rect(self) -> QRectF:
        if self._prepared is None:
            return QRectF()
        return rendered_image_rect(self.size(), self._prepared.pixmap.size())

    def interaction_allowed(
        self,
        gate: CameraSessionGate,
        now_ns: int,
    ) -> bool:
        result = self.displayed_result
        status = self._status
        return bool(
            result is not None
            and gate.accepts(result.frame.camera, result.frame.generation)
            and status is not None
            and status.camera is result.frame.camera
            and status.generation == result.frame.generation
            and status.state is CameraState.ONLINE
            and not is_camera_stale(
                status,
                now_ns,
                threshold_ns=self._stale_timeout_ns,
            )
        )

    def enterEvent(self, event: QEnterEvent) -> None:
        if self._preview:
            self._preview_hovered = True
            self._apply_swap_button_style()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self._preview:
            self._preview_hovered = False
            self._apply_swap_button_style()
        super().leaveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        button = self.swap_button
        if button is not None:
            button.move(max(0, self.width() - button.width() - 8), 8)
        super().resizeEvent(event)

    def _apply_swap_button_style(self) -> None:
        button = self.swap_button
        if button is None:
            return
        if self._preview_hovered:
            button.setStyleSheet(
                "QPushButton { background: rgba(45, 45, 45, 185); color: rgba(255, 255, 255, 225);"
                " border: 1px solid rgba(255, 255, 255, 125); border-radius: 4px; font-size: 15px; }"
                "QPushButton:hover { background: rgba(65, 65, 65, 220); color: white;"
                " border-color: rgba(255, 255, 255, 175); }"
                "QPushButton:pressed { background: rgba(20, 20, 20, 225); }"
            )
            return
        button.setStyleSheet(
            "QPushButton { background: rgba(35, 35, 35, 105); color: rgba(245, 245, 245, 135);"
            " border: 1px solid rgba(245, 245, 245, 55); border-radius: 4px; font-size: 15px; }"
            "QPushButton:hover { background: rgba(60, 60, 60, 205); color: white;"
            " border-color: rgba(255, 255, 255, 160); }"
            "QPushButton:pressed { background: rgba(20, 20, 20, 220); }"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton:
            event.accept()
            return
        if self._preview:
            if self._on_swap is not None:
                self._on_swap()
            event.accept()
            return
        result = self.displayed_result
        if result is None or self._on_main_click is None:
            event.accept()
            return
        height, width = result.frame.image.shape[:2]
        mapped = map_widget_to_source(
            event.position(),
            self.rendered_rect(),
            QSize(width, height),
        )
        if mapped is not None:
            self._on_main_click(result, *mapped)
        event.accept()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        prepared = self._prepared
        if prepared is not None:
            display_rect = self.rendered_rect()
            painter.drawPixmap(display_rect, prepared.pixmap, QRectF(prepared.pixmap.rect()))
            self._draw_objects(painter, display_rect, prepared.result)
            if self._stale:
                painter.fillRect(display_rect, QColor(0, 0, 0, 110))
                self._draw_center_message(painter, display_rect, STALE_MESSAGE)
        else:
            self._draw_center_message(painter, QRectF(self.rect()), "ОЖИДАНИЕ КАДРА")
        self._draw_camera_and_status(painter)

    def _draw_objects(
        self,
        painter: QPainter,
        display_rect: QRectF,
        result: VisionResult,
    ) -> None:
        height, width = result.frame.image.shape[:2]
        scale_x = display_rect.width() / width
        scale_y = display_rect.height() / height
        target = self._selected_target
        for tracked in result.tracked_objects:
            selected = bool(
                target is not None
                and target.camera is result.frame.camera
                and target.generation == result.frame.generation
                and target.track_id == tracked.track_id
            )
            painter.setPen(
                QPen(QColor(255, 210, 30) if selected else QColor(50, 230, 90), 4 if selected else 2)
            )
            bbox = tracked.bbox
            painter.drawRect(
                QRectF(
                    display_rect.left() + bbox.x * scale_x,
                    display_rect.top() + bbox.y * scale_y,
                    bbox.width * scale_x,
                    bbox.height * scale_y,
                )
            )

    def _draw_camera_and_status(self, painter: QPainter) -> None:
        name = _CAMERA_NAMES[self._camera]
        status = self._status
        status_text = "NO STATUS" if status is None else status.state.value.upper()
        if status is not None and status.state is CameraState.ERROR:
            detail = status.message or status.error_code
            if detail:
                status_text = f"{status_text}: {detail[:60]}"
        text = f"{name}  •  {status_text}"
        painter.setFont(QFont("Sans Serif", 11, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        box = metrics.boundingRect(text).adjusted(-8, -5, 8, 5)
        box.moveTopLeft(self.rect().topLeft() + QPointF(10, 10).toPoint())
        painter.fillRect(box, QColor(0, 0, 0, 165))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    @staticmethod
    def _draw_center_message(
        painter: QPainter,
        area: QRectF,
        text: str,
    ) -> None:
        painter.setFont(QFont("Sans Serif", 18, QFont.Weight.Bold))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, text)


__all__ = [
    "STALE_MESSAGE",
    "PreparedVisionFrame",
    "VideoView",
    "is_camera_stale",
    "map_widget_to_source",
    "prepare_vision_frame",
    "rendered_image_rect",
]
