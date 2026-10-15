"""Aspect-ratio-correct QWidget renderer for accepted Vision results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QLineF, QPointF, QRectF, QSize, Qt
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
    DistanceResult,
    DistanceSource,
    TargetRef,
    VisionResult,
)
from navmin.core import CameraSessionGate

STALE_MESSAGE = "НЕТ НОВЫХ КАДРОВ"

_RETICLE_GAP_PX = 4.0
_RETICLE_LENGTH_PX = 10.0
_RETICLE_PEN_WIDTH_PX = 3

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


def map_source_to_widget(
    point: tuple[float, float],
    display_rect: QRectF,
    source_size: QSize,
) -> QPointF | None:
    """Map working-frame pixel coordinates into the rendered image rectangle."""
    if display_rect.isEmpty() or source_size.isEmpty():
        return None
    x_px, y_px = point
    if not (0.0 <= x_px < source_size.width() and 0.0 <= y_px < source_size.height()):
        return None
    return QPointF(
        display_rect.left()
        + (x_px + 0.5) * display_rect.width() / source_size.width(),
        display_rect.top()
        + (y_px + 0.5) * display_rect.height() / source_size.height(),
    )


def reticle_segments(center: QPointF) -> tuple[QLineF, QLineF, QLineF, QLineF]:
    """Return the four fixed-size reticle segments around a clear center."""
    x = center.x()
    y = center.y()
    gap = _RETICLE_GAP_PX
    length = _RETICLE_LENGTH_PX
    return (
        QLineF(x - gap - length, y, x - gap, y),
        QLineF(x + gap, y, x + gap + length, y),
        QLineF(x, y - gap - length, x, y - gap),
        QLineF(x, y + gap, x, y + gap + length),
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
        self._target_distance: DistanceResult | None = None
        self._aim_point: tuple[float, float] | None = None
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
    def aim_point(self) -> tuple[float, float] | None:
        return self._aim_point

    @property
    def target_distance(self) -> DistanceResult | None:
        return self._target_distance

    @property
    def distance_annotation_text(self) -> str | None:
        result = self.displayed_result
        target = self._selected_target
        distance = self._target_distance
        if (
            self._preview
            or self._camera is not CameraRole.STEREO_LEFT
            or result is None
            or target is None
            or distance is None
            or result.frame.camera is not self._camera
            or distance.target != target
            or target.camera is not result.frame.camera
            or target.generation != result.frame.generation
            or not any(
                tracked.track_id == target.track_id for tracked in result.tracked_objects
            )
        ):
            return None
        return self._format_distance(distance)

    @property
    def stale_overlay_text(self) -> str | None:
        return STALE_MESSAGE if self._prepared is not None and self._stale else None

    def set_camera(self, camera: CameraRole) -> None:
        if camera is self._camera:
            return
        self._camera = camera
        self._target_distance = None
        self._aim_point = None
        self.update()

    def set_prepared_frame(self, prepared: PreparedVisionFrame | None) -> None:
        if prepared is self._prepared:
            return
        self._prepared = prepared
        if prepared is None:
            self._aim_point = None
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
        if self._target_distance is not None and self._target_distance.target != target:
            self._target_distance = None
        self.update()

    def set_target_distance(self, distance: DistanceResult | None) -> None:
        if distance is not None and not isinstance(distance, DistanceResult):
            raise TypeError("distance must be DistanceResult or None")
        if distance == self._target_distance:
            return
        self._target_distance = distance
        self.update()

    def set_aim_point(self, point: tuple[float, float] | None) -> None:
        if point == self._aim_point:
            return
        self._aim_point = point
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
            self._draw_reticle(painter, display_rect, prepared.result)
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
            bbox_rect = QRectF(
                display_rect.left() + bbox.x * scale_x,
                display_rect.top() + bbox.y * scale_y,
                bbox.width * scale_x,
                bbox.height * scale_y,
            )
            painter.drawRect(bbox_rect)
            if selected:
                self._draw_distance_annotation(painter, display_rect, bbox_rect)

    def _draw_distance_annotation(
        self,
        painter: QPainter,
        display_rect: QRectF,
        bbox_rect: QRectF,
    ) -> None:
        text = self.distance_annotation_text
        if text is None:
            return
        painter.setFont(QFont("Sans Serif", 11, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        label_rect = QRectF(metrics.boundingRect(text).adjusted(-6, -3, 6, 3))
        x = min(
            max(display_rect.left(), bbox_rect.left()),
            max(display_rect.left(), display_rect.right() - label_rect.width()),
        )
        above_y = bbox_rect.top() - label_rect.height() - 4
        below_y = bbox_rect.bottom() + 4
        y = above_y if above_y >= display_rect.top() else below_y
        y = min(
            max(display_rect.top(), y),
            max(display_rect.top(), display_rect.bottom() - label_rect.height()),
        )
        label_rect.moveTopLeft(QPointF(x, y))
        painter.fillRect(label_rect, QColor(0, 0, 0, 175))
        painter.setPen(QColor(255, 235, 90))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

    @staticmethod
    def _format_distance(distance: DistanceResult) -> str:
        if distance.source is DistanceSource.MANUAL:
            return f"{distance.distance_m:.0f} м"
        return f"{distance.distance_m:.1f} м"

    def _draw_reticle(
        self,
        painter: QPainter,
        display_rect: QRectF,
        result: VisionResult,
    ) -> None:
        if self._preview or self._aim_point is None:
            return
        height, width = result.frame.image.shape[:2]
        center = map_source_to_widget(
            self._aim_point,
            display_rect,
            QSize(width, height),
        )
        if center is None:
            return
        pen = QPen(QColor(255, 0, 0), _RETICLE_PEN_WIDTH_PX)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        for segment in reticle_segments(center):
            painter.drawLine(segment)

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
    "map_source_to_widget",
    "map_widget_to_source",
    "prepare_vision_frame",
    "rendered_image_rect",
    "reticle_segments",
]
