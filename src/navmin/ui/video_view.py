"""Aspect-ratio-correct QWidget renderer for accepted Vision results."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QLineF, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QBrush,
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
    BBox,
    CameraModel,
    CameraRay,
    CameraRole,
    CameraState,
    CameraStatus,
    DistanceResult,
    DistanceSource,
    TargetRef,
    VisionResult,
)
from navmin.core import CameraSessionGate, TrackingControlVisualization
from navmin.vision.processor import VisionTrackingDiagnostics

STALE_MESSAGE = "НЕТ НОВЫХ КАДРОВ"

_RETICLE_GAP_PX = 4.0
_RETICLE_LENGTH_PX = 10.0
_RETICLE_PEN_WIDTH_PX = 3
_TRACKING_OVERLAY_PEN_WIDTH_PX = 2
_SEARCH_REGION_COLOR = QColor(255, 80, 180)
_BEST_CANDIDATE_COLOR = QColor(255, 145, 40)
_CONTROL_PREDICTION_COLOR = QColor(40, 185, 255)
_CONTROL_PREDICTION_ENDPOINT_RADIUS_PX = 4.0

_CAMERA_NAMES = {
    CameraRole.OVERVIEW: "Overview",
    CameraRole.STEREO_LEFT: "Stereo Left",
}


@dataclass(frozen=True)
class PreparedVisionFrame:
    """One UI-thread conversion of an accepted VisionResult."""

    result: VisionResult
    pixmap: QPixmap


@dataclass(frozen=True)
class ControlPredictionOverlay:
    """Display-only prediction geometry derived from one authoritative snapshot."""

    observed_px: tuple[float, float]
    predicted_px: tuple[float, float]


def project_control_prediction(
    visualization: TrackingControlVisualization,
    camera_model: CameraModel,
) -> ControlPredictionOverlay | None:
    """Project raw angular forecast into pixels without feeding control state."""
    if not (visualization.rate_valid_x or visualization.rate_valid_y):
        return None

    delta_x_deg = (
        visualization.raw_prediction_offset_x_deg
        if visualization.rate_valid_x
        else 0.0
    )
    delta_y_deg = (
        visualization.raw_prediction_offset_y_deg
        if visualization.rate_valid_y
        else 0.0
    )
    if delta_x_deg == 0.0 and delta_y_deg == 0.0:
        return None

    try:
        observed_ray = camera_model.pixel_to_ray(*visualization.observed_center_px)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(
        math.isfinite(value)
        for value in (observed_ray.x, observed_ray.y, observed_ray.z)
    ) or (
        observed_ray.x == 0.0
        and observed_ray.y == 0.0
        and observed_ray.z == 0.0
    ):
        return None
    horizontal = math.atan2(observed_ray.x, observed_ray.z)
    vertical = -math.atan2(
        observed_ray.y,
        math.hypot(observed_ray.x, observed_ray.z),
    )
    horizontal += math.radians(delta_x_deg)
    vertical += math.radians(delta_y_deg)
    cos_vertical = math.cos(vertical)
    predicted_ray = CameraRay(
        cos_vertical * math.sin(horizontal),
        -math.sin(vertical),
        cos_vertical * math.cos(horizontal),
    )
    try:
        predicted_px = camera_model.ray_to_pixel(predicted_ray)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(value) for value in predicted_px):
        return None
    predicted_x = (
        float(predicted_px[0])
        if visualization.rate_valid_x
        else visualization.observed_center_px[0]
    )
    predicted_y = (
        float(predicted_px[1])
        if visualization.rate_valid_y
        else visualization.observed_center_px[1]
    )
    return ControlPredictionOverlay(
        observed_px=visualization.observed_center_px,
        predicted_px=(predicted_x, predicted_y),
    )


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


def _map_source_to_widget_unbounded(
    point: tuple[float, float],
    display_rect: QRectF,
    source_size: QSize,
) -> QPointF | None:
    if display_rect.isEmpty() or source_size.isEmpty():
        return None
    x_px, y_px = point
    return QPointF(
        display_rect.left()
        + (x_px + 0.5) * display_rect.width() / source_size.width(),
        display_rect.top()
        + (y_px + 0.5) * display_rect.height() / source_size.height(),
    )


def map_source_radius_to_widget(
    radius_px: float,
    display_rect: QRectF,
    source_size: QSize,
) -> tuple[float, float] | None:
    """Map a source-space radius into the aspect-preserving display rectangle."""
    if radius_px < 0 or display_rect.isEmpty() or source_size.isEmpty():
        return None
    return (
        radius_px * display_rect.width() / source_size.width(),
        radius_px * display_rect.height() / source_size.height(),
    )


def map_bbox_to_widget(
    bbox: BBox,
    display_rect: QRectF,
    source_size: QSize,
) -> QRectF | None:
    """Map an integer source bbox into the rendered image rectangle."""
    if display_rect.isEmpty() or source_size.isEmpty():
        return None
    return QRectF(
        display_rect.left() + bbox.x * display_rect.width() / source_size.width(),
        display_rect.top() + bbox.y * display_rect.height() / source_size.height(),
        bbox.width * display_rect.width() / source_size.width(),
        bbox.height * display_rect.height() / source_size.height(),
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
        self._tracking_diagnostics: VisionTrackingDiagnostics | None = None
        self._tracking_control_visualization: TrackingControlVisualization | None = None
        self._control_prediction_overlay: ControlPredictionOverlay | None = None
        self._show_search_region = False
        self._show_best_candidate = False
        self._show_control_prediction = False
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
    def tracking_diagnostics(self) -> VisionTrackingDiagnostics | None:
        return self._tracking_diagnostics

    @property
    def tracking_control_visualization(self) -> TrackingControlVisualization | None:
        return self._tracking_control_visualization

    @property
    def control_prediction_overlay(self) -> ControlPredictionOverlay | None:
        return self._control_prediction_overlay

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
        self._tracking_diagnostics = None
        self._tracking_control_visualization = None
        self._control_prediction_overlay = None
        self.update()

    def set_prepared_frame(self, prepared: PreparedVisionFrame | None) -> None:
        if prepared is self._prepared:
            return
        self._prepared = prepared
        if prepared is None:
            self._aim_point = None
            self._tracking_diagnostics = None
            self._tracking_control_visualization = None
            self._control_prediction_overlay = None
        else:
            if not self._diagnostics_match_result(
                self._tracking_diagnostics,
                prepared.result,
            ):
                self._tracking_diagnostics = None
            if not self._control_visualization_matches_result(
                self._tracking_control_visualization,
                prepared.result,
            ):
                self._tracking_control_visualization = None
                self._control_prediction_overlay = None
        self.update()

    def set_tracking_diagnostics(
        self,
        diagnostics: VisionTrackingDiagnostics | None,
    ) -> None:
        result = self.displayed_result
        if diagnostics is not None and (
            self._preview
            or result is None
            or not self._diagnostics_match_result(diagnostics, result)
        ):
            diagnostics = None
        if diagnostics == self._tracking_diagnostics:
            return
        self._tracking_diagnostics = diagnostics
        self.update()

    def set_tracking_control_visualization(
        self,
        visualization: TrackingControlVisualization | None,
        camera_model: CameraModel | None,
    ) -> None:
        result = self.displayed_result
        if visualization is not None and (
            self._preview
            or result is None
            or camera_model is None
            or not self._control_visualization_matches_result(visualization, result)
            or visualization.target != self._selected_target
        ):
            visualization = None
        overlay = (
            None
            if visualization is None or camera_model is None
            else project_control_prediction(visualization, camera_model)
        )
        if (
            visualization == self._tracking_control_visualization
            and overlay == self._control_prediction_overlay
        ):
            return
        self._tracking_control_visualization = visualization
        self._control_prediction_overlay = overlay
        self.update()

    def set_tracking_visualization(
        self,
        *,
        show_search_region: bool,
        show_best_candidate: bool,
        show_control_prediction: bool = False,
    ) -> None:
        changed = (
            self._show_search_region != show_search_region
            or self._show_best_candidate != show_best_candidate
            or self._show_control_prediction != show_control_prediction
        )
        self._show_search_region = show_search_region
        self._show_best_candidate = show_best_candidate
        self._show_control_prediction = show_control_prediction
        if changed:
            self.update()

    @staticmethod
    def _diagnostics_match_result(
        diagnostics: VisionTrackingDiagnostics | None,
        result: VisionResult,
    ) -> bool:
        return bool(
            diagnostics is not None
            and diagnostics.camera is result.frame.camera
            and diagnostics.generation == result.frame.generation
            and diagnostics.frame_id == result.frame.frame_id
        )

    @staticmethod
    def _control_visualization_matches_result(
        visualization: TrackingControlVisualization | None,
        result: VisionResult,
    ) -> bool:
        return bool(
            visualization is not None
            and visualization.camera is result.frame.camera
            and visualization.generation == result.frame.generation
            and visualization.frame_id == result.frame.frame_id
        )

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
        if (
            self._tracking_control_visualization is not None
            and self._tracking_control_visualization.target != target
        ):
            self._tracking_control_visualization = None
            self._control_prediction_overlay = None
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
            painter.drawPixmap(
                display_rect,
                prepared.pixmap,
                QRectF(prepared.pixmap.rect()),
            )
            self._draw_tracking_overlays(painter, display_rect, prepared.result)
            self._draw_control_prediction_overlay(
                painter,
                display_rect,
                prepared.result,
            )
            self._draw_objects(painter, display_rect, prepared.result)
            self._draw_reticle(painter, display_rect, prepared.result)
            if self._stale:
                painter.fillRect(display_rect, QColor(0, 0, 0, 110))
                self._draw_center_message(painter, display_rect, STALE_MESSAGE)
        else:
            self._draw_center_message(painter, QRectF(self.rect()), "ОЖИДАНИЕ КАДРА")
        self._draw_camera_and_status(painter)

    def _draw_tracking_overlays(
        self,
        painter: QPainter,
        display_rect: QRectF,
        result: VisionResult,
    ) -> None:
        if self._preview:
            return
        diagnostics = self._tracking_diagnostics
        if not self._diagnostics_match_result(diagnostics, result):
            return
        assert diagnostics is not None
        height, width = result.frame.image.shape[:2]
        source_size = QSize(width, height)

        if self._show_search_region:
            search_pen = QPen(
                _SEARCH_REGION_COLOR,
                _TRACKING_OVERLAY_PEN_WIDTH_PX,
                Qt.PenStyle.DashLine,
            )
            painter.setPen(search_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if diagnostics.full_frame_search:
                inset = _TRACKING_OVERLAY_PEN_WIDTH_PX / 2.0
                painter.drawRect(display_rect.adjusted(inset, inset, -inset, -inset))
            else:
                center = _map_source_to_widget_unbounded(
                    diagnostics.predicted_xy,
                    display_rect,
                    source_size,
                )
                radius = map_source_radius_to_widget(
                    diagnostics.search_radius_px,
                    display_rect,
                    source_size,
                )
                if center is not None and radius is not None:
                    painter.drawEllipse(center, radius[0], radius[1])

        if self._show_best_candidate and diagnostics.best_candidate_bbox is not None:
            if self._best_candidate_is_accepted(result, diagnostics.best_candidate_bbox):
                return
            candidate_rect = map_bbox_to_widget(
                diagnostics.best_candidate_bbox,
                display_rect,
                source_size,
            )
            if candidate_rect is not None:
                painter.setPen(
                    QPen(
                        _BEST_CANDIDATE_COLOR,
                        _TRACKING_OVERLAY_PEN_WIDTH_PX,
                        Qt.PenStyle.DashLine,
                    )
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(candidate_rect)

    def _best_candidate_is_accepted(
        self,
        result: VisionResult,
        candidate_bbox: BBox,
    ) -> bool:
        target = self._selected_target
        if (
            target is None
            or target.camera is not result.frame.camera
            or target.generation != result.frame.generation
        ):
            return False
        for tracked in result.tracked_objects:
            if tracked.track_id == target.track_id:
                return tracked.bbox == candidate_bbox
        return False

    def _draw_control_prediction_overlay(
        self,
        painter: QPainter,
        display_rect: QRectF,
        result: VisionResult,
    ) -> None:
        if self._preview or not self._show_control_prediction:
            return
        visualization = self._tracking_control_visualization
        overlay = self._control_prediction_overlay
        if (
            overlay is None
            or visualization is None
            or visualization.target != self._selected_target
            or not self._control_visualization_matches_result(visualization, result)
        ):
            return
        height, width = result.frame.image.shape[:2]
        source_size = QSize(width, height)
        start = _map_source_to_widget_unbounded(
            overlay.observed_px,
            display_rect,
            source_size,
        )
        end = _map_source_to_widget_unbounded(
            overlay.predicted_px,
            display_rect,
            source_size,
        )
        if start is None or end is None or start == end:
            return
        painter.setPen(
            QPen(
                _CONTROL_PREDICTION_COLOR,
                _TRACKING_OVERLAY_PEN_WIDTH_PX,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.setBrush(QBrush(_CONTROL_PREDICTION_COLOR))
        painter.drawLine(QLineF(start, end))
        painter.drawEllipse(
            end,
            _CONTROL_PREDICTION_ENDPOINT_RADIUS_PX,
            _CONTROL_PREDICTION_ENDPOINT_RADIUS_PX,
        )

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
