"""ROI geometry and operator selection data for VT-2."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoiRect:
    """Integer ROI in original-frame coordinates, using x/y/width/height."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + (self.width - 1) / 2.0, self.y + (self.height - 1) / 2.0


@dataclass(frozen=True)
class OperatorSelection:
    """Approximate operator indication that will be consumed by ACQUIRE in VT-3."""

    frame_index: int
    click_x: float
    click_y: float
    roi_radius_px: int
    roi: RoiRect


def clamp_point(x: float, y: float, frame_width: int, frame_height: int) -> tuple[float, float]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    return (
        min(max(float(x), 0.0), frame_width - 1.0),
        min(max(float(y), 0.0), frame_height - 1.0),
    )


def make_roi(click_xy: tuple[float, float], radius_px: int, frame_width: int, frame_height: int) -> RoiRect:
    """Build a square ROI centered on the click and clip it to frame boundaries.

    The nominal side is ``2*radius_px + 1``. Close to a frame edge the ROI is
    clipped rather than shifted; this preserves the click prior exactly.
    """
    if radius_px < 1:
        raise ValueError("ROI radius must be >= 1 px.")
    cx, cy = clamp_point(click_xy[0], click_xy[1], frame_width, frame_height)

    x1 = max(0, int(round(cx)) - int(radius_px))
    y1 = max(0, int(round(cy)) - int(radius_px))
    x2 = min(frame_width - 1, int(round(cx)) + int(radius_px))
    y2 = min(frame_height - 1, int(round(cy)) + int(radius_px))
    return RoiRect(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)


def make_operator_selection(
    frame_index: int,
    click_xy: tuple[float, float],
    radius_px: int,
    frame_width: int,
    frame_height: int,
) -> OperatorSelection:
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0.")
    cx, cy = clamp_point(click_xy[0], click_xy[1], frame_width, frame_height)
    roi = make_roi((cx, cy), radius_px, frame_width, frame_height)
    return OperatorSelection(
        frame_index=int(frame_index),
        click_x=cx,
        click_y=cy,
        roi_radius_px=int(radius_px),
        roi=roi,
    )
