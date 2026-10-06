"""VT-10 annotated-video exporter using OpenCV only."""
from __future__ import annotations

from pathlib import Path
import cv2

from navmin.vision.processors.vt11.evaluation.logger import session_rows


def _draw_cross(frame, xy, color, size=9, thickness=2):
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.line(frame, (x-size, y), (x+size, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y-size), (x, y+size), color, thickness, cv2.LINE_AA)


def export_annotated_video(
    *, frame_reader, frame_count: int, width: int, height: int, fps: float,
    session, path: str | Path, codec: str = "mp4v"
) -> Path:
    if len(codec) != 4:
        raise ValueError("codec must be a four-character code")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*codec), float(fps if fps > 0 else 25.0), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {p}")
    by_frame = {}
    for fi, mode, item in session_rows(session):
        # Prefer TRACK if both modes ever share the same frame.
        if fi not in by_frame or mode == "TRACK":
            by_frame[fi] = (mode, item)
    try:
        for fi in range(frame_count):
            _idx, frame = frame_reader(fi)
            out = frame.copy()
            entry = by_frame.get(fi)
            if entry is not None:
                mode, item = entry
                # prediction: magenta
                _draw_cross(out, item.predicted_xy, (255, 0, 255), size=6, thickness=1)
                cv2.circle(out, (int(round(item.predicted_xy[0])), int(round(item.predicted_xy[1]))), int(item.search_radius_px), (255, 0, 255), 1, cv2.LINE_AA)
                if item.selected is not None:
                    c = (0, 255, 0) if mode == "TRACK" else (0, 215, 255)
                    _draw_cross(out, item.selected.center, c, size=10, thickness=2)
                    b = item.selected.bbox
                    cv2.rectangle(out, (b.x, b.y), (b.x+b.width, b.y+b.height), c, 1, cv2.LINE_AA)
                cv2.putText(out, f"{mode} {item.status} conf={item.confidence:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(out, f"frame={fi} time={getattr(item, 'processing_time_ms', 0.0):.2f} ms", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            writer.write(out)
    finally:
        writer.release()
    return p
