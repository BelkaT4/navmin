"""VT-10 ground-truth format and loader.

CSV columns:
    frame,target_id,x,y,visible,is_target

Multiple objects may be present on the same frame. ``is_target=1`` marks the
operator-selected target. Other visible objects are used to distinguish
wrong-ID from generic localization error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class GroundTruthPoint:
    frame_index: int
    target_id: str
    x: float
    y: float
    visible: bool = True
    is_target: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class GroundTruthSequence:
    points: tuple[GroundTruthPoint, ...]

    def frames(self) -> tuple[int, ...]:
        return tuple(sorted({p.frame_index for p in self.points}))

    def objects_at(self, frame_index: int, *, visible_only: bool = True) -> tuple[GroundTruthPoint, ...]:
        out = [p for p in self.points if p.frame_index == frame_index]
        if visible_only:
            out = [p for p in out if p.visible]
        return tuple(out)

    def target_at(self, frame_index: int) -> GroundTruthPoint | None:
        items = [p for p in self.points if p.frame_index == frame_index and p.is_target]
        if not items:
            return None
        # Exactly one selected target per frame is preferred. Loader validates
        # duplicates to keep metrics deterministic.
        return items[0]

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(sorted({p.target_id for p in self.points if p.is_target}))


_REQUIRED = ("frame", "target_id", "x", "y", "visible", "is_target")
_TRUE = {"1", "true", "yes", "y", "да"}
_FALSE = {"0", "false", "no", "n", "нет", ""}


def _parse_bool(value: str, name: str) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"Invalid boolean in {name}: {value!r}")


def load_ground_truth_csv(path: str | Path) -> GroundTruthSequence:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [name for name in _REQUIRED if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Ground truth CSV missing columns: {', '.join(missing)}")
        points: list[GroundTruthPoint] = []
        seen_target_frames: set[int] = set()
        for line_no, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame"])
                x = float(row["x"])
                y = float(row["y"])
            except Exception as exc:
                raise ValueError(f"Invalid numeric value at CSV line {line_no}") from exc
            if frame < 0:
                raise ValueError(f"Negative frame at CSV line {line_no}")
            tid = str(row["target_id"]).strip()
            if not tid:
                raise ValueError(f"Empty target_id at CSV line {line_no}")
            visible = _parse_bool(row["visible"], "visible")
            is_target = _parse_bool(row["is_target"], "is_target")
            if is_target:
                if frame in seen_target_frames:
                    raise ValueError(f"More than one is_target row for frame {frame}")
                seen_target_frames.add(frame)
            points.append(GroundTruthPoint(frame, tid, x, y, visible, is_target))
    if not points:
        raise ValueError("Ground truth CSV is empty")
    points.sort(key=lambda q: (q.frame_index, not q.is_target, q.target_id))
    return GroundTruthSequence(tuple(points))


def write_ground_truth_template(path: str | Path, *, frame_count: int = 0) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_REQUIRED)
        if frame_count > 0:
            w.writerow([0, "TARGET", "", "", 1, 1])
            if frame_count > 1:
                w.writerow([frame_count - 1, "TARGET", "", "", 1, 1])
    return p
