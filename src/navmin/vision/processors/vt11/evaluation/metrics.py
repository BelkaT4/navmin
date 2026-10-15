"""VT-10 objective metrics against multi-object ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import mean, median
from typing import Mapping

from navmin.vision.processors.vt11.evaluation.ground_truth import GroundTruthSequence


@dataclass(frozen=True)
class PredictionPoint:
    frame_index: int
    x: float
    y: float
    mode: str = "TRACK"
    accepted: bool = True

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class FrameEvaluation:
    frame_index: int
    outcome: str
    predicted_xy: tuple[float, float] | None
    target_xy: tuple[float, float] | None
    matched_id: str | None
    center_error_px: float | None


@dataclass(frozen=True)
class EvaluationSummary:
    evaluated_visible_frames: int
    correct_frames: int
    lost_frames: int
    wrong_id_frames: int
    localization_error_frames: int
    false_positive_when_invisible: int
    id_switches: int
    p_correct: float
    p_lost: float
    p_wrong_id: float
    p_localization_error: float
    mean_center_error_px: float
    median_center_error_px: float
    max_center_error_px: float
    frames: tuple[FrameEvaluation, ...]


def predictions_from_session(session, *, include_reacq_tentative: bool = False) -> tuple[PredictionPoint, ...]:
    """Extract one deterministic prediction per frame from a TrackingSessionResult."""
    by_frame: dict[int, PredictionPoint] = {}
    for item in session.all_track_frames:
        if item.selected is not None:
            by_frame[item.frame_index] = PredictionPoint(
                item.frame_index, item.selected.center_x, item.selected.center_y, "TRACK", True
            )
    if include_reacq_tentative:
        for rr in session.reacquisitions:
            for item in rr.frames:
                if item.selected is not None and item.frame_index not in by_frame:
                    by_frame[item.frame_index] = PredictionPoint(
                        item.frame_index, item.selected.center_x, item.selected.center_y, "REACQ", True
                    )
    return tuple(by_frame[k] for k in sorted(by_frame))


def evaluate_predictions(
    gt: GroundTruthSequence,
    predictions: tuple[PredictionPoint, ...],
    *,
    tolerance_px: float = 20.0,
) -> EvaluationSummary:
    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be > 0")
    pred_map: Mapping[int, PredictionPoint] = {p.frame_index: p for p in predictions if p.accepted}
    eval_rows: list[FrameEvaluation] = []
    center_errors: list[float] = []
    correct = lost = wrong = locerr = false_invisible = 0
    wrong_state = False
    switches = 0

    for frame in gt.frames():
        target = gt.target_at(frame)
        if target is None:
            continue
        pred = pred_map.get(frame)
        if not target.visible:
            if pred is not None:
                false_invisible += 1
                eval_rows.append(FrameEvaluation(frame, "FALSE_POSITIVE_INVISIBLE", pred.center, target.center, None, None))
            else:
                eval_rows.append(FrameEvaluation(frame, "TARGET_INVISIBLE", None, target.center, None, None))
            wrong_state = False
            continue

        if pred is None:
            lost += 1
            eval_rows.append(FrameEvaluation(frame, "LOST", None, target.center, None, None))
            wrong_state = False
            continue

        d_target = hypot(pred.x - target.x, pred.y - target.y)
        if d_target <= tolerance_px:
            correct += 1
            center_errors.append(d_target)
            eval_rows.append(FrameEvaluation(frame, "CORRECT", pred.center, target.center, target.target_id, d_target))
            wrong_state = False
            continue

        distractors = [p for p in gt.objects_at(frame, visible_only=True) if not p.is_target]
        nearest = None
        nearest_d = float("inf")
        for obj in distractors:
            d = hypot(pred.x - obj.x, pred.y - obj.y)
            if d < nearest_d:
                nearest, nearest_d = obj, d
        if nearest is not None and nearest_d <= tolerance_px:
            wrong += 1
            if not wrong_state:
                switches += 1
            wrong_state = True
            eval_rows.append(FrameEvaluation(frame, "WRONG_ID", pred.center, target.center, nearest.target_id, d_target))
        else:
            locerr += 1
            wrong_state = False
            eval_rows.append(FrameEvaluation(frame, "LOCALIZATION_ERROR", pred.center, target.center, None, d_target))

    denom = correct + lost + wrong + locerr
    def frac(v: int) -> float:
        return float(v / denom) if denom else 0.0

    return EvaluationSummary(
        evaluated_visible_frames=denom,
        correct_frames=correct,
        lost_frames=lost,
        wrong_id_frames=wrong,
        localization_error_frames=locerr,
        false_positive_when_invisible=false_invisible,
        id_switches=switches,
        p_correct=frac(correct),
        p_lost=frac(lost),
        p_wrong_id=frac(wrong),
        p_localization_error=frac(locerr),
        mean_center_error_px=float(mean(center_errors)) if center_errors else 0.0,
        median_center_error_px=float(median(center_errors)) if center_errors else 0.0,
        max_center_error_px=max(center_errors, default=0.0),
        frames=tuple(eval_rows),
    )
