"""VT-10 automated acquisition and Fixed/EMA experiments."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, hypot, pi, sin
from statistics import mean
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig
from navmin.vision.processors.vt11.evaluation.metrics import EvaluationSummary, evaluate_predictions, predictions_from_session
from navmin.vision.processors.vt11.identity.scorer import AcquireScoringConfig, SCORING_MODES, score_candidates
from navmin.vision.processors.vt11.tracking.session import run_tracking_session


@dataclass(frozen=True)
class AcquisitionTrial:
    mode: str
    click_error_px: float
    direction_deg: float
    click_xy: tuple[float, float]
    best_xy: tuple[float, float] | None
    center_error_px: float | None
    success: bool
    confidence_gap: float
    candidates: int


@dataclass(frozen=True)
class AcquisitionExperimentSummary:
    trials: tuple[AcquisitionTrial, ...]

    @property
    def success_rate(self) -> float:
        return float(mean(1.0 if t.success else 0.0 for t in self.trials)) if self.trials else 0.0


@dataclass(frozen=True)
class SessionExperimentResult:
    mode: str
    elapsed_s: float
    processed_frames: int
    effective_fps: float
    session: object
    evaluation: EvaluationSummary | None


def run_acquisition_experiment(
    *, frame_bgr, frame_index: int, gt_center: tuple[float, float], roi_radius_px: int,
    frame_width: int, frame_height: int, detector_config: CandidateDetectorConfig,
    top_k: int, click_errors_px: Sequence[float], directions: int = 8,
    modes: Sequence[str] = SCORING_MODES, scoring_template: AcquireScoringConfig | None = None,
    identity_weights: Mapping[str, float] | None = None, success_tolerance_px: float = 20.0,
) -> AcquisitionExperimentSummary:
    if directions < 1:
        raise ValueError("directions must be >= 1")
    if success_tolerance_px <= 0:
        raise ValueError("success_tolerance_px must be > 0")
    cfg0 = scoring_template or AcquireScoringConfig()
    trials = []
    for error in click_errors_px:
        if error < 0:
            raise ValueError("click errors must be >= 0")
        dir_count = 1 if error == 0 else directions
        for di in range(dir_count):
            angle = 2*pi*di/dir_count
            click = (gt_center[0] + error*cos(angle), gt_center[1] + error*sin(angle))
            selection = make_operator_selection(frame_index=frame_index, click_xy=click, radius_px=roi_radius_px, frame_width=frame_width, frame_height=frame_height)
            detection = detect_local_candidates(frame_bgr, selection, top_k=top_k, config=detector_config)
            for mode in modes:
                if mode not in SCORING_MODES:
                    raise ValueError(f"Unknown mode: {mode}")
                scoring = score_candidates(frame_bgr, detection.candidates, config=replace(cfg0, mode=mode), identity_weights=identity_weights)
                best = scoring.best
                best_xy = None if best is None else best.candidate.center
                err = None if best_xy is None else hypot(best_xy[0]-gt_center[0], best_xy[1]-gt_center[1])
                trials.append(AcquisitionTrial(mode, float(error), float(angle*180/pi), click, best_xy, err, bool(err is not None and err <= success_tolerance_px), scoring.confidence_gap, len(detection.candidates)))
    return AcquisitionExperimentSummary(tuple(trials))


def run_session_experiment(*, label: str, ground_truth=None, gt_tolerance_px: float = 20.0, include_reacq_tentative: bool = False, **session_kwargs) -> SessionExperimentResult:
    t0 = perf_counter()
    session = run_tracking_session(**session_kwargs)
    elapsed = perf_counter() - t0
    processed = len(session.all_track_frames) + len(session.all_reacq_frames)
    eval_result = None
    if ground_truth is not None:
        preds = predictions_from_session(session, include_reacq_tentative=include_reacq_tentative)
        eval_result = evaluate_predictions(ground_truth, preds, tolerance_px=gt_tolerance_px)
    return SessionExperimentResult(label, elapsed, processed, float(processed/elapsed) if elapsed > 0 else 0.0, session, eval_result)


def compare_fixed_ema(*, adaptation_ema: EMAConfig, **kwargs) -> tuple[SessionExperimentResult, SessionExperimentResult]:
    fixed = run_session_experiment(label="fixed", adaptation_config=EMAConfig(mode="off"), **kwargs)
    ema = run_session_experiment(label="ema", adaptation_config=adaptation_ema, **kwargs)
    return fixed, ema


def summarize_acquisition(trials: Iterable[AcquisitionTrial]):
    """Return grouped dictionaries by (mode, click_error_px)."""
    groups: dict[tuple[str, float], list[AcquisitionTrial]] = {}
    for t in trials:
        groups.setdefault((t.mode, t.click_error_px), []).append(t)
    rows = []
    for (mode, error), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        errs = [t.center_error_px for t in items if t.center_error_px is not None]
        rows.append({
            "mode": mode,
            "click_error_px": error,
            "trials": len(items),
            "success_rate": mean(1.0 if t.success else 0.0 for t in items),
            "mean_center_error_px": mean(errs) if errs else 0.0,
        })
    return tuple(rows)
