"""VT-5 multi-frame INITIAL TRAIN.

Starts from the operator-selected frame and the best VT-4 candidate.  The
routine maintains a short reference bank, predicts the next search center,
re-detects local candidates, scores target-to-candidate identity and returns a
transparent CONFIRMED / OBSERVE_MORE / AMBIGUOUS decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot
from statistics import median
from typing import Callable, Mapping, Sequence

import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    Candidate, CandidateDetectorConfig, detect_local_candidates, detect_initial_acquire_candidates, _click_seeded_large_detection,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import DescriptorSimilarity, descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import ThermalProcessingConfig, ThermalProfile, build_thermal_profile, preprocess_frame, runtime_detector_config


@dataclass(frozen=True)
class InitialTrainConfig:
    initial_frames: int = 7
    top_k: int = 5
    reference_bank_size: int = 5
    motion_weight: float = 0.30
    identity_weight: float = 0.60
    size_weight: float = 0.10
    motion_sigma_px: float = 28.0
    bank_update_min_score: float = 0.62
    confirm_min_valid_fraction: float = 0.70
    confirm_min_median_score: float = 0.62
    confirm_min_median_gap: float = 0.06
    observe_min_valid_fraction: float = 0.50
    observe_min_median_score: float = 0.48

    # VT-11 HOTFIX-4 REV4: temporal click bootstrap for large THERMAL targets.
    # The first clicked frame may be smeared/fragmented by a short sensor shock.
    # Instead of making that one frame the immutable seed, inspect a few
    # consecutive click-linked hypotheses and start TRAIN from the earliest
    # member of the dominant scale cluster. Small/normal targets keep the
    # historical one-frame bootstrap path.
    temporal_click_acquire_enabled: bool = False
    temporal_click_bootstrap_frames: int = 5
    temporal_click_min_stable_hits: int = 3
    temporal_click_max_log_area_deviation: float = 0.50

    def validate(self) -> None:
        if self.initial_frames < 2:
            raise ValueError("initial_frames must be >= 2")
        if self.top_k < 2 or self.reference_bank_size < 1:
            raise ValueError("top_k/reference_bank_size invalid")
        if min(self.motion_weight, self.identity_weight, self.size_weight) < 0:
            raise ValueError("train weights must be >= 0")
        if self.motion_weight + self.identity_weight + self.size_weight <= 0:
            raise ValueError("at least one train weight must be > 0")
        if self.motion_sigma_px <= 0:
            raise ValueError("motion_sigma_px must be > 0")
        if self.temporal_click_bootstrap_frames < 2:
            raise ValueError("temporal_click_bootstrap_frames must be >=2")
        if self.temporal_click_min_stable_hits < 2 or self.temporal_click_min_stable_hits > self.temporal_click_bootstrap_frames:
            raise ValueError("temporal_click_min_stable_hits invalid")
        if self.temporal_click_max_log_area_deviation <= 0:
            raise ValueError("temporal_click_max_log_area_deviation must be >0")
        for name in (
            "bank_update_min_score", "confirm_min_valid_fraction", "confirm_min_median_score",
            "confirm_min_median_gap", "observe_min_valid_fraction", "observe_min_median_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class TrainCandidateScore:
    candidate: Candidate
    similarity: DescriptorSimilarity
    motion_score: float
    size_score: float
    total_score: float


@dataclass(frozen=True)
class TrainFrameResult:
    frame_index: int
    predicted_xy: tuple[float, float]
    selected: Candidate | None
    selected_score: float
    confidence_gap: float
    candidate_count: int
    identity_similarity: float
    motion_score: float
    size_score: float


@dataclass(frozen=True)
class InitialTrainResult:
    decision: str
    frames: tuple[TrainFrameResult, ...]
    reference_bank: ReferenceBank
    valid_fraction: float
    median_score: float
    median_gap: float
    last_center: tuple[float, float] | None
    last_bbox: tuple[int, int, int, int] | None
    video_mode: str = "visible"
    thermal_profile: ThermalProfile | None = None
    temporal_click_used: bool = False
    temporal_click_shift_frames: int = 0
    temporal_click_stable_hits: int = 0


FrameReader = Callable[[int], tuple[int, np.ndarray]]


def _is_large_acquire_candidate(candidate: Candidate, detector: CandidateDetectorConfig) -> bool:
    channel = str(getattr(candidate, "detection_channel", "NORMAL"))
    return bool(
        channel in {"ACQUIRE_CLICK_SEEDED", "ACQUIRE_MULTISCALE_LARGE"}
        or float(candidate.area_px) >= float(detector.acquire_large_target_trigger_area_px)
    )


def _temporal_click_bootstrap(
    *,
    raw_frames: Mapping[int, np.ndarray],
    start_frame: int,
    frame_width: int,
    frame_height: int,
    initial_candidate: Candidate,
    initial_click_xy: tuple[float, float] | None,
    roi_radius_px: int,
    runtime_detector: CandidateDetectorConfig,
    profile: ThermalProfile,
    config: InitialTrainConfig,
) -> tuple[int, Candidate, tuple[tuple[int, Candidate], ...]]:
    """Choose a robust large-target bootstrap across a short click-linked window.

    This is intentionally *not* a new general tracker.  It is used only for a
    large THERMAL acquisition and only before BaseBank is frozen.  Candidate
    proposals are chained spatially from the operator indication, then the
    dominant log-area cluster is used to reject one/few smeared startup frames.
    """
    if (
        not config.temporal_click_acquire_enabled
        or not profile.is_thermal
        or runtime_detector.method != "thermal_hot_blob"
        or not runtime_detector.acquire_click_seed_enabled
    ):
        return start_frame, initial_candidate, ()

    initial_large = _is_large_acquire_candidate(initial_candidate, runtime_detector)
    # If the current candidate is a normal/small target, do not activate the
    # more permissive temporal large-object path.
    if not initial_large:
        return start_frame, initial_candidate, ()

    proposals: list[tuple[int, Candidate]] = []
    search_xy = tuple(map(float, initial_click_xy or initial_candidate.center))
    last_center: tuple[float, float] | None = None
    stop_frame = start_frame + int(config.temporal_click_bootstrap_frames)
    for fi in range(start_frame, stop_frame):
        if fi not in raw_frames:
            break
        frame = preprocess_frame(raw_frames[fi], profile)
        click_xy = search_xy if last_center is None else last_center
        selection = make_operator_selection(
            frame_index=fi, click_xy=click_xy,
            radius_px=max(int(roi_radius_px), int(runtime_detector.acquire_click_seed_roi_radius_px)),
            frame_width=frame_width, frame_height=frame_height,
        )
        detected, _effective = detect_initial_acquire_candidates(
            frame, selection, top_k=max(5, config.top_k), config=runtime_detector,
        )
        large = [c for c in detected.candidates if _is_large_acquire_candidate(c, runtime_detector)]
        if fi == start_frame and _is_large_acquire_candidate(initial_candidate, runtime_detector):
            large.append(initial_candidate)
        if not large:
            continue
        # Prefer an explicitly click-supported large hypothesis; otherwise use
        # the largest coherent component near the propagated indication.
        large.sort(key=lambda c: (
            0 if str(getattr(c, "detection_channel", "")) == "ACQUIRE_CLICK_SEEDED" else 1,
            float(c.distance_to_click_px), -float(c.area_px), -float(c.mean_response),
        ))
        cand = large[0]
        proposals.append((fi, cand))
        last_center = cand.center

    if len(proposals) < int(config.temporal_click_min_stable_hits):
        return start_frame, initial_candidate, tuple(proposals)

    logs = [float(np.log(max(1.0, float(c.area_px)))) for _, c in proposals]
    med_log = float(median(logs))
    stable = [
        item for item, la in zip(proposals, logs)
        if abs(la - med_log) <= float(config.temporal_click_max_log_area_deviation)
    ]
    if len(stable) < int(config.temporal_click_min_stable_hits):
        return start_frame, initial_candidate, tuple(proposals)

    # Earliest stable frame avoids unnecessary delay.  The later stable members
    # are returned for diagnostics/tests; normal TRAIN still validates the full
    # sequence before CONFIRMED.
    stable.sort(key=lambda item: item[0])
    bootstrap_frame, bootstrap_candidate = stable[0]
    return int(bootstrap_frame), bootstrap_candidate, tuple(stable)


def _motion_prediction(history: Sequence[tuple[float, float]], fallback: tuple[float, float]) -> tuple[float, float]:
    if not history:
        return fallback
    if len(history) == 1:
        return history[-1]
    # Robust short constant-velocity estimate from the last 2-4 displacements.
    recent = history[-5:]
    dx = [recent[i][0] - recent[i - 1][0] for i in range(1, len(recent))]
    dy = [recent[i][1] - recent[i - 1][1] for i in range(1, len(recent))]
    vx = float(median(dx)) if dx else 0.0
    vy = float(median(dy)) if dy else 0.0
    return recent[-1][0] + vx, recent[-1][1] + vy


def _motion_score(center: tuple[float, float], predicted: tuple[float, float], sigma: float) -> float:
    d = hypot(center[0] - predicted[0], center[1] - predicted[1])
    return float(exp(-(d * d) / (2.0 * sigma * sigma)))


def _rank_candidates(
    frame: np.ndarray,
    candidates: Sequence[Candidate],
    bank: ReferenceBank,
    predicted_xy: tuple[float, float],
    *,
    cfg: InitialTrainConfig,
    identity_weights: Mapping[str, float] | None,
) -> list[TrainCandidateScore]:
    weights_sum = cfg.identity_weight + cfg.motion_weight + cfg.size_weight
    ranked: list[TrainCandidateScore] = []
    for candidate in candidates:
        desc = descriptor_from_candidate(frame, candidate)
        sim = bank.similarity(desc, identity_weights=identity_weights)
        motion = _motion_score(candidate.center, predicted_xy, cfg.motion_sigma_px)
        total = (
            cfg.identity_weight * sim.total
            + cfg.motion_weight * motion
            + cfg.size_weight * sim.size
        ) / weights_sum
        ranked.append(
            TrainCandidateScore(
                candidate=candidate,
                similarity=sim,
                motion_score=float(motion),
                size_score=float(sim.size),
                total_score=float(total),
            )
        )
    ranked.sort(key=lambda r: (-r.total_score, -r.similarity.total, r.candidate.distance_to_click_px))
    return ranked


def run_initial_train(
    *,
    frame_reader: FrameReader,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    initial_candidate: Candidate,
    roi_radius_px: int,
    detector_config: CandidateDetectorConfig,
    config: InitialTrainConfig,
    identity_weights: Mapping[str, float] | None = None,
    thermal_identity_weights: Mapping[str, float] | None = None,
    video_mode: str = "visible",
    thermal_config: ThermalProcessingConfig | None = None,
    initial_click_xy: tuple[float, float] | None = None,
) -> InitialTrainResult:
    config.validate()
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if not 0 <= start_frame < frame_count:
        raise ValueError("start_frame outside video")

    # Read a small look-ahead only when temporal click acquisition is enabled.
    # This lets a bad clicked frame be skipped without shortening the requested
    # INITIAL TRAIN validation window.
    lookahead = max(0, int(config.temporal_click_bootstrap_frames) - 1) if config.temporal_click_acquire_enabled else 0
    raw_end = min(frame_count, start_frame + config.initial_frames + lookahead)
    raw_frames: dict[int, np.ndarray] = {}
    for fi in range(start_frame, raw_end):
        _idx, fr = frame_reader(fi)
        raw_frames[fi] = fr
    profile_cfg = thermal_config or ThermalProcessingConfig(video_mode=video_mode)
    if profile_cfg.video_mode != video_mode:
        from dataclasses import replace
        profile_cfg = replace(profile_cfg, video_mode=video_mode)
    profile = build_thermal_profile(
        [raw_frames[fi] for fi in range(start_frame, raw_end)],
        config=profile_cfg, initial_candidate=initial_candidate,
    )
    active_weights = thermal_identity_weights if profile.is_thermal and thermal_identity_weights else identity_weights
    runtime_detector = runtime_detector_config(detector_config, profile)

    # REV4 temporal click bootstrap: if the clicked large target is corrupted on
    # the first frame, use the earliest member of the dominant short-term scale
    # cluster as the actual immutable TRAIN seed.
    clicked_start_frame = int(start_frame)
    start_frame, initial_candidate, _temporal_stable = _temporal_click_bootstrap(
        raw_frames=raw_frames, start_frame=start_frame, frame_width=frame_width, frame_height=frame_height,
        initial_candidate=initial_candidate, initial_click_xy=initial_click_xy, roi_radius_px=roi_radius_px,
        runtime_detector=runtime_detector, profile=profile, config=config,
    )
    end_frame = min(frame_count, start_frame + config.initial_frames)
    # Ensure the shifted validation window is available (normally already read
    # by the look-ahead above).
    for fi in range(start_frame, end_frame):
        if fi not in raw_frames:
            _idx, fr = frame_reader(fi)
            raw_frames[fi] = fr

    bank = ReferenceBank(max_size=config.reference_bank_size)
    first_frame = preprocess_frame(raw_frames[start_frame], profile)
    bank.add(descriptor_from_candidate(first_frame, initial_candidate))

    history: list[tuple[float, float]] = [initial_candidate.center]
    prev_center = initial_candidate.center
    large_target_train = bool(
        runtime_detector.method == "thermal_hot_blob"
        and runtime_detector.acquire_multiscale_enabled
        and _is_large_acquire_candidate(initial_candidate, runtime_detector)
    )
    train_roi_radius = int(roi_radius_px)
    area_history: list[float] = [float(initial_candidate.area_px)]
    if large_target_train:
        suggested = int(round(0.65 * max(initial_candidate.bbox.width, initial_candidate.bbox.height) + 30.0))
        train_roi_radius = min(
            int(runtime_detector.acquire_expand_roi_radius_px),
            max(int(roi_radius_px), suggested),
        )
    frame_results: list[TrainFrameResult] = [
        TrainFrameResult(
            frame_index=start_frame,
            predicted_xy=initial_candidate.center,
            selected=initial_candidate,
            selected_score=1.0,
            confidence_gap=1.0,
            candidate_count=1,
            identity_similarity=1.0,
            motion_score=1.0,
            size_score=1.0,
        )
    ]

    for frame_index in range(start_frame + 1, end_frame):
        frame = preprocess_frame(raw_frames[frame_index], profile)
        predicted = _motion_prediction(history, prev_center)
        predicted = (
            min(max(predicted[0], 0.0), frame_width - 1.0),
            min(max(predicted[1], 0.0), frame_height - 1.0),
        )
        selection = make_operator_selection(
            frame_index=frame_index,
            click_xy=predicted,
            radius_px=train_roi_radius,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        expected_train_area = float(median(area_history[-3:])) if large_target_train else None
        detected = detect_local_candidates(
            frame, selection, top_k=config.top_k, config=runtime_detector,
            expected_area_px=expected_train_area,
        )
        # REV2: if a target was already acquired as large but the contrast
        # detector fragments it on a TRAIN frame, retry with the same
        # click-seeded raw-thermal segmentation around the motion prediction.
        if large_target_train and expected_train_area is not None:
            largest_area = max((float(c.area_px) for c in detected.candidates), default=0.0)
            if largest_area < 0.45 * expected_train_area:
                seeded, _seed_sel = _click_seeded_large_detection(frame, selection, runtime_detector)
                if seeded is not None and seeded.candidates:
                    sa = float(seeded.candidates[0].area_px)
                    if 0.25 * expected_train_area <= sa <= 4.0 * expected_train_area:
                        detected = seeded
        if not detected.candidates:
            frame_results.append(
                TrainFrameResult(frame_index, predicted, None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
            )
            continue

        ranked = _rank_candidates(
            frame,
            detected.candidates,
            bank,
            predicted,
            cfg=config,
            identity_weights=active_weights,
        )
        best = ranked[0]
        gap = best.total_score - ranked[1].total_score if len(ranked) > 1 else best.total_score
        selected = best.candidate
        prev_center = selected.center
        history.append(prev_center)
        if large_target_train:
            area_history.append(float(selected.area_px))
        if best.total_score >= config.bank_update_min_score:
            bank.add(descriptor_from_candidate(frame, selected))
        frame_results.append(
            TrainFrameResult(
                frame_index=frame_index,
                predicted_xy=predicted,
                selected=selected,
                selected_score=best.total_score,
                confidence_gap=max(0.0, float(gap)),
                candidate_count=len(ranked),
                identity_similarity=best.similarity.total,
                motion_score=best.motion_score,
                size_score=best.size_score,
            )
        )

    valid = [r for r in frame_results if r.selected is not None]
    # The denominator is the requested training window, not merely the
    # number of frames still available before EOF. This prevents a click on
    # the final video frame from being falsely CONFIRMED from one sample.
    valid_fraction = len(valid) / max(1, config.initial_frames)
    subsequent_valid = [r for r in frame_results[1:] if r.selected is not None]
    scores = [r.selected_score for r in subsequent_valid]
    gaps = [r.confidence_gap for r in subsequent_valid]
    med_score = float(median(scores)) if scores else 0.0
    med_gap = float(median(gaps)) if gaps else 0.0

    if (
        valid_fraction >= config.confirm_min_valid_fraction
        and med_score >= config.confirm_min_median_score
        and med_gap >= config.confirm_min_median_gap
    ):
        decision = "CONFIRMED"
    elif valid_fraction >= config.observe_min_valid_fraction and med_score >= config.observe_min_median_score:
        decision = "OBSERVE_MORE"
    else:
        decision = "AMBIGUOUS"

    last = next((r for r in reversed(frame_results) if r.selected is not None), None)
    if last is None:
        last_center = None
        last_bbox = None
    else:
        last_center = last.selected.center
        b = last.selected.bbox
        last_bbox = (b.x, b.y, b.width, b.height)

    return InitialTrainResult(
        decision=decision,
        frames=tuple(frame_results),
        reference_bank=bank,
        valid_fraction=float(valid_fraction),
        median_score=med_score,
        median_gap=med_gap,
        last_center=last_center,
        last_bbox=last_bbox,
        video_mode=profile.video_mode,
        thermal_profile=profile,
        temporal_click_used=bool(start_frame != clicked_start_frame or len(_temporal_stable) >= config.temporal_click_min_stable_hits),
        temporal_click_shift_frames=int(start_frame - clicked_start_frame),
        temporal_click_stable_hits=int(len(_temporal_stable)),
    )
