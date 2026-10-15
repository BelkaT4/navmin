"""VT-4 single-frame ACQUIRE feature scoring and confidence gap.

Important: these are candidate-quality scores before INITIAL TRAIN. They do not
replace the target-reference identity distances planned for VT-5.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from navmin.vision.processors.vt11.features.appearance import appearance_score
from navmin.vision.processors.vt11.features.contrast import contrast_score
from navmin.vision.processors.vt11.features.edge import edge_score
from navmin.vision.processors.vt11.features.geometry import geometry_score
from navmin.vision.processors.vt11.features.projection import projection_score


SCORING_MODES = ("nearest", "appearance", "edge", "projection", "contrast", "geometry", "combined", "custom")


@dataclass(frozen=True)
class CandidateFeatureVector:
    candidate_id: int
    spatial: float
    appearance: float
    edge: float
    projection: float
    contrast: float
    geometry: float


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: object
    features: CandidateFeatureVector
    final_score: float
    final_rank: int = 0


@dataclass(frozen=True)
class ScoringResult:
    candidates: tuple[ScoredCandidate, ...]
    mode: str
    confidence_gap: float
    confidence_state: str

    @property
    def best(self) -> ScoredCandidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True)
class AcquireScoringConfig:
    mode: str = "combined"
    combined_spatial_weight: float = 0.45
    confidence_gap_low: float = 0.05
    confidence_gap_high: float = 0.12
    custom_weights: Mapping[str, float] | None = None

    def validate(self) -> None:
        if self.mode not in SCORING_MODES:
            raise ValueError(f"Unknown scoring mode: {self.mode}")
        if not 0.0 <= self.combined_spatial_weight <= 1.0:
            raise ValueError("combined_spatial_weight must be in [0,1]")
        if self.confidence_gap_low < 0 or self.confidence_gap_high < self.confidence_gap_low:
            raise ValueError("confidence gap thresholds are invalid")
        if self.custom_weights is not None:
            if any(v < 0 for v in self.custom_weights.values()):
                raise ValueError("custom weights must be >= 0")
            if sum(self.custom_weights.values()) <= 0:
                raise ValueError("custom weights need a positive sum")


def extract_candidate_features(frame_bgr, candidate) -> CandidateFeatureVector:
    return CandidateFeatureVector(
        candidate_id=int(candidate.candidate_id),
        spatial=float(candidate.spatial_prior),
        appearance=appearance_score(frame_bgr, candidate),
        edge=edge_score(frame_bgr, candidate),
        projection=projection_score(frame_bgr, candidate),
        contrast=contrast_score(frame_bgr, candidate),
        geometry=geometry_score(frame_bgr, candidate),
    )


def _weighted_average(features: CandidateFeatureVector, weights: Mapping[str, float]) -> float:
    names = ("spatial", "appearance", "edge", "projection", "contrast", "geometry")
    total = sum(max(0.0, float(weights.get(name, 0.0))) for name in names)
    if total <= 0:
        raise ValueError("At least one scoring weight must be > 0")
    return sum(max(0.0, float(weights.get(name, 0.0))) * float(getattr(features, name)) for name in names) / total


def score_feature_vector(
    features: CandidateFeatureVector,
    mode: str,
    *,
    identity_weights: Mapping[str, float] | None = None,
    combined_spatial_weight: float = 0.45,
    custom_weights: Mapping[str, float] | None = None,
) -> float:
    if mode not in SCORING_MODES:
        raise ValueError(f"Unknown scoring mode: {mode}")
    if mode == "nearest":
        return features.spatial
    if mode in {"appearance", "edge", "projection", "contrast", "geometry"}:
        return float(getattr(features, mode))
    if mode == "custom":
        return _weighted_average(features, custom_weights or {"spatial": 1.0})

    # combined: preserve the previously tested M2 relative weighting among
    # appearance/edge/projection, while keeping click proximity as a strong
    # independent ACQUIRE prior. contrast/geometry remain available through
    # individual and custom modes until separately validated.
    iw = dict(identity_weights or {"appearance": 0.603654, "edge": 0.288078, "projection": 0.108268})
    core_names = ("appearance", "edge", "projection")
    denom = sum(max(0.0, float(iw.get(name, 0.0))) for name in core_names)
    if denom <= 0:
        raise ValueError("combined mode needs positive appearance/edge/projection weights")
    core = sum(max(0.0, float(iw.get(name, 0.0))) * float(getattr(features, name)) for name in core_names) / denom
    sw = min(1.0, max(0.0, float(combined_spatial_weight)))
    return sw * features.spatial + (1.0 - sw) * core


def score_candidates(
    frame_bgr,
    candidates: Sequence[object],
    *,
    config: AcquireScoringConfig | None = None,
    identity_weights: Mapping[str, float] | None = None,
) -> ScoringResult:
    cfg = config or AcquireScoringConfig()
    cfg.validate()
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        features = extract_candidate_features(frame_bgr, candidate)
        final = score_feature_vector(
            features,
            cfg.mode,
            identity_weights=identity_weights,
            combined_spatial_weight=cfg.combined_spatial_weight,
            custom_weights=cfg.custom_weights,
        )
        scored.append(ScoredCandidate(candidate=candidate, features=features, final_score=float(final)))

    scored.sort(key=lambda s: (-s.final_score, s.candidate.distance_to_click_px, s.candidate.candidate_id))
    ranked = tuple(replace(s, final_rank=i) for i, s in enumerate(scored, start=1))
    if len(ranked) >= 2:
        gap = max(0.0, ranked[0].final_score - ranked[1].final_score)
    elif len(ranked) == 1:
        gap = max(0.0, ranked[0].final_score)
    else:
        gap = 0.0

    if not ranked:
        state = "NO_CANDIDATES"
    elif gap >= cfg.confidence_gap_high:
        state = "HIGH"
    elif gap >= cfg.confidence_gap_low:
        state = "MEDIUM"
    else:
        state = "LOW"
    return ScoringResult(candidates=ranked, mode=cfg.mode, confidence_gap=float(gap), confidence_state=state)
