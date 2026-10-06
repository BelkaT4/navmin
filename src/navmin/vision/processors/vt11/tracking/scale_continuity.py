"""Reliable scale anchor and nonlinear parameter-distance gate.

VT-11 HOTFIX-4 REV3.

The ordinary ``size_score`` remains a soft ranking cue based on CurrentArea.
This module adds an independent physical-plausibility relation to the last
trusted target scale.  A candidate that is tens/hundreds of times smaller or
larger cannot compensate that mismatch merely with a high visual identity.

The distance is symmetric in scale because it is measured in log-area:

    D_scale = abs(log(A_candidate / A_reliable))

The permitted distance grows slowly with elapsed frames.  Moderate excesses
can be accepted only after a short temporal quarantine; very large excesses
must first become plausible through elapsed time rather than being confirmed
in a few frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log


@dataclass(frozen=True)
class ScaleContinuityConfig:
    enabled: bool = False
    base_allowed_log_distance: float = 0.85
    growth_per_frame: float = 0.045
    max_allowed_log_distance: float = 1.60
    nonlinear_strength: float = 1.50
    quarantine_confirm_hits: int = 3
    quarantine_extreme_confirm_hits: int = 5
    quarantine_extreme_log_distance: float = 2.30
    quarantine_max_confirm_excess_log: float = 1.10
    quarantine_link_log_span: float = 0.45
    reliable_update_alpha: float = 0.20

    def validate(self) -> None:
        if self.base_allowed_log_distance <= 0:
            raise ValueError("base_allowed_log_distance must be >0")
        if self.growth_per_frame < 0:
            raise ValueError("growth_per_frame must be >=0")
        if self.max_allowed_log_distance < self.base_allowed_log_distance:
            raise ValueError("max_allowed_log_distance must be >= base_allowed_log_distance")
        if self.nonlinear_strength <= 0:
            raise ValueError("nonlinear_strength must be >0")
        if self.quarantine_confirm_hits < 2:
            raise ValueError("quarantine_confirm_hits must be >=2")
        if self.quarantine_extreme_confirm_hits < self.quarantine_confirm_hits:
            raise ValueError("quarantine_extreme_confirm_hits must be >= quarantine_confirm_hits")
        if self.quarantine_extreme_log_distance <= self.base_allowed_log_distance:
            raise ValueError("quarantine_extreme_log_distance too small")
        if self.quarantine_max_confirm_excess_log <= 0:
            raise ValueError("quarantine_max_confirm_excess_log must be >0")
        if self.quarantine_link_log_span <= 0:
            raise ValueError("quarantine_link_log_span must be >0")
        if not 0.0 < self.reliable_update_alpha <= 1.0:
            raise ValueError("reliable_update_alpha must be in (0,1]")


@dataclass(frozen=True)
class ScaleDistanceMetrics:
    reliable_area_px: float
    anchor_age_frames: int
    signed_log_ratio: float
    log_distance: float
    allowed_log_distance: float
    excess_log_distance: float
    nonlinear_gate: float
    suspicious: bool
    extreme: bool
    quick_confirmable: bool

    @property
    def area_ratio(self) -> float:
        return float(exp(self.log_distance))

    @property
    def is_expansion(self) -> bool:
        return bool(self.signed_log_ratio > 0.0)

    @property
    def is_shrink(self) -> bool:
        return bool(self.signed_log_ratio < 0.0)


class ReliableScaleAnchor:
    """Session-wide scale anchor changed only by trusted observations."""

    def __init__(self, area_px: float, frame_index: int, config: ScaleContinuityConfig | None = None) -> None:
        self.config = config or ScaleContinuityConfig()
        self.config.validate()
        self.reliable_area_px = max(1.0, float(area_px))
        self.reliable_frame = int(frame_index)

    def allowed_log_distance(self, frame_index: int) -> tuple[int, float]:
        age = max(0, int(frame_index) - int(self.reliable_frame))
        cfg = self.config
        allowed = min(
            float(cfg.max_allowed_log_distance),
            float(cfg.base_allowed_log_distance) + float(cfg.growth_per_frame) * age,
        )
        return age, float(allowed)

    def metrics(self, area_px: float, frame_index: int) -> ScaleDistanceMetrics:
        area = max(1.0, float(area_px))
        age, allowed = self.allowed_log_distance(frame_index)
        signed = log(area / max(1.0, self.reliable_area_px))
        d = abs(signed)
        excess = max(0.0, d - allowed)
        if excess <= 0:
            gate = 1.0
        else:
            # Nonlinear multiplicative penalty.  Normal scale evolution inside
            # the allowance remains exactly 1.0; only the excess is penalized.
            norm = excess / max(0.25, allowed)
            gate = exp(-float(self.config.nonlinear_strength) * norm * norm)
        return ScaleDistanceMetrics(
            reliable_area_px=float(self.reliable_area_px),
            anchor_age_frames=int(age),
            signed_log_ratio=float(signed),
            log_distance=float(d),
            allowed_log_distance=float(allowed),
            excess_log_distance=float(excess),
            nonlinear_gate=float(max(0.0, min(1.0, gate))),
            suspicious=bool(d > allowed),
            extreme=bool(d >= float(self.config.quarantine_extreme_log_distance)),
            quick_confirmable=bool(excess <= float(self.config.quarantine_max_confirm_excess_log)),
        )

    def update_trusted(self, area_px: float, frame_index: int) -> None:
        """Follow normal scale evolution slowly in log space."""
        area = max(1.0, float(area_px))
        a = float(self.config.reliable_update_alpha)
        new_log = (1.0 - a) * log(max(1.0, self.reliable_area_px)) + a * log(area)
        self.reliable_area_px = float(exp(new_log))
        self.reliable_frame = int(frame_index)

    def commit_confirmed_transition(self, areas_px: list[float], frame_index: int) -> None:
        """Commit a temporally confirmed new scale using geometric median-ish centre.

        Using the median in log-space prevents one segmentation outlier from
        redefining the reliable scale after quarantine.
        """
        if not areas_px:
            return
        logs = sorted(log(max(1.0, float(a))) for a in areas_px)
        n = len(logs)
        med = logs[n // 2] if n % 2 else 0.5 * (logs[n // 2 - 1] + logs[n // 2])
        self.reliable_area_px = float(exp(med))
        self.reliable_frame = int(frame_index)
