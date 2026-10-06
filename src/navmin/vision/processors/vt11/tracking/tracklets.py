"""Lightweight multi-hypothesis tracklets for robust REACQ (HOTFIX-6).

This module does not rerun detection or descriptors.  It only associates the
already-scored Top-K candidates over a short temporal window, so its CPU cost
is small compared with image processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, log
from statistics import mean, median
from typing import Sequence, Any


@dataclass
class TrackletEvidence:
    frame_index: int
    x: float
    y: float
    identity: float
    total_score: float
    size_score: float
    gap: float
    candidate: Any
    has_competitor: bool = True
    anchor_distance_px: float = 0.0
    far_jump: bool = False
    is_micro: bool = False
    background_dx_px: float = 0.0
    background_dy_px: float = 0.0
    background_valid: bool = False
    thermal_profile_score: float = 0.0
    scale_log_distance: float = 0.0
    scale_allowed_log_distance: float = 0.0
    scale_excess_log_distance: float = 0.0
    scale_nonlinear_gate: float = 1.0
    far_scale: bool = False
    scale_quick_confirmable: bool = True


@dataclass
class Tracklet:
    tracklet_id: int
    evidence: list[TrackletEvidence] = field(default_factory=list)
    last_seen_frame: int = -1

    @property
    def last(self) -> TrackletEvidence:
        return self.evidence[-1]

    def predicted_xy(self) -> tuple[float, float]:
        if len(self.evidence) < 2:
            return self.last.x, self.last.y
        a, b = self.evidence[-2], self.evidence[-1]
        dt = max(1, b.frame_index - a.frame_index)
        steps = max(1, b.frame_index + 1 - self.last.frame_index)
        vx = (b.x - a.x) / dt
        vy = (b.y - a.y) / dt
        return b.x + vx * steps, b.y + vy * steps

    def recent(self, window_frames: int, current_frame: int) -> list[TrackletEvidence]:
        lo = current_frame - max(1, window_frames) + 1
        return [e for e in self.evidence if e.frame_index >= lo]


@dataclass(frozen=True)
class TrackletConfig:
    max_tracklets: int = 3
    window_frames: int = 5
    confirm_hits: int = 3
    max_link_distance_px: float = 100.0
    max_stale_frames: int = 2
    min_mean_identity: float = 0.46
    min_mean_score: float = 0.50
    min_mean_gap: float = 0.025

    # REV2 identity-first confirmation. A tracklet that starts far from the
    # last reliable target needs more/stronger evidence than a local continuation.
    far_jump_confirm_hits: int = 5
    far_jump_min_mean_identity: float = 0.78
    far_jump_min_mean_gap: float = 0.06
    far_jump_min_median_thermal_profile: float = 0.68
    max_scale_log_span: float = 0.65
    scale_distance_confirm_hits: int = 3
    scale_distance_extreme_confirm_hits: int = 5
    scale_distance_extreme_log_distance: float = 2.30
    scale_distance_max_confirm_excess_log: float = 1.10
    micro_confirm_hits: int = 3
    micro_min_mean_identity: float = 0.52
    background_attachment_enabled: bool = True
    background_motion_min_px: float = 2.0
    background_attachment_tolerance_px: float = 2.5
    background_attachment_max_fraction: float = 0.80

    def validate(self) -> None:
        if self.max_tracklets < 1:
            raise ValueError("max_tracklets must be >=1")
        if self.window_frames < self.confirm_hits or self.confirm_hits < 2:
            raise ValueError("tracklet window/confirm_hits invalid")
        if self.max_link_distance_px <= 0 or self.max_stale_frames < 0:
            raise ValueError("tracklet geometry invalid")
        if self.far_jump_confirm_hits < self.confirm_hits:
            raise ValueError("far_jump_confirm_hits must be >= confirm_hits")
        if self.micro_confirm_hits < 2:
            raise ValueError("micro_confirm_hits must be >=2")
        for name in ("far_jump_min_mean_identity", "far_jump_min_mean_gap", "far_jump_min_median_thermal_profile", "micro_min_mean_identity", "background_attachment_max_fraction"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.background_motion_min_px < 0 or self.background_attachment_tolerance_px < 0:
            raise ValueError("background attachment thresholds must be >=0")
        if self.max_scale_log_span <= 0:
            raise ValueError("max_scale_log_span must be >0")
        if self.scale_distance_confirm_hits < 2 or self.scale_distance_extreme_confirm_hits < self.scale_distance_confirm_hits:
            raise ValueError("scale distance confirmation counts invalid")
        if self.scale_distance_extreme_log_distance <= 0 or self.scale_distance_max_confirm_excess_log <= 0:
            raise ValueError("scale distance thresholds invalid")


class TrackletManager:
    def __init__(self, config: TrackletConfig | None = None) -> None:
        self.config = config or TrackletConfig()
        self.config.validate()
        self.tracklets: list[Tracklet] = []
        self._next_id = 1

    def _spawn(self, ev: TrackletEvidence) -> Tracklet:
        tr = Tracklet(self._next_id, [ev], ev.frame_index)
        self._next_id += 1
        self.tracklets.append(tr)
        return tr

    def update(self, frame_index: int, evidences: Sequence[TrackletEvidence]) -> None:
        cfg = self.config
        # Remove stale hypotheses first.
        self.tracklets = [t for t in self.tracklets if frame_index - t.last_seen_frame <= cfg.max_stale_frames]
        remaining = set(range(len(evidences)))
        # Greedy nearest association, ordered by strongest existing tracklets.
        for tr in sorted(self.tracklets, key=lambda t: (-len(t.evidence), -mean(e.identity for e in t.evidence))):
            if not remaining:
                break
            px, py = tr.predicted_xy()
            best_idx = min(remaining, key=lambda i: hypot(evidences[i].x - px, evidences[i].y - py))
            ev = evidences[best_idx]
            dist = hypot(ev.x - px, ev.y - py)
            if dist <= cfg.max_link_distance_px:
                tr.evidence.append(ev)
                tr.last_seen_frame = frame_index
                # bounded memory
                if len(tr.evidence) > cfg.window_frames + 3:
                    del tr.evidence[:-(cfg.window_frames + 3)]
                remaining.remove(best_idx)
        # Spawn strongest unassociated evidence until capacity is reached.
        for idx in sorted(remaining, key=lambda i: (-evidences[i].identity, -evidences[i].total_score)):
            if len(self.tracklets) >= cfg.max_tracklets:
                break
            self._spawn(evidences[idx])

    def _background_attachment_fraction(self, recent: list[TrackletEvidence]) -> float:
        cfg = self.config
        if not cfg.background_attachment_enabled or len(recent) < 2:
            return 0.0
        tested = attached = 0
        for prev, cur in zip(recent, recent[1:]):
            if not cur.background_valid:
                continue
            bg_mag = hypot(cur.background_dx_px, cur.background_dy_px)
            if bg_mag < cfg.background_motion_min_px:
                continue
            tested += 1
            dx = cur.x - prev.x
            dy = cur.y - prev.y
            residual = hypot(dx - cur.background_dx_px, dy - cur.background_dy_px)
            if residual <= cfg.background_attachment_tolerance_px:
                attached += 1
        return float(attached / tested) if tested else 0.0

    def best_confirmed(self, frame_index: int) -> Tracklet | None:
        cfg = self.config
        eligible: list[tuple[float, Tracklet]] = []
        for tr in self.tracklets:
            recent = tr.recent(cfg.window_frames, frame_index)
            if not recent:
                continue
            is_far = any(e.far_jump for e in recent)
            is_far_scale = any(e.far_scale for e in recent)
            is_micro = sum(1 for e in recent if e.is_micro) >= max(1, len(recent) // 2)
            required_hits = cfg.far_jump_confirm_hits if is_far else (cfg.micro_confirm_hits if is_micro else cfg.confirm_hits)
            if is_far_scale:
                scale_hits = (
                    cfg.scale_distance_extreme_confirm_hits
                    if max(float(e.scale_log_distance) for e in recent) >= cfg.scale_distance_extreme_log_distance
                    else cfg.scale_distance_confirm_hits
                )
                required_hits = max(required_hits, scale_hits)
            if len(recent) < required_hits:
                continue
            # HOTFIX-1: aggregate evidence robustly. Projection/thermal scores can
            # fluctuate strongly frame-to-frame; a good temporal sequence should
            # not be rejected because one frame narrowly misses a hard threshold.
            mi = float(median(e.identity for e in recent))
            ms = float(mean(e.total_score for e in recent))
            measured_gaps = [float(e.gap) for e in recent if bool(e.has_competitor)]
            # Absence of a runner-up is neutral: it is neither evidence of a
            # huge confidence gap nor a reason to reject an otherwise coherent
            # temporal tracklet.
            mg = float(median(measured_gaps)) if measured_gaps else None
            mt = float(median(e.thermal_profile_score for e in recent))
            areas = [max(1.0, float(getattr(e.candidate, "area_px", 1.0))) for e in recent]
            scale_span = max(log(a) for a in areas) - min(log(a) for a in areas)
            min_i = cfg.far_jump_min_mean_identity if is_far else (cfg.micro_min_mean_identity if is_micro else cfg.min_mean_identity)
            min_g = cfg.far_jump_min_mean_gap if is_far else cfg.min_mean_gap
            if mi < min_i or ms < cfg.min_mean_score or (mg is not None and mg < min_g):
                continue
            if scale_span > cfg.max_scale_log_span:
                continue
            if is_far_scale:
                latest = recent[-1]
                if (not bool(latest.scale_quick_confirmable)
                        or float(latest.scale_excess_log_distance) > cfg.scale_distance_max_confirm_excess_log):
                    continue
            if (is_far and any(float(e.thermal_profile_score) > 0.0 for e in recent)
                    and mt < cfg.far_jump_min_median_thermal_profile):
                continue
            attached_fraction = self._background_attachment_fraction(recent)
            # A candidate that simply follows a moving background is not enough
            # evidence to declare a new target identity. Keep it tentative.
            if is_far and attached_fraction >= cfg.background_attachment_max_fraction:
                continue
            quality = 0.48 * mi + 0.22 * ms + 0.18 * mt + 0.12 * min(1.0, len(recent) / max(1, required_hits))
            if is_far:
                # Preserve a weak relation to the last reliable target. This is a
                # ranking penalty, not a hard motion veto.
                anchor_med = float(median(e.anchor_distance_px for e in recent))
                quality -= min(0.12, 0.00035 * anchor_med)
            if is_far_scale:
                # Scale distance is already a nonlinear candidate gate; this
                # small aggregate penalty only breaks ties between tracklets.
                scale_excess = float(median(e.scale_excess_log_distance for e in recent))
                quality -= min(0.10, 0.06 * scale_excess)
            eligible.append((quality, tr))
        if not eligible:
            return None
        eligible.sort(key=lambda p: (-p[0], p[1].tracklet_id))
        return eligible[0][1]

    def best_active(self, frame_index: int) -> Tracklet | None:
        if not self.tracklets:
            return None
        def quality(tr: Tracklet) -> tuple[float, int]:
            recent = tr.recent(self.config.window_frames, frame_index)
            if not recent:
                return -1.0, tr.tracklet_id
            return mean(e.identity for e in recent), -tr.tracklet_id
        return max(self.tracklets, key=quality)
