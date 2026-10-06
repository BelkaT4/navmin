"""VT-10 HOTFIX-5 adaptive memory.

Compatibility mode ``ema`` preserves the VT-9 cautious one-layer EMA.
New mode ``adaptive_memory`` uses four separated memories:

* immutable BaseBank;
* slow EMA for long-term appearance;
* fast EMA for short bursts of appearance drift;
* RecentBank FIFO containing the latest spatially trustworthy appearances.

Fast adaptation can continue when identity is only moderate, but only when
motion, prediction error and candidate margin strongly support continuity.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping

import numpy as np

from navmin.vision.processors.vt11.identity.descriptor import TargetDescriptor, descriptor_similarity
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank


@dataclass(frozen=True)
class EMAConfig:
    mode: str = "off"  # off | ema | adaptive_memory
    alpha: float = 0.05  # slow EMA / legacy EMA
    min_confidence: float = 0.68
    min_identity_similarity: float = 0.48
    min_confidence_gap: float = 0.10
    max_prediction_error_px: float = 12.0
    warmup_accepted_frames: int = 1
    update_interval_frames: int = 2
    max_updates: int = 10000

    # HOTFIX-5A: fast memory and RecentBank.
    fast_alpha: float = 0.25
    recent_bank_size: int = 5
    recent_update_interval_frames: int = 1
    fast_min_confidence: float = 0.52
    fast_min_identity_similarity: float = 0.30
    fast_min_confidence_gap: float = 0.08
    fast_min_motion_score: float = 0.88
    fast_max_prediction_error_px: float = 18.0
    fast_min_size_score: float = 0.04
    fast_burst_frames: int = 6
    drift_score_trigger: float = 0.34
    drift_similarity_trigger: float = 0.58
    drift_size_ratio_trigger: float = 1.35

    # VT-11 profile-dependent per-feature adaptation rates.  The scalar alpha
    # remains the temporal rate; these multipliers let THERMAL adapt thermal
    # statistics faster while keeping fine shape memory more conservative.
    shape_alpha_multiplier: float = 1.0
    thermal_alpha_multiplier: float = 1.0
    color_alpha_multiplier: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"off", "ema", "adaptive_memory"}:
            raise ValueError("adaptation mode must be 'off', 'ema' or 'adaptive_memory'")
        for name in ("alpha", "fast_alpha"):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        for name in (
            "min_confidence", "min_identity_similarity", "min_confidence_gap",
            "fast_min_confidence", "fast_min_identity_similarity", "fast_min_confidence_gap",
            "fast_min_motion_score", "fast_min_size_score", "drift_score_trigger",
            "drift_similarity_trigger",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.max_prediction_error_px <= 0 or self.fast_max_prediction_error_px <= 0:
            raise ValueError("prediction-error limits must be > 0")
        if self.warmup_accepted_frames < 0:
            raise ValueError("warmup_accepted_frames must be >= 0")
        if self.update_interval_frames < 1 or self.recent_update_interval_frames < 1:
            raise ValueError("update intervals must be >= 1")
        if self.max_updates < 1 or self.recent_bank_size < 1 or self.fast_burst_frames < 1:
            raise ValueError("memory sizes/counts must be >= 1")
        if self.drift_size_ratio_trigger <= 1.0:
            raise ValueError("drift_size_ratio_trigger must be > 1")
        for name in ("shape_alpha_multiplier", "thermal_alpha_multiplier", "color_alpha_multiplier"):
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class EMAUpdateEvent:
    frame_index: int
    updated: bool
    reason: str
    alpha: float
    confidence: float
    identity_similarity: float
    confidence_gap: float
    prediction_error_px: float | None
    similarity_before: float
    update_count: int
    layer: str = "NONE"  # NONE | SLOW | FAST | SLOW+FAST | RECENT | ...
    drift_score: float = 0.0
    drift_reason: str = ""
    recent_count: int = 0
    fast_remaining: int = 0
    slow_update_count: int = 0
    fast_update_count: int = 0
    recent_update_count: int = 0


def _blend_array(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(old, dtype=np.float32)
    b = np.asarray(new, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("descriptor arrays must have equal shapes for EMA")
    return ((1.0 - alpha) * a + alpha * b).astype(np.float32)


def ema_blend_descriptor(
    old: TargetDescriptor,
    new: TargetDescriptor,
    alpha: float,
    *,
    shape_alpha_multiplier: float = 1.0,
    thermal_alpha_multiplier: float = 1.0,
    color_alpha_multiplier: float = 1.0,
) -> TargetDescriptor:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0,1]")
    shape_alpha = min(1.0, max(1e-6, alpha * float(shape_alpha_multiplier)))
    thermal_alpha = min(1.0, max(1e-6, alpha * float(thermal_alpha_multiplier)))
    color_alpha = min(1.0, max(1e-6, alpha * float(color_alpha_multiplier)))
    if old.color_hist is None:
        color_hist = None if new.color_hist is None else np.asarray(new.color_hist, np.float32).copy()
    elif new.color_hist is None:
        color_hist = np.asarray(old.color_hist, np.float32).copy()
    else:
        color_hist = _blend_array(old.color_hist, new.color_hist, color_alpha)
        total = float(color_hist.sum())
        if total > 0:
            color_hist = (color_hist / total).astype(np.float32)
    return TargetDescriptor(
        appearance=_blend_array(old.appearance, new.appearance, shape_alpha),
        projection_x=_blend_array(old.projection_x, new.projection_x, shape_alpha),
        projection_y=_blend_array(old.projection_y, new.projection_y, shape_alpha),
        edge_map=_blend_array(old.edge_map, new.edge_map, shape_alpha),
        geometry=_blend_array(old.geometry, new.geometry, shape_alpha),
        size_px=float((1.0 - shape_alpha) * old.size_px + shape_alpha * new.size_px),
        bbox=tuple(int(v) for v in new.bbox),
        thermal_contrast=float((1.0 - thermal_alpha) * old.thermal_contrast + thermal_alpha * new.thermal_contrast),
        hot_area=float((1.0 - thermal_alpha) * old.hot_area + thermal_alpha * new.hot_area),
        color_hist=color_hist,
        peak_prominence=float((1.0 - thermal_alpha) * old.peak_prominence + thermal_alpha * new.peak_prominence),
        hot_core_compactness=float((1.0 - thermal_alpha) * old.hot_core_compactness + thermal_alpha * new.hot_core_compactness),
        radial_hotspot_profile=(
            None if old.radial_hotspot_profile is None and new.radial_hotspot_profile is None
            else (np.asarray(new.radial_hotspot_profile, np.float32).copy() if old.radial_hotspot_profile is None
                  else np.asarray(old.radial_hotspot_profile, np.float32).copy() if new.radial_hotspot_profile is None
                  else _blend_array(old.radial_hotspot_profile, new.radial_hotspot_profile, thermal_alpha))
        ),
        energy_concentration=float((1.0 - thermal_alpha) * old.energy_concentration + thermal_alpha * new.energy_concentration),
        thermal_background_clutter=float(new.thermal_background_clutter),
        thermal_2d_ratio=float((1.0 - shape_alpha) * old.thermal_2d_ratio + shape_alpha * new.thermal_2d_ratio),
    )


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


class EMAReferenceAdapter:
    """Stateful adapter; name retained for backward compatibility."""

    def __init__(
        self,
        bank: ReferenceBank,
        config: EMAConfig,
        *,
        identity_weights: Mapping[str, float] | None = None,
    ) -> None:
        config.validate()
        self.bank = bank
        self.config = config
        self.identity_weights = identity_weights
        self.bank.recent_max_size = int(config.recent_bank_size)
        self.accepted_streak = 0
        self.frames_since_update = 0
        self.frames_since_recent = 0
        self.update_count = 0
        self.slow_update_count = 0
        self.fast_update_count = 0
        self.recent_update_count = 0
        self.fast_remaining = 0
        self.events: list[EMAUpdateEvent] = []

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    @property
    def adaptive_memory_enabled(self) -> bool:
        return self.config.mode == "adaptive_memory"

    def reset_streak(self) -> None:
        self.accepted_streak = 0
        self.frames_since_update = 0
        self.frames_since_recent = 0
        self.fast_remaining = 0

    def _blend(self, old: TargetDescriptor, new: TargetDescriptor, alpha: float) -> TargetDescriptor:
        cfg = self.config
        return ema_blend_descriptor(
            old, new, alpha,
            shape_alpha_multiplier=cfg.shape_alpha_multiplier,
            thermal_alpha_multiplier=cfg.thermal_alpha_multiplier,
            color_alpha_multiplier=cfg.color_alpha_multiplier,
        )

    def _event(
        self, *, frame_index: int, updated: bool, reason: str, alpha: float,
        confidence: float, identity_similarity: float, confidence_gap: float,
        prediction_error_px: float | None, similarity_before: float = 0.0,
        layer: str = "NONE", drift_score: float = 0.0, drift_reason: str = "",
    ) -> EMAUpdateEvent:
        event = EMAUpdateEvent(
            frame_index=int(frame_index), updated=bool(updated), reason=str(reason), alpha=float(alpha),
            confidence=float(confidence), identity_similarity=float(identity_similarity),
            confidence_gap=float(confidence_gap), prediction_error_px=prediction_error_px,
            similarity_before=float(similarity_before), update_count=int(self.update_count),
            layer=str(layer), drift_score=float(drift_score), drift_reason=str(drift_reason),
            recent_count=int(self.bank.recent_count), fast_remaining=int(self.fast_remaining),
            slow_update_count=int(self.slow_update_count), fast_update_count=int(self.fast_update_count),
            recent_update_count=int(self.recent_update_count),
        )
        self.events.append(event)
        return event

    def blocked_event(self, frame_index: int, reason: str) -> EMAUpdateEvent:
        self.accepted_streak = 0
        self.frames_since_update = 0
        self.frames_since_recent = 0
        if reason in {"BLOCKED_REACQ", "BLOCKED_NOT_ACCEPTED"}:
            self.fast_remaining = 0
        return self._event(
            frame_index=frame_index, updated=False, reason=reason, alpha=self.config.alpha,
            confidence=0.0, identity_similarity=0.0, confidence_gap=0.0,
            prediction_error_px=None,
        )

    def _drift_metrics(self, descriptor: TargetDescriptor) -> tuple[float, str, float]:
        ref, sim = self.bank.best_long_term_match(descriptor, identity_weights=self.identity_weights)
        ratio = max(
            float(descriptor.size_px) / max(float(ref.size_px), 1.0),
            float(ref.size_px) / max(float(descriptor.size_px), 1.0),
        )
        size_cue = _clamp01(log(max(1.0, ratio)) / log(2.5))
        app_cue = 1.0 - float(sim.appearance)
        edge_cue = 1.0 - float(sim.edge)
        proj_cue = 1.0 - float(sim.projection)
        thermal_cue = max(
            1.0 - float(sim.thermal_contrast), 1.0 - float(sim.hot_area),
            1.0 - float(getattr(sim, "peak_prominence", 0.0)),
            1.0 - float(getattr(sim, "hot_core_compactness", 0.0)),
            1.0 - float(getattr(sim, "radial_hotspot_profile", 0.0)),
            1.0 - float(getattr(sim, "energy_concentration", 0.0)),
            1.0 - float(getattr(sim, "thermal_2d_ratio", 0.0)),
        )
        identity_cue = 1.0 - float(sim.total)
        thermal_weighted = any(float((self.identity_weights or {}).get(k, 0.0)) > 0 for k in (
            "thermal_contrast", "hot_area", "peak_prominence", "hot_core_compactness",
            "radial_hotspot_profile", "energy_concentration", "thermal_2d_ratio"))
        if thermal_weighted:
            drift_score = 0.30 * identity_cue + 0.20 * size_cue + 0.15 * app_cue + 0.10 * edge_cue + 0.25 * thermal_cue
        else:
            drift_score = 0.35 * identity_cue + 0.25 * size_cue + 0.20 * app_cue + 0.10 * edge_cue + 0.10 * proj_cue
        reasons: list[str] = []
        if sim.total < self.config.drift_similarity_trigger:
            reasons.append("LONG_TERM_SIM")
        if ratio >= self.config.drift_size_ratio_trigger:
            reasons.append("SIZE")
        if drift_score >= self.config.drift_score_trigger:
            reasons.append("FEATURE_DRIFT")
        return _clamp01(drift_score), "+".join(reasons), float(sim.total)

    def consider(
        self,
        *,
        frame_index: int,
        descriptor: TargetDescriptor,
        confidence: float,
        identity_similarity: float,
        confidence_gap: float,
        prediction_error_px: float | None,
        motion_score: float = 1.0,
        size_score: float = 1.0,
        force_fast: bool = False,
        phase: str = "TRACK",
    ) -> EMAUpdateEvent:
        cfg = self.config
        if not self.enabled:
            return self._event(
                frame_index=frame_index, updated=False, reason="ADAPT_OFF", alpha=cfg.alpha,
                confidence=confidence, identity_similarity=identity_similarity,
                confidence_gap=confidence_gap, prediction_error_px=prediction_error_px,
            )

        # Legacy VT-9 mode remains intentionally unchanged in behavior.
        if cfg.mode == "ema":
            gate_reasons: list[str] = []
            if confidence < cfg.min_confidence: gate_reasons.append("CONFIDENCE")
            if identity_similarity < cfg.min_identity_similarity: gate_reasons.append("IDENTITY")
            if confidence_gap < cfg.min_confidence_gap: gate_reasons.append("GAP")
            if prediction_error_px is None or prediction_error_px > cfg.max_prediction_error_px: gate_reasons.append("PREDICTION_ERROR")
            if self.update_count >= cfg.max_updates: gate_reasons.append("MAX_UPDATES")
            if gate_reasons:
                self.accepted_streak = 0
                self.frames_since_update = 0
                return self._event(
                    frame_index=frame_index, updated=False, reason="BLOCKED_" + "+".join(gate_reasons), alpha=cfg.alpha,
                    confidence=confidence, identity_similarity=identity_similarity, confidence_gap=confidence_gap,
                    prediction_error_px=prediction_error_px,
                )
            self.accepted_streak += 1
            self.frames_since_update += 1
            if self.accepted_streak <= cfg.warmup_accepted_frames:
                return self._event(frame_index=frame_index, updated=False, reason="WARMUP", alpha=cfg.alpha,
                    confidence=confidence, identity_similarity=identity_similarity, confidence_gap=confidence_gap,
                    prediction_error_px=prediction_error_px)
            if self.frames_since_update < cfg.update_interval_frames:
                return self._event(frame_index=frame_index, updated=False, reason="INTERVAL", alpha=cfg.alpha,
                    confidence=confidence, identity_similarity=identity_similarity, confidence_gap=confidence_gap,
                    prediction_error_px=prediction_error_px)
            old = self.bank.adaptive_descriptor
            if old is None:
                old, _ = self.bank.best_base_match(descriptor, identity_weights=self.identity_weights)
            sim_before = descriptor_similarity(old, descriptor, identity_weights=self.identity_weights).total
            self.bank.set_adaptive(self._blend(old, descriptor, cfg.alpha))
            self.update_count += 1; self.slow_update_count += 1; self.frames_since_update = 0
            return self._event(frame_index=frame_index, updated=True, reason="UPDATED", alpha=cfg.alpha,
                confidence=confidence, identity_similarity=identity_similarity, confidence_gap=confidence_gap,
                prediction_error_px=prediction_error_px, similarity_before=sim_before, layer="SLOW")

        # HOTFIX-5 adaptive memory.
        drift_score, drift_reason, long_sim = self._drift_metrics(descriptor)
        pred_ok_fast = prediction_error_px is not None and prediction_error_px <= cfg.fast_max_prediction_error_px
        spatial_reliable = (
            motion_score >= cfg.fast_min_motion_score
            and pred_ok_fast
            and confidence_gap >= cfg.fast_min_confidence_gap
            and size_score >= cfg.fast_min_size_score
            and identity_similarity >= cfg.fast_min_identity_similarity
            and confidence >= cfg.fast_min_confidence
        )
        drift_now = force_fast or bool(drift_reason)
        if drift_now and spatial_reliable:
            self.fast_remaining = max(self.fast_remaining, cfg.fast_burst_frames)

        layers: list[str] = []
        reasons: list[str] = []
        sim_before = long_sim
        updated = False
        chosen_alpha = cfg.alpha

        # RecentBank follows spatially trustworthy accepted states quickly.
        self.frames_since_recent += 1
        recent_ok = (
            spatial_reliable
            and self.frames_since_recent >= cfg.recent_update_interval_frames
        )
        if recent_ok:
            self.bank.add_recent(descriptor, max_size=cfg.recent_bank_size)
            self.recent_update_count += 1
            self.frames_since_recent = 0
            updated = True
            layers.append("RECENT")

        # Fast EMA is active for a short burst after drift is detected/rescued.
        fast_ok = spatial_reliable and self.fast_remaining > 0 and self.update_count < cfg.max_updates
        if fast_ok:
            old_fast = self.bank.fast_descriptor
            if old_fast is None:
                old_fast, _ = self.bank.best_long_term_match(descriptor, identity_weights=self.identity_weights)
            self.bank.set_fast(self._blend(old_fast, descriptor, cfg.fast_alpha))
            self.fast_update_count += 1
            self.update_count += 1
            self.fast_remaining = max(0, self.fast_remaining - 1)
            updated = True
            chosen_alpha = cfg.fast_alpha
            layers.append("FAST")
            reasons.append("DRIFT" if drift_now else "FAST_BURST")

        # Slow EMA remains conservative and creates long-term memory.
        slow_gate: list[str] = []
        if confidence < cfg.min_confidence: slow_gate.append("CONFIDENCE")
        if identity_similarity < cfg.min_identity_similarity: slow_gate.append("IDENTITY")
        if confidence_gap < cfg.min_confidence_gap: slow_gate.append("GAP")
        if prediction_error_px is None or prediction_error_px > cfg.max_prediction_error_px: slow_gate.append("PREDICTION_ERROR")
        if self.update_count >= cfg.max_updates: slow_gate.append("MAX_UPDATES")
        if not slow_gate:
            self.accepted_streak += 1
            self.frames_since_update += 1
            if self.accepted_streak > cfg.warmup_accepted_frames and self.frames_since_update >= cfg.update_interval_frames:
                old_slow = self.bank.adaptive_descriptor
                if old_slow is None:
                    old_slow, _ = self.bank.best_base_match(descriptor, identity_weights=self.identity_weights)
                self.bank.set_adaptive(self._blend(old_slow, descriptor, cfg.alpha))
                self.slow_update_count += 1
                self.update_count += 1
                self.frames_since_update = 0
                updated = True
                layers.append("SLOW")
            else:
                reasons.append("SLOW_WARMUP_OR_INTERVAL")
        else:
            self.accepted_streak = 0
            self.frames_since_update = 0
            reasons.append("SLOW_BLOCKED_" + "+".join(slow_gate))

        if phase == "TRACK_WARMUP" and recent_ok:
            reasons.append("WARMUP_RECENT")
        if drift_reason:
            reasons.append(drift_reason)
        if not updated and not reasons:
            reasons.append("NO_UPDATE")
        return self._event(
            frame_index=frame_index, updated=updated, reason=";".join(reasons), alpha=chosen_alpha,
            confidence=confidence, identity_similarity=identity_similarity, confidence_gap=confidence_gap,
            prediction_error_px=prediction_error_px, similarity_before=sim_before,
            layer="+".join(layers) if layers else "NONE", drift_score=drift_score, drift_reason=drift_reason,
        )
