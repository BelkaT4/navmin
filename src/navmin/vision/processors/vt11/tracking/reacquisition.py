"""LOST/REACQ with HOTFIX-6 robust motion-break recovery.

Normal REACQ keeps the HOTFIX-3 N-of-M confirmation and old motion prior.
When TRACK reports MOTION_BREAK, REACQ switches to a visual/self-consistent
mode: stale pre-break motion cannot veto a strong candidate, low-resolution
background motion updates the search anchor, and Top-K candidates are linked
into a few lightweight tracklets over 3-5 frames.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from math import hypot, sqrt
from statistics import mean
from time import perf_counter

import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    Candidate,
    CandidateDetectorConfig,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import (
    ThermalProfile,
    preprocess_frame,
    runtime_detector_config,
)
from navmin.vision.processors.vt11.tracking.background_motion import (
    BackgroundMotionConfig,
    estimate_background_motion,
    transform_point,
)
from navmin.vision.processors.vt11.tracking.frame_reliability import (
    FrameReliabilityConfig,
    estimate_frame_reliability,
)
from navmin.vision.processors.vt11.tracking.live_events import (
    LiveFrameEvent,
    LiveProgressCallback,
    StopRequested,
)
from navmin.vision.processors.vt11.tracking.motion_model import (
    MotionPoint,
    predict_position_timed,
)
from navmin.vision.processors.vt11.tracking.scale_continuity import (
    ReliableScaleAnchor,
    ScaleContinuityConfig,
)
from navmin.vision.processors.vt11.tracking.scale_model import (
    ScaleStabilityConfig,
    ScaleStabilityModel,
)
from navmin.vision.processors.vt11.tracking.tracker import (
    TrackingConfig,
    _diagnostic_confidence,
    _gate_reasons,
    _rank_candidates,
)
from navmin.vision.processors.vt11.tracking.tracklets import (
    TrackletConfig,
    TrackletEvidence,
    TrackletManager,
)

FrameReader = Callable[[int], tuple[int, np.ndarray]]


@dataclass(frozen=True)
class ReacquisitionConfig:
    top_k: int = 8
    initial_radius_px: int = 140
    max_radius_px: int = 360
    radius_growth_px_per_frame: float = 18.0
    uncertainty_radius_gain: float = 2.0
    speed_radius_gain: float = 0.75
    max_frames: int = 90

    confirm_frames: int = 3
    confirm_window_frames: int = 5
    max_confirmation_error_px: float = 65.0
    expected_area_update_alpha: float = 0.15

    identity_weight: float = 0.70
    motion_weight: float = 0.20
    size_weight: float = 0.10
    motion_sigma_px: float = 70.0
    velocity_window_frames: int = 5

    min_score: float = 0.50
    min_identity_similarity: float = 0.46
    min_confidence_gap: float = 0.025
    confidence_gap_reference: float = 0.12

    # HOTFIX-6: visual REACQ after a declared motion break.
    robust_motion_break_enabled: bool = False
    broken_identity_weight: float = 0.85
    broken_motion_weight: float = 0.05
    broken_size_weight: float = 0.10
    broken_motion_sigma_px: float = 180.0
    broken_min_score: float = 0.48
    broken_min_identity_similarity: float = 0.46
    broken_min_confidence_gap: float = 0.02
    tracklets_enabled: bool = False
    tracklet_top_k: int = 3
    tracklet_window_frames: int = 5
    tracklet_confirm_hits: int = 3
    tracklet_max_link_distance_px: float = 110.0
    full_frame_after_frames: int = 10
    full_frame_thermal_only: bool = True

    # HOTFIX-6 REV1: automatically promote normal REACQ to robust mode when
    # repeated visual evidence is strong but the stale motion prior disagrees.
    auto_promote_enabled: bool = True
    auto_promote_window_frames: int = 3
    auto_promote_hits_required: int = 2
    auto_promote_min_identity: float = 0.70
    auto_promote_min_gap: float = 0.06
    auto_promote_min_size_score: float = 0.04
    auto_promote_max_motion_score: float = 0.35
    auto_promote_max_static_overlay_penalty: float = 0.45

    # HOTFIX-6 REV2: IDENTITY-FIRST SAFE REACQ. The normal case is local
    # continuation of the same target; large whole-frame jumps are treated as
    # exceptional and require stronger evidence.
    identity_first_enabled: bool = True
    micro_detector_enabled: bool = True
    micro_min_thermal_contrast: float = 0.50
    micro_min_hot_area: float = 0.40
    far_jump_distance_px: float = 100.0
    far_jump_min_identity: float = 0.88
    far_jump_min_gap: float = 0.10
    far_jump_confirm_hits: int = 5
    far_jump_max_static_overlay_penalty: float = 0.35
    # HOTFIX-4: far distance is relative to elapsed time since the last reliable
    # observation, not a fixed radius forever.
    anchor_time_normalization_enabled: bool = False
    anchor_time_speed_multiplier: float = 1.50
    anchor_time_min_speed_px_per_frame: float = 3.0
    anchor_time_max_speed_px_per_frame: float = 22.0
    anchor_time_max_growth_px: float = 360.0
    anchor_time_uncertainty_gain: float = 1.0

    # HOTFIX-4 generic frame reliability during REACQ. Degraded frames neither
    # confirm nor destroy an otherwise coherent temporal hypothesis.
    thermal_frame_reliability_enabled: bool = False
    frame_reliability_max_dimension_px: int = 128
    frame_reliability_residual_scale: float = 0.75
    frame_reliability_phase_response_floor: float = 0.02
    frame_reliability_phase_response_good: float = 0.50
    frame_reliability_phase_weight: float = 0.15
    frame_reliability_enter_threshold: float = 0.55
    frame_reliability_exit_threshold: float = 0.72
    frame_reliability_max_hold_frames: int = 5

    size_update_min_identity: float = 0.60
    size_update_min_gap: float = 0.06
    size_update_max_step_ratio: float = 1.25
    global_emergency_after_frames: int = 15
    global_search_interval_frames: int = 3
    global_diversity_min_distance_px: float = 80.0

    # HOTFIX-1: separate permissive admission to a temporal tracklet from the
    # much stricter aggregate confirmation of that tracklet.
    tracklet_admission_min_identity: float = 0.55
    tracklet_admission_min_score: float = 0.44
    far_jump_admission_min_identity: float = 0.60
    far_jump_admission_min_gap: float = 0.0
    far_jump_admission_max_static_overlay_penalty: float = 0.45
    far_jump_min_thermal_profile: float = 0.68
    tracklet_max_scale_log_span: float = 0.65

    # THERMAL robust scale model mirrors TRACK and is shared by session when possible.
    thermal_scale_stability_enabled: bool = False
    scale_confirm_frames: int = 4
    scale_consistency_log_span: float = 0.35
    scale_max_log_change_per_frame: float = 0.025
    scale_large_change_ratio: float = 1.8
    scale_large_change_confirm_frames: int = 6
    scale_micro_max_area_px: float = 80.0
    scale_micro_confirm_frames: int = 3
    scale_micro_exit_confirm_frames: int = 4
    scale_micro_consistency_log_span: float = 0.55
    scale_micro_max_log_change_per_frame: float = 0.12

    # HOTFIX-4 REV3: nonlinear distance in scale-parameter space relative to
    # the session-wide last trusted area.  This is intentionally independent
    # of the ordinary soft ``size_score``.
    thermal_scale_parameter_distance_enabled: bool = False
    scale_anchor_base_allowed_log_distance: float = 0.85
    scale_anchor_growth_per_frame: float = 0.045
    scale_anchor_max_allowed_log_distance: float = 1.60
    scale_nonlinear_gate_strength: float = 1.50
    scale_quarantine_confirm_hits: int = 3
    scale_quarantine_extreme_confirm_hits: int = 5
    scale_quarantine_extreme_log_distance: float = 2.30
    scale_quarantine_max_confirm_excess_log: float = 1.10
    scale_quarantine_link_log_span: float = 0.45
    scale_reliable_update_alpha: float = 0.20

    # Event-driven background motion during broken REACQ only.
    global_shift_enabled: bool = False
    global_shift_downscale: float = 0.50
    global_shift_max_dimension_px: int = 320
    global_shift_phase_min_response: float = 0.08
    global_shift_phase_direct_max_fraction: float = 0.30
    global_shift_phase_max_change_from_previous_px: float = 90.0
    global_shift_feature_fallback: bool = True
    performance_target_fps: float = 25.0

    def validate(self) -> None:
        if self.top_k < 2:
            raise ValueError("top_k must be >= 2")
        if self.initial_radius_px < 5:
            raise ValueError("initial_radius_px must be >= 5")
        if self.max_radius_px < self.initial_radius_px:
            raise ValueError("max_radius_px must be >= initial_radius_px")
        if min(self.radius_growth_px_per_frame, self.uncertainty_radius_gain, self.speed_radius_gain) < 0:
            raise ValueError("REACQ radius gains must be >= 0")
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if self.confirm_frames < 2 or self.confirm_window_frames < self.confirm_frames:
            raise ValueError("confirm_frames/window invalid")
        if self.max_confirmation_error_px <= 0:
            raise ValueError("max_confirmation_error_px must be > 0")
        if not 0.0 < self.expected_area_update_alpha <= 1.0:
            raise ValueError("expected_area_update_alpha must be in (0,1]")
        if min(self.identity_weight, self.motion_weight, self.size_weight) < 0:
            raise ValueError("REACQ weights must be >= 0")
        if self.identity_weight + self.motion_weight + self.size_weight <= 0:
            raise ValueError("at least one REACQ weight must be > 0")
        if self.motion_sigma_px <= 0 or self.velocity_window_frames < 2:
            raise ValueError("motion settings invalid")
        for name in ("min_score", "min_identity_similarity", "min_confidence_gap", "broken_min_score", "broken_min_identity_similarity", "broken_min_confidence_gap"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.confidence_gap_reference <= 0:
            raise ValueError("confidence_gap_reference must be > 0")
        if min(self.broken_identity_weight, self.broken_motion_weight, self.broken_size_weight) < 0:
            raise ValueError("broken weights must be >=0")
        if self.broken_identity_weight + self.broken_motion_weight + self.broken_size_weight <= 0:
            raise ValueError("broken weights invalid")
        if self.broken_motion_sigma_px <= 0:
            raise ValueError("broken_motion_sigma_px must be >0")
        if self.tracklet_top_k < 1 or self.tracklet_window_frames < self.tracklet_confirm_hits or self.tracklet_confirm_hits < 2:
            raise ValueError("tracklet settings invalid")
        if self.tracklet_max_link_distance_px <= 0 or self.full_frame_after_frames < 0:
            raise ValueError("tracklet/full-frame settings invalid")
        if self.auto_promote_window_frames < 2 or self.auto_promote_hits_required < 1 or self.auto_promote_hits_required > self.auto_promote_window_frames:
            raise ValueError("auto-promote window/hits invalid")
        for name in ("auto_promote_min_identity", "auto_promote_min_gap", "auto_promote_min_size_score", "auto_promote_max_motion_score", "auto_promote_max_static_overlay_penalty"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("micro_min_thermal_contrast", "micro_min_hot_area", "far_jump_min_identity", "far_jump_min_gap", "far_jump_max_static_overlay_penalty", "size_update_min_identity", "size_update_min_gap"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.far_jump_distance_px <= 0 or self.far_jump_confirm_hits < self.tracklet_confirm_hits:
            raise ValueError("far-jump settings invalid")
        if self.anchor_time_speed_multiplier < 0 or self.anchor_time_min_speed_px_per_frame < 0:
            raise ValueError("anchor time-normalization speeds must be >=0")
        if self.anchor_time_max_speed_px_per_frame < self.anchor_time_min_speed_px_per_frame or self.anchor_time_max_growth_px < 0 or self.anchor_time_uncertainty_gain < 0:
            raise ValueError("anchor time-normalization limits invalid")
        FrameReliabilityConfig(
            enabled=self.thermal_frame_reliability_enabled,
            max_dimension_px=self.frame_reliability_max_dimension_px,
            residual_scale=self.frame_reliability_residual_scale,
            phase_response_floor=self.frame_reliability_phase_response_floor,
            phase_response_good=self.frame_reliability_phase_response_good,
            phase_weight=self.frame_reliability_phase_weight,
            enter_threshold=self.frame_reliability_enter_threshold,
            exit_threshold=self.frame_reliability_exit_threshold,
            max_hold_frames=self.frame_reliability_max_hold_frames,
        ).validate()
        if self.size_update_max_step_ratio < 1.0:
            raise ValueError("size_update_max_step_ratio must be >=1")
        if self.global_emergency_after_frames < 0:
            raise ValueError("global_emergency_after_frames must be >=0")
        if self.global_search_interval_frames < 1 or self.global_diversity_min_distance_px < 0:
            raise ValueError("global emergency scan settings invalid")
        for name in ("tracklet_admission_min_identity", "tracklet_admission_min_score", "far_jump_admission_min_identity",
                     "far_jump_admission_min_gap", "far_jump_admission_max_static_overlay_penalty", "far_jump_min_thermal_profile"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.tracklet_max_scale_log_span <= 0:
            raise ValueError("tracklet_max_scale_log_span must be >0")
        ScaleStabilityConfig(
            enabled=self.thermal_scale_stability_enabled, confirm_frames=self.scale_confirm_frames,
            consistency_log_span=self.scale_consistency_log_span, max_log_change_per_frame=self.scale_max_log_change_per_frame,
            large_change_ratio=self.scale_large_change_ratio, large_change_confirm_frames=self.scale_large_change_confirm_frames,
            micro_max_area_px=self.scale_micro_max_area_px, micro_confirm_frames=self.scale_micro_confirm_frames,
            micro_exit_confirm_frames=self.scale_micro_exit_confirm_frames, micro_consistency_log_span=self.scale_micro_consistency_log_span,
            micro_max_log_change_per_frame=self.scale_micro_max_log_change_per_frame,
        ).validate()
        ScaleContinuityConfig(
            enabled=self.thermal_scale_parameter_distance_enabled,
            base_allowed_log_distance=self.scale_anchor_base_allowed_log_distance,
            growth_per_frame=self.scale_anchor_growth_per_frame,
            max_allowed_log_distance=self.scale_anchor_max_allowed_log_distance,
            nonlinear_strength=self.scale_nonlinear_gate_strength,
            quarantine_confirm_hits=self.scale_quarantine_confirm_hits,
            quarantine_extreme_confirm_hits=self.scale_quarantine_extreme_confirm_hits,
            quarantine_extreme_log_distance=self.scale_quarantine_extreme_log_distance,
            quarantine_max_confirm_excess_log=self.scale_quarantine_max_confirm_excess_log,
            quarantine_link_log_span=self.scale_quarantine_link_log_span,
            reliable_update_alpha=self.scale_reliable_update_alpha,
        ).validate()
        if not 0.10 <= self.global_shift_downscale <= 1.0 or self.global_shift_max_dimension_px < 64:
            raise ValueError("global shift resolution settings invalid")
        if not 0.0 <= self.global_shift_phase_min_response <= 1.0:
            raise ValueError("global_shift_phase_min_response invalid")
        if not 0.05 <= self.global_shift_phase_direct_max_fraction <= 1.0:
            raise ValueError("global_shift_phase_direct_max_fraction invalid")
        if self.global_shift_phase_max_change_from_previous_px <= 0:
            raise ValueError("global_shift_phase_max_change_from_previous_px invalid")
        if self.performance_target_fps <= 0:
            raise ValueError("performance_target_fps must be >0")


@dataclass(frozen=True)
class ReacqFrameResult:
    frame_index: int
    predicted_xy: tuple[float, float]
    search_radius_px: int
    selected: Candidate | None
    selected_score: float
    identity_similarity: float
    motion_score: float
    size_score: float
    confidence_gap: float
    confidence: float
    candidate_count: int
    confirmation_streak: int
    confirmation_window_size: int
    status: str
    diagnostic_reason: str
    prediction_error_px: float | None
    best_candidate_center: tuple[float, float] | None = None
    best_candidate_bbox: tuple[int, int, int, int] | None = None
    best_candidate_area_px: int = 0
    best_candidate_mean_response: float = 0.0
    best_candidate_peak_response: float = 0.0
    second_score: float = 0.0
    appearance_similarity: float = 0.0
    edge_similarity: float = 0.0
    projection_similarity: float = 0.0
    geometry_similarity: float = 0.0
    color_similarity: float = 0.0
    thermal_contrast_similarity: float = 0.0
    hot_area_similarity: float = 0.0
    peak_prominence_similarity: float = 0.0
    hot_core_compactness_similarity: float = 0.0
    radial_hotspot_profile_similarity: float = 0.0
    energy_concentration_similarity: float = 0.0
    thermal_2d_ratio_similarity: float = 0.0
    static_overlay_penalty: float = 0.0
    hud_probability: float = 0.0
    hud_hard_overlap: float = 0.0
    hud_soft_overlap: float = 0.0
    hud_hard_reject: bool = False
    hud_negative_penalty: float = 0.0
    detector_threshold: float = 0.0
    detector_median_response: float = 0.0
    detector_robust_sigma: float = 0.0
    detector_raw_components: int = 0
    detector_valid_components: int = 0
    expected_area_px: float = 0.0
    detector_effective_min_area_px: int = 0
    detector_effective_max_area_px: int = 0
    adaptation_updated: bool = False
    adaptation_reason: str = "BLOCKED_REACQ"
    adaptation_update_count: int = 0
    motion_broken_mode: bool = False
    auto_promoted_motion_break: bool = False
    promotion_evidence_hits: int = 0
    promotion_reason: str = ""
    tracklet_id: int = 0
    active_tracklets: int = 0
    reacq_mode: str = "LOCAL_CONSERVATIVE"
    candidate_channel: str = "NONE"
    anchor_distance_px: float = 0.0
    anchor_distance_norm: float = 0.0
    anchor_age_frames: int = 0
    anchor_allowed_distance_px: float = 0.0
    frame_reliability_score: float = 1.0
    frame_reliability_residual: float = 0.0
    frame_reliability_phase_response: float = 1.0
    frame_degraded_active: bool = False
    frame_degraded_frames: int = 0
    far_jump_candidate: bool = False
    micro_candidate_count: int = 0
    selected_tracklet_x: float | None = None
    selected_tracklet_y: float | None = None
    selected_tracklet_area_px: int = 0
    selected_tracklet_identity: float = 0.0
    selected_tracklet_score: float = 0.0
    selected_tracklet_gap: float = 0.0
    selected_tracklet_thermal_profile: float = 0.0
    scale_state: str = "LEGACY"
    scale_base_area_px: float = 0.0
    scale_current_area_px: float = 0.0
    scale_pending_count: int = 0
    scale_update_reason: str = ""
    reliable_area_anchor_px: float = 0.0
    scale_log_distance: float = 0.0
    scale_allowed_log_distance: float = 0.0
    scale_excess_log_distance: float = 0.0
    scale_nonlinear_gate: float = 1.0
    scale_anchor_age_frames: int = 0
    scale_quarantine_active: bool = False
    scale_quarantine_hits: int = 0
    scale_quarantine_required_hits: int = 0
    scale_quarantine_confirmed: bool = False
    global_shift_dx_px: float = 0.0
    global_shift_dy_px: float = 0.0
    global_shift_response: float = 0.0
    global_shift_method: str = "OFF"
    global_shift_time_ms: float = 0.0
    detector_time_ms: float = 0.0
    identity_time_ms: float = 0.0
    performance_budget_ms: float = 0.0
    over_budget: bool = False
    processing_time_ms: float = 0.0


@dataclass(frozen=True)
class ReacquisitionResult:
    status: str
    lost_frame: int
    frames: tuple[ReacqFrameResult, ...]
    confirmed_frame: int | None
    confirmed_center: tuple[float, float] | None
    t_reacq_frames: int | None
    confirmation_points: tuple[MotionPoint, ...]
    attempted_frames: int
    mean_confidence: float
    max_search_radius_used_px: int
    confirmed_area_px: float | None = None
    last_expected_area_px: float = 0.0
    motion_broken_mode: bool = False
    motion_relocked: bool = False
    mean_global_shift_time_ms: float = 0.0
    over_budget_frames: int = 0
    auto_promoted: bool = False
    promotion_frame: int | None = None


@dataclass(frozen=True)
class ReacquisitionContinuitySnapshot:
    """Explicit REACQ state needed at a REACQ -> TRACK boundary."""

    confirmation_points: tuple[MotionPoint, ...]
    confirmed_frame: int | None
    confirmed_center: tuple[float, float] | None
    confirmed_area_px: float | None
    expected_area_px: float
    motion_relocked: bool
    motion_broken: bool
    auto_promoted: bool
    scale_model: ScaleStabilityModel | None
    reliable_scale_anchor: ReliableScaleAnchor | None


@dataclass
class _ReacquisitionContinuityState:
    confirmation_points: tuple[MotionPoint, ...] = ()
    confirmed_frame: int | None = None
    confirmed_center: tuple[float, float] | None = None
    confirmed_area_px: float | None = None
    expected_area_px: float = 0.0
    motion_relocked: bool = False
    motion_broken: bool = False
    auto_promoted: bool = False
    scale_model: ScaleStabilityModel | None = None
    reliable_scale_anchor: ReliableScaleAnchor | None = None


def _radius_for_frame(*, cfg: ReacquisitionConfig, elapsed_frames: int, speed_px_per_frame: float, uncertainty_px: float) -> int:
    raw = (
        float(cfg.initial_radius_px) + cfg.radius_growth_px_per_frame * max(0, elapsed_frames)
        + cfg.speed_radius_gain * max(0.0, speed_px_per_frame)
        + cfg.uncertainty_radius_gain * max(0.0, uncertainty_px)
    )
    return int(round(min(float(cfg.max_radius_px), max(float(cfg.initial_radius_px), raw))))


def _tracking_cfg(cfg: ReacquisitionConfig, *, motion_broken: bool = False) -> TrackingConfig:
    if motion_broken:
        iw, mw, sw, sigma = cfg.broken_identity_weight, cfg.broken_motion_weight, cfg.broken_size_weight, cfg.broken_motion_sigma_px
        min_s, min_i, min_g = cfg.broken_min_score, cfg.broken_min_identity_similarity, cfg.broken_min_confidence_gap
    else:
        iw, mw, sw, sigma = cfg.identity_weight, cfg.motion_weight, cfg.size_weight, cfg.motion_sigma_px
        min_s, min_i, min_g = cfg.min_score, cfg.min_identity_similarity, cfg.min_confidence_gap
    return TrackingConfig(
        top_k=cfg.top_k, search_radius_px=cfg.initial_radius_px,
        identity_weight=iw, motion_weight=mw, size_weight=sw, motion_sigma_px=sigma,
        velocity_window_frames=cfg.velocity_window_frames, min_track_score=min_s,
        min_identity_similarity=min_i, min_confidence_gap=min_g, lost_after_misses=1,
        dynamic_search_enabled=False, min_search_radius_px=min(50, cfg.initial_radius_px),
        max_search_radius_px=max(cfg.initial_radius_px, cfg.max_radius_px),
        confidence_gap_reference=cfg.confidence_gap_reference,
        expected_area_update_alpha=cfg.expected_area_update_alpha,
    )


def _window_points(window: Sequence[MotionPoint | None]) -> list[MotionPoint]:
    return [p for p in window if p is not None]


def _append_window(window: list[MotionPoint | None], point: MotionPoint | None, max_size: int) -> None:
    window.append(point)
    if len(window) > max_size:
        del window[:-max_size]


def _append_area_window(window: list[float | None], area_px: float | None, max_size: int) -> None:
    window.append(None if area_px is None else float(max(1.0, area_px)))
    if len(window) > max_size:
        del window[:-max_size]


def _normal_hypothesis_continuity(
    points: Sequence[MotionPoint | None],
    areas: Sequence[float | None],
    point: MotionPoint,
    area_px: float,
    *,
    config: ReacquisitionConfig,
) -> tuple[bool, str]:
    """Check whether a normal 3-of-5 hit belongs to the same hypothesis.

    REV5 fixes the old semantic bug where spatially/scale-incompatible hits
    could be counted together merely because they occurred inside the same
    N-of-M window. Missing frames remain allowed; a hypothesis switch resets
    the window instead of accumulating evidence across different objects.
    """
    previous: tuple[MotionPoint, float] | None = None
    for p, a in zip(reversed(points), reversed(areas)):
        if p is not None and a is not None:
            previous = (p, float(a))
            break
    if previous is None:
        return True, "NEW"
    prev_point, prev_area = previous
    dt = max(1, int(point.frame_index - prev_point.frame_index))
    spatial_distance = hypot(point.x - prev_point.x, point.y - prev_point.y)
    # Reuse the already validated tracklet link scale instead of introducing
    # another independent tuning parameter. A one-frame link is conservative;
    # a missed frame gets a sqrt(dt) spatial allowance, still bounded.
    spatial_allowance = float(config.tracklet_max_link_distance_px) * sqrt(float(dt))
    if spatial_distance > spatial_allowance:
        return False, "POSITION"
    log_area_delta = abs(float(np.log(max(1.0, area_px) / max(1.0, prev_area))))
    scale_allowance = min(
        float(config.tracklet_max_scale_log_span),
        float(config.scale_quarantine_link_log_span) * float(dt),
    )
    if log_area_delta > scale_allowance:
        return False, "SCALE"
    return True, "OK"


def _reacquisition_incremental_engine(
    *,
    frame_reader: FrameReader,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    lost_frame: int,
    reference_bank: ReferenceBank,
    motion_history: Sequence[MotionPoint],
    detector_config: CandidateDetectorConfig,
    config: ReacquisitionConfig,
    identity_weights: Mapping[str, float] | None = None,
    initial_expected_area_px: float | None = None,
    thermal_profile: ThermalProfile | None = None,
    motion_broken: bool = False,
    scale_model: ScaleStabilityModel | None = None,
    reliable_scale_anchor: ReliableScaleAnchor | None = None,
    continuity_state: _ReacquisitionContinuityState | None = None,
) -> Iterator[ReacqFrameResult]:
    """Attempt reacquisition; use robust visual tracklets after MOTION_BREAK."""
    config.validate()
    if frame_count < 1 or not 0 <= lost_frame < frame_count:
        raise ValueError("invalid frame_count/lost_frame")
    if len(reference_bank) < 1 or not motion_history:
        raise ValueError("reference_bank/motion_history must not be empty")
    base_history = list(motion_history)
    if any(b.frame_index <= a.frame_index for a, b in pairwise(base_history)):
        raise ValueError("motion_history frame indices must be strictly increasing")
    base_history = base_history[-max(2, config.velocity_window_frames):]

    broken = bool(motion_broken and config.robust_motion_break_enabled)
    expected_area_px = float(initial_expected_area_px if initial_expected_area_px and initial_expected_area_px > 0 else reference_bank.expected_size_px())
    if scale_model is None and config.thermal_scale_stability_enabled and thermal_profile is not None and thermal_profile.is_thermal:
        scale_model = ScaleStabilityModel(expected_area_px, ScaleStabilityConfig(
            enabled=True, confirm_frames=config.scale_confirm_frames, consistency_log_span=config.scale_consistency_log_span,
            max_log_change_per_frame=config.scale_max_log_change_per_frame, large_change_ratio=config.scale_large_change_ratio,
            large_change_confirm_frames=config.scale_large_change_confirm_frames, micro_max_area_px=config.scale_micro_max_area_px,
            micro_confirm_frames=config.scale_micro_confirm_frames, micro_exit_confirm_frames=config.scale_micro_exit_confirm_frames,
            micro_consistency_log_span=config.scale_micro_consistency_log_span, micro_max_log_change_per_frame=config.scale_micro_max_log_change_per_frame,
        ))
    if scale_model is not None and scale_model.config.enabled:
        expected_area_px = float(scale_model.effective_area_px)
    if (
        reliable_scale_anchor is None
        and config.thermal_scale_parameter_distance_enabled
        and thermal_profile is not None and thermal_profile.is_thermal
    ):
        reliable_scale_anchor = ReliableScaleAnchor(
            expected_area_px, base_history[-1].frame_index, ScaleContinuityConfig(
                enabled=True,
                base_allowed_log_distance=config.scale_anchor_base_allowed_log_distance,
                growth_per_frame=config.scale_anchor_growth_per_frame,
                max_allowed_log_distance=config.scale_anchor_max_allowed_log_distance,
                nonlinear_strength=config.scale_nonlinear_gate_strength,
                quarantine_confirm_hits=config.scale_quarantine_confirm_hits,
                quarantine_extreme_confirm_hits=config.scale_quarantine_extreme_confirm_hits,
                quarantine_extreme_log_distance=config.scale_quarantine_extreme_log_distance,
                quarantine_max_confirm_excess_log=config.scale_quarantine_max_confirm_excess_log,
                quarantine_link_log_span=config.scale_quarantine_link_log_span,
                reliable_update_alpha=config.scale_reliable_update_alpha,
            )
        )
    runtime_detector = runtime_detector_config(detector_config, thermal_profile)
    start_frame = lost_frame + 1
    if start_frame >= frame_count:
        return

    scorer_cfg = _tracking_cfg(config, motion_broken=broken)
    promotion_window: deque[bool] = deque(maxlen=max(2, int(config.auto_promote_window_frames)))
    auto_promoted = False
    promotion_reason = ""
    confirmation_window: list[MotionPoint | None] = []
    confirmation_area_window: list[float | None] = []
    max_frame_exclusive = min(frame_count, start_frame + config.max_frames)
    budget_ms = 1000.0 / max(1e-9, config.performance_target_fps)

    def make_tracklets() -> TrackletManager | None:
        if not config.tracklets_enabled:
            return None
        return TrackletManager(TrackletConfig(
            max_tracklets=config.tracklet_top_k,
            window_frames=config.tracklet_window_frames,
            confirm_hits=config.tracklet_confirm_hits,
            max_link_distance_px=config.tracklet_max_link_distance_px,
            min_mean_identity=config.broken_min_identity_similarity,
            min_mean_score=config.broken_min_score,
            min_mean_gap=config.broken_min_confidence_gap,
            far_jump_confirm_hits=config.far_jump_confirm_hits,
            far_jump_min_mean_identity=config.far_jump_min_identity,
            far_jump_min_mean_gap=config.far_jump_min_gap,
            far_jump_min_median_thermal_profile=config.far_jump_min_thermal_profile,
            max_scale_log_span=config.tracklet_max_scale_log_span,
            micro_confirm_hits=config.tracklet_confirm_hits,
            micro_min_mean_identity=max(config.broken_min_identity_similarity, 0.52),
            scale_distance_confirm_hits=config.scale_quarantine_confirm_hits,
            scale_distance_extreme_confirm_hits=config.scale_quarantine_extreme_confirm_hits,
            scale_distance_extreme_log_distance=config.scale_quarantine_extreme_log_distance,
            scale_distance_max_confirm_excess_log=config.scale_quarantine_max_confirm_excess_log,
        ))

    tracklets = make_tracklets() if broken else None
    global_emergency_active = False
    global_emergency_start_frame: int | None = None

    bg_cfg = BackgroundMotionConfig(
        enabled=bool(config.global_shift_enabled), downscale=config.global_shift_downscale,
        max_dimension_px=config.global_shift_max_dimension_px,
        phase_min_response=config.global_shift_phase_min_response,
        phase_direct_max_fraction=config.global_shift_phase_direct_max_fraction,
        phase_max_change_from_previous_px=config.global_shift_phase_max_change_from_previous_px,
        use_feature_fallback=config.global_shift_feature_fallback,
        target_exclusion_radius_px=max(20, int(round(1.8 * sqrt(max(1.0, expected_area_px))))),
    )
    _prev_idx, prev_frame = frame_reader(lost_frame)
    visual_anchor = base_history[-1].xy
    reliable_anchor_frame = int(base_history[-1].frame_index)
    if len(base_history) >= 2:
        _a, _b = base_history[-2], base_history[-1]
        _dt = max(1, _b.frame_index - _a.frame_index)
        reliable_anchor_speed = float(hypot(_b.x - _a.x, _b.y - _a.y) / _dt)
    else:
        reliable_anchor_speed = 0.0
    try:
        _qidx, quality_reference_frame = frame_reader(reliable_anchor_frame)
        quality_reference_frame = quality_reference_frame.copy()
    except Exception:
        quality_reference_frame = prev_frame.copy()
    frame_degraded_active = False
    frame_degraded_frames = 0
    quality_cfg = FrameReliabilityConfig(
        enabled=bool(config.thermal_frame_reliability_enabled and thermal_profile is not None and thermal_profile.is_thermal),
        max_dimension_px=config.frame_reliability_max_dimension_px, residual_scale=config.frame_reliability_residual_scale,
        phase_response_floor=config.frame_reliability_phase_response_floor, phase_response_good=config.frame_reliability_phase_response_good,
        phase_weight=config.frame_reliability_phase_weight, enter_threshold=config.frame_reliability_enter_threshold,
        exit_threshold=config.frame_reliability_exit_threshold, max_hold_frames=config.frame_reliability_max_hold_frames,
    )
    previous_global_shift_xy: tuple[float, float] | None = None
    previous_global_shift_frame: int | None = None

    def publish_continuity_state(
        points: Sequence[MotionPoint], item: ReacqFrameResult
    ) -> None:
        if continuity_state is None:
            return
        confirmed = points[-1] if item.status == "CONFIRMED" and points else None
        continuity_state.confirmation_points = tuple(points)
        continuity_state.confirmed_frame = (
            confirmed.frame_index if confirmed is not None else None
        )
        continuity_state.confirmed_center = confirmed.xy if confirmed is not None else None
        continuity_state.confirmed_area_px = (
            float(item.selected.area_px)
            if confirmed is not None and item.selected is not None
            else None
        )
        continuity_state.expected_area_px = float(expected_area_px)
        continuity_state.motion_relocked = bool(confirmed is not None and broken)
        continuity_state.motion_broken = bool(broken)
        continuity_state.auto_promoted = bool(auto_promoted)
        continuity_state.scale_model = scale_model
        continuity_state.reliable_scale_anchor = reliable_scale_anchor

    def thermal_profile_score(sim) -> float:
        vals = [
            float(getattr(sim, "thermal_contrast", 0.0)), float(getattr(sim, "hot_area", 0.0)),
            float(getattr(sim, "peak_prominence", 0.0)), float(getattr(sim, "hot_core_compactness", 0.0)),
            float(getattr(sim, "radial_hotspot_profile", 0.0)), float(getattr(sim, "energy_concentration", 0.0)),
        ]
        # Robust average: one projection-dependent thermal cue may dip without
        # destroying an otherwise coherent target history.
        vals.sort()
        return float(sum(vals[1:-1]) / max(1, len(vals) - 2))

    def anchor_allowance(frame_idx: int, speed_now: float, uncertainty_now: float) -> tuple[int, float]:
        age = max(0, int(frame_idx - reliable_anchor_frame))
        growth = 0.0
        if config.anchor_time_normalization_enabled:
            base_speed_now = max(float(reliable_anchor_speed), max(0.0, float(speed_now)))
            v_allow = max(
                float(config.anchor_time_min_speed_px_per_frame),
                min(float(config.anchor_time_max_speed_px_per_frame),
                    float(config.anchor_time_speed_multiplier) * base_speed_now),
            )
            growth = min(float(config.anchor_time_max_growth_px), v_allow * age)
        allowed = (
            float(config.far_jump_distance_px) + growth
            + float(config.anchor_time_uncertainty_gain) * max(0.0, float(uncertainty_now))
        )
        return age, max(float(config.far_jump_distance_px), float(allowed))

    for frame_index in range(start_frame, max_frame_exclusive):
        frame_t0 = perf_counter()
        _idx, frame = frame_reader(frame_index)
        analysis_frame = preprocess_frame(frame, thermal_profile)
        frame_quality = estimate_frame_reliability(quality_reference_frame, frame, config=quality_cfg)
        if quality_cfg.enabled and frame_quality.valid:
            if frame_degraded_active:
                if frame_quality.score >= quality_cfg.exit_threshold:
                    frame_degraded_active = False
                    frame_degraded_frames = 0
                else:
                    frame_degraded_frames += 1
            elif frame_quality.score < quality_cfg.enter_threshold:
                frame_degraded_active = True
                frame_degraded_frames = 1
        global_est = None
        elapsed = frame_index - start_frame
        thermal_now = bool(thermal_profile is not None and thermal_profile.is_thermal)
        emergency_after = max(int(config.full_frame_after_frames), int(config.global_emergency_after_frames))
        if thermal_now and emergency_after > 0 and elapsed >= emergency_after and not global_emergency_active:
            # Search may become global even if MOTION_BREAK was never triggered.
            # This only broadens candidate generation; confirmation remains temporal.
            global_emergency_active = True
            global_emergency_start_frame = frame_index
            if not broken:
                broken = True
                confirmation_window.clear()
                confirmation_area_window.clear()
                tracklets = make_tracklets()
                scorer_cfg = _tracking_cfg(config, motion_broken=True)

        if broken:
            # Update the visual search anchor with background/camera motion. This
            # is only paid during broken REACQ, not normal TRACK.
            if config.global_shift_enabled:
                global_est = estimate_background_motion(
                    prev_frame, frame, config=bg_cfg, target_xy=visual_anchor,
                    target_radius_px=max(20.0, 1.8 * sqrt(max(1.0, expected_area_px))),
                    reticle_mask=(thermal_profile.reticle_mask if thermal_profile is not None else None),
                    previous_shift_xy=(
                        previous_global_shift_xy
                        if previous_global_shift_frame is not None and previous_global_shift_frame == frame_index - 1
                        else None
                    ),
                )
                if global_est.valid:
                    visual_anchor = transform_point(visual_anchor, global_est)
                    previous_global_shift_xy = (float(global_est.dx_px), float(global_est.dy_px))
                    previous_global_shift_frame = frame_index
            active = tracklets.best_active(frame_index) if tracklets is not None else None
            if active is not None and active.evidence:
                predicted = active.predicted_xy()
            else:
                predicted = visual_anchor
            predicted = (min(max(predicted[0], 0.0), frame_width - 1.0), min(max(predicted[1], 0.0), frame_height - 1.0))
            speed = uncertainty = 0.0
        else:
            recent_points = _window_points(confirmation_window)
            working_history = base_history + recent_points
            estimate = predict_position_timed(
                working_history, target_frame_index=frame_index,
                fallback=working_history[-1].xy, window_points=config.velocity_window_frames,
            )
            predicted = (
                min(max(estimate.predicted_xy[0], 0.0), frame_width - 1.0),
                min(max(estimate.predicted_xy[1], 0.0), frame_height - 1.0),
            )
            speed, uncertainty = estimate.speed_px_per_frame, estimate.uncertainty_px

        # Global THERMAL scan is periodic (default 1 of 3 frames) to control CPU.
        full_frame = bool(
            global_emergency_active
            and (not config.full_frame_thermal_only or thermal_now)
            and global_emergency_start_frame is not None
            and ((frame_index - global_emergency_start_frame) % config.global_search_interval_frames == 0)
        )
        if full_frame:
            search_center = ((frame_width - 1) / 2.0, (frame_height - 1) / 2.0)
            radius = int(max(frame_width, frame_height))
        else:
            search_center = predicted
            radius = _radius_for_frame(cfg=config, elapsed_frames=elapsed, speed_px_per_frame=speed, uncertainty_px=uncertainty)

        selection = make_operator_selection(
            frame_index=frame_index, click_xy=search_center, radius_px=radius,
            frame_width=frame_width, frame_height=frame_height,
        )
        td = perf_counter()
        micro_enabled_now = bool(
            config.identity_first_enabled and config.micro_detector_enabled
            and thermal_profile is not None and thermal_profile.is_thermal
        )
        detection = detect_local_candidates(
            analysis_frame, selection, top_k=config.top_k, config=runtime_detector,
            expected_area_px=float(expected_area_px), include_micro=micro_enabled_now,
            global_search=full_frame, diversity_min_distance_px=config.global_diversity_min_distance_px,
        )
        detector_ms = (perf_counter() - td) * 1000.0

        if not detection.candidates:
            if not broken and not frame_degraded_active:
                _append_window(confirmation_window, None, config.confirm_window_frames)
                _append_area_window(confirmation_area_window, None, config.confirm_window_frames)
                promotion_window.append(False)
            hits = 0 if tracklets is not None else len(_window_points(confirmation_window))
            total_ms = (perf_counter() - frame_t0) * 1000.0
            item = ReacqFrameResult(
                frame_index, predicted, radius, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
                hits, config.tracklet_window_frames if tracklets is not None else len(confirmation_window),
                "SEARCHING", (f"FRAME_DEGRADED_{frame_degraded_frames}_OF_{quality_cfg.max_hold_frames}" if frame_degraded_active else "NO_CANDIDATES"), None,
                detector_threshold=float(detection.diagnostics.threshold),
                detector_median_response=float(detection.diagnostics.median_response),
                detector_robust_sigma=float(detection.diagnostics.robust_sigma),
                detector_raw_components=int(detection.diagnostics.raw_components),
                detector_valid_components=int(detection.diagnostics.valid_components),
                expected_area_px=float(expected_area_px),
                detector_effective_min_area_px=int(detection.diagnostics.effective_min_area_px),
                detector_effective_max_area_px=int(detection.diagnostics.effective_max_area_px),
                motion_broken_mode=broken, auto_promoted_motion_break=auto_promoted,
                promotion_evidence_hits=sum(promotion_window), promotion_reason=promotion_reason,
                active_tracklets=len(tracklets.tracklets) if tracklets is not None else 0,
                reacq_mode=("GLOBAL_EMERGENCY_SCAN" if full_frame else ("ROBUST_LOCAL" if broken else "LOCAL_CONSERVATIVE")),
                candidate_channel="NONE", anchor_distance_px=0.0, anchor_distance_norm=0.0,
                anchor_age_frames=max(0, frame_index - reliable_anchor_frame), anchor_allowed_distance_px=0.0,
                reliable_area_anchor_px=(float(reliable_scale_anchor.reliable_area_px) if reliable_scale_anchor is not None else 0.0),
                scale_anchor_age_frames=(max(0, int(frame_index - reliable_scale_anchor.reliable_frame)) if reliable_scale_anchor is not None else 0),
                frame_reliability_score=float(frame_quality.score), frame_reliability_residual=float(frame_quality.residual),
                frame_reliability_phase_response=float(frame_quality.phase_response),
                frame_degraded_active=bool(frame_degraded_active), frame_degraded_frames=int(frame_degraded_frames),
                far_jump_candidate=False,
                micro_candidate_count=int(getattr(detection.diagnostics, "micro_valid_components", 0)),
                global_shift_dx_px=float(global_est.dx_px) if global_est is not None else 0.0,
                global_shift_dy_px=float(global_est.dy_px) if global_est is not None else 0.0,
                global_shift_response=float(global_est.response) if global_est is not None else 0.0,
                global_shift_method=global_est.method if global_est is not None else "OFF",
                global_shift_time_ms=float(global_est.total_time_ms) if global_est is not None else 0.0,
                detector_time_ms=detector_ms, identity_time_ms=0.0,
                performance_budget_ms=budget_ms, over_budget=total_ms > budget_ms, processing_time_ms=total_ms,
            )
            prev_frame = frame.copy()
            if not frame_degraded_active and frame_quality.score >= quality_cfg.exit_threshold:
                quality_reference_frame = frame.copy()
            publish_continuity_state(_window_points(confirmation_window), item)
            yield item
            continue

        ti = perf_counter()
        ranked = _rank_candidates(
            analysis_frame, detection.candidates, reference_bank, predicted,
            cfg=scorer_cfg, identity_weights=identity_weights, thermal_profile=thermal_profile,
            expected_area_px=float(expected_area_px),
            reliable_scale_anchor=reliable_scale_anchor,
            frame_index=frame_index,
        )
        identity_ms = (perf_counter() - ti) * 1000.0
        best = ranked[0]
        second_score = float(ranked[1].total_score) if len(ranked) > 1 else 0.0
        gap = max(0.0, float(best.total_score - ranked[1].total_score if len(ranked) > 1 else best.total_score))
        evidence_gap = gap if len(ranked) > 1 else 0.0
        confidence = _diagnostic_confidence(best, gap, scorer_cfg)
        pred_error = float(hypot(best.candidate.center_x - predicted[0], best.candidate.center_y - predicted[1]))

        # HOTFIX-6 REV1: normal REACQ can promote itself to robust mode.
        # We only do so after repeated strong visual evidence whose position is
        # inconsistent with the stale pre-LOST motion prior. A single frame is
        # never sufficient, and static-overlay candidates are excluded.
        promoted_this_frame = False
        if not frame_degraded_active and not broken and config.robust_motion_break_enabled and config.auto_promote_enabled:
            bad_motion = (
                pred_error > config.max_confirmation_error_px
                or float(best.motion_score) <= config.auto_promote_max_motion_score
            )
            visual_hit = (
                float(best.similarity.total) >= config.auto_promote_min_identity
                and float(evidence_gap) >= config.auto_promote_min_gap
                and float(best.size_score) >= config.auto_promote_min_size_score
                and float(best.static_overlay_penalty) <= config.auto_promote_max_static_overlay_penalty
                and not bool(getattr(best, "hud_hard_reject", False))
                and float(getattr(best, "hud_probability", 0.0)) < 0.45
                and bad_motion
            )
            promotion_window.append(bool(visual_hit))
            if sum(promotion_window) >= config.auto_promote_hits_required:
                broken = True
                auto_promoted = True
                promoted_this_frame = True
                promotion_reason = (
                    f"AUTO_PROMOTE_VISUAL_MOTION_DISAGREEMENT_"
                    f"{sum(promotion_window)}_OF_{config.auto_promote_window_frames}"
                )
                confirmation_window.clear()
                confirmation_area_window.clear()
                tracklets = make_tracklets()
                scorer_cfg = _tracking_cfg(config, motion_broken=True)
                # The visually strong candidate becomes only a hypothesis.
                # REV2 deliberately keeps ``visual_anchor`` at the last reliable
                # target position (plus validated camera shift); a tentative
                # candidate may not redefine target identity.
                ranked = _rank_candidates(
                    analysis_frame, detection.candidates, reference_bank, visual_anchor,
                    cfg=scorer_cfg, identity_weights=identity_weights, thermal_profile=thermal_profile,
                    expected_area_px=expected_area_px, reliable_scale_anchor=reliable_scale_anchor, frame_index=frame_index,
                )
                best = ranked[0]
                second_score = float(ranked[1].total_score) if len(ranked) > 1 else 0.0
                gap = max(0.0, float(best.total_score - ranked[1].total_score if len(ranked) > 1 else best.total_score))
                evidence_gap = gap if len(ranked) > 1 else 0.0
                confidence = _diagnostic_confidence(best, gap, scorer_cfg)
                pred_error = float(hypot(best.candidate.center_x - visual_anchor[0], best.candidate.center_y - visual_anchor[1]))
        selected: Candidate | None = None
        status = "SEARCHING"
        reason = ""
        hits = 0
        tracklet_id = 0
        confirmation_points: list[MotionPoint] = []
        selected_tracklet_identity = 0.0
        selected_tracklet_score = 0.0
        selected_tracklet_gap = 0.0
        selected_tracklet_thermal = 0.0
        selected_tracklet_area = 0
        selected_tracklet_x: float | None = None
        selected_tracklet_y: float | None = None
        confirmed_scale_areas: list[float] = []
        selected_scale_quarantine = bool(getattr(best, "scale_distance_suspicious", False))
        selected_scale_confirmed = False
        scale_obs = scale_model.snapshot("REACQ_NO_CONFIRM") if scale_model is not None and scale_model.config.enabled else None

        if broken and tracklets is not None and not frame_degraded_active:
            evidences: list[TrackletEvidence] = []
            # Evaluate more than tracklet_top_k hypotheses before TrackletManager
            # keeps the strongest few; this prevents a plausible MICRO candidate
            # from being hidden by several large background blobs.
            for i, rc in enumerate(ranked[:max(config.tracklet_top_k, 7)]):
                has_runner_up = i + 1 < len(ranked)
                next_score = ranked[i + 1].total_score if has_runner_up else rc.total_score
                # No runner-up is neutral evidence, not proof of uniqueness.
                cg = max(0.0, float(rc.total_score - next_score)) if has_runner_up else 0.0
                # HOTFIX-3: a learned HUD/reticle hypothesis cannot enter a
                # confirmation tracklet. If the true target is hidden by HUD,
                # REACQ waits for it to emerge instead of learning the overlay.
                if bool(getattr(rc, "hud_hard_reject", False)) or float(getattr(rc, "hud_probability", 0.0)) >= 0.45:
                    continue
                # For a scale-mismatch hypothesis, the nonlinear gate is a safety
                # multiplier, not evidence that the visual target vanished.  Allow
                # it into a temporal tracklet only when the mismatch is still
                # physically quick-confirmable; use the pre-gate score for admission.
                if bool(getattr(rc, "scale_distance_suspicious", False)) and not bool(getattr(rc, "scale_quick_confirmable", True)):
                    continue
                admission_score = float(getattr(rc, "raw_total_score", rc.total_score)) if bool(getattr(rc, "scale_distance_suspicious", False)) else float(rc.total_score)
                if rc.similarity.total < config.tracklet_admission_min_identity or admission_score < config.tracklet_admission_min_score:
                    continue
                is_micro = getattr(rc.candidate, "detection_channel", "NORMAL") == "MICRO"
                if is_micro and (
                    float(rc.similarity.thermal_contrast) < config.micro_min_thermal_contrast
                    or float(rc.similarity.hot_area) < config.micro_min_hot_area
                ):
                    continue
                anchor_dist = float(hypot(rc.candidate.center_x - visual_anchor[0], rc.candidate.center_y - visual_anchor[1]))
                _anchor_age, _anchor_allowed = anchor_allowance(frame_index, speed, uncertainty)
                far_jump = bool(config.identity_first_enabled and anchor_dist > _anchor_allowed)
                if far_jump and (
                    float(rc.similarity.total) < config.far_jump_admission_min_identity
                    or cg < config.far_jump_admission_min_gap
                    or float(rc.static_overlay_penalty) > config.far_jump_admission_max_static_overlay_penalty
                ):
                    # HOTFIX-1: admission is deliberately softer than confirmation.
                    # Temporal aggregate evidence decides identity later.
                    continue
                evidences.append(TrackletEvidence(
                    frame_index, rc.candidate.center_x, rc.candidate.center_y,
                    float(rc.similarity.total), float(rc.total_score), float(rc.size_score), cg, rc.candidate,
                    has_competitor=has_runner_up, anchor_distance_px=anchor_dist, far_jump=far_jump, is_micro=is_micro,
                    background_dx_px=float(global_est.dx_px) if global_est is not None else 0.0,
                    background_dy_px=float(global_est.dy_px) if global_est is not None else 0.0,
                    background_valid=bool(global_est is not None and global_est.valid),
                    thermal_profile_score=thermal_profile_score(rc.similarity),
                    scale_log_distance=float(getattr(rc, "scale_log_distance", 0.0)),
                    scale_allowed_log_distance=float(getattr(rc, "scale_allowed_log_distance", 0.0)),
                    scale_excess_log_distance=float(getattr(rc, "scale_excess_log_distance", 0.0)),
                    scale_nonlinear_gate=float(getattr(rc, "scale_nonlinear_gate", 1.0)),
                    far_scale=bool(getattr(rc, "scale_distance_suspicious", False)),
                    scale_quick_confirmable=bool(getattr(rc, "scale_quick_confirmable", True)),
                ))
            tracklets.update(frame_index, evidences)
            active = tracklets.best_active(frame_index)
            confirmed = tracklets.best_confirmed(frame_index)
            if active is not None and active.evidence:
                cur = active.evidence[-1]
                selected = cur.candidate if cur.frame_index == frame_index else None
                tracklet_id = active.tracklet_id
                hits = len(active.recent(config.tracklet_window_frames, frame_index))
                if selected is not None:
                    selected_tracklet_identity = float(cur.identity); selected_tracklet_score = float(cur.total_score)
                    selected_tracklet_gap = float(cur.gap); selected_tracklet_thermal = float(cur.thermal_profile_score)
                    selected_tracklet_area = int(cur.candidate.area_px); selected_tracklet_x = float(cur.x); selected_tracklet_y = float(cur.y)
            if confirmed is not None and confirmed.evidence[-1].frame_index == frame_index:
                cur = confirmed.evidence[-1]
                selected = cur.candidate
                tracklet_id = confirmed.tracklet_id
                selected_tracklet_identity = float(cur.identity); selected_tracklet_score = float(cur.total_score)
                selected_tracklet_gap = float(cur.gap); selected_tracklet_thermal = float(cur.thermal_profile_score)
                selected_tracklet_area = int(cur.candidate.area_px); selected_tracklet_x = float(cur.x); selected_tracklet_y = float(cur.y)
                recent = confirmed.recent(config.tracklet_window_frames, frame_index)
                confirmation_points = [MotionPoint(e.frame_index, e.x, e.y) for e in recent]
                confirmed_scale_areas = [float(max(1, e.candidate.area_px)) for e in recent]
                selected_scale_quarantine = any(bool(getattr(e, "far_scale", False)) for e in recent)
                selected_scale_confirmed = bool(selected_scale_quarantine)
                hits = len(recent)
                status = "CONFIRMED"
                reason = "REACQ_CONFIRMED_TRACKLET_SCALE" if selected_scale_quarantine else "REACQ_CONFIRMED_TRACKLET"
            elif selected is not None:
                status = "TENTATIVE"
                prefix = "AUTO_PROMOTED_" if promoted_this_frame else ""
                reason = f"{prefix}TENTATIVE_TRACKLET_{hits}_OF_{config.tracklet_window_frames}"
            else:
                reason = "TRACKLETS_NO_VISUAL_MATCH"
        elif broken and tracklets is not None and frame_degraded_active:
            active = tracklets.best_active(frame_index)
            selected = None
            tracklet_id = active.tracklet_id if active is not None else 0
            hits = len(active.recent(config.tracklet_window_frames, frame_index)) if active is not None else 0
            status = "SEARCHING"
            reason = f"FRAME_DEGRADED_{frame_degraded_frames}_OF_{quality_cfg.max_hold_frames}"
        else:
            if frame_degraded_active:
                reasons = [f"FRAME_DEGRADED_Q_{frame_quality.score:.3f}"]
            else:
                reasons = _gate_reasons(best, gap, scorer_cfg)
                if bool(getattr(best, "scale_distance_suspicious", False)):
                    reasons.append("SCALE_PARAMETER_DISTANCE")
            if not frame_degraded_active and (bool(getattr(best, "hud_hard_reject", False)) or float(getattr(best, "hud_probability", 0.0)) >= 0.90):
                reasons.append("HUD_HARD_REJECT")
            if not frame_degraded_active and not broken and pred_error > config.max_confirmation_error_px:
                reasons.append("CONFIRMATION_MOTION_INCONSISTENT")
            hypothesis_reset = ""
            if reasons:
                if not frame_degraded_active:
                    _append_window(confirmation_window, None, config.confirm_window_frames)
                    _append_area_window(confirmation_area_window, None, config.confirm_window_frames)
                reason = "+".join(reasons)
            else:
                selected = best.candidate
                selected_tracklet_identity = float(best.similarity.total); selected_tracklet_score = float(best.total_score)
                selected_tracklet_gap = float(gap); selected_tracklet_thermal = thermal_profile_score(best.similarity)
                selected_tracklet_area = int(selected.area_px); selected_tracklet_x = float(selected.center_x); selected_tracklet_y = float(selected.center_y)
                point = MotionPoint(frame_index, selected.center_x, selected.center_y)
                continuous, continuity_reason = _normal_hypothesis_continuity(
                    confirmation_window, confirmation_area_window, point, float(selected.area_px), config=config
                )
                if not continuous:
                    # Different object/scale hypothesis: never add its hit to the
                    # preceding 3-of-5 evidence. Start a fresh hypothesis here.
                    confirmation_window.clear()
                    confirmation_area_window.clear()
                    hypothesis_reset = continuity_reason
                _append_window(confirmation_window, point, config.confirm_window_frames)
                _append_area_window(confirmation_area_window, float(selected.area_px), config.confirm_window_frames)
                status = "TENTATIVE"
                reason = "TENTATIVE_OK"
            confirmation_points = _window_points(confirmation_window)
            hits = len(confirmation_points)
            if selected is not None and hits >= config.confirm_frames:
                status = "CONFIRMED"; reason = "REACQ_CONFIRMED"
            elif selected is not None:
                prefix = f"TENTATIVE_HYPOTHESIS_RESET_{hypothesis_reset}_" if hypothesis_reset else "TENTATIVE_OK_"
                reason = f"{prefix}{hits}_OF_{config.confirm_window_frames}"

        # REV2: tentative candidates never change the size model or the reliable
        # spatial anchor. Only a fully CONFIRMED target may do so.
        if selected is not None and status == "CONFIRMED":
            # A confirmed far-scale tracklet is the only mechanism allowed to
            # redefine the session-wide ReliableAreaAnchor after a large change.
            # Ordinary confirmations only follow it slowly when trusted.
            trusted_scale = (
                (float(selected_tracklet_identity) if selected_tracklet_identity > 0 else float(best.similarity.total)) >= config.size_update_min_identity
                and (float(selected_tracklet_gap) if selected_tracklet_identity > 0 else float(evidence_gap)) >= config.size_update_min_gap
            )
            if reliable_scale_anchor is not None and reliable_scale_anchor.config.enabled:
                if selected_scale_confirmed:
                    reliable_scale_anchor.commit_confirmed_transition(
                        confirmed_scale_areas or [float(max(1, selected.area_px))], frame_index
                    )
                elif trusted_scale and not bool(getattr(best, "scale_distance_suspicious", False)):
                    reliable_scale_anchor.update_trusted(float(max(1, selected.area_px)), frame_index)
            if scale_model is not None and scale_model.config.enabled:
                if trusted_scale:
                    scale_obs = scale_model.commit_confirmed_reacquisition(
                        frame_index=frame_index, area_px=float(max(1, selected.area_px)),
                        is_micro=(getattr(selected, "detection_channel", "NORMAL") == "MICRO"),
                    )
                else:
                    scale_obs = scale_model.snapshot("REACQ_CONFIRMED_SCALE_UNTRUSTED")
                expected_area_px = float(scale_model.effective_area_px)
            elif trusted_scale:
                a = float(config.expected_area_update_alpha)
                proposed = (1.0 - a) * expected_area_px + a * float(max(1, selected.area_px))
                step = float(config.size_update_max_step_ratio)
                expected_area_px = min(expected_area_px * step, max(expected_area_px / step, proposed))

        total_ms = (perf_counter() - frame_t0) * 1000.0
        item = ReacqFrameResult(
            frame_index=frame_index, predicted_xy=predicted, search_radius_px=radius,
            selected=selected, selected_score=float(best.total_score), identity_similarity=float(best.similarity.total),
            motion_score=float(best.motion_score), size_score=float(best.size_score), confidence_gap=gap,
            confidence=confidence, candidate_count=len(ranked), confirmation_streak=hits,
            confirmation_window_size=(config.tracklet_window_frames if tracklets is not None else len(confirmation_window)),
            status=status, diagnostic_reason=reason,
            prediction_error_px=pred_error if selected is not None else None,
            best_candidate_center=(float(best.candidate.center_x), float(best.candidate.center_y)),
            best_candidate_bbox=(int(best.candidate.bbox.x), int(best.candidate.bbox.y), int(best.candidate.bbox.width), int(best.candidate.bbox.height)),
            best_candidate_area_px=int(best.candidate.area_px), best_candidate_mean_response=float(best.candidate.mean_response),
            best_candidate_peak_response=float(best.candidate.peak_response), second_score=second_score,
            appearance_similarity=float(best.similarity.appearance), edge_similarity=float(best.similarity.edge),
            projection_similarity=float(best.similarity.projection), geometry_similarity=float(best.similarity.geometry),
            color_similarity=float(getattr(best.similarity, "color", 0.0)),
            thermal_contrast_similarity=float(best.similarity.thermal_contrast), hot_area_similarity=float(best.similarity.hot_area),
            peak_prominence_similarity=float(getattr(best.similarity, "peak_prominence", 0.0)),
            hot_core_compactness_similarity=float(getattr(best.similarity, "hot_core_compactness", 0.0)),
            radial_hotspot_profile_similarity=float(getattr(best.similarity, "radial_hotspot_profile", 0.0)),
            energy_concentration_similarity=float(getattr(best.similarity, "energy_concentration", 0.0)),
            thermal_2d_ratio_similarity=float(getattr(best.similarity, "thermal_2d_ratio", 0.0)),
            static_overlay_penalty=float(best.static_overlay_penalty),
            hud_probability=float(getattr(best, "hud_probability", 0.0)),
            hud_hard_overlap=float(getattr(best, "hud_hard_overlap", 0.0)),
            hud_soft_overlap=float(getattr(best, "hud_soft_overlap", 0.0)),
            hud_hard_reject=bool(getattr(best, "hud_hard_reject", False)),
            hud_negative_penalty=float(getattr(best, "hud_negative_penalty", 0.0)),
            detector_threshold=float(detection.diagnostics.threshold),
            detector_median_response=float(detection.diagnostics.median_response), detector_robust_sigma=float(detection.diagnostics.robust_sigma),
            detector_raw_components=int(detection.diagnostics.raw_components), detector_valid_components=int(detection.diagnostics.valid_components),
            expected_area_px=float(expected_area_px), detector_effective_min_area_px=int(detection.diagnostics.effective_min_area_px),
            detector_effective_max_area_px=int(detection.diagnostics.effective_max_area_px),
            motion_broken_mode=broken, auto_promoted_motion_break=auto_promoted,
            promotion_evidence_hits=sum(promotion_window), promotion_reason=promotion_reason,
            tracklet_id=tracklet_id,
            active_tracklets=len(tracklets.tracklets) if tracklets is not None else 0,
            reacq_mode=("GLOBAL_EMERGENCY_SCAN" if full_frame else ("ROBUST_LOCAL" if broken else "LOCAL_CONSERVATIVE")),
            candidate_channel=getattr(best.candidate, "detection_channel", "NORMAL"),
            anchor_distance_px=float(hypot(best.candidate.center_x - visual_anchor[0], best.candidate.center_y - visual_anchor[1])),
            anchor_distance_norm=float(hypot(best.candidate.center_x - visual_anchor[0], best.candidate.center_y - visual_anchor[1]) / max(1e-9, anchor_allowance(frame_index, speed, uncertainty)[1])),
            anchor_age_frames=int(anchor_allowance(frame_index, speed, uncertainty)[0]),
            anchor_allowed_distance_px=float(anchor_allowance(frame_index, speed, uncertainty)[1]),
            frame_reliability_score=float(frame_quality.score), frame_reliability_residual=float(frame_quality.residual),
            frame_reliability_phase_response=float(frame_quality.phase_response),
            frame_degraded_active=bool(frame_degraded_active), frame_degraded_frames=int(frame_degraded_frames),
            far_jump_candidate=bool(config.identity_first_enabled and hypot(best.candidate.center_x - visual_anchor[0], best.candidate.center_y - visual_anchor[1]) > anchor_allowance(frame_index, speed, uncertainty)[1]),
            micro_candidate_count=int(getattr(detection.diagnostics, "micro_valid_components", 0)),
            selected_tracklet_x=selected_tracklet_x, selected_tracklet_y=selected_tracklet_y,
            selected_tracklet_area_px=selected_tracklet_area, selected_tracklet_identity=selected_tracklet_identity,
            selected_tracklet_score=selected_tracklet_score, selected_tracklet_gap=selected_tracklet_gap,
            selected_tracklet_thermal_profile=selected_tracklet_thermal,
            scale_state=(scale_obs.state if scale_obs is not None else "LEGACY"),
            scale_base_area_px=(scale_obs.base_area_px if scale_obs is not None else float(expected_area_px)),
            scale_current_area_px=(scale_obs.current_area_px if scale_obs is not None else float(expected_area_px)),
            scale_pending_count=(scale_obs.pending_count if scale_obs is not None else 0),
            scale_update_reason=(scale_obs.reason if scale_obs is not None else "LEGACY"),
            reliable_area_anchor_px=(float(reliable_scale_anchor.reliable_area_px) if reliable_scale_anchor is not None else 0.0),
            scale_log_distance=float(getattr(best, "scale_log_distance", 0.0)),
            scale_allowed_log_distance=float(getattr(best, "scale_allowed_log_distance", 0.0)),
            scale_excess_log_distance=float(getattr(best, "scale_excess_log_distance", 0.0)),
            scale_nonlinear_gate=float(getattr(best, "scale_nonlinear_gate", 1.0)),
            scale_anchor_age_frames=(int(reliable_scale_anchor.metrics(best.candidate.area_px, frame_index).anchor_age_frames) if reliable_scale_anchor is not None and reliable_scale_anchor.config.enabled else 0),
            scale_quarantine_active=bool(getattr(best, "scale_distance_suspicious", False) and status != "CONFIRMED"),
            scale_quarantine_hits=int(hits if selected_scale_quarantine else 0),
            scale_quarantine_required_hits=int(config.scale_quarantine_extreme_confirm_hits if bool(getattr(best, "scale_distance_extreme", False)) else (config.scale_quarantine_confirm_hits if selected_scale_quarantine else 0)),
            scale_quarantine_confirmed=bool(selected_scale_confirmed),
            global_shift_dx_px=float(global_est.dx_px) if global_est is not None else 0.0,
            global_shift_dy_px=float(global_est.dy_px) if global_est is not None else 0.0,
            global_shift_response=float(global_est.response) if global_est is not None else 0.0,
            global_shift_method=global_est.method if global_est is not None else "OFF",
            global_shift_time_ms=float(global_est.total_time_ms) if global_est is not None else 0.0,
            detector_time_ms=detector_ms, identity_time_ms=identity_ms,
            performance_budget_ms=budget_ms, over_budget=total_ms > budget_ms, processing_time_ms=total_ms,
        )
        prev_frame = frame.copy()
        if not frame_degraded_active and frame_quality.score >= quality_cfg.exit_threshold:
            quality_reference_frame = frame.copy()

        if status == "CONFIRMED" and selected is not None:
            if not confirmation_points:
                confirmation_points = [MotionPoint(frame_index, selected.center_x, selected.center_y)]
            publish_continuity_state(confirmation_points, item)
            yield item
            return
        publish_continuity_state(confirmation_points, item)
        yield item


class ReacquisitionRuntime:
    """Bounded, persistent one-frame REACQ engine."""

    _COLLECTION_STATE_NAMES = (
        "base_history",
        "promotion_window",
        "confirmation_window",
        "confirmation_area_window",
    )

    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        lost_frame: int,
        previous_frame: np.ndarray,
        reference_bank: ReferenceBank,
        motion_history: Sequence[MotionPoint],
        detector_config: CandidateDetectorConfig,
        config: ReacquisitionConfig,
        identity_weights: Mapping[str, float] | None = None,
        initial_expected_area_px: float | None = None,
        quality_reference_frame: np.ndarray | None = None,
        thermal_profile: ThermalProfile | None = None,
        motion_broken: bool = False,
        scale_model: ScaleStabilityModel | None = None,
        reliable_scale_anchor: ReliableScaleAnchor | None = None,
        frame_count: int | None = None,
    ) -> None:
        config.validate()
        if frame_width < 1 or frame_height < 1:
            raise ValueError("frame dimensions must be >= 1")
        if lost_frame < 0 or (frame_count is not None and not 0 <= lost_frame < frame_count):
            raise ValueError("invalid frame_count/lost_frame")
        if frame_count is not None and frame_count < 1:
            raise ValueError("invalid frame_count/lost_frame")
        if len(reference_bank) < 1 or not motion_history:
            raise ValueError("reference_bank/motion_history must not be empty")
        history = list(motion_history)
        if any(b.frame_index <= a.frame_index for a, b in pairwise(history)):
            raise ValueError("motion_history frame indices must be strictly increasing")

        self.lost_frame = int(lost_frame)
        self.start_frame = self.lost_frame + 1
        self.next_frame_index = self.start_frame
        budget_end = self.start_frame + int(config.max_frames)
        self.frame_count = None if frame_count is None else int(frame_count)
        self.end_frame_exclusive = min(self.frame_count, budget_end) if self.frame_count is not None else budget_end
        self.done = self.start_frame >= self.end_frame_exclusive
        self.last_result: ReacqFrameResult | None = None
        self.attempted_frames = 0
        self._pending_frame: np.ndarray | None = None
        self._continuity_state = _ReacquisitionContinuityState()
        self._previous_frame = previous_frame
        self._quality_reference_frame = (
            previous_frame if quality_reference_frame is None else quality_reference_frame
        )
        self._previous_served = False
        self._quality_served = False
        reliable_anchor_frame = int(history[-1].frame_index)

        def frame_reader(frame_index: int) -> tuple[int, np.ndarray]:
            if frame_index == self.lost_frame and not self._previous_served:
                self._previous_served = True
                return int(frame_index), self._previous_frame
            if frame_index == reliable_anchor_frame and not self._quality_served:
                self._quality_served = True
                return int(frame_index), self._quality_reference_frame
            if self._pending_frame is None:
                raise RuntimeError("ReacquisitionRuntime.step() did not supply a frame")
            frame = self._pending_frame
            self._pending_frame = None
            return int(frame_index), frame

        self._engine = _reacquisition_incremental_engine(
            frame_reader=frame_reader,
            frame_count=self.end_frame_exclusive,
            frame_width=frame_width,
            frame_height=frame_height,
            lost_frame=lost_frame,
            reference_bank=reference_bank,
            motion_history=history,
            detector_config=detector_config,
            config=config,
            identity_weights=identity_weights,
            initial_expected_area_px=initial_expected_area_px,
            thermal_profile=thermal_profile,
            motion_broken=motion_broken,
            scale_model=scale_model,
            reliable_scale_anchor=reliable_scale_anchor,
            continuity_state=self._continuity_state,
        )

    def step(self, frame: np.ndarray) -> ReacqFrameResult:
        """Process exactly the next REACQ frame."""
        if self.done:
            raise RuntimeError("reacquisition runtime is already complete")
        if self._pending_frame is not None:
            raise RuntimeError("reacquisition runtime already has a pending frame")
        self._pending_frame = frame
        try:
            item = next(self._engine)
        except StopIteration as exc:
            self.done = True
            self._pending_frame = None
            raise RuntimeError("reacquisition runtime completed without a frame result") from exc
        self.last_result = item
        self.attempted_frames += 1
        self.next_frame_index = int(item.frame_index) + 1
        if item.status == "CONFIRMED" or self.next_frame_index >= self.end_frame_exclusive:
            self.done = True
        return item

    @property
    def persistent_state(self) -> Mapping[str, object]:
        """Expose the live algorithm state for diagnostics and bound tests."""
        frame = self._engine.gi_frame
        return {} if frame is None else dict(frame.f_locals)

    @property
    def continuity_snapshot(self) -> ReacquisitionContinuitySnapshot:
        """Return the typed, read-only REACQ transition boundary."""

        state = self._continuity_state
        return ReacquisitionContinuitySnapshot(
            confirmation_points=state.confirmation_points,
            confirmed_frame=state.confirmed_frame,
            confirmed_center=state.confirmed_center,
            confirmed_area_px=state.confirmed_area_px,
            expected_area_px=float(state.expected_area_px),
            motion_relocked=bool(state.motion_relocked),
            motion_broken=bool(state.motion_broken),
            auto_promoted=bool(state.auto_promoted),
            scale_model=state.scale_model,
            reliable_scale_anchor=state.reliable_scale_anchor,
        )

    def collection_state_sizes(self) -> dict[str, int]:
        state = self.persistent_state
        sizes = {
            name: len(value)
            for name in self._COLLECTION_STATE_NAMES
            if (value := state.get(name)) is not None
        }
        tracklets = state.get("tracklets")
        if tracklets is not None:
            sizes["tracklets"] = len(tracklets.tracklets)
            sizes["tracklet_evidence_max"] = max(
                (len(tracklet.evidence) for tracklet in tracklets.tracklets),
                default=0,
            )
        return sizes

    def close(self) -> None:
        self.done = True
        self._pending_frame = None
        self._engine.close()


def run_reacquisition(
    *,
    frame_reader: FrameReader,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    lost_frame: int,
    reference_bank: ReferenceBank,
    motion_history: Sequence[MotionPoint],
    detector_config: CandidateDetectorConfig,
    config: ReacquisitionConfig,
    identity_weights: Mapping[str, float] | None = None,
    initial_expected_area_px: float | None = None,
    progress_callback: LiveProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
    thermal_profile: ThermalProfile | None = None,
    motion_broken: bool = False,
    scale_model: ScaleStabilityModel | None = None,
    reliable_scale_anchor: ReliableScaleAnchor | None = None,
) -> ReacquisitionResult:
    """Compatibility batch wrapper over :class:`ReacquisitionRuntime`."""
    config.validate()
    if frame_count < 1 or not 0 <= lost_frame < frame_count:
        raise ValueError("invalid frame_count/lost_frame")
    if len(reference_bank) < 1 or not motion_history:
        raise ValueError("reference_bank/motion_history must not be empty")
    history = list(motion_history)
    if any(b.frame_index <= a.frame_index for a, b in pairwise(history)):
        raise ValueError("motion_history frame indices must be strictly increasing")

    expected_area_px = float(
        initial_expected_area_px
        if initial_expected_area_px is not None and initial_expected_area_px > 0
        else reference_bank.expected_size_px()
    )
    if scale_model is not None and scale_model.config.enabled:
        expected_area_px = float(scale_model.effective_area_px)
    broken_initial = bool(motion_broken and config.robust_motion_break_enabled)
    start_frame = lost_frame + 1
    if start_frame >= frame_count:
        return ReacquisitionResult(
            "EOF", lost_frame, (), None, None, None, (), 0, 0.0, 0,
            None, expected_area_px, broken_initial, False, 0.0, 0,
        )

    _prev_idx, previous_frame = frame_reader(lost_frame)
    quality_reference_frame = previous_frame.copy()
    with suppress(Exception):
        _quality_idx, quality_reference_frame = frame_reader(history[-1].frame_index)
        quality_reference_frame = quality_reference_frame.copy()

    runtime = ReacquisitionRuntime(
        frame_count=frame_count,
        frame_width=frame_width,
        frame_height=frame_height,
        lost_frame=lost_frame,
        previous_frame=previous_frame,
        quality_reference_frame=quality_reference_frame,
        reference_bank=reference_bank,
        motion_history=history,
        detector_config=detector_config,
        config=config,
        identity_weights=identity_weights,
        initial_expected_area_px=initial_expected_area_px,
        thermal_profile=thermal_profile,
        motion_broken=motion_broken,
        scale_model=scale_model,
        reliable_scale_anchor=reliable_scale_anchor,
    )
    results: list[ReacqFrameResult] = []
    stopped = False
    try:
        while not runtime.done:
            frame_index = runtime.next_frame_index
            if stop_requested is not None and stop_requested():
                stopped = True
                break
            _idx, frame = frame_reader(frame_index)
            item = runtime.step(frame)
            results.append(item)
            if progress_callback is not None:
                progress_callback(
                    LiveFrameEvent(
                        mode="REACQ",
                        frame_index=int(frame_index),
                        frame_bgr=frame.copy(),
                        result=item,
                    )
                )
    finally:
        continuity = runtime.continuity_snapshot
        end_frame_exclusive = runtime.end_frame_exclusive
        runtime.close()

    confirmed_item = next((item for item in results if item.status == "CONFIRMED"), None)
    confirmation_points = continuity.confirmation_points if confirmed_item is not None else ()
    if confirmed_item is not None and not confirmation_points and confirmed_item.selected is not None:
        confirmation_points = (
            MotionPoint(
                confirmed_item.frame_index,
                confirmed_item.selected.center_x,
                confirmed_item.selected.center_y,
            ),
        )
    confirmed = confirmation_points[-1] if confirmation_points else None
    if stopped:
        status = "STOPPED"
    elif confirmed_item is not None:
        status = "CONFIRMED"
    else:
        status = "EOF" if end_frame_exclusive >= frame_count else "FAILED"
    shift_times = [
        item.global_shift_time_ms
        for item in results
        if item.global_shift_method != "OFF"
    ]
    broken = continuity.motion_broken if results else broken_initial
    final_expected_area = float(continuity.expected_area_px or expected_area_px)
    return ReacquisitionResult(
        status=status,
        lost_frame=lost_frame,
        frames=tuple(results),
        confirmed_frame=confirmed.frame_index if confirmed is not None else None,
        confirmed_center=confirmed.xy if confirmed is not None else None,
        t_reacq_frames=(confirmed.frame_index - lost_frame) if confirmed is not None else None,
        confirmation_points=confirmation_points,
        attempted_frames=len(results),
        mean_confidence=float(mean(item.confidence for item in results)) if results else 0.0,
        max_search_radius_used_px=max((item.search_radius_px for item in results), default=0),
        confirmed_area_px=(float(confirmed_item.selected.area_px) if confirmed_item is not None and confirmed_item.selected is not None else None),
        last_expected_area_px=final_expected_area,
        motion_broken_mode=broken,
        motion_relocked=bool(confirmed_item is not None and broken),
        mean_global_shift_time_ms=float(mean(shift_times)) if shift_times else 0.0,
        over_budget_frames=sum(1 for item in results if item.over_budget),
        auto_promoted=continuity.auto_promoted,
        promotion_frame=next(
            (item.frame_index for item in results if item.auto_promoted_motion_break),
            None,
        ),
    )
