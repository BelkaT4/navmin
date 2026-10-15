"""VT-9 continuous TRACK with motion diagnostics and optional cautious EMA.

Base INITIAL TRAIN references remain immutable; an optional EMA adapter may
update one separate adaptive descriptor only on high-confidence TRACK frames.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import exp, hypot, sqrt
from statistics import median
from time import perf_counter

import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    Candidate,
    CandidateDetectorConfig,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import (
    EMAReferenceAdapter,
    EMAUpdateEvent,
)
from navmin.vision.processors.vt11.identity.descriptor import (
    DescriptorSimilarity,
    TargetDescriptor,
    descriptor_from_candidate,
)
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import (
    ThermalProfile,
    hud_evidence,
    hud_point_membership,
    preprocess_frame,
    reticle_overlap_ratio,
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
    dynamic_search_radius,
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


@dataclass(frozen=True)
class TrackingConfig:
    top_k: int = 5
    search_radius_px: int = 90
    identity_weight: float = 0.60
    motion_weight: float = 0.30
    size_weight: float = 0.10
    motion_sigma_px: float = 30.0
    velocity_window_frames: int = 5
    min_track_score: float = 0.48
    min_identity_similarity: float = 0.42
    min_confidence_gap: float = 0.02
    lost_after_misses: int = 3

    # VT-7 dynamic motion/confidence parameters.
    dynamic_search_enabled: bool = True
    min_search_radius_px: int = 50
    max_search_radius_px: int = 180
    speed_radius_gain: float = 0.50
    uncertainty_radius_gain: float = 2.0
    miss_radius_growth_px: float = 25.0
    confidence_gap_reference: float = 0.12
    prediction_error_warn_px: float = 35.0
    expected_area_update_alpha: float = 0.15

    # HOTFIX-5B: soft TRAIN -> TRACK warm-up.
    track_warmup_frames: int = 0
    warmup_search_radius_px: int = 150
    warmup_motion_weight: float = 0.08
    warmup_identity_weight: float = 0.82
    warmup_size_weight: float = 0.10
    warmup_motion_sigma_px: float = 90.0
    warmup_min_track_score: float = 0.38
    warmup_min_identity_similarity: float = 0.32
    warmup_min_confidence_gap: float = 0.01
    # REV2: WARMUP may be spatially soft, but it must not accept an unexplained
    # one-frame jump to an unrelated object.
    identity_first_safe_warmup_enabled: bool = False
    warmup_max_unexplained_jump_px: float = 70.0

    # REV2: expected target size is updated only by trusted accepted frames and
    # may not jump arbitrarily because one wrong candidate was accepted.
    size_update_min_identity: float = 0.55
    size_update_min_gap: float = 0.04
    size_update_max_step_ratio: float = 1.25
    size_update_max_prediction_error_px: float = 1.0e9

    # VT-11 HOTFIX-2 THERMAL JUMP QUARANTINE. Disabled by default so DAY
    # remains behaviourally unchanged.
    thermal_jump_quarantine_enabled: bool = False
    jump_quarantine_distance_px: float = 60.0
    jump_extreme_distance_px: float = 85.0
    jump_confirm_hits: int = 2
    jump_extreme_confirm_hits: int = 3
    jump_single_candidate_extra_hits: int = 1
    jump_link_distance_px: float = 35.0
    jump_min_identity: float = 0.62
    jump_confirm_min_identity: float = 0.70
    jump_min_gap: float = 0.04
    jump_min_size_score: float = 0.03
    jump_max_static_overlay_penalty: float = 0.40
    jump_clutter_penalty_weight: float = 0.14
    jump_anchor_norm_threshold: float = 3.0
    jump_anchor_sigma_floor_px: float = 18.0
    jump_local_preference_max_identity_deficit: float = 0.25
    jump_global_motion_veto_response_min: float = 0.25
    jump_global_motion_veto_residual_px: float = 60.0
    reliable_anchor_update_max_prediction_error_px: float = 35.0

    # VT-11 HOTFIX-4: the spatial relation to ReliableAnchor relaxes with time,
    # but a one-frame long jump remains exceptional.  This preserves identity
    # without treating normal displacement accumulated over many frames as a jump.
    anchor_time_normalization_enabled: bool = False
    anchor_time_speed_multiplier: float = 1.50
    anchor_time_min_speed_px_per_frame: float = 3.0
    anchor_time_max_speed_px_per_frame: float = 22.0
    anchor_time_max_growth_px: float = 360.0

    # VT-11 HOTFIX-4 REV3: nonlinear distance in the SCALE parameter.
    # Ordinary size_score remains a soft cue, while a gross scale mismatch is
    # multiplicative negative evidence relative to a session-wide reliable area.
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
    scale_quarantine_min_identity: float = 0.62
    scale_quarantine_min_gap: float = 0.03
    scale_quarantine_max_prediction_error_px: float = 55.0
    scale_quarantine_link_distance_px: float = 45.0

    # VT-11 HOTFIX-4 REV4: asymmetric scale uncertainty. Sudden shrink remains
    # strongly gated, while moderate expansion may preserve position if visual
    # identity and motion remain strong. Scale/EMA memory stays frozen until
    # temporal scale confirmation.
    scale_expansion_uncertainty_enabled: bool = False
    scale_expansion_uncertainty_min_identity: float = 0.70
    scale_expansion_uncertainty_min_motion_score: float = 0.55
    scale_expansion_uncertainty_min_gap: float = 0.02
    scale_expansion_uncertainty_max_prediction_error_px: float = 45.0
    scale_expansion_uncertainty_max_log_distance: float = 1.85
    scale_expansion_uncertainty_min_raw_score: float = 0.44

    # VT-11 HOTFIX-4: generic whole-frame reliability.  It does not identify a
    # particular stripe/artefact.  During short degraded intervals target memory,
    # scale and ReliableAnchor are frozen.
    thermal_frame_reliability_enabled: bool = False
    frame_reliability_max_dimension_px: int = 128
    frame_reliability_residual_scale: float = 0.75
    frame_reliability_phase_response_floor: float = 0.02
    frame_reliability_phase_response_good: float = 0.50
    frame_reliability_phase_weight: float = 0.15
    frame_reliability_enter_threshold: float = 0.55
    frame_reliability_exit_threshold: float = 0.72
    frame_reliability_max_hold_frames: int = 5

    # VT-11 HOTFIX-3: explicit negative HUD / reticle model.  Enabled only for
    # THERMAL.  A known static overlay can be rejected even after a false lock
    # makes its motion score look perfect.  Genuine target passage through the
    # reticle is represented as temporary occlusion rather than a new target.
    thermal_hud_negative_enabled: bool = False
    hud_hard_probability_threshold: float = 0.90
    hud_soft_probability_threshold: float = 0.45
    hud_occlusion_max_frames: int = 18
    hud_occlusion_prediction_tolerance_px: float = 48.0
    hud_exit_confirm_hits: int = 2
    hud_exit_link_distance_px: float = 42.0
    hud_exit_min_identity: float = 0.52
    hud_exit_max_probability: float = 0.30

    # VT-11 HOTFIX-1 THERMAL-only robust two-time-scale model. DAY leaves it off.
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
    scale_robust_history_frames: int = 12

    # HOTFIX-5A: controlled drift rescue. It may override only SCORE/IDENTITY
    # gates when spatial continuity is very strong. BaseBank remains immutable.
    drift_rescue_enabled: bool = False
    drift_rescue_min_motion_score: float = 0.92
    drift_rescue_min_identity_similarity: float = 0.30
    drift_rescue_min_confidence_gap: float = 0.08
    drift_rescue_min_size_score: float = 0.04
    drift_rescue_max_prediction_error_px: float = 18.0
    drift_rescue_max_consecutive: int = 8

    # HOTFIX-6: event-driven camera/background motion and motion-break logic.
    motion_break_enabled: bool = False
    motion_break_error_px: float = 80.0
    motion_break_window_frames: int = 3
    motion_break_hits_required: int = 2
    motion_break_visual_identity_min: float = 0.72
    motion_break_visual_gap_min: float = 0.06
    motion_break_visual_size_min: float = 0.03
    motion_relock_frames: int = 3
    global_shift_enabled: bool = False
    global_shift_trigger_error_px: float = 45.0
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
        if self.search_radius_px < 5:
            raise ValueError("search_radius_px must be >= 5")
        if min(self.identity_weight, self.motion_weight, self.size_weight) < 0:
            raise ValueError("tracking weights must be >= 0")
        if self.identity_weight + self.motion_weight + self.size_weight <= 0:
            raise ValueError("at least one tracking weight must be > 0")
        if self.motion_sigma_px <= 0:
            raise ValueError("motion_sigma_px must be > 0")
        if self.velocity_window_frames < 2:
            raise ValueError("velocity_window_frames must be >= 2")
        for name in ("min_track_score", "min_identity_similarity", "min_confidence_gap"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.lost_after_misses < 1:
            raise ValueError("lost_after_misses must be >= 1")
        if self.min_search_radius_px < 5:
            raise ValueError("min_search_radius_px must be >= 5")
        if self.max_search_radius_px < self.min_search_radius_px:
            raise ValueError("max_search_radius_px must be >= min_search_radius_px")
        if self.dynamic_search_enabled and self.max_search_radius_px < self.search_radius_px:
            raise ValueError("max_search_radius_px must be >= search_radius_px")
        if min(self.speed_radius_gain, self.uncertainty_radius_gain, self.miss_radius_growth_px) < 0:
            raise ValueError("dynamic search gains must be >= 0")
        if self.confidence_gap_reference <= 0:
            raise ValueError("confidence_gap_reference must be > 0")
        if self.prediction_error_warn_px <= 0:
            raise ValueError("prediction_error_warn_px must be > 0")
        if not 0.0 < self.expected_area_update_alpha <= 1.0:
            raise ValueError("expected_area_update_alpha must be in (0,1]")
        if self.track_warmup_frames < 0:
            raise ValueError("track_warmup_frames must be >= 0")
        if self.warmup_search_radius_px < 5:
            raise ValueError("warmup_search_radius_px must be >= 5")
        if min(self.warmup_motion_weight, self.warmup_identity_weight, self.warmup_size_weight) < 0:
            raise ValueError("warmup weights must be >= 0")
        if self.warmup_motion_weight + self.warmup_identity_weight + self.warmup_size_weight <= 0:
            raise ValueError("at least one warmup weight must be > 0")
        if self.warmup_motion_sigma_px <= 0:
            raise ValueError("warmup_motion_sigma_px must be > 0")
        if self.warmup_max_unexplained_jump_px <= 0:
            raise ValueError("warmup_max_unexplained_jump_px must be >0")
        for name in ("size_update_min_identity", "size_update_min_gap"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.size_update_max_step_ratio < 1.0:
            raise ValueError("size_update_max_step_ratio must be >=1")
        if self.size_update_max_prediction_error_px <= 0:
            raise ValueError("size_update_max_prediction_error_px must be >0")
        if self.jump_quarantine_distance_px <= 0 or self.jump_extreme_distance_px < self.jump_quarantine_distance_px:
            raise ValueError("jump distance thresholds invalid")
        if self.jump_confirm_hits < 2 or self.jump_extreme_confirm_hits < self.jump_confirm_hits:
            raise ValueError("jump confirmation counts invalid")
        if self.jump_single_candidate_extra_hits < 0 or self.jump_link_distance_px <= 0:
            raise ValueError("jump temporal settings invalid")
        for name in ("jump_min_identity", "jump_confirm_min_identity", "jump_min_gap",
                     "jump_min_size_score", "jump_max_static_overlay_penalty",
                     "jump_clutter_penalty_weight", "jump_local_preference_max_identity_deficit",
                     "jump_global_motion_veto_response_min"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.jump_anchor_norm_threshold <= 0 or self.jump_anchor_sigma_floor_px <= 0:
            raise ValueError("jump anchor normalization settings invalid")
        if self.jump_global_motion_veto_residual_px <= 0 or self.reliable_anchor_update_max_prediction_error_px <= 0:
            raise ValueError("jump/global/reliable-anchor thresholds must be >0")
        if self.anchor_time_speed_multiplier < 0 or self.anchor_time_min_speed_px_per_frame < 0:
            raise ValueError("anchor time-normalization speeds must be >=0")
        if self.anchor_time_max_speed_px_per_frame < self.anchor_time_min_speed_px_per_frame or self.anchor_time_max_growth_px < 0:
            raise ValueError("anchor time-normalization limits invalid")
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
        for name in ("scale_quarantine_min_identity", "scale_quarantine_min_gap"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.scale_quarantine_max_prediction_error_px <= 0 or self.scale_quarantine_link_distance_px <= 0:
            raise ValueError("scale quarantine spatial thresholds must be >0")
        for name in ("scale_expansion_uncertainty_min_identity", "scale_expansion_uncertainty_min_motion_score",
                     "scale_expansion_uncertainty_min_gap", "scale_expansion_uncertainty_min_raw_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.scale_expansion_uncertainty_max_prediction_error_px <= 0 or self.scale_expansion_uncertainty_max_log_distance <= 0:
            raise ValueError("scale expansion uncertainty limits must be >0")
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
        for name in ("hud_hard_probability_threshold", "hud_soft_probability_threshold",
                     "hud_exit_min_identity", "hud_exit_max_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.hud_soft_probability_threshold > self.hud_hard_probability_threshold:
            raise ValueError("HUD soft threshold must not exceed hard threshold")
        if self.hud_occlusion_max_frames < 1 or self.hud_occlusion_prediction_tolerance_px <= 0:
            raise ValueError("HUD occlusion settings invalid")
        if self.hud_exit_confirm_hits < 1 or self.hud_exit_link_distance_px <= 0:
            raise ValueError("HUD exit confirmation settings invalid")
        ScaleStabilityConfig(
            enabled=self.thermal_scale_stability_enabled, confirm_frames=self.scale_confirm_frames,
            consistency_log_span=self.scale_consistency_log_span, max_log_change_per_frame=self.scale_max_log_change_per_frame,
            large_change_ratio=self.scale_large_change_ratio, large_change_confirm_frames=self.scale_large_change_confirm_frames,
            micro_max_area_px=self.scale_micro_max_area_px, micro_confirm_frames=self.scale_micro_confirm_frames,
            micro_exit_confirm_frames=self.scale_micro_exit_confirm_frames, micro_consistency_log_span=self.scale_micro_consistency_log_span,
            micro_max_log_change_per_frame=self.scale_micro_max_log_change_per_frame,
            robust_history_frames=self.scale_robust_history_frames,
        ).validate()
        for name in ("warmup_min_track_score", "warmup_min_identity_similarity", "warmup_min_confidence_gap",
                     "drift_rescue_min_motion_score", "drift_rescue_min_identity_similarity",
                     "drift_rescue_min_confidence_gap", "drift_rescue_min_size_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.drift_rescue_max_prediction_error_px <= 0:
            raise ValueError("drift_rescue_max_prediction_error_px must be > 0")
        if self.drift_rescue_max_consecutive < 1:
            raise ValueError("drift_rescue_max_consecutive must be >= 1")
        if self.motion_break_error_px <= 0 or self.global_shift_trigger_error_px <= 0:
            raise ValueError("HOTFIX-6 motion thresholds must be >0")
        if self.motion_break_window_frames < 2:
            raise ValueError("motion_break_window_frames must be >=2")
        if not 1 <= self.motion_break_hits_required <= self.motion_break_window_frames:
            raise ValueError("motion_break_hits_required must fit motion_break_window_frames")
        for name in ("motion_break_visual_identity_min", "motion_break_visual_gap_min", "motion_break_visual_size_min"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.motion_relock_frames < 2:
            raise ValueError("motion_relock_frames must be >=2")
        if not 0.10 <= self.global_shift_downscale <= 1.0:
            raise ValueError("global_shift_downscale must be in [0.10,1.0]")
        if self.global_shift_max_dimension_px < 64:
            raise ValueError("global_shift_max_dimension_px must be >=64")
        if not 0.0 <= self.global_shift_phase_min_response <= 1.0:
            raise ValueError("global_shift_phase_min_response must be in [0,1]")
        if not 0.05 <= self.global_shift_phase_direct_max_fraction <= 1.0:
            raise ValueError("global_shift_phase_direct_max_fraction must be in [0.05,1]")
        if self.global_shift_phase_max_change_from_previous_px <= 0:
            raise ValueError("global_shift_phase_max_change_from_previous_px must be >0")
        if self.performance_target_fps <= 0:
            raise ValueError("performance_target_fps must be >0")


@dataclass(frozen=True)
class TrackCandidateScore:
    candidate: Candidate
    descriptor: TargetDescriptor
    similarity: DescriptorSimilarity
    motion_score: float
    size_score: float
    total_score: float
    raw_total_score: float = 0.0
    scale_reliable_area_px: float = 0.0
    scale_log_distance: float = 0.0
    scale_allowed_log_distance: float = 0.0
    scale_excess_log_distance: float = 0.0
    scale_nonlinear_gate: float = 1.0
    scale_distance_suspicious: bool = False
    scale_distance_extreme: bool = False
    scale_quick_confirmable: bool = True
    scale_is_expansion: bool = False
    scale_expansion_uncertain_candidate: bool = False
    static_overlay_penalty: float = 0.0
    hud_probability: float = 0.0
    hud_hard_overlap: float = 0.0
    hud_soft_overlap: float = 0.0
    hud_hard_reject: bool = False
    hud_negative_penalty: float = 0.0


@dataclass(frozen=True)
class TrackFrameResult:
    frame_index: int
    predicted_xy: tuple[float, float]
    selected: Candidate | None
    selected_score: float
    confidence_gap: float
    candidate_count: int
    identity_similarity: float
    motion_score: float
    size_score: float
    status: str
    miss_count: int
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    speed_px_per_frame: float = 0.0
    motion_uncertainty_px: float = 0.0
    search_radius_px: int = 0
    prediction_error_px: float | None = None
    confidence: float = 0.0
    diagnostic_reason: str = ""
    # HOTFIX-2: retain diagnostics for the best evaluated candidate even when
    # it is rejected by gates. This is essential for distinguishing detector
    # failure from identity/threshold failure.
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
    thermal_background_clutter: float = 0.0
    reliable_anchor_xy: tuple[float, float] | None = None
    anchor_distance_px: float = 0.0
    anchor_distance_norm: float = 0.0
    anchor_age_frames: int = 0
    anchor_allowed_distance_px: float = 0.0
    frame_reliability_score: float = 1.0
    frame_reliability_residual: float = 0.0
    frame_reliability_phase_response: float = 1.0
    frame_degraded_active: bool = False
    frame_degraded_frames: int = 0
    jump_quarantine_active: bool = False
    jump_quarantine_hits: int = 0
    jump_quarantine_required_hits: int = 0
    jump_quarantine_confirmed: bool = False
    jump_global_motion_veto: bool = False
    anchor_continuity_override: bool = False
    hud_probability: float = 0.0
    hud_hard_overlap: float = 0.0
    hud_soft_overlap: float = 0.0
    hud_hard_reject: bool = False
    hud_negative_penalty: float = 0.0
    reticle_occlusion_active: bool = False
    reticle_occlusion_frames: int = 0
    reticle_exit_hits: int = 0
    detector_threshold: float = 0.0
    detector_median_response: float = 0.0
    detector_robust_sigma: float = 0.0
    detector_raw_components: int = 0
    detector_valid_components: int = 0
    expected_area_px: float = 0.0
    detector_effective_min_area_px: int = 0
    detector_effective_max_area_px: int = 0
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
    scale_expansion_uncertain: bool = False
    adaptation_updated: bool = False
    adaptation_reason: str = "ADAPT_OFF"
    adaptation_update_count: int = 0
    tracking_phase: str = "TRACK"
    adaptation_layer: str = "NONE"
    adaptation_drift_score: float = 0.0
    adaptation_drift_reason: str = ""
    adaptation_recent_count: int = 0
    adaptation_fast_remaining: int = 0
    motion_break_active: bool = False
    motion_break_triggered: bool = False
    motion_break_reason: str = ""
    global_shift_dx_px: float = 0.0
    global_shift_dy_px: float = 0.0
    global_shift_response: float = 0.0
    global_shift_method: str = "OFF"
    global_shift_time_ms: float = 0.0
    detector_time_ms: float = 0.0
    identity_time_ms: float = 0.0
    motion_time_ms: float = 0.0
    performance_budget_ms: float = 0.0
    over_budget: bool = False
    processing_time_ms: float = 0.0


@dataclass(frozen=True)
class TrackingResult:
    status: str
    frames: tuple[TrackFrameResult, ...]
    tracked_frames: int
    attempted_frames: int
    mean_score: float
    median_score: float
    lost_frame: int | None
    last_center: tuple[float, float] | None
    trajectory: tuple[tuple[int, float, float], ...]
    mean_confidence: float = 0.0
    median_prediction_error_px: float = 0.0
    max_search_radius_used_px: int = 0
    adaptation_updates: int = 0
    adaptation_events: tuple[EMAUpdateEvent, ...] = ()
    last_target_area_px: float = 0.0
    motion_break_events: int = 0
    motion_broken_at_lost: bool = False
    mean_global_shift_time_ms: float = 0.0
    over_budget_frames: int = 0


@dataclass(frozen=True)
class TrackingContinuitySnapshot:
    """Explicit TRACK state needed at a TRACK -> REACQ boundary.

    Frame arrays are read-only copies.  Scale objects intentionally retain
    identity because they are the session-wide mutable continuity objects.
    """

    motion_history: tuple[MotionPoint, ...]
    expected_area_px: float
    motion_broken_at_lost: bool
    previous_frame: np.ndarray | None
    quality_reference_frame: np.ndarray | None
    scale_model: ScaleStabilityModel | None
    reliable_scale_anchor: ReliableScaleAnchor | None
    motion_break_events: int


@dataclass
class _TrackingContinuityState:
    motion_history: tuple[MotionPoint, ...] = ()
    expected_area_px: float = 0.0
    motion_broken_at_lost: bool = False
    previous_frame: np.ndarray | None = None
    quality_reference_frame: np.ndarray | None = None
    scale_model: ScaleStabilityModel | None = None
    reliable_scale_anchor: ReliableScaleAnchor | None = None
    motion_break_events: int = 0


FrameReader = Callable[[int], tuple[int, np.ndarray]]


def _motion_score(center: tuple[float, float], predicted: tuple[float, float], sigma: float) -> float:
    distance = hypot(center[0] - predicted[0], center[1] - predicted[1])
    return float(exp(-(distance * distance) / (2.0 * sigma * sigma)))


def _rank_candidates(
    frame: np.ndarray,
    candidates: Sequence[Candidate],
    bank: ReferenceBank,
    predicted_xy: tuple[float, float],
    *,
    cfg: TrackingConfig,
    identity_weights: Mapping[str, float] | None,
    thermal_profile: ThermalProfile | None = None,
    identity_weight: float | None = None,
    motion_weight: float | None = None,
    size_weight: float | None = None,
    motion_sigma_px: float | None = None,
    expected_area_px: float | None = None,
    reliable_scale_anchor: ReliableScaleAnchor | None = None,
    frame_index: int | None = None,
) -> list[TrackCandidateScore]:
    iw = cfg.identity_weight if identity_weight is None else float(identity_weight)
    mw = cfg.motion_weight if motion_weight is None else float(motion_weight)
    sw = cfg.size_weight if size_weight is None else float(size_weight)
    msigma = cfg.motion_sigma_px if motion_sigma_px is None else float(motion_sigma_px)
    denom = iw + mw + sw
    ranked: list[TrackCandidateScore] = []
    for candidate in candidates:
        descriptor = descriptor_from_candidate(frame, candidate)
        similarity = bank.similarity(descriptor, identity_weights=identity_weights)
        motion = _motion_score(candidate.center, predicted_xy, msigma)
        if expected_area_px is not None and expected_area_px > 0:
            size_ratio = abs(np.log(max(float(candidate.area_px), 1.0) / max(float(expected_area_px), 1.0)))
            size_score = float(exp(-size_ratio / 0.65))
        else:
            size_score = float(similarity.size)
        base_total = (
            iw * similarity.total
            + mw * motion
            + sw * size_score
        ) / denom
        overlap = reticle_overlap_ratio(candidate, thermal_profile)
        # Legacy soft penalty is retained for compatibility, but HOTFIX-3 adds
        # an independent negative-HUD score.  Crucially, the new anti-function
        # is NOT multiplied by (1-motion): false self-consistent tracking of a
        # reticle can therefore no longer make the nuisance penalty vanish.
        static_penalty = 0.0
        if thermal_profile is not None and thermal_profile.is_thermal:
            static_penalty = (
                thermal_profile.static_overlay_penalty_weight
                * overlap
                * max(0.0, 1.0 - float(motion))
            )
        hud = hud_evidence(candidate, thermal_profile)
        hud_negative_penalty = (
            float(thermal_profile.hud_negative_penalty_weight) * float(hud.probability) ** 2
            if thermal_profile is not None and thermal_profile.is_thermal and thermal_profile.hud_negative_enabled
            else 0.0
        )
        raw_total = max(0.0, float(base_total) - float(static_penalty) - float(hud_negative_penalty))
        scale_metrics = (
            reliable_scale_anchor.metrics(candidate.area_px, int(frame_index))
            if reliable_scale_anchor is not None and reliable_scale_anchor.config.enabled and frame_index is not None
            else None
        )
        scale_gate = float(scale_metrics.nonlinear_gate) if scale_metrics is not None else 1.0
        scale_is_expansion = bool(scale_metrics is not None and scale_metrics.is_expansion)
        expansion_uncertain_candidate = bool(
            cfg.scale_expansion_uncertainty_enabled
            and scale_metrics is not None and scale_metrics.suspicious and scale_is_expansion
            and float(scale_metrics.log_distance) <= float(cfg.scale_expansion_uncertainty_max_log_distance)
            and float(similarity.total) >= float(cfg.scale_expansion_uncertainty_min_identity)
            and float(motion) >= float(cfg.scale_expansion_uncertainty_min_motion_score)
            and float(raw_total) >= float(cfg.scale_expansion_uncertainty_min_raw_score)
        )
        # REV4: only for a moderate expansion with strong visual/motion evidence,
        # do not let the scale gate erase the positional hypothesis before the
        # downstream uncertainty state can examine it.  Shrink is unchanged.
        effective_scale_gate = 1.0 if expansion_uncertain_candidate else scale_gate
        total = max(0.0, float(raw_total) * effective_scale_gate)
        ranked.append(
            TrackCandidateScore(
                candidate=candidate,
                descriptor=descriptor,
                similarity=similarity,
                motion_score=float(motion),
                size_score=float(size_score),
                total_score=float(total),
                raw_total_score=float(raw_total),
                scale_reliable_area_px=float(scale_metrics.reliable_area_px) if scale_metrics is not None else 0.0,
                scale_log_distance=float(scale_metrics.log_distance) if scale_metrics is not None else 0.0,
                scale_allowed_log_distance=float(scale_metrics.allowed_log_distance) if scale_metrics is not None else 0.0,
                scale_excess_log_distance=float(scale_metrics.excess_log_distance) if scale_metrics is not None else 0.0,
                scale_nonlinear_gate=float(scale_gate),
                scale_distance_suspicious=bool(scale_metrics.suspicious) if scale_metrics is not None else False,
                scale_distance_extreme=bool(scale_metrics.extreme) if scale_metrics is not None else False,
                scale_quick_confirmable=bool(scale_metrics.quick_confirmable) if scale_metrics is not None else True,
                scale_is_expansion=bool(scale_is_expansion),
                scale_expansion_uncertain_candidate=bool(expansion_uncertain_candidate),
                static_overlay_penalty=float(static_penalty),
                hud_probability=float(hud.probability),
                hud_hard_overlap=float(hud.hard_overlap),
                hud_soft_overlap=float(hud.soft_overlap),
                hud_hard_reject=bool(hud.hard_reject),
                hud_negative_penalty=float(hud_negative_penalty),
            )
        )
    ranked.sort(key=lambda r: (-r.total_score, -r.similarity.total, r.candidate.distance_to_click_px))
    return ranked


def _gate_reasons(
    best: TrackCandidateScore, gap: float, cfg: TrackingConfig, *, warmup: bool = False
) -> list[str]:
    reasons: list[str] = []
    min_score = cfg.warmup_min_track_score if warmup else cfg.min_track_score
    min_identity = cfg.warmup_min_identity_similarity if warmup else cfg.min_identity_similarity
    min_gap = cfg.warmup_min_confidence_gap if warmup else cfg.min_confidence_gap
    if best.total_score < min_score:
        reasons.append("SCORE_BELOW_THRESHOLD")
    if best.similarity.total < min_identity:
        reasons.append("IDENTITY_BELOW_THRESHOLD")
    if gap < min_gap:
        reasons.append("GAP_BELOW_THRESHOLD")
    return reasons


def _drift_rescue_allowed(
    best: TrackCandidateScore, gap: float, predicted_xy: tuple[float, float], cfg: TrackingConfig,
    *, consecutive_rescues: int, reasons: Sequence[str],
) -> bool:
    if not cfg.drift_rescue_enabled or consecutive_rescues >= cfg.drift_rescue_max_consecutive:
        return False
    # Never rescue an ambiguous candidate; only SCORE/IDENTITY degradation is
    # allowed to be overridden.
    if any(r not in {"SCORE_BELOW_THRESHOLD", "IDENTITY_BELOW_THRESHOLD"} for r in reasons):
        return False
    err = hypot(best.candidate.center_x - predicted_xy[0], best.candidate.center_y - predicted_xy[1])
    return bool(
        best.motion_score >= cfg.drift_rescue_min_motion_score
        and best.similarity.total >= cfg.drift_rescue_min_identity_similarity
        and gap >= cfg.drift_rescue_min_confidence_gap
        and best.size_score >= cfg.drift_rescue_min_size_score
        and err <= cfg.drift_rescue_max_prediction_error_px
    )


def _diagnostic_confidence(best: TrackCandidateScore, gap: float, cfg: TrackingConfig) -> float:
    gap_quality = min(1.0, max(0.0, gap) / max(cfg.confidence_gap_reference, cfg.min_confidence_gap, 1e-9))
    value = 0.50 * best.total_score + 0.30 * best.similarity.total + 0.20 * gap_quality
    return float(min(1.0, max(0.0, value)))


def _initial_timed_history(
    initial_history: Sequence[tuple[float, float]],
    *,
    start_frame: int,
    initial_history_frames: Sequence[int] | None,
) -> list[MotionPoint]:
    if initial_history_frames is None:
        first = start_frame - len(initial_history)
        frame_ids = list(range(first, start_frame))
    else:
        if len(initial_history_frames) != len(initial_history):
            raise ValueError("initial_history_frames length must match initial_history")
        frame_ids = [int(v) for v in initial_history_frames]
    points = [
        MotionPoint(frame_index=int(fi), x=float(xy[0]), y=float(xy[1]))
        for fi, xy in zip(frame_ids, initial_history)
    ]
    if any(b.frame_index <= a.frame_index for a, b in pairwise(points)):
        raise ValueError("initial history frame indices must be strictly increasing")
    if points and points[-1].frame_index >= start_frame:
        raise ValueError("initial history must end before start_frame")
    return points


def _tracking_incremental_engine(
    *,
    frame_reader: FrameReader,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    reference_bank: ReferenceBank,
    initial_history: Sequence[tuple[float, float]],
    detector_config: CandidateDetectorConfig,
    config: TrackingConfig,
    identity_weights: Mapping[str, float] | None = None,
    initial_history_frames: Sequence[int] | None = None,
    adaptation_adapter: EMAReferenceAdapter | None = None,
    initial_expected_area_px: float | None = None,
    thermal_profile: ThermalProfile | None = None,
    scale_model: ScaleStabilityModel | None = None,
    reliable_scale_anchor: ReliableScaleAnchor | None = None,
    continuity_state: _TrackingContinuityState | None = None,
) -> Iterator[TrackFrameResult]:
    """Track until EOF/LOST with event-driven HOTFIX-6 motion recovery.

    HOTFIX-6 adds no recurring heavy vision stage to the normal TRACK path.
    Background motion is estimated only when the previous prediction was poor,
    a miss occurred, MOTION_BREAK is active, or a visually strong candidate is
    far from the old prediction.  This preserves the HOTFIX-5 fast path.
    """
    config.validate()
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if not 0 <= start_frame <= frame_count:
        raise ValueError("start_frame outside video")
    if not initial_history:
        raise ValueError("initial_history must contain at least one center")
    if len(reference_bank) < 1:
        raise ValueError("reference_bank must not be empty")

    timed_history = _initial_timed_history(
        initial_history,
        start_frame=start_frame,
        initial_history_frames=initial_history_frames,
    )
    history_limit = max(config.velocity_window_frames + 3, 10)
    if len(timed_history) > history_limit:
        timed_history = timed_history[-history_limit:]
    miss_count = 0
    runtime_summary = {"motion_broken_at_lost": False}
    expected_area_px = float(
        initial_expected_area_px if initial_expected_area_px is not None and initial_expected_area_px > 0
        else reference_bank.expected_size_px()
    )
    if scale_model is None and config.thermal_scale_stability_enabled and thermal_profile is not None and thermal_profile.is_thermal:
        scale_model = ScaleStabilityModel(expected_area_px, ScaleStabilityConfig(
            enabled=True, confirm_frames=config.scale_confirm_frames, consistency_log_span=config.scale_consistency_log_span,
            max_log_change_per_frame=config.scale_max_log_change_per_frame, large_change_ratio=config.scale_large_change_ratio,
            large_change_confirm_frames=config.scale_large_change_confirm_frames, micro_max_area_px=config.scale_micro_max_area_px,
            micro_confirm_frames=config.scale_micro_confirm_frames, micro_exit_confirm_frames=config.scale_micro_exit_confirm_frames,
            micro_consistency_log_span=config.scale_micro_consistency_log_span, micro_max_log_change_per_frame=config.scale_micro_max_log_change_per_frame,
            robust_history_frames=config.scale_robust_history_frames,
        ))
    if scale_model is not None and scale_model.config.enabled:
        expected_area_px = float(scale_model.effective_area_px)
    if (
        reliable_scale_anchor is None
        and config.thermal_scale_parameter_distance_enabled
        and thermal_profile is not None and thermal_profile.is_thermal
    ):
        reliable_scale_anchor = ReliableScaleAnchor(
            expected_area_px, timed_history[-1].frame_index,
            ScaleContinuityConfig(
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
            ),
        )
    runtime_detector = runtime_detector_config(detector_config, thermal_profile)
    warmup_points: list[MotionPoint] = []
    drift_rescue_streak = 0

    # HOTFIX-6 state.
    recent_prediction_errors: deque[float] = deque(maxlen=max(2, config.motion_break_window_frames))
    motion_break_active = False
    motion_break_events = 0
    relock_points: list[MotionPoint] = []
    prev_raw_frame: np.ndarray | None = None
    previous_global_shift_xy: tuple[float, float] | None = None
    previous_global_shift_frame: int | None = None

    # HOTFIX-2 reliable spatial anchor: unlike normal motion history this is
    # changed only by a trusted observation or a temporally confirmed jump.
    reliable_anchor_xy: tuple[float, float] = timed_history[-1].xy
    reliable_anchor_frame = int(timed_history[-1].frame_index)
    jump_evidence: deque[dict] = deque(maxlen=max(6, config.jump_extreme_confirm_hits + config.jump_single_candidate_extra_hits + 2))
    scale_evidence: deque[dict] = deque(maxlen=max(8, config.scale_quarantine_extreme_confirm_hits + 3))

    # HOTFIX-4 frame-reliability state.  The reference frame is updated only
    # when image quality is good, preventing consecutive artefact frames from
    # legitimising one another.
    quality_reference_frame: np.ndarray | None = None
    frame_degraded_active = False
    frame_degraded_frames = 0
    quality_cfg = FrameReliabilityConfig(
        enabled=bool(config.thermal_frame_reliability_enabled and thermal_profile is not None and thermal_profile.is_thermal),
        max_dimension_px=config.frame_reliability_max_dimension_px,
        residual_scale=config.frame_reliability_residual_scale,
        phase_response_floor=config.frame_reliability_phase_response_floor,
        phase_response_good=config.frame_reliability_phase_response_good,
        phase_weight=config.frame_reliability_phase_weight,
        enter_threshold=config.frame_reliability_enter_threshold,
        exit_threshold=config.frame_reliability_exit_threshold,
        max_hold_frames=config.frame_reliability_max_hold_frames,
    )

    # HOTFIX-3 reticle-occlusion state.  During occlusion the last reliable
    # anchor, scale and adaptive identity memory are frozen.  Motion prediction
    # continues from pre-occlusion trusted points until the target emerges.
    reticle_occlusion_active = False
    reticle_occlusion_frames = 0
    reticle_exit_evidence: deque[dict] = deque(maxlen=max(3, config.hud_exit_confirm_hits + 1))

    budget_ms = 1000.0 / max(1e-9, config.performance_target_fps)
    bg_cfg = BackgroundMotionConfig(
        enabled=bool(config.global_shift_enabled),
        downscale=float(config.global_shift_downscale),
        max_dimension_px=int(config.global_shift_max_dimension_px),
        phase_min_response=float(config.global_shift_phase_min_response),
        phase_direct_max_fraction=float(config.global_shift_phase_direct_max_fraction),
        phase_max_change_from_previous_px=float(config.global_shift_phase_max_change_from_previous_px),
        use_feature_fallback=bool(config.global_shift_feature_fallback),
        target_exclusion_radius_px=max(18, int(round(max(1.0, sqrt(expected_area_px)) * 1.5))),
    )

    def publish_continuity_state() -> None:
        if continuity_state is None:
            return
        continuity_state.motion_history = tuple(timed_history)
        continuity_state.expected_area_px = float(expected_area_px)
        continuity_state.motion_broken_at_lost = bool(
            runtime_summary["motion_broken_at_lost"]
        )
        continuity_state.previous_frame = prev_raw_frame
        continuity_state.quality_reference_frame = quality_reference_frame
        continuity_state.scale_model = scale_model
        continuity_state.reliable_scale_anchor = reliable_scale_anchor
        continuity_state.motion_break_events = int(motion_break_events)

    def clipped(xy: tuple[float, float]) -> tuple[float, float]:
        return (
            min(max(float(xy[0]), 0.0), frame_width - 1.0),
            min(max(float(xy[1]), 0.0), frame_height - 1.0),
        )

    for frame_index in range(start_frame, frame_count):
        frame_t0 = perf_counter()
        _idx, frame = frame_reader(frame_index)
        analysis_frame = preprocess_frame(frame, thermal_profile)
        warmup = (frame_index - start_frame) < config.track_warmup_frames

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

        tm0 = perf_counter()
        if motion_break_active:
            prediction_history = relock_points if relock_points else timed_history[-1:]
        else:
            prediction_history = warmup_points if warmup and len(warmup_points) >= 2 else timed_history
        estimate = predict_position_timed(
            prediction_history,
            target_frame_index=frame_index,
            fallback=timed_history[-1].xy,
            window_points=config.velocity_window_frames,
        )
        predicted_raw = estimate.predicted_xy
        if (warmup and len(warmup_points) < 2) or (motion_break_active and len(relock_points) < 2):
            predicted_raw = prediction_history[-1].xy
        predicted = clipped(predicted_raw)
        motion_time_ms = (perf_counter() - tm0) * 1000.0

        if config.dynamic_search_enabled:
            base_radius = max(config.search_radius_px, config.warmup_search_radius_px) if warmup else config.search_radius_px
            radius = dynamic_search_radius(
                base_radius_px=base_radius,
                min_radius_px=config.min_search_radius_px,
                max_radius_px=max(config.max_search_radius_px, config.warmup_search_radius_px) if warmup else config.max_search_radius_px,
                speed_px_per_frame=estimate.speed_px_per_frame,
                uncertainty_px=estimate.uncertainty_px,
                miss_count=0 if warmup else miss_count,
                speed_gain=config.speed_radius_gain,
                uncertainty_gain=config.uncertainty_radius_gain,
                miss_growth_px=config.miss_radius_growth_px,
            )
        else:
            radius = int(max(config.search_radius_px, config.warmup_search_radius_px) if warmup else config.search_radius_px)

        global_est = None
        shift_attempted = False
        detector_time_ms = 0.0
        identity_time_ms = 0.0

        def try_global_shift() -> bool:
            nonlocal global_est, shift_attempted, predicted, previous_global_shift_xy, previous_global_shift_frame
            if shift_attempted or not config.global_shift_enabled or prev_raw_frame is None:
                return False
            shift_attempted = True
            target_xy = timed_history[-1].xy if timed_history else None
            target_radius = max(20.0, 1.8 * sqrt(max(1.0, expected_area_px)))
            global_est = estimate_background_motion(
                prev_raw_frame,
                frame,
                config=bg_cfg,
                target_xy=target_xy,
                target_radius_px=target_radius,
                reticle_mask=(thermal_profile.reticle_mask if thermal_profile is not None else None),
                previous_shift_xy=(
                    previous_global_shift_xy
                    if previous_global_shift_frame is not None and previous_global_shift_frame == frame_index - 1
                    else None
                ),
            )
            if global_est.valid:
                predicted = clipped(transform_point(predicted, global_est))
                previous_global_shift_xy = (float(global_est.dx_px), float(global_est.dy_px))
                previous_global_shift_frame = frame_index
                return True
            return False

        # Only the suspicious path pays for global motion before detection.
        prior_high = sum(1 for e in recent_prediction_errors if e >= config.global_shift_trigger_error_px)
        if config.motion_break_enabled and (motion_break_active or miss_count > 0 or prior_high >= 1):
            try_global_shift()

        def evaluate_at_prediction(pred_xy: tuple[float, float]):
            nonlocal detector_time_ms, identity_time_ms
            selection = make_operator_selection(
                frame_index=frame_index,
                click_xy=pred_xy,
                radius_px=radius,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            td = perf_counter()
            include_micro = bool(
                config.thermal_scale_stability_enabled
                and thermal_profile is not None and thermal_profile.is_thermal
                and (
                    expected_area_px <= 240.0
                    or miss_count > 0
                    or (scale_model is not None and scale_model.micro_mode)
                )
            )
            detection = detect_local_candidates(
                analysis_frame, selection, top_k=config.top_k, config=runtime_detector,
                expected_area_px=float(expected_area_px), include_micro=include_micro,
            )
            detector_time_ms += (perf_counter() - td) * 1000.0
            ranked = []
            if detection.candidates:
                ti = perf_counter()
                ranked = _rank_candidates(
                    analysis_frame,
                    detection.candidates,
                    reference_bank,
                    pred_xy,
                    cfg=config,
                    identity_weights=identity_weights,
                    thermal_profile=thermal_profile,
                    identity_weight=config.warmup_identity_weight if warmup else None,
                    expected_area_px=float(expected_area_px),
                    motion_weight=config.warmup_motion_weight if warmup else None,
                    size_weight=config.warmup_size_weight if warmup else None,
                    motion_sigma_px=config.warmup_motion_sigma_px if warmup else None,
                    reliable_scale_anchor=reliable_scale_anchor,
                    frame_index=frame_index,
                )
                identity_time_ms += (perf_counter() - ti) * 1000.0
            return detection, ranked

        detection, ranked = evaluate_at_prediction(predicted)
        # If old prediction yields nothing, try one cheap camera-shift correction
        # on this same frame before declaring a miss.
        if not ranked and try_global_shift():
            detection, ranked = evaluate_at_prediction(predicted)

        phase = (
            "TRACK_DEGRADED" if frame_degraded_active
            else ("RETICLE_OCCLUSION" if reticle_occlusion_active
                  else ("TRACK_WARMUP" if warmup else ("MOTION_BREAK" if motion_break_active else "TRACK")))
        )
        scale_obs = scale_model.snapshot("NO_ACCEPT") if scale_model is not None and scale_model.config.enabled else None
        common = dict(
            frame_index=frame_index,
            predicted_xy=predicted,
            velocity_xy=(estimate.vx_px_per_frame, estimate.vy_px_per_frame),
            speed_px_per_frame=estimate.speed_px_per_frame,
            motion_uncertainty_px=estimate.uncertainty_px,
            search_radius_px=radius,
            tracking_phase=phase,
            frame_reliability_score=float(frame_quality.score),
            frame_reliability_residual=float(frame_quality.residual),
            frame_reliability_phase_response=float(frame_quality.phase_response),
            frame_degraded_active=bool(frame_degraded_active),
            frame_degraded_frames=int(frame_degraded_frames),
        )

        if not ranked:
            _degraded_hold = bool(frame_degraded_active and frame_degraded_frames <= quality_cfg.max_hold_frames)
            _hud_guard_no_candidate = bool(
                config.thermal_hud_negative_enabled
                and thermal_profile is not None and thermal_profile.is_thermal
                and bool(getattr(thermal_profile, "hud_negative_enabled", False))
                and not warmup
                and not frame_degraded_active
            )
            _pred_hard, _pred_soft = hud_point_membership(predicted, thermal_profile)
            _hud_no_candidate_hold = bool(_hud_guard_no_candidate and (_pred_soft or reticle_occlusion_active))
            if _degraded_hold:
                status = "LOW_CONFIDENCE"
                warmup_miss = miss_count
                _no_candidate_reason = f"FRAME_DEGRADED_{frame_degraded_frames}_OF_{quality_cfg.max_hold_frames}"
            elif _hud_no_candidate_hold:
                reticle_occlusion_active = True
                reticle_occlusion_frames += 1
                warmup_miss = miss_count
                status = "LOW_CONFIDENCE"
                if reticle_occlusion_frames > config.hud_occlusion_max_frames:
                    miss_count += 1
                    warmup_miss = miss_count
                    status = "LOST" if miss_count >= config.lost_after_misses else "LOW_CONFIDENCE"
                _no_candidate_reason = (
                    "RETICLE_OCCLUSION_NO_CANDIDATE"
                    if reticle_occlusion_frames <= config.hud_occlusion_max_frames else "RETICLE_OCCLUSION_TIMEOUT"
                )
            elif warmup:
                status = "LOW_CONFIDENCE"
                warmup_miss = (frame_index - start_frame) + 1
                _no_candidate_reason = "NO_CANDIDATES"
            else:
                miss_count += 1
                status = "LOST" if miss_count >= config.lost_after_misses else "LOW_CONFIDENCE"
                warmup_miss = miss_count
                _no_candidate_reason = "NO_CANDIDATES"
            adapt_event = (
                adaptation_adapter.blocked_event(
                    frame_index,
                    "BLOCKED_FRAME_DEGRADED" if frame_degraded_active
                    else ("BLOCKED_HUD_OCCLUSION" if _hud_no_candidate_hold else "BLOCKED_NOT_ACCEPTED")
                )
                if adaptation_adapter is not None and adaptation_adapter.enabled else None
            )
            total_ms = (perf_counter() - frame_t0) * 1000.0
            item = TrackFrameResult(
                **common, selected=None, selected_score=0.0, confidence_gap=0.0,
                candidate_count=0, identity_similarity=0.0, motion_score=0.0, size_score=0.0,
                status=status, miss_count=warmup_miss if warmup else miss_count,
                prediction_error_px=None, confidence=0.0, diagnostic_reason=_no_candidate_reason,
                reticle_occlusion_active=bool(reticle_occlusion_active),
                reticle_occlusion_frames=int(reticle_occlusion_frames),
                detector_threshold=float(detection.diagnostics.threshold),
                detector_median_response=float(detection.diagnostics.median_response),
                detector_robust_sigma=float(detection.diagnostics.robust_sigma),
                detector_raw_components=int(detection.diagnostics.raw_components),
                detector_valid_components=int(detection.diagnostics.valid_components),
                expected_area_px=float(expected_area_px),
                reliable_area_anchor_px=(float(reliable_scale_anchor.reliable_area_px) if reliable_scale_anchor is not None else float(expected_area_px)),
                scale_anchor_age_frames=(max(0, int(frame_index - reliable_scale_anchor.reliable_frame)) if reliable_scale_anchor is not None else 0),
                detector_effective_min_area_px=int(detection.diagnostics.effective_min_area_px),
                detector_effective_max_area_px=int(detection.diagnostics.effective_max_area_px),
                adaptation_updated=False,
                adaptation_reason=adapt_event.reason if adapt_event is not None else "ADAPT_OFF",
                adaptation_update_count=adapt_event.update_count if adapt_event is not None else 0,
                adaptation_layer=adapt_event.layer if adapt_event is not None else "NONE",
                adaptation_drift_score=adapt_event.drift_score if adapt_event is not None else 0.0,
                adaptation_drift_reason=adapt_event.drift_reason if adapt_event is not None else "",
                adaptation_recent_count=adapt_event.recent_count if adapt_event is not None else 0,
                adaptation_fast_remaining=adapt_event.fast_remaining if adapt_event is not None else 0,
                motion_break_active=motion_break_active,
                motion_break_triggered=False,
                motion_break_reason="ACTIVE_NO_CANDIDATE" if motion_break_active else "",
                global_shift_dx_px=float(global_est.dx_px) if global_est is not None else 0.0,
                global_shift_dy_px=float(global_est.dy_px) if global_est is not None else 0.0,
                global_shift_response=float(global_est.response) if global_est is not None else 0.0,
                global_shift_method=global_est.method if global_est is not None else "OFF",
                global_shift_time_ms=float(global_est.total_time_ms) if global_est is not None else 0.0,
                detector_time_ms=detector_time_ms, identity_time_ms=identity_time_ms,
                motion_time_ms=motion_time_ms, performance_budget_ms=budget_ms,
                over_budget=total_ms > budget_ms, processing_time_ms=total_ms,
            )
            prev_raw_frame = frame.copy()
            if quality_reference_frame is None or (not frame_degraded_active and frame_quality.score >= quality_cfg.exit_threshold):
                quality_reference_frame = frame.copy()
            if status == "LOST":
                runtime_summary["motion_broken_at_lost"] = bool(
                    motion_break_active or prior_high >= 1
                )
                publish_continuity_state()
                yield item
                return
            publish_continuity_state()
            yield item
            continue

        best = ranked[0]
        second_score = float(ranked[1].total_score) if len(ranked) > 1 else 0.0
        has_competitor = len(ranked) > 1
        gap = best.total_score - ranked[1].total_score if has_competitor else best.total_score
        gap = max(0.0, float(gap))
        # No runner-up means no measured uniqueness. Keep legacy gap for normal
        # TRACK, but use neutral zero evidence for a suspicious far jump.
        jump_gap = gap if has_competitor else 0.0
        best_prediction_error = float(hypot(best.candidate.center_x - predicted[0], best.candidate.center_y - predicted[1]))

        # A large visual displacement is exactly where global camera motion can
        # help. Re-estimate once and re-rank, but only on that event.
        if (
            config.global_shift_enabled and not shift_attempted and prev_raw_frame is not None
            and best_prediction_error >= config.global_shift_trigger_error_px
            and best.similarity.total >= min(0.40, config.min_identity_similarity)
        ):
            old_best = best
            old_err = best_prediction_error
            if try_global_shift():
                detection2, ranked2 = evaluate_at_prediction(predicted)
                if ranked2:
                    new_best = ranked2[0]
                    new_err = hypot(new_best.candidate.center_x - predicted[0], new_best.candidate.center_y - predicted[1])
                    if new_err + 5.0 < old_err or new_best.total_score >= old_best.total_score - 0.02:
                        detection, ranked, best = detection2, ranked2, new_best
                        second_score = float(ranked[1].total_score) if len(ranked) > 1 else 0.0
                        has_competitor = len(ranked) > 1
                        gap = best.total_score - ranked[1].total_score if has_competitor else best.total_score
                        gap = max(0.0, float(gap))
                        jump_gap = gap if has_competitor else 0.0
                        best_prediction_error = float(new_err)

        # Anchor snapshot used for this frame's decision/diagnostics. The mutable
        # ReliableAnchor may be updated only after a trusted acceptance below.
        anchor_reference_xy = reliable_anchor_xy

        # HOTFIX-2: local continuity gets first refusal. If the raw Top-1 is a
        # large jump but a candidate still exists inside the ReliableAnchor /
        # prediction neighbourhood and passes the normal TRACK gates, prefer
        # that local hypothesis unless its identity is dramatically worse.
        # Far bright tree blobs therefore cannot suppress a still-visible local
        # target merely by having a slightly higher thermal identity score.
        anchor_sigma_for_local = max(
            float(config.jump_anchor_sigma_floor_px),
            float(estimate.uncertainty_px) + 0.50 * sqrt(max(1.0, float(expected_area_px))),
        )
        local_scope_px = min(
            float(config.jump_extreme_distance_px),
            max(float(config.jump_quarantine_distance_px),
                float(config.jump_anchor_norm_threshold) * anchor_sigma_for_local),
        )
        anchor_continuity_override = False
        thermal_guard_now = bool(
            config.thermal_jump_quarantine_enabled
            and thermal_profile is not None and thermal_profile.is_thermal and not warmup
        )
        raw_top_anchor_dist = float(hypot(best.candidate.center_x - anchor_reference_xy[0],
                                          best.candidate.center_y - anchor_reference_xy[1]))
        raw_top_suspicious = bool(
            thermal_guard_now
            and (best_prediction_error >= config.jump_quarantine_distance_px
                 or raw_top_anchor_dist >= local_scope_px)
        )
        if raw_top_suspicious and len(ranked) > 1:
            local_ranked = [
                rc for rc in ranked
                if (hypot(rc.candidate.center_x - anchor_reference_xy[0], rc.candidate.center_y - anchor_reference_xy[1]) <= local_scope_px
                    or hypot(rc.candidate.center_x - predicted[0], rc.candidate.center_y - predicted[1]) <= config.jump_quarantine_distance_px)
            ]
            if local_ranked:
                local_ranked.sort(key=lambda rc: (-rc.total_score, -rc.similarity.total, rc.candidate.distance_to_click_px))
                local_best = local_ranked[0]
                if local_best is not best:
                    local_has_competitor = len(local_ranked) > 1
                    local_gap = (max(0.0, float(local_best.total_score - local_ranked[1].total_score))
                                 if local_has_competitor
                                 else (config.min_confidence_gap if float(local_best.motion_score) >= 0.60 else 0.0))
                    local_reasons = _gate_reasons(local_best, local_gap, config, warmup=False)
                    identity_deficit = float(best.similarity.total - local_best.similarity.total)
                    if (not local_reasons
                            and identity_deficit <= config.jump_local_preference_max_identity_deficit
                            and float(local_best.size_score) >= config.jump_min_size_score):
                        best = local_best
                        best_prediction_error = float(hypot(best.candidate.center_x - predicted[0], best.candidate.center_y - predicted[1]))
                        has_competitor = local_has_competitor
                        second_score = float(local_ranked[1].total_score) if local_has_competitor else 0.0
                        gap = float(local_gap)
                        jump_gap = gap if local_has_competitor else 0.0
                        anchor_continuity_override = True

        # HOTFIX-2: continuous-coordinate analogue of a Hamming-distance guard.
        # anchor_distance_norm is a transparent isotropic Mahalanobis-like
        # innovation: distance from ReliableAnchor divided by uncertainty/scale.
        anchor_distance_px = float(hypot(best.candidate.center_x - anchor_reference_xy[0], best.candidate.center_y - anchor_reference_xy[1]))
        anchor_sigma_px = max(
            float(config.jump_anchor_sigma_floor_px),
            float(estimate.uncertainty_px) + 0.50 * sqrt(max(1.0, float(expected_area_px))),
        )
        anchor_age_frames = max(0, int(frame_index - reliable_anchor_frame))
        anchor_growth_px = 0.0
        if config.anchor_time_normalization_enabled:
            v_allow = max(
                float(config.anchor_time_min_speed_px_per_frame),
                min(float(config.anchor_time_max_speed_px_per_frame),
                    float(config.anchor_time_speed_multiplier) * max(0.0, float(estimate.speed_px_per_frame))),
            )
            anchor_growth_px = min(float(config.anchor_time_max_growth_px), v_allow * anchor_age_frames)
        # Preserve HOTFIX-2 threshold semantics: suspicious when norm >=
        # jump_anchor_norm_threshold.  Time growth is added to the corresponding
        # allowed distance, not used to excuse an instantaneous jump.
        norm_threshold = max(1e-6, float(config.jump_anchor_norm_threshold))
        anchor_norm_sigma_px = anchor_sigma_px + anchor_growth_px / norm_threshold
        anchor_distance_norm = float(anchor_distance_px / max(1e-9, anchor_norm_sigma_px))
        anchor_allowed_distance_px = float(norm_threshold * anchor_norm_sigma_px)
        thermal_jump_guard = bool(
            config.thermal_jump_quarantine_enabled
            and thermal_profile is not None and thermal_profile.is_thermal
            and not warmup
            and not frame_degraded_active
        )
        jump_suspicious = bool(
            thermal_jump_guard
            and (best_prediction_error >= config.jump_quarantine_distance_px
                 or anchor_distance_norm >= config.jump_anchor_norm_threshold)
        )
        jump_clutter = float(getattr(best.descriptor, "thermal_background_clutter", 0.0))
        jump_adjusted_identity = max(0.0, float(best.similarity.total) - config.jump_clutter_penalty_weight * jump_clutter)
        jump_global_motion_veto = bool(
            jump_suspicious
            and global_est is not None and bool(getattr(global_est, "valid", False))
            and float(global_est.response) >= config.jump_global_motion_veto_response_min
            and best_prediction_error >= config.jump_global_motion_veto_residual_px
        )
        jump_quarantine_hits = 0
        jump_quarantine_required_hits = 0
        jump_quarantine_confirmed = False
        jump_admissible = False
        if jump_suspicious:
            jump_admissible = bool(
                jump_adjusted_identity >= config.jump_min_identity
                and float(best.size_score) >= config.jump_min_size_score
                and float(best.static_overlay_penalty) <= config.jump_max_static_overlay_penalty
                and (not has_competitor or jump_gap >= config.jump_min_gap)
            )
            if jump_admissible:
                if jump_evidence:
                    prev = jump_evidence[-1]
                    link = hypot(best.candidate.center_x - float(prev["x"]), best.candidate.center_y - float(prev["y"]))
                    if link > config.jump_link_distance_px or frame_index != int(prev["frame"]) + 1:
                        jump_evidence.clear()
                jump_evidence.append({
                    "frame": int(frame_index), "x": float(best.candidate.center_x), "y": float(best.candidate.center_y),
                    "identity": float(jump_adjusted_identity), "gap": float(jump_gap),
                    "has_competitor": bool(has_competitor), "clutter": float(jump_clutter),
                })
                extreme = max(best_prediction_error, anchor_distance_px) >= config.jump_extreme_distance_px
                jump_quarantine_required_hits = int(config.jump_extreme_confirm_hits if extreme else config.jump_confirm_hits)
                if not any(bool(e["has_competitor"]) for e in jump_evidence):
                    jump_quarantine_required_hits += int(config.jump_single_candidate_extra_hits)
                if jump_global_motion_veto:
                    jump_quarantine_required_hits += 1
                jump_quarantine_hits = len(jump_evidence)
                recent_jump = list(jump_evidence)[-jump_quarantine_required_hits:]
                if len(recent_jump) >= jump_quarantine_required_hits:
                    med_identity = float(median(float(e["identity"]) for e in recent_jump))
                    competing_gaps = [float(e["gap"]) for e in recent_jump if bool(e["has_competitor"])]
                    gap_ok = (not competing_gaps) or float(median(competing_gaps)) >= config.jump_min_gap
                    jump_quarantine_confirmed = bool(med_identity >= config.jump_confirm_min_identity and gap_ok)
            else:
                jump_evidence.clear()
        elif jump_evidence:
            jump_evidence.clear()

        # HOTFIX-4 REV3: nonlinear distance in the SCALE parameter.  CurrentArea
        # remains useful for candidate generation, while ReliableAreaAnchor is a
        # separate trusted reference that cannot be silently dragged to a tiny
        # reticle. Gross mismatch is quarantined instead of being compensated by
        # a high visual Identity or motion score.
        scale_guard_now = bool(
            config.thermal_scale_parameter_distance_enabled
            and reliable_scale_anchor is not None and reliable_scale_anchor.config.enabled
            and thermal_profile is not None and thermal_profile.is_thermal
            and not frame_degraded_active
        )
        scale_suspicious = bool(scale_guard_now and best.scale_distance_suspicious)
        scale_quarantine_hits = 0
        scale_quarantine_required_hits = 0
        scale_quarantine_confirmed = False
        scale_quarantine_admissible = False
        scale_gap = (
            max(0.0, float(best.similarity.total - ranked[1].similarity.total))
            if len(ranked) > 1 else 0.0
        )
        scale_expansion_uncertain = bool(
            scale_suspicious
            and bool(getattr(best, "scale_expansion_uncertain_candidate", False))
            and best_prediction_error <= config.scale_expansion_uncertainty_max_prediction_error_px
            and (not has_competitor or scale_gap >= config.scale_expansion_uncertainty_min_gap)
            and not bool(getattr(best, "hud_hard_reject", False))
        )
        if scale_suspicious:
            scale_quarantine_admissible = bool(
                float(best.similarity.total) >= config.scale_quarantine_min_identity
                and best_prediction_error <= config.scale_quarantine_max_prediction_error_px
                and (not has_competitor or scale_gap >= config.scale_quarantine_min_gap)
                and not bool(getattr(best, "hud_hard_reject", False))
            )
            if scale_quarantine_admissible:
                log_area = float(np.log(max(1.0, float(best.candidate.area_px))))
                if scale_evidence:
                    prev = scale_evidence[-1]
                    link_xy = hypot(best.candidate.center_x - float(prev["x"]), best.candidate.center_y - float(prev["y"]))
                    link_scale = abs(log_area - float(prev["log_area"]))
                    if (frame_index != int(prev["frame"]) + 1
                            or link_xy > config.scale_quarantine_link_distance_px
                            or link_scale > config.scale_quarantine_link_log_span):
                        scale_evidence.clear()
                scale_evidence.append({
                    "frame": int(frame_index), "x": float(best.candidate.center_x), "y": float(best.candidate.center_y),
                    "area": float(max(1, best.candidate.area_px)), "log_area": log_area,
                    "identity": float(best.similarity.total), "gap": float(scale_gap),
                    "has_competitor": bool(has_competitor),
                })
                scale_quarantine_required_hits = int(
                    config.scale_quarantine_extreme_confirm_hits
                    if best.scale_distance_extreme else config.scale_quarantine_confirm_hits
                )
                scale_quarantine_hits = len(scale_evidence)
                recent_scale = list(scale_evidence)[-scale_quarantine_required_hits:]
                if len(recent_scale) >= scale_quarantine_required_hits and bool(best.scale_quick_confirmable):
                    med_identity = float(median(float(e["identity"]) for e in recent_scale))
                    measured_scale_gaps = [float(e["gap"]) for e in recent_scale if bool(e["has_competitor"])]
                    gap_ok = (not measured_scale_gaps) or float(median(measured_scale_gaps)) >= config.scale_quarantine_min_gap
                    logs = [float(e["log_area"]) for e in recent_scale]
                    stable_scale = (max(logs) - min(logs)) <= config.scale_quarantine_link_log_span
                    scale_quarantine_confirmed = bool(
                        med_identity >= config.scale_quarantine_min_identity and gap_ok and stable_scale
                    )
            else:
                scale_evidence.clear()
        elif scale_evidence:
            scale_evidence.clear()

        # HOTFIX-3 negative HUD / reticle logic.  The mask defines a known
        # nuisance object.  Outside a predicted crossing it is a hard negative;
        # if the genuine motion prediction itself enters the guard band, we
        # represent the event as temporary occlusion and wait for target exit.
        thermal_hud_guard = bool(
            config.thermal_hud_negative_enabled
            and thermal_profile is not None and thermal_profile.is_thermal
            and bool(getattr(thermal_profile, "hud_negative_enabled", False))
            and not warmup
            and not frame_degraded_active
        )
        predicted_hud_hard, predicted_hud_soft = hud_point_membership(predicted, thermal_profile)
        best_hud_probability = float(getattr(best, "hud_probability", 0.0))
        best_hud_hard = bool(getattr(best, "hud_hard_reject", False) or best_hud_probability >= config.hud_hard_probability_threshold)
        hud_candidate_near_prediction = bool(best_prediction_error <= config.hud_occlusion_prediction_tolerance_px)
        hud_occlusion_hold = bool(
            thermal_hud_guard
            and predicted_hud_soft
            and hud_candidate_near_prediction
            and best_hud_probability >= config.hud_soft_probability_threshold
        )
        hud_hard_reject = bool(
            thermal_hud_guard
            and best_hud_hard
            and not hud_occlusion_hold
        )

        # While already occluded, a clean non-HUD candidate near the propagated
        # motion path must be seen on more than one frame before TRACK resumes.
        hud_exit_candidate = bool(
            thermal_hud_guard and reticle_occlusion_active
            and best_hud_probability <= config.hud_exit_max_probability
            and float(best.similarity.total) >= config.hud_exit_min_identity
            and best_prediction_error <= config.hud_occlusion_prediction_tolerance_px
        )
        reticle_exit_hits = 0
        reticle_exit_confirmed = False
        if hud_exit_candidate:
            if reticle_exit_evidence:
                prev_exit = reticle_exit_evidence[-1]
                link = hypot(best.candidate.center_x - float(prev_exit["x"]), best.candidate.center_y - float(prev_exit["y"]))
                if link > config.hud_exit_link_distance_px or frame_index != int(prev_exit["frame"]) + 1:
                    reticle_exit_evidence.clear()
            reticle_exit_evidence.append({"frame": int(frame_index), "x": float(best.candidate.center_x), "y": float(best.candidate.center_y)})
            reticle_exit_hits = len(reticle_exit_evidence)
            reticle_exit_confirmed = reticle_exit_hits >= config.hud_exit_confirm_hits
        elif reticle_occlusion_active and not hud_occlusion_hold:
            reticle_exit_evidence.clear()

        reasons = _gate_reasons(best, gap, config, warmup=warmup)
        if frame_degraded_active:
            reasons.append(f"FRAME_DEGRADED_Q_{frame_quality.score:.3f}")
        if scale_suspicious and not scale_quarantine_confirmed and not scale_expansion_uncertain:
            if scale_quarantine_admissible:
                suffix = "WAIT_DISTANCE" if not bool(best.scale_quick_confirmable) else f"{scale_quarantine_hits}_OF_{max(1, scale_quarantine_required_hits)}"
                reasons.append(f"SCALE_QUARANTINE_{suffix}")
            else:
                reasons.append("SCALE_QUARANTINE_REJECTED")
        if scale_quarantine_confirmed:
            reasons = [r for r in reasons if r != "SCORE_BELOW_THRESHOLD"]
        if jump_suspicious and not jump_quarantine_confirmed:
            if jump_admissible:
                reasons.append(f"JUMP_QUARANTINE_{jump_quarantine_hits}_OF_{max(1, jump_quarantine_required_hits)}")
            else:
                reasons.append("JUMP_QUARANTINE_REJECTED")
        if config.identity_first_safe_warmup_enabled and warmup and best_prediction_error > config.warmup_max_unexplained_jump_px:
            # Global shift, when valid, has already corrected ``predicted`` and
            # triggered a re-rank above. A still-large residual jump is unsafe.
            reasons.append("WARMUP_LARGE_JUMP_UNEXPLAINED")
        rescue = (
            (not warmup) and bool(reasons)
            and _drift_rescue_allowed(best, gap, predicted, config, consecutive_rescues=drift_rescue_streak, reasons=reasons)
        )
        if rescue:
            reasons = []

        # HOTFIX-6 visual motion-break rescue.  It cannot override identity or
        # ambiguity: only a strong, distinctive candidate far from stale motion
        # can invalidate the old trajectory.
        visual_break = bool(
            config.motion_break_enabled and not warmup and not frame_degraded_active
            and best_prediction_error >= config.motion_break_error_px
            and best.similarity.total >= config.motion_break_visual_identity_min
            and gap >= config.motion_break_visual_gap_min
            and best.size_score >= config.motion_break_visual_size_min
            and (not jump_suspicious or jump_quarantine_confirmed)
            and (not scale_suspicious or scale_quarantine_confirmed)
        )
        if jump_quarantine_confirmed and not frame_degraded_active:
            reasons = [r for r in reasons if r.startswith("SCALE_") or r.startswith("HUD_") or r.startswith("FRAME_")]
            visual_break = bool(not scale_suspicious or scale_quarantine_confirmed)
        if visual_break:
            # All strong-visual conditions are checked explicitly above. Score
            # may be low solely because motion prior is stale.
            reasons = [
                r for r in reasons
                if r in {"IDENTITY_BELOW_THRESHOLD", "GAP_BELOW_THRESHOLD"}
                or r.startswith("SCALE_") or r.startswith("HUD_") or r.startswith("FRAME_")
            ]

        # Negative HUD evidence is safety evidence and cannot be cleared by
        # drift rescue or visual motion-break logic.
        if hud_hard_reject:
            reasons.append("HUD_HARD_REJECT")
        if reticle_occlusion_active and hud_exit_candidate and not reticle_exit_confirmed:
            reasons.append(f"HUD_EXIT_CONFIRM_{reticle_exit_hits}_OF_{config.hud_exit_confirm_hits}")
        if reticle_exit_confirmed:
            # Exit confirmation has its own identity/spatial checks above; do
            # not let a stale gap/score caused by the guard band keep us blind.
            reasons = [r for r in reasons if r == "HUD_HARD_REJECT" or r.startswith("SCALE_")]

        confidence = _diagnostic_confidence(best, gap, config)
        adapt_event = None
        motion_break_triggered = False
        motion_break_reason = ""

        if hud_occlusion_hold and not reticle_exit_confirmed:
            # Genuine trajectory is expected to pass through a learned overlay.
            # Do not accept the overlay as the target and do not train on it.
            reticle_occlusion_active = True
            reticle_occlusion_frames += 1
            selected = None
            prediction_error = None
            drift_rescue_streak = 0
            status = "LOW_CONFIDENCE"
            reason = f"RETICLE_OCCLUSION_{reticle_occlusion_frames}_OF_{config.hud_occlusion_max_frames}"
            # During the bounded occlusion grace period, do not count misses.
            # After the limit expires normal LOST logic is allowed to resume.
            if reticle_occlusion_frames > config.hud_occlusion_max_frames:
                miss_count += 1
                reason = "RETICLE_OCCLUSION_TIMEOUT"
                status = "LOST" if miss_count >= config.lost_after_misses else "LOW_CONFIDENCE"
            if adaptation_adapter is not None and adaptation_adapter.enabled:
                adapt_event = adaptation_adapter.blocked_event(frame_index, "BLOCKED_HUD_OCCLUSION")
        elif not reasons:
            miss_count = 0
            selected = best.candidate
            point = MotionPoint(frame_index, selected.center_x, selected.center_y)
            accepted_after_hud_occlusion = bool(reticle_occlusion_active and reticle_exit_confirmed)
            if accepted_after_hud_occlusion:
                reticle_occlusion_active = False
                reticle_occlusion_frames = 0
                reticle_exit_evidence.clear()
            if rescue:
                drift_rescue_streak += 1
            else:
                drift_rescue_streak = 0
            prediction_error = float(best_prediction_error)
            recent_prediction_errors.append(prediction_error)
            high_hits = sum(1 for e in recent_prediction_errors if e >= config.motion_break_error_px)
            history_break = bool(
                config.motion_break_enabled
                and len(recent_prediction_errors) >= config.motion_break_hits_required
                and high_hits >= config.motion_break_hits_required
            )
            if jump_quarantine_confirmed or visual_break or history_break:
                motion_break_triggered = not motion_break_active
                if motion_break_triggered:
                    motion_break_events += 1
                motion_break_active = True
                motion_break_reason = (
                    "JUMP_QUARANTINE_CONFIRMED" if jump_quarantine_confirmed
                    else ("VISUAL_JUMP" if visual_break else "PREDICTION_ERROR_WINDOW")
                )
                relock_points = [point]
                timed_history = [point]
                recent_prediction_errors.clear()
            elif motion_break_active:
                relock_points.append(point)
                timed_history = list(relock_points[-max(config.velocity_window_frames + 2, config.motion_relock_frames):])
                if len(relock_points) >= config.motion_relock_frames:
                    motion_break_active = False
                    motion_break_reason = "MOTION_RELOCKED"
                    timed_history = list(relock_points[-max(config.velocity_window_frames + 2, config.motion_relock_frames):])
                    relock_points = []
                    recent_prediction_errors.clear()
            else:
                timed_history.append(point)
                max_history = max(config.velocity_window_frames + 3, 10)
                if len(timed_history) > max_history:
                    timed_history = timed_history[-max_history:]

            if warmup:
                warmup_points.append(point)
                warmup_limit = max(config.velocity_window_frames + 2, 4)
                if len(warmup_points) > warmup_limit:
                    del warmup_points[:-warmup_limit]
            trusted_scale = (
                float(best.similarity.total) >= config.size_update_min_identity
                and float(gap) >= config.size_update_min_gap
                and prediction_error <= config.size_update_max_prediction_error_px
                and not motion_break_triggered
                and not best_hud_hard
                and best_hud_probability <= config.hud_soft_probability_threshold
                and (not warmup or prediction_error <= config.warmup_max_unexplained_jump_px)
                and (not scale_suspicious or scale_quarantine_confirmed)
            )
            trusted_anchor = bool(
                jump_quarantine_confirmed
                or (
                    prediction_error <= config.reliable_anchor_update_max_prediction_error_px
                    and float(best.similarity.total) >= config.size_update_min_identity
                    and ((has_competitor and gap >= config.min_confidence_gap)
                         or ((not has_competitor) and float(best.motion_score) >= 0.75))
                    and not best_hud_hard
                    and best_hud_probability <= config.hud_soft_probability_threshold
                )
            )
            if trusted_anchor:
                reliable_anchor_xy = (float(selected.center_x), float(selected.center_y))
                reliable_anchor_frame = int(frame_index)
            if reliable_scale_anchor is not None and reliable_scale_anchor.config.enabled:
                if scale_quarantine_confirmed:
                    recent_areas = [float(e["area"]) for e in list(scale_evidence)[-max(1, scale_quarantine_required_hits):]]
                    if bool(getattr(best, "scale_is_expansion", False)) and recent_areas:
                        # REV4: a confirmed expansion is followed gradually. This
                        # prevents a few smeared frames from redefining the scale
                        # anchor in one step, while still allowing a real zoom-in
                        # or approach to become the new normal over time.
                        target_area = float(median(recent_areas))
                        reliable_scale_anchor.update_trusted(target_area, frame_index)
                    else:
                        reliable_scale_anchor.commit_confirmed_transition(recent_areas, frame_index)
                    scale_evidence.clear()
                elif trusted_scale and trusted_anchor:
                    reliable_scale_anchor.update_trusted(float(max(1, selected.area_px)), frame_index)
            if scale_model is not None and scale_model.config.enabled:
                scale_obs = scale_model.observe(
                    frame_index=frame_index, area_px=float(max(1, selected.area_px)), trusted=trusted_scale,
                    is_micro=(getattr(selected, "detection_channel", "NORMAL") == "MICRO"),
                )
                expected_area_px = float(scale_model.effective_area_px)
            elif trusted_scale:
                a_area = float(config.expected_area_update_alpha)
                proposed_area = (1.0 - a_area) * expected_area_px + a_area * float(max(1, selected.area_px))
                step = float(config.size_update_max_step_ratio)
                expected_area_px = min(expected_area_px * step, max(expected_area_px / step, proposed_area))
            status = "TRACK"
            if accepted_after_hud_occlusion:
                reason = "ACCEPTED_HUD_EXIT_REACQ"
            elif scale_quarantine_confirmed:
                reason = "ACCEPTED_SCALE_QUARANTINE_CONFIRMED"
            elif scale_expansion_uncertain:
                reason = "ACCEPTED_SCALE_EXPANSION_UNCERTAIN"
            elif jump_quarantine_confirmed:
                reason = "ACCEPTED_JUMP_QUARANTINE_CONFIRMED"
            elif motion_break_triggered:
                reason = "ACCEPTED_MOTION_BREAK"
            elif motion_break_reason == "MOTION_RELOCKED":
                reason = "ACCEPTED_MOTION_RELOCK"
            elif motion_break_active:
                reason = "ACCEPTED_MOTION_RELOCKING"
            elif rescue:
                reason = "ACCEPTED_DRIFT_RESCUE"
            elif anchor_continuity_override:
                reason = "ACCEPTED_ANCHOR_CONTINUITY"
            elif warmup:
                reason = "ACCEPTED_TRACK_WARMUP"
            else:
                reason = "ACCEPTED" if prediction_error <= config.prediction_error_warn_px else "ACCEPTED_PREDICTION_ERROR_HIGH"
            if adaptation_adapter is not None and adaptation_adapter.enabled:
                if best_hud_probability >= config.hud_soft_probability_threshold:
                    adapt_event = adaptation_adapter.blocked_event(frame_index, "BLOCKED_HUD_NEGATIVE")
                elif scale_expansion_uncertain and not scale_quarantine_confirmed:
                    adapt_event = adaptation_adapter.blocked_event(frame_index, "BLOCKED_SCALE_EXPANSION_UNCERTAIN")
                elif scale_quarantine_confirmed:
                    adapt_event = adaptation_adapter.blocked_event(frame_index, "BLOCKED_SCALE_RELOCK")
                elif jump_quarantine_confirmed:
                    adapt_event = adaptation_adapter.blocked_event(frame_index, "BLOCKED_JUMP_RELOCK")
                else:
                    adapt_event = adaptation_adapter.consider(
                        frame_index=frame_index, descriptor=best.descriptor, confidence=confidence,
                        identity_similarity=float(best.similarity.total), confidence_gap=gap,
                        prediction_error_px=prediction_error, motion_score=float(best.motion_score),
                        size_score=float(best.size_score), force_fast=bool(rescue or motion_break_active), phase=phase,
                    )
        else:
            selected = None
            prediction_error = None
            drift_rescue_streak = 0
            if reticle_occlusion_active and hud_exit_candidate:
                # Exit confirmation is part of the bounded occlusion protocol,
                # not a normal tracking miss. Preserve the pre-HUD anchor while
                # the clean emerging target accumulates temporal evidence.
                status = "LOW_CONFIDENCE"
                reason = "+".join(reasons)
            elif frame_degraded_active and frame_degraded_frames <= quality_cfg.max_hold_frames:
                status = "LOW_CONFIDENCE"
                reason = "+".join(reasons)
            elif warmup:
                status = "LOW_CONFIDENCE"; reason = "WARMUP_" + "+".join(reasons)
            else:
                miss_count += 1
                lost_limit = int(config.lost_after_misses)
                if jump_suspicious and jump_admissible and jump_quarantine_required_hits > 0:
                    lost_limit = max(lost_limit, int(jump_quarantine_required_hits) + 1)
                if scale_suspicious and scale_quarantine_admissible and scale_quarantine_required_hits > 0:
                    lost_limit = max(lost_limit, int(scale_quarantine_required_hits) + 1)
                status = "LOST" if miss_count >= lost_limit else "LOW_CONFIDENCE"
                reason = "+".join(reasons)
            if adaptation_adapter is not None and adaptation_adapter.enabled:
                adapt_event = adaptation_adapter.blocked_event(
                    frame_index, "BLOCKED_FRAME_DEGRADED" if frame_degraded_active else "BLOCKED_NOT_ACCEPTED"
                )
            if status == "LOST":
                runtime_summary["motion_broken_at_lost"] = bool(
                    motion_break_active or visual_break or prior_high >= 1
                )

        total_ms = (perf_counter() - frame_t0) * 1000.0
        item = TrackFrameResult(
            **common, selected=selected, selected_score=float(best.total_score), confidence_gap=gap,
            candidate_count=len(ranked), identity_similarity=float(best.similarity.total),
            motion_score=float(best.motion_score), size_score=float(best.size_score), status=status,
            miss_count=((frame_index - start_frame) + 1) if warmup and selected is None else miss_count,
            prediction_error_px=prediction_error, confidence=confidence, diagnostic_reason=reason,
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
            thermal_background_clutter=float(jump_clutter), reliable_anchor_xy=anchor_reference_xy,
            anchor_distance_px=float(anchor_distance_px), anchor_distance_norm=float(anchor_distance_norm),
            anchor_age_frames=int(anchor_age_frames), anchor_allowed_distance_px=float(anchor_allowed_distance_px),
            jump_quarantine_active=bool(jump_suspicious and not jump_quarantine_confirmed),
            jump_quarantine_hits=int(jump_quarantine_hits), jump_quarantine_required_hits=int(jump_quarantine_required_hits),
            jump_quarantine_confirmed=bool(jump_quarantine_confirmed), jump_global_motion_veto=bool(jump_global_motion_veto),
            anchor_continuity_override=bool(anchor_continuity_override),
            hud_probability=float(best_hud_probability),
            hud_hard_overlap=float(getattr(best, "hud_hard_overlap", 0.0)),
            hud_soft_overlap=float(getattr(best, "hud_soft_overlap", 0.0)),
            hud_hard_reject=bool(hud_hard_reject),
            hud_negative_penalty=float(getattr(best, "hud_negative_penalty", 0.0)),
            reticle_occlusion_active=bool(reticle_occlusion_active or hud_occlusion_hold),
            reticle_occlusion_frames=int(reticle_occlusion_frames),
            reticle_exit_hits=int(reticle_exit_hits),
            detector_threshold=float(detection.diagnostics.threshold),
            detector_median_response=float(detection.diagnostics.median_response), detector_robust_sigma=float(detection.diagnostics.robust_sigma),
            detector_raw_components=int(detection.diagnostics.raw_components), detector_valid_components=int(detection.diagnostics.valid_components),
            expected_area_px=float(expected_area_px), detector_effective_min_area_px=int(detection.diagnostics.effective_min_area_px),
            detector_effective_max_area_px=int(detection.diagnostics.effective_max_area_px),
            scale_state=(scale_obs.state if scale_obs is not None else "LEGACY"),
            scale_base_area_px=(scale_obs.base_area_px if scale_obs is not None else float(expected_area_px)),
            scale_current_area_px=(scale_obs.current_area_px if scale_obs is not None else float(expected_area_px)),
            scale_pending_count=(scale_obs.pending_count if scale_obs is not None else 0),
            scale_update_reason=(scale_obs.reason if scale_obs is not None else "LEGACY"),
            reliable_area_anchor_px=float(best.scale_reliable_area_px if best.scale_reliable_area_px > 0 else expected_area_px),
            scale_log_distance=float(best.scale_log_distance),
            scale_allowed_log_distance=float(best.scale_allowed_log_distance),
            scale_excess_log_distance=float(best.scale_excess_log_distance),
            scale_nonlinear_gate=float(best.scale_nonlinear_gate),
            scale_anchor_age_frames=(int(reliable_scale_anchor.metrics(best.candidate.area_px, frame_index).anchor_age_frames) if reliable_scale_anchor is not None and reliable_scale_anchor.config.enabled else 0),
            scale_quarantine_active=bool(scale_suspicious and not scale_quarantine_confirmed),
            scale_quarantine_hits=int(scale_quarantine_hits),
            scale_quarantine_required_hits=int(scale_quarantine_required_hits),
            scale_quarantine_confirmed=bool(scale_quarantine_confirmed),
            scale_expansion_uncertain=bool(scale_expansion_uncertain and not scale_quarantine_confirmed),
            adaptation_updated=bool(adapt_event.updated) if adapt_event is not None else False,
            adaptation_reason=adapt_event.reason if adapt_event is not None else "ADAPT_OFF",
            adaptation_update_count=adapt_event.update_count if adapt_event is not None else 0,
            adaptation_layer=adapt_event.layer if adapt_event is not None else "NONE",
            adaptation_drift_score=adapt_event.drift_score if adapt_event is not None else 0.0,
            adaptation_drift_reason=adapt_event.drift_reason if adapt_event is not None else "",
            adaptation_recent_count=adapt_event.recent_count if adapt_event is not None else 0,
            adaptation_fast_remaining=adapt_event.fast_remaining if adapt_event is not None else 0,
            motion_break_active=motion_break_active, motion_break_triggered=motion_break_triggered,
            motion_break_reason=motion_break_reason,
            global_shift_dx_px=float(global_est.dx_px) if global_est is not None else 0.0,
            global_shift_dy_px=float(global_est.dy_px) if global_est is not None else 0.0,
            global_shift_response=float(global_est.response) if global_est is not None else 0.0,
            global_shift_method=global_est.method if global_est is not None else "OFF",
            global_shift_time_ms=float(global_est.total_time_ms) if global_est is not None else 0.0,
            detector_time_ms=detector_time_ms, identity_time_ms=identity_time_ms, motion_time_ms=motion_time_ms,
            performance_budget_ms=budget_ms, over_budget=total_ms > budget_ms, processing_time_ms=total_ms,
        )
        prev_raw_frame = frame.copy()
        if quality_reference_frame is None or (not frame_degraded_active and frame_quality.score >= quality_cfg.exit_threshold):
            quality_reference_frame = frame.copy()

        if warmup and (frame_index - start_frame + 1) == config.track_warmup_frames and len(warmup_points) >= 2:
            timed_history = list(warmup_points[-max(config.velocity_window_frames + 2, 4):])
        if status == "LOST":
            publish_continuity_state()
            yield item
            return
        publish_continuity_state()
        yield item


class TrackingRuntime:
    """Bounded, persistent one-frame TRACK engine.

    The generator frame owns the mechanically lifted locals from the original
    finite-video loop.  ``step`` advances that single algorithm body exactly
    once; it never stores the emitted frame results or a full trajectory.
    """

    _COLLECTION_STATE_NAMES = (
        "timed_history",
        "warmup_points",
        "recent_prediction_errors",
        "relock_points",
        "jump_evidence",
        "scale_evidence",
        "reticle_exit_evidence",
    )
    DEFAULT_ADAPTATION_EVENT_HISTORY = 32

    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        start_frame: int,
        reference_bank: ReferenceBank,
        initial_history: Sequence[tuple[float, float]],
        detector_config: CandidateDetectorConfig,
        config: TrackingConfig,
        identity_weights: Mapping[str, float] | None = None,
        initial_history_frames: Sequence[int] | None = None,
        adaptation_adapter: EMAReferenceAdapter | None = None,
        initial_expected_area_px: float | None = None,
        thermal_profile: ThermalProfile | None = None,
        scale_model: ScaleStabilityModel | None = None,
        reliable_scale_anchor: ReliableScaleAnchor | None = None,
        frame_count: int | None = None,
        _adaptation_event_history_limit: int | None = DEFAULT_ADAPTATION_EVENT_HISTORY,
    ) -> None:
        config.validate()
        if frame_width < 1 or frame_height < 1:
            raise ValueError("frame dimensions must be >= 1")
        if start_frame < 0 or (frame_count is not None and not 0 <= start_frame <= frame_count):
            raise ValueError("start_frame outside video")
        if frame_count is not None and frame_count < 1:
            raise ValueError("frame_count must be >= 1")
        if not initial_history:
            raise ValueError("initial_history must contain at least one center")
        if len(reference_bank) < 1:
            raise ValueError("reference_bank must not be empty")
        if _adaptation_event_history_limit is not None and _adaptation_event_history_limit < 1:
            raise ValueError("adaptation event history limit must be >= 1")
        validated_history = _initial_timed_history(
            initial_history,
            start_frame=start_frame,
            initial_history_frames=initial_history_frames,
        )

        self.start_frame = int(start_frame)
        self.next_frame_index = int(start_frame)
        self.frame_count = None if frame_count is None else int(frame_count)
        self.done = bool(frame_count is not None and start_frame >= frame_count)
        self.last_result: TrackFrameResult | None = None
        self._pending_frame: np.ndarray | None = None
        self._adaptation_adapter = adaptation_adapter
        self._adaptation_event_history_limit = _adaptation_event_history_limit
        self._continuity_state = _TrackingContinuityState()
        self.initial_last_center = validated_history[-1].xy
        self.initial_expected_area_px = float(
            initial_expected_area_px
            if initial_expected_area_px is not None and initial_expected_area_px > 0
            else reference_bank.expected_size_px()
        )
        if scale_model is not None and scale_model.config.enabled:
            self.initial_expected_area_px = float(scale_model.effective_area_px)

        def frame_reader(frame_index: int) -> tuple[int, np.ndarray]:
            if self._pending_frame is None:
                raise RuntimeError("TrackingRuntime.step() did not supply a frame")
            frame = self._pending_frame
            self._pending_frame = None
            return int(frame_index), frame

        engine_frame_count = self.frame_count if self.frame_count is not None else 2**63 - 1
        self._engine = _tracking_incremental_engine(
            frame_reader=frame_reader,
            frame_count=engine_frame_count,
            frame_width=frame_width,
            frame_height=frame_height,
            start_frame=start_frame,
            reference_bank=reference_bank,
            initial_history=initial_history,
            detector_config=detector_config,
            config=config,
            identity_weights=identity_weights,
            initial_history_frames=initial_history_frames,
            adaptation_adapter=adaptation_adapter,
            initial_expected_area_px=initial_expected_area_px,
            thermal_profile=thermal_profile,
            scale_model=scale_model,
            reliable_scale_anchor=reliable_scale_anchor,
            continuity_state=self._continuity_state,
        )

    def step(self, frame: np.ndarray) -> TrackFrameResult:
        """Process exactly the next TRACK frame."""
        if self.done:
            raise RuntimeError("tracking runtime is already complete")
        if self._pending_frame is not None:
            raise RuntimeError("tracking runtime already has a pending frame")
        self._pending_frame = frame
        try:
            item = next(self._engine)
        except StopIteration as exc:
            self.done = True
            self._pending_frame = None
            raise RuntimeError("tracking runtime completed without a frame result") from exc
        self.last_result = item
        self.next_frame_index = int(item.frame_index) + 1
        if (
            self._adaptation_adapter is not None
            and self._adaptation_event_history_limit is not None
            and len(self._adaptation_adapter.events) > self._adaptation_event_history_limit
        ):
            del self._adaptation_adapter.events[: -self._adaptation_event_history_limit]
        if item.status == "LOST" or (
            self.frame_count is not None and self.next_frame_index >= self.frame_count
        ):
            self.done = True
        return item

    @property
    def persistent_state(self) -> Mapping[str, object]:
        """Expose the live algorithm state for diagnostics and bound tests."""
        frame = self._engine.gi_frame
        return {} if frame is None else dict(frame.f_locals)

    @property
    def continuity_snapshot(self) -> TrackingContinuitySnapshot:
        """Return the typed, read-only TRACK transition boundary."""

        def readonly_copy(frame: np.ndarray | None) -> np.ndarray | None:
            if frame is None:
                return None
            copied = frame.copy()
            copied.setflags(write=False)
            return copied

        state = self._continuity_state
        return TrackingContinuitySnapshot(
            motion_history=state.motion_history,
            expected_area_px=float(
                state.expected_area_px or self.initial_expected_area_px
            ),
            motion_broken_at_lost=bool(state.motion_broken_at_lost),
            previous_frame=readonly_copy(state.previous_frame),
            quality_reference_frame=readonly_copy(state.quality_reference_frame),
            scale_model=state.scale_model,
            reliable_scale_anchor=state.reliable_scale_anchor,
            motion_break_events=int(state.motion_break_events),
        )

    def collection_state_sizes(self) -> dict[str, int]:
        state = self.persistent_state
        sizes = {
            name: len(value)
            for name in self._COLLECTION_STATE_NAMES
            if (value := state.get(name)) is not None
        }
        if self._adaptation_adapter is not None:
            sizes["adaptation_events"] = len(self._adaptation_adapter.events)
        return sizes

    def close(self) -> None:
        self.done = True
        self._pending_frame = None
        self._engine.close()


def run_tracking(
    *,
    frame_reader: FrameReader,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    reference_bank: ReferenceBank,
    initial_history: Sequence[tuple[float, float]],
    detector_config: CandidateDetectorConfig,
    config: TrackingConfig,
    identity_weights: Mapping[str, float] | None = None,
    initial_history_frames: Sequence[int] | None = None,
    adaptation_adapter: EMAReferenceAdapter | None = None,
    initial_expected_area_px: float | None = None,
    progress_callback: LiveProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
    thermal_profile: ThermalProfile | None = None,
    scale_model: ScaleStabilityModel | None = None,
    reliable_scale_anchor: ReliableScaleAnchor | None = None,
) -> TrackingResult:
    """Compatibility batch wrapper over :class:`TrackingRuntime`."""
    adapt_event_start = len(adaptation_adapter.events) if adaptation_adapter is not None else 0
    adapt_update_start = adaptation_adapter.update_count if adaptation_adapter is not None else 0
    runtime = TrackingRuntime(
        frame_count=frame_count,
        frame_width=frame_width,
        frame_height=frame_height,
        start_frame=start_frame,
        reference_bank=reference_bank,
        initial_history=initial_history,
        detector_config=detector_config,
        config=config,
        identity_weights=identity_weights,
        initial_history_frames=initial_history_frames,
        adaptation_adapter=adaptation_adapter,
        initial_expected_area_px=initial_expected_area_px,
        thermal_profile=thermal_profile,
        scale_model=scale_model,
        reliable_scale_anchor=reliable_scale_anchor,
        _adaptation_event_history_limit=None,
    )
    results: list[TrackFrameResult] = []
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
                        mode="TRACK",
                        frame_index=int(frame_index),
                        frame_bgr=frame.copy(),
                        result=item,
                    )
                )
    finally:
        continuity = runtime.continuity_snapshot
        runtime.close()

    accepted = [item for item in results if item.selected is not None and item.status == "TRACK"]
    accepted_scores = [item.selected_score for item in accepted]
    confidences = [item.confidence for item in results]
    prediction_errors = [
        item.prediction_error_px
        for item in accepted
        if item.prediction_error_px is not None
    ]
    shift_times = [
        item.global_shift_time_ms
        for item in results
        if item.global_shift_method != "OFF"
    ]
    trajectory = tuple(
        (item.frame_index, item.selected.center_x, item.selected.center_y)
        for item in accepted
        if item.selected is not None
    )
    timed_history = continuity.motion_history
    lost_frame = next((item.frame_index for item in results if item.status == "LOST"), None)
    status = "STOPPED" if stopped else ("LOST" if lost_frame is not None else "COMPLETED")
    tracked_frames = len(accepted_scores)
    return TrackingResult(
        status=status,
        frames=tuple(results),
        tracked_frames=tracked_frames,
        attempted_frames=len(results),
        mean_score=float(sum(accepted_scores) / tracked_frames) if tracked_frames else 0.0,
        median_score=float(median(accepted_scores)) if accepted_scores else 0.0,
        lost_frame=lost_frame,
        last_center=(
            timed_history[-1].xy if timed_history else runtime.initial_last_center
        ),
        trajectory=trajectory,
        mean_confidence=float(sum(confidences) / len(confidences)) if confidences else 0.0,
        median_prediction_error_px=float(median(prediction_errors)) if prediction_errors else 0.0,
        max_search_radius_used_px=max((item.search_radius_px for item in results), default=0),
        adaptation_updates=(adaptation_adapter.update_count - adapt_update_start) if adaptation_adapter is not None else 0,
        adaptation_events=tuple(adaptation_adapter.events[adapt_event_start:]) if adaptation_adapter is not None else (),
        last_target_area_px=float(
            continuity.expected_area_px
        ),
        motion_break_events=continuity.motion_break_events,
        motion_broken_at_lost=continuity.motion_broken_at_lost,
        mean_global_shift_time_ms=float(sum(shift_times) / len(shift_times)) if shift_times else 0.0,
        over_budget_frames=sum(1 for item in results if item.over_budget),
    )
