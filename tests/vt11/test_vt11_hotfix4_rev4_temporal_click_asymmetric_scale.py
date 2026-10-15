from math import log

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import Candidate, CandidateDetectorConfig, detect_initial_acquire_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import RoiRect, make_operator_selection
from navmin.vision.processors.vt11.config.loader import load_config
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.profiles.observation import effective_config
from navmin.vision.processors.vt11.tracking.scale_continuity import ReliableScaleAnchor, ScaleContinuityConfig
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, _rank_candidates


def _detector_cfg():
    return CandidateDetectorConfig(
        method='thermal_hot_blob', threshold_sigma=3.5, min_response=8.0,
        max_area_px=2500, thermal_top_hat_kernel_px=21, thermal_top_hat_max_px=81,
        acquire_multiscale_enabled=True, acquire_large_scale_sigma_px=5.0,
        acquire_large_target_trigger_area_px=2500.0, acquire_max_area_fraction_roi=0.70,
        acquire_expand_roi_radius_px=140, acquire_border_margin_px=5,
        acquire_click_seed_enabled=True, acquire_click_seed_roi_radius_px=180,
        acquire_click_seed_close_px=11, acquire_click_seed_max_support_distance_px=42.0,
        adaptive_min_area_ratio=0.10, adaptive_max_area_ratio=5.0,
    )


def _thermal_weights():
    return {
        'appearance': .04, 'edge': .015, 'projection': .07, 'geometry': .015,
        'thermal_contrast': .16, 'hot_area': .20, 'peak_prominence': .18,
        'hot_core_compactness': .14, 'radial_hotspot_profile': .14,
        'energy_concentration': .04, 'thermal_2d_ratio': 0.0,
    }


def _good_large_aircraft(dx=0):
    f = np.full((500, 800, 3), 80, np.uint8)
    cv2.rectangle(f, (220 + dx, 235), (590 + dx, 265), (125, 125, 125), -1)
    cv2.rectangle(f, (380 + dx, 145), (430 + dx, 355), (130, 130, 130), -1)
    return f


def _smeared_large_aircraft():
    f = np.full((500, 800, 3), 80, np.uint8)
    cv2.rectangle(f, (180, 205), (630, 305), (125, 125, 125), -1)
    cv2.rectangle(f, (350, 100), (455, 390), (130, 130, 130), -1)
    return f


def test_rev4_config_enables_temporal_click_and_asymmetric_expansion_only_for_thermal():
    cfg = load_config()
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    th = effective_config(cfg, 'thermal')
    day = effective_config(cfg, 'day')
    assert th['acquire']['initial_train']['temporal_click_acquire_enabled'] is True
    assert th['tracking']['scale_expansion_uncertainty_enabled'] is True
    assert day['acquire']['initial_train'].get('temporal_click_acquire_enabled', False) is False
    assert day['tracking'].get('scale_expansion_uncertainty_enabled', False) is False


def test_temporal_click_skips_single_bad_large_start_frame():
    frames = [_smeared_large_aircraft()] + [_good_large_aircraft(2 * i) for i in range(1, 11)]
    cfg = _detector_cfg()
    sel = make_operator_selection(0, (555, 250), 70, 800, 500)
    detected, _ = detect_initial_acquire_candidates(frames[0], sel, top_k=5, config=cfg)
    initial = detected.candidates[0]
    assert initial.area_px > 30000  # intentionally bad/smeared first representation

    def reader(i):
        return i, frames[i]

    result = run_initial_train(
        frame_reader=reader, frame_count=len(frames), frame_width=800, frame_height=500,
        start_frame=0, initial_candidate=initial, roi_radius_px=70, detector_config=cfg,
        config=InitialTrainConfig(
            initial_frames=7, top_k=5, bank_update_min_score=.40,
            confirm_min_valid_fraction=.70, confirm_min_median_score=.40,
            confirm_min_median_gap=0.0,
            temporal_click_acquire_enabled=True,
            temporal_click_bootstrap_frames=5,
            temporal_click_min_stable_hits=3,
            temporal_click_max_log_area_deviation=.35,
        ),
        identity_weights=_thermal_weights(), thermal_identity_weights=_thermal_weights(),
        video_mode='thermal', initial_click_xy=(555, 250),
    )
    assert result.decision == 'CONFIRMED'
    assert result.temporal_click_used
    assert result.temporal_click_shift_frames >= 1
    assert result.frames[0].frame_index >= 1
    assert result.reference_bank.expected_size_px() < 0.60 * initial.area_px


def _candidate(x, y, w, h):
    return Candidate(
        candidate_id=1, bbox=RoiRect(x=x, y=y, width=w, height=h),
        center_x=x + w / 2.0, center_y=y + h / 2.0,
        area_px=w * h, mean_response=30.0, peak_response=50.0,
        distance_to_click_px=0.0, spatial_prior=1.0, detector_strength=1.0,
        rank_score=1.0, size_prior=1.0,
    )


def _rank_pair(candidate, anchor_area):
    ref_frame = np.full((120, 120, 3), 50, np.uint8)
    ref_c = _candidate(50, 50, 20, 20)
    cv2.rectangle(ref_frame, (50, 50), (69, 69), (180, 180, 180), -1)
    bank = ReferenceBank(max_size=5)
    bank.add(descriptor_from_candidate(ref_frame, ref_c))

    frame = np.full((120, 120, 3), 50, np.uint8)
    b = candidate.bbox
    cv2.rectangle(frame, (b.x, b.y), (b.x2 - 1, b.y2 - 1), (180, 180, 180), -1)
    anchor = ReliableScaleAnchor(anchor_area, 0, ScaleContinuityConfig(
        enabled=True, base_allowed_log_distance=.85, growth_per_frame=0.0,
        max_allowed_log_distance=1.60, nonlinear_strength=1.50,
    ))
    tc = TrackingConfig(
        identity_weight=.70, motion_weight=.18, size_weight=.12,
        thermal_scale_parameter_distance_enabled=True,
        scale_expansion_uncertainty_enabled=True,
        scale_expansion_uncertainty_min_identity=.50,
        scale_expansion_uncertainty_min_motion_score=.50,
        scale_expansion_uncertainty_min_raw_score=.20,
        scale_expansion_uncertainty_max_log_distance=1.85,
    )
    return _rank_candidates(
        frame, [candidate], bank, ref_c.center, cfg=tc,
        identity_weights={'thermal_contrast': 1.0}, expected_area_px=anchor_area,
        reliable_scale_anchor=anchor, frame_index=1,
    )[0]


def test_moderate_fourfold_expansion_preserves_positional_hypothesis():
    # 20x20 -> 40x40 means 4x area and D=ln(4), suspicious but not absurd.
    c = _candidate(40, 40, 40, 40)
    ranked = _rank_pair(c, 400.0)
    assert abs(ranked.scale_log_distance - log(4.0)) < 0.05
    assert ranked.scale_distance_suspicious
    assert ranked.scale_is_expansion
    assert ranked.scale_expansion_uncertain_candidate
    assert abs(ranked.total_score - ranked.raw_total_score) < 1e-9


def test_equal_fourfold_shrink_keeps_nonlinear_scale_penalty():
    # Reference anchor 1600; a 20x20 candidate is a 4x shrink. The REV3
    # protection must remain active and must not get the expansion bypass.
    c = _candidate(50, 50, 20, 20)
    ranked = _rank_pair(c, 1600.0)
    assert abs(ranked.scale_log_distance - log(4.0)) < 0.05
    assert ranked.scale_distance_suspicious
    assert not ranked.scale_is_expansion
    assert not ranked.scale_expansion_uncertain_candidate
    assert ranked.total_score < ranked.raw_total_score


def test_scale_distance_reports_direction_without_changing_log_symmetry():
    a = ReliableScaleAnchor(1000.0, 0, ScaleContinuityConfig(enabled=True))
    small = a.metrics(500.0, 0)
    large = a.metrics(2000.0, 0)
    assert small.is_shrink and not small.is_expansion
    assert large.is_expansion and not large.is_shrink
    assert abs(small.log_distance - large.log_distance) < 1e-12
    assert small.signed_log_ratio < 0 < large.signed_log_ratio
