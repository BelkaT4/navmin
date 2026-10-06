import hashlib

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_initial_acquire_candidates, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG, load_config
from navmin.vision.processors.vt11.profiles import effective_config, profile_identity_weights


def _cfg(*, click_seed=True, broad_sigma=5.0):
    return CandidateDetectorConfig(
        method='thermal_hot_blob', threshold_sigma=3.5, min_response=8.0,
        max_area_px=2500, thermal_top_hat_kernel_px=21, thermal_top_hat_max_px=81,
        acquire_multiscale_enabled=True, acquire_large_scale_sigma_px=broad_sigma,
        acquire_large_target_trigger_area_px=2500.0, acquire_max_area_fraction_roi=0.70,
        acquire_expand_roi_radius_px=140, acquire_border_margin_px=5,
        acquire_click_seed_enabled=click_seed, acquire_click_seed_roi_radius_px=180,
        acquire_click_seed_close_px=11, acquire_click_seed_max_support_distance_px=42.0,
        adaptive_min_area_ratio=0.10, adaptive_max_area_ratio=5.0,
    )


def _flat_large_aircraft(dx=0, dy=0):
    f = np.full((500, 800, 3), 80, np.uint8)
    cv2.rectangle(f, (220+dx, 235+dy), (590+dx, 265+dy), (125,125,125), -1)
    cv2.rectangle(f, (380+dx, 145+dy), (430+dx, 355+dy), (130,130,130), -1)
    return f


def test_rev2_config_enables_click_seed_only_for_thermal_and_keeps_day_frozen():
    cfg = load_config()
    th = effective_config(cfg, 'thermal')
    day = effective_config(cfg, 'day')
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    assert th['acquire']['candidate_detector']['acquire_multiscale_enabled'] is True
    assert th['acquire']['candidate_detector']['acquire_click_seed_enabled'] is True
    assert day['acquire']['candidate_detector'].get('acquire_click_seed_enabled', False) is False
    assert th['tracking']['thermal_frame_reliability_enabled'] is False
    assert profile_identity_weights(cfg, 'thermal')['thermal_2d_ratio'] == 0.0
    assert hashlib.sha256(DEFAULT_CONFIG.with_name("day_profile.yaml").read_bytes()).hexdigest() == '902b15e22d2b1b2abb4da80ddd5ae1360577cdc16af1e7b2514a812703b994d4'


def test_click_seeded_fallback_captures_large_flat_target_when_contrast_path_fragments():
    f = _flat_large_aircraft()
    sel = make_operator_selection(0, (555, 250), 70, 800, 500)  # click on right wing
    legacy = detect_local_candidates(f, sel, top_k=5, config=_cfg(click_seed=False))
    assert legacy.candidates
    assert max(c.area_px for c in legacy.candidates) < 2500
    detected, used = detect_initial_acquire_candidates(f, sel, top_k=5, config=_cfg(click_seed=True))
    assert detected.candidates
    c = detected.candidates[0]
    assert c.detection_channel == 'ACQUIRE_CLICK_SEEDED'
    assert c.area_px > 10000
    assert c.distance_to_click_px <= 1.0
    assert used.roi_radius_px == 180


def test_large_multiscale_candidate_is_promoted_by_click_support_not_centroid_distance():
    f = np.full((500, 800, 3), 80, np.uint8)
    cv2.rectangle(f, (250,235), (570,265), (145,145,145), -1)
    cv2.rectangle(f, (385,150), (435,350), (155,155,155), -1)
    cv2.rectangle(f, (330,190), (490,310), (135,135,135), -1)
    sel = make_operator_selection(0, (525, 250), 70, 800, 500)
    detected, used = detect_initial_acquire_candidates(f, sel, top_k=5, config=_cfg(click_seed=True, broad_sigma=22.0))
    c = detected.candidates[0]
    assert c.area_px > 4000
    assert c.detection_channel in {'ACQUIRE_MULTISCALE_LARGE', 'ACQUIRE_CLICK_SEEDED'}
    assert c.distance_to_click_px <= 1.0
    assert used.roi_radius_px >= 140


def test_small_target_still_uses_historical_path():
    f = np.full((220, 320, 3), 50, np.uint8)
    cv2.circle(f, (160,110), 8, (220,220,220), -1)
    sel = make_operator_selection(0, (162,111), 70, 320, 220)
    cfg = _cfg(click_seed=True, broad_sigma=22.0)
    historical = detect_local_candidates(f, sel, top_k=5, config=cfg)
    revised, used = detect_initial_acquire_candidates(f, sel, top_k=5, config=cfg)
    assert used.roi_radius_px == 70
    assert [(c.area_px, c.center) for c in revised.candidates] == [(c.area_px, c.center) for c in historical.candidates]


def test_click_seeded_large_target_survives_initial_train():
    frames = [_flat_large_aircraft(3*i, i) for i in range(7)]
    sel = make_operator_selection(0, (555,250), 70, 800, 500)
    cfg = _cfg(click_seed=True)
    detected, _ = detect_initial_acquire_candidates(frames[0], sel, top_k=5, config=cfg)
    initial = detected.candidates[0]
    assert initial.area_px > 10000
    def reader(i):
        return i, frames[i]
    weights = {
        'appearance': .04, 'edge': .015, 'projection': .07, 'geometry': .015,
        'thermal_contrast': .16, 'hot_area': .20, 'peak_prominence': .18,
        'hot_core_compactness': .14, 'radial_hotspot_profile': .14,
        'energy_concentration': .04, 'thermal_2d_ratio': 0.0,
    }
    result = run_initial_train(
        frame_reader=reader, frame_count=len(frames), frame_width=800, frame_height=500,
        start_frame=0, initial_candidate=initial, roi_radius_px=70, detector_config=cfg,
        config=InitialTrainConfig(
            initial_frames=7, top_k=5, bank_update_min_score=.40,
            confirm_min_valid_fraction=.70, confirm_min_median_score=.40,
            confirm_min_median_gap=0.0,
        ),
        identity_weights=weights, thermal_identity_weights=weights, video_mode='thermal',
    )
    assert result.decision == 'CONFIRMED'
    assert result.valid_fraction == 1.0
    assert all(x.selected is not None and x.selected.area_px > 10000 for x in result.frames)
