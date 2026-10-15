import hashlib

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates, detect_initial_acquire_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG, load_config
from navmin.vision.processors.vt11.profiles import effective_config, profile_identity_weights
from navmin.vision.processors.vt11.features.thermal import thermal_feature_metrics


def _large_aircraft_frame(cx=240, cy=180):
    f = np.full((360, 480, 3), 50, np.uint8)
    cv2.rectangle(f, (cx - 70, cy - 8), (cx + 70, cy + 8), (220, 220, 220), -1)
    cv2.rectangle(f, (cx - 10, cy - 60), (cx + 10, cy + 60), (220, 220, 220), -1)
    return f


def _cfg(multiscale=True):
    return CandidateDetectorConfig(
        method='thermal_hot_blob', threshold_sigma=2.0, min_response=5.0,
        max_area_px=2500, thermal_top_hat_kernel_px=21, thermal_top_hat_max_px=81,
        acquire_multiscale_enabled=multiscale, acquire_large_scale_sigma_px=22.0,
        acquire_large_target_trigger_area_px=2500.0, acquire_max_area_fraction_roi=0.70,
        acquire_expand_roi_radius_px=140, acquire_border_margin_px=5,
        adaptive_min_area_ratio=0.10, adaptive_max_area_ratio=5.0,
    )


def test_rev1_checkpoint_conservative_defaults_and_day_frozen():
    cfg = load_config()
    th = effective_config(cfg, 'thermal')
    day = effective_config(cfg, 'day')
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    assert th['tracking']['anchor_time_normalization_enabled'] is True
    assert th['reacquisition']['anchor_time_normalization_enabled'] is True
    assert th['tracking']['thermal_frame_reliability_enabled'] is False
    assert th['reacquisition']['thermal_frame_reliability_enabled'] is False
    assert th['acquire']['candidate_detector']['acquire_multiscale_enabled'] is True
    assert day['acquire']['candidate_detector'].get('acquire_multiscale_enabled', False) is False
    assert hashlib.sha256(DEFAULT_CONFIG.with_name("day_profile.yaml").read_bytes()).hexdigest() == '902b15e22d2b1b2abb4da80ddd5ae1360577cdc16af1e7b2514a812703b994d4'


def test_k2d_is_computed_but_has_zero_decision_weight():
    cfg = load_config()
    w = profile_identity_weights(cfg, 'thermal')
    assert w['thermal_2d_ratio'] == 0.0
    assert w['hot_area'] == 0.20
    background = np.full((40, 40), 70, np.uint8)
    patch = np.full((24, 24), 70, np.uint8)
    cv2.circle(patch, (12, 12), 6, 220, -1)
    m = thermal_feature_metrics(patch, background)
    assert m.spatial_2d_ratio > 0.2


def test_small_thermal_acquire_preserves_historical_candidate_path():
    f = np.full((200, 300, 3), 50, np.uint8)
    cv2.circle(f, (150, 100), 8, (220, 220, 220), -1)
    sel = make_operator_selection(0, (153, 102), 70, 300, 200)
    cfg = _cfg(True)
    historical = detect_local_candidates(f, sel, top_k=5, config=cfg)
    conservative, used = detect_initial_acquire_candidates(f, sel, top_k=5, config=cfg)
    assert used.roi_radius_px == 70
    assert [(c.area_px, c.center) for c in conservative.candidates] == [(c.area_px, c.center) for c in historical.candidates]


def test_large_aircraft_acquire_forms_one_coherent_candidate_and_expands_roi():
    f = _large_aircraft_frame()
    sel = make_operator_selection(0, (242, 181), 70, 480, 360)
    historical = detect_local_candidates(f, sel, top_k=5, config=_cfg(False))
    revised, used = detect_initial_acquire_candidates(f, sel, top_k=5, config=_cfg(True))
    assert historical.candidates
    assert max(c.area_px for c in historical.candidates) < 2500
    assert revised.candidates
    assert revised.candidates[0].area_px > 4000
    assert revised.candidates[0].distance_to_click_px < 6.0
    assert used.roi_radius_px == 140


def test_large_aircraft_initial_train_keeps_large_scale_gate_alive():
    frames = [_large_aircraft_frame(240 + 3*i, 180 + i) for i in range(7)]
    def reader(i):
        return i, frames[i]
    cfg = _cfg(True)
    sel = make_operator_selection(0, (240, 180), 70, 480, 360)
    detected, _used = detect_initial_acquire_candidates(frames[0], sel, top_k=5, config=cfg)
    initial = detected.candidates[0]
    result = run_initial_train(
        frame_reader=reader, frame_count=len(frames), frame_width=480, frame_height=360,
        start_frame=0, initial_candidate=initial, roi_radius_px=70, detector_config=cfg,
        config=InitialTrainConfig(
            initial_frames=7, top_k=5, bank_update_min_score=0.40,
            confirm_min_valid_fraction=0.70, confirm_min_median_score=0.40,
            confirm_min_median_gap=0.0,
        ),
        identity_weights={'appearance': .2, 'edge': .1, 'projection': .1, 'geometry': .1, 'thermal_contrast': .2, 'hot_area': .3},
        thermal_identity_weights={
            'appearance': .04, 'edge': .015, 'projection': .07, 'geometry': .015,
            'thermal_contrast': .16, 'hot_area': .20, 'peak_prominence': .18,
            'hot_core_compactness': .14, 'radial_hotspot_profile': .14,
            'energy_concentration': .04, 'thermal_2d_ratio': 0.0,
        },
        video_mode='thermal',
    )
    assert result.decision == 'CONFIRMED'
    assert result.valid_fraction == 1.0
    assert all(x.selected is not None and x.selected.area_px > 3500 for x in result.frames)
