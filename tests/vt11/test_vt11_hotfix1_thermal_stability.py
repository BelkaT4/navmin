from types import SimpleNamespace
import hashlib
import numpy as np
import cv2

from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG, load_config
from navmin.vision.processors.vt11.profiles import effective_config, profile_identity_weights
from navmin.vision.processors.vt11.features.thermal import thermal_feature_metrics
from navmin.vision.processors.vt11.tracking.scale_model import ScaleStabilityConfig, ScaleStabilityModel
from navmin.vision.processors.vt11.tracking.tracklets import TrackletConfig, TrackletEvidence, TrackletManager
from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection


def test_hotfix1_checkpoint_and_day_profile_is_frozen():
    cfg = load_config()
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    day_bytes = DEFAULT_CONFIG.with_name("day_profile.yaml").read_bytes()
    assert hashlib.sha256(day_bytes).hexdigest() == '902b15e22d2b1b2abb4da80ddd5ae1360577cdc16af1e7b2514a812703b994d4'
    day = effective_config(cfg, 'day')
    assert day['acquire']['candidate_detector']['micro_enabled'] is False
    assert day['identity']['weights']['appearance'] == 0.5
    assert day['identity']['weights']['color'] == 0.1


def test_thermal_profile_uses_structural_features_and_stable_scale():
    cfg = load_config()
    th = effective_config(cfg, 'thermal')
    w = profile_identity_weights(cfg, 'thermal')
    for name in ('peak_prominence', 'hot_core_compactness', 'radial_hotspot_profile', 'energy_concentration'):
        assert w[name] > 0
    assert w['thermal_2d_ratio'] == 0.0  # REV1: retained in diagnostics, removed from decision score
    # hot_area remains useful but is no longer the 0.45-dominant cue from VT-11 baseline.
    assert w['hot_area'] <= 0.20
    assert th['tracking']['thermal_scale_stability_enabled'] is True
    assert th['reacquisition']['thermal_scale_stability_enabled'] is True
    assert th['reacquisition']['global_emergency_after_frames'] == 15
    assert th['reacquisition']['global_search_interval_frames'] == 3
    assert th['reacquisition']['tracklet_window_frames'] == 7


def test_thermal_structure_features_distinguish_compact_peak_from_broad_warm_patch():
    bg = np.full((40, 40), 80, np.uint8)
    compact = np.full((20, 20), 80, np.uint8)
    cv2.circle(compact, (10, 10), 3, 220, -1)
    broad = np.full((20, 20), 105, np.uint8)
    fc = thermal_feature_metrics(compact, bg)
    fb = thermal_feature_metrics(broad, bg)
    assert fc.peak_prominence > fb.peak_prominence
    assert fc.hot_core_compactness > 0.45
    assert np.isclose(float(fc.radial_profile.sum()), 1.0, atol=1e-5)
    assert 0.0 <= fc.energy_concentration <= 1.0
    assert 0.0 <= fb.energy_concentration <= 1.0


def test_scale_model_rejects_short_large_blob_run_and_keeps_immutable_base():
    cfg = ScaleStabilityConfig(
        enabled=True, confirm_frames=4, large_change_confirm_frames=6,
        max_log_change_per_frame=0.025, micro_max_area_px=80,
        micro_confirm_frames=3, micro_exit_confirm_frames=4,
    )
    model = ScaleStabilityModel(600.0, cfg)
    for f in range(5):
        obs = model.observe(frame_index=f, area_px=6000.0, trusted=True)
        assert not obs.updated
    assert model.base_area_px == 600.0
    assert model.current_area_px == 600.0
    # Sixth consistent large observation may move the model, but only by the log-rate limit.
    obs = model.observe(frame_index=5, area_px=6000.0, trusted=True)
    assert obs.updated
    assert model.current_area_px < 620.0
    assert model.base_area_px == 600.0


def test_scale_model_has_separate_micro_state_without_destroying_normal_scale():
    model = ScaleStabilityModel(500.0, ScaleStabilityConfig(enabled=True, micro_max_area_px=80, micro_confirm_frames=3))
    normal_before = model.current_area_px
    for f, area in enumerate((29.0, 31.0, 30.0), start=10):
        obs = model.observe(frame_index=f, area_px=area, trusted=True, is_micro=True)
    assert model.micro_mode is True
    assert 25.0 <= model.effective_area_px <= 35.0
    assert model.current_area_px == normal_before
    assert model.base_area_px == 500.0


def _ev(frame, x, identity, gap, thermal, area=45):
    c = SimpleNamespace(area_px=area)
    return TrackletEvidence(
        frame_index=frame, x=float(x), y=100.0 + frame,
        identity=float(identity), total_score=0.80, size_score=0.75, gap=float(gap), candidate=c,
        anchor_distance_px=150.0, far_jump=True, is_micro=True,
        background_dx_px=0.0, background_dy_px=0.0, background_valid=False,
        thermal_profile_score=float(thermal),
    )


def test_far_thermal_tracklet_uses_temporal_aggregate_not_old_per_frame_091_gate():
    mgr = TrackletManager(TrackletConfig(
        max_tracklets=3, window_frames=7, confirm_hits=4, max_link_distance_px=30,
        min_mean_identity=0.50, min_mean_score=0.50, min_mean_gap=0.02,
        far_jump_confirm_hits=6, far_jump_min_mean_identity=0.78, far_jump_min_mean_gap=0.06,
        far_jump_min_median_thermal_profile=0.68, max_scale_log_span=0.65,
        background_attachment_enabled=False,
    ))
    ids = (0.73, 0.84, 0.91, 0.80, 0.88, 0.82)
    gaps = (0.07, 0.13, 0.04, 0.18, 0.11, 0.09)
    therm = (0.72, 0.80, 0.76, 0.85, 0.78, 0.74)
    for f in range(6):
        mgr.update(f, [_ev(f, 180 + 3*f, ids[f], gaps[f], therm[f], area=42 + (f % 2))])
    # Several frames are below the old per-frame Identity=0.91 / Gap=0.11 gate,
    # but the six-frame temporal identity is strong and self-consistent.
    assert mgr.best_confirmed(5) is not None


def test_global_thermal_candidate_ranking_is_spatially_diverse():
    frame = np.full((220, 320, 3), 35, np.uint8)
    for pt in ((40, 40), (120, 55), (220, 60), (275, 170)):
        cv2.circle(frame, pt, 7, (230, 230, 230), -1)
    sel = make_operator_selection(0, (160, 110), 160, 320, 220)
    cfg = CandidateDetectorConfig(
        method='thermal_hot_blob', threshold_sigma=1.2, min_response=1.0,
        min_area_px=3, max_area_px=2500, spatial_sigma_px=35,
        thermal_top_hat_kernel_px=21, micro_enabled=True,
    )
    r = detect_local_candidates(
        frame, sel, top_k=4, config=cfg, expected_area_px=150.0,
        include_micro=True, global_search=True, diversity_min_distance_px=60.0,
    )
    assert len(r.candidates) >= 3
    centers = [(c.center_x, c.center_y) for c in r.candidates[:3]]
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            assert np.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1]) >= 50.0

def test_confirmed_micro_reacq_commits_micro_scale_immediately():
    model = ScaleStabilityModel(700.0, ScaleStabilityConfig(enabled=True, micro_max_area_px=80, micro_confirm_frames=3))
    obs = model.commit_confirmed_reacquisition(frame_index=40, area_px=28.0, is_micro=True)
    assert obs.updated is True
    assert model.micro_mode is True
    assert model.effective_area_px == 28.0
    assert model.base_area_px == 700.0
    assert model.current_area_px == 700.0
