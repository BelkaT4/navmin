from navmin.vision.processors.vt11.config.loader import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert cfg["project"]["config_version"] == 31
    assert cfg["acquire"]["roi_radius_px"] == 70
    assert cfg["acquire"]["initial_frames"] == 7
    assert cfg["acquire"]["top_k_candidates"] == 5
    assert cfg["acquire"]["scoring"]["mode"] == "combined"
    assert cfg["acquire"]["initial_train"]["reference_bank_size"] == 5
    assert cfg["tracking"]["search_radius_px"] == 90
    assert cfg["tracking"]["lost_after_misses"] == 3
    assert cfg["tracking"]["dynamic_search_enabled"] is True
    assert cfg["tracking"]["max_search_radius_px"] == 180
    assert cfg["adaptation"]["mode"] == "adaptive_memory"
    assert cfg["adaptation"]["ema"]["alpha"] == 0.05
    assert cfg["evaluation"]["ground_truth_tolerance_px"] == 20.0
    assert cfg["hotfix1"]["run_mode"] == "live"
    assert cfg["hotfix1"]["pace_to_source_fps"] is True
    assert cfg["hotfix2"]["diagnostics_enabled"] is True
    assert cfg["hotfix2"]["save_incident_frames"] is True
    assert cfg["hotfix2"]["make_zip_bundle"] is True
    assert cfg["acquire"]["candidate_detector"]["adaptive_area_enabled"] is True
    assert cfg["acquire"]["candidate_detector"]["adaptive_min_area_ratio"] == 0.15
    assert cfg["reacquisition"]["confirm_frames"] == 3
    assert cfg["reacquisition"]["confirm_window_frames"] == 5
    assert cfg["adaptation"]["ema"]["min_confidence"] == 0.68
    assert cfg["adaptation"]["ema"]["min_identity_similarity"] == 0.48
    assert cfg["thermal"]["video_mode"] == "auto"
    assert cfg["thermal"]["reticle_mode"] == "auto"
    assert cfg["identity"]["thermal_weights"]["hot_area"] > cfg["identity"]["thermal_weights"]["thermal_contrast"]
    assert cfg["tracking"]["track_warmup_frames"] == 6
    assert cfg["tracking"]["drift_rescue_enabled"] is True
    assert cfg["adaptation"]["ema"]["fast_alpha"] == 0.25
    assert cfg["adaptation"]["ema"]["recent_bank_size"] == 5


def test_m2_weights_are_normalized():
    cfg = load_config()
    weights = cfg["identity"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["appearance"] > weights["edge"] > weights["projection"] > 0


def test_custom_acquire_weights_are_normalized():
    cfg = load_config()
    weights = cfg["acquire"]["scoring"]["custom_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["spatial"] > weights["appearance"] > 0
