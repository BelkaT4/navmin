import numpy as np

from navmin.vision.processors.vt11.config.loader import load_config, load_profile_config
from navmin.vision.processors.vt11.thermal.processing import ThermalProcessingConfig, build_thermal_profile, config_from_dict


def test_rev1_operator_switch_defaults_off():
    cfg = load_config()
    assert cfg["project"]["checkpoint"] == "VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX"
    assert cfg["project"]["config_version"] == 31
    assert cfg["thermal"]["hud_suppression_enabled"] is False
    thermal = load_profile_config("thermal")
    assert thermal["thermal"]["hud_suppression_enabled"] is False
    assert thermal["tracking"]["thermal_jump_quarantine_enabled"] is True


def test_rev1_backend_master_gate_disables_mask_and_negative_model():
    frames = [np.full((40, 60, 3), 80, np.uint8) for _ in range(3)]
    off_cfg = config_from_dict({
        "video_mode": "thermal", "hud_suppression_enabled": False,
        "reticle_mode": "auto", "hud_negative_enabled": True,
    })
    off_profile = build_thermal_profile(frames, config=off_cfg)
    assert off_profile.reticle_mode == "off"
    assert off_profile.masked_pixels == 0
    assert off_profile.hud_negative_enabled is False

    on_cfg = ThermalProcessingConfig(
        video_mode="thermal", hud_suppression_enabled=True,
        reticle_mode="auto", hud_negative_enabled=True,
    )
    on_profile = build_thermal_profile(frames, config=on_cfg)
    assert on_profile.reticle_mode == "auto"
    assert on_profile.hud_negative_enabled is True
