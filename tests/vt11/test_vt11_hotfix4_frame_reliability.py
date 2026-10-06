import hashlib

import cv2
import numpy as np

from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG, load_config
from navmin.vision.processors.vt11.profiles import effective_config, profile_identity_weights
from navmin.vision.processors.vt11.features.thermal import thermal_feature_metrics
from navmin.vision.processors.vt11.tracking.frame_reliability import FrameReliabilityConfig, estimate_frame_reliability
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig


def test_hotfix4_checkpoint_and_day_profile_bytes_remain_frozen():
    cfg = load_config()
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    day_bytes = DEFAULT_CONFIG.with_name("day_profile.yaml").read_bytes()
    assert hashlib.sha256(day_bytes).hexdigest() == '902b15e22d2b1b2abb4da80ddd5ae1360577cdc16af1e7b2514a812703b994d4'


def test_thermal_profile_enables_only_compact_new_logic():
    cfg = load_config()
    th = effective_config(cfg, 'thermal')
    day = effective_config(cfg, 'day')
    tw = profile_identity_weights(cfg, 'thermal')
    assert th['tracking']['anchor_time_normalization_enabled'] is True
    assert th['tracking']['thermal_frame_reliability_enabled'] is False
    assert th['reacquisition']['anchor_time_normalization_enabled'] is True
    assert th['reacquisition']['thermal_frame_reliability_enabled'] is False
    assert day['tracking']['anchor_time_normalization_enabled'] is False
    assert day['tracking']['thermal_frame_reliability_enabled'] is False
    assert tw['thermal_2d_ratio'] == 0.0
    assert tw['hot_area'] == 0.20
    supported_sum = sum(v for k, v in tw.items() if not k.startswith('_'))
    assert abs(supported_sum - 1.0) < 1e-9


def test_frame_reliability_is_high_for_translation_and_lower_for_corruption():
    rng = np.random.default_rng(7)
    ref = rng.normal(115.0, 32.0, (120, 160)).clip(0, 255).astype(np.uint8)
    ref = cv2.GaussianBlur(ref, (0, 0), 1.1)
    shifted = cv2.warpAffine(
        ref, np.float32([[1, 0, 6], [0, 1, -4]]), (160, 120),
        borderMode=cv2.BORDER_REFLECT,
    )
    corrupted = rng.integers(0, 256, ref.shape, dtype=np.uint8)
    cfg = FrameReliabilityConfig(enabled=True)
    good = estimate_frame_reliability(ref, shifted, config=cfg)
    bad = estimate_frame_reliability(ref, corrupted, config=cfg)
    assert good.valid and bad.valid
    assert good.score > 0.90
    assert bad.score < good.score - 0.20


def test_k2d_is_soft_structural_cue_line_lower_than_compact_blob():
    background = np.full((60, 60), 70, np.uint8)
    line = np.full((24, 24), 70, np.uint8)
    line[11:13, 2:22] = 220
    compact = np.full((24, 24), 70, np.uint8)
    cv2.circle(compact, (12, 12), 6, 220, -1)
    f_line = thermal_feature_metrics(line, background)
    f_compact = thermal_feature_metrics(compact, background)
    assert 0.0 <= f_line.spatial_2d_ratio <= 1.0
    assert 0.0 <= f_compact.spatial_2d_ratio <= 1.0
    assert f_line.spatial_2d_ratio < 0.08
    assert f_compact.spatial_2d_ratio > f_line.spatial_2d_ratio + 0.30


def test_new_config_validation_accepts_conservative_settings():
    TrackingConfig(
        anchor_time_normalization_enabled=True,
        thermal_frame_reliability_enabled=True,
    ).validate()
    ReacquisitionConfig(
        anchor_time_normalization_enabled=True,
        thermal_frame_reliability_enabled=True,
    ).validate()
