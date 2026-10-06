import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import config_from_dict
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, ema_blend_descriptor
from navmin.vision.processors.vt11.config.loader import load_config, load_profile_config
from navmin.vision.processors.vt11.identity.descriptor import TargetDescriptor, descriptor_similarity
from navmin.vision.processors.vt11.profiles import effective_config, profile_identity_weights, resolve_profile_name


def _desc(*, color, area=120.0, thermal_contrast=0.4, hot_area=0.3):
    z = np.zeros((16, 16), np.float32)
    p = np.zeros(16, np.float32)
    g = np.zeros(2, np.float32)
    return TargetDescriptor(
        appearance=z.copy(), projection_x=p.copy(), projection_y=p.copy(), edge_map=z.copy(),
        geometry=g.copy(), size_px=area, bbox=(0, 0, 10, 10),
        thermal_contrast=thermal_contrast, hot_area=hot_area,
        color_hist=np.asarray(color, np.float32),
    )


def test_vt11_profiles_exist_and_auto_maps_domains():
    cfg = load_config()
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert set(cfg['profiles']) >= {'day', 'thermal'}
    assert resolve_profile_name('auto', inferred_video_mode='visible') == 'day'
    assert resolve_profile_name('auto', inferred_video_mode='thermal') == 'thermal'
    assert load_profile_config('day')['thermal']['video_mode'] == 'visible'
    assert load_profile_config('thermal')['thermal']['video_mode'] == 'thermal'


def test_day_profile_enables_color_geometry_and_disables_micro():
    cfg = load_config()
    eff = effective_config(cfg, 'day')
    w = profile_identity_weights(cfg, 'day')
    assert w['color'] > 0
    assert w['geometry'] > 0
    assert w['thermal_contrast'] == 0
    assert eff['acquire']['candidate_detector']['method'] == 'local_contrast_components'
    assert eff['acquire']['candidate_detector']['micro_enabled'] is False
    assert eff['reacquisition']['micro_detector_enabled'] is False


def test_thermal_profile_prioritizes_thermal_features_and_micro():
    cfg = load_config()
    eff = effective_config(cfg, 'thermal')
    w = profile_identity_weights(cfg, 'thermal')
    assert w['thermal_contrast'] > w['appearance']
    assert w['hot_area'] >= 0.10
    assert 'thermal_2d_ratio' in w
    assert w['thermal_2d_ratio'] == 0.0  # REV1: diagnostic only
    assert w['_thermal_scale_adaptive'] == 1.0
    assert eff['acquire']['candidate_detector']['method'] == 'thermal_hot_blob'
    assert eff['acquire']['candidate_detector']['micro_enabled'] is True
    assert eff['reacquisition']['micro_detector_enabled'] is True
    assert eff['reacquisition']['far_jump_confirm_hits'] >= 6


def test_day_color_histogram_affects_identity_similarity():
    # Two otherwise identical descriptors with matching vs opposite histogram bins.
    a = _desc(color=[1, 0, 0, 0])
    b = _desc(color=[1, 0, 0, 0])
    c = _desc(color=[0, 0, 0, 1])
    weights = {'color': 1.0}
    assert descriptor_similarity(a, b, identity_weights=weights).total > 0.99
    assert descriptor_similarity(a, c, identity_weights=weights).total < 0.2


def test_thermal_scale_adaptation_downweights_shape_for_micro_target():
    # Shape is deliberately dissimilar while thermal statistics match.  The
    # scale-adaptive thermal profile should score the MICRO candidate higher
    # than the same mismatch at large area.
    ref_small = _desc(color=[1, 0, 0, 0], area=50, thermal_contrast=0.6, hot_area=0.5)
    cand_small = _desc(color=[1, 0, 0, 0], area=50, thermal_contrast=0.6, hot_area=0.5)
    cand_small = TargetDescriptor(
        appearance=np.ones((16,16), np.float32)*3,
        projection_x=np.ones(16,np.float32), projection_y=np.ones(16,np.float32),
        edge_map=np.ones((16,16),np.float32)*3, geometry=np.ones(2,np.float32)*2,
        size_px=50, bbox=(0,0,10,10), thermal_contrast=0.6, hot_area=0.5,
        color_hist=np.asarray([1,0,0,0],np.float32),
    )
    ref_large = TargetDescriptor(**{**ref_small.__dict__, 'size_px': 600})
    cand_large = TargetDescriptor(**{**cand_small.__dict__, 'size_px': 600})
    w = {
        'appearance':0.20,'edge':0.10,'projection':0.10,'geometry':0.05,
        'thermal_contrast':0.25,'hot_area':0.30,
        '_thermal_scale_adaptive':1.0,'_micro_area_threshold_px':80,
        '_small_area_threshold_px':240,'_micro_shape_factor':0.2,
        '_small_shape_factor':0.6,'_micro_thermal_boost':1.4,'_small_thermal_boost':1.15,
    }
    s_small = descriptor_similarity(ref_small, cand_small, identity_weights=w).total
    s_large = descriptor_similarity(ref_large, cand_large, identity_weights=w).total
    assert s_small > s_large


def test_profiled_ema_can_adapt_thermal_features_faster_than_shape():
    old = _desc(color=[1,0,0,0], thermal_contrast=0.2, hot_area=0.2)
    new = TargetDescriptor(
        appearance=np.ones((16,16),np.float32), projection_x=np.ones(16,np.float32),
        projection_y=np.ones(16,np.float32), edge_map=np.ones((16,16),np.float32),
        geometry=np.ones(2,np.float32), size_px=200, bbox=(0,0,10,10),
        thermal_contrast=0.8, hot_area=0.8, color_hist=np.asarray([0,1,0,0],np.float32),
    )
    out = ema_blend_descriptor(old, new, 0.1, shape_alpha_multiplier=0.5, thermal_alpha_multiplier=1.5)
    # Shape alpha=0.05, thermal alpha=0.15.
    assert np.isclose(out.appearance.mean(), 0.05, atol=1e-5)
    assert np.isclose(out.thermal_contrast, 0.29, atol=1e-5)
    cfg = EMAConfig(shape_alpha_multiplier=0.7, thermal_alpha_multiplier=1.35)
    cfg.validate()


def test_profile_detector_dict_remains_valid_for_both_profiles():
    cfg = load_config()
    for name in ('day', 'thermal'):
        eff = effective_config(cfg, name)
        det = config_from_dict(eff['acquire']['candidate_detector'])
        det.validate()
