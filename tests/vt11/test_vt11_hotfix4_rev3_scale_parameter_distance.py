from dataclasses import dataclass
from math import log

from navmin.vision.processors.vt11.config.loader import load_config
from navmin.vision.processors.vt11.profiles.observation import effective_config
from navmin.vision.processors.vt11.tracking.scale_continuity import ReliableScaleAnchor, ScaleContinuityConfig
from navmin.vision.processors.vt11.tracking.tracklets import TrackletConfig, TrackletEvidence, TrackletManager


@dataclass
class DummyCandidate:
    area_px: int


def _ev(frame, area, *, quick=True, excess=0.3, d=1.2, extreme=False):
    return TrackletEvidence(
        frame_index=frame, x=100.0 + frame, y=100.0,
        identity=0.90, total_score=0.80, size_score=0.0, gap=0.20,
        candidate=DummyCandidate(area), has_competitor=True,
        thermal_profile_score=0.85, far_scale=True,
        scale_log_distance=(2.5 if extreme else d),
        scale_allowed_log_distance=0.9,
        scale_excess_log_distance=excess,
        scale_nonlinear_gate=0.25,
        scale_quick_confirmable=quick,
    )


def test_rev3_config_enables_scale_parameter_distance_only_for_thermal():
    cfg = load_config()
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    day = effective_config(cfg, 'day')
    th = effective_config(cfg, 'thermal')
    assert day['tracking']['thermal_scale_parameter_distance_enabled'] is False
    assert day['reacquisition']['thermal_scale_parameter_distance_enabled'] is False
    assert th['tracking']['thermal_scale_parameter_distance_enabled'] is True
    assert th['reacquisition']['thermal_scale_parameter_distance_enabled'] is True


def test_scale_distance_is_log_symmetric_and_normal_change_is_unpenalized():
    a = ReliableScaleAnchor(1000.0, 10, ScaleContinuityConfig(enabled=True))
    m_small = a.metrics(500.0, 10)
    m_large = a.metrics(2000.0, 10)
    assert abs(m_small.log_distance - log(2.0)) < 1e-9
    assert abs(m_small.log_distance - m_large.log_distance) < 1e-9
    assert m_small.nonlinear_gate == 1.0
    assert not m_small.suspicious


def test_gross_205x_scale_mismatch_is_strongly_suppressed_and_not_quick_confirmable():
    a = ReliableScaleAnchor(9638.0, 76, ScaleContinuityConfig(enabled=True))
    m = a.metrics(47.0, 77)
    assert m.log_distance > 5.0
    assert m.nonlinear_gate < 1e-6
    assert m.suspicious and m.extreme
    assert not m.quick_confirmable


def test_allowed_scale_distance_grows_with_time_but_is_capped():
    a = ReliableScaleAnchor(1000.0, 0, ScaleContinuityConfig(enabled=True))
    age0, d0 = a.allowed_log_distance(0)
    age10, d10 = a.allowed_log_distance(10)
    age100, d100 = a.allowed_log_distance(100)
    assert age0 == 0 and d0 == 0.85
    assert age10 == 10 and d10 > d0
    assert d100 == 1.60


def test_extreme_far_scale_tracklet_cannot_confirm_only_by_persisting():
    mgr = TrackletManager(TrackletConfig(
        max_tracklets=2, window_frames=7, confirm_hits=3,
        scale_distance_confirm_hits=3, scale_distance_extreme_confirm_hits=5,
        min_mean_identity=0.4, min_mean_score=0.4, min_mean_gap=0.0,
    ))
    for f in range(5):
        mgr.update(f, [_ev(f, 47, quick=False, excess=4.0, extreme=True)])
    assert mgr.best_confirmed(4) is None
