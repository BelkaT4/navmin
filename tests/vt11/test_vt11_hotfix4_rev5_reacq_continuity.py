import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.config.loader import load_config
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.profiles.observation import effective_config
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import (
    ReacquisitionConfig,
    _normal_hypothesis_continuity,
    run_reacquisition,
)
from navmin.vision.processors.vt11.tracking.scale_continuity import ReliableScaleAnchor, ScaleContinuityConfig


def _cfg() -> ReacquisitionConfig:
    return ReacquisitionConfig(
        initial_radius_px=80,
        max_radius_px=120,
        tracklet_max_link_distance_px=110.0,
        tracklet_max_scale_log_span=0.65,
        scale_quarantine_link_log_span=0.45,
    )


def test_rev5_checkpoint_and_temporal_click_threshold_are_conservative():
    cfg = load_config()
    assert cfg['project']['checkpoint'] == 'VT-11-HOTFIX-4-REV5-REACQ-CONTINUITY-FIX'
    assert cfg['project']['config_version'] == 31
    th = effective_config(cfg, 'thermal')
    assert th['acquire']['initial_train']['temporal_click_max_log_area_deviation'] == 0.35
    assert th['reacquisition']['thermal_scale_parameter_distance_enabled'] is True


def test_normal_reacq_hypothesis_accepts_same_position_and_scale_family():
    cfg = _cfg()
    points = [MotionPoint(10, 100.0, 80.0), MotionPoint(11, 108.0, 82.0)]
    areas = [5000.0, 5200.0]
    ok, why = _normal_hypothesis_continuity(
        points, areas, MotionPoint(12, 116.0, 84.0), 4900.0, config=cfg
    )
    assert ok and why == 'OK'


def test_normal_reacq_hypothesis_resets_on_5000_to_47_scale_switch():
    cfg = _cfg()
    ok, why = _normal_hypothesis_continuity(
        [MotionPoint(10, 100.0, 80.0)], [5000.0],
        MotionPoint(11, 102.0, 80.0), 47.0, config=cfg,
    )
    assert not ok and why == 'SCALE'


def test_normal_reacq_hypothesis_resets_on_different_spatial_object():
    cfg = _cfg()
    ok, why = _normal_hypothesis_continuity(
        [MotionPoint(10, 100.0, 80.0)], [5000.0],
        MotionPoint(11, 260.0, 80.0), 5100.0, config=cfg,
    )
    assert not ok and why == 'POSITION'


def test_normal_reacq_applies_reliable_scale_anchor_before_confirmation():
    # The visual descriptor deliberately matches the tiny candidate very well.
    # The only reason it must not confirm is the session-wide scale anchor:
    # ~100 px^2 candidate versus a 10000 px^2 reliable target.
    def blob_frame():
        im = np.full((100, 140, 3), 28, np.uint8)
        cv2.rectangle(im, (65, 51), (75, 59), (225, 225, 225), -1)
        return im

    detector_cfg = CandidateDetectorConfig(
        threshold_sigma=1.5, min_response=2.0, min_area_px=2, max_area_px=500,
        adaptive_area_enabled=False, thermal_polarity='bright',
    )
    f0 = blob_frame()
    sel = make_operator_selection(0, (70, 55), 40, 140, 100)
    det = detect_local_candidates(f0, sel, top_k=5, config=detector_cfg)
    assert det.candidates
    bank = ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(f0, det.candidates[0])])
    frames = [f0.copy() for _ in range(4)]

    def reader(i):
        return i, frames[i].copy()

    profile = ThermalProfile('thermal', 'off', np.zeros((100, 140), np.uint8), ())
    anchor = ReliableScaleAnchor(10000.0, 0, ScaleContinuityConfig(enabled=True))
    result = run_reacquisition(
        frame_reader=reader, frame_count=4, frame_width=140, frame_height=100,
        lost_frame=0, reference_bank=bank, motion_history=[MotionPoint(0, 70.0, 55.0)],
        detector_config=detector_cfg,
        config=ReacquisitionConfig(
            initial_radius_px=80, max_radius_px=100, max_frames=3,
            confirm_frames=2, confirm_window_frames=3,
            min_score=0.01, min_identity_similarity=0.01, min_confidence_gap=0.0,
            max_confirmation_error_px=100.0,
            thermal_scale_parameter_distance_enabled=True,
            tracklets_enabled=False, robust_motion_break_enabled=False,
        ),
        thermal_profile=profile,
        initial_expected_area_px=float(det.candidates[0].area_px),
        reliable_scale_anchor=anchor,
    )
    assert result.status != 'CONFIRMED'
    assert result.frames
    assert all(f.scale_log_distance > 4.0 for f in result.frames)
    assert all(f.scale_nonlinear_gate < 1e-6 for f in result.frames)
    assert all('SCALE_PARAMETER_DISTANCE' in f.diagnostic_reason for f in result.frames)


def test_normal_three_of_five_cannot_mix_large_and_tiny_hypotheses():
    def frame(size):
        im = np.full((120, 180, 3), 20, np.uint8)
        h = size // 2
        cv2.rectangle(im, (90-h, 60-h), (90+h, 60+h), (230, 230, 230), -1)
        return im

    detector_cfg = CandidateDetectorConfig(
        threshold_sigma=1.5, min_response=2.0, min_area_px=2, max_area_px=3000,
        adaptive_area_enabled=False,
    )
    f0 = frame(20)
    sel = make_operator_selection(0, (90, 60), 50, 180, 120)
    det = detect_local_candidates(f0, sel, top_k=5, config=detector_cfg)
    bank = ReferenceBank(max_size=3, descriptors=[descriptor_from_candidate(f0, det.candidates[0])])
    frames = [f0, frame(20), frame(4), frame(4)]

    def reader(i):
        return i, frames[i].copy()

    result = run_reacquisition(
        frame_reader=reader, frame_count=4, frame_width=180, frame_height=120,
        lost_frame=0, reference_bank=bank, motion_history=[MotionPoint(0, 90.0, 60.0)],
        detector_config=detector_cfg,
        config=ReacquisitionConfig(
            initial_radius_px=80, max_radius_px=100, max_frames=3,
            confirm_frames=3, confirm_window_frames=5,
            min_score=0.0, min_identity_similarity=0.0, min_confidence_gap=0.0,
            max_confirmation_error_px=200.0,
            identity_weight=1.0, motion_weight=0.0, size_weight=0.0,
            tracklets_enabled=False, robust_motion_break_enabled=False,
            thermal_scale_parameter_distance_enabled=False,
        ),
    )
    assert result.status != 'CONFIRMED'
    assert result.frames[0].diagnostic_reason == 'TENTATIVE_OK_1_OF_5'
    assert 'TENTATIVE_HYPOTHESIS_RESET_SCALE_1_OF_5' == result.frames[1].diagnostic_reason
    assert result.frames[2].diagnostic_reason == 'TENTATIVE_OK_2_OF_5'
