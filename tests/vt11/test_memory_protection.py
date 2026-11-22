import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectorConfig,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import (
    ReacquisitionConfig,
    run_reacquisition,
)
from navmin.vision.processors.vt11.tracking.session import Vt11TargetSession
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig


def _ellipse_frame(x: int | None, *, scale: float = 1.0) -> np.ndarray:
    frame = np.full((160, 320, 3), 28, np.uint8)
    if x is None:
        return frame
    cv2.ellipse(
        frame,
        (x, 80),
        (round(14 * scale), round(8 * scale)),
        0,
        0,
        360,
        (230, 230, 230),
        -1,
    )
    cv2.rectangle(
        frame,
        (x + round(9 * scale), 77),
        (x + round(18 * scale), 82),
        (145, 145, 145),
        -1,
    )
    return frame


def test_reacq_tentative_never_learns_and_memory_freeze_survives_confirmation():
    detector = CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3,
        max_area_px=2200,
        adaptive_max_area_ratio=4.0,
    )
    initial = _ellipse_frame(160)
    selection = make_operator_selection(
        frame_index=0,
        click_xy=(160, 80),
        radius_px=80,
        frame_width=320,
        frame_height=160,
    )
    candidate = detect_local_candidates(initial, selection, config=detector).candidates[0]
    reference = descriptor_from_candidate(initial, candidate)
    bank = ReferenceBank(descriptors=[reference])
    initial_area = float(candidate.area_px)
    history = [
        MotionPoint(0, 154.0, 80.0),
        MotionPoint(1, 157.0, 80.0),
        MotionPoint(2, 160.0, 80.0),
    ]
    frames = [
        initial,
        initial,
        initial,
        _ellipse_frame(164, scale=1.05),
        _ellipse_frame(168, scale=1.05),
        _ellipse_frame(172, scale=1.05),
    ]
    config = ReacquisitionConfig(
        initial_radius_px=90,
        max_radius_px=150,
        radius_growth_px_per_frame=10,
        confirm_frames=2,
        confirm_window_frames=3,
        max_frames=3,
        min_confidence_gap=0.0,
        max_confirmation_error_px=100,
        size_update_min_identity=0.0,
        size_update_min_gap=0.0,
        expected_area_update_alpha=1.0,
        size_update_max_step_ratio=10.0,
    )

    control = run_reacquisition(
        frame_reader=lambda i: (i, frames[i]),
        frame_count=len(frames),
        frame_width=320,
        frame_height=160,
        lost_frame=2,
        reference_bank=bank,
        motion_history=history,
        detector_config=detector,
        config=config,
        initial_expected_area_px=initial_area,
    )
    assert control.frames[0].status == "TENTATIVE"
    assert control.frames[0].expected_area_px == pytest.approx(initial_area)
    assert control.status == "CONFIRMED"
    assert control.last_expected_area_px != pytest.approx(initial_area)

    frozen = run_reacquisition(
        frame_reader=lambda i: (i, frames[i]),
        frame_count=len(frames),
        frame_width=320,
        frame_height=160,
        lost_frame=2,
        reference_bank=bank,
        motion_history=history,
        detector_config=detector,
        config=config,
        initial_expected_area_px=initial_area,
        initial_memory_freeze_until_frame=5,
    )
    assert frozen.frames[0].status == "TENTATIVE"
    assert frozen.frames[0].expected_area_px == pytest.approx(initial_area)
    assert frozen.status == "CONFIRMED"
    assert frozen.confirmed_frame == 4
    assert frozen.last_expected_area_px == pytest.approx(initial_area)
    assert bank.descriptors[0] is reference


def _maneuver_frame(x: int | None) -> np.ndarray:
    frame = np.full((180, 520, 3), 28, np.uint8)
    if x is None:
        return frame
    cv2.rectangle(frame, (x - 10, 80), (x + 10, 96), (225, 225, 225), -1)
    cv2.rectangle(frame, (x + 7, 83), (x + 16, 89), (150, 150, 150), -1)
    return frame


def test_target_session_carries_maneuver_freeze_across_track_to_reacq_boundary():
    detector = CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3,
        max_area_px=1200,
    )
    initial = _maneuver_frame(300)
    selection = make_operator_selection(
        frame_index=0,
        click_xy=(300, 88),
        radius_px=70,
        frame_width=520,
        frame_height=180,
    )
    candidate = detect_local_candidates(initial, selection, config=detector).candidates[0]
    bank = ReferenceBank(descriptors=[descriptor_from_candidate(initial, candidate)])
    session = Vt11TargetSession(
        frame_width=520,
        frame_height=180,
        start_frame=3,
        reference_bank=bank,
        initial_history=[(260, 88), (280, 88), (300, 88)],
        initial_history_frames=[0, 1, 2],
        detector_config=detector,
        tracking_config=TrackingConfig(
            day_maneuver_search_enabled=True,
            search_radius_px=70,
            dynamic_search_enabled=False,
            min_track_score=0.8,
            motion_sigma_px=8,
            lost_after_misses=2,
            track_warmup_frames=0,
        ),
        reacquisition_config=ReacquisitionConfig(
            initial_radius_px=120,
            max_radius_px=220,
            radius_growth_px_per_frame=10,
            confirm_frames=2,
            confirm_window_frames=3,
            max_frames=6,
            min_confidence_gap=0.0,
            max_confirmation_error_px=200,
        ),
        thermal_profile=ThermalProfile(
            "visible",
            "off",
            np.zeros((180, 520), np.uint8),
            (),
        ),
    )
    try:
        first = session.step(3, _maneuver_frame(260))
        assert first.result.status == "LOW_CONFIDENCE"
        confirmed = session.step(4, _maneuver_frame(220))
        assert confirmed.result.status == "TRACK"
        track_snapshot = session.tracking_continuity_snapshot
        assert track_snapshot is not None
        assert track_snapshot.memory_freeze_until_frame == 7

        session.step(5, _maneuver_frame(None))
        lost = session.step(6, _maneuver_frame(None))
        assert lost.result.status == "LOST"
        assert lost.phase_after == "REACQ"
        session.step(7, _maneuver_frame(140))
        reacq_snapshot = session.reacquisition_continuity_snapshot
        assert reacq_snapshot is not None
        assert reacq_snapshot.memory_freeze_until_frame == 7
    finally:
        session.close()
