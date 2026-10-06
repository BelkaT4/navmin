import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig, run_reacquisition
from navmin.vision.processors.vt11.tracking.session import TrackingSessionConfig, run_tracking_session
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def _detector_cfg():
    return CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3.0,
        min_area_px=2,
        max_area_px=1200,
        spatial_sigma_px=35.0,
    )


def _draw_target(frame, k):
    x = 38 + 3 * k
    cv2.rectangle(frame, (x, 53), (x + 9, 61), (225, 225, 225), -1)
    cv2.rectangle(frame, (x + 7, 55), (x + 12, 58), (150, 150, 150), -1)


def _frames(count=24, invisible=range(9, 13)):
    invisible = set(invisible)
    frames = []
    for k in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if k not in invisible:
            _draw_target(frame, k)
        frames.append(frame)
    return frames


def _train(frames):
    sel = make_operator_selection(
        frame_index=0,
        click_xy=(43, 57),
        radius_px=65,
        frame_width=180,
        frame_height=120,
    )
    detected = detect_local_candidates(frames[0], sel, top_k=5, config=_detector_cfg())
    assert detected.candidates

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=0,
        initial_candidate=detected.candidates[0],
        roi_radius_px=65,
        detector_config=_detector_cfg(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "CONFIRMED"
    return result, reader


def _history(train):
    accepted = [r for r in train.frames if r.selected is not None]
    return [MotionPoint(r.frame_index, r.selected.center_x, r.selected.center_y) for r in accepted]


def test_vt8_reacq_requires_multiframe_confirmation():
    frames = _frames()
    train, reader = _train(frames)
    history = _history(train)

    track = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[p.xy for p in history],
        initial_history_frames=[p.frame_index for p in history],
        detector_config=_detector_cfg(),
        config=TrackingConfig(search_radius_px=65, max_search_radius_px=130, min_confidence_gap=0.0, lost_after_misses=2),
    )
    assert track.status == "LOST"
    assert track.lost_frame == 10
    history.extend(
        MotionPoint(f.frame_index, f.selected.center_x, f.selected.center_y)
        for f in track.frames if f.selected is not None
    )

    reacq = run_reacquisition(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=track.lost_frame,
        reference_bank=train.reference_bank,
        motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=90,
            max_radius_px=170,
            radius_growth_px_per_frame=15,
            confirm_frames=3,
            max_frames=12,
            min_confidence_gap=0.0,
            max_confirmation_error_px=80,
        ),
    )
    assert reacq.status == "CONFIRMED"
    selected = [f for f in reacq.frames if f.selected is not None]
    assert len(selected) >= 3
    assert selected[-3].status == "TENTATIVE"
    assert selected[-2].status == "TENTATIVE"
    assert selected[-1].status == "CONFIRMED"
    assert reacq.confirmed_frame == 15
    assert reacq.t_reacq_frames == 5
    assert len(reacq.confirmation_points) == 3
    assert all(f.processing_time_ms > 0 for f in reacq.frames)


def test_vt8_session_resumes_track_after_reacq():
    frames = _frames(count=25)
    train, reader = _train(frames)
    accepted = [r for r in train.frames if r.selected is not None]

    session = run_tracking_session(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[r.selected.center for r in accepted],
        initial_history_frames=[r.frame_index for r in accepted],
        detector_config=_detector_cfg(),
        tracking_config=TrackingConfig(search_radius_px=65, max_search_radius_px=130, min_confidence_gap=0.0, lost_after_misses=2),
        reacquisition_config=ReacquisitionConfig(
            initial_radius_px=90,
            max_radius_px=170,
            radius_growth_px_per_frame=15,
            confirm_frames=3,
            max_frames=12,
            min_confidence_gap=0.0,
            max_confirmation_error_px=80,
        ),
        session_config=TrackingSessionConfig(max_reacq_cycles=3),
    )
    assert session.status == "COMPLETED"
    assert session.lost_events == 1
    assert session.reacq_successes == 1
    assert len(session.track_segments) == 2
    assert len(session.reacquisitions) == 1
    assert session.t_reacq_frames == (5,)
    assert session.mean_t_reacq_frames == pytest.approx(5.0)
    assert session.max_t_reacq_frames == 5
    assert session.final_frame == len(frames) - 1
    assert any(fi >= 16 for fi, _x, _y in session.trajectory)


def test_vt8_reacq_does_not_confirm_single_matching_frame():
    # Target reappears for exactly one frame and disappears again. The REACQ
    # rule must not promote that single match to a confirmed target.
    visible = {13}
    frames = []
    for k in range(20):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if k <= 8 or k in visible:
            _draw_target(frame, k)
        frames.append(frame)
    train, reader = _train(frames)
    history = _history(train)
    # Add the two valid tracking points before loss so motion prediction is realistic.
    for k in (7, 8):
        history.append(MotionPoint(k, 38 + 3 * k + 5.5, 57.0))

    reacq = run_reacquisition(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=10,
        reference_bank=train.reference_bank,
        motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=100,
            max_radius_px=180,
            confirm_frames=3,
            max_frames=8,
            min_confidence_gap=0.0,
            max_confirmation_error_px=90,
        ),
    )
    assert reacq.status != "CONFIRMED"
    assert reacq.confirmed_frame is None
    assert reacq.t_reacq_frames is None
    assert any(f.status == "TENTATIVE" for f in reacq.frames)


def test_vt8_reacq_search_radius_expands():
    frames = _frames(count=20, invisible=range(9, 20))
    train, reader = _train(frames)
    history = _history(train)
    reacq = run_reacquisition(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        lost_frame=10,
        reference_bank=train.reference_bank,
        motion_history=history,
        detector_config=_detector_cfg(),
        config=ReacquisitionConfig(
            initial_radius_px=70,
            max_radius_px=150,
            radius_growth_px_per_frame=20,
            max_frames=5,
            confirm_frames=3,
        ),
    )
    radii = [f.search_radius_px for f in reacq.frames]
    assert radii == sorted(radii)
    assert radii[-1] > radii[0]
    assert max(radii) <= 150


def test_vt8_config_validation():
    with pytest.raises(ValueError):
        ReacquisitionConfig(confirm_frames=1).validate()
    with pytest.raises(ValueError):
        ReacquisitionConfig(initial_radius_px=200, max_radius_px=100).validate()
    with pytest.raises(ValueError):
        TrackingSessionConfig(max_reacq_cycles=0).validate()
