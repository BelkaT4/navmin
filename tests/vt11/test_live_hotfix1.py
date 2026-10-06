import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig
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


def _frames(count=22, invisible=()):
    invisible = set(invisible)
    out = []
    for k in range(count):
        frame = np.full((120, 180, 3), 28, np.uint8)
        if k not in invisible:
            _draw_target(frame, k)
        out.append(frame)
    return out


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
    accepted = [x for x in result.frames if x.selected is not None]
    return result, accepted, reader


def test_hotfix1_tracking_emits_one_live_event_per_processed_frame():
    frames = _frames(15)
    train, accepted, reader = _train(frames)
    events = []
    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[x.selected.center for x in accepted],
        initial_history_frames=[x.frame_index for x in accepted],
        detector_config=_detector_cfg(),
        config=TrackingConfig(search_radius_px=65, max_search_radius_px=130, min_confidence_gap=0.0),
        progress_callback=events.append,
    )
    assert result.status == "COMPLETED"
    assert len(events) == result.attempted_frames
    assert [e.mode for e in events] == ["TRACK"] * len(events)
    assert [e.frame_index for e in events] == [x.frame_index for x in result.frames]
    assert all(e.frame_bgr.shape == (120, 180, 3) for e in events)
    assert all(e.result.frame_index == e.frame_index for e in events)


def test_hotfix1_stop_request_returns_stopped_not_lost():
    frames = _frames(18)
    train, accepted, reader = _train(frames)
    events = []

    result = run_tracking(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[x.selected.center for x in accepted],
        initial_history_frames=[x.frame_index for x in accepted],
        detector_config=_detector_cfg(),
        config=TrackingConfig(search_radius_px=65, max_search_radius_px=130, min_confidence_gap=0.0),
        progress_callback=events.append,
        stop_requested=lambda: len(events) >= 3,
    )
    assert result.status == "STOPPED"
    assert result.lost_frame is None
    assert result.attempted_frames == 3
    assert len(events) == 3


def test_hotfix1_session_stream_contains_track_lost_reacq_and_return_to_track():
    frames = _frames(25, invisible=range(9, 13))
    train, accepted, reader = _train(frames)
    events = []
    result = run_tracking_session(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[x.selected.center for x in accepted],
        initial_history_frames=[x.frame_index for x in accepted],
        detector_config=_detector_cfg(),
        tracking_config=TrackingConfig(
            search_radius_px=65,
            max_search_radius_px=130,
            min_confidence_gap=0.0,
            lost_after_misses=2,
        ),
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
        progress_callback=events.append,
    )
    assert result.status == "COMPLETED"
    modes = [e.mode for e in events]
    assert "TRACK" in modes and "REACQ" in modes
    lost_positions = [i for i, e in enumerate(events) if e.mode == "TRACK" and e.result.status == "LOST"]
    confirmed_positions = [i for i, e in enumerate(events) if e.mode == "REACQ" and e.result.status == "CONFIRMED"]
    assert lost_positions and confirmed_positions
    assert lost_positions[0] < confirmed_positions[0]
    assert any(i > confirmed_positions[0] and e.mode == "TRACK" for i, e in enumerate(events))


def test_hotfix1_session_stop_is_propagated():
    frames = _frames(20)
    train, accepted, reader = _train(frames)
    events = []
    result = run_tracking_session(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=180,
        frame_height=120,
        start_frame=7,
        reference_bank=train.reference_bank,
        initial_history=[x.selected.center for x in accepted],
        initial_history_frames=[x.frame_index for x in accepted],
        detector_config=_detector_cfg(),
        tracking_config=TrackingConfig(search_radius_px=65, max_search_radius_px=130, min_confidence_gap=0.0),
        reacquisition_config=ReacquisitionConfig(initial_radius_px=90, max_radius_px=170),
        progress_callback=events.append,
        stop_requested=lambda: len(events) >= 2,
    )
    assert result.status == "STOPPED"
    assert len(events) == 2
