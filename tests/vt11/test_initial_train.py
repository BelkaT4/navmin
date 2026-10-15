import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig, detect_local_candidates
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, _motion_prediction, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate, descriptor_similarity
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank


def _sequence(count=7, *, blank_after=None):
    frames = []
    for k in range(count):
        f = np.full((120, 160, 3), 30, np.uint8)
        if blank_after is None or k < blank_after:
            x = 45 + 3 * k
            cv2.rectangle(f, (x, 55), (x + 7, 62), (220, 220, 220), -1)
            cv2.circle(f, (95, 58), 5, (210, 210, 210), -1)
        frames.append(f)
    return frames


def _detector_cfg():
    return CandidateDetectorConfig(threshold_sigma=2.0, min_area_px=2, max_area_px=1000)


def _first_candidate(frame):
    sel = make_operator_selection(frame_index=0, click_xy=(49, 59), radius_px=70, frame_width=160, frame_height=120)
    return detect_local_candidates(frame, sel, top_k=5, config=_detector_cfg()).candidates[0]


def test_motion_prediction_constant_velocity():
    assert _motion_prediction([(10.0, 20.0), (13.0, 22.0), (16.0, 24.0)], (0.0, 0.0)) == (19.0, 26.0)


def test_descriptor_self_similarity_is_high():
    frame = _sequence(1)[0]
    cand = _first_candidate(frame)
    desc = descriptor_from_candidate(frame, cand)
    sim = descriptor_similarity(desc, desc)
    assert sim.total > 0.99
    assert sim.appearance > 0.99
    assert sim.edge > 0.99


def test_reference_bank_is_bounded():
    frame = _sequence(1)[0]
    desc = descriptor_from_candidate(frame, _first_candidate(frame))
    bank = ReferenceBank(max_size=3)
    for _ in range(5):
        bank.add(desc)
    assert len(bank) == 3


def test_initial_train_confirms_consistent_moving_target():
    frames = _sequence(7)
    first = _first_candidate(frames[0])

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=160,
        frame_height=120,
        start_frame=0,
        initial_candidate=first,
        roi_radius_px=70,
        detector_config=_detector_cfg(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "CONFIRMED"
    assert result.valid_fraction == 1.0
    assert len(result.reference_bank) == 5
    assert result.last_center is not None
    assert result.last_center[0] > first.center_x


def test_initial_train_rejects_sequence_that_disappears_immediately():
    frames = _sequence(7, blank_after=1)
    first = _first_candidate(frames[0])

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=len(frames),
        frame_width=160,
        frame_height=120,
        start_frame=0,
        initial_candidate=first,
        roi_radius_px=70,
        detector_config=_detector_cfg(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "AMBIGUOUS"
    assert result.valid_fraction < 0.5


def test_initial_train_does_not_confirm_from_single_frame_at_eof():
    frames = _sequence(1)
    first = _first_candidate(frames[0])

    def reader(index):
        return index, frames[index].copy()

    result = run_initial_train(
        frame_reader=reader,
        frame_count=1,
        frame_width=160,
        frame_height=120,
        start_frame=0,
        initial_candidate=first,
        roi_radius_px=70,
        detector_config=_detector_cfg(),
        config=InitialTrainConfig(initial_frames=7, top_k=5),
    )
    assert result.decision == "AMBIGUOUS"
    assert result.median_score == 0.0
