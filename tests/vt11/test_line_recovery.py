from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectorConfig, _suppress_lines, _thin_line_overlap_core,
    config_from_dict, detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig, run_initial_train
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig, run_reacquisition
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


def _mask(body=False):
    mask = np.zeros((180, 400), np.uint8)
    cv2.line(mask, (0, 90), (399, 90), 1, 2)
    if body:
        cv2.ellipse(mask, (200, 90), (28, 6), 0, 0, 360, 1, -1)
    return mask


def test_recovers_thin_body_from_wire_without_restoring_the_wire():
    _, labels, stats, _ = cv2.connectedComponentsWithStats(_mask(True))
    baseline, _, _ = _suppress_lines(labels, stats, 1600)
    recovered, flags, _ = _suppress_lines(labels, stats, 1600, recover_blurred_cores=True)
    assert not baseline.any()
    assert recovered[90, 200]
    assert not recovered[90, 30]
    assert cv2.boundingRect(recovered)[2] < 80
    assert np.count_nonzero(flags == 2) > 300


@pytest.mark.parametrize('kind', ('wire', 'cross', 'branches', 'thick_wire'))
def test_does_not_make_a_target_from_uniform_lines_or_simple_intersections(kind):
    mask = _mask()
    if kind == 'cross':
        cv2.line(mask, (200, 0), (200, 179), 1, 2)
    elif kind == 'branches':
        cv2.line(mask, (0, 40), (399, 140), 1, 2)
        cv2.line(mask, (0, 140), (399, 40), 1, 2)
    elif kind == 'thick_wire':
        cv2.line(mask, (0, 90), (399, 90), 1, 10)
    assert not _thin_line_overlap_core(mask.astype(bool), 1600).any()


def test_preserves_previously_recovered_compact_core():
    mask = _mask()
    cv2.ellipse(mask, (200, 90), (14, 9), 0, 0, 360, 1, -1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    before = _suppress_lines(labels, stats, 400)
    after = _suppress_lines(labels, stats, 400, recover_blurred_cores=True)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_option_defaults_to_off():
    assert not CandidateDetectorConfig().visible_line_recovery_enabled
    assert config_from_dict({'visible_line_recovery_enabled': True}).visible_line_recovery_enabled


def _sequence(x):
    frame = np.full((160, 320, 3), 28, np.uint8)
    if x is not None:
        cv2.ellipse(frame, (x, 80), (14, 8), 0, 0, 360, (230, 230, 230), -1)
        cv2.rectangle(frame, (x+9, 77), (x+18, 82), (145, 145, 145), -1)
    return frame


def _run_recovered_sequence(positions):
    detector = CandidateDetectorConfig(threshold_sigma=1.8, min_response=3, max_area_px=1500)
    initial = _sequence(160)
    sel = make_operator_selection(frame_index=0, click_xy=(160, 80), radius_px=70, frame_width=320, frame_height=160)
    candidate = detect_local_candidates(initial, sel, config=detector).candidates[0]
    reference = descriptor_from_candidate(initial, candidate)
    bank = ReferenceBank(descriptors=[reference])
    adapter = EMAReferenceAdapter(bank, EMAConfig(mode='adaptive_memory'))
    frames = [initial]*3 + [_sequence(x) for x in positions]

    def recovered(*args, **kwargs):
        result = detect_local_candidates(*args, **kwargs)
        return replace(result, candidates=tuple(replace(c, detection_channel='LINE_RECOVERY', separated_from_line=True)
                                                for c in result.candidates))

    with patch('navmin.vision.processors.vt11.tracking.tracker.detect_local_candidates', recovered):
        result = run_tracking(
            frame_reader=lambda i: (i, frames[i]), frame_count=len(frames), frame_width=320, frame_height=160,
            start_frame=3, reference_bank=bank, initial_history=[(154, 80), (157, 80), (160, 80)],
            detector_config=detector, config=TrackingConfig(search_radius_px=70, dynamic_search_enabled=False),
            thermal_profile=ThermalProfile('visible', 'off', np.zeros((160, 320), np.uint8), ()),
            adaptation_adapter=adapter,
        )
    assert bank.descriptors[0] is reference
    return result, adapter


def test_requires_two_observations_and_blocks_training_on_recovered_component():
    result, adapter = _run_recovered_sequence((163, 166, 169))
    assert [r.status for r in result.frames] == ['LOW_CONFIDENCE', 'TRACK', 'TRACK']
    assert result.frames[0].selected is None
    assert result.frames[1].diagnostic_reason == 'ACCEPTED_LINE_RECOVERY_CONFIRMED'
    assert result.frames[1].line_recovery_hits == 2
    assert result.frames[1].adaptation_reason == 'BLOCKED_LINE_OVERLAP'
    assert adapter.update_count == 0
    assert adapter.bank.adaptive_descriptor is None
    assert adapter.bank.fast_descriptor is None
    assert adapter.bank.recent_count == 0


def test_single_recovered_object_then_disappearance_is_never_accepted():
    result, adapter = _run_recovered_sequence((163, None, None))
    assert not any(r.selected is not None for r in result.frames)
    assert result.frames[-1].status == 'LOST'
    assert adapter.update_count == 0
    assert adapter.bank.adaptive_descriptor is None
    assert adapter.bank.fast_descriptor is None
    assert adapter.bank.recent_count == 0


def test_reacq_line_recovery_keeps_temporal_confirmation_without_scale_learning():
    detector = CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3,
        max_area_px=2200,
        adaptive_max_area_ratio=4.0,
    )
    initial = _sequence(160)
    sel = make_operator_selection(
        frame_index=0, click_xy=(160, 80), radius_px=80, frame_width=320, frame_height=160
    )
    initial_candidate = detect_local_candidates(initial, sel, config=detector).candidates[0]
    reference = descriptor_from_candidate(initial, initial_candidate)
    bank = ReferenceBank(descriptors=[reference])
    initial_area = float(initial_candidate.area_px)

    frames = [initial.copy() for _ in range(3)]
    for x in (164, 168, 172, 176):
        frame = np.full((160, 320, 3), 28, np.uint8)
        cv2.ellipse(frame, (x, 80), (18, 10), 0, 0, 360, (230, 230, 230), -1)
        cv2.rectangle(frame, (x + 12, 77), (x + 21, 82), (145, 145, 145), -1)
        frames.append(frame)

    original_detect = detect_local_candidates

    def recovered(*args, **kwargs):
        result = original_detect(*args, **kwargs)
        return replace(
            result,
            candidates=tuple(
                replace(c, detection_channel="LINE_RECOVERY", separated_from_line=True)
                for c in result.candidates
            ),
        )

    history = [
        MotionPoint(0, 154.0, 80.0),
        MotionPoint(1, 157.0, 80.0),
        MotionPoint(2, 160.0, 80.0),
    ]
    with patch(
        "navmin.vision.processors.vt11.tracking.reacquisition.detect_local_candidates",
        recovered,
    ):
        result = run_reacquisition(
            frame_reader=lambda i: (i, frames[i]),
            frame_count=len(frames),
            frame_width=320,
            frame_height=160,
            lost_frame=2,
            reference_bank=bank,
            motion_history=history,
            detector_config=detector,
            config=ReacquisitionConfig(
                initial_radius_px=90,
                max_radius_px=150,
                radius_growth_px_per_frame=10,
                confirm_frames=3,
                confirm_window_frames=5,
                max_frames=4,
                min_confidence_gap=0.0,
                max_confirmation_error_px=100,
            ),
            initial_expected_area_px=initial_area,
        )

    assert result.status == "CONFIRMED"
    assert result.confirmed_frame == 5
    assert result.frames[-1].candidate_channel == "LINE_RECOVERY"
    assert result.last_expected_area_px == pytest.approx(initial_area)


def test_initial_train_does_not_learn_line_separated_candidates():
    detector = CandidateDetectorConfig(threshold_sigma=1.8, min_response=3, max_area_px=1500)
    frames = [_sequence(160 + 2 * i) for i in range(7)]
    selection = make_operator_selection(
        frame_index=0, click_xy=(160, 80), radius_px=70, frame_width=320, frame_height=160
    )
    initial_candidate = detect_local_candidates(frames[0], selection, config=detector).candidates[0]
    original_detect = detect_local_candidates

    def separated(*args, **kwargs):
        result = original_detect(*args, **kwargs)
        return replace(
            result,
            candidates=tuple(replace(c, separated_from_line=True) for c in result.candidates),
        )

    with patch(
        "navmin.vision.processors.vt11.acquisition.initial_train.detect_local_candidates",
        separated,
    ):
        result = run_initial_train(
            frame_reader=lambda i: (i, frames[i]),
            frame_count=len(frames),
            frame_width=320,
            frame_height=160,
            start_frame=0,
            initial_candidate=initial_candidate,
            roi_radius_px=70,
            detector_config=detector,
            config=InitialTrainConfig(
                initial_frames=7,
                top_k=5,
                bank_update_min_score=0.0,
            ),
        )

    assert result.decision == "CONFIRMED"
    assert len(result.reference_bank.descriptors) == 1
