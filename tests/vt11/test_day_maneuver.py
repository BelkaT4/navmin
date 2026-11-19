import cv2
import numpy as np
import pytest

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectorConfig,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter
from navmin.vision.processors.vt11.identity.descriptor import descriptor_from_candidate
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile
from navmin.vision.processors.vt11.tracking.maneuver import (
    ManeuverEvidence,
    confirms_maneuver,
)
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig, run_tracking


@pytest.mark.parametrize(
    "center,frame,area,identity",
    [
        ((120, 88), 4, 350, 0.8),
        ((160, 88), 4, 350, 0.8),
        ((200, 88), 4, 350, 0.8),
        ((120, 88), 5, 350, 0.8),
        ((120, 88), 4, 1500, 0.8),
        ((120, 88), 4, 350, 0.4),
    ],
)
def test_temporal_confirmation_rejects_inconsistent_hypotheses(
    center, frame, area, identity
):
    first = ManeuverEvidence(3, (160, 88), 350, 0.8)
    second = ManeuverEvidence(frame, center, area, identity)
    expected = center == (120, 88) and frame == 4 and area == 350 and identity == 0.8
    assert confirms_maneuver(first, second, (200, 88)) == expected


def _image(x):
    image = np.full((180, 360, 3), 28, np.uint8)
    if x is not None:
        cv2.rectangle(image, (x - 10, 80), (x + 10, 96), (225, 225, 225), -1)
        cv2.rectangle(image, (x + 7, 83), (x + 16, 89), (150, 150, 150), -1)
    return image


def _run(enabled=True, positions=(160, 120, 80), mode="visible", distractor=False):
    detector = CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3,
        max_area_px=1200,
    )
    initial = _image(200)
    selection = make_operator_selection(
        frame_index=0,
        click_xy=(200, 88),
        radius_px=70,
        frame_width=360,
        frame_height=180,
    )
    candidate = detect_local_candidates(initial, selection, config=detector).candidates[0]
    descriptor = descriptor_from_candidate(initial, candidate)
    bank = ReferenceBank(descriptors=[descriptor])
    adapter = EMAReferenceAdapter(bank, EMAConfig(mode="adaptive_memory"))
    frames = [initial] * 3 + [_image(x) for x in positions]
    if distractor:
        for frame in frames[3:]:
            frame[130:150] = frame[80:100]
    result = run_tracking(
        frame_reader=lambda i: (i, frames[i]),
        frame_count=len(frames),
        frame_width=360,
        frame_height=180,
        start_frame=3,
        reference_bank=bank,
        initial_history=[(160, 88), (180, 88), (200, 88)],
        detector_config=detector,
        config=TrackingConfig(
            day_maneuver_search_enabled=enabled,
            search_radius_px=70,
            dynamic_search_enabled=False,
            min_track_score=0.8,
            motion_sigma_px=8,
        ),
        thermal_profile=ThermalProfile(
            mode,
            "off",
            np.zeros((180, 360), np.uint8),
            (),
        ),
        adaptation_adapter=adapter,
    )
    assert bank.descriptors[0] is descriptor
    return result, adapter


def test_two_observations_restore_target_and_seed_new_velocity_without_training():
    result, adapter = _run()
    assert [frame.status for frame in result.frames] == [
        "LOW_CONFIDENCE",
        "TRACK",
        "TRACK",
    ]
    assert [frame.maneuver_hits for frame in result.frames] == [1, 2, 0]
    assert result.frames[1].diagnostic_reason == "ACCEPTED_DAY_MANEUVER_RELOCK"
    assert result.frames[1].adaptation_reason == "BLOCKED_DAY_MANEUVER_RELOCK"
    assert result.frames[2].adaptation_reason == "BLOCKED_DAY_MANEUVER_RELOCK"
    assert adapter.update_count == 0
    assert abs(result.frames[2].selected.center_x - 80) < 3


def test_disabled_branch_keeps_original_loss_and_does_not_emit_hypotheses():
    result, _ = _run(enabled=False)
    assert result.frames[-1].status == "LOST"
    assert all(frame.maneuver_hits == 0 for frame in result.frames)


def test_one_observation_then_disappearance_cannot_be_confirmed():
    result, adapter = _run(positions=(160, None, None))
    assert result.frames[0].maneuver_hits == 1
    assert result.frames[-1].status == "LOST"
    assert all(frame.selected is None for frame in result.frames)
    assert adapter.update_count == 0


def test_branch_cannot_override_thermal_profile():
    result, _ = _run(mode="thermal")
    assert all(frame.maneuver_hits == 0 for frame in result.frames)


def test_stationary_distractor_cannot_confirm_new_motion():
    result, _ = _run(positions=(160, 160, 160))
    assert result.frames[-1].status == "LOST"
    assert not any(frame.maneuver_hits == 2 for frame in result.frames)


def test_similar_competing_objects_cannot_confirm_ambiguous_target():
    result, _ = _run(distractor=True)
    assert result.frames[-1].status == "LOST"
    assert not any(frame.maneuver_hits == 2 for frame in result.frames)


def test_adaptation_freeze_includes_confirmation_frame_plus_three():
    width = 520

    def image(x):
        frame = np.full((180, width, 3), 28, np.uint8)
        cv2.rectangle(frame, (x - 10, 80), (x + 10, 96), (225, 225, 225), -1)
        cv2.rectangle(frame, (x + 7, 83), (x + 16, 89), (150, 150, 150), -1)
        return frame

    detector = CandidateDetectorConfig(
        threshold_sigma=1.8,
        min_response=3,
        max_area_px=1200,
    )
    initial = image(300)
    selection = make_operator_selection(
        frame_index=0,
        click_xy=(300, 88),
        radius_px=70,
        frame_width=width,
        frame_height=180,
    )
    candidate = detect_local_candidates(initial, selection, config=detector).candidates[0]
    descriptor = descriptor_from_candidate(initial, candidate)
    bank = ReferenceBank(descriptors=[descriptor])
    adapter = EMAReferenceAdapter(bank, EMAConfig(mode="adaptive_memory"))
    frames = [initial] * 3 + [image(x) for x in (260, 220, 180, 140, 100, 60)]

    result = run_tracking(
        frame_reader=lambda i: (i, frames[i]),
        frame_count=len(frames),
        frame_width=width,
        frame_height=180,
        start_frame=3,
        reference_bank=bank,
        initial_history=[(260, 88), (280, 88), (300, 88)],
        detector_config=detector,
        config=TrackingConfig(
            day_maneuver_search_enabled=True,
            search_radius_px=70,
            dynamic_search_enabled=False,
            min_track_score=0.8,
            motion_sigma_px=8,
        ),
        thermal_profile=ThermalProfile(
            "visible",
            "off",
            np.zeros((180, width), np.uint8),
            (),
        ),
        adaptation_adapter=adapter,
    )

    assert result.frames[1].diagnostic_reason == "ACCEPTED_DAY_MANEUVER_RELOCK"
    assert [frame.adaptation_reason for frame in result.frames[1:5]] == [
        "BLOCKED_DAY_MANEUVER_RELOCK",
        "BLOCKED_DAY_MANEUVER_RELOCK",
        "BLOCKED_DAY_MANEUVER_RELOCK",
        "BLOCKED_DAY_MANEUVER_RELOCK",
    ]
    assert result.frames[5].adaptation_reason != "BLOCKED_DAY_MANEUVER_RELOCK"
