from __future__ import annotations

import numpy as np

from navmin.contracts import (
    CameraRole,
    FramePacket,
    TargetCandidateSelectionRequest,
    TargetSelectionRequest,
)
from navmin.vision.processors.vt11.processor import (
    Vt11ProcessorState,
    Vt11VisionProcessor,
)


def _packet(
    frame_id: int,
    *,
    target_visible: bool = True,
    camera: CameraRole = CameraRole.OVERVIEW,
) -> FramePacket:
    image = np.full((120, 160, 3), 30, dtype=np.uint8)
    if target_visible:
        image[48:60, 74:88] = 220
    image.flags.writeable = False
    return FramePacket(
        camera=camera,
        generation=1,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=frame_id * 10_000_000,
        image=image,
    )


def _request(
    *,
    auto: bool = False,
    tolerance: int = 15,
    camera: CameraRole = CameraRole.OVERVIEW,
) -> TargetSelectionRequest:
    return TargetSelectionRequest(
        selection_id=11,
        camera=camera,
        generation=1,
        frame_id=0,
        x_px=80.0,
        y_px=54.0,
        candidate_selection_enabled=True,
        auto_select_candidate_on_first_click=auto,
        candidate_click_tolerance_px=tolerance,
    )


def _manual_candidates(processor: Vt11VisionProcessor):
    processor.process(_packet(0), processing_enabled=False)
    assert processor.apply_target_control(_request()) == ()
    assert processor.state is Vt11ProcessorState.CANDIDATES
    output = processor.process(_packet(1))
    candidates = output.acquisition_candidates
    assert candidates is not None
    assert candidates.selection_id == 11
    assert candidates.frame_id == 1
    assert candidates.candidates
    return candidates


def test_vt11_candidate_discovery_is_camera_role_neutral_for_stereo_right() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    processor.process(
        _packet(0, camera=CameraRole.STEREO_RIGHT),
        processing_enabled=False,
    )
    assert processor.apply_target_control(
        _request(camera=CameraRole.STEREO_RIGHT)
    ) == ()
    assert processor.state is Vt11ProcessorState.CANDIDATES

    output = processor.process(_packet(1, camera=CameraRole.STEREO_RIGHT))
    candidates = output.acquisition_candidates
    assert candidates is not None
    assert candidates.camera is CameraRole.STEREO_RIGHT
    assert candidates.selection_id == 11
    assert candidates.candidates


def test_manual_candidate_selection_starts_train_from_exact_displayed_frame() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    candidates = _manual_candidates(processor)
    candidate = candidates.candidates[0]

    events = processor.apply_target_control(
        TargetCandidateSelectionRequest(
            selection_id=candidates.selection_id,
            camera=candidates.camera,
            generation=candidates.generation,
            frame_id=candidates.frame_id,
            candidate_id=candidate.candidate_id,
            x_px=82.0,
            y_px=55.0,
        )
    )

    assert events == ()
    assert processor.state is Vt11ProcessorState.ACQUIRING
    assert processor._acquisition is not None
    assert processor._acquisition.attempt.start_frame == candidates.frame_id
    assert processor._acquisition.attempt._train_click_xy == (82.0, 55.0)


def test_candidate_selection_rejects_stale_or_unknown_proposal_without_state_change() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    candidates = _manual_candidates(processor)
    candidate = candidates.candidates[0]

    stale = TargetCandidateSelectionRequest(
        selection_id=candidates.selection_id,
        camera=candidates.camera,
        generation=candidates.generation,
        frame_id=0,
        candidate_id=candidate.candidate_id,
        x_px=80.0,
        y_px=54.0,
    )
    unknown = TargetCandidateSelectionRequest(
        selection_id=candidates.selection_id,
        camera=candidates.camera,
        generation=candidates.generation,
        frame_id=candidates.frame_id,
        candidate_id=9999,
        x_px=80.0,
        y_px=54.0,
    )

    assert processor.apply_target_control(stale) == ()
    assert processor.apply_target_control(unknown) == ()
    assert processor.state is Vt11ProcessorState.CANDIDATES


def test_auto_select_starts_train_on_first_click_when_candidate_matches_click() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    processor.process(_packet(0), processing_enabled=False)

    assert processor.apply_target_control(_request(auto=True)) == ()

    assert processor.state is Vt11ProcessorState.ACQUIRING
    assert processor._acquisition is not None
    assert processor._acquisition.attempt.start_frame == 0


def test_failed_initial_train_returns_to_candidate_selection_without_terminal_failure() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    candidates = _manual_candidates(processor)
    candidate = candidates.candidates[0]
    assert processor.apply_target_control(
        TargetCandidateSelectionRequest(
            selection_id=candidates.selection_id,
            camera=candidates.camera,
            generation=candidates.generation,
            frame_id=candidates.frame_id,
            candidate_id=candidate.candidate_id,
            x_px=80.0,
            y_px=54.0,
        )
    ) == ()

    observed_events = []
    for frame_id in range(2, 8):
        output = processor.process(_packet(frame_id, target_visible=False))
        observed_events.extend(output.target_events)

    assert observed_events == []
    assert processor.state is Vt11ProcessorState.CANDIDATES
    assert processor.process(_packet(8, target_visible=False)).acquisition_candidates is not None
