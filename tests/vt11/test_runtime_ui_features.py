from __future__ import annotations

from navmin.contracts import CameraRole, TargetSelectionRequest
from navmin.vision.processor import VisionRuntimeSettings
from navmin.vision.processors.vt11.processor import (
    Vt11ProcessorState,
    Vt11VisionProcessor,
)

from .test_candidate_selection_runtime import _packet


def _acquire_target(processor: Vt11VisionProcessor) -> None:
    processor.process(_packet(0), processing_enabled=False)
    request = TargetSelectionRequest(
        selection_id=31,
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=0,
        x_px=80.0,
        y_px=54.0,
        candidate_selection_enabled=True,
        auto_select_candidate_on_first_click=True,
        candidate_click_tolerance_px=15,
    )
    assert processor.apply_target_control(request) == ()
    for frame_id in range(1, 7):
        processor.process(_packet(frame_id))
    assert processor.state is Vt11ProcessorState.TARGET


def test_day_low_light_setting_atomically_controls_exactly_three_live_flags() -> None:
    processor = Vt11VisionProcessor(profile_name="day")

    assert processor.apply_runtime_settings(VisionRuntimeSettings(True)) is None
    enabled = processor.live_feature_snapshot
    assert enabled.visible_blur_robust
    assert enabled.day_maneuver_search_enabled
    assert enabled.visible_line_recovery_enabled
    assert processor._target_identity_weights["_visible_blur_robust"] == 1.0

    assert processor.apply_runtime_settings(VisionRuntimeSettings(False)) is None
    disabled = processor.live_feature_snapshot
    assert not disabled.visible_blur_robust
    assert not disabled.day_maneuver_search_enabled
    assert not disabled.visible_line_recovery_enabled
    assert "_visible_blur_robust" not in processor._target_identity_weights


def test_thermal_ignores_low_light_setting() -> None:
    processor = Vt11VisionProcessor(profile_name="thermal")

    processor.apply_runtime_settings(VisionRuntimeSettings(True))

    snapshot = processor.live_feature_snapshot
    assert not snapshot.visible_blur_robust
    assert not snapshot.day_maneuver_search_enabled
    assert not snapshot.visible_line_recovery_enabled
    assert "_visible_blur_robust" not in processor._target_identity_weights


def test_low_light_runtime_update_does_not_reset_active_target_session() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    _acquire_target(processor)
    session = processor._target_session
    target = processor._current_target
    generation = processor._generation
    assert session is not None

    processor.apply_runtime_settings(VisionRuntimeSettings(True))

    assert processor.state is Vt11ProcessorState.TARGET
    assert processor._target_session is session
    assert processor._current_target == target
    assert processor._generation == generation
    assert session.live_features is processor._live_features
    assert session.identity_weights is processor._target_identity_weights
    assert session.identity_weights["_visible_blur_robust"] == 1.0

    output = processor.process(_packet(7))
    assert output.result.tracked_objects


def test_low_light_during_train_does_not_change_initial_train_identity_weights() -> None:
    processor = Vt11VisionProcessor(profile_name="day")
    processor.process(_packet(0), processing_enabled=False)
    request = TargetSelectionRequest(
        selection_id=32,
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=0,
        x_px=80.0,
        y_px=54.0,
        candidate_selection_enabled=True,
        auto_select_candidate_on_first_click=True,
        candidate_click_tolerance_px=15,
    )
    assert processor.apply_target_control(request) == ()
    assert processor.state is Vt11ProcessorState.ACQUIRING

    processor.apply_runtime_settings(VisionRuntimeSettings(True))

    assert processor._acquisition is not None
    attempt = processor._acquisition.attempt
    assert "_visible_blur_robust" not in attempt.resolved_config.identity_weights
    assert attempt.target_identity_weights is processor._target_identity_weights
    assert attempt.target_identity_weights["_visible_blur_robust"] == 1.0
