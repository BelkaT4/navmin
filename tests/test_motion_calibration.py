from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from navmin.calibration import load_stereo_calibration, stereo_camera_models
from navmin.config.models import AxesConfig, AxisMechanicsConfig
from navmin.contracts import CameraRay, CameraRole, FramePacket, MoveRelativeCommand
from navmin.motion_calibration import (
    AxisMotionCalibration,
    CameraMotionCalibration,
    CurvePoint,
    DirectionCurve,
    MechanicalIdentity,
    MotionCalibrationDocument,
    MotionCalibrationError,
    MotionCalibrationRuntime,
    MotionCalibrationWizard,
    MotionProfile,
    RawObservation,
    ReversalLoss,
    WizardPhase,
    WizardSettings,
    assess_calibration_plan,
    build_axis_calibration,
    build_calibration_sequence,
    calibration_click_guide,
    motion_calibration_from_mapping,
    motion_calibration_to_mapping,
    validate_angles,
)


def _mechanics(*, limit: float = 30.0, invert: bool = False) -> AxisMechanicsConfig:
    return AxisMechanicsConfig(
        invert=invert,
        full_steps_per_revolution=200,
        microstep_divider=16,
        max_relative_move_deg=limit,
    )


def _axes(*, limit: float = 30.0) -> AxesConfig:
    return AxesConfig(_mechanics(limit=limit), _mechanics(limit=limit))


def _curve(
    *, transition_enabled: bool = True, command_boundary: float = 14.0
) -> DirectionCurve:
    return DirectionCurve(
        (
            CurvePoint(0.0, 0.0),
            CurvePoint(3.5, 4.0),
            CurvePoint(7.0, 8.0),
            CurvePoint(12.0, command_boundary),
        ),
        transition_enabled=transition_enabled,
    )


def _axis_profile(
    *,
    positive: DirectionCurve | None = None,
    negative: DirectionCurve | None = None,
    reversal: ReversalLoss | None = None,
) -> AxisMotionCalibration:
    return AxisMotionCalibration(
        positive=positive or _curve(),
        negative=negative or _curve(command_boundary=13.0),
        reversal_loss=reversal or ReversalLoss(0.5, 0.8),
        observations=(),
        quality_max_repeat_spread_deg=0.1,
    )


def _camera_profile(
    camera: CameraRole,
    *,
    identity: str = "sha256:camera",
    x: AxisMotionCalibration | None = None,
    y: AxisMotionCalibration | None = None,
) -> CameraMotionCalibration:
    mechanical = MechanicalIdentity.from_config(_mechanics())
    return CameraMotionCalibration(
        camera=camera,
        camera_calibration_identity=identity,
        mechanical_identity_x=mechanical,
        mechanical_identity_y=mechanical,
        calibration_limit_x_deg=30.0,
        calibration_limit_y_deg=30.0,
        motion_profile=MotionProfile(5.0, 10.0, 500),
        axes={
            axis: profile
            for axis, profile in (("x", x), ("y", y))
            if profile is not None
        },
        created_at_utc="2026-11-01T12:00:00+00:00",
    )


def _runtime(
    tmp_path: Path,
    *,
    profiles: dict[CameraRole, CameraMotionCalibration],
    axes: AxesConfig | None = None,
    percent: float = 20.0,
) -> MotionCalibrationRuntime:
    return MotionCalibrationRuntime(
        path=tmp_path / "motion.json",
        axes=axes or _axes(),
        camera_identities={
            CameraRole.OVERVIEW: "sha256:camera",
            CameraRole.STEREO_LEFT: "sha256:stereo",
        },
        document=MotionCalibrationDocument(
            extrapolation_percent=percent,
            profiles=profiles,
        ),
    )


def test_curve_interpolates_positive_negative_and_preserves_zero(tmp_path: Path) -> None:
    profile = _camera_profile(
        CameraRole.OVERVIEW,
        x=_axis_profile(),
        y=_axis_profile(),
    )
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: profile})

    zero = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(0.0, 0.0))
    positive = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(5.25, 0.0))
    negative = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(-5.25, 0.0))

    assert zero == MoveRelativeCommand(0.0, 0.0)
    assert positive.delta_x_deg == pytest.approx(6.0)
    assert negative.delta_x_deg == pytest.approx(-6.0)


@pytest.mark.parametrize(
    "points",
    [
        (CurvePoint(1.0, 0.0), CurvePoint(2.0, 1.0), CurvePoint(3.0, 2.0)),
        (CurvePoint(0.0, 0.0), CurvePoint(2.0, 2.0), CurvePoint(1.0, 3.0)),
        (CurvePoint(0.0, 0.0), CurvePoint(1.0, 2.0), CurvePoint(2.0, 1.0)),
        (CurvePoint(0.0, 0.0), CurvePoint(-1.0, 1.0), CurvePoint(2.0, 2.0)),
    ],
)
def test_invalid_curve_is_rejected(points: tuple[CurvePoint, ...]) -> None:
    with pytest.raises(MotionCalibrationError):
        DirectionCurve(points)


def test_transition_fades_correction_continuously_to_baseline(tmp_path: Path) -> None:
    profile = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: profile})

    trusted = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(12.0, 0.0))
    middle = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(13.2, 0.0))
    extended = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(14.4, 0.0))
    outside = runtime.correct(CameraRole.OVERVIEW, MoveRelativeCommand(15.0, 0.0))

    assert trusted.delta_x_deg == pytest.approx(14.0)
    assert middle.delta_x_deg == pytest.approx(14.2)
    assert extended.delta_x_deg == pytest.approx(14.4)
    assert outside.delta_x_deg == pytest.approx(15.0)


def test_asymmetric_ranges_are_applied_independently(tmp_path: Path) -> None:
    shorter = DirectionCurve(
        (CurvePoint(0, 0), CurvePoint(2, 3), CurvePoint(4, 5))
    )
    profile = _camera_profile(
        CameraRole.OVERVIEW,
        x=_axis_profile(negative=shorter),
    )
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: profile})

    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(6.0, 0.0)
    ).delta_x_deg != 6.0
    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(-6.0, 0.0)
    ).delta_x_deg == -6.0


def test_disabled_transition_keeps_trusted_curve_and_uses_baseline_afterward(
    tmp_path: Path,
) -> None:
    disabled = replace(_curve(), transition_enabled=False)
    profile = _camera_profile(
        CameraRole.OVERVIEW,
        x=_axis_profile(positive=disabled),
    )
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: profile})

    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(7.0, 0.0)
    ).delta_x_deg == 8.0
    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(12.1, 0.0)
    ).delta_x_deg == 12.1


def test_camera_specific_profiles_and_generic_baseline_are_independent(
    tmp_path: Path,
) -> None:
    overview = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    stereo = _camera_profile(
        CameraRole.STEREO_LEFT,
        identity="sha256:stereo",
        x=_axis_profile(positive=DirectionCurve(
            (CurvePoint(0, 0), CurvePoint(4, 5), CurvePoint(8, 10))
        )),
    )
    runtime = _runtime(
        tmp_path,
        profiles={CameraRole.OVERVIEW: overview, CameraRole.STEREO_LEFT: stereo},
    )

    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(4.0, 0.0)
    ).delta_x_deg == pytest.approx(4.5714286)
    assert runtime.correct(
        CameraRole.STEREO_LEFT, MoveRelativeCommand(4.0, 0.0)
    ).delta_x_deg == 5.0
    assert runtime.correct(
        CameraRole.STEREO_RIGHT, MoveRelativeCommand(4.0, 0.0)
    ) == MoveRelativeCommand(4.0, 0.0)
    generic_command = MoveRelativeCommand(4.0, 0.0)
    assert generic_command == MoveRelativeCommand(4.0, 0.0)


def test_missing_axis_profile_falls_back_to_baseline(tmp_path: Path) -> None:
    profile = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: profile})
    command = MoveRelativeCommand(4.0, 4.0)

    corrected = runtime.correct(CameraRole.OVERVIEW, command)

    assert corrected.delta_x_deg != command.delta_x_deg
    assert corrected.delta_y_deg == command.delta_y_deg


@pytest.mark.parametrize("change", ["steps", "microstep", "invert"])
def test_changed_mechanical_identity_disables_calibration(
    tmp_path: Path, change: str
) -> None:
    profile = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    base = _mechanics()
    changed = {
        "steps": replace(base, full_steps_per_revolution=400),
        "microstep": replace(base, microstep_divider=8),
        "invert": replace(base, invert=True),
    }[change]
    runtime = _runtime(
        tmp_path,
        profiles={CameraRole.OVERVIEW: profile},
        axes=AxesConfig(changed, base),
    )

    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(4.0, 0.0)
    ) == MoveRelativeCommand(4.0, 0.0)


def test_changed_camera_identity_disables_only_that_camera(tmp_path: Path) -> None:
    profile = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: profile})
    runtime.camera_identities[CameraRole.OVERVIEW] = "sha256:new"

    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(4.0, 0.0)
    ) == MoveRelativeCommand(4.0, 0.0)


def test_reduced_relative_limit_constrains_usable_calibration(tmp_path: Path) -> None:
    profile = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    runtime = _runtime(
        tmp_path,
        profiles={CameraRole.OVERVIEW: profile},
        axes=_axes(limit=8.0),
    )
    runtime.document.profiles[CameraRole.OVERVIEW] = replace(
        profile,
        mechanical_identity_x=MechanicalIdentity.from_config(_mechanics(limit=8.0)),
        mechanical_identity_y=MechanicalIdentity.from_config(_mechanics(limit=8.0)),
    )

    assert runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(7.5, 0.0)
    ).delta_x_deg == 7.5


def test_increased_relative_limit_does_not_expand_old_calibration_domain(
    tmp_path: Path,
) -> None:
    profile = replace(
        _camera_profile(CameraRole.OVERVIEW, x=_axis_profile()),
        calibration_limit_x_deg=15.0,
    )
    runtime = _runtime(
        tmp_path,
        profiles={CameraRole.OVERVIEW: profile},
        axes=_axes(limit=30.0),
        percent=50.0,
    )

    inside_old_limit = runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(14.5, 0.0)
    )
    outside_old_limit = runtime.correct(
        CameraRole.OVERVIEW, MoveRelativeCommand(16.0, 0.0)
    )

    assert inside_old_limit.delta_x_deg != 14.5
    assert outside_old_limit.delta_x_deg == 16.0

def test_reversal_loss_metadata_never_changes_runtime_command(tmp_path: Path) -> None:
    first = _camera_profile(
        CameraRole.OVERVIEW,
        x=_axis_profile(reversal=ReversalLoss(0.1, 0.2)),
    )
    second = replace(
        first,
        axes={"x": replace(first.axes["x"], reversal_loss=ReversalLoss(4.0, None))},
    )
    command = MoveRelativeCommand(5.0, 0.0)

    one = _runtime(tmp_path / "one", profiles={CameraRole.OVERVIEW: first})
    two = _runtime(tmp_path / "two", profiles={CameraRole.OVERVIEW: second})

    assert one.correct(CameraRole.OVERVIEW, command) == two.correct(
        CameraRole.OVERVIEW, command
    )


def _observation(
    command: float,
    measured: float,
    *,
    axis: str = "x",
    reversal: bool = False,
) -> RawObservation:
    return RawObservation(
        camera=CameraRole.OVERVIEW,
        axis=axis,
        commanded_angle_deg=command,
        direction="positive" if command > 0 else "negative",
        reversal=reversal,
        before_generation=1,
        after_generation=1,
        before_frame_id=1,
        after_frame_id=2,
        before_receive_timestamp_ns=10,
        after_receive_timestamp_ns=20,
        before_click_x_px=100,
        before_click_y_px=50,
        after_click_x_px=90,
        after_click_y_px=50,
        before_angular_coordinate_deg=0,
        after_angular_coordinate_deg=-measured,
        measured_movement_deg=math.copysign(abs(measured), command),
        motion_profile=MotionProfile(5, 10, 500),
    )


def _complete_observations(axis: str = "x") -> tuple[RawObservation, ...]:
    values: list[RawObservation] = []
    for command, measured in ((4, 3.5), (8, 7), (14, 12)):
        values.extend(
            (_observation(command, measured, axis=axis),) * 2
        )
    for command, measured in ((-4, -3.4), (-8, -6.8), (-14, -11.8)):
        values.extend(
            (_observation(command, measured, axis=axis),) * 2
        )
    values.extend(
        [
            _observation(-8, -6.3, axis=axis, reversal=True),
            _observation(-14, -11.3, axis=axis, reversal=True),
            _observation(8, 6.5, axis=axis, reversal=True),
            _observation(14, 11.5, axis=axis, reversal=True),
        ]
    )
    return tuple(values)


def test_reversal_samples_are_excluded_and_directional_losses_are_separate() -> None:
    profile, _warnings = build_axis_calibration(
        "x",
        _complete_observations(),
        extrapolation_percent=20.0,
        mechanical_limit_deg=30.0,
    )

    assert profile.positive.points[-1] == CurvePoint(12.0, 14)
    assert profile.negative.points[-1] == CurvePoint(11.8, 14)
    assert profile.reversal_loss.positive_to_negative_deg == pytest.approx(0.5)
    assert profile.reversal_loss.negative_to_positive_deg == pytest.approx(0.5)


def test_unsafe_transition_is_disabled_without_invalidating_curve() -> None:
    observations = list(_complete_observations())
    observations = [
        replace(item, measured_movement_deg=math.copysign(
            abs(item.measured_movement_deg) * 0.7,
            item.measured_movement_deg,
        ))
        if not item.reversal and abs(item.commanded_angle_deg) == 14
        else item
        for item in observations
    ]
    profile, warnings = build_axis_calibration(
        "x", tuple(observations), extrapolation_percent=5.0, mechanical_limit_deg=30.0
    )

    assert not profile.positive.transition_enabled
    assert profile.positive.interpolate(profile.positive.trusted_boundary_deg) == 14
    assert any("transition disabled" in warning for warning in warnings)


def test_persistence_round_trip_preserves_both_cameras_axes_and_raw_evidence(
    tmp_path: Path,
) -> None:
    x, _ = build_axis_calibration(
        "x", _complete_observations("x"), extrapolation_percent=20, mechanical_limit_deg=30
    )
    y, _ = build_axis_calibration(
        "y", _complete_observations("y"), extrapolation_percent=20, mechanical_limit_deg=30
    )
    overview = _camera_profile(CameraRole.OVERVIEW, x=x, y=y)
    stereo_x = replace(
        x,
        observations=tuple(
            replace(item, camera=CameraRole.STEREO_LEFT)
            for item in x.observations
        ),
    )
    stereo = _camera_profile(
        CameraRole.STEREO_LEFT,
        identity="sha256:stereo",
        x=stereo_x,
    )
    document = MotionCalibrationDocument(
        extrapolation_percent=17.5,
        profiles={CameraRole.OVERVIEW: overview, CameraRole.STEREO_LEFT: stereo},
    )

    restored = motion_calibration_from_mapping(
        motion_calibration_to_mapping(document)
    )

    assert restored == document
    assert len(restored.profiles[CameraRole.OVERVIEW].axes["x"].observations) == 16

    runtime = _runtime(tmp_path, profiles={CameraRole.STEREO_LEFT: stereo})
    runtime.save_and_apply(overview, extrapolation_percent=17.5)
    loaded = MotionCalibrationRuntime.load(
        path=runtime.path,
        axes=_axes(),
        camera_identities=runtime.camera_identities,
    )
    assert set(loaded.document.profiles) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    }


def test_failed_atomic_save_does_not_live_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    new = replace(old, created_at_utc="new")
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: old})

    def fail_replace(_source, _destination):
        raise OSError("disk full")

    monkeypatch.setattr("navmin.motion_calibration.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        runtime.save_and_apply(new, extrapolation_percent=20)

    assert runtime.document.profiles[CameraRole.OVERVIEW] is old


def test_sequence_is_bounded_and_preserves_clean_and_reversal_samples() -> None:
    sequence = build_calibration_sequence((4, 8, 14), (3, 6, 10))

    for axis, a1 in (("x", 4.0), ("y", 3.0)):
        axis_steps = [step for step in sequence if step.axis == axis]
        clean = [step for step in axis_steps if not step.reversal]
        reversal = [step for step in axis_steps if step.reversal]

        assert len(clean) == 12
        assert len(reversal) == 4
        assert [abs(step.command_deg) for step in clean].count(a1) == 4
        assert sum(step.command_deg < 0 for step in reversal) == 2
        assert sum(step.command_deg > 0 for step in reversal) == 2

        for step in axis_steps:
            # Every measurement starts with only the smallest conditioning move
            # and returns to the approximate origin afterwards.
            assert abs(step.precondition_deg) == pytest.approx(a1)
            assert step.precondition_deg + step.command_deg + step.return_deg == pytest.approx(0.0)
            if step.reversal:
                assert step.precondition_deg * step.command_deg < 0.0
            else:
                assert step.precondition_deg * step.command_deg > 0.0


def test_plan_preflight_rejects_automatic_return_over_mechanical_limit() -> None:
    model = _LinearCameraModel()
    frame = _frame(frame_id=1, timestamp_ns=1_000_000_000)

    assessment = assess_calibration_plan(
        camera_model=model,
        frame=frame,
        x_angles=(4.0, 8.0, 14.0),
        y_angles=(3.0, 6.0, 10.0),
        mechanics=_axes(limit=15.0),
    )

    assert not assessment.valid
    assert any(
        "ось X" in issue and "возврат -18°" in issue and "15°" in issue
        for issue in assessment.issues
    )


@pytest.mark.parametrize(
    "angles",
    [(0, 2, 3), (1, 1, 3), (1, 3, 2), (1, 2, 31)],
)
def test_angle_validation_rejects_invalid_triplets(
    angles: tuple[float, float, float]
) -> None:
    with pytest.raises(MotionCalibrationError):
        validate_angles(angles, mechanical_limit_deg=30)


class _LinearCameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        horizontal = math.radians((x_px - 100.0) * 0.2)
        vertical = math.radians((50.0 - y_px) * 0.2)
        return CameraRay(
            math.sin(horizontal) * math.cos(vertical),
            -math.sin(vertical),
            math.cos(horizontal) * math.cos(vertical),
        )

    def ray_to_pixel(self, ray: CameraRay) -> tuple[float, float]:
        horizontal = math.degrees(math.atan2(ray.x, ray.z))
        vertical = -math.degrees(math.atan2(ray.y, math.hypot(ray.x, ray.z)))
        return 100.0 + horizontal / 0.2, 50.0 - vertical / 0.2


def _frame(*, frame_id: int, timestamp_ns: int, generation: int = 1) -> FramePacket:
    return FramePacket(
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=timestamp_ns,
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )



def test_real_stereo_left_plan_rejects_2_4_8_before_any_motion() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calibration = load_stereo_calibration(repo_root / "config/calibration/stereo.json")
    left_model, _right_model = stereo_camera_models(calibration)
    frame = FramePacket(
        CameraRole.STEREO_LEFT,
        1,
        1,
        None,
        1_000_000_000,
        np.zeros((calibration.image_height, calibration.image_width, 3), dtype=np.uint8),
    )

    assessment = assess_calibration_plan(
        camera_model=left_model,
        frame=frame,
        x_angles=(2.0, 4.0, 8.0),
        y_angles=(2.0, 4.0, 8.0),
        mechanics=_axes(limit=45.0),
    )

    assert not assessment.valid
    assert assessment.guide_for("x", 2.0) is not None
    assert assessment.guide_for("x", 4.0) is not None
    assert assessment.guide_for("x", 8.0) is None
    assert assessment.guide_for("x", -8.0) is None
    assert assessment.guide_for("y", 2.0) is not None
    assert assessment.guide_for("y", 4.0) is not None
    assert assessment.guide_for("y", 8.0) is None
    assert assessment.guide_for("y", -8.0) is None
    assert 4.0 < assessment.maximum_measurable_deg["x"] < 8.0
    assert 4.0 < assessment.maximum_measurable_deg["y"] < 8.0
    assert any("ось X, команда +8°" in issue for issue in assessment.issues)
    assert any("ось Y, команда -8°" in issue for issue in assessment.issues)


def test_click_guide_reports_narrower_band_for_larger_move() -> None:
    model = _LinearCameraModel()
    frame = _frame(frame_id=1, timestamp_ns=1_000_000_000)

    small = calibration_click_guide(
        camera_model=model, frame=frame, axis="x", command_deg=4.0
    )
    large = calibration_click_guide(
        camera_model=model, frame=frame, axis="x", command_deg=14.0
    )

    assert small is not None
    assert large is not None
    assert large.fraction < small.fraction

def test_wizard_first_click_move_return_and_next_precondition() -> None:
    now = [1_000_000_000]
    moves: list[MoveRelativeCommand] = []

    def submit(_camera, _generation, command):
        moves.append(command)
        return True

    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 100),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=500,
        submit_move=submit,
        clock_ns=lambda: now[0],
    )
    model = _LinearCameraModel()

    wizard.start(generation=1)
    assert wizard.phase == WizardPhase.CONDITIONING
    assert moves[-1] == MoveRelativeCommand(4, 0)
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.WAIT_FIRST_CLICK

    before = _frame(frame_id=1, timestamp_ns=now[0])
    assert wizard.handle_click(frame=before, camera_model=model, x_px=100, y_px=50)
    assert moves[-1] == MoveRelativeCommand(4, 0)
    assert wizard.phase == WizardPhase.MOVING
    assert not wizard.handle_click(
        frame=_frame(frame_id=2, timestamp_ns=now[0]),
        camera_model=model,
        x_px=80,
        y_px=50,
    )

    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.WAIT_SECOND_CLICK
    after = _frame(frame_id=2, timestamp_ns=now[0])
    assert wizard.handle_click(frame=after, camera_model=model, x_px=80, y_px=50)
    assert wizard.observations[-1].measured_movement_deg == pytest.approx(4)
    assert wizard.phase == WizardPhase.RETURNING
    assert moves[-1] == MoveRelativeCommand(-8, 0)
    assert wizard.step_index == 0

    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.step_index == 1
    assert wizard.phase == WizardPhase.CONDITIONING
    assert moves[-1] == MoveRelativeCommand(4, 0)

    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.WAIT_FIRST_CLICK


def test_return_completion_dispatches_next_axis_precondition_automatically() -> None:
    moves: list[MoveRelativeCommand] = []
    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 100),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=500,
        submit_move=lambda _camera, _generation, command: (moves.append(command) or True),
        clock_ns=lambda: 0,
    )
    wizard.start(generation=1)
    first_y_index = next(
        index for index, step in enumerate(wizard.steps) if step.axis == "y"
    )
    wizard.step_index = first_y_index - 1
    wizard.phase = WizardPhase.RETURNING
    wizard.settle_deadline_ns = 0

    wizard.tick()

    assert wizard.step_index == first_y_index
    assert wizard.phase == WizardPhase.CONDITIONING
    assert moves[-1] == MoveRelativeCommand(0.0, 3.0)


def test_repeat_problematic_step_keeps_evidence_and_third_sample_resolves_pair() -> None:
    now = [1_000_000_000]
    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 0),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=500,
        submit_move=lambda *_args: True,
        clock_ns=lambda: now[0],
    )
    wizard.start(generation=1)
    target_index = [
        index
        for index, step in enumerate(wizard.steps)
        if step.axis == "x"
        and not step.reversal
        and step.command_deg == 4
    ][1]
    wizard.step_index = target_index
    wizard.phase = WizardPhase.WAIT_FIRST_CLICK
    wizard.observations.append(_observation(4, 3.5))
    model = _LinearCameraModel()

    before = _frame(frame_id=10, timestamp_ns=now[0])
    wizard.handle_click(frame=before, camera_model=model, x_px=100, y_px=50)
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    wizard.handle_click(
        frame=_frame(frame_id=11, timestamp_ns=now[0]),
        camera_model=model,
        x_px=70,
        y_px=50,
    )
    assert wizard.phase == WizardPhase.RETURNING
    assert len(wizard.observations) == 2
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.REVIEW_REPEAT

    wizard.repeat_problematic_step()
    assert wizard.phase == WizardPhase.CONDITIONING
    assert len(wizard.observations) == 2
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.WAIT_FIRST_CLICK
    wizard.handle_click(
        frame=_frame(frame_id=12, timestamp_ns=now[0]),
        camera_model=model,
        x_px=100,
        y_px=50,
    )
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    wizard.handle_click(
        frame=_frame(frame_id=13, timestamp_ns=now[0]),
        camera_model=model,
        x_px=82,
        y_px=50,
    )

    assert wizard.phase == WizardPhase.RETURNING
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.CONDITIONING
    assert wizard.step_index == target_index + 1
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.phase == WizardPhase.WAIT_FIRST_CLICK
    assert len(wizard.observations) == 3
    assert any("resolved by additional sample" in warning for warning in wizard.warnings)

def test_wizard_full_synthetic_sequence_returns_to_origin_and_builds_profile() -> None:
    now = [1_000_000_000]
    moves: list[MoveRelativeCommand] = []
    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 0),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=500,
        submit_move=lambda _camera, _generation, command: (moves.append(command) or True),
        clock_ns=lambda: now[0],
    )
    model = _LinearCameraModel()
    frame_id = 0

    wizard.start(generation=1)
    while wizard.phase != WizardPhase.COMPLETE:
        if wizard.phase in (
            WizardPhase.CONDITIONING,
            WizardPhase.MOVING,
            WizardPhase.RETURNING,
        ):
            assert wizard.settle_deadline_ns is not None
            now[0] = wizard.settle_deadline_ns
            wizard.tick()
            continue
        step = wizard.current_step
        assert step is not None
        if wizard.phase == WizardPhase.WAIT_FIRST_CLICK:
            frame_id += 1
            if step.axis == "x":
                x_px, y_px = 100.0, 50.0
            else:
                x_px = 100.0
                y_px = 30.0 if step.command_deg > 0.0 else 70.0
            assert wizard.handle_click(
                frame=_frame(frame_id=frame_id, timestamp_ns=now[0]),
                camera_model=model,
                x_px=x_px,
                y_px=y_px,
            )
            continue
        if wizard.phase == WizardPhase.WAIT_SECOND_CLICK:
            first = wizard.first_click
            assert first is not None
            frame_id += 1
            if step.axis == "x":
                x_px = first.x_px - step.command_deg / 0.2
                y_px = first.y_px
            else:
                x_px = first.x_px
                y_px = first.y_px + step.command_deg / 0.2
            assert wizard.handle_click(
                frame=_frame(frame_id=frame_id, timestamp_ns=now[0]),
                camera_model=model,
                x_px=x_px,
                y_px=y_px,
            )
            continue
        pytest.fail(f"unexpected wizard phase: {wizard.phase}")

    assert len(wizard.observations) == len(wizard.steps) == 32
    assert wizard.first_click is None
    assert sum(command.delta_x_deg for command in moves) == pytest.approx(0.0)
    assert sum(command.delta_y_deg for command in moves) == pytest.approx(0.0)
    assert all(
        observation.measured_movement_deg
        == pytest.approx(observation.commanded_angle_deg)
        for observation in wizard.observations
    )
    profile = wizard.build_profile()
    assert profile.camera is CameraRole.OVERVIEW
    assert set(profile.axes) == {"x", "y"}


def test_return_rejection_cancels_wizard_after_recording_measurement() -> None:
    now = [1_000_000_000]
    calls: list[MoveRelativeCommand] = []

    def submit(_camera, _generation, command):
        calls.append(command)
        return len(calls) < 3

    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 0),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=500,
        submit_move=submit,
        clock_ns=lambda: now[0],
    )
    model = _LinearCameraModel()

    wizard.start(generation=1)
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    assert wizard.handle_click(
        frame=_frame(frame_id=1, timestamp_ns=now[0]),
        camera_model=model,
        x_px=100,
        y_px=50,
    )
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    with pytest.raises(MotionCalibrationError, match="return MOVE_RELATIVE"):
        wizard.handle_click(
            frame=_frame(frame_id=2, timestamp_ns=now[0]),
            camera_model=model,
            x_px=80,
            y_px=50,
        )

    assert calls == [
        MoveRelativeCommand(4, 0),
        MoveRelativeCommand(4, 0),
        MoveRelativeCommand(-8, 0),
    ]
    assert wizard.phase == WizardPhase.CANCELLED
    assert wizard.cancel_reason == "return MOVE_RELATIVE was rejected"
    assert len(wizard.observations) == 1


def test_wizard_rejects_stale_out_of_bounds_edge_and_generation_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [2_000_000_000]
    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 0),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=100,
        submit_move=lambda *_args: True,
        clock_ns=lambda: now[0],
    )
    model = _LinearCameraModel()
    wizard.start(generation=1)
    now[0] = wizard.settle_deadline_ns
    wizard.tick()

    with pytest.raises(MotionCalibrationError, match="stale"):
        wizard.handle_click(
            frame=_frame(frame_id=1, timestamp_ns=now[0] - 200_000_000),
            camera_model=model,
            x_px=100,
            y_px=50,
        )
    with pytest.raises(MotionCalibrationError, match="outside"):
        wizard.handle_click(
            frame=_frame(frame_id=1, timestamp_ns=now[0]),
            camera_model=model,
            x_px=300,
            y_px=50,
        )
    with (
        caplog.at_level("INFO", logger="navmin.motion_calibration"),
        pytest.raises(MotionCalibrationError, match="edge"),
    ):
        wizard.handle_click(
            frame=_frame(frame_id=1, timestamp_ns=now[0]),
            camera_model=model,
            x_px=10,
            y_px=50,
        )
    assert "first click rejected visibility" in caplog.text
    assert "command_deg=+4" in caplog.text

    wizard.handle_click(
        frame=_frame(frame_id=1, timestamp_ns=now[0]),
        camera_model=model,
        x_px=100,
        y_px=50,
    )
    now[0] = wizard.settle_deadline_ns
    wizard.tick()
    with pytest.raises(MotionCalibrationError, match="generation"):
        wizard.handle_click(
            frame=_frame(frame_id=2, timestamp_ns=now[0], generation=2),
            camera_model=model,
            x_px=80,
            y_px=50,
        )


def test_cancel_preserves_existing_document_and_stops_progression(tmp_path: Path) -> None:
    old = _camera_profile(CameraRole.OVERVIEW, x=_axis_profile())
    runtime = _runtime(tmp_path, profiles={CameraRole.OVERVIEW: old})
    wizard = MotionCalibrationWizard(
        settings=WizardSettings(
            CameraRole.OVERVIEW,
            (4, 8, 14),
            (3, 6, 10),
            20,
            MotionProfile(5, 10, 100),
        ),
        mechanics=_axes(),
        camera_calibration_identity="sha256:camera",
        stale_timeout_ms=500,
        submit_move=lambda *_args: True,
        clock_ns=lambda: 0,
    )
    wizard.start(generation=1)
    wizard.cancel("Emergency")
    wizard.tick()

    assert wizard.phase == WizardPhase.CANCELLED
    assert runtime.document.profiles[CameraRole.OVERVIEW] is old
