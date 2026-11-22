from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from navmin.calibration import overview_camera_model, stereo_camera_models
from navmin.contracts import CameraRole
from navmin.diagnostics.synthetic_tracking_calibration import (
    SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG,
    SYNTHETIC_TRACKING_STEREO_BASELINE_M,
    SYNTHETIC_TRACKING_STEREO_HFOV_DEG,
    synthetic_tracking_overview_calibration,
    synthetic_tracking_stereo_calibration,
)
from navmin.diagnostics.synthetic_tracking_world import (
    SYNTHETIC_TRACKING_INITIAL_STATIC_S,
    SYNTHETIC_TRACKING_INITIAL_YAW_ACCEL_DEG_S2,
    SYNTHETIC_TRACKING_TARGET_SIZE_PX,
    SYNTHETIC_TRACKING_WORLD_LIMIT_DEG,
    SyntheticTrackingWorld,
)
from navmin.vision.pipeline import (
    overview_corrector,
    stereo_left_corrector,
    stereo_right_corrector,
)


def _world() -> SyntheticTrackingWorld:
    return SyntheticTrackingWorld()


def test_tracking_world_geometry_matches_diagnostic_contract() -> None:
    world = _world()
    overview = world.geometries[CameraRole.OVERVIEW]
    left = world.geometries[CameraRole.STEREO_LEFT]
    right = world.geometries[CameraRole.STEREO_RIGHT]

    assert overview.horizontal_fov_deg == pytest.approx(
        SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG, abs=0.1
    )
    assert overview.vertical_fov_deg == pytest.approx(73.9, abs=0.2)
    assert left.horizontal_fov_deg == pytest.approx(
        SYNTHETIC_TRACKING_STEREO_HFOV_DEG, abs=0.01
    )
    assert left.vertical_fov_deg == pytest.approx(4.74, abs=0.02)
    assert left.stereo_offset_px == pytest.approx(-right.stereo_offset_px)
    assert 30.0 < world.disparity_px < 45.0


def test_canonical_simulation_frames_survive_production_correction_geometry() -> None:
    world = _world()
    rendered = world.render(220, turret_yaw_deg=0.25, turret_pitch_deg=-0.1)
    correctors = {
        CameraRole.OVERVIEW: overview_corrector(
            synthetic_tracking_overview_calibration()
        ),
        CameraRole.STEREO_LEFT: stereo_left_corrector(
            synthetic_tracking_stereo_calibration()
        ),
        CameraRole.STEREO_RIGHT: stereo_right_corrector(
            synthetic_tracking_stereo_calibration()
        ),
    }

    for role, corrector in correctors.items():
        corrected = corrector.correct(rendered.image(role))
        ys, xs = np.where(corrected[:, :, 0] > 200)
        projection = rendered.projection(role)
        assert xs.size > 0
        assert float(xs.mean()) == pytest.approx(projection.center_x_px, abs=1.0)
        assert float(ys.mean()) == pytest.approx(projection.center_y_px, abs=1.0)


def test_tracking_world_target_motion_is_continuous_bounded_and_repeats() -> None:
    world = _world()
    frames = round(world.cycle_duration_s * world.fps)
    truth = [world.truth(i) for i in range(frames + 1)]

    assert {item.phase for item in truth} >= {
        "STATIC",
        "YAW_SLOW_POS",
        "YAW_FAST_NEG",
        "PITCH_MEDIUM_POS",
        "PITCH_FAST_NEG",
        "DIAGONAL_RETURN",
        "COMBINED_SINE",
    }
    assert max(abs(item.target_yaw_deg) for item in truth) <= SYNTHETIC_TRACKING_WORLD_LIMIT_DEG + 1e-9
    assert max(abs(item.target_pitch_deg) for item in truth) <= SYNTHETIC_TRACKING_WORLD_LIMIT_DEG + 1e-9
    assert max(abs(item.target_yaw_rate_deg_s) for item in truth) <= 6.0 + 1e-9
    assert max(abs(item.target_pitch_rate_deg_s) for item in truth) < 5.9

    for before, after in pairwise(truth):
        assert abs(after.target_yaw_deg - before.target_yaw_deg) < 0.35
        assert abs(after.target_pitch_deg - before.target_pitch_deg) < 0.35

    assert truth[-1].target_yaw_deg == pytest.approx(truth[0].target_yaw_deg, abs=1e-9)
    assert truth[-1].target_pitch_deg == pytest.approx(truth[0].target_pitch_deg, abs=1e-9)


def test_tracking_world_starts_with_operator_selection_hold_and_gentle_acceleration() -> None:
    world = _world()
    before_motion = world.truth(round((SYNTHETIC_TRACKING_INITIAL_STATIC_S - 0.05) * world.fps))
    acceleration_midpoint = world.truth(round((SYNTHETIC_TRACKING_INITIAL_STATIC_S + 2.0) * world.fps))

    assert before_motion.phase == "STATIC"
    assert before_motion.target_yaw_deg == pytest.approx(0.0)
    assert before_motion.target_yaw_rate_deg_s == pytest.approx(0.0)
    assert acceleration_midpoint.phase == "YAW_ACCEL_POS"
    assert acceleration_midpoint.target_yaw_accel_deg_s2 == pytest.approx(
        SYNTHETIC_TRACKING_INITIAL_YAW_ACCEL_DEG_S2
    )
    assert acceleration_midpoint.target_yaw_rate_deg_s == pytest.approx(1.5)
    assert acceleration_midpoint.target_yaw_deg == pytest.approx(1.5)


def test_tracking_world_renders_shared_tick_and_constant_stereo_disparity() -> None:
    world = _world()
    rendered = world.render(0)

    assert rendered.capture_id == 0
    assert rendered.overview.shape == (976, 1296, 3)
    assert rendered.stereo_left.shape == (976, 1296, 3)
    assert rendered.stereo_right.shape == (976, 1296, 3)
    assert rendered.overview.flags.writeable is False
    assert rendered.stereo_left.flags.writeable is False
    assert rendered.stereo_right.flags.writeable is False

    left = rendered.projection(CameraRole.STEREO_LEFT)
    right = rendered.projection(CameraRole.STEREO_RIGHT)
    overview = rendered.projection(CameraRole.OVERVIEW)
    assert left.visible and right.visible and overview.visible
    assert left.center_x_px - right.center_x_px == pytest.approx(world.disparity_px)
    assert left.bbox is not None and left.bbox.width == SYNTHETIC_TRACKING_TARGET_SIZE_PX
    assert right.bbox is not None and right.bbox.width == SYNTHETIC_TRACKING_TARGET_SIZE_PX
    assert overview.bbox is not None and overview.bbox.width == SYNTHETIC_TRACKING_TARGET_SIZE_PX


def test_tracking_world_overview_keeps_full_plus_minus_30_degree_workspace_visible() -> None:
    world = _world()
    overview = world.geometries[CameraRole.OVERVIEW]
    half_target_deg_x = (SYNTHETIC_TRACKING_TARGET_SIZE_PX / 2.0) / overview.pixels_per_degree_x
    half_target_deg_y = (SYNTHETIC_TRACKING_TARGET_SIZE_PX / 2.0) / overview.pixels_per_degree_y

    assert SYNTHETIC_TRACKING_WORLD_LIMIT_DEG + half_target_deg_x < overview.horizontal_fov_deg / 2.0
    assert SYNTHETIC_TRACKING_WORLD_LIMIT_DEG + half_target_deg_y < overview.vertical_fov_deg / 2.0


def test_tracking_world_explicit_turret_pose_moves_background_and_relative_target() -> None:
    world = _world()
    stationary = world.render(0, turret_yaw_deg=0.0)
    turned = world.render(0, turret_yaw_deg=1.0)

    p0 = stationary.projection(CameraRole.STEREO_LEFT)
    p1 = turned.projection(CameraRole.STEREO_LEFT)
    geometry = world.geometries[CameraRole.STEREO_LEFT]
    expected_shift = geometry.focal_x_px * np.tan(np.radians(-1.0))
    assert p1.center_x_px - p0.center_x_px == pytest.approx(expected_shift)
    assert not np.array_equal(stationary.stereo_left, turned.stereo_left)


def test_tracking_world_background_is_low_contrast_away_from_target() -> None:
    world = _world()
    rendered = world.render(0)
    overview = rendered.overview
    bbox = rendered.projection(CameraRole.OVERVIEW).bbox
    assert bbox is not None
    mask = np.ones(overview.shape[:2], dtype=bool)
    mask[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    background = overview[:, :, 0][mask]

    assert int(background.max()) - int(background.min()) <= 20
    assert int(background.max()) < 100
    assert int(overview[bbox.y + bbox.height // 2, bbox.x + bbox.width // 2, 0]) == 238


def test_tracking_world_day_detector_sees_square_and_ignores_background() -> None:
    from navmin.vision.processors.vt11.acquisition.candidate_detector import (
        config_from_dict,
        detect_local_candidates,
    )
    from navmin.vision.processors.vt11.acquisition.roi import make_operator_selection
    from navmin.vision.processors.vt11.config.loader import load_profile_config

    world = _world()
    detector = config_from_dict(
        load_profile_config("day")["acquire"]["candidate_detector"]
    )

    with_target = world.render(0)
    projection = with_target.projection(CameraRole.STEREO_LEFT)
    selection = make_operator_selection(
        0,
        (round(projection.center_x_px), round(projection.center_y_px)),
        120,
        1296,
        976,
    )
    detected = detect_local_candidates(
        with_target.stereo_left,
        selection,
        top_k=5,
        config=detector,
    )
    assert len(detected.candidates) == 1
    candidate = detected.candidates[0]
    assert candidate.center_x == pytest.approx(projection.center_x_px, abs=2.0)
    assert candidate.center_y == pytest.approx(projection.center_y_px, abs=2.0)

    background_only = world.render(0, turret_yaw_deg=20.0)
    background_selection = make_operator_selection(0, (648, 488), 200, 1296, 976)
    background_detected = detect_local_candidates(
        background_only.stereo_left,
        background_selection,
        top_k=5,
        config=detector,
    )
    assert background_detected.candidates == ()


def _turret_config(*, invert_x: bool = False, watchdog_ms: int = 2000):
    from navmin.config.models import (
        AxesConfig,
        AxisMechanicsConfig,
        PidControllerConfig,
        SerialConfig,
        Stm32Config,
        TurretConfig,
    )

    x_axis = AxisMechanicsConfig(invert_x, 4000, 1, 45.0)
    y_axis = AxisMechanicsConfig(False, 4000, 1, 45.0)
    return TurretConfig(
        serial=SerialConfig("/dev/null", 9600, 10, 0, 0),
        axes=AxesConfig(x_axis, y_axis),
        controller=PidControllerConfig(2.0, 0.0, 0.0, 2.0, 0.0, 0.0),
        stm32=Stm32Config(40.0, 40.0, 20.0, 20.0, watchdog_ms),
        emulate_stm32=True,
    )


def _synthetic_turret_stack(*, invert_x: bool = False, watchdog_ms: int = 2000):
    from navmin.diagnostics.synthetic_tracking_turret import (
        SyntheticTrackingStm32Endpoint,
        SyntheticTurretPlant,
    )
    from navmin.turret.hal import TurretHal
    from navmin.turret.session import TurretSession
    from navmin.turret.simulator import FakeTransport

    config = _turret_config(invert_x=invert_x, watchdog_ms=watchdog_ms)
    plant = SyntheticTurretPlant(config)
    endpoint = SyntheticTrackingStm32Endpoint(plant)
    transport = FakeTransport(endpoint)
    transport.open()
    session = TurretSession(
        transport,
        response_timeout_s=0.01,
        max_retries=0,
        inter_request_delay_s=0.0,
    )
    hal = TurretHal(session, config)
    result = hal.sync_stm32_config()
    assert result is not None and result.response is not None
    result = hal.motor_on()
    assert result.response is not None
    return config, plant, endpoint, transport, session, hal


def test_synthetic_turret_plant_follows_executed_velocity_with_acceleration_limit() -> None:
    from navmin.contracts import AxisVelocitySetpoint

    _, plant, _, _, _, hal = _synthetic_turret_stack()
    hal.set_velocity(AxisVelocitySetpoint(30.0, -10.0, 0))

    commanded = plant.state
    assert commanded.configured is True
    assert commanded.motors_on is True
    assert commanded.requested_yaw_rate_deg_s == pytest.approx(29.97, abs=0.02)
    assert commanded.requested_pitch_rate_deg_s == pytest.approx(-9.99, abs=0.02)
    assert commanded.yaw_rate_deg_s == 0.0

    after = plant.advance(0.5)
    assert after.yaw_rate_deg_s == pytest.approx(9.99, abs=0.02)
    assert after.pitch_rate_deg_s == pytest.approx(-9.99, abs=0.02)
    assert after.yaw_deg == pytest.approx(2.4975, abs=0.01)
    assert after.pitch_deg == pytest.approx(-2.4975, abs=0.01)


def test_synthetic_turret_plant_uses_effective_payload_sign_and_clamps_speed() -> None:
    from navmin.contracts import AxisVelocitySetpoint
    from navmin.turret.protocol import CommandCode, SetVelocityPayload

    _, plant, endpoint, _, _, hal = _synthetic_turret_stack(invert_x=True)
    hal.set_velocity(AxisVelocitySetpoint(100.0, 0.0, 0))

    request = endpoint.executed_request_history[-1]
    assert request.command is CommandCode.SET_VELOCITY
    assert isinstance(request.payload, SetVelocityPayload)
    assert request.payload.velocity_x_steps_s < 0
    state = plant.state
    assert state.requested_yaw_rate_deg_s == pytest.approx(40.0, abs=0.1)


def test_synthetic_turret_zero_setpoint_decelerates_but_motor_off_hard_stops() -> None:
    from navmin.contracts import AxisVelocitySetpoint

    _, plant, _, _, _, hal = _synthetic_turret_stack()
    hal.set_velocity(AxisVelocitySetpoint(30.0, 0.0, 0))
    plant.advance(1.0)
    moving = plant.state
    assert moving.yaw_rate_deg_s == pytest.approx(19.98, abs=0.03)

    hal.set_velocity(AxisVelocitySetpoint(0.0, 0.0, 0))
    slowing = plant.advance(0.25)
    assert slowing.yaw_rate_deg_s == pytest.approx(14.985, abs=0.04)
    assert slowing.yaw_rate_deg_s > 0.0

    hal.motor_off()
    stopped = plant.state
    assert stopped.motors_on is False
    assert stopped.yaw_rate_deg_s == 0.0
    assert stopped.requested_yaw_rate_deg_s == 0.0
    yaw = stopped.yaw_deg
    assert plant.advance(1.0).yaw_deg == pytest.approx(yaw)


def test_synthetic_turret_emergency_hard_stops_without_turning_motors_off() -> None:
    from navmin.contracts import AxisVelocitySetpoint

    _, plant, _, _, _, hal = _synthetic_turret_stack()
    hal.set_velocity(AxisVelocitySetpoint(20.0, 5.0, 0))
    plant.advance(0.5)
    assert plant.state.yaw_rate_deg_s > 0.0

    response = hal.emergency_stop()
    assert response is not None
    state = plant.state
    assert state.motors_on is True
    assert state.yaw_rate_deg_s == 0.0
    assert state.pitch_rate_deg_s == 0.0
    assert state.requested_yaw_rate_deg_s == 0.0
    assert state.requested_pitch_rate_deg_s == 0.0


def test_synthetic_turret_ignores_failed_protocol_command() -> None:
    from navmin.contracts import AxisVelocitySetpoint
    from navmin.turret.protocol import ResultCode
    from navmin.turret.simulator import FakeResponseSpec

    _, plant, endpoint, _, _, hal = _synthetic_turret_stack()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INTERNAL_ERROR))
    result = hal.set_velocity(AxisVelocitySetpoint(20.0, 0.0, 0))

    assert result.response is not None
    assert result.response.result is ResultCode.INTERNAL_ERROR
    assert plant.state.requested_yaw_rate_deg_s == 0.0


def test_synthetic_tracking_simulation_moves_world_opposite_turret_rotation() -> None:
    from navmin.contracts import AxisVelocitySetpoint
    from navmin.diagnostics.synthetic_tracking_turret import SyntheticTrackingSimulation

    _, plant, _, _, _, hal = _synthetic_turret_stack()
    world = _world()
    simulation = SyntheticTrackingSimulation(world, plant)
    before = simulation.capture()
    before_projection = before.frame_set.projection(CameraRole.STEREO_LEFT)

    hal.set_velocity(AxisVelocitySetpoint(6.0, 0.0, 0))
    for _ in range(10):
        simulation.advance()
    after = simulation.capture()
    after_projection = after.frame_set.projection(CameraRole.STEREO_LEFT)

    assert after.turret.yaw_deg > 0.0
    assert after.frame_set.truth.turret_yaw_deg == pytest.approx(after.turret.yaw_deg)
    assert after_projection.center_x_px < before_projection.center_x_px
    assert not np.array_equal(before.frame_set.stereo_left, after.frame_set.stereo_left)


def test_synthetic_turret_velocity_watchdog_returns_target_to_zero() -> None:
    from navmin.contracts import AxisVelocitySetpoint

    _, plant, _, _, _, hal = _synthetic_turret_stack(watchdog_ms=200)
    hal.set_velocity(AxisVelocitySetpoint(30.0, 0.0, 0))

    plant.advance(0.15)
    assert plant.state.requested_yaw_rate_deg_s > 0.0
    assert plant.state.velocity_command_age_s == pytest.approx(0.15)

    plant.advance(0.10)
    state = plant.state
    assert state.velocity_command_age_s == pytest.approx(0.25)
    assert state.velocity_watchdog_timeout_s == pytest.approx(0.2)
    assert state.requested_yaw_rate_deg_s == 0.0
    assert state.yaw_rate_deg_s > 0.0


def test_synthetic_world_projection_matches_production_camera_models() -> None:
    world = _world()
    overview_model = overview_camera_model(synthetic_tracking_overview_calibration())
    left_model, right_model = stereo_camera_models(
        synthetic_tracking_stereo_calibration()
    )
    rendered = world.render(220, turret_yaw_deg=0.25, turret_pitch_deg=-0.1)
    truth = rendered.truth

    overview_projection = rendered.projection(CameraRole.OVERVIEW)
    overview_ray = overview_model.pixel_to_ray(
        overview_projection.center_x_px, overview_projection.center_y_px
    )
    assert overview_ray.x / overview_ray.z == pytest.approx(
        np.tan(np.radians(truth.true_error_yaw_deg)), abs=1e-9
    )
    assert overview_ray.y / overview_ray.z == pytest.approx(
        -np.tan(np.radians(truth.true_error_pitch_deg)), abs=1e-9
    )

    parallax = SYNTHETIC_TRACKING_STEREO_BASELINE_M / (
        2.0 * world.target_distance_m
    )
    for role, model, sign in (
        (CameraRole.STEREO_LEFT, left_model, +1.0),
        (CameraRole.STEREO_RIGHT, right_model, -1.0),
    ):
        projection = rendered.projection(role)
        ray = model.pixel_to_ray(projection.center_x_px, projection.center_y_px)
        assert ray.x / ray.z == pytest.approx(
            np.tan(np.radians(truth.true_error_yaw_deg)) + sign * parallax,
            abs=1e-9,
        )
        assert ray.y / ray.z == pytest.approx(
            -np.tan(np.radians(truth.true_error_pitch_deg)), abs=1e-9
        )


def test_synthetic_closed_loop_runs_real_vt11_aiming_estimator_and_turret_chain() -> None:
    from navmin.diagnostics.synthetic_tracking_closed_loop import (
        run_synthetic_tracking_closed_loop,
    )

    result = run_synthetic_tracking_closed_loop(
        config_path="config/navmin.json",
        frame_count=300,
        sample_stride=2,
    )

    assert result.status == "COMPLETE"
    assert result.target_acquired_frame is not None
    assert result.target_acquired_frame < 20
    assert result.target_end_frame is None
    assert result.missing_tracking_frames <= 2
    assert result.rate_valid_x_frames >= 45
    assert any(
        sample.commanded_velocity_x_deg_s is not None
        and abs(sample.commanded_velocity_x_deg_s) > 0.1
        for sample in result.samples
    )
    acceleration = next(
        phase for phase in result.phase_summaries if phase.phase == "YAW_ACCEL_POS"
    )
    assert acceleration.tracked_frames > 0
    assert acceleration.mean_abs_error_yaw_deg < 0.25


def test_synthetic_tracking_stereo_exact_pair_join_and_distance_path() -> None:
    from navmin.diagnostics.synthetic_tracking_stereo import (
        run_synthetic_tracking_stereo_diagnostic,
    )

    result = run_synthetic_tracking_stereo_diagnostic(
        frame_count=120,
        selection_frame=8,
        distance_stride=10,
    )

    assert result.status == "PASS"
    assert result.target_acquired_frame == 14
    assert result.target_end_frame is None
    assert result.exact_pairs == 120
    assert result.target_observations >= 100
    assert result.distance_attempts == result.distance_successes
    assert result.distance_successes >= 9
    assert result.mean_abs_distance_error_m is not None
    assert result.mean_abs_distance_error_m < 5.0
    assert result.max_abs_distance_error_m is not None
    assert result.max_abs_distance_error_m < 5.0
    assert all(sample.capture_id == sample.frame_index for sample in result.samples)
    expected_disparity = SyntheticTrackingWorld().disparity_px
    assert all(
        sample.measured_disparity_px == pytest.approx(expected_disparity, abs=2.0)
        for sample in result.samples
    )


def test_synthetic_tracking_stereo_world_keeps_common_capture_and_constant_disparity() -> None:
    world = _world()
    for frame_index in (0, 80, 240, 480, 720, 1000, 1300):
        truth = world.truth(frame_index)
        rendered = world.render(
            frame_index,
            turret_yaw_deg=truth.target_yaw_deg,
            turret_pitch_deg=truth.target_pitch_deg,
        )
        left = rendered.projection(CameraRole.STEREO_LEFT)
        right = rendered.projection(CameraRole.STEREO_RIGHT)
        assert rendered.capture_id == frame_index
        assert left.visible is True
        assert right.visible is True
        assert left.center_x_px - right.center_x_px == pytest.approx(world.disparity_px)
        assert left.center_y_px == pytest.approx(right.center_y_px)


def test_synthetic_tracking_background_uses_expected_turret_image_shift() -> None:
    world = _world()
    geometry = world.geometries[CameraRole.OVERVIEW]
    baseline = world.render(0)
    moved = world.render(
        0,
        turret_yaw_deg=1.0 / geometry.pixels_per_degree_x,
    )

    expected = np.roll(baseline.overview, shift=-1, axis=1)
    # The target is near frame center; this corner patch contains background only.
    assert np.array_equal(moved.overview[:100, :100], expected[:100, :100])


def test_realtime_tracking_rig_single_stereo_side_does_not_wait_for_peer() -> None:
    from navmin.diagnostics.synthetic_tracking_turret import (
        RealtimeSyntheticTrackingRig,
        SyntheticTrackingFrameSource,
        SyntheticTurretPlant,
    )

    now = [0.0]
    world = _world()
    rig = RealtimeSyntheticTrackingRig(
        world,
        SyntheticTurretPlant(_turret_config()),
        clock=lambda: now[0],
        synchronize_stereo_pair=False,
    )
    left = SyntheticTrackingFrameSource(rig, CameraRole.STEREO_LEFT)
    left.start()

    first = left.read()
    assert first is not None
    assert first.capture_id == 0
    now[0] = 1.0 / world.fps
    second = left.read()
    assert second is not None
    assert second.capture_id == 1


def test_realtime_tracking_rig_syncs_command_boundary_and_shared_stereo_tick() -> None:
    from navmin.contracts import AxisVelocitySetpoint
    from navmin.diagnostics.synthetic_tracking_turret import (
        RealtimeSyntheticTrackingRig,
        SyntheticTrackingFrameSource,
        SyntheticTrackingStm32Endpoint,
        SyntheticTurretPlant,
    )
    from navmin.turret.hal import TurretHal
    from navmin.turret.session import TurretSession
    from navmin.turret.simulator import FakeTransport

    now = [0.0]
    config = _turret_config(watchdog_ms=2000)
    world = _world()
    plant = SyntheticTurretPlant(config)
    rig = RealtimeSyntheticTrackingRig(world, plant, clock=lambda: now[0])
    endpoint = SyntheticTrackingStm32Endpoint(
        plant,
        request_sink=rig.apply_executed_request,
    )
    transport = FakeTransport(endpoint)
    transport.open()
    session = TurretSession(
        transport,
        response_timeout_s=0.01,
        max_retries=0,
        inter_request_delay_s=0.0,
        clock=lambda: now[0],
        wait_hook=lambda _seconds: None,
    )
    hal = TurretHal(session, config, command_clock_ns=lambda: int(now[0] * 1e9))
    assert hal.sync_stm32_config() is not None
    assert hal.motor_on() is not None

    left = SyntheticTrackingFrameSource(rig, CameraRole.STEREO_LEFT)
    right = SyntheticTrackingFrameSource(rig, CameraRole.STEREO_RIGHT)
    left.start()
    right.start()
    initial_left = left.read()
    initial_right = right.read()
    assert initial_left is not None and initial_right is not None
    assert initial_left.capture_id == initial_right.capture_id == 0
    assert initial_left.receive_timestamp_ns == initial_right.receive_timestamp_ns

    hal.set_velocity(AxisVelocitySetpoint(6.0, 0.0, 0))
    now[0] = 0.5
    moved_left = left.read()
    moved_right = right.read()
    assert moved_left is not None and moved_right is not None
    assert moved_left.capture_id == moved_right.capture_id == 10
    assert moved_left.receive_timestamp_ns == moved_right.receive_timestamp_ns
    assert plant.state.yaw_deg > 0.0
    assert not np.array_equal(initial_left.image, moved_left.image)
