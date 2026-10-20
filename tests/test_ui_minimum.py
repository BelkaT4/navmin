from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QSize, Qt
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from navmin.concurrency import (
    CameraSessionBarrierChannel,
    LatestValue,
    TargetControlChannel,
    TargetLifecycleEventChannel,
)
from navmin.config import load_config
from navmin.config.models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    PidControllerConfig,
    TrackingControlConfig,
    UiConfig,
)
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    CameraSessionStarted,
    CameraState,
    CameraStatus,
    DistanceResult,
    DistanceSource,
    FramePacket,
    MotorState,
    MoveRelativeCommand,
    PlatformMotionSnapshot,
    PlatformMotionSource,
    TargetAcquired,
    TargetRef,
    TrackedObject,
    TrackingControlSample,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.core import (
    AxisTrackingEstimate,
    DistanceMeasurementMode,
    Mediator,
    TrackingEstimateInvalidReason,
    TrackingEstimatorDiagnostics,
)
from navmin.imu import GyroBiasState, ImuRuntimeState, ImuSourceIdentity
from navmin.ui import (
    CameraUiBinding,
    MainWindow,
    PreparedVisionFrame,
    UiStatePump,
    VideoView,
    is_camera_stale,
    map_widget_to_source,
    prepare_vision_frame,
    rendered_image_rect,
)
from navmin.ui.video_view import map_source_to_widget, reticle_segments
from navmin.vision.stereo_distance_worker import StereoDistanceCandidate
from navmin.vision.stereo_target_join import StereoPairingKind, StereoTargetJoiner


class _Clock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _WallClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _CameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay((x_px - 50.0) / 50.0, (y_px - 25.0) / 25.0, 1.0)


class _FakeTurret:
    def __init__(self, state: TurretState) -> None:
        self._state = state
        self.relative_commands: list[MoveRelativeCommand] = []
        self.tracking_controls: list[TrackingControlSample] = []
        self.invalidations = 0
        self.mode_requests: list[TurretControlMode] = []
        self.stop_requests = 0
        self.motor_on_requests = 0
        self.motor_off_requests = 0
        self.emergency_requests = 0

    @property
    def current_state(self) -> TurretState:
        return self._state

    def submit_move_relative(self, command: MoveRelativeCommand) -> None:
        self.relative_commands.append(command)

    def submit_tracking_control(self, sample: TrackingControlSample) -> None:
        self.tracking_controls.append(sample)

    def invalidate_tracking_control(self) -> None:
        self.invalidations += 1

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot:
        return PlatformMotionSnapshot(PlatformMotionSource.COMMAND_MODEL, False, ())

    def set_control_mode(self, mode: TurretControlMode) -> bool:
        self.mode_requests.append(mode)
        return True

    def stop_motion(self) -> bool:
        self.stop_requests += 1
        return True

    def motor_on(self) -> bool:
        self.motor_on_requests += 1
        return True

    def motor_off(self) -> bool:
        self.motor_off_requests += 1
        return True

    def request_emergency(self) -> None:
        self.emergency_requests += 1


class _CountingMediator(Mediator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.vision_accept_calls = 0
        self.turret_accept_calls = 0
        self.accept_order: list[str] = []

    def accept_camera_session(self, session: CameraSessionStarted) -> bool:
        self.accept_order.append("session")
        return super().accept_camera_session(session)

    def accept_target_lifecycle_event(self, event) -> bool:
        self.accept_order.append("target_event")
        return super().accept_target_lifecycle_event(event)

    def accept_vision_result(self, result: VisionResult) -> bool:
        self.accept_order.append("vision_result")
        self.vision_accept_calls += 1
        return super().accept_vision_result(result)

    def accept_turret_state(self, state: TurretState) -> None:
        self.turret_accept_calls += 1
        super().accept_turret_state(state)


@pytest.fixture(scope="session")
def qt_application() -> QApplication:
    application = QApplication.instance() or QApplication([])
    return application


def _turret_state(
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    motor: MotorState = MotorState.ON,
    connection: TurretConnectionState = TurretConnectionState.READY,
) -> TurretState:
    return TurretState(connection, motor, mode, 40.0, 40.0, 80.0, 80.0)


def _mediator(
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    motor: MotorState = MotorState.ON,
    counting: bool = False,
    aim_points: AimPointsConfig | None = None,
    stereo_target_joiner: StereoTargetJoiner | None = None,
    stereo_distance_results: LatestValue[StereoDistanceCandidate] | None = None,
    imu_states: LatestValue[ImuRuntimeState] | None = None,
) -> tuple[Mediator, _FakeTurret]:
    turret = _FakeTurret(_turret_state(mode=mode, motor=motor))
    mediator_type = _CountingMediator if counting else Mediator
    mediator = mediator_type(
        aiming_config=AimingConfig(
            target_lost_timeout_ms=500,
            aim_points=aim_points
            or AimPointsConfig(
                overview=AimPointConfig(50, 25),
                stereo_left=AimPointConfig(50, 25),
            ),
        ),
        tracking_control_config=load_config(
            Path(__file__).resolve().parents[1]
            / "config"
            / "defaults"
            / "navmin.json"
        ).tracking_control,
        tracking_gap_timeout_ms=200,
        ui_config=UiConfig(CameraRole.OVERVIEW, False, False),
        turret=turret,
        target_controls={
            camera: TargetControlChannel(camera)
            for camera in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)
        },
        stereo_target_joiner=stereo_target_joiner,
        stereo_distance_results=stereo_distance_results,
    )
    return mediator, turret


def _session(camera: CameraRole, generation: int) -> CameraSessionStarted:
    return CameraSessionStarted(camera, generation, _CameraModel(), generation)


def _result(
    camera: CameraRole,
    generation: int,
    *,
    frame_id: int = 1,
    timestamp_ns: int = 1_000_000_000,
    objects: tuple[TrackedObject, ...] | None = None,
) -> VisionResult:
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    image[:, :, 1] = frame_id
    image.flags.writeable = False
    return VisionResult(
        FramePacket(camera, generation, frame_id, None, timestamp_ns, image),
        objects
        if objects is not None
        else (TrackedObject(7, BBox(40, 15, 20, 20), 0.0, 0.0, 3),),
        10,
    )


def _status(
    camera: CameraRole,
    generation: int,
    *,
    timestamp_ns: int = 1_000_000_000,
    state: CameraState = CameraState.ONLINE,
    message: str | None = None,
) -> CameraStatus:
    return CameraStatus(camera, state, generation, timestamp_ns, message=message)


def _bindings() -> dict[CameraRole, CameraUiBinding]:
    return {
        camera: CameraUiBinding(
            camera,
            CameraSessionBarrierChannel(),
            TargetLifecycleEventChannel(),
            LatestValue(),
            LatestValue(),
        )
        for camera in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)
    }


def _pid_config() -> PidControllerConfig:
    return PidControllerConfig(
        pid_kp_x=0.25,
        pid_ki_x=0.01,
        pid_kd_x=0.04,
        pid_kp_y=0.30,
        pid_ki_y=0.02,
        pid_kd_y=0.05,
    )


def _window(
    qt_application: QApplication,
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    motor: MotorState = MotorState.ON,
    clock: _Clock | None = None,
    wall_clock: _WallClock | None = None,
    show_diagnostic_clock: bool = False,
    frame_preparer=prepare_vision_frame,
    camera_stale_timeout_ms: int = 500,
    aim_points: AimPointsConfig | None = None,
    pid_controller_config: PidControllerConfig | None = None,
    apply_pid_tuning: Callable[[PidControllerConfig], None] | None = None,
    apply_tracking_control_tuning: (
        Callable[[TrackingControlConfig], None] | None
    ) = None,
    stereo_target_joiner: StereoTargetJoiner | None = None,
    stereo_distance_results: LatestValue[StereoDistanceCandidate] | None = None,
    imu_states: LatestValue[ImuRuntimeState] | None = None,
    imu_controls=None,
) -> tuple[
    MainWindow,
    Mediator,
    _FakeTurret,
    dict[CameraRole, CameraUiBinding],
    LatestValue[TurretState],
]:
    mediator, turret = _mediator(
        mode=mode,
        motor=motor,
        aim_points=aim_points,
        stereo_target_joiner=stereo_target_joiner,
        stereo_distance_results=stereo_distance_results,
    )
    bindings = _bindings()
    turret_states: LatestValue[TurretState] = LatestValue()
    window = MainWindow(
        mediator=mediator,
        camera_bindings=bindings,
        turret_states=turret_states,
        camera_stale_timeout_ms=camera_stale_timeout_ms,
        pid_controller_config=pid_controller_config or _pid_config(),
        apply_pid_tuning=apply_pid_tuning or (lambda _config: None),
        tracking_control_config=mediator.tracking_estimator.config,
        apply_tracking_control_tuning=(
            apply_tracking_control_tuning or (lambda _config: None)
        ),
        imu_states=imu_states,
        imu_controls=imu_controls,
        clock_ns=clock or _Clock(),
        wall_clock=wall_clock
        or _WallClock(datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)),
        show_diagnostic_clock=show_diagnostic_clock,
        frame_preparer=frame_preparer,
        start_timer=False,
        start_fullscreen=False,
    )
    window.resize(900, 650)
    window.show()
    qt_application.processEvents()
    return window, mediator, turret, bindings, turret_states



def _imu_state(
    *,
    running: bool = True,
    fresh: bool = True,
    gyro_bias_state: GyroBiasState = GyroBiasState.READY,
    orientation_valid: bool = True,
    euler: tuple[float, float, float] = (1.0, 2.0, 3.0),
    accel_faces_captured: tuple[int, ...] = (),
    mag_calibration_active: bool = False,
    mag_calibration_sample_count: int = 0,
    mag_calibration_octants: int = 0,
    calibration_message: str | None = None,
) -> ImuRuntimeState:
    return ImuRuntimeState(
        running=running,
        fresh=fresh,
        stale=not fresh,
        source=ImuSourceIdentity("192.0.2.10", 54321),
        sender_timestamp_s=12.5,
        receive_timestamp_s=33.25,
        accel_g=(0.0, 0.0, 1.0),
        gyro_raw_dps=(1.0, 2.0, 3.0),
        gyro_corrected_dps=(0.1, 0.2, 0.3),
        mag_raw=(100.0, 200.0, 300.0),
        mag_corrected=(10.0, 20.0, 30.0),
        gyro_bias_state=gyro_bias_state,
        gyro_bias_dps=(0.9, 1.8, 2.7),
        gyro_bias_sample_count=17,
        accel_calibrated=True,
        mag_calibrated=True,
        orientation_quaternion=(1.0, 0.0, 0.0, 0.0),
        relative_quaternion=(1.0, 0.0, 0.0, 0.0),
        euler_degrees=euler,
        orientation_valid=orientation_valid,
        heading_valid=True,
        mounting_axes=(1, 2, 3),
        relative_zero=(1.0, 0.0, 0.0, 0.0),
        invalid_packets=1,
        duplicate_packets=2,
        out_of_order_packets=3,
        foreign_packets=4,
        last_error=None,
        accel_faces_captured=accel_faces_captured,
        mag_calibration_active=mag_calibration_active,
        mag_calibration_sample_count=mag_calibration_sample_count,
        mag_calibration_octants=mag_calibration_octants,
        calibration_message=calibration_message,
    )

def _publish_camera(
    window: MainWindow,
    bindings: dict[CameraRole, CameraUiBinding],
    result: VisionResult,
    *,
    status: CameraStatus | None = None,
) -> None:
    binding = bindings[result.frame.camera]
    binding.session_barriers.publish(_session(result.frame.camera, result.frame.generation))
    binding.latest_result.publish(result)
    binding.status.publish(status or _status(result.frame.camera, result.frame.generation))
    window.state_pump.pump_once()


def _publish_acquired(
    window: MainWindow,
    mediator: Mediator,
    bindings: dict[CameraRole, CameraUiBinding],
    *,
    track_id: int = 7,
) -> TargetRef:
    request = mediator.pending_selection
    assert request is not None
    target = TargetRef(request.camera, request.generation, track_id)
    bindings[request.camera].target_events.publish(
        TargetAcquired(
            selection_id=request.selection_id,
            target=target,
            request_frame_id=request.frame_id,
            decision_frame_id=request.frame_id,
            decision_frame_timestamp_ns=1_000_000_000,
        )
    )
    window.state_pump.pump_once()
    return target


def test_rendered_rect_and_widget_mapping_cover_letterbox_boundaries() -> None:
    matching = rendered_image_rect(QSize(200, 100), QSize(100, 50))
    assert (
        matching.left(),
        matching.top(),
        matching.width(),
        matching.height(),
    ) == (0.0, 0.0, 200.0, 100.0)
    rect = rendered_image_rect(QSize(200, 200), QSize(100, 50))
    assert (rect.left(), rect.top(), rect.width(), rect.height()) == (0.0, 50.0, 200.0, 100.0)
    assert map_widget_to_source(QPointF(0, 50), rect, QSize(100, 50)) == (0.0, 0.0)
    bottom_right = map_widget_to_source(QPointF(199, 149), rect, QSize(100, 50))
    assert bottom_right == pytest.approx((99.0, 49.0))
    assert map_widget_to_source(QPointF(100, 49), rect, QSize(100, 50)) is None
    assert map_widget_to_source(QPointF(200, 100), rect, QSize(100, 50)) is None


def test_source_to_widget_and_reticle_geometry_keep_center_clear() -> None:
    display_rect = rendered_image_rect(QSize(200, 200), QSize(100, 50))
    center = map_source_to_widget((49.5, 24.5), display_rect, QSize(100, 50))
    assert center == QPointF(100.0, 100.0)

    segments = reticle_segments(center)
    assert len(segments) == 4
    assert [(line.x1(), line.y1(), line.x2(), line.y2()) for line in segments] == [
        (86.0, 100.0, 96.0, 100.0),
        (104.0, 100.0, 114.0, 100.0),
        (100.0, 86.0, 100.0, 96.0),
        (100.0, 104.0, 100.0, 114.0),
    ]


def test_main_reticle_draw_uses_red_three_pixel_pen_and_preview_draws_none(
    qt_application: QApplication,
) -> None:
    class _PainterSpy:
        def __init__(self) -> None:
            self.pen = None
            self.lines = []

        def setPen(self, pen) -> None:
            self.pen = pen

        def drawLine(self, line) -> None:
            self.lines.append(line)

    result = _result(CameraRole.OVERVIEW, 1)
    prepared = prepare_vision_frame(result)
    main = VideoView(
        camera=CameraRole.OVERVIEW,
        preview=False,
        stale_timeout_ns=500_000_000,
    )
    main.resize(200, 200)
    main.set_prepared_frame(prepared)
    main.set_aim_point((49.5, 24.5))
    painter = _PainterSpy()
    main._draw_reticle(painter, main.rendered_rect(), result)

    assert painter.pen is not None
    assert painter.pen.color().getRgb()[:3] == (255, 0, 0)
    assert painter.pen.widthF() == 3.0
    assert painter.pen.capStyle() == Qt.PenCapStyle.FlatCap
    assert len(painter.lines) == 4

    preview = VideoView(
        camera=CameraRole.OVERVIEW,
        preview=True,
        stale_timeout_ns=500_000_000,
    )
    preview.resize(200, 200)
    preview.set_prepared_frame(prepared)
    preview.set_aim_point((49.5, 24.5))
    preview_painter = _PainterSpy()
    preview._draw_reticle(preview_painter, preview.rendered_rect(), result)
    assert preview_painter.pen is None
    assert preview_painter.lines == []
    main.close()
    preview.close()


def test_main_reticle_uses_aiming_fallback_and_clears_on_generation_boundary(
    qt_application: QApplication,
) -> None:
    fallback = AimPointsConfig(
        overview=AimPointConfig(None, None),
        stereo_left=AimPointConfig(None, None),
    )
    window, _mediator_value, _turret, bindings, _states = _window(
        qt_application, aim_points=fallback
    )
    _publish_camera(window, bindings, _result(CameraRole.OVERVIEW, 1))

    assert window.main_view.aim_point == (49.5, 24.5)
    assert window.preview_view.aim_point is None

    bindings[CameraRole.OVERVIEW].session_barriers.publish(
        _session(CameraRole.OVERVIEW, 2)
    )
    window.state_pump.pump_once()
    assert window.main_view.displayed_result is None
    assert window.main_view.aim_point is None
    window.close()


def test_main_reticle_follows_camera_swap_and_dynamic_aiming_config(
    qt_application: QApplication,
) -> None:
    initial = AimPointsConfig(
        overview=AimPointConfig(20, 10),
        stereo_left=AimPointConfig(80, 40),
    )
    window, mediator, _turret, bindings, _states = _window(
        qt_application, aim_points=initial
    )
    _publish_camera(window, bindings, _result(CameraRole.OVERVIEW, 1))
    _publish_camera(window, bindings, _result(CameraRole.STEREO_LEFT, 1))
    assert window.main_view.camera is CameraRole.OVERVIEW
    assert window.main_view.aim_point == (20.0, 10.0)
    assert window.preview_view.aim_point is None

    window.swap_main_preview()
    assert window.main_view.camera is CameraRole.STEREO_LEFT
    assert window.main_view.aim_point == (80.0, 40.0)
    assert window.preview_view.aim_point is None

    mediator.aiming.replace_config(
        AimingConfig(
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(
                overview=AimPointConfig(25, 15),
                stereo_left=AimPointConfig(70, 30),
            ),
        )
    )
    window.refresh_presentation()
    assert window.main_view.aim_point == (70.0, 30.0)
    window.close()


def test_freshness_boundary_is_deterministic() -> None:
    status = _status(CameraRole.OVERVIEW, 1, timestamp_ns=1_000_000_000)
    assert not is_camera_stale(
        status,
        1_499_999_999,
        threshold_ns=500_000_000,
    )
    assert is_camera_stale(
        status,
        1_500_000_000,
        threshold_ns=500_000_000,
    )


def test_relative_click_uses_displayed_result_not_newer_unpumped_latest(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, turret, bindings, _states = _window(qt_application)
    first = _result(CameraRole.OVERVIEW, 1, frame_id=1)
    _publish_camera(window, bindings, first)
    newer = _result(CameraRole.OVERVIEW, 1, frame_id=2)
    bindings[CameraRole.OVERVIEW].latest_result.publish(newer)

    window._handle_main_click(first, 60.0, 25.0)

    assert window.main_view.displayed_result is first
    assert len(turret.relative_commands) == 1
    assert _mediator_value.selected_target is None
    window.close()


def test_tracking_click_sends_exact_displayed_frame_to_mediator(
    qt_application: QApplication,
) -> None:
    window, mediator, _turret, bindings, _states = _window(
        qt_application, mode=TurretControlMode.TRACKING
    )
    objects = (
        TrackedObject(9, BBox(30, 10, 40, 30), 0.0, 0.0, 1),
        TrackedObject(3, BBox(30, 10, 40, 30), 0.0, 0.0, 1),
    )
    displayed = _result(CameraRole.OVERVIEW, 1, objects=objects)
    _publish_camera(window, bindings, displayed)
    bindings[CameraRole.OVERVIEW].latest_result.publish(
        _result(CameraRole.OVERVIEW, 1, frame_id=2, objects=())
    )

    window._handle_main_click(displayed, 50.0, 25.0)
    request = mediator.pending_selection
    assert request is not None
    assert request.camera is CameraRole.OVERVIEW
    assert request.generation == 1
    assert request.frame_id == displayed.frame.frame_id
    assert (request.x_px, request.y_px) == (50.0, 25.0)
    assert mediator.selected_target is None

    selected = _publish_acquired(window, mediator, bindings, track_id=3)
    assert mediator.selected_target == selected
    window.close()


def test_tracking_empty_point_is_forwarded_without_ui_hit_test(
    qt_application: QApplication,
) -> None:
    window, mediator, _turret, bindings, _states = _window(
        qt_application, mode=TurretControlMode.TRACKING
    )
    displayed = _result(CameraRole.OVERVIEW, 1, objects=())
    _publish_camera(window, bindings, displayed)

    window._handle_main_click(displayed, 5.0, 5.0)

    request = mediator.pending_selection
    assert request is not None
    assert request.frame_id == displayed.frame.frame_id
    assert (request.x_px, request.y_px) == (5.0, 5.0)
    window.close()


def test_right_click_and_letterbox_do_nothing(qt_application: QApplication) -> None:
    window, _mediator_value, turret, bindings, _states = _window(qt_application)
    _publish_camera(window, bindings, _result(CameraRole.OVERVIEW, 1))
    window.main_view.resize(200, 200)

    QTest.mouseClick(window.main_view, Qt.MouseButton.RightButton, pos=QPoint(100, 100))
    QTest.mouseClick(window.main_view, Qt.MouseButton.LeftButton, pos=QPoint(100, 20))

    assert turret.relative_commands == []
    window.close()


def test_preview_click_and_action_swap_exactly_once_without_aiming(
    qt_application: QApplication,
) -> None:
    window, mediator, turret, bindings, _states = _window(qt_application)
    _publish_camera(window, bindings, _result(CameraRole.OVERVIEW, 1))
    _publish_camera(window, bindings, _result(CameraRole.STEREO_LEFT, 1))

    QTest.mouseClick(window.preview_view, Qt.MouseButton.LeftButton, pos=QPoint(50, 50))
    assert mediator.main_camera is CameraRole.STEREO_LEFT
    assert window.main_view.displayed_result is mediator.latest_vision_result(
        CameraRole.STEREO_LEFT
    )
    assert turret.relative_commands == []

    button = window.preview_view.swap_button
    assert button is not None
    assert button.width() < 42
    assert button.height() < 34

    inactive_style = button.styleSheet()
    QTest.mouseMove(window.preview_view, QPoint(50, 50))
    qt_application.processEvents()
    active_style = button.styleSheet()
    assert active_style != inactive_style

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert mediator.main_camera is CameraRole.OVERVIEW
    assert turret.relative_commands == []

    QTest.mouseMove(window.main_view, QPoint(50, 50))
    qt_application.processEvents()
    assert button.styleSheet() == inactive_style
    window.close()


def test_operational_bar_is_compact_and_emergency_remains_largest(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application
    )

    assert window.operational_bar.height() < 84
    ordinary_controls = (
        window.mode_button,
        window.motor_button,
        window.connection_label,
        window.now_label,
        window.distance_mode_combo,
        window.distance_slider,
        window.distance_value_label,
    )
    assert all(control.height() <= 42 for control in ordinary_controls)
    assert window.emergency_button.height() > max(
        control.height() for control in ordinary_controls
    )
    assert window.emergency_button.width() > max(
        control.width() for control in ordinary_controls
    )
    window.close()


def test_distance_controls_follow_main_camera_and_preserve_runtime_mode(
    qt_application: QApplication,
) -> None:
    window, mediator, _turret, _bindings, _states = _window(qt_application)

    assert [
        window.distance_mode_combo.itemText(index)
        for index in range(window.distance_mode_combo.count())
    ] == [
        "Константа",
        "Стерео — точная пара",
        "Стерео — по времени",
    ]
    assert mediator.distance_mode is DistanceMeasurementMode.CONSTANT
    assert not window.distance_mode_combo.isEnabled()
    assert window.distance_slider.isHidden()
    assert not window.distance_slider.isEnabled()
    assert window.distance_value_label.text() == "—"

    window.swap_main_preview()
    assert mediator.main_camera is CameraRole.STEREO_LEFT
    assert window.distance_mode_combo.isEnabled()
    assert window.distance_slider.isVisible()
    assert window.distance_slider.isEnabled()
    assert window.distance_slider.minimum() == 10
    assert window.distance_slider.maximum() == 500
    assert window.distance_slider.singleStep() == 10
    assert window.distance_slider.pageStep() == 10
    assert window.distance_slider.value() == 100
    assert window.distance_value_label.text() == "100 м"

    window.distance_slider.setValue(247)
    assert mediator.manual_distance_m == 250
    assert window.distance_slider.value() == 250
    assert window.distance_value_label.text() == "250 м"

    window.distance_mode_combo.setCurrentIndex(1)
    assert mediator.distance_mode is DistanceMeasurementMode.STEREO_EXACT
    assert window.distance_slider.isHidden()
    assert not window.distance_slider.isEnabled()
    assert window.distance_value_label.text() == "—"

    window.distance_mode_combo.setCurrentIndex(2)
    assert mediator.distance_mode is DistanceMeasurementMode.STEREO_TIMESTAMP
    assert window.distance_value_label.text() == "—"

    window.swap_main_preview()
    assert mediator.main_camera is CameraRole.OVERVIEW
    assert not window.distance_mode_combo.isEnabled()
    assert window.distance_value_label.text() == "—"

    window.swap_main_preview()
    assert mediator.main_camera is CameraRole.STEREO_LEFT
    assert mediator.distance_mode is DistanceMeasurementMode.STEREO_TIMESTAMP
    assert window.distance_mode_combo.currentIndex() == 2
    assert window.distance_slider.isHidden()
    assert window.distance_value_label.text() == "—"
    window.close()


def test_timestamp_distance_is_shown_without_approximation_marker(
    qt_application: QApplication,
) -> None:
    joiner = StereoTargetJoiner()
    results: LatestValue[StereoDistanceCandidate] = LatestValue()
    window, mediator, _turret, bindings, _states = _window(
        qt_application,
        mode=TurretControlMode.TRACKING,
        stereo_target_joiner=joiner,
        stereo_distance_results=results,
    )
    _publish_camera(window, bindings, _result(CameraRole.OVERVIEW, 1))
    stereo_result = _result(
        CameraRole.STEREO_LEFT,
        1,
        timestamp_ns=monotonic_ns(),
    )
    _publish_camera(window, bindings, stereo_result)

    window.swap_main_preview()
    window._handle_main_click(stereo_result, 50.0, 25.0)
    selected = _publish_acquired(window, mediator, bindings)
    window.distance_mode_combo.setCurrentIndex(2)
    assert mediator.distance_mode is DistanceMeasurementMode.STEREO_TIMESTAMP
    assert window.distance_value_label.text() == "—"

    results.publish(
        StereoDistanceCandidate(
            target=selected,
            intent_revision=joiner.state.intent_revision,
            distance_m=12.44,
            source_frame_id=stereo_result.frame.frame_id,
            capture_id=None,
            measured_timestamp_ns=stereo_result.frame.receive_timestamp_ns,
            pairing_kind=StereoPairingKind.TIMESTAMP,
        )
    )
    window.refresh_presentation()

    assert window.distance_value_label.text() == "12.4 м"
    assert "~" not in window.distance_value_label.text()
    assert window.main_view.distance_annotation_text == "12.4 м"
    assert window.preview_view.distance_annotation_text is None
    window.close()



def test_distance_slider_standard_wheel_step_is_ten_metres(
    qt_application: QApplication,
) -> None:
    window, mediator, _turret, _bindings, _states = _window(qt_application)
    window.swap_main_preview()
    assert window.distance_slider.value() == 100

    up = QWheelEvent(
        QPointF(20.0, 16.0),
        QPointF(20.0, 16.0),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(window.distance_slider, up)
    assert mediator.manual_distance_m == 110
    assert window.distance_slider.value() == 110

    down = QWheelEvent(
        QPointF(20.0, 16.0),
        QPointF(20.0, 16.0),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(window.distance_slider, down)
    assert mediator.manual_distance_m == 100
    assert window.distance_slider.value() == 100
    window.close()


def test_target_distance_annotation_is_main_stereo_target_bound(
    qt_application: QApplication,
) -> None:
    window, mediator, _turret, bindings, _states = _window(
        qt_application, mode=TurretControlMode.TRACKING
    )
    _publish_camera(window, bindings, _result(CameraRole.OVERVIEW, 1))
    stereo_result = _result(CameraRole.STEREO_LEFT, 1)
    _publish_camera(window, bindings, stereo_result)

    window.swap_main_preview()
    window._handle_main_click(stereo_result, 50.0, 25.0)
    selected = _publish_acquired(window, mediator, bindings)

    distance = mediator.current_distance
    assert distance is not None
    assert distance.target == selected
    assert distance.source is DistanceSource.MANUAL
    assert window.main_view.target_distance == distance
    assert window.main_view.distance_annotation_text == "100 м"
    assert window.preview_view.target_distance is None
    assert window.preview_view.distance_annotation_text is None

    window.main_view.set_target_distance(
        DistanceResult(
            target=selected,
            distance_m=12.44,
            source=DistanceSource.STEREO,
            source_frame_id=stereo_result.frame.frame_id,
            capture_id=7,
            measured_timestamp_ns=stereo_result.frame.receive_timestamp_ns,
        )
    )
    assert window.main_view.distance_annotation_text == "12.4 м"

    window.swap_main_preview()
    assert mediator.main_camera is CameraRole.OVERVIEW
    assert window.main_view.target_distance is None
    assert window.main_view.distance_annotation_text is None
    assert window.preview_view.target_distance is None
    window.close()


def test_now_clock_is_diagnostic_only_and_updates_in_operational_bar(
    qt_application: QApplication,
) -> None:
    normal, *_normal_dependencies = _window(qt_application)
    assert normal.now_label.isHidden()
    normal.close()

    wall_clock = _WallClock(
        datetime(2026, 9, 22, 14, 58, 21, 515_000, tzinfo=UTC)
    )
    diagnostic, *_diagnostic_dependencies = _window(
        qt_application,
        wall_clock=wall_clock,
        show_diagnostic_clock=True,
    )
    assert diagnostic.now_label.isVisible()
    assert diagnostic.now_label.text() == "NOW 14:58:21.515"
    assert diagnostic.now_label.parent() is diagnostic.operational_bar
    assert diagnostic.emergency_button.isVisible()

    wall_clock.value = datetime(
        2026,
        9,
        22,
        14,
        58,
        22,
        7_000,
        tzinfo=UTC,
    )
    diagnostic.refresh_presentation()
    assert diagnostic.now_label.text() == "NOW 14:58:22.007"
    diagnostic.close()


def test_mode_motor_connection_and_emergency_use_authoritative_state(
    qt_application: QApplication,
) -> None:
    window, mediator, turret, _bindings_value, states = _window(qt_application)
    assert "RELATIVE" in window.mode_button.text()
    assert "READY" in window.connection_label.text()

    QTest.mouseClick(window.mode_button, Qt.MouseButton.LeftButton)
    assert mediator.turret_state.control_mode is TurretControlMode.RELATIVE
    assert "RELATIVE" in window.mode_button.text()
    assert "TRACKING" in window.mode_button.text()
    assert not window.mode_button.isEnabled()
    QTest.mouseClick(window.emergency_button, Qt.MouseButton.LeftButton)
    assert turret.emergency_requests == 1

    states.publish(_turret_state(mode=TurretControlMode.TRACKING))
    window.state_pump.pump_once()
    assert window.mode_button.text() == "РЕЖИМ: TRACKING"
    assert window.mode_button.isEnabled()

    QTest.mouseClick(window.motor_button, Qt.MouseButton.LeftButton)
    assert turret.motor_off_requests == 1
    states.publish(
        _turret_state(
            mode=TurretControlMode.TRACKING,
            connection=TurretConnectionState.CONNECTING,
        )
    )
    window.state_pump.pump_once()
    assert "CONNECTING" in window.connection_label.text()
    window.close()


def test_motor_unknown_is_disabled_and_sends_nothing(qt_application: QApplication) -> None:
    window, _mediator_value, turret, _bindings_value, _states = _window(
        qt_application, motor=MotorState.UNKNOWN
    )
    assert not window.motor_button.isEnabled()
    window.motor_button.click()
    assert turret.motor_on_requests == 0
    assert turret.motor_off_requests == 0
    window.close()


def test_motor_off_click_requests_motor_on(qt_application: QApplication) -> None:
    window, _mediator_value, turret, _bindings_value, _states = _window(
        qt_application, motor=MotorState.OFF
    )
    QTest.mouseClick(window.motor_button, Qt.MouseButton.LeftButton)
    assert turret.motor_on_requests == 1
    assert turret.motor_off_requests == 0
    window.close()


def test_pump_drains_all_barriers_before_freshest_payloads_and_coalesces() -> None:
    mediator, _turret = _mediator(counting=True)
    assert isinstance(mediator, _CountingMediator)
    bindings = _bindings()
    states: LatestValue[TurretState] = LatestValue()
    accepted_sessions: list[tuple[CameraRole, int]] = []
    accepted_results: list[int] = []
    pump = UiStatePump(
        mediator=mediator,
        camera_bindings=bindings,
        turret_states=states,
        on_camera_session=lambda item: accepted_sessions.append((item.camera, item.generation)),
        on_vision_result=lambda item: accepted_results.append(item.frame.frame_id),
        on_camera_status=lambda _item: None,
        on_turret_state=lambda _item: None,
        on_presentation_tick=lambda: None,
    )
    overview = bindings[CameraRole.OVERVIEW]
    left = bindings[CameraRole.STEREO_LEFT]
    overview.session_barriers.publish(_session(CameraRole.OVERVIEW, 1))
    overview.session_barriers.publish(_session(CameraRole.OVERVIEW, 2))
    left.session_barriers.publish(_session(CameraRole.STEREO_LEFT, 4))
    overview.latest_result.publish(_result(CameraRole.OVERVIEW, 2, frame_id=1))
    overview.latest_result.publish(_result(CameraRole.OVERVIEW, 2, frame_id=2))
    left.latest_result.publish(_result(CameraRole.STEREO_LEFT, 4, frame_id=3))

    pump.pump_once()
    pump.pump_once()

    assert accepted_sessions == [
        (CameraRole.OVERVIEW, 1),
        (CameraRole.OVERVIEW, 2),
        (CameraRole.STEREO_LEFT, 4),
    ]
    assert accepted_results == [2, 3]
    assert mediator.vision_accept_calls == 2


def test_pump_orders_generation_barrier_then_lifecycle_then_latest_result() -> None:
    mediator, _turret = _mediator(
        mode=TurretControlMode.TRACKING,
        counting=True,
    )
    assert isinstance(mediator, _CountingMediator)
    bindings = _bindings()
    first = _result(CameraRole.OVERVIEW, 1)
    assert mediator.accept_camera_session(_session(CameraRole.OVERVIEW, 1))
    assert mediator.accept_vision_result(first)
    assert mediator.request_target_selection(first, 50.0, 25.0)
    pending = mediator.pending_selection
    assert pending is not None
    mediator.accept_order.clear()

    overview = bindings[CameraRole.OVERVIEW]
    overview.session_barriers.publish(_session(CameraRole.OVERVIEW, 2))
    overview.target_events.publish(
        TargetAcquired(
            selection_id=pending.selection_id,
            target=TargetRef(CameraRole.OVERVIEW, 1, 7),
            request_frame_id=pending.frame_id,
            decision_frame_id=pending.frame_id,
            decision_frame_timestamp_ns=1_000_000_000,
        )
    )
    overview.latest_result.publish(
        _result(CameraRole.OVERVIEW, 2, frame_id=1)
    )
    pump = UiStatePump(
        mediator=mediator,
        camera_bindings=bindings,
        turret_states=LatestValue(),
        on_camera_session=lambda _item: None,
        on_vision_result=lambda _item: None,
        on_camera_status=lambda _item: None,
        on_turret_state=lambda _item: None,
        on_presentation_tick=lambda: None,
    )

    pump.pump_once()

    assert mediator.accept_order == ["session", "target_event", "vision_result"]
    assert mediator.pending_selection is None
    assert mediator.selected_target is None
    assert mediator.latest_vision_result(CameraRole.OVERVIEW).frame.generation == 2


def test_rejected_new_generation_revision_is_retried_after_barrier() -> None:
    mediator, _turret = _mediator(counting=True)
    assert isinstance(mediator, _CountingMediator)
    bindings = _bindings()
    overview = bindings[CameraRole.OVERVIEW]
    overview.latest_result.publish(_result(CameraRole.OVERVIEW, 1))
    accepted: list[VisionResult] = []
    pump = UiStatePump(
        mediator=mediator,
        camera_bindings=bindings,
        turret_states=LatestValue(),
        on_camera_session=lambda _item: None,
        on_vision_result=accepted.append,
        on_camera_status=lambda _item: None,
        on_turret_state=lambda _item: None,
        on_presentation_tick=lambda: None,
    )

    pump.pump_once()
    assert accepted == []
    overview.session_barriers.publish(_session(CameraRole.OVERVIEW, 1))
    pump.pump_once()

    assert len(accepted) == 1
    assert mediator.vision_accept_calls == 2


def test_same_turret_revision_is_forwarded_once() -> None:
    mediator, _turret = _mediator(counting=True)
    assert isinstance(mediator, _CountingMediator)
    states: LatestValue[TurretState] = LatestValue()
    states.publish(_turret_state(connection=TurretConnectionState.CONNECTING))
    pump = UiStatePump(
        mediator=mediator,
        camera_bindings=_bindings(),
        turret_states=states,
        on_camera_session=lambda _item: None,
        on_vision_result=lambda _item: None,
        on_camera_status=lambda _item: None,
        on_turret_state=lambda _item: None,
        on_presentation_tick=lambda: None,
    )
    pump.pump_once()
    pump.pump_once()
    assert mediator.turret_accept_calls == 1


def test_new_generation_clears_display_and_old_result_cannot_return(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, _turret, bindings, _states = _window(qt_application)
    old = _result(CameraRole.OVERVIEW, 1)
    _publish_camera(window, bindings, old)
    assert window.main_view.displayed_result is old

    bindings[CameraRole.OVERVIEW].session_barriers.publish(
        _session(CameraRole.OVERVIEW, 2)
    )
    window.state_pump.pump_once()
    assert window.main_view.displayed_result is None

    bindings[CameraRole.OVERVIEW].latest_result.publish(old)
    window.state_pump.pump_once()
    assert window.main_view.displayed_result is None
    window.close()


def test_temporary_target_miss_keeps_authoritative_selection_without_old_bbox(
    qt_application: QApplication,
) -> None:
    window, mediator, _turret, bindings, _states = _window(
        qt_application, mode=TurretControlMode.TRACKING
    )
    initial = _result(CameraRole.OVERVIEW, 1)
    _publish_camera(window, bindings, initial)
    window._handle_main_click(initial, 50.0, 25.0)
    selected = _publish_acquired(window, mediator, bindings)

    missing = _result(CameraRole.OVERVIEW, 1, frame_id=2, objects=())
    bindings[CameraRole.OVERVIEW].latest_result.publish(missing)
    window.state_pump.pump_once()

    assert mediator.selected_target == selected
    assert window.main_view.displayed_result is missing
    assert missing.tracked_objects == ()
    window.close()


def test_same_revision_does_not_repeat_frame_conversion(
    qt_application: QApplication,
) -> None:
    conversions: list[int] = []

    def prepare(result: VisionResult) -> PreparedVisionFrame:
        conversions.append(result.frame.frame_id)
        return PreparedVisionFrame(result, QPixmap(1, 1))

    window, _mediator_value, _turret, bindings, _states = _window(
        qt_application, frame_preparer=prepare
    )
    binding = bindings[CameraRole.OVERVIEW]
    binding.session_barriers.publish(_session(CameraRole.OVERVIEW, 1))
    binding.latest_result.publish(_result(CameraRole.OVERVIEW, 1, frame_id=1))
    window.state_pump.pump_once()
    window.state_pump.pump_once()
    window.state_pump.pump_once()
    binding.latest_result.publish(_result(CameraRole.OVERVIEW, 1, frame_id=2))
    binding.latest_result.publish(_result(CameraRole.OVERVIEW, 1, frame_id=3))
    window.state_pump.pump_once()

    assert conversions == [1, 3]
    window.close()


def test_configured_stale_timeout_controls_presentation_and_interaction(
    qt_application: QApplication,
) -> None:
    clock = _Clock()
    window, mediator, turret, bindings, _states = _window(
        qt_application,
        clock=clock,
        camera_stale_timeout_ms=1_000,
    )
    main_result = _result(CameraRole.OVERVIEW, 1)
    _publish_camera(window, bindings, main_result)
    _publish_camera(window, bindings, _result(CameraRole.STEREO_LEFT, 1))
    displayed = window.main_view.displayed_result

    clock.now_ns = 1_999_999_999
    window.refresh_presentation()
    assert not window.main_view.is_stale
    window._handle_main_click(main_result, 50.0, 25.0)
    assert len(turret.relative_commands) == 1

    clock.now_ns = 2_000_000_000
    window.refresh_presentation()
    assert window.main_view.is_stale
    assert window.main_view.stale_overlay_text == "НЕТ НОВЫХ КАДРОВ"
    assert window.main_view.displayed_result is displayed
    window._handle_main_click(main_result, 50.0, 25.0)
    assert len(turret.relative_commands) == 1

    window.swap_main_preview()
    assert mediator.main_camera is CameraRole.STEREO_LEFT
    window.close()


def test_stale_tracking_does_not_select_or_deselect(qt_application: QApplication) -> None:
    clock = _Clock(1_000_000_000)
    window, mediator, _turret, bindings, _states = _window(
        qt_application, mode=TurretControlMode.TRACKING, clock=clock
    )
    result = _result(CameraRole.OVERVIEW, 1)
    _publish_camera(
        window,
        bindings,
        result,
        status=_status(CameraRole.OVERVIEW, 1, timestamp_ns=1_000_000_000),
    )
    window._handle_main_click(result, 50.0, 25.0)
    selected = _publish_acquired(window, mediator, bindings)
    clock.now_ns = 1_500_000_000
    window.refresh_presentation()
    window._handle_main_click(result, 5.0, 5.0)
    assert mediator.selected_target == selected
    window.close()


def test_fresh_result_clears_stale_and_error_keeps_frozen_frame(
    qt_application: QApplication,
) -> None:
    clock = _Clock(1_500_000_000)
    window, _mediator_value, _turret, bindings, _states = _window(
        qt_application, clock=clock
    )
    first = _result(CameraRole.OVERVIEW, 1)
    _publish_camera(
        window,
        bindings,
        first,
        status=_status(CameraRole.OVERVIEW, 1, timestamp_ns=1_000_000_000),
    )
    assert window.main_view.is_stale

    fresh = _result(CameraRole.OVERVIEW, 1, frame_id=2, timestamp_ns=1_500_000_000)
    bindings[CameraRole.OVERVIEW].latest_result.publish(fresh)
    bindings[CameraRole.OVERVIEW].status.publish(
        _status(CameraRole.OVERVIEW, 1, timestamp_ns=1_500_000_000)
    )
    window.state_pump.pump_once()
    assert not window.main_view.is_stale

    bindings[CameraRole.OVERVIEW].status.publish(
        _status(
            CameraRole.OVERVIEW,
            1,
            timestamp_ns=1_500_000_000,
            state=CameraState.ERROR,
            message="decoder failed",
        )
    )
    window.state_pump.pump_once()
    assert window.main_view.displayed_result is fresh
    assert window.main_view.camera_status is not None
    assert window.main_view.camera_status.state is CameraState.ERROR
    window._handle_main_click(fresh, 50.0, 25.0)
    assert _turret.relative_commands == []
    window.close()


def test_no_frame_uses_placeholder_state(qt_application: QApplication) -> None:
    view = VideoView(
        camera=CameraRole.OVERVIEW,
        preview=False,
        stale_timeout_ns=500_000_000,
    )
    view.resize(320, 200)
    view.show()
    qt_application.processEvents()
    assert view.displayed_result is None
    assert view.aim_point is None
    assert view.rendered_rect().isEmpty()
    status = _status(CameraRole.OVERVIEW, 1, state=CameraState.STARTING)
    view.set_camera_status(status)
    assert view.camera_status is status
    view.close()


def test_f11_toggles_fullscreen_repeatedly(qt_application: QApplication) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(qt_application)
    window.showFullScreen()
    qt_application.processEvents()
    assert window.isFullScreen()

    QTest.keyClick(window, Qt.Key.Key_F11)
    qt_application.processEvents()
    assert not window.isFullScreen()
    assert window.isVisible()

    QTest.keyClick(window, Qt.Key.Key_F11)
    qt_application.processEvents()
    assert window.isFullScreen()
    window.close()


def test_escape_toggles_single_modeless_operator_window(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application
    )
    operator = window.operator_window
    assert operator.parentWidget() is window
    assert operator.windowModality() is Qt.WindowModality.NonModal
    assert operator.windowFlags() & Qt.WindowType.Tool
    assert not operator.windowFlags() & Qt.WindowType.WindowStaysOnTopHint

    QTest.keyClick(window, Qt.Key.Key_Escape)
    qt_application.processEvents()
    assert operator.isVisible()
    assert window.operator_window is operator

    QTest.keyClick(window, Qt.Key.Key_Escape)
    qt_application.processEvents()
    assert not operator.isVisible()
    assert window.isVisible()

    QTest.keyClick(window, Qt.Key.Key_Escape)
    qt_application.processEvents()
    assert operator.isVisible()
    operator.close()
    qt_application.processEvents()
    assert not operator.isVisible()
    assert window.isVisible()
    window.close()



class _ImuControlRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    def request_gyro_bias_calibration(self) -> bool:
        self.events.append(("gyro", None))
        return True

    def request_accelerometer_face(self, face: int) -> bool:
        self.events.append(("accel", face))
        return True

    def request_magnetometer_start(self) -> bool:
        self.events.append(("mag-start", None))
        return True

    def request_magnetometer_finish(self) -> bool:
        self.events.append(("mag-finish", None))
        return True

    def request_relative_zero(self) -> bool:
        self.events.append(("zero", None))
        return True


def test_imu_calibration_controls_use_only_passive_imu_ingress(
    qt_application: QApplication,
) -> None:
    controls = _ImuControlRecorder()
    imu_states: LatestValue[ImuRuntimeState] = LatestValue()
    imu_states.publish(_imu_state())
    window, _mediator_value, turret, _bindings_value, _states = _window(
        qt_application,
        imu_states=imu_states,
        imu_controls=controls,
    )
    imu = window.imu_window
    imu.show()
    qt_application.processEvents()
    before = (
        len(turret.relative_commands),
        len(turret.tracking_controls),
        turret.invalidations,
        len(turret.mode_requests),
        turret.stop_requests,
        turret.motor_on_requests,
        turret.motor_off_requests,
        turret.emergency_requests,
    )

    QTest.mouseClick(imu.gyro_calibration_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(imu.accel_face_buttons[1], Qt.MouseButton.LeftButton)
    QTest.mouseClick(imu.relative_zero_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(imu.mag_start_button, Qt.MouseButton.LeftButton)

    imu_states.publish(
        _imu_state(
            accel_faces_captured=(1,),
            mag_calibration_active=True,
            mag_calibration_sample_count=321,
            mag_calibration_octants=6,
            calibration_message="Магнитометр: идёт сбор.",
        )
    )
    imu.refresh_once()
    assert not imu.gyro_calibration_button.isEnabled()
    assert not imu.accel_face_buttons[1].isEnabled()
    assert not imu.mag_start_button.isEnabled()
    assert imu.mag_finish_button.isEnabled()
    assert "X+ ✓" == imu.accel_face_buttons[1].text()
    assert "сбор MAG=321" in imu.quality_label.text()
    assert "охват≈6/8" in imu.quality_label.text()
    QTest.mouseClick(imu.mag_finish_button, Qt.MouseButton.LeftButton)

    assert controls.events == [
        ("gyro", None),
        ("accel", 1),
        ("zero", None),
        ("mag-start", None),
        ("mag-finish", None),
    ]
    after = (
        len(turret.relative_commands),
        len(turret.tracking_controls),
        turret.invalidations,
        len(turret.mode_requests),
        turret.stop_requests,
        turret.motor_on_requests,
        turret.motor_off_requests,
        turret.emergency_requests,
    )
    assert after == before

    imu_states.publish(_imu_state(fresh=False, orientation_valid=False))
    imu.refresh_once()
    assert not imu.gyro_calibration_button.isEnabled()
    assert not imu.accel_face_buttons[1].isEnabled()
    assert not imu.mag_start_button.isEnabled()
    assert not imu.relative_zero_button.isEnabled()
    window.close()


def test_position_button_reuses_one_modeless_imu_window_and_esc_does_not_hide_it(
    qt_application: QApplication,
) -> None:
    imu_states: LatestValue[ImuRuntimeState] = LatestValue()
    imu_states.publish(_imu_state())
    window, _mediator_value, turret, _bindings_value, _states = _window(
        qt_application,
        imu_states=imu_states,
    )
    operator = window.operator_window
    imu = window.imu_window
    before = (
        len(turret.relative_commands),
        len(turret.tracking_controls),
        turret.invalidations,
        len(turret.mode_requests),
        turret.stop_requests,
        turret.motor_on_requests,
        turret.motor_off_requests,
        turret.emergency_requests,
    )

    operator.show()
    QTest.mouseClick(operator.position_button, Qt.MouseButton.LeftButton)
    qt_application.processEvents()

    assert imu.isVisible()
    assert imu.parentWidget() is window
    assert imu.windowModality() is Qt.WindowModality.NonModal
    assert imu.windowFlags() & Qt.WindowType.Tool
    assert window.imu_window is imu
    assert imu.states is imu_states
    assert imu.status_label.text().startswith("LIVE")

    imu.close()
    qt_application.processEvents()
    assert not imu.isVisible()
    imu_states.publish(_imu_state(euler=(4.0, 5.0, 6.0)))

    QTest.mouseClick(operator.position_button, Qt.MouseButton.LeftButton)
    qt_application.processEvents()
    assert window.imu_window is imu
    assert imu.isVisible()
    assert "Крен: +4.0°" in imu.angles_label.text()

    QTest.keyClick(imu, Qt.Key.Key_Escape)
    qt_application.processEvents()
    assert not operator.isVisible()
    assert imu.isVisible()
    QTest.keyClick(imu, Qt.Key.Key_Escape)
    qt_application.processEvents()
    assert operator.isVisible()
    assert imu.isVisible()

    after = (
        len(turret.relative_commands),
        len(turret.tracking_controls),
        turret.invalidations,
        len(turret.mode_requests),
        turret.stop_requests,
        turret.motor_on_requests,
        turret.motor_off_requests,
        turret.emergency_requests,
    )
    assert after == before
    window.close()


def test_imu_window_distinguishes_calibrating_stale_error_and_disabled(
    qt_application: QApplication,
) -> None:
    imu_states: LatestValue[ImuRuntimeState] = LatestValue()
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application,
        imu_states=imu_states,
    )
    imu = window.imu_window

    imu_states.publish(
        _imu_state(
            gyro_bias_state=GyroBiasState.ACQUIRING,
            orientation_valid=False,
        )
    )
    imu.refresh_once()
    assert imu.status_label.text().startswith("CALIBRATING")
    assert not imu.attitude_view.fresh

    imu_states.publish(_imu_state(fresh=False, orientation_valid=False))
    imu.refresh_once()
    assert imu.status_label.text().startswith("STALE")
    assert not imu.attitude_view.fresh

    imu_states.publish(_imu_state(running=False, fresh=False, orientation_valid=False))
    imu.refresh_once()
    assert imu.status_label.text().startswith("ERROR")
    assert not imu.attitude_view.fresh
    window.close()

    disabled_window, *_ = _window(qt_application, imu_states=None)
    disabled_window.imu_window.refresh_once()
    assert disabled_window.imu_window.status_label.text() == (
        "ERROR — IMU отключён в конфигурации"
    )
    disabled_window.close()

def test_operator_window_statuses_follow_existing_ui_update_path(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, _turret, bindings, states = _window(qt_application)
    operator = window.operator_window
    overview = _status(CameraRole.OVERVIEW, 1, state=CameraState.ONLINE)
    left = _status(CameraRole.STEREO_LEFT, 3, state=CameraState.RECONNECTING)
    bindings[CameraRole.OVERVIEW].status.publish(overview)
    bindings[CameraRole.STEREO_LEFT].status.publish(left)
    states.publish(
        _turret_state(
            mode=TurretControlMode.TRACKING,
            motor=MotorState.OFF,
            connection=TurretConnectionState.CONNECTING,
        )
    )

    window.state_pump.pump_once()

    assert operator.camera_labels[CameraRole.OVERVIEW].text() == "Overview: ONLINE"
    assert (
        operator.camera_labels[CameraRole.STEREO_LEFT].text()
        == "Stereo Left: RECONNECTING"
    )
    assert operator.controller_label.text() == "Контроллер: CONNECTING"
    assert operator.mode_label.text() == "Режим: TRACKING"
    assert operator.motor_label.text() == "Мотор: OFF"
    window.close()


def test_operator_pid_panel_uses_initial_values_and_queues_full_typed_update(
    qt_application: QApplication,
) -> None:
    initial = _pid_config()
    queued: list[PidControllerConfig] = []
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application,
        pid_controller_config=initial,
        apply_pid_tuning=queued.append,
    )
    operator = window.operator_window

    assert operator.pid_inputs["pid_kp_x"].text() == "0.25"
    assert operator.pid_inputs["pid_ki_x"].text() == "0.01"
    assert operator.pid_inputs["pid_kd_x"].text() == "0.04"
    assert operator.pid_inputs["pid_kp_y"].text() == "0.3"
    assert operator.pid_inputs["pid_ki_y"].text() == "0.02"
    assert operator.pid_inputs["pid_kd_y"].text() == "0.05"

    values = {
        "pid_kp_x": "1.25",
        "pid_ki_x": "0.11",
        "pid_kd_x": "0.14",
        "pid_kp_y": "1.30",
        "pid_ki_y": "0.12",
        "pid_kd_y": "0.15",
    }
    for field, text in values.items():
        operator.pid_inputs[field].setText(text)
    QTest.mouseClick(operator.pid_apply_button, Qt.MouseButton.LeftButton)

    assert queued == [
        PidControllerConfig(
            pid_kp_x=1.25,
            pid_ki_x=0.11,
            pid_kd_x=0.14,
            pid_kp_y=1.30,
            pid_ki_y=0.12,
            pid_kd_y=0.15,
        )
    ]
    assert operator.pid_status_label.text().startswith("Отправлено:")
    assert "Применено" not in operator.pid_status_label.text()
    assert operator.pid_error_label.text() == ""

    operator.hide()
    operator.show()
    qt_application.processEvents()
    assert operator.pid_inputs["pid_kp_x"].text() == "1.25"
    window.close()


@pytest.mark.parametrize("invalid", ["-0.1", "nan", "inf", "-inf", "", "abc"])
def test_operator_pid_panel_rejects_invalid_input_without_changing_last_status(
    qt_application: QApplication,
    invalid: str,
) -> None:
    queued: list[PidControllerConfig] = []
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application,
        apply_pid_tuning=queued.append,
    )
    operator = window.operator_window

    QTest.mouseClick(operator.pid_apply_button, Qt.MouseButton.LeftButton)
    assert len(queued) == 1
    successful_status = operator.pid_status_label.text()

    operator.pid_inputs["pid_kp_x"].setText(invalid)
    QTest.mouseClick(operator.pid_apply_button, Qt.MouseButton.LeftButton)

    assert len(queued) == 1
    assert operator.pid_status_label.text() == successful_status
    assert operator.pid_error_label.text().startswith("Ошибка PID:")
    window.close()


def test_operator_tracking_panel_queues_only_exposed_typed_runtime_fields(
    qt_application: QApplication,
) -> None:
    queued: list[TrackingControlConfig] = []
    window, mediator, _turret, _bindings_value, _states = _window(
        qt_application,
        apply_tracking_control_tuning=queued.append,
    )
    operator = window.operator_window
    initial = mediator.tracking_estimator.config

    assert not operator.ff_enabled_checkbox.isChecked()
    assert operator.tracking_inputs["gain_x"].text() == "0"
    assert operator.tracking_inputs["gain_y"].text() == "0"
    assert not operator.prediction_enabled_checkbox.isChecked()
    assert operator.tracking_inputs["horizon_ms"].text() == "0"
    assert operator.tracking_inputs["window_ms"].text() == "200"

    operator.ff_enabled_checkbox.setChecked(True)
    operator.tracking_inputs["gain_x"].setText("0.4")
    operator.tracking_inputs["gain_y"].setText("0.5")
    operator.prediction_enabled_checkbox.setChecked(True)
    operator.tracking_inputs["horizon_ms"].setText("75")
    operator.tracking_inputs["window_ms"].setText("250")
    QTest.mouseClick(operator.tracking_apply_button, Qt.MouseButton.LeftButton)

    assert len(queued) == 1
    update = queued[0]
    assert update.feed_forward == replace(
        initial.feed_forward,
        enabled=True,
        gain_x=0.4,
        gain_y=0.5,
    )
    assert update.prediction == replace(
        initial.prediction,
        enabled=True,
        horizon_ms=75,
    )
    assert update.velocity_estimator == replace(
        initial.velocity_estimator,
        window_ms=250,
    )
    assert update.platform_motion == initial.platform_motion
    assert operator.tracking_status_label.text().startswith("Отправлено:")
    assert "сохран" not in operator.tracking_status_label.text().lower()
    assert operator.tracking_error_label.text() == ""
    window.close()


def test_operator_tracking_diagnostics_render_typed_axis_state(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application
    )
    operator = window.operator_window
    diagnostics = TrackingEstimatorDiagnostics(
        target=None,
        observation_timestamp_ns=123,
        platform_source=PlatformMotionSource.COMMAND_MODEL,
        x=AxisTrackingEstimate(
            relative_rate_deg_s=1.5,
            platform_rate_deg_s=2.0,
            target_rate_deg_s=3.5,
            fit_rms_deg=0.125,
            sample_count=5,
            actual_span_ms=160.0,
            valid=True,
            invalid_reason=None,
            prediction_offset_deg=0.25,
        ),
        y=AxisTrackingEstimate(
            relative_rate_deg_s=-1.0,
            platform_rate_deg_s=None,
            target_rate_deg_s=None,
            fit_rms_deg=0.5,
            sample_count=5,
            actual_span_ms=160.0,
            valid=False,
            invalid_reason=TrackingEstimateInvalidReason.PLATFORM_UNAVAILABLE,
            prediction_offset_deg=0.0,
        ),
    )

    operator.set_tracking_diagnostics(diagnostics)

    assert operator.platform_source_label.text() == "Platform source: COMMAND_MODEL"
    assert operator.tracking_diagnostic_labels["relative_rate_x"].text() == "1.5 deg/s"
    assert operator.tracking_diagnostic_labels["platform_rate_y"].text() == "—"
    assert operator.tracking_diagnostic_labels["sample_count_x"].text() == "5"
    assert operator.tracking_diagnostic_labels["actual_span_y"].text() == "160 ms"
    assert operator.tracking_diagnostic_labels["validity_x"].text() == "VALID"
    assert (
        operator.tracking_diagnostic_labels["validity_y"].text()
        == "INVALID (platform_unavailable)"
    )
    assert operator.tracking_diagnostic_labels["prediction_offset_x"].text() == "0.25 deg"
    window.close()


def test_operator_fullscreen_button_and_f11_keep_text_synchronized(
    qt_application: QApplication,
) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application
    )
    button = window.operator_window.fullscreen_button
    assert button.text() == "Полноэкранный режим: ВЫКЛ"

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qt_application.processEvents()
    assert window.isFullScreen()
    assert button.text() == "Полноэкранный режим: ВКЛ"

    QTest.keyClick(window, Qt.Key.Key_F11)
    qt_application.processEvents()
    assert not window.isFullScreen()
    assert button.text() == "Полноэкранный режим: ВЫКЛ"
    window.close()


def test_operator_exit_confirmation_controls_main_window_close(
    qt_application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(
        qt_application
    )
    operator = window.operator_window
    operator.show()
    monkeypatch.setattr(
        QMessageBox,
        "exec",
        lambda _dialog: QMessageBox.StandardButton.Cancel,
    )
    QTest.mouseClick(operator.exit_button, Qt.MouseButton.LeftButton)
    assert window.isVisible()

    monkeypatch.setattr(
        QMessageBox,
        "exec",
        lambda _dialog: QMessageBox.StandardButton.Yes,
    )
    QTest.mouseClick(operator.exit_button, Qt.MouseButton.LeftButton)
    qt_application.processEvents()
    assert not window.isVisible()


def test_main_window_can_request_fullscreen_at_startup(qt_application: QApplication) -> None:
    mediator, _turret = _mediator()
    window = MainWindow(
        mediator=mediator,
        camera_bindings=_bindings(),
        turret_states=LatestValue(),
        camera_stale_timeout_ms=500,
        pid_controller_config=_pid_config(),
        apply_pid_tuning=lambda _config: None,
        tracking_control_config=mediator.tracking_estimator.config,
        apply_tracking_control_tuning=lambda _config: None,
        start_timer=False,
        start_fullscreen=True,
    )
    qt_application.processEvents()
    assert window.isFullScreen()
    assert window.state_pump.timer.interval() == 16
    window.close()
