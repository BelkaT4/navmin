from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from navmin.concurrency import CameraSessionBarrierChannel, LatestValue
from navmin.config.models import AimingConfig, AimPointConfig, AimPointsConfig, UiConfig
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    CameraSessionStarted,
    CameraState,
    CameraStatus,
    FramePacket,
    MotorState,
    MoveRelativeCommand,
    TargetRef,
    TrackedObject,
    TrackingError,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.core import Mediator
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


class _Clock:
    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _CameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay((x_px - 50.0) / 50.0, (y_px - 25.0) / 25.0, 1.0)


class _FakeTurret:
    def __init__(self, state: TurretState) -> None:
        self._state = state
        self.relative_commands: list[MoveRelativeCommand] = []
        self.tracking_errors: list[TrackingError] = []
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

    def submit_tracking_error(self, error: TrackingError) -> None:
        self.tracking_errors.append(error)

    def invalidate_tracking_error(self) -> None:
        self.invalidations += 1

    def set_control_mode(self, mode: TurretControlMode) -> None:
        self.mode_requests.append(mode)

    def stop_motion(self) -> None:
        self.stop_requests += 1

    def motor_on(self) -> None:
        self.motor_on_requests += 1

    def motor_off(self) -> None:
        self.motor_off_requests += 1

    def request_emergency(self) -> None:
        self.emergency_requests += 1


class _CountingMediator(Mediator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.vision_accept_calls = 0
        self.turret_accept_calls = 0

    def accept_vision_result(self, result: VisionResult) -> bool:
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
) -> tuple[Mediator, _FakeTurret]:
    turret = _FakeTurret(_turret_state(mode=mode, motor=motor))
    mediator_type = _CountingMediator if counting else Mediator
    mediator = mediator_type(
        aiming_config=AimingConfig(
            lead_time_ms=100,
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(
                overview=AimPointConfig(50, 25),
                stereo_left=AimPointConfig(50, 25),
            ),
        ),
        ui_config=UiConfig(CameraRole.OVERVIEW, False, False),
        turret=turret,
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
            LatestValue(),
            LatestValue(),
        )
        for camera in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT)
    }


def _window(
    qt_application: QApplication,
    *,
    mode: TurretControlMode = TurretControlMode.RELATIVE,
    motor: MotorState = MotorState.ON,
    clock: _Clock | None = None,
    frame_preparer=prepare_vision_frame,
    camera_stale_timeout_ms: int = 500,
) -> tuple[
    MainWindow,
    Mediator,
    _FakeTurret,
    dict[CameraRole, CameraUiBinding],
    LatestValue[TurretState],
]:
    mediator, turret = _mediator(mode=mode, motor=motor)
    bindings = _bindings()
    turret_states: LatestValue[TurretState] = LatestValue()
    window = MainWindow(
        mediator=mediator,
        camera_bindings=bindings,
        turret_states=turret_states,
        camera_stale_timeout_ms=camera_stale_timeout_ms,
        clock_ns=clock or _Clock(),
        frame_preparer=frame_preparer,
        start_timer=False,
        start_fullscreen=False,
    )
    window.resize(900, 650)
    window.show()
    qt_application.processEvents()
    return window, mediator, turret, bindings, turret_states


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


def test_tracking_hit_test_uses_displayed_bbox_and_deterministic_tie(
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
    assert mediator.selected_target == TargetRef(CameraRole.OVERVIEW, 1, 3)

    window._handle_main_click(displayed, 5.0, 5.0)
    assert mediator.selected_target is None
    window.close()


def test_tracking_overlap_chooses_closest_center() -> None:
    objects = (
        TrackedObject(8, BBox(10, 10, 40, 30), 0.0, 0.0, 1),
        TrackedObject(4, BBox(25, 10, 40, 30), 0.0, 0.0, 1),
    )
    result = _result(CameraRole.OVERVIEW, 1, objects=objects)
    assert MainWindow._hit_test(result, 45.0, 25.0) == 4
    assert MainWindow._hit_test(result, 90.0, 25.0) is None


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
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert mediator.main_camera is CameraRole.OVERVIEW
    assert turret.relative_commands == []
    window.close()


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
    selected = mediator.selected_target
    assert selected is not None

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
    selected = mediator.selected_target
    assert selected is not None
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
    assert view.rendered_rect().isEmpty()
    status = _status(CameraRole.OVERVIEW, 1, state=CameraState.STARTING)
    view.set_camera_status(status)
    assert view.camera_status is status
    view.close()


def test_fullscreen_escape_leaves_window_running(qt_application: QApplication) -> None:
    window, _mediator_value, _turret, _bindings_value, _states = _window(qt_application)
    window.showFullScreen()
    qt_application.processEvents()
    assert window.isFullScreen()

    QTest.keyClick(window, Qt.Key.Key_Escape)
    qt_application.processEvents()
    assert not window.isFullScreen()
    assert window.isVisible()

    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.isVisible()
    window.close()


def test_main_window_can_request_fullscreen_at_startup(qt_application: QApplication) -> None:
    mediator, _turret = _mediator()
    window = MainWindow(
        mediator=mediator,
        camera_bindings=_bindings(),
        turret_states=LatestValue(),
        camera_stale_timeout_ms=500,
        start_timer=False,
        start_fullscreen=True,
    )
    qt_application.processEvents()
    assert window.isFullScreen()
    assert window.state_pump.timer.interval() == 16
    window.close()
