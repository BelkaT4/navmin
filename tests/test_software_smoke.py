from __future__ import annotations

import os
import runpy
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from time import monotonic, monotonic_ns, sleep

import numpy as np

from navmin.contracts import (
    CameraRole,
    CameraState,
    FramePacket,
    MotorState,
    TurretConnectionState,
    TurretControlMode,
)
from navmin.turret.protocol import (
    CommandCode,
    MoveRelativePayload,
    SetVelocityPayload,
)
from navmin.turret.simulator import (
    FakeReadFailure,
    FakeStm32Endpoint,
    FakeTransport,
)
from navmin.turret.transport import TransportDisconnectedError
from navmin.vision.pipeline import (
    InMemoryFrameSource,
    overview_corrector,
    stereo_left_corrector,
)
from navmin.vision.processors.legacy_14.processor import Legacy14VisionProcessor

_SMOKE_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_software_smoke.py"
_SMOKE = runpy.run_path(str(_SMOKE_TOOL), run_name="navmin_software_smoke")
FRAME_HEIGHT = _SMOKE["FRAME_HEIGHT"]
FRAME_WIDTH = _SMOKE["FRAME_WIDTH"]
SoftwareSmokeRuntime = _SMOKE["SoftwareSmokeRuntime"]
SyntheticFrameProducer = _SMOKE["SyntheticFrameProducer"]
_overview_frame = _SMOKE["_overview_frame"]
_stereo_left_frame = _SMOKE["_stereo_left_frame"]
_synthetic_target_center = _SMOKE["_synthetic_target_center"]
synthetic_overview_calibration = _SMOKE["synthetic_overview_calibration"]
synthetic_stereo_calibration = _SMOKE["synthetic_stereo_calibration"]


def _wait_for_frame(source: InMemoryFrameSource, timeout: float = 1.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        frame = source.read()
        if frame is not None:
            return frame
        sleep(0.005)
    raise AssertionError("timed out waiting for synthetic frame")


def test_synthetic_camera_producers_are_independent_and_stop_bounded() -> None:
    overview_source = InMemoryFrameSource()
    stereo_source = InMemoryFrameSource()
    overview = SyntheticFrameProducer(
        camera=CameraRole.OVERVIEW,
        source=overview_source,
    )
    stereo = SyntheticFrameProducer(
        camera=CameraRole.STEREO_LEFT,
        source=stereo_source,
    )

    overview.start()
    stereo.start()
    try:
        overview_frame = _wait_for_frame(overview_source)
        stereo_frame = _wait_for_frame(stereo_source)

        assert overview_frame.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        assert stereo_frame.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        assert overview_frame.image.dtype == np.uint8
        assert stereo_frame.image.dtype == np.uint8
        assert overview_frame.image is not stereo_frame.image
        assert not np.array_equal(overview_frame.image, stereo_frame.image)
        assert overview.frames_produced >= 1
        assert stereo.frames_produced >= 1
    finally:
        assert overview.stop(timeout=1.0)
        assert stereo.stop(timeout=1.0)

    assert not overview.is_alive()
    assert not stereo.is_alive()


def test_synthetic_scenes_produce_one_stable_real_track_per_camera() -> None:
    cases = (
        (
            CameraRole.OVERVIEW,
            _overview_frame,
            overview_corrector(synthetic_overview_calibration()),
            20.0,
        ),
        (
            CameraRole.STEREO_LEFT,
            _stereo_left_frame,
            stereo_left_corrector(synthetic_stereo_calibration()),
            15.0,
        ),
    )

    for camera, make_frame, corrector, fps in cases:
        processor = Legacy14VisionProcessor()
        tracked_by_frame = []
        for frame_id in range(240):
            image = corrector.correct(make_frame(frame_id))
            result = processor.process(
                FramePacket(
                    camera=camera,
                    generation=1,
                    frame_id=frame_id,
                    capture_id=None,
                    receive_timestamp_ns=round(frame_id * 1_000_000_000 / fps),
                    image=image,
                )
            )
            tracked_by_frame.append(result.tracked_objects)

        settled_tracks = tracked_by_frame[12:]
        assert all(len(tracked) == 1 for tracked in settled_tracks)
        assert len({tracked[0].track_id for tracked in settled_tracks}) == 1
        assert all(
            current[0].age_frames > previous[0].age_frames
            for previous, current in pairwise(settled_tracks)
        )

        stable_window = tracked_by_frame[12:32]
        assert len(stable_window) == 20
        assert all(len(tracked) == 1 for tracked in stable_window)
        stable_tracks = [tracked[0] for tracked in stable_window]
        assert len({tracked.track_id for tracked in stable_tracks}) == 1
        assert all(
            current.age_frames > previous.age_frames
            for previous, current in pairwise(stable_tracks)
        )
        assert all(abs(tracked.velocity_x_px_s) > 5.0 for tracked in stable_tracks)

        for frame_id, (tracked,) in enumerate(settled_tracks, start=12):
            target_x, target_y = _synthetic_target_center(camera, frame_id)
            bbox = tracked.bbox
            assert bbox.x - 8 <= target_x <= bbox.x + bbox.width + 8
            assert bbox.y - 8 <= target_y <= bbox.y + bbox.height + 8


def _wait_until(predicate, *, timeout: float = 3.0, message: str) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError(message)


def test_runtime_starts_real_vision_and_turret_and_shuts_down_cleanly() -> None:
    runtime = SoftwareSmokeRuntime()
    runtime.start()
    try:
        _wait_until(
            lambda: runtime.turret_worker.current_state.connection_state
            is TurretConnectionState.READY,
            message="TurretWorker did not become READY",
        )
        _wait_until(
            lambda: (
                runtime.overview_pipeline.generation >= 1
                and runtime.overview_pipeline.status.get() is not None
                and runtime.overview_pipeline.status.get().state is CameraState.ONLINE
                and runtime.overview_pipeline.latest_result.get() is not None
            ),
            message="Overview did not publish an ONLINE VisionResult",
        )
        _wait_until(
            lambda: (
                runtime.stereo_left_pipeline.generation >= 1
                and runtime.stereo_left_pipeline.status.get() is not None
                and runtime.stereo_left_pipeline.status.get().state is CameraState.ONLINE
                and runtime.stereo_left_pipeline.latest_result.get() is not None
            ),
            message="Stereo Left did not publish an ONLINE VisionResult",
        )
    finally:
        runtime.shutdown(timeout=1.0)

    assert not runtime.overview_producer.is_alive()
    assert not runtime.stereo_left_producer.is_alive()
    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()


def test_offscreen_ui_process_smoke_exits_cleanly() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    python_path = [str(root / "src"), str(root)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_software_smoke.py",
            "--auto-close-seconds",
            "1.0",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=12.0,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "Traceback (most recent call last)" not in output
    assert "software smoke shutdown complete" in output


def test_relative_ui_click_reaches_observable_fake_stm32_for_both_cameras() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from navmin.ui import MainWindow

    endpoint = FakeStm32Endpoint()

    def transport_factory(
        _port: str,
        baudrate: int,
        _emulate_stm32: bool,
    ) -> FakeTransport:
        return FakeTransport(endpoint, baudrate=baudrate)

    runtime = SoftwareSmokeRuntime(turret_transport_factory=transport_factory)
    application = QApplication.instance() or QApplication([])
    window = None

    def move_requests():
        return [
            request
            for request in list(endpoint.executed_request_history)
            if request.command is CommandCode.MOVE_RELATIVE
        ]

    try:
        runtime.start()
        window = MainWindow(
            mediator=runtime.mediator,
            camera_bindings=runtime.camera_bindings(),
            turret_states=runtime.turret_worker.state_updates,
            camera_stale_timeout_ms=500,
            start_timer=False,
            start_fullscreen=False,
        )
        window.resize(900, 650)
        window.show()
        application.processEvents()

        def pump_until(predicate, *, timeout: float, message: str) -> None:
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                window.state_pump.pump_once()
                application.processEvents()
                if predicate():
                    return
                sleep(0.005)
            raise AssertionError(message)

        def point_in_rendered_frame(x_fraction: float, y_fraction: float) -> QPoint:
            rendered = window.main_view.rendered_rect()
            assert not rendered.isEmpty()
            return QPoint(
                round(rendered.left() + rendered.width() * x_fraction),
                round(rendered.top() + rendered.height() * y_fraction),
            )

        pump_until(
            lambda: (
                runtime.mediator.turret_state.connection_state
                is TurretConnectionState.READY
                and runtime.mediator.turret_state.control_mode
                is TurretControlMode.RELATIVE
                and runtime.mediator.session_gate.accepted_generation(
                    CameraRole.OVERVIEW
                )
                is not None
                and runtime.mediator.session_gate.accepted_generation(
                    CameraRole.STEREO_LEFT
                )
                is not None
                and window.main_view.camera is CameraRole.OVERVIEW
                and window.main_view.displayed_result is not None
                and window.preview_view.camera is CameraRole.STEREO_LEFT
                and window.preview_view.displayed_result is not None
                and window.main_view.interaction_allowed(
                    runtime.mediator.session_gate,
                    monotonic_ns(),
                )
                and window.preview_view.interaction_allowed(
                    runtime.mediator.session_gate,
                    monotonic_ns(),
                )
            ),
            timeout=3.0,
            message="UI did not accept READY state and both live camera sessions",
        )

        assert runtime.mediator.selected_target is None
        assert move_requests() == []

        overview_click = point_in_rendered_frame(0.75, 0.25)
        QTest.mouseClick(
            window.main_view,
            Qt.MouseButton.LeftButton,
            pos=overview_click,
        )
        pump_until(
            lambda: len(move_requests()) == 1,
            timeout=1.0,
            message="Overview RELATIVE click did not reach fake STM32",
        )

        first_payload = move_requests()[0].payload
        assert isinstance(first_payload, MoveRelativePayload)
        assert first_payload.delta_x_steps > 0
        assert first_payload.delta_y_steps > 0
        assert runtime.mediator.selected_target is None

        move_count_before_swap = len(move_requests())
        QTest.mouseClick(
            window.preview_view,
            Qt.MouseButton.LeftButton,
            pos=window.preview_view.rect().center(),
        )
        application.processEvents()
        assert runtime.mediator.main_camera is CameraRole.STEREO_LEFT
        assert window.main_view.camera is CameraRole.STEREO_LEFT
        assert runtime.mediator.selected_target is None

        no_motion_deadline = monotonic() + 0.15
        while monotonic() < no_motion_deadline:
            window.state_pump.pump_once()
            application.processEvents()
            assert len(move_requests()) == move_count_before_swap
            sleep(0.005)

        pump_until(
            lambda: (
                window.main_view.camera is CameraRole.STEREO_LEFT
                and window.main_view.displayed_result is not None
                and window.main_view.interaction_allowed(
                    runtime.mediator.session_gate,
                    monotonic_ns(),
                )
            ),
            timeout=1.0,
            message="Stereo Left did not become the live main view after preview swap",
        )

        stereo_left_click = point_in_rendered_frame(0.25, 0.75)
        QTest.mouseClick(
            window.main_view,
            Qt.MouseButton.LeftButton,
            pos=stereo_left_click,
        )
        pump_until(
            lambda: len(move_requests()) == 2,
            timeout=1.0,
            message="Stereo Left RELATIVE click did not reach fake STM32",
        )

        second_payload = move_requests()[1].payload
        assert isinstance(second_payload, MoveRelativePayload)
        assert second_payload.delta_x_steps < 0
        assert second_payload.delta_y_steps < 0
        assert runtime.mediator.selected_target is None
        assert len(move_requests()) == 2
    finally:
        if window is not None:
            window.state_pump.stop()
            window.close()
            application.processEvents()
        runtime.shutdown(timeout=1.0)

    assert not runtime.overview_producer.is_alive()
    assert not runtime.stereo_left_producer.is_alive()
    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()


def test_tracking_ui_selection_drives_real_velocity_loop_to_fake_stm32() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from navmin.ui import MainWindow

    endpoint = FakeStm32Endpoint()

    def transport_factory(
        _port: str,
        baudrate: int,
        _emulate_stm32: bool,
    ) -> FakeTransport:
        return FakeTransport(endpoint, baudrate=baudrate)

    runtime = SoftwareSmokeRuntime(turret_transport_factory=transport_factory)
    application = QApplication.instance() or QApplication([])
    window = None

    def executed(command: CommandCode):
        return [
            request
            for request in list(endpoint.executed_request_history)
            if request.command is command
        ]

    try:
        runtime.start()
        window = MainWindow(
            mediator=runtime.mediator,
            camera_bindings=runtime.camera_bindings(),
            turret_states=runtime.turret_worker.state_updates,
            camera_stale_timeout_ms=500,
            start_timer=False,
            start_fullscreen=False,
        )
        window.resize(900, 650)
        window.show()
        application.processEvents()

        def pump_until(predicate, *, timeout: float, message: str) -> None:
            deadline = monotonic() + timeout
            while monotonic() < deadline:
                window.state_pump.pump_once()
                application.processEvents()
                if predicate():
                    return
                sleep(0.005)
            raise AssertionError(message)

        def stable_overview_track():
            result = window.main_view.displayed_result
            if result is None or len(result.tracked_objects) != 1:
                return None
            tracked = result.tracked_objects[0]
            center_x = tracked.bbox.x + tracked.bbox.width / 2.0
            center_y = tracked.bbox.y + tracked.bbox.height / 2.0
            if (
                abs(tracked.velocity_x_px_s) <= 5.0
                or center_x <= FRAME_WIDTH / 2.0
                or center_y >= FRAME_HEIGHT / 2.0
            ):
                return None
            return tracked

        pump_until(
            lambda: (
                runtime.mediator.turret_state.connection_state
                is TurretConnectionState.READY
                and runtime.mediator.turret_state.motor_state is MotorState.OFF
                and runtime.mediator.session_gate.accepted_generation(
                    CameraRole.OVERVIEW
                )
                is not None
                and runtime.mediator.session_gate.accepted_generation(
                    CameraRole.STEREO_LEFT
                )
                is not None
                and window.main_view.camera is CameraRole.OVERVIEW
                and stable_overview_track() is not None
                and window.main_view.interaction_allowed(
                    runtime.mediator.session_gate,
                    monotonic_ns(),
                )
            ),
            timeout=4.0,
            message="UI did not accept a stable real Overview track",
        )

        QTest.mouseClick(window.mode_button, Qt.MouseButton.LeftButton)
        pump_until(
            lambda: (
                runtime.mediator.turret_state.control_mode
                is TurretControlMode.TRACKING
                and runtime.mediator.pending_control_mode is None
                and "TRACKING" in window.mode_button.text()
                and "→" not in window.mode_button.text()
            ),
            timeout=1.0,
            message="real mode button did not confirm TRACKING",
        )

        QTest.mouseClick(window.motor_button, Qt.MouseButton.LeftButton)
        pump_until(
            lambda: (
                runtime.mediator.turret_state.motor_state is MotorState.ON
                and "ON" in window.motor_button.text()
                and len(executed(CommandCode.MOTOR_ON)) >= 1
            ),
            timeout=1.0,
            message="real motor button did not confirm MOTOR_ON",
        )

        displayed = window.main_view.displayed_result
        tracked = stable_overview_track()
        assert displayed is not None
        assert tracked is not None
        bbox_center = (
            tracked.bbox.x + tracked.bbox.width / 2.0,
            tracked.bbox.y + tracked.bbox.height / 2.0,
        )
        assert runtime.mediator.aiming.config.lead_time_ms > 0
        lead_point = runtime.mediator.aiming.lead_point(tracked)
        assert lead_point[0] != bbox_center[0]

        rendered = window.main_view.rendered_rect()
        assert not rendered.isEmpty()
        height, width = displayed.frame.image.shape[:2]
        click = QPoint(
            round(rendered.left() + bbox_center[0] * rendered.width() / width),
            round(rendered.top() + bbox_center[1] * rendered.height() / height),
        )
        velocity_baseline = len(executed(CommandCode.SET_VELOCITY))
        relative_baseline = len(executed(CommandCode.MOVE_RELATIVE))

        QTest.mouseClick(
            window.main_view,
            Qt.MouseButton.LeftButton,
            pos=click,
        )
        application.processEvents()
        selected = runtime.mediator.selected_target
        assert selected is not None
        assert selected.camera is CameraRole.OVERVIEW
        assert selected.generation == displayed.frame.generation
        assert selected.track_id == tracked.track_id

        pump_until(
            lambda: (
                runtime.mediator.selected_target == selected
                and len(executed(CommandCode.SET_VELOCITY))
                >= velocity_baseline + 2
            ),
            timeout=2.0,
            message="ongoing tracking did not produce multiple SET_VELOCITY requests",
        )

        tracking_requests = executed(CommandCode.SET_VELOCITY)[velocity_baseline:]
        assert len(tracking_requests) >= 2
        tracking_payloads = [request.payload for request in tracking_requests]
        assert all(isinstance(payload, SetVelocityPayload) for payload in tracking_payloads)
        nonzero_payloads = [
            payload
            for payload in tracking_payloads
            if isinstance(payload, SetVelocityPayload)
            and (payload.velocity_x_steps_s or payload.velocity_y_steps_s)
        ]
        assert nonzero_payloads
        assert all(payload.velocity_x_steps_s > 0 for payload in nonzero_payloads)
        assert all(payload.velocity_y_steps_s > 0 for payload in nonzero_payloads)
        assert len(executed(CommandCode.MOVE_RELATIVE)) == relative_baseline

        next_right_edge_frame = 45 + 90 * max(
            0,
            (displayed.frame.frame_id - 45) // 90 + 1,
        )
        reversal_deadline = monotonic() + 6.0
        last_checked_frame_id = displayed.frame.frame_id
        while monotonic() < reversal_deadline:
            window.state_pump.pump_once()
            application.processEvents()
            assert runtime.mediator.selected_target == selected
            current = window.main_view.displayed_result
            if current is None or current.frame.frame_id <= last_checked_frame_id:
                sleep(0.005)
                continue
            assert any(
                item.track_id == selected.track_id
                for item in current.tracked_objects
            ), (
                f"selected track {selected.track_id} missing at Overview "
                f"frame {current.frame.frame_id}: {current.tracked_objects!r}"
            )
            last_checked_frame_id = current.frame.frame_id
            if last_checked_frame_id >= next_right_edge_frame + 5:
                break
            sleep(0.005)
        else:
            raise AssertionError(
                "selected Overview track did not survive the right-edge reversal"
            )

        empty_source_point = (20.0, 20.0)
        empty_click = QPoint(
            round(rendered.left() + empty_source_point[0] * rendered.width() / width),
            round(rendered.top() + empty_source_point[1] * rendered.height() / height),
        )
        deselect_velocity_baseline = len(executed(CommandCode.SET_VELOCITY))
        QTest.mouseClick(
            window.main_view,
            Qt.MouseButton.LeftButton,
            pos=empty_click,
        )
        application.processEvents()
        assert runtime.mediator.selected_target is None
        pump_until(
            lambda: any(
                isinstance(request.payload, SetVelocityPayload)
                and request.payload.velocity_x_steps_s == 0
                and request.payload.velocity_y_steps_s == 0
                for request in executed(CommandCode.SET_VELOCITY)[
                    deselect_velocity_baseline:
                ]
            ),
            timeout=1.0,
            message="deselect did not produce the normal zero-velocity boundary",
        )
        assert len(executed(CommandCode.MOVE_RELATIVE)) == relative_baseline
    finally:
        if window is not None:
            window.state_pump.stop()
            window.close()
            application.processEvents()
        runtime.shutdown(timeout=1.0)

    assert not runtime.overview_producer.is_alive()
    assert not runtime.stereo_left_producer.is_alive()
    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()


class _FailOpenTransport(FakeTransport):
    def open(self) -> None:
        raise TransportDisconnectedError("simulated missing serial device")


class _RecordingTransportFactory:
    def __init__(self, endpoint: FakeStm32Endpoint | None = None) -> None:
        self.endpoint = endpoint or FakeStm32Endpoint()
        self.transports: list[FakeTransport] = []
        self.fail_open_count = 0
        self.always_fail = False

    def __call__(
        self,
        _port: str,
        baudrate: int,
        _emulate_stm32: bool,
    ) -> FakeTransport:
        fail_open = self.always_fail or self.fail_open_count > 0
        if self.fail_open_count > 0:
            self.fail_open_count -= 1
        transport_type = _FailOpenTransport if fail_open else FakeTransport
        transport = transport_type(self.endpoint, baudrate=baudrate)
        self.transports.append(transport)
        return transport


def _executed(endpoint: FakeStm32Endpoint, command: CommandCode):
    return [
        request
        for request in list(endpoint.executed_request_history)
        if request.command is command
    ]


def _pump_qt_until(
    window,
    application,
    predicate,
    *,
    timeout: float,
    message: str,
) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        window.state_pump.pump_once()
        application.processEvents()
        if predicate():
            return
        sleep(0.005)
    raise AssertionError(message)


def _pump_qt_for(window, application, duration: float) -> None:
    deadline = monotonic() + duration
    while monotonic() < deadline:
        window.state_pump.pump_once()
        application.processEvents()
        sleep(0.005)


def _stable_overview_track(window):
    result = window.main_view.displayed_result
    if result is None or len(result.tracked_objects) != 1:
        return None
    tracked = result.tracked_objects[0]
    center_x = tracked.bbox.x + tracked.bbox.width / 2.0
    center_y = tracked.bbox.y + tracked.bbox.height / 2.0
    if (
        abs(tracked.velocity_x_px_s) <= 5.0
        or center_x <= FRAME_WIDTH / 2.0
        or center_y >= FRAME_HEIGHT / 2.0
    ):
        return None
    return tracked


def _activate_real_overview_tracking(runtime, window, application, endpoint):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    _pump_qt_until(
        window,
        application,
        lambda: (
            runtime.mediator.turret_state.connection_state
            is TurretConnectionState.READY
            and runtime.mediator.session_gate.accepted_generation(
                CameraRole.OVERVIEW
            )
            is not None
            and window.main_view.camera is CameraRole.OVERVIEW
            and _stable_overview_track(window) is not None
            and window.main_view.interaction_allowed(
                runtime.mediator.session_gate,
                monotonic_ns(),
            )
        ),
        timeout=4.0,
        message="UI did not reach a stable live Overview tracking fixture",
    )

    if runtime.mediator.turret_state.control_mode is TurretControlMode.RELATIVE:
        QTest.mouseClick(window.mode_button, Qt.MouseButton.LeftButton)
        _pump_qt_until(
            window,
            application,
            lambda: (
                runtime.mediator.turret_state.control_mode
                is TurretControlMode.TRACKING
                and runtime.mediator.pending_control_mode is None
            ),
            timeout=1.0,
            message="real mode button did not confirm TRACKING",
        )

    if runtime.mediator.turret_state.motor_state is MotorState.OFF:
        QTest.mouseClick(window.motor_button, Qt.MouseButton.LeftButton)
        _pump_qt_until(
            window,
            application,
            lambda: runtime.mediator.turret_state.motor_state is MotorState.ON,
            timeout=1.0,
            message="real motor button did not confirm MOTOR_ON",
        )

    displayed = window.main_view.displayed_result
    tracked = _stable_overview_track(window)
    assert displayed is not None
    assert tracked is not None
    rendered = window.main_view.rendered_rect()
    assert not rendered.isEmpty()
    height, width = displayed.frame.image.shape[:2]
    center_x = tracked.bbox.x + tracked.bbox.width / 2.0
    center_y = tracked.bbox.y + tracked.bbox.height / 2.0
    click = QPoint(
        round(rendered.left() + center_x * rendered.width() / width),
        round(rendered.top() + center_y * rendered.height() / height),
    )
    velocity_baseline = len(_executed(endpoint, CommandCode.SET_VELOCITY))
    QTest.mouseClick(window.main_view, Qt.MouseButton.LeftButton, pos=click)
    application.processEvents()
    selected = runtime.mediator.selected_target
    assert selected is not None
    _pump_qt_until(
        window,
        application,
        lambda: any(
            isinstance(request.payload, SetVelocityPayload)
            and (request.payload.velocity_x_steps_s or request.payload.velocity_y_steps_s)
            for request in _executed(endpoint, CommandCode.SET_VELOCITY)[
                velocity_baseline:
            ]
        ),
        timeout=2.0,
        message="active TRACKING did not reach a nonzero SET_VELOCITY",
    )
    return selected


def _assert_runtime_threads_stopped(runtime: SoftwareSmokeRuntime) -> None:
    assert not runtime.overview_producer.is_alive()
    assert not runtime.stereo_left_producer.is_alive()
    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()


def test_camera_stale_resume_keeps_generation_and_recovers_interaction() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtWidgets import QApplication

    from navmin.ui import MainWindow

    runtime = SoftwareSmokeRuntime()
    application = QApplication.instance() or QApplication([])
    window = None
    try:
        runtime.start()
        window = MainWindow(
            mediator=runtime.mediator,
            camera_bindings=runtime.camera_bindings(),
            turret_states=runtime.turret_worker.state_updates,
            camera_stale_timeout_ms=500,
            start_timer=False,
            start_fullscreen=False,
        )
        window.resize(900, 650)
        window.show()
        application.processEvents()

        _pump_qt_until(
            window,
            application,
            lambda: (
                window.main_view.camera is CameraRole.OVERVIEW
                and window.main_view.displayed_result is not None
                and window.preview_view.displayed_result is not None
                and window.main_view.interaction_allowed(
                    runtime.mediator.session_gate,
                    monotonic_ns(),
                )
            ),
            timeout=3.0,
            message="both cameras did not become fresh before stale test",
        )
        generation_before = window.main_view.displayed_result.frame.generation
        overview_frame_before = window.main_view.displayed_result.frame.frame_id
        stereo_frame_before = window.preview_view.displayed_result.frame.frame_id

        runtime.overview_producer.pause()
        assert runtime.overview_producer.is_alive()
        assert runtime.overview_producer.is_paused
        _pump_qt_until(
            window,
            application,
            lambda: window.main_view.is_stale,
            timeout=1.5,
            message="Overview presentation did not become stale",
        )

        frozen = window.main_view.displayed_result
        status = window.main_view.camera_status
        assert frozen is not None
        assert frozen.frame.frame_id >= overview_frame_before
        assert frozen.frame.generation == generation_before
        assert status is not None
        assert status.state is CameraState.ONLINE
        assert status.generation == generation_before
        assert window.main_view.stale_overlay_text == "НЕТ НОВЫХ КАДРОВ"
        assert not window.main_view.interaction_allowed(
            runtime.mediator.session_gate,
            monotonic_ns(),
        )
        assert window.preview_view.displayed_result is not None
        assert window.preview_view.displayed_result.frame.frame_id > stereo_frame_before
        assert runtime.stereo_left_producer.is_alive()

        _pump_qt_for(window, application, 0.1)
        assert window.main_view.displayed_result is frozen
        assert runtime.overview_pipeline.generation == generation_before

        runtime.overview_producer.resume()
        assert not runtime.overview_producer.is_paused
        _pump_qt_until(
            window,
            application,
            lambda: (
                window.main_view.displayed_result is not None
                and window.main_view.displayed_result.frame.frame_id
                > frozen.frame.frame_id
                and not window.main_view.is_stale
                and window.main_view.interaction_allowed(
                    runtime.mediator.session_gate,
                    monotonic_ns(),
                )
            ),
            timeout=1.5,
            message="Overview did not recover after producer resume",
        )
        assert window.main_view.displayed_result.frame.generation == generation_before
        assert runtime.overview_pipeline.generation == generation_before
    finally:
        if window is not None:
            window.state_pump.stop()
            window.close()
            application.processEvents()
        runtime.shutdown(timeout=1.0)

    _assert_runtime_threads_stopped(runtime)


def test_generation_restart_and_preview_swap_clear_active_tracking() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from navmin.ui import MainWindow

    factory = _RecordingTransportFactory()
    runtime = SoftwareSmokeRuntime(turret_transport_factory=factory)
    application = QApplication.instance() or QApplication([])
    window = None
    try:
        runtime.start()
        window = MainWindow(
            mediator=runtime.mediator,
            camera_bindings=runtime.camera_bindings(),
            turret_states=runtime.turret_worker.state_updates,
            camera_stale_timeout_ms=500,
            start_timer=False,
            start_fullscreen=False,
        )
        window.resize(900, 650)
        window.show()
        application.processEvents()

        selected = _activate_real_overview_tracking(
            runtime, window, application, factory.endpoint
        )
        old_generation = selected.generation
        zero_boundary = len(_executed(factory.endpoint, CommandCode.SET_VELOCITY))

        runtime.overview_producer.pause()
        _pump_qt_for(window, application, 0.1)
        session = runtime.overview_pipeline.start()
        assert session.generation == old_generation + 1
        assert runtime.overview_pipeline.generation == old_generation + 1

        window.state_pump.pump_once()
        application.processEvents()
        assert runtime.mediator.session_gate.accepted_generation(
            CameraRole.OVERVIEW
        ) == old_generation + 1
        assert window.main_view.displayed_result is None
        assert runtime.mediator.selected_target is None

        _pump_qt_until(
            window,
            application,
            lambda: any(
                isinstance(request.payload, SetVelocityPayload)
                and request.payload.velocity_x_steps_s == 0
                and request.payload.velocity_y_steps_s == 0
                for request in _executed(factory.endpoint, CommandCode.SET_VELOCITY)[
                    zero_boundary:
                ]
            ),
            timeout=1.0,
            message="generation boundary did not drive the normal tracking stop",
        )

        runtime.overview_producer.resume()
        _pump_qt_until(
            window,
            application,
            lambda: (
                window.main_view.displayed_result is not None
                and window.main_view.displayed_result.frame.generation
                == old_generation + 1
            ),
            timeout=1.5,
            message="new Overview generation was not displayed",
        )
        _pump_qt_for(window, application, 0.1)
        assert window.main_view.displayed_result is not None
        assert window.main_view.displayed_result.frame.generation == old_generation + 1
        assert runtime.mediator.selected_target is None

        _activate_real_overview_tracking(runtime, window, application, factory.endpoint)
        assert runtime.mediator.selected_target is not None
        relative_before_swap = len(_executed(factory.endpoint, CommandCode.MOVE_RELATIVE))
        zero_boundary = len(_executed(factory.endpoint, CommandCode.SET_VELOCITY))

        QTest.mouseClick(
            window.preview_view,
            Qt.MouseButton.LeftButton,
            pos=window.preview_view.rect().center(),
        )
        application.processEvents()
        assert runtime.mediator.main_camera is CameraRole.STEREO_LEFT
        assert runtime.mediator.selected_target is None
        assert len(_executed(factory.endpoint, CommandCode.MOVE_RELATIVE)) == relative_before_swap
        _pump_qt_until(
            window,
            application,
            lambda: any(
                isinstance(request.payload, SetVelocityPayload)
                and request.payload.velocity_x_steps_s == 0
                and request.payload.velocity_y_steps_s == 0
                for request in _executed(factory.endpoint, CommandCode.SET_VELOCITY)[
                    zero_boundary:
                ]
            ),
            timeout=1.0,
            message="preview swap did not drive the normal tracking stop",
        )
    finally:
        runtime.overview_producer.resume()
        if window is not None:
            window.state_pump.stop()
            window.close()
            application.processEvents()
        runtime.shutdown(timeout=1.0)

    _assert_runtime_threads_stopped(runtime)


def test_turret_disconnect_clears_tracking_recovers_off_and_remains_usable() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from navmin.ui import MainWindow

    factory = _RecordingTransportFactory()
    runtime = SoftwareSmokeRuntime(turret_transport_factory=factory)
    application = QApplication.instance() or QApplication([])
    window = None
    try:
        runtime.start()
        window = MainWindow(
            mediator=runtime.mediator,
            camera_bindings=runtime.camera_bindings(),
            turret_states=runtime.turret_worker.state_updates,
            camera_stale_timeout_ms=500,
            start_timer=False,
            start_fullscreen=False,
        )
        window.resize(900, 650)
        window.show()
        application.processEvents()

        _activate_real_overview_tracking(runtime, window, application, factory.endpoint)
        assert runtime.mediator.selected_target is not None
        overview_before = runtime.overview_pipeline.latest_result.get().frame.frame_id
        stereo_before = runtime.stereo_left_pipeline.latest_result.get().frame.frame_id

        factory.fail_open_count = 1
        active_transport = factory.transports[-1]
        active_transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        _pump_qt_until(
            window,
            application,
            lambda: runtime.mediator.turret_state.connection_state
            is not TurretConnectionState.READY,
            timeout=1.0,
            message="Turret disconnect did not reach the UI/Core state pump",
        )
        lost_state = runtime.mediator.turret_state
        assert lost_state.connection_state is TurretConnectionState.CONNECTING
        assert lost_state.motor_state is MotorState.UNKNOWN
        assert runtime.mediator.selected_target is None
        assert runtime.mediator.pending_control_mode is None
        assert window.isVisible()

        _pump_qt_until(
            window,
            application,
            lambda: (
                runtime.overview_pipeline.latest_result.get() is not None
                and runtime.overview_pipeline.latest_result.get().frame.frame_id
                > overview_before
                and runtime.stereo_left_pipeline.latest_result.get() is not None
                and runtime.stereo_left_pipeline.latest_result.get().frame.frame_id
                > stereo_before
            ),
            timeout=1.0,
            message="camera streams did not continue during Turret recovery",
        )
        assert runtime.overview_pipeline.status.get().state is CameraState.ONLINE
        assert runtime.stereo_left_pipeline.status.get().state is CameraState.ONLINE

        _pump_qt_until(
            window,
            application,
            lambda: (
                runtime.mediator.turret_state.connection_state
                is TurretConnectionState.READY
                and runtime.mediator.turret_state.motor_state is MotorState.OFF
            ),
            timeout=2.0,
            message="Turret did not recover to READY with motors OFF",
        )
        assert len(factory.transports) >= 3
        assert runtime.mediator.selected_target is None

        QTest.mouseClick(window.motor_button, Qt.MouseButton.LeftButton)
        _pump_qt_until(
            window,
            application,
            lambda: runtime.mediator.turret_state.motor_state is MotorState.ON,
            timeout=1.0,
            message="Turret was not usable after reconnect",
        )
    finally:
        if window is not None:
            window.state_pump.stop()
            window.close()
            application.processEvents()
        runtime.shutdown(timeout=1.0)

    _assert_runtime_threads_stopped(runtime)


def test_emergency_click_during_tracking_crosses_full_stack_and_preserves_motor_on() -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from navmin.ui import MainWindow

    factory = _RecordingTransportFactory()
    runtime = SoftwareSmokeRuntime(turret_transport_factory=factory)
    application = QApplication.instance() or QApplication([])
    window = None
    try:
        runtime.start()
        window = MainWindow(
            mediator=runtime.mediator,
            camera_bindings=runtime.camera_bindings(),
            turret_states=runtime.turret_worker.state_updates,
            camera_stale_timeout_ms=500,
            start_timer=False,
            start_fullscreen=False,
        )
        window.resize(900, 650)
        window.show()
        application.processEvents()

        _activate_real_overview_tracking(runtime, window, application, factory.endpoint)
        assert runtime.mediator.selected_target is not None
        history_boundary = len(factory.endpoint.executed_request_history)

        QTest.mouseClick(window.emergency_button, Qt.MouseButton.LeftButton)
        application.processEvents()
        assert runtime.mediator.selected_target is None

        _pump_qt_until(
            window,
            application,
            lambda: any(
                request.command is CommandCode.EMERGENCY_STOP
                for request in list(factory.endpoint.executed_request_history)[
                    history_boundary:
                ]
            ),
            timeout=1.0,
            message="EMERGENCY_STOP did not reach FakeStm32Endpoint",
        )
        _pump_qt_until(
            window,
            application,
            lambda: (
                runtime.mediator.turret_state.connection_state
                is TurretConnectionState.READY
                and runtime.mediator.turret_state.motor_state is MotorState.ON
            ),
            timeout=1.0,
            message="Emergency did not preserve confirmed MotorState.ON while READY",
        )
        assert runtime.mediator.selected_target is None
    finally:
        if window is not None:
            window.state_pump.stop()
            window.close()
            application.processEvents()
        runtime.shutdown(timeout=1.0)

    _assert_runtime_threads_stopped(runtime)


def test_partial_start_cleanup_and_shutdown_during_turret_backoff(monkeypatch) -> None:
    partial = SoftwareSmokeRuntime()

    def fail_second_camera_start() -> None:
        raise RuntimeError("simulated Stereo Left start failure")

    monkeypatch.setattr(partial.stereo_left_worker, "start", fail_second_camera_start)
    start_failed = False
    try:
        partial.start()
    except RuntimeError as exc:
        start_failed = True
        assert "Stereo Left start failure" in str(exc)
    finally:
        shutdown_started = monotonic()
        partial.shutdown(timeout=0.5)
        partial.shutdown(timeout=0.5)
        partial_shutdown_elapsed = monotonic() - shutdown_started

    assert start_failed
    assert partial_shutdown_elapsed < 1.0
    _assert_runtime_threads_stopped(partial)

    factory = _RecordingTransportFactory()
    factory.always_fail = True
    reconnecting = SoftwareSmokeRuntime(turret_transport_factory=factory)
    reconnecting.start()
    try:
        _wait_until(
            lambda: (
                reconnecting.turret_worker.current_state.connection_state
                is TurretConnectionState.CONNECTING
                and reconnecting.overview_pipeline.latest_result.get() is not None
                and reconnecting.stereo_left_pipeline.latest_result.get() is not None
            ),
            timeout=1.5,
            message="runtime did not enter Turret backoff with live cameras",
        )
        shutdown_started = monotonic()
        reconnecting.shutdown(timeout=0.5)
        backoff_shutdown_elapsed = monotonic() - shutdown_started
    finally:
        reconnecting.shutdown(timeout=0.5)

    assert backoff_shutdown_elapsed < 1.0
    assert (
        reconnecting.turret_worker.current_state.connection_state
        is TurretConnectionState.DISCONNECTED
    )
    assert reconnecting.turret_worker.current_state.motor_state is MotorState.UNKNOWN
    _assert_runtime_threads_stopped(reconnecting)
