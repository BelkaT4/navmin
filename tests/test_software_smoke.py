from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from time import monotonic, monotonic_ns, sleep

import numpy as np

from navmin.contracts import (
    CameraRole,
    CameraState,
    TurretConnectionState,
    TurretControlMode,
)
from navmin.turret.protocol import CommandCode, MoveRelativePayload
from navmin.turret.simulator import FakeStm32Endpoint, FakeTransport
from navmin.vision.pipeline import InMemoryFrameSource

_SMOKE_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_software_smoke.py"
_SMOKE = runpy.run_path(str(_SMOKE_TOOL), run_name="navmin_software_smoke")
FRAME_HEIGHT = _SMOKE["FRAME_HEIGHT"]
FRAME_WIDTH = _SMOKE["FRAME_WIDTH"]
SoftwareSmokeRuntime = _SMOKE["SoftwareSmokeRuntime"]
SyntheticFrameProducer = _SMOKE["SyntheticFrameProducer"]


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
