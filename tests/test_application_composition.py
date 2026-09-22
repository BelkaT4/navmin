from __future__ import annotations

from time import monotonic, sleep

import pytest

from navmin.application import (
    ApplicationFactories,
    ApplicationShutdownError,
    build_application_runtime,
)
from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config.models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    AppConfig,
    AxesConfig,
    AxisMechanicsConfig,
    CameraConfig,
    CamerasConfig,
    PidControllerConfig,
    ProcessingScope,
    SerialConfig,
    StereoDistanceConfig,
    Stm32Config,
    TurretConfig,
    UiConfig,
    VisionConfig,
    VisionDistanceConfig,
)
from navmin.contracts import (
    CameraRole,
    DistanceSource,
    MotorState,
    TurretConnectionState,
)
from navmin.turret.protocol import CommandCode
from navmin.turret.simulator import FakeStm32Endpoint, FakeTransport
from navmin.turret.worker import TurretWorker
from navmin.vision.gstreamer_source import GStreamerRtpJpegSource
from navmin.vision.pipeline import InMemoryFrameSource

WIDTH = 32
HEIGHT = 24
K = ((20.0, 0.0, 15.0), (0.0, 20.0, 11.0), (0.0, 0.0, 1.0))
IDENTITY_3X3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _camera(port: int, *, enabled: bool = True) -> CameraConfig:
    return CameraConfig(
        enabled=enabled,
        address="0.0.0.0",
        port=port,
        rtp_enabled=True,
        buffer_size=1,
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _config() -> AppConfig:
    center = AimPointConfig(x_px=None, y_px=None)
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    return AppConfig(
        schema_version=1,
        vision=VisionConfig(
            processing_scope=ProcessingScope.MAIN_AND_PREVIEW,
            cameras=CamerasConfig(
                overview=_camera(8888),
                stereo_left=_camera(8889),
                stereo_right=_camera(8890),
            ),
            distance=VisionDistanceConfig(
                source=DistanceSource.MANUAL,
                manual_distance_m=100.0,
                distance_stale_timeout_ms=300,
                stereo=StereoDistanceConfig(
                    stereo_enabled=False,
                    right_frame_buffer_size=4,
                    pair_timeout_ms=100,
                ),
            ),
            camera_stale_timeout_ms=500,
            simulation_mode=False,
        ),
        aiming=AimingConfig(
            lead_time_ms=100,
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(overview=center, stereo_left=center),
        ),
        turret=TurretConfig(
            serial=SerialConfig(
                port="/dev/ttyUSB0",
                baudrate=9600,
                response_timeout_ms=25,
                max_retries=1,
                inter_request_delay_ms=1,
            ),
            axes=AxesConfig(x=axis, y=axis),
            controller=PidControllerConfig(
                pid_kp_x=1.0,
                pid_ki_x=0.0,
                pid_kd_x=0.0,
                pid_kp_y=1.0,
                pid_ki_y=0.0,
                pid_kd_y=0.0,
            ),
            stm32=Stm32Config(
                max_speed_x_deg_s=50.0,
                max_speed_y_deg_s=50.0,
                acceleration_x_deg_s2=100.0,
                acceleration_y_deg_s2=100.0,
                velocity_watchdog_timeout_ms=200,
            ),
            emulate_stm32=False,
        ),
        ui=UiConfig(
            default_camera=CameraRole.OVERVIEW,
            show_fps=False,
            show_stereo_right_diagnostics=False,
        ),
    )


def _overview_calibration() -> OverviewCalibration:
    return OverviewCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
        K=K,
        D=(0.0, 0.0, 0.0, 0.0),
        new_camera_matrix=K,
    )


def _stereo_calibration() -> StereoCalibration:
    projection = (
        (20.0, 0.0, 15.0, 0.0),
        (0.0, 20.0, 11.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    return StereoCalibration(
        schema_version=1,
        image_width=WIDTH,
        image_height=HEIGHT,
        K_left=K,
        D_left=(0.0, 0.0, 0.0, 0.0, 0.0),
        K_right=K,
        D_right=(0.0, 0.0, 0.0, 0.0, 0.0),
        R=IDENTITY_3X3,
        T=(-0.46, 0.0, 0.0),
        R1=IDENTITY_3X3,
        R2=IDENTITY_3X3,
        P1=projection,
        P2=projection,
        Q=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _injected_runtime():
    sources = {
        CameraRole.OVERVIEW: InMemoryFrameSource(),
        CameraRole.STEREO_LEFT: InMemoryFrameSource(),
    }
    endpoint = FakeStm32Endpoint()

    def overview_factory(_config: CameraConfig) -> InMemoryFrameSource:
        return sources[CameraRole.OVERVIEW]

    def stereo_left_factory(_config: CameraConfig) -> InMemoryFrameSource:
        return sources[CameraRole.STEREO_LEFT]

    def transport_factory(
        _port: str,
        baudrate: int,
        _emulate_stm32: bool,
    ) -> FakeTransport:
        return FakeTransport(endpoint, baudrate=baudrate)

    runtime = build_application_runtime(
        config=_config(),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
        factories=ApplicationFactories(
            overview_source_factory=overview_factory,
            stereo_left_source_factory=stereo_left_factory,
            turret_transport_factory=transport_factory,
        ),
    )
    return runtime, sources, endpoint


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError("condition did not become true")


def test_production_runtime_initializes_gstreamer_before_starting_workers(
    monkeypatch,
) -> None:
    runtime = build_application_runtime(
        config=_config(),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )
    events: list[str] = []

    monkeypatch.setattr(
        "navmin.application.initialize_gstreamer_runtime",
        lambda: events.append("gstreamer"),
    )
    monkeypatch.setattr(runtime.turret_worker, "start", lambda: events.append("turret"))
    monkeypatch.setattr(
        runtime.overview_worker,
        "start",
        lambda: events.append("overview"),
    )
    monkeypatch.setattr(
        runtime.stereo_left_worker,
        "start",
        lambda: events.append("stereo_left"),
    )

    runtime.start()

    assert events == ["gstreamer", "turret", "overview", "stereo_left"]


def test_default_construction_uses_only_production_transport_boundaries() -> None:
    factories = ApplicationFactories()
    runtime = build_application_runtime(
        config=_config(),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )

    assert set(runtime.camera_workers) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    }
    assert isinstance(runtime.overview_worker.source, GStreamerRtpJpegSource)
    assert isinstance(runtime.stereo_left_worker.source, GStreamerRtpJpegSource)
    assert isinstance(runtime.turret_worker, TurretWorker)
    assert factories.turret_transport_factory is None


def test_shared_runtime_starts_and_exposes_complete_ui_dependency_surface() -> None:
    runtime, sources, _endpoint = _injected_runtime()
    runtime.start()
    try:
        _wait_until(
            lambda: runtime.overview_worker.pipeline.generation == 1
            and runtime.stereo_left_worker.pipeline.generation == 1
            and runtime.turret_worker.current_state.connection_state
            is TurretConnectionState.READY
        )

        bindings = runtime.camera_bindings
        assert set(bindings) == {
            CameraRole.OVERVIEW,
            CameraRole.STEREO_LEFT,
        }
        assert runtime.overview_worker.source is sources[CameraRole.OVERVIEW]
        assert runtime.stereo_left_worker.source is sources[CameraRole.STEREO_LEFT]
        assert (
            bindings[CameraRole.OVERVIEW].latest_result
            is runtime.overview_worker.pipeline.latest_result
        )
        assert (
            bindings[CameraRole.STEREO_LEFT].session_barriers
            is runtime.stereo_left_worker.pipeline.session_barriers
        )

        for binding in bindings.values():
            session = binding.session_barriers.receive(timeout=0.5)
            assert runtime.mediator.accept_camera_session(session)
        assert runtime.mediator.session_gate.accepted_generation(
            CameraRole.OVERVIEW
        ) == 1
        assert runtime.mediator.session_gate.accepted_generation(
            CameraRole.STEREO_LEFT
        ) == 1

        ui = runtime.ui_dependencies
        assert ui.mediator is runtime.mediator
        assert ui.camera_bindings is runtime.camera_bindings
        assert ui.turret_states is runtime.turret_worker.state_updates
        assert ui.camera_stale_timeout_ms == 500
    finally:
        runtime.shutdown(timeout=1.0)
        runtime.shutdown(timeout=1.0)

    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()


def test_shutdown_stops_motion_and_confirms_motor_off_before_worker_shutdown(
    monkeypatch,
) -> None:
    runtime, _sources, endpoint = _injected_runtime()
    runtime.start()
    _wait_until(
        lambda: runtime.turret_worker.current_state.connection_state
        is TurretConnectionState.READY
    )
    assert runtime.turret_worker.motor_on()
    _wait_until(
        lambda: runtime.turret_worker.current_state.motor_state is MotorState.ON
    )

    original_shutdown = runtime.turret_worker.shutdown
    motor_state_at_worker_shutdown: list[MotorState] = []

    def checking_shutdown(timeout: float) -> None:
        motor_state_at_worker_shutdown.append(runtime.turret_worker.current_state.motor_state)
        original_shutdown(timeout)

    monkeypatch.setattr(runtime.turret_worker, "shutdown", checking_shutdown)
    history_start = len(endpoint.executed_request_history)
    runtime.shutdown(timeout=1.0)

    commands = [
        request.command for request in endpoint.executed_request_history[history_start:]
    ]
    assert commands[:2] == [CommandCode.SET_VELOCITY, CommandCode.MOTOR_OFF]
    assert motor_state_at_worker_shutdown == [MotorState.OFF]
    assert not runtime.turret_worker.is_alive()


def test_synchronous_partial_start_failure_rolls_back_started_workers(
    monkeypatch,
) -> None:
    runtime, _sources, _endpoint = _injected_runtime()

    def fail_stereo_left_start() -> None:
        raise RuntimeError("forced Stereo Left startup failure")

    monkeypatch.setattr(runtime.stereo_left_worker, "start", fail_stereo_left_start)

    with pytest.raises(RuntimeError, match="forced Stereo Left startup failure"):
        runtime.start()

    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()
    runtime.shutdown(timeout=1.0)
    runtime.shutdown(timeout=1.0)


def test_shutdown_attempts_all_workers_before_raising_aggregate_error(
    monkeypatch,
) -> None:
    runtime, _sources, _endpoint = _injected_runtime()
    runtime.start()
    original_overview_stop = runtime.overview_worker.stop
    original_stereo_stop = runtime.stereo_left_worker.stop
    original_turret_shutdown = runtime.turret_worker.shutdown
    calls: list[str] = []

    def failing_overview_stop(timeout: float) -> bool:
        calls.append("overview")
        assert original_overview_stop(timeout)
        raise RuntimeError("reported Overview shutdown failure")

    def stereo_stop(timeout: float) -> bool:
        calls.append("stereo-left")
        return original_stereo_stop(timeout)

    def turret_shutdown(timeout: float) -> None:
        calls.append("turret")
        original_turret_shutdown(timeout)

    monkeypatch.setattr(runtime.overview_worker, "stop", failing_overview_stop)
    monkeypatch.setattr(runtime.stereo_left_worker, "stop", stereo_stop)
    monkeypatch.setattr(runtime.turret_worker, "shutdown", turret_shutdown)

    with pytest.raises(ApplicationShutdownError) as exc_info:
        runtime.shutdown(timeout=1.0)

    assert calls == ["stereo-left", "overview", "turret"]
    assert [failure.component for failure in exc_info.value.failures] == [
        "Overview CameraWorker"
    ]
    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()

    monkeypatch.setattr(runtime.overview_worker, "stop", original_overview_stop)
    runtime.shutdown(timeout=1.0)
    runtime.shutdown(timeout=1.0)
