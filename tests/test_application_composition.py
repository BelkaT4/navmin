from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest

from navmin.application import (
    ApplicationFactories,
    ApplicationShutdownError,
    build_application_runtime,
)
from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config import load_config
from navmin.config.models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    AppConfig,
    AxesConfig,
    AxisMechanicsConfig,
    CameraConfig,
    CamerasConfig,
    OverviewCameraConfig,
    PidControllerConfig,
    ProcessingScope,
    RtpJpegSourceConfig,
    RtspDecoderMode,
    RtspProtocol,
    RtspSourceConfig,
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
    ConfigUpdate,
    DistanceSource,
    MotorState,
    PlatformMotionRate,
    TargetRef,
    TrackingError,
    TurretConnectionState,
)
from navmin.core import DistanceMeasurementMode
from navmin.turret.protocol import CommandCode
from navmin.turret.simulator import FakeStm32Endpoint, FakeTransport
from navmin.turret.worker import TurretWorker
from navmin.vision.gstreamer_source import (
    GStreamerRtpJpegSource,
    GStreamerRtspSource,
)
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
        source=RtpJpegSourceConfig(
            bind_address="0.0.0.0",
            port=port,
            buffer_size=1,
        ),
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _overview_camera(port: int) -> OverviewCameraConfig:
    camera = _camera(port)
    return OverviewCameraConfig(
        enabled=camera.enabled,
        source=camera.source,
        processing_enabled=camera.processing_enabled,
        vision_processor_class=camera.vision_processor_class,
        calibration_file="calibration/overview.json",
    )


def _rtsp_camera(uri: str) -> CameraConfig:
    return CameraConfig(
        enabled=True,
        source=RtspSourceConfig(
            uri=uri,
            protocol=RtspProtocol.TCP,
            decoder_mode=RtspDecoderMode.SOFTWARE,
            latency_ms=100,
            drop_on_latency=True,
            buffer_size=1,
        ),
        processing_enabled=False,
        vision_processor_class="Legacy14VisionProcessor",
    )


def _default_tracking_control_config():
    defaults = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "navmin.json"
    )
    return load_config(defaults).tracking_control


def _config(
    *,
    stereo_left_enabled: bool = True,
    stereo_right_enabled: bool = False,
    stereo_enabled: bool = False,
) -> AppConfig:
    center = AimPointConfig(x_px=None, y_px=None)
    axis = AxisMechanicsConfig(
        invert=False,
        full_steps_per_revolution=2000,
        microstep_divider=16,
        max_relative_move_deg=45.0,
    )
    return AppConfig(
        schema_version=3,
        vision=VisionConfig(
            processing_scope=ProcessingScope.MAIN_AND_PREVIEW,
            cameras=CamerasConfig(
                overview=_overview_camera(8888),
                stereo_left=_camera(8889, enabled=stereo_left_enabled),
                stereo_right=_camera(8890, enabled=stereo_right_enabled),
            ),
            distance=VisionDistanceConfig(
                source=DistanceSource.MANUAL,
                manual_distance_m=100.0,
                distance_stale_timeout_ms=300,
                stereo=StereoDistanceConfig(
                    stereo_enabled=stereo_enabled,
                    right_frame_buffer_size=4,
                    pair_timeout_ms=100,
                    calibration_file="calibration/stereo.json",
                ),
            ),
            camera_stale_timeout_ms=500,
            simulation_mode=False,
        ),
        aiming=AimingConfig(
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(overview=center, stereo_left=center),
        ),
        tracking_control=_default_tracking_control_config(),
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
        translation_unit="m",
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


def _injected_runtime(
    *,
    stereo_left_enabled: bool = True,
    stereo_right_enabled: bool = False,
    stereo_enabled: bool = False,
):
    sources = {
        CameraRole.OVERVIEW: InMemoryFrameSource(),
        CameraRole.STEREO_LEFT: InMemoryFrameSource(),
    }
    if stereo_right_enabled:
        sources[CameraRole.STEREO_RIGHT] = InMemoryFrameSource()
    endpoint = FakeStm32Endpoint()

    def overview_factory(_config: CameraConfig) -> InMemoryFrameSource:
        return sources[CameraRole.OVERVIEW]

    def stereo_left_factory(_config: CameraConfig) -> InMemoryFrameSource:
        return sources[CameraRole.STEREO_LEFT]

    def stereo_right_factory(_config: CameraConfig) -> InMemoryFrameSource:
        return sources[CameraRole.STEREO_RIGHT]

    def transport_factory(
        _port: str,
        baudrate: int,
        _emulate_stm32: bool,
    ) -> FakeTransport:
        return FakeTransport(endpoint, baudrate=baudrate)

    runtime = build_application_runtime(
        config=_config(
            stereo_left_enabled=stereo_left_enabled,
            stereo_right_enabled=stereo_right_enabled,
            stereo_enabled=stereo_enabled,
        ),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
        factories=ApplicationFactories(
            overview_source_factory=overview_factory,
            stereo_left_source_factory=stereo_left_factory,
            stereo_right_source_factory=stereo_right_factory,
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


def test_stereo_right_disabled_does_not_call_source_factory() -> None:
    calls: list[CameraConfig] = []

    def forbidden_right_factory(config: CameraConfig) -> InMemoryFrameSource:
        calls.append(config)
        raise AssertionError("disabled Stereo Right source factory was called")

    runtime = build_application_runtime(
        config=_config(stereo_right_enabled=False),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
        factories=ApplicationFactories(
            stereo_right_source_factory=forbidden_right_factory,
        ),
    )

    assert calls == []
    assert runtime.stereo_right_worker is None
    assert CameraRole.STEREO_RIGHT not in runtime.camera_workers
    assert CameraRole.STEREO_RIGHT not in runtime.camera_bindings


def test_stereo_right_enabled_is_owned_but_excluded_from_ui_and_targeting() -> None:
    runtime, sources, _endpoint = _injected_runtime(stereo_right_enabled=True)

    assert runtime.stereo_right_worker is not None
    assert runtime.stereo_right_worker.source is sources[CameraRole.STEREO_RIGHT]
    assert set(runtime.camera_workers) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
        CameraRole.STEREO_RIGHT,
    }
    assert set(runtime.camera_bindings) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    }
    assert set(runtime.mediator._target_controls) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    }
    assert set(runtime.mediator._vision_processing_enabled) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    }


@pytest.mark.parametrize(
    ("stereo_left_enabled", "stereo_right_enabled", "stereo_enabled"),
    [
        (True, False, True),
        (True, True, False),
        (False, True, True),
    ],
)
def test_stereo_pairer_is_inactive_when_any_prerequisite_is_disabled(
    stereo_left_enabled: bool,
    stereo_right_enabled: bool,
    stereo_enabled: bool,
) -> None:
    runtime, _sources, _endpoint = _injected_runtime(
        stereo_left_enabled=stereo_left_enabled,
        stereo_right_enabled=stereo_right_enabled,
        stereo_enabled=stereo_enabled,
    )

    assert runtime.stereo_pairer is None
    assert runtime.stereo_target_joiner is None
    assert runtime.stereo_distance_worker is None


def test_stereo_disabled_keeps_modes_selectable_without_compute_path() -> None:
    runtime, _sources, _endpoint = _injected_runtime(
        stereo_right_enabled=True,
        stereo_enabled=False,
    )

    assert runtime.stereo_pairer is None
    assert runtime.stereo_target_joiner is None
    assert runtime.stereo_distance_worker is None
    assert runtime.mediator.swap_main_preview() is CameraRole.STEREO_LEFT

    runtime.mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    assert runtime.mediator.distance_mode is DistanceMeasurementMode.STEREO_EXACT
    assert runtime.mediator.current_distance is None

    runtime.mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
    assert runtime.mediator.distance_mode is DistanceMeasurementMode.STEREO_TIMESTAMP
    assert runtime.mediator.current_distance is None


def test_enabled_stereo_pairer_receives_both_corrected_frame_paths_and_resets() -> None:
    runtime, sources, _endpoint = _injected_runtime(
        stereo_right_enabled=True,
        stereo_enabled=True,
    )
    pairer = runtime.stereo_pairer
    joiner = runtime.stereo_target_joiner
    distance_worker = runtime.stereo_distance_worker
    diagnostics_reporter = runtime.stereo_diagnostics_reporter
    assert pairer is not None
    assert runtime.stereo_timestamp_pairer is not None
    assert joiner is not None
    assert distance_worker is not None
    assert diagnostics_reporter is not None
    assert runtime.stereo_right_worker is not None
    assert set(runtime.camera_bindings) == {
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
    }
    assert CameraRole.STEREO_RIGHT not in runtime.mediator._target_controls
    assert CameraRole.STEREO_RIGHT not in runtime.mediator._vision_processing_enabled

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    runtime.mediator.set_distance_mode(DistanceMeasurementMode.STEREO_EXACT)
    runtime.start()
    try:
        sources[CameraRole.STEREO_LEFT].push(
            frame,
            capture_id=42,
            receive_timestamp_ns=1_000_000,
        )
        sources[CameraRole.STEREO_RIGHT].push(
            frame,
            capture_id=42,
            receive_timestamp_ns=2_000_000,
        )
        _wait_until(lambda: pairer.latest_pair is not None)

        pair = pairer.latest_pair
        assert pair is not None
        assert pair.identity.capture_id == 42
        assert pair.left.camera is CameraRole.STEREO_LEFT
        assert pair.right.camera is CameraRole.STEREO_RIGHT
        assert pair.left.generation == runtime.stereo_left_worker.pipeline.generation
        assert pair.right.generation == runtime.stereo_right_worker.pipeline.generation

        runtime.mediator.set_distance_mode(DistanceMeasurementMode.STEREO_TIMESTAMP)
        sources[CameraRole.STEREO_RIGHT].push(
            frame, capture_id=None, receive_timestamp_ns=10_000_000
        )
        sources[CameraRole.STEREO_LEFT].push(
            frame, capture_id=None, receive_timestamp_ns=10_010_000
        )
        timestamp_pairer = runtime.stereo_timestamp_pairer
        assert timestamp_pairer is not None
        _wait_until(lambda: timestamp_pairer.latest_pair is not None)
        timestamp_pair = timestamp_pairer.latest_pair
        assert timestamp_pair is not None
        assert timestamp_pair.left.capture_id is None
        assert timestamp_pair.right.capture_id is None
        assert pairer.latest_pair is None  # mode transition cleared old exact pair
    finally:
        runtime.shutdown()

    assert pairer.latest_pair is None
    assert runtime.stereo_timestamp_pairer is not None
    assert runtime.stereo_timestamp_pairer.latest_pair is None
    assert runtime.stereo_timestamp_pairer.state.left_generation is None
    assert runtime.stereo_timestamp_pairer.state.right_generation is None
    assert pairer.state.left_generation is None
    assert pairer.state.right_generation is None
    assert joiner.state.left_generation is None
    assert joiner.state.target is None
    assert joiner.observations.get() is None
    assert not distance_worker.is_alive()
    assert not diagnostics_reporter.is_alive()


def test_stereo_right_enabled_follows_startup_and_reverse_shutdown_order(
    monkeypatch,
) -> None:
    runtime, _sources, _endpoint = _injected_runtime(stereo_right_enabled=True)
    assert runtime.stereo_right_worker is not None
    events: list[str] = []

    monkeypatch.setattr(runtime.turret_worker, "start", lambda: events.append("turret"))
    monkeypatch.setattr(
        runtime.overview_worker,
        "start",
        lambda: events.append("overview"),
    )
    monkeypatch.setattr(
        runtime.stereo_left_worker,
        "start",
        lambda: events.append("stereo-left"),
    )
    monkeypatch.setattr(
        runtime.stereo_right_worker,
        "start",
        lambda: events.append("stereo-right"),
    )
    monkeypatch.setattr(
        runtime.stereo_right_worker,
        "stop",
        lambda *, timeout: events.append("stop-stereo-right") or True,
    )
    monkeypatch.setattr(
        runtime.stereo_left_worker,
        "stop",
        lambda *, timeout: events.append("stop-stereo-left") or True,
    )
    monkeypatch.setattr(
        runtime.overview_worker,
        "stop",
        lambda *, timeout: events.append("stop-overview") or True,
    )
    monkeypatch.setattr(
        runtime.turret_worker,
        "shutdown",
        lambda *, timeout: events.append("stop-turret"),
    )

    runtime.start()
    runtime.shutdown()

    assert events == [
        "turret",
        "overview",
        "stereo-left",
        "stereo-right",
        "stop-stereo-right",
        "stop-stereo-left",
        "stop-overview",
        "stop-turret",
    ]


def test_production_gstreamer_detection_includes_stereo_right(monkeypatch) -> None:
    sources = {
        CameraRole.OVERVIEW: InMemoryFrameSource(),
        CameraRole.STEREO_LEFT: InMemoryFrameSource(),
    }
    runtime = build_application_runtime(
        config=_config(stereo_right_enabled=True),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
        factories=ApplicationFactories(
            overview_source_factory=lambda _config: sources[CameraRole.OVERVIEW],
            stereo_left_source_factory=lambda _config: sources[CameraRole.STEREO_LEFT],
        ),
    )
    events: list[str] = []
    monkeypatch.setattr(
        "navmin.application.initialize_gstreamer_runtime",
        lambda: events.append("gstreamer"),
    )
    monkeypatch.setattr(runtime.turret_worker, "start", lambda: None)
    for worker in runtime.camera_workers.values():
        monkeypatch.setattr(worker, "start", lambda: None)

    runtime.start()

    assert events == ["gstreamer"]


def test_runtime_pid_tuning_dependencies_use_loaded_config_and_monotonic_updates(
    monkeypatch,
    caplog,
) -> None:
    config = _config()
    runtime = build_application_runtime(
        config=config,
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )
    queued: list[ConfigUpdate[TurretConfig]] = []
    monkeypatch.setattr(runtime.turret_worker, "submit_config_update", queued.append)
    caplog.set_level(logging.INFO, logger="navmin.application")

    ui = runtime.ui_dependencies
    assert ui.pid_controller_config is config.turret.controller

    first = PidControllerConfig(0.25, 0.01, 0.04, 0.30, 0.02, 0.05)
    second = PidControllerConfig(0.35, 0.03, 0.06, 0.40, 0.04, 0.07)
    third = PidControllerConfig(0.45, 0.05, 0.08, 0.50, 0.06, 0.09)
    for gains in (first, second, third):
        ui.apply_pid_tuning(gains)

    assert [update.revision for update in queued] == [1, 2, 3]
    assert [update.config.controller for update in queued] == [first, second, third]
    for update in queued:
        assert update.config.serial == config.turret.serial
        assert update.config.axes == config.turret.axes
        assert update.config.stm32 == config.turret.stm32
        assert update.config.emulate_stm32 == config.turret.emulate_stm32
    assert queued[-1].config == replace(config.turret, controller=third)
    assert "PID tuning update queued revision=3" in caplog.text


@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf")])
def test_runtime_pid_tuning_rejects_invalid_gains_without_publishing(
    monkeypatch,
    invalid: float,
) -> None:
    runtime = build_application_runtime(
        config=_config(),
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )
    queued: list[ConfigUpdate[TurretConfig]] = []
    monkeypatch.setattr(runtime.turret_worker, "submit_config_update", queued.append)
    gains = PidControllerConfig(invalid, 0.0, 0.0, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        runtime.apply_pid_tuning(gains)

    assert queued == []


def test_runtime_tracking_tuning_uses_one_revision_for_core_and_turret(
    monkeypatch,
    caplog,
) -> None:
    config = _config()
    runtime = build_application_runtime(
        config=config,
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )
    queued: list[ConfigUpdate] = []
    monkeypatch.setattr(
        runtime.turret_worker,
        "submit_tracking_control_config_update",
        queued.append,
    )
    caplog.set_level(logging.INFO, logger="navmin.application")
    estimator = runtime.mediator.tracking_estimator
    target = TargetRef(CameraRole.OVERVIEW, 1, 7)
    estimator.observe(
        TrackingError(target, 1.0, 2.0, 1_000_000_000),
        PlatformMotionRate(1_000_000_000, 0.0, 0.0),
    )

    control_only = replace(
        config.tracking_control,
        feed_forward=replace(
            config.tracking_control.feed_forward,
            enabled=True,
            gain_x=0.4,
            gain_y=0.5,
        ),
        prediction=replace(
            config.tracking_control.prediction,
            enabled=True,
            horizon_ms=75,
        ),
    )
    runtime.apply_tracking_control_tuning(control_only)

    assert estimator.sample_count == 1
    assert estimator.diagnostics.target == target
    assert queued == [ConfigUpdate(1, control_only)]

    estimator_change = replace(
        control_only,
        velocity_estimator=replace(
            control_only.velocity_estimator,
            window_ms=250,
        ),
    )
    runtime.apply_tracking_control_tuning(estimator_change)

    assert estimator.sample_count == 0
    assert estimator.diagnostics.target == target
    assert queued == [ConfigUpdate(1, control_only), ConfigUpdate(2, estimator_change)]
    messages = [
        record.getMessage()
        for record in caplog.records
        if "Tracking tuning update queued" in record.getMessage()
    ]
    assert len(messages) == 2
    assert "revision=2" in messages[-1]
    assert "gain=(0.4,0.5)" in messages[-1]
    assert "horizon=75" in messages[-1]
    assert "window=250" in messages[-1]


def test_default_composition_supports_overview_rtsp_and_stereo_left_rtp() -> None:
    base = _config()
    cameras = replace(
        base.vision.cameras,
        overview=_rtsp_camera("rtsp://overview.local/stream"),
    )
    config = replace(base, vision=replace(base.vision, cameras=cameras))

    runtime = build_application_runtime(
        config=config,
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )

    assert isinstance(runtime.overview_worker.source, GStreamerRtspSource)
    assert isinstance(runtime.stereo_left_worker.source, GStreamerRtpJpegSource)


def test_default_composition_supports_overview_rtp_and_stereo_left_rtsp() -> None:
    base = _config()
    cameras = replace(
        base.vision.cameras,
        stereo_left=_rtsp_camera("rtsp://stereo-left.local/stream"),
    )
    config = replace(base, vision=replace(base.vision, cameras=cameras))

    runtime = build_application_runtime(
        config=config,
        overview_calibration=_overview_calibration(),
        stereo_calibration=_stereo_calibration(),
    )

    assert isinstance(runtime.overview_worker.source, GStreamerRtpJpegSource)
    assert isinstance(runtime.stereo_left_worker.source, GStreamerRtspSource)

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
        assert (
            bindings[CameraRole.OVERVIEW].target_events
            is runtime.overview_worker.pipeline.target_events
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


def test_stereo_right_start_failure_rolls_back_started_workers(
    monkeypatch,
) -> None:
    runtime, _sources, _endpoint = _injected_runtime(stereo_right_enabled=True)
    assert runtime.stereo_right_worker is not None
    calls: list[str] = []

    original_stereo_left_stop = runtime.stereo_left_worker.stop
    original_overview_stop = runtime.overview_worker.stop
    original_turret_shutdown = runtime.turret_worker.shutdown

    def fail_stereo_right_start() -> None:
        raise RuntimeError("forced Stereo Right startup failure")

    def forbidden_stereo_right_stop(*, timeout: float) -> bool:
        calls.append("stereo-right")
        return True

    def stereo_left_stop(*, timeout: float) -> bool:
        calls.append("stereo-left")
        return original_stereo_left_stop(timeout=timeout)

    def overview_stop(*, timeout: float) -> bool:
        calls.append("overview")
        return original_overview_stop(timeout=timeout)

    def turret_shutdown(*, timeout: float) -> None:
        calls.append("turret")
        original_turret_shutdown(timeout=timeout)

    monkeypatch.setattr(runtime.stereo_right_worker, "start", fail_stereo_right_start)
    monkeypatch.setattr(runtime.stereo_right_worker, "stop", forbidden_stereo_right_stop)
    monkeypatch.setattr(runtime.stereo_left_worker, "stop", stereo_left_stop)
    monkeypatch.setattr(runtime.overview_worker, "stop", overview_stop)
    monkeypatch.setattr(runtime.turret_worker, "shutdown", turret_shutdown)

    with pytest.raises(RuntimeError, match="forced Stereo Right startup failure"):
        runtime.start()

    assert calls == ["stereo-left", "overview", "turret"]
    assert not runtime.stereo_right_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.overview_worker.is_alive()
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
