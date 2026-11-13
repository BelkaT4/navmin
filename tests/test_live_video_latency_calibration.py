from __future__ import annotations

from threading import Event, Thread
from time import monotonic, monotonic_ns, sleep

import numpy as np
import pytest

from navmin.concurrency import LatestValue
from navmin.contracts import (
    CameraRay,
    CameraRole,
    FramePacket,
    FrameSourceTimestamp,
    FrameTimestampSource,
    MotorState,
    PlatformMotionSnapshot,
    PlatformMotionSource,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
    VisionResult,
)
from navmin.diagnostics import live_video_latency_calibration as live
from navmin.diagnostics.live_video_latency_calibration import (
    InRuntimeVideoLatencyCalibration,
    LiveCalibrationPhase,
    LiveVideoLatencyCalibrationConfig,
    build_motion_profile,
    motion_profile_duration_s,
)
from navmin.diagnostics.video_latency_calibration import (
    LatencyCandidateScore,
    OpticalFlowRateObservationY,
    VideoLatencyEstimate,
)


class _CameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay((x_px - 79.5) / 500.0, (y_px - 59.5) / 500.0, 1.0)


class _FakeTurret:
    def __init__(self) -> None:
        self.active = False
        self.emergency = False
        self.rates: list[float] = []
        self.current_rate = 0.0
        self.current_state = TurretState(
            connection_state=TurretConnectionState.READY,
            motor_state=MotorState.ON,
            control_mode=TurretControlMode.RELATIVE,
            max_speed_x_deg_s=50.0,
            max_speed_y_deg_s=50.0,
            acceleration_x_deg_s2=100.0,
            acceleration_y_deg_s2=100.0,
        )

    @property
    def video_latency_calibration_active(self) -> bool:
        return self.active

    def begin_video_latency_calibration(self) -> bool:
        if self.active:
            return False
        self.active = True
        return True

    def submit_video_latency_calibration_velocity_y(self, rate_y_deg_s: float) -> bool:
        if not self.active:
            return False
        self.current_rate = float(rate_y_deg_s)
        self.rates.append(self.current_rate)
        return True

    def end_video_latency_calibration(self) -> bool:
        if not self.active:
            return False
        self.current_rate = 0.0
        self.active = False
        return True

    def request_emergency(self) -> None:
        self.emergency = True
        self.active = False

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot:
        return PlatformMotionSnapshot(
            source=PlatformMotionSource.COMMAND_MODEL,
            valid=True,
            anchors=(),
        )


def _publisher(results: LatestValue[VisionResult], stop: Event) -> None:
    image = np.random.default_rng(42).integers(
        0, 256, size=(120, 160, 3), dtype=np.uint8
    )
    index = 0
    while not stop.is_set():
        index += 1
        receive_ns = monotonic_ns()
        results.publish(
            VisionResult(
                frame=FramePacket(
                    camera=CameraRole.OVERVIEW,
                    generation=1,
                    frame_id=index,
                    capture_id=None,
                    receive_timestamp_ns=receive_ns,
                    image=image,
                    source_timestamp=FrameSourceTimestamp(
                        FrameTimestampSource.SENSOR_RTP_PTS,
                        index * 10_000_000,
                    ),
                ),
                tracked_objects=(),
                processing_time_ns=0,
            )
        )
        sleep(0.01)


def _wait_terminal(runtime: InRuntimeVideoLatencyCalibration, timeout: float = 2.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        state = runtime.state_updates.get()
        if state is not None and state.phase in (
            LiveCalibrationPhase.COMPLETE,
            LiveCalibrationPhase.FAILED,
            LiveCalibrationPhase.CANCELLED,
        ):
            return state
        sleep(0.005)
    raise AssertionError("calibration did not reach terminal state")


def test_live_calibration_motion_profile_uses_only_y_and_ends_zero() -> None:
    config = LiveVideoLatencyCalibrationConfig(
        command_rate_deg_s=6.0,
        segment_duration_s=0.8,
        run_count=3,
    )
    profile = build_motion_profile(config)

    assert [step.rate_y_deg_s for step in profile] == [6.0, -6.0, 6.0, -6.0, 0.0]
    assert [step.offset_s for step in profile] == pytest.approx(
        [0.0, 0.8, 1.6, 2.4, 3.2]
    )
    assert motion_profile_duration_s(config) == pytest.approx(4.0)


def test_live_calibration_rejects_excessive_diagnostic_rate() -> None:
    config = LiveVideoLatencyCalibrationConfig(
        command_rate_deg_s=20.1,
        max_command_rate_deg_s=20.0,
    )
    with pytest.raises(ValueError, match="safety bound"):
        config.validate()


def test_in_runtime_calibration_reuses_latest_overview_and_y_turret_channel(
    monkeypatch,
) -> None:
    results: LatestValue[VisionResult] = LatestValue()
    stop = Event()
    producer = Thread(target=_publisher, args=(results, stop), daemon=True)
    producer.start()
    turret = _FakeTurret()

    def fake_flow(_previous, _current, **kwargs):
        return OpticalFlowRateObservationY(
            receive_midpoint_timestamp_ns=(
                kwargs["previous_receive_timestamp_ns"]
                + kwargs["current_receive_timestamp_ns"]
            )
            // 2,
            source_interval_ns=kwargs["source_interval_ns"],
            background_rate_y_deg_s=-turret.current_rate,
            translation_y_px=0.0,
            inliers=80,
            tracked_points=100,
        )

    score = LatencyCandidateScore(
        latency_ms=150,
        observation_count=20,
        rmse_deg_s=0.01,
        normalized_rmse=0.01,
        correlation=0.999,
        platform_rate_std_deg_s=3.0,
    )

    monkeypatch.setattr(live, "estimate_optical_flow_rate_y", fake_flow)
    monkeypatch.setattr(
        live,
        "estimate_video_latency_y",
        lambda *_args, **_kwargs: VideoLatencyEstimate(
            valid=True,
            reason=None,
            estimated_latency_ms=150,
            observation_count=20,
            best_score=score,
            candidates=(score,),
        ),
    )

    config = LiveVideoLatencyCalibrationConfig(
        command_rate_deg_s=6.0,
        segment_duration_s=0.03,
        run_count=1,
        warmup_s=0.05,
        final_settle_s=0.02,
        poll_interval_s=0.001,
        ownership_timeout_s=0.2,
    )
    written = []
    runtime = InRuntimeVideoLatencyCalibration(
        overview_results=results,
        camera_model_provider=(
            lambda generation: _CameraModel() if generation == 1 else None
        ),
        turret=turret,
        configured_video_latency_ms=0,
        config=config,
        result_writer=written.append,
    )
    try:
        deadline = monotonic() + 0.5
        while results.get() is None:
            assert monotonic() < deadline
            sleep(0.005)
        assert runtime.start()
        state = _wait_terminal(runtime)
    finally:
        runtime.shutdown()
        stop.set()
        producer.join(0.5)

    assert state.phase is LiveCalibrationPhase.COMPLETE
    assert state.result is not None
    assert state.result.axis == "y"
    assert state.result.recommended_video_latency_ms == 150
    assert any(rate > 0.0 for rate in turret.rates)
    assert any(rate < 0.0 for rate in turret.rates)
    assert turret.rates[-1] == 0.0
    assert not turret.active
    assert not turret.emergency
    assert written == [state.result]


def test_cancel_releases_calibration_ownership(monkeypatch) -> None:
    results: LatestValue[VisionResult] = LatestValue()
    stop = Event()
    producer = Thread(target=_publisher, args=(results, stop), daemon=True)
    producer.start()
    turret = _FakeTurret()
    monkeypatch.setattr(
        live,
        "estimate_optical_flow_rate_y",
        lambda *_args, **kwargs: OpticalFlowRateObservationY(
            receive_midpoint_timestamp_ns=kwargs["current_receive_timestamp_ns"],
            source_interval_ns=kwargs["source_interval_ns"],
            background_rate_y_deg_s=0.0,
            translation_y_px=0.0,
            inliers=80,
            tracked_points=100,
        ),
    )
    runtime = InRuntimeVideoLatencyCalibration(
        overview_results=results,
        camera_model_provider=lambda _generation: _CameraModel(),
        turret=turret,
        configured_video_latency_ms=0,
        config=LiveVideoLatencyCalibrationConfig(
            segment_duration_s=0.2,
            run_count=1,
            warmup_s=0.2,
            poll_interval_s=0.001,
            ownership_timeout_s=0.2,
        ),
    )
    try:
        deadline = monotonic() + 0.5
        while results.get() is None:
            assert monotonic() < deadline
            sleep(0.005)
        assert runtime.start()
        sleep(0.03)
        runtime.cancel()
        state = _wait_terminal(runtime)
    finally:
        runtime.shutdown()
        stop.set()
        producer.join(0.5)

    assert state.phase is LiveCalibrationPhase.CANCELLED
    assert not turret.active
    assert not turret.emergency
