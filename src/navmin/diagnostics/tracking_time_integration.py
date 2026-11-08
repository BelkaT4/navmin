"""Integrated synthetic proof for camera source time used by tracking control.

The deterministic proof deliberately exercises the production timestamp contracts
without requiring PyGObject: synthetic transport PTS enter the real
``GStreamerRtpJpegSource`` through its injectable backend seam, then flow through
``Mediator``'s ``CameraClockTracker`` / ``ClockMapper`` ownership and the tracking
OFF/ON selector.  A separate optional localhost transport probe uses real
GStreamer so the same source mode can also be checked end-to-end on a host that
has the production multimedia runtime installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, monotonic_ns, sleep
from typing import Any

import cv2
import numpy as np

from navmin.concurrency import TargetControlChannel
from navmin.config.models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    FeedForwardConfig,
    PlatformMotionConfig,
    PredictionConfig,
    RtpJpegSourceConfig,
    RtpJpegTimestampMode,
    TrackingControlConfig,
    UiConfig,
    VelocityEstimatorConfig,
)
from navmin.contracts import (
    BBox,
    CameraRay,
    CameraRole,
    CameraSessionStarted,
    FramePacket,
    MotorState,
    PlatformMotionAnchor,
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
from navmin.core import Mediator
from navmin.time_sync import ClockMappingState
from navmin.vision.gstreamer_source import (
    CameraSourceError,
    GStreamerRtpJpegSource,
)
from navmin.vision.pipeline import DecodedFrame

_DEFAULT_WIDTH = 101
_DEFAULT_HEIGHT = 81
_DEFAULT_PORT = 18_948


class TrackingTimeIntegrationProofError(RuntimeError):
    """One required integrated tracking-time proof condition failed."""


class TrackingTimeIntegrationProofBlocked(RuntimeError):
    """The optional real-GStreamer proof cannot run on this host."""


@dataclass(frozen=True, slots=True)
class TrackingTimeGenerationResult:
    generation: int
    mapper_states: tuple[str, ...]
    reset_count: int
    fallback_observation_timestamp_ns: int | None
    fallback_receive_timestamp_ns: int | None
    mapped_observation_timestamp_ns: int
    mapped_timestamp_ns: int
    mapped_receive_timestamp_ns: int


@dataclass(frozen=True, slots=True)
class TrackingTimeDeterministicResult:
    status: str
    source_timestamp_type: str
    off_observation_timestamp_ns: int
    off_receive_timestamp_ns: int
    on_observation_timestamp_ns: int
    on_mapped_timestamp_ns: int
    on_receive_timestamp_ns: int
    after_off_observation_timestamp_ns: int
    after_off_receive_timestamp_ns: int
    first_generation: TrackingTimeGenerationResult
    second_generation: TrackingTimeGenerationResult
    source_restart_accepted_reset_pts: bool
    turret_stop_requests: int

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass(frozen=True, slots=True)
class TrackingTimeGStreamerResult:
    status: str
    source_timestamp_type: str
    mapper_states: tuple[str, ...]
    off_observation_timestamp_ns: int
    off_receive_timestamp_ns: int
    on_observation_timestamp_ns: int
    on_mapped_timestamp_ns: int
    on_receive_timestamp_ns: int

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass(frozen=True, slots=True)
class TrackingTimeIntegrationProofResult:
    deterministic: TrackingTimeDeterministicResult
    gstreamer: TrackingTimeGStreamerResult | None
    gstreamer_blocked_reason: str | None

    @property
    def passed(self) -> bool:
        return self.deterministic.passed and (
            self.gstreamer is None or self.gstreamer.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ProofCameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay((x_px - 50.0) / 50.0, (y_px - 40.0) / 40.0, 1.0)


class _ProofTurret:
    def __init__(self) -> None:
        self.tracking_controls: list[TrackingControlSample] = []
        self.invalidations = 0
        self.stop_requests = 0
        self._state = TurretState(
            connection_state=TurretConnectionState.READY,
            motor_state=MotorState.ON,
            control_mode=TurretControlMode.TRACKING,
            max_speed_x_deg_s=50.0,
            max_speed_y_deg_s=50.0,
            acceleration_x_deg_s2=100.0,
            acceleration_y_deg_s2=100.0,
        )
        self._platform = PlatformMotionSnapshot(
            source=PlatformMotionSource.COMMAND_MODEL,
            valid=True,
            anchors=(
                PlatformMotionAnchor(
                    timestamp_ns=0,
                    modeled_rate_x_deg_s=0.0,
                    modeled_rate_y_deg_s=0.0,
                    commanded_rate_x_deg_s=0.0,
                    commanded_rate_y_deg_s=0.0,
                    acceleration_x_deg_s2=100.0,
                    acceleration_y_deg_s2=100.0,
                ),
            ),
        )

    @property
    def current_state(self) -> TurretState:
        return self._state

    def submit_move_relative(self, _command) -> None:
        raise AssertionError("relative motion is outside this proof")

    def submit_tracking_control(self, sample: TrackingControlSample) -> None:
        self.tracking_controls.append(sample)

    def invalidate_tracking_control(self) -> None:
        self.invalidations += 1

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot:
        return self._platform

    def set_control_mode(self, _mode: TurretControlMode) -> bool:
        return True

    def stop_motion(self) -> bool:
        self.stop_requests += 1
        return True

    def motor_on(self) -> bool:
        return True

    def motor_off(self) -> bool:
        return True

    def request_emergency(self) -> None:
        return None


class _MutableClock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _SyntheticBackend:
    def __init__(self, on_frame, on_failure) -> None:
        self._on_frame = on_frame
        self._on_failure = on_failure
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def poll_failure(self) -> BaseException | None:
        return None

    def emit(self, image: np.ndarray, pts_ns: int) -> None:
        if not self.started:
            raise TrackingTimeIntegrationProofError("synthetic backend is not started")
        self._on_frame(image, pts_ns)


class _SyntheticBackendFactory:
    def __init__(self) -> None:
        self.backends: list[_SyntheticBackend] = []

    def __call__(self, _pipeline_description, on_frame, on_failure) -> _SyntheticBackend:
        backend = _SyntheticBackend(on_frame, on_failure)
        self.backends.append(backend)
        return backend

    @property
    def latest(self) -> _SyntheticBackend:
        if not self.backends:
            raise TrackingTimeIntegrationProofError("source backend was not created")
        return self.backends[-1]


@dataclass(frozen=True, slots=True)
class _FrameSchedule:
    source_ns: int
    receive_offset_ns: int


_GEN1 = (
    _FrameSchedule(0, 0),
    _FrameSchedule(400_000_000, 400_400_000),
    _FrameSchedule(800_000_000, 799_600_000),
    _FrameSchedule(1_200_000_000, 1_200_800_000),
    _FrameSchedule(1_600_000_000, 1_599_200_000),
    _FrameSchedule(2_000_000_000, 2_000_600_000),
)
_GEN2 = (
    _FrameSchedule(0, 0),
    _FrameSchedule(400_000_000, 400_500_000),
    _FrameSchedule(800_000_000, 799_700_000),
    _FrameSchedule(1_200_000_000, 1_200_600_000),
    _FrameSchedule(1_600_000_000, 1_600_100_000),
)


def _tracking_config() -> TrackingControlConfig:
    return TrackingControlConfig(
        velocity_estimator=VelocityEstimatorConfig(
            window_ms=200,
            min_span_ms=100,
            min_samples=4,
            max_samples=16,
            max_fit_rms_deg=1.0,
            max_target_rate_x_deg_s=90.0,
            max_target_rate_y_deg_s=90.0,
        ),
        feed_forward=FeedForwardConfig(enabled=False, gain_x=0.0, gain_y=0.0),
        prediction=PredictionConfig(
            enabled=False,
            horizon_ms=0,
            max_angle_x_deg=5.0,
            max_angle_y_deg=5.0,
        ),
        platform_motion=PlatformMotionConfig(source=PlatformMotionSource.COMMAND_MODEL),
    )


def _make_mediator() -> tuple[Mediator, _ProofTurret]:
    turret = _ProofTurret()
    controls = {
        CameraRole.OVERVIEW: TargetControlChannel(CameraRole.OVERVIEW),
        CameraRole.STEREO_LEFT: TargetControlChannel(CameraRole.STEREO_LEFT),
    }
    mediator = Mediator(
        aiming_config=AimingConfig(
            target_lost_timeout_ms=500,
            aim_points=AimPointsConfig(
                overview=AimPointConfig(50, 40),
                stereo_left=AimPointConfig(50, 40),
                stereo_right=AimPointConfig(50, 40),
            ),
        ),
        tracking_control_config=_tracking_config(),
        tracking_gap_timeout_ms=200,
        ui_config=UiConfig(
            default_camera=CameraRole.OVERVIEW,
            show_fps=False,
            show_stereo_right_diagnostics=False,
            show_stereo_right_camera=False,
        ),
        turret=turret,
        target_controls=controls,
    )
    return mediator, turret


def _session(generation: int) -> CameraSessionStarted:
    return CameraSessionStarted(
        camera=CameraRole.OVERVIEW,
        generation=generation,
        camera_model=_ProofCameraModel(),
        timestamp_ns=generation,
    )


def _image(frame_id: int) -> np.ndarray:
    return np.full((_DEFAULT_HEIGHT, _DEFAULT_WIDTH, 3), frame_id % 255, dtype=np.uint8)


def _result_from_decoded(
    decoded: DecodedFrame,
    *,
    generation: int,
    frame_id: int,
) -> VisionResult:
    image = np.array(decoded.image, copy=True, order="C")
    image.flags.writeable = False
    packet = FramePacket(
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=int(decoded.receive_timestamp_ns),
        image=image,
        source_timestamp=decoded.source_timestamp,
    )
    tracked = TrackedObject(
        track_id=7,
        bbox=BBox(55, 36, 10, 8),
        velocity_x_px_s=0.0,
        velocity_y_px_s=0.0,
        age_frames=max(frame_id, 1),
    )
    return VisionResult(packet, (tracked,), 0)


def _select_latest(mediator: Mediator, result: VisionResult) -> None:
    if not mediator.request_target_selection(result, 60.0, 40.0):
        raise TrackingTimeIntegrationProofError("target selection was rejected")
    request = mediator.pending_selection
    if request is None:
        raise TrackingTimeIntegrationProofError("target selection was not pending")
    event = TargetAcquired(
        selection_id=request.selection_id,
        target=TargetRef(CameraRole.OVERVIEW, result.frame.generation, 7),
        request_frame_id=request.frame_id,
        decision_frame_id=result.frame.frame_id,
        decision_frame_timestamp_ns=result.frame.receive_timestamp_ns,
    )
    if not mediator.accept_target_lifecycle_event(event):
        raise TrackingTimeIntegrationProofError("target acquisition was rejected")


def _emit_injected(
    *,
    source: GStreamerRtpJpegSource,
    factory: _SyntheticBackendFactory,
    receive_clock: _MutableClock,
    receive_ns: int,
    source_ns: int,
    frame_id: int,
) -> DecodedFrame:
    receive_clock.now_ns = receive_ns
    factory.latest.emit(_image(frame_id), source_ns)
    decoded = source.read()
    if decoded is None:
        raise TrackingTimeIntegrationProofError("production source did not publish frame")
    return decoded


def _accept_result(mediator: Mediator, result: VisionResult) -> str:
    if not mediator.accept_vision_result(result):
        raise TrackingTimeIntegrationProofError("mediator rejected synthetic vision result")
    observation = mediator.camera_clock_observation(CameraRole.OVERVIEW)
    if observation is None:
        raise TrackingTimeIntegrationProofError("camera clock observation is missing")
    return observation.diagnostics.state.value


def _last_control(turret: _ProofTurret) -> TrackingControlSample:
    if not turret.tracking_controls:
        raise TrackingTimeIntegrationProofError("tracking control was not published")
    return turret.tracking_controls[-1]


def run_deterministic_tracking_time_integration_proof() -> TrackingTimeDeterministicResult:
    """Run the full downstream timestamp/control proof without multimedia runtime."""
    base = monotonic_ns() - 30_000_000_000
    receive_clock = _MutableClock(base)
    factory = _SyntheticBackendFactory()
    source = GStreamerRtpJpegSource(
        RtpJpegSourceConfig(
            bind_address="127.0.0.1",
            port=_DEFAULT_PORT,
            buffer_size=1,
            timestamp_mode=RtpJpegTimestampMode.SENSOR_RTP_PTS,
        ),
        timestamp_clock_ns=receive_clock,
        backend_factory=factory,
    )
    mediator, turret = _make_mediator()

    if not mediator.accept_camera_session(_session(1)):
        raise TrackingTimeIntegrationProofError("generation 1 session was rejected")
    source.start()
    states1: list[str] = []
    latest: VisionResult | None = None
    try:
        for frame_id, item in enumerate(_GEN1[:4], start=1):
            decoded = _emit_injected(
                source=source,
                factory=factory,
                receive_clock=receive_clock,
                receive_ns=base + item.receive_offset_ns,
                source_ns=item.source_ns,
                frame_id=frame_id,
            )
            latest = _result_from_decoded(decoded, generation=1, frame_id=frame_id)
            states1.append(_accept_result(mediator, latest))

        if latest is None:
            raise TrackingTimeIntegrationProofError("generation 1 produced no frames")
        _select_latest(mediator, latest)
        off_control = _last_control(turret)
        if off_control.observation_timestamp_ns != latest.frame.receive_timestamp_ns:
            raise TrackingTimeIntegrationProofError("OFF mode did not use receive time")

        mediator.set_use_camera_time_for_tracking(True)
        item = _GEN1[4]
        decoded = _emit_injected(
            source=source,
            factory=factory,
            receive_clock=receive_clock,
            receive_ns=base + item.receive_offset_ns,
            source_ns=item.source_ns,
            frame_id=5,
        )
        on_result = _result_from_decoded(decoded, generation=1, frame_id=5)
        states1.append(_accept_result(mediator, on_result))
        on_control = _last_control(turret)
        on_observation = mediator.camera_clock_observation(CameraRole.OVERVIEW)
        if on_observation is None or on_observation.receive_anchored_timestamp is None:
            raise TrackingTimeIntegrationProofError("VALID mapper did not provide mapped time")
        mapped1 = on_observation.receive_anchored_timestamp
        if on_control.observation_timestamp_ns != mapped1.pc_timestamp_ns:
            raise TrackingTimeIntegrationProofError("ON mode did not use mapped camera time")

        mediator.set_use_camera_time_for_tracking(False)
        item = _GEN1[5]
        decoded = _emit_injected(
            source=source,
            factory=factory,
            receive_clock=receive_clock,
            receive_ns=base + item.receive_offset_ns,
            source_ns=item.source_ns,
            frame_id=6,
        )
        off_again_result = _result_from_decoded(decoded, generation=1, frame_id=6)
        states1.append(_accept_result(mediator, off_again_result))
        off_again_control = _last_control(turret)
        if off_again_control.observation_timestamp_ns != (
            off_again_result.frame.receive_timestamp_ns
        ):
            raise TrackingTimeIntegrationProofError(
                "OFF mode after toggle did not return to receive time"
            )

        first_observation = mediator.camera_clock_observation(CameraRole.OVERVIEW)
        if first_observation is None:
            raise TrackingTimeIntegrationProofError("generation 1 observation missing")
        first_generation = TrackingTimeGenerationResult(
            generation=1,
            mapper_states=tuple(states1),
            reset_count=first_observation.diagnostics.reset_count,
            fallback_observation_timestamp_ns=None,
            fallback_receive_timestamp_ns=None,
            mapped_observation_timestamp_ns=on_control.observation_timestamp_ns,
            mapped_timestamp_ns=mapped1.pc_timestamp_ns,
            mapped_receive_timestamp_ns=on_result.frame.receive_timestamp_ns,
        )

        source.stop()
        source.start()
        if not mediator.accept_camera_session(_session(2)):
            raise TrackingTimeIntegrationProofError("generation 2 session was rejected")
        mediator.set_use_camera_time_for_tracking(True)

        states2: list[str] = []
        gen2_base = base + 5_000_000_000
        first = _GEN2[0]
        decoded = _emit_injected(
            source=source,
            factory=factory,
            receive_clock=receive_clock,
            receive_ns=gen2_base + first.receive_offset_ns,
            source_ns=first.source_ns,
            frame_id=1,
        )
        first_result = _result_from_decoded(decoded, generation=2, frame_id=1)
        states2.append(_accept_result(mediator, first_result))
        _select_latest(mediator, first_result)
        fallback_control = _last_control(turret)
        if fallback_control.observation_timestamp_ns != first_result.frame.receive_timestamp_ns:
            raise TrackingTimeIntegrationProofError(
                "warming mapper did not fall back to receive time"
            )

        final_result = first_result
        for frame_id, item in enumerate(_GEN2[1:4], start=2):
            decoded = _emit_injected(
                source=source,
                factory=factory,
                receive_clock=receive_clock,
                receive_ns=gen2_base + item.receive_offset_ns,
                source_ns=item.source_ns,
                frame_id=frame_id,
            )
            final_result = _result_from_decoded(
                decoded, generation=2, frame_id=frame_id
            )
            states2.append(_accept_result(mediator, final_result))

        second_observation = mediator.camera_clock_observation(CameraRole.OVERVIEW)
        if second_observation is None or second_observation.receive_anchored_timestamp is None:
            raise TrackingTimeIntegrationProofError(
                "generation 2 mapper did not return to VALID mapped time"
            )
        mapped2 = second_observation.receive_anchored_timestamp
        mapped_control = _last_control(turret)
        if mapped_control.observation_timestamp_ns != mapped2.pc_timestamp_ns:
            raise TrackingTimeIntegrationProofError(
                "generation 2 ON mode did not resume mapped camera time"
            )
        if ClockMappingState.VALID.value not in states1 or ClockMappingState.VALID.value not in states2:
            raise TrackingTimeIntegrationProofError("mapper never reached VALID")
        if states2[0] != ClockMappingState.WARMING.value:
            raise TrackingTimeIntegrationProofError(
                "generation restart did not reset mapper to WARMING"
            )

        second_generation = TrackingTimeGenerationResult(
            generation=2,
            mapper_states=tuple(states2),
            reset_count=second_observation.diagnostics.reset_count,
            fallback_observation_timestamp_ns=fallback_control.observation_timestamp_ns,
            fallback_receive_timestamp_ns=first_result.frame.receive_timestamp_ns,
            mapped_observation_timestamp_ns=mapped_control.observation_timestamp_ns,
            mapped_timestamp_ns=mapped2.pc_timestamp_ns,
            mapped_receive_timestamp_ns=final_result.frame.receive_timestamp_ns,
        )
        return TrackingTimeDeterministicResult(
            status="PASS",
            source_timestamp_type="sensor-rtp-pts",
            off_observation_timestamp_ns=off_control.observation_timestamp_ns,
            off_receive_timestamp_ns=latest.frame.receive_timestamp_ns,
            on_observation_timestamp_ns=on_control.observation_timestamp_ns,
            on_mapped_timestamp_ns=mapped1.pc_timestamp_ns,
            on_receive_timestamp_ns=on_result.frame.receive_timestamp_ns,
            after_off_observation_timestamp_ns=off_again_control.observation_timestamp_ns,
            after_off_receive_timestamp_ns=off_again_result.frame.receive_timestamp_ns,
            first_generation=first_generation,
            second_generation=second_generation,
            source_restart_accepted_reset_pts=(
                first_result.frame.source_timestamp is not None
                and first_result.frame.source_timestamp.timestamp_ns == 0
            ),
            turret_stop_requests=turret.stop_requests,
        )
    finally:
        source.stop()


def _import_gstreamer() -> Any:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp

        Gst.init(None)
        _ = GstApp.AppSrc
    except (AttributeError, ImportError, ValueError) as exc:
        raise TrackingTimeIntegrationProofBlocked(
            "PyGObject with Gst 1.0/GstApp 1.0 is not available"
        ) from exc
    required = (
        "appsrc",
        "jpegparse",
        "rtpjpegpay",
        "udpsink",
        "udpsrc",
        "rtpjitterbuffer",
        "rtpjpegdepay",
        "jpegdec",
        "videoconvert",
        "appsink",
    )
    missing = tuple(name for name in required if Gst.ElementFactory.find(name) is None)
    if missing:
        raise TrackingTimeIntegrationProofBlocked(
            "missing GStreamer elements: " + ", ".join(missing)
        )
    return Gst


def _jpeg_payload(frame_id: int) -> bytes:
    image = _image(frame_id)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise TrackingTimeIntegrationProofError("OpenCV failed to encode proof JPEG")
    return encoded.tobytes()


def _sender_pipeline_description(*, port: int, fps: int) -> str:
    return (
        "appsrc name=source is-live=true format=time do-timestamp=false "
        f"caps=image/jpeg,framerate={fps}/1 "
        "! jpegparse ! rtpjpegpay pt=26 "
        f"! udpsink host=127.0.0.1 port={port} sync=false async=false"
    )


def _poll_decoded(source: GStreamerRtpJpegSource, *, timeout_s: float) -> DecodedFrame:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        failure = source.failure
        if failure is not None:
            raise TrackingTimeIntegrationProofError(
                f"production GStreamer source failed: {failure}"
            )
        decoded = source.read()
        if decoded is not None:
            return decoded
        sleep(0.005)
    raise TrackingTimeIntegrationProofError("timed out waiting for production RTP/JPEG frame")


def _sleep_until_monotonic(deadline_s: float) -> None:
    """Wait for an absolute monotonic deadline without accumulating loop overhead."""
    while True:
        remaining_s = deadline_s - monotonic()
        if remaining_s <= 0.0:
            return
        sleep(min(remaining_s, 0.01))


def run_gstreamer_tracking_time_integration_proof(
    *,
    port: int = _DEFAULT_PORT,
    fps: int = 5,
    timeout_s: float = 3.0,
) -> TrackingTimeGStreamerResult:
    """Run one real localhost RTP/JPEG -> production source -> tracking-time proof."""
    if fps < 4:
        raise ValueError("fps must be >= 4 so the mapper can warm within a short proof")
    Gst = _import_gstreamer()
    mediator, turret = _make_mediator()
    if not mediator.accept_camera_session(_session(1)):
        raise TrackingTimeIntegrationProofError("GStreamer proof session was rejected")

    source = GStreamerRtpJpegSource(
        RtpJpegSourceConfig(
            bind_address="127.0.0.1",
            port=port,
            buffer_size=1,
            timestamp_mode=RtpJpegTimestampMode.SENSOR_RTP_PTS,
        )
    )
    sender = Gst.parse_launch(_sender_pipeline_description(port=port, fps=fps))
    appsrc = sender.get_by_name("source")
    if appsrc is None:
        raise TrackingTimeIntegrationProofError("sender pipeline omitted appsrc")

    source_started = False
    sender_started = False
    states: list[str] = []
    latest: VisionResult | None = None
    try:
        source.start()
        source_started = True
        if sender.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise TrackingTimeIntegrationProofError("sender failed to enter PLAYING")
        sender_started = True
        sleep(0.15)
        interval_ns = 1_000_000_000 // fps
        # Pre-encode payloads so JPEG work cannot stretch the synthetic source clock.
        # Use a >2 s source span so the production drift estimate is not dominated by
        # sub-millisecond host scheduling/decode jitter.
        frame_count = fps * 2 + 2
        payloads = tuple(_jpeg_payload(index + 1) for index in range(frame_count + 1))
        schedule_origin_s = monotonic() + 0.1
        for index in range(frame_count):
            # PTS advances by exactly interval_ns.  Therefore sender wall time must
            # follow the same *absolute* schedule.  Sleeping 1/fps after encode,
            # push and decode would accumulate harness overhead and fabricate clock
            # drift that the production ClockMapper correctly rejects.
            _sleep_until_monotonic(schedule_origin_s + index / fps)
            payload = payloads[index]
            buffer = Gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = index * interval_ns
            buffer.dts = Gst.CLOCK_TIME_NONE
            buffer.duration = interval_ns
            flow = appsrc.emit("push-buffer", buffer)
            if flow != Gst.FlowReturn.OK:
                raise TrackingTimeIntegrationProofError(
                    f"GStreamer appsrc push failed: {flow}"
                )
            decoded = _poll_decoded(source, timeout_s=timeout_s)
            latest = _result_from_decoded(decoded, generation=1, frame_id=index + 1)
            states.append(_accept_result(mediator, latest))

        if latest is None:
            raise TrackingTimeIntegrationProofError("GStreamer proof produced no frames")
        if ClockMappingState.VALID.value not in states:
            observation = mediator.camera_clock_observation(CameraRole.OVERVIEW)
            diagnostics = None if observation is None else observation.diagnostics
            detail = (
                "no camera-clock diagnostics"
                if diagnostics is None
                else (
                    f"reason={None if diagnostics.invalid_reason is None else diagnostics.invalid_reason.value} "
                    f"samples={diagnostics.sample_count} "
                    f"span_ns={diagnostics.device_span_ns} "
                    f"drift_ppm={diagnostics.drift_ppm} "
                    f"fit_rms_ns={diagnostics.fit_rms_ns} "
                    f"rejected_outliers={diagnostics.rejected_outlier_count}"
                )
            )
            raise TrackingTimeIntegrationProofError(
                f"real GStreamer path did not reach VALID: {states!r}; {detail}"
            )
        _select_latest(mediator, latest)
        off_control = _last_control(turret)
        if off_control.observation_timestamp_ns != latest.frame.receive_timestamp_ns:
            raise TrackingTimeIntegrationProofError("GStreamer OFF mode was not legacy")

        mediator.set_use_camera_time_for_tracking(True)
        _sleep_until_monotonic(schedule_origin_s + frame_count / fps)
        payload = payloads[frame_count]
        buffer = Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, payload)
        buffer.pts = frame_count * interval_ns
        buffer.dts = Gst.CLOCK_TIME_NONE
        buffer.duration = interval_ns
        flow = appsrc.emit("push-buffer", buffer)
        if flow != Gst.FlowReturn.OK:
            raise TrackingTimeIntegrationProofError(
                f"GStreamer appsrc push failed: {flow}"
            )
        decoded = _poll_decoded(source, timeout_s=timeout_s)
        on_result = _result_from_decoded(
            decoded, generation=1, frame_id=frame_count + 1
        )
        states.append(_accept_result(mediator, on_result))
        observation = mediator.camera_clock_observation(CameraRole.OVERVIEW)
        if observation is None or observation.receive_anchored_timestamp is None:
            raise TrackingTimeIntegrationProofError("real GStreamer mapper has no mapped time")
        mapped = observation.receive_anchored_timestamp
        on_control = _last_control(turret)
        if on_control.observation_timestamp_ns != mapped.pc_timestamp_ns:
            raise TrackingTimeIntegrationProofError(
                "real GStreamer ON mode did not use mapped time"
            )
        source_timestamp = on_result.frame.source_timestamp
        if source_timestamp is None:
            raise TrackingTimeIntegrationProofError(
                "real GStreamer source omitted source timestamp"
            )
        return TrackingTimeGStreamerResult(
            status="PASS",
            source_timestamp_type=source_timestamp.source.value,
            mapper_states=tuple(states),
            off_observation_timestamp_ns=off_control.observation_timestamp_ns,
            off_receive_timestamp_ns=latest.frame.receive_timestamp_ns,
            on_observation_timestamp_ns=on_control.observation_timestamp_ns,
            on_mapped_timestamp_ns=mapped.pc_timestamp_ns,
            on_receive_timestamp_ns=on_result.frame.receive_timestamp_ns,
        )
    except CameraSourceError as exc:
        raise TrackingTimeIntegrationProofError(str(exc)) from exc
    finally:
        try:
            appsrc.emit("end-of-stream")
        except (AttributeError, TypeError):
            pass
        if sender_started:
            sender.set_state(Gst.State.NULL)
        if source_started:
            source.stop()


def run_tracking_time_integration_proof(
    *,
    include_gstreamer: bool = True,
    port: int = _DEFAULT_PORT,
) -> TrackingTimeIntegrationProofResult:
    deterministic = run_deterministic_tracking_time_integration_proof()
    gstreamer: TrackingTimeGStreamerResult | None = None
    blocked_reason: str | None = None
    if include_gstreamer:
        try:
            gstreamer = run_gstreamer_tracking_time_integration_proof(port=port)
        except TrackingTimeIntegrationProofBlocked as exc:
            blocked_reason = str(exc)
    return TrackingTimeIntegrationProofResult(
        deterministic=deterministic,
        gstreamer=gstreamer,
        gstreamer_blocked_reason=blocked_reason,
    )


def write_tracking_time_integration_json(
    result: TrackingTimeIntegrationProofResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "TrackingTimeDeterministicResult",
    "TrackingTimeGStreamerResult",
    "TrackingTimeGenerationResult",
    "TrackingTimeIntegrationProofBlocked",
    "TrackingTimeIntegrationProofError",
    "TrackingTimeIntegrationProofResult",
    "run_deterministic_tracking_time_integration_proof",
    "run_gstreamer_tracking_time_integration_proof",
    "run_tracking_time_integration_proof",
    "write_tracking_time_integration_json",
]
