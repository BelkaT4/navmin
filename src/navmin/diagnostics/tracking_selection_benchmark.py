"""Synthetic benchmark for tracking target-selection command discontinuities.

This diagnostic exercises the production ``Mediator`` target lifecycle and the
production Turret ``TurretController``/PID path synchronously over the fake STM32
transport.  It does not change control behaviour.

Two policies are evaluated:

* ``runtime``: the tracking-control config loaded from the selected NavMin config;
* ``effects-off-probe``: a diagnostic-only copy with prediction/FF disabled;
* ``active-effects-probe``: a diagnostic-only copy with prediction/FF enabled.

The two probe policies make the rate-invalid -> rate-valid boundary measurable
independently of the user's persisted runtime settings.

The synthetic camera model deliberately maps one horizontal pixel to one degree,
and the platform-motion snapshot is held at known zero.  Stage C therefore
isolates lifecycle/control discontinuities from the platform-motion timing bias
measured separately by Stage B.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from navmin.concurrency import TargetControlChannel
from navmin.config import load_config
from navmin.config.models import AppConfig, TrackingControlConfig
from navmin.contracts import (
    AxisVelocitySetpoint,
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
from navmin.turret.controller import TurretController
from navmin.turret.hal import TurretHal
from navmin.turret.session import TurretSession
from navmin.turret.simulator import FakeTransport

_BASE_NS = 10_000_000_000
_FRAME_INTERVAL_MS = 50
_AIM_X = 50
_AIM_Y = 40
_FRAME_WIDTH = 101
_FRAME_HEIGHT = 81
_OLD_TARGET_RATE_DEG_S = 20.0
_NEW_TARGET_RATE_DEG_S = -20.0
_PROBE_HORIZON_MS = 200
_PROBE_FF_GAIN = 0.25


@dataclass(frozen=True, slots=True)
class SelectionCommandEvent:
    kind: str
    timestamp_ms: int
    target_track_id: int | None
    command_x_deg_s: float | None
    real_error_x_deg: float | None
    rate_valid_x: bool | None
    target_rate_x_deg_s: float | None
    prediction_offset_x_deg: float | None
    proportional_offset_x_deg: float | None
    p_term_x_deg_s: float | None
    i_term_x_deg_s: float | None
    d_term_x_deg_s: float | None
    ff_term_x_deg_s: float | None


@dataclass(frozen=True, slots=True)
class SelectionScenarioResult:
    policy: str
    scenario: str
    prediction_enabled: bool
    prediction_horizon_ms: int
    feed_forward_enabled: bool
    ff_gain_x: float
    pid_kp_x: float
    stop_count: int
    invalidation_count: int
    pre_selection_command_x_deg_s: float
    post_stop_command_x_deg_s: float | None
    acquisition_command_x_deg_s: float
    first_valid_command_x_deg_s: float
    first_valid_elapsed_ms: int
    first_valid_rate_effect_x_deg_s: float
    stop_jump_deg_s: float
    acquisition_jump_deg_s: float
    first_valid_adjacent_jump_deg_s: float
    maximum_adjacent_command_jump_deg_s: float
    events: tuple[SelectionCommandEvent, ...]


@dataclass(frozen=True, slots=True)
class TrackingSelectionFindings:
    first_selection_has_no_stop_boundary: bool
    replacement_has_explicit_zero_stop_boundary: bool
    replacement_stop_interrupts_nonzero_command: bool
    replacement_restarts_with_rate_invalid_p_only: bool
    rate_validity_rewarms_after_replacement: bool
    active_effects_enter_at_valid_boundary: bool
    active_effects_increase_first_valid_jump: bool
    runtime_first_selection_initial_jump_deg_s: float
    runtime_replacement_stop_jump_deg_s: float
    runtime_replacement_acquisition_jump_deg_s: float
    runtime_replacement_first_valid_jump_deg_s: float
    probe_off_replacement_first_valid_jump_deg_s: float
    probe_replacement_first_valid_jump_deg_s: float
    probe_replacement_rate_effect_deg_s: float


@dataclass(frozen=True, slots=True)
class TrackingSelectionBenchmarkResult:
    status: str
    config_path: str
    fps: int
    runtime_prediction_enabled: bool
    runtime_prediction_horizon_ms: int
    runtime_feed_forward_enabled: bool
    runtime_ff_gain_x: float
    probe_prediction_horizon_ms: int
    probe_ff_gain_x: float
    scenarios: tuple[SelectionScenarioResult, ...]
    findings: TrackingSelectionFindings

    @property
    def completed(self) -> bool:
        return self.status == "COMPLETE"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _SecondsClock:
    def __init__(self) -> None:
        self.now = _BASE_NS / 1_000_000_000.0

    def set_ns(self, timestamp_ns: int) -> None:
        self.now = timestamp_ns / 1_000_000_000.0

    def __call__(self) -> float:
        return self.now


class _NanosecondsClock:
    def __init__(self) -> None:
        self.now = _BASE_NS

    def set_ns(self, timestamp_ns: int) -> None:
        self.now = timestamp_ns

    def __call__(self) -> int:
        return self.now


class _AngularCameraModel:
    """Synthetic camera: one horizontal pixel from aim point equals one degree."""

    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        horizontal_deg = float(x_px) - _AIM_X
        vertical_deg = float(y_px) - _AIM_Y
        horizontal_rad = math.radians(horizontal_deg)
        vertical_rad = math.radians(-vertical_deg)
        return CameraRay(
            math.tan(horizontal_rad),
            math.tan(vertical_rad),
            1.0,
        )

    def ray_to_pixel(self, ray: CameraRay) -> tuple[float, float]:
        horizontal_deg = math.degrees(math.atan2(ray.x, ray.z))
        vertical_deg = -math.degrees(
            math.atan2(ray.y, math.hypot(ray.x, ray.z))
        )
        return _AIM_X + horizontal_deg, _AIM_Y + vertical_deg


class _SynchronousTurretPort:
    """Mediator TurretPort backed by the real controller over FakeTransport."""

    def __init__(
        self,
        app_config: AppConfig,
        tracking_config: TrackingControlConfig,
    ) -> None:
        self._app_config = app_config
        self._seconds_clock = _SecondsClock()
        self._ns_clock = _NanosecondsClock()
        self._transport = FakeTransport()
        self._transport.open()
        self._session = TurretSession(
            self._transport,
            response_timeout_s=0.1,
            max_retries=0,
            inter_request_delay_s=0.0,
        )
        self._hal = TurretHal(
            self._session,
            app_config.turret,
            command_clock_ns=self._ns_clock,
        )
        self._controller = TurretController(
            self._hal,
            app_config.turret,
            tracking_config,
            clock=self._seconds_clock,
        )
        # Establish a known-zero platform model and enter TRACKING without a
        # velocity zero-settle transition: motor is OFF while the mode changes.
        self._controller.motor_off()
        self._controller.set_control_mode(TurretControlMode.TRACKING)
        self._controller.motor_on()
        self._transport.raw_write_history.clear()

        self._revision = 0
        self._timestamp_ns = _BASE_NS
        self._events: list[SelectionCommandEvent] = []
        self._stop_count = 0
        self._invalidation_count = 0
        self._last_command_x = 0.0
        self._zero_platform_snapshot = PlatformMotionSnapshot(
            source=PlatformMotionSource.COMMAND_MODEL,
            valid=True,
            anchors=(
                PlatformMotionAnchor(
                    timestamp_ns=_BASE_NS,
                    modeled_rate_x_deg_s=0.0,
                    modeled_rate_y_deg_s=0.0,
                    commanded_rate_x_deg_s=0.0,
                    commanded_rate_y_deg_s=0.0,
                    acceleration_x_deg_s2=app_config.turret.stm32.acceleration_x_deg_s2,
                    acceleration_y_deg_s2=app_config.turret.stm32.acceleration_y_deg_s2,
                ),
            ),
        )

    @property
    def current_state(self) -> TurretState:
        stm32 = self._app_config.turret.stm32
        return TurretState(
            connection_state=TurretConnectionState.READY,
            motor_state=MotorState.ON,
            control_mode=TurretControlMode.TRACKING,
            max_speed_x_deg_s=stm32.max_speed_x_deg_s,
            max_speed_y_deg_s=stm32.max_speed_y_deg_s,
            acceleration_x_deg_s2=stm32.acceleration_x_deg_s2,
            acceleration_y_deg_s2=stm32.acceleration_y_deg_s2,
        )

    @property
    def events(self) -> tuple[SelectionCommandEvent, ...]:
        return tuple(self._events)

    @property
    def stop_count(self) -> int:
        return self._stop_count

    @property
    def invalidation_count(self) -> int:
        return self._invalidation_count

    @property
    def last_command_x(self) -> float:
        return self._last_command_x

    def set_time_ms(self, timestamp_ms: int) -> None:
        self._timestamp_ns = _BASE_NS + timestamp_ms * 1_000_000
        self._seconds_clock.set_ns(self._timestamp_ns)
        self._ns_clock.set_ns(self._timestamp_ns)

    def submit_move_relative(self, command) -> None:  # pragma: no cover - not Stage C
        raise RuntimeError("Stage C does not submit relative moves")

    def submit_tracking_control(self, sample: TrackingControlSample) -> None:
        self.set_time_ns(sample.observation_timestamp_ns)
        self._revision += 1
        accepted = self._controller.submit_tracking_control(self._revision, sample)
        if not accepted:
            self._events.append(
                SelectionCommandEvent(
                    kind="tracking-rejected",
                    timestamp_ms=self._relative_ms(sample.observation_timestamp_ns),
                    target_track_id=sample.target.track_id,
                    command_x_deg_s=None,
                    real_error_x_deg=sample.real_error_x_deg,
                    rate_valid_x=sample.rate_valid_x,
                    target_rate_x_deg_s=sample.target_rate_x_deg_s,
                    prediction_offset_x_deg=sample.prediction_offset_x_deg,
                    proportional_offset_x_deg=None,
                    p_term_x_deg_s=None,
                    i_term_x_deg_s=None,
                    d_term_x_deg_s=None,
                    ff_term_x_deg_s=None,
                )
            )
            return
        pending = self._controller.pending_motion
        if not isinstance(pending, AxisVelocitySetpoint):
            raise TypeError("tracking control did not produce a velocity setpoint")
        step = self._controller.pid_x.last_step
        if step is None:
            raise RuntimeError("tracking control omitted PID diagnostics")
        self._last_command_x = pending.velocity_x_deg_s
        self._events.append(
            SelectionCommandEvent(
                kind="tracking",
                timestamp_ms=self._relative_ms(sample.observation_timestamp_ns),
                target_track_id=sample.target.track_id,
                command_x_deg_s=pending.velocity_x_deg_s,
                real_error_x_deg=sample.real_error_x_deg,
                rate_valid_x=sample.rate_valid_x,
                target_rate_x_deg_s=sample.target_rate_x_deg_s,
                prediction_offset_x_deg=sample.prediction_offset_x_deg,
                proportional_offset_x_deg=step.proportional_offset,
                p_term_x_deg_s=step.p_term,
                i_term_x_deg_s=step.i_term,
                d_term_x_deg_s=step.d_term,
                ff_term_x_deg_s=step.feed_forward,
            )
        )
        result = self._controller.flush_pending_motion()
        if result is None or result.response is None:
            raise RuntimeError("tracking velocity did not reach fake STM32")

    def invalidate_tracking_control(self) -> None:
        self._revision += 1
        self._controller.invalidate_tracking_control(self._revision)
        self._invalidation_count += 1
        self._events.append(
            SelectionCommandEvent(
                kind="invalidate",
                timestamp_ms=self._relative_ms(self._timestamp_ns),
                target_track_id=None,
                command_x_deg_s=None,
                real_error_x_deg=None,
                rate_valid_x=None,
                target_rate_x_deg_s=None,
                prediction_offset_x_deg=None,
                proportional_offset_x_deg=None,
                p_term_x_deg_s=None,
                i_term_x_deg_s=None,
                d_term_x_deg_s=None,
                ff_term_x_deg_s=None,
            )
        )

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot:
        # Deliberately fixed at zero to isolate selection lifecycle from the
        # platform-timing bias already covered by Stage B.
        return self._zero_platform_snapshot

    def set_control_mode(self, mode: TurretControlMode) -> bool:
        return mode is TurretControlMode.TRACKING

    def stop_motion(self) -> bool:
        result = self._controller.stop_motion()
        self._stop_count += 1
        self._last_command_x = 0.0
        self._events.append(
            SelectionCommandEvent(
                kind="stop",
                timestamp_ms=self._relative_ms(self._timestamp_ns),
                target_track_id=None,
                command_x_deg_s=0.0,
                real_error_x_deg=None,
                rate_valid_x=None,
                target_rate_x_deg_s=None,
                prediction_offset_x_deg=None,
                proportional_offset_x_deg=None,
                p_term_x_deg_s=None,
                i_term_x_deg_s=None,
                d_term_x_deg_s=None,
                ff_term_x_deg_s=None,
            )
        )
        return result is not None and result.response is not None

    def motor_on(self) -> bool:  # pragma: no cover - not Stage C
        return True

    def motor_off(self) -> bool:  # pragma: no cover - not Stage C
        return True

    def request_emergency(self) -> None:  # pragma: no cover - not Stage C
        return None

    def set_time_ns(self, timestamp_ns: int) -> None:
        self._timestamp_ns = timestamp_ns
        self._seconds_clock.set_ns(timestamp_ns)
        self._ns_clock.set_ns(timestamp_ns)

    @staticmethod
    def _relative_ms(timestamp_ns: int) -> int:
        return round((timestamp_ns - _BASE_NS) / 1_000_000)


def _tracking_config_for_policy(
    runtime: TrackingControlConfig,
    policy: str,
) -> TrackingControlConfig:
    if policy == "runtime":
        return runtime
    if policy == "effects-off-probe":
        return replace(
            runtime,
            prediction=replace(runtime.prediction, enabled=False, horizon_ms=0),
            feed_forward=replace(
                runtime.feed_forward,
                enabled=False,
                gain_x=0.0,
            ),
        )
    if policy == "active-effects-probe":
        return replace(
            runtime,
            prediction=replace(
                runtime.prediction,
                enabled=True,
                horizon_ms=_PROBE_HORIZON_MS,
            ),
            feed_forward=replace(
                runtime.feed_forward,
                enabled=True,
                gain_x=_PROBE_FF_GAIN,
            ),
        )
    raise ValueError(f"unsupported Stage C policy: {policy}")


def _mediator(
    app_config: AppConfig,
    tracking_config: TrackingControlConfig,
) -> tuple[Mediator, _SynchronousTurretPort, dict[CameraRole, TargetControlChannel]]:
    turret = _SynchronousTurretPort(app_config, tracking_config)
    controls = {
        CameraRole.OVERVIEW: TargetControlChannel(CameraRole.OVERVIEW),
        CameraRole.STEREO_LEFT: TargetControlChannel(CameraRole.STEREO_LEFT),
    }
    mediator = Mediator(
        aiming_config=app_config.aiming,
        tracking_control_config=tracking_config,
        tracking_gap_timeout_ms=app_config.turret.stm32.velocity_watchdog_timeout_ms,
        ui_config=replace(app_config.ui, default_camera=CameraRole.OVERVIEW),
        turret=turret,
        target_controls=controls,
        vision_processing_enabled={
            CameraRole.OVERVIEW: True,
            CameraRole.STEREO_LEFT: True,
        },
    )
    session = CameraSessionStarted(
        camera=CameraRole.OVERVIEW,
        generation=1,
        camera_model=_AngularCameraModel(),
        timestamp_ns=_BASE_NS,
    )
    if not mediator.accept_camera_session(session):
        raise RuntimeError("Stage C could not establish synthetic camera session")
    return mediator, turret, controls


def _track(track_id: int, error_deg: float, velocity_deg_s: float) -> TrackedObject:
    center_x = _AIM_X + error_deg
    bbox_x = round(center_x - 1.0)
    return TrackedObject(
        track_id=track_id,
        bbox=BBox(bbox_x, _AIM_Y - 1, 2, 2),
        velocity_x_px_s=velocity_deg_s,
        velocity_y_px_s=0.0,
        age_frames=10,
    )


def _vision_result(
    *,
    timestamp_ms: int,
    frame_id: int,
    tracks: tuple[tuple[int, float, float], ...],
) -> VisionResult:
    image = np.zeros((_FRAME_HEIGHT, _FRAME_WIDTH, 3), dtype=np.uint8)
    image.flags.writeable = False
    frame = FramePacket(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=frame_id,
        capture_id=None,
        receive_timestamp_ns=_BASE_NS + timestamp_ms * 1_000_000,
        image=image,
    )
    return VisionResult(
        frame=frame,
        tracked_objects=tuple(_track(*item) for item in tracks),
        processing_time_ns=0,
    )


def _acquire_pending(
    mediator: Mediator,
    *,
    track_id: int,
    timestamp_ms: int,
) -> None:
    request = mediator.pending_selection
    if request is None:
        raise RuntimeError("Stage C selection request was not published")
    event = TargetAcquired(
        selection_id=request.selection_id,
        target=TargetRef(request.camera, request.generation, track_id),
        request_frame_id=request.frame_id,
        decision_frame_id=request.frame_id,
        decision_frame_timestamp_ns=_BASE_NS + timestamp_ms * 1_000_000,
    )
    if not mediator.accept_target_lifecycle_event(event):
        raise RuntimeError("Stage C synthetic TargetAcquired was rejected")


def _request_target(
    mediator: Mediator,
    turret: _SynchronousTurretPort,
    result: VisionResult,
    *,
    x_px: float,
    timestamp_ms: int,
) -> None:
    turret.set_time_ms(timestamp_ms)
    if not mediator.request_target_selection(result, x_px, _AIM_Y):
        raise RuntimeError("Stage C synthetic target selection was rejected")


def _tracking_events(
    events: tuple[SelectionCommandEvent, ...],
    *,
    track_id: int,
) -> list[SelectionCommandEvent]:
    return [
        event
        for event in events
        if event.kind == "tracking" and event.target_track_id == track_id
    ]


def _first_valid(events: list[SelectionCommandEvent]) -> tuple[int, SelectionCommandEvent]:
    for index, event in enumerate(events):
        if event.rate_valid_x:
            return index, event
    raise RuntimeError("Stage C target never reached a valid rate estimate")


def _adjacent_command_jump(events: list[SelectionCommandEvent], index: int) -> float:
    if index <= 0:
        previous = 0.0
    else:
        previous_value = events[index - 1].command_x_deg_s
        if previous_value is None:
            raise RuntimeError("previous tracking event omitted command")
        previous = previous_value
    current = events[index].command_x_deg_s
    if current is None:
        raise RuntimeError("tracking event omitted command")
    return abs(current - previous)


def _max_adjacent_command_jump(events: tuple[SelectionCommandEvent, ...]) -> float:
    commands = [event.command_x_deg_s for event in events if event.command_x_deg_s is not None]
    if not commands:
        return 0.0
    maximum = abs(commands[0])
    for previous, current in pairwise(commands):
        maximum = max(maximum, abs(current - previous))
    return maximum


def _rate_effect(event: SelectionCommandEvent, kp_x: float) -> float:
    prediction = event.proportional_offset_x_deg or 0.0
    feed_forward = event.ff_term_x_deg_s or 0.0
    return kp_x * prediction + feed_forward


def _run_first_selection(
    app_config: AppConfig,
    tracking_config: TrackingControlConfig,
    *,
    policy: str,
) -> SelectionScenarioResult:
    mediator, turret, _controls = _mediator(app_config, tracking_config)
    first = _vision_result(
        timestamp_ms=0,
        frame_id=1,
        tracks=((1, 8.0, _OLD_TARGET_RATE_DEG_S),),
    )
    turret.set_time_ms(0)
    if not mediator.accept_vision_result(first):
        raise RuntimeError("Stage C initial frame rejected")
    before_stop = turret.stop_count
    before_invalidations = turret.invalidation_count
    _request_target(mediator, turret, first, x_px=58.0, timestamp_ms=0)
    _acquire_pending(mediator, track_id=1, timestamp_ms=0)

    frame_id = 2
    for timestamp_ms in range(_FRAME_INTERVAL_MS, 301, _FRAME_INTERVAL_MS):
        error = 8.0 + _OLD_TARGET_RATE_DEG_S * timestamp_ms / 1000.0
        turret.set_time_ms(timestamp_ms)
        result = _vision_result(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            tracks=((1, error, _OLD_TARGET_RATE_DEG_S),),
        )
        frame_id += 1
        if not mediator.accept_vision_result(result):
            raise RuntimeError("Stage C first-selection tracking frame rejected")

    tracking = _tracking_events(turret.events, track_id=1)
    if not tracking:
        raise RuntimeError("Stage C first selection produced no tracking control")
    acquisition = tracking[0]
    if acquisition.command_x_deg_s is None:
        raise RuntimeError("Stage C acquisition command unavailable")
    valid_index, first_valid = _first_valid(tracking)
    if first_valid.command_x_deg_s is None:
        raise RuntimeError("Stage C first valid command unavailable")
    kp = app_config.turret.controller.pid_kp_x
    return SelectionScenarioResult(
        policy=policy,
        scenario="first-selection",
        prediction_enabled=tracking_config.prediction.enabled,
        prediction_horizon_ms=tracking_config.prediction.horizon_ms,
        feed_forward_enabled=tracking_config.feed_forward.enabled,
        ff_gain_x=tracking_config.feed_forward.gain_x,
        pid_kp_x=kp,
        stop_count=turret.stop_count - before_stop,
        invalidation_count=turret.invalidation_count - before_invalidations,
        pre_selection_command_x_deg_s=0.0,
        post_stop_command_x_deg_s=None,
        acquisition_command_x_deg_s=acquisition.command_x_deg_s,
        first_valid_command_x_deg_s=first_valid.command_x_deg_s,
        first_valid_elapsed_ms=first_valid.timestamp_ms,
        first_valid_rate_effect_x_deg_s=_rate_effect(first_valid, kp),
        stop_jump_deg_s=0.0,
        acquisition_jump_deg_s=abs(acquisition.command_x_deg_s),
        first_valid_adjacent_jump_deg_s=_adjacent_command_jump(tracking, valid_index),
        maximum_adjacent_command_jump_deg_s=_max_adjacent_command_jump(turret.events),
        events=turret.events,
    )


def _run_replacement_selection(
    app_config: AppConfig,
    tracking_config: TrackingControlConfig,
    *,
    policy: str,
) -> SelectionScenarioResult:
    mediator, turret, _controls = _mediator(app_config, tracking_config)
    initial = _vision_result(
        timestamp_ms=0,
        frame_id=1,
        tracks=((1, 8.0, _OLD_TARGET_RATE_DEG_S),),
    )
    turret.set_time_ms(0)
    if not mediator.accept_vision_result(initial):
        raise RuntimeError("Stage C initial replacement frame rejected")
    _request_target(mediator, turret, initial, x_px=58.0, timestamp_ms=0)
    _acquire_pending(mediator, track_id=1, timestamp_ms=0)

    frame_id = 2
    for timestamp_ms in range(_FRAME_INTERVAL_MS, 351, _FRAME_INTERVAL_MS):
        old_error = 8.0 + _OLD_TARGET_RATE_DEG_S * timestamp_ms / 1000.0
        tracks: tuple[tuple[int, float, float], ...]
        if timestamp_ms == 350:
            tracks = (
                (1, old_error, _OLD_TARGET_RATE_DEG_S),
                (2, -8.0, _NEW_TARGET_RATE_DEG_S),
            )
        else:
            tracks = ((1, old_error, _OLD_TARGET_RATE_DEG_S),)
        turret.set_time_ms(timestamp_ms)
        latest = _vision_result(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            tracks=tracks,
        )
        frame_id += 1
        if not mediator.accept_vision_result(latest):
            raise RuntimeError("Stage C pre-replacement frame rejected")

    pre_selection_command = turret.last_command_x
    before_stop = turret.stop_count
    before_invalidations = turret.invalidation_count
    _request_target(mediator, turret, latest, x_px=42.0, timestamp_ms=350)
    _acquire_pending(mediator, track_id=2, timestamp_ms=350)

    for elapsed_ms in range(_FRAME_INTERVAL_MS, 301, _FRAME_INTERVAL_MS):
        timestamp_ms = 350 + elapsed_ms
        new_error = -8.0 + _NEW_TARGET_RATE_DEG_S * elapsed_ms / 1000.0
        turret.set_time_ms(timestamp_ms)
        result = _vision_result(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            tracks=((2, new_error, _NEW_TARGET_RATE_DEG_S),),
        )
        frame_id += 1
        if not mediator.accept_vision_result(result):
            raise RuntimeError("Stage C post-replacement frame rejected")

    tracking = _tracking_events(turret.events, track_id=2)
    if not tracking:
        raise RuntimeError("Stage C replacement produced no new-target control")
    acquisition = tracking[0]
    if acquisition.command_x_deg_s is None:
        raise RuntimeError("Stage C replacement acquisition command unavailable")
    valid_index, first_valid = _first_valid(tracking)
    if first_valid.command_x_deg_s is None:
        raise RuntimeError("Stage C replacement first valid command unavailable")
    stop_events = [event for event in turret.events if event.kind == "stop"]
    post_stop = stop_events[-1].command_x_deg_s if stop_events else None
    kp = app_config.turret.controller.pid_kp_x
    return SelectionScenarioResult(
        policy=policy,
        scenario="replacement-selection",
        prediction_enabled=tracking_config.prediction.enabled,
        prediction_horizon_ms=tracking_config.prediction.horizon_ms,
        feed_forward_enabled=tracking_config.feed_forward.enabled,
        ff_gain_x=tracking_config.feed_forward.gain_x,
        pid_kp_x=kp,
        stop_count=turret.stop_count - before_stop,
        invalidation_count=turret.invalidation_count - before_invalidations,
        pre_selection_command_x_deg_s=pre_selection_command,
        post_stop_command_x_deg_s=post_stop,
        acquisition_command_x_deg_s=acquisition.command_x_deg_s,
        first_valid_command_x_deg_s=first_valid.command_x_deg_s,
        first_valid_elapsed_ms=first_valid.timestamp_ms - 350,
        first_valid_rate_effect_x_deg_s=_rate_effect(first_valid, kp),
        stop_jump_deg_s=abs(pre_selection_command),
        acquisition_jump_deg_s=abs(acquisition.command_x_deg_s),
        first_valid_adjacent_jump_deg_s=_adjacent_command_jump(tracking, valid_index),
        maximum_adjacent_command_jump_deg_s=_max_adjacent_command_jump(turret.events),
        events=turret.events,
    )


def _scenario(
    scenarios: tuple[SelectionScenarioResult, ...],
    policy: str,
    scenario: str,
) -> SelectionScenarioResult:
    return next(
        item
        for item in scenarios
        if item.policy == policy and item.scenario == scenario
    )


def run_tracking_selection_benchmark(
    config_path: str | Path = "config/navmin.json",
) -> TrackingSelectionBenchmarkResult:
    path = Path(config_path)
    app_config = load_config(path)
    scenarios: list[SelectionScenarioResult] = []
    for policy in ("runtime", "effects-off-probe", "active-effects-probe"):
        tracking_config = _tracking_config_for_policy(app_config.tracking_control, policy)
        scenarios.append(
            _run_first_selection(app_config, tracking_config, policy=policy)
        )
        scenarios.append(
            _run_replacement_selection(app_config, tracking_config, policy=policy)
        )
    scenario_tuple = tuple(scenarios)
    runtime_first = _scenario(scenario_tuple, "runtime", "first-selection")
    runtime_replace = _scenario(
        scenario_tuple, "runtime", "replacement-selection"
    )
    off_replace = _scenario(
        scenario_tuple, "effects-off-probe", "replacement-selection"
    )
    probe_replace = _scenario(
        scenario_tuple, "active-effects-probe", "replacement-selection"
    )
    runtime_replacement_events = [
        event
        for event in runtime_replace.events
        if event.kind == "tracking" and event.target_track_id == 2
    ]
    runtime_replacement_first = runtime_replacement_events[0]
    findings = TrackingSelectionFindings(
        first_selection_has_no_stop_boundary=runtime_first.stop_count == 0,
        replacement_has_explicit_zero_stop_boundary=(
            runtime_replace.stop_count == 1
            and runtime_replace.post_stop_command_x_deg_s == 0.0
        ),
        replacement_stop_interrupts_nonzero_command=(
            runtime_replace.stop_jump_deg_s > 0.0
        ),
        replacement_restarts_with_rate_invalid_p_only=(
            runtime_replacement_first.rate_valid_x is False
            and (runtime_replacement_first.proportional_offset_x_deg or 0.0) == 0.0
            and (runtime_replacement_first.ff_term_x_deg_s or 0.0) == 0.0
        ),
        rate_validity_rewarms_after_replacement=(
            runtime_replace.first_valid_elapsed_ms >= app_config.tracking_control.velocity_estimator.min_span_ms
        ),
        active_effects_enter_at_valid_boundary=(
            abs(probe_replace.first_valid_rate_effect_x_deg_s) > 0.1
        ),
        active_effects_increase_first_valid_jump=(
            probe_replace.first_valid_adjacent_jump_deg_s
            > off_replace.first_valid_adjacent_jump_deg_s + 0.5
        ),
        runtime_first_selection_initial_jump_deg_s=runtime_first.acquisition_jump_deg_s,
        runtime_replacement_stop_jump_deg_s=runtime_replace.stop_jump_deg_s,
        runtime_replacement_acquisition_jump_deg_s=runtime_replace.acquisition_jump_deg_s,
        runtime_replacement_first_valid_jump_deg_s=(
            runtime_replace.first_valid_adjacent_jump_deg_s
        ),
        probe_off_replacement_first_valid_jump_deg_s=(
            off_replace.first_valid_adjacent_jump_deg_s
        ),
        probe_replacement_first_valid_jump_deg_s=(
            probe_replace.first_valid_adjacent_jump_deg_s
        ),
        probe_replacement_rate_effect_deg_s=(
            probe_replace.first_valid_rate_effect_x_deg_s
        ),
    )
    return TrackingSelectionBenchmarkResult(
        status="COMPLETE",
        config_path=str(path),
        fps=round(1000 / _FRAME_INTERVAL_MS),
        runtime_prediction_enabled=app_config.tracking_control.prediction.enabled,
        runtime_prediction_horizon_ms=app_config.tracking_control.prediction.horizon_ms,
        runtime_feed_forward_enabled=app_config.tracking_control.feed_forward.enabled,
        runtime_ff_gain_x=app_config.tracking_control.feed_forward.gain_x,
        probe_prediction_horizon_ms=_PROBE_HORIZON_MS,
        probe_ff_gain_x=_PROBE_FF_GAIN,
        scenarios=scenario_tuple,
        findings=findings,
    )


def write_tracking_selection_benchmark_json(
    result: TrackingSelectionBenchmarkResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "SelectionCommandEvent",
    "SelectionScenarioResult",
    "TrackingSelectionBenchmarkResult",
    "TrackingSelectionFindings",
    "run_tracking_selection_benchmark",
    "write_tracking_selection_benchmark_json",
]
