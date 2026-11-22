"""Deterministic virtual turret plant for the synthetic tracking world.

The plant is deliberately independent from ``CommandMotionModel``. It consumes
successfully executed STM32 protocol requests and integrates a separate virtual
physical state so closed-loop diagnostics can compare what NavMin predicts with
what the synthetic turret actually does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from time import monotonic

from navmin.config.models import AxisMechanicsConfig, TurretConfig
from navmin.contracts import CameraRole
from navmin.turret.protocol import (
    CommandCode,
    MoveRelativePayload,
    ProtocolRequest,
    ProtocolResponse,
    ResultCode,
    SetConfigPayload,
    SetVelocityPayload,
)
from navmin.turret.simulator import FakeStm32Endpoint
from navmin.vision.pipeline import DecodedFrame

from .synthetic_tracking_world import SyntheticTrackingFrameSet, SyntheticTrackingWorld

_RELATIVE_CONTROL_RATE_HZ = 20_000
_RELATIVE_CONTROL_TICK_S = 1.0 / _RELATIVE_CONTROL_RATE_HZ


@dataclass(frozen=True, slots=True)
class SyntheticTurretPlantState:
    configured: bool
    motors_on: bool
    yaw_deg: float
    pitch_deg: float
    yaw_rate_deg_s: float
    pitch_rate_deg_s: float
    requested_yaw_rate_deg_s: float
    requested_pitch_rate_deg_s: float
    max_speed_yaw_deg_s: float
    max_speed_pitch_deg_s: float
    acceleration_yaw_deg_s2: float
    acceleration_pitch_deg_s2: float
    velocity_watchdog_timeout_s: float
    velocity_command_age_s: float | None


class SyntheticTurretPlant:
    """Acceleration-limited yaw/pitch plant driven by executed protocol payloads."""

    def __init__(self, config: TurretConfig) -> None:
        if not isinstance(config, TurretConfig):
            raise TypeError("config must be TurretConfig")
        self._mechanics_x = config.axes.x
        self._mechanics_y = config.axes.y
        self._lock = Lock()
        self._configured = False
        self._motors_on = False
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._yaw_rate_deg_s = 0.0
        self._pitch_rate_deg_s = 0.0
        self._requested_yaw_rate_deg_s = 0.0
        self._requested_pitch_rate_deg_s = 0.0
        self._max_speed_yaw_deg_s = 0.0
        self._max_speed_pitch_deg_s = 0.0
        self._acceleration_yaw_deg_s2 = 0.0
        self._acceleration_pitch_deg_s2 = 0.0
        self._velocity_watchdog_timeout_s = 0.0
        self._velocity_command_age_s: float | None = None
        self._relative_target_yaw_deg: float | None = None
        self._relative_target_pitch_deg: float | None = None

    @property
    def state(self) -> SyntheticTurretPlantState:
        with self._lock:
            return self._state_locked()

    def apply_executed_request(self, request: ProtocolRequest) -> None:
        """Apply one successfully executed STM32 request to the virtual plant."""
        if not isinstance(request, ProtocolRequest):
            raise TypeError("request must be ProtocolRequest")
        with self._lock:
            if request.command is CommandCode.SET_CONFIG:
                payload = request.payload
                if not isinstance(payload, SetConfigPayload):
                    raise TypeError("SET_CONFIG requires SetConfigPayload")
                self._apply_config_locked(payload)
                return
            if request.command is CommandCode.MOTOR_ON:
                if self._configured:
                    self._motors_on = True
                return
            if request.command is CommandCode.MOTOR_OFF:
                self._motors_on = False
                self._hard_stop_locked()
                return
            if request.command is CommandCode.EMERGENCY_STOP:
                self._hard_stop_locked()
                return
            if request.command is CommandCode.MOVE_RELATIVE:
                payload = request.payload
                if not isinstance(payload, MoveRelativePayload):
                    raise TypeError("MOVE_RELATIVE requires MoveRelativePayload")
                if not self._configured or not self._motors_on:
                    return
                self._relative_target_yaw_deg = self._yaw_deg + self._logical_degrees_from_steps(
                    payload.delta_x_steps,
                    self._mechanics_x,
                )
                self._relative_target_pitch_deg = (
                    self._pitch_deg
                    + self._logical_degrees_from_steps(
                        payload.delta_y_steps,
                        self._mechanics_y,
                    )
                )
                # Firmware relative ownership disarms the velocity watchdog.  The
                # current physical rate is deliberately preserved and the planner
                # brakes/reverses from that state on subsequent control ticks.
                self._velocity_command_age_s = None
                return
            if request.command is CommandCode.SET_VELOCITY:
                payload = request.payload
                if not isinstance(payload, SetVelocityPayload):
                    raise TypeError("SET_VELOCITY requires SetVelocityPayload")
                if not self._configured or not self._motors_on:
                    return
                self._clear_relative_target_locked()
                self._requested_yaw_rate_deg_s = self._clamp(
                    self._logical_rate_from_steps_s(
                        payload.velocity_x_steps_s, self._mechanics_x
                    ),
                    self._max_speed_yaw_deg_s,
                )
                self._requested_pitch_rate_deg_s = self._clamp(
                    self._logical_rate_from_steps_s(
                        payload.velocity_y_steps_s, self._mechanics_y
                    ),
                    self._max_speed_pitch_deg_s,
                )
                self._velocity_command_age_s = 0.0

    def advance(self, dt_s: float) -> SyntheticTurretPlantState:
        """Advance the physical state by ``dt_s`` using exact ramp integration."""
        if not isfinite(dt_s) or dt_s < 0.0:
            raise ValueError("dt_s must be finite and >= 0")
        with self._lock:
            if dt_s == 0.0 or not self._configured or not self._motors_on:
                return self._state_locked()

            if self._relative_target_yaw_deg is not None:
                self._advance_relative_motion_locked(dt_s)
                return self._state_locked()

            remaining_s = dt_s
            if self._velocity_command_age_s is not None:
                until_watchdog_s = (
                    self._velocity_watchdog_timeout_s - self._velocity_command_age_s
                )
                if until_watchdog_s <= 0.0:
                    self._requested_yaw_rate_deg_s = 0.0
                    self._requested_pitch_rate_deg_s = 0.0
                elif until_watchdog_s < remaining_s:
                    self._advance_motion_locked(until_watchdog_s)
                    self._velocity_command_age_s += until_watchdog_s
                    remaining_s -= until_watchdog_s
                    self._requested_yaw_rate_deg_s = 0.0
                    self._requested_pitch_rate_deg_s = 0.0

            if remaining_s > 0.0:
                self._advance_motion_locked(remaining_s)
                if self._velocity_command_age_s is not None:
                    self._velocity_command_age_s += remaining_s
            return self._state_locked()

    def _advance_motion_locked(self, dt_s: float) -> None:
        self._yaw_deg, self._yaw_rate_deg_s = self._advance_axis(
            position_deg=self._yaw_deg,
            rate_deg_s=self._yaw_rate_deg_s,
            target_rate_deg_s=self._requested_yaw_rate_deg_s,
            acceleration_deg_s2=self._acceleration_yaw_deg_s2,
            dt_s=dt_s,
        )
        self._pitch_deg, self._pitch_rate_deg_s = self._advance_axis(
            position_deg=self._pitch_deg,
            rate_deg_s=self._pitch_rate_deg_s,
            target_rate_deg_s=self._requested_pitch_rate_deg_s,
            acceleration_deg_s2=self._acceleration_pitch_deg_s2,
            dt_s=dt_s,
        )

    def _advance_relative_motion_locked(self, dt_s: float) -> None:
        target_yaw = self._relative_target_yaw_deg
        target_pitch = self._relative_target_pitch_deg
        if target_yaw is None or target_pitch is None:
            raise RuntimeError("relative target must be complete")

        remaining_s = dt_s
        yaw_settled = False
        pitch_settled = False
        while remaining_s > 0.0:
            tick_s = min(remaining_s, _RELATIVE_CONTROL_TICK_S)
            (
                self._yaw_deg,
                self._yaw_rate_deg_s,
                self._requested_yaw_rate_deg_s,
                yaw_settled,
            ) = self._advance_relative_axis_tick(
                position_deg=self._yaw_deg,
                rate_deg_s=self._yaw_rate_deg_s,
                target_position_deg=target_yaw,
                max_speed_deg_s=self._max_speed_yaw_deg_s,
                acceleration_deg_s2=self._acceleration_yaw_deg_s2,
                mechanics=self._mechanics_x,
                dt_s=tick_s,
            )
            (
                self._pitch_deg,
                self._pitch_rate_deg_s,
                self._requested_pitch_rate_deg_s,
                pitch_settled,
            ) = self._advance_relative_axis_tick(
                position_deg=self._pitch_deg,
                rate_deg_s=self._pitch_rate_deg_s,
                target_position_deg=target_pitch,
                max_speed_deg_s=self._max_speed_pitch_deg_s,
                acceleration_deg_s2=self._acceleration_pitch_deg_s2,
                mechanics=self._mechanics_y,
                dt_s=tick_s,
            )
            remaining_s -= tick_s
            if yaw_settled and pitch_settled:
                self._clear_relative_target_locked()
                self._requested_yaw_rate_deg_s = 0.0
                self._requested_pitch_rate_deg_s = 0.0
                return

    @classmethod
    def _advance_relative_axis_tick(
        cls,
        *,
        position_deg: float,
        rate_deg_s: float,
        target_position_deg: float,
        max_speed_deg_s: float,
        acceleration_deg_s2: float,
        mechanics: AxisMechanicsConfig,
        dt_s: float,
    ) -> tuple[float, float, float, bool]:
        step_deg = 360.0 / cls._effective_steps_per_revolution(mechanics)
        settle_tolerance_deg = 0.5 * step_deg
        remaining_deg = target_position_deg - position_deg

        if abs(remaining_deg) <= settle_tolerance_deg and rate_deg_s == 0.0:
            return target_position_deg, 0.0, 0.0, True

        if acceleration_deg_s2 <= 0.0:
            raise RuntimeError("configured acceleration must be > 0")

        if remaining_deg > 0.0:
            direction = 1.0
        elif remaining_deg < 0.0:
            direction = -1.0
        else:
            direction = 0.0

        if direction == 0.0:
            target_rate_deg_s = 0.0
        elif rate_deg_s * direction < 0.0:
            # Match firmware: when motion is away from the new target, request
            # full speed toward it and let the acceleration limiter brake
            # through exact zero before opposite acceleration begins.
            target_rate_deg_s = direction * max_speed_deg_s
        else:
            braking_distance_deg = (
                rate_deg_s * rate_deg_s / (2.0 * acceleration_deg_s2)
            )
            target_rate_deg_s = (
                0.0
                if braking_distance_deg >= abs(remaining_deg)
                else direction * max_speed_deg_s
            )

        next_rate = cls._advance_relative_rate(
            rate_deg_s=rate_deg_s,
            target_rate_deg_s=target_rate_deg_s,
            acceleration_deg_s2=acceleration_deg_s2,
            dt_s=dt_s,
        )
        next_position = position_deg + next_rate * dt_s

        if (
            next_rate == 0.0
            and abs(target_position_deg - next_position) <= settle_tolerance_deg
        ):
            return target_position_deg, 0.0, 0.0, True
        return next_position, next_rate, target_rate_deg_s, False

    @staticmethod
    def _advance_relative_rate(
        *,
        rate_deg_s: float,
        target_rate_deg_s: float,
        acceleration_deg_s2: float,
        dt_s: float,
    ) -> float:
        delta = acceleration_deg_s2 * dt_s
        if rate_deg_s == target_rate_deg_s:
            return rate_deg_s

        # Firmware discards unused acceleration budget on a sign reversal.
        if (rate_deg_s > 0.0 > target_rate_deg_s) or (
            rate_deg_s < 0.0 < target_rate_deg_s
        ):
            if rate_deg_s > 0.0:
                return 0.0 if rate_deg_s <= delta else rate_deg_s - delta
            return 0.0 if rate_deg_s >= -delta else rate_deg_s + delta

        if rate_deg_s < target_rate_deg_s:
            return min(target_rate_deg_s, rate_deg_s + delta)
        return max(target_rate_deg_s, rate_deg_s - delta)

    def _apply_config_locked(self, payload: SetConfigPayload) -> None:
        self._max_speed_yaw_deg_s = self._magnitude_from_steps(
            payload.max_speed_x_steps_s,
            self._mechanics_x,
        )
        self._max_speed_pitch_deg_s = self._magnitude_from_steps(
            payload.max_speed_y_steps_s,
            self._mechanics_y,
        )
        self._acceleration_yaw_deg_s2 = self._magnitude_from_steps(
            payload.acceleration_x_steps_s2,
            self._mechanics_x,
        )
        self._acceleration_pitch_deg_s2 = self._magnitude_from_steps(
            payload.acceleration_y_steps_s2,
            self._mechanics_y,
        )
        self._velocity_watchdog_timeout_s = payload.velocity_watchdog_timeout_ms / 1000.0
        self._requested_yaw_rate_deg_s = self._clamp(
            self._requested_yaw_rate_deg_s,
            self._max_speed_yaw_deg_s,
        )
        self._requested_pitch_rate_deg_s = self._clamp(
            self._requested_pitch_rate_deg_s,
            self._max_speed_pitch_deg_s,
        )
        self._configured = True

    def _hard_stop_locked(self) -> None:
        self._clear_relative_target_locked()
        self._yaw_rate_deg_s = 0.0
        self._pitch_rate_deg_s = 0.0
        self._requested_yaw_rate_deg_s = 0.0
        self._requested_pitch_rate_deg_s = 0.0
        self._velocity_command_age_s = None

    def _clear_relative_target_locked(self) -> None:
        self._relative_target_yaw_deg = None
        self._relative_target_pitch_deg = None

    def _state_locked(self) -> SyntheticTurretPlantState:
        return SyntheticTurretPlantState(
            configured=self._configured,
            motors_on=self._motors_on,
            yaw_deg=self._yaw_deg,
            pitch_deg=self._pitch_deg,
            yaw_rate_deg_s=self._yaw_rate_deg_s,
            pitch_rate_deg_s=self._pitch_rate_deg_s,
            requested_yaw_rate_deg_s=self._requested_yaw_rate_deg_s,
            requested_pitch_rate_deg_s=self._requested_pitch_rate_deg_s,
            max_speed_yaw_deg_s=self._max_speed_yaw_deg_s,
            max_speed_pitch_deg_s=self._max_speed_pitch_deg_s,
            acceleration_yaw_deg_s2=self._acceleration_yaw_deg_s2,
            acceleration_pitch_deg_s2=self._acceleration_pitch_deg_s2,
            velocity_watchdog_timeout_s=self._velocity_watchdog_timeout_s,
            velocity_command_age_s=self._velocity_command_age_s,
        )

    @staticmethod
    def _effective_steps_per_revolution(mechanics: AxisMechanicsConfig) -> int:
        return mechanics.full_steps_per_revolution * mechanics.microstep_divider

    @classmethod
    def _logical_rate_from_steps_s(
        cls,
        physical_steps_s: int,
        mechanics: AxisMechanicsConfig,
    ) -> float:
        logical_steps_s = -physical_steps_s if mechanics.invert else physical_steps_s
        return logical_steps_s * 360.0 / cls._effective_steps_per_revolution(mechanics)

    @classmethod
    def _logical_degrees_from_steps(
        cls,
        physical_steps: int,
        mechanics: AxisMechanicsConfig,
    ) -> float:
        logical_steps = -physical_steps if mechanics.invert else physical_steps
        return logical_steps * 360.0 / cls._effective_steps_per_revolution(mechanics)

    @classmethod
    def _magnitude_from_steps(
        cls,
        steps: int,
        mechanics: AxisMechanicsConfig,
    ) -> float:
        return float(steps) * 360.0 / cls._effective_steps_per_revolution(mechanics)

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _advance_axis(
        *,
        position_deg: float,
        rate_deg_s: float,
        target_rate_deg_s: float,
        acceleration_deg_s2: float,
        dt_s: float,
    ) -> tuple[float, float]:
        delta = target_rate_deg_s - rate_deg_s
        if delta == 0.0:
            return position_deg + rate_deg_s * dt_s, rate_deg_s
        if acceleration_deg_s2 <= 0.0:
            raise RuntimeError("configured acceleration must be > 0")
        direction = 1.0 if delta > 0.0 else -1.0
        ramp_s = min(dt_s, abs(delta) / acceleration_deg_s2)
        next_rate = rate_deg_s + direction * acceleration_deg_s2 * ramp_s
        if ramp_s < dt_s:
            next_rate = target_rate_deg_s
        displacement = (
            rate_deg_s * ramp_s
            + 0.5 * direction * acceleration_deg_s2 * ramp_s * ramp_s
            + next_rate * (dt_s - ramp_s)
        )
        return position_deg + displacement, next_rate


@dataclass(frozen=True, slots=True)
class SyntheticTrackingSimulationSnapshot:
    frame_set: SyntheticTrackingFrameSet
    turret: SyntheticTurretPlantState


class SyntheticTrackingSimulation:
    """Single-clock bridge between the world renderer and virtual turret plant."""

    def __init__(
        self,
        world: SyntheticTrackingWorld,
        plant: SyntheticTurretPlant,
    ) -> None:
        if not isinstance(world, SyntheticTrackingWorld):
            raise TypeError("world must be SyntheticTrackingWorld")
        if not isinstance(plant, SyntheticTurretPlant):
            raise TypeError("plant must be SyntheticTurretPlant")
        self.world = world
        self.plant = plant
        self.frame_index = 0

    def capture(self) -> SyntheticTrackingSimulationSnapshot:
        state = self.plant.state
        frame_set = self.world.render(
            self.frame_index,
            turret_yaw_deg=state.yaw_deg,
            turret_pitch_deg=state.pitch_deg,
        )
        return SyntheticTrackingSimulationSnapshot(frame_set=frame_set, turret=state)

    def advance(self) -> SyntheticTurretPlantState:
        state = self.plant.advance(1.0 / self.world.fps)
        self.frame_index += 1
        return state


class RealtimeSyntheticTrackingRig:
    """Real-time shared world/plant clock for the normal full-SIM launcher.

    Camera workers and the fake STM32 execute on different threads.  This rig
    advances the plant to the same monotonic instant before either rendering a
    frame or applying a newly executed protocol command.  Frames rendered for
    the same simulation tick are cached, so Stereo Left/Right share capture_id,
    timestamp and turret pose.
    """

    def __init__(
        self,
        world: SyntheticTrackingWorld,
        plant: SyntheticTurretPlant,
        *,
        clock: Callable[[], float] = monotonic,
        synchronize_stereo_pair: bool = True,
    ) -> None:
        if not isinstance(world, SyntheticTrackingWorld):
            raise TypeError("world must be SyntheticTrackingWorld")
        if not isinstance(plant, SyntheticTurretPlant):
            raise TypeError("plant must be SyntheticTurretPlant")
        self.world = world
        self.plant = plant
        if type(synchronize_stereo_pair) is not bool:
            raise TypeError("synchronize_stereo_pair must be bool")
        self._clock = clock
        self._synchronize_stereo_pair = synchronize_stereo_pair
        self._lock = Lock()
        self._started_at_s: float | None = None
        self._last_sync_s: float | None = None
        self._cached_frame_index: int | None = None
        self._cached_timestamp_ns: int | None = None
        self._cached_frame_set: SyntheticTrackingFrameSet | None = None
        self._stereo_pending_frame_set: SyntheticTrackingFrameSet | None = None
        self._stereo_pending_timestamp_ns: int | None = None
        self._stereo_served: set[CameraRole] = set()

    def start(self) -> None:
        with self._lock:
            self._ensure_started_locked(self._clock())

    def apply_executed_request(self, request: ProtocolRequest) -> None:
        if not isinstance(request, ProtocolRequest):
            raise TypeError("request must be ProtocolRequest")
        with self._lock:
            now_s = self._clock()
            self._sync_locked(now_s)
            self.plant.apply_executed_request(request)

    def capture(
        self,
        camera: CameraRole,
    ) -> tuple[SyntheticTrackingFrameSet, int]:
        if camera not in (
            CameraRole.OVERVIEW,
            CameraRole.STEREO_LEFT,
            CameraRole.STEREO_RIGHT,
        ):
            raise ValueError("unsupported synthetic tracking camera")
        with self._lock:
            now_s = self._clock()
            self._sync_locked(now_s)
            if (
                self._synchronize_stereo_pair
                and camera in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT)
            ):
                return self._capture_stereo_locked(camera, now_s)
            return self._capture_current_locked(now_s)

    def _capture_stereo_locked(
        self,
        camera: CameraRole,
        now_s: float,
    ) -> tuple[SyntheticTrackingFrameSet, int]:
        if self._stereo_pending_frame_set is None:
            frame_set, timestamp_ns = self._capture_current_locked(now_s)
            self._stereo_pending_frame_set = frame_set
            self._stereo_pending_timestamp_ns = timestamp_ns
            self._stereo_served.clear()
        assert self._stereo_pending_timestamp_ns is not None
        frame_set = self._stereo_pending_frame_set
        timestamp_ns = self._stereo_pending_timestamp_ns
        self._stereo_served.add(camera)
        if self._stereo_served == {CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT}:
            self._stereo_pending_frame_set = None
            self._stereo_pending_timestamp_ns = None
            self._stereo_served.clear()
        return frame_set, timestamp_ns

    def _capture_current_locked(
        self,
        now_s: float,
    ) -> tuple[SyntheticTrackingFrameSet, int]:
        assert self._started_at_s is not None
        frame_index = max(0, int((now_s - self._started_at_s) * self.world.fps))
        if frame_index != self._cached_frame_index:
            state = self.plant.state
            self._cached_frame_set = self.world.render(
                frame_index,
                turret_yaw_deg=state.yaw_deg,
                turret_pitch_deg=state.pitch_deg,
            )
            self._cached_frame_index = frame_index
            self._cached_timestamp_ns = round(
                (self._started_at_s + frame_index / self.world.fps)
                * 1_000_000_000.0
            )
        assert self._cached_frame_set is not None
        assert self._cached_timestamp_ns is not None
        return self._cached_frame_set, self._cached_timestamp_ns

    def _ensure_started_locked(self, now_s: float) -> None:
        if self._started_at_s is None:
            self._started_at_s = now_s
            self._last_sync_s = now_s

    def _sync_locked(self, now_s: float) -> None:
        self._ensure_started_locked(now_s)
        assert self._last_sync_s is not None
        dt_s = max(0.0, now_s - self._last_sync_s)
        if dt_s > 0.0:
            self.plant.advance(dt_s)
            self._last_sync_s = now_s


class SyntheticTrackingFrameSource:
    """Decoded-frame source backed by one shared real-time tracking rig."""

    def __init__(
        self,
        rig: RealtimeSyntheticTrackingRig,
        camera: CameraRole,
    ) -> None:
        if not isinstance(rig, RealtimeSyntheticTrackingRig):
            raise TypeError("rig must be RealtimeSyntheticTrackingRig")
        if camera not in (
            CameraRole.OVERVIEW,
            CameraRole.STEREO_LEFT,
            CameraRole.STEREO_RIGHT,
        ):
            raise ValueError("unsupported synthetic tracking camera")
        self._rig = rig
        self._camera = camera
        self._started = False
        self._last_capture_id: int | None = None

    @property
    def supports_reconnect(self) -> bool:
        return False

    @property
    def failure(self) -> BaseException | None:
        return None

    def start(self) -> None:
        self._rig.start()
        self._started = True
        self._last_capture_id = None

    def stop(self) -> None:
        self._started = False

    def read(self) -> DecodedFrame | None:
        if not self._started:
            return None
        frame_set, timestamp_ns = self._rig.capture(self._camera)
        if frame_set.capture_id == self._last_capture_id:
            return None
        self._last_capture_id = frame_set.capture_id
        return DecodedFrame(
            image=frame_set.image(self._camera),
            capture_id=frame_set.capture_id,
            receive_timestamp_ns=timestamp_ns,
        )


class SyntheticTrackingStm32Endpoint(FakeStm32Endpoint):
    """Fake STM32 endpoint that drives a separate synthetic physical plant."""

    def __init__(
        self,
        plant: SyntheticTurretPlant,
        *,
        baudrate: int = 9600,
        request_sink: Callable[[ProtocolRequest], None] | None = None,
    ) -> None:
        if not isinstance(plant, SyntheticTurretPlant):
            raise TypeError("plant must be SyntheticTurretPlant")
        super().__init__(baudrate=baudrate)
        self.plant = plant
        self._request_sink = request_sink or plant.apply_executed_request

    def _on_executed_request(
        self,
        request: ProtocolRequest,
        response: ProtocolResponse,
    ) -> None:
        if (
            response.request_id == request.request_id
            and response.command is request.command
            and response.result is ResultCode.OK
        ):
            self._request_sink(request)


__all__ = [
    "RealtimeSyntheticTrackingRig",
    "SyntheticTrackingFrameSource",
    "SyntheticTrackingSimulation",
    "SyntheticTrackingSimulationSnapshot",
    "SyntheticTrackingStm32Endpoint",
    "SyntheticTurretPlant",
    "SyntheticTurretPlantState",
]
