"""Turret worker: UART ownership, reconnect/recovery, state publication."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Event, Lock, Thread

from navmin.concurrency import InvalidatableLatest, LatestValue
from navmin.config.models import SerialConfig, TurretConfig
from navmin.contracts import (
    ConfigUpdate,
    MotorState,
    MoveRelativeCommand,
    TrackingError,
    TurretConnectionState,
    TurretControlMode,
    TurretState,
)
from navmin.lifecycle import StopToken, stop_and_join

from .controller import ControllerError, TurretController
from .hal import TurretHal
from .protocol import ResultCode
from .session import (
    SessionError,
    SessionResult,
    TurretSession,
)
from .simulator import FakeStm32Endpoint, FakeTransport
from .transport import PhysicalTransport, SerialTransport, TransportError

_LOGGER = logging.getLogger(__name__)
_BACKOFF_SECONDS = (0.25, 0.5, 1.0, 2.0)
_IDLE_WAIT_SECONDS = 0.01
_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 2.0


class WorkerShutdownError(RuntimeError):
    """A Turret worker did not stop within its bounded shutdown deadline."""


class _ControlKind(Enum):
    SET_MODE = "set_mode"
    STOP_MOTION = "stop_motion"
    MOTOR_ON = "motor_on"
    MOTOR_OFF = "motor_off"


class _RecoveryConfigStatus(Enum):
    COMPLETE = "complete"
    EMERGENCY_HANDLED = "emergency_handled"
    FAILED = "failed"


@dataclass(frozen=True)
class _ControlIntent:
    kind: _ControlKind
    mode: TurretControlMode | None = None


type MotionInput = MoveRelativeCommand | TrackingError
type TransportFactory = Callable[[str, int, bool], PhysicalTransport]
type CooperativeWait = Callable[[StopToken, float], bool]


def _default_wait(stop_token: StopToken, timeout_s: float) -> bool:
    return stop_token.wait(timeout_s)


def _deduplicate_candidates(*values: int | None) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return tuple(result)


def _session_settings(serial: SerialConfig) -> tuple[float, int, float]:
    return (
        serial.response_timeout_ms / 1000.0,
        serial.max_retries,
        serial.inter_request_delay_ms / 1000.0,
    )


def _reconnect_signature(config: TurretConfig) -> tuple[str, int, int, int]:
    serial = config.serial
    return (
        serial.port,
        serial.response_timeout_ms,
        serial.max_retries,
        serial.inter_request_delay_ms,
    )


class TurretWorker:
    """Single owner of physical Turret I/O and automatic recovery."""

    def __init__(
        self,
        config: TurretConfig,
        *,
        transport_factory: TransportFactory | None = None,
        cooperative_wait: CooperativeWait = _default_wait,
    ) -> None:
        if not isinstance(config, TurretConfig):
            raise TypeError("config must be TurretConfig")
        self._startup_config = config
        self._desired_config = config
        self._startup_emulate_stm32 = config.emulate_stm32
        self._fake_endpoint = (
            FakeStm32Endpoint() if self._startup_emulate_stm32 else None
        )
        self._transport_factory = transport_factory or self._default_transport_factory
        self._cooperative_wait = cooperative_wait

        self._stop_token = StopToken()
        self._thread: Thread | None = None
        self._ready = False
        self._transport: PhysicalTransport | None = None
        self._session: TurretSession | None = None
        self._hal: TurretHal | None = None
        self._controller: TurretController | None = None
        self._last_known_port: str | None = None
        self._last_known_baud: int | None = None
        self._next_request_id = 0

        self._motion_inputs: InvalidatableLatest[MotionInput] = InvalidatableLatest()
        self._control_inputs: LatestValue[_ControlIntent] = LatestValue()
        self._config_updates: LatestValue[ConfigUpdate[TurretConfig]] = LatestValue()
        self._config_updates.publish(ConfigUpdate(0, config))
        self._emergency_signal = Event()
        self._emergency_signal_lock = Lock()
        self._normal_ingress_lock = Lock()
        self._normal_ingress_open = False
        self._last_motion_revision = 0
        self._last_control_revision = 0
        self._last_config_revision = 0

        self._state_updates: LatestValue[TurretState] = LatestValue()
        self._publish_state(TurretConnectionState.DISCONNECTED)

    @property
    def state_updates(self) -> LatestValue[TurretState]:
        return self._state_updates

    @property
    def current_state(self) -> TurretState:
        state = self._state_updates.get()
        if state is None:
            raise RuntimeError("TurretState has not been initialized")
        return state

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TurretWorker is one-shot and has already been started")
        self._thread = Thread(target=self._run, name="turret-worker", daemon=False)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_token.request_stop()

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def shutdown(self, timeout: float = _DEFAULT_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """Bounded application-facing cooperative shutdown boundary."""
        if stop_and_join(self, timeout):
            return
        _LOGGER.error("Turret worker failed to stop within %.3f s", timeout)
        raise WorkerShutdownError(
            f"Turret worker did not stop within {timeout:.3f} seconds"
        )

    # Cross-thread ingress. These methods only publish typed intent/state.
    def submit_move_relative(self, command: MoveRelativeCommand) -> None:
        if not isinstance(command, MoveRelativeCommand):
            raise TypeError("command must be MoveRelativeCommand")
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._motion_inputs.publish(command)

    def submit_tracking_error(self, error: TrackingError) -> None:
        if not isinstance(error, TrackingError):
            raise TypeError("error must be TrackingError")
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._motion_inputs.publish(error)

    def invalidate_tracking_error(self) -> None:
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._motion_inputs.invalidate()

    def set_control_mode(self, mode: TurretControlMode) -> None:
        if not isinstance(mode, TurretControlMode):
            raise TypeError("mode must be TurretControlMode")
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._control_inputs.publish(
                    _ControlIntent(_ControlKind.SET_MODE, mode)
                )

    def stop_motion(self) -> None:
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._control_inputs.publish(_ControlIntent(_ControlKind.STOP_MOTION))

    def motor_on(self) -> None:
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._control_inputs.publish(_ControlIntent(_ControlKind.MOTOR_ON))

    def motor_off(self) -> None:
        with self._normal_ingress_lock:
            if self._normal_ingress_open:
                self._control_inputs.publish(_ControlIntent(_ControlKind.MOTOR_OFF))

    def request_emergency(self) -> None:
        """Signal Emergency without performing UART I/O in the caller thread."""
        with self._emergency_signal_lock:
            session = self._session
            if session is None:
                self._emergency_signal.set()
                return
            if session.signal_emergency():
                self._emergency_signal.set()

    def submit_config_update(self, update: ConfigUpdate[TurretConfig]) -> None:
        if not isinstance(update, ConfigUpdate):
            raise TypeError("update must be ConfigUpdate[TurretConfig]")
        if not isinstance(update.config, TurretConfig):
            raise TypeError("update.config must be TurretConfig")
        self._config_updates.publish(update)

    def _run(self) -> None:
        _LOGGER.info("Turret worker started")
        self._publish_state(TurretConnectionState.CONNECTING)
        backoff_index = 0
        fatal_error = False
        try:
            while not self._stop_token.is_stop_requested():
                if not self._ready:
                    self._discard_normal_ingress()
                    try:
                        recovered = self._recover_once()
                    except Exception:
                        _LOGGER.exception("Unrecoverable local Turret recovery failure")
                        fatal_error = True
                        break
                    if recovered:
                        backoff_index = 0
                        continue
                    if self._stop_token.is_stop_requested():
                        break
                    delay = _BACKOFF_SECONDS[min(backoff_index, len(_BACKOFF_SECONDS) - 1)]
                    backoff_index += 1
                    _LOGGER.info("Turret reconnect backoff %.2f s", delay)
                    if self._cooperative_wait(self._stop_token, delay):
                        break
                    continue

                try:
                    did_work = self._process_ready_once()
                except (SessionError, TransportError) as exc:
                    self._begin_recovery(str(exc))
                    continue
                except ControllerError as exc:
                    _LOGGER.warning("Rejected Turret control intent: %s", exc)
                    continue
                except Exception:
                    _LOGGER.exception("Unrecoverable local Turret worker failure")
                    fatal_error = True
                    break

                if not did_work and self._stop_token.wait(_IDLE_WAIT_SECONDS):
                    break
        finally:
            self._discard_normal_ingress()
            self._ready = False
            controller = self._controller
            if controller is not None:
                controller.reset_for_transport_recovery()
            self._close_transport()
            final_state = (
                TurretConnectionState.ERROR
                if fatal_error
                else TurretConnectionState.DISCONNECTED
            )
            self._publish_state(final_state)
            _LOGGER.info("Turret worker stopped")

    def _recover_once(self) -> bool:
        update = self._latest_config_update()
        self._desired_config = update.config
        self._stage_latest_config(update)
        serial = self._desired_config.serial
        last_known = (
            self._last_known_baud
            if self._last_known_port == serial.port
            else None
        )
        candidates = _deduplicate_candidates(last_known, serial.baudrate, 9600)
        _LOGGER.info("Turret recovery started on port %s", serial.port)

        for candidate in candidates:
            if self._stop_token.is_stop_requested():
                return False
            _LOGGER.debug("Trying Turret baud candidate %d", candidate)
            transport: PhysicalTransport | None = None
            try:
                transport = self._transport_factory(
                    serial.port, candidate, self._startup_emulate_stm32
                )
                transport.open()
                session = self._make_session(transport, self._desired_config)
                self._transport = transport
                self._session = session
                self._bind_stack(session)
                with self._emergency_signal_lock:
                    if self._emergency_signal.is_set():
                        session.signal_emergency()

                if not self._complete_recovery_emergency_boundary():
                    self._close_transport()
                    return False
                self._last_known_port = serial.port
                self._last_known_baud = candidate
                _LOGGER.info(
                    "Turret physical connection found on %s at %d baud",
                    serial.port,
                    candidate,
                )
                self._publish_state(TurretConnectionState.CONNECTING)

                while not self._stop_token.is_stop_requested():
                    if not self._stabilize_recovery_config():
                        self._close_transport()
                        return False

                    config_status = self._sync_freshest_stm32_config()
                    if config_status is _RecoveryConfigStatus.FAILED:
                        self._close_transport()
                        return False
                    if config_status is _RecoveryConfigStatus.EMERGENCY_HANDLED:
                        if not self._complete_recovery_motor_off_boundary():
                            self._close_transport()
                            return False
                        continue

                    with self._emergency_signal_lock:
                        session = self._session_required()
                        if self._emergency_signal.is_set() or session.emergency_pending:
                            pending_emergency = True
                        else:
                            pending_emergency = False
                            with self._normal_ingress_lock:
                                self._ready = True
                                self._publish_state(TurretConnectionState.READY)
                                self._normal_ingress_open = True
                    if not pending_emergency:
                        _LOGGER.info("Turret READY")
                        return True
                    if not self._complete_recovery_emergency_boundary():
                        self._close_transport()
                        return False
            except (SessionError, TransportError) as exc:
                _LOGGER.debug("Turret baud candidate %d failed: %s", candidate, exc)
                self._close_transport()
                continue
            except ControllerError as exc:
                _LOGGER.warning("Turret recovery control boundary failed: %s", exc)
                self._close_transport()
                return False
            finally:
                if transport is not None and transport is not self._transport:
                    try:
                        transport.close()
                    except TransportError:
                        pass
        return False

    def _complete_recovery_emergency_boundary(self) -> bool:
        hal = self._hal_required()
        with hal.defer_pending_config_drain():
            emergency = self._controller_required().emergency_stop()
            if emergency is None or emergency.result is not ResultCode.OK:
                return False
            self._clear_emergency_signal_if_serviced()
            return self._complete_recovery_motor_off_boundary()

    def _complete_recovery_motor_off_boundary(self) -> bool:
        hal = self._hal_required()
        with hal.defer_pending_config_drain():
            motor_off = self._controller_required().motor_off()
        return self._session_result_ok(motor_off)

    def _stabilize_recovery_config(self) -> bool:
        """Apply freshest accepted config before final SET_CONFIG/READY."""
        while not self._stop_token.is_stop_requested():
            update = self._latest_config_update()
            if _reconnect_signature(update.config) != _reconnect_signature(
                self._desired_config
            ):
                self._desired_config = update.config
                self._stage_latest_config(update)
                return False

            self._desired_config = update.config
            self._stage_latest_config(update)
            desired_baud = update.config.serial.baudrate
            transport = self._transport_required()
            if transport.baudrate != desired_baud:
                response = self._session_required().set_baudrate(desired_baud)
                if response.result is not ResultCode.OK:
                    return False
                self._last_known_port = update.config.serial.port
                self._last_known_baud = desired_baud
                _LOGGER.info("Turret baud transitioned to %d", desired_baud)
                self._clear_emergency_signal_if_serviced()

            latest = self._latest_config_update()
            if latest.revision == update.revision:
                return True
        return False

    def _sync_freshest_stm32_config(self) -> _RecoveryConfigStatus:
        while not self._stop_token.is_stop_requested():
            update = self._latest_config_update()
            self._desired_config = update.config
            if _reconnect_signature(update.config) != _reconnect_signature(
                self._session_config_required()
            ):
                self._stage_latest_config(update)
                return _RecoveryConfigStatus.FAILED
            self._stage_latest_config(update)
            transport = self._transport_required()
            desired_baud = update.config.serial.baudrate
            if transport.baudrate != desired_baud:
                response = self._session_required().set_baudrate(desired_baud)
                if response.result is not ResultCode.OK:
                    return _RecoveryConfigStatus.FAILED
                self._last_known_port = update.config.serial.port
                self._last_known_baud = desired_baud
                _LOGGER.info("Turret baud transitioned to %d", desired_baud)
                self._clear_emergency_signal_if_serviced()
            result = self._hal_required().sync_stm32_config()
            if result is not None and result.preempted_by_emergency:
                emergency = result.emergency_response
                if emergency is not None and emergency.result is ResultCode.OK:
                    self._clear_emergency_signal_if_serviced()
                    return _RecoveryConfigStatus.EMERGENCY_HANDLED
                return _RecoveryConfigStatus.FAILED
            if not self._session_result_ok(result):
                return _RecoveryConfigStatus.FAILED
            latest = self._latest_config_update()
            if latest.revision == update.revision:
                return _RecoveryConfigStatus.COMPLETE
        return _RecoveryConfigStatus.FAILED

    def _process_ready_once(self) -> bool:
        if self._emergency_signal.is_set():
            self._service_emergency()
            return True

        config_snapshot = self._config_updates.snapshot()
        if config_snapshot.revision > self._last_config_revision:
            self._last_config_revision = config_snapshot.revision
            update = config_snapshot.value
            if update is not None:
                self._handle_config_update(update)
            return True

        control_snapshot = self._control_inputs.snapshot()
        if control_snapshot.revision > self._last_control_revision:
            self._last_control_revision = control_snapshot.revision
            if control_snapshot.value is not None:
                self._handle_control(control_snapshot.value)
            return True

        motion_snapshot = self._motion_inputs.snapshot()
        if motion_snapshot.revision > self._last_motion_revision:
            self._last_motion_revision = motion_snapshot.revision
            self._handle_motion(motion_snapshot.revision, motion_snapshot.value)
            return True

        return False

    def _handle_config_update(self, update: ConfigUpdate[TurretConfig]) -> None:
        old = self._desired_config
        new = update.config
        self._desired_config = new

        if _reconnect_signature(old) != _reconnect_signature(new):
            self._controller_required().apply_config_update(update, defer_stm32=True)
            self._begin_recovery("serial port/timing configuration changed")
            return

        result = self._controller_required().apply_config_update(update)
        if result is not None:
            self._handle_session_result(result)
            if (
                result.preempted_by_emergency
                and result.emergency_response is not None
                and result.emergency_response.result is ResultCode.OK
                and self._hal_required().pending_config_revision is not None
            ):
                pending_result = self._hal_required().flush_pending_config()
                if pending_result is not None:
                    self._handle_session_result(pending_result)
                if not self._session_result_ok(pending_result):
                    raise SessionError(
                        "SET_CONFIG remained unconfirmed after Emergency preemption"
                    )

        if old.serial.baudrate != new.serial.baudrate:
            if self._hal_required().motor_state is MotorState.OFF:
                self._apply_pending_baud_if_safe()
            else:
                _LOGGER.info(
                    "Turret baud transition to %d deferred until confirmed MOTOR_OFF",
                    new.serial.baudrate,
                )
        self._publish_state(TurretConnectionState.READY)

    def _handle_control(self, intent: _ControlIntent) -> None:
        controller = self._controller_required()
        if intent.kind is _ControlKind.SET_MODE:
            if intent.mode is None:
                raise RuntimeError("SET_MODE control intent requires mode")
            result = controller.set_control_mode(intent.mode)
            if result is not None:
                self._handle_session_result(result)
        elif intent.kind is _ControlKind.STOP_MOTION:
            result = controller.stop_motion()
            if result is not None:
                self._handle_session_result(result)
        elif intent.kind is _ControlKind.MOTOR_OFF:
            result = controller.motor_off()
            self._handle_session_result(result)
            if self._session_result_ok(result):
                self._apply_pending_baud_if_safe()
        elif intent.kind is _ControlKind.MOTOR_ON:
            if (
                self._hal_required().motor_state is MotorState.OFF
                and not self._apply_pending_baud_if_safe()
            ):
                self._publish_state(TurretConnectionState.READY)
                return
            result = controller.motor_on()
            self._handle_session_result(result)
        self._publish_state(TurretConnectionState.READY)

    def _handle_motion(self, revision: int, motion: MotionInput | None) -> None:
        controller = self._controller_required()
        if motion is None:
            if controller.control_mode is TurretControlMode.TRACKING:
                controller.invalidate_tracking_error(revision)
            return
        if isinstance(motion, MoveRelativeCommand):
            controller.submit_move_relative(motion)
        else:
            controller.submit_tracking_error(revision, motion)
        result = controller.flush_pending_motion()
        if result is not None:
            self._handle_session_result(result)
        self._publish_state(TurretConnectionState.READY)

    def _service_emergency(self) -> None:
        response = self._controller_required().emergency_stop()
        if response is not None and response.result is ResultCode.OK:
            self._clear_emergency_signal_if_serviced()
        self._publish_state(TurretConnectionState.READY)

    def _handle_session_result(self, result: SessionResult) -> None:
        if (
            result.emergency_response is not None
            and result.emergency_response.result is ResultCode.OK
        ):
            self._clear_emergency_signal_if_serviced()
        response = result.response
        if response is None or response.result is not ResultCode.INVALID_REQUEST_ID:
            return
        # Sequence loss is not a physical disconnect. Resync on the same link.
        emergency = self._controller_required().emergency_stop()
        if emergency is not None and emergency.result is ResultCode.OK:
            self._clear_emergency_signal_if_serviced()
            return
        raise SessionError("Emergency resync failed after INVALID_REQUEST_ID")

    def _clear_emergency_signal_if_serviced(self) -> None:
        with self._emergency_signal_lock:
            session = self._session
            if session is not None and not session.emergency_pending:
                self._emergency_signal.clear()

    def _apply_pending_baud_if_safe(self) -> bool:
        hal = self._hal_required()
        transport = self._transport_required()
        desired = self._desired_config.serial.baudrate
        if transport.baudrate == desired:
            return True
        if hal.motor_state is not MotorState.OFF:
            return False
        response = self._session_required().set_baudrate(desired)
        if response.result is not ResultCode.OK:
            return False
        self._clear_emergency_signal_if_serviced()
        self._last_known_port = self._desired_config.serial.port
        self._last_known_baud = desired
        _LOGGER.info("Turret baud transitioned to %d", desired)
        return True

    def _begin_recovery(self, reason: str) -> None:
        if (
            not self._ready
            and self.current_state.connection_state is TurretConnectionState.CONNECTING
        ):
            return
        _LOGGER.info("Turret entering recovery: %s", reason)
        self._discard_normal_ingress()
        self._ready = False
        controller = self._controller
        if controller is not None:
            controller.reset_for_transport_recovery()
        self._close_transport()
        self._publish_state(TurretConnectionState.CONNECTING)

    def _discard_normal_ingress(self) -> None:
        with self._normal_ingress_lock:
            self._normal_ingress_open = False
            self._motion_inputs.invalidate()
            self._last_motion_revision = self._motion_inputs.snapshot().revision
            self._last_control_revision = self._control_inputs.snapshot().revision

    def _bind_stack(self, session: TurretSession) -> None:
        if self._hal is None or self._controller is None:
            self._hal = TurretHal(session, self._startup_config)
            self._controller = TurretController(self._hal, self._startup_config)
        else:
            self._hal.rebind_session(session)
        self._stage_latest_config(self._latest_config_update())

    def _stage_latest_config(self, update: ConfigUpdate[TurretConfig]) -> None:
        controller = self._controller
        if controller is not None:
            controller.apply_config_update(update, defer_stm32=True)

    def _make_session(
        self, transport: PhysicalTransport, config: TurretConfig
    ) -> TurretSession:
        timeout_s, retries, delay_s = _session_settings(config.serial)
        session = TurretSession(
            transport,
            response_timeout_s=timeout_s,
            max_retries=retries,
            inter_request_delay_s=delay_s,
            initial_request_id=self._next_request_id,
        )
        # Store the serial/session snapshot used by this concrete boundary.
        self._session_config = config
        return session

    def _latest_config_update(self) -> ConfigUpdate[TurretConfig]:
        update = self._config_updates.get()
        if update is None:
            raise RuntimeError("Turret config update slot is empty")
        return update

    def _session_config_required(self) -> TurretConfig:
        config = getattr(self, "_session_config", None)
        if config is None:
            raise RuntimeError("session config is not available")
        return config

    def _publish_state(self, connection_state: TurretConnectionState) -> None:
        controller = self._controller
        hal = self._hal
        mode = (
            TurretControlMode.RELATIVE
            if controller is None
            else controller.control_mode
        )
        motor = MotorState.UNKNOWN if hal is None else hal.motor_state
        applied = None if hal is None else hal.applied_stm32_config
        self._state_updates.publish(
            TurretState(
                connection_state=connection_state,
                motor_state=motor,
                control_mode=mode,
                max_speed_x_deg_s=(
                    None if applied is None else applied.max_speed_x_deg_s
                ),
                max_speed_y_deg_s=(
                    None if applied is None else applied.max_speed_y_deg_s
                ),
                acceleration_x_deg_s2=(
                    None if applied is None else applied.acceleration_x_deg_s2
                ),
                acceleration_y_deg_s2=(
                    None if applied is None else applied.acceleration_y_deg_s2
                ),
            )
        )

    def _close_transport(self) -> None:
        transport = self._transport
        session = self._session
        if session is not None:
            self._next_request_id = session.next_request_id
        self._transport = None
        self._session = None
        if transport is None:
            return
        try:
            transport.close()
        except TransportError as exc:
            _LOGGER.warning("Failed to close Turret transport cleanly: %s", exc)

    def _default_transport_factory(
        self, port: str, baudrate: int, emulate_stm32: bool
    ) -> PhysicalTransport:
        if emulate_stm32:
            endpoint = self._fake_endpoint
            if endpoint is None:
                raise RuntimeError("fake STM32 endpoint is not initialized")
            return FakeTransport(endpoint, baudrate=baudrate)
        return SerialTransport(port, baudrate)

    def _controller_required(self) -> TurretController:
        if self._controller is None:
            raise RuntimeError("Turret Controller is not initialized")
        return self._controller

    def _hal_required(self) -> TurretHal:
        if self._hal is None:
            raise RuntimeError("Turret HAL is not initialized")
        return self._hal

    def _session_required(self) -> TurretSession:
        if self._session is None:
            raise RuntimeError("Turret session is not initialized")
        return self._session

    def _transport_required(self) -> PhysicalTransport:
        if self._transport is None:
            raise RuntimeError("Turret transport is not initialized")
        return self._transport

    @staticmethod
    def _session_result_ok(result: SessionResult | None) -> bool:
        return bool(
            result is not None
            and result.response is not None
            and result.response.result is ResultCode.OK
        )


__all__ = ["TransportFactory", "TurretWorker", "WorkerShutdownError"]
