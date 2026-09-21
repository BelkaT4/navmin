"""Deterministic hardware-free stability soak for the accepted software prototype."""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import threading
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from time import monotonic, monotonic_ns, sleep

from navmin.contracts import (
    CameraRole,
    MotorState,
    TurretConnectionState,
    TurretControlMode,
)
from navmin.turret.protocol import CommandCode, SetVelocityPayload
from navmin.turret.simulator import FakeReadFailure, FakeStm32Endpoint, FakeTransport
from navmin.turret.transport import TransportDisconnectedError


def _load_smoke_support():
    path = Path(__file__).with_name("run_software_smoke.py")
    spec = importlib.util.spec_from_file_location("navmin_software_smoke_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load software smoke support from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SMOKE_SUPPORT = _load_smoke_support()
FRAME_HEIGHT = _SMOKE_SUPPORT.FRAME_HEIGHT
FRAME_WIDTH = _SMOKE_SUPPORT.FRAME_WIDTH
SoftwareSmokeRuntime = _SMOKE_SUPPORT.SoftwareSmokeRuntime

_STALE_TIMEOUT_MS = 500
_POLL_S = 0.005
_HEARTBEAT_S = 30.0
_SAMPLE_S = 2.0
_DEFAULT_FAULT_CADENCE = (3, 6, 9, 12)  # stale, generation, disconnect, emergency
_OWNED_THREAD_PREFIXES = ("smoke-producer-", "vision-", "turret-worker")
_ENDPOINT_HISTORY_LIMIT = 32
_TRANSPORT_HISTORY_LIMIT = 32
_SAMPLE_WINDOW = 512
_CYCLE_WINDOW = 4096
_MIN_TRACK_AGE_FRAMES = 5


class SoakFailure(RuntimeError):
    """Functional/lifecycle invariant failure with phase context."""


def _trim_list(values: list[object], limit: int) -> None:
    if len(values) > limit:
        del values[:-limit]


class _SoakEndpoint(FakeStm32Endpoint):
    """Fake endpoint with full protocol fidelity and bounded soak diagnostics."""

    def __init__(self, *, history_limit: int = _ENDPOINT_HISTORY_LIMIT) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be > 0")
        super().__init__()
        self.history_limit = history_limit
        self._observability_lock = threading.Lock()
        self.command_counts: Counter[CommandCode] = Counter()
        self.set_velocity_execution_count = 0
        self.last_nonzero_velocity_sequence = 0
        self.last_zero_velocity_sequence = 0

    def handle_request(self, raw_frame: bytes) -> bytes | None:
        before = len(self.executed_request_history)
        response = super().handle_request(raw_frame)
        if len(self.executed_request_history) > before:
            request = self.executed_request_history[-1]
            with self._observability_lock:
                self.command_counts[request.command] += 1
                if request.command is CommandCode.SET_VELOCITY:
                    self.set_velocity_execution_count += 1
                    payload = request.payload
                    if isinstance(payload, SetVelocityPayload):
                        if payload.velocity_x_steps_s == 0 and payload.velocity_y_steps_s == 0:
                            self.last_zero_velocity_sequence = self.set_velocity_execution_count
                        else:
                            self.last_nonzero_velocity_sequence = self.set_velocity_execution_count
        _trim_list(self.raw_request_history, self.history_limit)
        _trim_list(self.request_history, self.history_limit)
        _trim_list(self.executed_request_history, self.history_limit)
        return response

    def command_count(self, command: CommandCode) -> int:
        with self._observability_lock:
            return self.command_counts[command]

    def has_nonzero_velocity_since(self, baseline: int) -> bool:
        with self._observability_lock:
            return self.last_nonzero_velocity_sequence > baseline

    def has_zero_velocity_since(self, baseline: int) -> bool:
        with self._observability_lock:
            return self.last_zero_velocity_sequence > baseline


class _BoundedFakeTransport(FakeTransport):
    """Real FakeTransport behavior with bounded diagnostic histories for soak."""

    def __init__(
        self,
        endpoint: FakeStm32Endpoint,
        *,
        baudrate: int,
        history_limit: int = _TRANSPORT_HISTORY_LIMIT,
    ) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be > 0")
        self.history_limit = history_limit
        super().__init__(endpoint, baudrate=baudrate)
        self._compact_histories()

    def set_baudrate(self, baudrate: int) -> None:
        super().set_baudrate(baudrate)
        self._compact_histories()

    def write_frame(self, frame: bytes, timeout_s: float) -> None:
        try:
            super().write_frame(frame, timeout_s)
        finally:
            self._compact_histories()

    def read_frame(self, timeout_s: float) -> bytes:
        try:
            return super().read_frame(timeout_s)
        finally:
            self._compact_histories()

    def _compact_histories(self) -> None:
        for name in (
            "baudrate_history",
            "raw_write_history",
            "raw_write_baudrate_history",
            "event_history",
        ):
            history = getattr(self, name, None)
            if history is not None:
                _trim_list(history, self.history_limit)


class _FailOpenTransport(_BoundedFakeTransport):
    def open(self) -> None:
        raise TransportDisconnectedError("software soak injected reconnect open failure")


class RecordingTransportFactory:
    """Soak-local observable FakeTransport factory used across reconnects."""

    def __init__(self) -> None:
        self.endpoint = _SoakEndpoint()
        self.current_transport: _BoundedFakeTransport | None = None
        self.total_transports_created = 0
        self.fail_open_count = 0

    @property
    def retained_transport_count(self) -> int:
        return int(self.current_transport is not None)

    def __call__(self, _port: str, baudrate: int, _emulate: bool) -> FakeTransport:
        transport_type = _FailOpenTransport if self.fail_open_count > 0 else _BoundedFakeTransport
        if self.fail_open_count > 0:
            self.fail_open_count -= 1
        transport = transport_type(self.endpoint, baudrate=baudrate)
        self.current_transport = transport
        self.total_transports_created += 1
        return transport


@dataclass(frozen=True)
class _ProcessSample:
    elapsed_s: float
    cycle: int
    rss_kib: int | None
    thread_count: int
    overview_revision: int
    overview_generation: int
    stereo_revision: int
    stereo_generation: int


@dataclass
class SoakMetrics:
    requested_s: float
    started_at: float = field(default_factory=monotonic)
    cycle_durations_s: deque[float] = field(
        default_factory=lambda: deque(maxlen=_CYCLE_WINDOW)
    )
    samples: deque[_ProcessSample] = field(
        default_factory=lambda: deque(maxlen=_SAMPLE_WINDOW)
    )
    cycle_rss_kib: deque[int] = field(
        default_factory=lambda: deque(maxlen=_CYCLE_WINDOW)
    )
    event_counts: Counter[str] = field(default_factory=Counter)
    cycle_duration_min_s: float | None = None
    cycle_duration_max_s: float | None = None
    thread_baseline: int = 0
    thread_warmup: int = 0
    thread_peak: int = 0
    thread_final: int = 0
    rss_initial_kib: int | None = None
    rss_warmup_kib: int | None = None
    rss_final_kib: int | None = None
    rss_peak_kib: int | None = None
    initial_overview_revision: int = 0
    initial_stereo_revision: int = 0

    @property
    def elapsed_s(self) -> float:
        return monotonic() - self.started_at


class SoftwareSoak:
    def __init__(
        self,
        *,
        duration_seconds: float,
        max_cycles: int | None = None,
        fault_cadence: tuple[int, int, int, int] = _DEFAULT_FAULT_CADENCE,
        heartbeat_seconds: float = _HEARTBEAT_S,
        sample_seconds: float = _SAMPLE_S,
    ) -> None:
        if duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be > 0")
        if max_cycles is not None and max_cycles <= 0:
            raise ValueError("max_cycles must be > 0 when provided")
        if any(value <= 0 for value in fault_cadence):
            raise ValueError("fault cadence entries must be > 0")
        self.duration_seconds = duration_seconds
        self.max_cycles = max_cycles
        self.stale_every, self.generation_every, self.disconnect_every, self.emergency_every = fault_cadence
        self.heartbeat_seconds = heartbeat_seconds
        self.sample_seconds = sample_seconds
        self.metrics = SoakMetrics(requested_s=duration_seconds)
        self.factory = RecordingTransportFactory()
        self.runtime = SoftwareSmokeRuntime(turret_transport_factory=self.factory)
        self.application = None
        self.window = None
        self._next_sample_at = 0.0
        self._next_heartbeat_at = 0.0
        self._warmup_thread_limit: int | None = None
        self._last_sample: _ProcessSample | None = None
        self._last_progress_sample: _ProcessSample | None = None
        self._intentional_stale = False
        self._phase = "init"
        self._cycle = 0

    def run(self) -> tuple[str, SoakMetrics]:
        # Qt can create stable process helper state; establish the baseline after QApplication.
        from PyQt6.QtWidgets import QApplication

        from navmin.ui import MainWindow

        self.application = QApplication.instance() or QApplication([])
        self.metrics.thread_baseline = len(threading.enumerate())
        self.metrics.rss_initial_kib = _read_rss_kib()
        primary_failure: Exception | None = None
        primary_failure_phase: str | None = None
        primary_failure_cycle = 0
        cleanup_failure: Exception | None = None
        result = "FAIL"
        try:
            self.runtime.start()
            self.window = MainWindow(
                mediator=self.runtime.mediator,
                camera_bindings=self.runtime.camera_bindings(),
                turret_states=self.runtime.turret_worker.state_updates,
                camera_stale_timeout_ms=_STALE_TIMEOUT_MS,
                start_timer=False,
                start_fullscreen=False,
            )
            self.window.resize(900, 650)
            self.window.show()
            self.application.processEvents()
            self._phase = "warm-up"
            self._wait_until(self._initial_ready, 5.0, "initial READY/live UI state was not reached")
            self.metrics.rss_warmup_kib = _read_rss_kib()
            self.metrics.initial_overview_revision = self.runtime.overview_pipeline.latest_result.snapshot().revision
            self.metrics.initial_stereo_revision = self.runtime.stereo_left_pipeline.latest_result.snapshot().revision
            self.metrics.thread_warmup = len(threading.enumerate())
            self.metrics.thread_peak = self.metrics.thread_warmup
            self._warmup_thread_limit = self.metrics.thread_warmup
            self._next_sample_at = monotonic()
            self._next_heartbeat_at = monotonic() + self.heartbeat_seconds
            run_deadline = monotonic() + self.duration_seconds

            while monotonic() < run_deadline:
                if self.max_cycles is not None and self._cycle >= self.max_cycles:
                    break
                self._cycle += 1
                self._phase = "nominal cycle"
                cycle_started = monotonic()
                self._nominal_cycle(self._cycle)
                cycle_duration = monotonic() - cycle_started
                self.metrics.cycle_durations_s.append(cycle_duration)
                self.metrics.cycle_duration_min_s = (
                    cycle_duration
                    if self.metrics.cycle_duration_min_s is None
                    else min(self.metrics.cycle_duration_min_s, cycle_duration)
                )
                self.metrics.cycle_duration_max_s = (
                    cycle_duration
                    if self.metrics.cycle_duration_max_s is None
                    else max(self.metrics.cycle_duration_max_s, cycle_duration)
                )
                self.metrics.event_counts["nominal_cycles"] += 1
                rss = _read_rss_kib()
                if rss is not None:
                    self.metrics.cycle_rss_kib.append(rss)
                    self._observe_rss_peak(rss)
                self._observe_periodic()
                if self._warmup_thread_limit is not None:
                    count = len(threading.enumerate())
                    if count > self._warmup_thread_limit:
                        raise SoakFailure(
                            f"thread count grew after warm-up: {count} > {self._warmup_thread_limit}"
                        )

            if self.max_cycles is None and monotonic() < run_deadline:
                raise SoakFailure("requested duration was not completed")
            self._phase = "final invariants"
            self._wait_until(self._final_operational_state, 2.0, "final operational state did not settle")
            self._assert_required_events()
            self._assert_progress_since_warmup()
            result = "SOAK SUSPECT" if self._memory_growth_suspect() else "PASS"
        except (RuntimeError, AssertionError) as exc:  # expected harness/invariant failure
            primary_failure = exc
            primary_failure_phase = self._phase
            primary_failure_cycle = self._cycle
            result = "FAIL"
        finally:
            self._phase = "cleanup"
            try:
                if self.window is not None:
                    self.window.state_pump.stop()
                    self.window.close()
                    self.application.processEvents()
                self.runtime.overview_producer.resume()
                self.runtime.shutdown(timeout=2.0)
            except RuntimeError as exc:
                cleanup_failure = exc
                result = "FAIL"
            self.metrics.rss_final_kib = _read_rss_kib()
            for value in (self.metrics.rss_initial_kib, self.metrics.rss_warmup_kib, self.metrics.rss_final_kib):
                if value is not None:
                    self._observe_rss_peak(value)
            self.metrics.thread_final = len(threading.enumerate())
            try:
                self._assert_shutdown_threads()
            except SoakFailure as exc:
                if cleanup_failure is None:
                    cleanup_failure = exc
                result = "FAIL"

        if primary_failure is not None:
            detail = f"cycle={primary_failure_cycle} phase={primary_failure_phase}: {primary_failure}"
            if cleanup_failure is not None:
                detail += f"; cleanup failure: {cleanup_failure}"
            raise SoakFailure(detail) from primary_failure
        if cleanup_failure is not None:
            raise SoakFailure(f"cleanup failure: {cleanup_failure}") from cleanup_failure
        return result, self.metrics

    def _nominal_cycle(self, cycle: int) -> None:
        self._ensure_main(CameraRole.OVERVIEW)
        self._ensure_mode(TurretControlMode.RELATIVE)
        self._ensure_motor(MotorState.ON)
        self._relative_move()

        if cycle % self.stale_every == 0:
            self._stale_resume()

        self._ensure_active_tracking()

        if cycle % self.generation_every == 0:
            self._generation_restart()
            self._ensure_active_tracking()
        if cycle % self.disconnect_every == 0:
            self._turret_disconnect_recovery()
            self._ensure_active_tracking()
        if cycle % self.emergency_every == 0:
            self._emergency_during_tracking()
            self._ensure_active_tracking()

        self._deselect_tracking()
        self._preview_swap_round_trip()
        self._ensure_mode(TurretControlMode.RELATIVE)
        self._ensure_motor(MotorState.OFF)
        self._assert_no_stale_selection_or_pending()

    def _pump_once(self) -> None:
        assert self.window is not None and self.application is not None
        self.window.state_pump.pump_once()
        self.application.processEvents()
        self._observe_periodic()

    def _wait_until(self, predicate, timeout: float, message: str) -> None:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            self._pump_once()
            if predicate():
                return
            sleep(_POLL_S)
        self._pump_once()
        if not predicate():
            raise SoakFailure(message)

    def _pump_for(self, duration: float) -> None:
        deadline = monotonic() + duration
        while monotonic() < deadline:
            self._pump_once()
            sleep(_POLL_S)

    def _initial_ready(self) -> bool:
        assert self.window is not None
        return (
            self.runtime.mediator.turret_state.connection_state is TurretConnectionState.READY
            and self.runtime.mediator.session_gate.accepted_generation(CameraRole.OVERVIEW) is not None
            and self.runtime.mediator.session_gate.accepted_generation(CameraRole.STEREO_LEFT) is not None
            and self.window.main_view.displayed_result is not None
            and self.window.preview_view.displayed_result is not None
        )

    def _final_operational_state(self) -> bool:
        return (
            self.runtime.mediator.turret_state.connection_state is TurretConnectionState.READY
            and self.runtime.mediator.turret_state.motor_state is MotorState.OFF
            and self.runtime.mediator.turret_state.control_mode is TurretControlMode.RELATIVE
            and self.runtime.mediator.pending_control_mode is None
            and self.runtime.mediator.selected_target is None
        )

    def _ensure_main(self, camera: CameraRole) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        if self.runtime.mediator.main_camera is camera:
            return
        QTest.mouseClick(self.window.preview_view, Qt.MouseButton.LeftButton, pos=self.window.preview_view.rect().center())
        self.application.processEvents()
        self._wait_until(lambda: self.runtime.mediator.main_camera is camera, 1.0, f"main camera did not become {camera.value}")

    def _ensure_mode(self, mode: TurretControlMode) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        if self.runtime.mediator.pending_control_mode is not None:
            self._wait_until(
                lambda: self.runtime.mediator.pending_control_mode is None,
                1.5,
                "previous control-mode request remained pending",
            )
        if self.runtime.mediator.turret_state.control_mode is mode:
            return
        QTest.mouseClick(self.window.mode_button, Qt.MouseButton.LeftButton)
        self._wait_until(
            lambda: self.runtime.mediator.turret_state.control_mode is mode and self.runtime.mediator.pending_control_mode is None,
            1.5,
            f"control mode did not settle to {mode.value}",
        )

    def _ensure_motor(self, state: MotorState) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        if self.runtime.mediator.turret_state.motor_state is state:
            return
        if self.runtime.mediator.turret_state.motor_state is MotorState.UNKNOWN:
            raise SoakFailure(f"cannot request motor {state.value} while motor state is UNKNOWN")
        QTest.mouseClick(self.window.motor_button, Qt.MouseButton.LeftButton)
        self._wait_until(lambda: self.runtime.mediator.turret_state.motor_state is state, 1.5, f"motor did not settle to {state.value}")

    def _relative_move(self) -> None:
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        baseline = self._command_count(CommandCode.MOVE_RELATIVE)
        self._wait_until(
            lambda: self.window.main_view.displayed_result is not None
            and self.window.main_view.interaction_allowed(self.runtime.mediator.session_gate, monotonic_ns()),
            2.0,
            "main view was not interaction-eligible for RELATIVE",
        )
        rendered = self.window.main_view.rendered_rect()
        click = QPoint(round(rendered.left() + rendered.width() * 0.75), round(rendered.top() + rendered.height() * 0.25))
        QTest.mouseClick(self.window.main_view, Qt.MouseButton.LeftButton, pos=click)
        self._wait_until(lambda: self._command_count(CommandCode.MOVE_RELATIVE) > baseline, 1.0, "RELATIVE command did not reach endpoint")

    def _stable_overview_track(self):
        assert self.window is not None
        result = self.window.main_view.displayed_result
        if result is None or len(result.tracked_objects) != 1:
            return None
        tracked = result.tracked_objects[0]
        center_x = tracked.bbox.x + tracked.bbox.width / 2.0
        center_y = tracked.bbox.y + tracked.bbox.height / 2.0
        if (
            tracked.age_frames < _MIN_TRACK_AGE_FRAMES
            or abs(tracked.velocity_x_px_s) <= 5.0
            or center_x <= FRAME_WIDTH / 2.0
            or center_y >= FRAME_HEIGHT / 2.0
        ):
            return None
        return tracked

    def _ensure_active_tracking(self) -> None:
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        self._ensure_main(CameraRole.OVERVIEW)
        self._ensure_mode(TurretControlMode.TRACKING)
        self._ensure_motor(MotorState.ON)
        if self.runtime.mediator.selected_target is not None:
            return
        self._wait_until(
            lambda: self._stable_overview_track() is not None
            and self.window.main_view.interaction_allowed(self.runtime.mediator.session_gate, monotonic_ns()),
            4.0,
            "stable Overview track was not available",
        )
        displayed = self.window.main_view.displayed_result
        tracked = self._stable_overview_track()
        assert displayed is not None and tracked is not None
        rendered = self.window.main_view.rendered_rect()
        height, width = displayed.frame.image.shape[:2]
        center_x = tracked.bbox.x + tracked.bbox.width / 2.0
        center_y = tracked.bbox.y + tracked.bbox.height / 2.0
        click = QPoint(
            round(rendered.left() + center_x * rendered.width() / width),
            round(rendered.top() + center_y * rendered.height() / height),
        )
        baseline = self._command_count(CommandCode.SET_VELOCITY)
        QTest.mouseClick(self.window.main_view, Qt.MouseButton.LeftButton, pos=click)
        self.application.processEvents()
        if self.runtime.mediator.selected_target is None:
            raise SoakFailure("TRACKING selection was not established")
        self._wait_until(lambda: self._has_nonzero_velocity_since(baseline), 2.0, "nonzero SET_VELOCITY did not reach endpoint")

    def _deselect_tracking(self) -> None:
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        if self.runtime.mediator.selected_target is None:
            return
        baseline = self._command_count(CommandCode.SET_VELOCITY)
        rendered = self.window.main_view.rendered_rect()
        click = QPoint(round(rendered.left() + rendered.width() * 0.08), round(rendered.top() + rendered.height() * 0.08))
        QTest.mouseClick(self.window.main_view, Qt.MouseButton.LeftButton, pos=click)
        self.application.processEvents()
        if self.runtime.mediator.selected_target is not None:
            raise SoakFailure("normal TRACKING deselect did not clear selection")
        self._wait_until(lambda: self._has_zero_velocity_since(baseline), 1.0, "normal TRACKING deselect did not stop motion")

    def _preview_swap_round_trip(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        start = self.runtime.mediator.main_camera
        QTest.mouseClick(self.window.preview_view, Qt.MouseButton.LeftButton, pos=self.window.preview_view.rect().center())
        self.application.processEvents()
        self._wait_until(lambda: self.runtime.mediator.main_camera is not start, 1.0, "preview swap did not change main camera")
        QTest.mouseClick(self.window.preview_view, Qt.MouseButton.LeftButton, pos=self.window.preview_view.rect().center())
        self.application.processEvents()
        self._wait_until(lambda: self.runtime.mediator.main_camera is start, 1.0, "preview round-trip did not restore main camera")

    def _stale_resume(self) -> None:
        assert self.window is not None
        self._phase = "camera stale/resume"
        generation = self.runtime.overview_pipeline.generation
        revision = self.runtime.overview_pipeline.latest_result.snapshot().revision
        self._intentional_stale = True
        self.runtime.overview_producer.pause()
        try:
            self._wait_until(lambda: self.window.main_view.is_stale, 1.2, "Overview stale state was not observed")
            if self.runtime.overview_pipeline.generation != generation:
                raise SoakFailure("Overview generation changed during stale pause")
            self.runtime.overview_producer.resume()
            self._wait_until(
                lambda: self.runtime.overview_pipeline.latest_result.snapshot().revision > revision
                and not self.window.main_view.is_stale,
                1.5,
                "Overview did not resume Vision progress after stale",
            )
            if self.runtime.overview_pipeline.generation != generation:
                raise SoakFailure("Overview generation changed across stale/resume")
            self.metrics.event_counts["stale_resume"] += 1
        finally:
            self.runtime.overview_producer.resume()
            self._intentional_stale = False
            # The pre-fault sample may be older than the progress interval.
            # Start a fresh progress window only after recovery has completed.
            self._last_progress_sample = None

    def _generation_restart(self) -> None:
        assert self.window is not None
        self._phase = "generation restart"
        old_generation = self.runtime.overview_pipeline.generation
        self._intentional_stale = True
        self.runtime.overview_producer.pause()
        try:
            self._pump_for(0.05)
            session = self.runtime.overview_pipeline.start()
            if session.generation != old_generation + 1:
                raise SoakFailure("Overview generation did not increment exactly once")
            self._wait_until(
                lambda: self.runtime.mediator.session_gate.accepted_generation(CameraRole.OVERVIEW) == old_generation + 1,
                1.0,
                "new Overview generation barrier was not accepted",
            )
            if self.runtime.mediator.selected_target is not None:
                raise SoakFailure("selection survived Overview generation restart")
            self.runtime.overview_producer.resume()
            self._wait_until(
                lambda: self.window.main_view.displayed_result is not None
                and self.window.main_view.displayed_result.frame.generation == old_generation + 1,
                1.5,
                "new Overview generation did not resume display progress",
            )
            self.metrics.event_counts["generation_restart"] += 1
        finally:
            self.runtime.overview_producer.resume()
            self._intentional_stale = False
            self._last_progress_sample = None

    def _turret_disconnect_recovery(self) -> None:
        self._phase = "turret disconnect/recovery"
        self.factory.fail_open_count = 1
        active_transport = self.factory.current_transport
        if active_transport is None:
            raise SoakFailure("no active FakeTransport available for disconnect injection")
        active_transport.queue_read_failure(FakeReadFailure.DISCONNECT)
        self._wait_until(
            lambda: self.runtime.mediator.turret_state.connection_state is not TurretConnectionState.READY,
            1.5,
            "Turret disconnect was not observed",
        )
        if self.runtime.mediator.selected_target is not None:
            raise SoakFailure("selection survived Turret disconnect")
        self._wait_until(
            lambda: self.runtime.mediator.turret_state.connection_state is TurretConnectionState.READY
            and self.runtime.mediator.turret_state.motor_state is MotorState.OFF,
            3.5,
            "Turret did not recover to READY/OFF",
        )
        self.metrics.event_counts["disconnect_recovery"] += 1

    def _emergency_during_tracking(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        assert self.window is not None
        self._phase = "Emergency during tracking"
        baseline = self._command_count(CommandCode.EMERGENCY_STOP)
        if self.runtime.mediator.selected_target is None or not self._has_any_nonzero_velocity():
            raise SoakFailure("Emergency precondition active TRACKING was not met")
        QTest.mouseClick(self.window.emergency_button, Qt.MouseButton.LeftButton)
        self.application.processEvents()
        if self.runtime.mediator.selected_target is not None:
            raise SoakFailure("Emergency did not clear selection immediately")
        self._wait_until(lambda: self._command_count(CommandCode.EMERGENCY_STOP) > baseline, 1.5, "EMERGENCY_STOP did not reach endpoint")
        self._wait_until(
            lambda: self.runtime.mediator.turret_state.connection_state is TurretConnectionState.READY
            and self.runtime.mediator.turret_state.motor_state is MotorState.ON,
            1.5,
            "Emergency did not preserve READY/MotorState.ON",
        )
        self.metrics.event_counts["emergency"] += 1

    def _observe_periodic(self) -> None:
        now = monotonic()
        if now < self._next_sample_at:
            return
        overview = self.runtime.overview_pipeline.latest_result.snapshot()
        stereo = self.runtime.stereo_left_pipeline.latest_result.snapshot()
        sample = _ProcessSample(
            elapsed_s=self.metrics.elapsed_s,
            cycle=self._cycle,
            rss_kib=_read_rss_kib(),
            thread_count=len(threading.enumerate()),
            overview_revision=overview.revision,
            overview_generation=self.runtime.overview_pipeline.generation,
            stereo_revision=stereo.revision,
            stereo_generation=self.runtime.stereo_left_pipeline.generation,
        )
        progress_base = self._last_progress_sample
        if progress_base is None:
            self._last_progress_sample = sample
        elif sample.elapsed_s - progress_base.elapsed_s >= self.sample_seconds * 0.9:
            if (
                not self._intentional_stale
                and self.runtime.overview_producer.is_alive()
                and not self.runtime.overview_producer.is_paused
                and sample.overview_revision <= progress_base.overview_revision
            ):
                raise SoakFailure("Overview producer is alive but Vision revision stopped progressing")
            if (
                self.runtime.stereo_left_producer.is_alive()
                and not self.runtime.stereo_left_producer.is_paused
                and sample.stereo_revision <= progress_base.stereo_revision
            ):
                raise SoakFailure("Stereo Left producer is alive but Vision revision stopped progressing")
            self._last_progress_sample = sample
        self.metrics.samples.append(sample)
        self._last_sample = sample
        self.metrics.thread_peak = max(self.metrics.thread_peak, sample.thread_count)
        if sample.rss_kib is not None:
            self._observe_rss_peak(sample.rss_kib)
        self._next_sample_at = now + self.sample_seconds
        if now >= self._next_heartbeat_at:
            state = self.runtime.mediator.turret_state
            rss = "unavailable" if sample.rss_kib is None else f"{sample.rss_kib / 1024:.1f} MiB"
            print(
                f"HEARTBEAT elapsed={sample.elapsed_s:.1f}s cycles={self._cycle} "
                f"Overview=rev{sample.overview_revision}/gen{sample.overview_generation} "
                f"StereoLeft=rev{sample.stereo_revision}/gen{sample.stereo_generation} "
                f"Turret={state.connection_state.value}/{state.motor_state.value}/{state.control_mode.value} "
                f"RSS={rss} threads={sample.thread_count}",
                flush=True,
            )
            self._next_heartbeat_at = now + self.heartbeat_seconds

    def _observe_rss_peak(self, value: int) -> None:
        self.metrics.rss_peak_kib = (
            value
            if self.metrics.rss_peak_kib is None
            else max(self.metrics.rss_peak_kib, value)
        )

    def _assert_required_events(self) -> None:
        for key in ("nominal_cycles", "stale_resume", "generation_restart", "disconnect_recovery", "emergency"):
            if self.metrics.event_counts[key] <= 0:
                raise SoakFailure(f"required event type was not executed: {key}")

    def _assert_progress_since_warmup(self) -> None:
        if self.runtime.overview_pipeline.latest_result.snapshot().revision <= self.metrics.initial_overview_revision:
            raise SoakFailure("Overview Vision made no post-warm-up progress")
        if self.runtime.stereo_left_pipeline.latest_result.snapshot().revision <= self.metrics.initial_stereo_revision:
            raise SoakFailure("Stereo Left Vision made no post-warm-up progress")
        if not self.runtime.overview_worker.is_alive() or not self.runtime.stereo_left_worker.is_alive() or not self.runtime.turret_worker.is_alive():
            raise SoakFailure("an owned worker died before soak completion")

    def _assert_no_stale_selection_or_pending(self) -> None:
        if self.runtime.mediator.selected_target is not None:
            raise SoakFailure("stale selected target remained after nominal cycle")
        if self.runtime.mediator.pending_control_mode is not None:
            raise SoakFailure("pending control mode remained after nominal cycle")

    def _assert_shutdown_threads(self) -> None:
        if self.runtime.overview_producer.is_alive() or self.runtime.stereo_left_producer.is_alive():
            raise SoakFailure("synthetic producer remained alive after shutdown")
        if self.runtime.overview_worker.is_alive() or self.runtime.stereo_left_worker.is_alive():
            raise SoakFailure("CameraWorker remained alive after shutdown")
        if self.runtime.turret_worker.is_alive():
            raise SoakFailure("TurretWorker remained alive after shutdown")
        owned = [thread.name for thread in threading.enumerate() if thread.name.startswith(_OWNED_THREAD_PREFIXES)]
        if owned:
            raise SoakFailure(f"owned worker threads remained after shutdown: {owned}")
        current_non_daemon = [thread for thread in threading.enumerate() if not thread.daemon]
        # MainThread existed in the baseline. New stable Qt/system daemon helpers are ignored.
        if len(current_non_daemon) > self.metrics.thread_baseline:
            raise SoakFailure(
                f"new non-daemon thread remained after shutdown: {len(current_non_daemon)} > baseline {self.metrics.thread_baseline}"
            )

    def _command_count(self, command: CommandCode) -> int:
        return self.factory.endpoint.command_count(command)

    def _has_nonzero_velocity_since(self, baseline: int) -> bool:
        return self.factory.endpoint.has_nonzero_velocity_since(baseline)

    def _has_zero_velocity_since(self, baseline: int) -> bool:
        return self.factory.endpoint.has_zero_velocity_since(baseline)

    def _has_any_nonzero_velocity(self) -> bool:
        return self._has_nonzero_velocity_since(0)

    def _memory_growth_suspect(self) -> bool:
        values = [sample.rss_kib for sample in self.metrics.samples if sample.rss_kib is not None]
        return _rss_growth_suspect(values)


def _read_rss_kib() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _linear_rss_trend_mib_per_minute(samples: Sequence[_ProcessSample]) -> float | None:
    points = [(sample.elapsed_s, sample.rss_kib) for sample in samples if sample.rss_kib is not None]
    if len(points) < 2:
        return None
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        return 0.0
    slope_kib_s = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope_kib_s * 60.0 / 1024.0


def _rss_growth_suspect(values: Sequence[int]) -> bool:
    """Detect sustained post-warm-up RSS growth without an absolute threshold."""
    if len(values) < 8:
        return False

    mean_x = (len(values) - 1) / 2.0
    mean_y = statistics.fmean(values)
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    if denominator == 0.0:
        return False
    slope = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values)
    ) / denominator
    if slope <= 0.0:
        return False

    segments: list[Sequence[int]] = []
    for segment_index in range(4):
        start = len(values) * segment_index // 4
        end = len(values) * (segment_index + 1) // 4
        segment = values[start:end]
        if not segment:
            return False
        segments.append(segment)
    medians = [statistics.median(segment) for segment in segments]
    transitions = list(pairwise(medians))
    strict_rises = sum(after > before for before, after in transitions)
    regressions = sum(after < before for before, after in transitions)
    first_half_median = statistics.median(values[: len(values) // 2])
    second_half_median = statistics.median(values[len(values) // 2 :])
    return (
        strict_rises >= 2
        and regressions <= 1
        and second_half_median > first_half_median
        and medians[-1] > medians[0]
    )


def _percentile95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[rank]


def _format_mib(kib: int | None) -> str:
    return "RSS measurement unavailable" if kib is None else f"{kib / 1024:.2f} MiB"


def _vision_rate(delta_revision: int, elapsed_s: float) -> float:
    return delta_revision / elapsed_s if elapsed_s > 0.0 else 0.0


def print_report(result: str, soak: SoftwareSoak) -> None:
    metrics = soak.metrics
    elapsed = metrics.elapsed_s
    overview_revision = soak.runtime.overview_pipeline.latest_result.snapshot().revision
    stereo_revision = soak.runtime.stereo_left_pipeline.latest_result.snapshot().revision
    overview_delta = max(0, overview_revision - metrics.initial_overview_revision)
    stereo_delta = max(0, stereo_revision - metrics.initial_stereo_revision)
    command_counts = {command.name: soak._command_count(command) for command in (
        CommandCode.MOTOR_ON,
        CommandCode.MOTOR_OFF,
        CommandCode.MOVE_RELATIVE,
        CommandCode.SET_VELOCITY,
        CommandCode.EMERGENCY_STOP,
    )}
    cycle_durations = metrics.cycle_durations_s
    trend = _linear_rss_trend_mib_per_minute(metrics.samples)
    warm_to_final = None
    if metrics.rss_warmup_kib is not None and metrics.rss_final_kib is not None:
        warm_to_final = metrics.rss_final_kib - metrics.rss_warmup_kib
    state = soak.runtime.mediator.turret_state

    print("\n=== SOFTWARE SOAK REPORT ===")
    print(f"requested / actual elapsed: {metrics.requested_s:.1f}s / {elapsed:.1f}s")
    print(f"cycles completed: {metrics.event_counts['nominal_cycles']}")
    print(f"nominal cycles: {metrics.event_counts['nominal_cycles']}")
    print(f"stale/resume count: {metrics.event_counts['stale_resume']}")
    print(f"generation restart count: {metrics.event_counts['generation_restart']}")
    print(f"turret disconnect/recovery count: {metrics.event_counts['disconnect_recovery']}")
    print(f"Emergency count: {metrics.event_counts['emergency']}")
    print("Overview:")
    print(f"  frames produced: {soak.runtime.overview_producer.frames_produced}")
    print(f"  Vision revision delta: {overview_delta}")
    print(f"  generation: {soak.runtime.overview_pipeline.generation}")
    print(f"  observed rate: {_vision_rate(overview_delta, elapsed):.2f} rev/s")
    print("Stereo Left:")
    print(f"  frames produced: {soak.runtime.stereo_left_producer.frames_produced}")
    print(f"  Vision revision delta: {stereo_delta}")
    print(f"  generation: {soak.runtime.stereo_left_pipeline.generation}")
    print(f"  observed rate: {_vision_rate(stereo_delta, elapsed):.2f} rev/s")
    print("Turret:")
    print(f"  final connection state: {state.connection_state.value}")
    print(f"  final motor state: {state.motor_state.value}")
    print(f"  final mode: {state.control_mode.value}")
    print(f"  reconnect transports created: {max(0, soak.factory.total_transports_created - 1)}")
    print("Endpoint command counts:")
    for name, count in command_counts.items():
        print(f"  {name}: {count}")
    print("Threads:")
    print(f"  baseline: {metrics.thread_baseline}")
    print(f"  warm-up: {metrics.thread_warmup}")
    print(f"  peak: {metrics.thread_peak}")
    print(f"  final-after-shutdown: {metrics.thread_final}")
    owned_threads = [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(_OWNED_THREAD_PREFIXES)
    ]
    owned_workers_stopped = (
        not soak.runtime.overview_producer.is_alive()
        and not soak.runtime.stereo_left_producer.is_alive()
        and not soak.runtime.overview_worker.is_alive()
        and not soak.runtime.stereo_left_worker.is_alive()
        and not soak.runtime.turret_worker.is_alive()
        and not owned_threads
    )
    print(f"  owned workers all stopped: {'yes' if owned_workers_stopped else 'NO'}")
    print("RSS:")
    print(f"  initial: {_format_mib(metrics.rss_initial_kib)}")
    print(f"  warm-up: {_format_mib(metrics.rss_warmup_kib)}")
    print(f"  final: {_format_mib(metrics.rss_final_kib)}")
    print(f"  peak: {_format_mib(metrics.rss_peak_kib)}")
    print(f"  warm-up→final delta: {'RSS measurement unavailable' if warm_to_final is None else f'{warm_to_final / 1024:.2f} MiB'}")
    print(
        "  approximate trend MiB/min "
        f"(recent <= {_SAMPLE_WINDOW} periodic samples): "
        f"{'RSS measurement unavailable' if trend is None else f'{trend:.3f}'}"
    )
    print("Cycle timing:")
    print(f"  median/p95 window: recent <= {_CYCLE_WINDOW} cycles")
    print(f"  min (full run): {metrics.cycle_duration_min_s or 0.0:.3f}s")
    print(f"  median: {statistics.median(cycle_durations) if cycle_durations else 0.0:.3f}s")
    print(f"  p95: {_percentile95(cycle_durations):.3f}s")
    print(f"  max (full run): {metrics.cycle_duration_max_s or 0.0:.3f}s")
    print(f"RESULT: {result}")


def run_soak(
    duration_seconds: float,
    *,
    max_cycles: int | None = None,
    fault_cadence: tuple[int, int, int, int] = _DEFAULT_FAULT_CADENCE,
    heartbeat_seconds: float = _HEARTBEAT_S,
    sample_seconds: float = _SAMPLE_S,
) -> tuple[str, SoftwareSoak]:
    soak = SoftwareSoak(
        duration_seconds=duration_seconds,
        max_cycles=max_cycles,
        fault_cadence=fault_cadence,
        heartbeat_seconds=heartbeat_seconds,
        sample_seconds=sample_seconds,
    )
    result = "FAIL"
    try:
        result, _metrics = soak.run()
    except RuntimeError as exc:
        print(f"SOAK FAILURE: {exc}", flush=True)
        result = "FAIL"
    print_report(result, soak)
    return result, soak


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NavMin hardware-free software stability soak.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=600.0,
        help="requested soak duration in seconds (> 0; default: 600)",
    )
    args = parser.parse_args(argv)
    if args.duration_seconds <= 0.0:
        parser.error("--duration-seconds must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result, _soak = run_soak(args.duration_seconds)
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
