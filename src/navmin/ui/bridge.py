"""Revision-aware main-thread bridge from worker-owned state to the UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from queue import Empty

from PyQt6.QtCore import QObject, QTimer

from navmin.concurrency import CameraSessionBarrierChannel, LatestValue
from navmin.contracts import (
    CameraRole,
    CameraSessionStarted,
    CameraStatus,
    TurretState,
    VisionResult,
)
from navmin.core import Mediator


@dataclass(frozen=True)
class CameraUiBinding:
    """Existing Vision channels consumed by one camera's UI presentation."""

    camera: CameraRole
    session_barriers: CameraSessionBarrierChannel
    latest_result: LatestValue[VisionResult]
    status: LatestValue[CameraStatus]


class UiStatePump(QObject):
    """Poll freshest worker state without creating queued per-frame callbacks."""

    INTERVAL_MS = 16

    def __init__(
        self,
        *,
        mediator: Mediator,
        camera_bindings: Mapping[CameraRole, CameraUiBinding],
        turret_states: LatestValue[TurretState],
        on_camera_session: Callable[[CameraSessionStarted], None],
        on_vision_result: Callable[[VisionResult], None],
        on_camera_status: Callable[[CameraStatus], None],
        on_turret_state: Callable[[TurretState], None],
        on_presentation_tick: Callable[[], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._mediator = mediator
        self._camera_bindings = dict(camera_bindings)
        self._turret_states = turret_states
        self._on_camera_session = on_camera_session
        self._on_vision_result = on_vision_result
        self._on_camera_status = on_camera_status
        self._on_turret_state = on_turret_state
        self._on_presentation_tick = on_presentation_tick
        self._vision_revisions = {camera: 0 for camera in camera_bindings}
        self._status_revisions = {camera: 0 for camera in camera_bindings}
        self._turret_revision = 0

        self.timer = QTimer(self)
        self.timer.setInterval(self.INTERVAL_MS)
        self.timer.timeout.connect(self.pump_once)

    def start(self) -> None:
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()

    def pump_once(self) -> None:
        """Drain every barrier first, then process each freshest latest value."""
        for binding in self._camera_bindings.values():
            self._drain_barriers(binding)

        for binding in self._camera_bindings.values():
            self._process_vision(binding)
        self._process_turret_state()
        for binding in self._camera_bindings.values():
            self._process_camera_status(binding)
        self._on_presentation_tick()

    def _drain_barriers(self, binding: CameraUiBinding) -> None:
        while True:
            try:
                session = binding.session_barriers.receive_nowait()
            except Empty:
                return
            if self._mediator.accept_camera_session(session):
                self._on_camera_session(session)

    def _process_vision(self, binding: CameraUiBinding) -> None:
        snapshot = binding.latest_result.snapshot()
        consumed = self._vision_revisions[binding.camera]
        if snapshot.revision <= consumed:
            return
        result = snapshot.value
        if result is None:
            self._vision_revisions[binding.camera] = snapshot.revision
            return

        accepted_generation = self._mediator.session_gate.accepted_generation(
            result.frame.camera
        )
        if accepted_generation is not None and result.frame.generation < accepted_generation:
            # A late result from an older session can never become acceptable.
            self._vision_revisions[binding.camera] = snapshot.revision
            return

        if not self._mediator.accept_vision_result(result):
            # A newer result can race ahead of its lossless barrier. Retain its
            # revision so the same freshest result is reconsidered next tick.
            return
        self._vision_revisions[binding.camera] = snapshot.revision
        self._on_vision_result(result)

    def _process_turret_state(self) -> None:
        snapshot = self._turret_states.snapshot()
        if snapshot.revision <= self._turret_revision or snapshot.value is None:
            return
        self._mediator.accept_turret_state(snapshot.value)
        self._turret_revision = snapshot.revision
        self._on_turret_state(snapshot.value)

    def _process_camera_status(self, binding: CameraUiBinding) -> None:
        snapshot = binding.status.snapshot()
        if snapshot.revision <= self._status_revisions[binding.camera]:
            return
        self._status_revisions[binding.camera] = snapshot.revision
        if snapshot.value is not None:
            self._on_camera_status(snapshot.value)


__all__ = ["CameraUiBinding", "UiStatePump"]
