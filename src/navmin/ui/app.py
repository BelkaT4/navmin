"""Reusable launch boundary for externally composed NavMin dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6.QtWidgets import QApplication

from navmin.concurrency import LatestValue
from navmin.contracts import CameraRole, TurretState
from navmin.core import Mediator

from .bridge import CameraUiBinding
from .main_window import MainWindow


def run_ui(
    *,
    mediator: Mediator,
    camera_bindings: Mapping[CameraRole, CameraUiBinding],
    turret_states: LatestValue[TurretState],
    camera_stale_timeout_ms: int,
    argv: Sequence[str] = (),
) -> int:
    """Run only the Qt UI; application composition owns all workers."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(list(argv))
    window = MainWindow(
        mediator=mediator,
        camera_bindings=camera_bindings,
        turret_states=turret_states,
        camera_stale_timeout_ms=camera_stale_timeout_ms,
    )
    window.showFullScreen()
    return application.exec()


__all__ = ["run_ui"]
