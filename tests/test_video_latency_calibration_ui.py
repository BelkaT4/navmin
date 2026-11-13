from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from navmin.concurrency import LatestValue
from navmin.diagnostics.live_video_latency_calibration import (
    LiveCalibrationPhase,
    LiveVideoLatencyCalibrationState,
)
from navmin.ui.video_latency_calibration_window import VideoLatencyCalibrationWindow


@pytest.fixture(scope="module")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_video_latency_window_starts_and_cancels_without_owning_hardware(
    qt_application: QApplication,
    monkeypatch,
) -> None:
    del qt_application
    states: LatestValue[LiveVideoLatencyCalibrationState] = LatestValue()
    states.publish(
        LiveVideoLatencyCalibrationState(
            LiveCalibrationPhase.IDLE,
            "ready",
            run_count=3,
        )
    )
    calls: list[str] = []
    window = VideoLatencyCalibrationWindow(
        owner=None,
        states=states,
        start_calibration=lambda: calls.append("start") or True,
        cancel_calibration=lambda: calls.append("cancel"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    window.start_button.click()
    assert calls == ["start"]

    states.publish(
        LiveVideoLatencyCalibrationState(
            LiveCalibrationPhase.RUNNING,
            "running",
            run_index=1,
            run_count=3,
            current_rate_y_deg_s=6.0,
            flow_observations=12,
        )
    )
    window._poll()
    assert "Y=+6.0" in window.progress_label.text()
    assert window.cancel_button.isEnabled()

    window.cancel_button.click()
    assert calls == ["start", "cancel"]
    window.close()
