from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from navmin.concurrency import LatestValue
from navmin.diagnostics.live_video_latency_calibration import (
    LiveCalibrationPhase,
    LiveVideoLatencyCalibrationResult,
    LiveVideoLatencyCalibrationState,
)
from navmin.diagnostics.video_latency_calibration import VideoLatencyAggregate
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


def test_video_latency_window_marks_result_as_manual_config_step(
    qt_application: QApplication,
) -> None:
    del qt_application
    states: LatestValue[LiveVideoLatencyCalibrationState] = LatestValue()
    window = VideoLatencyCalibrationWindow(
        owner=None,
        states=states,
        start_calibration=lambda: True,
        cancel_calibration=lambda: None,
    )
    states.publish(
        LiveVideoLatencyCalibrationState(
            LiveCalibrationPhase.COMPLETE,
            "complete",
            run_count=3,
            result=LiveVideoLatencyCalibrationResult(
                status="PASS",
                camera="overview",
                axis="y",
                configured_video_latency_ms=0,
                command_rate_deg_s=6.0,
                segment_duration_s=0.8,
                run_count=3,
                runs=(),
                aggregate=VideoLatencyAggregate(
                    valid_run_count=3,
                    mean_latency_ms=140.0,
                    median_latency_ms=140.0,
                    stddev_latency_ms=2.0,
                    mad_latency_ms=2.0,
                    min_latency_ms=138,
                    max_latency_ms=142,
                ),
                recommended_video_latency_ms=140,
                conclusion="pass",
            ),
        )
    )
    window._poll()

    text = window.result_label.text()
    assert "автоматически в конфигурацию не записывается" in text
    assert "tracking-control → platform-motion → video-latency-ms" in text
    assert "config/navmin.json" in text
    window.close()
