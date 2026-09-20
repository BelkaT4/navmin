from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from navmin.contracts import CameraRole, CameraState, TurretConnectionState
from navmin.vision.pipeline import InMemoryFrameSource

_SMOKE_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_software_smoke.py"
_SMOKE = runpy.run_path(str(_SMOKE_TOOL), run_name="navmin_software_smoke")
FRAME_HEIGHT = _SMOKE["FRAME_HEIGHT"]
FRAME_WIDTH = _SMOKE["FRAME_WIDTH"]
SoftwareSmokeRuntime = _SMOKE["SoftwareSmokeRuntime"]
SyntheticFrameProducer = _SMOKE["SyntheticFrameProducer"]


def _wait_for_frame(source: InMemoryFrameSource, timeout: float = 1.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        frame = source.read()
        if frame is not None:
            return frame
        sleep(0.005)
    raise AssertionError("timed out waiting for synthetic frame")


def test_synthetic_camera_producers_are_independent_and_stop_bounded() -> None:
    overview_source = InMemoryFrameSource()
    stereo_source = InMemoryFrameSource()
    overview = SyntheticFrameProducer(
        camera=CameraRole.OVERVIEW,
        source=overview_source,
    )
    stereo = SyntheticFrameProducer(
        camera=CameraRole.STEREO_LEFT,
        source=stereo_source,
    )

    overview.start()
    stereo.start()
    try:
        overview_frame = _wait_for_frame(overview_source)
        stereo_frame = _wait_for_frame(stereo_source)

        assert overview_frame.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        assert stereo_frame.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        assert overview_frame.image.dtype == np.uint8
        assert stereo_frame.image.dtype == np.uint8
        assert overview_frame.image is not stereo_frame.image
        assert not np.array_equal(overview_frame.image, stereo_frame.image)
        assert overview.frames_produced >= 1
        assert stereo.frames_produced >= 1
    finally:
        assert overview.stop(timeout=1.0)
        assert stereo.stop(timeout=1.0)

    assert not overview.is_alive()
    assert not stereo.is_alive()


def _wait_until(predicate, *, timeout: float = 3.0, message: str) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError(message)


def test_runtime_starts_real_vision_and_turret_and_shuts_down_cleanly() -> None:
    runtime = SoftwareSmokeRuntime()
    runtime.start()
    try:
        _wait_until(
            lambda: runtime.turret_worker.current_state.connection_state
            is TurretConnectionState.READY,
            message="TurretWorker did not become READY",
        )
        _wait_until(
            lambda: (
                runtime.overview_pipeline.generation >= 1
                and runtime.overview_pipeline.status.get() is not None
                and runtime.overview_pipeline.status.get().state is CameraState.ONLINE
                and runtime.overview_pipeline.latest_result.get() is not None
            ),
            message="Overview did not publish an ONLINE VisionResult",
        )
        _wait_until(
            lambda: (
                runtime.stereo_left_pipeline.generation >= 1
                and runtime.stereo_left_pipeline.status.get() is not None
                and runtime.stereo_left_pipeline.status.get().state is CameraState.ONLINE
                and runtime.stereo_left_pipeline.latest_result.get() is not None
            ),
            message="Stereo Left did not publish an ONLINE VisionResult",
        )
    finally:
        runtime.shutdown(timeout=1.0)

    assert not runtime.overview_producer.is_alive()
    assert not runtime.stereo_left_producer.is_alive()
    assert not runtime.overview_worker.is_alive()
    assert not runtime.stereo_left_worker.is_alive()
    assert not runtime.turret_worker.is_alive()


def test_offscreen_ui_process_smoke_exits_cleanly() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    python_path = [str(root / "src"), str(root)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_software_smoke.py",
            "--auto-close-seconds",
            "1.0",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=12.0,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "Traceback (most recent call last)" not in output
    assert "software smoke shutdown complete" in output
