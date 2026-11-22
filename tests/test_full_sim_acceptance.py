from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest

from navmin.contracts import (
    CameraRole,
    MotorState,
    MoveRelativeCommand,
    TurretConnectionState,
    TurretControlMode,
)
from navmin.diagnostics.rtp_jpeg_appsrc import check_appsrc_rtp_jpeg_sender_runtime
from navmin.launch_options import (
    CameraSourceMode,
    LaunchOptions,
    StereoRightMode,
    Stm32Mode,
)
from navmin.launch_runtime import build_launch_runtime_bindings
from navmin.launcher import LauncherPaths, load_application_inputs
from navmin.turret.worker import TurretWorker
from navmin.vision.camera_worker import CameraWorker, build_camera_worker
from navmin.vision.gstreamer_source import (
    RTP_JPEG_GSTREAMER_ELEMENTS,
    GStreamerUnavailableError,
    find_missing_gstreamer_elements,
    initialize_gstreamer_runtime,
)
from navmin.vision.pipeline import (
    overview_corrector,
    stereo_left_corrector,
    stereo_right_corrector,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gstreamer_blocker() -> str | None:
    sender_error = check_appsrc_rtp_jpeg_sender_runtime()
    if sender_error is not None:
        return sender_error
    try:
        initialize_gstreamer_runtime()
        missing = find_missing_gstreamer_elements(RTP_JPEG_GSTREAMER_ELEMENTS)
    except GStreamerUnavailableError as exc:
        return str(exc)
    if missing:
        return "missing receiver elements: " + ", ".join(missing)
    return None


def _wait_camera_result(worker: CameraWorker, *, timeout_s: float = 8.0):
    snapshot = worker.pipeline.latest_result.snapshot()
    deadline = monotonic() + timeout_s
    while snapshot.value is None:
        remaining = deadline - monotonic()
        assert remaining > 0.0, (
            f"timed out waiting for {worker.pipeline.camera.value} VisionResult; "
            f"status={worker.pipeline.status.get()!r}"
        )
        next_snapshot = worker.pipeline.latest_result.wait_for_revision(
            snapshot.revision,
            remaining,
        )
        assert next_snapshot is not None
        snapshot = next_snapshot
    return snapshot


def _wait_turret_state(worker: TurretWorker, predicate, *, timeout_s: float = 3.0):
    snapshot = worker.state_updates.snapshot()
    deadline = monotonic() + timeout_s
    while snapshot.value is None or not predicate(snapshot.value):
        remaining = deadline - monotonic()
        assert remaining > 0.0, f"timed out waiting for turret state: {snapshot.value!r}"
        next_snapshot = worker.state_updates.wait_for_revision(
            snapshot.revision,
            remaining,
        )
        assert next_snapshot is not None
        snapshot = next_snapshot
    return snapshot.value


def _wait_image_change(
    worker: CameraWorker,
    *,
    revision: int,
    baseline: np.ndarray,
    timeout_s: float = 8.0,
):
    deadline = monotonic() + timeout_s
    current_revision = revision
    while True:
        remaining = deadline - monotonic()
        assert remaining > 0.0, "timed out waiting for FULL SIM camera response to turret motion"
        snapshot = worker.pipeline.latest_result.wait_for_revision(
            current_revision,
            remaining,
        )
        assert snapshot is not None
        current_revision = snapshot.revision
        result = snapshot.value
        if result is not None and not np.array_equal(result.frame.image, baseline):
            return result


_GSTREAMER_BLOCKER = _gstreamer_blocker()


@pytest.mark.skipif(
    _GSTREAMER_BLOCKER is not None,
    reason=_GSTREAMER_BLOCKER or "GStreamer unavailable",
)
def test_canonical_full_sim_composition_closes_relative_move_through_rtp_and_vision() -> None:
    """Acceptance: launcher bindings close one FULL SIM world through production camera ingress."""
    inputs = load_application_inputs(
        LauncherPaths(config=REPO_ROOT / "config" / "defaults" / "navmin.json")
    )
    bindings = build_launch_runtime_bindings(
        inputs,
        LaunchOptions(
            overview=CameraSourceMode.SIM,
            stereo_left=CameraSourceMode.SIM,
            stereo_right=StereoRightMode.SIM,
            stm32=Stm32Mode.SIM,
        ),
    )
    simulation = bindings.simulation
    assert simulation is not None
    assert simulation.rig is not None
    assert simulation.stm32_endpoint is not None
    assert simulation.rig.plant is simulation.plant
    assert simulation.stm32_endpoint.plant is simulation.plant

    config = bindings.inputs.config
    cameras = config.vision.cameras
    workers = (
        build_camera_worker(
            camera=CameraRole.OVERVIEW,
            config=cameras.overview,
            corrector=overview_corrector(bindings.inputs.overview_calibration),
            source_factory=bindings.factories.overview_source_factory,
        ),
        build_camera_worker(
            camera=CameraRole.STEREO_LEFT,
            config=cameras.stereo_left,
            corrector=stereo_left_corrector(bindings.inputs.stereo_calibration),
            source_factory=bindings.factories.stereo_left_source_factory,
        ),
        build_camera_worker(
            camera=CameraRole.STEREO_RIGHT,
            config=cameras.stereo_right,
            corrector=stereo_right_corrector(bindings.inputs.stereo_calibration),
            source_factory=bindings.factories.stereo_right_source_factory,
        ),
    )
    assert bindings.factories.turret_transport_factory is not None
    turret = TurretWorker(
        config.turret,
        config.tracking_control,
        transport_factory=bindings.factories.turret_transport_factory,
    )

    started_workers: list[CameraWorker] = []
    turret_started = False
    try:
        for worker in workers:
            worker.start()
            started_workers.append(worker)
        turret.start()
        turret_started = True

        overview_before = _wait_camera_result(workers[0])
        _wait_camera_result(workers[1])
        _wait_camera_result(workers[2])
        for worker in workers:
            diagnostics = worker.pipeline.diagnostics
            assert diagnostics.frames_received >= 1
            assert (diagnostics.decoded_width, diagnostics.decoded_height) == (1296, 976)

        _wait_turret_state(
            turret,
            lambda state: state.connection_state is TurretConnectionState.READY,
        )
        assert turret.motor_on() is True
        _wait_turret_state(
            turret,
            lambda state: state.motor_state is MotorState.ON,
        )
        if turret.current_state.control_mode is not TurretControlMode.RELATIVE:
            assert turret.set_control_mode(TurretControlMode.RELATIVE) is True
            _wait_turret_state(
                turret,
                lambda state: state.control_mode is TurretControlMode.RELATIVE,
            )

        baseline = np.array(overview_before.value.frame.image, copy=True)
        yaw_before = simulation.plant.state.yaw_deg
        move_deg = config.turret.axes.x.max_relative_move_deg
        assert move_deg > 0.0
        observed_delta_deg = min(0.25, move_deg * 0.5)
        turret.submit_move_relative(MoveRelativeCommand(move_deg, 0.0))

        deadline = monotonic() + 5.0
        while simulation.plant.state.yaw_deg <= yaw_before + observed_delta_deg:
            assert monotonic() < deadline, "synthetic turret did not execute MOVE_RELATIVE"
            sleep(0.01)

        changed = _wait_image_change(
            workers[0],
            revision=overview_before.revision,
            baseline=baseline,
        )
        assert changed.frame.camera is CameraRole.OVERVIEW
        assert simulation.plant.state.yaw_deg > yaw_before
    finally:
        for worker in reversed(started_workers):
            assert worker.stop(3.0)
        if turret_started and turret.is_alive():
            turret.shutdown(3.0)
