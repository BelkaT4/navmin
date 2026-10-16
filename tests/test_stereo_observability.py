from __future__ import annotations

import json
import logging

from navmin.contracts import CameraRole
from navmin.core.mediator import StereoDistancePublicationDiagnostics
from navmin.diagnostics.stereo_runtime import (
    StereoDiagnosticsReporter,
    StereoRuntimeDiagnosticSnapshot,
)
from navmin.vision.pipeline import CameraRuntimeDiagnostics
from navmin.vision.stereo_diagnostics import BoundedTimingSnapshot
from navmin.vision.stereo_distance_worker import StereoDistanceWorkerDiagnostics
from navmin.vision.stereo_pairing import StereoPairerDiagnostics
from navmin.vision.stereo_target_join import StereoTargetJoinerDiagnostics
from navmin.vision.stereo_timestamp_pairing import TimestampStereoPairerDiagnostics


def _camera(camera: CameraRole, frames: int, fps: float) -> CameraRuntimeDiagnostics:
    return CameraRuntimeDiagnostics(
        camera=camera,
        generation=2,
        frames_received=frames,
        first_frame_timestamp_ns=1,
        latest_frame_timestamp_ns=2,
        decoded_width=1296,
        decoded_height=976,
        capture_id_present_count=0,
        capture_id_missing_count=frames,
        received_fps=fps,
    )


def _timing(count: int, latest: int) -> BoundedTimingSnapshot:
    return BoundedTimingSnapshot(count, count, latest, latest, latest, latest, latest)


def _snapshot(pair_count: int = 4) -> StereoRuntimeDiagnosticSnapshot:
    return StereoRuntimeDiagnosticSnapshot(
        captured_monotonic_ns=123,
        left_camera=_camera(CameraRole.STEREO_LEFT, 10, 20.0),
        right_camera=_camera(CameraRole.STEREO_RIGHT, 11, 21.0),
        exact_pairing=StereoPairerDiagnostics(10, 11, 0, 21, 1, 1, 0, 0, 2),
        timestamp_pairing=TimestampStereoPairerDiagnostics(
            10,
            11,
            pair_count,
            1,
            2,
            3,
            4,
            5,
            6,
            2,
            30_000_000,
            _timing(pair_count, 8_000_000),
        ),
        target_join=StereoTargetJoinerDiagnostics(
            4, 4, 3, 1, 1, 0, 0, 0, 0, 0, 2
        ),
        distance_worker=StereoDistanceWorkerDiagnostics(
            3,
            1,
            3,
            2,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            None,
            None,
            _timing(3, 50_000_000),
            None,
        ),
        publication=StereoDistancePublicationDiagnostics(
            2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1
        ),
    )


class _Snapshots:
    def __init__(self) -> None:
        self.pair_count = 4

    def snapshot(self) -> StereoRuntimeDiagnosticSnapshot:
        return _snapshot(self.pair_count)


class _RateClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        current = self.value
        self.value += 5.0
        return current


def test_periodic_summary_is_one_aggregated_machine_readable_line(
    caplog,
) -> None:
    snapshots = _Snapshots()
    reporter = StereoDiagnosticsReporter(
        snapshots,  # type: ignore[arg-type]
        rate_clock=_RateClock(),
        logger=logging.getLogger("navmin.test.stereo_observability"),
    )

    with caplog.at_level("INFO", logger="navmin.test.stereo_observability"):
        reporter.emit_once()
        snapshots.pair_count = 9
        reporter.emit_once()

    messages = [message for message in caplog.messages if message.startswith("STEREO_DIAG")]
    assert len(messages) == 2
    assert "left_fps=20.000" in messages[1]
    assert "timestamp_pairs=9" in messages[1]
    assert "pair_rate_hz=1.000" in messages[1]
    assert "skew_p95_ms=8.000" in messages[1]
    assert "sgbm_p95_ms=50.000" in messages[1]
    assert "distance_stale=2" in messages[1]


def test_structured_snapshot_is_json_serializable() -> None:
    payload = _snapshot().to_dict()

    encoded = json.dumps(payload, sort_keys=True)

    assert '"decoded_width": 1296' in encoded
    assert payload["timestamp_pairing"]["skew"]["retained_sample_count"] == 4
