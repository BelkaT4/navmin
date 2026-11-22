from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep

import pytest

from navmin.contracts import CameraRole, FrameTimestampSource
from navmin.diagnostics.rtp_jpeg_appsrc import (
    APPSRC_RTP_JPEG_SENDER_ELEMENTS,
    build_appsrc_rtp_jpeg_sender_pipeline_description,
    check_appsrc_rtp_jpeg_sender_runtime,
)
from navmin.diagnostics.synthetic_tracking_transport import (
    SyntheticTrackingRtpJpegSource,
    synthetic_tracking_rtp_source_config,
)
from navmin.diagnostics.synthetic_tracking_turret import (
    RealtimeSyntheticTrackingRig,
    SyntheticTurretPlant,
)
from navmin.diagnostics.synthetic_tracking_world import SyntheticTrackingWorld
from navmin.launcher import load_application_config
from navmin.vision.gstreamer_source import (
    RTP_JPEG_GSTREAMER_ELEMENTS,
    GStreamerUnavailableError,
    find_missing_gstreamer_elements,
    initialize_gstreamer_runtime,
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


def _read_one(source: SyntheticTrackingRtpJpegSource, timeout_s: float = 5.0):
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        failure = source.failure
        if failure is not None:
            raise AssertionError(f"synthetic RTP/JPEG source failed: {failure}")
        frame = source.read()
        if frame is not None:
            return frame
        sleep(0.01)
    raise AssertionError("timed out waiting for synthetic RTP/JPEG frame")


def test_appsrc_sender_description_reuses_established_rtp_jpeg_boundary() -> None:
    description = build_appsrc_rtp_jpeg_sender_pipeline_description(
        host="127.0.0.1",
        port=18_888,
        fps=20,
    )

    assert "appsrc name=source is-live=true format=time do-timestamp=false" in description
    assert "caps=image/jpeg,framerate=20/1" in description
    assert "! jpegparse ! rtpjpegpay pt=26" in description
    assert "udpsink host=127.0.0.1 port=18888 sync=false async=false" in description
    assert APPSRC_RTP_JPEG_SENDER_ELEMENTS == (
        "appsrc",
        "jpegparse",
        "rtpjpegpay",
        "udpsink",
    )


def test_synthetic_source_configs_are_fixed_localhost_rtp_endpoints() -> None:
    assert synthetic_tracking_rtp_source_config(CameraRole.OVERVIEW).port == 18_888
    assert synthetic_tracking_rtp_source_config(CameraRole.STEREO_LEFT).port == 18_889
    assert synthetic_tracking_rtp_source_config(CameraRole.STEREO_RIGHT).port == 18_890
    for camera in (
        CameraRole.OVERVIEW,
        CameraRole.STEREO_LEFT,
        CameraRole.STEREO_RIGHT,
    ):
        source = synthetic_tracking_rtp_source_config(camera)
        assert source.bind_address == "127.0.0.1"
        assert source.buffer_size == 1


_GSTREAMER_BLOCKER = _gstreamer_blocker()


@pytest.mark.skipif(
    _GSTREAMER_BLOCKER is not None,
    reason=_GSTREAMER_BLOCKER or "GStreamer unavailable",
)
def test_operator_sim_crosses_real_rtp_jpeg_receiver_for_all_cameras() -> None:
    config = load_application_config(REPO_ROOT / "config" / "defaults" / "navmin.json")
    rig = RealtimeSyntheticTrackingRig(
        SyntheticTrackingWorld(),
        SyntheticTurretPlant(config.turret),
        synchronize_stereo_pair=True,
    )
    sources = [
        SyntheticTrackingRtpJpegSource(
            rig,
            camera,
            synthetic_tracking_rtp_source_config(camera),
        )
        for camera in (
            CameraRole.OVERVIEW,
            CameraRole.STEREO_LEFT,
            CameraRole.STEREO_RIGHT,
        )
    ]
    started: list[SyntheticTrackingRtpJpegSource] = []
    try:
        for source in sources:
            source.start()
            started.append(source)
        frames = [_read_one(source) for source in sources]
    finally:
        for source in reversed(started):
            source.stop()

    for frame in frames:
        assert frame.image.shape == (976, 1296, 3)
        assert frame.capture_id is None
        assert frame.receive_timestamp_ns is not None
        assert frame.source_timestamp is not None
        assert frame.source_timestamp.source is FrameTimestampSource.RECEIVE_TIME
