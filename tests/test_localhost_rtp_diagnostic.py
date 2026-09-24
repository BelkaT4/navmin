from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from navmin.contracts import CameraRole, FramePacket
from navmin.diagnostics.localhost_rtp import (
    GST_RECEIVER_ELEMENTS,
    GST_SENDER_ELEMENTS,
    LocalhostRtpJpegSender,
    RtpJpegSenderConfig,
    SenderProcessError,
    SyntheticDiagnosticFrameGenerator,
    build_sender_command,
    check_gstreamer_runtime,
    check_gstreamer_sender_runtime,
    diagnostic_camera_config,
    diagnostic_overview_calibration,
    diagnostic_stereo_calibration,
)
from navmin.vision.pipeline import overview_corrector, stereo_left_corrector
from navmin.vision.processors.legacy_14.processor import Legacy14VisionProcessor


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"port": 0}, "port"),
        ({"port": 65_536}, "port"),
        ({"port": 9999, "width": 0}, "width"),
        ({"port": 9999, "height": -1}, "height"),
        ({"port": 9999, "fps": 0}, "fps"),
        ({"port": 9999, "host": "0.0.0.0"}, "127.0.0.1"),
        ({"port": 9999, "camera": CameraRole.STEREO_RIGHT}, "camera"),
    ],
)
def test_sender_config_rejects_invalid_transport_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RtpJpegSenderConfig(**kwargs)


def test_sender_command_is_shell_free_rtp_jpeg_pipeline() -> None:
    command = build_sender_command(
        RtpJpegSenderConfig(port=18_888),
        gst_launch="/usr/bin/gst-launch-1.0",
    )

    assert command == (
        "/usr/bin/gst-launch-1.0",
        "-q",
        "fdsrc",
        "fd=0",
        "do-timestamp=true",
        "!",
        "image/jpeg,framerate=20/1",
        "!",
        "jpegparse",
        "!",
        "image/jpeg,width=320,height=240,framerate=20/1",
        "!",
        "rtpjpegpay",
        "pt=26",
        "!",
        "udpsink",
        "host=127.0.0.1",
        "port=18888",
        "sync=false",
        "async=false",
    )


def test_sender_command_declares_jpeg_caps_before_parser() -> None:
    command = build_sender_command(RtpJpegSenderConfig(port=18_888, fps=17))

    fdsrc_index = command.index("fdsrc")
    assert command[fdsrc_index : fdsrc_index + 7] == (
        "fdsrc",
        "fd=0",
        "do-timestamp=true",
        "!",
        "image/jpeg,framerate=17/1",
        "!",
        "jpegparse",
    )


def test_preflight_checks_executables_bindings_and_each_required_element() -> None:
    inspected: list[str] = []

    def runner(command, **kwargs):
        del kwargs
        inspected.append(command[1])
        return SimpleNamespace(
            returncode=1 if command[1] == "jpegparse" else 0,
            stdout="",
            stderr="missing jpegparse",
        )

    result = check_gstreamer_runtime(
        which=lambda executable: f"/usr/bin/{executable}",
        runner=runner,
        python_binding_check=lambda: None,
    )

    assert not result.ok
    assert inspected == [*GST_RECEIVER_ELEMENTS, *GST_SENDER_ELEMENTS]
    assert [failure.name for failure in result.failures] == [
        "GStreamer element jpegparse"
    ]
    assert result.failures[0].detail == "missing jpegparse"


def test_preflight_reports_missing_runtime_without_running_gst_inspect() -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError((args, kwargs))

    result = check_gstreamer_runtime(
        which=lambda _executable: None,
        runner=must_not_run,
        python_binding_check=lambda: "ModuleNotFoundError: gi",
    )

    assert not result.ok
    assert result.checks[0].detail == "not found in PATH"
    assert result.checks[1].detail == "not found in PATH"
    assert result.checks[2].detail == "ModuleNotFoundError: gi"
    assert all(
        check.detail == "gst-inspect-1.0 unavailable"
        for check in result.checks[3:]
    )


class _FakeProcess:
    def __init__(self, *, initial_returncode: int | None = None) -> None:
        self._returncode = initial_returncode
        self.stdin = _FakeStdin()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float] = []
        self.timeout_on_terminate = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -9

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        if self.timeout_on_terminate and self.kill_calls == 0:
            raise subprocess.TimeoutExpired("gst-launch-1.0", timeout)
        self._returncode = 0 if self._returncode is None else self._returncode
        return self._returncode


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False
        self.bytes_written = 0

    def write(self, data) -> int:
        if self.closed:
            raise ValueError("closed")
        size = len(data)
        self.bytes_written += size
        return size

    def flush(self) -> None:
        if self.closed:
            raise ValueError("closed")

    def close(self) -> None:
        self.closed = True


class _ProcessFactory:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.command: list[str] | None = None
        self.kwargs = None

    def __call__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        return self.process


def test_sender_process_lifecycle_terminates_bounded_without_shell() -> None:
    process = _FakeProcess()
    factory = _ProcessFactory(process)
    sender = LocalhostRtpJpegSender(
        RtpJpegSenderConfig(port=18_888),
        process_factory=factory,
        startup_probe_seconds=0.0,
        terminate_timeout_seconds=0.25,
    )

    sender.start()
    assert sender.is_running
    assert factory.command == list(sender.command)
    assert factory.kwargs is not None
    assert "shell" not in factory.kwargs
    assert factory.kwargs["stdin"] == subprocess.PIPE
    assert factory.kwargs["stdout"] is subprocess.DEVNULL
    assert factory.kwargs["text"] is False

    sender.stop()

    assert not sender.is_running
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [0.25]
    assert process.stdin.closed


def test_sender_process_kills_only_after_bounded_terminate_timeout() -> None:
    process = _FakeProcess()
    process.timeout_on_terminate = True
    sender = LocalhostRtpJpegSender(
        RtpJpegSenderConfig(port=18_888),
        process_factory=_ProcessFactory(process),
        startup_probe_seconds=0.0,
        terminate_timeout_seconds=0.5,
    )
    sender.start()

    sender.stop()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == [0.5, 0.5]


def test_sender_startup_failure_includes_captured_stderr() -> None:
    process = _FakeProcess(initial_returncode=2)

    def factory(command, **kwargs):
        del command
        kwargs["stderr"].write("pipeline construction failed\n")
        kwargs["stderr"].flush()
        return process

    sender = LocalhostRtpJpegSender(
        RtpJpegSenderConfig(port=18_888),
        process_factory=factory,
        startup_probe_seconds=0.0,
    )

    with pytest.raises(SenderProcessError, match="pipeline construction failed"):
        sender.start()

    assert not sender.is_running
    sender.stop()


def test_diagnostic_calibrations_and_camera_config_match_synthetic_size() -> None:
    overview = diagnostic_overview_calibration()
    stereo = diagnostic_stereo_calibration()
    config = diagnostic_camera_config(18_889)

    assert (overview.image_width, overview.image_height) == (320, 240)
    assert overview.D == (0.0, 0.0, 0.0, 0.0)
    assert overview.K[0][0] == overview.K[1][1] == 1000.0
    assert (stereo.image_width, stereo.image_height) == (320, 240)
    assert stereo.D_left == stereo.D_right == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert stereo.K_left[0][0] == stereo.K_left[1][1] == 250.0
    assert config.source.bind_address == "127.0.0.1"
    assert config.source.port == 18_889
    assert config.source.buffer_size == 1
    assert not config.processing_enabled


def test_synthetic_scene_renders_target_source_time_and_advancing_frame_counter() -> None:
    fixed = datetime(2026, 9, 22, 14, 58, 21, 372_000, tzinfo=UTC)
    generator = SyntheticDiagnosticFrameGenerator(
        camera=CameraRole.OVERVIEW,
        width=320,
        height=240,
        wall_clock=lambda: fixed,
    )

    first = generator.next_frame()
    second = generator.next_frame()

    assert first.shape == (240, 320, 3)
    assert first.dtype == np.uint8
    assert generator.frames_generated == 2
    assert tuple(first[82, 190]) == (245, 245, 245)
    assert tuple(second[82, 192]) == (245, 245, 245)
    assert np.max(first[204:, :]) >= 240
    assert not np.array_equal(first[204:, :], second[204:, :])


def test_detector_friendly_scene_survives_jpeg_and_keeps_stable_track() -> None:
    fixed = datetime(2026, 9, 22, 14, 58, 21, 372_000, tzinfo=UTC)
    cases = (
        (
            CameraRole.OVERVIEW,
            overview_corrector(diagnostic_overview_calibration()),
        ),
        (
            CameraRole.STEREO_LEFT,
            stereo_left_corrector(diagnostic_stereo_calibration()),
        ),
    )

    for camera, corrector in cases:
        generator = SyntheticDiagnosticFrameGenerator(
            camera=camera,
            width=320,
            height=240,
            wall_clock=lambda: fixed,
        )
        processor = Legacy14VisionProcessor()
        tracked_by_frame = []
        for frame_id in range(120):
            encoded, jpeg = cv2.imencode(
                ".jpg",
                generator.next_frame(),
                (cv2.IMWRITE_JPEG_QUALITY, 85),
            )
            assert encoded
            decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            result = processor.process(
                FramePacket(
                    camera=camera,
                    generation=1,
                    frame_id=frame_id,
                    capture_id=None,
                    receive_timestamp_ns=frame_id * 50_000_000,
                    image=corrector.correct(decoded),
                )
            )
            tracked_by_frame.append(result.tracked_objects)

        settled = tracked_by_frame[20:]
        assert all(len(tracked) == 1 for tracked in settled)
        assert len({tracked[0].track_id for tracked in settled}) == 1
        stable = [tracked[0] for tracked in settled[:20]]
        assert all(
            current.age_frames > previous.age_frames
            for previous, current in pairwise(stable)
        )
        assert stable[0].bbox != stable[-1].bbox


def test_sender_only_preflight_does_not_inspect_receiver_elements() -> None:
    inspected: list[str] = []

    def runner(command, **kwargs):
        del kwargs
        inspected.append(command[1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = check_gstreamer_sender_runtime(
        which=lambda executable: f"/usr/bin/{executable}",
        runner=runner,
    )

    assert result.ok
    assert inspected == list(GST_SENDER_ELEMENTS)
    receiver_only = set(GST_RECEIVER_ELEMENTS) - set(GST_SENDER_ELEMENTS)
    assert not receiver_only.intersection(inspected)
