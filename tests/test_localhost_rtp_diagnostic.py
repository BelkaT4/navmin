from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from navmin.diagnostics.localhost_rtp import (
    GST_RECEIVER_ELEMENTS,
    GST_SENDER_ELEMENTS,
    LocalhostRtpJpegSender,
    RtpJpegSenderConfig,
    SenderProcessError,
    build_sender_command,
    check_gstreamer_runtime,
    diagnostic_camera_config,
    diagnostic_overview_calibration,
    diagnostic_stereo_calibration,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"port": 0}, "port"),
        ({"port": 65_536}, "port"),
        ({"port": 9999, "width": 0}, "width"),
        ({"port": 9999, "height": -1}, "height"),
        ({"port": 9999, "fps": 0}, "fps"),
        ({"port": 9999, "host": "0.0.0.0"}, "127.0.0.1"),
        ({"port": 9999, "pattern": "moving ball"}, "pattern"),
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
        "videotestsrc",
        "is-live=true",
        "pattern=ball",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=I420,width=320,height=240,framerate=20/1",
        "!",
        "jpegenc",
        "!",
        "jpegparse",
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


def test_preflight_checks_executables_bindings_and_each_required_element() -> None:
    inspected: list[str] = []

    def runner(command, **kwargs):
        del kwargs
        inspected.append(command[1])
        return SimpleNamespace(
            returncode=1 if command[1] == "jpegenc" else 0,
            stdout="",
            stderr="missing jpegenc",
        )

    result = check_gstreamer_runtime(
        which=lambda executable: f"/usr/bin/{executable}",
        runner=runner,
        python_binding_check=lambda: None,
    )

    assert not result.ok
    assert inspected == [*GST_RECEIVER_ELEMENTS, *GST_SENDER_ELEMENTS]
    assert [failure.name for failure in result.failures] == [
        "GStreamer element jpegenc"
    ]
    assert result.failures[0].detail == "missing jpegenc"


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
    assert factory.kwargs["stdin"] is subprocess.DEVNULL
    assert factory.kwargs["stdout"] is subprocess.DEVNULL

    sender.stop()

    assert not sender.is_running
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_calls == [0.25]


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
    assert (stereo.image_width, stereo.image_height) == (320, 240)
    assert stereo.D_left == stereo.D_right == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert config.address == "127.0.0.1"
    assert config.port == 18_889
    assert config.rtp_enabled
    assert config.buffer_size == 1
    assert not config.processing_enabled
