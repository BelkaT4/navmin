from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from navmin.calibration import (
    CalibrationError,
    CalibrationResolutionMismatchError,
    load_overview_calibration,
)
from navmin.contracts import (
    CameraRole,
    ClearTargetSelection,
    FramePacket,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TargetTrackingEndReason,
    VisionResult,
)
from navmin.diagnostics.localhost_rtp import SyntheticDiagnosticFrameGenerator
from navmin.diagnostics.vision_playback import (
    PlaybackBackend,
    PlaybackConfigurationError,
    PlaybackDiagnosticError,
    PlaybackOptions,
    PlaybackPacer,
    PlaybackTerminalState,
    VideoProbeError,
    VisionPlaybackSession,
    build_argument_parser,
    create_playback_processor,
    parse_playback_options,
    resolve_playback_fps,
)
from navmin.vision.pipeline import overview_corrector
from navmin.vision.processor import VisionProcessorOutput
from navmin.vision.processors import Legacy14VisionProcessor
from navmin.vision.processors.vt11 import Vt11VisionProcessor
from navmin.vision.processors.vt11.config.loader import DEFAULT_CONFIG

_WIDTH = 64
_HEIGHT = 48
_FPS = 12.0


class _RecordingProcessor:
    def __init__(self) -> None:
        self.frames: list[FramePacket] = []
        self.reset_calls = 0
        self.control_calls = 0
        self.commands: list[object] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def apply_target_control(self, command) -> tuple:
        self.control_calls += 1
        self.commands.append(command)
        return ()

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        assert processing_enabled
        self.frames.append(frame)
        return VisionProcessorOutput(
            result=VisionResult(frame=frame, tracked_objects=(), processing_time_ns=0)
        )


class _LifecycleProcessor(_RecordingProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.active_target: TargetRef | None = None

    def apply_target_control(self, command) -> tuple:
        super().apply_target_control(command)
        if isinstance(command, ClearTargetSelection):
            self.active_target = None
            return ()
        assert isinstance(command, TargetSelectionRequest)
        if command.selection_id == 1:
            frame = self.frames[command.frame_id]
            target = TargetRef(
                camera=command.camera,
                generation=command.generation,
                track_id=17,
            )
            self.active_target = target
            return (
                TargetAcquired(
                    selection_id=command.selection_id,
                    target=target,
                    request_frame_id=command.frame_id,
                    decision_frame_id=command.frame_id,
                    decision_frame_timestamp_ns=frame.receive_timestamp_ns,
                ),
            )
        frame = self.frames[command.frame_id]
        return (
            TargetAcquisitionFailed(
                selection_id=command.selection_id,
                camera=command.camera,
                generation=command.generation,
                request_frame_id=command.frame_id,
                reason=TargetAcquisitionFailureReason.AMBIGUOUS,
                decision_frame_id=command.frame_id,
                decision_frame_timestamp_ns=frame.receive_timestamp_ns,
            ),
        )

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        output = super().process(frame, processing_enabled=processing_enabled)
        if frame.frame_id == 1 and self.active_target is not None:
            event = TargetTrackingEnded(
                target=self.active_target,
                reason=TargetTrackingEndReason.LOST_UNRECOVERED,
                decision_frame_id=frame.frame_id,
                decision_frame_timestamp_ns=frame.receive_timestamp_ns,
            )
            self.active_target = None
            return VisionProcessorOutput(result=output.result, target_events=(event,))
        return output


class _IncrementingClock:
    def __init__(self, step_ns: int = 1_000_000) -> None:
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.value += self.step_ns
        return self.value


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, fps: float) -> None:
        self._frames = [frame.copy() for frame in frames]
        self._fps = fps
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def get(self, prop_id: int) -> float:
        assert prop_id == cv2.CAP_PROP_FPS
        return self._fps

    def release(self) -> None:
        self.released = True


def _frame(index: int) -> np.ndarray:
    image = np.full((_HEIGHT, _WIDTH, 3), 80 + index, dtype=np.uint8)
    x = 8 + index * 3
    cv2.rectangle(image, (x, 15), (x + 10, 27), (220, 220, 220), -1)
    return image


def _write_video(path: Path, *, frame_count: int = 5, fps: float = _FPS) -> list[np.ndarray]:
    frames = [_frame(index) for index in range(frame_count)]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (_WIDTH, _HEIGHT),
    )
    assert writer.isOpened(), "test environment cannot create an MJPG AVI"
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return frames


def _write_detector_friendly_video(path: Path, *, frame_count: int = 20) -> None:
    generator = SyntheticDiagnosticFrameGenerator(
        camera=CameraRole.OVERVIEW,
        width=320,
        height=240,
    )
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (320, 240),
    )
    assert writer.isOpened(), "test environment cannot create an MJPG AVI"
    try:
        for _ in range(frame_count):
            writer.write(generator.next_frame())
    finally:
        writer.release()


def _overview_calibration_json(path: Path, *, width: int = _WIDTH) -> None:
    payload = {
        "schema_version": 1,
        "image_width": width,
        "image_height": _HEIGHT,
        "K": [
            [45.0, 0.0, (_WIDTH - 1) / 2.0],
            [0.0, 45.0, (_HEIGHT - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "D": [0.0, 0.0, 0.0, 0.0],
        "new_camera_matrix": [
            [45.0, 0.0, (_WIDTH - 1) / 2.0],
            [0.0, 45.0, (_HEIGHT - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stereo_calibration_json(path: Path) -> None:
    k = [
        [45.0, 0.0, (_WIDTH - 1) / 2.0],
        [0.0, 45.0, (_HEIGHT - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    p = [
        [45.0, 0.0, (_WIDTH - 1) / 2.0, 0.0],
        [0.0, 45.0, (_HEIGHT - 1) / 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    payload = {
        "schema_version": 1,
        "translation-unit": "m",
        "image_width": _WIDTH,
        "image_height": _HEIGHT,
        "K_left": k,
        "D_left": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K_right": k,
        "D_right": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R": identity,
        "T": [-0.46, 0.0, 0.0],
        "R1": identity,
        "R2": identity,
        "P1": p,
        "P2": p,
        "Q": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("argv", "backend", "profile", "camera"),
    [
        (
            ["video.avi", "--backend", "legacy14"],
            PlaybackBackend.LEGACY14,
            None,
            CameraRole.OVERVIEW,
        ),
        (
            ["video.avi", "--backend", "vt11", "--profile", "day"],
            PlaybackBackend.VT11,
            "day",
            CameraRole.OVERVIEW,
        ),
        (
            [
                "video.avi",
                "--backend",
                "vt11",
                "--profile",
                "thermal",
                "--camera-role",
                "stereo-left",
            ],
            PlaybackBackend.VT11,
            "thermal",
            CameraRole.STEREO_LEFT,
        ),
        (
            [
                "video.avi",
                "--backend",
                "legacy14",
                "--camera-role",
                "stereo-right",
            ],
            PlaybackBackend.LEGACY14,
            None,
            CameraRole.STEREO_RIGHT,
        ),
    ],
)
def test_cli_accepts_supported_backend_profile_and_camera_combinations(
    argv,
    backend,
    profile,
    camera,
) -> None:
    options = parse_playback_options(argv)

    assert options.backend is backend
    assert options.profile == profile
    assert options.camera_role is camera


@pytest.mark.parametrize(
    "argv",
    [
        ["video.avi", "--backend", "vt11"],
        ["video.avi", "--backend", "legacy14", "--profile", "day"],
        ["video.avi", "--backend", "legacy14", "--vt11-config", "custom.yaml"],
    ],
)
def test_cli_rejects_backend_specific_argument_mismatches(argv) -> None:
    with pytest.raises(PlaybackConfigurationError):
        parse_playback_options(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["video.avi", "--backend", "legacy14", "--camera-role", "left"],
        ["video.avi", "--backend", "legacy14", "--fps", "0"],
        ["video.avi", "--backend", "legacy14", "--fps", "-1"],
        ["video.avi", "--backend", "legacy14", "--fps", "nan"],
        ["video.avi", "--backend", "legacy14", "--fps", "inf"],
    ],
)
def test_cli_parser_rejects_invalid_role_and_fps(argv) -> None:
    with pytest.raises(SystemExit):
        parse_playback_options(argv)


def test_fps_policy_uses_metadata_or_override_and_rejects_invalid_metadata() -> None:
    assert resolve_playback_fps(29.97, None) == pytest.approx(29.97)
    assert resolve_playback_fps(29.97, 15.0) == pytest.approx(15.0)
    assert resolve_playback_fps(float("nan"), 12.5) == pytest.approx(12.5)

    with pytest.raises(VideoProbeError):
        resolve_playback_fps(float("nan"), None)
    with pytest.raises(VideoProbeError):
        resolve_playback_fps(0.0, None)


def test_help_describes_interactive_controls_and_log_directory() -> None:
    help_text = build_argument_parser().format_help()

    assert "Left click" in help_text
    assert "Right click" in help_text
    assert "logs/vision-playback/<unique-run>/" in help_text
    assert "pause, seek, rewind, frame-step, or loop" in help_text


def test_probe_accepts_invalid_metadata_when_explicit_fps_is_supplied(tmp_path) -> None:
    path = tmp_path / "fake.avi"
    path.touch()
    capture = _FakeCapture([_frame(0)], fps=float("nan"))
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=20.0,
        ),
        capture_factory=lambda _: capture,
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    assert session.probe.metadata_fps is None
    assert session.probe.effective_fps == 20.0
    session.close()
    assert capture.released


def test_probe_rejects_invalid_metadata_without_override_and_releases_capture(tmp_path) -> None:
    path = tmp_path / "fake.avi"
    path.touch()
    capture = _FakeCapture([_frame(0)], fps=float("nan"))

    with pytest.raises(VideoProbeError):
        VisionPlaybackSession.open(
            PlaybackOptions(video_path=path, backend=PlaybackBackend.LEGACY14),
            capture_factory=lambda _: capture,
            log_root=tmp_path / "logs",
        )

    assert capture.released


def test_sequential_playback_preserves_every_frame_id_generation_and_camera(tmp_path) -> None:
    path = tmp_path / "sequence.avi"
    _write_video(path, frame_count=6)
    processor = _RecordingProcessor()
    outputs: list[VisionProcessorOutput] = []
    clock = _IncrementingClock()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            camera_role=CameraRole.STEREO_RIGHT,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=clock,
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    result = session.run(outputs.append)

    assert result.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED
    assert result.frames_processed == 6
    assert [frame.frame_id for frame in processor.frames] == list(range(6))
    assert {frame.generation for frame in processor.frames} == {1}
    assert {frame.camera for frame in processor.frames} == {CameraRole.STEREO_RIGHT}
    assert all(frame.capture_id is None for frame in processor.frames)
    assert all(not frame.image.flags.writeable for frame in processor.frames)
    assert all(
        current.receive_timestamp_ns > previous.receive_timestamp_ns
        for previous, current in zip(processor.frames, processor.frames[1:], strict=False)
    )
    assert [output.result.frame for output in outputs] == processor.frames
    assert processor.reset_calls == 1
    assert processor.control_calls == 0
    assert all(output.target_events == () for output in outputs)
    assert session.closed


def test_eof_returns_terminal_state_once_without_synthetic_target_control(tmp_path) -> None:
    path = tmp_path / "short.avi"
    _write_video(path, frame_count=2)
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    result = session.run()

    assert result.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED
    assert result.frames_processed == 2
    assert result.last_output is not None
    assert result.last_output.target_events == ()
    assert processor.control_calls == 0
    with pytest.raises(PlaybackDiagnosticError, match="only be run once"):
        session.run()


def test_pacer_sleeps_only_for_remaining_frame_period() -> None:
    sleeps: list[float] = []
    pacer = PlaybackPacer(
        10.0,
        clock_ns=lambda: 1_025_000_000,
        sleep_fn=sleeps.append,
    )

    pacer.wait_after(1_000_000_000)

    assert sleeps == [pytest.approx(0.075)]


def test_correction_is_off_by_default(tmp_path) -> None:
    path = tmp_path / "raw.avi"
    _write_video(path, frame_count=1)
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    assert not session.probe.correction_enabled
    session.run()
    assert len(processor.frames) == 1


def test_overview_correction_uses_production_corrector(tmp_path) -> None:
    path = tmp_path / "overview.avi"
    calibration_path = tmp_path / "overview.json"
    _write_video(path, frame_count=1)
    _overview_calibration_json(calibration_path)

    raw_capture = cv2.VideoCapture(str(path))
    try:
        ok, raw = raw_capture.read()
        assert ok
    finally:
        raw_capture.release()
    calibration = load_overview_calibration(
        calibration_path,
        source_size=(_WIDTH, _HEIGHT),
    )
    expected = overview_corrector(calibration).correct(raw)

    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            correction_config_path=calibration_path,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    assert session.probe.correction_enabled
    result = session.run()
    assert np.array_equal(processor.frames[0].image, expected)
    metadata = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["correction"]["enabled"] is True
    assert metadata["correction"]["config_path"] == str(calibration_path.resolve())
    assert metadata["correction"]["config_sha256"] == hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()


def test_stereo_left_correction_uses_production_path(tmp_path) -> None:
    path = tmp_path / "left.avi"
    calibration_path = tmp_path / "stereo.json"
    _write_video(path, frame_count=1)
    _stereo_calibration_json(calibration_path)
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            camera_role=CameraRole.STEREO_LEFT,
            correction_config_path=calibration_path,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    assert session.probe.correction_enabled
    session.run()
    assert processor.frames[0].camera is CameraRole.STEREO_LEFT
    assert not processor.frames[0].image.flags.writeable


def test_bad_or_mismatched_correction_fails_before_playback(tmp_path) -> None:
    path = tmp_path / "video.avi"
    _write_video(path, frame_count=1)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    with pytest.raises(CalibrationError):
        VisionPlaybackSession.open(
            PlaybackOptions(
                video_path=path,
                backend=PlaybackBackend.LEGACY14,
                correction_config_path=bad,
            ),
            log_root=tmp_path / "logs",
        )

    mismatch = tmp_path / "mismatch.json"
    _overview_calibration_json(mismatch, width=_WIDTH + 1)
    with pytest.raises(CalibrationResolutionMismatchError):
        VisionPlaybackSession.open(
            PlaybackOptions(
                video_path=path,
                backend=PlaybackBackend.LEGACY14,
                correction_config_path=mismatch,
            ),
            log_root=tmp_path / "logs",
        )


def test_stereo_right_rejects_correction_until_production_right_path_exists(tmp_path) -> None:
    with pytest.raises(PlaybackConfigurationError, match="Stereo Right"):
        parse_playback_options(
            [
                str(tmp_path / "video.avi"),
                "--backend",
                "legacy14",
                "--camera-role",
                "stereo-right",
                "--correction-config",
                str(tmp_path / "stereo.json"),
            ]
        )


def test_playback_factory_uses_real_legacy14_and_vt11_processors(tmp_path) -> None:
    legacy = create_playback_processor(
        PlaybackOptions(
            video_path=tmp_path / "video.avi",
            backend=PlaybackBackend.LEGACY14,
        )
    )
    assert isinstance(legacy, Legacy14VisionProcessor)

    custom_config = tmp_path / "vt11.yaml"
    custom_config.write_bytes(DEFAULT_CONFIG.read_bytes())
    vt11 = create_playback_processor(
        PlaybackOptions(
            video_path=tmp_path / "video.avi",
            backend=PlaybackBackend.VT11,
            profile="day",
            vt11_config_path=custom_config,
        )
    )
    assert isinstance(vt11, Vt11VisionProcessor)
    assert vt11.settings_snapshot.profile == "day"
    assert vt11.settings_snapshot.config_identity == str(custom_config.resolve())


def test_generated_video_runs_through_real_legacy14_processor(tmp_path) -> None:
    path = tmp_path / "legacy.avi"
    _write_video(path, frame_count=4)
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=1000.0,
        ),
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    assert isinstance(session.processor, Legacy14VisionProcessor)
    result = session.run()

    assert result.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED
    assert result.frames_processed == 4
    assert result.last_output is not None
    assert result.last_output.result.frame.frame_id == 3


@pytest.mark.parametrize("profile", ["day", "thermal"])
def test_generated_video_smoke_runs_through_real_vt11_processor(tmp_path, profile) -> None:
    path = tmp_path / f"vt11-{profile}.avi"
    _write_video(path, frame_count=2)
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.VT11,
            profile=profile,
            fps_override=1000.0,
        ),
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    assert isinstance(session.processor, Vt11VisionProcessor)
    result = session.run()

    assert result.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED
    assert result.frames_processed == 2
    assert result.last_output is not None
    assert result.last_output.result.frame.frame_id == 1


def test_controls_use_exact_current_processed_frame_and_monotonic_selection_ids(tmp_path) -> None:
    path = tmp_path / "controls.avi"
    _write_video(path, frame_count=3)
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            camera_role=CameraRole.STEREO_LEFT,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    def on_output(output: VisionProcessorOutput) -> None:
        if output.result.frame.frame_id != 0:
            return
        session.select_target(10.5, 11.5)
        session.select_target(12.5, 13.5)
        session.clear_target()

    session.run(on_output)

    first, second, clear = processor.commands
    assert isinstance(first, TargetSelectionRequest)
    assert isinstance(second, TargetSelectionRequest)
    assert isinstance(clear, ClearTargetSelection)
    assert [first.selection_id, second.selection_id] == [1, 2]
    assert [first.frame_id, second.frame_id] == [0, 0]
    assert first.camera is second.camera is clear.camera is CameraRole.STEREO_LEFT
    assert first.generation == second.generation == clear.generation == session.generation
    assert (first.x_px, first.y_px) == (10.5, 11.5)
    assert (second.x_px, second.y_px) == (12.5, 13.5)


def test_controls_require_an_active_processed_frame(tmp_path) -> None:
    path = tmp_path / "control-boundary.avi"
    _write_video(path, frame_count=1)
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: _RecordingProcessor(),
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    with pytest.raises(PlaybackDiagnosticError, match="active playback"):
        session.select_target(1.0, 1.0)

    session.run()

    with pytest.raises(PlaybackDiagnosticError, match="active playback"):
        session.clear_target()


def test_run_json_and_events_jsonl_capture_controls_lifecycle_and_terminal_state(tmp_path) -> None:
    path = tmp_path / "logged.avi"
    _write_video(path, frame_count=3)
    processor = _LifecycleProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            camera_role=CameraRole.OVERVIEW,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    def on_output(output: VisionProcessorOutput) -> None:
        if output.result.frame.frame_id == 0:
            session.select_target(20.0, 21.0)
        elif output.result.frame.frame_id == 1:
            session.select_target(22.0, 23.0)
            session.clear_target()

    result = session.run(on_output)

    metadata = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["tool"]["name"] == "vision-playback-diagnostic"
    assert metadata["video"]["path"] == str(path.resolve())
    assert metadata["video"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["video"]["resolution"] == {"width": _WIDTH, "height": _HEIGHT}
    assert metadata["video"]["effective_playback_fps"] == 1000.0
    assert metadata["camera_role"] == CameraRole.OVERVIEW.value
    assert metadata["backend"] == "legacy14"
    assert metadata["generation"] == 1
    assert metadata["vt11"] is None
    assert metadata["correction"] == {
        "enabled": False,
        "config_path": None,
        "config_sha256": None,
        "side": None,
    }

    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events] == [
        "PLAYBACK_STARTED",
        "SELECT",
        "TargetAcquired",
        "TargetTrackingEnded",
        "SELECT",
        "TargetAcquisitionFailed",
        "CLEAR",
        "PLAYBACK_FINISHED",
    ]
    selects = [event for event in events if event["type"] == "SELECT"]
    assert [event["selection_id"] for event in selects] == [1, 2]
    assert [event["frame_id"] for event in selects] == [0, 1]
    assert selects[0]["x_px"] == 20.0
    assert selects[1]["y_px"] == 23.0
    failed = next(event for event in events if event["type"] == "TargetAcquisitionFailed")
    assert failed["reason"] == TargetAcquisitionFailureReason.AMBIGUOUS.value
    assert sum(event["type"] == "PLAYBACK_FINISHED" for event in events) == 1


def test_eof_logging_does_not_synthesize_clear_or_target_loss(tmp_path) -> None:
    path = tmp_path / "eof-events.avi"
    _write_video(path, frame_count=2)
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=1000.0,
        ),
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    result = session.run()
    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [event["type"] for event in events] == [
        "PLAYBACK_STARTED",
        "PLAYBACK_FINISHED",
    ]
    assert not any(event["type"] == "CLEAR" for event in events)
    assert not any(event["type"] == "TargetTrackingEnded" for event in events)


def test_failed_initial_validation_creates_no_false_successful_run(tmp_path) -> None:
    log_root = tmp_path / "logs"

    with pytest.raises(VideoProbeError):
        VisionPlaybackSession.open(
            PlaybackOptions(
                video_path=tmp_path / "missing.avi",
                backend=PlaybackBackend.LEGACY14,
            ),
            log_root=log_root,
        )

    assert not log_root.exists()


def test_vt11_run_json_records_profile_and_settings_snapshot(tmp_path) -> None:
    path = tmp_path / "vt11-metadata.avi"
    _write_video(path, frame_count=1)
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.VT11,
            profile="thermal",
            fps_override=1000.0,
        ),
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    result = session.run()
    metadata = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))

    assert metadata["backend"] == "vt11"
    assert metadata["vt11"]["profile"] == "thermal"
    assert metadata["vt11"]["config_identity"] == "bundled:default.yaml"
    assert len(metadata["vt11"]["source_config_sha256"]) == 64
    assert isinstance(metadata["vt11"]["effective_config"], dict)
    assert isinstance(metadata["vt11"]["integration_settings"], dict)


def test_controls_pass_through_real_legacy14_processor_interface(tmp_path) -> None:
    path = tmp_path / "legacy-controls.avi"
    _write_detector_friendly_video(path, frame_count=20)
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
            fps_override=1000.0,
        ),
        clock_ns=_IncrementingClock(step_ns=1_000_000),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )
    acquired: list[TargetAcquired] = []

    def on_output(output: VisionProcessorOutput) -> None:
        if acquired or not output.result.tracked_objects:
            return
        tracked = output.result.tracked_objects[0]
        events = session.select_target(
            tracked.bbox.x + tracked.bbox.width / 2.0,
            tracked.bbox.y + tracked.bbox.height / 2.0,
        )
        assert len(events) == 1
        assert isinstance(events[0], TargetAcquired)
        acquired.append(events[0])

    result = session.run(on_output)

    assert len(acquired) == 1
    assert acquired[0].request_frame_id == acquired[0].decision_frame_id
    assert acquired[0].target.camera is CameraRole.OVERVIEW
    assert acquired[0].target.generation == session.generation
    logged = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "TargetAcquired" for event in logged)


def test_runs_with_same_wall_clock_create_distinct_directories(tmp_path) -> None:
    path = tmp_path / "unique.avi"
    _write_video(path, frame_count=1)
    fixed = datetime(2026, 10, 6, 3, 15, 0, tzinfo=UTC)
    run_dirs: list[Path] = []

    for _ in range(2):
        session = VisionPlaybackSession.open(
            PlaybackOptions(
                video_path=path,
                backend=PlaybackBackend.LEGACY14,
                fps_override=1000.0,
            ),
            processor_factory=lambda _: _RecordingProcessor(),
            clock_ns=_IncrementingClock(),
            sleep_fn=lambda _: None,
            wall_clock=lambda: fixed,
            log_root=tmp_path / "logs",
        )
        run_dirs.append(session.run().run_dir)

    assert run_dirs[0] != run_dirs[1]
    assert all(path.is_dir() for path in run_dirs)


def test_incremental_process_next_preserves_sequence_and_finishes_once(tmp_path) -> None:
    path = tmp_path / "incremental.avi"
    path.touch()
    capture = _FakeCapture([_frame(0), _frame(1), _frame(2)], fps=25.0)
    processor = _RecordingProcessor()
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
        ),
        capture_factory=lambda _: capture,
        processor_factory=lambda _: processor,
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    session.start()
    steps = [session.process_next() for _ in range(3)]
    terminal = session.process_next()
    terminal_again = session.process_next()

    assert [step.output.result.frame.frame_id for step in steps if step.output] == [
        0,
        1,
        2,
    ]
    assert terminal.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED
    assert terminal_again.terminal_state is PlaybackTerminalState.PLAYBACK_FINISHED
    assert session.finished
    assert session.frames_processed == 3
    assert not session.closed

    assert session.run_dir is not None
    events = [
        json.loads(line)
        for line in (session.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["type"] == "PLAYBACK_FINISHED" for event in events) == 1

    session.close()
    assert capture.released


def test_manual_close_before_eof_logs_playback_closed_without_target_loss(tmp_path) -> None:
    path = tmp_path / "closed.avi"
    path.touch()
    capture = _FakeCapture([_frame(0), _frame(1)], fps=25.0)
    session = VisionPlaybackSession.open(
        PlaybackOptions(
            video_path=path,
            backend=PlaybackBackend.LEGACY14,
        ),
        capture_factory=lambda _: capture,
        processor_factory=lambda _: _RecordingProcessor(),
        clock_ns=_IncrementingClock(),
        sleep_fn=lambda _: None,
        log_root=tmp_path / "logs",
    )

    session.start()
    session.process_next()
    run_dir = session.run_dir
    assert run_dir is not None
    session.close(log_closed=True)

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events] == [
        "PLAYBACK_STARTED",
        "PLAYBACK_CLOSED",
    ]
    assert not any(event["type"] == "TargetTrackingEnded" for event in events)
