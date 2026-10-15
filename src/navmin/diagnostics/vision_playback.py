"""Headless sequential video playback through production NavMin Vision processors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic_ns, sleep
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from navmin.calibration import (
    CalibrationError,
    load_overview_calibration,
    load_stereo_calibration,
)
from navmin.contracts import (
    CameraRole,
    ClearTargetSelection,
    FramePacket,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetLifecycleEvent,
    TargetSelectionRequest,
    TargetTrackingEnded,
)
from navmin.vision.pipeline import overview_corrector, stereo_left_corrector
from navmin.vision.processor import VisionProcessor, VisionProcessorOutput
from navmin.vision.processors import Legacy14VisionProcessor
from navmin.vision.processors.vt11 import Vt11VisionProcessor

_GENERATION = 1
_DEFAULT_LOG_ROOT = Path("logs/vision-playback")
_TOOL_NAME = "vision-playback-diagnostic"
_CAMERA_ROLE_BY_CLI = {
    "overview": CameraRole.OVERVIEW,
    "stereo-left": CameraRole.STEREO_LEFT,
    "stereo-right": CameraRole.STEREO_RIGHT,
}


class PlaybackDiagnosticError(RuntimeError):
    """Base error for the standalone vision playback diagnostic."""


class PlaybackConfigurationError(PlaybackDiagnosticError):
    """Playback options are contradictory or unsupported."""


class VideoProbeError(PlaybackDiagnosticError):
    """The input video cannot be validated for sequential playback."""


class PlaybackBackend(Enum):
    LEGACY14 = "legacy14"
    VT11 = "vt11"


class PlaybackTerminalState(Enum):
    PLAYBACK_FINISHED = "PLAYBACK_FINISHED"


@dataclass(frozen=True)
class PlaybackOptions:
    video_path: Path
    backend: PlaybackBackend
    camera_role: CameraRole = CameraRole.OVERVIEW
    profile: Literal["day", "thermal"] | None = None
    vt11_config_path: Path | None = None
    fps_override: float | None = None
    correction_config_path: Path | None = None


@dataclass(frozen=True)
class PlaybackProbe:
    source_size: tuple[int, int]
    metadata_fps: float | None
    effective_fps: float
    correction_enabled: bool


@dataclass(frozen=True)
class PlaybackRunResult:
    terminal_state: PlaybackTerminalState
    frames_processed: int
    last_output: VisionProcessorOutput | None
    run_dir: Path


@dataclass(frozen=True)
class PlaybackStepResult:
    """One strict sequential playback step for event-loop driven frontends."""

    output: VisionProcessorOutput | None
    terminal_state: PlaybackTerminalState | None
    frames_processed: int
    frame_started_ns: int | None


class _VideoCapture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


class _FrameCorrector(Protocol):
    def correct(self, image: np.ndarray) -> np.ndarray: ...


ProcessorFactory = Callable[[PlaybackOptions], VisionProcessor]
CaptureFactory = Callable[[str], _VideoCapture]
ClockNs = Callable[[], int]
SleepFn = Callable[[float], None]
WallClock = Callable[[], datetime]
OutputCallback = Callable[[VisionProcessorOutput], None]


def validate_playback_options(options: PlaybackOptions) -> None:
    if options.backend is PlaybackBackend.VT11:
        if options.profile not in {"day", "thermal"}:
            raise PlaybackConfigurationError(
                "--profile day|thermal is required for --backend vt11"
            )
    else:
        if options.profile is not None:
            raise PlaybackConfigurationError(
                "--profile is only valid with --backend vt11"
            )
        if options.vt11_config_path is not None:
            raise PlaybackConfigurationError(
                "--vt11-config is only valid with --backend vt11"
            )

    if options.fps_override is not None and (
        not math.isfinite(options.fps_override) or options.fps_override <= 0.0
    ):
        raise PlaybackConfigurationError("--fps must be a finite value greater than 0")

    if (
        options.correction_config_path is not None
        and options.camera_role is CameraRole.STEREO_RIGHT
    ):
        raise PlaybackConfigurationError(
            "the baseline production correction path has no Stereo Right corrector; "
            "use --camera-role stereo-right only with correction disabled"
        )


def resolve_playback_fps(metadata_fps: float, fps_override: float | None) -> float:
    if fps_override is not None:
        if not math.isfinite(fps_override) or fps_override <= 0.0:
            raise PlaybackConfigurationError(
                "--fps must be a finite value greater than 0"
            )
        return float(fps_override)
    if not math.isfinite(metadata_fps) or metadata_fps <= 0.0:
        raise VideoProbeError(
            "video metadata FPS is invalid; provide a positive finite --fps override"
        )
    return float(metadata_fps)


def create_playback_processor(options: PlaybackOptions) -> VisionProcessor:
    validate_playback_options(options)
    if options.backend is PlaybackBackend.LEGACY14:
        return Legacy14VisionProcessor()
    assert options.profile is not None
    return Vt11VisionProcessor(
        profile_name=options.profile,
        config_path=options.vt11_config_path,
    )


def _validate_decoded_frame(image: np.ndarray | None) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise VideoProbeError("decoded video frame is not a numpy.ndarray")
    if image.dtype != np.uint8:
        raise VideoProbeError("decoded video frame must use uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise VideoProbeError(
            f"decoded video frame must be BGR HxWx3, got shape={image.shape!r}"
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise VideoProbeError("decoded video frame has invalid resolution")
    return image


def _freeze_uncorrected_frame(image: np.ndarray) -> np.ndarray:
    working = np.array(_validate_decoded_frame(image), copy=True, order="C")
    working.flags.writeable = False
    return working


def _build_corrector(
    options: PlaybackOptions,
    *,
    source_size: tuple[int, int],
) -> _FrameCorrector | None:
    path = options.correction_config_path
    if path is None:
        return None
    if options.camera_role is CameraRole.OVERVIEW:
        calibration = load_overview_calibration(path, source_size=source_size)
        return overview_corrector(calibration)
    if options.camera_role is CameraRole.STEREO_LEFT:
        calibration = load_stereo_calibration(path, source_size=source_size)
        return stereo_left_corrector(calibration)
    raise PlaybackConfigurationError(
        "production correction is unavailable for the selected camera role"
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(
                _json_value(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _create_unique_run_dir(log_root: Path, started: datetime) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    candidate_time = _as_utc(started)
    while True:
        run_id = f"vision-playback-{candidate_time.strftime('%Y%m%d-%H%M%S-%f')}"
        run_dir = log_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            candidate_time += timedelta(microseconds=1)
            continue
        return run_dir


def _navmin_version() -> str | None:
    try:
        return version("navmin")
    except PackageNotFoundError:
        return None


def _lifecycle_event_identity(event: TargetLifecycleEvent) -> tuple[CameraRole, int]:
    if isinstance(event, TargetAcquired | TargetTrackingEnded):
        return event.target.camera, event.target.generation
    if isinstance(event, TargetAcquisitionFailed):
        return event.camera, event.generation
    raise TypeError("processor returned an unknown target lifecycle event")


def _lifecycle_event_mapping(event: TargetLifecycleEvent) -> dict[str, Any]:
    if isinstance(event, TargetAcquired):
        return {
            "type": "TargetAcquired",
            "selection_id": event.selection_id,
            "target": {
                "camera": event.target.camera.value,
                "generation": event.target.generation,
                "track_id": event.target.track_id,
            },
            "request_frame_id": event.request_frame_id,
            "decision_frame_id": event.decision_frame_id,
            "decision_frame_timestamp_ns": event.decision_frame_timestamp_ns,
        }
    if isinstance(event, TargetAcquisitionFailed):
        return {
            "type": "TargetAcquisitionFailed",
            "selection_id": event.selection_id,
            "camera": event.camera.value,
            "generation": event.generation,
            "request_frame_id": event.request_frame_id,
            "reason": event.reason.value,
            "decision_frame_id": event.decision_frame_id,
            "decision_frame_timestamp_ns": event.decision_frame_timestamp_ns,
        }
    if isinstance(event, TargetTrackingEnded):
        return {
            "type": "TargetTrackingEnded",
            "target": {
                "camera": event.target.camera.value,
                "generation": event.target.generation,
                "track_id": event.target.track_id,
            },
            "reason": event.reason.value,
            "decision_frame_id": event.decision_frame_id,
            "decision_frame_timestamp_ns": event.decision_frame_timestamp_ns,
        }
    raise TypeError("event must be a TargetLifecycleEvent")


class _PlaybackRunLogger:
    """Own run.json/events.jsonl for one validated standalone playback run."""

    def __init__(
        self,
        *,
        run_dir: Path,
        started: datetime,
        monotonic_clock_ns: ClockNs,
        wall_clock: WallClock,
    ) -> None:
        self.run_dir = run_dir
        self._monotonic_clock_ns = monotonic_clock_ns
        self._wall_clock = wall_clock
        self._events_path = run_dir / "events.jsonl"
        self._events_stream = self._events_path.open("x", encoding="utf-8")
        self._finished_logged = False
        self._closed = False
        self.started = _as_utc(started)

    @classmethod
    def create(
        cls,
        *,
        log_root: Path,
        metadata: Mapping[str, Any],
        started: datetime,
        monotonic_clock_ns: ClockNs,
        wall_clock: WallClock,
    ) -> _PlaybackRunLogger:
        run_dir = _create_unique_run_dir(log_root, started)
        try:
            _atomic_write_json(run_dir / "run.json", metadata)
            return cls(
                run_dir=run_dir,
                started=started,
                monotonic_clock_ns=monotonic_clock_ns,
                wall_clock=wall_clock,
            )
        except Exception:
            for child in run_dir.iterdir():
                child.unlink()
            run_dir.rmdir()
            raise

    def _write(self, payload: Mapping[str, Any]) -> None:
        if self._closed:
            raise PlaybackDiagnosticError("playback event log is already closed")
        event = {
            "timestamp_utc": _iso_utc(self._wall_clock()),
            "timestamp_monotonic_ns": self._monotonic_clock_ns(),
            **payload,
        }
        json.dump(_json_value(event), self._events_stream, ensure_ascii=False, sort_keys=True)
        self._events_stream.write("\n")
        self._events_stream.flush()

    def playback_started(self) -> None:
        self._write({"type": "PLAYBACK_STARTED"})

    def select(self, request: TargetSelectionRequest) -> None:
        self._write(
            {
                "type": "SELECT",
                "selection_id": request.selection_id,
                "camera": request.camera.value,
                "generation": request.generation,
                "frame_id": request.frame_id,
                "x_px": request.x_px,
                "y_px": request.y_px,
            }
        )

    def clear(self, command: ClearTargetSelection) -> None:
        self._write(
            {
                "type": "CLEAR",
                "camera": command.camera.value,
                "generation": command.generation,
            }
        )

    def lifecycle(self, event: TargetLifecycleEvent) -> None:
        self._write(_lifecycle_event_mapping(event))

    def playback_finished(self, *, frames_processed: int) -> None:
        if self._finished_logged:
            return
        self._write(
            {
                "type": PlaybackTerminalState.PLAYBACK_FINISHED.value,
                "frames_processed": frames_processed,
            }
        )
        self._finished_logged = True

    def playback_closed(self, *, frames_processed: int) -> None:
        if self._finished_logged:
            return
        self._write(
            {
                "type": "PLAYBACK_CLOSED",
                "frames_processed": frames_processed,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._events_stream.close()
        self._closed = True


class PlaybackPacer:
    """Pace each processed frame without catch-up skipping or frame discard."""

    def __init__(
        self,
        fps: float,
        *,
        clock_ns: ClockNs = monotonic_ns,
        sleep_fn: SleepFn = sleep,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and greater than 0")
        self._period_ns = 1_000_000_000.0 / fps
        self._clock_ns = clock_ns
        self._sleep_fn = sleep_fn

    def wait_after(self, frame_started_ns: int) -> None:
        remaining_ns = self.remaining_ns_after(frame_started_ns)
        if remaining_ns > 0:
            self._sleep_fn(remaining_ns / 1_000_000_000.0)

    def remaining_ns_after(self, frame_started_ns: int) -> float:
        elapsed_ns = self._clock_ns() - frame_started_ns
        return max(0.0, self._period_ns - max(0, elapsed_ns))

    def remaining_delay_ms_after(self, frame_started_ns: int) -> int:
        """Return a non-negative Qt-timer delay without catch-up skipping."""
        return math.ceil(self.remaining_ns_after(frame_started_ns) / 1_000_000.0)


class VisionPlaybackSession:
    """Single-owner, forward-only, no-drop playback of one video file."""

    def __init__(
        self,
        *,
        options: PlaybackOptions,
        capture: _VideoCapture,
        processor: VisionProcessor,
        corrector: _FrameCorrector | None,
        first_working_frame: np.ndarray,
        probe: PlaybackProbe,
        clock_ns: ClockNs,
        sleep_fn: SleepFn,
        wall_clock: WallClock,
        log_root: Path,
    ) -> None:
        self.options = options
        self.probe = probe
        self.processor = processor
        self._capture = capture
        self._corrector = corrector
        self._first_working_frame: np.ndarray | None = first_working_frame
        self._clock_ns = clock_ns
        self._wall_clock = wall_clock
        self._log_root = log_root
        self._pacer = PlaybackPacer(
            probe.effective_fps,
            clock_ns=clock_ns,
            sleep_fn=sleep_fn,
        )
        self._closed = False
        self._ran = False
        self._finished = False
        self._frames_processed = 0
        self._next_selection_id = 1
        self._run_logger: _PlaybackRunLogger | None = None
        self.last_output: VisionProcessorOutput | None = None

    @classmethod
    def open(
        cls,
        options: PlaybackOptions,
        *,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        processor_factory: ProcessorFactory = create_playback_processor,
        clock_ns: ClockNs = monotonic_ns,
        sleep_fn: SleepFn = sleep,
        wall_clock: WallClock = _utc_now,
        log_root: Path = _DEFAULT_LOG_ROOT,
    ) -> VisionPlaybackSession:
        validate_playback_options(options)
        video_path = options.video_path.expanduser()
        if not video_path.is_file():
            raise VideoProbeError(f"video file not found: {video_path}")

        capture = capture_factory(str(video_path))
        try:
            if not capture.isOpened():
                raise VideoProbeError(f"cannot open video: {video_path}")
            metadata_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
            effective_fps = resolve_playback_fps(
                metadata_fps_raw,
                options.fps_override,
            )
            ok, first_raw = capture.read()
            if not ok:
                raise VideoProbeError("video contains no readable first frame")
            first_raw = _validate_decoded_frame(first_raw)
            source_size = (int(first_raw.shape[1]), int(first_raw.shape[0]))
            corrector = _build_corrector(options, source_size=source_size)
            first_working = (
                _freeze_uncorrected_frame(first_raw)
                if corrector is None
                else corrector.correct(first_raw)
            )
            processor = processor_factory(options)
            probe = PlaybackProbe(
                source_size=source_size,
                metadata_fps=(
                    metadata_fps_raw
                    if math.isfinite(metadata_fps_raw) and metadata_fps_raw > 0.0
                    else None
                ),
                effective_fps=effective_fps,
                correction_enabled=corrector is not None,
            )
            return cls(
                options=options,
                capture=capture,
                processor=processor,
                corrector=corrector,
                first_working_frame=first_working,
                probe=probe,
                clock_ns=clock_ns,
                sleep_fn=sleep_fn,
                wall_clock=wall_clock,
                log_root=Path(log_root),
            )
        except Exception:
            capture.release()
            raise

    @property
    def generation(self) -> int:
        return _GENERATION

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def started(self) -> bool:
        return self._ran

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def run_dir(self) -> Path | None:
        return None if self._run_logger is None else self._run_logger.run_dir

    @property
    def current_frame(self) -> FramePacket | None:
        return None if self.last_output is None else self.last_output.result.frame

    def _run_metadata(self, started: datetime) -> dict[str, Any]:
        video_path = self.options.video_path.expanduser().resolve()
        correction_path = self.options.correction_config_path
        correction_mapping: dict[str, Any] = {
            "enabled": self.probe.correction_enabled,
            "config_path": None,
            "config_sha256": None,
            "side": None,
        }
        if correction_path is not None:
            resolved_correction = correction_path.expanduser().resolve()
            correction_mapping["config_path"] = str(resolved_correction)
            correction_mapping["config_sha256"] = _file_sha256(resolved_correction)
            if self.options.camera_role is CameraRole.STEREO_LEFT:
                correction_mapping["side"] = "left"

        vt11_mapping: dict[str, Any] | None = None
        if isinstance(self.processor, Vt11VisionProcessor):
            snapshot = self.processor.settings_snapshot
            vt11_mapping = {
                "profile": snapshot.profile,
                "config_identity": snapshot.config_identity,
                "source_config_sha256": snapshot.source_config_sha256,
                "checkpoint": snapshot.checkpoint,
                "effective_config": _json_value(snapshot.effective_config),
                "integration_settings": _json_value(snapshot.integration_settings),
            }

        return {
            "tool": {
                "name": _TOOL_NAME,
                "navmin_version": _navmin_version(),
            },
            "start_timestamp_utc": _iso_utc(started),
            "generation": self.generation,
            "video": {
                "path": str(video_path),
                "sha256": _file_sha256(video_path),
                "resolution": {
                    "width": self.probe.source_size[0],
                    "height": self.probe.source_size[1],
                },
                "metadata_fps": self.probe.metadata_fps,
                "effective_playback_fps": self.probe.effective_fps,
            },
            "camera_role": self.options.camera_role.value,
            "backend": self.options.backend.value,
            "vt11": vt11_mapping,
            "correction": correction_mapping,
        }

    def _start_run_logging(self) -> None:
        started = _as_utc(self._wall_clock())
        metadata = self._run_metadata(started)
        self._run_logger = _PlaybackRunLogger.create(
            log_root=self._log_root,
            metadata=metadata,
            started=started,
            monotonic_clock_ns=self._clock_ns,
            wall_clock=self._wall_clock,
        )
        self._run_logger.playback_started()

    def start(self) -> None:
        """Start one validated run without processing the first frame yet."""
        if self._ran:
            raise PlaybackDiagnosticError("a playback session can only be started once")
        if self._closed:
            raise PlaybackDiagnosticError("playback session is already closed")
        self._ran = True
        self._start_run_logging()
        self.processor.reset()

    def close(self, *, log_closed: bool = False) -> None:
        if self._closed:
            return
        self._capture.release()
        if self._run_logger is not None:
            if log_closed and not self._finished:
                self._run_logger.playback_closed(
                    frames_processed=self._frames_processed
                )
            self._run_logger.close()
        self._closed = True

    def _next_working_frame(self) -> np.ndarray | None:
        if self._first_working_frame is not None:
            frame = self._first_working_frame
            self._first_working_frame = None
            return frame
        ok, raw = self._capture.read()
        if not ok:
            return None
        raw = _validate_decoded_frame(raw)
        if self._corrector is None:
            return _freeze_uncorrected_frame(raw)
        return self._corrector.correct(raw)

    def _require_control_frame(self) -> FramePacket:
        if not self._ran or self._closed or self._finished:
            raise PlaybackDiagnosticError(
                "target control is only available during active playback"
            )
        frame = self.current_frame
        if frame is None:
            raise PlaybackDiagnosticError(
                "target control requires an already processed current frame"
            )
        return frame

    def _log_lifecycle_events(
        self,
        events: tuple[TargetLifecycleEvent, ...],
    ) -> None:
        if self._run_logger is None:
            raise PlaybackDiagnosticError("playback logging has not started")
        for event in events:
            event_camera, event_generation = _lifecycle_event_identity(event)
            if (
                event_camera is not self.options.camera_role
                or event_generation != self.generation
            ):
                raise PlaybackDiagnosticError(
                    "processor returned a target event for another playback session"
                )
            self._run_logger.lifecycle(event)

    def select_target(
        self,
        x_px: float,
        y_px: float,
    ) -> tuple[TargetLifecycleEvent, ...]:
        frame = self._require_control_frame()
        selection_id = self._next_selection_id
        self._next_selection_id += 1
        request = TargetSelectionRequest(
            selection_id=selection_id,
            camera=self.options.camera_role,
            generation=self.generation,
            frame_id=frame.frame_id,
            x_px=float(x_px),
            y_px=float(y_px),
        )
        assert self._run_logger is not None
        self._run_logger.select(request)
        events = self.processor.apply_target_control(request)
        self._log_lifecycle_events(events)
        return events

    def clear_target(self) -> tuple[TargetLifecycleEvent, ...]:
        self._require_control_frame()
        command = ClearTargetSelection(
            camera=self.options.camera_role,
            generation=self.generation,
        )
        assert self._run_logger is not None
        self._run_logger.clear(command)
        events = self.processor.apply_target_control(command)
        self._log_lifecycle_events(events)
        return events

    def remaining_delay_ms_after(self, frame_started_ns: int) -> int:
        return self._pacer.remaining_delay_ms_after(frame_started_ns)

    def process_next(self) -> PlaybackStepResult:
        """Process exactly one next decoded frame, or expose EOF once reached."""
        if not self._ran:
            raise PlaybackDiagnosticError("playback session has not been started")
        if self._closed:
            raise PlaybackDiagnosticError("playback session is already closed")
        if self._finished:
            return PlaybackStepResult(
                output=None,
                terminal_state=PlaybackTerminalState.PLAYBACK_FINISHED,
                frames_processed=self._frames_processed,
                frame_started_ns=None,
            )

        working = self._next_working_frame()
        if working is None:
            self._finished = True
            assert self._run_logger is not None
            self._run_logger.playback_finished(
                frames_processed=self._frames_processed
            )
            return PlaybackStepResult(
                output=None,
                terminal_state=PlaybackTerminalState.PLAYBACK_FINISHED,
                frames_processed=self._frames_processed,
                frame_started_ns=None,
            )

        frame_started_ns = self._clock_ns()
        packet = FramePacket(
            camera=self.options.camera_role,
            generation=self.generation,
            frame_id=self._frames_processed,
            capture_id=None,
            receive_timestamp_ns=frame_started_ns,
            image=working,
        )
        output = self.processor.process(packet)
        self.last_output = output
        self._frames_processed += 1
        self._log_lifecycle_events(output.target_events)
        return PlaybackStepResult(
            output=output,
            terminal_state=None,
            frames_processed=self._frames_processed,
            frame_started_ns=frame_started_ns,
        )

    def run(self, on_output: OutputCallback | None = None) -> PlaybackRunResult:
        if self._ran:
            raise PlaybackDiagnosticError("a playback session can only be run once")
        self.start()
        try:
            while True:
                step = self.process_next()
                if step.terminal_state is not None:
                    assert self._run_logger is not None
                    return PlaybackRunResult(
                        terminal_state=step.terminal_state,
                        frames_processed=self._frames_processed,
                        last_output=self.last_output,
                        run_dir=self._run_logger.run_dir,
                    )
                output = step.output
                assert output is not None
                if on_output is not None:
                    on_output(output)
                assert step.frame_started_ns is not None
                self._pacer.wait_after(step.frame_started_ns)
        finally:
            self.close()


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially play one video through a production NavMin Vision processor "
            "without frame dropping in a standalone interactive PyQt6 diagnostic."
        ),
        epilog=(
            "Examples:\n"
            "  python tools/run_vision_playback_diagnostic.py video.mp4 --backend legacy14\n"
            "  python tools/run_vision_playback_diagnostic.py video.mp4 "
            "--backend vt11 --profile day\n"
            "  python tools/run_vision_playback_diagnostic.py video.mp4 "
            "--backend vt11 --profile thermal --fps 25\n\n"
            "Mouse controls:\n"
            "  Left click inside the displayed image: SELECT on that processed frame\n"
            "  Right click in the video area: CLEAR\n\n"
            "Playback is forward-only: no pause, seek, rewind, frame-step, or loop.\n"
            "A validated run creates run.json and events.jsonl under "
            "logs/vision-playback/<unique-run>/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", metavar="VIDEO_PATH", type=Path)
    parser.add_argument(
        "--backend",
        choices=tuple(item.value for item in PlaybackBackend),
        required=True,
    )
    parser.add_argument("--profile", choices=("day", "thermal"))
    parser.add_argument("--vt11-config", type=Path)
    parser.add_argument("--fps", type=_positive_finite_float)
    parser.add_argument(
        "--camera-role",
        choices=tuple(_CAMERA_ROLE_BY_CLI),
        default="overview",
    )
    parser.add_argument(
        "--correction-config",
        type=Path,
        help=(
            "Enable production NavMin geometric correction with this calibration JSON. "
            "Correction is OFF when omitted."
        ),
    )
    return parser


def parse_playback_options(argv: Sequence[str] | None = None) -> PlaybackOptions:
    args = build_argument_parser().parse_args(argv)
    options = PlaybackOptions(
        video_path=args.video_path,
        backend=PlaybackBackend(args.backend),
        camera_role=_CAMERA_ROLE_BY_CLI[args.camera_role],
        profile=args.profile,
        vt11_config_path=args.vt11_config,
        fps_override=args.fps,
        correction_config_path=args.correction_config,
    )
    validate_playback_options(options)
    return options


def run_cli(argv: Sequence[str] | None = None) -> int:
    try:
        options = parse_playback_options(argv)
        session = VisionPlaybackSession.open(options)
    except (PlaybackDiagnosticError, CalibrationError, ValueError, OSError) as exc:
        print(f"VISION PLAYBACK: FAIL — {exc}", file=sys.stderr)
        return 1

    try:
        from navmin.diagnostics.vision_playback_ui import run_interactive_session
    except ImportError as exc:
        session.close()
        print(f"VISION PLAYBACK: FAIL — PyQt6 UI unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        return run_interactive_session(session)
    except (PlaybackDiagnosticError, RuntimeError, OSError) as exc:
        session.close()
        print(f"VISION PLAYBACK: FAIL — {exc}", file=sys.stderr)
        return 1


__all__ = [
    "PlaybackBackend",
    "PlaybackConfigurationError",
    "PlaybackDiagnosticError",
    "PlaybackOptions",
    "PlaybackPacer",
    "PlaybackProbe",
    "PlaybackRunResult",
    "PlaybackStepResult",
    "PlaybackTerminalState",
    "VideoProbeError",
    "VisionPlaybackSession",
    "build_argument_parser",
    "create_playback_processor",
    "parse_playback_options",
    "resolve_playback_fps",
    "run_cli",
    "validate_playback_options",
]
