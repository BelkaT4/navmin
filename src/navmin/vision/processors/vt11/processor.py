"""Composite NavMin VisionProcessor shell for the VT11 vision backend."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from statistics import median
from time import monotonic_ns, perf_counter_ns
from types import MappingProxyType
from typing import Literal

from navmin.contracts import (
    BBox,
    CameraRole,
    ClearTargetSelection,
    FramePacket,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetCandidateProposal,
    TargetCandidateSelectionRequest,
    TargetCandidateSet,
    TargetLifecycleEvent,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TargetTrackingEndReason,
    TrackedObject,
    VisionResult,
    VisionTargetControlCommand,
)
from navmin.vision.candidate_selection import auto_select_candidate_at_point
from navmin.vision.processor import VisionProcessorOutput, VisionTrackingDiagnostics
from navmin.vision.tracking import PublicTrackIdAllocator

from ..legacy_14 import Legacy14VisionProcessor
from .acquisition.acquire import (
    Vt11AcquireCandidates,
    Vt11AcquisitionAttempt,
    Vt11AcquisitionResult,
    discover_acquire_candidates,
)
from .acquisition.candidate_detector import Candidate
from .config.loader import DEFAULT_CONFIG, load_config
from .frame_history import FramePacketHistory
from .profiles import effective_config
from .runtime_config import Vt11ResolvedConfig, Vt11RuntimeSettings
from .tracking.reacquisition import ReacqFrameResult
from .tracking.session import Vt11TargetSession
from .tracking.tracker import TrackFrameResult

_HISTORY_MAX_FRAMES = 12
_HISTORY_MAX_AGE_NS = 400_000_000
_TIMING_HISTORY_MAX_FRAMES = 12
_BACKEND_NAME = "vt11"


class Vt11ProcessorState(Enum):
    """Only the integration states owned by the composite adapter."""

    DISCOVERY = "discovery"
    CANDIDATES = "candidates"
    ACQUIRING = "acquiring"
    TARGET = "target"


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class Vt11ProcessorSettingsSnapshot:
    """Read-only resolved identity and effective settings for future evidence."""

    backend: str
    profile: str
    checkpoint: str
    config_identity: str
    source_config_sha256: str
    effective_config: Mapping[str, object]
    integration_settings: Mapping[str, object]


@dataclass
class _CandidateSelectionContext:
    request: TargetSelectionRequest
    latest_candidates: TargetCandidateSet | None = None


@dataclass
class _AcquisitionContext:
    request: TargetSelectionRequest
    attempt: Vt11AcquisitionAttempt
    receive_timestamps_ns: dict[int, int]


@dataclass(frozen=True)
class _ProcessedFrameTiming:
    frame_id: int
    receive_timestamp_ns: int


class Vt11VisionProcessor:
    """Compose Legacy14 discovery with the incremental VT11 core."""

    def __init__(
        self,
        *,
        profile_name: Literal["day", "thermal"],
        config_path: Path | None = None,
        runtime_settings: Vt11RuntimeSettings | None = None,
        duration_clock_ns: Callable[[], int] = perf_counter_ns,
        history_clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        profile = str(profile_name).lower()
        if profile not in {"day", "thermal"}:
            raise ValueError("profile_name must be 'day' or 'thermal'")

        source_path = DEFAULT_CONFIG if config_path is None else Path(config_path)
        raw_config = load_config(source_path)
        effective = effective_config(raw_config, profile)
        settings = runtime_settings or Vt11RuntimeSettings()
        settings.validate()

        self._runtime_settings = settings
        self._resolved_config = Vt11ResolvedConfig.from_effective_config(
            profile,
            effective,
        )
        self._duration_clock_ns = duration_clock_ns
        self._history_clock_ns = history_clock_ns
        self._public_track_ids = PublicTrackIdAllocator()
        self._legacy = Legacy14VisionProcessor(
            public_track_id_allocator=self._public_track_ids
        )
        self._frame_history = FramePacketHistory(
            max_frames=_HISTORY_MAX_FRAMES,
            max_age_ns=_HISTORY_MAX_AGE_NS,
        )
        self._state = Vt11ProcessorState.DISCOVERY
        self._camera: CameraRole | None = None
        self._generation: int | None = None
        self._candidate_selection: _CandidateSelectionContext | None = None
        self._acquisition: _AcquisitionContext | None = None
        self._target_session: Vt11TargetSession | None = None
        self._current_target: TargetRef | None = None
        self._current_public_target_id: int | None = None
        self._initial_acquisition_decision_frame_id: int | None = None
        self._timing_history: deque[_ProcessedFrameTiming] = deque(
            maxlen=_TIMING_HISTORY_MAX_FRAMES
        )

        config_identity = (
            "bundled:default.yaml"
            if config_path is None
            else str(source_path.resolve())
        )
        source_hash = sha256(source_path.read_bytes()).hexdigest()
        checkpoint = str(raw_config["project"]["checkpoint"])
        frozen_effective = _freeze(effective)
        frozen_integration = _freeze(
            {
                "history_max_frames": _HISTORY_MAX_FRAMES,
                "history_max_age_ms": _HISTORY_MAX_AGE_NS // 1_000_000,
                "timing_history_max_frames": _TIMING_HISTORY_MAX_FRAMES,
                "max_retrospective_replay_frames": (
                    settings.max_retrospective_replay_frames
                ),
            }
        )
        if not isinstance(frozen_effective, Mapping):
            raise TypeError("effective VT11 config must be a mapping")
        if not isinstance(frozen_integration, Mapping):
            raise TypeError("VT11 integration settings must be a mapping")
        self._settings_snapshot = Vt11ProcessorSettingsSnapshot(
            backend=_BACKEND_NAME,
            profile=profile,
            checkpoint=checkpoint,
            config_identity=config_identity,
            source_config_sha256=source_hash,
            effective_config=frozen_effective,
            integration_settings=frozen_integration,
        )

    @property
    def state(self) -> Vt11ProcessorState:
        return self._state

    @property
    def settings_snapshot(self) -> Vt11ProcessorSettingsSnapshot:
        return self._settings_snapshot

    def reset(self) -> None:
        """Clear every generation-local component while preserving configuration."""
        if self._target_session is not None:
            self._target_session.close()
        self._legacy.reset()
        self._public_track_ids.reset()
        self._frame_history.clear()
        self._timing_history.clear()
        self._state = Vt11ProcessorState.DISCOVERY
        self._camera = None
        self._generation = None
        self._candidate_selection = None
        self._acquisition = None
        self._target_session = None
        self._current_target = None
        self._current_public_target_id = None
        self._initial_acquisition_decision_frame_id = None

    def _ensure_generation(self, frame: FramePacket) -> None:
        if self._camera is None and self._generation is None:
            self._camera = frame.camera
            self._generation = frame.generation
            return
        if self._camera is frame.camera and self._generation == frame.generation:
            return
        self.reset()
        self._camera = frame.camera
        self._generation = frame.generation

    def _selection_sequence(
        self,
        command: TargetSelectionRequest | TargetCandidateSelectionRequest,
    ) -> tuple[FramePacket, ...]:
        return self._frame_history.sequence_from_exact(
            camera=command.camera,
            generation=command.generation,
            frame_id=command.frame_id,
            now_ns=self._history_clock_ns(),
        )

    def _discard_active_episode(self, *, reset_legacy: bool) -> None:
        if self._target_session is not None:
            self._target_session.close()
        self._candidate_selection = None
        self._acquisition = None
        self._target_session = None
        self._current_target = None
        self._current_public_target_id = None
        self._initial_acquisition_decision_frame_id = None
        self._state = Vt11ProcessorState.DISCOVERY
        if reset_legacy:
            self._legacy.reset()

    @staticmethod
    def _public_candidate_set(
        request: TargetSelectionRequest,
        frame: FramePacket,
        scoring,
    ) -> TargetCandidateSet:
        proposals = tuple(
            TargetCandidateProposal(
                candidate_id=int(item.candidate.candidate_id),
                bbox=Vt11VisionProcessor._public_bbox(item.candidate),
                score=float(item.final_score),
                rank=int(item.final_rank),
            )
            for item in scoring.candidates
        )
        return TargetCandidateSet(
            selection_id=request.selection_id,
            camera=request.camera,
            generation=request.generation,
            frame_id=frame.frame_id,
            candidates=proposals,
        )

    def _discover_candidates_for_frame(
        self,
        request: TargetSelectionRequest,
        frame: FramePacket,
    ) -> tuple[TargetCandidateSet, Vt11AcquireCandidates]:
        discovered = discover_acquire_candidates(
            frame_index=frame.frame_id,
            frame_bgr=frame.image,
            click_xy=(request.x_px, request.y_px),
            frame_width=frame.image.shape[1],
            frame_height=frame.image.shape[0],
            resolved_config=self._resolved_config,
        )
        return self._public_candidate_set(request, frame, discovered.scoring), discovered

    def _start_acquisition_from_candidate(
        self,
        *,
        request: TargetSelectionRequest,
        candidate_id: int,
        sequence: Sequence[FramePacket],
        train_click_xy: tuple[float, float] | None = None,
    ) -> tuple[TargetLifecycleEvent, ...]:
        if not sequence:
            return (
                TargetAcquisitionFailed(
                    selection_id=request.selection_id,
                    camera=request.camera,
                    generation=request.generation,
                    request_frame_id=request.frame_id,
                    reason=TargetAcquisitionFailureReason.STALE_SELECTION,
                    decision_frame_id=None,
                    decision_frame_timestamp_ns=None,
                ),
            )
        selection_frame = sequence[0]
        attempt = Vt11AcquisitionAttempt(
            frame_index=selection_frame.frame_id,
            frame_bgr=selection_frame.image,
            click_xy=(request.x_px, request.y_px),
            frame_width=selection_frame.image.shape[1],
            frame_height=selection_frame.image.shape[0],
            resolved_config=self._resolved_config,
            runtime_settings=self._runtime_settings,
            candidate_id=candidate_id,
            train_click_xy=train_click_xy,
        )
        self._candidate_selection = None
        self._acquisition = _AcquisitionContext(
            request=request,
            attempt=attempt,
            receive_timestamps_ns={
                selection_frame.frame_id: selection_frame.receive_timestamp_ns
            },
        )
        self._state = Vt11ProcessorState.ACQUIRING
        return self._advance_acquisition(sequence[1:])

    def _candidate_selection_failure(
        self,
        reason: TargetAcquisitionFailureReason,
        *,
        decision_identity: tuple[int, int] | None,
    ) -> tuple[TargetLifecycleEvent, ...]:
        context = self._candidate_selection
        if context is None:
            raise RuntimeError("VT11 candidate-selection context is missing")
        request = context.request
        event = TargetAcquisitionFailed(
            selection_id=request.selection_id,
            camera=request.camera,
            generation=request.generation,
            request_frame_id=request.frame_id,
            reason=reason,
            decision_frame_id=(
                None if decision_identity is None else decision_identity[0]
            ),
            decision_frame_timestamp_ns=(
                None if decision_identity is None else decision_identity[1]
            ),
        )
        self._discard_active_episode(reset_legacy=True)
        return (event,)

    def _retry_candidate_selection(
        self,
        result: Vt11AcquisitionResult | None = None,
    ) -> tuple[TargetLifecycleEvent, ...]:
        context = self._acquisition
        if context is None:
            raise RuntimeError("VT11 acquisition context is missing")
        if result is not None and result.target_session is not None:
            result.target_session.close()
        request = context.request
        self._acquisition = None
        self._target_session = None
        self._current_target = None
        self._current_public_target_id = None
        self._initial_acquisition_decision_frame_id = None
        self._candidate_selection = _CandidateSelectionContext(request=request)
        self._state = Vt11ProcessorState.CANDIDATES
        return ()

    def _reject_or_retry_acquisition(
        self,
        reason: TargetAcquisitionFailureReason,
        *,
        result: Vt11AcquisitionResult | None,
        decision_identity: tuple[int, int] | None,
    ) -> tuple[TargetLifecycleEvent, ...]:
        context = self._acquisition
        if context is None:
            raise RuntimeError("VT11 acquisition context is missing")
        if context.request.candidate_selection_enabled:
            return self._retry_candidate_selection(result)
        return self._acquisition_failure(
            reason,
            decision_identity=decision_identity,
        )

    def _acquisition_failure(
        self,
        reason: TargetAcquisitionFailureReason,
        *,
        decision_identity: tuple[int, int] | None,
    ) -> tuple[TargetLifecycleEvent, ...]:
        context = self._acquisition
        if context is None:
            raise RuntimeError("VT11 acquisition context is missing")
        request = context.request
        event = TargetAcquisitionFailed(
            selection_id=request.selection_id,
            camera=request.camera,
            generation=request.generation,
            request_frame_id=request.frame_id,
            reason=reason,
            decision_frame_id=(
                None if decision_identity is None else decision_identity[0]
            ),
            decision_frame_timestamp_ns=(
                None
                if decision_identity is None
                else decision_identity[1]
            ),
        )
        self._discard_active_episode(reset_legacy=True)
        return (event,)

    def _acquisition_decision_identity(
        self,
        result: Vt11AcquisitionResult,
    ) -> tuple[int, int]:
        context = self._acquisition
        if context is None:
            raise RuntimeError("VT11 acquisition context is missing")
        train = result.train_result
        frame_id = (
            context.attempt.start_frame
            if train is None
            else train.frames[-1].frame_index
        )
        try:
            timestamp_ns = context.receive_timestamps_ns[frame_id]
        except KeyError as exc:
            raise RuntimeError(
                f"missing receive timestamp for VT11 decision frame {frame_id}"
            ) from exc
        return frame_id, timestamp_ns

    def _finish_acquisition(
        self,
        result: Vt11AcquisitionResult,
        remaining_frames: Sequence[FramePacket],
    ) -> tuple[TargetLifecycleEvent, ...]:
        context = self._acquisition
        if context is None:
            raise RuntimeError("VT11 acquisition context is missing")
        decision_identity = self._acquisition_decision_identity(result)

        if result.scoring.best is None:
            return self._reject_or_retry_acquisition(
                TargetAcquisitionFailureReason.NO_TARGET_AT_POINT,
                result=result,
                decision_identity=decision_identity,
            )
        if result.decision == "AMBIGUOUS":
            return self._reject_or_retry_acquisition(
                TargetAcquisitionFailureReason.AMBIGUOUS,
                result=result,
                decision_identity=decision_identity,
            )
        if result.decision == "OBSERVE_MORE":
            return self._reject_or_retry_acquisition(
                TargetAcquisitionFailureReason.INSUFFICIENT_EVIDENCE,
                result=result,
                decision_identity=decision_identity,
            )
        if result.decision != "CONFIRMED" or result.target_session is None:
            return self._reject_or_retry_acquisition(
                TargetAcquisitionFailureReason.ACQUISITION_FAILED,
                result=result,
                decision_identity=decision_identity,
            )

        replay_count = result.retrospective_replay_frames + len(remaining_frames)
        if replay_count > self._runtime_settings.max_retrospective_replay_frames:
            failure_identity = (
                decision_identity
                if not remaining_frames
                else (
                    remaining_frames[-1].frame_id,
                    remaining_frames[-1].receive_timestamp_ns,
                )
            )
            if context.request.candidate_selection_enabled:
                return self._retry_candidate_selection(result)
            result.target_session.close()
            return self._acquisition_failure(
                TargetAcquisitionFailureReason.ACQUISITION_FAILED,
                decision_identity=failure_identity,
            )

        session = result.target_session
        for frame in remaining_frames:
            context.receive_timestamps_ns[frame.frame_id] = (
                frame.receive_timestamp_ns
            )
            if frame.frame_id != session.next_frame_index:
                if context.request.candidate_selection_enabled:
                    return self._retry_candidate_selection(result)
                session.close()
                return self._acquisition_failure(
                    TargetAcquisitionFailureReason.ACQUISITION_FAILED,
                    decision_identity=(
                        frame.frame_id,
                        frame.receive_timestamp_ns,
                    ),
                )
            session.step(frame.frame_id, frame.image)
            if session.terminal_status is not None:
                if context.request.candidate_selection_enabled:
                    return self._retry_candidate_selection(result)
                session.close()
                return self._acquisition_failure(
                    TargetAcquisitionFailureReason.ACQUISITION_FAILED,
                    decision_identity=(
                        frame.frame_id,
                        frame.receive_timestamp_ns,
                    ),
                )

        public_track_id = self._public_track_ids.allocate()
        request = context.request
        target = TargetRef(
            camera=request.camera,
            generation=request.generation,
            track_id=public_track_id,
        )
        self._target_session = session
        self._current_target = target
        self._current_public_target_id = public_track_id
        self._initial_acquisition_decision_frame_id = decision_identity[0]
        self._acquisition = None
        self._state = Vt11ProcessorState.TARGET
        return (
            TargetAcquired(
                selection_id=request.selection_id,
                target=target,
                request_frame_id=request.frame_id,
                decision_frame_id=decision_identity[0],
                decision_frame_timestamp_ns=decision_identity[1],
            ),
        )

    def _advance_acquisition(
        self,
        frames: Sequence[FramePacket],
    ) -> tuple[TargetLifecycleEvent, ...]:
        context = self._acquisition
        if context is None:
            raise RuntimeError("VT11 acquisition context is missing")
        attempt = context.attempt
        if attempt.done:
            if attempt.result is None:
                raise RuntimeError("completed VT11 acquisition has no result")
            return self._finish_acquisition(attempt.result, frames)

        for index, frame in enumerate(frames):
            context.receive_timestamps_ns[frame.frame_id] = (
                frame.receive_timestamp_ns
            )
            if frame.frame_id != attempt.next_frame_index:
                return self._acquisition_failure(
                    TargetAcquisitionFailureReason.ACQUISITION_FAILED,
                    decision_identity=(
                        frame.frame_id,
                        frame.receive_timestamp_ns,
                    ),
                )
            result = attempt.step(frame.frame_id, frame.image)
            if result is not None:
                return self._finish_acquisition(result, frames[index + 1 :])
        return ()

    def _append_timing(self, frame: FramePacket) -> None:
        self._timing_history.append(
            _ProcessedFrameTiming(
                frame_id=frame.frame_id,
                receive_timestamp_ns=frame.receive_timestamp_ns,
            )
        )

    def _frames_per_second(self) -> float:
        seconds_per_frame: list[float] = []
        for previous, current in pairwise(self._timing_history):
            frame_delta = current.frame_id - previous.frame_id
            timestamp_delta = (
                current.receive_timestamp_ns - previous.receive_timestamp_ns
            )
            if frame_delta <= 0 or timestamp_delta <= 0:
                continue
            seconds_per_frame.append(
                timestamp_delta / 1_000_000_000.0 / frame_delta
            )
        if not seconds_per_frame:
            return 0.0
        interval_s = float(median(seconds_per_frame))
        return 0.0 if interval_s <= 0.0 else 1.0 / interval_s

    @staticmethod
    def _public_bbox(candidate: Candidate) -> BBox:
        bbox = candidate.bbox
        return BBox(
            x=int(bbox.x),
            y=int(bbox.y),
            width=int(bbox.width),
            height=int(bbox.height),
        )

    def _public_target_object(
        self,
        *,
        candidate: Candidate,
        frame_id: int,
        velocity_xy_px_per_frame: tuple[float, float],
    ) -> TrackedObject:
        target = self._current_target
        decision_frame_id = self._initial_acquisition_decision_frame_id
        if target is None or decision_frame_id is None:
            raise RuntimeError("confirmed VT11 target identity is missing")
        frames_per_second = self._frames_per_second()
        return TrackedObject(
            track_id=target.track_id,
            bbox=self._public_bbox(candidate),
            velocity_x_px_s=(
                float(velocity_xy_px_per_frame[0]) * frames_per_second
            ),
            velocity_y_px_s=(
                float(velocity_xy_px_per_frame[1]) * frames_per_second
            ),
            age_frames=frame_id - decision_frame_id + 1,
        )

    def _finish_target_loss(
        self,
        frame: FramePacket,
    ) -> tuple[TargetLifecycleEvent, ...]:
        target = self._current_target
        if target is None:
            raise RuntimeError("terminal VT11 target has no public identity")
        event = TargetTrackingEnded(
            target=target,
            reason=TargetTrackingEndReason.LOST_UNRECOVERED,
            decision_frame_id=frame.frame_id,
            decision_frame_timestamp_ns=frame.receive_timestamp_ns,
        )
        self._discard_active_episode(reset_legacy=True)
        return (event,)

    @staticmethod
    def _tracking_diagnostics(
        frame: FramePacket,
        result: TrackFrameResult | ReacqFrameResult,
        *,
        phase: str,
    ) -> VisionTrackingDiagnostics:
        best_bbox = result.best_candidate_bbox
        return VisionTrackingDiagnostics(
            camera=frame.camera,
            generation=frame.generation,
            frame_id=frame.frame_id,
            phase=phase,
            status=result.status,
            predicted_xy=(
                float(result.predicted_xy[0]),
                float(result.predicted_xy[1]),
            ),
            search_radius_px=int(result.search_radius_px),
            full_frame_search=(
                isinstance(result, ReacqFrameResult)
                and result.reacq_mode == "GLOBAL_EMERGENCY_SCAN"
            ),
            search_mode=(
                result.tracking_phase
                if isinstance(result, TrackFrameResult)
                else result.reacq_mode
            ),
            candidate_count=int(result.candidate_count),
            best_candidate_center=(
                None
                if result.best_candidate_center is None
                else (
                    float(result.best_candidate_center[0]),
                    float(result.best_candidate_center[1]),
                )
            ),
            best_candidate_bbox=(
                None
                if best_bbox is None
                else BBox(
                    x=int(best_bbox[0]),
                    y=int(best_bbox[1]),
                    width=int(best_bbox[2]),
                    height=int(best_bbox[3]),
                )
            ),
            confidence=float(result.confidence),
            prediction_error_px=(
                None
                if result.prediction_error_px is None
                else float(result.prediction_error_px)
            ),
            diagnostic_reason=result.diagnostic_reason,
        )

    def _process_target_frame(
        self,
        frame: FramePacket,
    ) -> tuple[
        tuple[TrackedObject, ...],
        tuple[TargetLifecycleEvent, ...],
        VisionTrackingDiagnostics | None,
    ]:
        session = self._target_session
        if session is None:
            raise RuntimeError("TARGET state has no VT11 target session")
        step = session.step(frame.frame_id, frame.image)
        tracked_objects: tuple[TrackedObject, ...] = ()
        tracking_diagnostics: VisionTrackingDiagnostics

        if step.phase == "TRACK":
            result = step.result
            if not isinstance(result, TrackFrameResult):
                raise TypeError("TRACK phase emitted a non-TRACK result")
            tracking_diagnostics = self._tracking_diagnostics(
                frame, result, phase="TRACK"
            )
            if result.status == "TRACK" and result.selected is not None:
                tracked_objects = (
                    self._public_target_object(
                        candidate=result.selected,
                        frame_id=frame.frame_id,
                        velocity_xy_px_per_frame=result.velocity_xy,
                    ),
                )
        elif step.phase == "REACQ":
            result = step.result
            if not isinstance(result, ReacqFrameResult):
                raise TypeError("REACQ phase emitted a non-REACQ result")
            tracking_diagnostics = self._tracking_diagnostics(
                frame, result, phase="REACQ"
            )
            if result.status == "CONFIRMED" and result.selected is not None:
                tracked_objects = (
                    self._public_target_object(
                        candidate=result.selected,
                        frame_id=frame.frame_id,
                        velocity_xy_px_per_frame=(0.0, 0.0),
                    ),
                )
        else:
            raise RuntimeError(f"unknown VT11 target phase: {step.phase}")

        if step.terminal_status in {"LOST_UNRECOVERED", "REACQ_LIMIT"}:
            return (), self._finish_target_loss(frame), None
        if step.terminal_status is not None:
            raise RuntimeError(
                f"unexpected live VT11 terminal status: {step.terminal_status}"
            )
        return tracked_objects, (), tracking_diagnostics

    def apply_target_control(
        self,
        command: VisionTargetControlCommand,
    ) -> tuple[TargetLifecycleEvent, ...]:
        if isinstance(command, ClearTargetSelection):
            if self._state is Vt11ProcessorState.DISCOVERY:
                return ()
            self._discard_active_episode(reset_legacy=True)
            return ()

        if isinstance(command, TargetCandidateSelectionRequest):
            context = self._candidate_selection
            if (
                self._state is not Vt11ProcessorState.CANDIDATES
                or context is None
                or context.request.selection_id != command.selection_id
                or context.request.camera is not command.camera
                or context.request.generation != command.generation
            ):
                return ()
            latest = context.latest_candidates
            if (
                latest is None
                or latest.frame_id != command.frame_id
                or not any(
                    item.candidate_id == command.candidate_id
                    for item in latest.candidates
                )
            ):
                return ()
            sequence = self._selection_sequence(command)
            if not sequence:
                return self._candidate_selection_failure(
                    TargetAcquisitionFailureReason.STALE_SELECTION,
                    decision_identity=None,
                )
            return self._start_acquisition_from_candidate(
                request=context.request,
                candidate_id=command.candidate_id,
                sequence=sequence,
                train_click_xy=(command.x_px, command.y_px),
            )

        if not isinstance(command, TargetSelectionRequest):
            raise TypeError("unsupported target-control command")
        if (
            isinstance(command.candidate_click_tolerance_px, bool)
            or not isinstance(command.candidate_click_tolerance_px, int)
            or not 0 <= command.candidate_click_tolerance_px <= 50
        ):
            raise ValueError("candidate_click_tolerance_px must be in [0, 50]")

        was_active = self._state is not Vt11ProcessorState.DISCOVERY
        sequence = self._selection_sequence(command)
        if was_active:
            self._discard_active_episode(reset_legacy=True)
        if not sequence:
            return (
                TargetAcquisitionFailed(
                    selection_id=command.selection_id,
                    camera=command.camera,
                    generation=command.generation,
                    request_frame_id=command.frame_id,
                    reason=TargetAcquisitionFailureReason.STALE_SELECTION,
                    decision_frame_id=None,
                    decision_frame_timestamp_ns=None,
                ),
            )

        if not was_active:
            self._legacy.reset()
        selection_frame = sequence[0]
        if not command.candidate_selection_enabled:
            attempt = Vt11AcquisitionAttempt(
                frame_index=selection_frame.frame_id,
                frame_bgr=selection_frame.image,
                click_xy=(command.x_px, command.y_px),
                frame_width=selection_frame.image.shape[1],
                frame_height=selection_frame.image.shape[0],
                resolved_config=self._resolved_config,
                runtime_settings=self._runtime_settings,
            )
            self._acquisition = _AcquisitionContext(
                request=command,
                attempt=attempt,
                receive_timestamps_ns={
                    selection_frame.frame_id: selection_frame.receive_timestamp_ns
                },
            )
            self._state = Vt11ProcessorState.ACQUIRING
            return self._advance_acquisition(sequence[1:])

        candidate_set, _discovered = self._discover_candidates_for_frame(
            command,
            selection_frame,
        )
        if not candidate_set.candidates:
            self._state = Vt11ProcessorState.DISCOVERY
            return (
                TargetAcquisitionFailed(
                    selection_id=command.selection_id,
                    camera=command.camera,
                    generation=command.generation,
                    request_frame_id=command.frame_id,
                    reason=TargetAcquisitionFailureReason.NO_TARGET_AT_POINT,
                    decision_frame_id=selection_frame.frame_id,
                    decision_frame_timestamp_ns=selection_frame.receive_timestamp_ns,
                ),
            )

        if command.auto_select_candidate_on_first_click:
            selected = auto_select_candidate_at_point(
                candidate_set.candidates,
                command.x_px,
                command.y_px,
                tolerance_px=command.candidate_click_tolerance_px,
            )
            if selected is not None:
                return self._start_acquisition_from_candidate(
                    request=command,
                    candidate_id=selected.candidate_id,
                    sequence=sequence,
                )

        self._candidate_selection = _CandidateSelectionContext(request=command)
        self._state = Vt11ProcessorState.CANDIDATES
        return ()

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        started_ns = self._duration_clock_ns()
        self._ensure_generation(frame)
        self._frame_history.append(
            frame,
            inserted_timestamp_ns=self._history_clock_ns(),
        )
        self._append_timing(frame)

        if self._state is Vt11ProcessorState.DISCOVERY and processing_enabled:
            return self._legacy.process(frame, processing_enabled=True)

        target_events: tuple[TargetLifecycleEvent, ...] = ()
        tracked_objects: tuple[TrackedObject, ...] = ()
        tracking_diagnostics: VisionTrackingDiagnostics | None = None
        acquisition_candidates: TargetCandidateSet | None = None
        if self._state is Vt11ProcessorState.CANDIDATES:
            context = self._candidate_selection
            if context is None:
                raise RuntimeError("CANDIDATES state has no selection context")
            acquisition_candidates, _discovered = self._discover_candidates_for_frame(
                context.request,
                frame,
            )
            context.latest_candidates = acquisition_candidates
        elif self._state is Vt11ProcessorState.ACQUIRING:
            target_events = self._advance_acquisition((frame,))
        elif self._state is Vt11ProcessorState.TARGET:
            (
                tracked_objects,
                target_events,
                tracking_diagnostics,
            ) = self._process_target_frame(frame)

        elapsed_ns = max(0, self._duration_clock_ns() - started_ns)
        return VisionProcessorOutput(
            result=VisionResult(
                frame=frame,
                tracked_objects=tracked_objects,
                processing_time_ns=elapsed_ns,
            ),
            target_events=target_events,
            tracking_diagnostics=tracking_diagnostics,
            acquisition_candidates=acquisition_candidates,
        )


__all__ = [
    "Vt11ProcessorSettingsSnapshot",
    "Vt11ProcessorState",
    "Vt11VisionProcessor",
]
