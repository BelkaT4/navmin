"""VT-9 end-to-end TRACK -> LOST -> REACQ -> TRACK with optional EMA."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Mapping, Sequence

import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import CandidateDetectorConfig
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig, EMAReferenceAdapter, EMAUpdateEvent
from navmin.vision.processors.vt11.identity.reference_bank import ReferenceBank
from navmin.vision.processors.vt11.tracking.motion_model import MotionPoint
from navmin.vision.processors.vt11.tracking.live_events import (
    LiveFrameEvent,
    LiveProgressCallback,
    StopRequested,
)
from navmin.vision.processors.vt11.tracking.reacquisition import (
    ReacqFrameResult,
    ReacquisitionConfig,
    ReacquisitionContinuitySnapshot,
    ReacquisitionResult,
    ReacquisitionRuntime,
)
from navmin.vision.processors.vt11.tracking.tracker import (
    FrameReader,
    TrackFrameResult,
    TrackingConfig,
    TrackingContinuitySnapshot,
    TrackingResult,
    TrackingRuntime,
)
from navmin.vision.processors.vt11.tracking.scale_model import ScaleStabilityConfig, ScaleStabilityModel
from navmin.vision.processors.vt11.tracking.scale_continuity import ScaleContinuityConfig, ReliableScaleAnchor
from navmin.vision.processors.vt11.thermal.processing import ThermalProfile


@dataclass(frozen=True)
class TrackingSessionConfig:
    max_reacq_cycles: int = 5

    def validate(self) -> None:
        if self.max_reacq_cycles < 1:
            raise ValueError("max_reacq_cycles must be >= 1")


@dataclass(frozen=True)
class TrackingSessionResult:
    status: str
    track_segments: tuple[TrackingResult, ...]
    reacquisitions: tuple[ReacquisitionResult, ...]
    lost_events: int
    reacq_successes: int
    t_reacq_frames: tuple[int, ...]
    mean_t_reacq_frames: float
    max_t_reacq_frames: int
    trajectory: tuple[tuple[int, float, float], ...]
    final_frame: int | None
    adaptation_mode: str = "off"
    adaptation_updates: int = 0
    adaptation_events: tuple[EMAUpdateEvent, ...] = ()
    adaptation_slow_updates: int = 0
    adaptation_fast_updates: int = 0
    adaptation_recent_updates: int = 0
    adaptation_recent_count: int = 0
    mean_track_confidence: float = 0.0
    track_accepted_frames: int = 0
    final_reference_bank: ReferenceBank | None = None
    final_expected_area_px: float = 0.0
    motion_break_events: int = 0
    robust_reacq_count: int = 0
    auto_promoted_reacq_count: int = 0
    over_budget_frames: int = 0

    @property
    def all_track_frames(self):
        return tuple(frame for segment in self.track_segments for frame in segment.frames)

    @property
    def all_reacq_frames(self):
        return tuple(frame for result in self.reacquisitions for frame in result.frames)


@dataclass(frozen=True)
class SessionComparison:
    fixed: TrackingSessionResult
    ema: TrackingSessionResult

    @property
    def accepted_delta(self) -> int:
        return self.ema.track_accepted_frames - self.fixed.track_accepted_frames

    @property
    def lost_delta(self) -> int:
        return self.ema.lost_events - self.fixed.lost_events

    @property
    def mean_confidence_delta(self) -> float:
        return self.ema.mean_track_confidence - self.fixed.mean_track_confidence


def _append_motion_history(history: list[MotionPoint], points: Sequence[MotionPoint], *, max_points: int = 40) -> None:
    for p in points:
        if history and p.frame_index <= history[-1].frame_index:
            continue
        history.append(p)
    if len(history) > max_points:
        del history[:-max_points]


@dataclass(frozen=True)
class TargetSessionStep:
    phase: str
    result: TrackFrameResult | ReacqFrameResult
    phase_after: str
    segment_ended: bool
    terminal_status: str | None = None


class Vt11TargetSession:
    """Bounded live TRACK/REACQ coordinator using the shared runtimes."""

    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        start_frame: int,
        reference_bank: ReferenceBank,
        initial_history: Sequence[tuple[float, float]],
        initial_history_frames: Sequence[int],
        detector_config: CandidateDetectorConfig,
        tracking_config: TrackingConfig,
        reacquisition_config: ReacquisitionConfig,
        identity_weights: Mapping[str, float] | None = None,
        adaptation_config: EMAConfig | None = None,
        thermal_profile: ThermalProfile | None = None,
        frame_count: int | None = None,
        max_reacq_cycles: int | None = None,
        _collect_compatibility_history: bool = False,
    ) -> None:
        tracking_config.validate()
        reacquisition_config.validate()
        adapt_cfg = adaptation_config or EMAConfig(mode="off")
        adapt_cfg.validate()
        if frame_width < 1 or frame_height < 1:
            raise ValueError("frame dimensions must be >= 1")
        if len(initial_history) != len(initial_history_frames):
            raise ValueError("initial_history and initial_history_frames must have equal length")
        if not initial_history:
            raise ValueError("initial_history must not be empty")
        if max_reacq_cycles is not None and max_reacq_cycles < 1:
            raise ValueError("max_reacq_cycles must be >= 1")

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.frame_count = None if frame_count is None else int(frame_count)
        self.detector_config = detector_config
        self.tracking_config = tracking_config
        self.reacquisition_config = reacquisition_config
        self.identity_weights = identity_weights
        self.thermal_profile = thermal_profile
        self.max_reacq_cycles = max_reacq_cycles
        self._collect_compatibility_history = bool(_collect_compatibility_history)

        self.reference_bank = reference_bank.clone(include_adaptive=False)
        self.adapter = (
            EMAReferenceAdapter(
                self.reference_bank, adapt_cfg, identity_weights=identity_weights
            )
            if adapt_cfg.mode != "off"
            else None
        )
        self.adaptation_config = adapt_cfg
        self.motion_history = [
            MotionPoint(int(fi), float(xy[0]), float(xy[1]))
            for fi, xy in zip(initial_history_frames, initial_history)
        ]
        if any(
            later.frame_index <= earlier.frame_index
            for earlier, later in zip(self.motion_history, self.motion_history[1:])
        ):
            raise ValueError("initial history frame indices must be strictly increasing")
        if len(self.motion_history) > 40:
            self.motion_history = self.motion_history[-40:]

        self.expected_area_px = float(self.reference_bank.expected_size_px())
        self.scale_model: ScaleStabilityModel | None = None
        self.reliable_scale_anchor: ReliableScaleAnchor | None = None
        self._initialize_scale_continuity()

        self.phase = "TRACK"
        self.terminal_status: str | None = None
        self.next_frame_index = int(start_frame)
        self.final_frame: int | None = self.motion_history[-1].frame_index
        self.lost_events = 0
        self.reacq_attempts = 0
        self.reacq_successes = 0
        self.reacq_frames_total = 0
        self.reacq_frames_max = 0
        self.motion_break_events = 0
        self.auto_promoted_reacq_count = 0
        self.robust_reacq_count = 0
        self.over_budget_frames = 0
        self.track_accepted_frames = 0
        self.track_confidence_sum = 0.0
        self.track_attempted_frames = 0
        self._tracking_runtime: TrackingRuntime | None = None
        self._reacquisition_runtime: ReacquisitionRuntime | None = None
        self._last_tracking_snapshot: TrackingContinuitySnapshot | None = None
        self._last_reacquisition_snapshot: ReacquisitionContinuitySnapshot | None = None
        self._start_tracking_runtime(int(start_frame))

    def _initialize_scale_continuity(self) -> None:
        cfg = self.tracking_config
        thermal = self.thermal_profile is not None and self.thermal_profile.is_thermal
        if cfg.thermal_scale_parameter_distance_enabled and thermal:
            self.reliable_scale_anchor = ReliableScaleAnchor(
                self.expected_area_px,
                self.motion_history[-1].frame_index,
                ScaleContinuityConfig(
                    enabled=True,
                    base_allowed_log_distance=cfg.scale_anchor_base_allowed_log_distance,
                    growth_per_frame=cfg.scale_anchor_growth_per_frame,
                    max_allowed_log_distance=cfg.scale_anchor_max_allowed_log_distance,
                    nonlinear_strength=cfg.scale_nonlinear_gate_strength,
                    quarantine_confirm_hits=cfg.scale_quarantine_confirm_hits,
                    quarantine_extreme_confirm_hits=cfg.scale_quarantine_extreme_confirm_hits,
                    quarantine_extreme_log_distance=cfg.scale_quarantine_extreme_log_distance,
                    quarantine_max_confirm_excess_log=cfg.scale_quarantine_max_confirm_excess_log,
                    quarantine_link_log_span=cfg.scale_quarantine_link_log_span,
                    reliable_update_alpha=cfg.scale_reliable_update_alpha,
                ),
            )
        if cfg.thermal_scale_stability_enabled and thermal:
            self.scale_model = ScaleStabilityModel(
                self.expected_area_px,
                ScaleStabilityConfig(
                    enabled=True,
                    confirm_frames=cfg.scale_confirm_frames,
                    consistency_log_span=cfg.scale_consistency_log_span,
                    max_log_change_per_frame=cfg.scale_max_log_change_per_frame,
                    large_change_ratio=cfg.scale_large_change_ratio,
                    large_change_confirm_frames=cfg.scale_large_change_confirm_frames,
                    micro_max_area_px=cfg.scale_micro_max_area_px,
                    micro_confirm_frames=cfg.scale_micro_confirm_frames,
                    micro_exit_confirm_frames=cfg.scale_micro_exit_confirm_frames,
                    micro_consistency_log_span=cfg.scale_micro_consistency_log_span,
                    micro_max_log_change_per_frame=cfg.scale_micro_max_log_change_per_frame,
                    robust_history_frames=cfg.scale_robust_history_frames,
                ),
            )

    def _start_tracking_runtime(self, start_frame: int) -> None:
        self.phase = "TRACK"
        self.next_frame_index = int(start_frame)
        self._tracking_runtime = TrackingRuntime(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame_count=self.frame_count,
            start_frame=start_frame,
            reference_bank=self.reference_bank,
            initial_history=[p.xy for p in self.motion_history],
            initial_history_frames=[p.frame_index for p in self.motion_history],
            detector_config=self.detector_config,
            config=self.tracking_config,
            identity_weights=self.identity_weights,
            adaptation_adapter=self.adapter,
            initial_expected_area_px=self.expected_area_px,
            thermal_profile=self.thermal_profile,
            scale_model=self.scale_model,
            reliable_scale_anchor=self.reliable_scale_anchor,
            _adaptation_event_history_limit=(
                None
                if self._collect_compatibility_history
                else TrackingRuntime.DEFAULT_ADAPTATION_EVENT_HISTORY
            ),
        )
        self._reacquisition_runtime = None

    def _start_reacquisition_runtime(
        self, lost_frame: int, snapshot: TrackingContinuitySnapshot
    ) -> None:
        if snapshot.previous_frame is None:
            raise RuntimeError("TRACK continuity snapshot has no previous frame")
        self.phase = "REACQ"
        self.reacq_attempts += 1
        self._reacquisition_runtime = ReacquisitionRuntime(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame_count=self.frame_count,
            lost_frame=lost_frame,
            previous_frame=snapshot.previous_frame,
            quality_reference_frame=snapshot.quality_reference_frame,
            reference_bank=self.reference_bank,
            motion_history=self.motion_history,
            detector_config=self.detector_config,
            config=self.reacquisition_config,
            identity_weights=self.identity_weights,
            initial_expected_area_px=self.expected_area_px,
            thermal_profile=self.thermal_profile,
            motion_broken=snapshot.motion_broken_at_lost,
            scale_model=self.scale_model,
            reliable_scale_anchor=self.reliable_scale_anchor,
        )
        self._tracking_runtime = None
        self.next_frame_index = lost_frame + 1

    def step(self, frame_index: int, frame: np.ndarray) -> TargetSessionStep:
        if self.terminal_status is not None:
            raise RuntimeError("target session is terminal")
        if int(frame_index) != self.next_frame_index:
            raise ValueError(
                f"expected frame {self.next_frame_index}, got {int(frame_index)}"
            )
        phase = self.phase
        if phase == "TRACK":
            runtime = self._tracking_runtime
            if runtime is None:
                raise RuntimeError("TRACK runtime is not active")
            item = runtime.step(frame)
            self.next_frame_index = runtime.next_frame_index
            self.final_frame = item.frame_index
            self.track_attempted_frames += 1
            self.track_confidence_sum += float(item.confidence)
            self.over_budget_frames += int(item.over_budget)
            if item.selected is not None:
                _append_motion_history(
                    self.motion_history,
                    [MotionPoint(item.frame_index, item.selected.center_x, item.selected.center_y)],
                )
                if item.status == "TRACK":
                    self.track_accepted_frames += 1
            if item.status == "LOST":
                snapshot = runtime.continuity_snapshot
                self._last_tracking_snapshot = snapshot
                self.expected_area_px = float(snapshot.expected_area_px)
                self.scale_model = snapshot.scale_model
                self.reliable_scale_anchor = snapshot.reliable_scale_anchor
                self.motion_break_events += int(snapshot.motion_break_events)
                self.lost_events += 1
                if (
                    self.max_reacq_cycles is not None
                    and self.reacq_attempts >= self.max_reacq_cycles
                ):
                    self.terminal_status = "REACQ_LIMIT"
                else:
                    if self.adapter is not None:
                        self.adapter.reset_streak()
                    self._start_reacquisition_runtime(item.frame_index, snapshot)
                return TargetSessionStep(
                    phase, item, self.phase, True, self.terminal_status
                )
            if runtime.done and self.frame_count is not None:
                self._last_tracking_snapshot = runtime.continuity_snapshot
                self.expected_area_px = float(
                    self._last_tracking_snapshot.expected_area_px
                )
                self.terminal_status = "COMPLETED"
                return TargetSessionStep(
                    phase, item, self.phase, True, self.terminal_status
                )
            return TargetSessionStep(phase, item, self.phase, False)

        runtime = self._reacquisition_runtime
        if runtime is None:
            raise RuntimeError("REACQ runtime is not active")
        item = runtime.step(frame)
        self.next_frame_index = runtime.next_frame_index
        self.final_frame = item.frame_index
        self.over_budget_frames += int(item.over_budget)
        if item.status == "CONFIRMED":
            snapshot = runtime.continuity_snapshot
            self._last_reacquisition_snapshot = snapshot
            self.expected_area_px = float(snapshot.expected_area_px)
            self.scale_model = snapshot.scale_model
            self.reliable_scale_anchor = snapshot.reliable_scale_anchor
            if snapshot.motion_relocked and snapshot.confirmation_points:
                self.motion_history = list(snapshot.confirmation_points)
            else:
                _append_motion_history(
                    self.motion_history, snapshot.confirmation_points
                )
            self.reacq_successes += 1
            t_reacq = int(item.frame_index - runtime.lost_frame)
            self.reacq_frames_total += t_reacq
            self.reacq_frames_max = max(self.reacq_frames_max, t_reacq)
            self.robust_reacq_count += int(snapshot.motion_broken)
            self.auto_promoted_reacq_count += int(snapshot.auto_promoted)
            if self.adapter is not None:
                self.adapter.reset_streak()
            next_track = int(item.frame_index) + 1
            if self.frame_count is not None and next_track >= self.frame_count:
                self.terminal_status = "COMPLETED"
            else:
                self._start_tracking_runtime(next_track)
            return TargetSessionStep(
                phase, item, self.phase, True, self.terminal_status
            )

        if runtime.done:
            snapshot = runtime.continuity_snapshot
            self._last_reacquisition_snapshot = snapshot
            self.expected_area_px = float(
                snapshot.expected_area_px or self.expected_area_px
            )
            self.scale_model = snapshot.scale_model
            self.reliable_scale_anchor = snapshot.reliable_scale_anchor
            finite_eof = (
                self.frame_count is not None
                and runtime.end_frame_exclusive >= self.frame_count
                and runtime.attempted_frames < self.reacquisition_config.max_frames
            )
            self.terminal_status = (
                "EOF_DURING_REACQ" if finite_eof else "LOST_UNRECOVERED"
            )
            self.robust_reacq_count += int(snapshot.motion_broken)
            self.auto_promoted_reacq_count += int(snapshot.auto_promoted)
            return TargetSessionStep(
                phase, item, self.phase, True, self.terminal_status
            )
        return TargetSessionStep(phase, item, self.phase, False)

    @property
    def tracking_continuity_snapshot(self) -> TrackingContinuitySnapshot | None:
        if self._tracking_runtime is not None:
            return self._tracking_runtime.continuity_snapshot
        return self._last_tracking_snapshot

    @property
    def reacquisition_continuity_snapshot(
        self,
    ) -> ReacquisitionContinuitySnapshot | None:
        if self._reacquisition_runtime is not None:
            return self._reacquisition_runtime.continuity_snapshot
        return self._last_reacquisition_snapshot

    def collection_state_sizes(self) -> dict[str, int]:
        sizes = {"motion_history": len(self.motion_history)}
        if self.adapter is not None:
            sizes["adaptation_events"] = len(self.adapter.events)
        runtime = self._tracking_runtime or self._reacquisition_runtime
        if runtime is not None:
            sizes.update(
                {f"runtime.{key}": value for key, value in runtime.collection_state_sizes().items()}
            )
        return sizes

    def stop(self) -> None:
        if self.terminal_status is None:
            self.terminal_status = "STOPPED"

    def close(self) -> None:
        if self._tracking_runtime is not None:
            self._tracking_runtime.close()
        if self._reacquisition_runtime is not None:
            self._reacquisition_runtime.close()

def _tracking_segment_result(
    *,
    frames: Sequence[TrackFrameResult],
    snapshot: TrackingContinuitySnapshot,
    initial_last_center: tuple[float, float],
    status: str,
    adaptation_adapter: EMAReferenceAdapter | None,
    adaptation_event_start: int,
    adaptation_update_start: int,
) -> TrackingResult:
    accepted = [
        item for item in frames if item.selected is not None and item.status == "TRACK"
    ]
    scores = [item.selected_score for item in accepted]
    confidences = [item.confidence for item in frames]
    prediction_errors = [
        item.prediction_error_px
        for item in accepted
        if item.prediction_error_px is not None
    ]
    shift_times = [
        item.global_shift_time_ms
        for item in frames
        if item.global_shift_method != "OFF"
    ]
    lost_frame = next(
        (item.frame_index for item in frames if item.status == "LOST"), None
    )
    history = snapshot.motion_history
    return TrackingResult(
        status=status,
        frames=tuple(frames),
        tracked_frames=len(scores),
        attempted_frames=len(frames),
        mean_score=float(sum(scores) / len(scores)) if scores else 0.0,
        median_score=float(median(scores)) if scores else 0.0,
        lost_frame=lost_frame,
        last_center=history[-1].xy if history else initial_last_center,
        trajectory=tuple(
            (item.frame_index, item.selected.center_x, item.selected.center_y)
            for item in accepted
            if item.selected is not None
        ),
        mean_confidence=(
            float(sum(confidences) / len(confidences)) if confidences else 0.0
        ),
        median_prediction_error_px=(
            float(median(prediction_errors)) if prediction_errors else 0.0
        ),
        max_search_radius_used_px=max(
            (item.search_radius_px for item in frames), default=0
        ),
        adaptation_updates=(
            adaptation_adapter.update_count - adaptation_update_start
            if adaptation_adapter is not None
            else 0
        ),
        adaptation_events=(
            tuple(adaptation_adapter.events[adaptation_event_start:])
            if adaptation_adapter is not None
            else ()
        ),
        last_target_area_px=float(snapshot.expected_area_px),
        motion_break_events=int(snapshot.motion_break_events),
        motion_broken_at_lost=bool(snapshot.motion_broken_at_lost),
        mean_global_shift_time_ms=(
            float(sum(shift_times) / len(shift_times)) if shift_times else 0.0
        ),
        over_budget_frames=sum(1 for item in frames if item.over_budget),
    )


def _reacquisition_segment_result(
    *,
    lost_frame: int,
    frames: Sequence[ReacqFrameResult],
    snapshot: ReacquisitionContinuitySnapshot,
    status: str,
    expected_area_fallback: float,
) -> ReacquisitionResult:
    confirmed = snapshot.confirmed_frame is not None and status == "CONFIRMED"
    shift_times = [
        item.global_shift_time_ms
        for item in frames
        if item.global_shift_method != "OFF"
    ]
    return ReacquisitionResult(
        status=status,
        lost_frame=lost_frame,
        frames=tuple(frames),
        confirmed_frame=snapshot.confirmed_frame if confirmed else None,
        confirmed_center=snapshot.confirmed_center if confirmed else None,
        t_reacq_frames=(
            snapshot.confirmed_frame - lost_frame
            if confirmed and snapshot.confirmed_frame is not None
            else None
        ),
        confirmation_points=(
            snapshot.confirmation_points if confirmed else ()
        ),
        attempted_frames=len(frames),
        mean_confidence=(
            float(mean(item.confidence for item in frames)) if frames else 0.0
        ),
        max_search_radius_used_px=max(
            (item.search_radius_px for item in frames), default=0
        ),
        confirmed_area_px=snapshot.confirmed_area_px if confirmed else None,
        last_expected_area_px=float(
            snapshot.expected_area_px or expected_area_fallback
        ),
        motion_broken_mode=bool(snapshot.motion_broken),
        motion_relocked=bool(snapshot.motion_relocked and confirmed),
        mean_global_shift_time_ms=(
            float(mean(shift_times)) if shift_times else 0.0
        ),
        over_budget_frames=sum(1 for item in frames if item.over_budget),
        auto_promoted=bool(snapshot.auto_promoted),
        promotion_frame=next(
            (
                item.frame_index
                for item in frames
                if item.auto_promoted_motion_break
            ),
            None,
        ),
    )


def run_tracking_session(
    *,
    frame_reader: FrameReader,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    reference_bank: ReferenceBank,
    initial_history: Sequence[tuple[float, float]],
    initial_history_frames: Sequence[int],
    detector_config: CandidateDetectorConfig,
    tracking_config: TrackingConfig,
    reacquisition_config: ReacquisitionConfig,
    session_config: TrackingSessionConfig | None = None,
    identity_weights: Mapping[str, float] | None = None,
    adaptation_config: EMAConfig | None = None,
    progress_callback: LiveProgressCallback | None = None,
    stop_requested: StopRequested | None = None,
    thermal_profile: ThermalProfile | None = None,
) -> TrackingSessionResult:
    """Finite-video compatibility wrapper over :class:`Vt11TargetSession`."""
    session_cfg = session_config or TrackingSessionConfig()
    session_cfg.validate()
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")

    session = Vt11TargetSession(
        frame_width=frame_width,
        frame_height=frame_height,
        frame_count=frame_count,
        start_frame=start_frame,
        reference_bank=reference_bank,
        initial_history=initial_history,
        initial_history_frames=initial_history_frames,
        detector_config=detector_config,
        tracking_config=tracking_config,
        reacquisition_config=reacquisition_config,
        identity_weights=identity_weights,
        adaptation_config=adaptation_config,
        thermal_profile=thermal_profile,
        max_reacq_cycles=session_cfg.max_reacq_cycles,
        _collect_compatibility_history=True,
    )
    track_segments: list[TrackingResult] = []
    reacquisitions: list[ReacquisitionResult] = []
    trajectory: list[tuple[int, float, float]] = []
    t_reacq: list[int] = []
    track_frames: list[TrackFrameResult] = []
    reacq_frames: list[ReacqFrameResult] = []
    track_initial_center = session.motion_history[-1].xy
    adaptation_event_start = (
        len(session.adapter.events) if session.adapter is not None else 0
    )
    adaptation_update_start = (
        session.adapter.update_count if session.adapter is not None else 0
    )
    current_lost_frame: int | None = None

    try:
        if start_frame >= frame_count:
            session.terminal_status = "COMPLETED"
        while session.terminal_status is None:
            if stop_requested is not None and stop_requested():
                session.stop()
                if session.phase == "TRACK":
                    snapshot = session.tracking_continuity_snapshot
                    if snapshot is None:
                        raise RuntimeError("missing TRACK continuity snapshot")
                    track_segments.append(
                        _tracking_segment_result(
                            frames=track_frames,
                            snapshot=snapshot,
                            initial_last_center=track_initial_center,
                            status="STOPPED",
                            adaptation_adapter=session.adapter,
                            adaptation_event_start=adaptation_event_start,
                            adaptation_update_start=adaptation_update_start,
                        )
                    )
                else:
                    snapshot = session.reacquisition_continuity_snapshot
                    if snapshot is None or current_lost_frame is None:
                        raise RuntimeError("missing REACQ continuity snapshot")
                    reacquisitions.append(
                        _reacquisition_segment_result(
                            lost_frame=current_lost_frame,
                            frames=reacq_frames,
                            snapshot=snapshot,
                            status="STOPPED",
                            expected_area_fallback=session.expected_area_px,
                        )
                    )
                break

            frame_index = session.next_frame_index
            _idx, frame = frame_reader(frame_index)
            event = session.step(frame_index, frame)
            if progress_callback is not None:
                progress_callback(
                    LiveFrameEvent(
                        mode=event.phase,
                        frame_index=frame_index,
                        frame_bgr=frame.copy(),
                        result=event.result,
                    )
                )

            if event.phase == "TRACK":
                item = event.result
                if not isinstance(item, TrackFrameResult):
                    raise TypeError("TRACK phase emitted a non-TRACK result")
                track_frames.append(item)
                if not event.segment_ended:
                    continue
                snapshot = session.tracking_continuity_snapshot
                if snapshot is None:
                    raise RuntimeError("missing TRACK continuity snapshot")
                segment_status = (
                    "LOST" if item.status == "LOST" else "COMPLETED"
                )
                track_segments.append(
                    _tracking_segment_result(
                        frames=track_frames,
                        snapshot=snapshot,
                        initial_last_center=track_initial_center,
                        status=segment_status,
                        adaptation_adapter=session.adapter,
                        adaptation_event_start=adaptation_event_start,
                        adaptation_update_start=adaptation_update_start,
                    )
                )
                accepted_points = [
                    MotionPoint(
                        frame_item.frame_index,
                        frame_item.selected.center_x,
                        frame_item.selected.center_y,
                    )
                    for frame_item in track_frames
                    if frame_item.selected is not None
                ]
                trajectory.extend(
                    (point.frame_index, point.x, point.y)
                    for point in accepted_points
                )
                track_frames = []
                if item.status == "LOST" and session.terminal_status is None:
                    current_lost_frame = item.frame_index
                    reacq_frames = []
                continue

            item = event.result
            if not isinstance(item, ReacqFrameResult):
                raise TypeError("REACQ phase emitted a non-REACQ result")
            reacq_frames.append(item)
            if not event.segment_ended:
                continue
            snapshot = session.reacquisition_continuity_snapshot
            if snapshot is None or current_lost_frame is None:
                raise RuntimeError("missing REACQ continuity snapshot")
            if item.status == "CONFIRMED":
                reacq_status = "CONFIRMED"
            elif session.terminal_status == "EOF_DURING_REACQ":
                reacq_status = "EOF"
            else:
                reacq_status = "FAILED"
            reacq = _reacquisition_segment_result(
                lost_frame=current_lost_frame,
                frames=reacq_frames,
                snapshot=snapshot,
                status=reacq_status,
                expected_area_fallback=session.expected_area_px,
            )
            reacquisitions.append(reacq)
            trajectory.extend(
                (point.frame_index, point.x, point.y)
                for point in reacq.confirmation_points
            )
            if reacq.t_reacq_frames is not None:
                t_reacq.append(int(reacq.t_reacq_frames))
            reacq_frames = []
            current_lost_frame = None
            if reacq.status == "CONFIRMED" and session.terminal_status is None:
                track_initial_center = session.motion_history[-1].xy
                adaptation_event_start = (
                    len(session.adapter.events)
                    if session.adapter is not None
                    else 0
                )
                adaptation_update_start = (
                    session.adapter.update_count
                    if session.adapter is not None
                    else 0
                )
    finally:
        session.close()

    all_track = [
        frame for segment in track_segments for frame in segment.frames
    ]
    accepted = [frame for frame in all_track if frame.selected is not None]
    adapt_cfg = adaptation_config or EMAConfig(mode="off")
    events = tuple(session.adapter.events) if session.adapter is not None else ()
    updates = session.adapter.update_count if session.adapter is not None else 0
    final_status = session.terminal_status or "COMPLETED"
    return TrackingSessionResult(
        status=final_status,
        track_segments=tuple(track_segments),
        reacquisitions=tuple(reacquisitions),
        lost_events=sum(
            1 for segment in track_segments if segment.status == "LOST"
        ),
        reacq_successes=sum(
            1 for result in reacquisitions if result.status == "CONFIRMED"
        ),
        t_reacq_frames=tuple(t_reacq),
        mean_t_reacq_frames=float(mean(t_reacq)) if t_reacq else 0.0,
        max_t_reacq_frames=max(t_reacq, default=0),
        trajectory=tuple(trajectory),
        final_frame=session.final_frame,
        adaptation_mode=adapt_cfg.mode,
        adaptation_updates=updates,
        adaptation_events=events,
        adaptation_slow_updates=(
            session.adapter.slow_update_count
            if session.adapter is not None
            else 0
        ),
        adaptation_fast_updates=(
            session.adapter.fast_update_count
            if session.adapter is not None
            else 0
        ),
        adaptation_recent_updates=(
            session.adapter.recent_update_count
            if session.adapter is not None
            else 0
        ),
        adaptation_recent_count=session.reference_bank.recent_count,
        mean_track_confidence=(
            float(mean(frame.confidence for frame in all_track))
            if all_track
            else 0.0
        ),
        track_accepted_frames=len(accepted),
        final_reference_bank=session.reference_bank,
        final_expected_area_px=float(session.expected_area_px),
        motion_break_events=sum(
            segment.motion_break_events for segment in track_segments
        ),
        robust_reacq_count=sum(
            1 for result in reacquisitions if result.motion_broken_mode
        ),
        auto_promoted_reacq_count=sum(
            1 for result in reacquisitions if result.auto_promoted
        ),
        over_budget_frames=(
            sum(segment.over_budget_frames for segment in track_segments)
            + sum(result.over_budget_frames for result in reacquisitions)
        ),
    )
