"""Robust two-time-scale area model for THERMAL tracking.

The model prevents one wrong large hot blob from exponentially inflating the
candidate-area gate.  BaseArea is immutable; CurrentArea changes only after a
short consistent run.  MICRO scale is a separate state and does not overwrite
the normal target scale.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, log
from statistics import median


@dataclass(frozen=True)
class ScaleStabilityConfig:
    enabled: bool = False
    confirm_frames: int = 4
    consistency_log_span: float = 0.35
    max_log_change_per_frame: float = 0.025
    large_change_ratio: float = 1.8
    large_change_confirm_frames: int = 6
    micro_enabled: bool = True
    micro_max_area_px: float = 80.0
    micro_confirm_frames: int = 3
    micro_exit_confirm_frames: int = 4
    micro_consistency_log_span: float = 0.55
    micro_max_log_change_per_frame: float = 0.12
    robust_history_frames: int = 12

    def validate(self) -> None:
        if self.confirm_frames < 2 or self.large_change_confirm_frames < self.confirm_frames:
            raise ValueError("scale confirmation settings invalid")
        if self.consistency_log_span <= 0 or self.micro_consistency_log_span <= 0:
            raise ValueError("scale consistency spans must be >0")
        if self.max_log_change_per_frame <= 0 or self.micro_max_log_change_per_frame <= 0:
            raise ValueError("scale log-rate limits must be >0")
        if self.large_change_ratio <= 1.0:
            raise ValueError("large_change_ratio must be >1")
        if self.micro_max_area_px < 1 or self.micro_confirm_frames < 2 or self.micro_exit_confirm_frames < 2:
            raise ValueError("micro scale settings invalid")
        if self.robust_history_frames < self.large_change_confirm_frames:
            raise ValueError("robust_history_frames must be >= large_change_confirm_frames")


@dataclass(frozen=True)
class ScaleObservation:
    updated: bool
    reason: str
    state: str
    base_area_px: float
    current_area_px: float
    effective_area_px: float
    pending_count: int


class ScaleStabilityModel:
    def __init__(self, base_area_px: float, config: ScaleStabilityConfig | None = None) -> None:
        cfg = config or ScaleStabilityConfig()
        cfg.validate()
        self.config = cfg
        self.base_area_px = float(max(1.0, base_area_px))
        self.current_area_px = self.base_area_px
        self.micro_area_px = min(self.base_area_px, cfg.micro_max_area_px)
        self.micro_mode = False
        self._normal: deque[tuple[int, float]] = deque(maxlen=max(cfg.robust_history_frames, cfg.large_change_confirm_frames, cfg.confirm_frames))
        self._micro: deque[tuple[int, float]] = deque(maxlen=cfg.micro_confirm_frames + 2)
        self._exit_micro: deque[tuple[int, float]] = deque(maxlen=cfg.micro_exit_confirm_frames + 2)
        self._last_update_frame: int | None = None

    @property
    def state(self) -> str:
        return "MICRO_SCALE" if self.micro_mode else "NORMAL_SCALE"

    @property
    def effective_area_px(self) -> float:
        return float(self.micro_area_px if self.micro_mode else self.current_area_px)

    @staticmethod
    def _consistent(items: list[tuple[int, float]], max_span: float) -> bool:
        if not items:
            return False
        logs = [log(max(1.0, a)) for _, a in items]
        return max(logs) - min(logs) <= max_span

    def _bounded_update(self, old: float, target: float, frame_index: int, max_rate: float) -> float:
        dt = 1 if self._last_update_frame is None else max(1, int(frame_index) - int(self._last_update_frame))
        delta = log(max(1.0, target) / max(1.0, old))
        limit = max_rate * dt
        delta = max(-limit, min(limit, delta))
        self._last_update_frame = int(frame_index)
        return float(max(1.0, old * exp(delta)))

    def snapshot(self, reason: str = "NO_UPDATE", *, updated: bool = False, pending: int = 0) -> ScaleObservation:
        return ScaleObservation(updated, reason, self.state, self.base_area_px, self.current_area_px, self.effective_area_px, pending)


    def commit_confirmed_reacquisition(self, *, frame_index: int, area_px: float, is_micro: bool) -> ScaleObservation:
        """Commit scale only after REACQ has already passed temporal confirmation.

        REACQ confirmation itself represents several consistent frames.  For a
        MICRO target we therefore may enter MICRO_SCALE immediately; otherwise
        the next TRACK frame would still use the old large area gate and could
        lose the just-reacquired target.  Normal-scale changes remain bounded.
        """
        if not self.config.enabled:
            return self.snapshot("DISABLED")
        area = float(max(1.0, area_px))
        cfg = self.config
        if cfg.micro_enabled and (is_micro or area <= cfg.micro_max_area_px):
            self.micro_mode = True
            self.micro_area_px = area
            self._micro.clear(); self._normal.clear(); self._exit_micro.clear()
            self._last_update_frame = int(frame_index)
            return self.snapshot("REACQ_CONFIRMED_ENTER_MICRO", updated=True, pending=cfg.micro_confirm_frames)
        if self.micro_mode:
            self.micro_mode = False
            self.current_area_px = self._bounded_update(self.current_area_px, area, frame_index, cfg.max_log_change_per_frame)
            self._micro.clear(); self._normal.clear(); self._exit_micro.clear()
            return self.snapshot("REACQ_CONFIRMED_EXIT_MICRO", updated=True, pending=cfg.micro_exit_confirm_frames)
        # A normal target still obeys the temporal/rate-limited normal model.
        return self.observe(frame_index=frame_index, area_px=area, trusted=True, is_micro=False)

    def observe(self, *, frame_index: int, area_px: float, trusted: bool, is_micro: bool = False) -> ScaleObservation:
        if not self.config.enabled:
            return self.snapshot("DISABLED")
        area = float(max(1.0, area_px))
        if not trusted:
            return self.snapshot("UNTRUSTED")
        cfg = self.config

        if cfg.micro_enabled and (is_micro or area <= cfg.micro_max_area_px):
            self._normal.clear(); self._exit_micro.clear()
            self._micro.append((int(frame_index), area))
            recent = list(self._micro)[-cfg.micro_confirm_frames:]
            if len(recent) < cfg.micro_confirm_frames:
                return self.snapshot("MICRO_PENDING", pending=len(recent))
            if not self._consistent(recent, cfg.micro_consistency_log_span):
                return self.snapshot("MICRO_INCONSISTENT", pending=len(recent))
            target = float(median(a for _, a in recent))
            if not self.micro_mode:
                self.micro_mode = True
                self.micro_area_px = target
                self._last_update_frame = int(frame_index)
                return self.snapshot("ENTER_MICRO_SCALE", updated=True, pending=len(recent))
            self.micro_area_px = self._bounded_update(self.micro_area_px, target, frame_index, cfg.micro_max_log_change_per_frame)
            return self.snapshot("UPDATE_MICRO_SCALE", updated=True, pending=len(recent))

        self._micro.clear()
        if self.micro_mode:
            self._exit_micro.append((int(frame_index), area))
            recent = list(self._exit_micro)[-cfg.micro_exit_confirm_frames:]
            if len(recent) < cfg.micro_exit_confirm_frames:
                return self.snapshot("MICRO_EXIT_PENDING", pending=len(recent))
            if not self._consistent(recent, cfg.consistency_log_span):
                return self.snapshot("MICRO_EXIT_INCONSISTENT", pending=len(recent))
            target = float(median(a for _, a in recent))
            self.micro_mode = False
            self.current_area_px = self._bounded_update(self.current_area_px, target, frame_index, cfg.max_log_change_per_frame)
            self._exit_micro.clear()
            return self.snapshot("EXIT_MICRO_SCALE", updated=True, pending=len(recent))

        self._normal.append((int(frame_index), area))
        ratio = max(area / max(self.current_area_px, 1.0), self.current_area_px / max(area, 1.0))
        required = cfg.large_change_confirm_frames if ratio >= cfg.large_change_ratio else cfg.confirm_frames
        recent = list(self._normal)[-required:]
        if len(recent) < required:
            return self.snapshot("NORMAL_PENDING", pending=len(recent))
        if not self._consistent(recent, cfg.consistency_log_span):
            return self.snapshot("NORMAL_INCONSISTENT", pending=len(recent))
        # HOTFIX-2: after the short consistency gate, move toward a longer
        # robust median instead of recursively following every short run.
        robust_recent = list(self._normal)[-cfg.robust_history_frames:]
        target = float(median(a for _, a in robust_recent))
        self.current_area_px = self._bounded_update(self.current_area_px, target, frame_index, cfg.max_log_change_per_frame)
        return self.snapshot("UPDATE_NORMAL_SCALE", updated=True, pending=len(recent))
