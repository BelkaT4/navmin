"""Bounded original VT11 ACQUIRE -> INITIAL TRAIN coordination."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectionResult,
    detect_initial_acquire_candidates,
    detect_local_candidates,
)
from navmin.vision.processors.vt11.acquisition.initial_train import (
    InitialTrainResult,
    run_initial_train,
)
from navmin.vision.processors.vt11.acquisition.roi import (
    OperatorSelection,
    make_operator_selection,
)
from navmin.vision.processors.vt11.identity.scorer import (
    ScoringResult,
    score_candidates,
)
from navmin.vision.processors.vt11.runtime_config import (
    Vt11ResolvedConfig,
    Vt11RuntimeSettings,
)
from navmin.vision.processors.vt11.tracking.session import Vt11TargetSession


@dataclass(frozen=True)
class Vt11AcquisitionResult:
    decision: str
    selection: OperatorSelection
    detection: CandidateDetectionResult
    scoring: ScoringResult
    train_result: InitialTrainResult | None
    target_session: Vt11TargetSession | None
    retrospective_replay_frames: int = 0
    continuation_error: str | None = None


class Vt11AcquisitionAttempt:
    """Collect exactly the bounded window required by original INITIAL TRAIN."""

    def __init__(
        self,
        *,
        frame_index: int,
        frame_bgr: np.ndarray,
        click_xy: tuple[float, float],
        frame_width: int,
        frame_height: int,
        resolved_config: Vt11ResolvedConfig,
        runtime_settings: Vt11RuntimeSettings | None = None,
    ) -> None:
        settings = runtime_settings or Vt11RuntimeSettings()
        settings.validate()
        if frame_width < 1 or frame_height < 1:
            raise ValueError("frame dimensions must be >= 1")
        if frame_bgr.shape[:2] != (frame_height, frame_width):
            raise ValueError("frame dimensions do not match frame_bgr")

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.start_frame = int(frame_index)
        self.resolved_config = resolved_config
        self.runtime_settings = settings
        acquire_cfg = resolved_config.effective_config["acquire"]
        self.selection = make_operator_selection(
            frame_index=frame_index,
            click_xy=click_xy,
            radius_px=int(acquire_cfg["roi_radius_px"]),
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if resolved_config.profile_name == "thermal":
            detection, _effective_selection = detect_initial_acquire_candidates(
                frame_bgr,
                self.selection,
                top_k=resolved_config.initial_train.top_k,
                config=resolved_config.detector,
            )
        else:
            detection = detect_local_candidates(
                frame_bgr,
                self.selection,
                top_k=resolved_config.initial_train.top_k,
                config=resolved_config.detector,
            )
        self.detection = detection
        self.scoring = score_candidates(
            frame_bgr,
            detection.candidates,
            config=resolved_config.acquire_scoring,
            identity_weights=resolved_config.identity_weights,
        )

        lookahead = (
            resolved_config.initial_train.temporal_click_bootstrap_frames - 1
            if resolved_config.initial_train.temporal_click_acquire_enabled
            else 0
        )
        self.required_window_frames = (
            resolved_config.initial_train.initial_frames + max(0, lookahead)
        )
        self.next_frame_index = self.start_frame + 1
        self._frames: dict[int, np.ndarray] = {
            self.start_frame: frame_bgr.copy()
        }
        self.result: Vt11AcquisitionResult | None = None
        self.done = False
        if self.scoring.best is None:
            self.result = Vt11AcquisitionResult(
                decision="AMBIGUOUS",
                selection=self.selection,
                detection=self.detection,
                scoring=self.scoring,
                train_result=None,
                target_session=None,
                continuation_error="NO_ACQUIRE_CANDIDATES",
            )
            self.done = True
            self._frames.clear()
        elif self.required_window_frames == 1:
            self._finish()

    @property
    def collected_frame_count(self) -> int:
        return len(self._frames)

    def collection_state_sizes(self) -> dict[str, int]:
        return {
            "raw_frames": len(self._frames),
            "required_window_frames": self.required_window_frames,
        }

    def step(self, frame_index: int, frame_bgr: np.ndarray) -> Vt11AcquisitionResult | None:
        if self.done:
            raise RuntimeError("acquisition attempt is already complete")
        if int(frame_index) != self.next_frame_index:
            raise ValueError(
                f"expected frame {self.next_frame_index}, got {int(frame_index)}"
            )
        if frame_bgr.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError("frame dimensions do not match acquisition attempt")
        self._frames[int(frame_index)] = frame_bgr.copy()
        self.next_frame_index += 1
        if len(self._frames) >= self.required_window_frames:
            self._finish()
        return self.result

    def _finish(self) -> None:
        best = self.scoring.best
        if best is None:
            raise RuntimeError("cannot run INITIAL TRAIN without an ACQUIRE candidate")

        def reader(frame_index: int) -> tuple[int, np.ndarray]:
            try:
                return int(frame_index), self._frames[int(frame_index)]
            except KeyError as exc:
                raise IndexError(
                    f"frame {frame_index} is outside acquisition window"
                ) from exc

        last_collected = max(self._frames)
        cfg = self.resolved_config
        train = run_initial_train(
            frame_reader=reader,
            frame_count=last_collected + 1,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            start_frame=self.start_frame,
            initial_candidate=best.candidate,
            roi_radius_px=self.selection.roi_radius_px,
            detector_config=cfg.detector,
            config=cfg.initial_train,
            identity_weights=cfg.identity_weights,
            thermal_identity_weights=cfg.thermal_identity_weights,
            video_mode="thermal" if cfg.profile_name == "thermal" else "visible",
            thermal_config=cfg.thermal,
            initial_click_xy=(self.selection.click_x, self.selection.click_y),
        )
        target_session: Vt11TargetSession | None = None
        replayed = 0
        continuation_error: str | None = None
        if train.decision == "CONFIRMED":
            selected_train = [
                item for item in train.frames if item.selected is not None
            ]
            if not selected_train:
                continuation_error = "CONFIRMED_WITHOUT_SELECTED_TRAIN_HISTORY"
            else:
                last_train_frame = train.frames[-1].frame_index
                catch_up = [
                    (index, self._frames[index])
                    for index in sorted(self._frames)
                    if index > last_train_frame
                ]
                limit = self.runtime_settings.max_retrospective_replay_frames
                if len(catch_up) > limit:
                    continuation_error = (
                        "RETROSPECTIVE_REPLAY_LIMIT_EXCEEDED: "
                        f"required={len(catch_up)}, limit={limit}"
                    )
                else:
                    target_session = Vt11TargetSession(
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                        start_frame=last_train_frame + 1,
                        reference_bank=train.reference_bank,
                        initial_history=[
                            item.selected.center
                            for item in selected_train
                            if item.selected is not None
                        ],
                        initial_history_frames=[
                            item.frame_index for item in selected_train
                        ],
                        detector_config=cfg.detector,
                        tracking_config=cfg.tracking,
                        reacquisition_config=cfg.reacquisition,
                        identity_weights=cfg.identity_weights,
                        adaptation_config=cfg.adaptation,
                        thermal_profile=train.thermal_profile,
                    )
                    for replay_index, replay_frame in catch_up:
                        target_session.step(replay_index, replay_frame)
                        replayed += 1
                        if target_session.terminal_status is not None:
                            continuation_error = (
                                "TARGET_SESSION_TERMINATED_DURING_CATCH_UP: "
                                f"{target_session.terminal_status}"
                            )
                            target_session.close()
                            target_session = None
                            break

        self.result = Vt11AcquisitionResult(
            decision=train.decision,
            selection=self.selection,
            detection=self.detection,
            scoring=self.scoring,
            train_result=train,
            target_session=target_session,
            retrospective_replay_frames=replayed,
            continuation_error=continuation_error,
        )
        self.done = True
        self._frames.clear()
