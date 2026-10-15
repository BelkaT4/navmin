from __future__ import annotations

from dataclasses import dataclass
from .states import TrackerState


_ALLOWED = {
    TrackerState.SELECT: {TrackerState.ACQUIRE},
    TrackerState.ACQUIRE: {TrackerState.TRAIN, TrackerState.SELECT},
    TrackerState.TRAIN: {TrackerState.TRACK, TrackerState.SELECT},
    TrackerState.TRACK: {TrackerState.TRACK_DEGRADED, TrackerState.ADAPT, TrackerState.LOST, TrackerState.SELECT},
    TrackerState.TRACK_DEGRADED: {TrackerState.TRACK, TrackerState.LOST, TrackerState.SELECT},
    TrackerState.ADAPT: {TrackerState.TRACK, TrackerState.TRACK_DEGRADED, TrackerState.LOST, TrackerState.SELECT},
    TrackerState.LOST: {TrackerState.REACQ, TrackerState.SELECT},
    TrackerState.REACQ: {TrackerState.TRACK, TrackerState.LOST, TrackerState.SELECT},
}


@dataclass
class TrackingStateMachine:
    state: TrackerState = TrackerState.SELECT

    def can_transition(self, new_state: TrackerState) -> bool:
        return new_state in _ALLOWED[self.state]

    def transition(self, new_state: TrackerState) -> None:
        if not self.can_transition(new_state):
            raise ValueError(f"Invalid transition: {self.state.value} -> {new_state.value}")
        self.state = new_state

    def reset(self) -> None:
        self.state = TrackerState.SELECT
