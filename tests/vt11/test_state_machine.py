import pytest

from navmin.vision.processors.vt11.core.state_machine import TrackingStateMachine
from navmin.vision.processors.vt11.core.states import TrackerState


def test_nominal_state_sequence():
    sm = TrackingStateMachine()
    for state in [
        TrackerState.ACQUIRE,
        TrackerState.TRAIN,
        TrackerState.TRACK,
        TrackerState.LOST,
        TrackerState.REACQ,
        TrackerState.TRACK,
    ]:
        sm.transition(state)
    assert sm.state is TrackerState.TRACK


def test_invalid_transition_is_rejected():
    sm = TrackingStateMachine()
    with pytest.raises(ValueError):
        sm.transition(TrackerState.TRACK)
