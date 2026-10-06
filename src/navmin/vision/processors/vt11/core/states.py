from __future__ import annotations

from enum import Enum


class TrackerState(str, Enum):
    SELECT = "SELECT"
    ACQUIRE = "ACQUIRE"
    TRAIN = "TRAIN"
    TRACK = "TRACK"
    TRACK_DEGRADED = "TRACK_DEGRADED"
    ADAPT = "ADAPT"
    LOST = "LOST"
    REACQ = "REACQ"
