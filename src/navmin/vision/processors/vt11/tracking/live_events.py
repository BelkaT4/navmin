"""UI-agnostic live progress events for VT-10 HOTFIX-1.

The tracking core stays independent from Qt.  A caller may subscribe to
per-frame events and display them, log them, or block between frames for
pause/resume control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class LiveFrameEvent:
    mode: str  # TRACK or REACQ
    frame_index: int
    frame_bgr: np.ndarray
    result: Any


LiveProgressCallback = Callable[[LiveFrameEvent], None]
StopRequested = Callable[[], bool]
