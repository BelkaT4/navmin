"""Bounded DAY maneuver hypothesis confirmation for VT11 tracking."""
from dataclasses import dataclass
from math import hypot, log


@dataclass(frozen=True)
class ManeuverEvidence:
    frame: int
    center: tuple[float, float]
    area: float
    identity: float


def confirms_maneuver(
    previous: ManeuverEvidence,
    current: ManeuverEvidence,
    origin: tuple[float, float],
) -> bool:
    """Require two adjacent, moving, size-consistent visual observations."""
    a = (previous.center[0] - origin[0], previous.center[1] - origin[1])
    b = (
        current.center[0] - previous.center[0],
        current.center[1] - previous.center[1],
    )
    first, second = hypot(*a), hypot(*b)
    return bool(
        current.frame == previous.frame + 1
        and min(previous.identity, current.identity) >= 0.50
        and (previous.identity + current.identity) / 2 >= 0.52
        and 10 <= first <= 180
        and 3 <= second <= 180
        and 0.15 <= second / max(first, 1) <= 3.5
        and (a[0] * b[0] + a[1] * b[1]) / max(1, first * second) >= 0.5
        and abs(log(max(1, current.area) / max(1, previous.area))) <= 0.5
    )
