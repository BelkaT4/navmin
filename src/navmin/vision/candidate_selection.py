"""Backend-neutral hit testing for operator candidate selection."""

from __future__ import annotations

from math import hypot

from navmin.contracts import BBox, TargetCandidateProposal

AUTO_SELECTION_AMBIGUITY_PX = 5.0


def _bbox_contains_point(x_px: float, y_px: float, bbox: BBox) -> bool:
    """Return whether a point is inside the public half-open bbox."""
    x = float(x_px)
    y = float(y_px)
    return (
        float(bbox.x) <= x < float(bbox.x + bbox.width)
        and float(bbox.y) <= y < float(bbox.y + bbox.height)
    )


def point_to_bbox_distance(x_px: float, y_px: float, bbox: BBox) -> float:
    """Return Euclidean distance to the half-open bbox geometry.

    Points inside the bbox have zero distance. A point on the right/bottom
    geometric boundary is outside by hit-test semantics but still has zero
    geometric distance to the rectangle.
    """
    left = float(bbox.x)
    top = float(bbox.y)
    right = float(bbox.x + bbox.width)
    bottom = float(bbox.y + bbox.height)
    x = float(x_px)
    y = float(y_px)

    if x < left:
        dx = left - x
    elif x > right:
        dx = x - right
    else:
        dx = 0.0

    if y < top:
        dy = top - y
    elif y > bottom:
        dy = y - bottom
    else:
        dy = 0.0

    return hypot(dx, dy)


def _validate_tolerance(tolerance_px: int) -> None:
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, int):
        raise TypeError("tolerance_px must be int")
    if not 0 <= tolerance_px <= 50:
        raise ValueError("tolerance_px must be in [0, 50]")


def candidate_at_point(
    candidates: tuple[TargetCandidateProposal, ...],
    x_px: float,
    y_px: float,
    *,
    tolerance_px: int,
) -> TargetCandidateProposal | None:
    """Resolve an explicit second-click selection.

    A bbox containing the click outranks nearby boxes. If several containing
    boxes exist, VT11 rank provides a deterministic tie-break. Otherwise the
    nearest bbox within tolerance wins, with rank as the final tie-break.
    """
    _validate_tolerance(tolerance_px)
    containing = [
        item for item in candidates if _bbox_contains_point(x_px, y_px, item.bbox)
    ]
    if containing:
        return min(containing, key=lambda item: (item.rank, item.candidate_id))

    nearby = [
        (point_to_bbox_distance(x_px, y_px, item.bbox), item)
        for item in candidates
    ]
    nearby = [
        (distance, item)
        for distance, item in nearby
        if distance <= float(tolerance_px)
    ]
    if not nearby:
        return None
    return min(
        nearby,
        key=lambda pair: (pair[0], pair[1].rank, pair[1].candidate_id),
    )[1]


def auto_select_candidate_at_point(
    candidates: tuple[TargetCandidateProposal, ...],
    x_px: float,
    y_px: float,
    *,
    tolerance_px: int,
    ambiguity_px: float = AUTO_SELECTION_AMBIGUITY_PX,
) -> TargetCandidateProposal | None:
    """Conservatively resolve a first-click candidate when intent is clear.

    Auto-selection never uses VT11 rank to break an ambiguous operator click.
    Multiple containing boxes are ambiguous. For nearby boxes, the nearest
    candidate must beat the second-nearest by more than ``ambiguity_px``.
    """
    _validate_tolerance(tolerance_px)
    if ambiguity_px < 0:
        raise ValueError("ambiguity_px must be >= 0")

    containing = [
        item for item in candidates if _bbox_contains_point(x_px, y_px, item.bbox)
    ]
    if len(containing) == 1:
        return containing[0]
    if len(containing) > 1:
        return None

    nearby = sorted(
        (
            (point_to_bbox_distance(x_px, y_px, item.bbox), item)
            for item in candidates
            if point_to_bbox_distance(x_px, y_px, item.bbox)
            <= float(tolerance_px)
        ),
        key=lambda pair: (pair[0], pair[1].rank, pair[1].candidate_id),
    )
    if not nearby:
        return None
    if len(nearby) == 1:
        return nearby[0][1]
    if nearby[1][0] - nearby[0][0] <= float(ambiguity_px):
        return None
    return nearby[0][1]


__all__ = [
    "AUTO_SELECTION_AMBIGUITY_PX",
    "auto_select_candidate_at_point",
    "candidate_at_point",
    "point_to_bbox_distance",
]
