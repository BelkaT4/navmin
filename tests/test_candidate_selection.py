from __future__ import annotations

import pytest

from navmin.contracts import BBox, TargetCandidateProposal
from navmin.vision.candidate_selection import (
    auto_select_candidate_at_point,
    candidate_at_point,
    point_to_bbox_distance,
)


def _proposal(candidate_id: int, bbox: BBox, *, rank: int) -> TargetCandidateProposal:
    return TargetCandidateProposal(candidate_id, bbox, 1.0 / rank, rank)


def test_point_to_bbox_distance_and_half_open_hit_testing() -> None:
    bbox = BBox(10, 20, 11, 9)
    assert point_to_bbox_distance(10, 20, bbox) == 0.0
    assert point_to_bbox_distance(20.999, 28.999, bbox) == 0.0
    assert point_to_bbox_distance(21, 29, bbox) == 0.0
    assert point_to_bbox_distance(26, 29, bbox) == pytest.approx(5.0)
    assert point_to_bbox_distance(24, 33, bbox) == pytest.approx(5.0)

    adjacent = (
        _proposal(1, BBox(10, 10, 10, 10), rank=1),
        _proposal(2, BBox(20, 10, 10, 10), rank=2),
    )
    selected = candidate_at_point(adjacent, 20, 15, tolerance_px=0)
    assert selected is not None
    assert selected.candidate_id == 2


def test_candidate_hit_test_prefers_box_containing_click_over_nearby_rank() -> None:
    candidates = (
        _proposal(1, BBox(10, 10, 10, 10), rank=2),
        _proposal(2, BBox(22, 10, 10, 10), rank=1),
    )
    selected = candidate_at_point(candidates, 18, 15, tolerance_px=15)
    assert selected is not None
    assert selected.candidate_id == 1


def test_candidate_hit_test_uses_nearest_bbox_then_rank_as_tie_break() -> None:
    candidates = (
        _proposal(1, BBox(10, 10, 10, 10), rank=2),
        _proposal(2, BBox(30, 10, 10, 10), rank=1),
    )
    assert candidate_at_point(candidates, 25, 15, tolerance_px=5).candidate_id == 2

    tied = (
        _proposal(3, BBox(10, 10, 10, 10), rank=2),
        _proposal(4, BBox(30, 10, 10, 10), rank=1),
    )
    assert candidate_at_point(tied, 25, 15, tolerance_px=5).candidate_id == 4


def test_auto_select_rejects_overlapping_boxes_even_when_rank_differs() -> None:
    candidates = (
        _proposal(1, BBox(10, 10, 15, 15), rank=2),
        _proposal(2, BBox(15, 10, 15, 15), rank=1),
    )
    assert auto_select_candidate_at_point(
        candidates, 20, 15, tolerance_px=15
    ) is None


def test_auto_select_rejects_near_equal_distances_within_five_pixels() -> None:
    candidates = (
        _proposal(1, BBox(10, 10, 10, 10), rank=2),
        _proposal(2, BBox(32, 10, 10, 10), rank=1),
    )
    # Distances are 5 px and 7 px. Rank must not override the 5 px ambiguity rule.
    assert auto_select_candidate_at_point(
        candidates, 25, 15, tolerance_px=15
    ) is None


def test_auto_select_accepts_clearly_nearest_candidate() -> None:
    candidates = (
        _proposal(1, BBox(10, 10, 10, 10), rank=2),
        _proposal(2, BBox(38, 10, 10, 10), rank=1),
    )
    selected = auto_select_candidate_at_point(
        candidates, 25, 15, tolerance_px=15
    )
    assert selected is not None
    assert selected.candidate_id == 1


def test_candidate_hit_test_returns_none_outside_tolerance() -> None:
    candidates = (_proposal(1, BBox(10, 10, 10, 10), rank=1),)
    assert candidate_at_point(candidates, 40, 40, tolerance_px=15) is None
    assert auto_select_candidate_at_point(
        candidates, 40, 40, tolerance_px=15
    ) is None


def test_candidate_hit_test_validates_tolerance_and_ambiguity() -> None:
    candidates = (_proposal(1, BBox(10, 10, 10, 10), rank=1),)
    with pytest.raises(TypeError):
        candidate_at_point(candidates, 10, 10, tolerance_px=True)
    with pytest.raises(ValueError):
        candidate_at_point(candidates, 10, 10, tolerance_px=51)
    with pytest.raises(ValueError):
        auto_select_candidate_at_point(
            candidates, 10, 10, tolerance_px=15, ambiguity_px=-1
        )
