from __future__ import annotations

import pytest

from navmin.contracts import BBox
from navmin.vision.tracking import SimpleTracker


def box(x: int, y: int = 20, width: int = 20, height: int = 20) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


def confirm(
    tracker: SimpleTracker,
    first: BBox,
    second: BBox,
    *,
    start_ns: int = 0,
    step_ns: int = 1_000_000_000,
):
    assert tracker.update([first], timestamp_ns=start_ns) == ()
    result = tracker.update([second], timestamp_ns=start_ns + step_ns)
    assert len(result) == 1
    return result[0]


def test_first_detection_is_tentative_second_consecutive_match_confirms() -> None:
    tracker = SimpleTracker()

    assert tracker.update([box(10)], timestamp_ns=0) == ()
    result = tracker.update([box(20)], timestamp_ns=1_000_000_000)

    assert len(result) == 1
    assert result[0].track_id == 1
    assert result[0].bbox == box(20)
    assert result[0].age_frames == 2
    assert result[0].velocity_x_px_s == pytest.approx(10.0)
    assert result[0].velocity_y_px_s == pytest.approx(0.0)


def test_tentative_confirmation_requires_consecutive_matches() -> None:
    tracker = SimpleTracker()

    assert tracker.update([box(10)], timestamp_ns=0) == ()
    assert tracker.update([], timestamp_ns=1_000_000_000) == ()
    assert tracker.update([box(12)], timestamp_ns=2_000_000_000) == ()
    result = tracker.update([box(14)], timestamp_ns=3_000_000_000)

    assert len(result) == 1
    assert result[0].track_id == 1
    assert result[0].age_frames == 4


def test_track_id_is_stable_and_velocity_uses_real_timestamps() -> None:
    tracker = SimpleTracker()
    confirmed = confirm(
        tracker,
        box(10),
        box(20),
        start_ns=5_000_000_000,
        step_ns=500_000_000,
    )

    next_result = tracker.update([box(30)], timestamp_ns=6_500_000_000)

    assert next_result[0].track_id == confirmed.track_id == 1
    assert next_result[0].velocity_x_px_s == pytest.approx(15.0)


def test_velocity_is_robust_median_over_matched_history() -> None:
    tracker = SimpleTracker()
    positions = (0, 10, 20, 120, 130)
    published = ()
    for second, x in enumerate(positions):
        published = tracker.update([box(x)], timestamp_ns=second * 1_000_000_000)

    assert published
    assert published[0].velocity_x_px_s == pytest.approx(10.0)


def test_nonpositive_timestamp_delta_is_ignored_safely() -> None:
    tracker = SimpleTracker()

    assert tracker.update([box(0)], timestamp_ns=10) == ()
    result = tracker.update([box(10)], timestamp_ns=10)
    assert result[0].velocity_x_px_s == 0.0

    result = tracker.update([box(20)], timestamp_ns=9)
    assert result[0].velocity_x_px_s == 0.0


def test_matched_history_is_capped_at_five_observations() -> None:
    tracker = SimpleTracker()
    for index in range(9):
        tracker.update([box(index * 5)], timestamp_ns=index * 1_000_000_000)

    assert len(tracker._tracks) == 1
    assert len(tracker._tracks[0].history) == 5


def test_misses_do_not_publish_prediction_and_third_miss_deletes() -> None:
    tracker = SimpleTracker()
    confirmed = confirm(tracker, box(0), box(10))

    assert tracker.update([], timestamp_ns=2_000_000_000) == ()
    assert tracker._tracks[0].misses == 1
    assert tracker.update([], timestamp_ns=3_000_000_000) == ()
    assert tracker._tracks[0].misses == 2

    reacquired = tracker.update([box(30)], timestamp_ns=4_000_000_000)
    assert len(reacquired) == 1
    assert reacquired[0].track_id == confirmed.track_id
    assert reacquired[0].age_frames == 5

    tracker.update([], timestamp_ns=5_000_000_000)
    tracker.update([], timestamp_ns=6_000_000_000)
    tracker.update([], timestamp_ns=7_000_000_000)
    assert tracker._tracks == []


def test_reacquired_after_deletion_gets_new_monotonic_id() -> None:
    tracker = SimpleTracker()
    first = confirm(tracker, box(0), box(10))
    for index in range(3):
        tracker.update([], timestamp_ns=(index + 2) * 1_000_000_000)

    assert tracker.update([box(10)], timestamp_ns=5_000_000_000) == ()
    second = tracker.update([box(20)], timestamp_ns=6_000_000_000)[0]

    assert first.track_id == 1
    assert second.track_id == 2


def test_reset_starts_independent_generation_id_space() -> None:
    tracker = SimpleTracker()
    assert confirm(tracker, box(0), box(10)).track_id == 1

    tracker.reset()

    assert confirm(tracker, box(100), box(110)).track_id == 1


def test_one_detection_cannot_update_two_tracks() -> None:
    tracker = SimpleTracker()
    assert tracker.update([box(10), box(50)], timestamp_ns=0) == ()
    confirmed = tracker.update([box(20), box(40)], timestamp_ns=1_000_000_000)
    assert {item.track_id for item in confirmed} == {1, 2}

    result = tracker.update([box(30)], timestamp_ns=2_000_000_000)

    assert len(result) == 1
    misses = sorted(track.misses for track in tracker._tracks)
    assert misses == [0, 1]


def test_nearby_crossing_tracks_are_associated_deterministically() -> None:
    tracker = SimpleTracker()
    tracker.update([box(10), box(90)], timestamp_ns=0)
    second = tracker.update([box(20), box(80)], timestamp_ns=1_000_000_000)
    assert [(item.track_id, item.bbox.x) for item in second] == [(1, 20), (2, 80)]

    third = tracker.update([box(45), box(55)], timestamp_ns=2_000_000_000)
    fourth = tracker.update([box(70), box(30)], timestamp_ns=3_000_000_000)

    assert [(item.track_id, item.bbox.x) for item in third] == [(1, 45), (2, 55)]
    assert [(item.track_id, item.bbox.x) for item in fourth] == [(1, 70), (2, 30)]


@pytest.mark.parametrize(
    "wrong_bbox",
    [
        box(400),
        box(12, width=80, height=80),
    ],
)
def test_distance_and_size_gates_reject_clearly_wrong_association(wrong_bbox: BBox) -> None:
    tracker = SimpleTracker()
    original = confirm(tracker, box(10), box(12))

    assert tracker.update([wrong_bbox], timestamp_ns=2_000_000_000) == ()
    result = tracker.update([wrong_bbox], timestamp_ns=3_000_000_000)

    assert len(result) == 1
    assert result[0].track_id != original.track_id


def test_unmatched_detection_creates_new_tentative_track() -> None:
    tracker = SimpleTracker()
    confirm(tracker, box(0), box(5))

    result = tracker.update([box(5), box(300)], timestamp_ns=2_000_000_000)

    assert len(result) == 1
    assert len(tracker._tracks) == 2
    assert sum(track.confirmed for track in tracker._tracks) == 1
