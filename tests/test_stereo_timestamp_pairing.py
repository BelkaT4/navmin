"""Deterministic compatibility pairing contracts; no camera or time sleeps."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_stereo_pairing import FakeClock, frame

from navmin.contracts import CameraRole
from navmin.vision.stereo_pairing import StereoPairContractError, make_exact_stereo_pair
from navmin.vision.stereo_timestamp_pairing import (
    TimestampStereoFramePair,
    TimestampStereoPairer,
)

LEFT = CameraRole.STEREO_LEFT
RIGHT = CameraRole.STEREO_RIGHT
MS = 1_000_000


def pairer(*, clock=None, capacity=3, timeout=100, skew=30):
    result = TimestampStereoPairer(
        right_frame_buffer_size=capacity, pair_timeout_ms=timeout,
        max_timestamp_skew_ms=skew, clock_ns=clock or FakeClock(),
    )
    result.start_generation(LEFT, 1)
    result.start_generation(RIGHT, 1)
    return result


def f(camera, index, time_ms, *, generation=1, capture_id=None):
    return frame(camera, frame_id=index, generation=generation,
                 capture_id=capture_id, receive_timestamp_ns=time_ms * MS)


def test_both_arrival_orders_and_exact_remains_strict():
    p = pairer()
    assert p.offer_frame(f(RIGHT, 1, 101)) is None
    pair = p.offer_frame(f(LEFT, 1, 100))
    assert pair is not None and pair.identity.timestamp_skew_ns == MS
    assert pair.left.capture_id is None and pair.right.capture_id is None
    assert make_exact_stereo_pair(pair.left, pair.right) is None
    assert p.offer_frame(f(LEFT, 2, 200)) is None
    second = p.offer_frame(f(RIGHT, 2, 201))
    assert second is not None and second.identity.left_frame_id == 2
    assert second.identity.right_frame_id == 2
    assert p.state.latest_skew_ns == MS


def test_nearest_boundary_and_tie_break():
    p = pairer(skew=30)
    for index, ts in enumerate((130, 90, 110), 1):
        assert p.offer_frame(f(RIGHT, index, ts)) is None
    pair = p.offer_frame(f(LEFT, 1, 100))
    assert pair is not None and pair.identity.right_frame_id == 2  # 90 beats 110.
    assert pair.identity.timestamp_skew_ns == 10 * MS
    boundary = p.offer_frame(f(LEFT, 2, 160))
    assert boundary is not None and boundary.identity.timestamp_skew_ns == 30 * MS
    assert p.offer_frame(f(LEFT, 3, 300)) is None
    assert p.offer_frame(f(RIGHT, 4, 260)) is None  # outside skew
    assert p.offer_frame(f(RIGHT, 5, 330)) is not None  # boundary +30 ms



def test_left_prunes_only_right_frames_that_are_too_old_by_timestamp():
    p = pairer(capacity=5, timeout=1_000, skew=30)
    assert p.offer_frame(f(RIGHT, 1, 50)) is None
    assert p.offer_frame(f(RIGHT, 2, 80)) is None
    assert p.offer_frame(f(RIGHT, 3, 160)) is None

    # For Left=120 ms, 50/80 ms can never satisfy the 30 ms threshold.
    # The 160 ms frame is too far in the future for this Left, but may still
    # be useful for a later Left and therefore must remain buffered.
    assert p.offer_frame(f(LEFT, 1, 120)) is None
    assert p.state.right_buffer_size == 1

    pair = p.offer_frame(f(LEFT, 2, 150))
    assert pair is not None
    assert pair.identity.right_frame_id == 3
    assert pair.identity.timestamp_skew_ns == 10 * MS
    assert p.state.right_buffer_size == 0

def test_right_not_reused_and_pending_left_latest_only():
    p = pairer()
    p.offer_frame(f(RIGHT, 1, 100))
    assert p.offer_frame(f(LEFT, 1, 100)) is not None
    assert p.offer_frame(f(LEFT, 2, 200)) is None
    assert p.state.pending_left_frame_id == 2
    assert p.offer_frame(f(LEFT, 3, 300)) is None
    assert p.state.pending_left_frame_id == 3
    assert p.offer_frame(f(RIGHT, 2, 200)) is None
    assert p.offer_frame(f(RIGHT, 3, 300)) is not None
    assert p.offer_frame(f(LEFT, 4, 300)) is None


def test_bounds_timeout_and_generation_barriers():
    clock = FakeClock()
    p = pairer(clock=clock, capacity=2, timeout=10)
    for i in range(3):
        p.offer_frame(f(RIGHT, i, i * 100))
    assert p.state.right_buffer_size == 2
    p.offer_frame(f(LEFT, 1, 500))
    assert p.state.pending_left_frame_id == 1
    clock.advance_ms(10)
    p.expire()
    assert p.state.pending_left_frame_id is None
    assert p.state.right_buffer_size == 0
    p.offer_frame(f(RIGHT, 3, 600))
    assert p.offer_frame(f(LEFT, 2, 600)) is not None
    p.start_generation(RIGHT, 2)
    assert p.state.latest_pair_identity is None
    assert p.state.right_buffer_size == 0
    assert p.offer_frame(f(LEFT, 3, 600)) is None
    assert p.offer_frame(f(RIGHT, 0, 600, generation=1)) is None
    assert p.offer_frame(f(RIGHT, 0, 600, generation=2)) is not None
    p.start_generation(LEFT, 2)
    assert p.latest_pair is None


def test_contract_rejections_and_no_fake_capture_id():
    p = pairer()
    with pytest.raises(StereoPairContractError):
        p.offer_frame(f(CameraRole.OVERVIEW, 1, 100))
    with pytest.raises(StereoPairContractError):
        p.start_generation(CameraRole.OVERVIEW, 1)
    with pytest.raises(ValueError, match='max_timestamp_skew_ms'):
        pairer(skew=-1)
    with pytest.raises(ValueError, match='right_frame_buffer_size'):
        pairer(capacity=0)
    with pytest.raises(ValueError, match='pair_timeout_ms'):
        pairer(timeout=0)
    right = f(RIGHT, 2, 101, capture_id=23)
    left = f(LEFT, 1, 100, capture_id=22)
    p.offer_frame(right)
    result = p.offer_frame(left)
    assert result is not None
    assert not hasattr(result.identity, 'capture_id')
    with pytest.raises(StereoPairContractError):
        TimestampStereoFramePair(replace(result.identity, timestamp_skew_ns=0), left, right)
    with pytest.raises(StereoPairContractError):
        TimestampStereoFramePair(result.identity, right, left)


def test_exact_equal_timestamps_without_capture_ids_still_rejected():
    assert make_exact_stereo_pair(f(LEFT, 1, 100), f(RIGHT, 1, 100)) is None
    assert make_exact_stereo_pair(f(LEFT, 1, 100, capture_id=8),
                                  f(RIGHT, 1, 100, capture_id=9)) is None
    assert make_exact_stereo_pair(f(LEFT, 1, 100, capture_id=8),
                                  f(RIGHT, 1, 800, capture_id=8)) is not None


def test_timestamp_diagnostics_count_bounds_timeouts_and_skew_rejections():
    clock = FakeClock()
    p = pairer(clock=clock, capacity=1, timeout=10, skew=30)
    p.offer_frame(f(LEFT, 1, 100))
    p.offer_frame(f(LEFT, 2, 200))
    p.offer_frame(f(RIGHT, 1, 100))
    p.offer_frame(f(RIGHT, 2, 400))
    p.offer_frame(f(LEFT, 3, 350))
    clock.advance_ms(10)
    p.expire()

    diagnostics = p.diagnostics
    assert diagnostics.left_frames_seen == 3
    assert diagnostics.right_frames_seen == 2
    assert diagnostics.timestamp_pairs_emitted == 0
    assert diagnostics.pending_left_superseded == 2
    assert diagnostics.pair_attempts_expired == 1
    assert diagnostics.right_frames_evicted_by_capacity == 1
    assert diagnostics.right_frames_timed_out == 1
    assert diagnostics.skew_rejected_candidates == 3


def test_timestamp_skew_statistics_use_fixed_recent_window_and_nearest_rank():
    p = TimestampStereoPairer(
        right_frame_buffer_size=2,
        pair_timeout_ms=100,
        max_timestamp_skew_ms=30,
        clock_ns=FakeClock(),
        diagnostic_window_size=4,
    )
    p.start_generation(LEFT, 1)
    p.start_generation(RIGHT, 1)
    for index, skew_ms in enumerate((1, 2, 3, 4, 20), 1):
        p.offer_frame(f(RIGHT, index, index * 100 + skew_ms))
        assert p.offer_frame(f(LEFT, index, index * 100)) is not None

    diagnostics = p.diagnostics
    assert diagnostics.timestamp_pairs_emitted == 5
    assert diagnostics.skew.sample_count == 5
    assert diagnostics.skew.retained_sample_count == 4
    assert diagnostics.skew.latest_ns == 20 * MS
    assert diagnostics.skew.minimum_ns == 1 * MS
    assert diagnostics.skew.maximum_ns == 20 * MS
    assert diagnostics.skew.p50_ns == 3 * MS
    assert diagnostics.skew.p95_ns == 20 * MS
