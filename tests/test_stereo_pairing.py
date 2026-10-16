from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from navmin.contracts import (
    BBox,
    CameraRole,
    FramePacket,
    TargetRef,
    TrackedObject,
    VisionResult,
)
from navmin.vision.stereo_pairing import (
    StereoFramePair,
    StereoPairContractError,
    StereoPairIdentity,
    make_exact_stereo_pair,
)
from navmin.vision.stereo_target_join import (
    StereoPairingKind,
    StereoPairingTargetJoinSink,
    StereoTargetJoiner,
)
from navmin.vision.stereo_timestamp_pairing import (
    TimestampStereoFramePair,
    TimestampStereoPairIdentity,
)


def frame(
    camera: CameraRole,
    *,
    generation: int = 1,
    frame_id: int = 0,
    capture_id: int | None = 42,
    receive_timestamp_ns: int = 1_000,
) -> FramePacket:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image.flags.writeable = False
    return FramePacket(
        camera=camera,
        generation=generation,
        frame_id=frame_id,
        capture_id=capture_id,
        receive_timestamp_ns=receive_timestamp_ns,
        image=image,
    )


def test_valid_exact_pair_uses_capture_id_and_both_generations() -> None:
    left = frame(
        CameraRole.STEREO_LEFT,
        generation=7,
        frame_id=101,
        receive_timestamp_ns=10_000,
    )
    right = frame(
        CameraRole.STEREO_RIGHT,
        generation=11,
        frame_id=9,
        receive_timestamp_ns=99_000,
    )

    pair = make_exact_stereo_pair(left, right)

    assert pair is not None
    assert pair.identity == StereoPairIdentity(
        left_generation=7,
        right_generation=11,
        capture_id=42,
    )
    assert pair.left is left
    assert pair.right is right
    assert pair.left.image is left.image
    assert pair.right.image is right.image


@pytest.mark.parametrize(
    ("left_camera", "right_camera"),
    [
        (CameraRole.OVERVIEW, CameraRole.STEREO_RIGHT),
        (CameraRole.STEREO_RIGHT, CameraRole.STEREO_LEFT),
        (CameraRole.STEREO_LEFT, CameraRole.OVERVIEW),
    ],
)
def test_wrong_camera_roles_are_rejected(
    left_camera: CameraRole,
    right_camera: CameraRole,
) -> None:
    with pytest.raises(StereoPairContractError):
        make_exact_stereo_pair(frame(left_camera), frame(right_camera))


@pytest.mark.parametrize(
    ("left_capture_id", "right_capture_id"),
    [
        (None, 42),
        (42, None),
        (None, None),
    ],
)
def test_none_capture_id_cannot_form_exact_pair(
    left_capture_id: int | None,
    right_capture_id: int | None,
) -> None:
    pair = make_exact_stereo_pair(
        frame(CameraRole.STEREO_LEFT, capture_id=left_capture_id),
        frame(CameraRole.STEREO_RIGHT, capture_id=right_capture_id),
    )

    assert pair is None


def test_different_capture_ids_do_not_pair_even_with_near_timestamps() -> None:
    left = frame(
        CameraRole.STEREO_LEFT,
        capture_id=41,
        receive_timestamp_ns=1_000_000,
    )
    right = frame(
        CameraRole.STEREO_RIGHT,
        capture_id=42,
        receive_timestamp_ns=1_000_001,
    )

    assert make_exact_stereo_pair(left, right) is None


def test_frame_id_and_timestamp_equality_are_not_required_for_exact_pair() -> None:
    left = frame(
        CameraRole.STEREO_LEFT,
        frame_id=1,
        capture_id=55,
        receive_timestamp_ns=100,
    )
    right = frame(
        CameraRole.STEREO_RIGHT,
        frame_id=999,
        capture_id=55,
        receive_timestamp_ns=9_000_000,
    )

    assert make_exact_stereo_pair(left, right) is not None


def test_same_capture_id_across_generations_has_distinct_pair_identity() -> None:
    first = make_exact_stereo_pair(
        frame(CameraRole.STEREO_LEFT, generation=2, capture_id=42),
        frame(CameraRole.STEREO_RIGHT, generation=3, capture_id=42),
    )
    second = make_exact_stereo_pair(
        frame(CameraRole.STEREO_LEFT, generation=4, capture_id=42),
        frame(CameraRole.STEREO_RIGHT, generation=5, capture_id=42),
    )

    assert first is not None
    assert second is not None
    assert first.identity != second.identity


@pytest.mark.parametrize(
    "invalid_pair",
    [
        lambda left, right: StereoFramePair(
            identity=StereoPairIdentity(1, 1, 42),
            left=replace(left, camera=CameraRole.OVERVIEW),
            right=right,
        ),
        lambda left, right: StereoFramePair(
            identity=StereoPairIdentity(1, 1, 42),
            left=left,
            right=replace(right, capture_id=None),
        ),
        lambda left, right: StereoFramePair(
            identity=StereoPairIdentity(1, 1, 42),
            left=left,
            right=replace(right, capture_id=43),
        ),
        lambda left, right: StereoFramePair(
            identity=StereoPairIdentity(1, 1, 99),
            left=left,
            right=right,
        ),
        lambda left, right: StereoFramePair(
            identity=StereoPairIdentity(9, 1, 42),
            left=left,
            right=right,
        ),
        lambda left, right: StereoFramePair(
            identity=StereoPairIdentity(1, 9, 42),
            left=left,
            right=right,
        ),
    ],
)
def test_typed_pair_rejects_invariant_violations(invalid_pair) -> None:
    left = frame(CameraRole.STEREO_LEFT)
    right = frame(CameraRole.STEREO_RIGHT)

    with pytest.raises(StereoPairContractError):
        invalid_pair(left, right)


class FakeClock:
    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


def pairer(
    *,
    clock: FakeClock | None = None,
    capacity: int = 4,
    timeout_ms: int = 100,
):
    from navmin.vision.stereo_pairing import StereoPairer

    return StereoPairer(
        right_frame_buffer_size=capacity,
        pair_timeout_ms=timeout_ms,
        clock_ns=clock if clock is not None else FakeClock(),
    )


def start_pairer_sessions(stereo_pairer, *, left: int = 1, right: int = 1) -> None:
    stereo_pairer.start_generation(CameraRole.STEREO_LEFT, left)
    stereo_pairer.start_generation(CameraRole.STEREO_RIGHT, right)


def test_pairer_rejects_invalid_bounds() -> None:
    from navmin.vision.stereo_pairing import StereoPairer

    with pytest.raises(ValueError, match="right_frame_buffer_size"):
        StereoPairer(right_frame_buffer_size=0, pair_timeout_ms=100)
    with pytest.raises(ValueError, match="pair_timeout_ms"):
        StereoPairer(right_frame_buffer_size=4, pair_timeout_ms=0)


def test_pairer_rejects_non_stereo_session_and_frame_inputs() -> None:
    stereo_pairer = pairer()

    with pytest.raises(StereoPairContractError):
        stereo_pairer.start_generation(CameraRole.OVERVIEW, 1)
    with pytest.raises(StereoPairContractError):
        stereo_pairer.offer_frame(frame(CameraRole.OVERVIEW))


def test_right_before_left_pairs_exactly() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=7, right=9)
    right = frame(CameraRole.STEREO_RIGHT, generation=9, capture_id=42)
    left = frame(CameraRole.STEREO_LEFT, generation=7, capture_id=42)

    assert stereo_pairer.offer_frame(right) is None
    paired = stereo_pairer.offer_frame(left)

    assert paired is not None
    assert paired.left is left
    assert paired.right is right
    assert paired.identity == StereoPairIdentity(7, 9, 42)
    assert stereo_pairer.latest_pair is paired
    assert stereo_pairer.state.right_buffer_size == 0


def test_left_before_right_pairs_within_timeout() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=100)
    start_pairer_sessions(stereo_pairer)
    left = frame(CameraRole.STEREO_LEFT, capture_id=12)
    right = frame(CameraRole.STEREO_RIGHT, capture_id=12)

    assert stereo_pairer.offer_frame(left) is None
    clock.advance_ms(99)
    paired = stereo_pairer.offer_frame(right)

    assert paired is not None
    assert paired.left is left
    assert paired.right is right


def test_pair_timeout_is_attempt_lifetime_not_frame_timestamp_delta() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=100)
    start_pairer_sessions(stereo_pairer)
    right = frame(
        CameraRole.STEREO_RIGHT,
        capture_id=90,
        receive_timestamp_ns=9_000_000_000,
    )
    left = frame(
        CameraRole.STEREO_LEFT,
        capture_id=90,
        receive_timestamp_ns=10,
    )

    stereo_pairer.offer_frame(right)
    clock.advance_ms(10)

    assert stereo_pairer.offer_frame(left) is not None


def test_left_attempt_expires_at_timeout_and_late_right_does_not_revive_it() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=100)
    start_pairer_sessions(stereo_pairer)

    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=5))
    clock.advance_ms(100)
    result = stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=5))

    assert result is None
    assert stereo_pairer.latest_pair is None
    assert stereo_pairer.state.pending_left_capture_id is None
    assert stereo_pairer.state.right_buffer_size == 1


def test_unmatched_right_expires_and_cannot_pair_with_later_left() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=100)
    start_pairer_sessions(stereo_pairer)

    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=5))
    clock.advance_ms(100)
    result = stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=5))

    assert result is None
    assert stereo_pairer.state.right_buffer_size == 0
    assert stereo_pairer.state.pending_left_capture_id == 5


def test_expire_clears_unmatched_state_without_sleep() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=25)
    start_pairer_sessions(stereo_pairer)
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=1))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=2))

    clock.advance_ms(25)
    stereo_pairer.expire()

    assert stereo_pairer.state.right_buffer_size == 0
    assert stereo_pairer.state.pending_left_capture_id is None


def test_right_buffer_is_bounded_and_evicts_oldest_arrival() -> None:
    stereo_pairer = pairer(capacity=2)
    start_pairer_sessions(stereo_pairer)

    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=30))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=10))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=20))

    assert stereo_pairer.state.right_buffer_size == 2
    assert stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=30)) is None
    assert stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=10)) is not None


def test_out_of_order_right_capture_ids_match_by_exact_id_not_order() -> None:
    stereo_pairer = pairer(capacity=4)
    start_pairer_sessions(stereo_pairer)

    for capture_id in (100, 3, 77):
        stereo_pairer.offer_frame(
            frame(CameraRole.STEREO_RIGHT, capture_id=capture_id)
        )

    paired = stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, capture_id=3)
    )

    assert paired is not None
    assert paired.identity.capture_id == 3
    assert stereo_pairer.state.right_buffer_size == 2


def test_pending_left_is_latest_only_and_strictly_bounded_to_one() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer)

    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=1))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=2))

    assert stereo_pairer.state.pending_left_capture_id == 2
    assert stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=1)) is None
    paired = stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=2))
    assert paired is not None
    assert paired.identity.capture_id == 2


def test_duplicate_left_does_not_extend_pending_timeout() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=100)
    start_pairer_sessions(stereo_pairer)
    original = frame(CameraRole.STEREO_LEFT, frame_id=1, capture_id=8)
    duplicate = frame(CameraRole.STEREO_LEFT, frame_id=2, capture_id=8)

    stereo_pairer.offer_frame(original)
    clock.advance_ms(90)
    assert stereo_pairer.offer_frame(duplicate) is None
    clock.advance_ms(10)

    assert stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=8)) is None
    assert stereo_pairer.latest_pair is None


def test_duplicate_right_does_not_extend_buffer_timeout() -> None:
    clock = FakeClock()
    stereo_pairer = pairer(clock=clock, timeout_ms=100)
    start_pairer_sessions(stereo_pairer)
    original = frame(CameraRole.STEREO_RIGHT, frame_id=1, capture_id=8)
    duplicate = frame(CameraRole.STEREO_RIGHT, frame_id=2, capture_id=8)

    stereo_pairer.offer_frame(original)
    clock.advance_ms(90)
    assert stereo_pairer.offer_frame(duplicate) is None
    clock.advance_ms(10)

    assert stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=8)) is None
    assert stereo_pairer.latest_pair is None


def test_immediate_duplicate_inputs_do_not_republish_latest_identity() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=4, right=6)
    left = frame(CameraRole.STEREO_LEFT, generation=4, capture_id=99)
    right = frame(CameraRole.STEREO_RIGHT, generation=6, capture_id=99)

    first = stereo_pairer.offer_frame(left)
    assert first is None
    first = stereo_pairer.offer_frame(right)
    assert first is not None

    assert stereo_pairer.offer_frame(replace(left, frame_id=10)) is None
    assert stereo_pairer.offer_frame(replace(right, frame_id=10)) is None
    assert stereo_pairer.latest_pair is first


def test_none_capture_id_is_not_buffered_or_pending() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer)

    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, capture_id=None)
    ) is None
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_RIGHT, capture_id=None)
    ) is None

    assert stereo_pairer.state.pending_left_capture_id is None
    assert stereo_pairer.state.right_buffer_size == 0


def test_wrong_generation_frames_are_ignored() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=10, right=20)

    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, generation=9, capture_id=1)
    ) is None
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_RIGHT, generation=19, capture_id=1)
    ) is None

    assert stereo_pairer.state.pending_left_capture_id is None
    assert stereo_pairer.state.right_buffer_size == 0


def test_new_left_generation_invalidates_pending_right_buffer_and_latest_pair() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=1, right=1)
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, generation=1, capture_id=1))
    first = stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_RIGHT, generation=1, capture_id=1)
    )
    assert first is not None
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, generation=1, capture_id=2))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, generation=1, capture_id=3))

    stereo_pairer.start_generation(CameraRole.STEREO_LEFT, 2)

    state = stereo_pairer.state
    assert state.left_generation == 2
    assert state.right_generation == 1
    assert state.right_buffer_size == 0
    assert state.pending_left_capture_id is None
    assert stereo_pairer.latest_pair is None
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, generation=2, capture_id=2)
    ) is None


def test_new_right_generation_invalidates_pending_buffer_and_latest_pair() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=1, right=1)
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, generation=1, capture_id=1))
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, generation=1, capture_id=1)
    ) is not None
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, generation=1, capture_id=2))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, generation=1, capture_id=3))

    stereo_pairer.start_generation(CameraRole.STEREO_RIGHT, 2)

    state = stereo_pairer.state
    assert state.left_generation == 1
    assert state.right_generation == 2
    assert state.right_buffer_size == 0
    assert state.pending_left_capture_id is None
    assert stereo_pairer.latest_pair is None


def test_same_capture_id_after_generation_change_never_cross_pairs_old_state() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=1, right=1)
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, generation=1, capture_id=42))

    stereo_pairer.start_generation(CameraRole.STEREO_LEFT, 2)
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, generation=2, capture_id=42)
    ) is None

    stereo_pairer.start_generation(CameraRole.STEREO_RIGHT, 2)
    assert stereo_pairer.latest_pair is None
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_RIGHT, generation=2, capture_id=42)
    ) is None
    paired = stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, generation=2, capture_id=42)
    )
    assert paired is not None
    assert paired.identity == StereoPairIdentity(2, 2, 42)


def test_reannouncing_same_generation_is_idempotent() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=3, right=4)
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, generation=4, capture_id=7))

    stereo_pairer.start_generation(CameraRole.STEREO_RIGHT, 4)

    assert stereo_pairer.state.right_buffer_size == 1
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, generation=3, capture_id=7)
    ) is not None


def test_new_valid_pair_replaces_old_latest_pair() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer)

    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=1))
    first = stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=1))
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_RIGHT, capture_id=2))
    second = stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=2))

    assert first is not None
    assert second is not None
    assert second.identity.capture_id == 2
    assert stereo_pairer.latest_pair is second


def test_reset_invalidates_latest_and_requires_fresh_session_boundaries() -> None:
    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer)
    stereo_pairer.offer_frame(frame(CameraRole.STEREO_LEFT, capture_id=1))
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_RIGHT, capture_id=1)
    ) is not None

    stereo_pairer.reset()

    assert stereo_pairer.latest_pair is None
    assert stereo_pairer.state.left_generation is None
    assert stereo_pairer.state.right_generation is None
    assert stereo_pairer.offer_frame(
        frame(CameraRole.STEREO_LEFT, capture_id=2)
    ) is None


def test_capture_id_order_and_wrap_are_not_used_for_pairing() -> None:
    stereo_pairer = pairer(capacity=4)
    start_pairer_sessions(stereo_pairer)

    for capture_id in (2**32 - 1, 0, 17):
        stereo_pairer.offer_frame(
            frame(CameraRole.STEREO_RIGHT, capture_id=capture_id)
        )

    for capture_id in (0, 2**32 - 1, 17):
        paired = stereo_pairer.offer_frame(
            frame(CameraRole.STEREO_LEFT, capture_id=capture_id)
        )
        assert paired is not None
        assert paired.identity.capture_id == capture_id


def test_pairer_is_thread_safe_for_simultaneous_left_and_right_arrival() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    stereo_pairer = pairer()
    start_pairer_sessions(stereo_pairer, left=8, right=9)
    barrier = Barrier(2)
    left = frame(CameraRole.STEREO_LEFT, generation=8, capture_id=123)
    right = frame(CameraRole.STEREO_RIGHT, generation=9, capture_id=123)

    def offer(packet: FramePacket):
        barrier.wait()
        return stereo_pairer.offer_frame(packet)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(offer, (left, right)))

    pairs = [result for result in results if result is not None]
    assert len(pairs) == 1
    assert pairs[0].identity == StereoPairIdentity(8, 9, 123)
    assert stereo_pairer.latest_pair is pairs[0]


def _vision_result_for(
    packet: FramePacket,
    *,
    track_ids: tuple[int, ...] = (7,),
) -> VisionResult:
    tracked = tuple(
        TrackedObject(
            track_id=track_id,
            bbox=BBox(10 + index * 5, 11, 12, 13),
            velocity_x_px_s=0.0,
            velocity_y_px_s=0.0,
            age_frames=3,
        )
        for index, track_id in enumerate(track_ids)
    )
    return VisionResult(frame=packet, tracked_objects=tracked, processing_time_ns=5)


def _exact_pair_for_left(left: FramePacket, *, right_generation: int = 2) -> StereoFramePair:
    pair = make_exact_stereo_pair(
        left,
        frame(
            CameraRole.STEREO_RIGHT,
            generation=right_generation,
            frame_id=99,
            capture_id=left.capture_id,
        ),
    )
    assert pair is not None
    return pair


def _active_joiner(*, generation: int = 1, track_id: int = 7) -> StereoTargetJoiner:
    joiner = StereoTargetJoiner()
    joiner.start_generation(generation)
    joiner.set_target(TargetRef(CameraRole.STEREO_LEFT, generation, track_id))
    return joiner


def test_target_join_accepts_pair_then_exact_result_for_selected_track() -> None:
    joiner = _active_joiner(generation=3, track_id=8)
    left = frame(
        CameraRole.STEREO_LEFT,
        generation=3,
        frame_id=12,
        capture_id=44,
    )
    pair = _exact_pair_for_left(left, right_generation=9)

    assert joiner.offer_pair(pair) is None
    observation = joiner.offer_vision_result(
        _vision_result_for(left, track_ids=(4, 8, 12))
    )

    assert observation is not None
    assert observation.target == TargetRef(CameraRole.STEREO_LEFT, 3, 8)
    assert observation.pair is pair
    assert observation.roi == BBox(15, 11, 12, 13)
    assert joiner.observations.get() is observation
    assert joiner.state.pending_pair_frame_id is None
    assert joiner.state.pending_result_frame_id is None


def test_target_join_accepts_result_then_pair_without_timestamp_fallback() -> None:
    joiner = _active_joiner(generation=2)
    left = frame(
        CameraRole.STEREO_LEFT,
        generation=2,
        frame_id=5,
        capture_id=70,
        receive_timestamp_ns=10,
    )
    pair = _exact_pair_for_left(left)

    assert joiner.offer_vision_result(_vision_result_for(left)) is None
    observation = joiner.offer_pair(pair)

    assert observation is not None
    assert observation.pair.left.frame_id == 5
    assert observation.pair.identity.capture_id == 70

    mismatched = frame(
        CameraRole.STEREO_LEFT,
        generation=2,
        frame_id=6,
        capture_id=70,
        receive_timestamp_ns=11,
    )
    published_revision = joiner.observations.snapshot().revision
    assert joiner.offer_vision_result(_vision_result_for(mismatched)) is None
    assert joiner.offer_pair(pair) is None
    assert joiner.observations.snapshot().revision == published_revision
    assert joiner.observations.get() is observation


def test_target_join_requires_exact_capture_identity_and_selected_track() -> None:
    joiner = _active_joiner()
    left = frame(CameraRole.STEREO_LEFT, frame_id=10, capture_id=20)
    pair = _exact_pair_for_left(left)
    wrong_capture_result = _vision_result_for(replace(left, capture_id=21))

    joiner.offer_pair(pair)
    assert joiner.offer_vision_result(wrong_capture_result) is None
    assert joiner.observations.get() is None

    joiner.offer_pair(pair)
    assert joiner.offer_vision_result(_vision_result_for(left, track_ids=(99,))) is None
    assert joiner.observations.get() is None


def test_target_join_state_is_single_slot_latest_useful_and_generation_aware() -> None:
    joiner = _active_joiner(generation=4)
    first = frame(CameraRole.STEREO_LEFT, generation=4, frame_id=1, capture_id=1)
    second = frame(CameraRole.STEREO_LEFT, generation=4, frame_id=2, capture_id=2)

    joiner.offer_vision_result(_vision_result_for(first))
    joiner.offer_vision_result(_vision_result_for(second))
    assert joiner.state.pending_result_frame_id == 2

    joiner.offer_pair(_exact_pair_for_left(first))
    assert joiner.state.pending_pair_frame_id is None
    assert joiner.state.pending_result_frame_id == 2

    observation = joiner.offer_pair(_exact_pair_for_left(second))
    assert observation is not None
    assert observation.pair.left.frame_id == 2

    third = frame(CameraRole.STEREO_LEFT, generation=4, frame_id=3, capture_id=3)
    fourth = frame(CameraRole.STEREO_LEFT, generation=4, frame_id=4, capture_id=4)
    joiner.offer_pair(_exact_pair_for_left(third))
    joiner.offer_pair(_exact_pair_for_left(fourth))
    assert joiner.state.pending_pair_frame_id == 4
    joiner.offer_vision_result(_vision_result_for(third))
    assert joiner.state.pending_pair_frame_id == 4
    assert joiner.state.pending_result_frame_id is None

    joiner.start_generation(5)
    assert joiner.state.left_generation == 5
    assert joiner.state.target is None
    assert joiner.state.pending_pair_frame_id is None
    assert joiner.state.pending_result_frame_id is None
    assert joiner.observations.get() is None


def test_target_change_or_disable_clears_pending_and_published_join_state() -> None:
    joiner = _active_joiner()
    left = frame(CameraRole.STEREO_LEFT, frame_id=4, capture_id=4)
    observation = joiner.offer_pair(_exact_pair_for_left(left))
    assert observation is None
    observation = joiner.offer_vision_result(_vision_result_for(left))
    assert observation is not None

    joiner.set_target(None)
    assert joiner.state.target is None
    assert joiner.observations.get() is None

    joiner.set_target(TargetRef(CameraRole.STEREO_LEFT, 1, 8))
    joiner.offer_pair(_exact_pair_for_left(left))
    joiner.set_target(TargetRef(CameraRole.STEREO_LEFT, 1, 9))
    assert joiner.state.pending_pair_frame_id is None
    assert joiner.observations.get() is None


def test_pairing_target_join_sink_forwards_only_new_exact_pairs() -> None:
    stereo_pairer = pairer()
    joiner = _active_joiner(generation=6)
    sink = StereoPairingTargetJoinSink(stereo_pairer, joiner)
    stereo_pairer.start_generation(CameraRole.STEREO_LEFT, 6)
    stereo_pairer.start_generation(CameraRole.STEREO_RIGHT, 7)
    left = frame(
        CameraRole.STEREO_LEFT,
        generation=6,
        frame_id=20,
        capture_id=55,
    )
    right = frame(
        CameraRole.STEREO_RIGHT,
        generation=7,
        frame_id=30,
        capture_id=55,
    )

    sink.on_corrected_frame(left)
    assert joiner.offer_vision_result(_vision_result_for(left)) is None
    sink.on_corrected_frame(right)

    observation = joiner.observations.get()
    assert observation is not None
    assert observation.pair.identity == StereoPairIdentity(6, 7, 55)


def _timestamp_pair_for_left(left: FramePacket) -> TimestampStereoFramePair:
    right = frame(
        CameraRole.STEREO_RIGHT,
        generation=8,
        frame_id=29,
        capture_id=None,
        receive_timestamp_ns=left.receive_timestamp_ns + 12_000_000,
    )
    return TimestampStereoFramePair(
        identity=TimestampStereoPairIdentity(
            left.generation,
            right.generation,
            left.frame_id,
            right.frame_id,
            left.receive_timestamp_ns,
            right.receive_timestamp_ns,
            12_000_000,
        ),
        left=left,
        right=right,
    )


@pytest.mark.parametrize("result_first", [False, True])
def test_timestamp_target_join_only_matches_same_left_observation(result_first: bool) -> None:
    joiner = _active_joiner(generation=3, track_id=7)
    left = frame(
        CameraRole.STEREO_LEFT,
        generation=3,
        frame_id=5,
        capture_id=None,
        receive_timestamp_ns=90_000_000,
    )
    pair = _timestamp_pair_for_left(left)
    result = _vision_result_for(left, track_ids=(7, 8))
    if result_first:
        assert joiner.offer_vision_result(result) is None
        observation = joiner.offer_pair(pair)
    else:
        assert joiner.offer_pair(pair) is None
        observation = joiner.offer_vision_result(result)
    assert observation is not None
    assert observation.pairing_kind is StereoPairingKind.TIMESTAMP
    assert observation.pair is pair
    assert observation.target.track_id == 7


@pytest.mark.parametrize("change", ["frame_id", "generation", "receive_timestamp_ns", "track"])
def test_timestamp_join_does_not_guess_left_target(change: str) -> None:
    joiner = _active_joiner(generation=3)
    left = frame(
        CameraRole.STEREO_LEFT,
        generation=3,
        frame_id=5,
        capture_id=None,
        receive_timestamp_ns=90_000_000,
    )
    result_left = left
    tracks = (7,)
    if change == "frame_id":
        result_left = replace(left, frame_id=6)
    elif change == "generation":
        result_left = replace(left, generation=4)
    elif change == "receive_timestamp_ns":
        result_left = replace(left, receive_timestamp_ns=90_000_001)
    else:
        tracks = (22,)
    assert joiner.offer_pair(_timestamp_pair_for_left(left)) is None
    assert joiner.offer_vision_result(_vision_result_for(result_left, track_ids=tracks)) is None
    assert joiner.observations.get() is None


def test_production_stereo_sink_routes_only_selected_pairing_kind() -> None:
    """Timestamp frames with no capture ID cannot become exact distance input."""
    from navmin.vision.stereo_pairing import StereoPairer
    from navmin.vision.stereo_timestamp_pairing import TimestampStereoPairer

    class RecordingJoiner:
        def __init__(self) -> None:
            self.pairs: list[StereoFramePair | TimestampStereoFramePair] = []

        def offer_pair(self, pair: StereoFramePair | TimestampStereoFramePair) -> None:
            self.pairs.append(pair)

    exact = StereoPairer(right_frame_buffer_size=3, pair_timeout_ms=200)
    timed = TimestampStereoPairer(right_frame_buffer_size=3, pair_timeout_ms=200)
    joiner = RecordingJoiner()
    selected: list[StereoPairingKind | None] = [StereoPairingKind.EXACT]
    sink = StereoPairingTargetJoinSink(
        exact, joiner, timestamp_pairer=timed, active_pairing_kind=lambda: selected[0]
    )
    # Camera sessions are injected here without creating hardware/Qt workers.
    for camera in (CameraRole.STEREO_LEFT, CameraRole.STEREO_RIGHT):
        exact.start_generation(camera, 1)
        timed.start_generation(camera, 1)
        sink._generations[camera] = 1

    sink.on_corrected_frame(frame(CameraRole.STEREO_RIGHT, frame_id=1, capture_id=None))
    sink.on_corrected_frame(frame(CameraRole.STEREO_LEFT, frame_id=1, capture_id=None))
    assert not joiner.pairs

    selected[0] = StereoPairingKind.TIMESTAMP
    sink.on_corrected_frame(
        frame(CameraRole.STEREO_RIGHT, frame_id=2, capture_id=None, receive_timestamp_ns=10_000)
    )
    sink.on_corrected_frame(
        frame(CameraRole.STEREO_LEFT, frame_id=2, capture_id=None, receive_timestamp_ns=10_010)
    )
    assert len(joiner.pairs) == 1
    assert isinstance(joiner.pairs[0], TimestampStereoFramePair)
    assert joiner.pairs[0].left.capture_id is None

    selected[0] = StereoPairingKind.EXACT
    sink.on_corrected_frame(
        frame(CameraRole.STEREO_RIGHT, frame_id=3, capture_id=99, receive_timestamp_ns=15_000)
    )
    sink.on_corrected_frame(
        frame(CameraRole.STEREO_LEFT, frame_id=3, capture_id=99, receive_timestamp_ns=999_000)
    )
    assert len(joiner.pairs) == 2
    assert isinstance(joiner.pairs[1], StereoFramePair)
    assert joiner.pairs[1].identity.capture_id == 99

    selected[0] = None
    sink.on_corrected_frame(frame(CameraRole.STEREO_RIGHT, frame_id=4, capture_id=100))
    sink.on_corrected_frame(frame(CameraRole.STEREO_LEFT, frame_id=4, capture_id=100))
    assert len(joiner.pairs) == 2
