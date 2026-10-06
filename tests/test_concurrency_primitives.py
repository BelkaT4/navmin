from __future__ import annotations

from dataclasses import dataclass
from queue import Empty
from threading import Thread

import pytest

from navmin.concurrency import (
    CameraSessionBarrierChannel,
    InvalidatableLatest,
    LatestValue,
    TargetControlChannel,
    TargetLifecycleEventChannel,
    TargetLifecycleEventChannelOverflowError,
)
from navmin.contracts import (
    CameraRay,
    CameraRole,
    CameraSessionStarted,
    ClearTargetSelection,
    TargetAcquired,
    TargetAcquisitionFailed,
    TargetAcquisitionFailureReason,
    TargetRef,
    TargetSelectionRequest,
    TargetTrackingEnded,
    TargetTrackingEndReason,
)


@dataclass(frozen=True)
class _Pair:
    left: int
    right: int


@dataclass(frozen=True)
class _CameraModel:
    def pixel_to_ray(self, x_px: float, y_px: float) -> CameraRay:
        return CameraRay(x_px, y_px, 1.0)


def test_latest_store_keeps_only_newest_value() -> None:
    store: LatestValue[int] = LatestValue()

    for value in range(100):
        store.publish(value)

    snapshot = store.snapshot()
    assert snapshot.value == 99
    assert snapshot.revision == 100


def test_concurrent_publish_preserves_atomic_state_and_revision() -> None:
    store: LatestValue[_Pair] = LatestValue()
    publish_count = 1_000
    worker_count = 4

    def publish(worker_index: int) -> None:
        base = worker_index * publish_count
        for offset in range(publish_count):
            value = base + offset
            store.publish(_Pair(value, value))

    workers = [Thread(target=publish, args=(index,)) for index in range(worker_count)]
    for worker in workers:
        worker.start()

    while any(worker.is_alive() for worker in workers):
        value = store.get()
        if value is not None:
            assert value.left == value.right

    for worker in workers:
        worker.join()

    snapshot = store.snapshot()
    assert snapshot.revision == publish_count * worker_count
    assert snapshot.value is not None
    assert snapshot.value.left == snapshot.value.right


def test_invalidation_is_observable_and_advances_revision() -> None:
    store: InvalidatableLatest[str] = InvalidatableLatest()

    store.publish("target")
    published = store.snapshot()
    store.invalidate()
    invalidated = store.snapshot()
    store.clear()
    invalidated_again = store.snapshot()

    assert published.value == "target"
    assert invalidated.value is None
    assert invalidated.revision == published.revision + 1
    assert invalidated_again.value is None
    assert invalidated_again.revision == invalidated.revision + 1


def test_wait_for_revision_observes_latest_without_replaying_intermediate_values() -> None:
    store: LatestValue[int] = LatestValue()
    initial_revision = store.snapshot().revision

    store.publish(1)
    store.publish(2)
    store.publish(3)

    observed = store.wait_for_revision(initial_revision, timeout=0.1)
    assert observed is not None
    assert observed.value == 3
    assert observed.revision == 3


def test_camera_session_barriers_are_ordered_while_data_can_coalesce() -> None:
    barriers = CameraSessionBarrierChannel()
    latest_target: LatestValue[TargetRef] = LatestValue()
    model = _CameraModel()

    for generation in (1, 2):
        barriers.publish(
            CameraSessionStarted(
                camera=CameraRole.OVERVIEW,
                generation=generation,
                camera_model=model,
                timestamp_ns=generation,
            )
        )
        latest_target.publish(TargetRef(CameraRole.OVERVIEW, generation, track_id=7))

    assert barriers.receive_nowait().generation == 1
    assert barriers.receive_nowait().generation == 2
    with pytest.raises(Empty):
        barriers.receive_nowait()

    target = latest_target.get()
    assert target is not None
    assert target.generation == 2


def _selection(selection_id: int, *, generation: int = 1) -> TargetSelectionRequest:
    return TargetSelectionRequest(
        selection_id=selection_id,
        camera=CameraRole.OVERVIEW,
        generation=generation,
        frame_id=selection_id,
        x_px=10.0,
        y_px=20.0,
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_selection(1), _selection(2)),
        (ClearTargetSelection(CameraRole.OVERVIEW, 1), _selection(2)),
        (_selection(1), ClearTargetSelection(CameraRole.OVERVIEW, 1)),
    ],
)
def test_target_control_channel_consumes_only_latest_intent(first, second) -> None:
    channel = TargetControlChannel(CameraRole.OVERVIEW)

    channel.publish(first)
    channel.publish(second)

    assert channel.consume_latest(generation=1) == second
    assert channel.consume_latest(generation=1) is None


def test_target_control_channel_rejects_old_generation_and_discarded_intent() -> None:
    channel = TargetControlChannel(CameraRole.OVERVIEW)
    channel.publish(_selection(1, generation=1))

    assert channel.consume_latest(generation=2) is None

    current = _selection(2, generation=2)
    channel.publish(current)
    channel.discard_pending()
    assert channel.consume_latest(generation=2) is None

    channel.publish(current)
    assert channel.consume_latest(generation=2) == current


def test_target_control_channel_rejects_another_camera() -> None:
    channel = TargetControlChannel(CameraRole.OVERVIEW)

    with pytest.raises(ValueError):
        channel.publish(ClearTargetSelection(CameraRole.STEREO_LEFT, 1))


def test_target_lifecycle_channel_preserves_fifo_order() -> None:
    channel = TargetLifecycleEventChannel(capacity=3)
    target = TargetRef(CameraRole.OVERVIEW, generation=1, track_id=8)
    events = (
        TargetAcquired(1, target, 5, 5, 100),
        TargetAcquisitionFailed(
            2,
            CameraRole.OVERVIEW,
            1,
            6,
            TargetAcquisitionFailureReason.NO_TARGET_AT_POINT,
            6,
            200,
        ),
        TargetTrackingEnded(
            target,
            TargetTrackingEndReason.LOST_UNRECOVERED,
            7,
            300,
        ),
    )

    for event in events:
        channel.publish(event)

    assert tuple(channel.receive_nowait() for _ in events) == events
    with pytest.raises(Empty):
        channel.receive_nowait()


def test_target_lifecycle_channel_overflow_is_explicit_and_keeps_queued_event() -> None:
    channel = TargetLifecycleEventChannel(capacity=1)
    first = TargetAcquisitionFailed(
        1,
        CameraRole.OVERVIEW,
        1,
        5,
        TargetAcquisitionFailureReason.STALE_SELECTION,
        None,
        None,
    )
    second = TargetAcquisitionFailed(
        2,
        CameraRole.OVERVIEW,
        1,
        6,
        TargetAcquisitionFailureReason.STALE_SELECTION,
        None,
        None,
    )
    channel.publish(first)

    with pytest.raises(TargetLifecycleEventChannelOverflowError):
        channel.publish(second)

    assert channel.receive_nowait() is first
