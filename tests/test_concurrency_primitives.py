from __future__ import annotations

from dataclasses import dataclass
from queue import Empty
from threading import Thread

import pytest

from navmin.concurrency import (
    CameraSessionBarrierChannel,
    InvalidatableLatest,
    LatestValue,
)
from navmin.contracts import CameraRay, CameraRole, CameraSessionStarted, TargetRef


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
