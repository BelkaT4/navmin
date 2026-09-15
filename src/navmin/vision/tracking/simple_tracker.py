"""Small deterministic multi-object tracker for baseline Vision processors."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import median

from navmin.contracts import BBox, TrackedObject

from . import settings

type Center = tuple[float, float]


@dataclass(frozen=True)
class _Observation:
    timestamp_ns: int
    bbox: BBox
    center: Center


@dataclass
class _Track:
    internal_id: int
    bbox: BBox
    center: Center
    age_frames: int = 1
    misses: int = 0
    consecutive_matches: int = 1
    confirmed: bool = False
    public_track_id: int | None = None
    velocity_x_px_s: float = 0.0
    velocity_y_px_s: float = 0.0
    history: deque[_Observation] = field(
        default_factory=lambda: deque(maxlen=settings.MATCHED_HISTORY_LENGTH)
    )


class SimpleTracker:
    """2-hit tracker with short miss persistence and real-time velocity."""

    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._next_internal_id = 1
        self._next_public_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_internal_id = 1
        self._next_public_track_id = 1

    @staticmethod
    def _center(bbox: BBox) -> Center:
        return bbox.x + bbox.width * 0.5, bbox.y + bbox.height * 0.5

    @staticmethod
    def _distance(a: Center, b: Center) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _size_ratio(a: int, b: int) -> float:
        smaller = max(1, min(a, b))
        larger = max(1, max(a, b))
        return larger / smaller

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.width, b.x + b.width)
        y2 = min(a.y + a.height, b.y + b.height)
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        if intersection <= 0:
            return 0.0
        union = a.width * a.height + b.width * b.height - intersection
        return intersection / max(1, union)

    @staticmethod
    def _predicted_center(track: _Track, timestamp_ns: int) -> Center:
        if not track.history:
            return track.center
        dt_ns = timestamp_ns - track.history[-1].timestamp_ns
        if dt_ns <= 0:
            return track.center
        dt_s = dt_ns / 1_000_000_000.0
        return (
            track.center[0] + track.velocity_x_px_s * dt_s,
            track.center[1] + track.velocity_y_px_s * dt_s,
        )

    @staticmethod
    def _update_velocity(track: _Track) -> None:
        velocity_x: list[float] = []
        velocity_y: list[float] = []
        observations = tuple(track.history)
        for previous, current in pairwise(observations):
            dt_ns = current.timestamp_ns - previous.timestamp_ns
            if dt_ns <= 0:
                continue
            dt_s = dt_ns / 1_000_000_000.0
            velocity_x.append((current.center[0] - previous.center[0]) / dt_s)
            velocity_y.append((current.center[1] - previous.center[1]) / dt_s)
        if not velocity_x:
            track.velocity_x_px_s = 0.0
            track.velocity_y_px_s = 0.0
            return
        track.velocity_x_px_s = float(median(velocity_x))
        track.velocity_y_px_s = float(median(velocity_y))

    @classmethod
    def _association_cost(
        cls,
        track: _Track,
        detection: BBox,
        timestamp_ns: int,
    ) -> tuple[float, float] | None:
        width_ratio = cls._size_ratio(track.bbox.width, detection.width)
        height_ratio = cls._size_ratio(track.bbox.height, detection.height)
        if (
            width_ratio > settings.MAX_BBOX_SIZE_RATIO
            or height_ratio > settings.MAX_BBOX_SIZE_RATIO
        ):
            return None

        predicted = cls._predicted_center(track, timestamp_ns)
        distance = cls._distance(predicted, cls._center(detection))
        if distance > settings.MAX_ASSOCIATION_DISTANCE_PX:
            return None

        iou = cls._iou(track.bbox, detection)
        size_penalty = (
            math.log(width_ratio) + math.log(height_ratio)
        ) / (2.0 * math.log(settings.MAX_BBOX_SIZE_RATIO))
        distance_cost = distance / settings.MAX_ASSOCIATION_DISTANCE_PX
        cost = (
            distance_cost * settings.DISTANCE_COST_WEIGHT
            + (1.0 - iou) * settings.IOU_COST_WEIGHT
            + size_penalty * settings.SIZE_COST_WEIGHT
        )
        return cost, distance

    def _match(self, track: _Track, bbox: BBox, timestamp_ns: int) -> None:
        center = self._center(bbox)
        track.bbox = bbox
        track.center = center
        track.misses = 0
        track.consecutive_matches += 1
        track.history.append(_Observation(timestamp_ns, bbox, center))
        self._update_velocity(track)
        if not track.confirmed and track.consecutive_matches >= settings.CONFIRMATION_MATCHES:
            track.confirmed = True
            track.public_track_id = self._next_public_track_id
            self._next_public_track_id += 1

    def _new_track(self, bbox: BBox, timestamp_ns: int) -> None:
        center = self._center(bbox)
        track = _Track(
            internal_id=self._next_internal_id,
            bbox=bbox,
            center=center,
        )
        track.history.append(_Observation(timestamp_ns, bbox, center))
        self._tracks.append(track)
        self._next_internal_id += 1

    @staticmethod
    def _public_object(track: _Track) -> TrackedObject:
        if track.public_track_id is None:
            raise RuntimeError("confirmed track must have a public track ID")
        return TrackedObject(
            track_id=track.public_track_id,
            bbox=track.bbox,
            velocity_x_px_s=track.velocity_x_px_s,
            velocity_y_px_s=track.velocity_y_px_s,
            age_frames=track.age_frames,
        )

    def update(
        self,
        detections: tuple[BBox, ...] | list[BBox],
        *,
        timestamp_ns: int,
    ) -> tuple[TrackedObject, ...]:
        """Update tracks and publish only confirmed tracks matched this frame."""
        for track in self._tracks:
            track.age_frames += 1

        candidate_pairs: list[tuple[float, float, int, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                association = self._association_cost(track, detection, timestamp_ns)
                if association is None:
                    continue
                cost, distance = association
                candidate_pairs.append(
                    (cost, distance, track.internal_id, track_index, detection_index)
                )
        candidate_pairs.sort()

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        published: list[TrackedObject] = []
        for _cost, _distance, _internal_id, track_index, detection_index in candidate_pairs:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks[track_index]
            self._match(track, detections[detection_index], timestamp_ns)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            if track.confirmed:
                published.append(self._public_object(track))

        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks:
                continue
            track.misses += 1
            track.consecutive_matches = 0

        self._tracks = [
            track
            for track in self._tracks
            if track.misses < settings.MAX_CONSECUTIVE_MISSES
        ]

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._new_track(detection, timestamp_ns)

        published.sort(key=lambda item: item.track_id)
        return tuple(published)


__all__ = ["SimpleTracker"]
