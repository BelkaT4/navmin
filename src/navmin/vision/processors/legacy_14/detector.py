"""Clean port of legacy ``variant_1_sky_mog2_1_4`` candidate generation.

The original detector mixed its MOG2/compact-body candidate generation with a
small detector-local guide tracker used to search the current frame after MOG2
misses.  That guide remains private here because it is part of Variant 1.4's
candidate-generation behaviour.  It does not allocate NavMin public track IDs
and it never emits predicted/held boxes: public identity is owned exclusively by
``SimpleTracker``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from navmin.contracts import BBox

from . import settings

type DetectionCandidate = tuple[int, int, int, int, int]
type Center = tuple[float, float]


def _build_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if settings.MORPH_OPEN > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _build_kernel(settings.MORPH_OPEN))
    if settings.MORPH_CLOSE > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _build_kernel(settings.MORPH_CLOSE))
    return mask


def _extract_candidates(mask: np.ndarray) -> list[DetectionCandidate]:
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]
    detections: list[DetectionCandidate] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < settings.MIN_AREA or area > settings.MAX_AREA:
            continue
        if w < settings.MIN_WIDTH or h < settings.MIN_HEIGHT:
            continue
        if w > width * settings.MAX_WIDTH_RATIO or h > height * settings.MAX_HEIGHT_RATIO:
            continue
        aspect = w / max(h, 1)
        if not settings.ASPECT_RATIO_MIN <= aspect <= settings.ASPECT_RATIO_MAX:
            continue
        fill_ratio = area / float(max(w * h, 1))
        if fill_ratio < settings.MIN_FILL_RATIO:
            continue
        detections.append((x, y, w, h, area))
    return detections


def _median_u8(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    mid = flat.size // 2
    partitioned = np.partition(flat, mid)
    return float(partitioned[mid])


def _candidate_center(detection: DetectionCandidate) -> Center:
    x, y, w, h, _area = detection
    return float(x) + float(w) * 0.5, float(y) + float(h) * 0.5


def _center_distance(a: Center, b: Center) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _candidate_current_frame_contrast(gray: np.ndarray, detection: DetectionCandidate) -> float:
    x, y, w, h, _area = detection
    frame_h, frame_w = gray.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, x1 + max(1, int(w)))
    y2 = min(frame_h, y1 + max(1, int(h)))
    if x1 >= x2 or y1 >= y2:
        return 0.0

    pad = settings.CONTRAST_CONTEXT_PAD
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(frame_w, x2 + pad)
    cy2 = min(frame_h, y2 + pad)
    inner = gray[y1:y2, x1:x2]
    context = gray[cy1:cy2, cx1:cx2]
    if inner.size == 0 or context.size == 0:
        return 0.0

    context_mask = np.ones(context.shape, dtype=bool)
    context_mask[y1 - cy1 : y2 - cy1, x1 - cx1 : x2 - cx1] = False
    ring = context[context_mask]
    if ring.size < max(8, inner.size // 2):
        ring = context.reshape(-1)

    background = _median_u8(ring)
    inner_values = inner.reshape(-1)
    low = float(int(inner_values.min()))
    high = float(int(inner_values.max()))
    median = _median_u8(inner_values)
    return max(abs(low - background), abs(high - background), abs(median - background))


def _is_probable_thin_line_candidate(detection: DetectionCandidate) -> bool:
    _x, _y, w, h, area = detection
    short_side = max(1, min(int(w), int(h)))
    long_side = max(int(w), int(h))
    if (
        short_side <= settings.THIN_LINE_MAX_SHORT_SIDE
        and long_side >= short_side * settings.THIN_LINE_MIN_LONG_TO_SHORT
    ):
        return int(area) <= max(120, short_side * long_side)
    return int(h) >= int(w) * 3 and int(w) <= 28 and int(area) <= 900


def _is_probable_frame_edge_artifact(
    detection: DetectionCandidate,
    shape: tuple[int, ...],
) -> bool:
    x, y, w, h, _area = detection
    frame_h, frame_w = shape[:2]
    if frame_w <= 0 or frame_h <= 0:
        return False
    if x <= 1 or y <= 1 or x + w >= frame_w - 1 or y + h >= frame_h - 1:
        return True
    cx, cy = _candidate_center(detection)
    return cx >= frame_w * 0.88 or cy >= frame_h * 0.88


def _candidate_iou(a: DetectionCandidate, b: DetectionCandidate) -> float:
    ax, ay, aw, ah, _aa = a
    bx, by, bw, bh, _ba = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    union = max(1, aw * ah + bw * bh - intersection)
    return float(intersection) / float(union)


def _merge_candidate_lists(
    primary: list[DetectionCandidate],
    secondary: list[DetectionCandidate],
) -> list[DetectionCandidate]:
    merged = list(primary)
    for candidate in secondary:
        candidate_center = _candidate_center(candidate)
        duplicate = False
        for existing in merged:
            close_distance = max(
                10.0,
                (candidate[2] + candidate[3] + existing[2] + existing[3]) * 0.20,
            )
            if (
                _candidate_iou(candidate, existing) >= 0.15
                or _center_distance(candidate_center, _candidate_center(existing)) <= close_distance
            ):
                duplicate = True
                break
        if not duplicate:
            merged.append(candidate)
    return merged


def _odd_kernel_size(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _local_contrast_map(gray: np.ndarray) -> np.ndarray:
    blur_size = _odd_kernel_size(settings.REACQUIRE_BLUR_KSIZE)
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    return cv2.absdiff(gray, background)


def _extract_reacquire_candidates_from_search_box(
    gray: np.ndarray,
    contrast_map: np.ndarray,
    search_box: tuple[int, int, int, int],
    *,
    predicted_center: Center,
    max_candidates: int = 1,
    distance_penalty: float = 0.025,
) -> list[DetectionCandidate]:
    sx, sy, sw, sh = search_box
    if sw <= 0 or sh <= 0:
        return []
    roi = contrast_map[sy : sy + sh, sx : sx + sw]
    if roi.size == 0:
        return []

    _, binary = cv2.threshold(roi, settings.REACQUIRE_MIN_CONTRAST, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8, copy=False)
    open_size = _odd_kernel_size(settings.REACQUIRE_BODY_OPEN_KSIZE)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    frame_h, frame_w = gray.shape[:2]
    max_area = int(settings.MAX_AREA * settings.REACQUIRE_MAX_AREA_MULTIPLIER)
    candidates: list[tuple[float, DetectionCandidate]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT]) + sx
        y = int(stats[label, cv2.CC_STAT_TOP]) + sy
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        detection = (x, y, w, h, area)
        if area < settings.REACQUIRE_MIN_AREA or area > max_area:
            continue
        if _is_probable_frame_edge_artifact(detection, gray.shape[:2]):
            continue
        if w < settings.MIN_WIDTH or h < settings.MIN_HEIGHT:
            continue
        if w > frame_w * settings.MAX_WIDTH_RATIO or h > frame_h * settings.MAX_HEIGHT_RATIO:
            continue
        aspect = w / max(h, 1)
        if not settings.ASPECT_RATIO_MIN <= aspect <= settings.ASPECT_RATIO_MAX:
            continue
        if _is_probable_thin_line_candidate(detection):
            continue
        contrast = _candidate_current_frame_contrast(gray, detection)
        if contrast < settings.REACQUIRE_MIN_CONTRAST:
            continue
        distance = _center_distance(_candidate_center(detection), predicted_center)
        fill = area / float(max(w * h, 1))
        compact_bonus = min(1.0, fill) * 8.0
        size_bonus = min(1.0, area / float(max(1, settings.MIN_AREA))) * 4.0
        score = contrast + compact_bonus + size_bonus - distance * distance_penalty
        candidates.append((score, detection))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in candidates[: max(1, int(max_candidates))]]


def _expand_candidate_box(
    detection: DetectionCandidate,
    shape: tuple[int, ...],
    *,
    pad: int,
) -> tuple[int, int, int, int]:
    x, y, w, h, _area = detection
    frame_h, frame_w = shape[:2]
    x1 = max(0, int(x) - pad)
    y1 = max(0, int(y) - pad)
    x2 = min(frame_w, int(x) + int(w) + pad)
    y2 = min(frame_h, int(y) + int(h) + pad)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _refine_candidate_to_current_frame_body(
    gray: np.ndarray,
    contrast_map: np.ndarray,
    detection: DetectionCandidate,
) -> DetectionCandidate | None:
    _x, _y, w, h, _area = detection
    pad = max(10, min(48, int(max(w, h) * 0.35)))
    refined = _extract_reacquire_candidates_from_search_box(
        gray,
        contrast_map,
        _expand_candidate_box(detection, gray.shape[:2], pad=pad),
        predicted_center=_candidate_center(detection),
    )
    if refined:
        return refined[0]
    if _is_probable_frame_edge_artifact(detection, gray.shape[:2]):
        return None
    if _is_probable_thin_line_candidate(detection):
        return None
    if _candidate_current_frame_contrast(gray, detection) < settings.MIN_CURRENT_CONTRAST:
        return None
    return detection


def _refine_candidates_by_current_frame_body(
    gray: np.ndarray,
    detections: list[DetectionCandidate],
) -> list[DetectionCandidate]:
    if not detections:
        return []
    contrast_map = _local_contrast_map(gray)
    refined: list[DetectionCandidate] = []
    for detection in detections:
        candidate = _refine_candidate_to_current_frame_body(gray, contrast_map, detection)
        if candidate is not None:
            refined = _merge_candidate_lists(refined, [candidate])
    return refined


def _suppress_tiny_artifacts_when_body_present(
    detections: list[DetectionCandidate],
) -> list[DetectionCandidate]:
    has_body = any(
        area >= settings.BODY_PRESENT_MIN_AREA
        and min(w, h) >= settings.BODY_PRESENT_MIN_SHORT_SIDE
        for _x, _y, w, h, area in detections
    )
    if not has_body:
        return detections
    return [
        detection
        for detection in detections
        if not (
            detection[4] <= settings.TINY_ARTIFACT_MAX_AREA
            or min(detection[2], detection[3]) <= settings.TINY_ARTIFACT_MAX_SHORT_SIDE
        )
    ]


def _clip_search_box(
    center: Center,
    radius: float,
    shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = shape[:2]
    cx, cy = center
    r = max(1, round(radius))
    x1 = max(0, round(cx) - r)
    y1 = max(0, round(cy) - r)
    x2 = min(frame_w, round(cx) + r)
    y2 = min(frame_h, round(cy) + r)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _scale_detection_to_shape(
    detection: DetectionCandidate,
    inv_scale: float,
    shape: tuple[int, ...],
) -> DetectionCandidate:
    x, y, w, h, area = detection
    frame_h, frame_w = shape[:2]
    sx = max(0, min(frame_w - 1, round(x * inv_scale)))
    sy = max(0, min(frame_h - 1, round(y * inv_scale)))
    sw = max(1, round(w * inv_scale))
    sh = max(1, round(h * inv_scale))
    if sx + sw > frame_w:
        sw = max(1, frame_w - sx)
    if sy + sh > frame_h:
        sh = max(1, frame_h - sy)
    scaled_area = max(1, round(area * inv_scale * inv_scale))
    return sx, sy, sw, sh, scaled_area


def _find_global_compact_body_candidates(gray: np.ndarray) -> list[DetectionCandidate]:
    frame_h, frame_w = gray.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return []

    scan_gray = gray
    scan_scale = 1.0
    if frame_w > settings.GLOBAL_BODY_SCAN_MAX_WIDTH:
        scan_scale = settings.GLOBAL_BODY_SCAN_MAX_WIDTH / float(frame_w)
        scan_gray = cv2.resize(
            gray,
            (
                settings.GLOBAL_BODY_SCAN_MAX_WIDTH,
                max(1, round(frame_h * scan_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    scan_h, scan_w = scan_gray.shape[:2]
    candidates = _extract_reacquire_candidates_from_search_box(
        scan_gray,
        _local_contrast_map(scan_gray),
        (0, 0, scan_w, scan_h),
        predicted_center=(scan_w * 0.5, scan_h * 0.5),
        max_candidates=8,
        distance_penalty=0.0,
    )
    min_body_area = max(24, round(settings.BODY_PRESENT_MIN_AREA * scan_scale * scan_scale))
    min_short_side = max(6, round(settings.BODY_PRESENT_MIN_SHORT_SIDE * scan_scale))
    bodies = [
        candidate
        for candidate in candidates
        if candidate[4] >= min_body_area and min(candidate[2], candidate[3]) >= min_short_side
    ]
    bodies.sort(
        key=lambda candidate: (
            candidate[4],
            _candidate_current_frame_contrast(scan_gray, candidate),
            min(candidate[2], candidate[3]),
        ),
        reverse=True,
    )
    if not bodies:
        return []
    if scan_scale == 1.0:
        return bodies[:1]
    return [_scale_detection_to_shape(bodies[0], 1.0 / scan_scale, gray.shape[:2])]


@dataclass
class _GuideTrack:
    bbox: tuple[int, int, int, int]
    center: Center
    hits: int
    misses: int
    area: int
    velocity: Center


class _CandidateGuideTracker:
    """Legacy detector-local guide state used only for current-frame reacquire."""

    def __init__(self) -> None:
        self._tracks: list[_GuideTrack] = []

    def reset(self) -> None:
        self._tracks = []

    @staticmethod
    def _area_ratio(a: int, b: int) -> float:
        smaller = max(1, min(int(a), int(b)))
        larger = max(1, max(int(a), int(b)))
        return float(larger) / float(smaller)

    @staticmethod
    def _predicted_center(track: _GuideTrack) -> Center:
        steps = max(1, int(track.misses) + 1)
        return (
            track.center[0] + track.velocity[0] * steps,
            track.center[1] + track.velocity[1] * steps,
        )

    def confirmed_tracks_for_reacquire(self) -> list[_GuideTrack]:
        return [
            track
            for track in self._tracks
            if track.hits >= settings.GUIDE_MIN_CONFIRMED_HITS
            and track.misses <= settings.GUIDE_MAX_MISSED_FRAMES
        ]

    def search_radius_for(self, track: _GuideTrack) -> float:
        velocity_px = _center_distance((0.0, 0.0), track.velocity)
        box_size = max(track.bbox[2], track.bbox[3])
        radius = max(
            settings.REACQUIRE_MIN_WINDOW_RADIUS,
            self._gate_for(track, track.area) + velocity_px * 0.8 + box_size * 1.5,
        )
        return min(settings.REACQUIRE_MAX_WINDOW_RADIUS, radius)

    def _gate_for(self, track: _GuideTrack, area: int) -> float:
        base = settings.GUIDE_MAX_MATCH_DISTANCE
        velocity_px = _center_distance((0.0, 0.0), track.velocity)
        multiplier = 1.0
        if track.hits < settings.GUIDE_MIN_CONFIRMED_HITS:
            multiplier = max(multiplier, settings.GUIDE_BOOTSTRAP_DISTANCE_MULTIPLIER)
        if velocity_px >= base * 0.45:
            multiplier = max(multiplier, settings.GUIDE_FAST_DISTANCE_MULTIPLIER)
        if track.misses > 0:
            multiplier = max(
                multiplier,
                1.0 + min(1.0, track.misses * settings.GUIDE_MISSED_DISTANCE_BONUS),
            )
        if self._area_ratio(track.area, area) > 4.0:
            multiplier = min(multiplier, 1.35)
        return base * multiplier

    def update(self, detections: list[DetectionCandidate]) -> None:
        det_centers = [_candidate_center(detection) for detection in detections]
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        pairs: list[tuple[float, float, int, int]] = []

        for track_index, track in enumerate(self._tracks):
            predicted = self._predicted_center(track)
            for detection_index, center in enumerate(det_centers):
                area = detections[detection_index][4]
                current_distance = _center_distance(track.center, center)
                predicted_distance = _center_distance(predicted, center)
                if min(current_distance, predicted_distance) > self._gate_for(track, area):
                    continue
                area_ratio = self._area_ratio(track.area, area)
                area_penalty = min(1.5, area_ratio - 1.0) * settings.GUIDE_MAX_MATCH_DISTANCE * 0.20
                score = predicted_distance + current_distance * 0.18 + area_penalty
                pairs.append((score, predicted_distance, track_index, detection_index))

        def apply_match(track_index: int, detection_index: int) -> None:
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            x, y, w, h, area = detections[detection_index]
            new_center = det_centers[detection_index]
            track = self._tracks[track_index]
            measured_velocity = (
                new_center[0] - track.center[0],
                new_center[1] - track.center[1],
            )
            if track.hits <= 1 and track.velocity == (0.0, 0.0):
                velocity = measured_velocity
            else:
                alpha = settings.GUIDE_VELOCITY_EMA_ALPHA
                velocity = (
                    track.velocity[0] * alpha + measured_velocity[0] * (1.0 - alpha),
                    track.velocity[1] * alpha + measured_velocity[1] * (1.0 - alpha),
                )
            track.bbox = (x, y, w, h)
            track.center = new_center
            track.velocity = velocity
            track.hits += 1
            track.misses = 0
            track.area = area

        pairs.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for _score, _predicted_distance, track_index, detection_index in pairs:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            apply_match(track_index, detection_index)

        if detections and not matched_tracks:
            bridge_options: list[tuple[float, int, int]] = []
            for track_index, track in enumerate(self._tracks):
                if (
                    track.hits < settings.GUIDE_MIN_CONFIRMED_HITS
                    or track.misses > settings.GUIDE_MAX_MISSED_FRAMES
                ):
                    continue
                for detection_index, center in enumerate(det_centers):
                    if detection_index in matched_detections:
                        continue
                    area = detections[detection_index][4]
                    if self._area_ratio(track.area, area) > 8.0:
                        continue
                    distance = min(
                        _center_distance(track.center, center),
                        _center_distance(self._predicted_center(track), center),
                    )
                    if distance <= settings.GUIDE_SINGLE_CANDIDATE_REACQUIRE_GATE:
                        bridge_options.append((distance, track_index, detection_index))
            if bridge_options:
                _distance, track_index, detection_index = min(bridge_options)
                apply_match(track_index, detection_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.misses += 1
        self._tracks = [
            track for track in self._tracks if track.misses <= settings.GUIDE_MAX_MISSED_FRAMES
        ]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            x, y, w, h, area = detection
            self._tracks.append(
                _GuideTrack(
                    bbox=(x, y, w, h),
                    center=det_centers[detection_index],
                    hits=1,
                    misses=0,
                    area=area,
                    velocity=(0.0, 0.0),
                )
            )


class Legacy14Detector:
    """Variant 1.4 candidate detector without public tracking identity."""

    def __init__(self) -> None:
        self._guide = _CandidateGuideTracker()
        self._frame_index = 0
        self._background = self._new_background_subtractor()

    @staticmethod
    def _new_background_subtractor() -> cv2.BackgroundSubtractor:
        return cv2.createBackgroundSubtractorMOG2(
            history=settings.MOG2_HISTORY,
            varThreshold=settings.MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )

    def reset(self) -> None:
        self._guide.reset()
        self._frame_index = 0
        self._background = self._new_background_subtractor()

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be numpy.ndarray")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use uint8 pixels")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected BGR HxWx3 frame, got shape={frame.shape!r}")

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        Legacy14Detector._validate_frame(frame)
        if frame.shape[1] <= settings.RESIZE_WIDTH:
            work_frame = frame
            scale = 1.0
        else:
            scale = settings.RESIZE_WIDTH / float(frame.shape[1])
            work_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
        if settings.BLUR_KSIZE >= 3:
            gray = cv2.GaussianBlur(gray, (settings.BLUR_KSIZE, settings.BLUR_KSIZE), 0)
        return work_frame, gray, scale

    def _find_reacquire_candidates(self, gray: np.ndarray) -> list[DetectionCandidate]:
        tracks = self._guide.confirmed_tracks_for_reacquire()
        if not tracks:
            return []
        contrast_map = _local_contrast_map(gray)
        reacquired: list[DetectionCandidate] = []
        for track in tracks:
            predicted = self._guide._predicted_center(track)
            candidates = _extract_reacquire_candidates_from_search_box(
                gray,
                contrast_map,
                _clip_search_box(
                    predicted,
                    self._guide.search_radius_for(track),
                    gray.shape[:2],
                ),
                predicted_center=predicted,
            )
            reacquired = _merge_candidate_lists(reacquired, candidates)
        return reacquired

    @staticmethod
    def _to_public_bbox(
        detection: DetectionCandidate,
        scale: float,
        frame_shape: tuple[int, ...],
    ) -> BBox | None:
        x, y, w, h, _area = detection
        inv_scale = 1.0 / scale
        x = round(x * inv_scale)
        y = round(y * inv_scale)
        w = round(w * inv_scale)
        h = round(h * inv_scale)
        frame_h, frame_w = frame_shape[:2]
        x1 = max(0, min(x, frame_w))
        y1 = max(0, min(y, frame_h))
        x2 = max(x1, min(x + max(0, w), frame_w))
        y2 = max(y1, min(y + max(0, h), frame_h))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None
        return BBox(x=x1, y=y1, width=width, height=height)

    def detect(self, frame: np.ndarray) -> tuple[BBox, ...]:
        """Return only current-frame candidate boxes in working-frame coordinates."""
        work_frame, gray, scale = self._prepare_frame(frame)
        learning_rate = 0.02 if self._frame_index < settings.WARMUP_FRAMES else -1
        foreground = self._background.apply(gray, learningRate=learning_rate)
        _, mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        mask = _postprocess_mask(mask)

        coverage = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
        if coverage >= settings.OVERLOAD_MASK_COVERAGE_THRESHOLD:
            self._guide.reset()
            self._frame_index += 1
            return ()

        detections = _extract_candidates(mask)
        if len(detections) > settings.OVERLOAD_CANDIDATE_COUNT_THRESHOLD:
            self._guide.reset()
            self._frame_index += 1
            return ()

        current_gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
        body_assist = _find_global_compact_body_candidates(current_gray)
        if body_assist:
            detections = body_assist
        else:
            detections = _refine_candidates_by_current_frame_body(current_gray, detections)
            reacquired = self._find_reacquire_candidates(current_gray)
            if reacquired:
                detections = reacquired
            else:
                detections = _suppress_tiny_artifacts_when_body_present(detections)

        self._guide.update(detections)
        self._frame_index += 1
        boxes = (
            self._to_public_bbox(detection, scale, frame.shape)
            for detection in detections
        )
        return tuple(box for box in boxes if box is not None)


__all__ = ["Legacy14Detector"]
