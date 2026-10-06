"""VT-10 HOTFIX-3 local candidate detector.

The detector extracts multiple local hypotheses inside the search ROI using an
adaptive local-contrast response and connected components.  HOTFIX-3 adds a
*target-relative area gate* used by TRACK/REACQ: once a target size estimate is
available, implausibly tiny/huge components are removed before Top-K ranking.
This directly addresses the observed failure where REACQ chose 2-4 px noise
components while the tracked object occupied roughly 1.3-2.4 kpx.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, hypot, log

import cv2
import numpy as np

from navmin.vision.processors.vt11.acquisition.roi import OperatorSelection, RoiRect, make_operator_selection


@dataclass(frozen=True)
class CandidateDetectorConfig:
    method: str = "local_contrast_components"
    blur_sigma_px: float = 3.0
    threshold_sigma: float = 2.5
    min_response: float = 5.0
    min_area_px: int = 2
    max_area_px: int = 2500
    morph_close_px: int = 3
    spatial_sigma_px: float = 35.0
    spatial_weight: float = 0.80
    detector_weight: float = 0.20

    # HOTFIX-3 target-relative size filtering/ranking.  These parameters are
    # ignored during the first ACQUIRE click where expected_area_px is unknown.
    adaptive_area_enabled: bool = True
    adaptive_hard_max_area_px: int = 50000
    adaptive_min_area_ratio: float = 0.15
    adaptive_max_area_ratio: float = 4.0
    expected_area_sigma_log: float = 0.80
    expected_area_weight: float = 0.50

    # HOTFIX-4 thermal hot-blob detector.
    thermal_polarity: str = "bright"  # bright | dark
    thermal_top_hat_kernel_px: int = 21
    thermal_top_hat_scale: float = 2.2
    thermal_top_hat_max_px: int = 81

    # VT-11 HOTFIX-4 REV1: conservative multiscale support for the first
    # thermal ACQUIRE and for already-known large targets.  The extra scale is
    # deliberately dormant for normal/small tracked targets so old THERMAL
    # behaviour is preserved as much as possible.
    acquire_multiscale_enabled: bool = False
    acquire_large_scale_sigma_px: float = 22.0
    acquire_large_target_trigger_area_px: float = 2500.0
    acquire_max_area_fraction_roi: float = 0.70
    acquire_expand_roi_radius_px: int = 140
    acquire_border_margin_px: int = 5

    # VT-11 HOTFIX-4 REV2: click-seeded raw-thermal fallback.  This path is
    # used only during initial THERMAL ACQUIRE when contrast/top-hat passes
    # cannot form one coherent large object.  It deliberately does not affect
    # normal TRACK/REACQ candidate generation.
    acquire_click_seed_enabled: bool = False
    acquire_click_seed_roi_radius_px: int = 180
    acquire_click_seed_close_px: int = 11
    acquire_click_seed_max_support_distance_px: float = 42.0

    # HOTFIX-6 REV2: secondary MICRO channel for tiny thermal targets.
    # It reuses the same connected-components pass, so enabling it does not
    # duplicate the expensive image-response calculation.
    micro_enabled: bool = True
    micro_min_area_px: int = 3
    micro_max_area_px: int = 80
    micro_response_multiplier: float = 1.20
    micro_top_k: int = 4

    def validate(self) -> None:
        if self.method not in {"local_contrast_components", "thermal_hot_blob"}:
            raise ValueError("Unknown candidate detector method")
        if self.thermal_polarity not in {"bright", "dark"}:
            raise ValueError("thermal_polarity must be bright or dark")
        if self.thermal_top_hat_kernel_px < 3 or self.thermal_top_hat_max_px < self.thermal_top_hat_kernel_px:
            raise ValueError("thermal top-hat kernel settings invalid")
        if self.thermal_top_hat_scale <= 0:
            raise ValueError("thermal_top_hat_scale must be > 0")
        if self.acquire_large_scale_sigma_px <= 0:
            raise ValueError("acquire_large_scale_sigma_px must be > 0")
        if self.acquire_large_target_trigger_area_px <= 0:
            raise ValueError("acquire_large_target_trigger_area_px must be > 0")
        if not 0.05 <= self.acquire_max_area_fraction_roi <= 0.95:
            raise ValueError("acquire_max_area_fraction_roi must be in [0.05,0.95]")
        if self.acquire_expand_roi_radius_px < 1 or self.acquire_border_margin_px < 0:
            raise ValueError("ACQUIRE ROI expansion settings invalid")
        if self.acquire_click_seed_roi_radius_px < 1 or self.acquire_click_seed_close_px < 0:
            raise ValueError("click-seeded ACQUIRE settings invalid")
        if self.acquire_click_seed_max_support_distance_px < 0:
            raise ValueError("acquire_click_seed_max_support_distance_px must be >= 0")
        if self.micro_min_area_px < 1 or self.micro_max_area_px < self.micro_min_area_px:
            raise ValueError("micro area settings invalid")
        if self.micro_response_multiplier < 1.0:
            raise ValueError("micro_response_multiplier must be >= 1")
        if self.micro_top_k < 1:
            raise ValueError("micro_top_k must be >= 1")
        if self.blur_sigma_px <= 0:
            raise ValueError("blur_sigma_px must be > 0")
        if self.threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be > 0")
        if self.min_response < 0:
            raise ValueError("min_response must be >= 0")
        if self.min_area_px < 1:
            raise ValueError("min_area_px must be >= 1")
        if self.max_area_px < self.min_area_px:
            raise ValueError("max_area_px must be >= min_area_px")
        if self.morph_close_px < 0:
            raise ValueError("morph_close_px must be >= 0")
        if self.spatial_sigma_px <= 0:
            raise ValueError("spatial_sigma_px must be > 0")
        if self.spatial_weight < 0 or self.detector_weight < 0 or self.expected_area_weight < 0:
            raise ValueError("ranking weights must be >= 0")
        if self.spatial_weight + self.detector_weight <= 0:
            raise ValueError("at least one base ranking weight must be > 0")
        if self.adaptive_hard_max_area_px < self.min_area_px:
            raise ValueError("adaptive_hard_max_area_px must be >= min_area_px")
        if self.adaptive_min_area_ratio <= 0:
            raise ValueError("adaptive_min_area_ratio must be > 0")
        if self.adaptive_max_area_ratio < self.adaptive_min_area_ratio:
            raise ValueError("adaptive_max_area_ratio must be >= adaptive_min_area_ratio")
        if self.expected_area_sigma_log <= 0:
            raise ValueError("expected_area_sigma_log must be > 0")


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    bbox: RoiRect
    center_x: float
    center_y: float
    area_px: int
    mean_response: float
    peak_response: float
    distance_to_click_px: float
    spatial_prior: float
    detector_strength: float
    rank_score: float
    # 1.0 when target-relative filtering is not active.
    size_prior: float = 1.0
    detection_channel: str = "NORMAL"  # NORMAL | MICRO

    @property
    def center(self) -> tuple[float, float]:
        return self.center_x, self.center_y


@dataclass(frozen=True)
class DetectionDiagnostics:
    threshold: float
    median_response: float
    robust_sigma: float
    raw_components: int
    valid_components: int
    expected_area_px: float | None = None
    effective_min_area_px: int = 0
    effective_max_area_px: int = 0
    micro_valid_components: int = 0


@dataclass(frozen=True)
class CandidateDetectionResult:
    candidates: tuple[Candidate, ...]
    diagnostics: DetectionDiagnostics


def config_from_dict(data: dict | None) -> CandidateDetectorConfig:
    data = data or {}
    cfg = CandidateDetectorConfig(
        method=str(data.get("method", "local_contrast_components")),
        blur_sigma_px=float(data.get("blur_sigma_px", 3.0)),
        threshold_sigma=float(data.get("threshold_sigma", 2.5)),
        min_response=float(data.get("min_response", 5.0)),
        min_area_px=int(data.get("min_area_px", 2)),
        max_area_px=int(data.get("max_area_px", 2500)),
        morph_close_px=int(data.get("morph_close_px", 3)),
        spatial_sigma_px=float(data.get("spatial_sigma_px", 35.0)),
        spatial_weight=float(data.get("spatial_weight", 0.80)),
        detector_weight=float(data.get("detector_weight", 0.20)),
        adaptive_area_enabled=bool(data.get("adaptive_area_enabled", True)),
        adaptive_hard_max_area_px=int(data.get("adaptive_hard_max_area_px", 50000)),
        adaptive_min_area_ratio=float(data.get("adaptive_min_area_ratio", 0.15)),
        adaptive_max_area_ratio=float(data.get("adaptive_max_area_ratio", 4.0)),
        expected_area_sigma_log=float(data.get("expected_area_sigma_log", 0.80)),
        expected_area_weight=float(data.get("expected_area_weight", 0.50)),
        thermal_polarity=str(data.get("thermal_polarity", "bright")),
        thermal_top_hat_kernel_px=int(data.get("thermal_top_hat_kernel_px", 21)),
        thermal_top_hat_scale=float(data.get("thermal_top_hat_scale", 2.2)),
        thermal_top_hat_max_px=int(data.get("thermal_top_hat_max_px", 81)),
        acquire_multiscale_enabled=bool(data.get("acquire_multiscale_enabled", False)),
        acquire_large_scale_sigma_px=float(data.get("acquire_large_scale_sigma_px", 22.0)),
        acquire_large_target_trigger_area_px=float(data.get("acquire_large_target_trigger_area_px", 2500.0)),
        acquire_max_area_fraction_roi=float(data.get("acquire_max_area_fraction_roi", 0.70)),
        acquire_expand_roi_radius_px=int(data.get("acquire_expand_roi_radius_px", 140)),
        acquire_border_margin_px=int(data.get("acquire_border_margin_px", 5)),
        acquire_click_seed_enabled=bool(data.get("acquire_click_seed_enabled", False)),
        acquire_click_seed_roi_radius_px=int(data.get("acquire_click_seed_roi_radius_px", 180)),
        acquire_click_seed_close_px=int(data.get("acquire_click_seed_close_px", 11)),
        acquire_click_seed_max_support_distance_px=float(data.get("acquire_click_seed_max_support_distance_px", 42.0)),
        micro_enabled=bool(data.get("micro_enabled", True)),
        micro_min_area_px=int(data.get("micro_min_area_px", 3)),
        micro_max_area_px=int(data.get("micro_max_area_px", 80)),
        micro_response_multiplier=float(data.get("micro_response_multiplier", 1.20)),
        micro_top_k=int(data.get("micro_top_k", 4)),
    )
    cfg.validate()
    return cfg


def _odd_kernel(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def _robust_threshold(response: np.ndarray, cfg: CandidateDetectorConfig) -> tuple[float, float, float]:
    vals = response.astype(np.float32, copy=False).reshape(-1)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(cfg.min_response, median + cfg.threshold_sigma * robust_sigma)
    return float(threshold), median, robust_sigma


def _local_contrast_response(gray_roi: np.ndarray, cfg: CandidateDetectorConfig) -> np.ndarray:
    gray_f = gray_roi.astype(np.float32)
    background = cv2.GaussianBlur(
        gray_f,
        ksize=(0, 0),
        sigmaX=cfg.blur_sigma_px,
        sigmaY=cfg.blur_sigma_px,
        borderType=cv2.BORDER_REPLICATE,
    )
    return cv2.absdiff(gray_f, background)



def _thermal_blob_response(
    gray_roi: np.ndarray,
    cfg: CandidateDetectorConfig,
    expected_area_px: float | None,
    *,
    acquire_large_scale_pass: bool = False,
) -> np.ndarray:
    """Polarity-aware thermal response.

    REV1 keeps the historical top-hat path untouched for normal/small targets.
    During the first ACQUIRE (unknown area), or once the target is already
    known to be large, one additional broad Gaussian-background response is
    fused with max().  This is scale-generic and avoids a dedicated
    "aircraft" detector.
    """
    linear = 0.0
    if expected_area_px is not None and np.isfinite(expected_area_px) and expected_area_px > 0:
        linear = float(np.sqrt(expected_area_px))
    desired = max(float(cfg.thermal_top_hat_kernel_px), cfg.thermal_top_hat_scale * linear)
    k = int(round(min(float(cfg.thermal_top_hat_max_px), desired)))
    if k % 2 == 0:
        k += 1
    k = max(3, min(k, cfg.thermal_top_hat_max_px if cfg.thermal_top_hat_max_px % 2 == 1 else cfg.thermal_top_hat_max_px - 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if cfg.thermal_polarity == "dark":
        primary = cv2.morphologyEx(gray_roi, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    else:
        primary = cv2.morphologyEx(gray_roi, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    use_large_scale = bool(
        cfg.acquire_multiscale_enabled
        and (
            acquire_large_scale_pass
            or (expected_area_px is not None and np.isfinite(expected_area_px) and float(expected_area_px) >= cfg.acquire_large_target_trigger_area_px)
        )
    )
    if not use_large_scale:
        return primary

    gray_f = gray_roi.astype(np.float32)
    background = cv2.GaussianBlur(
        gray_f, ksize=(0, 0),
        sigmaX=cfg.acquire_large_scale_sigma_px,
        sigmaY=cfg.acquire_large_scale_sigma_px,
        borderType=cv2.BORDER_REPLICATE,
    )
    if cfg.thermal_polarity == "dark":
        broad = np.maximum(background - gray_f, 0.0)
    else:
        broad = np.maximum(gray_f - background, 0.0)
    return np.maximum(primary, broad.astype(np.float32, copy=False))


def _effective_area_limits(
    cfg: CandidateDetectorConfig,
    expected_area_px: float | None,
    *,
    roi_area_px: int | None = None,
    acquire_large_scale_pass: bool = False,
) -> tuple[int, int, float | None]:
    expected = None
    if expected_area_px is not None and np.isfinite(expected_area_px) and expected_area_px > 0:
        expected = float(expected_area_px)

    if not cfg.adaptive_area_enabled or expected is None:
        hi = int(cfg.max_area_px)
        if (
            expected is None
            and cfg.method == "thermal_hot_blob"
            and cfg.acquire_multiscale_enabled
            and acquire_large_scale_pass
            and roi_area_px is not None
            and roi_area_px > 0
        ):
            hi = max(hi, int(round(float(roi_area_px) * cfg.acquire_max_area_fraction_roi)))
            hi = min(hi, int(cfg.adaptive_hard_max_area_px))
        return int(cfg.min_area_px), int(hi), expected

    lo = int(round(expected * cfg.adaptive_min_area_ratio))
    hi = int(round(expected * cfg.adaptive_max_area_ratio))
    lo = max(int(cfg.min_area_px), max(1, lo))
    # ``max_area_px`` remains the legacy ACQUIRE cap. During TRACK/REACQ the
    # target-relative upper limit may legitimately exceed it as the object
    # grows; only the much wider hard safety cap is applied.
    hi = min(int(cfg.adaptive_hard_max_area_px), max(lo, hi))
    return lo, hi, expected


def _expected_area_prior(area_px: int, expected_area_px: float | None, sigma_log: float) -> float:
    if expected_area_px is None or expected_area_px <= 0:
        return 1.0
    ratio = max(float(area_px), 1.0) / max(float(expected_area_px), 1.0)
    return float(exp(-abs(log(ratio)) / max(1e-9, sigma_log)))


def detect_local_candidates(
    frame_bgr: np.ndarray,
    selection: OperatorSelection,
    *,
    top_k: int = 5,
    config: CandidateDetectorConfig | None = None,
    expected_area_px: float | None = None,
    include_micro: bool = False,
    global_search: bool = False,
    diversity_min_distance_px: float = 70.0,
    acquire_large_scale_pass: bool = False,
) -> CandidateDetectionResult:
    """Detect and rank local object hypotheses inside ``selection.roi``.

    Candidate bounding boxes and centers are returned in original-frame
    coordinates.  When ``expected_area_px`` is supplied by TRACK/REACQ, a
    target-relative area gate is applied *before* Top-K selection and area
    plausibility contributes to shortlist ordering.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR HxWx3 frame")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if diversity_min_distance_px < 0:
        raise ValueError("diversity_min_distance_px must be >= 0")

    cfg = config or CandidateDetectorConfig()
    cfg.validate()

    frame_h, frame_w = frame_bgr.shape[:2]
    roi = selection.roi
    if roi.x < 0 or roi.y < 0 or roi.x2 > frame_w or roi.y2 > frame_h:
        raise ValueError("Selection ROI lies outside the frame")

    crop = frame_bgr[roi.y:roi.y2, roi.x:roi.x2]
    if crop.size == 0:
        raise ValueError("Selection ROI is empty")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cfg.method == "thermal_hot_blob":
        response = _thermal_blob_response(
            gray, cfg, expected_area_px, acquire_large_scale_pass=acquire_large_scale_pass
        )
    else:
        response = _local_contrast_response(gray, cfg)
    threshold, median, robust_sigma = _robust_threshold(response, cfg)
    mask = (response >= threshold).astype(np.uint8) * 255

    kernel_size = _odd_kernel(cfg.morph_close_px)
    if kernel_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw_components = max(0, count - 1)
    effective_min_area, effective_max_area, expected = _effective_area_limits(
        cfg, expected_area_px, roi_area_px=int(roi.width * roi.height),
        acquire_large_scale_pass=acquire_large_scale_pass,
    )

    provisional: list[Candidate] = []
    micro_provisional: list[Candidate] = []
    use_size_prior = cfg.adaptive_area_enabled and expected is not None and cfg.expected_area_weight > 0
    rank_weight_sum = cfg.spatial_weight + cfg.detector_weight + (cfg.expected_area_weight if use_size_prior else 0.0)
    spatial_w = cfg.spatial_weight / rank_weight_sum
    detector_w = cfg.detector_weight / rank_weight_sum
    size_w = (cfg.expected_area_weight / rank_weight_sum) if use_size_prior else 0.0

    for label_id in range(1, count):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        normal_valid = effective_min_area <= area <= effective_max_area
        micro_area_valid = bool(
            include_micro and cfg.micro_enabled
            and cfg.micro_min_area_px <= area <= cfg.micro_max_area_px
        )
        if not normal_valid and not micro_area_valid:
            continue

        cx_local, cy_local = centroids[label_id]
        cx = float(roi.x + cx_local)
        cy = float(roi.y + cy_local)
        distance = hypot(cx - selection.click_x, cy - selection.click_y)
        spatial_prior = exp(-(distance * distance) / (2.0 * cfg.spatial_sigma_px * cfg.spatial_sigma_px))

        component_values = response[labels == label_id]
        mean_response = float(component_values.mean()) if component_values.size else 0.0
        peak_response = float(component_values.max()) if component_values.size else 0.0
        detector_strength = min(1.0, mean_response / max(threshold * 2.0, 1e-6))

        # A component may be accepted by the normal target-relative size gate,
        # by the MICRO gate, or by both. NORMAL wins on overlap to avoid duplicates.
        if normal_valid:
            size_prior = _expected_area_prior(area, expected, cfg.expected_area_sigma_log)
            rank_score = spatial_w * spatial_prior + detector_w * detector_strength + size_w * size_prior
            channel = "NORMAL"
            target_list = provisional
        else:
            if mean_response < threshold * cfg.micro_response_multiplier:
                continue
            # MICRO candidates deliberately ignore the stale expected target area.
            # Spatial proximity and a strong hot-blob response dominate pre-ranking.
            size_prior = 1.0
            rank_score = 0.55 * spatial_prior + 0.45 * detector_strength
            channel = "MICRO"
            target_list = micro_provisional

        target_list.append(
            Candidate(
                candidate_id=0,
                bbox=RoiRect(x=roi.x + x, y=roi.y + y, width=w, height=h),
                center_x=cx,
                center_y=cy,
                area_px=area,
                mean_response=mean_response,
                peak_response=peak_response,
                distance_to_click_px=float(distance),
                spatial_prior=float(spatial_prior),
                detector_strength=float(detector_strength),
                rank_score=float(rank_score),
                size_prior=float(size_prior),
                detection_channel=channel,
            )
        )

    sort_key = lambda c: (-c.rank_score, c.distance_to_click_px, -c.size_prior, -c.mean_response, c.bbox.y, c.bbox.x)

    def global_key(c: Candidate):
        # Global THERMAL search must not be dominated by distance to frame centre.
        # Detector strength and size plausibility define the pool; spatial diversity
        # is enforced afterwards. MICRO deliberately ignores stale expected area.
        if c.detection_channel == "MICRO":
            score = 0.85 * c.detector_strength + 0.15 * min(1.0, c.mean_response / max(threshold, 1e-6))
        else:
            score = 0.58 * c.detector_strength + 0.42 * c.size_prior
        return (-score, -c.mean_response, c.bbox.y, c.bbox.x)

    def diverse(items: list[Candidate], count: int) -> list[Candidate]:
        if count <= 0:
            return []
        ordered = sorted(items, key=global_key if global_search else sort_key)
        if not global_search or diversity_min_distance_px <= 0:
            return ordered[:count]
        picked: list[Candidate] = []
        deferred: list[Candidate] = []
        for c in ordered:
            if all(hypot(c.center_x - p.center_x, c.center_y - p.center_y) >= diversity_min_distance_px for p in picked):
                picked.append(c)
                if len(picked) >= count:
                    break
            else:
                deferred.append(c)
        if len(picked) < count:
            for c in deferred:
                if c not in picked:
                    picked.append(c)
                    if len(picked) >= count:
                        break
        return picked

    provisional.sort(key=global_key if global_search else sort_key)
    micro_provisional.sort(key=global_key if global_search else sort_key)
    shortlist = diverse(provisional, top_k)
    if include_micro and cfg.micro_enabled:
        shortlist = shortlist + diverse(micro_provisional, cfg.micro_top_k)
    selected_candidates = []
    for rank, c in enumerate(shortlist, start=1):
        selected_candidates.append(
            Candidate(
                candidate_id=rank,
                bbox=c.bbox,
                center_x=c.center_x,
                center_y=c.center_y,
                area_px=c.area_px,
                mean_response=c.mean_response,
                peak_response=c.peak_response,
                distance_to_click_px=c.distance_to_click_px,
                spatial_prior=c.spatial_prior,
                detector_strength=c.detector_strength,
                rank_score=c.rank_score,
                size_prior=c.size_prior,
                detection_channel=c.detection_channel,
            )
        )

    return CandidateDetectionResult(
        candidates=tuple(selected_candidates),
        diagnostics=DetectionDiagnostics(
            threshold=threshold,
            median_response=median,
            robust_sigma=robust_sigma,
            raw_components=raw_components,
            valid_components=len(provisional),
            expected_area_px=expected,
            effective_min_area_px=effective_min_area,
            effective_max_area_px=effective_max_area,
            micro_valid_components=len(micro_provisional),
        ),
    )



def _point_to_component_distance(labels: np.ndarray, label_id: int, x: float, y: float) -> float:
    yy, xx = np.nonzero(labels == int(label_id))
    if xx.size == 0:
        return float("inf")
    dx = xx.astype(np.float32) - float(x)
    dy = yy.astype(np.float32) - float(y)
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def _click_seeded_large_detection(
    frame_bgr: np.ndarray,
    selection: OperatorSelection,
    cfg: CandidateDetectorConfig,
) -> tuple[CandidateDetectionResult | None, OperatorSelection]:
    """Raw-thermal fallback for a large target indicated by the operator click.

    Unlike top-hat detection, this fallback segments the absolute thermal
    intensity in an expanded ROI.  It tests two conservative robust thresholds
    and accepts only a *large* connected region whose actual pixels lie close
    to the click.  Thus it can join a warm wing/fuselage silhouette that the
    contrast detector fragments, without changing normal small-target ACQUIRE.
    """
    if not cfg.acquire_click_seed_enabled or cfg.method != "thermal_hot_blob":
        return None, selection

    h, w = frame_bgr.shape[:2]
    radius = max(int(selection.roi_radius_px), int(cfg.acquire_click_seed_roi_radius_px))
    expanded = make_operator_selection(
        frame_index=selection.frame_index,
        click_xy=(selection.click_x, selection.click_y),
        radius_px=radius,
        frame_width=w,
        frame_height=h,
    )
    roi = expanded.roi
    crop = frame_bgr[roi.y:roi.y2, roi.x:roi.x2]
    if crop.size == 0:
        return None, expanded
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Mild smoothing suppresses isolated hot pixels while preserving a large
    # aircraft silhouette.
    gray_s = cv2.GaussianBlur(gray, (0, 0), 1.1, borderType=cv2.BORDER_REPLICATE)
    vals = gray_s.reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    rs = max(1.0, 1.4826 * mad)
    if cfg.thermal_polarity == "dark":
        extreme = float(np.percentile(vals, 5.0))
        contrast = max(4.0, med - extreme, 2.5 * rs)
        thresholds = (med - 0.28 * contrast, med - 0.42 * contrast)
    else:
        extreme = float(np.percentile(vals, 95.0))
        contrast = max(4.0, extreme - med, 2.5 * rs)
        thresholds = (med + 0.28 * contrast, med + 0.42 * contrast)

    close_k = _odd_kernel(int(cfg.acquire_click_seed_close_px))
    kernel = None
    if close_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    click_x_local = float(selection.click_x - roi.x)
    click_y_local = float(selection.click_y - roi.y)
    min_large_area = max(int(cfg.min_area_px), int(round(0.45 * cfg.acquire_large_target_trigger_area_px)))
    max_large_area = min(
        int(cfg.adaptive_hard_max_area_px),
        max(min_large_area, int(round(float(roi.width * roi.height) * cfg.acquire_max_area_fraction_roi))),
    )

    best = None
    best_key = None
    best_diag = None
    for threshold in thresholds:
        if cfg.thermal_polarity == "dark":
            mask = (gray_s <= float(threshold)).astype(np.uint8) * 255
            response = np.maximum(med - gray_s, 0.0)
        else:
            mask = (gray_s >= float(threshold)).astype(np.uint8) * 255
            response = np.maximum(gray_s - med, 0.0)
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label_id in range(1, count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if not (min_large_area <= area <= max_large_area):
                continue
            support_dist = _point_to_component_distance(labels, label_id, click_x_local, click_y_local)
            if support_dist > float(cfg.acquire_click_seed_max_support_distance_px):
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            cx_l, cy_l = centroids[label_id]
            cx = float(roi.x + cx_l)
            cy = float(roi.y + cy_l)
            comp_vals = response[labels == label_id]
            mean_response = float(comp_vals.mean()) if comp_vals.size else 0.0
            peak_response = float(comp_vals.max()) if comp_vals.size else 0.0
            spatial_prior = float(exp(-(support_dist * support_dist) / (2.0 * cfg.spatial_sigma_px * cfg.spatial_sigma_px)))
            detector_strength = float(min(1.0, mean_response / max(0.5 * contrast, 1.0)))
            # Prefer actual click support first, then a coherent larger region.
            key = (support_dist, -area, -mean_response)
            if best_key is None or key < best_key:
                best_key = key
                best = Candidate(
                    candidate_id=1,
                    bbox=RoiRect(x=roi.x + x, y=roi.y + y, width=bw, height=bh),
                    center_x=cx,
                    center_y=cy,
                    area_px=area,
                    mean_response=mean_response,
                    peak_response=peak_response,
                    # For a large object the click need not be near its centroid;
                    # spatial evidence is therefore distance to the segmented
                    # object itself, not to its geometric centre.
                    distance_to_click_px=float(support_dist),
                    spatial_prior=spatial_prior,
                    detector_strength=detector_strength,
                    rank_score=float(0.75 * spatial_prior + 0.25 * detector_strength),
                    size_prior=1.0,
                    detection_channel="ACQUIRE_CLICK_SEEDED",
                )
                best_diag = DetectionDiagnostics(
                    threshold=float(threshold),
                    median_response=med,
                    robust_sigma=rs,
                    raw_components=max(0, count - 1),
                    valid_components=1,
                    expected_area_px=None,
                    effective_min_area_px=min_large_area,
                    effective_max_area_px=max_large_area,
                    micro_valid_components=0,
                )
    if best is None or best_diag is None:
        return None, expanded
    return CandidateDetectionResult(candidates=(best,), diagnostics=best_diag), expanded


def _touches_roi_border(candidate: Candidate, roi: RoiRect, margin_px: int) -> bool:
    m = max(0, int(margin_px))
    b = candidate.bbox
    return bool(
        b.x <= roi.x + m
        or b.y <= roi.y + m
        or b.x2 >= roi.x2 - m
        or b.y2 >= roi.y2 - m
    )


def _point_to_bbox_distance(x: float, y: float, bbox: RoiRect) -> float:
    dx = max(float(bbox.x) - float(x), 0.0, float(x) - float(bbox.x2))
    dy = max(float(bbox.y) - float(y), 0.0, float(y) - float(bbox.y2))
    return float(hypot(dx, dy))


def _as_click_supported_acquire_candidate(candidate: Candidate, selection: OperatorSelection) -> Candidate:
    support = _point_to_bbox_distance(selection.click_x, selection.click_y, candidate.bbox)
    spatial = exp(-(support * support) / (2.0 * 35.0 * 35.0))
    return replace(
        candidate, candidate_id=1, distance_to_click_px=float(support),
        spatial_prior=float(max(candidate.spatial_prior, spatial)),
        detection_channel='ACQUIRE_MULTISCALE_LARGE',
    )


def detect_initial_acquire_candidates(
    frame_bgr: np.ndarray,
    selection: OperatorSelection,
    *,
    top_k: int = 5,
    config: CandidateDetectorConfig | None = None,
) -> tuple[CandidateDetectionResult, OperatorSelection]:
    """Conservative multiscale click-to-object ACQUIRE for THERMAL.

    Pass 1 is byte-for-byte the historical detector path.  A broad-scale pass
    is only allowed to replace it when it finds a coherent *large* component
    near the click (or when pass 1 finds nothing).  If that large component is
    clipped by the click ROI, one expanded-ROI pass is made.  Thus small/normal
    thermal acquisitions keep their previous candidate list and ranking.
    """
    cfg = config or CandidateDetectorConfig()
    first = detect_local_candidates(frame_bgr, selection, top_k=top_k, config=cfg)
    if cfg.method != "thermal_hot_blob" or not cfg.acquire_multiscale_enabled:
        return first, selection

    broad = detect_local_candidates(
        frame_bgr, selection, top_k=top_k, config=cfg, acquire_large_scale_pass=True
    )

    first_best = first.candidates[0] if first.candidates else None
    large_candidates = [
        c for c in broad.candidates
        if float(c.area_px) >= cfg.acquire_large_target_trigger_area_px
        and (
            first_best is None
            or c.distance_to_click_px <= max(35.0, float(first_best.distance_to_click_px) + 12.0)
        )
        and (
            first_best is None
            or float(c.area_px) >= 1.35 * max(1.0, float(first_best.area_px))
        )
    ]
    large = large_candidates[0] if large_candidates else None
    if large is None:
        seeded, seeded_selection = _click_seeded_large_detection(frame_bgr, selection, cfg)
        if seeded is not None:
            return seeded, seeded_selection
        return first, selection

    chosen_large = large
    chosen_diag = broad.diagnostics
    chosen_selection = selection
    if _touches_roi_border(large, selection.roi, cfg.acquire_border_margin_px):
        expanded_radius = max(int(selection.roi_radius_px), int(cfg.acquire_expand_roi_radius_px))
        if expanded_radius > int(selection.roi_radius_px):
            h, w = frame_bgr.shape[:2]
            expanded = make_operator_selection(
                frame_index=selection.frame_index,
                click_xy=(selection.click_x, selection.click_y),
                radius_px=expanded_radius,
                frame_width=w, frame_height=h,
            )
            second = detect_local_candidates(
                frame_bgr, expanded, top_k=max(top_k, 8), config=cfg, acquire_large_scale_pass=True
            )
            second_large = [
                c for c in second.candidates
                if float(c.area_px) >= cfg.acquire_large_target_trigger_area_px
                and _point_to_bbox_distance(selection.click_x, selection.click_y, c.bbox) <= cfg.acquire_click_seed_max_support_distance_px
            ]
            if second_large:
                chosen_large = max(second_large, key=lambda c: (c.area_px, c.mean_response))
                chosen_diag = second.diagnostics
                chosen_selection = expanded
            else:
                seeded, seeded_selection = _click_seeded_large_detection(frame_bgr, selection, cfg)
                if seeded is not None:
                    return seeded, seeded_selection

    # For a coherent large object the operator click is evidence about the
    # object support, not about its centroid. Return this hypothesis explicitly
    # so a wing-tip click cannot be re-ranked below one of its fragments.
    promoted = _as_click_supported_acquire_candidate(chosen_large, selection)
    return CandidateDetectionResult(candidates=(promoted,), diagnostics=chosen_diag), chosen_selection
