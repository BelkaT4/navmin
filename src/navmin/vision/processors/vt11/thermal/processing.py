"""VT-10 HOTFIX-4 thermal-specific preprocessing.

The universal visible-light pipeline remains unchanged.  For thermal imagery we
learn a *static overlay mask* (reticle/HUD) from a short frame stack, remove the
overlay only from the analysis copy using inpainting, and switch candidate
extraction to a polarity-aware hot-blob detector.

The mask is intentionally a set of stable high-contrast pixels, not a forbidden
circular zone.  Therefore a moving target may still pass through the reticle
area; only the fixed graphics themselves are suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Sequence

import cv2
import numpy as np


VIDEO_MODES = ("auto", "visible", "thermal")
RETICLE_MODES = ("off", "auto", "manual")


@dataclass(frozen=True)
class ThermalProcessingConfig:
    video_mode: str = "auto"
    # REV1 master gate.  It deliberately separates "how to locate HUD" from
    # "whether HUD suppression is allowed at all".  Default OFF.
    hud_suppression_enabled: bool = False
    reticle_mode: str = "auto"
    auto_channel_corr_min: float = 0.997
    auto_channel_spread_p90_max: float = 35.0
    auto_reticle_search_fraction: float = 0.34
    static_contrast_min: float = 6.0
    static_contrast_percentile: float = 98.5
    static_std_max: float = 3.0
    static_dilate_px: int = 3
    static_max_component_area_px: int = 1200
    manual_center_x: float = -1.0
    manual_center_y: float = -1.0
    manual_search_radius_px: int = 110
    inpaint_radius_px: int = 3
    thermal_top_hat_kernel_px: int = 21
    thermal_top_hat_scale: float = 2.2
    thermal_top_hat_max_px: int = 81
    static_overlay_penalty_weight: float = 0.25

    # VT-11 HOTFIX-3: explicit negative HUD/reticle evidence.  The learned
    # static mask is no longer merely a weak motion-dependent penalty: it also
    # defines a hard nuisance core and a soft guard band around known overlays.
    hud_negative_enabled: bool = False
    hud_soft_dilate_px: int = 15
    hud_negative_penalty_weight: float = 0.42
    hud_hard_overlap_ratio: float = 0.08
    hud_soft_overlap_ratio: float = 0.10

    def validate(self) -> None:
        if self.video_mode not in VIDEO_MODES:
            raise ValueError(f"video_mode must be one of {VIDEO_MODES}")
        if self.reticle_mode not in RETICLE_MODES:
            raise ValueError(f"reticle_mode must be one of {RETICLE_MODES}")
        if not 0.0 < self.auto_channel_corr_min <= 1.0:
            raise ValueError("auto_channel_corr_min must be in (0,1]")
        if self.auto_channel_spread_p90_max <= 0:
            raise ValueError("auto_channel_spread_p90_max must be > 0")
        if not 0.05 <= self.auto_reticle_search_fraction <= 1.0:
            raise ValueError("auto_reticle_search_fraction must be in [0.05,1]")
        if self.static_contrast_min < 0 or not 0.0 < self.static_contrast_percentile < 100.0:
            raise ValueError("static contrast settings are invalid")
        if self.static_std_max < 0 or self.static_dilate_px < 0:
            raise ValueError("static mask settings are invalid")
        if self.static_max_component_area_px < 1:
            raise ValueError("static_max_component_area_px must be >=1")
        if self.manual_search_radius_px < 5 or self.inpaint_radius_px < 1:
            raise ValueError("manual_search_radius_px/inpaint_radius_px invalid")
        if self.thermal_top_hat_kernel_px < 3 or self.thermal_top_hat_max_px < self.thermal_top_hat_kernel_px:
            raise ValueError("thermal top-hat kernel settings invalid")
        if self.thermal_top_hat_scale <= 0:
            raise ValueError("thermal_top_hat_scale must be >0")
        if not 0.0 <= self.static_overlay_penalty_weight <= 1.0:
            raise ValueError("static_overlay_penalty_weight must be in [0,1]")
        if self.hud_soft_dilate_px < 0:
            raise ValueError("hud_soft_dilate_px must be >=0")
        for name in ("hud_negative_penalty_weight", "hud_hard_overlap_ratio", "hud_soft_overlap_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class ThermalProfile:
    video_mode: str
    reticle_mode: str
    reticle_mask: np.ndarray
    static_centers: tuple[tuple[float, float], ...]
    target_polarity: str = "bright"  # bright | dark
    auto_score_corr: float = 0.0
    auto_score_spread_p90: float = 0.0
    static_overlay_penalty_weight: float = 0.25

    # VT-11 HOTFIX-3: explicit negative HUD/reticle evidence.  The learned
    # static mask is no longer merely a weak motion-dependent penalty: it also
    # defines a hard nuisance core and a soft guard band around known overlays.
    hud_negative_enabled: bool = False
    hud_hard_mask: np.ndarray | None = None
    hud_soft_mask: np.ndarray | None = None
    hud_soft_dilate_px: int = 15
    hud_negative_penalty_weight: float = 0.42
    hud_hard_overlap_ratio: float = 0.08
    hud_soft_overlap_ratio: float = 0.10
    thermal_top_hat_kernel_px: int = 21
    thermal_top_hat_scale: float = 2.2
    thermal_top_hat_max_px: int = 81
    inpaint_radius_px: int = 3

    @property
    def is_thermal(self) -> bool:
        return self.video_mode == "thermal"

    @property
    def masked_pixels(self) -> int:
        return int(np.count_nonzero(self.reticle_mask))

    def summary(self) -> dict:
        return {
            "video_mode": self.video_mode,
            "reticle_mode": self.reticle_mode,
            "masked_pixels": self.masked_pixels,
            "static_centers": [[float(x), float(y)] for x, y in self.static_centers],
            "target_polarity": self.target_polarity,
            "auto_score_corr": float(self.auto_score_corr),
            "auto_score_spread_p90": float(self.auto_score_spread_p90),
            "static_overlay_penalty_weight": float(self.static_overlay_penalty_weight),
            "hud_negative_enabled": bool(self.hud_negative_enabled),
            "hud_negative_penalty_weight": float(self.hud_negative_penalty_weight),
        }


def config_from_dict(data: dict | None) -> ThermalProcessingConfig:
    data = data or {}
    cfg = ThermalProcessingConfig(
        video_mode=str(data.get("video_mode", "auto")),
        hud_suppression_enabled=bool(data.get("hud_suppression_enabled", False)),
        reticle_mode=str(data.get("reticle_mode", "auto")),
        auto_channel_corr_min=float(data.get("auto_channel_corr_min", 0.997)),
        auto_channel_spread_p90_max=float(data.get("auto_channel_spread_p90_max", 35.0)),
        auto_reticle_search_fraction=float(data.get("auto_reticle_search_fraction", 0.34)),
        static_contrast_min=float(data.get("static_contrast_min", 6.0)),
        static_contrast_percentile=float(data.get("static_contrast_percentile", 98.5)),
        static_std_max=float(data.get("static_std_max", 3.0)),
        static_dilate_px=int(data.get("static_dilate_px", 3)),
        static_max_component_area_px=int(data.get("static_max_component_area_px", 1200)),
        manual_center_x=float(data.get("manual_center_x", -1.0)),
        manual_center_y=float(data.get("manual_center_y", -1.0)),
        manual_search_radius_px=int(data.get("manual_search_radius_px", 110)),
        inpaint_radius_px=int(data.get("inpaint_radius_px", 3)),
        thermal_top_hat_kernel_px=int(data.get("thermal_top_hat_kernel_px", 21)),
        thermal_top_hat_scale=float(data.get("thermal_top_hat_scale", 2.2)),
        thermal_top_hat_max_px=int(data.get("thermal_top_hat_max_px", 81)),
        static_overlay_penalty_weight=float(data.get("static_overlay_penalty_weight", 0.25)),
        hud_negative_enabled=bool(data.get("hud_negative_enabled", False)),
        hud_soft_dilate_px=int(data.get("hud_soft_dilate_px", 15)),
        hud_negative_penalty_weight=float(data.get("hud_negative_penalty_weight", 0.42)),
        hud_hard_overlap_ratio=float(data.get("hud_hard_overlap_ratio", 0.08)),
        hud_soft_overlap_ratio=float(data.get("hud_soft_overlap_ratio", 0.10)),
    )
    cfg.validate()
    return cfg


def _mode_scores(frame_bgr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(frame_bgr, dtype=np.float32).reshape(-1, 3)
    if len(arr) < 16:
        return 0.0, 999.0
    # Subsample deterministically for speed.
    step = max(1, len(arr) // 20000)
    arr = arr[::step]
    corr = np.corrcoef(arr.T)
    offdiag = [corr[0, 1], corr[0, 2], corr[1, 2]]
    corr_score = float(np.nanmin(offdiag)) if np.isfinite(offdiag).all() else 0.0
    spread = arr.max(axis=1) - arr.min(axis=1)
    return corr_score, float(np.percentile(spread, 90.0))


def infer_video_mode(frames: Sequence[np.ndarray], config: ThermalProcessingConfig | None = None) -> str:
    cfg = config or ThermalProcessingConfig()
    cfg.validate()
    if cfg.video_mode in {"visible", "thermal"}:
        return cfg.video_mode
    valid = [f for f in frames if isinstance(f, np.ndarray) and f.ndim == 3 and f.shape[2] == 3 and f.size]
    if not valid:
        return "visible"
    scores = [_mode_scores(f) for f in valid[:5]]
    corr = float(np.median([s[0] for s in scores]))
    spread90 = float(np.median([s[1] for s in scores]))
    # Monochrome thermal cameras frequently encode gray level with a fixed RGB
    # offset/tint: channel correlation remains almost perfect while channel
    # spread stays bounded.  Visible imagery rarely satisfies both conditions.
    return "thermal" if corr >= cfg.auto_channel_corr_min and spread90 <= cfg.auto_channel_spread_p90_max else "visible"


def _target_polarity(frame_bgr: np.ndarray, candidate) -> str:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = candidate.bbox
    x1, y1, x2, y2 = int(b.x), int(b.y), int(b.x + b.width), int(b.y + b.height)
    obj = gray[max(0, y1):min(gray.shape[0], y2), max(0, x1):min(gray.shape[1], x2)]
    pad = max(4, int(round(max(b.width, b.height) * 0.75)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(gray.shape[1], x2 + pad), min(gray.shape[0], y2 + pad)
    ring = gray[ry1:ry2, rx1:rx2].copy()
    if ring.size == 0 or obj.size == 0:
        return "bright"
    mask = np.ones(ring.shape, dtype=bool)
    ox1, oy1 = x1 - rx1, y1 - ry1
    ox2, oy2 = ox1 + max(0, x2 - x1), oy1 + max(0, y2 - y1)
    mask[max(0, oy1):min(mask.shape[0], oy2), max(0, ox1):min(mask.shape[1], ox2)] = False
    bg = ring[mask]
    return "bright" if float(np.mean(obj)) >= (float(np.mean(bg)) if bg.size else float(np.mean(obj))) else "dark"


def _search_region(shape: tuple[int, int], cfg: ThermalProcessingConfig) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    if cfg.reticle_mode == "manual":
        cx = cfg.manual_center_x if cfg.manual_center_x >= 0 else (w - 1) / 2.0
        cy = cfg.manual_center_y if cfg.manual_center_y >= 0 else (h - 1) / 2.0
        r = float(cfg.manual_search_radius_px)
    else:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = float(cfg.auto_reticle_search_fraction) * min(h, w)
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r)


def _static_mask(frames: Sequence[np.ndarray], cfg: ThermalProcessingConfig, initial_candidate=None) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    if not frames or not cfg.hud_suppression_enabled or cfg.reticle_mode == "off":
        if not frames:
            return np.zeros((1, 1), dtype=np.uint8), ()
        h, w = frames[0].shape[:2]
        return np.zeros((h, w), dtype=np.uint8), ()
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames], axis=0)
    median_img = np.median(grays, axis=0).astype(np.float32)
    temporal_std = np.std(grays, axis=0)
    local = cv2.GaussianBlur(median_img, (0, 0), sigmaX=3.0, sigmaY=3.0, borderType=cv2.BORDER_REPLICATE)
    response = np.abs(median_img - local)
    region = _search_region(median_img.shape, cfg)
    vals = response[region]
    percentile = float(np.percentile(vals, cfg.static_contrast_percentile)) if vals.size else cfg.static_contrast_min
    threshold = max(float(cfg.static_contrast_min), percentile)
    raw = ((response >= threshold) & (temporal_std <= cfg.static_std_max) & region).astype(np.uint8) * 255
    # Never learn the operator-selected target itself as static HUD.  Only a
    # modest dilation of its first bbox is excluded; the rest of the central
    # search region remains available for reticle discovery.
    if initial_candidate is not None:
        b = initial_candidate.bbox
        pad = max(3, int(round(0.5 * max(b.width, b.height))))
        x1, y1 = max(0, int(b.x) - pad), max(0, int(b.y) - pad)
        x2 = min(raw.shape[1], int(b.x + b.width) + pad)
        y2 = min(raw.shape[0], int(b.y + b.height) + pad)
        raw[y1:y2, x1:x2] = 0
    if cfg.static_dilate_px > 0:
        k = cfg.static_dilate_px if cfg.static_dilate_px % 2 == 1 else cfg.static_dilate_px + 1
        k = max(3, k)
        raw = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(raw, connectivity=8)
    clean = np.zeros_like(raw)
    centers: list[tuple[float, float]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 2 <= area <= cfg.static_max_component_area_px:
            clean[labels == i] = 255
            centers.append((float(cents[i][0]), float(cents[i][1])))
    return clean, tuple(centers)


def build_thermal_profile(
    frames: Sequence[np.ndarray],
    *,
    config: ThermalProcessingConfig | None = None,
    initial_candidate=None,
) -> ThermalProfile:
    cfg = config or ThermalProcessingConfig()
    cfg.validate()
    if not frames:
        raise ValueError("At least one frame is required to build thermal profile")
    mode = infer_video_mode(frames, cfg)
    corr_scores = [_mode_scores(f)[0] for f in frames[:5]]
    spread_scores = [_mode_scores(f)[1] for f in frames[:5]]
    h, w = frames[0].shape[:2]
    if mode == "thermal":
        mask, centers = _static_mask(frames, cfg, initial_candidate=initial_candidate)
        polarity = _target_polarity(frames[0], initial_candidate) if initial_candidate is not None else "bright"
    else:
        mask, centers, polarity = np.zeros((h, w), dtype=np.uint8), (), "bright"
    if mode == "thermal" and cfg.hud_suppression_enabled and cfg.hud_negative_enabled and mask.size:
        hud_hard_mask = mask.copy()
        if cfg.hud_soft_dilate_px > 0:
            k = 2 * int(cfg.hud_soft_dilate_px) + 1
            hud_soft_mask = cv2.dilate(hud_hard_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        else:
            hud_soft_mask = hud_hard_mask.copy()
    else:
        hud_hard_mask = np.zeros((h, w), dtype=np.uint8)
        hud_soft_mask = np.zeros((h, w), dtype=np.uint8)

    return ThermalProfile(
        video_mode=mode,
        reticle_mode=(cfg.reticle_mode if (mode == "thermal" and cfg.hud_suppression_enabled) else "off"),
        reticle_mask=mask,
        static_centers=centers,
        target_polarity=polarity,
        auto_score_corr=float(np.median(corr_scores)) if corr_scores else 0.0,
        auto_score_spread_p90=float(np.median(spread_scores)) if spread_scores else 0.0,
        static_overlay_penalty_weight=float(cfg.static_overlay_penalty_weight),
        hud_negative_enabled=bool(cfg.hud_suppression_enabled and cfg.hud_negative_enabled and mode == "thermal"),
        hud_hard_mask=hud_hard_mask,
        hud_soft_mask=hud_soft_mask,
        hud_negative_penalty_weight=float(cfg.hud_negative_penalty_weight),
        hud_hard_overlap_ratio=float(cfg.hud_hard_overlap_ratio),
        hud_soft_overlap_ratio=float(cfg.hud_soft_overlap_ratio),
        thermal_top_hat_kernel_px=int(cfg.thermal_top_hat_kernel_px),
        thermal_top_hat_scale=float(cfg.thermal_top_hat_scale),
        thermal_top_hat_max_px=int(cfg.thermal_top_hat_max_px),
        inpaint_radius_px=int(cfg.inpaint_radius_px),
    )


def preprocess_frame(frame_bgr: np.ndarray, profile: ThermalProfile | None) -> np.ndarray:
    if profile is None or not profile.is_thermal or profile.masked_pixels == 0:
        return frame_bgr
    if profile.reticle_mask.shape != frame_bgr.shape[:2]:
        return frame_bgr
    # Inpainting is applied only to the analysis copy. The displayed/raw frame
    # remains untouched for diagnostics and operator inspection.
    return cv2.inpaint(frame_bgr, profile.reticle_mask, profile.inpaint_radius_px, cv2.INPAINT_TELEA)


def reticle_overlap_ratio(candidate, profile: ThermalProfile | None) -> float:
    if profile is None or not profile.is_thermal or profile.masked_pixels == 0:
        return 0.0
    b = candidate.bbox
    h, w = profile.reticle_mask.shape
    x1, y1 = max(0, int(b.x)), max(0, int(b.y))
    x2, y2 = min(w, int(b.x + b.width)), min(h, int(b.y + b.height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = profile.reticle_mask[y1:y2, x1:x2]
    return float(np.count_nonzero(patch)) / max(1.0, float(patch.size))



@dataclass(frozen=True)
class HudEvidence:
    """Negative evidence that a candidate is a learned static HUD/reticle object."""

    hard_overlap: float = 0.0
    soft_overlap: float = 0.0
    center_in_hard: bool = False
    center_in_soft: bool = False
    probability: float = 0.0
    hard_reject: bool = False


def _mask_overlap(candidate, mask: np.ndarray | None) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    b = candidate.bbox
    h, w = mask.shape
    x1, y1 = max(0, int(b.x)), max(0, int(b.y))
    x2, y2 = min(w, int(b.x + b.width)), min(h, int(b.y + b.height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(patch)) / max(1.0, float(patch.size))


def hud_point_membership(xy: tuple[float, float], profile: ThermalProfile | None) -> tuple[bool, bool]:
    """Return (inside hard HUD core, inside soft guard band) for one image point."""
    if profile is None or not profile.is_thermal or not profile.hud_negative_enabled:
        return False, False
    hard = profile.hud_hard_mask
    soft = profile.hud_soft_mask
    if hard is None or soft is None or hard.size == 0 or soft.size == 0:
        return False, False
    x = int(round(float(xy[0]))); y = int(round(float(xy[1])))
    h, w = hard.shape
    if not (0 <= x < w and 0 <= y < h):
        return False, False
    return bool(hard[y, x] > 0), bool(soft[y, x] > 0)


def hud_evidence(candidate, profile: ThermalProfile | None) -> HudEvidence:
    """Compute an anti-target score from the learned static HUD geometry.

    Unlike the old ``static_overlay_penalty``, this evidence is intentionally
    independent of target-motion agreement.  Once a tracker has accidentally
    locked to a reticle, a high motion score must not erase the knowledge that
    the candidate lies on a known static overlay.
    """
    if profile is None or not profile.is_thermal or not profile.hud_negative_enabled:
        return HudEvidence()
    hard = profile.hud_hard_mask
    soft = profile.hud_soft_mask
    if hard is None or soft is None or hard.size == 0 or soft.size == 0:
        return HudEvidence()
    hard_overlap = _mask_overlap(candidate, hard)
    soft_overlap = _mask_overlap(candidate, soft)
    hard_center, soft_center = hud_point_membership(candidate.center, profile)
    # Center membership is strong evidence for thin reticle strokes, where bbox
    # overlap alone can be deceptively small.  Soft membership is weaker.
    probability = max(
        min(1.0, hard_overlap / max(1e-9, float(profile.hud_hard_overlap_ratio))),
        1.0 if hard_center else 0.0,
        0.70 * min(1.0, soft_overlap / max(1e-9, float(profile.hud_soft_overlap_ratio))),
        0.55 if soft_center else 0.0,
    )
    hard_reject = bool(
        hard_center
        or hard_overlap >= float(profile.hud_hard_overlap_ratio)
    )
    return HudEvidence(
        hard_overlap=float(hard_overlap), soft_overlap=float(soft_overlap),
        center_in_hard=bool(hard_center), center_in_soft=bool(soft_center),
        probability=float(min(1.0, max(0.0, probability))), hard_reject=hard_reject,
    )

def runtime_detector_config(detector_config, profile: ThermalProfile | None):
    """Return a detector config adjusted for the resolved video mode."""
    if profile is None or not profile.is_thermal:
        return replace(detector_config, method="local_contrast_components")
    return replace(
        detector_config,
        method="thermal_hot_blob",
        thermal_polarity=profile.target_polarity,
        thermal_top_hat_kernel_px=profile.thermal_top_hat_kernel_px,
        thermal_top_hat_scale=profile.thermal_top_hat_scale,
        thermal_top_hat_max_px=profile.thermal_top_hat_max_px,
    )
