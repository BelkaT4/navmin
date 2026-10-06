"""Target descriptors used by VT-5 INITIAL TRAIN.

The descriptor is intentionally compact and transparent: normalized grayscale
appearance, edge map, 1-D projections and simple geometry.  It is designed for
reference-bank matching, not for category classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Mapping

import cv2
import numpy as np

from navmin.vision.processors.vt11.features.common import candidate_patch_and_background, clamp01
from navmin.vision.processors.vt11.features.thermal import thermal_feature_metrics


@dataclass(frozen=True)
class TargetDescriptor:
    appearance: np.ndarray
    projection_x: np.ndarray
    projection_y: np.ndarray
    edge_map: np.ndarray
    geometry: np.ndarray
    size_px: float
    bbox: tuple[int, int, int, int]
    # HOTFIX-4 thermal descriptors; always computed, but only weighted in thermal mode.
    thermal_contrast: float = 0.0
    hot_area: float = 0.0
    # VT-11 DAY profile: normalized HSV H×S histogram (illumination-robust colour cue).
    color_hist: np.ndarray | None = None
    # VT-11 HOTFIX-1 richer thermal structure descriptors.
    peak_prominence: float = 0.0
    hot_core_compactness: float = 0.0
    radial_hotspot_profile: np.ndarray | None = None
    energy_concentration: float = 0.0
    thermal_background_clutter: float = 0.0
    thermal_2d_ratio: float = 0.0


@dataclass(frozen=True)
class DescriptorSimilarity:
    appearance: float
    edge: float
    projection: float
    geometry: float
    size: float
    total: float
    thermal_contrast: float = 0.0
    hot_area: float = 0.0
    color: float = 0.0
    peak_prominence: float = 0.0
    hot_core_compactness: float = 0.0
    radial_hotspot_profile: float = 0.0
    energy_concentration: float = 0.0
    thermal_2d_ratio: float = 0.0


def _normalize_image(gray: np.ndarray, size: tuple[int, int] = (16, 16)) -> np.ndarray:
    if gray.size == 0:
        return np.zeros(size[::-1], dtype=np.float32)
    resized = cv2.resize(gray.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    std = float(resized.std())
    return ((resized - mean) / (std + 8.0)).astype(np.float32)


def _unit_vector(values: np.ndarray, length: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, length)
    interp = np.interp(x_new, x_old, values).astype(np.float32)
    norm = float(np.linalg.norm(interp))
    return interp / (norm + 1e-8)


def _color_histogram(frame_bgr: np.ndarray, bbox) -> np.ndarray:
    """Compact HSV histogram used only by the DAY profile.

    Hue and saturation are used instead of value so moderate illumination
    changes do not dominate the descriptor.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return np.zeros(48, dtype=np.float32)
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(int(bbox.x), w - 1)); y1 = max(0, min(int(bbox.y), h - 1))
    x2 = max(x1 + 1, min(int(bbox.x + bbox.width), w)); y2 = max(y1 + 1, min(int(bbox.y + bbox.height), h))
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def _hist_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    # Bhattacharyya distance is [0,1] for normalized non-negative histograms.
    d = float(cv2.compareHist(aa, bb, cv2.HISTCMP_BHATTACHARYYA))
    return clamp01(1.0 - d)


def descriptor_from_candidate(frame_bgr: np.ndarray, candidate) -> TargetDescriptor:
    patch = candidate_patch_and_background(frame_bgr, candidate)
    gray = patch.gray.astype(np.float32)
    app = _normalize_image(gray)

    if gray.size:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
    else:
        edge = np.zeros((1, 1), dtype=np.float32)
    edge_norm = _normalize_image(edge)

    bg_mean = float(np.mean(patch.background)) if patch.background.size else 0.0
    deviation = np.abs(gray - bg_mean) if gray.size else np.zeros((1, 1), dtype=np.float32)
    proj_x = _unit_vector(deviation.sum(axis=0), 16)
    proj_y = _unit_vector(deviation.sum(axis=1), 16)

    b = candidate.bbox
    aspect = float(b.width) / max(1.0, float(b.height))
    fill = float(candidate.area_px) / max(1.0, float(b.width * b.height))
    geom = np.asarray([np.log(max(aspect, 1e-6)), fill], dtype=np.float32)
    thermal = thermal_feature_metrics(gray, patch.background)
    color_hist = _color_histogram(frame_bgr, b)
    return TargetDescriptor(
        appearance=app,
        projection_x=proj_x,
        projection_y=proj_y,
        edge_map=edge_norm,
        geometry=geom,
        size_px=float(max(1, candidate.area_px)),
        bbox=(int(b.x), int(b.y), int(b.width), int(b.height)),
        thermal_contrast=float(thermal.thermal_contrast),
        hot_area=float(thermal.hot_area),
        color_hist=color_hist,
        peak_prominence=float(thermal.peak_prominence),
        hot_core_compactness=float(thermal.hot_core_compactness),
        radial_hotspot_profile=thermal.radial_profile,
        energy_concentration=float(thermal.energy_concentration),
        thermal_background_clutter=float(thermal.background_clutter),
        thermal_2d_ratio=float(thermal.spatial_2d_ratio),
    )


def _rmse_similarity(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    rmse = float(np.sqrt(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2)))
    return clamp01(exp(-rmse / max(1e-6, scale)))


def descriptor_similarity(
    ref: TargetDescriptor,
    cand: TargetDescriptor,
    *,
    identity_weights: Mapping[str, float] | None = None,
) -> DescriptorSimilarity:
    app = _rmse_similarity(ref.appearance, cand.appearance, 0.85)
    edge = _rmse_similarity(ref.edge_map, cand.edge_map, 0.95)
    px = _rmse_similarity(ref.projection_x, cand.projection_x, 0.28)
    py = _rmse_similarity(ref.projection_y, cand.projection_y, 0.28)
    projection = 0.5 * (px + py)
    geometry = _rmse_similarity(ref.geometry, cand.geometry, 0.65)
    size_ratio = abs(np.log(max(cand.size_px, 1.0) / max(ref.size_px, 1.0)))
    size = clamp01(exp(-size_ratio / 0.55))

    thermal_contrast = clamp01(exp(-abs(float(ref.thermal_contrast) - float(cand.thermal_contrast)) / 0.22))
    hot_area = clamp01(exp(-abs(float(ref.hot_area) - float(cand.hot_area)) / 0.18))
    color = _hist_similarity(ref.color_hist, cand.color_hist)
    peak_prominence = clamp01(exp(-abs(float(ref.peak_prominence) - float(cand.peak_prominence)) / 0.20))
    hot_core_compactness = clamp01(exp(-abs(float(ref.hot_core_compactness) - float(cand.hot_core_compactness)) / 0.22))
    radial_hotspot_profile = _rmse_similarity(
        np.zeros(6, np.float32) if ref.radial_hotspot_profile is None else ref.radial_hotspot_profile,
        np.zeros(6, np.float32) if cand.radial_hotspot_profile is None else cand.radial_hotspot_profile,
        0.24,
    )
    energy_concentration = clamp01(exp(-abs(float(ref.energy_concentration) - float(cand.energy_concentration)) / 0.20))
    thermal_2d_ratio = clamp01(exp(-abs(float(ref.thermal_2d_ratio) - float(cand.thermal_2d_ratio)) / 0.22))

    weights = dict(identity_weights or {"appearance": 0.603654, "edge": 0.288078, "projection": 0.108268})

    # VT-11 THERMAL profile: as the target shrinks, fine silhouette/edge
    # descriptors become unstable.  Shift weight toward thermal statistics
    # instead of globally loosening identity thresholds.
    if float(weights.get("_thermal_scale_adaptive", 0.0)) > 0.0:
        area = max(1.0, float(cand.size_px))
        micro_t = max(1.0, float(weights.get("_micro_area_threshold_px", 80.0)))
        small_t = max(micro_t, float(weights.get("_small_area_threshold_px", 240.0)))
        if area <= micro_t:
            shape_factor = float(weights.get("_micro_shape_factor", 0.25))
            thermal_boost = float(weights.get("_micro_thermal_boost", 1.35))
        elif area <= small_t:
            shape_factor = float(weights.get("_small_shape_factor", 0.60))
            thermal_boost = float(weights.get("_small_thermal_boost", 1.15))
        else:
            shape_factor = 1.0
            thermal_boost = 1.0
        for key in ("appearance", "edge", "projection", "geometry", "thermal_2d_ratio"):
            if key in weights:
                weights[key] = max(0.0, float(weights[key])) * shape_factor
        for key in ("thermal_contrast", "hot_area", "peak_prominence", "hot_core_compactness", "radial_hotspot_profile", "energy_concentration"):
            if key in weights:
                weights[key] = max(0.0, float(weights[key])) * thermal_boost

    score_map = {
        "appearance": app, "edge": edge, "projection": projection,
        "geometry": geometry, "color": color,
        "thermal_contrast": thermal_contrast, "hot_area": hot_area,
        "peak_prominence": peak_prominence, "hot_core_compactness": hot_core_compactness,
        "radial_hotspot_profile": radial_hotspot_profile, "energy_concentration": energy_concentration,
        "thermal_2d_ratio": thermal_2d_ratio,
    }
    names = tuple(name for name in score_map if max(0.0, float(weights.get(name, 0.0))) > 0.0)
    denom = sum(max(0.0, float(weights.get(n, 0.0))) for n in names)
    if denom <= 0:
        raise ValueError("identity weights need at least one positive supported feature")
    total = sum(max(0.0, float(weights.get(n, 0.0))) * score_map[n] for n in names) / denom
    return DescriptorSimilarity(
        appearance=float(app),
        edge=float(edge),
        projection=float(projection),
        geometry=float(geometry),
        size=float(size),
        total=float(clamp01(total)),
        thermal_contrast=float(thermal_contrast),
        hot_area=float(hot_area),
        color=float(color),
        peak_prominence=float(peak_prominence),
        hot_core_compactness=float(hot_core_compactness),
        radial_hotspot_profile=float(radial_hotspot_profile),
        energy_concentration=float(energy_concentration),
        thermal_2d_ratio=float(thermal_2d_ratio),
    )
