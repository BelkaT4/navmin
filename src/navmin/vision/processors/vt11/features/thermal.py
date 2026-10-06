"""Transparent thermal-domain features.

VT-11 HOTFIX-1 extends the original two scalar thermal features with structure
cues that remain useful when a target changes projection or becomes very small.
All computations are local to the candidate patch; no neural network or second
full-frame pass is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np

from navmin.vision.processors.vt11.features.common import clamp01


@dataclass(frozen=True)
class ThermalFeatureSet:
    thermal_contrast: float
    hot_area: float
    peak_prominence: float
    hot_core_compactness: float
    radial_profile: np.ndarray
    energy_concentration: float
    # HOTFIX-2: robust heterogeneity of the immediate background ring.
    # Used only as a soft far-jump penalty, not as a hard identity feature.
    background_clutter: float = 0.0
    # HOTFIX-4: compact two-dimensionality of the hot/cold target support.
    # 0 -> line-like, 1 -> isotropic/2-D. It is an identity cue, not a hard filter.
    spatial_2d_ratio: float = 0.0


def _background_stats(obj: np.ndarray, background: np.ndarray) -> tuple[float, float]:
    bg = np.asarray(background, dtype=np.float32).reshape(-1)
    if bg.size == 0:
        return float(np.mean(obj)), 8.0
    return float(np.mean(bg)), float(np.std(bg))


def _radial_energy_profile(energy: np.ndarray, bins: int = 6) -> np.ndarray:
    """Scale-normalised radial distribution of polarity-aligned thermal energy."""
    e = np.asarray(energy, dtype=np.float32)
    if e.size == 0 or float(e.sum()) <= 1e-9:
        return np.zeros(bins, dtype=np.float32)
    h, w = e.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    total = float(e.sum())
    cx = float((e * xx).sum() / total)
    cy = float((e * yy).sum() / total)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = float(rr.max())
    if rmax <= 1e-6:
        out = np.zeros(bins, dtype=np.float32)
        out[0] = 1.0
        return out
    rn = np.clip(rr / rmax, 0.0, 1.0)
    out = np.zeros(bins, dtype=np.float32)
    # Last edge includes 1.0.
    idx = np.minimum(bins - 1, (rn * bins).astype(np.int32))
    for i in range(bins):
        out[i] = float(e[idx == i].sum())
    s = float(out.sum())
    if s > 0:
        out /= s
    return out


def _compactness(mask: np.ndarray) -> float:
    """4*pi*A/P^2 of the dominant hot core, bounded to [0,1]."""
    m = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(m)) < 3:
        return 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(max(cv2.contourArea(c), 0.0))
    perimeter = float(cv2.arcLength(c, True))
    if area <= 0.0 or perimeter <= 1e-6:
        return 0.0
    return clamp01(4.0 * pi * area / (perimeter * perimeter))


def thermal_feature_metrics(gray: np.ndarray, background: np.ndarray) -> ThermalFeatureSet:
    """Return local thermal descriptors for one candidate patch.

    The signs of bright/dark objects are handled automatically.  Except for the
    legacy signed ``thermal_contrast``, the added quantities are [0,1].
    """
    if gray.size == 0:
        return ThermalFeatureSet(0.0, 0.0, 0.0, 0.0, np.zeros(6, np.float32), 0.0, 0.0, 0.0)

    obj2 = np.asarray(gray, dtype=np.float32)
    obj = obj2.reshape(-1)
    bg_mean, bg_std = _background_stats(obj, background)
    obj_mean = float(np.mean(obj))
    scale = max(5.0, bg_std)
    signed_z = (obj_mean - bg_mean) / scale
    thermal_contrast = float(np.tanh(signed_z / 3.0))

    polarity = 1.0 if obj_mean >= bg_mean else -1.0
    aligned = polarity * (obj2 - bg_mean)
    threshold_delta = max(4.0, 1.25 * bg_std)
    hot_mask = aligned >= threshold_delta
    hot_area = clamp01(float(np.mean(hot_mask)))

    # Robust peak rather than max: less sensitive to one defective pixel.
    q95 = float(np.percentile(aligned, 95.0))
    peak_prominence = clamp01(float(np.tanh(max(0.0, q95) / (4.0 * scale))))

    hot_core_compactness = _compactness(hot_mask)

    # HOTFIX-4: one generic structural scalar instead of a catalogue of
    # artefact-specific filters.  PCA of the thresholded thermal support gives
    # lambda_min/lambda_max: thin stripes approach 0, genuinely two-dimensional
    # supports are larger.  Small supports are left neutral at 0 to avoid noise.
    yy_hot, xx_hot = np.nonzero(hot_mask)
    if xx_hot.size < 4:
        spatial_2d_ratio = 0.0
    else:
        pts = np.column_stack((xx_hot.astype(np.float32), yy_hot.astype(np.float32)))
        pts -= pts.mean(axis=0, keepdims=True)
        cov = (pts.T @ pts) / max(1.0, float(len(pts) - 1))
        eig = np.linalg.eigvalsh(cov)
        lmin = max(0.0, float(eig[0])); lmax = max(lmin, float(eig[-1]))
        spatial_2d_ratio = clamp01(lmin / max(1e-9, lmax))

    # Energy is polarity-aligned and only positive deviations contribute.
    energy = np.maximum(aligned, 0.0)
    radial_profile = _radial_energy_profile(energy, bins=6)
    total_energy = float(energy.sum())
    if total_energy <= 1e-9:
        energy_concentration = 0.0
    else:
        # Fraction of energy above the background-derived core threshold. Broad
        # warm regions tend to be lower than compact point-like hot targets.
        core_energy = float(np.maximum(aligned - threshold_delta, 0.0).sum())
        energy_concentration = clamp01(core_energy / total_energy)

    bg = np.asarray(background, dtype=np.float32).reshape(-1)
    if bg.size < 4:
        background_clutter = 0.0
    else:
        med = float(np.median(bg))
        mad = float(np.median(np.abs(bg - med)))
        q10, q90 = np.percentile(bg, (10.0, 90.0))
        robust_sigma = max(1.4826 * mad, float(q90 - q10) / 2.563)
        background_clutter = clamp01(float(np.tanh(robust_sigma / 18.0)))

    return ThermalFeatureSet(
        thermal_contrast=float(thermal_contrast),
        hot_area=float(hot_area),
        peak_prominence=float(peak_prominence),
        hot_core_compactness=float(hot_core_compactness),
        radial_profile=radial_profile.astype(np.float32),
        energy_concentration=float(energy_concentration),
        background_clutter=float(background_clutter),
        spatial_2d_ratio=float(spatial_2d_ratio),
    )


def thermal_metrics(gray: np.ndarray, background: np.ndarray) -> tuple[float, float]:
    """Backward-compatible HOTFIX-4 API: ``(thermal_contrast, hot_area)``."""
    f = thermal_feature_metrics(gray, background)
    return f.thermal_contrast, f.hot_area
