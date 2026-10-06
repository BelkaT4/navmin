"""Event-driven background/camera motion estimation for HOTFIX-6.

The normal TRACK path does not pay this cost.  Estimation is requested only
when prediction quality degrades, a candidate is visually strong but far from
its old prediction, or REACQ is operating after a declared MOTION_BREAK.

Primary estimator: low-resolution phase correlation on gradient magnitude.
Fallback: sparse LK optical flow + RANSAC affine-partial transform.  Both are
CPU friendly and work with the standard opencv-python package.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class BackgroundMotionConfig:
    enabled: bool = True
    downscale: float = 0.50
    max_dimension_px: int = 320
    phase_min_response: float = 0.08
    max_translation_px: float = 500.0
    # HOTFIX-6 REV1: a large FFT shift is not trusted directly. It must be
    # confirmed by the feature-based fallback if it exceeds this fraction of
    # the longest image side, or jumps too far from the previous adjacent-frame
    # shift estimate.
    phase_direct_max_fraction: float = 0.30
    phase_max_change_from_previous_px: float = 90.0
    use_feature_fallback: bool = True
    feature_max_corners: int = 180
    feature_quality: float = 0.01
    feature_min_distance_px: float = 7.0
    feature_ransac_threshold_px: float = 3.0
    min_feature_inliers: int = 8
    target_exclusion_radius_px: int = 36

    def validate(self) -> None:
        if not 0.10 <= self.downscale <= 1.0:
            raise ValueError("downscale must be in [0.10,1.0]")
        if self.max_dimension_px < 64:
            raise ValueError("max_dimension_px must be >=64")
        if not 0.0 <= self.phase_min_response <= 1.0:
            raise ValueError("phase_min_response must be in [0,1]")
        if self.max_translation_px <= 0:
            raise ValueError("max_translation_px must be >0")
        if not 0.05 <= self.phase_direct_max_fraction <= 1.0:
            raise ValueError("phase_direct_max_fraction must be in [0.05,1]")
        if self.phase_max_change_from_previous_px <= 0:
            raise ValueError("phase_max_change_from_previous_px must be >0")
        if self.feature_max_corners < 20:
            raise ValueError("feature_max_corners must be >=20")
        if not 0.0 < self.feature_quality < 1.0:
            raise ValueError("feature_quality must be in (0,1)")
        if self.feature_min_distance_px <= 0 or self.feature_ransac_threshold_px <= 0:
            raise ValueError("feature geometry settings must be >0")
        if self.min_feature_inliers < 4:
            raise ValueError("min_feature_inliers must be >=4")
        if self.target_exclusion_radius_px < 0:
            raise ValueError("target_exclusion_radius_px must be >=0")


@dataclass(frozen=True)
class BackgroundMotionEstimate:
    valid: bool
    method: str
    dx_px: float
    dy_px: float
    response: float
    rotation_deg: float = 0.0
    scale: float = 1.0
    affine_2x3: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    phase_time_ms: float = 0.0
    fallback_time_ms: float = 0.0
    total_time_ms: float = 0.0
    feature_inliers: int = 0

    @property
    def translation_norm_px(self) -> float:
        return float((self.dx_px * self.dx_px + self.dy_px * self.dy_px) ** 0.5)


def _gray_gradient(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr
    gray = gray.astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mean, std = cv2.meanStdDev(mag)
    s = float(std.ravel()[0])
    if s > 1e-6:
        mag = (mag - float(mean.ravel()[0])) / s
    return mag.astype(np.float32)


def _effective_scale(shape: tuple[int, int], cfg: BackgroundMotionConfig) -> float:
    h, w = shape
    maxdim = max(h, w)
    scale = min(float(cfg.downscale), float(cfg.max_dimension_px) / max(1.0, float(maxdim)))
    return max(0.10, min(1.0, scale))


def _analysis_mask(
    shape: tuple[int, int],
    *,
    target_xy: tuple[float, float] | None,
    target_radius_px: float,
    reticle_mask: np.ndarray | None,
) -> np.ndarray:
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    if reticle_mask is not None and reticle_mask.shape == (h, w):
        mask[np.asarray(reticle_mask) > 0] = 0
    if target_xy is not None and target_radius_px > 0:
        cv2.circle(mask, (int(round(target_xy[0])), int(round(target_xy[1]))), int(round(target_radius_px)), 0, -1)
    # Avoid borders that can dominate correlation after large pans.
    border = max(2, int(round(0.02 * min(h, w))))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def _prepare_pair(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    cfg: BackgroundMotionConfig,
    *,
    target_xy: tuple[float, float] | None,
    target_radius_px: float,
    reticle_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    prev = _gray_gradient(prev_frame)
    curr = _gray_gradient(curr_frame)
    if prev.shape != curr.shape:
        raise ValueError("background motion frames must have equal size")
    mask = _analysis_mask(prev.shape, target_xy=target_xy, target_radius_px=target_radius_px, reticle_mask=reticle_mask)
    scale = _effective_scale(prev.shape, cfg)
    if scale < 0.999:
        size = (max(32, int(round(prev.shape[1] * scale))), max(32, int(round(prev.shape[0] * scale))))
        prev = cv2.resize(prev, size, interpolation=cv2.INTER_AREA)
        curr = cv2.resize(curr, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    valid = mask > 0
    # Replace masked pixels with zero in normalized gradient space.
    prev = prev.copy(); curr = curr.copy()
    prev[~valid] = 0.0; curr[~valid] = 0.0
    # Hanning suppresses FFT edge effects.
    window = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
    prev *= window
    curr *= window
    return prev, curr, mask, scale


def _phase_estimate(prev: np.ndarray, curr: np.ndarray, scale: float) -> tuple[float, float, float]:
    (dx_small, dy_small), response = cv2.phaseCorrelate(prev, curr)
    inv = 1.0 / max(scale, 1e-9)
    return float(dx_small * inv), float(dy_small * inv), float(response)


def _feature_affine(
    prev: np.ndarray,
    curr: np.ndarray,
    mask: np.ndarray,
    scale: float,
    cfg: BackgroundMotionConfig,
) -> tuple[np.ndarray | None, int]:
    p8 = cv2.normalize(prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c8 = cv2.normalize(curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(
        p8,
        maxCorners=cfg.feature_max_corners,
        qualityLevel=cfg.feature_quality,
        minDistance=cfg.feature_min_distance_px,
        mask=mask,
        blockSize=5,
        useHarrisDetector=False,
    )
    if pts is None or len(pts) < cfg.min_feature_inliers:
        return None, 0
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        p8, c8, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if nxt is None or status is None:
        return None, 0
    good = status.reshape(-1) > 0
    src = pts.reshape(-1, 2)[good]
    dst = nxt.reshape(-1, 2)[good]
    if len(src) < cfg.min_feature_inliers:
        return None, 0
    mat, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=max(1.0, cfg.feature_ransac_threshold_px * scale),
        maxIters=500, confidence=0.98, refineIters=10,
    )
    if mat is None or inliers is None:
        return None, 0
    n_inliers = int(np.count_nonzero(inliers))
    if n_inliers < cfg.min_feature_inliers:
        return None, n_inliers
    # Convert small-image transform to full-resolution coordinate system.
    full = np.asarray(mat, dtype=np.float64).copy()
    full[0, 2] /= max(scale, 1e-9)
    full[1, 2] /= max(scale, 1e-9)
    return full, n_inliers


def estimate_background_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    *,
    config: BackgroundMotionConfig | None = None,
    target_xy: tuple[float, float] | None = None,
    target_radius_px: float | None = None,
    reticle_mask: np.ndarray | None = None,
    previous_shift_xy: tuple[float, float] | None = None,
) -> BackgroundMotionEstimate:
    cfg = config or BackgroundMotionConfig()
    cfg.validate()
    if not cfg.enabled:
        return BackgroundMotionEstimate(False, "OFF", 0.0, 0.0, 0.0)
    t0 = perf_counter()
    radius = float(cfg.target_exclusion_radius_px if target_radius_px is None else target_radius_px)
    prev, curr, mask, scale = _prepare_pair(
        prev_frame, curr_frame, cfg,
        target_xy=target_xy, target_radius_px=radius, reticle_mask=reticle_mask,
    )
    tp = perf_counter()
    dx, dy, response = _phase_estimate(prev, curr, scale)
    phase_ms = (perf_counter() - tp) * 1000.0
    norm = float((dx * dx + dy * dy) ** 0.5)
    h_full, w_full = prev_frame.shape[:2]
    direct_limit = min(float(cfg.max_translation_px), float(cfg.phase_direct_max_fraction) * float(max(h_full, w_full)))
    previous_jump = 0.0
    if previous_shift_xy is not None:
        previous_jump = float(((dx - float(previous_shift_xy[0])) ** 2 + (dy - float(previous_shift_xy[1])) ** 2) ** 0.5)
    phase_plausible = (
        response >= cfg.phase_min_response
        and norm <= direct_limit
        and (previous_shift_xy is None or previous_jump <= cfg.phase_max_change_from_previous_px)
    )
    if phase_plausible:
        affine = ((1.0, 0.0, dx), (0.0, 1.0, dy))
        return BackgroundMotionEstimate(
            True, "PHASE", dx, dy, response,
            affine_2x3=affine,
            phase_time_ms=phase_ms,
            total_time_ms=(perf_counter() - t0) * 1000.0,
        )

    # A suspicious phase result is never applied directly. The more expensive
    # feature/RANSAC path is used only here to confirm a large or discontinuous
    # camera shift.
    fallback_ms = 0.0
    if cfg.use_feature_fallback:
        tf = perf_counter()
        mat, inliers = _feature_affine(prev, curr, mask, scale, cfg)
        fallback_ms = (perf_counter() - tf) * 1000.0
        if mat is not None:
            a, b, tx = mat[0]
            c, d, ty = mat[1]
            sc = float((a * a + c * c) ** 0.5)
            rot = float(degrees(atan2(c, a)))
            norm = float((tx * tx + ty * ty) ** 0.5)
            if norm <= cfg.max_translation_px:
                # A simple inlier fraction is a more interpretable response here.
                response2 = min(1.0, max(response, inliers / max(1.0, float(cfg.feature_max_corners))))
                return BackgroundMotionEstimate(
                    True, "FEATURE_AFFINE", float(tx), float(ty), float(response2),
                    rotation_deg=rot, scale=sc,
                    affine_2x3=((float(a), float(b), float(tx)), (float(c), float(d), float(ty))),
                    phase_time_ms=phase_ms, fallback_time_ms=fallback_ms,
                    total_time_ms=(perf_counter() - t0) * 1000.0,
                    feature_inliers=int(inliers),
                )
    return BackgroundMotionEstimate(
        False, "FAILED", dx, dy, response,
        phase_time_ms=phase_ms, fallback_time_ms=fallback_ms,
        total_time_ms=(perf_counter() - t0) * 1000.0,
    )


def transform_point(point_xy: tuple[float, float], estimate: BackgroundMotionEstimate) -> tuple[float, float]:
    if not estimate.valid or estimate.affine_2x3 is None:
        return float(point_xy[0]), float(point_xy[1])
    m = estimate.affine_2x3
    x, y = float(point_xy[0]), float(point_xy[1])
    return (
        float(m[0][0] * x + m[0][1] * y + m[0][2]),
        float(m[1][0] * x + m[1][1] * y + m[1][2]),
    )
