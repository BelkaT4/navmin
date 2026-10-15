"""Lightweight thermal frame-reliability estimator (VT-11 HOTFIX-4).

The estimator is deliberately generic: it does not search for a specific
horizontal stripe, reticle or other named artefact.  Instead it measures how
well the current frame can be explained by a globally translated, robustly
normalised reference frame.  Sudden vibration artefacts, smear, AGC glitches
and other whole-frame degradations tend to leave a large residual even after
translation compensation.

Only a small downscaled grayscale image is processed, so the normal TRACK path
is not burdened by another full-resolution detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameReliabilityConfig:
    enabled: bool = False
    max_dimension_px: int = 128
    residual_scale: float = 0.75
    phase_response_floor: float = 0.02
    phase_response_good: float = 0.50
    phase_weight: float = 0.15
    enter_threshold: float = 0.55
    exit_threshold: float = 0.72
    max_hold_frames: int = 5

    def validate(self) -> None:
        if self.max_dimension_px < 64:
            raise ValueError("frame reliability max_dimension_px must be >=64")
        if self.residual_scale <= 0:
            raise ValueError("frame reliability residual_scale must be >0")
        if not 0.0 <= self.phase_response_floor < self.phase_response_good <= 1.0:
            raise ValueError("frame reliability phase response thresholds invalid")
        if not 0.0 <= self.phase_weight <= 1.0:
            raise ValueError("frame reliability phase_weight must be in [0,1]")
        if not 0.0 <= self.enter_threshold < self.exit_threshold <= 1.0:
            raise ValueError("frame reliability enter/exit thresholds invalid")
        if self.max_hold_frames < 1:
            raise ValueError("frame reliability max_hold_frames must be >=1")


@dataclass(frozen=True)
class FrameReliabilityResult:
    score: float = 1.0
    residual: float = 0.0
    phase_response: float = 1.0
    shift_dx_px: float = 0.0
    shift_dy_px: float = 0.0
    valid: bool = False


def _gray_downscaled(frame: np.ndarray, max_dimension_px: int) -> tuple[np.ndarray, float]:
    # Resize before grayscale conversion: on 1024x768 thermal recordings this
    # avoids a needless full-resolution colour conversion and keeps Q_frame
    # comfortably cheaper than the candidate detector.
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_dimension_px) / max(1.0, float(max(h, w))))
    if scale < 0.999:
        nw = max(16, int(round(w * scale)))
        nh = max(16, int(round(h * scale)))
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = np.asarray(arr, dtype=np.float32)
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med)))
    robust_sigma = max(6.0, 1.4826 * mad)
    return ((gray - med) / robust_sigma).astype(np.float32), float(scale)


def estimate_frame_reliability(
    reference_frame: np.ndarray | None,
    current_frame: np.ndarray,
    *,
    config: FrameReliabilityConfig,
) -> FrameReliabilityResult:
    """Estimate generic frame reliability in [0,1].

    A high score means the current frame is well explained by a translated,
    robustly normalised reference image.  The score intentionally ignores the
    target identity itself.
    """
    config.validate()
    if not config.enabled or reference_frame is None or current_frame is None:
        return FrameReliabilityResult(score=1.0, valid=False)
    if reference_frame.size == 0 or current_frame.size == 0:
        return FrameReliabilityResult(score=0.0, residual=9.0, phase_response=0.0, valid=True)

    ref, scale = _gray_downscaled(reference_frame, config.max_dimension_px)
    cur, _ = _gray_downscaled(current_frame, config.max_dimension_px)
    if ref.shape != cur.shape:
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ref, cur, window)
    except cv2.error:
        dx = dy = 0.0
        response = 0.0

    if not np.isfinite(dx) or not np.isfinite(dy):
        dx = dy = 0.0
    if not np.isfinite(response):
        response = 0.0

    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    warped = cv2.warpAffine(
        ref, matrix, (ref.shape[1], ref.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    residual = float(np.median(np.abs(cur - warped)))
    residual_score = float(exp(-max(0.0, residual) / max(1e-9, config.residual_scale)))
    phase_score = float(np.clip(
        (float(response) - config.phase_response_floor)
        / max(1e-9, config.phase_response_good - config.phase_response_floor),
        0.0, 1.0,
    ))
    score = (1.0 - config.phase_weight) * residual_score + config.phase_weight * phase_score
    inv_scale = 1.0 / max(1e-9, scale)
    return FrameReliabilityResult(
        score=float(np.clip(score, 0.0, 1.0)),
        residual=float(residual),
        phase_response=float(response),
        shift_dx_px=float(dx * inv_scale),
        shift_dy_px=float(dy * inv_scale),
        valid=True,
    )
