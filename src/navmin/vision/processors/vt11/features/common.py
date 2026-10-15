"""Shared helpers for VT-4 single-frame ACQUIRE feature scoring."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidatePatch:
    gray: np.ndarray
    background: np.ndarray


def _clip_bbox(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x), frame_w - 1))
    y1 = max(0, min(int(y), frame_h - 1))
    x2 = max(x1 + 1, min(int(x + w), frame_w))
    y2 = max(y1 + 1, min(int(y + h), frame_h))
    return x1, y1, x2, y2


def candidate_patch_and_background(frame_bgr: np.ndarray, candidate, pad_px: int = 4) -> CandidatePatch:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR HxWx3 frame")
    fh, fw = frame_bgr.shape[:2]
    b = candidate.bbox
    x1, y1, x2, y2 = _clip_bbox(b.x, b.y, b.width, b.height, fw, fh)
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    obj = gray_full[y1:y2, x1:x2]

    px1 = max(0, x1 - pad_px)
    py1 = max(0, y1 - pad_px)
    px2 = min(fw, x2 + pad_px)
    py2 = min(fh, y2 + pad_px)
    local = gray_full[py1:py2, px1:px2]
    mask = np.ones(local.shape, dtype=bool)
    ox1, oy1 = x1 - px1, y1 - py1
    ox2, oy2 = x2 - px1, y2 - py1
    mask[oy1:oy2, ox1:ox2] = False
    bg = local[mask]
    if bg.size == 0:
        bg = gray_full.reshape(-1)
    return CandidatePatch(gray=obj, background=bg)


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
