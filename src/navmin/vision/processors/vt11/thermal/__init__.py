"""Thermal/video-mode helpers introduced by VT-10 HOTFIX-4."""
from navmin.vision.processors.vt11.thermal.processing import (
    ThermalProcessingConfig,
    ThermalProfile,
    build_thermal_profile,
    infer_video_mode,
    preprocess_frame,
    reticle_overlap_ratio,
    hud_evidence,
    hud_point_membership,
    runtime_detector_config,
)

__all__ = [
    "ThermalProcessingConfig", "ThermalProfile", "build_thermal_profile",
    "infer_video_mode", "preprocess_frame", "reticle_overlap_ratio",
    "hud_evidence", "hud_point_membership", "runtime_detector_config",
]
