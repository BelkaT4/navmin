"""THERMAL/NIGHT observation profile description for VT-11."""
from __future__ import annotations

PROFILE_NAME = "thermal"
VIDEO_MODE = "thermal"
PRIMARY_IDENTITY_FEATURES = (
    "thermal_contrast",
    "hot_area",
    "peak_prominence",
    "hot_core_compactness",
    "radial_hotspot_profile",
    "energy_concentration",
    "thermal_2d_ratio",
    "projection",
    "appearance",
    "edge",
    "geometry",
)
CANDIDATE_METHOD = "thermal_hot_blob"
MICRO_CHANNEL = True


def description() -> dict:
    return {
        "profile": PROFILE_NAME,
        "video_mode": VIDEO_MODE,
        "candidate_method": CANDIDATE_METHOD,
        "identity_features": PRIMARY_IDENTITY_FEATURES,
        "micro_channel": MICRO_CHANNEL,
        "scale_adaptive_identity": True,
        "principle": "thermal structural identity + time-normalized ReliableAnchor + frame reliability + identity-first reacquisition",
    }
