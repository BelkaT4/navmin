"""DAY observation profile description for VT-11.

This module intentionally contains no duplicate tracking implementation.  It
publishes the feature priorities used by the shared core and is useful for
embedding/documentation code that wants to inspect the active method.
"""
from __future__ import annotations

PROFILE_NAME = "day"
VIDEO_MODE = "visible"
PRIMARY_IDENTITY_FEATURES = (
    "appearance",
    "edge",
    "projection",
    "geometry",
    "color",
)
CANDIDATE_METHOD = "local_contrast_components"
MICRO_CHANNEL = False


def description() -> dict:
    return {
        "profile": PROFILE_NAME,
        "video_mode": VIDEO_MODE,
        "candidate_method": CANDIDATE_METHOD,
        "identity_features": PRIMARY_IDENTITY_FEATURES,
        "micro_channel": MICRO_CHANNEL,
        "principle": "preserve proven visible-light tracking; motion is auxiliary to visual identity",
    }
