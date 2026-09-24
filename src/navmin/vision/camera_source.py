"""Transport-aware composition for production decoded camera sources."""

from __future__ import annotations

from navmin.config.models import (
    CameraConfig,
    RtpJpegSourceConfig,
    RtspSourceConfig,
)

from .gstreamer_source import GStreamerRtpJpegSource, GStreamerRtspSource
from .pipeline import DecodedFrameSource


def create_camera_source(config: CameraConfig) -> DecodedFrameSource:
    """Create the production decoded source selected by typed camera config."""
    source = config.source
    if isinstance(source, RtpJpegSourceConfig):
        return GStreamerRtpJpegSource(source)
    if isinstance(source, RtspSourceConfig):
        return GStreamerRtspSource(source)
    raise TypeError(f"unsupported camera source config: {type(source).__name__}")


__all__ = ["create_camera_source"]
