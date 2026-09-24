"""Vision processing and minimal camera-pipeline boundaries."""

from .camera_source import create_camera_source
from .camera_worker import CameraWorker, build_camera_worker
from .gstreamer_source import (
    CameraSourceError,
    GStreamerRtpJpegSource,
    GStreamerRtspSource,
    GStreamerUnavailableError,
    UnsupportedCameraTransportError,
    build_rtp_jpeg_pipeline_description,
    build_rtsp_pipeline_description,
)
from .pipeline import (
    DecodedFrame,
    DecodedFrameSource,
    InMemoryFrameSource,
    MissingCalibrationError,
    VisionPipeline,
    VisionPipelineError,
    WorkingFrameError,
    build_working_frame_corrector,
    overview_corrector,
    stereo_left_corrector,
)
from .processor import (
    DEFAULT_VISION_PROCESSOR_CLASS,
    VisionProcessor,
    create_vision_processor,
)
from .processors import Legacy14VisionProcessor

__all__ = [
    "DEFAULT_VISION_PROCESSOR_CLASS",
    "CameraSourceError",
    "CameraWorker",
    "DecodedFrame",
    "DecodedFrameSource",
    "GStreamerRtpJpegSource",
    "GStreamerRtspSource",
    "GStreamerUnavailableError",
    "InMemoryFrameSource",
    "Legacy14VisionProcessor",
    "MissingCalibrationError",
    "UnsupportedCameraTransportError",
    "VisionPipeline",
    "VisionPipelineError",
    "VisionProcessor",
    "WorkingFrameError",
    "build_camera_worker",
    "build_rtp_jpeg_pipeline_description",
    "build_rtsp_pipeline_description",
    "build_working_frame_corrector",
    "create_camera_source",
    "create_vision_processor",
    "overview_corrector",
    "stereo_left_corrector",
]
