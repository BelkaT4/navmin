"""Public processing boundary for Vision implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from navmin.contracts import (
    BBox,
    CameraRole,
    FramePacket,
    TargetLifecycleEvent,
    VisionResult,
    VisionTargetControlCommand,
)

DEFAULT_VISION_PROCESSOR_CLASS = "Legacy14VisionProcessor"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisionTrackingDiagnostics:
    """Backend-neutral read-only tracking diagnostics for one exact frame."""

    camera: CameraRole
    generation: int
    frame_id: int
    phase: str
    status: str
    predicted_xy: tuple[float, float]
    search_radius_px: int
    full_frame_search: bool
    search_mode: str
    candidate_count: int
    best_candidate_center: tuple[float, float] | None
    best_candidate_bbox: BBox | None
    confidence: float
    prediction_error_px: float | None
    diagnostic_reason: str


@dataclass(frozen=True)
class VisionProcessorOutput:
    """Explicit internal output from a vision backend to its pipeline."""

    result: VisionResult
    target_events: tuple[TargetLifecycleEvent, ...] = ()
    tracking_diagnostics: VisionTrackingDiagnostics | None = None


class VisionProcessor(Protocol):
    """Only processing boundary intended for consumers outside Vision."""

    def reset(self) -> None:
        """Clear all generation-local processing state."""
        ...

    def apply_target_control(
        self,
        command: VisionTargetControlCommand,
    ) -> tuple[TargetLifecycleEvent, ...]:
        """Apply target-control intent on the processor-owner thread."""
        ...

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionProcessorOutput:
        """Process one corrected working frame."""
        ...


def create_vision_processor(
    class_name: str = DEFAULT_VISION_PROCESSOR_CLASS,
) -> VisionProcessor:
    """Create a configured processor without exposing detector/tracker internals."""
    if class_name == DEFAULT_VISION_PROCESSOR_CLASS:
        from .processors import Legacy14VisionProcessor

        processor = Legacy14VisionProcessor()
        LOGGER.info("Vision processor: %s", DEFAULT_VISION_PROCESSOR_CLASS)
        return processor
    if class_name == "Vt11VisionProcessor":
        from .processors.vt11 import Vt11VisionProcessor
        from .processors.vt11.settings import VT11_PROFILE

        profile = VT11_PROFILE
        if profile not in {"day", "thermal"}:
            raise ValueError("VT11_PROFILE must be 'day' or 'thermal'")
        processor = Vt11VisionProcessor(profile_name=profile)
        LOGGER.info(
            "Vision processor: Vt11VisionProcessor profile=%s",
            processor.settings_snapshot.profile,
        )
        return processor
    raise ValueError(f"unsupported VisionProcessor class in this checkpoint: {class_name}")


__all__ = [
    "DEFAULT_VISION_PROCESSOR_CLASS",
    "VisionProcessor",
    "VisionProcessorOutput",
    "VisionTrackingDiagnostics",
    "create_vision_processor",
]
