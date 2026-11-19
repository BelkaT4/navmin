"""Public processing boundary for Vision implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from navmin.contracts import (
    BBox,
    CameraRole,
    FramePacket,
    TargetCandidateSet,
    TargetLifecycleEvent,
    VisionResult,
    VisionTargetControlCommand,
)

DEFAULT_VISION_PROCESSOR_CLASS = "Legacy14VisionProcessor"
VT11_VISION_PROCESSOR_CLASS = "Vt11VisionProcessor"
LEGACY14_PROFILES = ("default",)
VT11_PROFILES = ("day", "thermal")
VISION_PROCESSOR_PROFILES = {
    DEFAULT_VISION_PROCESSOR_CLASS: LEGACY14_PROFILES,
    VT11_VISION_PROCESSOR_CLASS: VT11_PROFILES,
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisionProcessorSelection:
    """Runtime-selectable Vision backend and one explicit backend profile."""

    class_name: str
    profile_name: str

    def __post_init__(self) -> None:
        profiles = VISION_PROCESSOR_PROFILES.get(self.class_name)
        if profiles is None:
            raise ValueError(f"unsupported VisionProcessor class: {self.class_name}")
        if self.profile_name not in profiles:
            expected = ", ".join(repr(profile) for profile in profiles)
            raise ValueError(
                f"{self.class_name} profile must be one of: {expected}"
            )


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
    acquisition_candidates: TargetCandidateSet | None = None


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


def default_vision_processor_selection(class_name: str) -> VisionProcessorSelection:
    """Return the production default profile for one configured backend."""
    if class_name == DEFAULT_VISION_PROCESSOR_CLASS:
        return VisionProcessorSelection(class_name, "default")
    if class_name == VT11_VISION_PROCESSOR_CLASS:
        from .processors.vt11.settings import VT11_PROFILE

        if VT11_PROFILE not in VT11_PROFILES:
            raise ValueError("VT11_PROFILE must be 'day' or 'thermal'")
        return VisionProcessorSelection(class_name, VT11_PROFILE)
    raise ValueError(f"unsupported VisionProcessor class in this checkpoint: {class_name}")


def vision_processor_profiles(class_name: str) -> tuple[str, ...]:
    profiles = VISION_PROCESSOR_PROFILES.get(class_name)
    if profiles is None:
        raise ValueError(f"unsupported VisionProcessor class: {class_name}")
    return profiles


def create_vision_processor(
    class_name: str = DEFAULT_VISION_PROCESSOR_CLASS,
    *,
    profile_name: str | None = None,
) -> VisionProcessor:
    """Create one processor from the same typed runtime selection used by UI."""
    selection = (
        default_vision_processor_selection(class_name)
        if profile_name is None
        else VisionProcessorSelection(class_name, profile_name)
    )
    if selection.class_name == DEFAULT_VISION_PROCESSOR_CLASS:
        from .processors import Legacy14VisionProcessor

        processor = Legacy14VisionProcessor()
        LOGGER.info("Vision processor: %s", DEFAULT_VISION_PROCESSOR_CLASS)
        return processor

    from .processors.vt11 import Vt11VisionProcessor

    processor = Vt11VisionProcessor(profile_name=selection.profile_name)
    LOGGER.info(
        "Vision processor: Vt11VisionProcessor profile=%s",
        processor.settings_snapshot.profile,
    )
    return processor


def create_selected_vision_processor(selection: VisionProcessorSelection) -> VisionProcessor:
    if not isinstance(selection, VisionProcessorSelection):
        raise TypeError("selection must be VisionProcessorSelection")
    return create_vision_processor(
        selection.class_name,
        profile_name=selection.profile_name,
    )


__all__ = [
    "DEFAULT_VISION_PROCESSOR_CLASS",
    "LEGACY14_PROFILES",
    "VISION_PROCESSOR_PROFILES",
    "VT11_PROFILES",
    "VT11_VISION_PROCESSOR_CLASS",
    "VisionProcessor",
    "VisionProcessorOutput",
    "VisionProcessorSelection",
    "VisionTrackingDiagnostics",
    "create_selected_vision_processor",
    "create_vision_processor",
    "default_vision_processor_selection",
    "vision_processor_profiles",
]
