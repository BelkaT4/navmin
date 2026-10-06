"""Public processing boundary for Vision implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from navmin.contracts import (
    FramePacket,
    TargetLifecycleEvent,
    VisionResult,
    VisionTargetControlCommand,
)

DEFAULT_VISION_PROCESSOR_CLASS = "Legacy14VisionProcessor"


@dataclass(frozen=True)
class VisionProcessorOutput:
    """Explicit internal output from a vision backend to its pipeline."""

    result: VisionResult
    target_events: tuple[TargetLifecycleEvent, ...] = ()


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

        return Legacy14VisionProcessor()
    raise ValueError(f"unsupported VisionProcessor class in this checkpoint: {class_name}")


__all__ = [
    "DEFAULT_VISION_PROCESSOR_CLASS",
    "VisionProcessor",
    "VisionProcessorOutput",
    "create_vision_processor",
]
