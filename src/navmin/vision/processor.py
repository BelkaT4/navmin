"""Public processing boundary for Vision implementations."""

from __future__ import annotations

from typing import Protocol

from navmin.contracts import FramePacket, VisionResult

from .processors import Legacy14VisionProcessor

DEFAULT_VISION_PROCESSOR_CLASS = "Legacy14VisionProcessor"


class VisionProcessor(Protocol):
    """Only processing boundary intended for consumers outside Vision."""

    def reset(self) -> None:
        """Clear all generation-local processing state."""
        ...

    def process(
        self,
        frame: FramePacket,
        *,
        processing_enabled: bool = True,
    ) -> VisionResult:
        """Process one corrected working frame."""
        ...


def create_vision_processor(
    class_name: str = DEFAULT_VISION_PROCESSOR_CLASS,
) -> VisionProcessor:
    """Create a configured processor without exposing detector/tracker internals."""
    if class_name == DEFAULT_VISION_PROCESSOR_CLASS:
        return Legacy14VisionProcessor()
    raise ValueError(f"unsupported VisionProcessor class in this checkpoint: {class_name}")


__all__ = [
    "DEFAULT_VISION_PROCESSOR_CLASS",
    "VisionProcessor",
    "create_vision_processor",
]
