"""Minimal stoppable camera worker joining a decoded-frame source to VisionPipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Thread
from typing import Protocol

import numpy as np

from navmin.config.models import CameraConfig
from navmin.contracts import CameraModel, CameraRole, CameraState
from navmin.lifecycle import StopToken

from .camera_source import create_camera_source
from .gstreamer_source import CameraSourceError
from .pipeline import DecodedFrameSource, VisionPipeline, VisionPipelineError
from .processor import create_vision_processor

LOGGER = logging.getLogger(__name__)

class _FrameCorrector(Protocol):
    camera_model: CameraModel

    def correct(self, image: np.ndarray) -> np.ndarray: ...


SourceFactory = Callable[[CameraConfig], DecodedFrameSource]



class CameraWorker:
    """One application worker for one camera source and one VisionPipeline."""

    def __init__(
        self,
        *,
        source: DecodedFrameSource,
        pipeline: VisionPipeline,
        idle_wait_s: float = 0.01,
        thread_name: str | None = None,
    ) -> None:
        if idle_wait_s <= 0.0:
            raise ValueError("idle_wait_s must be > 0")
        self._source = source
        self._pipeline = pipeline
        self._idle_wait_s = idle_wait_s
        self._stop_token = StopToken()
        self._thread = Thread(
            target=self._run,
            name=thread_name or f"vision-{pipeline.camera.value}",
            daemon=False,
        )

    @property
    def source(self) -> DecodedFrameSource:
        return self._source

    @property
    def pipeline(self) -> VisionPipeline:
        return self._pipeline

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_token.request_stop()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 2.0) -> bool:
        """Cooperatively stop and perform a bounded join."""
        self.request_stop()
        self.join(timeout)
        return not self.is_alive()

    def _run(self) -> None:
        LOGGER.info("Vision camera worker starting camera=%s", self._pipeline.camera.value)
        try:
            # The ordered generation barrier exists before source data can be published.
            self._pipeline.start()
            self._source.start()
            while not self._stop_token.is_stop_requested():
                failure = self._source.failure
                if failure is not None:
                    self._pipeline.report_source_failure(failure)
                    break
                if self._pipeline.submit_from_source(self._source):
                    self._pipeline.process_latest()
                    continue
                self._stop_token.wait(self._idle_wait_s)
        except (CameraSourceError, VisionPipelineError) as exc:
            self._pipeline.report_source_failure(exc)
        finally:
            try:
                self._source.stop()
            except CameraSourceError as exc:
                self._pipeline.report_source_failure(exc)
            if self._pipeline.status.get() is not None:
                current = self._pipeline.status.get()
                if current is not None and current.state is not CameraState.ERROR:
                    self._pipeline.stop()
            LOGGER.info("Vision camera worker stopped camera=%s", self._pipeline.camera.value)


def build_camera_worker(
    *,
    camera: CameraRole,
    config: CameraConfig,
    corrector: _FrameCorrector,
    source_factory: SourceFactory = create_camera_source,
    idle_wait_s: float = 0.01,
) -> CameraWorker:
    """Bind typed camera config to one decoded source and VisionPipeline."""
    source = source_factory(config)
    pipeline = VisionPipeline(
        camera=camera,
        corrector=corrector,
        processor=create_vision_processor(config.vision_processor_class),
        processing_enabled=config.processing_enabled,
    )
    return CameraWorker(
        source=source,
        pipeline=pipeline,
        idle_wait_s=idle_wait_s,
    )


__all__ = ["CameraWorker", "build_camera_worker"]
