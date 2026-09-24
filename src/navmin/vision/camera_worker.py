"""Minimal stoppable camera worker joining a decoded-frame source to VisionPipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock, Thread
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

_RTSP_RECONNECT_DELAYS_S = (0.25, 0.5, 1.0, 2.0)


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
        reconnect_delays_s: tuple[float, ...] = _RTSP_RECONNECT_DELAYS_S,
        thread_name: str | None = None,
    ) -> None:
        if idle_wait_s <= 0.0:
            raise ValueError("idle_wait_s must be > 0")
        if not reconnect_delays_s or any(delay <= 0.0 for delay in reconnect_delays_s):
            raise ValueError("reconnect_delays_s must contain positive delays")
        self._source = source
        self._pipeline = pipeline
        self._idle_wait_s = idle_wait_s
        self._reconnect_delays_s = reconnect_delays_s
        self._stop_token = StopToken()
        self._lifecycle_lock = Lock()
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
        with self._lifecycle_lock:
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
            if bool(getattr(self._source, "supports_reconnect", False)):
                self._run_with_reconnect()
            else:
                self._run_without_reconnect()
        except (CameraSourceError, VisionPipelineError) as exc:
            self._pipeline.report_source_failure(exc)
        finally:
            try:
                self._source.stop()
            except CameraSourceError as exc:
                self._pipeline.report_source_failure(exc)
            current = self._pipeline.status.get()
            if current is not None and current.state is not CameraState.ERROR:
                self._pipeline.stop()
            LOGGER.info("Vision camera worker stopped camera=%s", self._pipeline.camera.value)

    def _run_without_reconnect(self) -> None:
        # Preserve the established RTP/JPEG ordering: the generation barrier exists
        # before source data can be published.
        self._pipeline.start()
        self._source.start()
        while not self._stop_token.is_stop_requested():
            failure = self._source.failure
            if failure is not None:
                self._pipeline.report_source_failure(failure)
                return
            if self._process_source_once():
                continue
            self._stop_token.wait(self._idle_wait_s)

    def _run_with_reconnect(self) -> None:
        delay_index = 0
        while not self._stop_token.is_stop_requested():
            try:
                self._source.start()
            except CameraSourceError as exc:
                self._pipeline.report_source_reconnecting(exc)
                self._source.stop()
                if self._wait_before_reconnect(delay_index):
                    return
                delay_index = self._next_delay_index(delay_index)
                continue

            # A successful source start establishes the next camera session.
            if not self._start_pipeline_session():
                return

            while not self._stop_token.is_stop_requested():
                failure = self._source.failure
                if failure is not None:
                    self._pipeline.report_source_reconnecting(failure)
                    self._source.stop()
                    if self._wait_before_reconnect(delay_index):
                        return
                    delay_index = self._next_delay_index(delay_index)
                    break
                if self._process_source_once():
                    # A real accepted frame proves recovery; later losses restart
                    # backoff from the shortest delay.
                    delay_index = 0
                    continue
                self._stop_token.wait(self._idle_wait_s)

    def _start_pipeline_session(self) -> bool:
        with self._lifecycle_lock:
            if self._stop_token.is_stop_requested():
                return False
            self._pipeline.start()
            return True

    def _process_source_once(self) -> bool:
        if not self._pipeline.submit_from_source(self._source):
            return False
        self._pipeline.process_latest()
        return True

    def _wait_before_reconnect(self, delay_index: int) -> bool:
        delay_s = self._reconnect_delays_s[delay_index]
        LOGGER.info(
            "Vision source reconnect backoff camera=%s delay_s=%.2f",
            self._pipeline.camera.value,
            delay_s,
        )
        return self._stop_token.wait(delay_s)

    def _next_delay_index(self, delay_index: int) -> int:
        return min(delay_index + 1, len(self._reconnect_delays_s) - 1)


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
