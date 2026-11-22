"""Minimal stoppable camera worker joining a decoded-frame source to VisionPipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock, Thread
from time import monotonic
from typing import Protocol

import numpy as np

from navmin.concurrency import LatestValue
from navmin.config.models import CameraConfig
from navmin.contracts import CameraModel, CameraRole, CameraState, ConfigUpdate
from navmin.lifecycle import StopToken

from .camera_source import create_camera_source
from .gstreamer_source import CameraSourceError
from .pipeline import (
    CorrectedFrameSink,
    DecodedFrameSource,
    VisionPipeline,
    VisionPipelineError,
)
from .processor import (
    VisionProcessorSelection,
    VisionRuntimeSettings,
    create_selected_vision_processor,
)

LOGGER = logging.getLogger(__name__)

_RTSP_RECONNECT_DELAYS_S = (0.25, 0.5, 1.0, 2.0)
_FRAME_STALE_RECONNECT_S = 5.0


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
        stale_reconnect_timeout_s: float = _FRAME_STALE_RECONNECT_S,
        clock: Callable[[], float] = monotonic,
        thread_name: str | None = None,
        processor_selection: VisionProcessorSelection | None = None,
        runtime_settings: VisionRuntimeSettings | None = None,
    ) -> None:
        if idle_wait_s <= 0.0:
            raise ValueError("idle_wait_s must be > 0")
        if not reconnect_delays_s or any(delay <= 0.0 for delay in reconnect_delays_s):
            raise ValueError("reconnect_delays_s must contain positive delays")
        if stale_reconnect_timeout_s <= 0.0:
            raise ValueError("stale_reconnect_timeout_s must be > 0")
        self._source = source
        self._pipeline = pipeline
        self._idle_wait_s = idle_wait_s
        self._reconnect_delays_s = reconnect_delays_s
        self._stale_reconnect_timeout_s = float(stale_reconnect_timeout_s)
        self._clock = clock
        self._stop_token = StopToken()
        self._processor_selection = processor_selection
        self._processor_updates: LatestValue[VisionProcessorSelection] = LatestValue()
        self._processor_update_revision = 0
        self._runtime_settings = runtime_settings or VisionRuntimeSettings()
        self._runtime_setting_updates: LatestValue[
            ConfigUpdate[VisionRuntimeSettings]
        ] = LatestValue()
        self._runtime_settings_update_revision = 0
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

    @property
    def processor_selection(self) -> VisionProcessorSelection | None:
        return self._processor_selection

    def submit_processor_selection(self, selection: VisionProcessorSelection) -> None:
        if not isinstance(selection, VisionProcessorSelection):
            raise TypeError("selection must be VisionProcessorSelection")
        self._processor_updates.publish(selection)

    def submit_runtime_settings(
        self, update: ConfigUpdate[VisionRuntimeSettings]
    ) -> None:
        if not isinstance(update, ConfigUpdate):
            raise TypeError("update must be ConfigUpdate[VisionRuntimeSettings]")
        if not isinstance(update.config, VisionRuntimeSettings):
            raise TypeError("update.config must be VisionRuntimeSettings")
        self._runtime_setting_updates.publish(update)

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
        self._apply_latest_processor_selection(restart_session=False)
        self._apply_latest_runtime_settings(force=True)
        self._pipeline.start()
        self._source.start()
        while not self._stop_token.is_stop_requested():
            processor_changed = self._apply_latest_processor_selection(
                restart_session=True
            )
            self._apply_latest_runtime_settings(force=processor_changed)
            self._pipeline.apply_latest_target_control()
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
            self._apply_latest_processor_selection(restart_session=False)
            self._apply_latest_runtime_settings(force=True)
            if not self._start_pipeline_session():
                return

            last_frame_time_s: float | None = None
            while not self._stop_token.is_stop_requested():
                processor_changed = self._apply_latest_processor_selection(
                    restart_session=True
                )
                self._apply_latest_runtime_settings(force=processor_changed)
                self._pipeline.apply_latest_target_control()
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
                    last_frame_time_s = self._clock()
                    delay_index = 0
                    continue
                if (
                    last_frame_time_s is not None
                    and self._clock() - last_frame_time_s
                    >= self._stale_reconnect_timeout_s
                ):
                    stale_error = CameraSourceError(
                        "camera frame stream remained stale for "
                        f"{self._stale_reconnect_timeout_s:.3f} s"
                    )
                    LOGGER.warning(
                        "Vision source stale; restarting camera=%s timeout_s=%.3f",
                        self._pipeline.camera.value,
                        self._stale_reconnect_timeout_s,
                    )
                    self._pipeline.report_source_reconnecting(stale_error)
                    self._source.stop()
                    if self._wait_before_reconnect(delay_index):
                        return
                    delay_index = self._next_delay_index(delay_index)
                    break
                self._stop_token.wait(self._idle_wait_s)

    def _apply_latest_processor_selection(self, *, restart_session: bool) -> bool:
        snapshot = self._processor_updates.snapshot()
        if snapshot.revision <= self._processor_update_revision or snapshot.value is None:
            return False
        selection = snapshot.value
        processor = create_selected_vision_processor(selection)
        self._pipeline.replace_processor(processor)
        self._pipeline.apply_processor_runtime_settings(self._runtime_settings)
        self._processor_selection = selection
        self._processor_update_revision = snapshot.revision
        if restart_session:
            self._pipeline.start()
        LOGGER.info(
            "Vision processor update applied camera=%s class=%s profile=%s",
            self._pipeline.camera.value,
            selection.class_name,
            selection.profile_name or "—",
        )
        return True

    def _apply_latest_runtime_settings(self, *, force: bool = False) -> bool:
        snapshot = self._runtime_setting_updates.snapshot()
        changed = False
        if (
            snapshot.revision > self._runtime_settings_update_revision
            and snapshot.value is not None
        ):
            self._runtime_settings = snapshot.value.config
            self._runtime_settings_update_revision = snapshot.revision
            changed = True
        if not changed and not force:
            return False
        self._pipeline.apply_processor_runtime_settings(self._runtime_settings)
        return True

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
    corrected_frame_sink: CorrectedFrameSink | None = None,
    idle_wait_s: float = 0.01,
    runtime_settings: VisionRuntimeSettings | None = None,
) -> CameraWorker:
    """Bind typed camera config to one decoded source and VisionPipeline."""
    source = source_factory(config)
    processor_selection = VisionProcessorSelection(
        config.vision_processor_class,
        config.vision_processor_profile,
    )
    pipeline = VisionPipeline(
        camera=camera,
        corrector=corrector,
        processor=create_selected_vision_processor(processor_selection),
        processing_enabled=config.processing_enabled,
        corrected_frame_sink=corrected_frame_sink,
    )
    return CameraWorker(
        source=source,
        pipeline=pipeline,
        idle_wait_s=idle_wait_s,
        processor_selection=processor_selection,
        runtime_settings=runtime_settings,
    )


__all__ = ["CameraWorker", "build_camera_worker"]
