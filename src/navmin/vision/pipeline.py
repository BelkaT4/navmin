"""Minimal reusable camera pipeline core for corrected working-frame Vision data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic_ns
from typing import Protocol

import cv2
import numpy as np

from navmin.calibration import (
    OverviewCalibration,
    StereoCalibration,
    overview_camera_model,
    stereo_camera_models,
)
from navmin.concurrency import (
    CameraSessionBarrierChannel,
    InvalidatableLatest,
    LatestValue,
)
from navmin.contracts import (
    CameraModel,
    CameraRole,
    CameraSessionStarted,
    CameraState,
    CameraStatus,
    FramePacket,
    VisionResult,
)

from .processor import VisionProcessor, create_vision_processor

LOGGER = logging.getLogger(__name__)


class VisionPipelineError(RuntimeError):
    """Base error for the prototype Vision camera pipeline."""


class WorkingFrameError(VisionPipelineError):
    """Decoded source frame cannot be converted into the calibrated working frame."""


class MissingCalibrationError(VisionPipelineError):
    """Required calibration was not supplied for the camera role."""


class _FrameCorrector(Protocol):
    camera_model: CameraModel

    def correct(self, image: np.ndarray) -> np.ndarray: ...


class _OpenCvMapCorrector:
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        map_x: np.ndarray,
        map_y: np.ndarray,
        camera_model: CameraModel,
    ) -> None:
        self._width = image_width
        self._height = image_height
        self._map_x = map_x
        self._map_y = map_y
        self.camera_model = camera_model

    def correct(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise WorkingFrameError("decoded frame must be numpy.ndarray")
        if image.dtype != np.uint8:
            raise WorkingFrameError("decoded frame must use uint8 pixels")
        if image.ndim != 3 or image.shape[2] != 3:
            raise WorkingFrameError(
                f"decoded frame must be BGR HxWx3, got shape={image.shape!r}"
            )
        actual = (image.shape[1], image.shape[0])
        expected = (self._width, self._height)
        if actual != expected:
            raise WorkingFrameError(
                f"source resolution {actual[0]}x{actual[1]} does not match "
                f"calibration {expected[0]}x{expected[1]}"
            )
        corrected = cv2.remap(
            image,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if corrected.shape != image.shape or corrected.dtype != np.uint8:
            raise WorkingFrameError("OpenCV correction returned an invalid working frame")
        corrected = np.ascontiguousarray(corrected)
        corrected.flags.writeable = False
        return corrected


def overview_corrector(calibration: OverviewCalibration) -> _FrameCorrector:
    size = (calibration.image_width, calibration.image_height)
    map_x, map_y = cv2.initUndistortRectifyMap(
        np.asarray(calibration.K, dtype=np.float64),
        np.asarray(calibration.D, dtype=np.float64),
        None,
        np.asarray(calibration.new_camera_matrix, dtype=np.float64),
        size,
        cv2.CV_32FC1,
    )
    return _OpenCvMapCorrector(
        image_width=calibration.image_width,
        image_height=calibration.image_height,
        map_x=map_x,
        map_y=map_y,
        camera_model=overview_camera_model(calibration),
    )


def stereo_left_corrector(calibration: StereoCalibration) -> _FrameCorrector:
    size = (calibration.image_width, calibration.image_height)
    map_x, map_y = cv2.initUndistortRectifyMap(
        np.asarray(calibration.K_left, dtype=np.float64),
        np.asarray(calibration.D_left, dtype=np.float64),
        np.asarray(calibration.R1, dtype=np.float64),
        np.asarray(calibration.P1, dtype=np.float64),
        size,
        cv2.CV_32FC1,
    )
    left_model, _right_model = stereo_camera_models(calibration)
    return _OpenCvMapCorrector(
        image_width=calibration.image_width,
        image_height=calibration.image_height,
        map_x=map_x,
        map_y=map_y,
        camera_model=left_model,
    )


def build_working_frame_corrector(
    camera: CameraRole,
    *,
    overview_calibration: OverviewCalibration | None = None,
    stereo_calibration: StereoCalibration | None = None,
) -> _FrameCorrector:
    if camera is CameraRole.OVERVIEW:
        if overview_calibration is None:
            raise MissingCalibrationError("Overview calibration is required")
        return overview_corrector(overview_calibration)
    if camera is CameraRole.STEREO_LEFT:
        if stereo_calibration is None:
            raise MissingCalibrationError("Stereo calibration is required for Stereo Left")
        return stereo_left_corrector(stereo_calibration)
    raise ValueError(f"prototype checkpoint does not build a pipeline for {camera.value}")


@dataclass(frozen=True)
class DecodedFrame:
    """Already-decoded BGR source frame accepted by an injectable source."""

    image: np.ndarray
    capture_id: int | None = None


class InMemoryFrameSource:
    """Deterministic hardware-free latest-frame source for prototype tests."""

    def __init__(self) -> None:
        self._latest: LatestValue[DecodedFrame] = LatestValue()
        self._last_read_revision = 0

    def push(self, image: np.ndarray, *, capture_id: int | None = None) -> None:
        self._latest.publish(DecodedFrame(image=image, capture_id=capture_id))

    def read(self) -> DecodedFrame | None:
        snapshot = self._latest.snapshot()
        if snapshot.value is None or snapshot.revision <= self._last_read_revision:
            return None
        self._last_read_revision = snapshot.revision
        return snapshot.value


@dataclass(frozen=True)
class _PendingFrame:
    generation: int
    image: np.ndarray
    capture_id: int | None
    receive_timestamp_ns: int


class VisionPipeline:
    """Owner-local prototype pipeline with barrier and latest-only result semantics."""

    def __init__(
        self,
        *,
        camera: CameraRole,
        corrector: _FrameCorrector,
        processor: VisionProcessor | None = None,
        session_barriers: CameraSessionBarrierChannel | None = None,
        latest_result: InvalidatableLatest[VisionResult] | None = None,
        status: LatestValue[CameraStatus] | None = None,
        timestamp_clock_ns: Callable[[], int] = monotonic_ns,
        processing_enabled: bool = True,
    ) -> None:
        if camera not in (CameraRole.OVERVIEW, CameraRole.STEREO_LEFT):
            raise ValueError("prototype pipeline supports Overview and Stereo Left only")
        self.camera = camera
        self._corrector = corrector
        self._processor = processor or create_vision_processor()
        self.session_barriers = session_barriers or CameraSessionBarrierChannel()
        self.latest_result = latest_result or InvalidatableLatest()
        self.status = status or LatestValue()
        self._timestamp_clock_ns = timestamp_clock_ns
        self._processing_enabled = processing_enabled
        self._pending: LatestValue[_PendingFrame] = LatestValue()
        self._last_processed_revision = 0
        self._generation = 0
        self._frame_id = 0
        self._state_lock = Lock()

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    def set_processing_enabled(self, enabled: bool) -> None:
        self._processing_enabled = bool(enabled)

    def start(self) -> CameraSessionStarted:
        with self._state_lock:
            self._generation += 1
            generation = self._generation
            self._frame_id = 0
            self._last_processed_revision = self._pending.snapshot().revision
            self._processor.reset()
            self.latest_result.invalidate()
            timestamp_ns = self._timestamp_clock_ns()
            self.status.publish(
                CameraStatus(
                    camera=self.camera,
                    state=CameraState.STARTING,
                    generation=generation,
                    last_receive_timestamp_ns=None,
                )
            )
            session = CameraSessionStarted(
                camera=self.camera,
                generation=generation,
                camera_model=self._corrector.camera_model,
                timestamp_ns=timestamp_ns,
            )
            self.session_barriers.publish(session)
        LOGGER.info("Vision pipeline started camera=%s generation=%d", self.camera.value, generation)
        return session

    def submit_decoded_frame(
        self,
        image: np.ndarray,
        *,
        capture_id: int | None = None,
    ) -> None:
        with self._state_lock:
            generation = self._generation
        if generation <= 0:
            raise VisionPipelineError("pipeline must be started before frames are submitted")
        if not isinstance(image, np.ndarray):
            raise WorkingFrameError("decoded frame must be numpy.ndarray")
        stable_image = np.array(image, copy=True, order="C")
        received_ns = self._timestamp_clock_ns()
        self._pending.publish(
            _PendingFrame(
                generation=generation,
                image=stable_image,
                capture_id=capture_id,
                receive_timestamp_ns=received_ns,
            )
        )

    def submit_from_source(self, source: InMemoryFrameSource) -> bool:
        decoded = source.read()
        if decoded is None:
            return False
        self.submit_decoded_frame(decoded.image, capture_id=decoded.capture_id)
        return True

    def process_latest(self) -> VisionResult | None:
        snapshot = self._pending.snapshot()
        if snapshot.value is None or snapshot.revision <= self._last_processed_revision:
            return None
        self._last_processed_revision = snapshot.revision
        pending = snapshot.value
        with self._state_lock:
            if pending.generation != self._generation:
                return None
            frame_id = self._frame_id

        try:
            image = self._corrector.correct(pending.image)
            packet = FramePacket(
                camera=self.camera,
                generation=pending.generation,
                frame_id=frame_id,
                capture_id=(
                    pending.capture_id
                    if self.camera is CameraRole.STEREO_LEFT
                    else None
                ),
                receive_timestamp_ns=pending.receive_timestamp_ns,
                image=image,
            )
            result = self._processor.process(
                packet,
                processing_enabled=self._processing_enabled,
            )
        except Exception as exc:
            with self._state_lock:
                is_current_generation = pending.generation == self._generation
                if is_current_generation:
                    self.status.publish(
                        CameraStatus(
                            camera=self.camera,
                            state=CameraState.ERROR,
                            generation=pending.generation,
                            last_receive_timestamp_ns=pending.receive_timestamp_ns,
                            error_code=type(exc).__name__,
                            message=str(exc),
                        )
                    )
            if is_current_generation:
                LOGGER.warning(
                    "Vision pipeline frame failed camera=%s generation=%d: %s",
                    self.camera.value,
                    pending.generation,
                    exc,
                )
            raise

        with self._state_lock:
            if pending.generation != self._generation:
                return None
            self._frame_id += 1
            self.latest_result.publish(result)
            self.status.publish(
                CameraStatus(
                    camera=self.camera,
                    state=CameraState.ONLINE,
                    generation=pending.generation,
                    last_receive_timestamp_ns=pending.receive_timestamp_ns,
                )
            )
        return result


__all__ = [
    "DecodedFrame",
    "InMemoryFrameSource",
    "MissingCalibrationError",
    "VisionPipeline",
    "VisionPipelineError",
    "WorkingFrameError",
    "build_working_frame_corrector",
    "overview_corrector",
    "stereo_left_corrector",
]
