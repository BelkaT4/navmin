"""Camera source-clock observations anchored to PC receive time.

This module deliberately stops short of defining a PC-domain *capture* timestamp.
For a sensor-derived source timestamp, the only PC-domain observation currently
available is the frame receive time.  That pair is sufficient to estimate clock
rate/continuity, but a constant one-way transport delay is indistinguishable from
clock offset and must not be silently treated as zero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from time import monotonic_ns

from navmin.contracts import CameraRole, FramePacket, FrameTimestampSource
from navmin.time_sync import (
    ClockMapper,
    ClockMapperConfig,
    ClockMappingDiagnostics,
    ClockSampleOutcome,
    ClockSyncSample,
    MappedTimestamp,
)


class CameraPcTimeBasis(Enum):
    RECEIVE_TIME = "receive-time"
    RECEIVE_ANCHORED_SOURCE_CLOCK = "receive-anchored-source-clock"


@dataclass(frozen=True, slots=True)
class CameraReceiveAnchoredTimestamp:
    """PC-domain timestamp anchored to frame receipt, not sensor capture.

    For ``SENSOR_RTP_PTS`` the affine mapper is fitted against PC receive
    timestamps, so any constant encode/transport/decode latency is absorbed into
    the fitted offset.  This value therefore MUST NOT be treated as a true
    capture/exposure timestamp without an independently established latency
    correction.
    """

    camera: CameraRole
    generation: int
    source: FrameTimestampSource
    source_timestamp_ns: int
    pc_timestamp_ns: int
    estimated_error_ns: int
    basis: CameraPcTimeBasis


class CameraClockSessionError(RuntimeError):
    """A frame violates the active camera clock-session contract."""


@dataclass(frozen=True, slots=True)
class CameraClockObservation:
    """Read-only result of observing one frame timestamp pair."""

    camera: CameraRole
    generation: int
    source: FrameTimestampSource
    source_timestamp_ns: int
    receive_timestamp_ns: int
    sync_sample: ClockSyncSample | None
    sample_outcome: ClockSampleOutcome | None
    diagnostics: ClockMappingDiagnostics
    receive_anchored_timestamp: CameraReceiveAnchoredTimestamp | None


class CameraClockTracker:
    """Own one ClockMapper for one camera across generation-bounded sessions.

    ``SENSOR_RTP_PTS`` frames contribute ``source timestamp -> receive time``
    observations.  ``RECEIVE_TIME`` frames are already in the PC monotonic domain
    and therefore do not need to warm or fit a ClockMapper.

    Generation changes are the authoritative clock-session boundary.  A source
    timestamp reset inside one generation is rejected instead of being allowed to
    trigger ClockMapper's generic backstep reset semantics.
    """

    def __init__(
        self,
        camera: CameraRole,
        *,
        mapper_config: ClockMapperConfig | None = None,
        receive_anchor_uncertainty_ns: int = 0,
        observation_clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not isinstance(camera, CameraRole):
            raise TypeError("camera must be CameraRole")
        if type(receive_anchor_uncertainty_ns) is not int:
            raise TypeError("receive_anchor_uncertainty_ns must be an integer")
        if receive_anchor_uncertainty_ns < 0:
            raise ValueError("receive_anchor_uncertainty_ns must be >= 0")
        self.camera = camera
        self._mapper = ClockMapper(mapper_config)
        self._receive_anchor_uncertainty_ns = receive_anchor_uncertainty_ns
        self._observation_clock_ns = observation_clock_ns
        self._generation: int | None = None
        self._source: FrameTimestampSource | None = None
        self._last_source_timestamp_ns: int | None = None

    @property
    def generation(self) -> int | None:
        return self._generation

    @property
    def mapper(self) -> ClockMapper:
        return self._mapper

    def observe_frame(self, frame: FramePacket) -> CameraClockObservation:
        if not isinstance(frame, FramePacket):
            raise TypeError("frame must be FramePacket")
        if frame.camera is not self.camera:
            raise CameraClockSessionError(
                "frame belongs to another camera: "
                f"tracker={self.camera.value} frame={frame.camera.value}"
            )
        if frame.generation <= 0:
            raise CameraClockSessionError("frame generation must be > 0")

        self._enter_generation(frame.generation)
        source_timestamp = frame.source_timestamp
        source = (
            FrameTimestampSource.RECEIVE_TIME
            if source_timestamp is None
            else source_timestamp.source
        )
        source_timestamp_ns = (
            frame.receive_timestamp_ns
            if source_timestamp is None
            else source_timestamp.timestamp_ns
        )
        self._check_source_contract(source, source_timestamp_ns)

        observed_at_pc_ns = self._observation_clock_ns()
        sync_sample: ClockSyncSample | None = None
        sample_outcome: ClockSampleOutcome | None = None
        receive_anchored_timestamp: CameraReceiveAnchoredTimestamp | None = None
        now_pc_ns = max(frame.receive_timestamp_ns, observed_at_pc_ns)
        if source is FrameTimestampSource.SENSOR_RTP_PTS:
            if observed_at_pc_ns < frame.receive_timestamp_ns:
                raise CameraClockSessionError(
                    "observation clock precedes frame receive timestamp: "
                    f"observed={observed_at_pc_ns} receive={frame.receive_timestamp_ns}"
                )
            sync_sample = ClockSyncSample(
                device_timestamp_ns=source_timestamp_ns,
                pc_timestamp_ns=frame.receive_timestamp_ns,
                observed_at_pc_ns=observed_at_pc_ns,
                uncertainty_ns=self._receive_anchor_uncertainty_ns,
            )
            sample_outcome = self._mapper.observe(sync_sample)
            mapped = self._mapper.map_timestamp(
                source_timestamp_ns,
                now_pc_ns=now_pc_ns,
            )
            receive_anchored_timestamp = self._sensor_receive_anchor(
                frame=frame,
                source_timestamp_ns=source_timestamp_ns,
                mapped=mapped,
            )
        else:
            receive_anchored_timestamp = CameraReceiveAnchoredTimestamp(
                camera=frame.camera,
                generation=frame.generation,
                source=source,
                source_timestamp_ns=source_timestamp_ns,
                pc_timestamp_ns=frame.receive_timestamp_ns,
                estimated_error_ns=0,
                basis=CameraPcTimeBasis.RECEIVE_TIME,
            )

        diagnostics = self._mapper.diagnostics(now_pc_ns=now_pc_ns)
        return CameraClockObservation(
            camera=frame.camera,
            generation=frame.generation,
            source=source,
            source_timestamp_ns=source_timestamp_ns,
            receive_timestamp_ns=frame.receive_timestamp_ns,
            sync_sample=sync_sample,
            sample_outcome=sample_outcome,
            diagnostics=diagnostics,
            receive_anchored_timestamp=receive_anchored_timestamp,
        )

    def _sensor_receive_anchor(
        self,
        *,
        frame: FramePacket,
        source_timestamp_ns: int,
        mapped: MappedTimestamp | None,
    ) -> CameraReceiveAnchoredTimestamp | None:
        if mapped is None:
            return None
        return CameraReceiveAnchoredTimestamp(
            camera=frame.camera,
            generation=frame.generation,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=source_timestamp_ns,
            pc_timestamp_ns=mapped.pc_timestamp_ns,
            estimated_error_ns=mapped.estimated_error_ns,
            basis=CameraPcTimeBasis.RECEIVE_ANCHORED_SOURCE_CLOCK,
        )

    def diagnostics(self, *, now_pc_ns: int) -> ClockMappingDiagnostics:
        return self._mapper.diagnostics(now_pc_ns=now_pc_ns)

    def _enter_generation(self, generation: int) -> None:
        previous = self._generation
        if previous is None:
            self._generation = generation
            return
        if generation < previous:
            raise CameraClockSessionError(
                "frame generation moved backwards: "
                f"previous={previous} current={generation}"
            )
        if generation == previous:
            return
        self._mapper.reset()
        self._generation = generation
        self._source = None
        self._last_source_timestamp_ns = None

    def _check_source_contract(
        self,
        source: FrameTimestampSource,
        source_timestamp_ns: int,
    ) -> None:
        if type(source_timestamp_ns) is not int or source_timestamp_ns < 0:
            raise CameraClockSessionError("source timestamp must be an integer >= 0")
        if self._source is None:
            self._source = source
        elif source is not self._source:
            raise CameraClockSessionError(
                "timestamp source changed inside one camera generation: "
                f"previous={self._source.value} current={source.value}"
            )
        previous = self._last_source_timestamp_ns
        if previous is not None and source_timestamp_ns <= previous:
            raise CameraClockSessionError(
                "source timestamp is not strictly increasing inside one generation: "
                f"previous={previous} current={source_timestamp_ns}"
            )
        self._last_source_timestamp_ns = source_timestamp_ns


__all__ = [
    "CameraClockObservation",
    "CameraClockSessionError",
    "CameraClockTracker",
    "CameraPcTimeBasis",
    "CameraReceiveAnchoredTimestamp",
]
