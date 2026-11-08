from __future__ import annotations

import numpy as np
import pytest

from navmin.camera_time_sync import (
    CameraClockSessionError,
    CameraClockTracker,
    CameraPcTimeBasis,
)
from navmin.contracts import (
    CameraRole,
    FramePacket,
    FrameSourceTimestamp,
    FrameTimestampSource,
)
from navmin.time_sync import (
    ClockMapperConfig,
    ClockMappingState,
    ClockSampleOutcome,
)


def frame(
    *,
    generation: int,
    source: FrameTimestampSource,
    source_timestamp_ns: int,
    receive_timestamp_ns: int,
    camera: CameraRole = CameraRole.OVERVIEW,
) -> FramePacket:
    return FramePacket(
        camera=camera,
        generation=generation,
        frame_id=0,
        capture_id=None,
        receive_timestamp_ns=receive_timestamp_ns,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        source_timestamp=FrameSourceTimestamp(
            source=source,
            timestamp_ns=source_timestamp_ns,
        ),
    )


def mapper_config() -> ClockMapperConfig:
    return ClockMapperConfig(
        capacity=16,
        min_samples=4,
        min_span_ns=300_000_000,
        max_fit_rms_ns=2_000_000,
        max_abs_drift_ppm=2000.0,
        max_mapping_age_ns=1_000_000_000,
        reset_backstep_ns=500_000_000,
        outlier_floor_ns=2_000_000,
        outlier_sigma=6.0,
    )


def test_sensor_frame_forms_receive_anchored_clock_sync_sample() -> None:
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        receive_anchor_uncertainty_ns=3_000_000,
        observation_clock_ns=lambda: 10_020_000_000,
    )

    observation = tracker.observe_frame(
        frame(
            generation=1,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=1_000_000_000,
            receive_timestamp_ns=10_000_000_000,
        )
    )

    assert observation.sample_outcome is ClockSampleOutcome.ACCEPTED
    assert observation.sync_sample is not None
    assert observation.sync_sample.device_timestamp_ns == 1_000_000_000
    assert observation.sync_sample.pc_timestamp_ns == 10_000_000_000
    assert observation.sync_sample.observed_at_pc_ns == 10_020_000_000
    assert observation.sync_sample.uncertainty_ns == 3_000_000
    assert observation.diagnostics.state is ClockMappingState.WARMING


def test_receive_time_source_is_direct_pc_domain_and_does_not_feed_mapper() -> None:
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=lambda: 5_100_000_000,
    )

    observation = tracker.observe_frame(
        frame(
            generation=1,
            source=FrameTimestampSource.RECEIVE_TIME,
            source_timestamp_ns=5_000_000_000,
            receive_timestamp_ns=5_000_000_000,
        )
    )

    assert observation.sync_sample is None
    assert observation.sample_outcome is None
    assert observation.diagnostics.state is ClockMappingState.UNAVAILABLE
    assert observation.diagnostics.sample_count == 0


def test_missing_source_timestamp_falls_back_to_receive_time_contract() -> None:
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=lambda: 5_100_000_000,
    )
    packet = FramePacket(
        camera=CameraRole.OVERVIEW,
        generation=1,
        frame_id=0,
        capture_id=None,
        receive_timestamp_ns=5_000_000_000,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        source_timestamp=None,
    )

    observation = tracker.observe_frame(packet)

    assert observation.source is FrameTimestampSource.RECEIVE_TIME
    assert observation.source_timestamp_ns == 5_000_000_000
    assert observation.sync_sample is None


def test_sensor_samples_warm_and_validate_mapper_without_defining_capture_time(
) -> None:
    observations = iter(
        (10_010_000_000, 10_110_000_000, 10_210_000_000, 10_310_000_000)
    )
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=observations.__next__,
    )

    latest = None
    for index in range(4):
        latest = tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=index * 100_000_000,
                receive_timestamp_ns=10_000_000_000 + index * 100_000_000,
            )
        )

    assert latest is not None
    assert latest.diagnostics.state is ClockMappingState.VALID
    assert latest.diagnostics.slope == pytest.approx(1.0)
    assert latest.diagnostics.offset_ns == pytest.approx(10_000_000_000.0)


def test_generation_change_resets_sensor_clock_session() -> None:
    observations = iter(
        (
            10_000_000_000,
            10_100_000_000,
            10_200_000_000,
            10_300_000_000,
            20_000_000_000,
        )
    )
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=observations.__next__,
    )
    for index in range(4):
        tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=index * 100_000_000,
                receive_timestamp_ns=10_000_000_000 + index * 100_000_000,
            )
        )
    diagnostics = tracker.diagnostics(now_pc_ns=10_300_000_000)
    assert diagnostics.state is ClockMappingState.VALID

    next_session = tracker.observe_frame(
        frame(
            generation=2,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=0,
            receive_timestamp_ns=20_000_000_000,
        )
    )

    assert next_session.sample_outcome is ClockSampleOutcome.ACCEPTED
    assert next_session.diagnostics.state is ClockMappingState.WARMING
    assert next_session.diagnostics.sample_count == 1
    assert next_session.diagnostics.reset_count == 1


def test_source_change_inside_generation_is_rejected() -> None:
    observations = iter((10_000_000_000, 10_100_000_000))
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=observations.__next__,
    )
    tracker.observe_frame(
        frame(
            generation=1,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=0,
            receive_timestamp_ns=10_000_000_000,
        )
    )

    with pytest.raises(CameraClockSessionError, match="timestamp source changed"):
        tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.RECEIVE_TIME,
                source_timestamp_ns=10_100_000_000,
                receive_timestamp_ns=10_100_000_000,
            )
        )


def test_nonincreasing_source_time_requires_new_generation() -> None:
    observations = iter((10_000_000_000, 10_100_000_000))
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=observations.__next__,
    )
    tracker.observe_frame(
        frame(
            generation=1,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=100_000_000,
            receive_timestamp_ns=10_000_000_000,
        )
    )

    with pytest.raises(CameraClockSessionError, match="not strictly increasing"):
        tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=100_000_000,
                receive_timestamp_ns=10_100_000_000,
            )
        )


def test_tracker_rejects_wrong_camera_and_backward_generation() -> None:
    observations = iter((10_000_000_000, 20_000_000_000))
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=observations.__next__,
    )
    tracker.observe_frame(
        frame(
            generation=2,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=0,
            receive_timestamp_ns=10_000_000_000,
        )
    )

    with pytest.raises(CameraClockSessionError, match="another camera"):
        tracker.observe_frame(
            frame(
                generation=2,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=10_000_000,
                receive_timestamp_ns=10_100_000_000,
                camera=CameraRole.STEREO_LEFT,
            )
        )

    with pytest.raises(CameraClockSessionError, match="generation moved backwards"):
        tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=0,
                receive_timestamp_ns=20_000_000_000,
            )
        )


def test_observation_clock_cannot_precede_receive_timestamp() -> None:
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=lambda: 9_999_999_999,
    )

    with pytest.raises(CameraClockSessionError, match="precedes frame receive"):
        tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=0,
                receive_timestamp_ns=10_000_000_000,
            )
        )


def test_receive_time_observation_has_direct_receive_anchor_without_mapper() -> None:
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=lambda: 10_000_000_010,
    )

    observation = tracker.observe_frame(
        frame(
            generation=1,
            source=FrameTimestampSource.RECEIVE_TIME,
            source_timestamp_ns=10_000_000_000,
            receive_timestamp_ns=10_000_000_000,
        )
    )

    mapped = observation.receive_anchored_timestamp
    assert mapped is not None
    assert mapped.basis is CameraPcTimeBasis.RECEIVE_TIME
    assert mapped.pc_timestamp_ns == 10_000_000_000
    assert mapped.estimated_error_ns == 0
    assert observation.sync_sample is None


def test_sensor_receive_anchor_is_unavailable_until_mapper_is_valid() -> None:
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=lambda: 10_010_000_000,
    )

    observation = tracker.observe_frame(
        frame(
            generation=1,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=0,
            receive_timestamp_ns=10_000_000_000,
        )
    )

    assert observation.receive_anchored_timestamp is None
    assert observation.diagnostics.state is ClockMappingState.WARMING


def test_sensor_receive_anchor_preserves_constant_transport_latency_as_offset() -> None:
    """Lock the semantic boundary: receive anchoring is not capture time.

    Synthetic true capture time is 10 s + device time.  Every frame arrives 80 ms
    later.  Because ClockMapper sees only device time paired with receive time, the
    fitted PC-domain timestamp must also remain 80 ms later than the synthetic
    capture time.  Future code must not silently interpret this receive-anchored
    estimate as exposure/capture time.
    """

    observations = iter(
        (
            10_081_000_000,
            10_181_000_000,
            10_281_000_000,
            10_381_000_000,
        )
    )
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        receive_anchor_uncertainty_ns=2_000_000,
        observation_clock_ns=observations.__next__,
    )

    latest = None
    for index in range(4):
        source_ns = index * 100_000_000
        synthetic_capture_pc_ns = 10_000_000_000 + source_ns
        receive_ns = synthetic_capture_pc_ns + 80_000_000
        latest = tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=source_ns,
                receive_timestamp_ns=receive_ns,
            )
        )

    assert latest is not None
    mapped = latest.receive_anchored_timestamp
    assert mapped is not None
    assert mapped.basis is CameraPcTimeBasis.RECEIVE_ANCHORED_SOURCE_CLOCK
    assert mapped.pc_timestamp_ns == 10_380_000_000
    assert mapped.pc_timestamp_ns - 10_300_000_000 == 80_000_000
    assert latest.diagnostics.state is ClockMappingState.VALID


def test_generation_change_invalidates_sensor_receive_anchor_until_rewarmed() -> None:
    observations = iter(
        (
            10_010_000_000,
            10_110_000_000,
            10_210_000_000,
            10_310_000_000,
            20_010_000_000,
        )
    )
    tracker = CameraClockTracker(
        CameraRole.OVERVIEW,
        mapper_config=mapper_config(),
        observation_clock_ns=observations.__next__,
    )

    latest = None
    for index in range(4):
        source_ns = index * 100_000_000
        latest = tracker.observe_frame(
            frame(
                generation=1,
                source=FrameTimestampSource.SENSOR_RTP_PTS,
                source_timestamp_ns=source_ns,
                receive_timestamp_ns=10_000_000_000 + source_ns,
            )
        )

    assert latest is not None
    assert latest.receive_anchored_timestamp is not None

    restarted = tracker.observe_frame(
        frame(
            generation=2,
            source=FrameTimestampSource.SENSOR_RTP_PTS,
            source_timestamp_ns=0,
            receive_timestamp_ns=20_000_000_000,
        )
    )

    assert restarted.receive_anchored_timestamp is None
    assert restarted.diagnostics.state is ClockMappingState.WARMING
