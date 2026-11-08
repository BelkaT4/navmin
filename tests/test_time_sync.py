from __future__ import annotations

import math

import pytest

from navmin.time_sync import (
    ClockMapper,
    ClockMapperConfig,
    ClockMappingInvalidReason,
    ClockMappingState,
    ClockSampleOutcome,
    ClockSyncSample,
)


def sample(
    device_ns: int,
    pc_ns: int,
    *,
    observed_at_pc_ns: int | None = None,
    uncertainty_ns: int = 0,
) -> ClockSyncSample:
    return ClockSyncSample(
        device_timestamp_ns=device_ns,
        pc_timestamp_ns=pc_ns,
        observed_at_pc_ns=pc_ns if observed_at_pc_ns is None else observed_at_pc_ns,
        uncertainty_ns=uncertainty_ns,
    )


def valid_config(**changes) -> ClockMapperConfig:
    defaults = {
        "capacity": 16,
        "min_samples": 4,
        "min_span_ns": 300_000_000,
        "max_fit_rms_ns": 2_000_000,
        "max_abs_drift_ppm": 2000.0,
        "max_mapping_age_ns": 1_000_000_000,
        "reset_backstep_ns": 500_000_000,
        "outlier_floor_ns": 2_000_000,
        "outlier_sigma": 6.0,
    }
    defaults.update(changes)
    return ClockMapperConfig(**defaults)


def seed_linear(
    mapper: ClockMapper,
    *,
    slope: float = 1.0,
    offset_ns: int = 2_000_000_000,
    count: int = 5,
    step_ns: int = 100_000_000,
    uncertainty_ns: int = 0,
) -> None:
    for index in range(count):
        device_ns = index * step_ns
        pc_ns = round(slope * device_ns + offset_ns)
        assert mapper.observe(
            sample(
                device_ns,
                pc_ns,
                uncertainty_ns=uncertainty_ns,
            )
        ) is ClockSampleOutcome.ACCEPTED


def test_clock_mapper_config_defaults_match_accepted_t2a_contract() -> None:
    config = ClockMapperConfig()

    assert config.capacity == 32
    assert config.min_samples == 4
    assert config.min_span_ns == 1_000_000_000
    assert config.max_fit_rms_ns == 5_000_000
    assert config.max_abs_drift_ppm == 2000.0
    assert config.max_mapping_age_ns == 5_000_000_000
    assert config.reset_backstep_ns == 1_000_000_000
    assert config.outlier_floor_ns == 10_000_000
    assert config.outlier_sigma == 6.0




def test_sync_sample_and_config_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="min_samples"):
        ClockMapperConfig(min_samples=1)
    with pytest.raises(ValueError, match="capacity"):
        ClockMapperConfig(capacity=3, min_samples=4)
    with pytest.raises(ValueError, match="max_abs_drift_ppm"):
        ClockMapperConfig(max_abs_drift_ppm=math.inf)
    with pytest.raises(ValueError, match="outlier_floor_ns"):
        ClockMapperConfig(outlier_floor_ns=0)
    with pytest.raises(TypeError, match="config"):
        ClockMapper(0)  # type: ignore[arg-type]

def test_clock_sync_sample_validates_monotonic_domain_fields() -> None:
    with pytest.raises(ValueError, match="observed_at_pc_ns"):
        sample(10, 20, observed_at_pc_ns=19)
    with pytest.raises(ValueError, match="device_timestamp_ns"):
        sample(-1, 20)
    with pytest.raises(TypeError, match="uncertainty_ns"):
        ClockSyncSample(1, 2, 2, uncertainty_ns=True)


def test_diagnostics_progress_unavailable_warming_span_then_valid() -> None:
    mapper = ClockMapper(valid_config(min_span_ns=300_000_000))

    diagnostics = mapper.diagnostics(now_pc_ns=0)
    assert diagnostics.state is ClockMappingState.UNAVAILABLE
    assert diagnostics.invalid_reason is ClockMappingInvalidReason.NO_SAMPLES

    for device_ns in (0, 100_000_000, 200_000_000):
        mapper.observe(sample(device_ns, device_ns + 1_000_000_000))
    diagnostics = mapper.diagnostics(now_pc_ns=1_200_000_000)
    assert diagnostics.state is ClockMappingState.WARMING
    assert diagnostics.invalid_reason is ClockMappingInvalidReason.INSUFFICIENT_SAMPLES

    mapper.observe(sample(250_000_000, 1_250_000_000))
    diagnostics = mapper.diagnostics(now_pc_ns=1_250_000_000)
    assert diagnostics.state is ClockMappingState.WARMING
    assert diagnostics.invalid_reason is ClockMappingInvalidReason.INSUFFICIENT_SPAN

    mapper.observe(sample(300_000_000, 1_300_000_000))
    diagnostics = mapper.diagnostics(now_pc_ns=1_300_000_000)
    assert diagnostics.state is ClockMappingState.VALID
    assert diagnostics.invalid_reason is None
    assert diagnostics.slope == pytest.approx(1.0)
    assert diagnostics.offset_ns == pytest.approx(1_000_000_000.0)
    assert diagnostics.fit_rms_ns == pytest.approx(0.0)


def test_map_timestamp_maps_offset_and_reports_uncertainty_floor() -> None:
    mapper = ClockMapper(valid_config())
    seed_linear(mapper, offset_ns=2_500_000_000, uncertainty_ns=750_000)

    mapped = mapper.map_timestamp(450_000_000, now_pc_ns=2_400_000_000)

    assert mapped is not None
    assert mapped.device_timestamp_ns == 450_000_000
    assert mapped.pc_timestamp_ns == 2_950_000_000
    assert mapped.estimated_error_ns == 750_000


def test_small_clock_drift_is_fitted_and_reported() -> None:
    mapper = ClockMapper(valid_config(max_abs_drift_ppm=500.0))
    seed_linear(mapper, slope=1.0001)

    diagnostics = mapper.diagnostics(now_pc_ns=2_400_040_000)
    mapped = mapper.map_timestamp(450_000_000, now_pc_ns=2_400_040_000)

    assert diagnostics.state is ClockMappingState.VALID
    assert diagnostics.drift_ppm == pytest.approx(100.0, abs=0.01)
    assert mapped is not None
    assert mapped.pc_timestamp_ns == pytest.approx(2_450_045_000, abs=1)


def test_drift_limit_invalidates_mapping() -> None:
    mapper = ClockMapper(valid_config(max_abs_drift_ppm=100.0))
    seed_linear(mapper, slope=1.001)

    diagnostics = mapper.diagnostics(now_pc_ns=2_400_400_000)

    assert diagnostics.state is ClockMappingState.INVALID
    assert diagnostics.invalid_reason is ClockMappingInvalidReason.DRIFT_LIMIT
    assert mapper.map_timestamp(450_000_000, now_pc_ns=2_400_400_000) is None


def test_fit_rms_limit_invalidates_mapping() -> None:
    mapper = ClockMapper(
        valid_config(max_fit_rms_ns=10_000, outlier_floor_ns=100_000_000)
    )
    points = (
        (0, 2_000_000_000),
        (100_000_000, 2_100_100_000),
        (200_000_000, 2_199_900_000),
        (300_000_000, 2_300_100_000),
        (400_000_000, 2_399_900_000),
    )
    for device_ns, pc_ns in points:
        mapper.observe(sample(device_ns, pc_ns))

    diagnostics = mapper.diagnostics(now_pc_ns=2_399_900_000)

    assert diagnostics.state is ClockMappingState.INVALID
    assert diagnostics.invalid_reason is ClockMappingInvalidReason.FIT_RMS
    assert diagnostics.fit_rms_ns is not None
    assert diagnostics.fit_rms_ns > 10_000


def test_robust_filter_rejects_single_large_outlier_without_poisoning_fit() -> None:
    mapper = ClockMapper(valid_config(min_span_ns=300_000_000))
    seed_linear(mapper, count=5)

    outcome = mapper.observe(sample(500_000_000, 2_900_000_000))
    diagnostics = mapper.diagnostics(now_pc_ns=2_400_000_000)

    assert outcome is ClockSampleOutcome.REJECTED_OUTLIER
    assert diagnostics.rejected_outlier_count == 1
    assert diagnostics.sample_count == 5
    assert diagnostics.state is ClockMappingState.VALID
    assert diagnostics.slope == pytest.approx(1.0)


def test_out_of_order_sample_is_rejected_but_large_backstep_resets_session() -> None:
    mapper = ClockMapper(valid_config(reset_backstep_ns=500_000_000))
    for device_ns in (1_000_000_000, 1_100_000_000, 1_200_000_000, 1_300_000_000):
        mapper.observe(sample(device_ns, device_ns + 2_000_000_000))

    assert mapper.observe(
        sample(1_250_000_000, 3_250_000_000)
    ) is ClockSampleOutcome.REJECTED_OUT_OF_ORDER
    assert mapper.observe(
        sample(100_000_000, 4_000_000_000)
    ) is ClockSampleOutcome.RESET_ACCEPTED

    diagnostics = mapper.diagnostics(now_pc_ns=4_000_000_000)
    assert diagnostics.state is ClockMappingState.WARMING
    assert diagnostics.invalid_reason is ClockMappingInvalidReason.INSUFFICIENT_SAMPLES
    assert diagnostics.sample_count == 1
    assert diagnostics.submitted_sample_count == 6
    assert diagnostics.rejected_out_of_order_count == 1
    assert diagnostics.reset_count == 1


def test_explicit_reset_retains_lifetime_counters_and_clears_session_fit() -> None:
    mapper = ClockMapper(valid_config())
    seed_linear(mapper)
    mapper.reset()

    diagnostics = mapper.diagnostics(now_pc_ns=3_000_000_000)

    assert diagnostics.state is ClockMappingState.UNAVAILABLE
    assert diagnostics.sample_count == 0
    assert diagnostics.submitted_sample_count == 5
    assert diagnostics.reset_count == 1
    assert diagnostics.slope is None
    assert mapper.map_timestamp(1, now_pc_ns=3_000_000_000) is None


def test_mapping_becomes_stale_from_last_observation_age() -> None:
    mapper = ClockMapper(valid_config(max_mapping_age_ns=200_000_000))
    seed_linear(mapper)
    fresh = mapper.diagnostics(now_pc_ns=2_400_000_000)
    stale = mapper.diagnostics(now_pc_ns=2_600_000_001)

    assert fresh.state is ClockMappingState.VALID
    assert stale.state is ClockMappingState.STALE
    assert stale.invalid_reason is ClockMappingInvalidReason.STALE
    assert stale.age_ns == 200_000_001
    assert mapper.map_timestamp(450_000_000, now_pc_ns=2_600_000_001) is None


def test_history_is_bounded_and_mapping_remains_finite() -> None:
    mapper = ClockMapper(valid_config(capacity=5, min_samples=4))
    for index in range(12):
        device_ns = index * 100_000_000
        mapper.observe(sample(device_ns, device_ns + 3_000_000_000))

    diagnostics = mapper.diagnostics(now_pc_ns=4_100_000_000)
    mapped = mapper.map_timestamp(1_150_000_000, now_pc_ns=4_100_000_000)

    assert diagnostics.sample_count == 5
    assert diagnostics.submitted_sample_count == 12
    assert diagnostics.device_span_ns == 400_000_000
    assert diagnostics.state is ClockMappingState.VALID
    assert mapped is not None
    assert math.isfinite(mapped.pc_timestamp_ns)
    assert mapped.pc_timestamp_ns == 4_150_000_000
