from __future__ import annotations

import pytest

from navmin.turret.platform_motion import CommandMotionModel


def _model(*, capacity: int = 16) -> CommandMotionModel:
    return CommandMotionModel(
        acceleration_x_deg_s2=100.0,
        acceleration_y_deg_s2=50.0,
        capacity=capacity,
    )


def test_command_model_reconstructs_acceleration_deceleration_and_reversal() -> None:
    model = _model()
    model.mark_known_zero(timestamp_ns=0)
    model.record_velocity_command(
        timestamp_ns=0,
        commanded_rate_x_deg_s=20.0,
        commanded_rate_y_deg_s=-10.0,
    )

    rate = model.snapshot().rate_at(100_000_000)
    assert rate is not None
    assert rate.rate_x_deg_s == pytest.approx(10.0)
    assert rate.rate_y_deg_s == pytest.approx(-5.0)

    model.record_velocity_command(
        timestamp_ns=100_000_000,
        commanded_rate_x_deg_s=0.0,
        commanded_rate_y_deg_s=0.0,
    )
    rate = model.snapshot().rate_at(150_000_000)
    assert rate is not None
    assert rate.rate_x_deg_s == pytest.approx(5.0)
    assert rate.rate_y_deg_s == pytest.approx(-2.5)

    model.record_velocity_command(
        timestamp_ns=200_000_000,
        commanded_rate_x_deg_s=-10.0,
        commanded_rate_y_deg_s=10.0,
    )
    rate = model.snapshot().rate_at(250_000_000)
    assert rate is not None
    assert rate.rate_x_deg_s == pytest.approx(-5.0)
    assert rate.rate_y_deg_s == pytest.approx(2.5)


def test_snapshot_uses_the_anchor_covering_observation_timestamp() -> None:
    model = _model()
    model.mark_known_zero(timestamp_ns=1_000_000_000)
    model.record_velocity_command(
        timestamp_ns=1_000_000_000,
        commanded_rate_x_deg_s=30.0,
        commanded_rate_y_deg_s=0.0,
    )
    model.record_velocity_command(
        timestamp_ns=1_200_000_000,
        commanded_rate_x_deg_s=-30.0,
        commanded_rate_y_deg_s=0.0,
    )

    snapshot = model.snapshot()
    before_reversal = snapshot.rate_at(1_100_000_000)
    after_reversal = snapshot.rate_at(1_250_000_000)
    assert before_reversal is not None and after_reversal is not None
    assert before_reversal.rate_x_deg_s == pytest.approx(10.0)
    assert after_reversal.rate_x_deg_s == pytest.approx(15.0)


def test_snapshot_is_bounded_and_rejects_uncovered_old_timestamp() -> None:
    model = _model(capacity=3)
    model.mark_known_zero(timestamp_ns=0)
    for index in range(1, 5):
        model.record_velocity_command(
            timestamp_ns=index * 100_000_000,
            commanded_rate_x_deg_s=float(index),
            commanded_rate_y_deg_s=0.0,
        )

    snapshot = model.snapshot()
    assert len(snapshot.anchors) == 3
    assert snapshot.rate_at(100_000_000) is None
    assert snapshot.rate_at(200_000_000) is not None


def test_invalid_model_recovers_only_from_known_zero() -> None:
    model = _model()
    assert not model.snapshot().valid
    model.record_velocity_command(
        timestamp_ns=10,
        commanded_rate_x_deg_s=5.0,
        commanded_rate_y_deg_s=5.0,
    )
    assert not model.snapshot().valid

    model.mark_known_zero(timestamp_ns=20)
    assert model.snapshot().valid
    model.invalidate()
    assert not model.snapshot().valid
    assert model.snapshot().anchors == ()


def test_acceleration_change_preserves_rate_and_changes_future_ramp() -> None:
    model = _model()
    model.mark_known_zero(timestamp_ns=0)
    model.record_velocity_command(
        timestamp_ns=0,
        commanded_rate_x_deg_s=100.0,
        commanded_rate_y_deg_s=100.0,
    )
    model.record_acceleration_limits(
        timestamp_ns=100_000_000,
        acceleration_x_deg_s2=20.0,
        acceleration_y_deg_s2=10.0,
    )

    rate = model.snapshot().rate_at(200_000_000)
    assert rate is not None
    assert rate.rate_x_deg_s == pytest.approx(12.0)
    assert rate.rate_y_deg_s == pytest.approx(6.0)


def test_snapshot_angle_at_integrates_acceleration_plateau_and_reversal() -> None:
    model = _model()
    model.mark_known_zero(timestamp_ns=0)
    model.record_velocity_command(
        timestamp_ns=0,
        commanded_rate_x_deg_s=20.0,
        commanded_rate_y_deg_s=-10.0,
    )
    model.record_velocity_command(
        timestamp_ns=300_000_000,
        commanded_rate_x_deg_s=-10.0,
        commanded_rate_y_deg_s=10.0,
    )

    snapshot = model.snapshot()
    at_100 = snapshot.angle_at(100_000_000)
    at_300 = snapshot.angle_at(300_000_000)
    at_400 = snapshot.angle_at(400_000_000)

    assert at_100 is not None
    assert at_100.angle_x_deg == pytest.approx(0.5)
    assert at_100.angle_y_deg == pytest.approx(-0.25)

    # X reaches +20 deg/s after 0.2 s and then holds it for 0.1 s.
    # Y is still ramping toward -10 deg/s for the whole 0.2 s, then holds.
    assert at_300 is not None
    assert at_300.angle_x_deg == pytest.approx(4.0)
    assert at_300.angle_y_deg == pytest.approx(-2.0)

    # At 0.3 s the X command reverses. With 100 deg/s^2 it falls from
    # +20 to +10 deg/s during the next 0.1 s, adding another 1.5 deg.
    assert at_400 is not None
    assert at_400.angle_x_deg == pytest.approx(5.5)
    # Y rises from -10 to -5 deg/s at 50 deg/s^2, adding -0.75 deg.
    assert at_400.angle_y_deg == pytest.approx(-2.75)


def test_snapshot_angle_at_rejects_uncovered_or_invalid_motion() -> None:
    model = _model()
    assert model.snapshot().angle_at(0) is None

    model.mark_known_zero(timestamp_ns=100_000_000)
    snapshot = model.snapshot()
    assert snapshot.angle_at(99_999_999) is None
    at_anchor = snapshot.angle_at(100_000_000)
    assert at_anchor is not None
    assert at_anchor.angle_x_deg == 0.0
    assert at_anchor.angle_y_deg == 0.0
