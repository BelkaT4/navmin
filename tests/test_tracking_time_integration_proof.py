from __future__ import annotations

from pathlib import Path

from navmin.diagnostics import tracking_time_integration
from navmin.diagnostics.tracking_time_integration import (
    run_deterministic_tracking_time_integration_proof,
    run_tracking_time_integration_proof,
    write_tracking_time_integration_json,
)
from navmin.time_sync import ClockMappingState


def test_deterministic_proof_covers_off_on_restart_and_revalidation() -> None:
    result = run_deterministic_tracking_time_integration_proof()

    assert result.passed
    assert result.source_timestamp_type == "sensor-rtp-pts"
    assert result.off_observation_timestamp_ns == result.off_receive_timestamp_ns
    assert result.on_observation_timestamp_ns == result.on_mapped_timestamp_ns
    assert result.after_off_observation_timestamp_ns == result.after_off_receive_timestamp_ns
    assert result.source_restart_accepted_reset_pts

    assert result.first_generation.mapper_states[0] == ClockMappingState.WARMING.value
    assert ClockMappingState.VALID.value in result.first_generation.mapper_states
    assert result.second_generation.mapper_states[0] == ClockMappingState.WARMING.value
    assert ClockMappingState.VALID.value in result.second_generation.mapper_states
    assert result.second_generation.reset_count >= 1
    assert (
        result.second_generation.fallback_observation_timestamp_ns
        == result.second_generation.fallback_receive_timestamp_ns
    )
    assert (
        result.second_generation.mapped_observation_timestamp_ns
        == result.second_generation.mapped_timestamp_ns
    )


def test_wrapper_can_run_without_multimedia_runtime() -> None:
    result = run_tracking_time_integration_proof(include_gstreamer=False)

    assert result.passed
    assert result.deterministic.passed
    assert result.gstreamer is None
    assert result.gstreamer_blocked_reason is None


def test_result_json_is_stable_and_machine_readable(tmp_path: Path) -> None:
    result = run_tracking_time_integration_proof(include_gstreamer=False)
    output = tmp_path / "tracking-time-proof.json"

    write_tracking_time_integration_json(result, output)

    text = output.read_text(encoding="utf-8")
    assert '"status": "PASS"' in text
    assert '"sensor-rtp-pts"' in text
    assert '"source_restart_accepted_reset_pts": true' in text


def test_absolute_sender_schedule_does_not_accumulate_processing_overhead(monkeypatch) -> None:
    now = [10.0]
    send_times: list[float] = []

    def fake_monotonic() -> float:
        return now[0]

    def fake_sleep(duration_s: float) -> None:
        now[0] += duration_s

    monkeypatch.setattr(tracking_time_integration, "monotonic", fake_monotonic)
    monkeypatch.setattr(tracking_time_integration, "sleep", fake_sleep)

    origin_s = now[0]
    fps = 5
    for index in range(4):
        tracking_time_integration._sleep_until_monotonic(origin_s + index / fps)
        send_times.append(now[0])
        now[0] += 0.03  # synthetic JPEG/decode/poll overhead

    assert send_times == [10.0, 10.2, 10.4, 10.6]
