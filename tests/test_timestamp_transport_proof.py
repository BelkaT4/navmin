from navmin.diagnostics.timestamp_transport import (
    RtpSenderIdentity,
    TimestampProvenanceProofResult,
    TimestampReceiverRecoveryProofResult,
    assess_timestamp_sequence,
    build_timestamp_receiver_pipeline_description,
    build_timestamp_sender_pipeline_description,
)


def test_sender_pipeline_matches_colleague_timestamp_semantics() -> None:
    description = build_timestamp_sender_pipeline_description(
        host="127.0.0.1",
        port=18_898,
        fps=20,
    )

    assert "appsrc name=source is-live=true format=time do-timestamp=false" in description
    assert "caps=image/jpeg,framerate=20/1" in description
    assert "! jpegparse ! rtpjpegpay pt=26" in description
    assert "udpsink host=127.0.0.1 port=18898 sync=false async=false" in description


def test_receiver_pipeline_matches_colleague_candidate_chain() -> None:
    description = build_timestamp_receiver_pipeline_description(
        address="127.0.0.1",
        port=18_898,
    )

    assert "clock-rate=90000" in description
    assert "! rtpjitterbuffer latency=0 mode=0 ! rtpjpegdepay" in description
    assert "! jpegdec ! videoconvert ! video/x-raw,format=BGR" in description
    assert "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false" in description


def test_assessment_accepts_exact_timestamp_preservation() -> None:
    source = (5_000_000_000, 5_050_000_000, 5_100_000_000)

    assessment = assess_timestamp_sequence(source, source, tolerance_ns=1_000)

    assert assessment.passed
    assert assessment.relative_timing_preserved
    assert assessment.constant_offset
    assert assessment.origin_preserved
    assert assessment.delta_errors_ns == (0, 0)
    assert assessment.per_frame_offsets_ns == (0, 0, 0)


def test_assessment_accepts_stable_origin_shift_when_deltas_are_preserved() -> None:
    source = (9_000_000_000, 9_020_000_000, 9_055_000_000, 9_095_000_000)
    received = tuple(timestamp + 123_000_000 for timestamp in source)

    assessment = assess_timestamp_sequence(source, received, tolerance_ns=1_000)

    assert assessment.passed
    assert assessment.relative_timing_preserved
    assert assessment.constant_offset
    assert not assessment.origin_preserved
    assert assessment.delta_errors_ns == (0, 0, 0)
    assert assessment.offset_span_ns == 0


def test_assessment_rejects_distorted_relative_timing() -> None:
    source = (0, 20_000_000, 55_000_000, 95_000_000)
    received = (100_000_000, 120_000_000, 156_500_000, 196_500_000)

    assessment = assess_timestamp_sequence(source, received, tolerance_ns=1_000_000)

    assert not assessment.passed
    assert not assessment.relative_timing_preserved
    assert not assessment.constant_offset
    assert assessment.max_abs_delta_error_ns == 1_500_000
    assert assessment.offset_span_ns == 1_500_000


def test_assessment_accepts_explicit_omitted_frame_gap() -> None:
    source = (1_000_000_000, 1_050_000_000, 1_150_000_000, 1_200_000_000)
    received = (0, 50_000_000, 150_000_000, 200_000_000)

    assessment = assess_timestamp_sequence(source, received, tolerance_ns=1_000)

    assert assessment.passed
    assert assessment.input_deltas_ns == (50_000_000, 100_000_000, 50_000_000)
    assert assessment.output_deltas_ns == assessment.input_deltas_ns
    assert not assessment.origin_preserved


def test_assessment_accepts_90khz_wrap_crossing_when_relative_time_is_continuous() -> None:
    rtp_wrap_ns = ((1 << 32) * 1_000_000_000) // 90_000
    source = (
        rtp_wrap_ns - 100_000_000,
        rtp_wrap_ns - 50_000_000,
        rtp_wrap_ns + 20_000_000,
        rtp_wrap_ns + 70_000_000,
    )
    received = (0, 50_000_000, 120_000_000, 170_000_000)

    assessment = assess_timestamp_sequence(source, received, tolerance_ns=1_000)

    assert assessment.passed
    assert assessment.delta_errors_ns == (0, 0, 0)
    assert assessment.offset_span_ns == 0


def test_assessment_accepts_fresh_origin_after_receiver_restart() -> None:
    first_source = (3_000_000_000, 3_050_000_000, 3_100_000_000)
    second_source = (17_000_000_000, 17_020_000_000, 17_070_000_000)

    first = assess_timestamp_sequence(first_source, (0, 50_000_000, 100_000_000))
    second = assess_timestamp_sequence(second_source, (0, 20_000_000, 70_000_000))

    assert first.passed
    assert second.passed
    assert not first.origin_preserved
    assert not second.origin_preserved


def test_provenance_result_passes_only_for_explicit_ambiguity() -> None:
    result = TimestampProvenanceProofResult(
        status="PTS_PROVENANCE_NOT_DISTINGUISHABLE",
        gstreamer_version="test",
        variants=(),
        pts_presence_is_provenance=False,
        conclusion="ambiguous",
    )

    assert result.passed
    assert not result.pts_presence_is_provenance


def test_provenance_result_does_not_accept_inconclusive_state() -> None:
    result = TimestampProvenanceProofResult(
        status="INCONCLUSIVE",
        gstreamer_version="test",
        variants=(),
        pts_presence_is_provenance=False,
        conclusion="not proven",
    )

    assert not result.passed


def test_receiver_recovery_result_passes_only_for_recovered_same_sender_timeline() -> None:
    identity_before = RtpSenderIdentity(
        ssrc=123,
        seqnum=10,
        timestamp=1000,
        seqnum_offset=1,
        timestamp_offset=100,
    )
    identity_after = RtpSenderIdentity(
        ssrc=123,
        seqnum=20,
        timestamp=2000,
        seqnum_offset=1,
        timestamp_offset=100,
    )
    result = TimestampReceiverRecoveryProofResult(
        status="PC_RECEIVER_RESTART_RECOVERS_TIMELINE",
        gstreamer_version="test",
        tolerance_ns=1_000_000,
        first_source_pts_ns=(1, 2),
        first_output_pts_ns=(0, 1),
        second_source_before_reset_pts_ns=(10, 20),
        second_output_before_reset_pts_ns=(1, 1),
        second_before_reset_received_all_frames=True,
        second_before_reset_relative_timing_preserved=False,
        recovery_source_pts_ns=(30, 40),
        recovery_output_pts_ns=(0, 10),
        recovery_received_all_frames=True,
        recovery_relative_timing_preserved=True,
        second_sender_identity_before_reset=identity_before,
        second_sender_identity_after_reset=identity_after,
        same_second_sender_ssrc=True,
        conclusion="recovered",
    )

    assert result.passed
    assert result.same_second_sender_ssrc
    assert result.recovery_relative_timing_preserved


def test_receiver_recovery_result_rejects_failed_recovery() -> None:
    identity = RtpSenderIdentity(
        ssrc=123,
        seqnum=10,
        timestamp=1000,
        seqnum_offset=1,
        timestamp_offset=100,
    )
    result = TimestampReceiverRecoveryProofResult(
        status="PC_RECEIVER_RESTART_DID_NOT_RECOVER_TIMELINE",
        gstreamer_version="test",
        tolerance_ns=1_000_000,
        first_source_pts_ns=(1, 2),
        first_output_pts_ns=(0, 1),
        second_source_before_reset_pts_ns=(10, 20),
        second_output_before_reset_pts_ns=(1, 1),
        second_before_reset_received_all_frames=True,
        second_before_reset_relative_timing_preserved=False,
        recovery_source_pts_ns=(30, 40),
        recovery_output_pts_ns=(5, 5),
        recovery_received_all_frames=True,
        recovery_relative_timing_preserved=False,
        second_sender_identity_before_reset=identity,
        second_sender_identity_after_reset=identity,
        same_second_sender_ssrc=True,
        conclusion="not recovered",
    )

    assert not result.passed
