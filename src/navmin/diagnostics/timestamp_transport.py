"""Synthetic RTP/JPEG timestamp transport proof used during camera integration audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from threading import Event, Lock
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np

from navmin.vision.gstreamer_source import RTP_JPEG_CAPS

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 20
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_TOLERANCE_NS = 1_000_000

_REQUIRED_GSTREAMER_ELEMENTS = (
    "appsrc",
    "jpegparse",
    "rtpjpegpay",
    "udpsink",
    "udpsrc",
    "rtpjitterbuffer",
    "rtpjpegdepay",
    "jpegdec",
    "videoconvert",
    "appsink",
)


class TimestampTransportProofError(RuntimeError):
    """One required synthetic timestamp transport observation failed."""


@dataclass(frozen=True)
class TimestampSequenceAssessment:
    input_pts_ns: tuple[int, ...]
    output_pts_ns: tuple[int, ...]
    input_deltas_ns: tuple[int, ...]
    output_deltas_ns: tuple[int, ...]
    delta_errors_ns: tuple[int, ...]
    per_frame_offsets_ns: tuple[int, ...]
    max_abs_delta_error_ns: int
    offset_span_ns: int
    relative_timing_preserved: bool
    constant_offset: bool
    origin_preserved: bool

    @property
    def passed(self) -> bool:
        return self.relative_timing_preserved and self.constant_offset


@dataclass(frozen=True)
class TimestampTransportCaseResult:
    name: str
    assessment: TimestampSequenceAssessment
    decoded_width: int
    decoded_height: int

    @property
    def passed(self) -> bool:
        return self.assessment.passed


@dataclass(frozen=True)
class TimestampTransportProofResult:
    status: str
    gstreamer_version: str
    tolerance_ns: int
    cases: tuple[TimestampTransportCaseResult, ...]
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimestampTransportObservationCaseResult:
    name: str
    input_pts_ns: tuple[int, ...]
    output_pts_ns: tuple[int, ...]
    decoded_width: int
    decoded_height: int
    received_all_frames: bool
    output_monotonic: bool
    note: str


@dataclass(frozen=True)
class TimestampTransportRestartResult:
    name: str
    first_assessment: TimestampSequenceAssessment
    second_assessment: TimestampSequenceAssessment
    second_session_origin_preserved: bool
    note: str

    @property
    def passed(self) -> bool:
        return self.first_assessment.passed and self.second_assessment.passed


@dataclass(frozen=True)
class RtpSenderIdentity:
    ssrc: int
    seqnum: int
    timestamp: int
    seqnum_offset: int
    timestamp_offset: int


@dataclass(frozen=True)
class TimestampSenderRestartVariantResult:
    name: str
    second_sender_controls: dict[str, int]
    first_sender_identity: RtpSenderIdentity
    second_sender_identity: RtpSenderIdentity
    first_output_pts_ns: tuple[int, ...]
    second_output_pts_ns: tuple[int, ...]
    jitter_stats_before_restart: dict[str, int]
    jitter_stats_after_restart: dict[str, int]
    second_received_all_frames: bool
    second_relative_timing_preserved: bool | None
    note: str


@dataclass(frozen=True)
class TimestampSenderRestartProofResult:
    status: str
    gstreamer_version: str
    variants: tuple[TimestampSenderRestartVariantResult, ...]
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status in {"FINDING_CONFIRMED", "RANDOM_RESTART_RECOVERED"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimestampProvenanceVariantResult:
    name: str
    source_timestamp_present: bool
    appsrc_do_timestamp: bool
    input_pts_ns: tuple[int | None, ...]
    output_pts_ns: tuple[int, ...]
    received_all_frames: bool
    output_pts_available: bool
    output_monotonic: bool
    output_strictly_increasing: bool
    note: str


@dataclass(frozen=True)
class TimestampProvenanceProofResult:
    status: str
    gstreamer_version: str
    variants: tuple[TimestampProvenanceVariantResult, ...]
    pts_presence_is_provenance: bool
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status == "PTS_PROVENANCE_NOT_DISTINGUISHABLE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimestampReceiverRecoveryProofResult:
    status: str
    gstreamer_version: str
    tolerance_ns: int
    first_source_pts_ns: tuple[int, ...]
    first_output_pts_ns: tuple[int, ...]
    second_source_before_reset_pts_ns: tuple[int, ...]
    second_output_before_reset_pts_ns: tuple[int, ...]
    second_before_reset_received_all_frames: bool
    second_before_reset_relative_timing_preserved: bool | None
    recovery_source_pts_ns: tuple[int, ...]
    recovery_output_pts_ns: tuple[int, ...]
    recovery_received_all_frames: bool
    recovery_relative_timing_preserved: bool
    second_sender_identity_before_reset: RtpSenderIdentity
    second_sender_identity_after_reset: RtpSenderIdentity
    same_second_sender_ssrc: bool
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status == "PC_RECEIVER_RESTART_RECOVERS_TIMELINE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimestampTransportLifecycleProofResult:
    status: str
    gstreamer_version: str
    tolerance_ns: int
    deterministic_cases: tuple[TimestampTransportCaseResult, ...]
    observation_cases: tuple[TimestampTransportObservationCaseResult, ...]
    restart_cases: tuple[TimestampTransportRestartResult, ...]
    limitations: tuple[str, ...]
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_timestamp_sender_pipeline_description(*, host: str, port: int, fps: int) -> str:
    """Build the colleague-style appsrc JPEG -> RTP/JPEG sender pipeline."""
    return (
        "appsrc name=source is-live=true format=time do-timestamp=false "
        f"caps=image/jpeg,framerate={fps}/1 "
        "! jpegparse "
        "! rtpjpegpay pt=26 "
        f"! udpsink host={host} port={port} sync=false async=false"
    )


def build_timestamp_receiver_pipeline_description(*, address: str, port: int) -> str:
    """Build the candidate receiver chain from the colleague timestamp archive."""
    quoted_caps = '"' + RTP_JPEG_CAPS.replace('"', '\\"') + '"'
    return (
        f'udpsrc address="{address}" port={port} caps={quoted_caps} '
        "! rtpjitterbuffer latency=0 mode=0 "
        "! rtpjpegdepay "
        "! jpegdec "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    )


def assess_timestamp_sequence(
    input_pts_ns: tuple[int, ...],
    output_pts_ns: tuple[int, ...],
    *,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
) -> TimestampSequenceAssessment:
    """Compare source and received PTS without assuming a shared absolute origin."""
    if len(input_pts_ns) != len(output_pts_ns):
        raise ValueError("input and output timestamp counts must match")
    if len(input_pts_ns) < 2:
        raise ValueError("at least two timestamps are required")
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be >= 0")

    input_deltas = tuple(
        current - previous for previous, current in pairwise(input_pts_ns)
    )
    output_deltas = tuple(
        current - previous for previous, current in pairwise(output_pts_ns)
    )
    delta_errors = tuple(
        output - expected for expected, output in zip(input_deltas, output_deltas)
    )
    offsets = tuple(output - source for source, output in zip(input_pts_ns, output_pts_ns))
    max_abs_delta_error = max(abs(error) for error in delta_errors)
    offset_span = max(offsets) - min(offsets)

    return TimestampSequenceAssessment(
        input_pts_ns=input_pts_ns,
        output_pts_ns=output_pts_ns,
        input_deltas_ns=input_deltas,
        output_deltas_ns=output_deltas,
        delta_errors_ns=delta_errors,
        per_frame_offsets_ns=offsets,
        max_abs_delta_error_ns=max_abs_delta_error,
        offset_span_ns=offset_span,
        relative_timing_preserved=max_abs_delta_error <= tolerance_ns,
        constant_offset=offset_span <= tolerance_ns,
        origin_preserved=max(abs(offset) for offset in offsets) <= tolerance_ns,
    )


def _import_gstreamer() -> Any:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp

        Gst.init(None)
        _ = GstApp.AppSink
    except (AttributeError, ImportError, ValueError) as exc:
        raise TimestampTransportProofError(
            "PyGObject with Gst 1.0 and GstApp 1.0 is required"
        ) from exc
    missing = tuple(
        name for name in _REQUIRED_GSTREAMER_ELEMENTS if Gst.ElementFactory.find(name) is None
    )
    if missing:
        raise TimestampTransportProofError(
            "missing required GStreamer elements: " + ", ".join(missing)
        )
    return Gst


def _jpeg_payload(frame_index: int, *, width: int, height: int) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (
        (31 + frame_index * 17) % 255,
        (71 + frame_index * 29) % 255,
        (113 + frame_index * 37) % 255,
    )
    cv2.putText(
        frame,
        f"TSP1 {frame_index}",
        (16, max(32, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise TimestampTransportProofError("OpenCV failed to encode synthetic JPEG")
    return encoded.tobytes()


def _bus_failure(gst: Any, pipeline: Any) -> TimestampTransportProofError | None:
    bus = pipeline.get_bus()
    message = bus.timed_pop_filtered(
        0,
        gst.MessageType.ERROR | gst.MessageType.EOS,
    )
    if message is None:
        return None
    if message.type == gst.MessageType.ERROR:
        error, debug = message.parse_error()
        suffix = f": {debug}" if debug else ""
        return TimestampTransportProofError(f"GStreamer error: {error}{suffix}")
    return TimestampTransportProofError("GStreamer pipeline reached EOS unexpectedly")


def _run_case(
    gst: Any,
    *,
    name: str,
    source_pts_ns: tuple[int, ...],
    port: int,
    tolerance_ns: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
) -> TimestampTransportCaseResult:
    sender = gst.parse_launch(
        build_timestamp_sender_pipeline_description(host="127.0.0.1", port=port, fps=fps)
    )
    receiver = gst.parse_launch(
        build_timestamp_receiver_pipeline_description(address="127.0.0.1", port=port)
    )
    source = sender.get_by_name("source")
    sink = receiver.get_by_name("sink")
    if source is None or sink is None:
        raise TimestampTransportProofError("GStreamer proof pipeline omitted appsrc/appsink")

    received_pts: list[int] = []
    decoded_sizes: list[tuple[int, int]] = []
    receive_event = Event()
    receive_lock = Lock()

    def on_sample(appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                raise TimestampTransportProofError("appsink returned no sample")
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            if buffer is None or caps is None or caps.get_size() < 1:
                raise TimestampTransportProofError("decoded sample omitted buffer/caps")
            pts = int(buffer.pts)
            if pts == int(gst.CLOCK_TIME_NONE):
                raise TimestampTransportProofError("decoded sample PTS is unavailable")
            structure = caps.get_structure(0)
            decoded_width = int(structure.get_value("width"))
            decoded_height = int(structure.get_value("height"))
            with receive_lock:
                received_pts.append(pts)
                decoded_sizes.append((decoded_width, decoded_height))
                receive_event.set()
            return gst.FlowReturn.OK
        except TimestampTransportProofError:
            return gst.FlowReturn.ERROR
        except (AttributeError, TypeError, ValueError):
            return gst.FlowReturn.ERROR

    handler_id = sink.connect("new-sample", on_sample)
    receiver_started = False
    sender_started = False
    try:
        if receiver.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError("receiver failed to enter PLAYING")
        receiver_started = True
        if sender.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError("sender failed to enter PLAYING")
        sender_started = True
        sleep(0.15)

        for frame_index, pts_ns in enumerate(source_pts_ns):
            payload = _jpeg_payload(frame_index, width=width, height=height)
            buffer = gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = pts_ns
            buffer.dts = gst.CLOCK_TIME_NONE
            buffer.duration = gst.SECOND // fps
            with receive_lock:
                before = len(received_pts)
                receive_event.clear()
            flow = source.emit("push-buffer", buffer)
            if flow != gst.FlowReturn.OK:
                raise TimestampTransportProofError(f"appsrc push failed: {flow}")

            deadline = monotonic() + timeout_seconds
            while monotonic() < deadline:
                sender_failure = _bus_failure(gst, sender)
                receiver_failure = _bus_failure(gst, receiver)
                if sender_failure is not None:
                    raise sender_failure
                if receiver_failure is not None:
                    raise receiver_failure
                with receive_lock:
                    if len(received_pts) > before:
                        break
                receive_event.wait(0.01)
            else:
                raise TimestampTransportProofError(
                    f"case {name}: frame {frame_index} did not reach appsink"
                )

        with receive_lock:
            output_pts = tuple(received_pts)
            sizes = tuple(decoded_sizes)
        if len(output_pts) != len(source_pts_ns):
            raise TimestampTransportProofError(
                f"case {name}: received {len(output_pts)} frames, expected {len(source_pts_ns)}"
            )
        if not sizes or any(size != (width, height) for size in sizes):
            raise TimestampTransportProofError(
                f"case {name}: decoded dimensions differ from {width}x{height}: {sizes!r}"
            )
        assessment = assess_timestamp_sequence(
            source_pts_ns,
            output_pts,
            tolerance_ns=tolerance_ns,
        )
        return TimestampTransportCaseResult(
            name=name,
            assessment=assessment,
            decoded_width=width,
            decoded_height=height,
        )
    finally:
        try:
            source.emit("end-of-stream")
        except (AttributeError, TypeError):
            pass
        if sender_started:
            sender.set_state(gst.State.NULL)
        if receiver_started:
            receiver.set_state(gst.State.NULL)
        sink.disconnect(handler_id)



def _run_observation_case(
    gst: Any,
    *,
    name: str,
    source_pts_ns: tuple[int, ...],
    port: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
) -> TimestampTransportObservationCaseResult:
    """Observe duplicate/backward PTS without assuming delivery semantics."""
    sender = gst.parse_launch(
        build_timestamp_sender_pipeline_description(host="127.0.0.1", port=port, fps=fps)
    )
    receiver = gst.parse_launch(
        build_timestamp_receiver_pipeline_description(address="127.0.0.1", port=port)
    )
    source = sender.get_by_name("source")
    sink = receiver.get_by_name("sink")
    if source is None or sink is None:
        raise TimestampTransportProofError(
            "GStreamer observation pipeline omitted appsrc/appsink"
        )

    received_pts: list[int] = []
    decoded_sizes: list[tuple[int, int]] = []
    receive_lock = Lock()

    def on_sample(appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            if buffer is None or caps is None or caps.get_size() < 1:
                return gst.FlowReturn.ERROR
            pts = int(buffer.pts)
            if pts == int(gst.CLOCK_TIME_NONE):
                return gst.FlowReturn.ERROR
            structure = caps.get_structure(0)
            size = (
                int(structure.get_value("width")),
                int(structure.get_value("height")),
            )
            with receive_lock:
                received_pts.append(pts)
                decoded_sizes.append(size)
            return gst.FlowReturn.OK
        except (AttributeError, TypeError, ValueError):
            return gst.FlowReturn.ERROR

    handler_id = sink.connect("new-sample", on_sample)
    receiver_started = False
    sender_started = False
    try:
        if receiver.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError("receiver failed to enter PLAYING")
        receiver_started = True
        if sender.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError("sender failed to enter PLAYING")
        sender_started = True
        sleep(0.15)

        for frame_index, pts_ns in enumerate(source_pts_ns):
            payload = _jpeg_payload(frame_index, width=width, height=height)
            buffer = gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = pts_ns
            buffer.dts = gst.CLOCK_TIME_NONE
            buffer.duration = gst.SECOND // fps
            flow = source.emit("push-buffer", buffer)
            if flow != gst.FlowReturn.OK:
                raise TimestampTransportProofError(f"appsrc push failed: {flow}")
            sleep(min(0.03, 0.5 / fps))

        deadline = monotonic() + timeout_seconds
        previous_count = -1
        stable_since = monotonic()
        while monotonic() < deadline:
            sender_failure = _bus_failure(gst, sender)
            receiver_failure = _bus_failure(gst, receiver)
            if sender_failure is not None:
                raise sender_failure
            if receiver_failure is not None:
                raise receiver_failure
            with receive_lock:
                current_count = len(received_pts)
            if current_count != previous_count:
                previous_count = current_count
                stable_since = monotonic()
            elif monotonic() - stable_since >= 0.2:
                break
            sleep(0.01)

        with receive_lock:
            output_pts = tuple(received_pts)
            sizes = tuple(decoded_sizes)
        if sizes and any(size != (width, height) for size in sizes):
            raise TimestampTransportProofError(
                f"case {name}: decoded dimensions differ from {width}x{height}: {sizes!r}"
            )
        output_monotonic = all(
            current >= previous for previous, current in pairwise(output_pts)
        )
        return TimestampTransportObservationCaseResult(
            name=name,
            input_pts_ns=source_pts_ns,
            output_pts_ns=output_pts,
            decoded_width=width,
            decoded_height=height,
            received_all_frames=len(output_pts) == len(source_pts_ns),
            output_monotonic=output_monotonic,
            note=(
                "Observation only: duplicate/backward input PTS may be dropped, "
                "coalesced, reordered, or normalized by RTP/jitterbuffer semantics."
            ),
        )
    finally:
        try:
            source.emit("end-of-stream")
        except (AttributeError, TypeError):
            pass
        if sender_started:
            sender.set_state(gst.State.NULL)
        if receiver_started:
            receiver.set_state(gst.State.NULL)
        sink.disconnect(handler_id)


def _run_receiver_restart_case(
    gst: Any,
    *,
    port: int,
    tolerance_ns: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
) -> TimestampTransportRestartResult:
    first = _run_case(
        gst,
        name="receiver-restart-first",
        source_pts_ns=(3_000_000_000, 3_050_000_000, 3_100_000_000),
        port=port,
        tolerance_ns=tolerance_ns,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
    )
    sleep(0.1)
    second = _run_case(
        gst,
        name="receiver-restart-second",
        source_pts_ns=(17_000_000_000, 17_020_000_000, 17_070_000_000),
        port=port,
        tolerance_ns=tolerance_ns,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
    )
    return TimestampTransportRestartResult(
        name="receiver-restart",
        first_assessment=first.assessment,
        second_assessment=second.assessment,
        second_session_origin_preserved=second.assessment.origin_preserved,
        note=(
            "A fresh receiver pipeline is expected to establish a fresh local PTS origin; "
            "relative timing must remain preserved in both sessions."
        ),
    )


def _run_sender_restart_case(
    gst: Any,
    *,
    port: int,
    tolerance_ns: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
) -> TimestampTransportRestartResult:
    receiver = gst.parse_launch(
        build_timestamp_receiver_pipeline_description(address="127.0.0.1", port=port)
    )
    sink = receiver.get_by_name("sink")
    if sink is None:
        raise TimestampTransportProofError("sender-restart receiver omitted appsink")

    received_pts: list[int] = []
    receive_event = Event()
    receive_lock = Lock()

    def on_sample(appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            if buffer is None or int(buffer.pts) == int(gst.CLOCK_TIME_NONE):
                return gst.FlowReturn.ERROR
            with receive_lock:
                received_pts.append(int(buffer.pts))
                receive_event.set()
            return gst.FlowReturn.OK
        except (AttributeError, TypeError, ValueError):
            return gst.FlowReturn.ERROR

    handler_id = sink.connect("new-sample", on_sample)
    receiver_started = False

    def send_session(
        source_pts_ns: tuple[int, ...],
        session_index: int,
    ) -> tuple[int, ...]:
        sender = gst.parse_launch(
            build_timestamp_sender_pipeline_description(
                host="127.0.0.1",
                port=port,
                fps=fps,
            )
        )
        source = sender.get_by_name("source")
        if source is None:
            raise TimestampTransportProofError("sender-restart pipeline omitted appsrc")
        if sender.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                "sender restart session failed to enter PLAYING"
            )
        try:
            sleep(0.15)
            with receive_lock:
                start_count = len(received_pts)
            for frame_index, pts_ns in enumerate(source_pts_ns):
                payload = _jpeg_payload(
                    session_index * 100 + frame_index,
                    width=width,
                    height=height,
                )
                buffer = gst.Buffer.new_allocate(None, len(payload), None)
                buffer.fill(0, payload)
                buffer.pts = pts_ns
                buffer.dts = gst.CLOCK_TIME_NONE
                buffer.duration = gst.SECOND // fps
                with receive_lock:
                    before = len(received_pts)
                    receive_event.clear()
                flow = source.emit("push-buffer", buffer)
                if flow != gst.FlowReturn.OK:
                    raise TimestampTransportProofError(f"appsrc push failed: {flow}")
                deadline = monotonic() + timeout_seconds
                while monotonic() < deadline:
                    sender_failure = _bus_failure(gst, sender)
                    receiver_failure = _bus_failure(gst, receiver)
                    if sender_failure is not None:
                        raise sender_failure
                    if receiver_failure is not None:
                        raise receiver_failure
                    with receive_lock:
                        if len(received_pts) > before:
                            break
                    receive_event.wait(0.01)
                else:
                    raise TimestampTransportProofError(
                        f"sender restart session {session_index} frame {frame_index} "
                        "did not reach appsink"
                    )
            with receive_lock:
                return tuple(received_pts[start_count:])
        finally:
            try:
                source.emit("end-of-stream")
            except (AttributeError, TypeError):
                pass
            sender.set_state(gst.State.NULL)

    try:
        if receiver.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                "sender-restart receiver failed to enter PLAYING"
            )
        receiver_started = True
        first_source = (4_000_000_000, 4_050_000_000, 4_100_000_000)
        second_source = (25_000_000_000, 25_020_000_000, 25_070_000_000)
        first_output = send_session(first_source, 1)
        sleep(0.2)
        second_output = send_session(second_source, 2)
        if len(first_output) != len(first_source) or len(second_output) != len(second_source):
            raise TimestampTransportProofError(
                "sender restart did not deliver all frames in both sessions"
            )
        first_assessment = assess_timestamp_sequence(
            first_source,
            first_output,
            tolerance_ns=tolerance_ns,
        )
        second_assessment = assess_timestamp_sequence(
            second_source,
            second_output,
            tolerance_ns=tolerance_ns,
        )
        return TimestampTransportRestartResult(
            name="sender-restart-persistent-receiver",
            first_assessment=first_assessment,
            second_assessment=second_assessment,
            second_session_origin_preserved=second_assessment.origin_preserved,
            note=(
                "The receiver remains alive while a new RTP sender pipeline/session starts. "
                "Both source-relative timelines must remain internally coherent."
            ),
        )
    finally:
        if receiver_started:
            receiver.set_state(gst.State.NULL)
        sink.disconnect(handler_id)


def _gst_structure_int(structure: Any, field: str) -> int:
    try:
        value = structure.get_value(field)
    except (AttributeError, TypeError) as exc:
        raise TimestampTransportProofError(
            f"GStreamer stats omitted required field {field!r}"
        ) from exc
    if value is None:
        raise TimestampTransportProofError(
            f"GStreamer stats omitted required field {field!r}"
        )
    return int(value)


def _sender_identity(payloader: Any) -> RtpSenderIdentity:
    stats = payloader.get_property("stats")
    if stats is None:
        raise TimestampTransportProofError("RTP payloader stats are unavailable")
    return RtpSenderIdentity(
        ssrc=_gst_structure_int(stats, "ssrc"),
        seqnum=_gst_structure_int(stats, "seqnum"),
        timestamp=_gst_structure_int(stats, "timestamp"),
        seqnum_offset=_gst_structure_int(stats, "seqnum-offset"),
        timestamp_offset=_gst_structure_int(stats, "timestamp-offset"),
    )


def _jitter_stats(jitterbuffer: Any) -> dict[str, int]:
    stats = jitterbuffer.get_property("stats")
    if stats is None:
        return {}
    fields = (
        "num-pushed",
        "num-lost",
        "num-late",
        "num-duplicates",
        "avg-jitter",
    )
    result: dict[str, int] = {}
    for field in fields:
        try:
            value = stats.get_value(field)
        except (AttributeError, TypeError):
            continue
        if value is not None:
            result[field] = int(value)
    return result


def _build_restart_sender_pipeline_description(*, host: str, port: int, fps: int) -> str:
    return (
        "appsrc name=source is-live=true format=time do-timestamp=false "
        f"caps=image/jpeg,framerate={fps}/1 "
        "! jpegparse "
        "! rtpjpegpay name=pay pt=26 "
        f"! udpsink host={host} port={port} sync=false async=false"
    )


def _build_restart_receiver_pipeline_description(*, address: str, port: int) -> str:
    quoted_caps = '"' + RTP_JPEG_CAPS.replace('"', '\\"') + '"'
    return (
        f'udpsrc address="{address}" port={port} caps={quoted_caps} '
        "! rtpjitterbuffer name=jitter latency=0 mode=0 "
        "! rtpjpegdepay "
        "! jpegdec "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    )


def _continuation_controls(identity: RtpSenderIdentity, *, gap_ns: int) -> dict[str, int]:
    if gap_ns < 0:
        raise ValueError("gap_ns must be >= 0")
    timestamp_step = round(gap_ns * 90_000 / 1_000_000_000)
    return {
        "ssrc": identity.ssrc,
        "seqnum-offset": (identity.seqnum + 1) & 0xFFFF,
        "timestamp-offset": (identity.timestamp + timestamp_step) & 0xFFFFFFFF,
    }


def _run_sender_restart_variant(
    gst: Any,
    *,
    name: str,
    port: int,
    second_control_mode: str,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
    tolerance_ns: int,
) -> TimestampSenderRestartVariantResult:
    receiver = gst.parse_launch(
        _build_restart_receiver_pipeline_description(address="127.0.0.1", port=port)
    )
    sink = receiver.get_by_name("sink")
    jitter = receiver.get_by_name("jitter")
    if sink is None or jitter is None:
        raise TimestampTransportProofError(
            "sender-restart diagnostic receiver omitted appsink/jitterbuffer"
        )

    received_pts: list[int] = []
    receive_lock = Lock()

    def on_sample(appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            if buffer is None or int(buffer.pts) == int(gst.CLOCK_TIME_NONE):
                return gst.FlowReturn.ERROR
            with receive_lock:
                received_pts.append(int(buffer.pts))
            return gst.FlowReturn.OK
        except (AttributeError, TypeError, ValueError):
            return gst.FlowReturn.ERROR

    handler_id = sink.connect("new-sample", on_sample)
    receiver_started = False

    def send_session(
        source_pts_ns: tuple[int, ...],
        *,
        session_index: int,
        controls: dict[str, int] | None,
    ) -> tuple[tuple[int, ...], RtpSenderIdentity]:
        sender = gst.parse_launch(
            _build_restart_sender_pipeline_description(
                host="127.0.0.1",
                port=port,
                fps=fps,
            )
        )
        source = sender.get_by_name("source")
        pay = sender.get_by_name("pay")
        if source is None or pay is None:
            raise TimestampTransportProofError(
                "sender-restart diagnostic sender omitted appsrc/payloader"
            )
        if controls:
            for property_name, value in controls.items():
                pay.set_property(property_name, value)
        if sender.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                f"sender restart variant {name} session {session_index} failed to enter PLAYING"
            )
        try:
            sleep(0.15)
            with receive_lock:
                start_count = len(received_pts)
            for frame_index, pts_ns in enumerate(source_pts_ns):
                payload = _jpeg_payload(
                    session_index * 100 + frame_index,
                    width=width,
                    height=height,
                )
                buffer = gst.Buffer.new_allocate(None, len(payload), None)
                buffer.fill(0, payload)
                buffer.pts = pts_ns
                buffer.dts = gst.CLOCK_TIME_NONE
                buffer.duration = gst.SECOND // fps
                flow = source.emit("push-buffer", buffer)
                if flow != gst.FlowReturn.OK:
                    raise TimestampTransportProofError(f"appsrc push failed: {flow}")
                sleep(min(0.03, 0.5 / fps))

            deadline = monotonic() + timeout_seconds
            previous_count = -1
            stable_since = monotonic()
            while monotonic() < deadline:
                sender_failure = _bus_failure(gst, sender)
                receiver_failure = _bus_failure(gst, receiver)
                if sender_failure is not None:
                    raise sender_failure
                if receiver_failure is not None:
                    raise receiver_failure
                with receive_lock:
                    current_count = len(received_pts)
                if current_count != previous_count:
                    previous_count = current_count
                    stable_since = monotonic()
                elif monotonic() - stable_since >= 0.25:
                    break
                sleep(0.01)
            identity = _sender_identity(pay)
            with receive_lock:
                output = tuple(received_pts[start_count:])
            return output, identity
        finally:
            try:
                source.emit("end-of-stream")
            except (AttributeError, TypeError):
                pass
            sender.set_state(gst.State.NULL)

    try:
        if receiver.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                f"sender restart variant {name}: receiver failed to enter PLAYING"
            )
        receiver_started = True
        first_source = (4_000_000_000, 4_050_000_000, 4_100_000_000)
        second_source = (25_000_000_000, 25_020_000_000, 25_070_000_000)
        first_output, first_identity = send_session(
            first_source,
            session_index=1,
            controls=None,
        )
        if len(first_output) != len(first_source):
            raise TimestampTransportProofError(
                f"sender restart variant {name}: first session received "
                f"{len(first_output)}/{len(first_source)} frames"
            )
        jitter_before = _jitter_stats(jitter)
        sleep(0.2)
        continuity = _continuation_controls(first_identity, gap_ns=50_000_000)
        if second_control_mode == "random":
            second_controls: dict[str, int] = {}
        elif second_control_mode == "same-ssrc":
            second_controls = {"ssrc": first_identity.ssrc}
        elif second_control_mode == "continued-seq-ts-new-ssrc":
            second_controls = {
                "seqnum-offset": continuity["seqnum-offset"],
                "timestamp-offset": continuity["timestamp-offset"],
            }
        elif second_control_mode == "full-continuity":
            second_controls = continuity
        else:
            raise ValueError(f"unknown second_control_mode: {second_control_mode}")
        second_output, second_identity = send_session(
            second_source,
            session_index=2,
            controls=second_controls,
        )
        jitter_after = _jitter_stats(jitter)
        second_preserved: bool | None = None
        if len(second_output) == len(second_source):
            second_preserved = assess_timestamp_sequence(
                second_source,
                second_output,
                tolerance_ns=tolerance_ns,
            ).passed
        return TimestampSenderRestartVariantResult(
            name=name,
            second_sender_controls=second_controls,
            first_sender_identity=first_identity,
            second_sender_identity=second_identity,
            first_output_pts_ns=first_output,
            second_output_pts_ns=second_output,
            jitter_stats_before_restart=jitter_before,
            jitter_stats_after_restart=jitter_after,
            second_received_all_frames=len(second_output) == len(second_source),
            second_relative_timing_preserved=second_preserved,
            note=(
                "Each variant uses a fresh receiver. Only the second sender's RTP identity/"
                "continuity controls differ; the candidate receiver chain is otherwise unchanged."
            ),
        )
    finally:
        if receiver_started:
            receiver.set_state(gst.State.NULL)
        sink.disconnect(handler_id)


def run_timestamp_sender_restart_proof(
    *,
    base_port: int = 18_918,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TimestampSenderRestartProofResult:
    """Determine whether persistent-receiver recovery depends on RTP identity continuity."""
    if type(base_port) is not int or not 1 <= base_port <= 65_531:
        raise ValueError("base_port must be an integer in range 1..65531")
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be >= 0")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    gst = _import_gstreamer()
    variants = tuple(
        _run_sender_restart_variant(
            gst,
            name=name,
            port=base_port + index,
            second_control_mode=mode,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
            tolerance_ns=tolerance_ns,
        )
        for index, (name, mode) in enumerate(
            (
                ("random-new-sender", "random"),
                ("same-ssrc-only", "same-ssrc"),
                ("continued-seq-ts-new-ssrc", "continued-seq-ts-new-ssrc"),
                ("full-rtp-continuity", "full-continuity"),
            )
        )
    )
    by_name = {variant.name: variant for variant in variants}
    random_ok = by_name["random-new-sender"].second_received_all_frames
    full = by_name["full-rtp-continuity"]
    full_ok = full.second_received_all_frames and full.second_relative_timing_preserved is True
    if random_ok:
        status = "RANDOM_RESTART_RECOVERED"
        conclusion = (
            "The fresh random RTP sender recovered through the persistent candidate receiver; "
            "the previously observed block was not reproduced in this controlled matrix."
        )
    elif full_ok:
        status = "FINDING_CONFIRMED"
        conclusion = (
            "A fresh random RTP sender did not recover through the persistent single-stream "
            "receiver, while explicit RTP continuity did. Sender restart therefore depends on "
            "RTP session identity/sequence/timestamp continuity in this candidate chain."
        )
    else:
        status = "INCONCLUSIVE"
        conclusion = (
            "The random sender restart did not recover and explicit RTP continuity also failed; "
            "the current proof cannot isolate the restart boundary."
        )
    return TimestampSenderRestartProofResult(
        status=status,
        gstreamer_version=gst.version_string(),
        variants=variants,
        conclusion=conclusion,
    )


def write_timestamp_sender_restart_json(
    result: TimestampSenderRestartProofResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_timestamp_transport_proof(
    *,
    base_port: int = 18_898,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TimestampTransportProofResult:
    """Run regular and irregular localhost timestamp transport cases."""
    if type(base_port) is not int or not 1 <= base_port <= 65_534:
        raise ValueError("base_port must be an integer in range 1..65534")
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be >= 0")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    gst = _import_gstreamer()
    cases = (
        _run_case(
            gst,
            name="regular-50ms-nonzero-origin",
            source_pts_ns=(
                5_000_000_000,
                5_050_000_000,
                5_100_000_000,
                5_150_000_000,
                5_200_000_000,
            ),
            port=base_port,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
        _run_case(
            gst,
            name="irregular-nonzero-origin",
            source_pts_ns=(
                9_000_000_000,
                9_020_000_000,
                9_055_000_000,
                9_095_000_000,
                9_160_000_000,
            ),
            port=base_port + 1,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
    )
    passed = all(case.passed for case in cases)
    if passed:
        if all(case.assessment.origin_preserved for case in cases):
            conclusion = (
                "RTP/JPEG preserved source timestamp deltas and absolute PTS origin "
                "within the configured tolerance."
            )
        else:
            conclusion = (
                "RTP/JPEG preserved source timestamp deltas with a stable per-case "
                "offset; absolute source PTS origin was not preserved."
            )
    else:
        conclusion = (
            "RTP/JPEG did not preserve the source timestamp deltas with a stable "
            "offset within the configured tolerance."
        )
    return TimestampTransportProofResult(
        status="PASS" if passed else "FAIL",
        gstreamer_version=gst.version_string(),
        tolerance_ns=tolerance_ns,
        cases=cases,
        conclusion=conclusion,
    )



def run_timestamp_transport_lifecycle_proof(
    *,
    base_port: int = 18_908,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TimestampTransportLifecycleProofResult:
    """Run TSP-2 timing-gap, wrap, restart, and invalid-PTS observations."""
    if type(base_port) is not int or not 1 <= base_port <= 65_528:
        raise ValueError("base_port must be an integer in range 1..65528")
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be >= 0")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    gst = _import_gstreamer()
    rtp_wrap_ns = ((1 << 32) * 1_000_000_000) // 90_000
    deterministic_cases = (
        _run_case(
            gst,
            name="omitted-source-frame-gap",
            source_pts_ns=(
                1_000_000_000,
                1_050_000_000,
                1_150_000_000,
                1_200_000_000,
            ),
            port=base_port,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
        _run_case(
            gst,
            name="rtp-90khz-wrap-boundary",
            source_pts_ns=(
                rtp_wrap_ns - 100_000_000,
                rtp_wrap_ns - 50_000_000,
                rtp_wrap_ns + 20_000_000,
                rtp_wrap_ns + 70_000_000,
            ),
            port=base_port + 1,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
        _run_case(
            gst,
            name="large-source-pts",
            source_pts_ns=(
                6 * rtp_wrap_ns + 3_000_000_000,
                6 * rtp_wrap_ns + 3_020_000_000,
                6 * rtp_wrap_ns + 3_055_000_000,
                6 * rtp_wrap_ns + 3_095_000_000,
            ),
            port=base_port + 2,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
    )
    observation_cases = (
        _run_observation_case(
            gst,
            name="duplicate-source-pts",
            source_pts_ns=(
                2_000_000_000,
                2_050_000_000,
                2_050_000_000,
                2_100_000_000,
            ),
            port=base_port + 3,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
        _run_observation_case(
            gst,
            name="backward-source-pts",
            source_pts_ns=(
                8_000_000_000,
                8_050_000_000,
                8_030_000_000,
                8_100_000_000,
            ),
            port=base_port + 4,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
    )
    restart_cases = (
        _run_receiver_restart_case(
            gst,
            port=base_port + 5,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
        _run_sender_restart_case(
            gst,
            port=base_port + 6,
            tolerance_ns=tolerance_ns,
            width=width,
            height=height,
            fps=fps,
            timeout_seconds=timeout_seconds,
        ),
    )
    passed = all(case.passed for case in deterministic_cases) and all(
        case.passed for case in restart_cases
    )
    limitations = (
        (
            "Duplicate/backward PTS cases are observational and do not define a pass/fail "
            "delivery contract for rtpjitterbuffer."
        ),
        (
            "Deterministic network packet-loss/jitter injection is not performed in TSP-2; "
            "doing so without altering the production transport boundary requires a separate proof."
        ),
        (
            "TSP-2 does not establish timestamp provenance; legacy-vs-SensorTimestamp "
            "distinguishability is the TSP-3 scope."
        ),
    )
    conclusion = (
        "Gap/wrap/large-PTS and restart timing semantics preserved source-relative "
        "timing within tolerance; duplicate/backward PTS behavior was recorded "
        "without treating it as an accepted camera timestamp contract."
        if passed
        else "One or more required TSP-2 deterministic/restart timing invariants failed."
    )
    return TimestampTransportLifecycleProofResult(
        status="PASS" if passed else "FAIL",
        gstreamer_version=gst.version_string(),
        tolerance_ns=tolerance_ns,
        deterministic_cases=deterministic_cases,
        observation_cases=observation_cases,
        restart_cases=restart_cases,
        limitations=limitations,
        conclusion=conclusion,
    )


def write_timestamp_transport_lifecycle_json(
    result: TimestampTransportLifecycleProofResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_timestamp_transport_proof_json(
    result: TimestampTransportProofResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_provenance_sender_pipeline_description(
    *,
    host: str,
    port: int,
    fps: int,
    do_timestamp: bool,
) -> str:
    do_timestamp_value = "true" if do_timestamp else "false"
    return (
        "appsrc name=source is-live=true format=time "
        f"do-timestamp={do_timestamp_value} "
        f"caps=image/jpeg,framerate={fps}/1 "
        "! jpegparse "
        "! rtpjpegpay pt=26 "
        f"! udpsink host={host} port={port} sync=false async=false"
    )


def _run_provenance_variant(
    gst: Any,
    *,
    name: str,
    source_pts_ns: tuple[int | None, ...],
    source_timestamp_present: bool,
    do_timestamp: bool,
    port: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
    push_interval_seconds: float,
) -> TimestampProvenanceVariantResult:
    sender = gst.parse_launch(
        _build_provenance_sender_pipeline_description(
            host="127.0.0.1",
            port=port,
            fps=fps,
            do_timestamp=do_timestamp,
        )
    )
    receiver = gst.parse_launch(
        build_timestamp_receiver_pipeline_description(address="127.0.0.1", port=port)
    )
    source = sender.get_by_name("source")
    sink = receiver.get_by_name("sink")
    if source is None or sink is None:
        raise TimestampTransportProofError(
            "timestamp provenance pipeline omitted appsrc/appsink"
        )

    received_pts: list[int] = []
    receive_lock = Lock()

    def on_sample(appsink):
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return gst.FlowReturn.ERROR
            buffer = sample.get_buffer()
            if buffer is None:
                return gst.FlowReturn.ERROR
            pts = int(buffer.pts)
            with receive_lock:
                received_pts.append(pts)
            return gst.FlowReturn.OK
        except (AttributeError, TypeError, ValueError):
            return gst.FlowReturn.ERROR

    handler_id = sink.connect("new-sample", on_sample)
    receiver_started = False
    sender_started = False
    try:
        if receiver.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                f"provenance variant {name}: receiver failed to enter PLAYING"
            )
        receiver_started = True
        if sender.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                f"provenance variant {name}: sender failed to enter PLAYING"
            )
        sender_started = True
        sleep(0.15)

        for frame_index, pts_ns in enumerate(source_pts_ns):
            payload = _jpeg_payload(frame_index, width=width, height=height)
            buffer = gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = gst.CLOCK_TIME_NONE if pts_ns is None else pts_ns
            buffer.dts = gst.CLOCK_TIME_NONE
            buffer.duration = gst.SECOND // fps
            flow = source.emit("push-buffer", buffer)
            if flow != gst.FlowReturn.OK:
                raise TimestampTransportProofError(
                    f"provenance variant {name}: appsrc push failed: {flow}"
                )
            if push_interval_seconds > 0:
                sleep(push_interval_seconds)

        deadline = monotonic() + timeout_seconds
        previous_count = -1
        stable_since = monotonic()
        while monotonic() < deadline:
            sender_failure = _bus_failure(gst, sender)
            receiver_failure = _bus_failure(gst, receiver)
            if sender_failure is not None:
                raise sender_failure
            if receiver_failure is not None:
                raise receiver_failure
            with receive_lock:
                current_count = len(received_pts)
            if current_count != previous_count:
                previous_count = current_count
                stable_since = monotonic()
            elif monotonic() - stable_since >= 0.25:
                break
            sleep(0.01)

        with receive_lock:
            output_pts = tuple(received_pts)
        output_available = bool(output_pts) and all(
            pts != int(gst.CLOCK_TIME_NONE) for pts in output_pts
        )
        monotonic_output = all(
            current >= previous for previous, current in pairwise(output_pts)
        )
        strictly_increasing = all(
            current > previous for previous, current in pairwise(output_pts)
        )
        return TimestampProvenanceVariantResult(
            name=name,
            source_timestamp_present=source_timestamp_present,
            appsrc_do_timestamp=do_timestamp,
            input_pts_ns=source_pts_ns,
            output_pts_ns=output_pts,
            received_all_frames=len(output_pts) == len(source_pts_ns),
            output_pts_available=output_available,
            output_monotonic=monotonic_output,
            output_strictly_increasing=strictly_increasing,
            note=(
                "The candidate receiver observes decoded buffer PTS only; this variant "
                "changes whether that PTS originated in an explicit source timestamp or "
                "was generated by appsrc/GStreamer."
            ),
        )
    finally:
        try:
            source.emit("end-of-stream")
        except (AttributeError, TypeError):
            pass
        if sender_started:
            sender.set_state(gst.State.NULL)
        if receiver_started:
            receiver.set_state(gst.State.NULL)
        sink.disconnect(handler_id)


def run_timestamp_provenance_proof(
    *,
    base_port: int = 18_928,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TimestampProvenanceProofResult:
    """Prove whether decoded PTS presence identifies a real camera source timestamp."""
    if type(base_port) is not int or not 1 <= base_port <= 65_533:
        raise ValueError("base_port must be an integer in range 1..65533")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    gst = _import_gstreamer()
    explicit = _run_provenance_variant(
        gst,
        name="explicit-source-pts",
        source_pts_ns=(
            7_000_000_000,
            7_050_000_000,
            7_100_000_000,
            7_150_000_000,
        ),
        source_timestamp_present=True,
        do_timestamp=False,
        port=base_port,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
        push_interval_seconds=max(0.03, 1.0 / fps),
    )
    generated = _run_provenance_variant(
        gst,
        name="gstreamer-generated-pts-without-source-timestamp",
        source_pts_ns=(None, None, None, None),
        source_timestamp_present=False,
        do_timestamp=True,
        port=base_port + 1,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
        push_interval_seconds=max(0.03, 1.0 / fps),
    )
    no_timestamp = _run_provenance_variant(
        gst,
        name="no-source-pts-no-appsrc-timestamp",
        source_pts_ns=(None, None, None, None),
        source_timestamp_present=False,
        do_timestamp=False,
        port=base_port + 2,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
        push_interval_seconds=max(0.03, 1.0 / fps),
    )
    variants = (explicit, generated, no_timestamp)

    ambiguity_proven = (
        explicit.received_all_frames
        and explicit.output_pts_available
        and generated.received_all_frames
        and generated.output_pts_available
    )
    if ambiguity_proven:
        status = "PTS_PROVENANCE_NOT_DISTINGUISHABLE"
        conclusion = (
            "Decoded appsink PTS is present both when the sender supplies an explicit "
            "source timestamp and when GStreamer generates timestamps without any source "
            "timestamp. PTS presence alone therefore cannot prove SensorTimestamp provenance."
        )
    else:
        status = "INCONCLUSIVE"
        conclusion = (
            "The synthetic variants did not establish both explicit-source and "
            "GStreamer-generated decoded PTS, so timestamp provenance remains unproven."
        )
    return TimestampProvenanceProofResult(
        status=status,
        gstreamer_version=gst.version_string(),
        variants=variants,
        pts_presence_is_provenance=not ambiguity_proven,
        conclusion=conclusion,
    )


def write_timestamp_provenance_json(
    result: TimestampProvenanceProofResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )



def _run_receiver_recovery_proof(
    gst: Any,
    *,
    port: int,
    tolerance_ns: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: float,
) -> TimestampReceiverRecoveryProofResult:
    """Keep sender session 2 alive while replacing only the PC receiver pipeline."""

    def create_receiver() -> tuple[Any, Any, list[int], Lock, Any]:
        receiver = gst.parse_launch(
            build_timestamp_receiver_pipeline_description(
                address="127.0.0.1",
                port=port,
            )
        )
        sink = receiver.get_by_name("sink")
        if sink is None:
            raise TimestampTransportProofError(
                "receiver-recovery diagnostic receiver omitted appsink"
            )
        received_pts: list[int] = []
        receive_lock = Lock()

        def on_sample(appsink):
            try:
                sample = appsink.emit("pull-sample")
                if sample is None:
                    return gst.FlowReturn.ERROR
                buffer = sample.get_buffer()
                if buffer is None or int(buffer.pts) == int(gst.CLOCK_TIME_NONE):
                    return gst.FlowReturn.ERROR
                with receive_lock:
                    received_pts.append(int(buffer.pts))
                return gst.FlowReturn.OK
            except (AttributeError, TypeError, ValueError):
                return gst.FlowReturn.ERROR

        handler_id = sink.connect("new-sample", on_sample)
        return receiver, sink, received_pts, receive_lock, handler_id

    def create_sender() -> tuple[Any, Any, Any]:
        sender = gst.parse_launch(
            _build_restart_sender_pipeline_description(
                host="127.0.0.1",
                port=port,
                fps=fps,
            )
        )
        source = sender.get_by_name("source")
        pay = sender.get_by_name("pay")
        if source is None or pay is None:
            raise TimestampTransportProofError(
                "receiver-recovery diagnostic sender omitted appsrc/payloader"
            )
        if sender.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                "receiver-recovery sender failed to enter PLAYING"
            )
        return sender, source, pay

    def send_frames(
        sender: Any,
        source: Any,
        receiver: Any,
        received_pts: list[int],
        receive_lock: Lock,
        source_pts_ns: tuple[int, ...],
        *,
        frame_seed: int,
    ) -> tuple[int, ...]:
        with receive_lock:
            start_count = len(received_pts)
        for frame_index, pts_ns in enumerate(source_pts_ns):
            payload = _jpeg_payload(
                frame_seed + frame_index,
                width=width,
                height=height,
            )
            buffer = gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = pts_ns
            buffer.dts = gst.CLOCK_TIME_NONE
            buffer.duration = gst.SECOND // fps
            flow = source.emit("push-buffer", buffer)
            if flow != gst.FlowReturn.OK:
                raise TimestampTransportProofError(
                    f"receiver-recovery appsrc push failed: {flow}"
                )
            sleep(min(0.03, 0.5 / fps))

        deadline = monotonic() + timeout_seconds
        previous_count = -1
        stable_since = monotonic()
        while monotonic() < deadline:
            sender_failure = _bus_failure(gst, sender)
            receiver_failure = _bus_failure(gst, receiver)
            if sender_failure is not None:
                raise sender_failure
            if receiver_failure is not None:
                raise receiver_failure
            with receive_lock:
                current_count = len(received_pts)
            if current_count != previous_count:
                previous_count = current_count
                stable_since = monotonic()
            elif monotonic() - stable_since >= 0.25:
                break
            sleep(0.01)
        with receive_lock:
            return tuple(received_pts[start_count:])

    receiver_1, sink_1, received_1, lock_1, handler_1 = create_receiver()
    receiver_1_started = False
    receiver_2 = None
    sink_2 = None
    handler_2 = None
    sender_1 = None
    source_1 = None
    sender_2 = None
    source_2 = None
    try:
        if receiver_1.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                "receiver-recovery first receiver failed to enter PLAYING"
            )
        receiver_1_started = True
        sleep(0.15)

        first_source = (4_000_000_000, 4_050_000_000, 4_100_000_000)
        sender_1, source_1, _ = create_sender()
        sleep(0.15)
        first_output = send_frames(
            sender_1,
            source_1,
            receiver_1,
            received_1,
            lock_1,
            first_source,
            frame_seed=100,
        )
        if len(first_output) != len(first_source):
            raise TimestampTransportProofError(
                "receiver-recovery first sender session did not deliver all frames"
            )
        first_assessment = assess_timestamp_sequence(
            first_source,
            first_output,
            tolerance_ns=tolerance_ns,
        )
        if not first_assessment.passed:
            raise TimestampTransportProofError(
                "receiver-recovery first sender session did not establish a valid timeline"
            )
        try:
            source_1.emit("end-of-stream")
        except (AttributeError, TypeError):
            pass
        sender_1.set_state(gst.State.NULL)
        sender_1 = None
        source_1 = None
        sleep(0.2)

        sender_2, source_2, pay_2 = create_sender()
        sleep(0.15)
        second_before_reset_source = (
            25_000_000_000,
            25_020_000_000,
            25_070_000_000,
        )
        second_before_reset_output = send_frames(
            sender_2,
            source_2,
            receiver_1,
            received_1,
            lock_1,
            second_before_reset_source,
            frame_seed=200,
        )
        identity_before_reset = _sender_identity(pay_2)
        before_reset_received_all = (
            len(second_before_reset_output) == len(second_before_reset_source)
        )
        before_reset_preserved: bool | None = None
        if before_reset_received_all:
            before_reset_preserved = assess_timestamp_sequence(
                second_before_reset_source,
                second_before_reset_output,
                tolerance_ns=tolerance_ns,
            ).passed

        receiver_1.set_state(gst.State.NULL)
        receiver_1_started = False
        sink_1.disconnect(handler_1)
        handler_1 = None
        sleep(0.2)

        receiver_2, sink_2, received_2, lock_2, handler_2 = create_receiver()
        if receiver_2.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise TimestampTransportProofError(
                "receiver-recovery replacement receiver failed to enter PLAYING"
            )
        sleep(0.15)

        recovery_source = (
            25_120_000_000,
            25_170_000_000,
            25_220_000_000,
            25_270_000_000,
        )
        recovery_output = send_frames(
            sender_2,
            source_2,
            receiver_2,
            received_2,
            lock_2,
            recovery_source,
            frame_seed=300,
        )
        identity_after_reset = _sender_identity(pay_2)
        recovery_received_all = len(recovery_output) == len(recovery_source)
        recovery_preserved = False
        if recovery_received_all:
            recovery_preserved = assess_timestamp_sequence(
                recovery_source,
                recovery_output,
                tolerance_ns=tolerance_ns,
            ).passed

        same_ssrc = identity_before_reset.ssrc == identity_after_reset.ssrc
        if recovery_received_all and recovery_preserved and same_ssrc:
            status = "PC_RECEIVER_RESTART_RECOVERS_TIMELINE"
            conclusion = (
                "A fresh PC receiver pipeline recovered source-relative timestamp timing "
                "while sender session 2 remained alive with the same RTP SSRC. The Raspberry "
                "Pi sender therefore does not need a protocol change for this recovery path; "
                "the PC can terminate the old camera-clock session and re-anchor a fresh one."
            )
        else:
            status = "PC_RECEIVER_RESTART_DID_NOT_RECOVER_TIMELINE"
            conclusion = (
                "Replacing only the PC receiver pipeline did not restore a complete coherent "
                "source-relative timestamp timeline from the still-running second sender session."
            )
        return TimestampReceiverRecoveryProofResult(
            status=status,
            gstreamer_version=gst.version_string(),
            tolerance_ns=tolerance_ns,
            first_source_pts_ns=first_source,
            first_output_pts_ns=first_output,
            second_source_before_reset_pts_ns=second_before_reset_source,
            second_output_before_reset_pts_ns=second_before_reset_output,
            second_before_reset_received_all_frames=before_reset_received_all,
            second_before_reset_relative_timing_preserved=before_reset_preserved,
            recovery_source_pts_ns=recovery_source,
            recovery_output_pts_ns=recovery_output,
            recovery_received_all_frames=recovery_received_all,
            recovery_relative_timing_preserved=recovery_preserved,
            second_sender_identity_before_reset=identity_before_reset,
            second_sender_identity_after_reset=identity_after_reset,
            same_second_sender_ssrc=same_ssrc,
            conclusion=conclusion,
        )
    finally:
        if source_1 is not None:
            try:
                source_1.emit("end-of-stream")
            except (AttributeError, TypeError):
                pass
        if sender_1 is not None:
            sender_1.set_state(gst.State.NULL)
        if source_2 is not None:
            try:
                source_2.emit("end-of-stream")
            except (AttributeError, TypeError):
                pass
        if sender_2 is not None:
            sender_2.set_state(gst.State.NULL)
        if receiver_1_started:
            receiver_1.set_state(gst.State.NULL)
        if handler_1 is not None:
            sink_1.disconnect(handler_1)
        if receiver_2 is not None:
            receiver_2.set_state(gst.State.NULL)
        if sink_2 is not None and handler_2 is not None:
            sink_2.disconnect(handler_2)


def run_timestamp_receiver_recovery_proof(
    *,
    port: int = 18_938,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TimestampReceiverRecoveryProofResult:
    """Run TSP-3R: replace only the PC receiver after sender-session discontinuity."""
    if type(port) is not int or not 1 <= port <= 65_535:
        raise ValueError("port must be an integer in range 1..65535")
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be >= 0")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be > 0")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    gst = _import_gstreamer()
    return _run_receiver_recovery_proof(
        gst,
        port=port,
        tolerance_ns=tolerance_ns,
        width=width,
        height=height,
        fps=fps,
        timeout_seconds=timeout_seconds,
    )


def write_timestamp_receiver_recovery_json(
    result: TimestampReceiverRecoveryProofResult,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
