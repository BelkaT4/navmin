from __future__ import annotations

from pathlib import Path

import pytest

from navmin.diagnostics.pty_stm32 import (
    PtyDiagnosticError,
    PtyFault,
    PtyFaultKind,
    PtyFaultQueue,
    PtyPreflightResult,
    RequestFrameAssembler,
    StableDiagnosticPort,
)
from navmin.turret.protocol import (
    MAX_FRAME_LENGTH,
    START_BYTES,
    CommandCode,
    ProtocolRequest,
    encode_request,
)


def test_request_assembler_accepts_fragmented_and_coalesced_frames() -> None:
    first = encode_request(ProtocolRequest(7, CommandCode.PING))
    second = encode_request(ProtocolRequest(8, CommandCode.MOTOR_OFF))
    assembler = RequestFrameAssembler()

    assert assembler.feed(b"noise" + first[:1]) == ()
    assert assembler.feed(first[1:5]) == ()
    assert assembler.feed(first[5:] + second) == (first, second)
    assert assembler.buffered_bytes == 0


def test_request_assembler_resyncs_false_length_and_overlapping_prefix() -> None:
    frame = encode_request(ProtocolRequest(9, CommandCode.PING))
    assembler = RequestFrameAssembler()

    raw = START_BYTES + b"\x07" + START_BYTES[:1] + frame

    assert assembler.feed(raw) == (frame,)
    assert assembler.buffered_bytes == 0


def test_request_assembler_keeps_buffer_bounded_for_garbage() -> None:
    assembler = RequestFrameAssembler()

    assert assembler.feed(b"x" * (MAX_FRAME_LENGTH * 20) + START_BYTES[:1]) == ()
    assert assembler.buffered_bytes == 1


def test_fault_queue_is_fifo_and_each_fault_is_consumed_once() -> None:
    queue = PtyFaultQueue()
    fragmented = PtyFault(PtyFaultKind.FRAGMENTED_RESPONSE)
    dropped = PtyFault(PtyFaultKind.DROP_RESPONSE)

    queue.put(fragmented)
    queue.put(dropped)

    assert len(queue) == 2
    assert queue.pop() is fragmented
    assert queue.pop() is dropped
    assert queue.pop() is None
    assert len(queue) == 0


def test_stable_port_repoints_atomically_and_cleanup_is_idempotent() -> None:
    stable = StableDiagnosticPort()
    directory = Path(stable.directory)
    path = Path(stable.path)
    try:
        stable.repoint("/dev/pts/100")
        assert path.is_symlink()
        assert stable.read_target() == "/dev/pts/100"

        stable.repoint("/dev/pts/101")
        assert path.is_symlink()
        assert stable.read_target() == "/dev/pts/101"
        assert list(directory.iterdir()) == [path]
    finally:
        stable.cleanup()
        stable.cleanup()

    assert not directory.exists()
    with pytest.raises(PtyDiagnosticError, match="cleaned"):
        stable.repoint("/dev/pts/102")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (PtyPreflightResult(True, True, True), True),
        (PtyPreflightResult(False, True, True), False),
        (PtyPreflightResult(True, False, True), False),
        (PtyPreflightResult(True, True, False), False),
    ],
)
def test_preflight_requires_linux_openpty_and_pyserial(result, expected) -> None:
    assert result.ok is expected
