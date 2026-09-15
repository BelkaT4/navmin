from __future__ import annotations

from struct import pack

import pytest

from navmin.turret.protocol import (
    INT32_MAX,
    INT32_MIN,
    MIN_RESPONSE_LENGTH,
    START_BYTES,
    UINT32_MAX,
    CommandCode,
    CrcMismatchError,
    FrameFormatError,
    MoveRelativePayload,
    PayloadFormatError,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolValueError,
    ResultCode,
    SetBaudratePayload,
    SetConfigPayload,
    SetVelocityPayload,
    UnsupportedCommandCodeError,
    UnsupportedResultCodeError,
    crc16_modbus,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from navmin.turret.simulator import (
    FakeReadFailure,
    FakeResponseSpec,
    FakeStm32Endpoint,
    FakeTransport,
)
from navmin.turret.transport import (
    PhysicalTransport,
    SerialTransport,
    TransportDisconnectedError,
    TransportIOError,
    TransportTimeoutError,
)


def _raw_frame(body_without_crc: bytes) -> bytes:
    return b"\xAA\x55" + body_without_crc + pack("<H", crc16_modbus(body_without_crc))


def test_crc16_modbus_known_vector() -> None:
    assert crc16_modbus(b"123456789") == 0x4B37


@pytest.mark.parametrize(
    ("protocol_request", "expected_hex"),
    [
        (ProtocolRequest(0x1234, CommandCode.PING), "aa5508341202ceeb"),
        (
            ProtocolRequest(
                1,
                CommandCode.SET_CONFIG,
                SetConfigPayload(1, 2, 3, 4, 5),
            ),
            "aa551c010010010000000200000003000000040000000500000044db",
        ),
        (
            ProtocolRequest(
                0,
                CommandCode.MOVE_RELATIVE,
                MoveRelativePayload(-1, INT32_MAX),
            ),
            "aa5510000020ffffffffffffff7f7c18",
        ),
        (ProtocolRequest(3, CommandCode.MOTOR_OFF), "aa55080300327391"),
    ],
)
def test_request_wire_layout_matches_protocol(protocol_request, expected_hex) -> None:
    raw = encode_request(protocol_request)

    assert raw.hex() == expected_hex
    assert decode_request(raw) == protocol_request


def test_response_wire_layout_matches_protocol() -> None:
    response = ProtocolResponse(0x1234, CommandCode.PING, ResultCode.OK)

    raw = encode_response(response)

    assert raw.hex() == "aa5509341202005754"
    assert decode_response(raw) == response


@pytest.mark.parametrize(
    "protocol_request",
    [
        ProtocolRequest(0, CommandCode.PING),
        ProtocolRequest(
            1,
            CommandCode.SET_CONFIG,
            SetConfigPayload(100, 200, 300, 400, 500),
        ),
        ProtocolRequest(
            2,
            CommandCode.SET_BAUDRATE,
            SetBaudratePayload(115200),
        ),
        ProtocolRequest(
            3,
            CommandCode.MOVE_RELATIVE,
            MoveRelativePayload(-1000, 2000),
        ),
        ProtocolRequest(
            4,
            CommandCode.SET_VELOCITY,
            SetVelocityPayload(-3000, 4000),
        ),
        ProtocolRequest(5, CommandCode.EMERGENCY_STOP),
        ProtocolRequest(6, CommandCode.MOTOR_ON),
        ProtocolRequest(7, CommandCode.MOTOR_OFF),
    ],
)
def test_request_encode_decode_roundtrip_for_all_v1_command_layouts(
    protocol_request,
) -> None:
    assert decode_request(encode_request(protocol_request)) == protocol_request


@pytest.mark.parametrize("request_id", [0, 0xFFFF])
def test_request_id_uint16_boundaries_are_accepted(request_id) -> None:
    request = ProtocolRequest(request_id, CommandCode.PING)

    assert decode_request(encode_request(request)).request_id == request_id


@pytest.mark.parametrize("request_id", [-1, 0x10000, True])
def test_request_id_outside_strict_uint16_is_rejected(request_id) -> None:
    with pytest.raises(ProtocolValueError):
        ProtocolRequest(request_id, CommandCode.PING)


def test_int32_min_max_are_accepted() -> None:
    payload = MoveRelativePayload(INT32_MIN, INT32_MAX)
    request = ProtocolRequest(0, CommandCode.MOVE_RELATIVE, payload)

    assert decode_request(encode_request(request)).payload == payload


@pytest.mark.parametrize("value", [INT32_MIN - 1, INT32_MAX + 1, True])
def test_int32_overflow_underflow_and_bool_are_rejected(value) -> None:
    with pytest.raises(ProtocolValueError):
        SetVelocityPayload(value, 0)


def test_uint32_max_is_accepted() -> None:
    payload = SetConfigPayload(UINT32_MAX, 1, 2, 3, 4)
    request = ProtocolRequest(0, CommandCode.SET_CONFIG, payload)

    assert decode_request(encode_request(request)).payload == payload


@pytest.mark.parametrize("value", [-1, UINT32_MAX + 1, True])
def test_uint32_negative_overflow_and_bool_are_rejected(value) -> None:
    with pytest.raises(ProtocolValueError):
        SetConfigPayload(value, 1, 2, 3, 4)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xAA\x55\x08",
        bytes.fromhex("005508341202ceeb"),
        bytes.fromhex("aa5509341202ceeb"),
    ],
)
def test_malformed_request_frame_is_rejected(raw) -> None:
    with pytest.raises(FrameFormatError):
        decode_request(raw)


def test_bad_crc_is_rejected() -> None:
    raw = bytearray(
        encode_response(ProtocolResponse(1, CommandCode.PING, ResultCode.OK))
    )
    raw[-1] ^= 0xFF

    with pytest.raises(CrcMismatchError):
        decode_response(raw)


def test_unsupported_command_code_is_rejected() -> None:
    raw = _raw_frame(bytes((8, 0, 0, 0x99)))

    with pytest.raises(UnsupportedCommandCodeError):
        decode_request(raw)


def test_unsupported_result_code_is_rejected() -> None:
    raw = _raw_frame(bytes((9, 0, 0, CommandCode.PING, 0x99)))

    with pytest.raises(UnsupportedResultCodeError):
        decode_response(raw)


def test_malformed_payload_length_is_rejected_after_valid_frame_crc() -> None:
    raw = _raw_frame(bytes((11, 0, 0, CommandCode.SET_BAUDRATE)) + b"\x00\x01\x02")

    with pytest.raises(PayloadFormatError):
        decode_request(raw)


def test_nonempty_v1_response_events_are_rejected() -> None:
    raw = _raw_frame(bytes((10, 0, 0, CommandCode.PING, ResultCode.OK, 0x01)))

    with pytest.raises(FrameFormatError, match="EVENTS"):
        decode_response(raw)


def test_command_payload_type_mismatch_is_rejected() -> None:
    with pytest.raises(PayloadFormatError):
        ProtocolRequest(0, CommandCode.PING, SetBaudratePayload(9600))


def test_fake_transport_success_uses_real_encoded_frames() -> None:
    endpoint = FakeStm32Endpoint()
    transport = FakeTransport(endpoint)
    request = ProtocolRequest(0, CommandCode.PING)
    raw_request = encode_request(request)

    transport.open()
    transport.write_frame(raw_request, timeout_s=0.1)
    response = decode_response(transport.read_frame(timeout_s=0.1))

    assert response == ProtocolResponse(0, CommandCode.PING, ResultCode.OK)
    assert endpoint.raw_request_history == [raw_request]
    assert endpoint.request_history == [request]


def test_fake_transport_timeout_is_a_physical_transport_error() -> None:
    transport = FakeTransport()
    transport.open()
    transport.queue_read_failure(FakeReadFailure.TIMEOUT)
    transport.write_frame(encode_request(ProtocolRequest(1, CommandCode.PING)), 0.1)

    with pytest.raises(TransportTimeoutError):
        transport.read_frame(0.1)


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (FakeReadFailure.IO_ERROR, TransportIOError),
        (FakeReadFailure.DISCONNECT, TransportDisconnectedError),
    ],
)
def test_fake_transport_can_model_physical_io_and_disconnect_failures(
    failure, expected_error
) -> None:
    transport = FakeTransport()
    transport.open()
    transport.queue_read_failure(failure)
    transport.write_frame(encode_request(ProtocolRequest(1, CommandCode.PING)), 0.1)

    with pytest.raises(expected_error):
        transport.read_frame(0.1)


def test_fake_endpoint_can_generate_bad_crc_response() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(corrupt_crc=True))
    transport = FakeTransport(endpoint)
    transport.open()
    transport.write_frame(encode_request(ProtocolRequest(0, CommandCode.PING)), 0.1)

    with pytest.raises(CrcMismatchError):
        decode_response(transport.read_frame(0.1))


@pytest.mark.parametrize(
    ("spec", "expected_id", "expected_command"),
    [
        (FakeResponseSpec(request_id=99), 99, CommandCode.PING),
        (
            FakeResponseSpec(command=CommandCode.MOTOR_OFF),
            0,
            CommandCode.MOTOR_OFF,
        ),
    ],
)
def test_fake_endpoint_can_generate_mismatched_response_identity(
    spec, expected_id, expected_command
) -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(spec)
    transport = FakeTransport(endpoint)
    transport.open()
    transport.write_frame(encode_request(ProtocolRequest(0, CommandCode.PING)), 0.1)

    response = decode_response(transport.read_frame(0.1))

    assert response.request_id == expected_id
    assert response.command is expected_command


def test_fake_endpoint_can_return_malformed_raw_response() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(raw_response=b"not-a-frame"))
    transport = FakeTransport(endpoint)
    transport.open()
    transport.write_frame(encode_request(ProtocolRequest(0, CommandCode.PING)), 0.1)

    with pytest.raises(FrameFormatError):
        decode_response(transport.read_frame(0.1))


def test_fake_endpoint_exact_retry_returns_cached_bytes_without_reexecution() -> None:
    endpoint = FakeStm32Endpoint()
    transport = FakeTransport(endpoint)
    raw = encode_request(
        ProtocolRequest(
            0,
            CommandCode.SET_VELOCITY,
            SetVelocityPayload(100, -100),
        )
    )
    transport.open()

    responses = []
    for _ in range(2):
        transport.write_frame(raw, 0.1)
        responses.append(transport.read_frame(0.1))

    assert responses[0] == responses[1]
    assert transport.raw_write_history == [raw, raw]
    assert endpoint.raw_request_history == [raw, raw]
    assert endpoint.request_history[0] == endpoint.request_history[1]
    assert endpoint.executed_request_history == [endpoint.request_history[0]]
    assert endpoint.expected_request_id == 1


def test_fake_endpoint_models_ordinary_request_sequence_and_signature_collision() -> None:
    endpoint = FakeStm32Endpoint()
    transport = FakeTransport(endpoint)
    transport.open()

    ping_0 = encode_request(ProtocolRequest(0, CommandCode.PING))
    transport.write_frame(ping_0, 0.1)
    assert decode_response(transport.read_frame(0.1)).result is ResultCode.OK
    assert endpoint.expected_request_id == 1

    same_id_other_command = encode_request(ProtocolRequest(0, CommandCode.MOTOR_OFF))
    transport.write_frame(same_id_other_command, 0.1)
    assert (
        decode_response(transport.read_frame(0.1)).result
        is ResultCode.INVALID_REQUEST_ID
    )
    assert endpoint.expected_request_id == 1

    endpoint_payload = FakeStm32Endpoint()
    payload_transport = FakeTransport(endpoint_payload)
    payload_transport.open()
    velocity_a = encode_request(
        ProtocolRequest(
            0,
            CommandCode.SET_VELOCITY,
            SetVelocityPayload(10, -10),
        )
    )
    velocity_b = encode_request(
        ProtocolRequest(
            0,
            CommandCode.SET_VELOCITY,
            SetVelocityPayload(11, -10),
        )
    )
    payload_transport.write_frame(velocity_a, 0.1)
    assert decode_response(payload_transport.read_frame(0.1)).result is ResultCode.OK
    payload_transport.write_frame(velocity_b, 0.1)
    assert (
        decode_response(payload_transport.read_frame(0.1)).result
        is ResultCode.INVALID_REQUEST_ID
    )
    assert len(endpoint_payload.executed_request_history) == 1
    assert endpoint_payload.expected_request_id == 1

    wrong_next_id = encode_request(ProtocolRequest(2, CommandCode.PING))
    transport.write_frame(wrong_next_id, 0.1)
    assert (
        decode_response(transport.read_frame(0.1)).result
        is ResultCode.INVALID_REQUEST_ID
    )
    assert endpoint.expected_request_id == 1

    motor_off_1 = encode_request(ProtocolRequest(1, CommandCode.MOTOR_OFF))
    transport.write_frame(motor_off_1, 0.1)
    assert decode_response(transport.read_frame(0.1)).result is ResultCode.OK
    assert endpoint.expected_request_id == 2


def test_fake_endpoint_emergency_resync_overrides_ordinary_sequence_and_cache_collision() -> None:
    endpoint = FakeStm32Endpoint()
    transport = FakeTransport(endpoint)
    transport.open()

    ping_0 = encode_request(ProtocolRequest(0, CommandCode.PING))
    transport.write_frame(ping_0, 0.1)
    assert decode_response(transport.read_frame(0.1)).result is ResultCode.OK

    emergency_0 = encode_request(ProtocolRequest(0, CommandCode.EMERGENCY_STOP))
    transport.write_frame(emergency_0, 0.1)
    emergency_response = decode_response(transport.read_frame(0.1))
    assert emergency_response.result is ResultCode.OK
    assert endpoint.expected_request_id == 1
    assert [request.command for request in endpoint.executed_request_history] == [
        CommandCode.PING,
        CommandCode.EMERGENCY_STOP,
    ]

    ping_1 = encode_request(ProtocolRequest(1, CommandCode.PING))
    transport.write_frame(ping_1, 0.1)
    assert decode_response(transport.read_frame(0.1)).result is ResultCode.OK
    assert endpoint.expected_request_id == 2


def test_command_level_error_remains_valid_protocol_response() -> None:
    endpoint = FakeStm32Endpoint()
    endpoint.queue_response(FakeResponseSpec(result=ResultCode.INVALID_ARGUMENT))
    transport = FakeTransport(endpoint)
    transport.open()
    transport.write_frame(encode_request(ProtocolRequest(0, CommandCode.PING)), 0.1)

    response = decode_response(transport.read_frame(0.1))

    assert response.result is ResultCode.INVALID_ARGUMENT


class _FakeSerialBackend:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.is_open = True
        self.baudrate = kwargs["baudrate"]
        self.timeout = kwargs["timeout"]
        self.write_timeout = kwargs["write_timeout"]
        self.writes: list[bytes] = []
        self.incoming = bytearray()

    def write(self, data: bytes) -> int:
        raw = bytes(data)
        self.writes.append(raw)
        return len(raw)

    def read(self, size: int) -> bytes:
        if not self.incoming:
            return b""
        chunk = bytes(self.incoming[:size])
        del self.incoming[:size]
        return chunk

    def close(self) -> None:
        self.is_open = False


def test_serial_transport_uses_common_boundary_without_real_serial_device() -> None:
    created: list[_FakeSerialBackend] = []

    def factory(**kwargs):
        backend = _FakeSerialBackend(**kwargs)
        created.append(backend)
        return backend

    transport = SerialTransport("/dev/fake", 9600, serial_factory=factory)
    assert isinstance(transport, PhysicalTransport)

    transport.open()
    backend = created[0]
    request = encode_request(ProtocolRequest(5, CommandCode.PING))
    backend.incoming.extend(
        encode_response(ProtocolResponse(5, CommandCode.PING, ResultCode.OK))
    )

    transport.write_frame(request, 0.1)
    response = decode_response(transport.read_frame(0.1))
    transport.set_baudrate(115200)

    assert backend.writes == [request]
    assert response == ProtocolResponse(5, CommandCode.PING, ResultCode.OK)
    assert transport.baudrate == 115200
    assert backend.baudrate == 115200

    transport.close()
    assert not transport.is_open


def _serial_transport_with_backend() -> tuple[SerialTransport, _FakeSerialBackend]:
    backend = _FakeSerialBackend(
        port="/dev/fake",
        baudrate=9600,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=0,
        write_timeout=0,
    )
    transport = SerialTransport(
        "/dev/fake",
        9600,
        serial_factory=lambda **_: backend,
    )
    transport.open()
    return transport, backend


def test_serial_transport_skips_leading_garbage_before_valid_response() -> None:
    transport, backend = _serial_transport_with_backend()
    response = encode_response(ProtocolResponse(8, CommandCode.PING, ResultCode.OK))
    backend.incoming.extend(b"garbage" + response)

    assert transport.read_frame(0.1) == response


def test_serial_transport_resyncs_overlapping_start_prefix() -> None:
    transport, backend = _serial_transport_with_backend()
    response = encode_response(ProtocolResponse(9, CommandCode.PING, ResultCode.OK))
    backend.incoming.extend(bytes((START_BYTES[0],)) + response)

    assert transport.read_frame(0.1) == response


def test_serial_transport_skips_false_start_with_too_short_length() -> None:
    transport, backend = _serial_transport_with_backend()
    response = encode_response(ProtocolResponse(10, CommandCode.PING, ResultCode.OK))
    backend.incoming.extend(START_BYTES + bytes((MIN_RESPONSE_LENGTH - 1,)) + response)

    assert transport.read_frame(0.1) == response


def test_serial_transport_times_out_when_no_valid_start_is_found() -> None:
    transport, backend = _serial_transport_with_backend()
    backend.incoming.extend(b"garbage without a frame")

    with pytest.raises(TransportTimeoutError):
        transport.read_frame(0.01)


def test_serial_transport_read_timeout_is_bounded_error() -> None:
    backend = _FakeSerialBackend(
        port="/dev/fake",
        baudrate=9600,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=0,
        write_timeout=0,
    )
    transport = SerialTransport(
        "/dev/fake",
        9600,
        serial_factory=lambda **_: backend,
    )
    transport.open()

    with pytest.raises(TransportTimeoutError):
        transport.read_frame(0.01)
