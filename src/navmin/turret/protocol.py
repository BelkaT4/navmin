"""Binary STM32 protocol codec for the NavMin turret link."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import pack, unpack
from typing import cast

START_BYTES = b"\xAA\x55"
MIN_REQUEST_LENGTH = 8
MIN_RESPONSE_LENGTH = 9
MAX_FRAME_LENGTH = 0xFF
UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
SUPPORTED_BAUDRATES = (9600, 19200, 38400, 57600, 115200)


class ProtocolError(ValueError):
    """Base class for invalid protocol values or frames."""


class ProtocolValueError(ProtocolError):
    """A value cannot be represented by the declared wire type."""


class FrameFormatError(ProtocolError):
    """A frame violates the v1 framing contract."""


class CrcMismatchError(ProtocolError):
    """A frame CRC does not match its contents."""


class UnsupportedCommandCodeError(ProtocolError):
    """A frame contains a command code outside the v1 command set."""


class UnsupportedResultCodeError(ProtocolError):
    """A response contains a result code outside the v1 result set."""


class PayloadFormatError(ProtocolError):
    """A command payload has the wrong type or encoded length."""


class CommandCode(IntEnum):
    PING = 0x02
    SET_CONFIG = 0x10
    SET_BAUDRATE = 0x11
    MOVE_RELATIVE = 0x20
    SET_VELOCITY = 0x21
    EMERGENCY_STOP = 0x30
    MOTOR_ON = 0x31
    MOTOR_OFF = 0x32


class ResultCode(IntEnum):
    OK = 0x00
    UNKNOWN_COMMAND = 0x01
    PARSE_ERROR = 0x02
    INVALID_ARGUMENT = 0x03
    INVALID_REQUEST_ID = 0x04
    MOTORS_OFF = 0x05
    NOT_CONFIGURED = 0x06
    INVALID_STATE = 0x07
    INTERNAL_ERROR = 0x08


def _require_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValueError(
            f"{name} must be an integer, not {type(value).__name__}"
        )
    return value


def require_uint16(value: int, *, name: str = "value") -> int:
    value = _require_integer(value, name=name)
    if not 0 <= value <= UINT16_MAX:
        raise ProtocolValueError(f"{name} must fit uint16")
    return value


def require_uint32(value: int, *, name: str = "value") -> int:
    value = _require_integer(value, name=name)
    if not 0 <= value <= UINT32_MAX:
        raise ProtocolValueError(f"{name} must fit uint32")
    return value


def require_int32(value: int, *, name: str = "value") -> int:
    value = _require_integer(value, name=name)
    if not INT32_MIN <= value <= INT32_MAX:
        raise ProtocolValueError(f"{name} must fit int32")
    return value


@dataclass(frozen=True)
class SetConfigPayload:
    max_speed_x_steps_s: int
    max_speed_y_steps_s: int
    acceleration_x_steps_s2: int
    acceleration_y_steps_s2: int
    velocity_watchdog_timeout_ms: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_speed_x_steps_s",
            "max_speed_y_steps_s",
            "acceleration_x_steps_s2",
            "acceleration_y_steps_s2",
            "velocity_watchdog_timeout_ms",
        ):
            require_uint32(getattr(self, field_name), name=field_name)


@dataclass(frozen=True)
class SetBaudratePayload:
    baudrate: int

    def __post_init__(self) -> None:
        require_uint32(self.baudrate, name="baudrate")


@dataclass(frozen=True)
class MoveRelativePayload:
    delta_x_steps: int
    delta_y_steps: int

    def __post_init__(self) -> None:
        require_int32(self.delta_x_steps, name="delta_x_steps")
        require_int32(self.delta_y_steps, name="delta_y_steps")


@dataclass(frozen=True)
class SetVelocityPayload:
    velocity_x_steps_s: int
    velocity_y_steps_s: int

    def __post_init__(self) -> None:
        require_int32(self.velocity_x_steps_s, name="velocity_x_steps_s")
        require_int32(self.velocity_y_steps_s, name="velocity_y_steps_s")


type RequestPayload = (
    SetConfigPayload
    | SetBaudratePayload
    | MoveRelativePayload
    | SetVelocityPayload
    | None
)

_PAYLOAD_TYPE_BY_COMMAND: dict[CommandCode, type[object] | None] = {
    CommandCode.PING: None,
    CommandCode.SET_CONFIG: SetConfigPayload,
    CommandCode.SET_BAUDRATE: SetBaudratePayload,
    CommandCode.MOVE_RELATIVE: MoveRelativePayload,
    CommandCode.SET_VELOCITY: SetVelocityPayload,
    CommandCode.EMERGENCY_STOP: None,
    CommandCode.MOTOR_ON: None,
    CommandCode.MOTOR_OFF: None,
}

_PAYLOAD_LENGTH_BY_COMMAND: dict[CommandCode, int] = {
    CommandCode.PING: 0,
    CommandCode.SET_CONFIG: 20,
    CommandCode.SET_BAUDRATE: 4,
    CommandCode.MOVE_RELATIVE: 8,
    CommandCode.SET_VELOCITY: 8,
    CommandCode.EMERGENCY_STOP: 0,
    CommandCode.MOTOR_ON: 0,
    CommandCode.MOTOR_OFF: 0,
}


@dataclass(frozen=True)
class ProtocolRequest:
    request_id: int
    command: CommandCode
    payload: RequestPayload = None

    def __post_init__(self) -> None:
        require_uint16(self.request_id, name="request_id")
        if not isinstance(self.command, CommandCode):
            raise UnsupportedCommandCodeError("command must be a v1 CommandCode")
        expected_type = _PAYLOAD_TYPE_BY_COMMAND[self.command]
        if expected_type is None:
            if self.payload is not None:
                raise PayloadFormatError(f"{self.command.name} payload must be empty")
        elif not isinstance(self.payload, expected_type):
            raise PayloadFormatError(
                f"{self.command.name} requires {expected_type.__name__}"
            )


@dataclass(frozen=True)
class ProtocolResponse:
    request_id: int
    command: CommandCode
    result: ResultCode

    def __post_init__(self) -> None:
        require_uint16(self.request_id, name="request_id")
        if not isinstance(self.command, CommandCode):
            raise UnsupportedCommandCodeError("command must be a v1 CommandCode")
        if not isinstance(self.result, ResultCode):
            raise UnsupportedResultCodeError("result must be a v1 ResultCode")


def crc16_modbus(data: bytes | bytearray | memoryview) -> int:
    """Return CRC-16/MODBUS for ``data``."""
    crc = 0xFFFF
    for byte in bytes(data):
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def _encode_request_payload(request: ProtocolRequest) -> bytes:
    payload = request.payload
    match request.command:
        case CommandCode.SET_CONFIG:
            payload = cast(SetConfigPayload, payload)
            return pack(
                "<IIIII",
                payload.max_speed_x_steps_s,
                payload.max_speed_y_steps_s,
                payload.acceleration_x_steps_s2,
                payload.acceleration_y_steps_s2,
                payload.velocity_watchdog_timeout_ms,
            )
        case CommandCode.SET_BAUDRATE:
            payload = cast(SetBaudratePayload, payload)
            return pack("<I", payload.baudrate)
        case CommandCode.MOVE_RELATIVE:
            payload = cast(MoveRelativePayload, payload)
            return pack("<ii", payload.delta_x_steps, payload.delta_y_steps)
        case CommandCode.SET_VELOCITY:
            payload = cast(SetVelocityPayload, payload)
            return pack(
                "<ii", payload.velocity_x_steps_s, payload.velocity_y_steps_s
            )
        case _:
            return b""


def encode_request(request: ProtocolRequest) -> bytes:
    payload = _encode_request_payload(request)
    length = MIN_REQUEST_LENGTH + len(payload)
    if length > MAX_FRAME_LENGTH:
        raise FrameFormatError("request frame exceeds uint8 LENGTH")
    crc_input = bytes((length,)) + pack("<H", request.request_id) + bytes(
        (request.command,)
    ) + payload
    crc = crc16_modbus(crc_input)
    return START_BYTES + crc_input + pack("<H", crc)


def encode_response(response: ProtocolResponse) -> bytes:
    length = MIN_RESPONSE_LENGTH
    crc_input = (
        bytes((length,))
        + pack("<H", response.request_id)
        + bytes((response.command, response.result))
    )
    crc = crc16_modbus(crc_input)
    return START_BYTES + crc_input + pack("<H", crc)


def _as_frame_bytes(frame: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise FrameFormatError("frame must be bytes-like")
    return bytes(frame)


def _validate_frame(frame: bytes, *, minimum_length: int) -> None:
    if len(frame) < minimum_length:
        raise FrameFormatError(
            f"frame is shorter than minimum {minimum_length} bytes"
        )
    if len(frame) > MAX_FRAME_LENGTH:
        raise FrameFormatError("frame exceeds uint8 LENGTH")
    if frame[:2] != START_BYTES:
        raise FrameFormatError("invalid START bytes")
    declared_length = frame[2]
    if declared_length != len(frame):
        raise FrameFormatError(
            f"declared LENGTH {declared_length} does not match {len(frame)} bytes"
        )
    expected_crc = int.from_bytes(frame[-2:], "little")
    actual_crc = crc16_modbus(frame[2:-2])
    if expected_crc != actual_crc:
        raise CrcMismatchError(
            f"CRC mismatch: expected 0x{expected_crc:04X}, computed 0x{actual_crc:04X}"
        )


def _decode_command(raw_code: int) -> CommandCode:
    try:
        return CommandCode(raw_code)
    except ValueError as exc:
        raise UnsupportedCommandCodeError(
            f"unsupported command code 0x{raw_code:02X}"
        ) from exc


def _decode_result(raw_code: int) -> ResultCode:
    try:
        return ResultCode(raw_code)
    except ValueError as exc:
        raise UnsupportedResultCodeError(
            f"unsupported result code 0x{raw_code:02X}"
        ) from exc


def _decode_request_payload(command: CommandCode, payload: bytes) -> RequestPayload:
    expected_length = _PAYLOAD_LENGTH_BY_COMMAND[command]
    if len(payload) != expected_length:
        raise PayloadFormatError(
            f"{command.name} payload must be {expected_length} bytes, "
            f"got {len(payload)}"
        )

    match command:
        case CommandCode.SET_CONFIG:
            return SetConfigPayload(*unpack("<IIIII", payload))
        case CommandCode.SET_BAUDRATE:
            (baudrate,) = unpack("<I", payload)
            return SetBaudratePayload(baudrate)
        case CommandCode.MOVE_RELATIVE:
            return MoveRelativePayload(*unpack("<ii", payload))
        case CommandCode.SET_VELOCITY:
            return SetVelocityPayload(*unpack("<ii", payload))
        case _:
            return None


def decode_request(frame: bytes | bytearray | memoryview) -> ProtocolRequest:
    raw = _as_frame_bytes(frame)
    _validate_frame(raw, minimum_length=MIN_REQUEST_LENGTH)
    request_id = int.from_bytes(raw[3:5], "little")
    command = _decode_command(raw[5])
    payload = _decode_request_payload(command, raw[6:-2])
    return ProtocolRequest(request_id=request_id, command=command, payload=payload)


def decode_response(frame: bytes | bytearray | memoryview) -> ProtocolResponse:
    raw = _as_frame_bytes(frame)
    _validate_frame(raw, minimum_length=MIN_RESPONSE_LENGTH)
    if raw[7:-2]:
        raise FrameFormatError(
            "response EVENTS section must be empty in protocol v1"
        )
    request_id = int.from_bytes(raw[3:5], "little")
    command = _decode_command(raw[5])
    result = _decode_result(raw[6])
    return ProtocolResponse(request_id=request_id, command=command, result=result)


__all__ = [
    "INT32_MAX",
    "INT32_MIN",
    "MAX_FRAME_LENGTH",
    "MIN_REQUEST_LENGTH",
    "MIN_RESPONSE_LENGTH",
    "START_BYTES",
    "SUPPORTED_BAUDRATES",
    "UINT16_MAX",
    "UINT32_MAX",
    "CommandCode",
    "CrcMismatchError",
    "FrameFormatError",
    "MoveRelativePayload",
    "PayloadFormatError",
    "ProtocolError",
    "ProtocolRequest",
    "ProtocolResponse",
    "ProtocolValueError",
    "ResultCode",
    "SetBaudratePayload",
    "SetConfigPayload",
    "SetVelocityPayload",
    "UnsupportedCommandCodeError",
    "UnsupportedResultCodeError",
    "crc16_modbus",
    "decode_request",
    "decode_response",
    "encode_request",
    "encode_response",
    "require_int32",
    "require_uint16",
    "require_uint32",
]
