from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_sender_module() -> ModuleType:
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "send_gy85_udp.py"
    spec = importlib.util.spec_from_file_location("navmin_test_send_gy85_udp", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load GY-85 sender tool from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


send_gy85_udp = _load_sender_module()
Gy85 = send_gy85_udp.Gy85
build_payload = send_gy85_udp.build_payload
encode_payload = send_gy85_udp.encode_payload
main = send_gy85_udp.main
parse_args = send_gy85_udp.parse_args
remap = send_gy85_udp.remap
signed_axes = send_gy85_udp.signed_axes


class Bus:
    def __init__(self) -> None:
        self.status = 1

    def read_byte_data(self, device: int, register: int) -> int:
        return self.status

    def read_i2c_block_data(self, device: int, register: int, count: int) -> list[int]:
        if device == 0x68:
            return list(struct.pack(">hhh", 14375, -14375, 0))
        if device == 0x53:
            return list(struct.pack("<hhh", 256, -256, 0))
        return list(struct.pack("<hhh", 1200, -1300, 500))


def test_sensor_byte_order_scale_and_magnetic_freshness() -> None:
    bus = Bus()
    sensor = Gy85(bus)
    accel, gyro, mag = sensor.read()
    assert gyro == (1000, -1000, 0)
    assert accel == pytest.approx((0.9984, -0.9984, 0))
    assert mag == (1200, -1300, 500)
    bus.status = 3
    assert sensor.read()[2] is None
    bus.status = 0
    assert sensor.read() is None
    assert remap((1, 2, 3), signed_axes("y,-x,z")) == [2, -1, 3]


def test_sensor_rejects_gyro_saturation() -> None:
    class SaturatedBus(Bus):
        def read_i2c_block_data(self, device: int, register: int, count: int) -> list[int]:
            if device == 0x68:
                return list(struct.pack(">hhh", 32760, 0, 0))
            return super().read_i2c_block_data(device, register, count)

    with pytest.raises(RuntimeError, match="saturated"):
        Gy85(SaturatedBus()).read()


def test_sensor_initialization_sets_expected_registers(monkeypatch) -> None:
    class InitializationBus:
        def __init__(self) -> None:
            self.writes: list[tuple[int, int, int]] = []

        def read_byte_data(self, device: int, register: int) -> int:
            return 0xE5 if device == 0x53 else 0x68

        def write_byte_data(self, device: int, register: int, value: int) -> None:
            self.writes.append((device, register, value))

    bus = InitializationBus()
    monkeypatch.setattr(send_gy85_udp.time, "sleep", lambda _seconds: None)
    Gy85(bus).initialize()
    assert (0x53, 0x31, 0x08) in bus.writes
    assert (0x53, 0x2C, 0x0A) in bus.writes
    assert (0x68, 0x16, 0x1C) in bus.writes
    assert (0x68, 0x15, 19) in bus.writes
    assert (0x68, 0x17, 1) in bus.writes
    assert (0x0D, 0x09, 0x1D) in bus.writes

    bus.read_byte_data = lambda device, register: 0
    with pytest.raises(RuntimeError, match="ADXL345"):
        Gy85(bus).initialize()


def test_protocol_payload_and_json_match_v1_contract() -> None:
    payload = build_payload(
        ((0.1, -0.2, 1.0), (1.0, 2.0, 3.0), (100, 200, 300)),
        timestamp=123.456,
        gyro_axes=signed_axes("y,-x,z"),
        mag_axes=signed_axes("-y,x,z"),
    )
    assert payload == {
        "type": "navmin-gy85",
        "version": 1,
        "t": 123.456,
        "accel_g": (0.1, -0.2, 1.0),
        "gyro_dps": [2.0, -1.0, 3.0],
        "mag_raw": [-200, 100, 300],
    }
    assert json.loads(encode_payload(payload)) == {
        "type": "navmin-gy85",
        "version": 1,
        "t": 123.456,
        "accel_g": [0.1, -0.2, 1.0],
        "gyro_dps": [2.0, -1.0, 3.0],
        "mag_raw": [-200, 100, 300],
    }


def test_protocol_payload_allows_missing_fresh_magnetometer() -> None:
    payload = build_payload(
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), None),
        timestamp=5.0,
    )
    assert payload["mag_raw"] is None
    assert json.loads(encode_payload(payload))["mag_raw"] is None


def test_sender_cli_requires_host_and_has_generic_port_bus_defaults() -> None:
    args = parse_args(["--host", "127.0.0.1"])
    assert args.host == "127.0.0.1"
    assert args.port == 8891
    assert args.bus == 1
    assert args.gyro_axes == (1, 2, 3)
    assert args.mag_axes == (1, 2, 3)

    args = parse_args(
        [
            "--host",
            "192.0.2.10",
            "--port",
            "12345",
            "--bus",
            "2",
            "--gyro-axes",
            "y,-x,z",
        ]
    )
    assert args.port == 12345
    assert args.bus == 2
    assert args.gyro_axes == (2, -1, 3)


@pytest.mark.parametrize("axes", ["x,x,z", "x,y,-z", "bad,y,z"])
def test_sender_cli_rejects_invalid_axis_mapping(axes: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--host", "127.0.0.1", "--gyro-axes", axes])


def test_sender_loop_uses_monotonic_timestamp_and_survives_temporary_i2c_error(
    monkeypatch,
) -> None:
    class RuntimeBus:
        def __init__(self, _bus_id: int) -> None:
            self.read_calls = 0
            self.closed = False

        def read_byte_data(self, device: int, register: int) -> int:
            if register == 0x00:
                return 0xE5 if device == 0x53 else 0x68
            if device == 0x68 and register == 0x1A:
                self.read_calls += 1
                if self.read_calls == 1:
                    raise OSError("temporary I2C failure")
                if self.read_calls == 2:
                    return 1
                raise KeyboardInterrupt
            if device == 0x0D and register == 0x06:
                return 1
            return 0

        def write_byte_data(self, device: int, register: int, value: int) -> None:
            return None

        def read_i2c_block_data(self, device: int, register: int, count: int) -> list[int]:
            if device == 0x68:
                return list(struct.pack(">hhh", 0, 0, 0))
            if device == 0x53:
                return list(struct.pack("<hhh", 0, 0, 256))
            return list(struct.pack("<hhh", 100, 0, 0))

        def close(self) -> None:
            self.closed = True

    class RuntimeSocket:
        def __init__(self, *_args, **_kwargs) -> None:
            self.sent: list[tuple[bytes, tuple[str, int]]] = []
            self.closed = False

        def settimeout(self, _timeout: float) -> None:
            return None

        def sendto(self, data: bytes, address: tuple[str, int]) -> None:
            self.sent.append((data, address))

        def close(self) -> None:
            self.closed = True

    runtime_socket = RuntimeSocket()
    monotonic_values = iter([10.0, 10.0, 10.05, 10.10, 10.15, 10.20])
    sleeps: list[float] = []

    monkeypatch.setitem(sys.modules, "smbus", SimpleNamespace(SMBus=RuntimeBus))
    monkeypatch.setattr(send_gy85_udp.socket, "socket", lambda *_args, **_kwargs: runtime_socket)
    monkeypatch.setattr(send_gy85_udp.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(send_gy85_udp.time, "sleep", sleeps.append)

    assert main(["--host", "127.0.0.1", "--port", "8891"]) == 0
    assert 0.05 in sleeps
    assert len(runtime_socket.sent) == 1
    encoded, address = runtime_socket.sent[0]
    assert address == ("127.0.0.1", 8891)
    payload = json.loads(encoded)
    assert payload["type"] == "navmin-gy85"
    assert payload["version"] == 1
    assert payload["t"] == 10.05
    assert runtime_socket.closed
