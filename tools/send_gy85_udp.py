"""Standalone Raspberry Pi GY-85 raw telemetry sender; requires python3-smbus."""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import time
from collections.abc import Sequence

PROTOCOL_TYPE = "navmin-gy85"
PROTOCOL_VERSION = 1


def signed_axes(text: str) -> tuple[int, int, int]:
    names = {"x": 1, "y": 2, "z": 3, "-x": -1, "-y": -2, "-z": -3}
    try:
        axes = tuple(names[part.strip().lower()] for part in text.split(","))
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "use x,y,z or signed/permuted axes, e.g. y,-x,z"
        ) from exc
    if len(axes) != 3 or sorted(abs(axis) for axis in axes) != [1, 2, 3]:
        raise argparse.ArgumentTypeError("each axis must appear once")
    inversions = sum(
        abs(axes[i]) > abs(axes[j]) for i in range(3) for j in range(i + 1, 3)
    )
    handedness = (-1) ** inversions * math.prod(1 if axis > 0 else -1 for axis in axes)
    if handedness != 1:
        raise argparse.ArgumentTypeError("axes must be right-handed")
    return axes


def remap(
    values: tuple[float, float, float], axes: tuple[int, int, int]
) -> list[float]:
    return [
        values[abs(axis) - 1] * (1 if axis > 0 else -1)
        for axis in axes
    ]


class Gy85:
    """Minimal register-level contract for the supported GY-85 chip combination."""

    def __init__(self, bus) -> None:
        self.bus = bus

    def initialize(self) -> None:
        if self.bus.read_byte_data(0x53, 0x00) != 0xE5:
            raise RuntimeError("0x53 is not an ADXL345")
        identity = self.bus.read_byte_data(0x68, 0x00)
        if identity & 0x7E != 0x68:
            raise RuntimeError(f"unexpected ITG WHO_AM_I: {identity:#x}")
        # ADXL345: full resolution +/-2g, 100 Hz, measurement mode.
        self.bus.write_byte_data(0x53, 0x2D, 0x00)
        self.bus.write_byte_data(0x53, 0x31, 0x08)
        self.bus.write_byte_data(0x53, 0x2C, 0x0A)
        self.bus.write_byte_data(0x53, 0x2D, 0x08)
        # ITG3205: reset, PLL X reference, +/-2000 deg/s, DLPF 20 Hz, output 50 Hz.
        self.bus.write_byte_data(0x68, 0x3E, 0x80)
        time.sleep(0.1)
        self.bus.write_byte_data(0x68, 0x3E, 0x01)
        self.bus.write_byte_data(0x68, 0x16, 0x1C)
        self.bus.write_byte_data(0x68, 0x15, 19)
        self.bus.write_byte_data(0x68, 0x17, 0x01)
        # QMC5883L-compatible: soft reset, set/reset period, OSR512, 8G, 200Hz.
        self.bus.write_byte_data(0x0D, 0x0A, 0x80)
        time.sleep(0.02)
        self.bus.write_byte_data(0x0D, 0x0B, 0x01)
        self.bus.write_byte_data(0x0D, 0x09, 0x1D)
        time.sleep(0.2)

    def read(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...] | None] | None:
        if not self.bus.read_byte_data(0x68, 0x1A) & 1:
            return None
        gyro_raw = struct.unpack(
            ">hhh", bytes(self.bus.read_i2c_block_data(0x68, 0x1D, 6))
        )
        if max(abs(value) for value in gyro_raw) >= 32760:
            raise RuntimeError("gyro saturated; sample rejected")
        gyro = tuple(value / 14.375 for value in gyro_raw)
        accel = tuple(
            value * 0.0039
            for value in struct.unpack(
                "<hhh", bytes(self.bus.read_i2c_block_data(0x53, 0x32, 6))
            )
        )
        mag = None
        status = self.bus.read_byte_data(0x0D, 0x06)
        if status & 3:
            values = struct.unpack(
                "<hhh", bytes(self.bus.read_i2c_block_data(0x0D, 0x00, 6))
            )
            if status & 1 and not status & 2:
                mag = values
        return accel, gyro, mag


def build_payload(
    reading: tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...] | None],
    *,
    timestamp: float,
    gyro_axes: tuple[int, int, int] = (1, 2, 3),
    mag_axes: tuple[int, int, int] = (1, 2, 3),
) -> dict[str, object]:
    """Build one protocol-v1 payload from a fresh hardware reading."""
    accel, gyro, mag = reading
    return {
        "type": PROTOCOL_TYPE,
        "version": PROTOCOL_VERSION,
        "t": timestamp,
        "accel_g": accel,
        "gyro_dps": remap(gyro, gyro_axes),
        "mag_raw": None if mag is None else remap(mag, mag_axes),
    }


def encode_payload(payload: dict[str, object]) -> bytes:
    """Encode strict JSON; NaN/Infinity are never emitted on the wire."""
    return json.dumps(payload, allow_nan=False).encode("ascii")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GY-85 -> NavMin UDP (raw sensors, no motor control)"
    )
    parser.add_argument("--host", required=True, help="NavMin PC IPv4 address")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument(
        "--gyro-axes",
        type=signed_axes,
        default=(1, 2, 3),
        help="map gyro chip axes into accelerometer axes, default x,y,z",
    )
    parser.add_argument(
        "--mag-axes",
        type=signed_axes,
        default=(1, 2, 3),
        help="map magnetometer chip axes into accelerometer axes, default x,y,z",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.bus < 0:
        parser.error("invalid UDP port or I2C bus")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from smbus import SMBus
    except ImportError:
        build_argument_parser().error("python3-smbus is required on Raspberry Pi")
    bus = SMBus(args.bus)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.settimeout(0.2)
    try:
        sensor = Gy85(bus)
        sensor.initialize()
        print(
            f"GY-85 -> {args.host}:{args.port}; gyro 50 Hz. Keep still for startup calibration.",
            flush=True,
        )
        last_sample = time.monotonic()
        last_error = 0.0
        while True:
            try:
                reading = sensor.read()
                now = time.monotonic()
                if reading is None:
                    if now - last_sample > 2:
                        raise RuntimeError("no fresh gyro data for 2 seconds")
                    time.sleep(0.002)
                    continue
                last_sample = now
                payload = build_payload(
                    reading,
                    timestamp=now,
                    gyro_axes=args.gyro_axes,
                    mag_axes=args.mag_axes,
                )
                udp.sendto(encode_payload(payload), (args.host, args.port))
            except (OSError, RuntimeError) as exc:
                now = time.monotonic()
                if now - last_error > 1:
                    print(f"GY-85: {exc}", flush=True)
                    last_error = now
                time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        udp.close()
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())
