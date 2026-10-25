"""Launcher-owned visual motion simulator; never connected to real equipment."""

from __future__ import annotations

from math import copysign, cos, radians, sin
from threading import Lock
from time import monotonic

import cv2
import numpy as np

from navmin.contracts import CameraRole
from navmin.turret.protocol import (
    CommandCode,
    MoveRelativePayload,
    SetConfigPayload,
    SetVelocityPayload,
)
from navmin.turret.simulator import FakeStm32Endpoint, FakeTransport


class DemoPlatform(FakeStm32Endpoint):
    """Execute accepted protocol commands once and share pose with camera senders."""

    def __init__(self, axes, *, clock=monotonic):
        super().__init__()
        self._clock = clock
        self._lock = Lock()
        self._updated = clock()
        self._position = [0.0, 0.0]
        self._velocity = [0.0, 0.0]
        self._target = None
        self._motors = False
        self._speed = [20.0, 20.0]
        self._watchdog = 0.2
        self._velocity_deadline = 0.0
        self._degrees_per_step = [
            360.0
            / (a.full_steps_per_revolution * a.microstep_divider)
            * (-1 if a.invert else 1)
            for a in (axes.x, axes.y)
        ]

    def _advance(self):
        now = self._clock()
        dt = max(0.0, now - self._updated)
        if self._motors:
            for axis in range(2):
                if self._target is not None:
                    delta = self._target[axis] - self._position[axis]
                    self._position[axis] += copysign(
                        min(abs(delta), self._speed[axis] * dt), delta
                    )
                else:
                    active_dt = max(
                        0.0, min(now, self._velocity_deadline) - self._updated
                    )
                    self._position[axis] += self._velocity[axis] * active_dt
        self._updated = now

    def pose(self):
        with self._lock:
            self._advance()
            return tuple(self._position)

    def handle_request(self, raw_frame):
        with self._lock:
            self._advance()
            response = super().handle_request(raw_frame)
            # The base endpoint records only new executions, excluding exact retries.
            if self.executed_request_history:
                request = self.executed_request_history[-1]
                command, payload = request.command, request.payload
                if command in (CommandCode.EMERGENCY_STOP, CommandCode.MOTOR_OFF):
                    self._motors = False
                    self._target = None
                    self._velocity = [0.0, 0.0]
                elif command is CommandCode.MOTOR_ON:
                    self._motors = True
                elif isinstance(payload, SetConfigPayload):
                    self._speed = [
                        abs(v * scale)
                        for v, scale in zip(
                            (payload.max_speed_x_steps_s, payload.max_speed_y_steps_s),
                            self._degrees_per_step,
                        )
                    ]
                    self._watchdog = payload.velocity_watchdog_timeout_ms / 1000.0
                elif isinstance(payload, MoveRelativePayload) and self._motors:
                    self._target = [
                        p + v * scale
                        for p, v, scale in zip(
                            self._position,
                            (payload.delta_x_steps, payload.delta_y_steps),
                            self._degrees_per_step,
                        )
                    ]
                    self._velocity = [0.0, 0.0]
                elif isinstance(payload, SetVelocityPayload):
                    self._target = None
                    self._velocity = (
                        [
                            v * scale
                            for v, scale in zip(
                                (
                                    payload.velocity_x_steps_s,
                                    payload.velocity_y_steps_s,
                                ),
                                self._degrees_per_step,
                            )
                        ]
                        if self._motors
                        else [0.0, 0.0]
                    )
                    self._velocity_deadline = self._clock() + self._watchdog
            # Histories are test diagnostics, not an unbounded runtime recording.
            self.executed_request_history.clear()
            self.request_history.clear()
            self.raw_request_history.clear()
            return response

    def transport(self, _port, baudrate, _emulate):
        return DemoTransport(self, baudrate=baudrate)


class DemoTransport(FakeTransport):
    def write_frame(self, frame, timeout_s):
        super().write_frame(frame, timeout_s)
        self.raw_write_history.clear()
        self.raw_write_baudrate_history.clear()
        self.event_history.clear()


class DemoScene:
    """Fixed world landmarks and a moving bright tracking target, projected to raw video."""

    def __init__(self, platform, camera, calibration, *, clock=monotonic):
        self.platform = platform
        self.camera = camera
        self.calibration = calibration
        self._clock = clock
        self._started = clock()
        self.frames_generated = 0

    def project(self, yaw, pitch, pose):
        # Yaw then pitch rotation gives the same ray-angle sign as Core/Aiming.
        yaw, pitch = radians(yaw), radians(pitch)
        x, y, z = sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch)
        pan, tilt = map(radians, pose)
        x, z = cos(pan) * x - sin(pan) * z, sin(pan) * x + cos(pan) * z
        y, z = cos(tilt) * y + sin(tilt) * z, -sin(tilt) * y + cos(tilt) * z
        if z <= 0.01:
            return None
        c = self.calibration
        points = np.array([[[x, y, z]]], dtype=np.float64)
        zero = np.zeros(3)
        if self.camera is CameraRole.OVERVIEW:
            projected, _ = cv2.fisheye.projectPoints(
                points, zero, zero, np.array(c.K), np.array(c.D)
            )
        else:
            projected, _ = cv2.projectPoints(
                points, zero, zero, np.array(c.K_left), np.array(c.D_left)
            )
        return tuple(round(v) for v in projected.reshape(2))

    def next_frame(self):
        c = self.calibration
        frame = np.full(
            (c.image_height, c.image_width, 3), (30, 40, 48), dtype=np.uint8
        )
        pose = self.platform.pose()
        # Dense numbered landmarks remain fixed in world coordinates for calibration.
        # Cull distant landmarks before OpenCV projection, keeping rendering bounded.
        pan = round(pose[0] / 4) * 4
        tilt = round(pose[1] / 4) * 4
        for yaw in range(pan - 40, pan + 41, 4):
            for pitch in range(max(-80, tilt - 32), min(80, tilt + 32) + 1, 4):
                point = self.project(yaw, pitch, pose)
                if (
                    point is not None
                    and 5 < point[0] < c.image_width - 5
                    and 20 < point[1] < c.image_height - 20
                ):
                    cv2.drawMarker(frame, point, (60, 150, 170), cv2.MARKER_CROSS, 9, 1)
                    cv2.putText(
                        frame,
                        f"{yaw},{pitch}",
                        (point[0] + 4, point[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (60, 150, 170),
                        1,
                    )
        elapsed = self._clock() - self._started
        point = self.project(5 * sin(elapsed * 0.25), 2 * sin(elapsed * 0.17), pose)
        if (
            point is not None
            and 0 <= point[0] < c.image_width
            and 0 <= point[1] < c.image_height
        ):
            cv2.circle(frame, point, 6, (245, 245, 245), -1)
        cv2.putText(
            frame,
            f"DEMO X {pose[0]:+.1f} Y {pose[1]:+.1f}",
            (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            frame,
            "CALIB: fixed crosses; TRACK: white dot",
            (3, c.image_height - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (220, 220, 220),
            1,
        )
        self.frames_generated += 1
        return frame
