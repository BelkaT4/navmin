"""Deterministic 2.5D synthetic world for closed-loop tracking diagnostics.

The world owns only scene/trajectory/rendering truth.  It never consumes
Turret commands directly; callers supply the virtual turret pose, keeping ground
truth independent from the production control path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, degrees, pi, sin

import numpy as np

from navmin.calibration import StereoCalibration
from navmin.contracts import BBox, CameraRole

SYNTHETIC_TRACKING_WIDTH = 1296
SYNTHETIC_TRACKING_HEIGHT = 976
SYNTHETIC_TRACKING_FPS = 20
SYNTHETIC_TRACKING_TARGET_SIZE_PX = 48
SYNTHETIC_TRACKING_WORLD_LIMIT_DEG = 30.0
SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG = 90.0
SYNTHETIC_TRACKING_OVERVIEW_VFOV_DEG = (
    SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG
    * SYNTHETIC_TRACKING_HEIGHT
    / SYNTHETIC_TRACKING_WIDTH
)
SYNTHETIC_TRACKING_DISTANCE_M = 150.0
SYNTHETIC_TRACKING_INITIAL_STATIC_S = 10.0
SYNTHETIC_TRACKING_INITIAL_YAW_ACCEL_DEG_S2 = 0.75

_TARGET_BGR = np.asarray((238, 238, 238), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class SyntheticTrackingCameraGeometry:
    camera: CameraRole
    width: int
    height: int
    horizontal_fov_deg: float
    vertical_fov_deg: float
    stereo_offset_px: float = 0.0

    @property
    def pixels_per_degree_x(self) -> float:
        return self.width / self.horizontal_fov_deg

    @property
    def pixels_per_degree_y(self) -> float:
        return self.height / self.vertical_fov_deg

    @property
    def center_x_px(self) -> float:
        return self.width / 2.0

    @property
    def center_y_px(self) -> float:
        return self.height / 2.0


@dataclass(frozen=True, slots=True)
class SyntheticTrackingTruth:
    frame_index: int
    cycle_time_s: float
    phase: str
    target_yaw_deg: float
    target_pitch_deg: float
    target_yaw_rate_deg_s: float
    target_pitch_rate_deg_s: float
    target_yaw_accel_deg_s2: float
    target_pitch_accel_deg_s2: float
    turret_yaw_deg: float
    turret_pitch_deg: float
    true_error_yaw_deg: float
    true_error_pitch_deg: float


@dataclass(frozen=True, slots=True)
class SyntheticTrackingProjection:
    camera: CameraRole
    center_x_px: float
    center_y_px: float
    bbox: BBox | None
    visible: bool


@dataclass(frozen=True, slots=True)
class SyntheticTrackingFrameSet:
    capture_id: int
    overview: np.ndarray
    stereo_left: np.ndarray
    stereo_right: np.ndarray
    truth: SyntheticTrackingTruth
    projections: tuple[SyntheticTrackingProjection, ...]

    def image(self, camera: CameraRole) -> np.ndarray:
        if camera is CameraRole.OVERVIEW:
            return self.overview
        if camera is CameraRole.STEREO_LEFT:
            return self.stereo_left
        if camera is CameraRole.STEREO_RIGHT:
            return self.stereo_right
        raise ValueError(f"unsupported camera role: {camera}")

    def projection(self, camera: CameraRole) -> SyntheticTrackingProjection:
        for item in self.projections:
            if item.camera is camera:
                return item
        raise ValueError(f"unsupported camera role: {camera}")


@dataclass(frozen=True, slots=True)
class _MotionState:
    phase: str
    yaw_deg: float
    pitch_deg: float
    yaw_rate_deg_s: float
    pitch_rate_deg_s: float
    yaw_accel_deg_s2: float
    pitch_accel_deg_s2: float


@dataclass(frozen=True, slots=True)
class _Phase:
    name: str
    duration_s: float


_PHASES = (
    _Phase("STATIC", SYNTHETIC_TRACKING_INITIAL_STATIC_S),
    _Phase("YAW_ACCEL_POS", 4.0),
    _Phase("YAW_SLOW_POS", 6.0),
    _Phase("YAW_BRAKE_POS", 2.0),
    _Phase("YAW_REVERSE_ACCEL", 4.0),
    _Phase("YAW_FAST_NEG", 6.0),
    _Phase("YAW_BRAKE_NEG", 2.0),
    _Phase("PITCH_ACCEL_POS", 2.0),
    _Phase("PITCH_MEDIUM_POS", 8.0),
    _Phase("PITCH_BRAKE_POS", 2.0),
    _Phase("PITCH_REVERSE_ACCEL", 4.0),
    _Phase("PITCH_FAST_NEG", 7.0),
    _Phase("PITCH_BRAKE_NEG", 2.0),
    _Phase("DIAGONAL_RETURN", 8.0),
    _Phase("COMBINED_SINE", 16.0),
    _Phase("STATIC_END", 2.0),
)
_CYCLE_DURATION_S = sum(phase.duration_s for phase in _PHASES)


def _motion_state_at(time_s: float) -> _MotionState:
    t = time_s % _CYCLE_DURATION_S
    starts: list[tuple[_Phase, float]] = []
    cursor = 0.0
    for phase in _PHASES:
        starts.append((phase, cursor))
        cursor += phase.duration_s

    for phase, start in starts:
        if t < start + phase.duration_s:
            u = t - start
            name = phase.name
            break
    else:  # floating-point boundary only
        name = "STATIC_END"
        u = 0.0

    if name == "STATIC":
        return _MotionState(name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if name == "YAW_ACCEL_POS":
        accel = SYNTHETIC_TRACKING_INITIAL_YAW_ACCEL_DEG_S2
        return _MotionState(
            name,
            0.5 * accel * u * u,
            0.0,
            accel * u,
            0.0,
            accel,
            0.0,
        )
    if name == "YAW_SLOW_POS":
        return _MotionState(name, 6.0 + 3.0 * u, 0.0, 3.0, 0.0, 0.0, 0.0)
    if name == "YAW_BRAKE_POS":
        return _MotionState(
            name,
            24.0 + 3.0 * u - 0.75 * u * u,
            0.0,
            3.0 - 1.5 * u,
            0.0,
            -1.5,
            0.0,
        )
    if name == "YAW_REVERSE_ACCEL":
        return _MotionState(name, 27.0 - 0.75 * u * u, 0.0, -1.5 * u, 0.0, -1.5, 0.0)
    if name == "YAW_FAST_NEG":
        return _MotionState(name, 15.0 - 6.0 * u, 0.0, -6.0, 0.0, 0.0, 0.0)
    if name == "YAW_BRAKE_NEG":
        return _MotionState(
            name,
            -21.0 - 6.0 * u + 1.5 * u * u,
            0.0,
            -6.0 + 3.0 * u,
            0.0,
            3.0,
            0.0,
        )
    if name == "PITCH_ACCEL_POS":
        return _MotionState(name, -27.0, 0.625 * u * u, 0.0, 1.25 * u, 0.0, 1.25)
    if name == "PITCH_MEDIUM_POS":
        return _MotionState(name, -27.0, 2.5 + 2.5 * u, 0.0, 2.5, 0.0, 0.0)
    if name == "PITCH_BRAKE_POS":
        return _MotionState(
            name,
            -27.0,
            22.5 + 2.5 * u - 0.625 * u * u,
            0.0,
            2.5 - 1.25 * u,
            0.0,
            -1.25,
        )
    if name == "PITCH_REVERSE_ACCEL":
        return _MotionState(name, -27.0, 25.0 - 0.625 * u * u, 0.0, -1.25 * u, 0.0, -1.25)
    if name == "PITCH_FAST_NEG":
        return _MotionState(name, -27.0, 15.0 - 5.0 * u, 0.0, -5.0, 0.0, 0.0)
    if name == "PITCH_BRAKE_NEG":
        return _MotionState(
            name,
            -27.0,
            -20.0 - 5.0 * u + 1.25 * u * u,
            0.0,
            -5.0 + 2.5 * u,
            0.0,
            2.5,
        )
    if name == "DIAGONAL_RETURN":
        duration = 8.0
        q = u / duration
        smooth = 3.0 * q * q - 2.0 * q * q * q
        ds_dt = (6.0 * q - 6.0 * q * q) / duration
        d2s_dt2 = (6.0 - 12.0 * q) / (duration * duration)
        return _MotionState(
            name,
            -27.0 + 27.0 * smooth,
            -25.0 + 25.0 * smooth,
            27.0 * ds_dt,
            25.0 * ds_dt,
            27.0 * d2s_dt2,
            25.0 * d2s_dt2,
        )
    if name == "COMBINED_SINE":
        duration = 16.0
        omega = 2.0 * pi / duration
        theta_yaw = omega * u
        theta_pitch = 2.0 * omega * u

        def wave(amplitude: float, theta: float, rate: float) -> tuple[float, float, float]:
            s = sin(theta)
            c = cos(theta)
            position = amplitude * s**3
            velocity = 3.0 * amplitude * rate * s * s * c
            acceleration = 3.0 * amplitude * rate * rate * s * (2.0 * c * c - s * s)
            return position, velocity, acceleration

        yaw, yaw_rate, yaw_accel = wave(10.0, theta_yaw, omega)
        pitch, pitch_rate, pitch_accel = wave(6.0, theta_pitch, 2.0 * omega)
        return _MotionState(name, yaw, pitch, yaw_rate, pitch_rate, yaw_accel, pitch_accel)
    return _MotionState("STATIC_END", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _stereo_fov(reference: StereoCalibration) -> tuple[float, float]:
    fx = float(reference.P1[0][0])
    fy = float(reference.P1[1][1])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("stereo projection focal lengths must be > 0")
    return (
        degrees(2.0 * atan(reference.image_width / (2.0 * fx))),
        degrees(2.0 * atan(reference.image_height / (2.0 * fy))),
    )


def _stereo_disparity_px(reference: StereoCalibration, *, distance_m: float) -> float:
    if distance_m <= 0.0:
        raise ValueError("distance_m must be > 0")
    fx = float(reference.P1[0][0])
    baseline_native = abs(float(reference.P2[0][3]) / fx)
    baseline_m = baseline_native * reference.length_scale_to_m
    if baseline_m <= 0.0:
        raise ValueError("stereo calibration must provide a non-zero baseline")
    return fx * baseline_m / distance_m


class SyntheticTrackingWorld:
    """Render one deterministic angular 2.5D world into three camera views."""

    def __init__(
        self,
        stereo_calibration: StereoCalibration,
        *,
        fps: int = SYNTHETIC_TRACKING_FPS,
        target_distance_m: float = SYNTHETIC_TRACKING_DISTANCE_M,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if (
            stereo_calibration.image_width != SYNTHETIC_TRACKING_WIDTH
            or stereo_calibration.image_height != SYNTHETIC_TRACKING_HEIGHT
        ):
            raise ValueError("synthetic tracking world requires 1296x976 stereo calibration")
        self.stereo_calibration = stereo_calibration
        self.fps = fps
        self.target_distance_m = target_distance_m
        stereo_hfov, stereo_vfov = _stereo_fov(stereo_calibration)
        disparity_px = _stereo_disparity_px(
            stereo_calibration,
            distance_m=target_distance_m,
        )
        self.disparity_px = disparity_px
        self.geometries = {
            CameraRole.OVERVIEW: SyntheticTrackingCameraGeometry(
                CameraRole.OVERVIEW,
                SYNTHETIC_TRACKING_WIDTH,
                SYNTHETIC_TRACKING_HEIGHT,
                SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG,
                SYNTHETIC_TRACKING_OVERVIEW_VFOV_DEG,
            ),
            CameraRole.STEREO_LEFT: SyntheticTrackingCameraGeometry(
                CameraRole.STEREO_LEFT,
                SYNTHETIC_TRACKING_WIDTH,
                SYNTHETIC_TRACKING_HEIGHT,
                stereo_hfov,
                stereo_vfov,
                +disparity_px / 2.0,
            ),
            CameraRole.STEREO_RIGHT: SyntheticTrackingCameraGeometry(
                CameraRole.STEREO_RIGHT,
                SYNTHETIC_TRACKING_WIDTH,
                SYNTHETIC_TRACKING_HEIGHT,
                stereo_hfov,
                stereo_vfov,
                -disparity_px / 2.0,
            ),
        }
        self._background_bases = {
            camera: self._make_background_base(geometry)
            for camera, geometry in self.geometries.items()
        }

    @property
    def cycle_duration_s(self) -> float:
        return _CYCLE_DURATION_S

    def truth(
        self,
        frame_index: int,
        *,
        turret_yaw_deg: float = 0.0,
        turret_pitch_deg: float = 0.0,
    ) -> SyntheticTrackingTruth:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        cycle_time_s = (frame_index / self.fps) % _CYCLE_DURATION_S
        state = _motion_state_at(cycle_time_s)
        return SyntheticTrackingTruth(
            frame_index=frame_index,
            cycle_time_s=cycle_time_s,
            phase=state.phase,
            target_yaw_deg=state.yaw_deg,
            target_pitch_deg=state.pitch_deg,
            target_yaw_rate_deg_s=state.yaw_rate_deg_s,
            target_pitch_rate_deg_s=state.pitch_rate_deg_s,
            target_yaw_accel_deg_s2=state.yaw_accel_deg_s2,
            target_pitch_accel_deg_s2=state.pitch_accel_deg_s2,
            turret_yaw_deg=float(turret_yaw_deg),
            turret_pitch_deg=float(turret_pitch_deg),
            true_error_yaw_deg=state.yaw_deg - float(turret_yaw_deg),
            true_error_pitch_deg=state.pitch_deg - float(turret_pitch_deg),
        )

    def render(
        self,
        frame_index: int,
        *,
        turret_yaw_deg: float = 0.0,
        turret_pitch_deg: float = 0.0,
    ) -> SyntheticTrackingFrameSet:
        truth = self.truth(
            frame_index,
            turret_yaw_deg=turret_yaw_deg,
            turret_pitch_deg=turret_pitch_deg,
        )
        images: dict[CameraRole, np.ndarray] = {}
        projections: list[SyntheticTrackingProjection] = []
        for camera in (
            CameraRole.OVERVIEW,
            CameraRole.STEREO_LEFT,
            CameraRole.STEREO_RIGHT,
        ):
            geometry = self.geometries[camera]
            image = self._background(
                geometry,
                turret_yaw_deg=truth.turret_yaw_deg,
                turret_pitch_deg=truth.turret_pitch_deg,
            )
            projection = self._project_target(geometry, truth)
            self._draw_target(image, projection)
            image.flags.writeable = False
            images[camera] = image
            projections.append(projection)
        return SyntheticTrackingFrameSet(
            capture_id=frame_index,
            overview=images[CameraRole.OVERVIEW],
            stereo_left=images[CameraRole.STEREO_LEFT],
            stereo_right=images[CameraRole.STEREO_RIGHT],
            truth=truth,
            projections=tuple(projections),
        )

    def _project_target(
        self,
        geometry: SyntheticTrackingCameraGeometry,
        truth: SyntheticTrackingTruth,
    ) -> SyntheticTrackingProjection:
        x = (
            geometry.center_x_px
            + truth.true_error_yaw_deg * geometry.pixels_per_degree_x
            + geometry.stereo_offset_px
        )
        y = geometry.center_y_px - truth.true_error_pitch_deg * geometry.pixels_per_degree_y
        half = SYNTHETIC_TRACKING_TARGET_SIZE_PX / 2.0
        x0 = round(x - half)
        y0 = round(y - half)
        x1 = x0 + SYNTHETIC_TRACKING_TARGET_SIZE_PX
        y1 = y0 + SYNTHETIC_TRACKING_TARGET_SIZE_PX
        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(geometry.width, x1)
        clip_y1 = min(geometry.height, y1)
        visible = clip_x0 < clip_x1 and clip_y0 < clip_y1
        bbox = (
            BBox(
                x=clip_x0,
                y=clip_y0,
                width=clip_x1 - clip_x0,
                height=clip_y1 - clip_y0,
            )
            if visible
            else None
        )
        return SyntheticTrackingProjection(camera=geometry.camera, center_x_px=x, center_y_px=y, bbox=bbox, visible=visible)

    @staticmethod
    def _draw_target(image: np.ndarray, projection: SyntheticTrackingProjection) -> None:
        if projection.bbox is None:
            return
        bbox = projection.bbox
        image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = _TARGET_BGR

    def _background(
        self,
        geometry: SyntheticTrackingCameraGeometry,
        *,
        turret_yaw_deg: float,
        turret_pitch_deg: float,
    ) -> np.ndarray:
        # The texture is periodic, so turret rotation is exactly represented as
        # an image translation.  Precomputing the expensive sin/tanh field once
        # keeps the normal full-SIM path within its 20 FPS frame budget.
        shift_x_px = -round(turret_yaw_deg * geometry.pixels_per_degree_x)
        shift_y_px = round(turret_pitch_deg * geometry.pixels_per_degree_y)
        return np.roll(
            self._background_bases[geometry.camera],
            shift=(shift_y_px, shift_x_px),
            axis=(0, 1),
        )

    @staticmethod
    def _make_background_base(
        geometry: SyntheticTrackingCameraGeometry,
    ) -> np.ndarray:
        x = np.arange(geometry.width, dtype=np.float32)
        y = np.arange(geometry.height, dtype=np.float32)
        world_yaw = (x - geometry.center_x_px) / geometry.pixels_per_degree_x
        world_pitch = -(y - geometry.center_y_px) / geometry.pixels_per_degree_y
        yaw_grid = world_yaw[None, :]
        pitch_grid = world_pitch[:, None]

        # Low-contrast, band-limited texture: visible during camera motion but
        # intentionally far below the bright square's local contrast.
        intensity = (
            78.0
            + 2.6 * np.sin(2.0 * pi * yaw_grid / 7.0)
            + 2.0 * np.sin(2.0 * pi * pitch_grid / 5.0)
            + 1.4 * np.sin(2.0 * pi * (yaw_grid + pitch_grid) / 11.0)
            + 3.5 * np.tanh(pitch_grid / 1.8)
        )
        intensity = np.clip(np.rint(intensity), 0, 255).astype(np.uint8)
        background = np.repeat(intensity[:, :, None], 3, axis=2)
        background.flags.writeable = False
        return background


__all__ = [
    "SYNTHETIC_TRACKING_DISTANCE_M",
    "SYNTHETIC_TRACKING_FPS",
    "SYNTHETIC_TRACKING_HEIGHT",
    "SYNTHETIC_TRACKING_OVERVIEW_HFOV_DEG",
    "SYNTHETIC_TRACKING_OVERVIEW_VFOV_DEG",
    "SYNTHETIC_TRACKING_TARGET_SIZE_PX",
    "SYNTHETIC_TRACKING_WIDTH",
    "SYNTHETIC_TRACKING_WORLD_LIMIT_DEG",
    "SyntheticTrackingCameraGeometry",
    "SyntheticTrackingFrameSet",
    "SyntheticTrackingProjection",
    "SyntheticTrackingTruth",
    "SyntheticTrackingWorld",
]
