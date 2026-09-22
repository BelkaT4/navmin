"""Immutable configuration models and owner-facing comparison metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from navmin.contracts import CameraRole, DistanceSource


class ProcessingScope(Enum):
    MAIN_ONLY = "main-only"
    MAIN_AND_PREVIEW = "main-and-preview"


class ConfigApplyPolicy(Enum):
    """Architecturally fixed apply classifications, without owner actions."""

    DYNAMIC = "dynamic"
    CAMERA_PIPELINE_RESTART = "camera-pipeline-restart"
    TURRET_RECONNECT = "turret-reconnect"
    CONTROLLED_SERIAL_TRANSITION = "controlled-serial-transition"
    APPLICATION_RESTART = "application-restart"


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool
    address: str
    port: int
    rtp_enabled: bool
    buffer_size: int
    processing_enabled: bool
    vision_processor_class: str


@dataclass(frozen=True)
class CamerasConfig:
    overview: CameraConfig
    stereo_left: CameraConfig
    stereo_right: CameraConfig


@dataclass(frozen=True)
class StereoDistanceConfig:
    stereo_enabled: bool
    right_frame_buffer_size: int
    pair_timeout_ms: int


@dataclass(frozen=True)
class VisionDistanceConfig:
    source: DistanceSource
    manual_distance_m: float
    distance_stale_timeout_ms: int
    stereo: StereoDistanceConfig


@dataclass(frozen=True)
class VisionConfig:
    processing_scope: ProcessingScope
    cameras: CamerasConfig
    distance: VisionDistanceConfig
    camera_stale_timeout_ms: int
    simulation_mode: bool


@dataclass(frozen=True)
class AimPointConfig:
    x_px: int | None
    y_px: int | None


@dataclass(frozen=True)
class AimPointsConfig:
    overview: AimPointConfig
    stereo_left: AimPointConfig


@dataclass(frozen=True)
class AimingConfig:
    lead_time_ms: int
    target_lost_timeout_ms: int
    aim_points: AimPointsConfig


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baudrate: int
    response_timeout_ms: int
    max_retries: int
    inter_request_delay_ms: int


@dataclass(frozen=True)
class AxisMechanicsConfig:
    invert: bool
    full_steps_per_revolution: int
    microstep_divider: int
    max_relative_move_deg: float


@dataclass(frozen=True)
class AxesConfig:
    x: AxisMechanicsConfig
    y: AxisMechanicsConfig


@dataclass(frozen=True)
class PidControllerConfig:
    pid_kp_x: float
    pid_ki_x: float
    pid_kd_x: float
    pid_kp_y: float
    pid_ki_y: float
    pid_kd_y: float


@dataclass(frozen=True)
class Stm32Config:
    max_speed_x_deg_s: float
    max_speed_y_deg_s: float
    acceleration_x_deg_s2: float
    acceleration_y_deg_s2: float
    velocity_watchdog_timeout_ms: int


@dataclass(frozen=True)
class TurretConfig:
    serial: SerialConfig
    axes: AxesConfig
    controller: PidControllerConfig
    stm32: Stm32Config
    emulate_stm32: bool


@dataclass(frozen=True)
class UiConfig:
    default_camera: CameraRole
    show_fps: bool
    show_stereo_right_diagnostics: bool


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    vision: VisionConfig
    aiming: AimingConfig
    turret: TurretConfig
    ui: UiConfig


_DYNAMIC_EXACT_PATHS = frozenset(
    {
        "vision.processing-scope",
        "aiming.lead-time-ms",
        "aiming.target-lost-timeout-ms",
        "ui.show-fps",
        "ui.show-stereo-right-diagnostics",
    }
)


def config_apply_policy(path: str) -> ConfigApplyPolicy | None:
    """Return only classifications already fixed by configuration.md.

    ``None`` deliberately means that Stage 2 does not invent a policy for that
    field. Owners may only act on policy once the architecture defines it.
    """
    if path in _DYNAMIC_EXACT_PATHS:
        return ConfigApplyPolicy.DYNAMIC

    parts = path.split(".")
    if (
        parts[:2] == ["turret", "controller"]
        and len(parts) == 3
        and parts[2].startswith("pid-")
    ):
        return ConfigApplyPolicy.DYNAMIC

    if parts[:2] == ["turret", "stm32"] and len(parts) == 3:
        return ConfigApplyPolicy.DYNAMIC

    if len(parts) == 4 and parts[:2] == ["vision", "cameras"]:
        field = parts[3]
        if field == "processing-enabled":
            return ConfigApplyPolicy.DYNAMIC
        if field in {
            "address",
            "port",
            "rtp-enabled",
            "buffer-size",
            "vision-processor-class",
        }:
            return ConfigApplyPolicy.CAMERA_PIPELINE_RESTART

    if path in {
        "turret.serial.port",
        "turret.serial.response-timeout-ms",
        "turret.serial.max-retries",
        "turret.serial.inter-request-delay-ms",
    }:
        return ConfigApplyPolicy.TURRET_RECONNECT
    if path == "turret.serial.baudrate":
        return ConfigApplyPolicy.CONTROLLED_SERIAL_TRANSITION
    if path == "turret.emulate-stm32":
        return ConfigApplyPolicy.APPLICATION_RESTART

    if (
        len(parts) == 4
        and parts[:2] == ["turret", "axes"]
        and parts[3] in {
            "invert",
            "full-steps-per-revolution",
            "microstep-divider",
            "max-relative-move-deg",
        }
    ):
        return ConfigApplyPolicy.APPLICATION_RESTART

    # Aim-point paths have four components, e.g.
    # aiming.aim-points.overview.x-px.
    if (
        len(parts) == 4
        and parts[:2] == ["aiming", "aim-points"]
        and parts[3] in {"x-px", "y-px"}
    ):
        return ConfigApplyPolicy.DYNAMIC

    return None


def _changed_leaf_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        paths: list[str] = []
        for key in sorted(old.keys() | new.keys()):
            child = f"{prefix}.{key}" if prefix else key
            if key not in old or key not in new:
                paths.append(child)
            else:
                paths.extend(_changed_leaf_paths(old[key], new[key], child))
        return paths
    return [] if old == new else [prefix]


def changed_config_paths(old: AppConfig, new: AppConfig) -> tuple[str, ...]:
    """Return changed persisted leaf paths without deciding owner actions."""
    from .parser import config_to_mapping

    return tuple(_changed_leaf_paths(config_to_mapping(old), config_to_mapping(new)))


__all__ = [
    "AimPointConfig",
    "AimPointsConfig",
    "AimingConfig",
    "AppConfig",
    "AxesConfig",
    "AxisMechanicsConfig",
    "CameraConfig",
    "CamerasConfig",
    "ConfigApplyPolicy",
    "PidControllerConfig",
    "ProcessingScope",
    "SerialConfig",
    "StereoDistanceConfig",
    "Stm32Config",
    "TurretConfig",
    "UiConfig",
    "VisionConfig",
    "VisionDistanceConfig",
    "changed_config_paths",
    "config_apply_policy",
]
