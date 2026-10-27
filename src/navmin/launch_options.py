"""Typed launcher source selection independent from CLI and runtime composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from navmin.config import AppConfig
from navmin.contracts import CameraRole


class CameraSourceMode(Enum):
    """Runtime source choice for mandatory Overview and Stereo Left cameras."""

    ON = "on"
    SIM = "sim"


class StereoRightMode(Enum):
    """Runtime source choice for optional Stereo Right."""

    ON = "on"
    SIM = "sim"
    OFF = "off"


class Stm32Mode(Enum):
    """Runtime source choice for the turret controller."""

    ON = "on"
    SIM = "sim"


class LaunchSelectionError(ValueError):
    """Raised when persistent config cannot provide a valid launcher default."""


@dataclass(frozen=True)
class LaunchOptions:
    """Fully resolved operator choices for one application launch."""

    overview: CameraSourceMode
    stereo_left: CameraSourceMode
    stereo_right: StereoRightMode
    stm32: Stm32Mode
    preflight_only: bool = False
    verbose: bool = False
    config_path: Path = Path("config/navmin.json")


@dataclass(frozen=True)
class EffectiveLaunchSelection:
    """Immutable effective config plus source-selection diagnostics."""

    options: LaunchOptions
    config: AppConfig
    warnings: tuple[str, ...] = ()

    @property
    def real_camera_roles(self) -> tuple[CameraRole, ...]:
        roles: list[CameraRole] = []
        if self.options.overview is CameraSourceMode.ON:
            roles.append(CameraRole.OVERVIEW)
        if self.options.stereo_left is CameraSourceMode.ON:
            roles.append(CameraRole.STEREO_LEFT)
        if self.options.stereo_right is StereoRightMode.ON:
            roles.append(CameraRole.STEREO_RIGHT)
        return tuple(roles)

    @property
    def simulated_camera_roles(self) -> tuple[CameraRole, ...]:
        roles: list[CameraRole] = []
        if self.options.overview is CameraSourceMode.SIM:
            roles.append(CameraRole.OVERVIEW)
        if self.options.stereo_left is CameraSourceMode.SIM:
            roles.append(CameraRole.STEREO_LEFT)
        if self.options.stereo_right is StereoRightMode.SIM:
            roles.append(CameraRole.STEREO_RIGHT)
        return tuple(roles)

    @property
    def requires_camera_network(self) -> bool:
        return bool(self.real_camera_roles)


def default_launch_options(
    config: AppConfig,
    *,
    config_path: Path = Path("config/navmin.json"),
) -> LaunchOptions:
    """Resolve launcher defaults from persistent config without mutating it.

    Overview and Stereo Left are mandatory in the current ApplicationRuntime.
    A persisted disabled value for either camera therefore cannot be represented
    as a supported launcher default and is rejected instead of silently changing
    the configured intent.
    """
    cameras = config.vision.cameras
    if not cameras.overview.enabled:
        raise LaunchSelectionError(
            "Overview is disabled in config, but the current runtime requires it"
        )
    if not cameras.stereo_left.enabled:
        raise LaunchSelectionError(
            "Stereo Left is disabled in config, but the current runtime requires it"
        )

    return LaunchOptions(
        overview=CameraSourceMode.ON,
        stereo_left=CameraSourceMode.ON,
        stereo_right=(
            StereoRightMode.ON if cameras.stereo_right.enabled else StereoRightMode.OFF
        ),
        stm32=Stm32Mode.SIM if config.turret.emulate_stm32 else Stm32Mode.ON,
        config_path=config_path,
    )


def resolve_launch_options(
    config: AppConfig,
    *,
    config_path: Path = Path("config/navmin.json"),
    overview: CameraSourceMode | None = None,
    stereo_left: CameraSourceMode | None = None,
    stereo_right: StereoRightMode | None = None,
    stm32: Stm32Mode | None = None,
    preflight_only: bool = False,
    verbose: bool = False,
) -> LaunchOptions:
    """Resolve partial non-interactive overrides against persistent defaults.

    Explicit ON/SIM values may enable mandatory cameras for this launch even if
    their persisted ``enabled`` flag is false. An omitted mandatory source still
    has no representable OFF default and therefore fails clearly.
    """
    cameras = config.vision.cameras
    if overview is None:
        if not cameras.overview.enabled:
            raise LaunchSelectionError(
                "Overview is disabled in config; pass --overview on or --overview sim"
            )
        overview = CameraSourceMode.ON
    if stereo_left is None:
        if not cameras.stereo_left.enabled:
            raise LaunchSelectionError(
                "Stereo Left is disabled in config; pass --stereo-left on or "
                "--stereo-left sim"
            )
        stereo_left = CameraSourceMode.ON
    if stereo_right is None:
        stereo_right = (
            StereoRightMode.ON
            if cameras.stereo_right.enabled
            else StereoRightMode.OFF
        )
    if stm32 is None:
        stm32 = Stm32Mode.SIM if config.turret.emulate_stm32 else Stm32Mode.ON
    return LaunchOptions(
        overview=overview,
        stereo_left=stereo_left,
        stereo_right=stereo_right,
        stm32=stm32,
        preflight_only=preflight_only,
        verbose=verbose,
        config_path=config_path,
    )


def apply_launch_options(
    config: AppConfig,
    options: LaunchOptions,
) -> EffectiveLaunchSelection:
    """Build immutable effective runtime settings for one resolved launch.

    This stage intentionally does not replace real camera source objects with
    synthetic sources. Source factories and synthetic calibration belong to the
    next composition checkpoint. Here only config-backed enablement, STM32
    emulation and stereo feature availability are resolved.
    """
    cameras = config.vision.cameras
    effective_cameras = replace(
        cameras,
        overview=replace(cameras.overview, enabled=True),
        stereo_left=replace(cameras.stereo_left, enabled=True),
        stereo_right=replace(
            cameras.stereo_right,
            enabled=options.stereo_right is not StereoRightMode.OFF,
        ),
    )

    warnings: list[str] = []
    stereo_sources_compatible = True
    if options.stereo_right is StereoRightMode.OFF:
        stereo_sources_compatible = False
        warnings.append(
            "Stereo Right is OFF; stereo pairing and distance are disabled for "
            "this launch."
        )
    else:
        left_is_sim = options.stereo_left is CameraSourceMode.SIM
        right_is_sim = options.stereo_right is StereoRightMode.SIM
        if left_is_sim != right_is_sim:
            stereo_sources_compatible = False
            warnings.append(
                "Stereo Left and Stereo Right use mixed real/sim sources; stereo "
                "pairing and distance are disabled for this launch."
            )

    stereo = config.vision.distance.stereo
    effective_stereo = replace(
        stereo,
        stereo_enabled=stereo.stereo_enabled and stereo_sources_compatible,
    )
    effective_vision = replace(
        config.vision,
        cameras=effective_cameras,
        distance=replace(config.vision.distance, stereo=effective_stereo),
    )
    effective_config = replace(
        config,
        vision=effective_vision,
        turret=replace(
            config.turret,
            emulate_stm32=options.stm32 is Stm32Mode.SIM,
        ),
    )
    return EffectiveLaunchSelection(
        options=options,
        config=effective_config,
        warnings=tuple(warnings),
    )


__all__ = [
    "CameraSourceMode",
    "EffectiveLaunchSelection",
    "LaunchOptions",
    "LaunchSelectionError",
    "StereoRightMode",
    "Stm32Mode",
    "apply_launch_options",
    "default_launch_options",
    "resolve_launch_options",
]
