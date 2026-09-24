"""Strict parser and serializer for NavMin config schema v1."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlsplit

from navmin.contracts import CameraRole, DistanceSource

from .models import (
    AimingConfig,
    AimPointConfig,
    AimPointsConfig,
    AppConfig,
    AxesConfig,
    AxisMechanicsConfig,
    CameraConfig,
    CamerasConfig,
    PidControllerConfig,
    ProcessingScope,
    RtpJpegSourceConfig,
    RtspDecoderMode,
    RtspProtocol,
    RtspSourceConfig,
    SerialConfig,
    StereoDistanceConfig,
    Stm32Config,
    TurretConfig,
    UiConfig,
    VisionConfig,
    VisionDistanceConfig,
)

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_BAUDRATES = frozenset({9600, 19200, 38400, 57600, 115200})


class ConfigError(ValueError):
    """Base class for configuration loading, validation, and persistence errors."""


class ConfigFileNotFoundError(ConfigError):
    pass


class ConfigJsonError(ConfigError):
    def __init__(self, message: str) -> None:
        self.path = "$"
        self.reason = message
        super().__init__(f"$: {message}")


class ConfigValidationError(ConfigError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class UnsupportedConfigSchemaError(ConfigValidationError):
    def __init__(self, actual: int) -> None:
        self.actual = actual
        super().__init__(
            "schema-version",
            f"unsupported schema version {actual}; expected {SUPPORTED_SCHEMA_VERSION}",
        )


class ConfigPersistenceError(ConfigError):
    pass


def _validation(path: str, reason: str) -> NoReturn:
    raise ConfigValidationError(path, reason)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _validation(path, "expected object")
    return value


def _fields(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    obj = _object(value, path)
    optional = optional or set()
    allowed = required | optional
    unknown = [key for key in obj if key not in allowed]
    if unknown:
        key = str(unknown[0])
        child = f"{path}.{key}" if path else key
        _validation(child, "unknown field")
    missing = sorted(required - set(obj))
    if missing:
        child = f"{path}.{missing[0]}" if path else missing[0]
        _validation(child, "missing required field")
    return obj


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _validation(path, "expected boolean")
    return value


def _int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _validation(path, "expected integer")
    if minimum is not None and value < minimum:
        _validation(path, f"must be >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in (int, float):
        _validation(path, "expected number")
    try:
        result = float(value)
    except OverflowError:
        _validation(path, "number must be finite")
    if not math.isfinite(result):
        _validation(path, "number must be finite")
    if strictly_positive and result <= 0:
        _validation(path, "must be > 0")
    if minimum is not None and result < minimum:
        _validation(path, f"must be >= {minimum:g}")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if type(value) is not str:
        _validation(path, "expected string")
    if value == "":
        _validation(path, "must be non-empty")
    return value


def _enum_string(value: Any, path: str, allowed: set[str]) -> str:
    if type(value) is not str:
        _validation(path, "expected string")
    if value not in allowed:
        _validation(path, f"expected one of {', '.join(sorted(allowed))}")
    return value


def _parse_camera_source(value: Any, path: str) -> RtpJpegSourceConfig | RtspSourceConfig:
    obj = _object(value, path)
    if "type" not in obj:
        _validation(f"{path}.type", "missing required field")
    source_type = _enum_string(obj["type"], f"{path}.type", {"rtp-jpeg", "rtsp"})

    if source_type == "rtp-jpeg":
        obj = _fields(
            obj,
            path,
            required={"type", "bind-address", "port", "buffer-size"},
        )
        port = _int(obj["port"], f"{path}.port", minimum=1)
        if port > 65535:
            _validation(f"{path}.port", "must be <= 65535")
        return RtpJpegSourceConfig(
            bind_address=_nonempty_string(
                obj["bind-address"], f"{path}.bind-address"
            ),
            port=port,
            buffer_size=_int(
                obj["buffer-size"], f"{path}.buffer-size", minimum=1
            ),
        )

    obj = _fields(
        obj,
        path,
        required={
            "type",
            "uri",
            "protocol",
            "decoder-mode",
            "latency-ms",
            "drop-on-latency",
            "buffer-size",
        },
    )
    uri = _nonempty_string(obj["uri"], f"{path}.uri")
    try:
        parsed_uri = urlsplit(uri)
        port = parsed_uri.port
    except ValueError as exc:
        _validation(f"{path}.uri", f"invalid RTSP URI: {exc}")
    if parsed_uri.scheme.lower() != "rtsp" or parsed_uri.hostname is None:
        _validation(f"{path}.uri", "expected rtsp://host[:port]/path URI")
    if port is not None and not 1 <= port <= 65535:
        _validation(f"{path}.uri", "RTSP URI port must be in range 1..65535")
    protocol = _enum_string(
        obj["protocol"],
        f"{path}.protocol",
        {item.value for item in RtspProtocol},
    )
    decoder_mode = _enum_string(
        obj["decoder-mode"],
        f"{path}.decoder-mode",
        {item.value for item in RtspDecoderMode},
    )
    return RtspSourceConfig(
        uri=uri,
        protocol=RtspProtocol(protocol),
        decoder_mode=RtspDecoderMode(decoder_mode),
        latency_ms=_int(obj["latency-ms"], f"{path}.latency-ms", minimum=0),
        drop_on_latency=_bool(
            obj["drop-on-latency"], f"{path}.drop-on-latency"
        ),
        buffer_size=_int(obj["buffer-size"], f"{path}.buffer-size", minimum=1),
    )


def _parse_camera(value: Any, path: str) -> CameraConfig:
    obj = _fields(
        value,
        path,
        required={
            "enabled",
            "source",
            "processing-enabled",
            "vision-processor-class",
        },
    )
    return CameraConfig(
        enabled=_bool(obj["enabled"], f"{path}.enabled"),
        source=_parse_camera_source(obj["source"], f"{path}.source"),
        processing_enabled=_bool(
            obj["processing-enabled"], f"{path}.processing-enabled"
        ),
        vision_processor_class=_nonempty_string(
            obj["vision-processor-class"], f"{path}.vision-processor-class"
        ),
    )


def _parse_vision(value: Any) -> VisionConfig:
    path = "vision"
    obj = _fields(
        value,
        path,
        required={
            "processing-scope",
            "cameras",
            "distance",
            "camera-stale-timeout-ms",
            "simulation-mode",
        },
    )
    scope = _enum_string(
        obj["processing-scope"],
        f"{path}.processing-scope",
        {item.value for item in ProcessingScope},
    )

    cameras_obj = _fields(
        obj["cameras"],
        f"{path}.cameras",
        required={"overview", "stereo-left", "stereo-right"},
    )
    cameras = CamerasConfig(
        overview=_parse_camera(cameras_obj["overview"], f"{path}.cameras.overview"),
        stereo_left=_parse_camera(
            cameras_obj["stereo-left"], f"{path}.cameras.stereo-left"
        ),
        stereo_right=_parse_camera(
            cameras_obj["stereo-right"], f"{path}.cameras.stereo-right"
        ),
    )

    distance_obj = _fields(
        obj["distance"],
        f"{path}.distance",
        required={
            "source",
            "manual-distance-m",
            "distance-stale-timeout-ms",
            "stereo",
        },
    )
    source = _enum_string(
        distance_obj["source"],
        f"{path}.distance.source",
        {item.value for item in DistanceSource},
    )
    stereo_obj = _fields(
        distance_obj["stereo"],
        f"{path}.distance.stereo",
        required={"stereo-enabled", "right-frame-buffer-size", "pair-timeout-ms"},
    )
    distance = VisionDistanceConfig(
        source=DistanceSource(source),
        manual_distance_m=_number(
            distance_obj["manual-distance-m"],
            f"{path}.distance.manual-distance-m",
            strictly_positive=True,
        ),
        distance_stale_timeout_ms=_int(
            distance_obj["distance-stale-timeout-ms"],
            f"{path}.distance.distance-stale-timeout-ms",
            minimum=1,
        ),
        stereo=StereoDistanceConfig(
            stereo_enabled=_bool(
                stereo_obj["stereo-enabled"],
                f"{path}.distance.stereo.stereo-enabled",
            ),
            right_frame_buffer_size=_int(
                stereo_obj["right-frame-buffer-size"],
                f"{path}.distance.stereo.right-frame-buffer-size",
                minimum=1,
            ),
            pair_timeout_ms=_int(
                stereo_obj["pair-timeout-ms"],
                f"{path}.distance.stereo.pair-timeout-ms",
                minimum=1,
            ),
        ),
    )

    return VisionConfig(
        processing_scope=ProcessingScope(scope),
        cameras=cameras,
        distance=distance,
        camera_stale_timeout_ms=_int(
            obj["camera-stale-timeout-ms"],
            f"{path}.camera-stale-timeout-ms",
            minimum=1,
        ),
        simulation_mode=_bool(obj["simulation-mode"], f"{path}.simulation-mode"),
    )


def _parse_aim_point(value: Any, path: str) -> AimPointConfig:
    obj = _fields(value, path, required=set(), optional={"x-px", "y-px"})

    def coordinate(name: str) -> int | None:
        if name not in obj or obj[name] is None:
            return None
        return _int(obj[name], f"{path}.{name}", minimum=0)

    return AimPointConfig(x_px=coordinate("x-px"), y_px=coordinate("y-px"))


def _parse_aiming(value: Any) -> AimingConfig:
    path = "aiming"
    obj = _fields(
        value,
        path,
        required={"lead-time-ms", "target-lost-timeout-ms", "aim-points"},
    )
    aim_points_obj = _fields(
        obj["aim-points"],
        f"{path}.aim-points",
        required={"overview", "stereo-left"},
    )
    return AimingConfig(
        lead_time_ms=_int(obj["lead-time-ms"], f"{path}.lead-time-ms", minimum=0),
        target_lost_timeout_ms=_int(
            obj["target-lost-timeout-ms"],
            f"{path}.target-lost-timeout-ms",
            minimum=1,
        ),
        aim_points=AimPointsConfig(
            overview=_parse_aim_point(
                aim_points_obj["overview"], f"{path}.aim-points.overview"
            ),
            stereo_left=_parse_aim_point(
                aim_points_obj["stereo-left"], f"{path}.aim-points.stereo-left"
            ),
        ),
    )


def _parse_axis(value: Any, path: str) -> AxisMechanicsConfig:
    obj = _fields(
        value,
        path,
        required={
            "invert",
            "full-steps-per-revolution",
            "microstep-divider",
            "max-relative-move-deg",
        },
    )
    return AxisMechanicsConfig(
        invert=_bool(obj["invert"], f"{path}.invert"),
        full_steps_per_revolution=_int(
            obj["full-steps-per-revolution"],
            f"{path}.full-steps-per-revolution",
            minimum=1,
        ),
        microstep_divider=_int(
            obj["microstep-divider"], f"{path}.microstep-divider", minimum=1
        ),
        max_relative_move_deg=_number(
            obj["max-relative-move-deg"],
            f"{path}.max-relative-move-deg",
            strictly_positive=True,
        ),
    )


def _parse_turret(value: Any) -> TurretConfig:
    path = "turret"
    obj = _fields(
        value,
        path,
        required={"serial", "axes", "controller", "stm32", "emulate-stm32"},
    )
    serial_obj = _fields(
        obj["serial"],
        f"{path}.serial",
        required={
            "port",
            "baudrate",
            "response-timeout-ms",
            "max-retries",
            "inter-request-delay-ms",
        },
    )
    baudrate = _int(serial_obj["baudrate"], f"{path}.serial.baudrate")
    if baudrate not in SUPPORTED_BAUDRATES:
        _validation(
            f"{path}.serial.baudrate",
            f"expected one of {', '.join(str(x) for x in sorted(SUPPORTED_BAUDRATES))}",
        )
    serial = SerialConfig(
        port=_nonempty_string(serial_obj["port"], f"{path}.serial.port"),
        baudrate=baudrate,
        response_timeout_ms=_int(
            serial_obj["response-timeout-ms"],
            f"{path}.serial.response-timeout-ms",
            minimum=1,
        ),
        max_retries=_int(
            serial_obj["max-retries"], f"{path}.serial.max-retries", minimum=0
        ),
        inter_request_delay_ms=_int(
            serial_obj["inter-request-delay-ms"],
            f"{path}.serial.inter-request-delay-ms",
            minimum=1,
        ),
    )

    axes_obj = _fields(obj["axes"], f"{path}.axes", required={"x", "y"})
    axes = AxesConfig(
        x=_parse_axis(axes_obj["x"], f"{path}.axes.x"),
        y=_parse_axis(axes_obj["y"], f"{path}.axes.y"),
    )

    controller_obj = _fields(
        obj["controller"],
        f"{path}.controller",
        required={
            "pid-kp-x",
            "pid-ki-x",
            "pid-kd-x",
            "pid-kp-y",
            "pid-ki-y",
            "pid-kd-y",
        },
    )
    controller = PidControllerConfig(
        pid_kp_x=_number(
            controller_obj["pid-kp-x"], f"{path}.controller.pid-kp-x", minimum=0
        ),
        pid_ki_x=_number(
            controller_obj["pid-ki-x"], f"{path}.controller.pid-ki-x", minimum=0
        ),
        pid_kd_x=_number(
            controller_obj["pid-kd-x"], f"{path}.controller.pid-kd-x", minimum=0
        ),
        pid_kp_y=_number(
            controller_obj["pid-kp-y"], f"{path}.controller.pid-kp-y", minimum=0
        ),
        pid_ki_y=_number(
            controller_obj["pid-ki-y"], f"{path}.controller.pid-ki-y", minimum=0
        ),
        pid_kd_y=_number(
            controller_obj["pid-kd-y"], f"{path}.controller.pid-kd-y", minimum=0
        ),
    )

    stm32_obj = _fields(
        obj["stm32"],
        f"{path}.stm32",
        required={
            "max-speed-x-deg-s",
            "max-speed-y-deg-s",
            "acceleration-x-deg-s2",
            "acceleration-y-deg-s2",
            "velocity-watchdog-timeout-ms",
        },
    )
    stm32 = Stm32Config(
        max_speed_x_deg_s=_number(
            stm32_obj["max-speed-x-deg-s"],
            f"{path}.stm32.max-speed-x-deg-s",
            strictly_positive=True,
        ),
        max_speed_y_deg_s=_number(
            stm32_obj["max-speed-y-deg-s"],
            f"{path}.stm32.max-speed-y-deg-s",
            strictly_positive=True,
        ),
        acceleration_x_deg_s2=_number(
            stm32_obj["acceleration-x-deg-s2"],
            f"{path}.stm32.acceleration-x-deg-s2",
            strictly_positive=True,
        ),
        acceleration_y_deg_s2=_number(
            stm32_obj["acceleration-y-deg-s2"],
            f"{path}.stm32.acceleration-y-deg-s2",
            strictly_positive=True,
        ),
        velocity_watchdog_timeout_ms=_int(
            stm32_obj["velocity-watchdog-timeout-ms"],
            f"{path}.stm32.velocity-watchdog-timeout-ms",
            minimum=1,
        ),
    )

    return TurretConfig(
        serial=serial,
        axes=axes,
        controller=controller,
        stm32=stm32,
        emulate_stm32=_bool(obj["emulate-stm32"], f"{path}.emulate-stm32"),
    )


def _parse_ui(value: Any) -> UiConfig:
    path = "ui"
    obj = _fields(
        value,
        path,
        required={"default-camera", "show-fps", "show-stereo-right-diagnostics"},
    )
    default_camera = _enum_string(
        obj["default-camera"],
        f"{path}.default-camera",
        {"overview", "stereo-left"},
    )
    camera_role = (
        CameraRole.OVERVIEW
        if default_camera == "overview"
        else CameraRole.STEREO_LEFT
    )
    return UiConfig(
        default_camera=camera_role,
        show_fps=_bool(obj["show-fps"], f"{path}.show-fps"),
        show_stereo_right_diagnostics=_bool(
            obj["show-stereo-right-diagnostics"],
            f"{path}.show-stereo-right-diagnostics",
        ),
    )


def parse_config(data: Any) -> AppConfig:
    """Validate an already decoded JSON value and return schema-v1 snapshot."""
    obj = _fields(
        data,
        "",
        required={"schema-version", "vision", "aiming", "turret", "ui"},
    )
    schema_version = _int(obj["schema-version"], "schema-version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise UnsupportedConfigSchemaError(schema_version)

    return AppConfig(
        schema_version=schema_version,
        vision=_parse_vision(obj["vision"]),
        aiming=_parse_aiming(obj["aiming"]),
        turret=_parse_turret(obj["turret"]),
        ui=_parse_ui(obj["ui"]),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load and strictly validate config.json without modifying it."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigFileNotFoundError(f"config file not found: {config_path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigJsonError(f"cannot read config JSON: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigJsonError(
            f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return parse_config(data)


def _camera_source_to_mapping(
    source: RtpJpegSourceConfig | RtspSourceConfig,
) -> dict[str, Any]:
    if isinstance(source, RtpJpegSourceConfig):
        return {
            "type": "rtp-jpeg",
            "bind-address": source.bind_address,
            "port": source.port,
            "buffer-size": source.buffer_size,
        }
    if isinstance(source, RtspSourceConfig):
        return {
            "type": "rtsp",
            "uri": source.uri,
            "protocol": source.protocol.value,
            "decoder-mode": source.decoder_mode.value,
            "latency-ms": source.latency_ms,
            "drop-on-latency": source.drop_on_latency,
            "buffer-size": source.buffer_size,
        }
    raise TypeError(f"unsupported camera source config: {type(source).__name__}")


def _camera_to_mapping(config: CameraConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "source": _camera_source_to_mapping(config.source),
        "processing-enabled": config.processing_enabled,
        "vision-processor-class": config.vision_processor_class,
    }


def _aim_point_to_mapping(config: AimPointConfig) -> dict[str, Any]:
    return {"x-px": config.x_px, "y-px": config.y_px}


def config_to_mapping(config: AppConfig) -> dict[str, Any]:
    """Convert a typed schema-v1 snapshot to its persisted JSON structure."""
    return {
        "schema-version": config.schema_version,
        "vision": {
            "processing-scope": config.vision.processing_scope.value,
            "cameras": {
                "overview": _camera_to_mapping(config.vision.cameras.overview),
                "stereo-left": _camera_to_mapping(config.vision.cameras.stereo_left),
                "stereo-right": _camera_to_mapping(config.vision.cameras.stereo_right),
            },
            "distance": {
                "source": config.vision.distance.source.value,
                "manual-distance-m": config.vision.distance.manual_distance_m,
                "distance-stale-timeout-ms": (
                    config.vision.distance.distance_stale_timeout_ms
                ),
                "stereo": {
                    "stereo-enabled": config.vision.distance.stereo.stereo_enabled,
                    "right-frame-buffer-size": (
                        config.vision.distance.stereo.right_frame_buffer_size
                    ),
                    "pair-timeout-ms": config.vision.distance.stereo.pair_timeout_ms,
                },
            },
            "camera-stale-timeout-ms": config.vision.camera_stale_timeout_ms,
            "simulation-mode": config.vision.simulation_mode,
        },
        "aiming": {
            "lead-time-ms": config.aiming.lead_time_ms,
            "target-lost-timeout-ms": config.aiming.target_lost_timeout_ms,
            "aim-points": {
                "overview": _aim_point_to_mapping(config.aiming.aim_points.overview),
                "stereo-left": _aim_point_to_mapping(config.aiming.aim_points.stereo_left),
            },
        },
        "turret": {
            "serial": {
                "port": config.turret.serial.port,
                "baudrate": config.turret.serial.baudrate,
                "response-timeout-ms": config.turret.serial.response_timeout_ms,
                "max-retries": config.turret.serial.max_retries,
                "inter-request-delay-ms": config.turret.serial.inter_request_delay_ms,
            },
            "axes": {
                "x": {
                    "invert": config.turret.axes.x.invert,
                    "full-steps-per-revolution": (
                        config.turret.axes.x.full_steps_per_revolution
                    ),
                    "microstep-divider": config.turret.axes.x.microstep_divider,
                    "max-relative-move-deg": config.turret.axes.x.max_relative_move_deg,
                },
                "y": {
                    "invert": config.turret.axes.y.invert,
                    "full-steps-per-revolution": (
                        config.turret.axes.y.full_steps_per_revolution
                    ),
                    "microstep-divider": config.turret.axes.y.microstep_divider,
                    "max-relative-move-deg": config.turret.axes.y.max_relative_move_deg,
                },
            },
            "controller": {
                "pid-kp-x": config.turret.controller.pid_kp_x,
                "pid-ki-x": config.turret.controller.pid_ki_x,
                "pid-kd-x": config.turret.controller.pid_kd_x,
                "pid-kp-y": config.turret.controller.pid_kp_y,
                "pid-ki-y": config.turret.controller.pid_ki_y,
                "pid-kd-y": config.turret.controller.pid_kd_y,
            },
            "stm32": {
                "max-speed-x-deg-s": config.turret.stm32.max_speed_x_deg_s,
                "max-speed-y-deg-s": config.turret.stm32.max_speed_y_deg_s,
                "acceleration-x-deg-s2": config.turret.stm32.acceleration_x_deg_s2,
                "acceleration-y-deg-s2": config.turret.stm32.acceleration_y_deg_s2,
                "velocity-watchdog-timeout-ms": (
                    config.turret.stm32.velocity_watchdog_timeout_ms
                ),
            },
            "emulate-stm32": config.turret.emulate_stm32,
        },
        "ui": {
            "default-camera": (
                "overview"
                if config.ui.default_camera is CameraRole.OVERVIEW
                else "stereo-left"
            ),
            "show-fps": config.ui.show_fps,
            "show-stereo-right-diagnostics": config.ui.show_stereo_right_diagnostics,
        },
    }


__all__ = [
    "SUPPORTED_BAUDRATES",
    "SUPPORTED_SCHEMA_VERSION",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigJsonError",
    "ConfigPersistenceError",
    "ConfigValidationError",
    "UnsupportedConfigSchemaError",
    "config_to_mapping",
    "load_config",
    "parse_config",
]
