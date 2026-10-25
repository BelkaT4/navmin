from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from navmin.config import (
    ConfigApplyPolicy,
    ConfigFileNotFoundError,
    ConfigJsonError,
    ConfigManager,
    ConfigValidationError,
    UnsupportedConfigSchemaError,
    changed_config_paths,
    config_apply_policy,
    config_to_mapping,
    load_config,
    parse_config,
    save_config,
)
from navmin.contracts import CameraRole, DistanceSource


def _valid_config() -> dict:
    return {
        "schema-version": 3,
        "vision": {
            "processing-scope": "main-only",
            "cameras": {
                "overview": {
                    "enabled": True,
                    "source": {
                        "type": "rtp-jpeg",
                        "bind-address": "192.168.1.101",
                        "port": 5001,
                        "buffer-size": 1,
                    },
                    "processing-enabled": True,
                    "vision-processor-class": "DefaultVisionProcessor",
                    "calibration-file": "calibration/overview.json",
                },
                "stereo-left": {
                    "enabled": True,
                    "source": {
                        "type": "rtp-jpeg",
                        "bind-address": "192.168.1.102",
                        "port": 5002,
                        "buffer-size": 1,
                    },
                    "processing-enabled": True,
                    "vision-processor-class": "DefaultVisionProcessor",
                },
                "stereo-right": {
                    "enabled": True,
                    "source": {
                        "type": "rtp-jpeg",
                        "bind-address": "192.168.1.103",
                        "port": 5003,
                        "buffer-size": 1,
                    },
                    "processing-enabled": False,
                    "vision-processor-class": "DefaultVisionProcessor",
                },
            },
            "distance": {
                "source": "manual",
                "manual-distance-m": 100.0,
                "distance-stale-timeout-ms": 300,
                "stereo": {
                    "stereo-enabled": False,
                    "right-frame-buffer-size": 4,
                    "pair-timeout-ms": 100,
                    "calibration-file": "calibration/stereo.json",
                },
            },
            "camera-stale-timeout-ms": 500,
            "simulation-mode": False,
        },
        "aiming": {
            "target-lost-timeout-ms": 500,
            "aim-points": {
                "overview": {"x-px": None, "y-px": None},
                "stereo-left": {"x-px": None, "y-px": None},
            },
        },
        "tracking-control": {
            "velocity-estimator": {
                "window-ms": 200,
                "min-span-ms": 100,
                "min-samples": 4,
                "max-samples": 16,
                "max-fit-rms-deg": 1.0,
                "max-target-rate-x-deg-s": 90.0,
                "max-target-rate-y-deg-s": 90.0,
            },
            "feed-forward": {
                "enabled": False,
                "gain-x": 0.0,
                "gain-y": 0.0,
            },
            "prediction": {
                "enabled": False,
                "horizon-ms": 0,
                "max-angle-x-deg": 1.0,
                "max-angle-y-deg": 1.0,
            },
            "platform-motion": {"source": "command-model"},
        },
        "turret": {
            "serial": {
                "port": "/dev/ttyUSB0",
                "baudrate": 115200,
                "response-timeout-ms": 100,
                "max-retries": 2,
                "inter-request-delay-ms": 2,
            },
            "axes": {
                "x": {
                    "invert": False,
                    "full-steps-per-revolution": 2000,
                    "microstep-divider": 16,
                    "max-relative-move-deg": 45.0,
                },
                "y": {
                    "invert": False,
                    "full-steps-per-revolution": 2000,
                    "microstep-divider": 16,
                    "max-relative-move-deg": 45.0,
                },
            },
            "controller": {
                "pid-kp-x": 1.0,
                "pid-ki-x": 0.0,
                "pid-kd-x": 0.0,
                "pid-kp-y": 1.0,
                "pid-ki-y": 0.0,
                "pid-kd-y": 0.0,
            },
            "stm32": {
                "max-speed-x-deg-s": 50.0,
                "max-speed-y-deg-s": 50.0,
                "acceleration-x-deg-s2": 100.0,
                "acceleration-y-deg-s2": 100.0,
                "velocity-watchdog-timeout-ms": 200,
            },
            "emulate-stm32": False,
        },
        "ui": {
            "default-camera": "overview",
            "show-fps": True,
            "show-stereo-right-diagnostics": False,
        },
    }


def _write_json(path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_schema_v3_builds_typed_immutable_snapshot() -> None:
    config = parse_config(_valid_config())

    assert config.schema_version == 3
    assert (
        config.vision.cameras.overview.calibration_file == "calibration/overview.json"
    )
    assert not hasattr(config.vision.cameras.stereo_left, "calibration_file")
    assert not hasattr(config.vision.cameras.stereo_right, "calibration_file")
    assert config.vision.distance.source is DistanceSource.MANUAL
    assert config.ui.default_camera is CameraRole.OVERVIEW
    assert config.turret.serial.baudrate == 115200

    with pytest.raises(FrozenInstanceError):
        config.tracking_control.velocity_estimator.window_ms = 10  # type: ignore[misc]


def test_tracking_control_round_trips_through_mapping() -> None:
    config = parse_config(_valid_config())

    assert parse_config(config_to_mapping(config)) == config


def test_schema_v3_requires_tracking_control_and_rejects_old_pixel_lead() -> None:
    missing = _valid_config()
    missing.pop("tracking-control")
    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(missing)
    assert exc_info.value.path == "tracking-control"

    old_lead = _valid_config()
    old_lead["aiming"]["lead-time-ms"] = 100
    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(old_lead)
    assert exc_info.value.path == "aiming.lead-time-ms"
    assert exc_info.value.reason == "unknown field"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("velocity-estimator", "window-ms"), 0),
        (("velocity-estimator", "min-span-ms"), 0),
        (("velocity-estimator", "min-samples"), 1),
        (("velocity-estimator", "max-samples"), 1),
        (("velocity-estimator", "max-fit-rms-deg"), 0),
        (("velocity-estimator", "max-target-rate-x-deg-s"), 0),
        (("velocity-estimator", "max-target-rate-y-deg-s"), 0),
        (("feed-forward", "enabled"), 1),
        (("feed-forward", "gain-x"), -0.1),
        (("feed-forward", "gain-y"), -0.1),
        (("prediction", "enabled"), 1),
        (("prediction", "horizon-ms"), -1),
        (("prediction", "max-angle-x-deg"), 0),
        (("prediction", "max-angle-y-deg"), 0),
        (("platform-motion", "source"), "gyro"),
    ],
)
def test_tracking_control_field_validation(path, value) -> None:
    data = _valid_config()
    section, field = path
    data["tracking-control"][section][field] = value

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == f"tracking-control.{section}.{field}"


def test_tracking_control_cross_field_validation() -> None:
    min_span = _valid_config()
    min_span["tracking-control"]["velocity-estimator"]["min-span-ms"] = 201
    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(min_span)
    assert exc_info.value.path == "tracking-control.velocity-estimator.min-span-ms"

    max_samples = _valid_config()
    max_samples["tracking-control"]["velocity-estimator"]["max-samples"] = 3
    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(max_samples)
    assert exc_info.value.path == "tracking-control.velocity-estimator.max-samples"


def test_tracking_control_unknown_field_is_rejected() -> None:
    data = _valid_config()
    data["tracking-control"]["prediction"]["surprise"] = 1

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == "tracking-control.prediction.surprise"


def test_rtsp_camera_source_is_strict_typed_union() -> None:
    data = _valid_config()
    data["vision"]["cameras"]["overview"]["source"] = {
        "type": "rtsp",
        "uri": "rtsp://camera.local:8554/stream",
        "protocol": "tcp",
        "decoder-mode": "software",
        "latency-ms": 100,
        "drop-on-latency": True,
        "buffer-size": 1,
    }

    config = parse_config(data)
    source = config.vision.cameras.overview.source

    assert source.uri == "rtsp://camera.local:8554/stream"
    assert source.protocol.value == "tcp"
    assert source.decoder_mode.value == "software"
    assert source.latency_ms == 100
    assert source.drop_on_latency is True


@pytest.mark.parametrize("protocol", ["tcp", "udp"])
def test_rtsp_protocol_tcp_and_udp_are_supported(protocol: str) -> None:
    data = _valid_config()
    data["vision"]["cameras"]["overview"]["source"] = {
        "type": "rtsp",
        "uri": "rtsp://camera.local/stream",
        "protocol": protocol,
        "decoder-mode": "software",
        "latency-ms": 0,
        "drop-on-latency": False,
        "buffer-size": 2,
    }

    source = parse_config(data).vision.cameras.overview.source

    assert source.protocol.value == protocol


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda d: d["vision"]["cameras"]["overview"]["source"].__setitem__(
                "type", "gstreamer"
            ),
            "vision.cameras.overview.source.type",
        ),
        (
            lambda d: d["vision"]["cameras"]["overview"]["source"].__setitem__(
                "uri", "rtsp://not-valid-for-rtp"
            ),
            "vision.cameras.overview.source.uri",
        ),
    ],
)
def test_camera_source_invalid_type_or_mixed_fields_are_rejected(
    mutate, expected_path
) -> None:
    data = _valid_config()
    mutate(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == expected_path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decoder-mode", "hardware"),
        ("protocol", "auto"),
        ("codec", "h265"),
    ],
)
def test_rtsp_unsupported_modes_and_codec_field_are_rejected(field, value) -> None:
    data = _valid_config()
    source = {
        "type": "rtsp",
        "uri": "rtsp://camera.local/stream",
        "protocol": "tcp",
        "decoder-mode": "software",
        "latency-ms": 100,
        "drop-on-latency": True,
        "buffer-size": 1,
    }
    source[field] = value
    data["vision"]["cameras"]["overview"]["source"] = source

    with pytest.raises(ConfigValidationError):
        parse_config(data)


def test_legacy_flat_camera_schema_is_rejected_without_migration() -> None:
    data = _valid_config()
    data["vision"]["cameras"]["overview"] = {
        "enabled": True,
        "address": "0.0.0.0",
        "port": 8888,
        "rtp-enabled": True,
        "buffer-size": 1,
        "processing-enabled": True,
        "vision-processor-class": "Legacy14VisionProcessor",
    }

    with pytest.raises(ConfigValidationError):
        parse_config(data)


@pytest.mark.parametrize(
    ("field", "value", "expected_path"),
    [
        ("uri", "http://camera.local/stream", "vision.cameras.overview.source.uri"),
        ("latency-ms", -1, "vision.cameras.overview.source.latency-ms"),
        (
            "drop-on-latency",
            1,
            "vision.cameras.overview.source.drop-on-latency",
        ),
    ],
)
def test_rtsp_invalid_uri_latency_or_boolean_is_rejected(
    field, value, expected_path
) -> None:
    data = _valid_config()
    source = {
        "type": "rtsp",
        "uri": "rtsp://camera.local/stream",
        "protocol": "tcp",
        "decoder-mode": "software",
        "latency-ms": 100,
        "drop-on-latency": True,
        "buffer-size": 1,
    }
    source[field] = value
    data["vision"]["cameras"]["overview"]["source"] = source

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == expected_path


def test_missing_config_is_explicit_error(tmp_path) -> None:
    with pytest.raises(ConfigFileNotFoundError):
        load_config(tmp_path / "config.json")


def test_malformed_json_is_preserved(tmp_path) -> None:
    path = tmp_path / "config.json"
    original = b'{"schema-version": 3, broken}'
    path.write_bytes(original)

    with pytest.raises(ConfigJsonError):
        load_config(path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda data: data.pop("schema-version"), "schema-version"),
        (lambda data: data.__setitem__("schema-version", True), "schema-version"),
        (lambda data: data.__setitem__("schema-version", "3"), "schema-version"),
    ],
)
def test_missing_or_invalid_schema_version_is_validation_error(mutation, path) -> None:
    data = _valid_config()
    mutation(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == path
    assert not isinstance(exc_info.value, UnsupportedConfigSchemaError)


@pytest.mark.parametrize("unsupported", [1, 2, 4])
def test_unsupported_schema_is_distinct_error(unsupported: int) -> None:
    data = _valid_config()
    data["schema-version"] = unsupported

    with pytest.raises(UnsupportedConfigSchemaError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == "schema-version"
    assert exc_info.value.actual == unsupported
    assert "expected 3" in str(exc_info.value)


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda d: d.__setitem__("surprise", 1), "surprise"),
        (
            lambda d: d["vision"].__setitem__("surprise", 1),
            "vision.surprise",
        ),
        (
            lambda d: d["vision"]["cameras"]["overview"].__setitem__("surprise", 1),
            "vision.cameras.overview.surprise",
        ),
        (
            lambda d: d["aiming"]["aim-points"]["overview"].__setitem__("surprise", 1),
            "aiming.aim-points.overview.surprise",
        ),
    ],
)
def test_unknown_fields_are_rejected_at_multiple_levels(mutate, expected_path) -> None:
    data = _valid_config()
    mutate(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == expected_path


def test_missing_required_field_reports_exact_path() -> None:
    data = _valid_config()
    del data["turret"]["serial"]["response-timeout-ms"]

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == "turret.serial.response-timeout-ms"
    assert "missing required" in exc_info.value.reason


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda d: d["vision"]["cameras"]["overview"].pop("calibration-file"),
            "vision.cameras.overview.calibration-file",
        ),
        (
            lambda d: d["vision"]["cameras"]["overview"].__setitem__(
                "calibration-file", ""
            ),
            "vision.cameras.overview.calibration-file",
        ),
        (
            lambda d: d["vision"]["distance"]["stereo"].pop("calibration-file"),
            "vision.distance.stereo.calibration-file",
        ),
        (
            lambda d: d["vision"]["distance"]["stereo"].__setitem__(
                "calibration-file", ""
            ),
            "vision.distance.stereo.calibration-file",
        ),
        (
            lambda d: d["vision"]["cameras"]["stereo-left"].__setitem__(
                "calibration-file", "calibration/left.json"
            ),
            "vision.cameras.stereo-left.calibration-file",
        ),
        (
            lambda d: d["vision"]["cameras"]["stereo-right"].__setitem__(
                "calibration-file", "calibration/right.json"
            ),
            "vision.cameras.stereo-right.calibration-file",
        ),
    ],
)
def test_schema_v3_calibration_references_are_required_and_role_owned(
    mutate, expected_path
) -> None:
    data = _valid_config()
    mutate(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == expected_path


def test_config_to_mapping_round_trips_calibration_references() -> None:
    config = parse_config(_valid_config())
    mapped = config_to_mapping(config)

    assert (
        mapped["vision"]["cameras"]["overview"]["calibration-file"]
        == "calibration/overview.json"
    )
    assert (
        mapped["vision"]["distance"]["stereo"]["calibration-file"]
        == "calibration/stereo.json"
    )
    assert "calibration-file" not in mapped["vision"]["cameras"]["stereo-left"]
    assert "calibration-file" not in mapped["vision"]["cameras"]["stereo-right"]


def test_repository_defaults_parse_with_production_loaders() -> None:
    from pathlib import Path

    from navmin.calibration import load_overview_calibration, load_stereo_calibration

    root = Path(__file__).resolve().parents[1]
    defaults = root / "config" / "defaults"

    config = load_config(defaults / "navmin.json")
    overview = load_overview_calibration(defaults / "calibration" / "overview.json")
    stereo = load_stereo_calibration(defaults / "calibration" / "stereo.json")

    assert config.schema_version == 3
    assert (
        config.vision.cameras.overview.calibration_file == "calibration/overview.json"
    )
    assert config.vision.distance.stereo.calibration_file == "calibration/stereo.json"
    assert not config.tracking_control.feed_forward.enabled
    assert config.tracking_control.feed_forward.gain_x == 0.0
    assert config.tracking_control.feed_forward.gain_y == 0.0
    assert not config.tracking_control.prediction.enabled
    assert config.tracking_control.prediction.horizon_ms == 0
    assert overview.schema_version == 1
    assert stereo.schema_version == 1


def test_optional_aim_point_missing_and_null_both_mean_center() -> None:
    missing = _valid_config()
    del missing["aiming"]["aim-points"]["overview"]["x-px"]
    del missing["aiming"]["aim-points"]["overview"]["y-px"]
    explicit_null = _valid_config()

    missing_config = parse_config(missing)
    null_config = parse_config(explicit_null)

    assert missing_config.aiming.aim_points.overview.x_px is None
    assert missing_config.aiming.aim_points.overview.y_px is None
    assert null_config.aiming.aim_points.overview.x_px is None
    assert null_config.aiming.aim_points.overview.y_px is None


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda d: d["vision"]["cameras"]["overview"]["source"].__setitem__(
                "port", True
            ),
            "vision.cameras.overview.source.port",
        ),
        (
            lambda d: d["turret"]["controller"].__setitem__("pid-kp-x", True),
            "turret.controller.pid-kp-x",
        ),
        (
            lambda d: d["aiming"]["aim-points"]["overview"].__setitem__("x-px", True),
            "aiming.aim-points.overview.x-px",
        ),
    ],
)
def test_bool_is_not_accepted_as_integer_or_number(mutate, expected_path) -> None:
    data = _valid_config()
    mutate(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == expected_path


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected_with_field_path(invalid) -> None:
    data = _valid_config()
    data["vision"]["distance"]["manual-distance-m"] = invalid

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == "vision.distance.manual-distance-m"
    assert "finite" in exc_info.value.reason


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda d: d["vision"].__setitem__("processing-scope", "everything"),
            "vision.processing-scope",
        ),
        (
            lambda d: d["vision"]["cameras"]["overview"]["source"].__setitem__(
                "port", 0
            ),
            "vision.cameras.overview.source.port",
        ),
        (
            lambda d: d["turret"]["serial"].__setitem__("baudrate", 230400),
            "turret.serial.baudrate",
        ),
        (
            lambda d: d["ui"].__setitem__("default-camera", "stereo-right"),
            "ui.default-camera",
        ),
    ],
)
def test_invalid_enum_or_range_is_rejected(mutate, expected_path) -> None:
    data = _valid_config()
    mutate(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == expected_path


def test_watchdog_less_than_target_lost_is_not_an_undocumented_hard_rule() -> None:
    data = _valid_config()
    data["turret"]["stm32"]["velocity-watchdog-timeout-ms"] = 900
    data["aiming"]["target-lost-timeout-ms"] = 500

    config = parse_config(data)

    assert config.turret.stm32.velocity_watchdog_timeout_ms == 900


def test_runtime_invalid_update_does_not_replace_snapshot_or_revision(tmp_path) -> None:
    path = tmp_path / "config.json"
    initial_data = _valid_config()
    _write_json(path, initial_data)
    manager = ConfigManager.load(path)
    before = manager.config
    before_vision_update = manager.vision_updates.get()

    invalid = copy.deepcopy(initial_data)
    invalid["vision"]["distance"]["manual-distance-m"] = -1
    with pytest.raises(ConfigValidationError):
        manager.apply_runtime_update(invalid, persist=False)

    assert manager.config == before
    assert manager.revision == 0
    assert manager.vision_updates.get() == before_vision_update


def test_revision_advances_only_for_actual_accepted_changes(tmp_path) -> None:
    path = tmp_path / "config.json"
    initial_data = _valid_config()
    _write_json(path, initial_data)
    manager = ConfigManager.load(path)

    assert manager.revision == 0
    assert manager.vision_updates.get().revision == 0  # type: ignore[union-attr]
    assert manager.aiming_updates.get().revision == 0  # type: ignore[union-attr]

    changed = copy.deepcopy(initial_data)
    changed["aiming"]["target-lost-timeout-ms"] = 600
    assert manager.apply_runtime_update(changed, persist=False)
    assert manager.revision == 1
    assert manager.aiming_updates.get().revision == 1  # type: ignore[union-attr]
    assert manager.vision_updates.get().revision == 0  # type: ignore[union-attr]

    assert not manager.apply_runtime_update(changed, persist=False)
    assert manager.revision == 1


def test_config_manager_publishes_but_does_not_own_restart_reconnect_actions(
    tmp_path,
) -> None:
    path = tmp_path / "config.json"
    data = _valid_config()
    _write_json(path, data)
    manager = ConfigManager.load(path)

    changed = copy.deepcopy(data)
    changed["turret"]["serial"]["port"] = "/dev/ttyUSB1"
    assert manager.apply_runtime_update(changed, persist=False)
    update = manager.turret_updates.get()

    assert update is not None
    assert update.revision == 1
    assert update.config.serial.port == "/dev/ttyUSB1"
    assert not hasattr(manager, "requires_restart")
    assert not hasattr(manager, "restart")
    assert not hasattr(manager, "reconnect")


def test_atomic_save_round_trips_in_temp_directory(tmp_path) -> None:
    path = tmp_path / "config.json"
    config = parse_config(_valid_config())

    save_config(path, config)

    assert load_config(path) == config
    assert not list(tmp_path.glob(".config.json.*.tmp"))


def test_rtsp_config_round_trip_preserves_source_specific_fields(tmp_path) -> None:
    path = tmp_path / "config.json"
    data = _valid_config()
    data["vision"]["cameras"]["overview"]["source"] = {
        "type": "rtsp",
        "uri": "rtsp://camera.local:8554/stream",
        "protocol": "udp",
        "decoder-mode": "software",
        "latency-ms": 75,
        "drop-on-latency": False,
        "buffer-size": 2,
    }
    config = parse_config(data)

    save_config(path, config)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema-version"] == 3
    assert (
        saved["vision"]["cameras"]["overview"]["source"]
        == data["vision"]["cameras"]["overview"]["source"]
    )
    assert load_config(path) == config


def test_loading_valid_config_does_not_rewrite_optional_fields(tmp_path) -> None:
    path = tmp_path / "config.json"
    data = _valid_config()
    del data["aiming"]["aim-points"]["overview"]["x-px"]
    original = json.dumps(data, indent=3).encode()
    path.write_bytes(original)

    loaded = load_config(path)

    assert loaded.aiming.aim_points.overview.x_px is None
    assert path.read_bytes() == original


def test_change_metadata_reports_only_fixed_architecture_classification() -> None:
    old = parse_config(_valid_config())
    data = _valid_config()
    data["aiming"]["target-lost-timeout-ms"] = 600
    data["vision"]["cameras"]["overview"]["vision-processor-class"] = "Other"
    data["vision"]["cameras"]["overview"]["calibration-file"] = (
        "calibration/overview-b.json"
    )
    data["vision"]["distance"]["stereo"]["calibration-file"] = (
        "calibration/stereo-b.json"
    )
    data["turret"]["serial"]["port"] = "/dev/ttyUSB1"
    data["turret"]["serial"]["baudrate"] = 57600
    data["turret"]["serial"]["response-timeout-ms"] = 150
    data["turret"]["serial"]["max-retries"] = 3
    data["turret"]["serial"]["inter-request-delay-ms"] = 3
    data["turret"]["emulate-stm32"] = True
    data["turret"]["axes"]["x"]["microstep-divider"] = 8
    data["turret"]["axes"]["x"]["max-relative-move-deg"] = 60.0
    data["turret"]["axes"]["y"]["max-relative-move-deg"] = 75.0
    data["vision"]["camera-stale-timeout-ms"] = 700
    new = parse_config(data)

    paths = changed_config_paths(old, new)

    assert "aiming.target-lost-timeout-ms" in paths
    assert "turret.axes.x.max-relative-move-deg" in paths
    assert "turret.axes.y.max-relative-move-deg" in paths
    assert "vision.cameras.overview.calibration-file" in paths
    assert "vision.distance.stereo.calibration-file" in paths
    assert (
        config_apply_policy("aiming.target-lost-timeout-ms")
        is ConfigApplyPolicy.DYNAMIC
    )
    assert (
        config_apply_policy("vision.cameras.overview.vision-processor-class")
        is ConfigApplyPolicy.CAMERA_PIPELINE_RESTART
    )
    assert (
        config_apply_policy("vision.cameras.overview.calibration-file")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )
    assert (
        config_apply_policy("vision.distance.stereo.calibration-file")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )
    assert (
        config_apply_policy("vision.cameras.overview.source.type")
        is ConfigApplyPolicy.CAMERA_PIPELINE_RESTART
    )
    assert (
        config_apply_policy("vision.cameras.overview.source.port")
        is ConfigApplyPolicy.CAMERA_PIPELINE_RESTART
    )
    assert (
        config_apply_policy("turret.serial.port") is ConfigApplyPolicy.TURRET_RECONNECT
    )
    assert (
        config_apply_policy("turret.serial.baudrate")
        is ConfigApplyPolicy.CONTROLLED_SERIAL_TRANSITION
    )
    for path in (
        "turret.serial.response-timeout-ms",
        "turret.serial.max-retries",
        "turret.serial.inter-request-delay-ms",
    ):
        assert config_apply_policy(path) is ConfigApplyPolicy.TURRET_RECONNECT
    assert (
        config_apply_policy("turret.emulate-stm32")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )
    assert (
        config_apply_policy("turret.axes.x.microstep-divider")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )
    assert (
        config_apply_policy("turret.axes.x.max-relative-move-deg")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )
    assert (
        config_apply_policy("turret.axes.y.max-relative-move-deg")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )
    assert config_apply_policy("vision.camera-stale-timeout-ms") is None


@pytest.mark.parametrize(
    "path",
    [
        "tracking-control.velocity-estimator.window-ms",
        "tracking-control.velocity-estimator.min-span-ms",
        "tracking-control.velocity-estimator.min-samples",
        "tracking-control.velocity-estimator.max-samples",
        "tracking-control.velocity-estimator.max-fit-rms-deg",
        "tracking-control.velocity-estimator.max-target-rate-x-deg-s",
        "tracking-control.velocity-estimator.max-target-rate-y-deg-s",
        "tracking-control.feed-forward.enabled",
        "tracking-control.feed-forward.gain-x",
        "tracking-control.feed-forward.gain-y",
        "tracking-control.prediction.enabled",
        "tracking-control.prediction.horizon-ms",
        "tracking-control.prediction.max-angle-x-deg",
        "tracking-control.prediction.max-angle-y-deg",
    ],
)
def test_tracking_control_runtime_fields_are_dynamic(path: str) -> None:
    assert config_apply_policy(path) is ConfigApplyPolicy.DYNAMIC


def test_tracking_control_unknown_fields_have_no_implicit_apply_policy() -> None:
    assert config_apply_policy("tracking-control.prediction.future-field") is None
    assert config_apply_policy("tracking-control.feed-forward.future-field") is None
    assert (
        config_apply_policy("tracking-control.velocity-estimator.future-field")
        is None
    )


def test_platform_motion_source_requires_application_restart() -> None:
    assert (
        config_apply_policy("tracking-control.platform-motion.source")
        is ConfigApplyPolicy.APPLICATION_RESTART
    )


def test_persistence_failure_does_not_publish_partial_runtime_update(
    tmp_path, monkeypatch
) -> None:
    from navmin.config.parser import ConfigPersistenceError

    path = tmp_path / "config.json"
    data = _valid_config()
    _write_json(path, data)
    manager = ConfigManager.load(path)
    before = manager.config
    before_update = manager.aiming_updates.get()

    changed = copy.deepcopy(data)
    changed["aiming"]["target-lost-timeout-ms"] = 600

    def fail_save(*args, **kwargs) -> None:
        raise ConfigPersistenceError("simulated persistence failure")

    monkeypatch.setattr("navmin.config.manager.save_config", fail_save)

    with pytest.raises(ConfigPersistenceError):
        manager.apply_runtime_update(changed, persist=True)

    assert manager.config == before
    assert manager.revision == 0
    assert manager.aiming_updates.get() == before_update


def test_atomic_replace_failure_preserves_existing_file(tmp_path, monkeypatch) -> None:
    from navmin.config.parser import ConfigPersistenceError

    path = tmp_path / "config.json"
    original = b"existing config bytes\n"
    path.write_bytes(original)
    config = parse_config(_valid_config())

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("navmin.config.manager.os.replace", fail_replace)

    with pytest.raises(ConfigPersistenceError):
        save_config(path, config)

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".config.json.*.tmp"))


def test_relative_motion_curve_roundtrip_and_dynamic_policy() -> None:
    data = _valid_config()
    points = [(-7, -10), (-4, -5), (0, 0), (3, 5), (6, 10)]
    data['turret']['axes']['x']['relative-motion-curve'] = [
        {'measured-deg': measured, 'command-deg': command} for measured, command in points
    ]
    parsed = parse_config(data)
    assert parsed.turret.axes.x.relative_motion_curve == tuple(points)
    assert parse_config(config_to_mapping(parsed)) == parsed
    assert config_apply_policy('turret.axes.x.relative-motion-curve') is ConfigApplyPolicy.DYNAMIC
    assert changed_config_paths(parse_config(_valid_config()), parsed) == ('turret.axes.x.relative-motion-curve',)


@pytest.mark.parametrize('points', [
    [(0, 0), (1, 1)],
    [(-1, -1), (0, 0), (1, -1)],
    [(-1, -1), (0, 0), (1, 2), (2, 1)],
    [(-1, -1), (0, 0), (1, 46)],
    [(-1, -1), (0, 0), (float('nan'), 1)],
    [(-1, -1), (0, 0), (1, 1), (1, 2)],
])
def test_relative_motion_curve_rejects_invalid_mapping(points) -> None:
    data = _valid_config()
    data['turret']['axes']['x']['relative-motion-curve'] = [
        {'measured-deg': measured, 'command-deg': command} for measured, command in points
    ]
    with pytest.raises(ConfigValidationError):
        parse_config(data)
