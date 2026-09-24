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
    load_config,
    parse_config,
    save_config,
)
from navmin.contracts import CameraRole, DistanceSource


def _valid_config() -> dict:
    return {
        "schema-version": 1,
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
                },
            },
            "camera-stale-timeout-ms": 500,
            "simulation-mode": False,
        },
        "aiming": {
            "lead-time-ms": 150,
            "target-lost-timeout-ms": 500,
            "aim-points": {
                "overview": {"x-px": None, "y-px": None},
                "stereo-left": {"x-px": None, "y-px": None},
            },
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


def test_valid_schema_v1_builds_typed_immutable_snapshot() -> None:
    config = parse_config(_valid_config())

    assert config.schema_version == 1
    assert config.vision.distance.source is DistanceSource.MANUAL
    assert config.ui.default_camera is CameraRole.OVERVIEW
    assert config.turret.serial.baudrate == 115200

    with pytest.raises(FrozenInstanceError):
        config.aiming.lead_time_ms = 10  # type: ignore[misc]



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
    original = b'{"schema-version": 1, broken}'
    path.write_bytes(original)

    with pytest.raises(ConfigJsonError):
        load_config(path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda data: data.pop("schema-version"), "schema-version"),
        (lambda data: data.__setitem__("schema-version", True), "schema-version"),
        (lambda data: data.__setitem__("schema-version", "1"), "schema-version"),
    ],
)
def test_missing_or_invalid_schema_version_is_validation_error(mutation, path) -> None:
    data = _valid_config()
    mutation(data)

    with pytest.raises(ConfigValidationError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == path
    assert not isinstance(exc_info.value, UnsupportedConfigSchemaError)


def test_unsupported_schema_is_distinct_error() -> None:
    data = _valid_config()
    data["schema-version"] = 2

    with pytest.raises(UnsupportedConfigSchemaError) as exc_info:
        parse_config(data)

    assert exc_info.value.path == "schema-version"
    assert exc_info.value.actual == 2


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (lambda d: d.__setitem__("surprise", 1), "surprise"),
        (
            lambda d: d["vision"].__setitem__("surprise", 1),
            "vision.surprise",
        ),
        (
            lambda d: d["vision"]["cameras"]["overview"].__setitem__(
                "surprise", 1
            ),
            "vision.cameras.overview.surprise",
        ),
        (
            lambda d: d["aiming"]["aim-points"]["overview"].__setitem__(
                "surprise", 1
            ),
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
            lambda d: d["vision"]["cameras"]["overview"]["source"].__setitem__("port", True),
            "vision.cameras.overview.source.port",
        ),
        (
            lambda d: d["turret"]["controller"].__setitem__("pid-kp-x", True),
            "turret.controller.pid-kp-x",
        ),
        (
            lambda d: d["aiming"]["aim-points"]["overview"].__setitem__(
                "x-px", True
            ),
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
            lambda d: d["vision"]["cameras"]["overview"]["source"].__setitem__("port", 0),
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
    changed["aiming"]["lead-time-ms"] = 200
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
    assert saved["schema-version"] == 1
    assert saved["vision"]["cameras"]["overview"]["source"] == data["vision"]["cameras"]["overview"]["source"]
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
    data["aiming"]["lead-time-ms"] = 250
    data["vision"]["cameras"]["overview"]["vision-processor-class"] = "Other"
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

    assert "aiming.lead-time-ms" in paths
    assert "turret.axes.x.max-relative-move-deg" in paths
    assert "turret.axes.y.max-relative-move-deg" in paths
    assert (
        config_apply_policy("aiming.lead-time-ms")
        is ConfigApplyPolicy.DYNAMIC
    )
    assert (
        config_apply_policy("vision.cameras.overview.vision-processor-class")
        is ConfigApplyPolicy.CAMERA_PIPELINE_RESTART
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
        config_apply_policy("turret.serial.port")
        is ConfigApplyPolicy.TURRET_RECONNECT
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
    changed["aiming"]["lead-time-ms"] = 250

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
