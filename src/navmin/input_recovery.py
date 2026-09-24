"""Explicit operator-approved recovery for local config and calibration inputs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from navmin.calibration import load_overview_calibration, load_stereo_calibration
from navmin.config import load_config
from navmin.launcher import LauncherPaths

_DEFAULT_WIDTH = 320
_DEFAULT_HEIGHT = 240


class InputRecoveryError(RuntimeError):
    """Raised when a requested recovery cannot be completed safely."""


@dataclass(frozen=True)
class InputRecoveryResult:
    timestamp: str
    restored_paths: tuple[Path, ...]
    backups: tuple[tuple[Path, Path], ...]


def _default_config_mapping() -> dict[str, Any]:
    camera_common = {
        "enabled": True,
        "source": {
            "type": "rtp-jpeg",
            "bind-address": "0.0.0.0",
            "buffer-size": 1,
        },
        "processing-enabled": True,
        "vision-processor-class": "Legacy14VisionProcessor",
    }
    return {
        "schema-version": 1,
        "vision": {
            "processing-scope": "main-and-preview",
            "cameras": {
                "overview": {
                    **camera_common,
                    "source": {**camera_common["source"], "port": 8888},
                },
                "stereo-left": {
                    **camera_common,
                    "source": {**camera_common["source"], "port": 8889},
                },
                "stereo-right": {
                    **camera_common,
                    "enabled": False,
                    "processing-enabled": False,
                    "source": {**camera_common["source"], "port": 8890},
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
            "lead-time-ms": 0,
            "target-lost-timeout-ms": 500,
            "aim-points": {
                "overview": {"x-px": None, "y-px": None},
                "stereo-left": {"x-px": None, "y-px": None},
            },
        },
        "turret": {
            "serial": {
                # Deliberately nonexistent: restored defaults must not pass real
                # hardware preflight until an operator edits the local config.
                "port": "/dev/navmin-configure-serial-port",
                "baudrate": 9600,
                "response-timeout-ms": 100,
                "max-retries": 2,
                "inter-request-delay-ms": 2,
            },
            "axes": {
                "x": {
                    "invert": False,
                    "full-steps-per-revolution": 200,
                    "microstep-divider": 1,
                    "max-relative-move-deg": 1.0,
                },
                "y": {
                    "invert": False,
                    "full-steps-per-revolution": 200,
                    "microstep-divider": 1,
                    "max-relative-move-deg": 1.0,
                },
            },
            "controller": {
                "pid-kp-x": 0.0,
                "pid-ki-x": 0.0,
                "pid-kd-x": 0.0,
                "pid-kp-y": 0.0,
                "pid-ki-y": 0.0,
                "pid-kd-y": 0.0,
            },
            "stm32": {
                "max-speed-x-deg-s": 1.0,
                "max-speed-y-deg-s": 1.0,
                "acceleration-x-deg-s2": 2.0,
                "acceleration-y-deg-s2": 2.0,
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


def _default_overview_mapping() -> dict[str, Any]:
    width = _DEFAULT_WIDTH
    height = _DEFAULT_HEIGHT
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    focal = 1000.0
    matrix = [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    return {
        "schema_version": 1,
        "image_width": width,
        "image_height": height,
        "K": matrix,
        "D": [0.0, 0.0, 0.0, 0.0],
        "new_camera_matrix": matrix,
    }


def _default_stereo_mapping() -> dict[str, Any]:
    width = _DEFAULT_WIDTH
    height = _DEFAULT_HEIGHT
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    focal = 250.0
    matrix = [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    projection = [
        [focal, 0.0, cx, 0.0],
        [0.0, focal, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    return {
        "schema_version": 1,
        "image_width": width,
        "image_height": height,
        "K_left": matrix,
        "D_left": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K_right": matrix,
        "D_right": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R": identity,
        "T": [0.0, 0.0, 0.0],
        "R1": identity,
        "R2": identity,
        "P1": projection,
        "P2": projection,
        "Q": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _write_json_temp(path: Path, data: dict[str, Any]) -> Path:
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.recovery-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(data, temp_file, indent=2, ensure_ascii=False, allow_nan=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
    except (OSError, TypeError, ValueError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise InputRecoveryError(f"cannot stage default {path}: {exc}") from exc
    return temp_path


def _create_backup(source: Path, destination: Path) -> None:
    try:
        with source.open("rb") as source_file, destination.open("xb") as backup_file:
            shutil.copyfileobj(source_file, backup_file)
            backup_file.flush()
            os.fsync(backup_file.fileno())
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise InputRecoveryError(
            f"cannot back up {source} to {destination}: {exc}"
        ) from exc


def _backup_token(existing_paths: tuple[Path, ...], timestamp: str) -> str:
    suffix_index = 0
    while True:
        suffix = "" if suffix_index == 0 else f"-{suffix_index:02d}"
        token = f"{timestamp}{suffix}"
        if all(
            not path.with_name(f"{path.name}-{token}.bak").exists()
            for path in existing_paths
        ):
            return token
        suffix_index += 1


def _restore_original(path: Path, backup: Path | None) -> None:
    if backup is None:
        path.unlink(missing_ok=True)
        return
    rollback_temp = path.with_name(f".{path.name}.rollback.tmp")
    try:
        shutil.copyfile(backup, rollback_temp)
        os.replace(rollback_temp, path)
    finally:
        rollback_temp.unlink(missing_ok=True)


def recover_default_inputs(
    paths: LauncherPaths,
    *,
    now: datetime | None = None,
) -> InputRecoveryResult:
    """Back up local inputs and replace the full set with validated safe defaults.

    The timestamp is local system time by project/operator policy. The strict
    loaders themselves remain unchanged; this function is only called after
    explicit operator confirmation.
    """
    local_now = now or datetime.now(UTC).astimezone()
    timestamp = local_now.strftime("%Y%m%d-%H%M%S")
    targets = (
        paths.config,
        paths.overview_calibration,
        paths.stereo_calibration,
    )
    mappings = (
        _default_config_mapping(),
        _default_overview_mapping(),
        _default_stereo_mapping(),
    )
    validators: tuple[Callable[[Path], object], ...] = (
        load_config,
        load_overview_calibration,
        load_stereo_calibration,
    )

    staged: dict[Path, Path] = {}
    try:
        for target, mapping, validator in zip(
            targets, mappings, validators, strict=True
        ):
            temp_path = _write_json_temp(target, mapping)
            staged[target] = temp_path
            validator(temp_path)
    except InputRecoveryError:
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)
        raise InputRecoveryError(
            f"generated default inputs failed validation: {exc}"
        ) from exc

    existing = tuple(path for path in targets if path.exists())
    token = _backup_token(existing, timestamp)
    backups: dict[Path, Path] = {}
    created_backups: list[Path] = []
    try:
        for source in existing:
            backup = source.with_name(f"{source.name}-{token}.bak")
            _create_backup(source, backup)
            backups[source] = backup
            created_backups.append(backup)
    except InputRecoveryError:
        for backup in created_backups:
            backup.unlink(missing_ok=True)
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)
        raise

    replaced: list[Path] = []
    try:
        for target in targets:
            os.replace(staged[target], target)
            replaced.append(target)
        # Validate the committed set once more before reporting success.
        load_config(paths.config)
        load_overview_calibration(paths.overview_calibration)
        load_stereo_calibration(paths.stereo_calibration)
    except (OSError, ValueError) as exc:
        rollback_failures: list[str] = []
        for target in reversed(replaced):
            try:
                _restore_original(target, backups.get(target))
            except OSError as rollback_exc:
                rollback_failures.append(f"{target}: {rollback_exc}")
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)
        if not rollback_failures:
            for backup in created_backups:
                backup.unlink(missing_ok=True)
        detail = ""
        if rollback_failures:
            detail = "; rollback failures: " + "; ".join(rollback_failures)
        raise InputRecoveryError(
            f"failed to install default inputs: {exc}{detail}"
        ) from exc
    finally:
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)

    return InputRecoveryResult(
        timestamp=token,
        restored_paths=targets,
        backups=tuple((source, backups[source]) for source in existing),
    )


__all__ = [
    "InputRecoveryError",
    "InputRecoveryResult",
    "recover_default_inputs",
]
