"""Explicit operator-approved recovery for local config and calibration inputs."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from navmin.calibration import load_overview_calibration, load_stereo_calibration
from navmin.config import load_config
from navmin.imu import ImuCalibration
from navmin.launcher import LauncherPaths, resolve_selected_config_path


class InputRecoveryError(RuntimeError):
    """Raised when a requested recovery cannot be completed safely."""


@dataclass(frozen=True)
class InputRecoveryResult:
    timestamp: str
    restored_paths: tuple[Path, ...]
    backups: tuple[tuple[Path, Path], ...]


def _repository_defaults_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "defaults"


def _canonical_recovery_targets(paths: LauncherPaths) -> tuple[Path, Path, Path, Path]:
    config_path = resolve_selected_config_path(paths.config)
    return (
        config_path,
        config_path.parent / "calibration" / "overview.json",
        config_path.parent / "calibration" / "stereo.json",
        config_path.parent / "calibration" / "imu.json",
    )


def _default_sources(defaults_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        defaults_dir / "navmin.json",
        defaults_dir / "calibration" / "overview.json",
        defaults_dir / "calibration" / "stereo.json",
        defaults_dir / "calibration" / "imu.json",
    )


def _stage_default_file(source: Path, target: Path) -> Path:
    temp_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_file, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.recovery-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(source_file, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
    except OSError as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise InputRecoveryError(
            f"cannot stage recovery default {source} for {target}: {exc}"
        ) from exc
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
    defaults_dir: Path | None = None,
) -> InputRecoveryResult:
    """Install vetted repository defaults at canonical targets and then stop.

    Explicit CLI calibration overrides are never recovery targets. Recovery
    always installs ``navmin.json`` plus canonical Overview/Stereo/IMU
    calibration files next to the selected root config and never searches
    arbitrary custom paths.
    """
    local_now = now or datetime.now(UTC).astimezone()
    timestamp = local_now.strftime("%Y%m%d-%H%M%S")
    targets = _canonical_recovery_targets(paths)
    defaults = _default_sources(defaults_dir or _repository_defaults_dir())
    validators = (
        load_config,
        load_overview_calibration,
        load_stereo_calibration,
        ImuCalibration.load,
    )

    for source in defaults:
        if not source.is_file():
            raise InputRecoveryError(f"recovery default not found: {source}")

    staged: dict[Path, Path] = {}
    try:
        for target, source, validator in zip(targets, defaults, validators, strict=True):
            temp_path = _stage_default_file(source, target)
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
            f"repository recovery defaults failed validation: {exc}"
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
        load_config(targets[0])
        load_overview_calibration(targets[1])
        load_stereo_calibration(targets[2])
        ImuCalibration.load(targets[3])
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
