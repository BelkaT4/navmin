"""Config Manager: validated snapshots, atomic persistence, and publication."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from navmin.concurrency import LatestValue
from navmin.contracts import ConfigUpdate

from .models import AimingConfig, AppConfig, TurretConfig, UiConfig, VisionConfig
from .parser import ConfigPersistenceError, config_to_mapping, load_config, parse_config


def save_config(path: str | Path, config: AppConfig) -> None:
    """Atomically persist one fully validated config snapshot."""
    config_path = Path(path)
    parent = config_path.parent
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                config_to_mapping(config),
                temp_file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, config_path)
        temp_path = None
        _fsync_directory(parent)
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigPersistenceError(f"failed to save {config_path}: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    """Persist the directory entry where the platform supports directory fsync."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            # Some target filesystems/platforms do not support directory fsync.
            pass
    finally:
        os.close(directory_fd)


class ConfigManager:
    """Authoritative typed config source; owners retain apply/restart decisions."""

    def __init__(self, path: str | Path, initial: AppConfig) -> None:
        self._path = Path(path)
        self._lock = RLock()
        self._config = initial
        self._revision = 0

        self._vision_updates: LatestValue[ConfigUpdate[VisionConfig]] = LatestValue()
        self._aiming_updates: LatestValue[ConfigUpdate[AimingConfig]] = LatestValue()
        self._turret_updates: LatestValue[ConfigUpdate[TurretConfig]] = LatestValue()
        self._ui_updates: LatestValue[ConfigUpdate[UiConfig]] = LatestValue()
        self._publish_all(initial, revision=0)

    @classmethod
    def load(cls, path: str | Path = "config.json") -> ConfigManager:
        return cls(path, load_config(path))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    @property
    def vision_updates(self) -> LatestValue[ConfigUpdate[VisionConfig]]:
        return self._vision_updates

    @property
    def aiming_updates(self) -> LatestValue[ConfigUpdate[AimingConfig]]:
        return self._aiming_updates

    @property
    def turret_updates(self) -> LatestValue[ConfigUpdate[TurretConfig]]:
        return self._turret_updates

    @property
    def ui_updates(self) -> LatestValue[ConfigUpdate[UiConfig]]:
        return self._ui_updates

    def save(self) -> None:
        with self._lock:
            save_config(self._path, self._config)

    def apply_runtime_update(
        self,
        data: Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> bool:
        """Validate atomically, optionally persist, then publish changed modules.

        Returns ``True`` only for an actual accepted change. Validation or
        persistence failures leave the active snapshot and global revision
        unchanged.
        """
        candidate = parse_config(data)

        with self._lock:
            previous = self._config
            if candidate == previous:
                return False

            if persist:
                save_config(self._path, candidate)

            revision = self._revision + 1
            self._config = candidate
            self._revision = revision
            self._publish_changed(previous, candidate, revision)
            return True

    def _publish_all(self, config: AppConfig, revision: int) -> None:
        self._vision_updates.publish(ConfigUpdate(revision, config.vision))
        self._aiming_updates.publish(ConfigUpdate(revision, config.aiming))
        self._turret_updates.publish(ConfigUpdate(revision, config.turret))
        self._ui_updates.publish(ConfigUpdate(revision, config.ui))

    def _publish_changed(
        self,
        previous: AppConfig,
        current: AppConfig,
        revision: int,
    ) -> None:
        if current.vision != previous.vision:
            self._vision_updates.publish(ConfigUpdate(revision, current.vision))
        if current.aiming != previous.aiming:
            self._aiming_updates.publish(ConfigUpdate(revision, current.aiming))
        if current.turret != previous.turret:
            self._turret_updates.publish(ConfigUpdate(revision, current.turret))
        if current.ui != previous.ui:
            self._ui_updates.publish(ConfigUpdate(revision, current.ui))


__all__ = ["ConfigManager", "save_config"]
