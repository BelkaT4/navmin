"""Launcher-owned offline evidence for one NavMin application session."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from navmin.calibration import OverviewCalibration, StereoCalibration
from navmin.config import AppConfig, config_to_mapping

_MANIFEST_SCHEMA_VERSION = 1
_JSON_INDENT = 2
_GIT_TIMEOUT_SECONDS = 2.0


class SessionStatus(StrEnum):
    STARTING = "starting"
    INPUT_FAILED = "input-failed"
    RUNTIME_FAILED = "runtime-failed"
    CLEANUP_FAILED = "cleanup-failed"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class SourceInputPaths:
    config: Path
    overview_calibration: Path
    stereo_calibration: Path


@dataclass
class SessionArtifacts:
    """Own the bounded on-disk evidence directory for one launcher invocation."""

    session_id: str
    session_dir: Path
    runtime_log_path: Path
    manifest_path: Path
    inputs_dir: Path
    _manifest: dict[str, Any]

    @classmethod
    def create(
        cls,
        parent_dir: Path,
        *,
        mode: str,
        overview_backend: str,
        stereo_left_backend: str,
        turret_backend: str,
        input_mode: str,
        now: datetime | None = None,
        git_root: Path | None = None,
    ) -> SessionArtifacts:
        if mode not in {"normal", "diagnostic"}:
            raise ValueError("mode must be 'normal' or 'diagnostic'")
        started = _utc_datetime(now)
        session_dir, session_id = _create_unique_session_dir(
            Path(parent_dir),
            mode=mode,
            started=started,
        )
        inputs_dir = session_dir / "inputs"
        inputs_dir.mkdir()
        runtime_log_path = session_dir / "runtime.log"
        manifest_path = session_dir / "manifest.json"

        manifest: dict[str, Any] = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "session_id": session_id,
            "mode": mode,
            "started_at_utc": _iso_utc(started),
            "finished_at_utc": None,
            "status": SessionStatus.STARTING.value,
            "exit_code": None,
            "overview_backend": overview_backend,
            "stereo_left_backend": stereo_left_backend,
            "turret_backend": turret_backend,
            "input_mode": input_mode,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "failure_type": None,
            "failure_message": None,
            "cleanup_failure_type": None,
            "cleanup_failure_message": None,
        }
        manifest.update(_git_metadata(git_root or Path.cwd()))
        artifacts = cls(
            session_id=session_id,
            session_dir=session_dir,
            runtime_log_path=runtime_log_path,
            manifest_path=manifest_path,
            inputs_dir=inputs_dir,
            _manifest=manifest,
        )
        artifacts.write_manifest()
        return artifacts

    def write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self._manifest)

    def write_effective_inputs(
        self,
        *,
        config: AppConfig,
        overview_calibration: OverviewCalibration,
        stereo_calibration: StereoCalibration,
        source_paths: SourceInputPaths | None,
    ) -> None:
        _atomic_write_json(
            self.inputs_dir / "effective-config.json",
            config_to_mapping(config),
        )
        _atomic_write_json(
            self.inputs_dir / "overview-calibration.json",
            _json_normalize(asdict(overview_calibration)),
        )
        _atomic_write_json(
            self.inputs_dir / "stereo-calibration.json",
            _json_normalize(asdict(stereo_calibration)),
        )
        if source_paths is None:
            source_evidence = {
                "config": _synthetic_source(),
                "overview_calibration": _synthetic_source(),
                "stereo_calibration": _synthetic_source(),
            }
        else:
            source_evidence = {
                "config": _file_source(source_paths.config),
                "overview_calibration": _file_source(
                    source_paths.overview_calibration
                ),
                "stereo_calibration": _file_source(source_paths.stereo_calibration),
            }
        _atomic_write_json(self.inputs_dir / "source-hashes.json", source_evidence)

    def finalize(
        self,
        *,
        status: SessionStatus,
        exit_code: int,
        failure: BaseException | None = None,
        cleanup_failure: BaseException | None = None,
        now: datetime | None = None,
    ) -> None:
        if status is SessionStatus.STARTING:
            raise ValueError("final status cannot be starting")
        self._manifest["finished_at_utc"] = _iso_utc(_utc_datetime(now))
        self._manifest["status"] = status.value
        self._manifest["exit_code"] = int(exit_code)
        if failure is not None:
            self._manifest["failure_type"] = type(failure).__name__
            self._manifest["failure_message"] = str(failure)
        if cleanup_failure is not None:
            self._manifest["cleanup_failure_type"] = type(cleanup_failure).__name__
            self._manifest["cleanup_failure_message"] = str(cleanup_failure)
        self.write_manifest()


def _create_unique_session_dir(
    parent_dir: Path,
    *,
    mode: str,
    started: datetime,
) -> tuple[Path, str]:
    parent_dir.mkdir(parents=True, exist_ok=True)
    candidate_time = started
    while True:
        session_id = f"navmin-{mode}-{candidate_time.strftime('%Y%m%d-%H%M%S-%f')}"
        path = parent_dir / session_id
        try:
            path.mkdir()
        except FileExistsError:
            candidate_time += timedelta(microseconds=1)
            continue
        return path, session_id


def _utc_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    timestamp = value.astimezone(UTC).isoformat(timespec="microseconds")
    return timestamp.replace("+00:00", "Z")


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_normalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_normalize(item) for item in value]
    return value


def _file_source(path: Path) -> dict[str, str | None]:
    input_path = Path(path)
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "source": "file",
        "path": str(input_path),
        "sha256": digest.hexdigest(),
    }


def _synthetic_source() -> dict[str, str | None]:
    return {"source": "synthetic", "path": None, "sha256": None}


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=_JSON_INDENT,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _git_metadata(root: Path) -> dict[str, str | bool | None]:
    result: dict[str, str | bool | None] = {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit is None:
        return result
    result["git_commit"] = commit
    result["git_branch"] = _git_output(root, "branch", "--show-current") or None
    status = _git_output(root, "status", "--porcelain")
    if status is not None:
        result["git_dirty"] = bool(status)
    return result


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


__all__ = ["SessionArtifacts", "SessionStatus", "SourceInputPaths"]
