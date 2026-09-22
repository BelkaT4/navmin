from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import navmin.session_artifacts as session_artifacts_module
from navmin.diagnostic_launcher import _synthetic_diagnostic_inputs
from navmin.logging_setup import (
    SESSION_LOG_BACKUP_COUNT,
    SESSION_LOG_MAX_BYTES,
    session_file_logging,
)
from navmin.session_artifacts import (
    SessionArtifacts,
    SessionStatus,
    SourceInputPaths,
)


def _artifacts(
    tmp_path: Path,
    *,
    mode: str = "normal",
    now: datetime | None = None,
) -> SessionArtifacts:
    return SessionArtifacts.create(
        tmp_path / "logs",
        mode=mode,
        overview_backend="real" if mode == "normal" else "localhost",
        stereo_left_backend="real" if mode == "normal" else "localhost",
        turret_backend="real" if mode == "normal" else "pty",
        input_mode="files" if mode == "normal" else "synthetic",
        now=now,
        git_root=tmp_path / "not-a-repository",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_session_directories_are_unique_and_have_expected_initial_layout(tmp_path) -> None:
    now = datetime(2026, 9, 22, 4, 5, 6, 123456, tzinfo=UTC)
    first = _artifacts(tmp_path, now=now)
    second = _artifacts(tmp_path, now=now)

    assert first.session_id == "navmin-normal-20260922-040506-123456"
    assert second.session_id == "navmin-normal-20260922-040506-123457"
    assert first.session_dir != second.session_dir
    diagnostic_first = _artifacts(tmp_path, mode="diagnostic", now=now)
    diagnostic_second = _artifacts(tmp_path, mode="diagnostic", now=now)
    assert diagnostic_first.session_id == "navmin-diagnostic-20260922-040506-123456"
    assert diagnostic_second.session_id == "navmin-diagnostic-20260922-040506-123457"
    assert diagnostic_first.session_dir != diagnostic_second.session_dir
    assert first.manifest_path.is_file()
    assert first.inputs_dir.is_dir()
    assert first.runtime_log_path == first.session_dir / "runtime.log"

    manifest = _read_json(first.manifest_path)
    assert manifest["schema_version"] == 1
    assert manifest["session_id"] == first.session_id
    assert manifest["mode"] == "normal"
    assert manifest["status"] == "starting"
    assert manifest["finished_at_utc"] is None
    assert manifest["exit_code"] is None
    assert manifest["overview_backend"] == "real"
    assert manifest["stereo_left_backend"] == "real"
    assert manifest["turret_backend"] == "real"
    assert manifest["input_mode"] == "files"
    assert manifest["started_at_utc"].endswith("Z")


def test_manifest_finalizes_completed_and_failure_states(tmp_path) -> None:
    completed = _artifacts(tmp_path)
    completed.finalize(status=SessionStatus.COMPLETED, exit_code=0)
    completed_manifest = _read_json(completed.manifest_path)
    assert completed_manifest["status"] == "completed"
    assert completed_manifest["exit_code"] == 0
    assert completed_manifest["finished_at_utc"].endswith("Z")

    failed = _artifacts(tmp_path, mode="diagnostic")
    error = RuntimeError("runtime exploded")
    failed.finalize(
        status=SessionStatus.RUNTIME_FAILED,
        exit_code=2,
        failure=error,
    )
    failed_manifest = _read_json(failed.manifest_path)
    assert failed_manifest["status"] == "runtime-failed"
    assert failed_manifest["failure_type"] == "RuntimeError"
    assert failed_manifest["failure_message"] == "runtime exploded"


def test_manifest_updates_use_same_directory_atomic_replace(monkeypatch, tmp_path) -> None:
    artifacts = _artifacts(tmp_path)
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination) -> None:
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(session_artifacts_module.os, "replace", recording_replace)
    artifacts.finalize(status=SessionStatus.COMPLETED, exit_code=0)

    source, destination = calls[-1]
    assert source.parent == artifacts.manifest_path.parent
    assert destination == artifacts.manifest_path
    assert not source.exists()
    assert _read_json(destination)["status"] == "completed"


def test_session_file_logging_is_bounded_rotating_and_preserves_console(tmp_path) -> None:
    artifacts = _artifacts(tmp_path, mode="diagnostic")
    logger = logging.getLogger("navmin.session-test")

    with session_file_logging(artifacts.runtime_log_path, level=logging.DEBUG):
        navmin_logger = logging.getLogger("navmin")
        rotating = [
            item for item in navmin_logger.handlers if isinstance(item, RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == SESSION_LOG_MAX_BYTES == 10 * 1024 * 1024
        assert rotating[0].backupCount == SESSION_LOG_BACKUP_COUNT == 5
        console = [item for item in navmin_logger.handlers if item not in rotating]
        assert console
        assert any(item.level == logging.INFO for item in console)
        logger.debug("diagnostic detail")

    assert "diagnostic detail" in artifacts.runtime_log_path.read_text(encoding="utf-8")

    normal = _artifacts(tmp_path)
    with session_file_logging(normal.runtime_log_path, level=logging.INFO):
        rotating = [
            item
            for item in logging.getLogger("navmin").handlers
            if isinstance(item, RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].level == logging.INFO


def test_small_injected_rotation_keeps_runtime_log_bounded(tmp_path) -> None:
    artifacts = _artifacts(tmp_path)
    logger = logging.getLogger("navmin.rotation-test")

    with session_file_logging(
        artifacts.runtime_log_path,
        level=logging.INFO,
        max_bytes=120,
        backup_count=2,
    ):
        for index in range(30):
            logger.info("rotation-line-%02d-xxxxxxxxxxxxxxxx", index)

    assert artifacts.runtime_log_path.is_file()
    assert (artifacts.session_dir / "runtime.log.1").is_file()
    assert len(list(artifacts.session_dir.glob("runtime.log*"))) <= 3


def test_effective_synthetic_inputs_and_allowlisted_manifest_evidence(tmp_path) -> None:
    artifacts = _artifacts(tmp_path, mode="diagnostic")
    inputs = _synthetic_diagnostic_inputs()
    artifacts.write_effective_inputs(
        config=inputs.config,
        overview_calibration=inputs.overview_calibration,
        stereo_calibration=inputs.stereo_calibration,
        source_paths=None,
    )

    assert (artifacts.inputs_dir / "effective-config.json").is_file()
    assert (artifacts.inputs_dir / "overview-calibration.json").is_file()
    assert (artifacts.inputs_dir / "stereo-calibration.json").is_file()
    sources = _read_json(artifacts.inputs_dir / "source-hashes.json")
    assert all(item["source"] == "synthetic" for item in sources.values())
    assert all(item["sha256"] is None for item in sources.values())

    manifest = _read_json(artifacts.manifest_path)
    for key in (
        "python_version",
        "platform_system",
        "platform_release",
        "platform_machine",
        "git_commit",
        "git_branch",
        "git_dirty",
    ):
        assert key in manifest
    assert manifest["git_commit"] is None
    assert manifest["git_branch"] is None
    assert manifest["git_dirty"] is None
    forbidden = {"environment", "environ", "path", "home", "credentials"}
    assert forbidden.isdisjoint({key.lower() for key in manifest})


def test_file_backed_source_hashes_match_real_files(tmp_path) -> None:
    artifacts = _artifacts(tmp_path)
    inputs = _synthetic_diagnostic_inputs()
    config_path = tmp_path / "operator-config.json"
    overview_path = tmp_path / "overview.json"
    stereo_path = tmp_path / "stereo.json"
    config_path.write_bytes(b"config-evidence")
    overview_path.write_bytes(b"overview-evidence")
    stereo_path.write_bytes(b"stereo-evidence")

    artifacts.write_effective_inputs(
        config=inputs.config,
        overview_calibration=inputs.overview_calibration,
        stereo_calibration=inputs.stereo_calibration,
        source_paths=SourceInputPaths(config_path, overview_path, stereo_path),
    )

    sources = _read_json(artifacts.inputs_dir / "source-hashes.json")
    expected = {
        "config": config_path,
        "overview_calibration": overview_path,
        "stereo_calibration": stereo_path,
    }
    for key, path in expected.items():
        assert sources[key]["source"] == "file"
        assert sources[key]["path"] == str(path)
        assert sources[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
