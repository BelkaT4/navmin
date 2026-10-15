#!/usr/bin/env python3
"""Create a compact project ZIP for handing the current NavMin workspace to ChatGPT.

The archive is intentionally independent from Git and ``.gitignore``. It packs
all regular project files except the explicit denylist below, so local working
configuration and calibration files are preserved in the handoff snapshot.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Files/directories excluded only from ChatGPT handoff snapshots.
# This policy is intentionally independent from .gitignore.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "logs",
    "_chat_exports",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}
EXCLUDED_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.zip",
}

EXPORT_DIR_NAME = "_chat_exports"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_cleanup(root: Path) -> None:
    cleanup_script = root / "tools" / "clean_workspace.py"
    if not cleanup_script.is_file():
        raise SystemExit(f"ERROR: Cleanup script is missing: {cleanup_script}")
    try:
        subprocess.run(
            [sys.executable, str(cleanup_script)],
            cwd=root,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ERROR: Workspace cleanup failed with code {exc.returncode}")


def _excluded_file(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def _collect_files(root: Path) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []

    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)

        # Prune excluded directories before os.walk descends into them.
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name not in EXCLUDED_DIR_NAMES
            and not name.endswith(".egg-info")
            and not (current / name).is_symlink()
        )

        for file_name in sorted(file_names):
            source = current / file_name
            if source.is_symlink() or not source.is_file() or _excluded_file(source):
                continue

            rel = source.relative_to(root)
            files.append((source, rel))

    files.sort(key=lambda item: item[1].as_posix())
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise SystemExit(f"ERROR: ZIP integrity check failed at: {bad_member}")


def main() -> int:
    root = _project_root()
    _run_cleanup(root)

    export_dir = root / EXPORT_DIR_NAME
    export_dir.mkdir(exist_ok=True)

    files = _collect_files(root)

    timestamp = dt.datetime.now(dt.UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    project_name = root.name.replace(" ", "_") or "project"
    archive_path = export_dir / f"{project_name}_for_chat_{timestamp}.zip"

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for source, rel in files:
            archive.write(source, rel.as_posix())

    _verify_zip(archive_path)
    digest = _sha256(archive_path)
    size_mb = archive_path.stat().st_size / (1024 * 1024)

    print()
    print("Archive created successfully.")
    print(f"Files:   {len(files)}")
    print(f"Size:    {size_mb:.2f} MiB")
    print(f"Archive: {archive_path}")
    print(f"SHA-256: {digest}")
    print("ZIP test: PASS")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
