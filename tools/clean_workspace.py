#!/usr/bin/env python3
"""Remove rebuildable cache and build artifacts from the NavMin workspace."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# These trees are workspace state, not cleanup targets. In particular, do not
# walk through a virtual environment just to remove Python bytecode caches.
PROTECTED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "logs",
    "_chat_exports",
}
REMOVABLE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}
REMOVABLE_FILE_SUFFIXES = {".pyc", ".pyo"}
REMOVABLE_DIR_SUFFIXES = {".egg-info"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_workspace(root: Path) -> tuple[int, int]:
    """Remove only known rebuildable artifacts below *root*.

    Returns ``(removed_files, removed_directories)``. The function intentionally
    does not inspect ``.gitignore`` and never removes local configs, calibrations,
    virtual environments, logs, snapshots, archives, or IDE metadata.
    """

    root = root.resolve()
    removed_files = 0
    removed_directories = 0

    for current_root, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_root)

        kept_dirs: list[str] = []
        for name in dir_names:
            directory = current / name
            if name in PROTECTED_DIR_NAMES or directory.is_symlink():
                continue
            if name in REMOVABLE_DIR_NAMES or any(
                name.endswith(suffix) for suffix in REMOVABLE_DIR_SUFFIXES
            ):
                shutil.rmtree(directory)
                removed_directories += 1
                continue
            kept_dirs.append(name)
        dir_names[:] = kept_dirs

        for name in file_names:
            path = current / name
            if path.is_symlink():
                continue
            if path.suffix in REMOVABLE_FILE_SUFFIXES:
                path.unlink()
                removed_files += 1

    return removed_files, removed_directories


def main() -> int:
    files, directories = clean_workspace(_project_root())
    print(
        "Workspace cleanup complete: "
        f"{files} files and {directories} directories removed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
