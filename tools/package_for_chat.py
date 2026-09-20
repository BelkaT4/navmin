#!/usr/bin/env python3
"""Create a clean project ZIP for handing the current working tree to ChatGPT.

The file list comes from Git:
- tracked files are always included;
- untracked files are included only when they are not ignored by Git;
- ignored files and .git/ are excluded.

The archive uses the current working-tree contents, not HEAD, so uncommitted
changes are included.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


EXPORT_DIR_NAME = "_chat_exports"


def run_git(root_hint: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root_hint,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise SystemExit("ERROR: Git is not installed or is not available in PATH.")
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(f"ERROR: Git command failed: {message or exc.returncode}")
    return result.stdout


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    root_text = run_git(script_dir, "rev-parse", "--show-toplevel").strip()
    if not root_text:
        raise SystemExit("ERROR: Cannot determine Git repository root.")

    root = Path(root_text).resolve()
    export_dir = root / EXPORT_DIR_NAME
    export_dir.mkdir(exist_ok=True)

    # Use the Git index + standard ignore rules instead of reimplementing
    # .gitignore parsing. The current file contents are packed from disk.
    output = run_git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    relative_files = [Path(line) for line in output.splitlines() if line.strip()]

    files: list[tuple[Path, Path]] = []
    for rel in relative_files:
        # Never recursively package previous exports, even if ignore rules change.
        if rel.parts and rel.parts[0] == EXPORT_DIR_NAME:
            continue

        source = root / rel
        if source.is_file():
            files.append((source, rel))

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    sha256 = hashlib.sha256()
    with archive_path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            sha256.update(chunk)

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print()
    print("Archive created successfully.")
    print(f"Files:   {len(files)}")
    print(f"Size:    {size_mb:.2f} MiB")
    print(f"Archive: {archive_path}")
    print(f"SHA-256: {sha256.hexdigest()}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
