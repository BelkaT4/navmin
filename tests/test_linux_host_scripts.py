from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINUX_TOOLS = REPO_ROOT / "tools" / "platform" / "linux"
EXECUTABLE_SHELL_SCRIPTS = (
    REPO_ROOT / "run_navmin.sh",
    LINUX_TOOLS / "setup_serial_access.sh",
    LINUX_TOOLS / "setup_camera_network.sh",
    LINUX_TOOLS / "camera_network_up.sh",
    LINUX_TOOLS / "camera_network_down.sh",
)
SHELL_FILES = (*EXECUTABLE_SHELL_SCRIPTS, LINUX_TOOLS / "host_config.sh")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _read_state(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _write_host_config(
    tmp_path: Path,
    *,
    serial_device: str = "/dev/ttyUSB0",
    serial_alias: str = "navmin-turret",
    interface: str = "wlan0",
    address: str = "192.168.42.2/24",
    profile: str = "NavMin Cameras",
) -> Path:
    config = tmp_path / "host.local.env"
    config.write_text(
        "\n".join(
            (
                "# Test host settings",
                f"SERIAL_SETUP_DEVICE={serial_device}",
                f"SERIAL_ALIAS={serial_alias}",
                f"CAMERA_INTERFACE={interface}",
                f"CAMERA_HOST_ADDRESS={address}",
                f"CAMERA_PROFILE_NAME={profile}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config


def _write_fake_nmcli(
    tmp_path: Path,
    *,
    active: str,
    interface: str = "wlan0",
    address: str = "192.168.42.2/24",
    profile: str = "NavMin Cameras",
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nm_state = tmp_path / "nm-state"
    nm_state.write_text(
        "\n".join(
            (
                f"profile={profile}",
                "target=target-uuid",
                "previous=previous-uuid",
                f"interface={interface}",
                f"address={address}",
                f"active={active}",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_executable(
        bin_dir / "nmcli",
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

state_path = Path(os.environ["FAKE_NM_STATE"])


def load():
    values = {}
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def save(values):
    state_path.write_text(
        "".join(f"{key}={value}\\n" for key, value in values.items()),
        encoding="utf-8",
    )


values = load()
args = sys.argv[1:]
profile = values["profile"]
target = values["target"]
previous = values["previous"]
interface = values["interface"]
address = values["address"]

if args[:4] == ["-g", "connection.uuid", "connection", "show"]:
    identifier = " ".join(args[4:])
    if identifier in (profile, target, f"uuid {target}"):
        print(target)
        raise SystemExit(0)
    if identifier in (previous, f"uuid {previous}"):
        print(previous)
        raise SystemExit(0)
    raise SystemExit(10)

if args[:4] == ["-g", "connection.interface-name", "connection", "show"]:
    print(interface)
    raise SystemExit(0)

if args[:4] == ["-g", "ipv4.addresses", "connection", "show"]:
    print(address)
    raise SystemExit(0)

if args[:5] == ["-t", "-f", "UUID,DEVICE", "connection", "show"] and args[5:] == ["--active"]:
    active = values.get("active", "")
    if active:
        print(f"{active}:{interface}")
    raise SystemExit(0)

if args[:4] == ["-g", "IP4.ADDRESS", "device", "show"]:
    if values.get("active") == target:
        print(address)
    else:
        print("10.0.0.2/24")
    raise SystemExit(0)

if args[:3] == ["connection", "up", "uuid"]:
    values["active"] = args[3]
    save(values)
    raise SystemExit(0)

if args[:3] == ["connection", "down", "uuid"]:
    if values.get("active") == args[3]:
        values["active"] = ""
    save(values)
    raise SystemExit(0)

if args[:4] == ["-g", "connection.id", "connection", "show"]:
    identifier = " ".join(args[4:])
    if previous in identifier:
        print("Previous connection")
        raise SystemExit(0)
    if target in identifier:
        print(profile)
        raise SystemExit(0)
    raise SystemExit(10)

print(f"unhandled fake nmcli arguments: {args!r}", file=sys.stderr)
raise SystemExit(99)
""",
    )
    host_config = _write_host_config(
        tmp_path,
        interface=interface,
        address=address,
        profile=profile,
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_NM_STATE": str(nm_state),
            "NAVMIN_NETWORK_STATE_FILE": str(tmp_path / "camera-network.state"),
            "NAVMIN_HOST_CONFIG": str(host_config),
        }
    )
    return env, nm_state


def test_linux_host_shell_files_are_posix_syntax_valid() -> None:
    for script in SHELL_FILES:
        result = subprocess.run(
            ["sh", "-n", str(script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"

    for script in EXECUTABLE_SHELL_SCRIPTS:
        assert script.stat().st_mode & 0o111


def test_example_host_config_is_tracked_but_local_config_is_ignored() -> None:
    example = (REPO_ROOT / "config" / "host.example.env").read_text(encoding="utf-8")
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "SERIAL_SETUP_DEVICE=/dev/ttyUSB0" in example
    assert "SERIAL_ALIAS=navmin-turret" in example
    assert "CAMERA_INTERFACE=enp3s0" in example
    assert "CAMERA_HOST_ADDRESS=192.168.42.2/24" in example
    assert "CAMERA_PROFILE_NAME=NavMin Cameras" in example
    assert "/config/host.local.env" in ignore


def test_camera_network_up_and_down_restore_previous_profile(tmp_path: Path) -> None:
    env, nm_state = _write_fake_nmcli(tmp_path, active="previous-uuid")
    restore_state = Path(env["NAVMIN_NETWORK_STATE_FILE"])

    up = subprocess.run(
        [str(LINUX_TOOLS / "camera_network_up.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert up.returncode == 0, up.stderr
    assert _read_state(nm_state)["active"] == "target-uuid"
    saved = _read_state(restore_state)
    assert saved == {
        "interface": "wlan0",
        "target_uuid": "target-uuid",
        "previous_uuid": "previous-uuid",
    }

    down = subprocess.run(
        [str(LINUX_TOOLS / "camera_network_down.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert down.returncode == 0, down.stderr
    assert _read_state(nm_state)["active"] == "previous-uuid"
    assert not restore_state.exists()


def test_camera_network_down_leaves_profile_that_was_already_active(tmp_path: Path) -> None:
    env, nm_state = _write_fake_nmcli(tmp_path, active="target-uuid")

    subprocess.run(
        [str(LINUX_TOOLS / "camera_network_up.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(LINUX_TOOLS / "camera_network_down.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert _read_state(nm_state)["active"] == "target-uuid"


def test_camera_network_up_rejects_stale_restore_state(tmp_path: Path) -> None:
    env, nm_state = _write_fake_nmcli(tmp_path, active="previous-uuid")
    restore_state = Path(env["NAVMIN_NETWORK_STATE_FILE"])
    restore_state.write_text(
        "interface=wlan0\ntarget_uuid=other-target\nprevious_uuid=old-uuid\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(LINUX_TOOLS / "camera_network_up.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "stale network state" in result.stderr
    assert _read_state(nm_state)["active"] == "previous-uuid"


@pytest.mark.parametrize(
    "script_name",
    ["setup_serial_access.sh", "setup_camera_network.sh"],
)
def test_setup_scripts_require_local_host_config(
    tmp_path: Path,
    script_name: str,
) -> None:
    env = os.environ.copy()
    env["NAVMIN_HOST_CONFIG"] = str(tmp_path / "missing.env")

    result = subprocess.run(
        [str(LINUX_TOOLS / script_name)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "host config not found" in result.stderr


def test_camera_network_up_requires_local_host_config(tmp_path: Path) -> None:
    env, _ = _write_fake_nmcli(tmp_path, active="previous-uuid")
    env["NAVMIN_HOST_CONFIG"] = str(tmp_path / "missing.env")

    result = subprocess.run(
        [str(LINUX_TOOLS / "camera_network_up.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "host config not found" in result.stderr


def test_camera_network_up_rejects_profile_that_differs_from_host_config(
    tmp_path: Path,
) -> None:
    env, nm_state = _write_fake_nmcli(tmp_path, active="previous-uuid")
    host_config = Path(env["NAVMIN_HOST_CONFIG"])
    host_config.write_text(
        host_config.read_text(encoding="utf-8").replace(
            "CAMERA_HOST_ADDRESS=192.168.42.2/24",
            "CAMERA_HOST_ADDRESS=192.168.42.99/24",
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(LINUX_TOOLS / "camera_network_up.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "host config expects '192.168.42.99/24'" in result.stderr
    assert "run setup_camera_network.sh again" in result.stderr
    assert _read_state(nm_state)["active"] == "previous-uuid"


@pytest.mark.parametrize("app_returncode", [0, 7])
def test_run_navmin_restores_network_and_preserves_application_result(
    tmp_path: Path,
    app_returncode: int,
) -> None:
    env, nm_state = _write_fake_nmcli(tmp_path, active="previous-uuid")
    bin_dir = Path(env["PATH"].split(os.pathsep, 1)[0])
    _write_executable(
        bin_dir / "uv",
        f"#!/bin/sh\nexit {app_returncode}\n",
    )

    result = subprocess.run(
        [str(REPO_ROOT / "run_navmin.sh"), "--preflight-only"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == app_returncode
    assert _read_state(nm_state)["active"] == "previous-uuid"
    assert not Path(env["NAVMIN_NETWORK_STATE_FILE"]).exists()
    assert "verified IPv4: 192.168.42.2/24" in result.stdout
    assert "network restore complete" in result.stdout
