from __future__ import annotations

from pathlib import Path

import pytest

from navmin.host_environment import (
    HostCommandError,
    HostConfigurationCreated,
    HostEnvironmentError,
    LinuxHostEnvironment,
    WindowsHostEnvironment,
    create_host_environment,
)


def _write_linux_tree(tmp_path: Path, *, with_working_config: bool = True) -> Path:
    repo = tmp_path / "repo"
    linux = repo / "tools" / "platform" / "linux"
    defaults = repo / "config" / "defaults"
    linux.mkdir(parents=True)
    defaults.mkdir(parents=True)
    for name in ("camera_network_up.sh", "camera_network_down.sh"):
        path = linux / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    (linux / "host_config.sh").write_text("# helper\n", encoding="utf-8")
    (defaults / "host.env").write_text("SAFE=1\n", encoding="utf-8")
    if with_working_config:
        (repo / "config" / "host.env").write_text("SITE=1\n", encoding="utf-8")
    return repo


def test_factory_selects_linux_and_windows_without_running_helpers(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command, _cwd, _env) -> int:
        calls.append(tuple(command))
        return 0

    linux = create_host_environment(
        tmp_path,
        platform_name="linux",
        command_runner=runner,
    )
    windows = create_host_environment(
        tmp_path,
        platform_name="win32",
        command_runner=runner,
    )

    assert isinstance(linux, LinuxHostEnvironment)
    assert isinstance(windows, WindowsHostEnvironment)
    assert calls == []


def test_factory_rejects_unknown_platform(tmp_path: Path) -> None:
    with pytest.raises(HostEnvironmentError, match="unsupported host platform"):
        create_host_environment(tmp_path, platform_name="plan9")


def test_linux_without_real_camera_needs_no_helpers_or_host_config(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command, _cwd, _env) -> int:
        calls.append(tuple(command))
        return 0

    host = LinuxHostEnvironment(tmp_path, environ={}, command_runner=runner)

    with host.activation(camera_network_required=False):
        pass

    assert calls == []
    assert host.network_prepared is False
    assert not (tmp_path / "config" / "host.env").exists()


def test_linux_missing_working_config_creates_safe_template_and_stops(
    tmp_path: Path,
) -> None:
    repo = _write_linux_tree(tmp_path, with_working_config=False)
    calls: list[tuple[str, ...]] = []
    host = LinuxHostEnvironment(
        repo,
        environ={},
        command_runner=lambda command, _cwd, _env: calls.append(tuple(command)) or 0,
    )

    with pytest.raises(HostConfigurationCreated) as error:
        host.prepare(camera_network_required=True)

    working = repo / "config" / "host.env"
    assert error.value.path == working
    assert working.read_text(encoding="utf-8") == "SAFE=1\n"
    assert calls == []
    assert host.network_prepared is False


def test_linux_missing_explicit_config_is_not_recovered(tmp_path: Path) -> None:
    repo = _write_linux_tree(tmp_path, with_working_config=False)
    explicit = Path("site") / "custom.env"
    host = LinuxHostEnvironment(
        repo,
        environ={"NAVMIN_HOST_CONFIG": str(explicit)},
        command_runner=lambda *_args: pytest.fail("network helper must not run"),
    )

    with pytest.raises(HostEnvironmentError, match="explicit host config not found"):
        host.prepare(camera_network_required=True)

    assert not (repo / explicit).exists()
    assert not (repo / "config" / "host.env").exists()


def test_linux_existing_host_config_is_not_replaced_when_prepare_fails(
    tmp_path: Path,
) -> None:
    repo = _write_linux_tree(tmp_path)
    working = repo / "config" / "host.env"
    working.write_text("BROKEN_BUT_OPERATOR_OWNED=1\n", encoding="utf-8")
    original = working.read_bytes()

    host = LinuxHostEnvironment(
        repo,
        environ={},
        command_runner=lambda _command, _cwd, _env: 9,
    )

    with pytest.raises(HostCommandError) as error:
        host.prepare(camera_network_required=True)

    assert error.value.returncode == 9
    assert working.read_bytes() == original
    assert host.network_prepared is False


def test_linux_prepare_and_restore_use_existing_helpers(tmp_path: Path) -> None:
    repo = _write_linux_tree(tmp_path)
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []
    environ = {"NAVMIN_NETWORK_STATE_FILE": "/tmp/navmin-test.state"}

    def runner(command, cwd, env) -> int:
        calls.append((tuple(command), cwd, dict(env)))
        return 0

    host = LinuxHostEnvironment(repo, environ=environ, command_runner=runner)
    host.prepare(camera_network_required=True)
    assert host.network_prepared is True
    host.restore()

    assert [Path(call[0][0]).name for call in calls] == [
        "camera_network_up.sh",
        "camera_network_down.sh",
    ]
    assert all(call[1] == repo.resolve() for call in calls)
    assert all(call[2] == environ for call in calls)
    assert host.network_prepared is False


def test_linux_prepare_failure_does_not_call_restore(tmp_path: Path) -> None:
    repo = _write_linux_tree(tmp_path)
    calls: list[str] = []

    def runner(command, _cwd, _env) -> int:
        calls.append(Path(command[0]).name)
        return 7

    host = LinuxHostEnvironment(repo, environ={}, command_runner=runner)

    with pytest.raises(HostCommandError), host.activation(camera_network_required=True):
        pytest.fail("body must not start after prepare failure")

    assert calls == ["camera_network_up.sh"]


def test_activation_restores_after_success_and_runtime_failure(tmp_path: Path) -> None:
    repo = _write_linux_tree(tmp_path)
    calls: list[str] = []

    def runner(command, _cwd, _env) -> int:
        calls.append(Path(command[0]).name)
        return 0

    host = LinuxHostEnvironment(repo, environ={}, command_runner=runner)
    with host.activation(camera_network_required=True):
        pass

    assert calls == ["camera_network_up.sh", "camera_network_down.sh"]

    calls.clear()
    with pytest.raises(
        RuntimeError, match="runtime failed"
    ), host.activation(camera_network_required=True):
        raise RuntimeError("runtime failed")

    assert calls == ["camera_network_up.sh", "camera_network_down.sh"]
    assert host.network_prepared is False



@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(130)])
def test_activation_restores_for_controlled_base_exceptions(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    repo = _write_linux_tree(tmp_path)
    calls: list[str] = []

    def runner(command, _cwd, _env) -> int:
        calls.append(Path(command[0]).name)
        return 0

    host = LinuxHostEnvironment(repo, environ={}, command_runner=runner)
    with pytest.raises(type(failure)), host.activation(camera_network_required=True):
        raise failure

    assert calls == ["camera_network_up.sh", "camera_network_down.sh"]
    assert host.network_prepared is False

def test_activation_preserves_body_failure_when_restore_also_fails(
    tmp_path: Path,
) -> None:
    repo = _write_linux_tree(tmp_path)
    calls: list[str] = []

    def runner(command, _cwd, _env) -> int:
        name = Path(command[0]).name
        calls.append(name)
        return 11 if name == "camera_network_down.sh" else 0

    host = LinuxHostEnvironment(repo, environ={}, command_runner=runner)
    with pytest.raises(
        RuntimeError, match="runtime failed"
    ) as error, host.activation(camera_network_required=True):
        raise RuntimeError("runtime failed")

    assert calls == ["camera_network_up.sh", "camera_network_down.sh"]
    assert any("Host restore also failed" in note for note in error.value.__notes__)
    assert host.network_prepared is True


def test_activation_reports_restore_failure_after_success(tmp_path: Path) -> None:
    repo = _write_linux_tree(tmp_path)

    def runner(command, _cwd, _env) -> int:
        return 12 if Path(command[0]).name == "camera_network_down.sh" else 0

    host = LinuxHostEnvironment(repo, environ={}, command_runner=runner)

    with pytest.raises(HostCommandError) as error, host.activation(
        camera_network_required=True
    ):
        pass

    assert error.value.returncode == 12
    assert host.network_prepared is True


def test_windows_backend_never_invokes_linux_helpers(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command, _cwd, _env) -> int:
        calls.append(tuple(command))
        return 0

    host = create_host_environment(
        tmp_path,
        platform_name="win32",
        command_runner=runner,
    )
    with host.activation(camera_network_required=True):
        pass

    assert calls == []
