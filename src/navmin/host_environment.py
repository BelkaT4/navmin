"""Platform host preparation owned by the unified launcher boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], int]


class HostEnvironmentError(RuntimeError):
    """Base error for host preparation or restoration failures."""


class HostConfigurationCreated(HostEnvironmentError):
    """Safe local host configuration was created and must be reviewed first."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Created safe {path} from config/defaults/host.env; "
            "review and fill host parameters, then run NavMin again"
        )


class HostCommandError(HostEnvironmentError):
    """A platform helper returned a non-zero status."""

    def __init__(self, command: Sequence[str], returncode: int) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        super().__init__(
            f"host command failed with exit status {returncode}: {' '.join(command)}"
        )


def _subprocess_runner(
    command: Sequence[str],
    cwd: Path,
    environ: Mapping[str, str],
) -> int:
    return subprocess.run(
        tuple(command),
        cwd=cwd,
        env=dict(environ),
        check=False,
    ).returncode


class HostEnvironment:
    """Minimal prepare/restore boundary for one launcher invocation."""

    def prepare(self, *, camera_network_required: bool) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError

    @contextmanager
    def activation(self, *, camera_network_required: bool) -> Iterator[None]:
        """Prepare host state and restore it on every controlled exit path.

        A failure in the application body remains authoritative. If restoration
        also fails, its error is attached as a note instead of masking the
        original exception. When the body succeeds, restoration failure is
        propagated normally.
        """
        self.prepare(camera_network_required=camera_network_required)
        try:
            yield
        except BaseException as error:
            try:
                self.restore()
            except HostEnvironmentError as restore_error:
                error.add_note(f"Host restore also failed: {restore_error}")
            raise
        else:
            self.restore()


class LinuxHostEnvironment(HostEnvironment):
    """Linux camera-network preparation backed by existing shell helpers."""

    def __init__(
        self,
        repo_root: Path,
        *,
        environ: Mapping[str, str] | None = None,
        command_runner: CommandRunner = _subprocess_runner,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._environ = dict(os.environ if environ is None else environ)
        self._run_command = command_runner
        self._network_prepared = False

    @property
    def network_prepared(self) -> bool:
        return self._network_prepared

    @property
    def _linux_tools(self) -> Path:
        return self._repo_root / "tools" / "platform" / "linux"

    @property
    def _network_up(self) -> Path:
        return self._linux_tools / "camera_network_up.sh"

    @property
    def _network_down(self) -> Path:
        return self._linux_tools / "camera_network_down.sh"

    @property
    def _host_helpers(self) -> Path:
        return self._linux_tools / "host_config.sh"

    def _require_helpers(self) -> None:
        for path in (self._network_up, self._network_down):
            if not path.is_file() or not os.access(path, os.X_OK):
                raise HostEnvironmentError(
                    f"network helper is not executable: {path}"
                )
        if not self._host_helpers.is_file():
            raise HostEnvironmentError(
                f"host config helper not found: {self._host_helpers}"
            )

    def _host_config_path(self) -> tuple[Path, bool]:
        explicit_value = self._environ.get("NAVMIN_HOST_CONFIG", "")
        if explicit_value:
            explicit = Path(explicit_value)
            if not explicit.is_absolute():
                explicit = self._repo_root / explicit
            return explicit, True
        return self._repo_root / "config" / "host.env", False

    def _ensure_host_config(self) -> None:
        host_config, explicit = self._host_config_path()
        if host_config.is_file():
            return
        if explicit:
            raise HostEnvironmentError(
                f"explicit host config not found: {host_config}"
            )

        default = self._repo_root / "config" / "defaults" / "host.env"
        if not default.is_file():
            raise HostEnvironmentError(f"safe host default not found: {default}")
        host_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default, host_config)
        raise HostConfigurationCreated(host_config)

    def prepare(self, *, camera_network_required: bool) -> None:
        if not camera_network_required or self._network_prepared:
            return
        self._require_helpers()
        self._ensure_host_config()
        command = (str(self._network_up),)
        returncode = self._run_command(command, self._repo_root, self._environ)
        if returncode != 0:
            raise HostCommandError(command, returncode)
        self._network_prepared = True

    def restore(self) -> None:
        if not self._network_prepared:
            return
        command = (str(self._network_down),)
        returncode = self._run_command(command, self._repo_root, self._environ)
        if returncode != 0:
            raise HostCommandError(command, returncode)
        self._network_prepared = False


class WindowsHostEnvironment(HostEnvironment):
    """Current Windows host backend intentionally performs no network setup."""

    def prepare(self, *, camera_network_required: bool) -> None:
        del camera_network_required

    def restore(self) -> None:
        return


def create_host_environment(
    repo_root: Path,
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _subprocess_runner,
) -> HostEnvironment:
    """Create the platform backend without invoking host preparation."""
    platform_value = sys.platform if platform_name is None else platform_name
    if platform_value.startswith("linux"):
        return LinuxHostEnvironment(
            repo_root,
            environ=environ,
            command_runner=command_runner,
        )
    if platform_value.startswith("win"):
        return WindowsHostEnvironment()
    raise HostEnvironmentError(f"unsupported host platform: {platform_value}")


__all__ = [
    "HostCommandError",
    "HostConfigurationCreated",
    "HostEnvironment",
    "HostEnvironmentError",
    "LinuxHostEnvironment",
    "WindowsHostEnvironment",
    "create_host_environment",
]
