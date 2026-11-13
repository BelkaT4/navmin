"""Platform host preparation owned by the unified launcher boundary."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

LOGGER = logging.getLogger(__name__)

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
    completed = subprocess.run(
        tuple(command),
        cwd=cwd,
        env=dict(environ),
        check=False,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        LOGGER.info("host-helper stdout: %s", line)
    stderr_level = logging.ERROR if completed.returncode else logging.WARNING
    for line in completed.stderr.splitlines():
        LOGGER.log(stderr_level, "host-helper stderr: %s", line)
    return completed.returncode


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
        self._network_lock_file = None

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


    def _network_state_path(self) -> Path:
        explicit = self._environ.get("NAVMIN_NETWORK_STATE_FILE", "")
        if explicit:
            return Path(explicit)
        runtime_dir = self._environ.get("XDG_RUNTIME_DIR", "")
        if runtime_dir:
            return Path(runtime_dir) / "navmin" / "camera-network.state"
        return Path("/tmp") / f"navmin-{os.getuid()}" / "camera-network.state"

    def _acquire_network_lock(self) -> None:
        if self._network_lock_file is not None:
            return
        import fcntl

        lock_path = self._network_state_path().with_name("camera-network.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise HostEnvironmentError(
                "camera network is already owned by another NavMin process: "
                f"{lock_path}"
            ) from exc
        self._network_lock_file = lock_file
        LOGGER.info("Camera network lifecycle lock acquired: %s", lock_path)

    def _release_network_lock(self) -> None:
        lock_file = self._network_lock_file
        if lock_file is None:
            return
        import fcntl

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            self._network_lock_file = None
        LOGGER.info("Camera network lifecycle lock released")

    def _run_network_helper(self, path: Path, *, action: str) -> None:
        command = (str(path),)
        LOGGER.info("Camera network %s starting: %s", action, path.name)
        returncode = self._run_command(command, self._repo_root, self._environ)
        if returncode != 0:
            LOGGER.error(
                "Camera network %s failed helper=%s exit_status=%d",
                action,
                path.name,
                returncode,
            )
            raise HostCommandError(command, returncode)
        LOGGER.info("Camera network %s complete: %s", action, path.name)

    def prepare(self, *, camera_network_required: bool) -> None:
        if not camera_network_required or self._network_prepared:
            return
        self._require_helpers()
        self._ensure_host_config()
        self._acquire_network_lock()
        try:
            # Recover stale state left by a crashed/aborted previous invocation.
            # The process-wide lock makes this safe: a live NavMin owner cannot
            # have its camera network torn down by a second instance.
            self._run_network_helper(self._network_down, action="pre-start recovery")
            self._run_network_helper(self._network_up, action="activation")
        except HostEnvironmentError:
            self._release_network_lock()
            raise
        self._network_prepared = True

    def restore(self) -> None:
        if not self._network_prepared:
            self._release_network_lock()
            return
        try:
            self._run_network_helper(self._network_down, action="restore")
        finally:
            self._network_prepared = False
            self._release_network_lock()


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
