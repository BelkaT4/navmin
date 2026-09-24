from __future__ import annotations

import logging
import subprocess
import sys
from io import StringIO
from threading import Thread

from navmin.lifecycle import StopToken, stop_and_join
from navmin.logging_setup import configure_logging


class _CooperativeWorker:
    def __init__(self) -> None:
        self._stop = StopToken()
        self._thread = Thread(target=self._run, name="test-worker")

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.request_stop()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        self._stop.wait()


def test_cooperative_worker_can_stop_and_join() -> None:
    worker = _CooperativeWorker()
    worker.start()

    assert stop_and_join(worker, timeout=1.0)
    assert not worker.is_alive()


def test_stop_token_is_idempotent_and_waitable() -> None:
    token = StopToken()
    assert not token.is_stop_requested()
    assert not token.wait(0.0)

    token.request_stop()
    token.request_stop()

    assert token.is_stop_requested()
    assert token.wait(0.0)


def test_logging_bootstrap_is_idempotent_for_navmin_logger_tree() -> None:
    app_logger = logging.getLogger("navmin")
    old_handlers = list(app_logger.handlers)
    old_level = app_logger.level
    old_propagate = app_logger.propagate
    stream = StringIO()

    try:
        app_logger.handlers.clear()
        configure_logging(logging.DEBUG, stream=stream)
        configure_logging(logging.DEBUG, stream=stream)

        logging.getLogger("navmin.test").debug("hello")
        output = stream.getvalue()

        assert output.count("hello") == 1
        assert len(app_logger.handlers) == 1
        assert "navmin.test" in output
    finally:
        app_logger.handlers.clear()
        app_logger.handlers.extend(old_handlers)
        app_logger.setLevel(old_level)
        app_logger.propagate = old_propagate


def test_package_module_help_executes_without_ui_or_hardware(tmp_path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "navmin", "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0
    assert "configured real camera sources" in completed.stdout
    assert completed.stderr == ""
