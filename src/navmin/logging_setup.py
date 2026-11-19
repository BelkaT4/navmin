"""Application logging bootstrap for NavMin."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic, process_time, sleep
from typing import TextIO

from navmin.diagnostic_logging import (
    build_event_message,
    configure_diagnostic_events,
    diagnostic_integrity_snapshot,
    diagnostic_performance_snapshot,
    emit_event,
)

_LOGGER_NAME = "navmin"
_HANDLER_NAME = "navmin-default"
_SESSION_HANDLER_NAME = "navmin-session-file"
_SESSION_QUEUE_HANDLER_NAME = "navmin-session-queue"
_CONFIG_LOCK = Lock()
_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"
SESSION_LOG_MAX_BYTES = 10 * 1024 * 1024
SESSION_LOG_BACKUP_COUNT = 5
VERBOSE_SESSION_LOG_MAX_BYTES = 100 * 1024 * 1024
VERBOSE_SESSION_LOG_BACKUP_COUNT = 10
VERBOSE_LOG_QUEUE_CAPACITY = 8192
VERBOSE_LOG_SHUTDOWN_DRAIN_SECONDS = 2.0
VERBOSE_PERF_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class VerboseLoggingHealth:
    queue_depth: int
    queue_high_watermark: int
    queue_capacity_records: int
    dropped_total: int


class _QueueStats:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._lock = Lock()
        self.high_watermark = 0
        self.dropped_total = 0
        self.dropped_by_level: Counter[str] = Counter()
        self.dropped_by_event: Counter[str] = Counter()

    def note_enqueue(self, depth: int) -> None:
        with self._lock:
            self.high_watermark = max(self.high_watermark, depth)

    def note_drop(self, record: logging.LogRecord) -> None:
        message = record.msg if isinstance(record.msg, str) else ""
        token = message.split(" ", 1)[0] if message else "UNSTRUCTURED"
        if not token or token.upper() != token:
            token = "UNSTRUCTURED"
        with self._lock:
            self.dropped_total += 1
            self.dropped_by_level[record.levelname] += 1
            self.dropped_by_event[token] += 1

    def snapshot(self) -> tuple[int, dict[str, int], dict[str, int], int]:
        with self._lock:
            return (
                self.dropped_total,
                dict(self.dropped_by_level),
                dict(self.dropped_by_event),
                self.high_watermark,
            )


class _VerbosePerfReporter:
    """Emit low-rate VERBOSE performance snapshots without blocking producers."""

    def __init__(self, *, interval_s: float = VERBOSE_PERF_INTERVAL_SECONDS) -> None:
        if interval_s <= 0.0:
            raise ValueError("interval_s must be > 0")
        self._interval_s = float(interval_s)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_wall_s = monotonic()
        self._last_cpu_s = process_time()
        self._last_frame_counts: dict[str, int] = {}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VERBOSE performance reporter is one-shot")
        self._thread = Thread(
            target=self._run,
            name="verbose-log-health",
            daemon=False,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    def emit_once(self) -> None:
        now_wall_s = monotonic()
        now_cpu_s = process_time()
        elapsed_s = now_wall_s - self._last_wall_s
        cpu_delta_s = now_cpu_s - self._last_cpu_s
        cpu_percent = (
            None
            if elapsed_s <= 0.0
            else max(0.0, cpu_delta_s / elapsed_s * 100.0)
        )
        self._last_wall_s = now_wall_s
        self._last_cpu_s = now_cpu_s

        metrics = diagnostic_performance_snapshot()
        frame_counts = metrics.pop("frame_counts_by_camera")
        camera_fps: dict[str, float] = {}
        if elapsed_s > 0.0:
            for camera, count in frame_counts.items():
                previous = self._last_frame_counts.get(camera, 0)
                camera_fps[camera] = max(0, count - previous) / elapsed_s
        self._last_frame_counts = dict(frame_counts)

        health = verbose_logging_health()
        emit_event(
            logging.getLogger("navmin.logging_setup"),
            "PERF_SNAPSHOT",
            source_instance="logging:performance",
            process_cpu_percent=cpu_percent,
            process_rss_bytes=_current_process_rss_bytes(),
            thread_timing_summary=None,
            camera_fps=camera_fps,
            vision_processing_p50_ms=metrics["vision_processing_p50_ms"],
            vision_processing_p95_ms=metrics["vision_processing_p95_ms"],
            vision_processing_p99_ms=metrics["vision_processing_p99_ms"],
            control_dt_p50_ms=metrics["control_dt_p50_ms"],
            control_dt_p95_ms=metrics["control_dt_p95_ms"],
            control_dt_p99_ms=metrics["control_dt_p99_ms"],
            uart_rtt_p50_ms=metrics["uart_rtt_p50_ms"],
            uart_rtt_p95_ms=metrics["uart_rtt_p95_ms"],
            logging_queue_depth=None if health is None else health.queue_depth,
            logging_queue_high_watermark=(
                None if health is None else health.queue_high_watermark
            ),
            recording_drop_count=None,
            pipeline_drop_counters=metrics["pipeline_drop_counters"],
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self.emit_once()


def _current_process_rss_bytes() -> int | None:
    """Best-effort current RSS without adding a runtime dependency."""
    try:
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(statm) < 2:
            return None
        return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


class _NonBlockingQueueHandler(QueueHandler):
    def __init__(self, queue: Queue[logging.LogRecord], stats: _QueueStats) -> None:
        super().__init__(queue)
        self._stats = stats

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except Full:
            self._stats.note_drop(record)
            return
        self._stats.note_enqueue(self.queue.qsize())

    def handleError(self, record: logging.LogRecord) -> None:
        # VERBOSE diagnostics must never disturb the runtime because stderr itself
        # is slow or because logging.raiseExceptions is enabled in development.
        self._stats.note_drop(record)


_ACTIVE_QUEUE: Queue[logging.LogRecord] | None = None
_ACTIVE_QUEUE_STATS: _QueueStats | None = None
_ACTIVE_QUEUE_LOCK = Lock()


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one reusable stderr/stream handler for the NavMin logger tree."""
    logger = logging.getLogger(_LOGGER_NAME)

    with _CONFIG_LOCK:
        handler = next(
            (item for item in logger.handlers if item.get_name() == _HANDLER_NAME),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler(stream)
            handler.set_name(_HANDLER_NAME)
            handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(handler)
        elif stream is not None and isinstance(handler, logging.StreamHandler):
            handler.setStream(stream)

        logger.setLevel(level)
        handler.setLevel(level)
        logger.propagate = False

    return logger


def verbose_logging_health() -> VerboseLoggingHealth | None:
    """Return a bounded snapshot of the active VERBOSE file queue, if any."""
    with _ACTIVE_QUEUE_LOCK:
        queue = _ACTIVE_QUEUE
        stats = _ACTIVE_QUEUE_STATS
    if queue is None or stats is None:
        return None
    dropped_total, _by_level, _by_event, high_watermark = stats.snapshot()
    return VerboseLoggingHealth(
        queue_depth=queue.qsize(),
        queue_high_watermark=high_watermark,
        queue_capacity_records=stats.capacity,
        dropped_total=dropped_total,
    )


def _set_active_queue(
    queue: Queue[logging.LogRecord] | None,
    stats: _QueueStats | None,
) -> None:
    global _ACTIVE_QUEUE, _ACTIVE_QUEUE_STATS
    with _ACTIVE_QUEUE_LOCK:
        _ACTIVE_QUEUE = queue
        _ACTIVE_QUEUE_STATS = stats


def _direct_event(
    handler: logging.Handler,
    event_name: str,
    *,
    source_instance: str,
    **fields: object,
) -> None:
    message = build_event_message(
        event_name,
        source_instance=source_instance,
        **fields,
    )
    if message is None:
        return
    record = logging.LogRecord(
        name="navmin.logging_setup",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    handler.handle(record)


def _drain_or_discard(
    queue: Queue[logging.LogRecord], *, timeout_s: float
) -> int:
    deadline = monotonic() + timeout_s
    while not queue.empty() and monotonic() < deadline:
        sleep(0.005)
    if queue.empty():
        return 0
    undrained = 0
    while True:
        try:
            queue.get_nowait()
        except Empty:
            return undrained
        else:
            undrained += 1
            try:
                queue.task_done()
            except ValueError:
                pass


@contextmanager
def session_file_logging(
    path: Path,
    *,
    level: int,
    run_id: str | None = None,
    verbose: bool | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    integrity_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Iterator[Path]:
    """Attach one bounded session file while retaining the INFO console handler.

    NORMAL keeps the historical synchronous rotating file. VERBOSE routes the
    DEBUG-capable file stream through a bounded non-blocking queue so disk I/O
    cannot stall Vision/Core/Turret producer threads.
    """
    verbose_enabled = level <= logging.DEBUG if verbose is None else bool(verbose)
    selected_max_bytes = (
        VERBOSE_SESSION_LOG_MAX_BYTES
        if verbose_enabled and max_bytes is None
        else SESSION_LOG_MAX_BYTES
        if max_bytes is None
        else max_bytes
    )
    selected_backup_count = (
        VERBOSE_SESSION_LOG_BACKUP_COUNT
        if verbose_enabled and backup_count is None
        else SESSION_LOG_BACKUP_COUNT
        if backup_count is None
        else backup_count
    )
    if selected_max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if selected_backup_count < 1:
        raise ValueError("backup_count must be >= 1")

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    with _CONFIG_LOCK:
        previous_level = logger.level
        previous_propagate = logger.propagate
        previous_console = next(
            (item for item in logger.handlers if item.get_name() == _HANDLER_NAME),
            None,
        )
        previous_console_level = (
            previous_console.level if previous_console is not None else None
        )

    logger = configure_logging(logging.INFO)
    file_handler = RotatingFileHandler(
        path,
        maxBytes=selected_max_bytes,
        backupCount=selected_backup_count,
        encoding="utf-8",
    )
    file_handler.set_name(_SESSION_HANDLER_NAME)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FORMAT))

    effective_run_id = run_id or path.parent.name
    configure_diagnostic_events(
        enabled=verbose_enabled,
        run_id=effective_run_id if verbose_enabled else None,
    )

    queue: Queue[logging.LogRecord] | None = None
    queue_handler: _NonBlockingQueueHandler | None = None
    listener: QueueListener | None = None
    stats: _QueueStats | None = None
    perf_reporter: _VerbosePerfReporter | None = None
    attached_handler: logging.Handler
    if verbose_enabled:
        queue = Queue(maxsize=VERBOSE_LOG_QUEUE_CAPACITY)
        stats = _QueueStats(VERBOSE_LOG_QUEUE_CAPACITY)
        queue_handler = _NonBlockingQueueHandler(queue, stats)
        queue_handler.set_name(_SESSION_QUEUE_HANDLER_NAME)
        queue_handler.setLevel(level)
        listener = QueueListener(queue, file_handler, respect_handler_level=True)
        listener.start()
        _set_active_queue(queue, stats)
        perf_reporter = _VerbosePerfReporter()
        perf_reporter.start()
        attached_handler = queue_handler
    else:
        attached_handler = file_handler

    with _CONFIG_LOCK:
        logger.setLevel(min(logging.INFO, level))
        logger.addHandler(attached_handler)

    if verbose_enabled:
        emit_event(
            logging.getLogger("navmin.logging_setup"),
            "LOGGING_SESSION_START",
            source_instance="logging:session",
            logging_mode="verbose",
            file_level="debug",
            console_level="info",
            queue_capacity_records=VERBOSE_LOG_QUEUE_CAPACITY,
            rotation_max_bytes=selected_max_bytes,
            rotation_backup_count=selected_backup_count,
            format_version=1,
        )

    try:
        yield path
    finally:
        if perf_reporter is not None:
            perf_reporter.stop()
        with _CONFIG_LOCK:
            logger.removeHandler(attached_handler)

        undrained = 0
        if verbose_enabled and queue is not None and listener is not None and stats is not None:
            undrained = _drain_or_discard(
                queue,
                timeout_s=VERBOSE_LOG_SHUTDOWN_DRAIN_SECONDS,
            )
            listener.stop()
            dropped_total, dropped_by_level, dropped_by_event, high_watermark = (
                stats.snapshot()
            )
            if dropped_total or undrained:
                _direct_event(
                    file_handler,
                    "LOGGING_DROP_SUMMARY",
                    source_instance="logging:session",
                    dropped_total=dropped_total,
                    dropped_by_level=dropped_by_level,
                    dropped_by_event=dropped_by_event,
                    queue_high_watermark=high_watermark,
                    queue_capacity_records=VERBOSE_LOG_QUEUE_CAPACITY,
                    undrained_on_shutdown=undrained,
                )

            integrity = diagnostic_integrity_snapshot()
            outcome = integrity.get("run_outcome") or {}
            notes: list[str] = []
            if not outcome:
                notes.append("launcher outcome unavailable before logging shutdown")
            notes.extend(
                [
                    "recording drop aggregate unavailable in Stage 2 PC logging",
                    "IMU integrity counters unavailable until IMU instrumentation stage",
                    "STM32 execution evidence unavailable until firmware telemetry stage",
                ]
            )
            integrity_summary: dict[str, object] = {
                "completed": bool(outcome.get("completed", False)),
                "status": outcome.get("status", "unfinalized"),
                "exit_code": outcome.get("exit_code"),
                "failure_type": outcome.get("failure_type"),
                "failure_message": outcome.get("failure_message"),
                "cleanup_failure_type": outcome.get("cleanup_failure_type"),
                "cleanup_failure_message": outcome.get("cleanup_failure_message"),
                "logging_dropped_total": dropped_total,
                "logging_undrained_total": undrained,
                "logging_queue_high_watermark": high_watermark,
                "logging_queue_capacity_records": VERBOSE_LOG_QUEUE_CAPACITY,
                "frame_drop_counters": integrity["frame_drop_counters"],
                "recording_drop_counters": {},
                "uart_error_counters": integrity["uart_error_counters"],
                "uart_retry_counters": integrity["uart_retry_counters"],
                "imu_drop_counters": {},
                "imu_error_counters": {},
                "max_observed_queue_depths": {"logging": high_watermark},
                "last_frame_keys": integrity["last_frame_keys"],
                "last_control_id": integrity["last_control_id"],
                "last_request_id_by_connection_epoch": integrity[
                    "last_request_id_by_connection_epoch"
                ],
                "event_counts": integrity["event_counts"],
                "notes": notes,
            }
            _direct_event(
                file_handler,
                "RUN_INTEGRITY",
                source_instance="logging:session",
                **integrity_summary,
            )
            if integrity_callback is not None:
                try:
                    integrity_callback(integrity_summary)
                except OSError:
                    logging.getLogger("navmin.logging_setup").exception(
                        "Failed to persist VERBOSE integrity summary to manifest"
                    )
            _set_active_queue(None, None)

        with _CONFIG_LOCK:
            if previous_console is None:
                current_console = next(
                    (
                        item
                        for item in logger.handlers
                        if item.get_name() == _HANDLER_NAME
                    ),
                    None,
                )
                if current_console is not None:
                    logger.removeHandler(current_console)
                    current_console.close()
            elif previous_console_level is not None:
                previous_console.setLevel(previous_console_level)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
        file_handler.close()
        configure_diagnostic_events(enabled=False, run_id=None)


__all__ = [
    "SESSION_LOG_BACKUP_COUNT",
    "SESSION_LOG_MAX_BYTES",
    "VERBOSE_LOG_QUEUE_CAPACITY",
    "VERBOSE_PERF_INTERVAL_SECONDS",
    "VERBOSE_SESSION_LOG_BACKUP_COUNT",
    "VERBOSE_SESSION_LOG_MAX_BYTES",
    "VerboseLoggingHealth",
    "configure_logging",
    "session_file_logging",
    "verbose_logging_health",
]
