"""Structured VERBOSE diagnostic events for NavMin runtime evidence."""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import asdict, is_dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from threading import Lock
from time import monotonic_ns
from typing import Any

_SCHEMA_VERSION = 1
_METRIC_WINDOW_SIZE = 2048
_STATE_LOCK = Lock()
_ENABLED = False
_RUN_ID: str | None = None
_SEQUENCES: dict[str, int] = {}
_EVENT_COUNTS: Counter[str] = Counter()
_FRAME_COUNTS_BY_CAMERA: Counter[str] = Counter()
_LAST_FRAME_KEYS: dict[str, dict[str, int | str]] = {}
_FRAME_DROP_COUNTERS: Counter[str] = Counter()
_LAST_CONTROL_ID: int | None = None
_LAST_REQUEST_ID_BY_CONNECTION_EPOCH: dict[str, int] = {}
_UART_ERROR_COUNTERS: Counter[str] = Counter()
_UART_RETRY_COUNTERS: Counter[str] = Counter()
_VISION_PROCESSING_MS: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)
_CONTROL_DT_MS: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)
_UART_RTT_MS: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)
_RUN_OUTCOME: dict[str, Any] | None = None


def configure_diagnostic_events(*, enabled: bool, run_id: str | None) -> None:
    """Set one process-local VERBOSE event context for the current launcher run."""
    global _ENABLED, _RUN_ID, _LAST_CONTROL_ID, _RUN_OUTCOME
    if enabled and not run_id:
        raise ValueError("run_id is required when diagnostic events are enabled")
    with _STATE_LOCK:
        _ENABLED = bool(enabled)
        _RUN_ID = run_id if enabled else None
        _SEQUENCES.clear()
        _EVENT_COUNTS.clear()
        _FRAME_COUNTS_BY_CAMERA.clear()
        _LAST_FRAME_KEYS.clear()
        _FRAME_DROP_COUNTERS.clear()
        _LAST_CONTROL_ID = None
        _LAST_REQUEST_ID_BY_CONNECTION_EPOCH.clear()
        _UART_ERROR_COUNTERS.clear()
        _UART_RETRY_COUNTERS.clear()
        _VISION_PROCESSING_MS.clear()
        _CONTROL_DT_MS.clear()
        _UART_RTT_MS.clear()
        _RUN_OUTCOME = None


def diagnostic_events_enabled() -> bool:
    with _STATE_LOCK:
        return _ENABLED and _RUN_ID is not None


def record_run_outcome(
    *,
    status: str,
    exit_code: int,
    completed: bool,
    failure_type: str | None = None,
    failure_message: str | None = None,
    cleanup_failure_type: str | None = None,
    cleanup_failure_message: str | None = None,
) -> None:
    """Remember the launcher-owned final outcome for the VERBOSE integrity event."""
    global _RUN_OUTCOME
    with _STATE_LOCK:
        if not _ENABLED or _RUN_ID is None:
            return
        _RUN_OUTCOME = {
            "status": status,
            "exit_code": int(exit_code),
            "completed": bool(completed),
            "failure_type": failure_type,
            "failure_message": failure_message,
            "cleanup_failure_type": cleanup_failure_type,
            "cleanup_failure_message": cleanup_failure_message,
        }


def diagnostic_performance_snapshot() -> dict[str, Any]:
    """Return bounded rolling metrics derived from already-emitted events."""
    with _STATE_LOCK:
        return {
            "frame_counts_by_camera": dict(_FRAME_COUNTS_BY_CAMERA),
            "vision_processing_p50_ms": _percentile(_VISION_PROCESSING_MS, 0.50),
            "vision_processing_p95_ms": _percentile(_VISION_PROCESSING_MS, 0.95),
            "vision_processing_p99_ms": _percentile(_VISION_PROCESSING_MS, 0.99),
            "control_dt_p50_ms": _percentile(_CONTROL_DT_MS, 0.50),
            "control_dt_p95_ms": _percentile(_CONTROL_DT_MS, 0.95),
            "control_dt_p99_ms": _percentile(_CONTROL_DT_MS, 0.99),
            "uart_rtt_p50_ms": _percentile(_UART_RTT_MS, 0.50),
            "uart_rtt_p95_ms": _percentile(_UART_RTT_MS, 0.95),
            "pipeline_drop_counters": dict(_FRAME_DROP_COUNTERS),
        }


def diagnostic_integrity_snapshot() -> dict[str, Any]:
    """Return the current bounded run-integrity evidence accumulated on the PC."""
    with _STATE_LOCK:
        outcome = None if _RUN_OUTCOME is None else dict(_RUN_OUTCOME)
        return {
            "run_outcome": outcome,
            "event_counts": dict(_EVENT_COUNTS),
            "frame_drop_counters": dict(_FRAME_DROP_COUNTERS),
            "last_frame_keys": {
                camera: dict(frame_key)
                for camera, frame_key in _LAST_FRAME_KEYS.items()
            },
            "last_control_id": _LAST_CONTROL_ID,
            "last_request_id_by_connection_epoch": dict(
                _LAST_REQUEST_ID_BY_CONNECTION_EPOCH
            ),
            "uart_error_counters": dict(_UART_ERROR_COUNTERS),
            "uart_retry_counters": dict(_UART_RETRY_COUNTERS),
        }


def build_event_message(
    event_name: str,
    *,
    source_instance: str,
    event_monotonic_ns: int | None = None,
    **fields: Any,
) -> str | None:
    """Build one compact single-line JSON event, or ``None`` outside VERBOSE."""
    if (
        not event_name
        or not event_name.replace("_", "").isalnum()
        or event_name.upper() != event_name
    ):
        raise ValueError("event_name must be non-empty UPPER_SNAKE_CASE")
    if not source_instance:
        raise ValueError("source_instance must not be empty")
    with _STATE_LOCK:
        if not _ENABLED or _RUN_ID is None:
            return None
        sequence = _SEQUENCES.get(source_instance, 0) + 1
        _SEQUENCES[source_instance] = sequence
        run_id = _RUN_ID
        _note_event_locked(event_name, fields)
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "source_instance": source_instance,
        "sequence": sequence,
        "event_monotonic_ns": (
            monotonic_ns() if event_monotonic_ns is None else int(event_monotonic_ns)
        ),
    }
    payload.update(fields)
    normalized = _normalize(payload)
    return f"{event_name} " + json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        sort_keys=True,
    )


def emit_event(
    logger: logging.Logger,
    event_name: str,
    *,
    source_instance: str,
    event_monotonic_ns: int | None = None,
    **fields: Any,
) -> None:
    """Emit one structured DEBUG event when the VERBOSE contract is active."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    message = build_event_message(
        event_name,
        source_instance=source_instance,
        event_monotonic_ns=event_monotonic_ns,
        **fields,
    )
    if message is not None:
        logger.debug("%s", message)


def _note_event_locked(event_name: str, fields: dict[str, Any]) -> None:
    global _LAST_CONTROL_ID
    _EVENT_COUNTS[event_name] += 1

    if event_name == "FRAME_INGRESS":
        camera = fields.get("camera")
        generation = fields.get("generation")
        frame_id = fields.get("frame_id")
        if isinstance(camera, str):
            _FRAME_COUNTS_BY_CAMERA[camera] += 1
            if isinstance(generation, int) and isinstance(frame_id, int):
                _LAST_FRAME_KEYS[camera] = {
                    "camera": camera,
                    "generation": generation,
                    "frame_id": frame_id,
                }
    elif event_name == "FRAME_DROP":
        camera = str(fields.get("camera") or "unknown")
        reason = str(
            fields.get("drop_reason")
            or fields.get("reason")
            or fields.get("result_drop_reason")
            or "unknown"
        )
        _FRAME_DROP_COUNTERS[f"{camera}:{reason}"] += 1
    elif event_name == "FRAME_PROCESS":
        _append_finite(_VISION_PROCESSING_MS, fields.get("processing_duration_ms"))
    elif event_name == "TRACK_CONTROL":
        control_id = fields.get("control_id")
        if isinstance(control_id, int):
            _LAST_CONTROL_ID = control_id
        _append_finite(_CONTROL_DT_MS, fields.get("dt_ms"))
    elif event_name in {"UART_TX", "UART_RX", "UART_RETRY"}:
        connection_epoch = fields.get("connection_epoch")
        request_id = fields.get("request_id")
        if isinstance(connection_epoch, int) and isinstance(request_id, int):
            _LAST_REQUEST_ID_BY_CONNECTION_EPOCH[str(connection_epoch)] = request_id
        if event_name == "UART_RX":
            _append_finite(_UART_RTT_MS, fields.get("round_trip_ms"))
            result = fields.get("transaction_result")
            if result not in (None, "ok"):
                _UART_ERROR_COUNTERS[str(result)] += 1
        elif event_name == "UART_RETRY":
            reason = str(fields.get("reason") or "unknown")
            _UART_RETRY_COUNTERS[reason] += 1


def _append_finite(target: deque[float], value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return
    numeric = float(value)
    if isfinite(numeric):
        target.append(numeric)


def _percentile(values: deque[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_normalize(item) for item in value]
    return str(value)


__all__ = [
    "build_event_message",
    "configure_diagnostic_events",
    "diagnostic_events_enabled",
    "diagnostic_integrity_snapshot",
    "diagnostic_performance_snapshot",
    "emit_event",
    "record_run_outcome",
]
