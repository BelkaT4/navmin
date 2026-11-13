"""In-runtime effective video-latency calibration using production Overview frames.

The calibration never opens a second camera or serial transport.  It observes
``VisionPipeline.latest_result`` while ``TurretWorker`` retains sole UART
ownership and grants a temporary calibration-only Y-velocity channel.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from statistics import median
from threading import Event, Thread
from time import monotonic, sleep
from typing import Any, Protocol

from navmin.concurrency import LatestValue
from navmin.contracts import (
    BBox,
    CameraModel,
    CameraRole,
    MotorState,
    PlatformMotionSnapshot,
    TurretConnectionState,
    TurretState,
    VisionResult,
)
from navmin.diagnostics.video_latency_calibration import (
    OpticalFlowLatencyConfig,
    OpticalFlowRateObservationY,
    VideoLatencyAggregate,
    VideoLatencyCalibrationError,
    VideoLatencyEstimate,
    aggregate_latency_estimates,
    estimate_optical_flow_rate_y,
    estimate_video_latency_y,
)

_LOGGER = logging.getLogger(__name__)


class LiveVideoLatencyCalibrationError(RuntimeError):
    """In-runtime calibration could not be completed safely or validly."""


class LiveCalibrationPhase(Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class LiveVideoLatencyCalibrationConfig:
    command_rate_deg_s: float = 6.0
    segment_duration_s: float = 0.8
    run_count: int = 3
    warmup_s: float = 1.0
    final_settle_s: float = 0.8
    poll_interval_s: float = 0.004
    ownership_timeout_s: float = 2.0
    max_command_rate_deg_s: float = 20.0

    def validate(self) -> None:
        for name in (
            "command_rate_deg_s",
            "segment_duration_s",
            "warmup_s",
            "final_settle_s",
            "poll_interval_s",
            "ownership_timeout_s",
            "max_command_rate_deg_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.command_rate_deg_s > self.max_command_rate_deg_s:
            raise ValueError("command_rate_deg_s exceeds diagnostic safety bound")
        if isinstance(self.run_count, bool) or not isinstance(self.run_count, int):
            raise TypeError("run_count must be an integer")
        if not 1 <= self.run_count <= 10:
            raise ValueError("run_count must be in [1, 10]")


@dataclass(frozen=True, slots=True)
class MotionCommandStep:
    offset_s: float
    rate_y_deg_s: float


@dataclass(frozen=True, slots=True)
class LiveCalibrationRunResult:
    run_index: int
    flow_observations: int
    rejected_frame_pairs: int
    median_inliers: float | None
    timestamp_basis: str
    estimate: VideoLatencyEstimate


@dataclass(frozen=True, slots=True)
class LiveVideoLatencyCalibrationResult:
    status: str
    camera: str
    axis: str
    configured_video_latency_ms: int
    command_rate_deg_s: float
    segment_duration_s: float
    run_count: int
    runs: tuple[LiveCalibrationRunResult, ...]
    aggregate: VideoLatencyAggregate
    recommended_video_latency_ms: int | None
    conclusion: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LiveVideoLatencyCalibrationState:
    phase: LiveCalibrationPhase
    message: str
    run_index: int = 0
    run_count: int = 0
    current_rate_y_deg_s: float = 0.0
    flow_observations: int = 0
    result: LiveVideoLatencyCalibrationResult | None = None


class CalibrationTurretPort(Protocol):
    @property
    def current_state(self) -> TurretState: ...

    @property
    def video_latency_calibration_active(self) -> bool: ...

    def begin_video_latency_calibration(self) -> bool: ...

    def submit_video_latency_calibration_velocity_y(
        self, rate_y_deg_s: float
    ) -> bool: ...

    def end_video_latency_calibration(self) -> bool: ...

    def request_emergency(self) -> None: ...

    def platform_motion_snapshot(self) -> PlatformMotionSnapshot: ...


CameraModelProvider = Callable[[int], CameraModel | None]
ResultWriter = Callable[[LiveVideoLatencyCalibrationResult], None]


def build_motion_profile(
    config: LiveVideoLatencyCalibrationConfig,
) -> tuple[MotionCommandStep, ...]:
    """Build a symmetric logical-Y profile ending at zero velocity."""
    config.validate()
    d = float(config.segment_duration_s)
    v = float(config.command_rate_deg_s)
    return (
        MotionCommandStep(0.0, +v),
        MotionCommandStep(1.0 * d, -v),
        MotionCommandStep(2.0 * d, +v),
        MotionCommandStep(3.0 * d, -v),
        MotionCommandStep(4.0 * d, 0.0),
    )


def motion_profile_duration_s(config: LiveVideoLatencyCalibrationConfig) -> float:
    return build_motion_profile(config)[-1].offset_s + config.final_settle_s


def _source_interval(previous: VisionResult, current: VisionResult) -> tuple[int, str]:
    previous_source = previous.frame.source_timestamp
    current_source = current.frame.source_timestamp
    if (
        previous_source is not None
        and current_source is not None
        and previous_source.source is current_source.source
    ):
        interval = current_source.timestamp_ns - previous_source.timestamp_ns
        if interval > 0:
            return interval, current_source.source.value
    interval = current.frame.receive_timestamp_ns - previous.frame.receive_timestamp_ns
    if interval <= 0:
        raise LiveVideoLatencyCalibrationError("camera timestamps are not increasing")
    return interval, "receive-time"


def _exclusions(previous: VisionResult, current: VisionResult) -> tuple[BBox, ...]:
    return tuple(
        tracked.bbox
        for result in (previous, current)
        for tracked in result.tracked_objects
    )


class InRuntimeVideoLatencyCalibration:
    """One-shot-at-a-time calibration service sharing production owners."""

    def __init__(
        self,
        *,
        overview_results: LatestValue[VisionResult],
        camera_model_provider: CameraModelProvider,
        turret: CalibrationTurretPort,
        configured_video_latency_ms: int,
        config: LiveVideoLatencyCalibrationConfig | None = None,
        optical_config: OpticalFlowLatencyConfig | None = None,
        result_writer: ResultWriter | None = None,
    ) -> None:
        if not isinstance(overview_results, LatestValue):
            raise TypeError("overview_results must be LatestValue[VisionResult]")
        if isinstance(configured_video_latency_ms, bool) or not isinstance(
            configured_video_latency_ms, int
        ):
            raise TypeError("configured_video_latency_ms must be an integer")
        if configured_video_latency_ms < 0:
            raise ValueError("configured_video_latency_ms must be >= 0")
        self._overview_results = overview_results
        self._camera_model_provider = camera_model_provider
        self._turret = turret
        self._configured_video_latency_ms = configured_video_latency_ms
        self._config = config or LiveVideoLatencyCalibrationConfig()
        self._config.validate()
        self._optical_config = optical_config or OpticalFlowLatencyConfig()
        self._optical_config.validate()
        self._result_writer = result_writer
        self._cancel = Event()
        self._thread: Thread | None = None
        self._states: LatestValue[LiveVideoLatencyCalibrationState] = LatestValue()
        self._states.publish(
            LiveVideoLatencyCalibrationState(
                LiveCalibrationPhase.IDLE,
                "Готово к калибровке задержки видео "
                "по оси Y.",
                run_count=self._config.run_count,
            )
        )

    @property
    def state_updates(self) -> LatestValue[LiveVideoLatencyCalibrationState]:
        return self._states

    @property
    def active(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def set_result_writer(self, writer: ResultWriter | None) -> None:
        if writer is not None and not callable(writer):
            raise TypeError("writer must be callable or None")
        if self.active:
            raise RuntimeError("cannot replace result writer during calibration")
        self._result_writer = writer

    def start(self) -> bool:
        if self.active:
            return False
        state = self._turret.current_state
        if (
            state.connection_state is not TurretConnectionState.READY
            or state.motor_state is not MotorState.ON
        ):
            raise LiveVideoLatencyCalibrationError(
                "калибровка требует READY и MOTOR ON"
            )
        latest = self._overview_results.get()
        if latest is None or latest.frame.camera is not CameraRole.OVERVIEW:
            raise LiveVideoLatencyCalibrationError(
                "нет текущего кадра Overview"
            )
        if not self._turret.begin_video_latency_calibration():
            raise LiveVideoLatencyCalibrationError(
                "TurretWorker не предоставил эксклюзивное "
                "управление калибратору"
            )
        self._cancel.clear()
        self._thread = Thread(
            target=self._run,
            name="video-latency-calibration",
            daemon=False,
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def shutdown(self, timeout: float = 2.0) -> None:
        self.cancel()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                self._turret.request_emergency()
                raise LiveVideoLatencyCalibrationError(
                    "video latency calibration did not stop within timeout"
                )

    def _publish(
        self,
        phase: LiveCalibrationPhase,
        message: str,
        *,
        run_index: int = 0,
        current_rate_y_deg_s: float = 0.0,
        flow_observations: int = 0,
        result: LiveVideoLatencyCalibrationResult | None = None,
    ) -> None:
        self._states.publish(
            LiveVideoLatencyCalibrationState(
                phase=phase,
                message=message,
                run_index=run_index,
                run_count=self._config.run_count,
                current_rate_y_deg_s=current_rate_y_deg_s,
                flow_observations=flow_observations,
                result=result,
            )
        )

    def _run(self) -> None:
        try:
            self._wait_for_ownership()
            self._publish(
                LiveCalibrationPhase.WARMUP,
                "Проверка текущего Overview-потока "
                "и фоновых признаков…",
            )
            self._warmup()
            runs = tuple(
                self._collect_run(run_index)
                for run_index in range(1, self._config.run_count + 1)
            )
            aggregate = aggregate_latency_estimates(
                tuple(run.estimate for run in runs)
            )
            minimum_valid = max(1, (self._config.run_count + 1) // 2)
            passed = aggregate.valid_run_count >= minimum_valid
            recommended = (
                None
                if not passed or aggregate.median_latency_ms is None
                else round(aggregate.median_latency_ms)
            )
            result = LiveVideoLatencyCalibrationResult(
                status="PASS" if passed else "FAIL",
                camera=CameraRole.OVERVIEW.value,
                axis="y",
                configured_video_latency_ms=self._configured_video_latency_ms,
                command_rate_deg_s=self._config.command_rate_deg_s,
                segment_duration_s=self._config.segment_duration_s,
                run_count=self._config.run_count,
                runs=runs,
                aggregate=aggregate,
                recommended_video_latency_ms=recommended,
                conclusion=(
                    "Candidate measured on the active production Overview pipeline; "
                    "validate on hardware before persisting it."
                    if passed
                    else "Not enough valid runs; keep the existing video-latency-ms."
                ),
            )
            if self._result_writer is not None:
                self._result_writer(result)
            self._publish(
                LiveCalibrationPhase.COMPLETE,
                (
                    f"Готово: median={aggregate.median_latency_ms} мс, "
                    f"MAD={aggregate.mad_latency_ms} мс."
                    if passed
                    else (
                        "Калибровка завершилась без надёжной оценки."
                    )
                ),
                run_index=self._config.run_count,
                result=result,
            )
        except _Cancelled:
            self._publish(
                LiveCalibrationPhase.CANCELLED,
                "Калибровка отменена; команда скорости "
                "сброшена в ноль.",
            )
        except (LiveVideoLatencyCalibrationError, VideoLatencyCalibrationError) as exc:
            _LOGGER.warning("Video latency calibration failed: %s", exc)
            self._publish(LiveCalibrationPhase.FAILED, str(exc))
        finally:
            self._finish_ownership_safely()

    def _wait_for_ownership(self) -> None:
        deadline = monotonic() + self._config.ownership_timeout_s
        while monotonic() < deadline:
            self._check_cancel()
            if self._turret.video_latency_calibration_active:
                return
            sleep(self._config.poll_interval_s)
        raise LiveVideoLatencyCalibrationError(
            "тайм-аут подтверждения эксклюзивного "
            "управления TurretWorker"
        )

    def _next_result(
        self,
        previous_revision: int,
        *,
        deadline: float,
    ) -> tuple[int, VisionResult]:
        while monotonic() < deadline:
            self._check_cancel()
            if not self._turret.video_latency_calibration_active:
                raise LiveVideoLatencyCalibrationError(
                    "эксклюзивное управление TurretWorker "
                    "потеряно"
                )
            snapshot = self._overview_results.snapshot()
            result = snapshot.value
            if snapshot.revision > previous_revision and result is not None:
                if result.frame.camera is not CameraRole.OVERVIEW:
                    raise LiveVideoLatencyCalibrationError(
                        "калибратор получил кадр не от "
                        "Overview"
                    )
                return snapshot.revision, result
            sleep(self._config.poll_interval_s)
        raise _FrameWaitExpired

    def _warmup(self) -> None:
        deadline = monotonic() + self._config.warmup_s
        revision = self._overview_results.snapshot().revision
        previous: VisionResult | None = None
        validated = False
        generation: int | None = None
        while monotonic() < deadline:
            try:
                revision, current = self._next_result(revision, deadline=deadline)
            except _FrameWaitExpired:
                break
            if generation is None:
                generation = current.frame.generation
            elif current.frame.generation != generation:
                raise LiveVideoLatencyCalibrationError(
                    "generation Overview изменилась "
                    "во время калибровки"
                )
            camera_model = self._camera_model_provider(current.frame.generation)
            if camera_model is None:
                raise LiveVideoLatencyCalibrationError(
                    "camera model Overview недоступна"
                )
            if previous is not None:
                try:
                    interval_ns, _basis = _source_interval(previous, current)
                    estimate_optical_flow_rate_y(
                        previous.frame.image,
                        current.frame.image,
                        previous_receive_timestamp_ns=(
                            previous.frame.receive_timestamp_ns
                        ),
                        current_receive_timestamp_ns=(
                            current.frame.receive_timestamp_ns
                        ),
                        source_interval_ns=interval_ns,
                        camera_model=camera_model,
                        exclusions=_exclusions(previous, current),
                        config=self._optical_config,
                    )
                except VideoLatencyCalibrationError as exc:
                    _LOGGER.debug("Warm-up optical-flow sample rejected: %s", exc)
                else:
                    validated = True
            previous = current
        if previous is None:
            raise LiveVideoLatencyCalibrationError(
                "Overview не выдаёт кадры"
            )
        if not validated:
            raise LiveVideoLatencyCalibrationError(
                "недостаточно фоновых признаков; наведите Overview "
                "на неподвижную "
                "детализированную сцену"
            )

    def _collect_run(self, run_index: int) -> LiveCalibrationRunResult:
        profile = build_motion_profile(self._config)
        observations: list[OpticalFlowRateObservationY] = []
        inliers: list[int] = []
        rejected = 0
        timestamp_bases: list[str] = []
        previous: VisionResult | None = None
        revision = self._overview_results.snapshot().revision
        generation: int | None = None
        start_s = monotonic()
        end_s = start_s + motion_profile_duration_s(self._config)
        command_index = 0
        current_rate = 0.0

        while monotonic() < end_s:
            self._check_cancel()
            elapsed = monotonic() - start_s
            while (
                command_index < len(profile)
                and elapsed >= profile[command_index].offset_s
            ):
                step = profile[command_index]
                if not self._turret.submit_video_latency_calibration_velocity_y(
                    step.rate_y_deg_s
                ):
                    raise LiveVideoLatencyCalibrationError(
                        "TurretWorker отклонил "
                        "Y-velocity калибратора"
                    )
                current_rate = step.rate_y_deg_s
                command_index += 1
                elapsed = monotonic() - start_s
                self._publish(
                    LiveCalibrationPhase.RUNNING,
                    (
                        f"Прогон {run_index}/{self._config.run_count}: "
                        "движение по Y."
                    ),
                    run_index=run_index,
                    current_rate_y_deg_s=current_rate,
                    flow_observations=len(observations),
                )

            frame_deadline = min(end_s, monotonic() + 0.25)
            try:
                revision, current = self._next_result(
                    revision, deadline=frame_deadline
                )
            except _FrameWaitExpired as exc:
                if monotonic() >= end_s:
                    break
                raise LiveVideoLatencyCalibrationError(
                    "тайм-аут ожидания нового Overview-кадра"
                ) from exc
            if generation is None:
                generation = current.frame.generation
            elif current.frame.generation != generation:
                raise LiveVideoLatencyCalibrationError(
                    "generation Overview изменилась "
                    "во время прогона"
                )
            camera_model = self._camera_model_provider(current.frame.generation)
            if camera_model is None:
                raise LiveVideoLatencyCalibrationError(
                    "camera model Overview недоступна"
                )
            if previous is not None:
                try:
                    interval_ns, basis = _source_interval(previous, current)
                    observation = estimate_optical_flow_rate_y(
                        previous.frame.image,
                        current.frame.image,
                        previous_receive_timestamp_ns=(
                            previous.frame.receive_timestamp_ns
                        ),
                        current_receive_timestamp_ns=(
                            current.frame.receive_timestamp_ns
                        ),
                        source_interval_ns=interval_ns,
                        camera_model=camera_model,
                        exclusions=_exclusions(previous, current),
                        config=self._optical_config,
                    )
                except (VideoLatencyCalibrationError, LiveVideoLatencyCalibrationError):
                    rejected += 1
                else:
                    observations.append(observation)
                    inliers.append(observation.inliers)
                    timestamp_bases.append(basis)
            previous = current

        if not self._turret.submit_video_latency_calibration_velocity_y(0.0):
            raise LiveVideoLatencyCalibrationError(
                "не удалось отправить финальную Y=0"
            )
        snapshot = self._turret.platform_motion_snapshot()
        estimate = estimate_video_latency_y(
            tuple(observations), snapshot, config=self._optical_config
        )
        basis = (
            "mixed"
            if len(set(timestamp_bases)) > 1
            else (timestamp_bases[0] if timestamp_bases else "unknown")
        )
        return LiveCalibrationRunResult(
            run_index=run_index,
            flow_observations=len(observations),
            rejected_frame_pairs=rejected,
            median_inliers=None if not inliers else float(median(inliers)),
            timestamp_basis=basis,
            estimate=estimate,
        )

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled

    def _finish_ownership_safely(self) -> None:
        if not self._turret.video_latency_calibration_active:
            return
        try:
            self._turret.submit_video_latency_calibration_velocity_y(0.0)
            self._turret.end_video_latency_calibration()
            deadline = monotonic() + self._config.ownership_timeout_s
            while (
                self._turret.video_latency_calibration_active
                and monotonic() < deadline
            ):
                sleep(self._config.poll_interval_s)
            if self._turret.video_latency_calibration_active:
                raise LiveVideoLatencyCalibrationError(
                    "TurretWorker не освободил calibration ownership"
                )
        except (RuntimeError, ValueError, LiveVideoLatencyCalibrationError) as exc:
            _LOGGER.error("Calibration cleanup failed; requesting Emergency: %s", exc)
            self._turret.request_emergency()


class _Cancelled(Exception):
    pass


class _FrameWaitExpired(Exception):
    pass


def write_live_video_latency_calibration_json(
    result: LiveVideoLatencyCalibrationResult, path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


__all__ = [
    "InRuntimeVideoLatencyCalibration",
    "LiveCalibrationPhase",
    "LiveCalibrationRunResult",
    "LiveVideoLatencyCalibrationConfig",
    "LiveVideoLatencyCalibrationError",
    "LiveVideoLatencyCalibrationResult",
    "LiveVideoLatencyCalibrationState",
    "MotionCommandStep",
    "build_motion_profile",
    "motion_profile_duration_s",
    "write_live_video_latency_calibration_json",
]
