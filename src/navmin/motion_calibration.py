"""Camera-specific empirical correction for click-driven relative motion."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, Literal, Protocol

import numpy as np

from navmin.config.models import AxesConfig, AxisMechanicsConfig
from navmin.contracts import (
    CameraModel,
    CameraRay,
    CameraRole,
    FramePacket,
    MoveRelativeCommand,
)

LOGGER = logging.getLogger(__name__)

MOTION_CALIBRATION_SCHEMA_VERSION = 2
SUPPORTED_MOTION_CALIBRATION_CAMERAS = (
    CameraRole.OVERVIEW,
    CameraRole.STEREO_LEFT,
)
type AxisName = Literal["x", "y"]
type DirectionName = Literal["positive", "negative"]
type ReversalName = Literal["positive-to-negative", "negative-to-positive"]


class MotionCalibrationError(ValueError):
    """Motion calibration data or an operator measurement is invalid."""


@dataclass(frozen=True, slots=True)
class MechanicalIdentity:
    full_steps_per_revolution: int
    microstep_divider: int
    invert: bool

    @classmethod
    def from_config(cls, config: AxisMechanicsConfig) -> MechanicalIdentity:
        return cls(
            full_steps_per_revolution=config.full_steps_per_revolution,
            microstep_divider=config.microstep_divider,
            invert=config.invert,
        )


@dataclass(frozen=True, slots=True)
class CurvePoint:
    physical_deg: float
    command_deg: float


@dataclass(frozen=True, slots=True)
class DirectionCurve:
    points: tuple[CurvePoint, ...]
    transition_enabled: bool = True

    def __post_init__(self) -> None:
        _validate_direction_curve(self.points)

    @property
    def trusted_boundary_deg(self) -> float:
        return self.points[-1].physical_deg

    @property
    def command_boundary_deg(self) -> float:
        return self.points[-1].command_deg

    def interpolate(self, physical_deg: float) -> float:
        value = _positive_finite(physical_deg, "physical_deg", allow_zero=True)
        if value > self.trusted_boundary_deg:
            raise MotionCalibrationError("physical angle is outside the trusted range")
        for left, right in pairwise(self.points):
            if value <= right.physical_deg:
                span = right.physical_deg - left.physical_deg
                fraction = (value - left.physical_deg) / span
                return left.command_deg + fraction * (
                    right.command_deg - left.command_deg
                )
        return self.command_boundary_deg


@dataclass(frozen=True, slots=True)
class ReversalLoss:
    positive_to_negative_deg: float | None = None
    negative_to_positive_deg: float | None = None


@dataclass(frozen=True, slots=True)
class MotionProfile:
    max_speed_deg_s: float
    acceleration_deg_s2: float
    settle_margin_ms: int

    def __post_init__(self) -> None:
        _positive_finite(self.max_speed_deg_s, "max_speed_deg_s")
        _positive_finite(self.acceleration_deg_s2, "acceleration_deg_s2")
        if (
            isinstance(self.settle_margin_ms, bool)
            or not isinstance(self.settle_margin_ms, int)
            or self.settle_margin_ms < 0
        ):
            raise MotionCalibrationError("settle_margin_ms must be an integer >= 0")


@dataclass(frozen=True, slots=True)
class RawObservation:
    camera: CameraRole
    axis: AxisName
    commanded_angle_deg: float
    direction: DirectionName
    reversal: bool
    before_generation: int
    after_generation: int
    before_frame_id: int
    after_frame_id: int
    before_receive_timestamp_ns: int
    after_receive_timestamp_ns: int
    before_click_x_px: float
    before_click_y_px: float
    after_click_x_px: float
    after_click_y_px: float
    before_angular_coordinate_deg: float
    after_angular_coordinate_deg: float
    measured_movement_deg: float
    motion_profile: MotionProfile


@dataclass(frozen=True, slots=True)
class AxisMotionCalibration:
    positive: DirectionCurve
    negative: DirectionCurve
    reversal_loss: ReversalLoss
    observations: tuple[RawObservation, ...]
    quality_max_repeat_spread_deg: float


@dataclass(frozen=True, slots=True)
class CameraMotionCalibration:
    camera: CameraRole
    camera_calibration_identity: str
    mechanical_identity_x: MechanicalIdentity
    mechanical_identity_y: MechanicalIdentity
    calibration_limit_x_deg: float
    calibration_limit_y_deg: float
    motion_profile: MotionProfile
    axes: dict[AxisName, AxisMotionCalibration]
    created_at_utc: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.camera not in SUPPORTED_MOTION_CALIBRATION_CAMERAS:
            raise MotionCalibrationError("camera profile must be Overview or Stereo Left")
        if not self.camera_calibration_identity:
            raise MotionCalibrationError("camera calibration identity must not be empty")
        _positive_finite(self.calibration_limit_x_deg, "calibration_limit_x_deg")
        _positive_finite(self.calibration_limit_y_deg, "calibration_limit_y_deg")
        if not self.axes or any(axis not in ("x", "y") for axis in self.axes):
            raise MotionCalibrationError("camera profile must contain X and/or Y")


@dataclass(frozen=True, slots=True)
class MotionCalibrationDocument:
    schema_version: int = MOTION_CALIBRATION_SCHEMA_VERSION
    extrapolation_percent: float = 20.0
    profiles: dict[CameraRole, CameraMotionCalibration] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != MOTION_CALIBRATION_SCHEMA_VERSION:
            raise MotionCalibrationError(
                f"unsupported motion calibration schema {self.schema_version}"
            )
        value = _finite(self.extrapolation_percent, "extrapolation_percent")
        if value < 0.0 or value > 100.0:
            raise MotionCalibrationError("extrapolation_percent must be in [0, 100]")
        if any(camera not in SUPPORTED_MOTION_CALIBRATION_CAMERAS for camera in self.profiles):
            raise MotionCalibrationError("unsupported camera profile")


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    command: MoveRelativeCommand
    zone_x: str
    zone_y: str


def calibration_identity(value: object) -> str:
    """Return a stable content identity for an already validated camera calibration."""
    normalized = _json_normalize(value)
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_normalize(value: object) -> object:
    if is_dataclass(value):
        return _json_normalize(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, CameraRole):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


class MotionCalibrationRuntime:
    """Own the active file snapshot and correct only camera-originated commands."""

    def __init__(
        self,
        *,
        path: Path,
        axes: AxesConfig,
        camera_identities: dict[CameraRole, str],
        document: MotionCalibrationDocument | None = None,
    ) -> None:
        self.path = Path(path)
        self.axes = axes
        self.camera_identities = dict(camera_identities)
        self._document = document or MotionCalibrationDocument()

    @classmethod
    def load(
        cls,
        *,
        path: Path,
        axes: AxesConfig,
        camera_identities: dict[CameraRole, str],
    ) -> MotionCalibrationRuntime:
        candidate = Path(path)
        if not candidate.exists():
            return cls(path=candidate, axes=axes, camera_identities=camera_identities)
        try:
            document = motion_calibration_from_mapping(
                json.loads(candidate.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, MotionCalibrationError, TypeError) as exc:
            LOGGER.warning(
                "Motion calibration disabled because %s could not be loaded: %s",
                candidate,
                exc,
            )
            document = MotionCalibrationDocument()
        return cls(
            path=candidate,
            axes=axes,
            camera_identities=camera_identities,
            document=document,
        )

    @property
    def document(self) -> MotionCalibrationDocument:
        return self._document

    def correct(
        self,
        camera: CameraRole,
        command: MoveRelativeCommand,
    ) -> MoveRelativeCommand:
        if camera not in SUPPORTED_MOTION_CALIBRATION_CAMERAS:
            return command
        profile = self._document.profiles.get(camera)
        if profile is None:
            return command
        expected_identity = self.camera_identities.get(camera)
        if expected_identity != profile.camera_calibration_identity:
            LOGGER.warning(
                "Motion calibration ignored camera=%s reason=camera_identity_mismatch",
                camera.value,
            )
            return command
        corrected_x, zone_x = self._correct_axis(
            camera, "x", command.delta_x_deg, profile
        )
        corrected_y, zone_y = self._correct_axis(
            camera, "y", command.delta_y_deg, profile
        )
        LOGGER.debug(
            "Motion calibration camera=%s requested=(%.6f,%.6f) corrected=(%.6f,%.6f) zones=(%s,%s)",
            camera.value,
            command.delta_x_deg,
            command.delta_y_deg,
            corrected_x,
            corrected_y,
            zone_x,
            zone_y,
        )
        return MoveRelativeCommand(corrected_x, corrected_y)

    def _correct_axis(
        self,
        camera: CameraRole,
        axis: AxisName,
        requested_deg: float,
        profile: CameraMotionCalibration,
    ) -> tuple[float, str]:
        requested = _finite(requested_deg, f"requested_{axis}_deg")
        if requested == 0.0:
            return 0.0, "zero"
        axis_profile = profile.axes.get(axis)
        if axis_profile is None:
            return requested, "baseline-missing-axis"
        mechanics = self.axes.x if axis == "x" else self.axes.y
        identity = (
            profile.mechanical_identity_x
            if axis == "x"
            else profile.mechanical_identity_y
        )
        if identity != MechanicalIdentity.from_config(mechanics):
            LOGGER.warning(
                "Motion calibration ignored camera=%s axis=%s reason=mechanical_identity_mismatch",
                camera.value,
                axis,
            )
            return requested, "baseline-incompatible"
        magnitude = abs(requested)
        if magnitude > mechanics.max_relative_move_deg:
            return requested, "baseline-limit"
        calibration_limit = (
            profile.calibration_limit_x_deg
            if axis == "x"
            else profile.calibration_limit_y_deg
        )
        effective_limit = min(mechanics.max_relative_move_deg, calibration_limit)
        direction = axis_profile.positive if requested > 0.0 else axis_profile.negative
        trusted = direction.trusted_boundary_deg
        sign = 1.0 if requested > 0.0 else -1.0
        if magnitude > effective_limit:
            self._warn_fallback(
                camera, axis, requested, trusted, effective_limit
            )
            return requested, "baseline-calibration-limit"
        if magnitude <= trusted:
            corrected = direction.interpolate(magnitude)
            if corrected > mechanics.max_relative_move_deg:
                return requested, "baseline-current-limit"
            return sign * corrected, "trusted"

        extended = _extended_boundary(
            trusted,
            self._document.extrapolation_percent,
            effective_limit,
        )
        if not direction.transition_enabled or extended <= trusted:
            self._warn_fallback(camera, axis, requested, trusted, extended)
            return requested, "baseline-transition-disabled"
        if magnitude <= extended:
            correction = direction.command_boundary_deg - trusted
            fade = (extended - magnitude) / (extended - trusted)
            corrected = magnitude + correction * fade
            if (
                corrected < direction.command_boundary_deg
                or corrected > mechanics.max_relative_move_deg
                or not math.isfinite(corrected)
            ):
                self._warn_fallback(camera, axis, requested, trusted, extended)
                return requested, "baseline-unsafe-transition"
            return sign * corrected, "transition"
        self._warn_fallback(camera, axis, requested, trusted, extended)
        return requested, "baseline-outside"

    @staticmethod
    def _warn_fallback(
        camera: CameraRole,
        axis: AxisName,
        requested: float,
        trusted: float,
        extended: float,
    ) -> None:
        LOGGER.warning(
            "Motion calibration fallback=baseline camera=%s axis=%s requested_deg=%.6f trusted_deg=%.6f extended_deg=%.6f",
            camera.value,
            axis,
            requested,
            trusted,
            extended,
        )

    def save_and_apply(
        self,
        profile: CameraMotionCalibration,
        *,
        extrapolation_percent: float,
    ) -> None:
        """Atomically persist a merged document, then make it authoritative."""
        candidate = MotionCalibrationDocument(
            extrapolation_percent=extrapolation_percent,
            profiles={**self._document.profiles, profile.camera: profile},
        )
        payload = json.dumps(
            motion_calibration_to_mapping(candidate),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        self._document = candidate


def _extended_boundary(trusted: float, percent: float, limit: float) -> float:
    return min(limit, trusted * (1.0 + percent / 100.0))


def _transition_is_safe(
    curve: DirectionCurve,
    *,
    extrapolation_percent: float,
    mechanical_limit_deg: float,
) -> bool:
    trusted = curve.trusted_boundary_deg
    extended = _extended_boundary(trusted, extrapolation_percent, mechanical_limit_deg)
    if extended <= trusted:
        return False
    correction = curve.command_boundary_deg - trusted
    slope = 1.0 - correction / (extended - trusted)
    return slope > 0.0 and curve.command_boundary_deg <= mechanical_limit_deg


def _validate_direction_curve(points: tuple[CurvePoint, ...]) -> None:
    if len(points) < 3:
        raise MotionCalibrationError("curve requires zero plus at least two amplitudes")
    first = points[0]
    if first.physical_deg != 0.0 or first.command_deg != 0.0:
        raise MotionCalibrationError("curve must start with 0 deg -> 0 deg")
    previous_physical = -1.0
    previous_command = -1.0
    for point in points:
        physical = _positive_finite(
            point.physical_deg, "curve physical_deg", allow_zero=True
        )
        command = _positive_finite(
            point.command_deg, "curve command_deg", allow_zero=True
        )
        if physical <= previous_physical or command <= previous_command:
            raise MotionCalibrationError("curve coordinates must be strictly monotonic")
        previous_physical = physical
        previous_command = command


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MotionCalibrationError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MotionCalibrationError(f"{name} must be finite")
    return numeric


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    numeric = _finite(value, name)
    if numeric < 0.0 or (not allow_zero and numeric == 0.0):
        relation = ">= 0" if allow_zero else "> 0"
        raise MotionCalibrationError(f"{name} must be {relation}")
    return numeric


def validate_angles(
    angles: tuple[float, float, float],
    *,
    mechanical_limit_deg: float,
) -> tuple[float, float, float]:
    if len(angles) != 3:
        raise MotionCalibrationError("exactly A1, A2 and A3 are required")
    values = tuple(_positive_finite(value, "calibration angle") for value in angles)
    if not values[0] < values[1] < values[2]:
        raise MotionCalibrationError("calibration angles must satisfy 0 < A1 < A2 < A3")
    if values[2] > mechanical_limit_deg:
        raise MotionCalibrationError("A3 exceeds the mechanical relative limit")
    return values


@dataclass(frozen=True, slots=True)
class CalibrationPlanStep:
    axis: AxisName
    command_deg: float
    reversal: bool
    conditioning: bool = False


def build_calibration_sequence(
    x_angles: tuple[float, float, float],
    y_angles: tuple[float, float, float],
) -> tuple[CalibrationPlanStep, ...]:
    steps: list[CalibrationPlanStep] = []
    for axis, angles in (("x", x_angles), ("y", y_angles)):
        a1, a2, a3 = angles
        steps.append(CalibrationPlanStep(axis, a3, reversal=False, conditioning=True))
        steps.extend(CalibrationPlanStep(axis, angle, False) for angle in (a1, a2, a3))
        steps.append(CalibrationPlanStep(axis, -a2, True))
        steps.extend(CalibrationPlanStep(axis, -angle, False) for angle in (a1, a2, a3))
        steps.append(CalibrationPlanStep(axis, a2, True))
        steps.extend(CalibrationPlanStep(axis, angle, False) for angle in (a1, a2, a3))
        steps.append(CalibrationPlanStep(axis, -a3, True))
        steps.extend(CalibrationPlanStep(axis, -angle, False) for angle in (a1, a2, a3))
        steps.append(CalibrationPlanStep(axis, a3, True))
    return tuple(steps)


class CalibrationMovePort(Protocol):
    def __call__(
        self,
        camera: CameraRole,
        generation: int,
        command: MoveRelativeCommand,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class WizardSettings:
    camera: CameraRole
    x_angles: tuple[float, float, float]
    y_angles: tuple[float, float, float]
    extrapolation_percent: float
    motion_profile: MotionProfile


class WizardPhase:
    IDLE = "idle"
    MOVING = "moving"
    WAIT_FIRST_CLICK = "wait-first-click"
    WAIT_SECOND_CLICK = "wait-second-click"
    REVIEW_REPEAT = "review-repeat"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class _FirstClick:
    frame: FramePacket
    x_px: float
    y_px: float
    angular_coordinate_deg: float


class MotionCalibrationWizard:
    """Deterministic click/settle state machine independent of Qt."""

    def __init__(
        self,
        *,
        settings: WizardSettings,
        mechanics: AxesConfig,
        camera_calibration_identity: str,
        stale_timeout_ms: int,
        submit_move: CalibrationMovePort,
        clock_ns,
    ) -> None:
        if settings.camera not in SUPPORTED_MOTION_CALIBRATION_CAMERAS:
            raise MotionCalibrationError("unsupported calibration camera")
        validate_angles(
            settings.x_angles,
            mechanical_limit_deg=mechanics.x.max_relative_move_deg,
        )
        validate_angles(
            settings.y_angles,
            mechanical_limit_deg=mechanics.y.max_relative_move_deg,
        )
        if settings.extrapolation_percent < 0.0 or settings.extrapolation_percent > 100.0:
            raise MotionCalibrationError("extrapolation percent must be in [0, 100]")
        if stale_timeout_ms <= 0:
            raise MotionCalibrationError("stale timeout must be positive")
        self.settings = settings
        self.mechanics = mechanics
        self.camera_calibration_identity = camera_calibration_identity
        self.stale_timeout_ns = stale_timeout_ms * 1_000_000
        self._submit_move = submit_move
        self._clock_ns = clock_ns
        self.steps = build_calibration_sequence(settings.x_angles, settings.y_angles)
        self.step_index = 0
        self.phase = WizardPhase.IDLE
        self.settle_deadline_ns: int | None = None
        self.first_click: _FirstClick | None = None
        self.observations: list[RawObservation] = []
        self.cancel_reason: str | None = None
        self.warnings: list[str] = []
        self._generation: int | None = None
        self._repeat_resolution_key: tuple[AxisName, DirectionName, float] | None = None

    @property
    def current_step(self) -> CalibrationPlanStep | None:
        if self.step_index >= len(self.steps):
            return None
        return self.steps[self.step_index]

    def start(self, *, generation: int) -> None:
        if self.phase != WizardPhase.IDLE:
            raise MotionCalibrationError("wizard already started")
        self._generation = generation
        self._dispatch_current(generation)

    def tick(self) -> None:
        if self.phase != WizardPhase.MOVING or self.settle_deadline_ns is None:
            return
        if self._clock_ns() < self.settle_deadline_ns:
            return
        step = self.current_step
        if step is None:
            self.phase = WizardPhase.COMPLETE
        elif step.conditioning:
            self.step_index += 1
            self.phase = WizardPhase.WAIT_FIRST_CLICK
            self.settle_deadline_ns = None
        else:
            self.phase = WizardPhase.WAIT_SECOND_CLICK
            self.settle_deadline_ns = None

    def handle_click(
        self,
        *,
        frame: FramePacket,
        camera_model: CameraModel,
        x_px: float,
        y_px: float,
    ) -> bool:
        self._validate_frame(frame, x_px, y_px)
        step = self.current_step
        if step is None:
            return False
        if self.phase == WizardPhase.WAIT_FIRST_CLICK:
            coordinate = _axis_coordinate(camera_model, step.axis, x_px, y_px)
            if not landmark_remains_visible(
                camera_model=camera_model,
                frame=frame,
                x_px=x_px,
                y_px=y_px,
                axis=step.axis,
                command_deg=step.command_deg,
            ):
                raise MotionCalibrationError(
                    "selected landmark is too close to the image edge for this move"
                )
            first = _FirstClick(frame, float(x_px), float(y_px), coordinate)
            command = _axis_command(step.axis, step.command_deg)
            if not self._submit_move(frame.camera, frame.generation, command):
                raise MotionCalibrationError("calibration MOVE_RELATIVE was rejected")
            self.first_click = first
            self.phase = WizardPhase.MOVING
            self.settle_deadline_ns = self._clock_ns() + minimum_move_time_ns(
                abs(step.command_deg), self.settings.motion_profile
            )
            return True
        if self.phase != WizardPhase.WAIT_SECOND_CLICK:
            return False
        first = self.first_click
        if first is None:
            raise MotionCalibrationError("missing first click")
        if frame.generation != first.frame.generation:
            raise MotionCalibrationError("camera generation changed during measurement")
        if frame.frame_id <= first.frame.frame_id:
            raise MotionCalibrationError("second frame must be newer than first frame")
        coordinate = _axis_coordinate(camera_model, step.axis, x_px, y_px)
        measured = first.angular_coordinate_deg - coordinate
        if not math.isfinite(measured) or measured * step.command_deg <= 0.0:
            raise MotionCalibrationError(
                "measured movement has the wrong direction; repeat the measurement"
            )
        observation = RawObservation(
            camera=frame.camera,
            axis=step.axis,
            commanded_angle_deg=step.command_deg,
            direction="positive" if step.command_deg > 0.0 else "negative",
            reversal=step.reversal,
            before_generation=first.frame.generation,
            after_generation=frame.generation,
            before_frame_id=first.frame.frame_id,
            after_frame_id=frame.frame_id,
            before_receive_timestamp_ns=first.frame.receive_timestamp_ns,
            after_receive_timestamp_ns=frame.receive_timestamp_ns,
            before_click_x_px=first.x_px,
            before_click_y_px=first.y_px,
            after_click_x_px=float(x_px),
            after_click_y_px=float(y_px),
            before_angular_coordinate_deg=first.angular_coordinate_deg,
            after_angular_coordinate_deg=coordinate,
            measured_movement_deg=measured,
            motion_profile=self.settings.motion_profile,
        )
        self.observations.append(observation)
        self.first_click = None
        if not step.reversal:
            if self._repeat_resolution_key is not None:
                if self._repeat_group_has_agreeing_pair(observation):
                    self.warnings.append(
                        f"repeat disagreement resolved by additional sample at step {self.step_index + 1}"
                    )
                    self._repeat_resolution_key = None
                else:
                    self.phase = WizardPhase.REVIEW_REPEAT
                    return True
            elif self._repeat_disagrees(observation):
                self._repeat_resolution_key = self._observation_key(observation)
                self.phase = WizardPhase.REVIEW_REPEAT
                return True
        self._advance_after_measurement()
        return True

    def accept_disagreement(self) -> None:
        if self.phase != WizardPhase.REVIEW_REPEAT:
            raise MotionCalibrationError("there is no measurement disagreement to accept")
        self.warnings.append(
            f"operator accepted repeat disagreement at step {self.step_index + 1}"
        )
        self._repeat_resolution_key = None
        self._advance_after_measurement()

    def repeat_problematic_step(self) -> None:
        if self.phase != WizardPhase.REVIEW_REPEAT:
            raise MotionCalibrationError("there is no problematic step to repeat")
        self.first_click = None
        self.phase = WizardPhase.WAIT_FIRST_CLICK

    def cancel(self, reason: str) -> None:
        if self.phase in (WizardPhase.COMPLETE, WizardPhase.CANCELLED):
            return
        self.first_click = None
        self.settle_deadline_ns = None
        self._repeat_resolution_key = None
        self.phase = WizardPhase.CANCELLED
        self.cancel_reason = reason

    def build_profile(self) -> CameraMotionCalibration:
        if self.phase != WizardPhase.COMPLETE:
            raise MotionCalibrationError("calibration is not complete")
        axes: dict[AxisName, AxisMotionCalibration] = {}
        warnings = list(self.warnings)
        for axis in ("x", "y"):
            mechanics = self.mechanics.x if axis == "x" else self.mechanics.y
            axis_profile, axis_warnings = build_axis_calibration(
                axis,
                tuple(item for item in self.observations if item.axis == axis),
                extrapolation_percent=self.settings.extrapolation_percent,
                mechanical_limit_deg=mechanics.max_relative_move_deg,
            )
            axes[axis] = axis_profile
            warnings.extend(axis_warnings)
        return CameraMotionCalibration(
            camera=self.settings.camera,
            camera_calibration_identity=self.camera_calibration_identity,
            mechanical_identity_x=MechanicalIdentity.from_config(self.mechanics.x),
            mechanical_identity_y=MechanicalIdentity.from_config(self.mechanics.y),
            calibration_limit_x_deg=self.mechanics.x.max_relative_move_deg,
            calibration_limit_y_deg=self.mechanics.y.max_relative_move_deg,
            motion_profile=self.settings.motion_profile,
            axes=axes,
            created_at_utc=datetime.now(UTC).isoformat(),
            warnings=tuple(warnings),
        )

    def _dispatch_current(self, generation: int) -> None:
        step = self.current_step
        if step is None:
            self.phase = WizardPhase.COMPLETE
            return
        if not step.conditioning:
            self.phase = WizardPhase.WAIT_FIRST_CLICK
            return
        if not self._submit_move(
            self.settings.camera,
            generation,
            _axis_command(step.axis, step.command_deg),
        ):
            raise MotionCalibrationError("conditioning MOVE_RELATIVE was rejected")
        self.phase = WizardPhase.MOVING
        self.settle_deadline_ns = self._clock_ns() + minimum_move_time_ns(
            abs(step.command_deg), self.settings.motion_profile
        )

    def _advance_after_measurement(self) -> None:
        self.step_index += 1
        if self.step_index >= len(self.steps):
            self.phase = WizardPhase.COMPLETE
            return
        step = self.current_step
        if step is not None and step.conditioning:
            generation = self._generation
            if generation is None:
                raise MotionCalibrationError("calibration generation is unavailable")
            try:
                self._dispatch_current(generation)
            except MotionCalibrationError:
                self.cancel("automatic conditioning MOVE_RELATIVE was rejected")
                raise
            return
        self.phase = WizardPhase.WAIT_FIRST_CLICK

    @staticmethod
    def _observation_key(
        observation: RawObservation,
    ) -> tuple[AxisName, DirectionName, float]:
        return (
            observation.axis,
            observation.direction,
            abs(observation.commanded_angle_deg),
        )

    def _repeat_group_has_agreeing_pair(self, observation: RawObservation) -> bool:
        key = self._repeat_resolution_key
        if key is None or self._observation_key(observation) != key:
            return False
        values = sorted(
            abs(item.measured_movement_deg)
            for item in self.observations
            if not item.reversal and self._observation_key(item) == key
        )
        if len(values) < 3:
            return False
        for left, right in pairwise(values):
            center = median((left, right))
            tolerance = max(0.25, center * 0.12)
            if right - left <= tolerance:
                return True
        return False

    def _repeat_disagrees(self, observation: RawObservation) -> bool:
        peers = [
            item
            for item in self.observations[:-1]
            if not item.reversal
            and item.axis == observation.axis
            and item.direction == observation.direction
            and abs(item.commanded_angle_deg) == abs(observation.commanded_angle_deg)
        ]
        if not peers:
            return False
        previous = median(abs(item.measured_movement_deg) for item in peers)
        difference = abs(abs(observation.measured_movement_deg) - previous)
        tolerance = max(0.25, previous * 0.12)
        return difference > tolerance

    def _validate_frame(self, frame: FramePacket, x_px: float, y_px: float) -> None:
        if frame.camera is not self.settings.camera:
            raise MotionCalibrationError("click came from the wrong camera")
        if self._clock_ns() - frame.receive_timestamp_ns > self.stale_timeout_ns:
            raise MotionCalibrationError("camera frame is stale")
        height, width = frame.image.shape[:2]
        if not (0.0 <= x_px < width and 0.0 <= y_px < height):
            raise MotionCalibrationError("click is outside the frame")


def minimum_move_time_ns(angle_deg: float, profile: MotionProfile) -> int:
    distance = _positive_finite(angle_deg, "angle_deg", allow_zero=True)
    speed = profile.max_speed_deg_s
    acceleration = profile.acceleration_deg_s2
    ramp_distance = speed * speed / acceleration
    if distance <= ramp_distance:
        duration_s = 2.0 * math.sqrt(distance / acceleration)
    else:
        duration_s = 2.0 * speed / acceleration + (distance - ramp_distance) / speed
    duration_s += profile.settle_margin_ms / 1000.0
    return math.ceil(duration_s * 1_000_000_000)


def _axis_command(axis: AxisName, angle_deg: float) -> MoveRelativeCommand:
    return (
        MoveRelativeCommand(angle_deg, 0.0)
        if axis == "x"
        else MoveRelativeCommand(0.0, angle_deg)
    )


def _axis_coordinate(
    camera_model: CameraModel,
    axis: AxisName,
    x_px: float,
    y_px: float,
) -> float:
    try:
        ray = camera_model.pixel_to_ray(x_px, y_px)
    except (ArithmeticError, ValueError) as exc:
        raise MotionCalibrationError("camera model rejected the selected pixel") from exc
    horizontal, vertical = _ray_angles(ray)
    return horizontal if axis == "x" else vertical


def _ray_angles(ray: CameraRay) -> tuple[float, float]:
    values = (ray.x, ray.y, ray.z)
    if not all(math.isfinite(value) for value in values) or values == (0.0, 0.0, 0.0):
        raise MotionCalibrationError("camera model returned an invalid ray")
    return (
        math.degrees(math.atan2(ray.x, ray.z)),
        -math.degrees(math.atan2(ray.y, math.hypot(ray.x, ray.z))),
    )


def landmark_remains_visible(
    *,
    camera_model: CameraModel,
    frame: FramePacket,
    x_px: float,
    y_px: float,
    axis: AxisName,
    command_deg: float,
) -> bool:
    try:
        source_ray = camera_model.pixel_to_ray(x_px, y_px)
        horizontal, vertical = _ray_angles(source_ray)
    except (ArithmeticError, ValueError, MotionCalibrationError):
        return False
    if axis == "x":
        horizontal -= command_deg
    else:
        vertical -= command_deg
    h = math.radians(horizontal)
    v = math.radians(vertical)
    ray = CameraRay(
        x=math.sin(h) * math.cos(v),
        y=-math.sin(v),
        z=math.cos(h) * math.cos(v),
    )
    try:
        projected_x, projected_y = camera_model.ray_to_pixel(ray)
    except (ArithmeticError, ValueError):
        return False
    height, width = frame.image.shape[:2]
    margin_x = max(8.0, width * 0.05)
    margin_y = max(8.0, height * 0.05)
    return bool(
        math.isfinite(projected_x)
        and math.isfinite(projected_y)
        and margin_x <= projected_x < width - margin_x
        and margin_y <= projected_y < height - margin_y
    )


def build_axis_calibration(
    axis: AxisName,
    observations: tuple[RawObservation, ...],
    *,
    extrapolation_percent: float,
    mechanical_limit_deg: float,
) -> tuple[AxisMotionCalibration, tuple[str, ...]]:
    clean = [item for item in observations if not item.reversal]
    reversal = [item for item in observations if item.reversal]
    positive, positive_spread = _build_curve(clean, "positive", mechanical_limit_deg)
    negative, negative_spread = _build_curve(clean, "negative", mechanical_limit_deg)
    warnings: list[str] = []
    positive_safe = _transition_is_safe(
        positive,
        extrapolation_percent=extrapolation_percent,
        mechanical_limit_deg=mechanical_limit_deg,
    )
    negative_safe = _transition_is_safe(
        negative,
        extrapolation_percent=extrapolation_percent,
        mechanical_limit_deg=mechanical_limit_deg,
    )
    if not positive_safe:
        warnings.append(f"{axis}+ transition disabled")
    if not negative_safe:
        warnings.append(f"{axis}- transition disabled")
    positive = replace(positive, transition_enabled=positive_safe)
    negative = replace(negative, transition_enabled=negative_safe)
    loss_positive_to_negative = _reversal_loss(
        reversal,
        direction="negative",
        clean_curve=negative,
    )
    loss_negative_to_positive = _reversal_loss(
        reversal,
        direction="positive",
        clean_curve=positive,
    )
    if loss_positive_to_negative is None:
        warnings.append(f"{axis} positive-to-negative reversal loss unavailable")
    if loss_negative_to_positive is None:
        warnings.append(f"{axis} negative-to-positive reversal loss unavailable")
    return (
        AxisMotionCalibration(
            positive=positive,
            negative=negative,
            reversal_loss=ReversalLoss(
                positive_to_negative_deg=loss_positive_to_negative,
                negative_to_positive_deg=loss_negative_to_positive,
            ),
            observations=observations,
            quality_max_repeat_spread_deg=max(positive_spread, negative_spread),
        ),
        tuple(warnings),
    )


def _build_curve(
    observations: list[RawObservation],
    direction: DirectionName,
    mechanical_limit_deg: float,
) -> tuple[DirectionCurve, float]:
    grouped: dict[float, list[float]] = {}
    for item in observations:
        if item.direction != direction:
            continue
        command = abs(item.commanded_angle_deg)
        measured = abs(item.measured_movement_deg)
        if command > mechanical_limit_deg:
            raise MotionCalibrationError("curve command exceeds mechanical limit")
        grouped.setdefault(command, []).append(measured)
    if len(grouped) < 3:
        raise MotionCalibrationError(
            f"{direction} curve requires three distinct clean command amplitudes"
        )
    points = [CurvePoint(0.0, 0.0)]
    max_spread = 0.0
    for command in sorted(grouped):
        values = grouped[command]
        measured = median(values)
        max_spread = max(max_spread, max(values) - min(values))
        points.append(CurvePoint(measured, command))
    curve = DirectionCurve(tuple(points))
    if curve.trusted_boundary_deg <= 0.0:
        raise MotionCalibrationError("curve has insufficient measured span")
    return curve, max_spread


def _reversal_loss(
    observations: list[RawObservation],
    *,
    direction: DirectionName,
    clean_curve: DirectionCurve,
) -> float | None:
    estimates: list[float] = []
    for item in observations:
        if item.direction != direction:
            continue
        command = abs(item.commanded_angle_deg)
        expected = _physical_for_command(clean_curve, command)
        estimates.append(max(0.0, expected - abs(item.measured_movement_deg)))
    if len(estimates) < 2:
        return None
    center = median(estimates)
    if max(estimates) - min(estimates) > max(0.5, center * 0.5):
        return None
    return center


def _physical_for_command(curve: DirectionCurve, command: float) -> float:
    points = curve.points
    if command > points[-1].command_deg:
        return points[-1].physical_deg
    for left, right in pairwise(points):
        if command <= right.command_deg:
            fraction = (command - left.command_deg) / (
                right.command_deg - left.command_deg
            )
            return left.physical_deg + fraction * (
                right.physical_deg - left.physical_deg
            )
    return points[-1].physical_deg


def motion_calibration_to_mapping(document: MotionCalibrationDocument) -> dict[str, Any]:
    return {
        "schema_version": document.schema_version,
        "extrapolation_percent": document.extrapolation_percent,
        "profiles": {
            camera.value: _camera_profile_to_mapping(profile)
            for camera, profile in document.profiles.items()
        },
    }


def _camera_profile_to_mapping(profile: CameraMotionCalibration) -> dict[str, Any]:
    return {
        "camera_calibration_identity": profile.camera_calibration_identity,
        "mechanical_identity": {
            "x": asdict(profile.mechanical_identity_x),
            "y": asdict(profile.mechanical_identity_y),
        },
        "calibration_relative_limit_deg": {
            "x": profile.calibration_limit_x_deg,
            "y": profile.calibration_limit_y_deg,
        },
        "motion_profile": asdict(profile.motion_profile),
        "created_at_utc": profile.created_at_utc,
        "warnings": list(profile.warnings),
        "axes": {
            axis: {
                "positive": _curve_to_mapping(value.positive),
                "negative": _curve_to_mapping(value.negative),
                "reversal_loss": asdict(value.reversal_loss),
                "quality_max_repeat_spread_deg": value.quality_max_repeat_spread_deg,
                "raw_observations": [_observation_to_mapping(item) for item in value.observations],
            }
            for axis, value in profile.axes.items()
        },
    }


def _curve_to_mapping(curve: DirectionCurve) -> dict[str, Any]:
    return {
        "points": [asdict(point) for point in curve.points],
        "transition_enabled": curve.transition_enabled,
    }


def _observation_to_mapping(observation: RawObservation) -> dict[str, Any]:
    value = asdict(observation)
    value["camera"] = observation.camera.value
    value["motion_profile"] = asdict(observation.motion_profile)
    return value


def motion_calibration_from_mapping(value: Any) -> MotionCalibrationDocument:
    root = _mapping(value, "$")
    _exact_fields(root, {"schema_version", "extrapolation_percent", "profiles"}, "$")
    profiles_obj = _mapping(root["profiles"], "$.profiles")
    profiles: dict[CameraRole, CameraMotionCalibration] = {}
    for key, profile_value in profiles_obj.items():
        try:
            camera = CameraRole(key)
        except ValueError as exc:
            raise MotionCalibrationError(f"unsupported camera profile {key!r}") from exc
        profiles[camera] = _camera_profile_from_mapping(camera, profile_value)
    return MotionCalibrationDocument(
        schema_version=_integer(root["schema_version"], "$.schema_version"),
        extrapolation_percent=_finite(
            root["extrapolation_percent"], "$.extrapolation_percent"
        ),
        profiles=profiles,
    )


def _camera_profile_from_mapping(
    camera: CameraRole, value: Any
) -> CameraMotionCalibration:
    path = f"$.profiles.{camera.value}"
    obj = _mapping(value, path)
    _exact_fields(
        obj,
        {
            "camera_calibration_identity",
            "mechanical_identity",
            "calibration_relative_limit_deg",
            "motion_profile",
            "created_at_utc",
            "warnings",
            "axes",
        },
        path,
    )
    mechanical = _mapping(obj["mechanical_identity"], f"{path}.mechanical_identity")
    _exact_fields(mechanical, {"x", "y"}, f"{path}.mechanical_identity")
    calibration_limits = _mapping(
        obj["calibration_relative_limit_deg"],
        f"{path}.calibration_relative_limit_deg",
    )
    _exact_fields(
        calibration_limits, {"x", "y"}, f"{path}.calibration_relative_limit_deg"
    )
    axes_obj = _mapping(obj["axes"], f"{path}.axes")
    axes = {
        axis: _axis_profile_from_mapping(
            camera, axis, axis_value, f"{path}.axes.{axis}"
        )
        for axis, axis_value in axes_obj.items()
        if axis in ("x", "y")
    }
    if set(axes_obj) != set(axes):
        raise MotionCalibrationError(f"{path}.axes contains unsupported axis")
    return CameraMotionCalibration(
        camera=camera,
        camera_calibration_identity=_string(
            obj["camera_calibration_identity"], f"{path}.camera_calibration_identity"
        ),
        mechanical_identity_x=_mechanical_from_mapping(
            mechanical["x"], f"{path}.mechanical_identity.x"
        ),
        mechanical_identity_y=_mechanical_from_mapping(
            mechanical["y"], f"{path}.mechanical_identity.y"
        ),
        calibration_limit_x_deg=_positive_finite(
            calibration_limits["x"], f"{path}.calibration_relative_limit_deg.x"
        ),
        calibration_limit_y_deg=_positive_finite(
            calibration_limits["y"], f"{path}.calibration_relative_limit_deg.y"
        ),
        motion_profile=_motion_profile_from_mapping(
            obj["motion_profile"], f"{path}.motion_profile"
        ),
        axes=axes,
        created_at_utc=_string(obj["created_at_utc"], f"{path}.created_at_utc"),
        warnings=tuple(_string_list(obj["warnings"], f"{path}.warnings")),
    )


def _axis_profile_from_mapping(
    camera: CameraRole, axis: str, value: Any, path: str
) -> AxisMotionCalibration:
    obj = _mapping(value, path)
    _exact_fields(
        obj,
        {
            "positive",
            "negative",
            "reversal_loss",
            "quality_max_repeat_spread_deg",
            "raw_observations",
        },
        path,
    )
    observations_value = obj["raw_observations"]
    if not isinstance(observations_value, list):
        raise MotionCalibrationError(f"{path}.raw_observations must be an array")
    return AxisMotionCalibration(
        positive=_curve_from_mapping(obj["positive"], f"{path}.positive"),
        negative=_curve_from_mapping(obj["negative"], f"{path}.negative"),
        reversal_loss=_reversal_from_mapping(
            obj["reversal_loss"], f"{path}.reversal_loss"
        ),
        observations=tuple(
            _observation_from_mapping(camera, axis, item, f"{path}.raw_observations")
            for item in observations_value
        ),
        quality_max_repeat_spread_deg=_finite(
            obj["quality_max_repeat_spread_deg"],
            f"{path}.quality_max_repeat_spread_deg",
        ),
    )


def _curve_from_mapping(value: Any, path: str) -> DirectionCurve:
    obj = _mapping(value, path)
    _exact_fields(obj, {"points", "transition_enabled"}, path)
    raw_points = obj["points"]
    if not isinstance(raw_points, list):
        raise MotionCalibrationError(f"{path}.points must be an array")
    points: list[CurvePoint] = []
    for index, raw in enumerate(raw_points):
        point = _mapping(raw, f"{path}.points[{index}]")
        _exact_fields(point, {"physical_deg", "command_deg"}, path)
        points.append(
            CurvePoint(
                _finite(point["physical_deg"], f"{path}.physical_deg"),
                _finite(point["command_deg"], f"{path}.command_deg"),
            )
        )
    enabled = obj["transition_enabled"]
    if type(enabled) is not bool:
        raise MotionCalibrationError(f"{path}.transition_enabled must be bool")
    return DirectionCurve(tuple(points), enabled)


def _reversal_from_mapping(value: Any, path: str) -> ReversalLoss:
    obj = _mapping(value, path)
    _exact_fields(obj, {"positive_to_negative_deg", "negative_to_positive_deg"}, path)
    return ReversalLoss(
        positive_to_negative_deg=_optional_nonnegative(
            obj["positive_to_negative_deg"], f"{path}.positive_to_negative_deg"
        ),
        negative_to_positive_deg=_optional_nonnegative(
            obj["negative_to_positive_deg"], f"{path}.negative_to_positive_deg"
        ),
    )


def _observation_from_mapping(
    camera: CameraRole, axis: str, value: Any, path: str
) -> RawObservation:
    obj = _mapping(value, path)
    fields = {
        "camera",
        "axis",
        "commanded_angle_deg",
        "direction",
        "reversal",
        "before_generation",
        "after_generation",
        "before_frame_id",
        "after_frame_id",
        "before_receive_timestamp_ns",
        "after_receive_timestamp_ns",
        "before_click_x_px",
        "before_click_y_px",
        "after_click_x_px",
        "after_click_y_px",
        "before_angular_coordinate_deg",
        "after_angular_coordinate_deg",
        "measured_movement_deg",
        "motion_profile",
    }
    _exact_fields(obj, fields, path)
    if obj["camera"] != camera.value or obj["axis"] != axis:
        raise MotionCalibrationError(f"{path} camera/axis does not match owner")
    direction = obj["direction"]
    if direction not in ("positive", "negative"):
        raise MotionCalibrationError(f"{path}.direction is invalid")
    reversal = obj["reversal"]
    if type(reversal) is not bool:
        raise MotionCalibrationError(f"{path}.reversal must be bool")
    return RawObservation(
        camera=camera,
        axis=axis,  # type: ignore[arg-type]
        commanded_angle_deg=_finite(obj["commanded_angle_deg"], path),
        direction=direction,
        reversal=reversal,
        before_generation=_integer(obj["before_generation"], path),
        after_generation=_integer(obj["after_generation"], path),
        before_frame_id=_integer(obj["before_frame_id"], path),
        after_frame_id=_integer(obj["after_frame_id"], path),
        before_receive_timestamp_ns=_integer(obj["before_receive_timestamp_ns"], path),
        after_receive_timestamp_ns=_integer(obj["after_receive_timestamp_ns"], path),
        before_click_x_px=_finite(obj["before_click_x_px"], path),
        before_click_y_px=_finite(obj["before_click_y_px"], path),
        after_click_x_px=_finite(obj["after_click_x_px"], path),
        after_click_y_px=_finite(obj["after_click_y_px"], path),
        before_angular_coordinate_deg=_finite(obj["before_angular_coordinate_deg"], path),
        after_angular_coordinate_deg=_finite(obj["after_angular_coordinate_deg"], path),
        measured_movement_deg=_finite(obj["measured_movement_deg"], path),
        motion_profile=_motion_profile_from_mapping(obj["motion_profile"], path),
    )


def _mechanical_from_mapping(value: Any, path: str) -> MechanicalIdentity:
    obj = _mapping(value, path)
    _exact_fields(obj, {"full_steps_per_revolution", "microstep_divider", "invert"}, path)
    invert = obj["invert"]
    if type(invert) is not bool:
        raise MotionCalibrationError(f"{path}.invert must be bool")
    return MechanicalIdentity(
        _integer(obj["full_steps_per_revolution"], path),
        _integer(obj["microstep_divider"], path),
        invert,
    )


def _motion_profile_from_mapping(value: Any, path: str) -> MotionProfile:
    obj = _mapping(value, path)
    _exact_fields(obj, {"max_speed_deg_s", "acceleration_deg_s2", "settle_margin_ms"}, path)
    return MotionProfile(
        max_speed_deg_s=_finite(obj["max_speed_deg_s"], path),
        acceleration_deg_s2=_finite(obj["acceleration_deg_s2"], path),
        settle_margin_ms=_integer(obj["settle_margin_ms"], path),
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MotionCalibrationError(f"{path} must be an object")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise MotionCalibrationError(f"{path} missing field {min(missing)}")
    if unknown:
        raise MotionCalibrationError(f"{path} unknown field {min(unknown)}")


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise MotionCalibrationError(f"{path} must be an integer")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise MotionCalibrationError(f"{path} must be a non-empty string")
    return value


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MotionCalibrationError(f"{path} must be an array of strings")
    return value


def _optional_nonnegative(value: Any, path: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, path)
    if result < 0.0:
        raise MotionCalibrationError(f"{path} must be >= 0")
    return result


__all__ = [
    "AxisMotionCalibration",
    "CalibrationPlanStep",
    "CameraMotionCalibration",
    "CurvePoint",
    "DirectionCurve",
    "MechanicalIdentity",
    "MotionCalibrationDocument",
    "MotionCalibrationError",
    "MotionCalibrationRuntime",
    "MotionCalibrationWizard",
    "MotionProfile",
    "RawObservation",
    "ReversalLoss",
    "WizardPhase",
    "WizardSettings",
    "build_axis_calibration",
    "build_calibration_sequence",
    "calibration_identity",
    "landmark_remains_visible",
    "minimum_move_time_ns",
    "motion_calibration_from_mapping",
    "motion_calibration_to_mapping",
    "validate_angles",
]
