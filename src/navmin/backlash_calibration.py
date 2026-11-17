"""Standalone gyro-based gearbox backlash calibration and profile storage."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from navmin.config.models import AxesConfig

AxisName = Literal["x", "y"]


class BacklashCalibrationError(ValueError):
    """Invalid or incompatible backlash calibration data."""


@dataclass(frozen=True, slots=True)
class AxisBacklashDelay:
    positive_to_negative_s: float
    negative_to_positive_s: float


@dataclass(frozen=True, slots=True)
class BacklashCalibrationProfile:
    axes_identity: str
    x: AxisBacklashDelay
    y: AxisBacklashDelay
    schema_version: int = 1


def axes_identity(axes: AxesConfig) -> str:
    payload = {
        name: {
            "invert": axis.invert,
            "full_steps_per_revolution": axis.full_steps_per_revolution,
            "microstep_divider": axis.microstep_divider,
            "max_relative_move_deg": axis.max_relative_move_deg,
        }
        for name, axis in (("x", axes.x), ("y", axes.y))
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


class BacklashCalibrationRuntime:
    """Owns a mechanically scoped profile separate from camera calibration."""

    def __init__(
        self,
        *,
        path: Path,
        expected_axes_identity: str,
        max_relative_move: dict[str, float],
        profile: BacklashCalibrationProfile | None = None,
    ) -> None:
        self.path = Path(path)
        self.expected_axes_identity = expected_axes_identity
        self.max_relative_move = dict(max_relative_move)
        self._profile = profile

    @classmethod
    def load(cls, *, path: Path, axes: AxesConfig) -> BacklashCalibrationRuntime:
        identity = axes_identity(axes)
        limits = {"x": axes.x.max_relative_move_deg, "y": axes.y.max_relative_move_deg}
        target = Path(path)
        if not target.exists():
            return cls(
                path=target, expected_axes_identity=identity, max_relative_move=limits
            )
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            profile = _profile_from_mapping(raw)
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise BacklashCalibrationError(
                f"cannot read backlash profile {target}: {exc}"
            ) from exc
        if profile.axes_identity != identity:
            return cls(
                path=target, expected_axes_identity=identity, max_relative_move=limits
            )
        return cls(
            path=target,
            expected_axes_identity=identity,
            max_relative_move=limits,
            profile=profile,
        )

    @property
    def profile(self) -> BacklashCalibrationProfile | None:
        return self._profile

    def delay(self, axis: AxisName, from_direction: int) -> float | None:
        if axis not in ("x", "y"):
            raise ValueError("axis must be x or y")
        if from_direction not in (-1, 1):
            raise ValueError("from_direction must be -1 or 1")
        if self._profile is None:
            return None
        values = self._profile.x if axis == "x" else self._profile.y
        return (
            values.positive_to_negative_s
            if from_direction > 0
            else values.negative_to_positive_s
        )

    def save(
        self,
        x: AxisBacklashDelay,
        y: AxisBacklashDelay,
    ) -> BacklashCalibrationProfile:
        profile = BacklashCalibrationProfile(
            axes_identity=self.expected_axes_identity,
            x=_validate_axis_delay(x),
            y=_validate_axis_delay(y),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(asdict(profile), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        self._profile = profile
        return profile


def _validate_axis_delay(value: AxisBacklashDelay) -> AxisBacklashDelay:
    if not isinstance(value, AxisBacklashDelay):
        raise TypeError("axis delay must be AxisBacklashDelay")
    for delay in (value.positive_to_negative_s, value.negative_to_positive_s):
        if not isfinite(delay) or not 0.0 <= delay <= 5.0:
            raise BacklashCalibrationError("backlash delay must be within 0..5 seconds")
    return value


def _profile_from_mapping(value: Any) -> BacklashCalibrationProfile:
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "axes_identity", "x", "y"
    }:
        raise BacklashCalibrationError("backlash profile has unsupported fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise BacklashCalibrationError("unsupported backlash profile schema")
    identity = value["axes_identity"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise BacklashCalibrationError("invalid axes identity")
    return BacklashCalibrationProfile(
        axes_identity=identity,
        x=_axis_delay_from_mapping(value["x"]),
        y=_axis_delay_from_mapping(value["y"]),
    )


def _axis_delay_from_mapping(value: Any) -> AxisBacklashDelay:
    if not isinstance(value, dict) or set(value) != {
        "positive_to_negative_s", "negative_to_positive_s"
    }:
        raise BacklashCalibrationError("invalid axis backlash delay")
    if any(
        isinstance(value[field], bool)
        or not isinstance(value[field], int | float)
        for field in ("positive_to_negative_s", "negative_to_positive_s")
    ):
        raise BacklashCalibrationError("invalid axis backlash delay")
    try:
        delay = AxisBacklashDelay(
            positive_to_negative_s=float(value["positive_to_negative_s"]),
            negative_to_positive_s=float(value["negative_to_positive_s"]),
        )
    except (TypeError, ValueError) as exc:
        raise BacklashCalibrationError("invalid axis backlash delay") from exc
    return _validate_axis_delay(delay)


__all__ = [
    "AxisBacklashDelay",
    "AxisName",
    "BacklashCalibrationError",
    "BacklashCalibrationProfile",
    "BacklashCalibrationRuntime",
    "axes_identity",
]
