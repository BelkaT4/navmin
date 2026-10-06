"""Resolved VT11 algorithm configuration for incremental embedding."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Mapping

from navmin.vision.processors.vt11.acquisition.candidate_detector import (
    CandidateDetectorConfig,
    config_from_dict as detector_config_from_dict,
)
from navmin.vision.processors.vt11.acquisition.initial_train import InitialTrainConfig
from navmin.vision.processors.vt11.adaptation.ema import EMAConfig
from navmin.vision.processors.vt11.identity.scorer import AcquireScoringConfig
from navmin.vision.processors.vt11.thermal.processing import (
    ThermalProcessingConfig,
    config_from_dict as thermal_config_from_dict,
)
from navmin.vision.processors.vt11.tracking.reacquisition import ReacquisitionConfig
from navmin.vision.processors.vt11.tracking.session import TrackingSessionConfig
from navmin.vision.processors.vt11.tracking.tracker import TrackingConfig


def _config_dataclass[ConfigT](
    cls: type[ConfigT], data: Mapping[str, Any], *, aliases: Mapping[str, str] | None = None
) -> ConfigT:
    aliases = aliases or {}
    values = dict(data)
    for source, destination in aliases.items():
        if source in values:
            values[destination] = values[source]
    allowed = {field.name for field in fields(cls) if field.init}
    config = cls(**{key: value for key, value in values.items() if key in allowed})
    validate = getattr(config, "validate", None)
    if validate is not None:
        validate()
    return config


@dataclass(frozen=True)
class Vt11RuntimeSettings:
    max_retrospective_replay_frames: int = 6

    def validate(self) -> None:
        if not 0 <= self.max_retrospective_replay_frames <= 6:
            raise ValueError("max_retrospective_replay_frames must be in [0, 6]")


@dataclass(frozen=True)
class Vt11ResolvedConfig:
    profile_name: str
    effective_config: Mapping[str, Any]
    detector: CandidateDetectorConfig
    acquire_scoring: AcquireScoringConfig
    initial_train: InitialTrainConfig
    tracking: TrackingConfig
    reacquisition: ReacquisitionConfig
    finite_session: TrackingSessionConfig
    adaptation: EMAConfig
    thermal: ThermalProcessingConfig
    identity_weights: Mapping[str, float]
    thermal_identity_weights: Mapping[str, float]

    @classmethod
    def from_effective_config(
        cls, profile_name: str, effective_config: Mapping[str, Any]
    ) -> Vt11ResolvedConfig:
        profile = str(profile_name).lower()
        if profile not in {"day", "thermal"}:
            raise ValueError("profile_name must be 'day' or 'thermal'")
        cfg = deepcopy(dict(effective_config))
        acquire = cfg["acquire"]
        thermal_data = dict(cfg.get("thermal", {}))
        detector_data = dict(acquire["candidate_detector"])
        for key in (
            "thermal_top_hat_kernel_px",
            "thermal_top_hat_scale",
            "thermal_top_hat_max_px",
        ):
            if key in thermal_data:
                detector_data[key] = thermal_data[key]
        scoring = _config_dataclass(
            AcquireScoringConfig, acquire["scoring"]
        )
        train_data = dict(acquire["initial_train"])
        train_data["initial_frames"] = acquire["initial_frames"]
        train_data["top_k"] = acquire["top_k_candidates"]
        tracking = _config_dataclass(
            TrackingConfig,
            cfg["tracking"],
            aliases={"top_k_candidates": "top_k"},
        )
        reacquisition = _config_dataclass(
            ReacquisitionConfig,
            cfg["reacquisition"],
            aliases={"top_k_candidates": "top_k"},
        )
        adaptation_data = dict(cfg["adaptation"]["ema"])
        adaptation_data["mode"] = cfg["adaptation"].get("mode", "off")
        identity_weights = {
            str(key): float(value)
            for key, value in cfg["identity"]["weights"].items()
        }
        thermal_identity_weights = {
            str(key): float(value)
            for key, value in cfg["identity"].get("thermal_weights", {}).items()
        }
        return cls(
            profile_name=profile,
            effective_config=MappingProxyType(cfg),
            detector=detector_config_from_dict(detector_data),
            acquire_scoring=scoring,
            initial_train=_config_dataclass(InitialTrainConfig, train_data),
            tracking=tracking,
            reacquisition=reacquisition,
            finite_session=TrackingSessionConfig(
                max_reacq_cycles=int(
                    cfg["reacquisition"].get("max_reacq_cycles", 5)
                )
            ),
            adaptation=_config_dataclass(EMAConfig, adaptation_data),
            thermal=thermal_config_from_dict(thermal_data),
            identity_weights=MappingProxyType(identity_weights),
            thermal_identity_weights=MappingProxyType(thermal_identity_weights),
        )
