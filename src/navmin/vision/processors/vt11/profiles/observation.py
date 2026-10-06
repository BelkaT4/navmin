"""VT-11 observation profiles.

The tracking state machine remains common.  A profile changes only the
image-domain assumptions: candidate generation, identity weights, motion/
reacquisition conservatism and adaptation rates.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

PROFILE_NAMES = ("day", "thermal")
REQUESTED_PROFILES = ("auto",) + PROFILE_NAMES


def profile_from_video_mode(video_mode: str) -> str:
    """Map the legacy resolved video mode to a VT-11 observation profile."""
    return "thermal" if str(video_mode).lower() == "thermal" else "day"


def video_mode_from_profile(profile: str) -> str:
    p = str(profile).lower()
    if p == "thermal":
        return "thermal"
    if p == "day":
        return "visible"
    return "auto"


def resolve_profile_name(requested: str, *, inferred_video_mode: str = "visible") -> str:
    req = str(requested).lower()
    if req == "auto":
        return profile_from_video_mode(inferred_video_mode)
    if req not in PROFILE_NAMES:
        raise ValueError(f"Unknown observation profile: {requested}")
    return req


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def effective_config(cfg: Mapping[str, Any], profile_name: str) -> dict[str, Any]:
    """Return config with profile-specific overrides applied.

    ``profiles.<name>.overrides`` may contain partial copies of any top-level
    runtime sections (acquire, tracking, reacquisition, adaptation, thermal,
    identity).  The original config is never modified.
    """
    name = str(profile_name).lower()
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown observation profile: {profile_name}")
    profiles = cfg.get("profiles", {})
    spec = profiles.get(name)
    if not isinstance(spec, Mapping):
        raise ValueError(f"Missing profiles.{name}")
    overrides = spec.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError(f"profiles.{name}.overrides must be a mapping")
    return _deep_merge(dict(cfg), overrides)


def profile_identity_weights(cfg: Mapping[str, Any], profile_name: str) -> dict[str, float]:
    eff = effective_config(cfg, profile_name)
    weights = eff.get("identity", {}).get("weights", {})
    return {str(k): float(v) for k, v in weights.items()}


def profile_label(cfg: Mapping[str, Any], profile_name: str) -> str:
    spec = cfg.get("profiles", {}).get(profile_name, {})
    return str(spec.get("label", profile_name.upper()))


def profile_summary(cfg: Mapping[str, Any], profile_name: str) -> str:
    spec = cfg.get("profiles", {}).get(profile_name, {})
    return str(spec.get("summary", ""))
