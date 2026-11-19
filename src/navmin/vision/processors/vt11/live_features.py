"""Owner-thread runtime flags for REV5 DAY observation aids."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vt11LiveFeatureSnapshot:
    """One atomic snapshot of the three low-light/blur-controlled REV5 flags."""

    revision: int
    visible_blur_robust: bool
    day_maneuver_search_enabled: bool
    visible_line_recovery_enabled: bool


class Vt11LiveFeatures:
    """Mutable owner-thread feature state shared by one VT11 target episode."""

    def __init__(self, profile_name: str) -> None:
        profile = str(profile_name).lower()
        if profile not in {"day", "thermal"}:
            raise ValueError("profile_name must be 'day' or 'thermal'")
        self._profile_name = profile
        self._revision = 0
        self._visible_blur_robust = False
        self._day_maneuver_search_enabled = False
        self._visible_line_recovery_enabled = False

    @property
    def profile_name(self) -> str:
        return self._profile_name

    @property
    def snapshot(self) -> Vt11LiveFeatureSnapshot:
        return Vt11LiveFeatureSnapshot(
            revision=self._revision,
            visible_blur_robust=self._visible_blur_robust,
            day_maneuver_search_enabled=self._day_maneuver_search_enabled,
            visible_line_recovery_enabled=self._visible_line_recovery_enabled,
        )

    def set_low_light_blur_mode(self, enabled: bool) -> bool:
        """Atomically map one operator setting to exactly three DAY-only flags."""
        active = bool(enabled) and self._profile_name == "day"
        next_values = (active, active, active)
        current = (
            self._visible_blur_robust,
            self._day_maneuver_search_enabled,
            self._visible_line_recovery_enabled,
        )
        if next_values == current:
            return False
        (
            self._visible_blur_robust,
            self._day_maneuver_search_enabled,
            self._visible_line_recovery_enabled,
        ) = next_values
        self._revision += 1
        return True


__all__ = ["Vt11LiveFeatureSnapshot", "Vt11LiveFeatures"]
