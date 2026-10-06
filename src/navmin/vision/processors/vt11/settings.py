"""Temporary owner-local production settings for the VT11 backend."""

from typing import Literal

VT11_PROFILE: Literal["day", "thermal"] = "day"

__all__ = ["VT11_PROFILE"]
