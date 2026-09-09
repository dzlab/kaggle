"""Dependency-free identity values and validation shared by runtime code."""

from __future__ import annotations

from typing import Any


ACTION_REPRESENTATIONS = ("current_v1", "target_first_v1")
DEFAULT_ACTION_REPRESENTATION = "current_v1"


def validate_action_representation(value: Any, *, source: str = "") -> None:
    prefix = f"{source} " if source else ""
    if value not in ACTION_REPRESENTATIONS:
        choices = ", ".join(ACTION_REPRESENTATIONS)
        raise ValueError(f"{prefix}action_representation must be one of: {choices}")
