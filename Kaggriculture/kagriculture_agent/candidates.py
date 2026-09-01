"""Stable route-candidate factories used by evaluation tooling."""

from collections.abc import Callable, Mapping
from typing import Any

from .policy import Policy


CANDIDATES = ("current", "melon", "premium", "mixed")


def candidate_policy(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Return a fresh stateful route policy's observation callable."""
    if name not in CANDIDATES:
        raise ValueError(f"unsupported candidate: {name}")
    return Policy(strategy=name).act
