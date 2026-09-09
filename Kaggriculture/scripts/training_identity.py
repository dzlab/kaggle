"""Canonical experiment identity values and validation."""

from __future__ import annotations

from typing import Any

DEFAULT_EXPERIMENT_ID = "orbit-policy-v1"
FEATURE_VARIANTS = ("production_v1", "experimental_context_v1")
ACTION_REPRESENTATIONS = ("current_v1", "target_first_v1")
DEFAULT_ACTION_REPRESENTATION = "current_v1"
TRAINING_MODES = (
    "behavior_clone_then_ppo",
    "pure_ppo",
    "reduced_behavior_clone_then_ppo",
)


def validate_experiment_id(value: Any, *, source: str = "") -> None:
    prefix = f"{source} " if source else ""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{prefix}experiment_id must be a non-empty string")


def validate_feature_variant(value: Any, *, source: str = "") -> None:
    prefix = f"{source} " if source else ""
    if value not in FEATURE_VARIANTS:
        choices = ", ".join(FEATURE_VARIANTS)
        raise ValueError(f"{prefix}feature_variant must be one of: {choices}")


def validate_training_mode(value: Any, *, source: str = "") -> None:
    prefix = f"{source} " if source else ""
    if value not in TRAINING_MODES:
        choices = ", ".join(TRAINING_MODES)
        raise ValueError(f"{prefix}training_mode must be one of: {choices}")


def validate_action_representation(value: Any, *, source: str = "") -> None:
    prefix = f"{source} " if source else ""
    if value not in ACTION_REPRESENTATIONS:
        choices = ", ".join(ACTION_REPRESENTATIONS)
        raise ValueError(f"{prefix}action_representation must be one of: {choices}")


def validate_training_identity(
    experiment_id: Any, feature_variant: Any, training_mode: Any, *, source: str = "",
) -> None:
    validate_experiment_id(experiment_id, source=source)
    validate_feature_variant(feature_variant, source=source)
    validate_training_mode(training_mode, source=source)
