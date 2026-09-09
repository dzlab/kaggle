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
IDENTITY_FIELDS = (
    "experiment_id",
    "feature_variant",
    "training_mode",
    "action_representation",
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


def validate_identity_consistency(
    top_level: Any, nested: Any, *, source: str = "identity",
) -> None:
    """Reject contradictory duplicated identity fields while allowing legacy omissions."""
    if top_level is None or nested is None:
        return
    if not isinstance(top_level, dict) or not isinstance(nested, dict):
        raise ValueError(f"{source} identity layers must be objects")
    for field in IDENTITY_FIELDS:
        if field in top_level and field in nested:
            left = top_level[field]
            right = nested[field]
            if field == "action_representation":
                left = left or DEFAULT_ACTION_REPRESENTATION
                right = right or DEFAULT_ACTION_REPRESENTATION
            if left != right:
                raise ValueError(
                    f"{source} {field} conflicts between top-level and nested identity"
                )


def validate_training_identity(
    experiment_id: Any, feature_variant: Any, training_mode: Any, *, source: str = "",
) -> None:
    validate_experiment_id(experiment_id, source=source)
    validate_feature_variant(feature_variant, source=source)
    validate_training_mode(training_mode, source=source)
