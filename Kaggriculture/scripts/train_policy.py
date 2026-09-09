"""Train compact Kaggriculture policies with behavior cloning and PPO helpers."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.checkpoints import (
    CheckpointError,
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
    restore_checkpoint,
    save_checkpoint,
)
from kagriculture_agent.action_objectives import conditional_action_objectives
from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION, extract_features
from kagriculture_agent.league import (
    DEFAULT_OPPONENT_PROBABILITIES,
    LeagueSampler,
    OpponentMatch,
)
from kagriculture_agent.model import (
    ACTION_VOCAB,
    DEFAULT_MODEL_DEPTH,
    DEFAULT_MODEL_WIDTH,
    MODEL_VERSION,
    CompactPolicyNet,
    model_parameter_count,
    require_torch,
    resolve_device,
    set_training_seed,
    validate_model_shape,
)
from kagriculture_agent.reward_shaping import shaped_transition_reward, should_bootstrap_truncate
from scripts.training_identity import (
    DEFAULT_EXPERIMENT_ID,
    FEATURE_VARIANTS,
    TRAINING_MODES,
    validate_experiment_id,
    validate_feature_variant,
    validate_training_mode,
)

PROMOTION_MATCH_SIZE = 100
LOG_RATIO_CLAMP = 20.0
# PPO starts from a behavior-cloned policy and uses a small trust-region step.
# The BC optimizer state is cleared before this phase; this rate keeps the first
# on-policy update below the default target-KL gate on the compact network.
PPO_LEARNING_RATE = 1e-5
_DIRECTION_DELTAS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}
_CURRENT_TILE_KINDS = {
    "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "DROP", "SELL", "DIG", "WEED",
}


def resolve_behavior_clone_steps(training_mode: str, configured_steps: int) -> int:
    """Resolve the effective BC epoch budget for a supported training mode."""
    _validate_training_mode(training_mode, source="requested")
    if type(configured_steps) is not int or configured_steps < 0:
        raise ValueError("behavior_clone_steps must be a nonnegative integer")
    if training_mode == "pure_ppo":
        return 0
    if training_mode == "reduced_behavior_clone_then_ppo":
        return max(1, configured_steps // 4)
    return configured_steps


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    target_kl: float = 0.03
    rollout_steps: int = 64
    kl_coef: float = 0.10
    prior_ce_coef: float = 0.01
    ppo_epochs: int = 1
    potential_reward_coef: float = 0.0
    no_progress_window: int = 0
    resolved_margin: float = 0.0
    training_action_mask: bool = False

    def __post_init__(self) -> None:
        for name in (
            "gamma", "gae_lambda", "clip_epsilon", "value_coef", "entropy_coef",
            "target_kl", "rollout_steps", "kl_coef", "prior_ce_coef", "ppo_epochs",
            "potential_reward_coef", "no_progress_window", "resolved_margin",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("rollout_steps", "ppo_epochs"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if type(self.no_progress_window) is not int:
            raise ValueError("no_progress_window must be an integer")
        if type(self.training_action_mask) is not bool:
            raise ValueError("training_action_mask must be boolean")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]; gamma=1.0 is for explicit experiments only")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        for name in (
            "value_coef", "entropy_coef", "kl_coef", "prior_ce_coef",
            "potential_reward_coef", "resolved_margin",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive")
        if self.rollout_steps < 1 or self.ppo_epochs < 1:
            raise ValueError("rollout_steps and ppo_epochs must be positive")
        if self.no_progress_window < 0:
            raise ValueError("no_progress_window must be nonnegative")


@dataclass(frozen=True)
class TrainingContract:
    """Validated configuration and input count required to resume training."""

    configuration: dict[str, Any]
    transition_count: int


_RESUME_CONFIGURATION_FIELDS = (
    "experiment_id",
    "feature_variant",
    "training_mode",
    "input_trajectory",
    "steps",
    "batch_size",
    "seed",
    "ppo_steps",
    "behavior_clone_steps",
    "model_width",
    "model_depth",
    "prior_checkpoint",
    "offline_ppo_fallback",
    "ppo_config",
)
_RUNTIME_CONFIGURATION_FIELDS = {"device", "checkpoint_interval"}
_PPO_FLOAT_FIELDS = {
    "gamma", "gae_lambda", "clip_epsilon", "value_coef", "entropy_coef",
    "target_kl", "kl_coef", "prior_ce_coef", "potential_reward_coef", "resolved_margin",
}
_PPO_INTEGER_FIELDS = {"rollout_steps", "ppo_epochs", "no_progress_window"}
_PPO_BOOLEAN_FIELDS = {"training_action_mask"}
_PPO_BACKWARDS_COMPATIBLE_DEFAULTS = {
    "potential_reward_coef": 0.0,
    "no_progress_window": 0,
    "resolved_margin": 0.0,
    "training_action_mask": False,
}
_PPO_RESUME_METRIC_FIELDS = {
    "ppo_updates", "rollout_count", "early_stopped", "last_metrics",
    "promotion", "completed_steps",
}
_PPO_OPTIONAL_RESUME_METRIC_FIELDS = {
    "shaping_count", "truncation_count", "league_composition",
    "league_checkpoint_identities",
}
_PPO_LEAGUE_COMPOSITION_DEFAULT = {
    "current": 0, "mixed": 0, "random": 0, "starter": 0, "checkpoint": 0,
}


def _validate_experiment_id(value: Any, *, source: str) -> None:
    validate_experiment_id(value, source=f"{source} configuration")


def _validate_feature_variant(value: Any, *, source: str) -> None:
    validate_feature_variant(value, source=f"{source} configuration")


def _validate_training_mode(value: Any, *, source: str) -> None:
    validate_training_mode(value, source=f"{source} configuration")


def _validate_content_identity(value: Any, *, source: str, label: str) -> None:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError(f"{source} {label} must be a content identity object")
    path = value["path"]
    digest = value["sha256"]
    if type(path) is not str or not path or not Path(path).is_absolute():
        raise ValueError(f"{source} {label} path must be an absolute string")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{source} {label} sha256 must be a lowercase digest")


def _validate_prior_checkpoint_identity(value: Any, *, source: str) -> None:
    if value is None:
        return
    _validate_content_identity(value, source=source, label="prior_checkpoint")


def _validate_ppo_configuration(value: Any, *, source: str) -> dict[str, Any]:
    expected_fields = _PPO_FLOAT_FIELDS | _PPO_INTEGER_FIELDS | _PPO_BOOLEAN_FIELDS
    if type(value) is not dict:
        raise ValueError(f"{source} ppo_config must be an object")
    normalized = dict(value)
    missing = expected_fields - set(normalized)
    unsupported_missing = missing - set(_PPO_BACKWARDS_COMPATIBLE_DEFAULTS)
    if unsupported_missing:
        missing = unsupported_missing
    else:
        for field in missing:
            normalized[field] = _PPO_BACKWARDS_COMPATIBLE_DEFAULTS[field]
    missing = sorted(missing)
    unexpected = sorted(set(normalized) - expected_fields)
    if missing:
        raise ValueError(f"{source} ppo_config is missing required fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{source} ppo_config has unexpected fields: {', '.join(unexpected)}")
    for field in sorted(_PPO_FLOAT_FIELDS):
        if type(normalized[field]) is not float:
            raise ValueError(f"{source} ppo_config {field} must be a float")
    for field in sorted(_PPO_INTEGER_FIELDS):
        if type(normalized[field]) is not int:
            raise ValueError(f"{source} ppo_config {field} must be an integer")
    for field in sorted(_PPO_BOOLEAN_FIELDS):
        if type(normalized[field]) is not bool:
            raise ValueError(f"{source} ppo_config {field} must be a boolean")
    try:
        PPOConfig(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} ppo_config is invalid: {exc}") from exc
    return normalized


def _validate_configuration_shape(configuration: Any, *, source: str) -> None:
    expected_fields = set(_RESUME_CONFIGURATION_FIELDS) | _RUNTIME_CONFIGURATION_FIELDS
    if type(configuration) is not dict:
        raise ValueError(f"{source} configuration must be an object")
    for field, default in (
        ("experiment_id", DEFAULT_EXPERIMENT_ID),
        ("feature_variant", FEATURE_VARIANTS[0]),
        ("training_mode", TRAINING_MODES[0]),
    ):
        # Older checkpoints and test fixtures predate the identity contract;
        # treat omitted fields as the production defaults while validating all
        # newly-created contracts strictly below.
        configuration.setdefault(field, default)
    configuration.setdefault(
        "behavior_clone_steps",
        resolve_behavior_clone_steps(configuration["training_mode"], configuration.get("steps", 1)),
    )
    configuration.setdefault("model_width", DEFAULT_MODEL_WIDTH)
    configuration.setdefault("model_depth", DEFAULT_MODEL_DEPTH)
    missing = sorted(expected_fields - set(configuration))
    unexpected = sorted(set(configuration) - expected_fields)
    if missing:
        raise ValueError(f"{source} configuration is missing required fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{source} configuration has unexpected fields: {', '.join(unexpected)}")
    _validate_content_identity(
        configuration["input_trajectory"], source=source, label="input trajectory",
    )
    for field, minimum in (
        ("steps", 1), ("batch_size", 1), ("ppo_steps", 0),
        ("behavior_clone_steps", 0),
        ("checkpoint_interval", 1),
    ):
        value = configuration[field]
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"{source} configuration {field} must be an integer at least {minimum}"
            )
    try:
        validate_model_shape(
            configuration["model_width"], configuration["model_depth"], source=f"{source} configuration",
        )
    except ValueError:
        raise
    if type(configuration["seed"]) is not int:
        raise ValueError(f"{source} configuration seed must be an integer")
    _validate_experiment_id(configuration["experiment_id"], source=source)
    _validate_feature_variant(configuration["feature_variant"], source=source)
    _validate_training_mode(configuration["training_mode"], source=source)
    if type(configuration["device"]) is not str or not configuration["device"]:
        raise ValueError(f"{source} configuration device must be a nonempty string")
    if type(configuration["offline_ppo_fallback"]) is not bool:
        raise ValueError(f"{source} configuration offline_ppo_fallback must be boolean")
    _validate_prior_checkpoint_identity(configuration["prior_checkpoint"], source=source)
    configuration["ppo_config"] = _validate_ppo_configuration(
        configuration["ppo_config"], source=source,
    )


def build_training_contract(
    *, input_path: str | Path, steps: int, batch_size: int, seed: int = 0,
    ppo_steps: int = 0, device: str = "auto", checkpoint_interval: int = 100,
    prior_checkpoint: str | Path | None = None,
    offline_ppo_fallback: bool = False, resolved_device: Any | None = None,
    ppo_config: PPOConfig | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    behavior_clone_steps: int | None = None,
    model_width: int = DEFAULT_MODEL_WIDTH,
    model_depth: int = DEFAULT_MODEL_DEPTH,
) -> TrainingContract:
    """Build the canonical input/configuration contract used by training and resume.

    ``steps`` and ``batch_size`` retain the trainer's valid integer normalization
    where nonpositive values become one. All other public scalar inputs must
    already have their declared types; strings, floats, and booleans are not
    silently coerced.
    """
    if type(steps) is not int:
        raise ValueError("steps must be an integer")
    if type(batch_size) is not int:
        raise ValueError("batch_size must be an integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(ppo_steps) is not int:
        raise ValueError("ppo_steps must be an integer")
    if type(device) is not str:
        raise ValueError("device must be a string")
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    if behavior_clone_steps is not None and (
        type(behavior_clone_steps) is not int or behavior_clone_steps < 0
    ):
        raise ValueError("behavior_clone_steps must be a nonnegative integer")
    validate_model_shape(model_width, model_depth, source="requested")
    if type(offline_ppo_fallback) is not bool:
        raise ValueError("offline_ppo_fallback must be boolean")
    _validate_experiment_id(experiment_id, source="requested")
    _validate_feature_variant(feature_variant, source="requested")
    _validate_training_mode(training_mode, source="requested")
    if ppo_config is not None and not isinstance(ppo_config, PPOConfig):
        raise ValueError("ppo_config must be a PPOConfig or None")
    resolved = resolve_device(device) if resolved_device is None else resolved_device
    normalized_batch_size = max(1, batch_size)
    normalized_steps = max(1, steps)
    configured_behavior_clone_steps = (
        normalized_steps if behavior_clone_steps is None else behavior_clone_steps
    )
    effective_behavior_clone_steps = resolve_behavior_clone_steps(
        training_mode, configured_behavior_clone_steps,
    )
    input_identity = _trajectory_identity(input_path)
    transitions = _read_transitions(input_path)
    configuration = {
        "experiment_id": experiment_id,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "input_trajectory": input_identity,
        "steps": normalized_steps,
        "batch_size": normalized_batch_size,
        "seed": seed,
        "ppo_steps": ppo_steps,
        "behavior_clone_steps": effective_behavior_clone_steps,
        "model_width": model_width,
        "model_depth": model_depth,
        "device": str(resolved),
        "prior_checkpoint": _checkpoint_identity(prior_checkpoint),
        "offline_ppo_fallback": offline_ppo_fallback,
        "ppo_config": asdict(ppo_config or PPOConfig()),
        "checkpoint_interval": checkpoint_interval,
    }
    _validate_configuration_shape(configuration, source="requested")
    return TrainingContract(configuration=configuration, transition_count=len(transitions))


def _validate_resume_configuration(
    saved: dict[str, Any], requested: dict[str, Any], *,
    allow_ppo_extension: bool = False,
) -> None:
    if type(allow_ppo_extension) is not bool:
        raise ValueError("allow_ppo_extension must be boolean")
    _validate_configuration_shape(saved, source="saved")
    _validate_configuration_shape(requested, source="requested")
    for field in _RESUME_CONFIGURATION_FIELDS:
        if field == "ppo_steps" and allow_ppo_extension:
            if saved[field] >= requested[field]:
                raise ValueError(
                    "resume checkpoint configuration mismatch for ppo_steps: "
                    "allow_ppo_extension requires the requested target to be greater "
                    f"than the saved target ({requested[field]} <= {saved[field]})"
                )
            continue
        if type(saved[field]) is not type(requested[field]) or saved[field] != requested[field]:
            label = "input trajectory" if field == "input_trajectory" else field
            raise ValueError(
                f"resume checkpoint configuration mismatch for {label}: "
                f"saved {saved[field]!r}, requested {requested[field]!r}"
            )


def _checkpoint_identity(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"prior checkpoint does not exist: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(checkpoint_path), "sha256": digest.hexdigest()}


def _trajectory_identity(path: str | Path) -> dict[str, str]:
    trajectory_path = Path(path).resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"input trajectory does not exist: {trajectory_path}")
    digest = hashlib.sha256()
    with trajectory_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(trajectory_path), "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class WorkerLabels:
    act: list[int]
    target: list[int]
    kind: list[int]


@dataclass(frozen=True)
class RolloutBatch:
    transitions: list[dict[str, Any]]
    rewards: list[float]
    dones: list[bool]
    values: list[float]
    old_log_probs: list[float]
    advantages: list[float]
    returns: list[float]
    worker: WorkerLabels
    market_items: list[int]
    market_quantities: list[int]
    market_active: list[int]
    bootstrap_values: list[float] = field(default_factory=list)
    bootstrap_truncated: list[bool] = field(default_factory=list)
    shaping_count: int = 0
    truncation_count: int = 0


class OpponentPool:
    """Deterministic league sampler for self-play PPO rollouts."""

    mixed_opponents = ("current", "random", "starter")

    probabilities = {
        "current": 0.40,
        "mixed": 0.15,
        "random": 0.10,
        "starter": 0.10,
        "checkpoint": 0.25,
    }

    def __init__(
        self, previous_checkpoints: Sequence[str | Path] | None = None, *,
        league_sampler: LeagueSampler | None = None,
        sampler: LeagueSampler | None = None,
        checkpoint_window: int = 5,
        probabilities: Mapping[str, object] | None = None,
        configured_checkpoints: Sequence[str | Path] | None = None,
    ) -> None:
        if league_sampler is not None and sampler is not None:
            raise ValueError("provide only one of league_sampler or sampler")
        selected_sampler = league_sampler if league_sampler is not None else sampler
        if selected_sampler is not None and not isinstance(selected_sampler, LeagueSampler):
            raise ValueError("league_sampler must be a LeagueSampler or None")
        if type(checkpoint_window) is not int or checkpoint_window < 0:
            raise ValueError("checkpoint_window must be a nonnegative integer")
        if previous_checkpoints is None:
            previous_checkpoints = ()
        elif isinstance(previous_checkpoints, (str, bytes)) or not isinstance(
            previous_checkpoints, Sequence,
        ):
            raise ValueError("previous_checkpoints must be a sequence or None")
        candidates = tuple(str(path) for path in previous_checkpoints)
        if configured_checkpoints is not None:
            if isinstance(configured_checkpoints, (str, bytes)) or not isinstance(
                configured_checkpoints, Sequence,
            ):
                raise ValueError("configured_checkpoints must be a sequence of paths")
            configured_candidates = tuple(str(path) for path in configured_checkpoints)
        else:
            configured_candidates = candidates
        self.configured_checkpoint_candidates = configured_candidates
        self.checkpoint_window = checkpoint_window
        self.checkpoint_candidates = candidates[-checkpoint_window:] if checkpoint_window else ()
        configured_probabilities = (
            dict(DEFAULT_OPPONENT_PROBABILITIES)
            if probabilities is None else dict(probabilities)
        )
        self.league_sampler = selected_sampler or LeagueSampler(
            probabilities=configured_probabilities,
            checkpoint_candidates=self.checkpoint_candidates,
        )
        self.configured_probabilities = dict(configured_probabilities)
        if selected_sampler is not None:
            self.configured_probabilities = dict(
                getattr(selected_sampler, "configured_probabilities", selected_sampler.probabilities)
            )
            if configured_checkpoints is None:
                self.configured_checkpoint_candidates = tuple(
                    getattr(
                        selected_sampler,
                        "configured_checkpoint_candidates",
                        self.configured_checkpoint_candidates,
                    )
                )
        self.probabilities = dict(self.league_sampler.probabilities)

    @property
    def league_configuration(self) -> dict[str, Any]:
        """Return the original league inputs for rollout-manifest provenance."""
        return {
            "league_probabilities": dict(self.configured_probabilities),
            "league_checkpoint_window": self.checkpoint_window,
            "league_checkpoints": list(self.configured_checkpoint_candidates),
        }

    def sample(self, index: int, *, seed: int = 0) -> OpponentMatch:
        return self.league_sampler.sample(index, seed=seed)

    def schedule(self, *, count: int, seed: int = 0) -> list[OpponentMatch]:
        return self.league_sampler.schedule(count, seed=seed)


def should_promote(
    *, wins: int, games: int, threshold: float = 0.70,
    match_size: int = PROMOTION_MATCH_SIZE,
) -> bool:
    """Promote only after the fixed 100-game match exceeds 70% wins."""
    return games == match_size and games > 0 and (wins / games) > threshold


def terminal_bank_margin_reward(final_bank: Any, opponent_final_bank: Any) -> float:
    try:
        final = float(final_bank)
        opponent = float(opponent_final_bank)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(final) or not math.isfinite(opponent):
        return 0.0
    return math.tanh((final - opponent) / 1000.0)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_advantages(advantages: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    values = [_finite_float(value, "advantages") for value in advantages]
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= epsilon:
        return [0.0 for _value in values]
    return [(value - mean) / std for value in values]


def generalized_advantage_estimate(
    *, rewards: Sequence[float], values: Sequence[float], dones: Sequence[bool],
    gamma: float = PPOConfig.gamma, gae_lambda: float = PPOConfig.gae_lambda,
    bootstrap_values: Sequence[float] | None = None,
    bootstrap_truncated: Sequence[bool] | None = None,
) -> tuple[list[float], list[float]]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have the same length")
    gamma = _finite_float(gamma, "gamma")
    gae_lambda = _finite_float(gae_lambda, "gae_lambda")
    rewards = [_finite_float(reward, "rewards") for reward in rewards]
    values = [_finite_float(value, "values") for value in values]
    if bootstrap_values is None:
        final_bootstrap_values = [0.0 for _value in rewards]
    else:
        if len(bootstrap_values) != len(rewards):
            raise ValueError("bootstrap_values must have the same length as rewards")
        final_bootstrap_values = [
            _finite_float(value, "bootstrap_values") for value in bootstrap_values
        ]
    if bootstrap_truncated is None:
        truncation_flags = [False for _value in rewards]
        if bootstrap_values is not None and truncation_flags:
            # Preserve the old single-final-bootstrap API for direct callers.
            truncation_flags[-1] = True
    else:
        if len(bootstrap_truncated) != len(rewards):
            raise ValueError("bootstrap_truncated must have the same length as rewards")
        if any(type(value) is not bool for value in bootstrap_truncated):
            raise ValueError("bootstrap_truncated must contain only booleans")
        truncation_flags = list(bootstrap_truncated)
    advantages = [0.0 for _ in rewards]
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        is_terminal = bool(dones[index])
        is_truncated = truncation_flags[index] and not is_terminal
        delta_nonterminal = 0.0 if is_terminal else 1.0
        continuation = 0.0 if is_terminal or is_truncated else 1.0
        if is_terminal:
            next_value = 0.0
        elif is_truncated:
            next_value = final_bootstrap_values[index]
        elif index + 1 < len(rewards):
            next_value = values[index + 1]
        else:
            next_value = 0.0
        delta = rewards[index] + gamma * next_value * delta_nonterminal - values[index]
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def clipped_policy_terms(
    *, new_log_probs: Sequence[float], old_log_probs: Sequence[float],
    advantages: Sequence[float], clip_epsilon: float = PPOConfig.clip_epsilon,
) -> dict[str, list[float] | float]:
    if not (len(new_log_probs) == len(old_log_probs) == len(advantages)):
        raise ValueError("new_log_probs, old_log_probs, and advantages must have the same length")
    ratios = [
        math.exp(max(-LOG_RATIO_CLAMP, min(LOG_RATIO_CLAMP, _finite_float(new, "new_log_probs") - _finite_float(old, "old_log_probs"))))
        for new, old in zip(new_log_probs, old_log_probs)
    ]
    low = 1.0 - clip_epsilon
    high = 1.0 + clip_epsilon
    clipped = [min(high, max(low, ratio)) for ratio in ratios]
    objectives = [
        min(ratio * _finite_float(advantage, "advantages"), clipped_ratio * _finite_float(advantage, "advantages"))
        for ratio, clipped_ratio, advantage in zip(ratios, clipped, advantages)
    ]
    loss = -sum(objectives) / len(objectives) if objectives else 0.0
    return {"ratios": ratios, "clipped_ratios": clipped, "loss": loss}


def clipped_value_loss(
    *, values: Sequence[float], old_values: Sequence[float], returns: Sequence[float],
    clip_epsilon: float = PPOConfig.clip_epsilon,
) -> float:
    if not (len(values) == len(old_values) == len(returns)):
        raise ValueError("values, old_values, and returns must have the same length")
    losses = []
    for value, old, target in zip(values, old_values, returns):
        clipped = float(old) + min(clip_epsilon, max(-clip_epsilon, float(value) - float(old)))
        losses.append(max((float(value) - float(target)) ** 2, (clipped - float(target)) ** 2))
    return sum(losses) / len(losses) if losses else 0.0


def ppo_total_loss(
    *, policy_loss: float, value_loss: float, entropy: float, kl_to_prior: float,
    cross_entropy_to_prior: float, config: PPOConfig,
) -> float:
    """Combine PPO clipped losses with previous-checkpoint regularization."""
    return (
        float(policy_loss)
        + config.value_coef * float(value_loss)
        - config.entropy_coef * float(entropy)
        + config.kl_coef * float(kl_to_prior)
        + config.prior_ce_coef * float(cross_entropy_to_prior)
    )


def approximate_kl(new_log_probs: Sequence[float], old_log_probs: Sequence[float]) -> float:
    if len(new_log_probs) != len(old_log_probs):
        raise ValueError("new_log_probs and old_log_probs must have the same length")
    if not new_log_probs:
        return 0.0
    new_values = [_finite_float(value, "new_log_probs") for value in new_log_probs]
    old_values = [_finite_float(value, "old_log_probs") for value in old_log_probs]
    return sum(old - new for new, old in zip(new_values, old_values)) / len(new_values)


def _read_transitions(path: str | Path) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON transition at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"transition at line {line_number} must be an object")
            transitions.append(row)
    if not transitions:
        raise ValueError("at least one transition is required")
    return transitions


def epoch_minibatches(*, count: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Return a deterministic complete epoch partition over all transition indices."""
    if count < 1:
        raise ValueError("count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = list(range(count))
    random.Random((int(seed) << 16) + int(epoch)).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, count, batch_size)]


def _command_kind(command: Any) -> str:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
        return "PASS"
    name = str(command[0]).upper()
    if name in {"NORTH", "SOUTH", "EAST", "WEST"}:
        return "MOVE"
    return name if name in ACTION_VOCAB["worker_kinds"] else "PASS"


def _selected_farm(observation: dict[str, Any]) -> dict[str, Any]:
    farm = observation.get("farm")
    if isinstance(farm, dict):
        return farm
    farms = observation.get("farms")
    if isinstance(farms, Sequence) and not isinstance(farms, (str, bytes)):
        try:
            player = int(observation.get("player", 0))
        except (TypeError, ValueError, OverflowError):
            player = 0
        if 0 <= player < len(farms) and isinstance(farms[player], dict):
            return farms[player]
    return {}


def _worker_positions(observation: dict[str, Any]) -> list[tuple[int, int]]:
    workers = _selected_farm(observation).get("workers", [])
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)):
        workers = []
    positions: list[tuple[int, int]] = []
    for worker in list(workers)[:10]:
        raw = worker.get("position", worker) if isinstance(worker, dict) else worker
        if isinstance(raw, dict):
            x_value, y_value = raw.get("x", 0), raw.get("y", 0)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            x_value, y_value = raw[0], raw[1]
        else:
            x_value, y_value = 0, 0
        try:
            x, y = int(x_value), int(y_value)
        except (TypeError, ValueError, OverflowError):
            x, y = 0, 0
        positions.append((max(0, min(9, x)), max(0, min(9, y))))
    positions.extend([(0, 0)] * (10 - len(positions)))
    return positions[:10]


def _target_index(command: Any, position: tuple[int, int]) -> int:
    x, y = position
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)) and command:
        name = str(command[0]).upper()
        if name in _DIRECTION_DELTAS:
            dx, dy = _DIRECTION_DELTAS[name]
            x = max(0, min(9, x + dx))
            y = max(0, min(9, y + dy))
        elif name not in _CURRENT_TILE_KINDS and len(command) >= 3:
            raw = command[2]
            if isinstance(raw, dict) and "x" in raw and "y" in raw:
                try:
                    x = max(0, min(9, int(raw["x"])))
                    y = max(0, min(9, int(raw["y"])))
                except (TypeError, ValueError, OverflowError):
                    pass
    return y * 10 + x


def worker_labels(action: dict[str, Any], observation: dict[str, Any] | None = None) -> WorkerLabels:
    commands = [action.get("farmer", ["PASS"])]
    hands = action.get("hands", [])
    if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes)):
        commands.extend(hands)
    commands = (commands + [["PASS"]] * 10)[:10]
    positions = _worker_positions(observation or {})
    kind_lookup = {kind: index for index, kind in enumerate(ACTION_VOCAB["worker_kinds"])}
    act_labels: list[int] = []
    target_labels: list[int] = []
    kind_labels: list[int] = []
    for command, position in zip(commands, positions):
        kind = _command_kind(command)
        act_labels.append(0 if kind == "PASS" else 1)
        target_labels.append(_target_index(command, position))
        kind_labels.append(kind_lookup.get(kind, 0))
    return WorkerLabels(act_labels, target_labels, kind_labels)


def _worker_labels(action: dict[str, Any], observation: dict[str, Any] | None = None) -> tuple[list[int], list[int], list[int]]:
    labels = worker_labels(action, observation)
    return labels.act, labels.target, labels.kind


def _market_labels(action: dict[str, Any]) -> tuple[int, int]:
    market = action.get("market", [])
    if not isinstance(market, Sequence) or isinstance(market, (str, bytes)) or not market:
        return 0, 0
    first = market[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)) or len(first) < 2:
        return 0, 0
    item_lookup = {item: index for index, item in enumerate(ACTION_VOCAB["market_items"])}
    quantity_lookup = {quantity: index for index, quantity in enumerate(ACTION_VOCAB["market_quantities"])}
    item = str(first[1]).upper()
    try:
        quantity = int(first[2]) if len(first) > 2 else 1
    except (TypeError, ValueError, OverflowError):
        quantity = 0
    quantity = min(ACTION_VOCAB["market_quantities"], key=lambda candidate: abs(candidate - quantity))
    return item_lookup.get(item, 0), quantity_lookup.get(quantity, 0)


def _market_active_label(action: dict[str, Any]) -> int:
    market = action.get("market", [])
    return int(
        isinstance(market, Sequence)
        and not isinstance(market, (str, bytes))
        and bool(market)
    )


def _transition_base_reward(transition: Mapping[str, Any], *, done: bool, config: PPOConfig) -> float:
    if done:
        terminal = terminal_bank_margin_reward(
            transition.get("final_bank"), transition.get("opponent_final_bank"),
        )
        if transition.get("final_bank") is None and transition.get("opponent_final_bank") is None:
            try:
                reward = float(transition.get("reward", 0.0))
            except (TypeError, ValueError, OverflowError):
                reward = 0.0
            return reward if math.isfinite(reward) else 0.0
        return terminal
    if config.potential_reward_coef == 0.0:
        # Preserve the legacy trajectory contract: nonterminal base rewards
        # are zero unless shaping is explicitly enabled.
        return 0.0
    try:
        reward = float(transition.get("reward", 0.0))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return reward if math.isfinite(reward) else 0.0


def _resolved_transition(transition: Mapping[str, Any], *, config: PPOConfig) -> bool:
    if bool(transition.get("done")):
        return False
    if bool(transition.get("bootstrap_truncated")):
        return True
    if should_bootstrap_truncate(
        transition.get("no_progress_steps", 0), config.no_progress_window,
    ):
        return True
    if config.resolved_margin <= 0.0:
        return False
    raw_margin = transition.get("bank_differential")
    if raw_margin is None:
        try:
            raw_margin = float(transition.get("final_bank")) - float(
                transition.get("opponent_final_bank"),
            )
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        margin = abs(float(raw_margin))
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(margin) and margin >= config.resolved_margin


def build_rollout_batch(
    transitions: Sequence[dict[str, Any]], *, config: PPOConfig,
    value_estimates: Sequence[float] | None = None,
    old_log_probs: Sequence[float] | None = None,
    bootstrap_values: Sequence[float] | None = None,
) -> RolloutBatch:
    rows = list(transitions)
    if not rows:
        raise ValueError("rollout batch requires at least one transition")
    values = [0.0 for _ in rows] if value_estimates is None else [float(value) for value in value_estimates]
    old = [0.0 for _ in rows] if old_log_probs is None else [float(value) for value in old_log_probs]
    if bootstrap_values is None:
        next_values = [0.0 for _ in rows]
    else:
        next_values = [float(value) for value in bootstrap_values]
    if (
        len(values) != len(rows)
        or len(old) != len(rows)
        or len(next_values) != len(rows)
    ):
        raise ValueError(
            "value_estimates, old_log_probs, and bootstrap_values must match transition count"
        )
    rewards: list[float] = []
    dones: list[bool] = []
    worker_act: list[list[int]] = []
    worker_target: list[list[int]] = []
    worker_kind: list[list[int]] = []
    market_items: list[int] = []
    market_quantities: list[int] = []
    market_active: list[int] = []
    truncation_flags: list[bool] = []
    shaping_count = 0
    truncation_count = 0
    for transition in rows:
        if not isinstance(transition, Mapping):
            raise ValueError("each transition must be a mapping")
        done = bool(transition.get("done"))
        bootstrap_truncated = _resolved_transition(transition, config=config)
        if bootstrap_truncated:
            truncation_count += 1
        truncation_flags.append(bootstrap_truncated)
        dones.append(done)
        base_reward = _transition_base_reward(transition, done=done, config=config)
        if config.potential_reward_coef != 0.0:
            shaped_reward = shaped_transition_reward(
                {**transition, "reward": base_reward},
                gamma=config.gamma,
                coefficient=config.potential_reward_coef,
            )
            if shaped_reward != base_reward:
                shaping_count += 1
            rewards.append(shaped_reward)
        else:
            rewards.append(base_reward)
        action = transition.get("action", {})
        if not isinstance(action, dict):
            action = {}
        observation = transition.get("observation", {})
        if not isinstance(observation, dict):
            observation = {}
        labels = worker_labels(action, observation)
        worker_act.append(labels.act)
        worker_target.append(labels.target)
        worker_kind.append(labels.kind)
        item, quantity = _market_labels(action)
        market_items.append(item)
        market_quantities.append(quantity)
        market_active.append(_market_active_label(action))
    advantages, returns = generalized_advantage_estimate(
        rewards=rewards, values=values, dones=dones,
        gamma=config.gamma, gae_lambda=config.gae_lambda,
        bootstrap_values=next_values,
        bootstrap_truncated=truncation_flags,
    )
    return RolloutBatch(
        transitions=rows,
        rewards=rewards,
        dones=dones,
        values=values,
        old_log_probs=old,
        advantages=normalize_advantages(advantages),
        returns=returns,
        worker=WorkerLabels(worker_act, worker_target, worker_kind),
        market_items=market_items,
        market_quantities=market_quantities,
        market_active=market_active,
        bootstrap_values=next_values,
        bootstrap_truncated=truncation_flags,
        shaping_count=shaping_count,
        truncation_count=truncation_count,
    )


def _select_outputs(
    outputs: dict[str, Any], batch: RolloutBatch, *, device: Any = None,
    training_action_mask: bool = False,
) -> tuple[Any, Any]:
    th = require_torch()
    if "market_active_logits" not in outputs:
        # Older injected/test policies predate the training-only head.  Give
        # them a neutral market-intent distribution while keeping the strict
        # helper contract intact for all objective calculations.
        outputs = dict(outputs)
        market_items = outputs.get("market_item_logits")
        if market_items is None or market_items.ndim != 2:
            raise ValueError("outputs is missing market_active_logits")
        outputs["market_active_logits"] = market_items.new_zeros(
            (market_items.shape[0], 2),
        )
    device = outputs["value"].device if device is None else device
    worker_act = th.tensor(batch.worker.act, dtype=th.long, device=device)
    worker_target = th.tensor(batch.worker.target, dtype=th.long, device=device)
    worker_kind = th.tensor(batch.worker.kind, dtype=th.long, device=device)
    market_items = th.tensor(batch.market_items, dtype=th.long, device=device)
    market_quantities = th.tensor(batch.market_quantities, dtype=th.long, device=device)
    market_active = th.tensor(batch.market_active, dtype=th.long, device=device)
    masks: dict[str, Any] = {}
    if training_action_mask:
        raw_masks = [
            row.get("action_masks", row.get("training_action_masks"))
            if isinstance(row, Mapping) else None
            for row in batch.transitions
        ]
        if any(mask is not None for mask in raw_masks):
            if not all(isinstance(mask, Mapping) for mask in raw_masks):
                raise ValueError("training action masks must be present for every transition")
            for output_name, objective_name in (
                ("worker_target_logits", "worker_target_mask"),
                ("worker_kind_logits", "worker_kind_mask"),
                ("market_item_logits", "market_item_mask"),
                ("market_quantity_logits", "market_quantity_mask"),
            ):
                values = [mask.get(objective_name, mask.get(objective_name.removesuffix("_mask"))) for mask in raw_masks]
                if any(value is not None for value in values):
                    if not all(value is not None for value in values):
                        raise ValueError(f"{objective_name} must be present for every transition")
                    masks[objective_name] = th.tensor(values, dtype=th.bool, device=device)
    return conditional_action_objectives(
        outputs,
        worker_active=worker_act,
        worker_target=worker_target,
        worker_kind=worker_kind,
        market_active=market_active,
        market_item=market_items,
        market_quantity=market_quantities,
        **masks,
    )


def _distribution_regularization(
    outputs: dict[str, Any], prior_outputs: dict[str, Any] | None,
    *, batch: RolloutBatch | None = None,
) -> tuple[Any, Any]:
    th = require_torch()
    zero = outputs["value"].sum() * 0.0
    if prior_outputs is None:
        return zero, zero
    worker_active = None
    market_active = None
    if batch is not None:
        worker_active = th.tensor(
            batch.worker.act, dtype=th.bool, device=outputs["value"].device,
        )
        market_active = th.tensor(
            batch.market_active, dtype=th.bool, device=outputs["value"].device,
        )

    def masked_mean(values: Any, mask: Any | None) -> Any:
        if mask is None:
            return values.mean()
        mask = mask.to(dtype=values.dtype, device=values.device)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denominator = mask.expand_as(values).sum().clamp_min(1.0)
        return (values * mask).sum() / denominator

    kl_terms = []
    ce_terms = []
    branch_masks = {
        "worker_act_logits": None,
        "worker_target_logits": worker_active,
        "worker_kind_logits": worker_active,
        "market_active_logits": None,
        "market_item_logits": market_active,
        "market_quantity_logits": market_active,
    }
    for name, mask in branch_masks.items():
        log_probs = outputs[name].log_softmax(dim=-1)
        with th.no_grad():
            prior_log_probs = prior_outputs[name].log_softmax(dim=-1)
            prior_probs = prior_log_probs.exp()
        kl_terms.append(masked_mean((prior_probs * (prior_log_probs - log_probs)).sum(dim=-1), mask))
        ce_terms.append(masked_mean(-(prior_probs * log_probs).sum(dim=-1), mask))
    return sum(kl_terms) / len(kl_terms), sum(ce_terms) / len(ce_terms)


def validate_prior_checkpoint_metadata(
    metadata: Any, *, model_width: int = DEFAULT_MODEL_WIDTH,
    model_depth: int = DEFAULT_MODEL_DEPTH,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("prior checkpoint metadata must be an object")
    expected = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"prior checkpoint {field} mismatch")
    expected_vocab = {key: list(value) for key, value in ACTION_VOCAB.items()}
    if metadata.get("action_vocab") != expected_vocab:
        raise ValueError("prior checkpoint action_vocab mismatch")
    for field, expected_value in (
        ("model_width", model_width), ("model_depth", model_depth),
    ):
        if field in metadata and metadata[field] != expected_value:
            raise ValueError(f"prior checkpoint {field} mismatch")


def _load_prior_network(
    prior_checkpoint: str | Path | None, *, device: Any,
    model_width: int = DEFAULT_MODEL_WIDTH, model_depth: int = DEFAULT_MODEL_DEPTH,
) -> Any:
    if prior_checkpoint is None:
        return None
    th = require_torch()
    checkpoint = th.load(
        prior_checkpoint, map_location="cpu", weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise ValueError("prior checkpoint metadata is required")
    validate_prior_checkpoint_metadata(
        checkpoint["metadata"], model_width=model_width, model_depth=model_depth,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError("prior checkpoint model_state_dict is required")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("prior checkpoint model_state_dict must be an object")
    prior = CompactPolicyNet(hidden_width=model_width, depth=model_depth).to(device)
    prior.load_state_dict(state)
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    return prior


def _ensure_finite_outputs(outputs: dict[str, Any]) -> None:
    th = require_torch()
    for name, tensor in outputs.items():
        if not th.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")


def _bootstrap_value_estimates(
    network: Any, transitions: Sequence[dict[str, Any]], *, config: PPOConfig,
) -> list[float]:
    """Evaluate next-state values only for nonterminal bootstrap truncations."""
    th = require_torch()
    rows = list(transitions)
    values = [0.0 for _row in rows]
    indices = [
        index for index, row in enumerate(rows)
        if _resolved_transition(row, config=config)
    ]
    if not indices:
        return values
    next_features = [
        extract_features(rows[index].get("next_observation", {})) for index in indices
    ]
    with th.no_grad():
        next_outputs = network(next_features)
        _ensure_finite_outputs(next_outputs)
        next_values = next_outputs["value"].detach().tolist()
    for index, value in zip(indices, next_values):
        values[index] = float(value)
    return values


def _parameter_norm(network: Any) -> float:
    th = require_torch()
    squared_norms = [
        parameter.detach().norm(2).pow(2)
        for parameter in network.parameters()
    ]
    if not squared_norms:
        return 0.0
    return float(th.stack(squared_norms).sum().sqrt())


def _gradient_norm(network: Any) -> float:
    th = require_torch()
    squared_norms = [
        parameter.grad.detach().norm(2).pow(2)
        for parameter in network.parameters()
        if parameter.grad is not None
    ]
    if not squared_norms:
        return 0.0
    return float(th.stack(squared_norms).sum().sqrt())


def _active_learning_rate(optimizer: Any) -> float:
    for parameter_group in optimizer.param_groups:
        if "lr" in parameter_group:
            return float(parameter_group["lr"])
    return 0.0


def _tensor_mean_std(values: Sequence[float], *, device: Any) -> tuple[float, float]:
    th = require_torch()
    tensor = th.tensor(values, dtype=th.float32, device=device)
    return float(tensor.mean()), float(tensor.std(unbiased=False))


def _explained_variance(*, values: Sequence[float], returns: Sequence[float], device: Any) -> float:
    th = require_torch()
    predictions = th.tensor(values, dtype=th.float32, device=device)
    targets = th.tensor(returns, dtype=th.float32, device=device)
    target_variance = targets.var(unbiased=False)
    if float(target_variance) == 0.0:
        return 0.0
    residual_variance = (targets - predictions).var(unbiased=False)
    return float(1.0 - residual_variance / target_variance)


def ppo_update(
    network: Any, optimizer: Any, transitions: Sequence[dict[str, Any]], *,
    config: PPOConfig, batch_size: int | None = None, seed: int = 0,
    prior_checkpoint: str | Path | None = None, device: Any = None,
    model_width: int = DEFAULT_MODEL_WIDTH, model_depth: int = DEFAULT_MODEL_DEPTH,
) -> dict[str, float | int | bool]:
    """Run clipped PPO updates over one rollout batch."""
    th = require_torch()
    rows = list(transitions)
    if not rows:
        raise ValueError("ppo_update requires at least one transition")
    features = [extract_features(row.get("observation", {})) for row in rows]
    bootstrap = build_rollout_batch(rows, config=config)
    with th.no_grad():
        old_outputs = network(features)
        _ensure_finite_outputs(old_outputs)
        device = old_outputs["value"].device if device is None else device
        old_log_probs, _old_entropy = _select_outputs(
            old_outputs, bootstrap, device=device,
            training_action_mask=config.training_action_mask,
        )
        bootstrap_values = _bootstrap_value_estimates(network, rows, config=config)
    rollout = build_rollout_batch(
        rows, config=config,
        value_estimates=old_outputs["value"].detach().tolist(),
        old_log_probs=old_log_probs.detach().tolist(),
        bootstrap_values=bootstrap_values,
    )
    prior = _load_prior_network(
        prior_checkpoint, device=device, model_width=model_width, model_depth=model_depth,
    )
    return_mean, return_std = _tensor_mean_std(
        rollout.returns, device=device,
    )
    advantage_mean, advantage_std = _tensor_mean_std(
        rollout.advantages, device=device,
    )
    metrics: dict[str, float | int | bool] = {
        "updates": 0,
        "early_stopped": False,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "kl_to_prior": 0.0,
        "prior_cross_entropy": 0.0,
        "loss": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": _explained_variance(
            values=rollout.values,
            returns=rollout.returns,
            device=device,
        ),
        "return_mean": return_mean,
        "return_std": return_std,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "gradient_norm": 0.0,
        "parameter_norm": _parameter_norm(network),
        "learning_rate": _active_learning_rate(optimizer),
        "shaping_count": rollout.shaping_count,
        "truncation_count": rollout.truncation_count,
    }
    size = max(1, int(batch_size or len(rows)))
    for epoch in range(config.ppo_epochs):
        for indices in epoch_minibatches(count=len(rows), batch_size=size, seed=seed, epoch=epoch):
            mini_features = [features[index] for index in indices]
            mini = RolloutBatch(
                transitions=[rollout.transitions[index] for index in indices],
                rewards=[rollout.rewards[index] for index in indices],
                dones=[rollout.dones[index] for index in indices],
                values=[rollout.values[index] for index in indices],
                old_log_probs=[rollout.old_log_probs[index] for index in indices],
                advantages=[rollout.advantages[index] for index in indices],
                returns=[rollout.returns[index] for index in indices],
                worker=WorkerLabels(
                    [rollout.worker.act[index] for index in indices],
                    [rollout.worker.target[index] for index in indices],
                    [rollout.worker.kind[index] for index in indices],
                ),
                market_items=[rollout.market_items[index] for index in indices],
                market_quantities=[rollout.market_quantities[index] for index in indices],
                market_active=[rollout.market_active[index] for index in indices],
                bootstrap_values=[rollout.bootstrap_values[index] for index in indices],
                bootstrap_truncated=[rollout.bootstrap_truncated[index] for index in indices],
                shaping_count=0,
                truncation_count=0,
            )
            outputs = network(mini_features)
            _ensure_finite_outputs(outputs)
            log_probs, entropy = _select_outputs(
                outputs, mini, device=device,
                training_action_mask=config.training_action_mask,
            )
            old_log = th.tensor(mini.old_log_probs, dtype=th.float32, device=device)
            advantages = th.tensor(mini.advantages, dtype=th.float32, device=device)
            returns = th.tensor(mini.returns, dtype=th.float32, device=device)
            old_values = th.tensor(mini.values, dtype=th.float32, device=device)
            log_ratio = th.clamp(log_probs - old_log, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
            ratio = log_ratio.exp()
            policy_loss = -th.minimum(
                ratio * advantages,
                th.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages,
            ).mean()
            clip_fraction = (
                ((ratio < 1.0 - config.clip_epsilon) | (ratio > 1.0 + config.clip_epsilon))
                .to(dtype=th.float32)
                .mean()
            )
            clipped_values = old_values + th.clamp(
                outputs["value"] - old_values, -config.clip_epsilon, config.clip_epsilon,
            )
            value_loss = th.maximum((outputs["value"] - returns) ** 2, (clipped_values - returns) ** 2).mean()
            prior_outputs = prior(mini_features) if prior is not None else None
            kl_to_prior, prior_ce = _distribution_regularization(
                outputs, prior_outputs, batch=mini,
            )
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + config.kl_coef * kl_to_prior
                + config.prior_ce_coef * prior_ce
            )
            approx = (old_log - log_probs).mean().detach()
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = _gradient_norm(network)
            optimizer.step()
            parameter_norm = _parameter_norm(network)
            learning_rate = _active_learning_rate(optimizer)
            with th.no_grad():
                post_outputs = network(mini_features)
                _ensure_finite_outputs(post_outputs)
                post_log_probs, _post_entropy = _select_outputs(
                    post_outputs, mini, device=device,
                    training_action_mask=config.training_action_mask,
                )
                post_log_ratio = th.clamp(post_log_probs - old_log, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
                post_step_kl = ((post_log_ratio.exp() - 1.0) - post_log_ratio).mean()
            metrics.update({
                "updates": int(metrics["updates"]) + 1,
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "approx_kl": float(post_step_kl.detach()),
                "kl_to_prior": float(kl_to_prior.detach()),
                "prior_cross_entropy": float(prior_ce.detach()),
                "loss": float(loss.detach()),
                "clip_fraction": float(clip_fraction.detach()),
                "gradient_norm": gradient_norm,
                "parameter_norm": parameter_norm,
                "learning_rate": learning_rate,
            })
            if float(post_step_kl) > config.target_kl:
                metrics["early_stopped"] = True
                return metrics
    return metrics


def _accepts_keyword_argument(callback: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return (
        parameter is not None
        and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    ) or any(item.kind == item.VAR_KEYWORD for item in parameters.values())


def _rollout_checkpoint_identity(path: str | Path | None) -> str | None:
    if path is None:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return None
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"checkpoint:{digest.hexdigest()}"


def run_ppo_training(
    *, network: Any, optimizer: Any, transitions: Sequence[dict[str, Any]],
    ppo_steps: int, config: PPOConfig, batch_size: int | None = None,
    seed: int = 0, prior_checkpoint: str | Path | None = None, device: Any = None,
    opponent_pool: Any | None = None, rollout_fn: Any | None = None,
    offline_ppo_fallback: bool = False, update_fn: Any | None = None,
    promotion_match_fn: Any | None = None,
    candidate_checkpoint: str | Path | None = None,
    best_checkpoint_path: str | Path | None = None,
    checkpoint_registry: dict[str, Any] | None = None,
    candidate_artifact: str | Path | None = None,
    candidate_identity: str | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    save_candidate_fn: Any | None = None,
    cleanup_candidate_fn: Any | None = None,
    start_step: int = 0,
    initial_ppo_updates: int = 0,
    initial_rollout_count: int = 0,
    initial_shaping_count: int = 0,
    initial_truncation_count: int = 0,
    initial_league_composition: Mapping[str, int] | None = None,
    initial_league_checkpoint_identities: Sequence[str] | None = None,
    progress_fn: Any | None = None,
    telemetry_callback: Any | None = None,
    model_width: int = DEFAULT_MODEL_WIDTH,
    model_depth: int = DEFAULT_MODEL_DEPTH,
) -> dict[str, Any]:
    """Run PPO with fresh scheduled league rollouts or explicit offline fallback."""
    _validate_experiment_id(experiment_id, source="PPO")
    steps = max(0, int(ppo_steps))
    if type(start_step) is not int or not 0 <= start_step <= steps:
        raise ValueError("start_step must be an integer between zero and ppo_steps")
    if type(initial_ppo_updates) is not int or initial_ppo_updates < 0:
        raise ValueError("initial_ppo_updates must be a nonnegative integer")
    if type(initial_rollout_count) is not int or initial_rollout_count < 0:
        raise ValueError("initial_rollout_count must be a nonnegative integer")
    if type(initial_shaping_count) is not int or initial_shaping_count < 0:
        raise ValueError("initial_shaping_count must be a nonnegative integer")
    if type(initial_truncation_count) is not int or initial_truncation_count < 0:
        raise ValueError("initial_truncation_count must be a nonnegative integer")
    if initial_league_composition is None:
        league_composition = dict(_PPO_LEAGUE_COMPOSITION_DEFAULT)
    else:
        if not isinstance(initial_league_composition, Mapping):
            raise ValueError("initial_league_composition must be a mapping")
        if any(
            type(name) is not str or not name
            or type(count) is not int or count < 0
            for name, count in initial_league_composition.items()
        ):
            raise ValueError(
                "initial_league_composition must map names to nonnegative integers"
            )
        league_composition = dict(_PPO_LEAGUE_COMPOSITION_DEFAULT)
        league_composition.update(initial_league_composition)
    if initial_league_checkpoint_identities is None:
        league_checkpoint_identities = []
    else:
        if (
            isinstance(initial_league_checkpoint_identities, (str, bytes))
            or not isinstance(initial_league_checkpoint_identities, Sequence)
            or any(
                type(identity) is not str or not identity
                for identity in initial_league_checkpoint_identities
            )
        ):
            raise ValueError(
                "initial_league_checkpoint_identities must be a sequence of "
                "nonempty strings"
            )
        league_checkpoint_identities = list(initial_league_checkpoint_identities)
    if steps == 0:
        return {
            "ppo_updates": initial_ppo_updates,
            "rollout_count": initial_rollout_count,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
            "completed_steps": 0,
            "shaping_count": initial_shaping_count,
            "truncation_count": initial_truncation_count,
            "league_composition": dict(league_composition),
            "league_checkpoint_identities": list(league_checkpoint_identities),
        }
    updater = update_fn or ppo_update
    if rollout_fn is None and not offline_ppo_fallback:
        raise ValueError("rollout_fn is required for PPO unless offline_ppo_fallback is explicitly selected")
    pool = opponent_pool if opponent_pool is not None else OpponentPool()
    schedule = pool.schedule(count=steps, seed=seed) if rollout_fn is not None else []
    total_updates = initial_ppo_updates
    rollout_count = initial_rollout_count
    shaping_count = initial_shaping_count
    truncation_count = initial_truncation_count
    last_metrics: dict[str, Any] | None = None
    offline_rows = list(transitions)
    if offline_ppo_fallback and not offline_rows:
        raise ValueError("offline_ppo_fallback requires collected transitions")
    ran_step = False
    for step in range(start_step, steps):
        ran_step = True
        match = None
        checkpoint_identity = None
        if rollout_fn is None:
            rollout = offline_rows[:config.rollout_steps]
        else:
            match = schedule[step]
            if isinstance(pool, OpponentPool) and match.opponent == "checkpoint" and (
                match.checkpoint is None or not Path(match.checkpoint).is_file()
            ):
                raise FileNotFoundError(
                    f"league checkpoint does not exist: {match.checkpoint}"
                )
            rollout_kwargs = {
                "step": step,
                "opponent": match.opponent,
                "seat": match.seat,
                "checkpoint": match.checkpoint,
                "rollout_steps": config.rollout_steps,
            }
            checkpoint_identity = getattr(match, "checkpoint_identity", None)
            if checkpoint_identity is None and match.checkpoint is not None:
                checkpoint_identity = _rollout_checkpoint_identity(match.checkpoint)
            if _accepts_keyword_argument(rollout_fn, "candidate_artifact"):
                rollout_kwargs["candidate_artifact"] = candidate_artifact
            if _accepts_keyword_argument(rollout_fn, "candidate_identity"):
                rollout_kwargs["candidate_identity"] = candidate_identity
            if _accepts_keyword_argument(rollout_fn, "seed"):
                rollout_kwargs["seed"] = int(seed) + step
            if _accepts_keyword_argument(rollout_fn, "round_index"):
                rollout_kwargs["round_index"] = step
            if _accepts_keyword_argument(rollout_fn, "opponent_identity"):
                rollout_kwargs["opponent_identity"] = match.opponent
            if _accepts_keyword_argument(rollout_fn, "checkpoint_identity"):
                rollout_kwargs["checkpoint_identity"] = checkpoint_identity
            if _accepts_keyword_argument(rollout_fn, "fallback_reason"):
                rollout_kwargs["fallback_reason"] = getattr(match, "fallback_reason", None)
            if _accepts_keyword_argument(rollout_fn, "mixed_opponent"):
                rollout_kwargs["mixed_opponent"] = getattr(match, "mixed_opponent", None)
            if _accepts_keyword_argument(rollout_fn, "network"):
                rollout_kwargs["network"] = network
            if _accepts_keyword_argument(rollout_fn, "experiment_id"):
                rollout_kwargs["experiment_id"] = experiment_id
            rollout = rollout_fn(**rollout_kwargs)
            rollout_count += 1
            league_composition[match.opponent] += 1
            if checkpoint_identity is not None:
                league_checkpoint_identities.append(checkpoint_identity)
        if not isinstance(rollout, Sequence) or isinstance(rollout, (str, bytes)):
            raise ValueError("rollout_fn must return a sequence of transitions")
        if not rollout:
            raise ValueError("PPO rollout produced no transitions")
        update_kwargs = {
            "network": network,
            "optimizer": optimizer,
            "transitions": list(rollout),
            "config": config,
            "batch_size": batch_size,
            "seed": int(seed) + step,
            "prior_checkpoint": prior_checkpoint,
        }
        if _accepts_keyword_argument(updater, "device"):
            update_kwargs["device"] = device
        if _accepts_keyword_argument(updater, "model_width"):
            update_kwargs["model_width"] = model_width
        if _accepts_keyword_argument(updater, "model_depth"):
            update_kwargs["model_depth"] = model_depth
        last_metrics = updater(**update_kwargs)
        total_updates += int(last_metrics.get("updates", 0))
        shaping_count += int(last_metrics.get("shaping_count", 0))
        truncation_count += int(last_metrics.get("truncation_count", 0))
        step_summary = {
            "ppo_updates": total_updates,
            "rollout_count": rollout_count,
            "early_stopped": bool(last_metrics.get("early_stopped")),
            "last_metrics": last_metrics,
            "shaping_count": shaping_count,
            "truncation_count": truncation_count,
            "promotion": None,
            "completed_steps": step + 1,
            "league_composition": dict(league_composition),
            "league_checkpoint_identities": list(league_checkpoint_identities),
        }
        if telemetry_callback is not None:
            telemetry_payload = {
                "step": step + 1,
                "ppo_updates": total_updates,
                "ppo_updates_step": int(last_metrics.get("updates", 0)),
                "rollout_count": rollout_count,
                "early_stopped": bool(last_metrics.get("early_stopped")),
                "policy_loss": last_metrics.get("policy_loss"),
                "value_loss": last_metrics.get("value_loss"),
                "entropy": last_metrics.get("entropy"),
                "approx_kl": last_metrics.get("approx_kl"),
            }
            if match is not None:
                telemetry_payload.update({
                    "experiment_id": experiment_id,
                    "league/opponent": match.opponent,
                    "league/seat": match.seat,
                    "league/seed": int(seed) + step,
                    "league/checkpoint_identity": checkpoint_identity,
                    "league/fallback_reason": getattr(match, "fallback_reason", None),
                    **{
                        f"league/{name}": count
                        for name, count in league_composition.items()
                    },
                })
            for name in (
                "clip_fraction", "explained_variance", "return_mean", "return_std",
                "advantage_mean", "advantage_std", "gradient_norm", "parameter_norm",
                "learning_rate", "shaping_count", "truncation_count",
            ):
                if name in last_metrics:
                    telemetry_payload[name] = last_metrics[name]
            telemetry_callback("ppo", telemetry_payload)
        if progress_fn is not None:
            progress_fn(completed_step=step + 1, metrics=step_summary)
        if last_metrics.get("early_stopped"):
            return step_summary
    promotion = None
    summary = {
        "ppo_updates": total_updates,
        "rollout_count": rollout_count,
        "early_stopped": False,
        "last_metrics": last_metrics,
        "completed_steps": steps,
        "shaping_count": shaping_count,
        "truncation_count": truncation_count,
        "league_composition": dict(league_composition),
        "league_checkpoint_identities": list(league_checkpoint_identities),
    }
    if promotion_match_fn is not None and ran_step:
        if candidate_checkpoint is None:
            raise ValueError("candidate_checkpoint is required when promotion_match_fn is provided")
        final_candidate = Path(candidate_checkpoint)
        candidate_for_match = _temporary_candidate_path(final_candidate)

        def save_temp_candidate(path: str | Path) -> str:
            return _save_candidate(save_candidate_fn, path, ppo_metrics=summary)

        registry_best_checkpoint = (
            _registry_best_checkpoint_path(final_candidate, checkpoint_registry)
            if checkpoint_registry is not None and best_checkpoint_path is None else None
        )
        promotion = maybe_promote_checkpoint(
            match_fn=promotion_match_fn,
            candidate_checkpoint=candidate_for_match,
            registry=checkpoint_registry,
            save_candidate_fn=save_temp_candidate,
            cleanup_candidate_fn=cleanup_candidate_fn,
            best_checkpoint_path=best_checkpoint_path,
            promoted_checkpoint_path=registry_best_checkpoint,
        )
        if promotion["promoted"]:
            try:
                _persist_best_checkpoint(candidate_for_match, final_candidate)
            finally:
                _cleanup_checkpoint(candidate_for_match)
    return {**summary, "promotion": promotion}


def _checkpoint_metadata(
    transition_count: int, config: PPOConfig | None = None, *, device: Any = "cpu",
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    behavior_clone_steps: int | None = None,
    behavior_clone_updates: int = 0,
    model_width: int = DEFAULT_MODEL_WIDTH,
    model_depth: int = DEFAULT_MODEL_DEPTH,
) -> dict[str, Any]:
    _validate_experiment_id(experiment_id, source="checkpoint metadata")
    _validate_feature_variant(feature_variant, source="checkpoint metadata")
    _validate_training_mode(training_mode, source="checkpoint metadata")
    validate_model_shape(model_width, model_depth, source="checkpoint metadata")
    if type(behavior_clone_updates) is not int or behavior_clone_updates < 0:
        raise ValueError("checkpoint metadata behavior_clone_updates must be a nonnegative integer")
    if behavior_clone_steps is None:
        effective_bc_steps = 0 if training_mode == "pure_ppo" else 1
    else:
        if type(behavior_clone_steps) is not int or behavior_clone_steps < 0:
            raise ValueError(
                "checkpoint metadata behavior_clone_steps must be a nonnegative integer"
            )
        effective_bc_steps = behavior_clone_steps
    model = CompactPolicyNet(hidden_width=model_width, depth=model_depth)
    return {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
        "engine_version": ENGINE_VERSION,
        "transition_count": int(transition_count),
        "ppo_config": asdict(config or PPOConfig()),
        "device": str(device),
        "experiment_id": experiment_id,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "behavior_clone_steps": effective_bc_steps,
        "behavior_clone_updates": behavior_clone_updates,
        "model_width": model_width,
        "model_depth": model_depth,
        "parameter_count": model_parameter_count(model),
    }


def checkpoint_metadata(
    transition_count: int, config: PPOConfig | None = None, *, device: Any = "cpu",
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    behavior_clone_steps: int | None = None,
    behavior_clone_updates: int = 0,
    model_width: int = DEFAULT_MODEL_WIDTH,
    model_depth: int = DEFAULT_MODEL_DEPTH,
) -> dict[str, Any]:
    if behavior_clone_steps is None:
        configured_bc_steps = 0 if training_mode == "pure_ppo" else 1
    else:
        configured_bc_steps = behavior_clone_steps
    return _checkpoint_metadata(
        transition_count, config, device=device,
        experiment_id=experiment_id,
        feature_variant=feature_variant,
        training_mode=training_mode,
        behavior_clone_steps=resolve_behavior_clone_steps(training_mode, configured_bc_steps),
        behavior_clone_updates=behavior_clone_updates,
        model_width=model_width,
        model_depth=model_depth,
    )


def _candidate_won(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        if "candidate_win" in result:
            if type(result["candidate_win"]) is bool:
                return result["candidate_win"]
            raise ValueError("promotion match result candidate_win must be boolean")
        if "winner" in result:
            winner = result["winner"]
            if not isinstance(winner, str):
                raise ValueError("promotion match result winner must be a string")
            normalized = winner.lower()
            if normalized in {"candidate", "learned", "policy", "agent"}:
                return True
            if normalized in {"opponent", "best", "baseline"}:
                return False
            raise ValueError("promotion match result winner must be candidate or opponent")
    raise ValueError("promotion match result must be boolean or contain candidate_win/winner")


def run_promotion_match(
    match_fn: Any, *, match_size: int = PROMOTION_MATCH_SIZE,
) -> dict[str, int | bool]:
    """Run the fixed promotion match and apply the strict >70% win gate."""
    if match_size != PROMOTION_MATCH_SIZE:
        raise ValueError(f"promotion match must run exactly {PROMOTION_MATCH_SIZE} games")
    wins = 0
    for index in range(PROMOTION_MATCH_SIZE):
        try:
            wins += int(_candidate_won(match_fn(index)))
        except ValueError as exc:
            raise ValueError(f"promotion match result {index} is malformed: {exc}") from exc
    return {
        "games": PROMOTION_MATCH_SIZE,
        "wins": wins,
        "promoted": should_promote(wins=wins, games=PROMOTION_MATCH_SIZE),
    }


def _cleanup_checkpoint(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _persist_best_checkpoint(candidate: str | Path, best: str | Path) -> str:
    candidate_path = Path(candidate)
    best_path = Path(best)
    if not candidate_path.exists():
        raise OSError(f"candidate checkpoint does not exist: {candidate_path}")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = best_path.with_name(f".{best_path.name}.publish.tmp")
    try:
        shutil.copyfile(candidate_path, temporary_path)
        temporary_path.replace(best_path)
    except Exception:
        _cleanup_checkpoint(temporary_path)
        raise
    return str(best_path)


def _temporary_candidate_path(final_path: str | Path) -> Path:
    path = Path(final_path)
    return path.with_name(f".{path.name}.promotion-candidate.tmp")


def _registry_best_checkpoint_path(final_path: str | Path, registry: dict[str, Any], best_key: str = "best") -> Path:
    current_best = registry.get(best_key)
    if current_best:
        return Path(current_best)
    path = Path(final_path)
    return path.with_name(f".{path.name}.registry-best.pt")


def _save_candidate(save_candidate_fn: Any, path: str | Path, *, ppo_metrics: dict[str, Any] | None) -> str:
    if save_candidate_fn is None:
        return str(path)
    try:
        signature = inspect.signature(save_candidate_fn)
    except (TypeError, ValueError):
        return str(save_candidate_fn(path))
    if "ppo_metrics" in signature.parameters:
        return str(save_candidate_fn(path, ppo_metrics=ppo_metrics))
    return str(save_candidate_fn(path))


def _promoted_checkpoint_path(
    saved_candidate: str,
    *,
    best_checkpoint_path: str | Path | None,
    promoted_checkpoint_path: str | Path | None,
) -> str:
    if best_checkpoint_path is not None:
        return _persist_best_checkpoint(saved_candidate, best_checkpoint_path)
    if promoted_checkpoint_path is not None:
        return _persist_best_checkpoint(saved_candidate, promoted_checkpoint_path)
    return saved_candidate


def maybe_promote_checkpoint(
    *, match_fn: Any, candidate_checkpoint: str | Path,
    registry: dict[str, Any] | None = None,
    save_candidate_fn: Any | None = None,
    cleanup_candidate_fn: Any | None = None,
    best_checkpoint_path: str | Path | None = None,
    promoted_checkpoint_path: str | Path | None = None,
    best_key: str = "best",
) -> dict[str, Any]:
    """Register a candidate, run the fixed promotion match, and update best only on promotion."""
    registry = registry if registry is not None else {best_key: None, "candidates": []}
    previous_best = str(best_checkpoint_path) if best_checkpoint_path is not None else registry.get(best_key)
    saved_candidate = _save_candidate(save_candidate_fn, candidate_checkpoint, ppo_metrics=None)
    if best_checkpoint_path is not None and not Path(saved_candidate).exists():
        raise OSError(f"candidate checkpoint does not exist: {saved_candidate}")
    entry = {"path": saved_candidate, "status": "candidate"}
    candidates = registry.setdefault("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("checkpoint registry candidates must be a list")
    candidates.append(entry)

    def candidate_match(index: int) -> Any:
        return match_fn(
            index,
            candidate_checkpoint=saved_candidate,
            best_checkpoint=previous_best,
            registry_entry=dict(entry),
        )

    try:
        result = run_promotion_match(candidate_match)
    except Exception:
        entry["status"] = "error"
        registry[best_key] = previous_best
        (cleanup_candidate_fn or _cleanup_checkpoint)(saved_candidate)
        raise
    if result["promoted"]:
        entry["status"] = "promoted"
        registry[best_key] = _promoted_checkpoint_path(
            saved_candidate,
            best_checkpoint_path=best_checkpoint_path,
            promoted_checkpoint_path=promoted_checkpoint_path,
        )
    else:
        entry["status"] = "rejected"
        registry[best_key] = previous_best
        (cleanup_candidate_fn or _cleanup_checkpoint)(saved_candidate)
    return {"candidate_checkpoint": saved_candidate, **result}


def _validate_resume_payload(
    payload: dict[str, Any], *, configuration: dict[str, Any],
    transition_count: int, allow_ppo_extension: bool = False,
) -> None:
    if type(payload) is not dict:
        raise CheckpointError("resume checkpoint payload must be an object")
    for field in ("configuration", "progress", "metrics", "metadata"):
        if field not in payload:
            raise CheckpointError(f"resume checkpoint is missing {field}")
    _validate_resume_configuration(
        payload["configuration"], configuration,
        allow_ppo_extension=allow_ppo_extension,
    )
    progress = payload["progress"]
    if type(progress) is not dict:
        raise CheckpointError("resume checkpoint progress must be an object")
    epoch = progress["epoch"]
    cursor = progress["cursor"]
    round_index = progress["round"]
    epochs = configuration["behavior_clone_steps"]
    saved_ppo_steps = payload["configuration"]["ppo_steps"]
    if round_index > saved_ppo_steps:
        raise ValueError(
            f"resume checkpoint progress.round {round_index} exceeds saved PPO target "
            f"{saved_ppo_steps}"
        )
    if epoch > epochs:
        raise ValueError(
            f"resume checkpoint epoch {epoch} exceeds requested steps {epochs}"
        )
    if epoch == epochs and cursor != 0:
        raise ValueError(
            f"resume checkpoint cursor {cursor} must be zero when epoch "
            f"{epoch} has completed requested steps"
        )
    if epoch < epochs:
        batch_count = math.ceil(transition_count / configuration["batch_size"])
        if cursor > batch_count:
            raise ValueError(
                f"resume checkpoint cursor {cursor} exceeds epoch batch count {batch_count}"
            )
        if round_index != 0:
            raise ValueError("resume checkpoint PPO round must be zero before behavior cloning completes")
    if round_index > configuration["ppo_steps"]:
        raise ValueError(
            f"resume checkpoint PPO round {round_index} exceeds requested ppo_steps "
            f"{configuration['ppo_steps']}"
        )
    metrics = payload["metrics"]
    if type(metrics) is not dict:
        raise CheckpointError("resume checkpoint metrics must be an object")
    expected_metrics = {"behavior_clone_updates", "ppo_updates", "ppo_metrics"}
    if set(metrics) != expected_metrics:
        missing = sorted(expected_metrics - set(metrics))
        unexpected = sorted(set(metrics) - expected_metrics)
        detail = missing or unexpected
        label = "missing required" if missing else "unexpected"
        raise ValueError(
            f"resume checkpoint metrics has {label} fields: {', '.join(detail)}"
        )
    for field in ("behavior_clone_updates", "ppo_updates"):
        if type(metrics[field]) is not int or metrics[field] < 0:
            raise ValueError(f"resume checkpoint metrics {field} must be a nonnegative integer")
    ppo_metrics = metrics["ppo_metrics"]
    if isinstance(ppo_metrics, dict) and type(ppo_metrics.get("completed_steps")) is int:
        completed_steps = ppo_metrics["completed_steps"]
        if completed_steps > saved_ppo_steps:
            raise ValueError(
                "resume checkpoint ppo_metrics completed_steps "
                f"{completed_steps} exceeds saved PPO target {saved_ppo_steps}"
            )
        if completed_steps > configuration["ppo_steps"]:
            raise ValueError(
                "resume checkpoint ppo_metrics completed_steps "
                f"{completed_steps} exceeds requested PPO target "
                f"{configuration['ppo_steps']}"
            )
    if round_index == 0:
        if ppo_metrics is not None:
            raise ValueError("resume checkpoint metrics ppo_metrics must be null before PPO progress")
        if metrics["ppo_updates"] != 0:
            raise ValueError("resume checkpoint metrics ppo_updates must be zero before PPO progress")
    else:
        if type(ppo_metrics) is not dict:
            raise ValueError("resume checkpoint metrics ppo_metrics is required after PPO progress")
        ppo_metrics.setdefault("league_composition", dict(_PPO_LEAGUE_COMPOSITION_DEFAULT))
        ppo_metrics.setdefault("league_checkpoint_identities", [])
        missing = sorted(_PPO_RESUME_METRIC_FIELDS - set(ppo_metrics))
        allowed_fields = _PPO_RESUME_METRIC_FIELDS | _PPO_OPTIONAL_RESUME_METRIC_FIELDS
        unexpected = sorted(set(ppo_metrics) - allowed_fields)
        if missing:
            raise ValueError(
                "resume checkpoint ppo_metrics is missing required fields: "
                + ", ".join(missing)
            )
        if unexpected:
            raise ValueError(
                "resume checkpoint ppo_metrics has unexpected fields: "
                + ", ".join(unexpected)
            )
        for field in ("ppo_updates", "rollout_count", "completed_steps"):
            if type(ppo_metrics[field]) is not int or ppo_metrics[field] < 0:
                raise ValueError(
                    f"resume checkpoint ppo_metrics {field} must be a nonnegative integer"
                )
        for field in ("shaping_count", "truncation_count"):
            if field in ppo_metrics and (
                type(ppo_metrics[field]) is not int or ppo_metrics[field] < 0
            ):
                raise ValueError(
                    f"resume checkpoint ppo_metrics {field} must be a nonnegative integer"
                )
        if type(ppo_metrics["early_stopped"]) is not bool:
            raise ValueError("resume checkpoint ppo_metrics early_stopped must be boolean")
        for field in ("last_metrics", "promotion"):
            if ppo_metrics[field] is not None and type(ppo_metrics[field]) is not dict:
                raise ValueError(
                    f"resume checkpoint ppo_metrics {field} must be an object or null"
                )
        league_composition = ppo_metrics["league_composition"]
        if type(league_composition) is not dict or any(
            type(name) is not str or not name
            or type(count) is not int or count < 0
            for name, count in league_composition.items()
        ):
            raise ValueError(
                "resume checkpoint ppo_metrics league_composition must map names to "
                "nonnegative integers"
            )
        checkpoint_identities = ppo_metrics["league_checkpoint_identities"]
        if type(checkpoint_identities) is not list or any(
            type(identity) is not str or not identity
            for identity in checkpoint_identities
        ):
            raise ValueError(
                "resume checkpoint ppo_metrics league_checkpoint_identities must be a "
                "list of nonempty strings"
            )
        if ppo_metrics["ppo_updates"] != metrics["ppo_updates"]:
            raise ValueError(
                "resume checkpoint ppo_metrics ppo_updates does not match metrics ppo_updates"
            )
        if ppo_metrics["completed_steps"] != round_index:
            raise ValueError(
                "resume checkpoint ppo_metrics completed_steps does not match PPO round"
            )
    metadata = payload["metadata"]
    if type(metadata) is not dict:
        raise CheckpointError("resume checkpoint metadata must be an object")
    if type(metadata.get("transition_count")) is not int:
        raise ValueError("resume checkpoint metadata transition_count must be an integer")
    if metadata["transition_count"] != transition_count:
        raise ValueError(
            "resume checkpoint transition_count does not match the requested input"
        )
    if type(metadata.get("device")) is not str or not metadata["device"]:
        raise ValueError("resume checkpoint metadata device must be a nonempty string")
    saved_configuration = payload["configuration"]
    for field in ("experiment_id", "feature_variant", "training_mode"):
        if field in metadata and metadata[field] != saved_configuration[field]:
            raise ValueError(
                f"resume checkpoint metadata {field} does not match saved configuration"
            )
    for field in ("model_width", "model_depth", "behavior_clone_steps"):
        if field in metadata:
            value = metadata[field]
            minimum = 0 if field == "behavior_clone_steps" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"resume checkpoint metadata {field} is invalid")
            if value != saved_configuration[field]:
                raise ValueError(
                    f"resume checkpoint metadata {field} does not match saved configuration"
                )
    if "parameter_count" in metadata:
        parameter_count = metadata["parameter_count"]
        expected_count = model_parameter_count(
            CompactPolicyNet(
                hidden_width=saved_configuration["model_width"],
                depth=saved_configuration["model_depth"],
            )
        )
        if type(parameter_count) is not int or parameter_count != expected_count:
            raise ValueError("resume checkpoint metadata parameter_count is incompatible")
    if "ppo_steps" in metadata:
        metadata_ppo_steps = metadata["ppo_steps"]
        if type(metadata_ppo_steps) is not int or metadata_ppo_steps < 0:
            raise ValueError(
                "resume checkpoint metadata ppo_steps must be a nonnegative integer"
            )
        saved_ppo_steps = saved_configuration["ppo_steps"]
        if metadata_ppo_steps != saved_ppo_steps:
            raise ValueError(
                "resume checkpoint metadata ppo_steps does not match saved configuration"
            )


def validate_training_checkpoint(
    payload: dict[str, Any], *, contract: TrainingContract,
    allow_ppo_extension: bool = False,
) -> None:
    """Validate a loaded checkpoint against the canonical training contract."""
    if not isinstance(contract, TrainingContract):
        raise TypeError("contract must be a TrainingContract")
    try:
        _validate_resume_payload(
            payload,
            configuration=contract.configuration,
            transition_count=contract.transition_count,
            allow_ppo_extension=allow_ppo_extension,
        )
    except CheckpointError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint is incompatible with training contract: {exc}") from exc


def make_fresh_rollout_fn(
    *, run_directory: str | Path, candidate_artifact: str | Path | None = None,
    candidate_artifact_callback: Any | None = None,
    seeds: Sequence[int], steps: int, workers: int = 1,
    game_timeout: float = 120.0, candidate_identity: str | None = None,
    no_progress_window: int = 0, resolved_margin: float = 0.0,
    league_probabilities: Mapping[str, object] | None = None,
    league_checkpoint_window: int | None = None,
    league_checkpoints: Sequence[str | Path] | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
) -> Any:
    """Build a collector-backed callback for one fresh PPO rollout per step.

    ``candidate_artifact`` preserves the original static-artifact API.  For
    fresh PPO rounds, ``candidate_artifact_callback`` may create or refresh a
    dependency-free artifact immediately before collection.  It receives
    keyword arguments for ``network``, ``output_path``, ``candidate_artifact``,
    ``step``, and ``round_index`` and may return the artifact path; returning
    ``None`` means that ``output_path`` was updated in place.
    """
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("seeds must not be empty")
    if candidate_artifact is None and candidate_artifact_callback is None:
        raise ValueError(
            "candidate_artifact or candidate_artifact_callback is required"
        )
    artifact = (
        Path(candidate_artifact).expanduser().resolve()
        if candidate_artifact is not None else None
    )
    if artifact is not None and candidate_artifact_callback is None and not artifact.is_file():
        raise FileNotFoundError(f"candidate artifact does not exist: {artifact}")
    if candidate_artifact_callback is not None and not callable(candidate_artifact_callback):
        raise TypeError("candidate_artifact_callback must be callable")
    _validate_experiment_id(experiment_id, source="rollout")
    _validate_feature_variant(feature_variant, source="rollout")
    _validate_training_mode(training_mode, source="rollout")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be at least 2 to produce a transition")
    if type(no_progress_window) is not int or no_progress_window < 0:
        raise ValueError("no_progress_window must be a nonnegative integer")
    if (
        isinstance(resolved_margin, bool)
        or not isinstance(resolved_margin, (int, float))
        or not math.isfinite(float(resolved_margin))
        or resolved_margin < 0
    ):
        raise ValueError("resolved_margin must be a nonnegative finite number")
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)
    configured_league_probabilities = (
        dict(league_probabilities) if league_probabilities is not None else None
    )
    configured_league_checkpoints = (
        [str(path) for path in league_checkpoints]
        if league_checkpoints is not None else None
    )

    def fresh_rollout(
        *, step: int, opponent: str, seat: int, checkpoint: str | None,
        rollout_steps: int, candidate_artifact: str | Path | None = None,
        opponent_identity: str | None = None,
        checkpoint_identity: str | None = None,
        fallback_reason: str | None = None,
        mixed_opponent: str | None = None,
        seed: int | None = None, round_index: int | None = None,
        network: Any = None,
    ) -> list[dict[str, Any]]:
        from scripts.collect_trajectories import collect

        selected_step = int(step if round_index is None else round_index)
        selected_seed = int(seed) if seed is not None else normalized_seeds[selected_step % len(normalized_seeds)]
        selected_artifact = (
            Path(candidate_artifact).expanduser().resolve()
            if candidate_artifact is not None else artifact
        )
        if candidate_artifact_callback is not None:
            target_artifact = selected_artifact or run_path / f"candidate-step-{selected_step:05d}.json"
            callback_kwargs = {
                "network": network,
                "output_path": target_artifact,
                "candidate_artifact": target_artifact,
                "step": int(step),
                "round_index": selected_step,
            }
            result = candidate_artifact_callback(**{
                key: value for key, value in callback_kwargs.items()
                if _accepts_keyword_argument(candidate_artifact_callback, key)
            })
            selected_artifact = Path(result or target_artifact).expanduser().resolve()
        if selected_artifact is None or not selected_artifact.is_file():
            raise FileNotFoundError(
                f"candidate artifact does not exist: {selected_artifact}"
            )
        digest = hashlib.sha256(selected_artifact.read_bytes()).hexdigest()
        identity = candidate_identity or candidate_identity_outer or f"artifact:{digest}"
        output = run_path / f"ppo-step-{selected_step:05d}.jsonl"
        collector_opponent = opponent
        opponent_artifact = None
        checkpoint_provenance = checkpoint_identity or checkpoint
        if opponent == "checkpoint":
            if checkpoint is None:
                raise ValueError("checkpoint opponent requires a checkpoint path")
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"league checkpoint does not exist: {checkpoint_path}")
            from scripts.export_policy import export_checkpoint

            opponent_artifact = run_path / f"opponent-{selected_step:05d}.json"
            export_checkpoint(checkpoint_path, opponent_artifact)
            collector_opponent = "pass"
        elif opponent == "mixed":
            collector_opponent = mixed_opponent or "current"
            if collector_opponent not in {"current", "random", "starter"}:
                raise ValueError(
                    "mixed_opponent must be one of current, random, starter"
                )
        collect(
            seeds=[selected_seed], opponents=[collector_opponent], seats=[int(seat)],
            steps=max(int(rollout_steps) + 1, steps), output=output,
            workers=workers, game_timeout=game_timeout,
            experiment_id=experiment_id,
            feature_variant=feature_variant,
            training_mode=training_mode,
            candidate_artifact=selected_artifact, candidate_identity=identity,
            opponent_artifact=opponent_artifact,
            opponent_checkpoint_identity=checkpoint_provenance,
            checkpoint_identity=checkpoint_provenance,
            fallback_reason=fallback_reason,
            opponent_identity=opponent_identity or opponent,
            league_round=selected_step,
            league_seed=selected_seed,
            league_composition={
                "current": int(opponent == "current"),
                "mixed": int(opponent == "mixed"),
                "random": int(opponent == "random"),
                "starter": int(opponent == "starter"),
                "checkpoint": int(opponent == "checkpoint"),
            },
            league_probabilities=configured_league_probabilities,
            league_checkpoint_window=league_checkpoint_window,
            league_checkpoints=configured_league_checkpoints,
            source_policy_identity=identity,
            no_progress_window=no_progress_window,
            resolved_margin=resolved_margin,
        )
        return [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    candidate_identity_outer = candidate_identity
    return fresh_rollout


def train_behavior_clone(
    *, input_path: str | Path, output_path: str | Path, steps: int,
    batch_size: int, seed: int = 0, ppo_steps: int = 0,
    device: str = "auto", checkpoint_interval: int = 100,
    ppo_config: PPOConfig | None = None,
    resume_checkpoint: str | Path | None = None,
    allow_ppo_extension: bool = False,
    prior_checkpoint: str | Path | None = None, rollout_fn: Any | None = None,
    opponent_pool: Any | None = None, offline_ppo_fallback: bool = False,
    promotion_match_fn: Any | None = None,
    best_checkpoint_path: str | Path | None = None,
    checkpoint_registry: dict[str, Any] | None = None,
    candidate_artifact: str | Path | None = None,
    telemetry_callback: Any | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    behavior_clone_steps: int | None = None,
    model_width: int = DEFAULT_MODEL_WIDTH,
    model_depth: int = DEFAULT_MODEL_DEPTH,
) -> dict[str, Any]:
    """Run complete behavior-cloning epochs, optional PPO, and checkpoint."""
    if type(allow_ppo_extension) is not bool:
        raise ValueError("allow_ppo_extension must be boolean")
    th = require_torch()
    resolved_device = resolve_device(device)
    batch_size = max(1, int(batch_size))
    normalized_steps = max(1, int(steps))
    if behavior_clone_steps is not None and (
        type(behavior_clone_steps) is not int or behavior_clone_steps < 0
    ):
        raise ValueError("behavior_clone_steps must be a nonnegative integer")
    configured_behavior_clone_steps = (
        normalized_steps if behavior_clone_steps is None else behavior_clone_steps
    )
    seed = int(seed)
    ppo_steps = int(ppo_steps)
    ppo_config = ppo_config or PPOConfig()
    if not isinstance(ppo_config, PPOConfig):
        raise ValueError("ppo_config must be a PPOConfig or None")
    offline_ppo_fallback = bool(offline_ppo_fallback)
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    contract = build_training_contract(
        input_path=input_path,
        steps=normalized_steps,
        batch_size=batch_size,
        seed=seed,
        ppo_steps=ppo_steps,
        device=device,
        checkpoint_interval=checkpoint_interval,
        prior_checkpoint=prior_checkpoint,
        offline_ppo_fallback=offline_ppo_fallback,
        resolved_device=resolved_device,
        ppo_config=ppo_config,
        experiment_id=experiment_id,
        feature_variant=feature_variant,
        training_mode=training_mode,
        behavior_clone_steps=configured_behavior_clone_steps,
        model_width=model_width,
        model_depth=model_depth,
    )
    configuration = contract.configuration
    epochs = configuration["behavior_clone_steps"]
    transitions = _read_transitions(input_path)
    features = [extract_features(transition.get("observation", {})) for transition in transitions]
    actions = [transition.get("action", {}) if isinstance(transition.get("action"), dict) else {} for transition in transitions]
    start_epoch = 0
    start_cursor = 0
    round_index = 0
    bc_updates = 0
    initial_ppo_updates = 0
    initial_rollout_count = 0
    initial_shaping_count = 0
    initial_truncation_count = 0
    initial_league_composition = dict(_PPO_LEAGUE_COMPOSITION_DEFAULT)
    initial_league_checkpoint_identities: list[str] = []
    previous_ppo_metrics: dict[str, Any] | None = None
    resumed = None
    if resume_checkpoint is not None:
        resumed = read_checkpoint(resume_checkpoint, map_location="cpu")
        validate_training_checkpoint(
            resumed, contract=contract, allow_ppo_extension=allow_ppo_extension,
        )
        start_epoch = resumed["progress"]["epoch"]
        start_cursor = resumed["progress"]["cursor"]
        round_index = resumed["progress"]["round"]
        bc_updates = resumed["metrics"]["behavior_clone_updates"]
        initial_ppo_updates = resumed["metrics"]["ppo_updates"]
        previous_ppo_metrics = resumed["metrics"]["ppo_metrics"]
        if previous_ppo_metrics is not None:
            initial_rollout_count = previous_ppo_metrics.get("rollout_count", 0)
            initial_shaping_count = previous_ppo_metrics.get("shaping_count", 0)
            initial_truncation_count = previous_ppo_metrics.get("truncation_count", 0)
            initial_league_composition = dict(
                previous_ppo_metrics.get(
                    "league_composition", _PPO_LEAGUE_COMPOSITION_DEFAULT,
                )
            )
            initial_league_checkpoint_identities = list(
                previous_ppo_metrics.get("league_checkpoint_identities", [])
            )
    rng_before_resume = capture_rng_state() if resumed is not None else None
    try:
        set_training_seed(seed)
        network = CompactPolicyNet(
            hidden_width=model_width, depth=model_depth,
        ).to(resolved_device)
        optimizer = th.optim.AdamW(network.parameters(), lr=1e-3)
        if resumed is not None:
            restore_checkpoint(
                resumed, model=network, optimizer=optimizer, restore_rng=True,
            )
    except Exception:
        if rng_before_resume is not None:
            restore_rng_state(rng_before_resume)
        raise
    metadata = _checkpoint_metadata(
        len(transitions), ppo_config, device=resolved_device,
        experiment_id=experiment_id,
        feature_variant=feature_variant,
        training_mode=training_mode,
        behavior_clone_steps=epochs,
        behavior_clone_updates=bc_updates,
        model_width=model_width,
        model_depth=model_depth,
    )
    metadata["ppo_steps"] = int(ppo_steps)
    metadata["behavior_clone_epochs"] = epochs
    destination = Path(output_path)
    last_bc_loss: float | None = None
    last_bc_entropy: float | None = None
    last_bc_gradient_norm: float | None = None
    last_bc_parameter_norm: float | None = None
    last_bc_learning_rate: float | None = None
    last_bc_telemetry_update = 0

    def checkpoint_metrics(ppo_result: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "behavior_clone_updates": bc_updates,
            "ppo_updates": (
                initial_ppo_updates
                if ppo_result is None else int(ppo_result["ppo_updates"])
            ),
            "ppo_metrics": ppo_result,
        }

    def save_current_checkpoint(
        path: str | Path, *, ppo_result: dict[str, Any] | None,
        completed_epoch: int, completed_round: int, cursor: int,
    ) -> str:
        current_metrics = checkpoint_metrics(ppo_result)
        candidate_metadata = dict(metadata)
        candidate_metadata.update(current_metrics)
        return save_checkpoint(
            path,
            model=network,
            optimizer=optimizer,
            configuration=configuration,
            epoch=completed_epoch,
            round_index=completed_round,
            cursor=cursor,
            metrics=current_metrics,
            metadata=candidate_metadata,
        )

    for epoch in range(start_epoch, epochs):
        minibatches = epoch_minibatches(
            count=len(features), batch_size=batch_size, seed=int(seed), epoch=epoch,
        )
        cursor = start_cursor if epoch == start_epoch else 0
        if cursor > len(minibatches):
            raise ValueError(
                f"resume checkpoint cursor {cursor} exceeds epoch batch count {len(minibatches)}"
            )
        for batch_index, permutation in enumerate(minibatches):
            if batch_index < cursor:
                continue
            batch_features = [features[index] for index in permutation]
            batch_actions = [actions[index] for index in permutation]
            batch_observations = [
                transitions[index].get("observation", {})
                if isinstance(transitions[index].get("observation", {}), dict) else {}
                for index in permutation
            ]
            outputs = network(batch_features)
            labels = [
                worker_labels(action, observation)
                for action, observation in zip(batch_actions, batch_observations)
            ]
            act_target = th.tensor([label.act for label in labels], dtype=th.long, device=resolved_device)
            target_target = th.tensor([label.target for label in labels], dtype=th.long, device=resolved_device)
            kind_target = th.tensor([label.kind for label in labels], dtype=th.long, device=resolved_device)
            market_items, market_quantities = zip(*(_market_labels(action) for action in batch_actions))
            market_active = th.tensor(
                [_market_active_label(action) for action in batch_actions],
                dtype=th.long,
                device=resolved_device,
            )
            item_target = th.tensor(market_items, dtype=th.long, device=resolved_device)
            quantity_target = th.tensor(market_quantities, dtype=th.long, device=resolved_device)
            bc_batch = RolloutBatch(
                transitions=[transitions[index] for index in permutation],
                rewards=[0.0 for _ in permutation],
                dones=[False for _ in permutation],
                values=[0.0 for _ in permutation],
                old_log_probs=[0.0 for _ in permutation],
                advantages=[0.0 for _ in permutation],
                returns=[0.0 for _ in permutation],
                worker=WorkerLabels(
                    [label.act for label in labels],
                    [label.target for label in labels],
                    [label.kind for label in labels],
                ),
                market_items=list(market_items),
                market_quantities=list(market_quantities),
                market_active=market_active.detach().tolist(),
            )
            log_probs, entropy = _select_outputs(
                outputs, bc_batch, device=resolved_device,
                training_action_mask=ppo_config.training_action_mask,
            )
            loss = -log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            last_bc_gradient_norm = _gradient_norm(network)
            optimizer.step()
            last_bc_parameter_norm = _parameter_norm(network)
            last_bc_learning_rate = _active_learning_rate(optimizer)
            bc_updates += 1
            last_bc_loss = float(loss.detach())
            last_bc_entropy = float(entropy.detach())
            if bc_updates % checkpoint_interval == 0:
                completed_cursor = batch_index + 1
                completed_epoch = epoch
                if completed_cursor == len(minibatches):
                    completed_epoch += 1
                    completed_cursor = 0
                save_current_checkpoint(
                    destination,
                    ppo_result=None,
                    completed_epoch=completed_epoch,
                    completed_round=0,
                    cursor=completed_cursor,
                )
                if telemetry_callback is not None:
                    telemetry_callback("behavior_clone", {
                        "step": bc_updates,
                        "update_count": bc_updates,
                        "epoch": completed_epoch,
                        "loss": last_bc_loss,
                        "learning_rate": last_bc_learning_rate,
                        "gradient_norm": last_bc_gradient_norm,
                        "parameter_norm": last_bc_parameter_norm,
                        "entropy": last_bc_entropy,
                    })
                    last_bc_telemetry_update = bc_updates
    metadata["behavior_clone_updates"] = bc_updates
    metadata["ppo_updates"] = 0
    metadata["ppo_metrics"] = None
    save_current_checkpoint(
        destination,
        ppo_result=previous_ppo_metrics,
        completed_epoch=epochs,
        completed_round=round_index,
        cursor=0,
    )
    if (
        telemetry_callback is not None
        and last_bc_loss is not None
        and last_bc_telemetry_update != bc_updates
    ):
        telemetry_callback("behavior_clone", {
            "step": bc_updates,
            "update_count": bc_updates,
            "epoch": epochs,
            "loss": last_bc_loss,
            "learning_rate": last_bc_learning_rate,
            "gradient_norm": last_bc_gradient_norm,
            "parameter_norm": last_bc_parameter_norm,
            "entropy": last_bc_entropy,
        })
        last_bc_telemetry_update = bc_updates

    def save_current_candidate(path: str | Path, *, ppo_metrics: dict[str, Any] | None = None) -> str:
        completed_round = (
            round_index
            if ppo_metrics is None
            else int(ppo_metrics.get("completed_steps", ppo_steps))
        )
        return save_current_checkpoint(
            path,
            ppo_result=ppo_metrics,
            completed_epoch=epochs,
            completed_round=completed_round,
            cursor=0,
        )

    def save_ppo_progress(*, completed_step: int, metrics: dict[str, Any]) -> None:
        save_current_checkpoint(
            destination,
            ppo_result=metrics,
            completed_epoch=epochs,
            completed_round=completed_step,
            cursor=0,
        )

    ppo_metrics = previous_ppo_metrics
    if ppo_steps and round_index < int(ppo_steps):
        # Behavior cloning benefits from a larger step size, but carrying that
        # rate into on-policy updates can move the policy far outside the
        # trust region in a single minibatch and trip target_kl immediately.
        # Adam's accumulated behavior-cloning moments can have the same effect
        # even after changing the scalar learning rate, so the first PPO phase
        # always starts with fresh optimizer state.  Later PPO rounds keep the
        # state so checkpoint resume remains continuous.
        if round_index == 0 and initial_ppo_updates == 0 and initial_rollout_count == 0:
            optimizer.state.clear()
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = PPO_LEARNING_RATE
        config = ppo_config
        ppo_metrics = run_ppo_training(
            network=network,
            optimizer=optimizer,
            transitions=transitions,
            ppo_steps=int(ppo_steps),
            config=config,
            batch_size=batch_size,
            seed=int(seed),
            prior_checkpoint=prior_checkpoint,
            device=resolved_device,
            opponent_pool=opponent_pool,
            rollout_fn=rollout_fn,
            offline_ppo_fallback=offline_ppo_fallback,
            promotion_match_fn=promotion_match_fn,
            candidate_checkpoint=output_path,
            best_checkpoint_path=best_checkpoint_path,
            checkpoint_registry=checkpoint_registry,
            candidate_artifact=candidate_artifact,
            candidate_identity=(str(candidate_artifact) if candidate_artifact is not None else None),
            experiment_id=experiment_id,
            save_candidate_fn=save_current_candidate,
            start_step=round_index,
            initial_ppo_updates=initial_ppo_updates,
            initial_rollout_count=initial_rollout_count,
            initial_shaping_count=initial_shaping_count,
            initial_truncation_count=initial_truncation_count,
            initial_league_composition=initial_league_composition,
            initial_league_checkpoint_identities=initial_league_checkpoint_identities,
            progress_fn=save_ppo_progress,
            telemetry_callback=telemetry_callback,
            model_width=model_width,
            model_depth=model_depth,
        )
    metadata["ppo_updates"] = initial_ppo_updates if ppo_metrics is None else ppo_metrics["ppo_updates"]
    metadata["ppo_metrics"] = ppo_metrics
    completed_round = (
        round_index
        if ppo_metrics is None
        else int(ppo_metrics.get("completed_steps", ppo_steps))
    )
    save_current_checkpoint(
        destination,
        ppo_result=ppo_metrics,
        completed_epoch=epochs,
        completed_round=completed_round,
        cursor=0,
    )
    return metadata


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
    return number


def _cli_training_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.ppo_steps > 0 and not args.offline_ppo_fallback:
        raise ValueError(
            "--ppo-steps requires --offline-ppo-fallback in the standalone CLI; "
            "fresh league rollouts are available through the rollout_fn API"
        )
    probabilities = {
        name: getattr(args, f"league_{name}_probability", None)
        for name in ("current", "mixed", "random", "starter", "checkpoint")
    }
    if any(value is None for value in probabilities.values()):
        probabilities = dict(DEFAULT_OPPONENT_PROBABILITIES)
        for name in probabilities:
            value = getattr(args, f"league_{name}_probability", None)
            if value is not None:
                probabilities[name] = value
    return {
        "offline_ppo_fallback": bool(args.offline_ppo_fallback),
        "allow_ppo_extension": bool(args.allow_ppo_extension),
        "opponent_pool": OpponentPool(
            previous_checkpoints=args.league_checkpoints,
            checkpoint_window=args.league_checkpoint_window,
            probabilities=probabilities,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="input_path")
    parser.add_argument("--output", type=Path, required=True, dest="output_path")
    parser.add_argument("--steps", type=_positive_int, default=1)
    parser.add_argument("--behavior-clone-steps", type=_nonnegative_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--checkpoint-interval", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None, dest="resume_checkpoint")
    parser.add_argument(
        "--allow-ppo-extension",
        action="store_true",
        help="allow a resumed checkpoint to increase its PPO training target",
    )
    parser.add_argument("--ppo-steps", type=_nonnegative_int, default=0)
    parser.add_argument("--prior-checkpoint", type=Path, default=None)
    parser.add_argument("--best-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--training-mode", choices=TRAINING_MODES, default="behavior_clone_then_ppo",
    )
    parser.add_argument("--model-width", type=_positive_int, default=DEFAULT_MODEL_WIDTH)
    parser.add_argument("--model-depth", type=_positive_int, default=DEFAULT_MODEL_DEPTH)
    parser.add_argument("--league-checkpoints", type=Path, action="append", default=[])
    parser.add_argument("--league-checkpoint-window", type=_nonnegative_int, default=5)
    parser.add_argument("--league-current-probability", type=_nonnegative_float, default=None)
    parser.add_argument("--league-mixed-probability", type=_nonnegative_float, default=None)
    parser.add_argument("--league-random-probability", type=_nonnegative_float, default=None)
    parser.add_argument("--league-starter-probability", type=_nonnegative_float, default=None)
    parser.add_argument("--league-checkpoint-probability", type=_nonnegative_float, default=None)
    parser.add_argument(
        "--offline-ppo-fallback",
        action="store_true",
        help="reuse collected input transitions for PPO instead of requiring a fresh rollout callback",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        options = _cli_training_options(args)
        metadata = train_behavior_clone(
            input_path=args.input_path,
            output_path=args.output_path,
            steps=args.steps,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
            device=args.device,
            resume_checkpoint=args.resume_checkpoint,
            allow_ppo_extension=options["allow_ppo_extension"],
            ppo_steps=args.ppo_steps,
            prior_checkpoint=args.prior_checkpoint,
            offline_ppo_fallback=options["offline_ppo_fallback"],
            best_checkpoint_path=args.best_checkpoint,
            opponent_pool=options["opponent_pool"],
            experiment_id=args.experiment_id,
            training_mode=args.training_mode,
            behavior_clone_steps=args.behavior_clone_steps,
            model_width=args.model_width,
            model_depth=args.model_depth,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
