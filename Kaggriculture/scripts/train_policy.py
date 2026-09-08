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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.checkpoints import (
    capture_rng_state,
    read_checkpoint,
    restore_rng_state,
    restore_checkpoint,
    save_checkpoint,
)
from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION, extract_features
from kagriculture_agent.model import (
    ACTION_VOCAB,
    MODEL_VERSION,
    CompactPolicyNet,
    require_torch,
    resolve_device,
    set_training_seed,
)

PROMOTION_MATCH_SIZE = 100
LOG_RATIO_CLAMP = 20.0
PPO_LEARNING_RATE = 1e-4
_DIRECTION_DELTAS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}
_CURRENT_TILE_KINDS = {
    "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "DROP", "SELL", "DIG", "WEED",
}


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

    def __post_init__(self) -> None:
        for name in (
            "gamma", "gae_lambda", "clip_epsilon", "value_coef", "entropy_coef",
            "target_kl", "rollout_steps", "kl_coef", "prior_ce_coef", "ppo_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("rollout_steps", "ppo_epochs"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]; gamma=1.0 is for explicit experiments only")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        for name in ("value_coef", "entropy_coef", "kl_coef", "prior_ce_coef"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive")
        if self.rollout_steps < 1 or self.ppo_epochs < 1:
            raise ValueError("rollout_steps and ppo_epochs must be positive")


_RESUME_CONFIGURATION_FIELDS = (
    "input_trajectory",
    "steps",
    "batch_size",
    "seed",
    "ppo_steps",
    "prior_checkpoint",
    "offline_ppo_fallback",
    "ppo_config",
)
_RUNTIME_CONFIGURATION_FIELDS = {"device", "checkpoint_interval"}
_PPO_FLOAT_FIELDS = {
    "gamma", "gae_lambda", "clip_epsilon", "value_coef", "entropy_coef",
    "target_kl", "kl_coef", "prior_ce_coef",
}
_PPO_INTEGER_FIELDS = {"rollout_steps", "ppo_epochs"}
_PPO_RESUME_METRIC_FIELDS = {
    "ppo_updates", "rollout_count", "early_stopped", "last_metrics",
    "promotion", "completed_steps",
}


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


def _validate_ppo_configuration(value: Any, *, source: str) -> None:
    expected_fields = _PPO_FLOAT_FIELDS | _PPO_INTEGER_FIELDS
    if type(value) is not dict:
        raise ValueError(f"{source} ppo_config must be an object")
    missing = sorted(expected_fields - set(value))
    unexpected = sorted(set(value) - expected_fields)
    if missing:
        raise ValueError(f"{source} ppo_config is missing required fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{source} ppo_config has unexpected fields: {', '.join(unexpected)}")
    for field in sorted(_PPO_FLOAT_FIELDS):
        if type(value[field]) is not float:
            raise ValueError(f"{source} ppo_config {field} must be a float")
    for field in sorted(_PPO_INTEGER_FIELDS):
        if type(value[field]) is not int:
            raise ValueError(f"{source} ppo_config {field} must be an integer")
    try:
        PPOConfig(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} ppo_config is invalid: {exc}") from exc


def _validate_configuration_shape(configuration: Any, *, source: str) -> None:
    expected_fields = set(_RESUME_CONFIGURATION_FIELDS) | _RUNTIME_CONFIGURATION_FIELDS
    if type(configuration) is not dict:
        raise ValueError(f"{source} configuration must be an object")
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
        ("checkpoint_interval", 1),
    ):
        value = configuration[field]
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"{source} configuration {field} must be an integer at least {minimum}"
            )
    if type(configuration["seed"]) is not int:
        raise ValueError(f"{source} configuration seed must be an integer")
    if type(configuration["device"]) is not str or not configuration["device"]:
        raise ValueError(f"{source} configuration device must be a nonempty string")
    if type(configuration["offline_ppo_fallback"]) is not bool:
        raise ValueError(f"{source} configuration offline_ppo_fallback must be boolean")
    _validate_prior_checkpoint_identity(configuration["prior_checkpoint"], source=source)
    _validate_ppo_configuration(configuration["ppo_config"], source=source)


def _validate_resume_configuration(
    saved: dict[str, Any], requested: dict[str, Any],
) -> None:
    _validate_configuration_shape(saved, source="saved")
    _validate_configuration_shape(requested, source="requested")
    for field in _RESUME_CONFIGURATION_FIELDS:
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
class OpponentMatch:
    opponent: str
    seat: int
    checkpoint: str | None = None
    mixed_opponent: str | None = None


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

    def __init__(self, previous_checkpoints: Sequence[str | Path] = ()) -> None:
        self.checkpoint_candidates = tuple(str(path) for path in previous_checkpoints[-5:])
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-12:
            raise ValueError("opponent pool probabilities must sum to one")

    def sample(self, index: int) -> OpponentMatch:
        rng = random.Random(int(index))
        draw = rng.random()
        cumulative = 0.0
        selected = "checkpoint"
        for opponent, probability in self.probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                selected = opponent
                break
        checkpoint = None
        if selected == "checkpoint":
            if self.checkpoint_candidates:
                checkpoint = self.checkpoint_candidates[rng.randrange(len(self.checkpoint_candidates))]
            else:
                selected = "current"
        mixed_opponent = None
        if selected == "mixed":
            mixed_opponent = self.mixed_opponents[rng.randrange(len(self.mixed_opponents))]
        return OpponentMatch(
            opponent=selected, seat=int(index) % 2, checkpoint=checkpoint,
            mixed_opponent=mixed_opponent,
        )

    def schedule(self, *, count: int, seed: int = 0) -> list[OpponentMatch]:
        """Return a deterministic stratified schedule with alternating seats."""
        if count < 1:
            raise ValueError("count must be positive")
        counts = {
            opponent: int(count * probability)
            for opponent, probability in self.probabilities.items()
        }
        missing = count - sum(counts.values())
        remainders = sorted(
            (
                (count * probability - counts[opponent], opponent)
                for opponent, probability in self.probabilities.items()
            ),
            reverse=True,
        )
        for _fraction, opponent in remainders[:missing]:
            counts[opponent] += 1
        matches: list[OpponentMatch] = []
        for opponent in self.probabilities:
            if opponent != "checkpoint":
                for index in range(counts[opponent]):
                    mixed_opponent = None
                    if opponent == "mixed":
                        mixed_rng = random.Random(int(seed) + index)
                        mixed_opponent = self.mixed_opponents[
                            mixed_rng.randrange(len(self.mixed_opponents))
                        ]
                    matches.append(
                        OpponentMatch(
                            opponent=opponent, seat=0,
                            mixed_opponent=mixed_opponent,
                        )
                    )
        if self.checkpoint_candidates:
            matches.extend(
                OpponentMatch(
                    opponent="checkpoint", seat=0,
                    checkpoint=self.checkpoint_candidates[index % len(self.checkpoint_candidates)],
                )
                for index in range(counts["checkpoint"])
            )
        else:
            matches.extend(OpponentMatch(opponent="current", seat=0) for _ in range(counts["checkpoint"]))
        random.Random(int(seed)).shuffle(matches)
        return [
            OpponentMatch(
                match.opponent, index % 2, match.checkpoint, match.mixed_opponent,
            )
            for index, match in enumerate(matches)
        ]


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
) -> tuple[list[float], list[float]]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have the same length")
    gamma = _finite_float(gamma, "gamma")
    gae_lambda = _finite_float(gae_lambda, "gae_lambda")
    rewards = [_finite_float(reward, "rewards") for reward in rewards]
    values = [_finite_float(value, "values") for value in values]
    advantages = [0.0 for _ in rewards]
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
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


def build_rollout_batch(
    transitions: Sequence[dict[str, Any]], *, config: PPOConfig,
    value_estimates: Sequence[float] | None = None,
    old_log_probs: Sequence[float] | None = None,
) -> RolloutBatch:
    rows = list(transitions)
    if not rows:
        raise ValueError("rollout batch requires at least one transition")
    values = [0.0 for _ in rows] if value_estimates is None else [float(value) for value in value_estimates]
    old = [0.0 for _ in rows] if old_log_probs is None else [float(value) for value in old_log_probs]
    if len(values) != len(rows) or len(old) != len(rows):
        raise ValueError("value_estimates and old_log_probs must match transition count")
    rewards: list[float] = []
    dones: list[bool] = []
    worker_act: list[list[int]] = []
    worker_target: list[list[int]] = []
    worker_kind: list[list[int]] = []
    market_items: list[int] = []
    market_quantities: list[int] = []
    for transition in rows:
        done = bool(transition.get("done"))
        dones.append(done)
        rewards.append(
            terminal_bank_margin_reward(
                transition.get("final_bank"), transition.get("opponent_final_bank"),
            )
            if done else 0.0
        )
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
    advantages, returns = generalized_advantage_estimate(
        rewards=rewards, values=values, dones=dones,
        gamma=config.gamma, gae_lambda=config.gae_lambda,
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
    )


def _select_outputs(
    outputs: dict[str, Any], batch: RolloutBatch, *, device: Any = None,
) -> tuple[Any, Any]:
    th = require_torch()
    device = outputs["value"].device if device is None else device
    worker_act = th.tensor(batch.worker.act, dtype=th.long, device=device)
    worker_target = th.tensor(batch.worker.target, dtype=th.long, device=device)
    worker_kind = th.tensor(batch.worker.kind, dtype=th.long, device=device)
    market_items = th.tensor(batch.market_items, dtype=th.long, device=device)
    market_quantities = th.tensor(batch.market_quantities, dtype=th.long, device=device)
    act_log = outputs["worker_act_logits"].log_softmax(dim=-1)
    target_log = outputs["worker_target_logits"].log_softmax(dim=-1)
    kind_log = outputs["worker_kind_logits"].log_softmax(dim=-1)
    item_log = outputs["market_item_logits"].log_softmax(dim=-1)
    quantity_log = outputs["market_quantity_logits"].log_softmax(dim=-1)
    log_probs = (
        act_log.gather(-1, worker_act.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        + target_log.gather(-1, worker_target.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        + kind_log.gather(-1, worker_kind.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        + item_log.gather(-1, market_items.unsqueeze(-1)).squeeze(-1)
        + quantity_log.gather(-1, market_quantities.unsqueeze(-1)).squeeze(-1)
    )
    entropy = (
        -(act_log.exp() * act_log).sum(dim=-1).sum(dim=1)
        - (target_log.exp() * target_log).sum(dim=-1).sum(dim=1)
        - (kind_log.exp() * kind_log).sum(dim=-1).sum(dim=1)
        - (item_log.exp() * item_log).sum(dim=-1)
        - (quantity_log.exp() * quantity_log).sum(dim=-1)
    ).mean()
    return log_probs, entropy


def _distribution_regularization(outputs: dict[str, Any], prior_outputs: dict[str, Any] | None) -> tuple[Any, Any]:
    th = require_torch()
    zero = outputs["value"].sum() * 0.0
    if prior_outputs is None:
        return zero, zero
    kl_terms = []
    ce_terms = []
    for name in (
        "worker_act_logits", "worker_target_logits", "worker_kind_logits",
        "market_item_logits", "market_quantity_logits",
    ):
        log_probs = outputs[name].log_softmax(dim=-1)
        with th.no_grad():
            prior_log_probs = prior_outputs[name].log_softmax(dim=-1)
            prior_probs = prior_log_probs.exp()
        kl_terms.append((prior_probs * (prior_log_probs - log_probs)).sum(dim=-1).mean())
        ce_terms.append(-(prior_probs * log_probs).sum(dim=-1).mean())
    return sum(kl_terms) / len(kl_terms), sum(ce_terms) / len(ce_terms)


def validate_prior_checkpoint_metadata(metadata: Any) -> None:
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


def _load_prior_network(prior_checkpoint: str | Path | None, *, device: Any) -> Any:
    if prior_checkpoint is None:
        return None
    th = require_torch()
    checkpoint = th.load(
        prior_checkpoint, map_location="cpu", weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise ValueError("prior checkpoint metadata is required")
    validate_prior_checkpoint_metadata(checkpoint["metadata"])
    if "model_state_dict" not in checkpoint:
        raise ValueError("prior checkpoint model_state_dict is required")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("prior checkpoint model_state_dict must be an object")
    prior = CompactPolicyNet().to(device)
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


def ppo_update(
    network: Any, optimizer: Any, transitions: Sequence[dict[str, Any]], *,
    config: PPOConfig, batch_size: int | None = None, seed: int = 0,
    prior_checkpoint: str | Path | None = None, device: Any = None,
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
        old_log_probs, _old_entropy = _select_outputs(old_outputs, bootstrap, device=device)
    rollout = build_rollout_batch(
        rows, config=config,
        value_estimates=old_outputs["value"].detach().tolist(),
        old_log_probs=old_log_probs.detach().tolist(),
    )
    prior = _load_prior_network(prior_checkpoint, device=device)
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
            )
            outputs = network(mini_features)
            _ensure_finite_outputs(outputs)
            log_probs, entropy = _select_outputs(outputs, mini, device=device)
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
            clipped_values = old_values + th.clamp(
                outputs["value"] - old_values, -config.clip_epsilon, config.clip_epsilon,
            )
            value_loss = th.maximum((outputs["value"] - returns) ** 2, (clipped_values - returns) ** 2).mean()
            prior_outputs = prior(mini_features) if prior is not None else None
            kl_to_prior, prior_ce = _distribution_regularization(outputs, prior_outputs)
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
            optimizer.step()
            with th.no_grad():
                post_outputs = network(mini_features)
                _ensure_finite_outputs(post_outputs)
                post_log_probs, _post_entropy = _select_outputs(
                    post_outputs, mini, device=device,
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
    save_candidate_fn: Any | None = None,
    cleanup_candidate_fn: Any | None = None,
    start_step: int = 0,
    initial_ppo_updates: int = 0,
    initial_rollout_count: int = 0,
    progress_fn: Any | None = None,
) -> dict[str, Any]:
    """Run PPO with fresh scheduled league rollouts or explicit offline fallback."""
    steps = max(0, int(ppo_steps))
    if type(start_step) is not int or not 0 <= start_step <= steps:
        raise ValueError("start_step must be an integer between zero and ppo_steps")
    if type(initial_ppo_updates) is not int or initial_ppo_updates < 0:
        raise ValueError("initial_ppo_updates must be a nonnegative integer")
    if type(initial_rollout_count) is not int or initial_rollout_count < 0:
        raise ValueError("initial_rollout_count must be a nonnegative integer")
    if steps == 0:
        return {
            "ppo_updates": initial_ppo_updates,
            "rollout_count": initial_rollout_count,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
            "completed_steps": 0,
        }
    updater = update_fn or ppo_update
    if rollout_fn is None and not offline_ppo_fallback:
        raise ValueError("rollout_fn is required for PPO unless offline_ppo_fallback is explicitly selected")
    pool = opponent_pool or OpponentPool()
    schedule = pool.schedule(count=steps, seed=seed) if rollout_fn is not None else []
    total_updates = initial_ppo_updates
    rollout_count = initial_rollout_count
    last_metrics: dict[str, Any] | None = None
    offline_rows = list(transitions)
    if offline_ppo_fallback and not offline_rows:
        raise ValueError("offline_ppo_fallback requires collected transitions")
    ran_step = False
    for step in range(start_step, steps):
        ran_step = True
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
                rollout_kwargs["checkpoint_identity"] = match.checkpoint
            if _accepts_keyword_argument(rollout_fn, "mixed_opponent"):
                rollout_kwargs["mixed_opponent"] = getattr(match, "mixed_opponent", None)
            if _accepts_keyword_argument(rollout_fn, "network"):
                rollout_kwargs["network"] = network
            rollout = rollout_fn(**rollout_kwargs)
            rollout_count += 1
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
        last_metrics = updater(**update_kwargs)
        total_updates += int(last_metrics.get("updates", 0))
        step_summary = {
            "ppo_updates": total_updates,
            "rollout_count": rollout_count,
            "early_stopped": bool(last_metrics.get("early_stopped")),
            "last_metrics": last_metrics,
            "promotion": None,
            "completed_steps": step + 1,
        }
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
) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
        "engine_version": ENGINE_VERSION,
        "transition_count": int(transition_count),
        "ppo_config": asdict(config or PPOConfig()),
        "device": str(device),
    }


def checkpoint_metadata(
    transition_count: int, config: PPOConfig | None = None, *, device: Any = "cpu",
) -> dict[str, Any]:
    return _checkpoint_metadata(transition_count, config, device=device)


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
    transition_count: int,
) -> None:
    _validate_resume_configuration(payload["configuration"], configuration)
    progress = payload["progress"]
    epoch = progress["epoch"]
    cursor = progress["cursor"]
    round_index = progress["round"]
    epochs = configuration["steps"]
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
    if round_index == 0:
        if ppo_metrics is not None:
            raise ValueError("resume checkpoint metrics ppo_metrics must be null before PPO progress")
        if metrics["ppo_updates"] != 0:
            raise ValueError("resume checkpoint metrics ppo_updates must be zero before PPO progress")
    else:
        if type(ppo_metrics) is not dict:
            raise ValueError("resume checkpoint metrics ppo_metrics is required after PPO progress")
        missing = sorted(_PPO_RESUME_METRIC_FIELDS - set(ppo_metrics))
        unexpected = sorted(set(ppo_metrics) - _PPO_RESUME_METRIC_FIELDS)
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
        if type(ppo_metrics["early_stopped"]) is not bool:
            raise ValueError("resume checkpoint ppo_metrics early_stopped must be boolean")
        for field in ("last_metrics", "promotion"):
            if ppo_metrics[field] is not None and type(ppo_metrics[field]) is not dict:
                raise ValueError(
                    f"resume checkpoint ppo_metrics {field} must be an object or null"
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
    if type(metadata.get("transition_count")) is not int:
        raise ValueError("resume checkpoint metadata transition_count must be an integer")
    if metadata["transition_count"] != transition_count:
        raise ValueError(
            "resume checkpoint transition_count does not match the requested input"
        )
    if type(metadata.get("device")) is not str or not metadata["device"]:
        raise ValueError("resume checkpoint metadata device must be a nonempty string")


def make_fresh_rollout_fn(
    *, run_directory: str | Path, candidate_artifact: str | Path | None = None,
    candidate_artifact_callback: Any | None = None,
    seeds: Sequence[int], steps: int, workers: int = 1,
    game_timeout: float = 120.0, candidate_identity: str | None = None,
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
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be at least 2 to produce a transition")
    run_path = Path(run_directory)
    run_path.mkdir(parents=True, exist_ok=True)

    def fresh_rollout(
        *, step: int, opponent: str, seat: int, checkpoint: str | None,
        rollout_steps: int, candidate_artifact: str | Path | None = None,
        opponent_identity: str | None = None,
        checkpoint_identity: str | None = None,
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
            candidate_artifact=selected_artifact, candidate_identity=identity,
            opponent_artifact=opponent_artifact,
            opponent_checkpoint_identity=checkpoint_provenance,
            source_policy_identity=identity,
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
    resume_checkpoint: str | Path | None = None,
    prior_checkpoint: str | Path | None = None, rollout_fn: Any | None = None,
    opponent_pool: Any | None = None, offline_ppo_fallback: bool = False,
    promotion_match_fn: Any | None = None,
    best_checkpoint_path: str | Path | None = None,
    checkpoint_registry: dict[str, Any] | None = None,
    candidate_artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Run complete behavior-cloning epochs, optional PPO, and checkpoint."""
    th = require_torch()
    resolved_device = resolve_device(device)
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(steps))
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    input_identity = _trajectory_identity(input_path)
    transitions = _read_transitions(input_path)
    features = [extract_features(transition.get("observation", {})) for transition in transitions]
    actions = [transition.get("action", {}) if isinstance(transition.get("action"), dict) else {} for transition in transitions]
    configuration = {
        "input_trajectory": input_identity,
        "steps": epochs,
        "batch_size": batch_size,
        "seed": int(seed),
        "ppo_steps": int(ppo_steps),
        "device": str(resolved_device),
        "prior_checkpoint": _checkpoint_identity(prior_checkpoint),
        "offline_ppo_fallback": bool(offline_ppo_fallback),
        "ppo_config": asdict(PPOConfig()),
        "checkpoint_interval": checkpoint_interval,
    }
    _validate_configuration_shape(configuration, source="requested")
    start_epoch = 0
    start_cursor = 0
    round_index = 0
    bc_updates = 0
    initial_ppo_updates = 0
    initial_rollout_count = 0
    previous_ppo_metrics: dict[str, Any] | None = None
    resumed = None
    if resume_checkpoint is not None:
        resumed = read_checkpoint(resume_checkpoint, map_location="cpu")
        _validate_resume_payload(
            resumed,
            configuration=configuration,
            transition_count=len(transitions),
        )
        start_epoch = resumed["progress"]["epoch"]
        start_cursor = resumed["progress"]["cursor"]
        round_index = resumed["progress"]["round"]
        bc_updates = resumed["metrics"]["behavior_clone_updates"]
        initial_ppo_updates = resumed["metrics"]["ppo_updates"]
        previous_ppo_metrics = resumed["metrics"]["ppo_metrics"]
        if previous_ppo_metrics is not None:
            initial_rollout_count = previous_ppo_metrics.get("rollout_count", 0)
    rng_before_resume = capture_rng_state() if resumed is not None else None
    try:
        set_training_seed(seed)
        network = CompactPolicyNet().to(resolved_device)
        optimizer = th.optim.AdamW(network.parameters(), lr=1e-3)
        if resumed is not None:
            restore_checkpoint(
                resumed, model=network, optimizer=optimizer, restore_rng=True,
            )
    except Exception:
        if rng_before_resume is not None:
            restore_rng_state(rng_before_resume)
        raise
    metadata = _checkpoint_metadata(len(transitions), device=resolved_device)
    metadata["behavior_clone_epochs"] = epochs
    destination = Path(output_path)

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
            item_target = th.tensor(market_items, dtype=th.long, device=resolved_device)
            quantity_target = th.tensor(market_quantities, dtype=th.long, device=resolved_device)
            loss = (
                th.nn.functional.cross_entropy(outputs["worker_act_logits"].reshape(-1, 2), act_target.reshape(-1))
                + th.nn.functional.cross_entropy(outputs["worker_target_logits"].reshape(-1, 100), target_target.reshape(-1))
                + th.nn.functional.cross_entropy(
                    outputs["worker_kind_logits"].reshape(-1, len(ACTION_VOCAB["worker_kinds"])),
                    kind_target.reshape(-1),
                )
                + th.nn.functional.cross_entropy(outputs["market_item_logits"], item_target)
                + th.nn.functional.cross_entropy(outputs["market_quantity_logits"], quantity_target)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bc_updates += 1
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
        config = PPOConfig()
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
            save_candidate_fn=save_current_candidate,
            start_step=round_index,
            initial_ppo_updates=initial_ppo_updates,
            initial_rollout_count=initial_rollout_count,
            progress_fn=save_ppo_progress,
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


def _cli_training_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.ppo_steps > 0 and not args.offline_ppo_fallback:
        raise ValueError(
            "--ppo-steps requires --offline-ppo-fallback in the standalone CLI; "
            "fresh league rollouts are available through the rollout_fn API"
        )
    return {
        "offline_ppo_fallback": bool(args.offline_ppo_fallback),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="input_path")
    parser.add_argument("--output", type=Path, required=True, dest="output_path")
    parser.add_argument("--steps", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--checkpoint-interval", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None, dest="resume_checkpoint")
    parser.add_argument("--ppo-steps", type=_nonnegative_int, default=0)
    parser.add_argument("--prior-checkpoint", type=Path, default=None)
    parser.add_argument("--best-checkpoint", type=Path, default=None)
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
            ppo_steps=args.ppo_steps,
            prior_checkpoint=args.prior_checkpoint,
            offline_ppo_fallback=options["offline_ppo_fallback"],
            best_checkpoint_path=args.best_checkpoint,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
