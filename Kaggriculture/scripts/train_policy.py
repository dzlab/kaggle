"""Train compact Kaggriculture policies with behavior cloning and PPO helpers."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION, extract_features
from kagriculture_agent.model import ACTION_VOCAB, MODEL_VERSION, CompactPolicyNet, require_torch, set_training_seed

PROMOTION_MATCH_SIZE = 100
LOG_RATIO_CLAMP = 20.0
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


@dataclass(frozen=True)
class OpponentMatch:
    opponent: str
    seat: int
    checkpoint: str | None = None


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
        return OpponentMatch(opponent=selected, seat=int(index) % 2, checkpoint=checkpoint)

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
                matches.extend(OpponentMatch(opponent=opponent, seat=0) for _ in range(counts[opponent]))
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
            OpponentMatch(match.opponent, index % 2, match.checkpoint)
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


def _select_outputs(outputs: dict[str, Any], batch: RolloutBatch) -> tuple[Any, Any]:
    th = require_torch()
    worker_act = th.tensor(batch.worker.act, dtype=th.long)
    worker_target = th.tensor(batch.worker.target, dtype=th.long)
    worker_kind = th.tensor(batch.worker.kind, dtype=th.long)
    market_items = th.tensor(batch.market_items, dtype=th.long)
    market_quantities = th.tensor(batch.market_quantities, dtype=th.long)
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


def _load_prior_network(prior_checkpoint: str | Path | None) -> Any:
    if prior_checkpoint is None:
        return None
    th = require_torch()
    prior = CompactPolicyNet()
    checkpoint = th.load(prior_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise ValueError("prior checkpoint metadata is required")
    validate_prior_checkpoint_metadata(checkpoint["metadata"])
    if "model_state_dict" not in checkpoint:
        raise ValueError("prior checkpoint model_state_dict is required")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
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
    prior_checkpoint: str | Path | None = None,
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
        old_log_probs, _old_entropy = _select_outputs(old_outputs, bootstrap)
    rollout = build_rollout_batch(
        rows, config=config,
        value_estimates=old_outputs["value"].detach().tolist(),
        old_log_probs=old_log_probs.detach().tolist(),
    )
    prior = _load_prior_network(prior_checkpoint)
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
            log_probs, entropy = _select_outputs(outputs, mini)
            old_log = th.tensor(mini.old_log_probs, dtype=th.float32)
            advantages = th.tensor(mini.advantages, dtype=th.float32)
            returns = th.tensor(mini.returns, dtype=th.float32)
            old_values = th.tensor(mini.values, dtype=th.float32)
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
                post_log_probs, _post_entropy = _select_outputs(post_outputs, mini)
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


def run_ppo_training(
    *, network: Any, optimizer: Any, transitions: Sequence[dict[str, Any]],
    ppo_steps: int, config: PPOConfig, batch_size: int | None = None,
    seed: int = 0, prior_checkpoint: str | Path | None = None,
    opponent_pool: Any | None = None, rollout_fn: Any | None = None,
    offline_ppo_fallback: bool = False, update_fn: Any | None = None,
    promotion_match_fn: Any | None = None,
    candidate_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run PPO with fresh scheduled league rollouts or explicit offline fallback."""
    steps = max(0, int(ppo_steps))
    if steps == 0:
        return {
            "ppo_updates": 0,
            "rollout_count": 0,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
        }
    updater = update_fn or ppo_update
    if rollout_fn is None and not offline_ppo_fallback:
        raise ValueError("rollout_fn is required for PPO unless offline_ppo_fallback is explicitly selected")
    pool = opponent_pool or OpponentPool()
    schedule = pool.schedule(count=steps, seed=seed) if rollout_fn is not None else []
    total_updates = 0
    rollout_count = 0
    last_metrics: dict[str, Any] | None = None
    offline_rows = list(transitions)
    if offline_ppo_fallback and not offline_rows:
        raise ValueError("offline_ppo_fallback requires collected transitions")
    for step in range(steps):
        if rollout_fn is None:
            rollout = offline_rows[:config.rollout_steps]
        else:
            match = schedule[step]
            rollout = rollout_fn(
                step=step,
                opponent=match.opponent,
                seat=match.seat,
                checkpoint=match.checkpoint,
                rollout_steps=config.rollout_steps,
            )
            rollout_count += 1
        if not isinstance(rollout, Sequence) or isinstance(rollout, (str, bytes)):
            raise ValueError("rollout_fn must return a sequence of transitions")
        if not rollout:
            raise ValueError("PPO rollout produced no transitions")
        last_metrics = updater(
            network=network,
            optimizer=optimizer,
            transitions=list(rollout),
            config=config,
            batch_size=batch_size,
            seed=int(seed) + step,
            prior_checkpoint=prior_checkpoint,
        )
        total_updates += int(last_metrics.get("updates", 0))
        if last_metrics.get("early_stopped"):
            return {
                "ppo_updates": total_updates,
                "rollout_count": rollout_count,
                "early_stopped": True,
                "last_metrics": last_metrics,
                "promotion": None,
            }
    promotion = None
    if promotion_match_fn is not None:
        if candidate_checkpoint is None:
            raise ValueError("candidate_checkpoint is required when promotion_match_fn is provided")
        promotion = maybe_promote_checkpoint(
            match_fn=promotion_match_fn,
            candidate_checkpoint=candidate_checkpoint,
        )
    return {
        "ppo_updates": total_updates,
        "rollout_count": rollout_count,
        "early_stopped": False,
        "last_metrics": last_metrics,
        "promotion": promotion,
    }


def _checkpoint_metadata(transition_count: int, config: PPOConfig | None = None) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
        "engine_version": ENGINE_VERSION,
        "transition_count": int(transition_count),
        "ppo_config": asdict(config or PPOConfig()),
    }


def checkpoint_metadata(transition_count: int, config: PPOConfig | None = None) -> dict[str, Any]:
    return _checkpoint_metadata(transition_count, config)


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


def maybe_promote_checkpoint(
    *, match_fn: Any, candidate_checkpoint: str | Path,
    registry: dict[str, Any] | None = None,
    save_candidate_fn: Any | None = None,
    cleanup_candidate_fn: Any | None = None,
    best_key: str = "best",
) -> dict[str, Any]:
    """Register a candidate, run the fixed promotion match, and update best only on promotion."""
    registry = registry if registry is not None else {best_key: None, "candidates": []}
    previous_best = registry.get(best_key)
    saved_candidate = (
        str(save_candidate_fn(candidate_checkpoint))
        if save_candidate_fn is not None else str(candidate_checkpoint)
    )
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
        if cleanup_candidate_fn is not None:
            cleanup_candidate_fn(saved_candidate)
        raise
    if result["promoted"]:
        entry["status"] = "promoted"
        registry[best_key] = saved_candidate
    else:
        entry["status"] = "rejected"
        registry[best_key] = previous_best
        if cleanup_candidate_fn is not None:
            cleanup_candidate_fn(saved_candidate)
    return {"candidate_checkpoint": saved_candidate, **result}


def train_behavior_clone(
    *, input_path: str | Path, output_path: str | Path, steps: int,
    batch_size: int, seed: int = 0, ppo_steps: int = 0,
    prior_checkpoint: str | Path | None = None, rollout_fn: Any | None = None,
    opponent_pool: Any | None = None, offline_ppo_fallback: bool = False,
    promotion_match_fn: Any | None = None,
) -> dict[str, Any]:
    """Run complete behavior-cloning epochs, optional PPO, and checkpoint."""
    th = require_torch()
    set_training_seed(seed)
    transitions = _read_transitions(input_path)
    features = [extract_features(transition.get("observation", {})) for transition in transitions]
    actions = [transition.get("action", {}) if isinstance(transition.get("action"), dict) else {} for transition in transitions]
    network = CompactPolicyNet()
    optimizer = th.optim.AdamW(network.parameters(), lr=1e-3)
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(steps))
    bc_updates = 0
    for epoch in range(epochs):
        for permutation in epoch_minibatches(
            count=len(features), batch_size=batch_size, seed=int(seed), epoch=epoch,
        ):
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
            act_target = th.tensor([label.act for label in labels], dtype=th.long)
            target_target = th.tensor([label.target for label in labels], dtype=th.long)
            kind_target = th.tensor([label.kind for label in labels], dtype=th.long)
            market_items, market_quantities = zip(*(_market_labels(action) for action in batch_actions))
            item_target = th.tensor(market_items, dtype=th.long)
            quantity_target = th.tensor(market_quantities, dtype=th.long)
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
    ppo_metrics = None
    if ppo_steps:
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
            opponent_pool=opponent_pool,
            rollout_fn=rollout_fn,
            offline_ppo_fallback=offline_ppo_fallback,
            promotion_match_fn=promotion_match_fn,
            candidate_checkpoint=output_path,
        )
        if isinstance(ppo_metrics.get("last_metrics"), dict):
            ppo_metrics = {**ppo_metrics, **ppo_metrics["last_metrics"]}
    metadata = _checkpoint_metadata(len(transitions))
    metadata["behavior_clone_epochs"] = epochs
    metadata["behavior_clone_updates"] = bc_updates
    metadata["ppo_updates"] = 0 if ppo_metrics is None else ppo_metrics["ppo_updates"]
    metadata["ppo_metrics"] = ppo_metrics
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    th.save({"metadata": metadata, "model_state_dict": network.state_dict()}, destination)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ppo-steps", type=_nonnegative_int, default=0)
    parser.add_argument("--prior-checkpoint", type=Path, default=None)
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
            seed=args.seed,
            ppo_steps=args.ppo_steps,
            prior_checkpoint=args.prior_checkpoint,
            offline_ppo_fallback=options["offline_ppo_fallback"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
