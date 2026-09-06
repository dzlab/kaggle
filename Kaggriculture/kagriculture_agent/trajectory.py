"""Validated replay-to-trajectory conversion for Kaggriculture."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from unittest.mock import patch

TRANSITION_SCHEMA_VERSION = 1


class ReplayValidationError(ValueError):
    """A replay cannot be converted into a trustworthy trajectory."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.details = dict(details)
        super().__init__(f"{code}: {message}")

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": copy.deepcopy(self.details)}


@dataclass
class Transition:
    """One candidate action and the state transition it caused."""

    observation: dict[str, Any]
    action: dict[str, Any]
    next_observation: dict[str, Any]
    done: bool
    reward: float
    final_bank: float
    opponent_final_bank: float
    safety_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible representation."""
        return copy.deepcopy(asdict(self))

    def to_json(self) -> str:
        """Serialize deterministically for stable JSONL output and tests."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return None


def _failure(message: str, **details: Any) -> ReplayValidationError:
    return ReplayValidationError("malformed_replay", message, **details)


def _validated_player_states(
    replay: Mapping[str, Any], candidate_player: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any]]:
    if type(candidate_player) is not int or candidate_player not in (0, 1):
        raise ReplayValidationError(
            "invalid_candidate_player",
            "candidate_player must be 0 or 1",
            candidate_player=candidate_player,
        )
    steps = _sequence(replay.get("steps"))
    if not isinstance(replay, Mapping) or steps is None or len(steps) < 2:
        raise _failure("replay must contain at least two ordered engine states", reason="incomplete_sequence")

    # These helpers are the evaluator's single source of truth for replay
    # shape, engine provenance, action legality, and transition effects.
    from scripts.evaluate import _final_bank, _player_states, _valid_replay

    own_states = _player_states(replay, candidate_player)
    other_states = _player_states(replay, 1 - candidate_player)
    configuration = _mapping(replay.get("configuration"))
    info = _mapping(replay.get("info"))
    expected_seed = info.get("seed") if type(info.get("seed")) is int else None
    try:
        # Trajectory data may retain terminal goods for later analysis. Keep
        # every evaluator structural, action-schema, and transition-effect
        # check, but do not apply the evaluator's selection-only full-season
        # liquidation gate to collection.
        with patch("scripts.evaluate._requires_full_liquidation", return_value=False):
            valid = _valid_replay(
                replay,
                own_states,
                other_states,
                configuration,
                expected_seed=expected_seed,
                candidate_player=candidate_player,
            )
    except Exception as exc:
        raise _failure(
            "replay validation raised an exception",
            reason="replay_validation_exception",
            exception_type=type(exc).__name__,
        ) from exc
    if not valid:
        raise _failure(
            "replay failed evaluator validation",
            reason="replay_validation_failed",
            candidate_player=candidate_player,
            candidate_state_count=len(own_states),
            opponent_state_count=len(other_states),
            step_count=len(steps),
        )
    if len(own_states) != len(steps) or len(other_states) != len(steps):
        raise _failure(
            "replay does not contain one state for each player at every step",
            reason="incomplete_pairing",
            candidate_state_count=len(own_states),
            opponent_state_count=len(other_states),
            step_count=len(steps),
        )
    candidate_bank = _final_bank(own_states[-1])
    opponent_bank = _final_bank(other_states[-1])
    if candidate_bank is None or opponent_bank is None:
        raise _failure(
            "terminal banks are not finite numeric values",
            reason="missing_terminal_banks",
        )
    return own_states, other_states, configuration


def transitions_from_replay(replay: Mapping[str, Any], candidate_player: int = 0) -> list[Transition]:
    """Convert a validated engine replay into candidate-player transitions.

    Kaggle stores the action chosen from state ``n`` on the state record at
    ``n + 1``.  The bootstrap record's placeholder action is therefore omitted
    so every returned transition represents exactly one engine step.
    """
    if not isinstance(replay, Mapping):
        raise _failure("replay must be a JSON object", reason="invalid_replay_type")
    own_states, other_states, _configuration = _validated_player_states(replay, candidate_player)

    from scripts.evaluate import _final_bank

    candidate_bank = _final_bank(own_states[-1])
    opponent_bank = _final_bank(other_states[-1])
    assert candidate_bank is not None and opponent_bank is not None
    terminal_reward = math.tanh((candidate_bank - opponent_bank) / 1000.0)
    transitions: list[Transition] = []
    for index in range(len(own_states) - 1):
        preceding = _mapping(own_states[index].get("observation"))
        state = own_states[index + 1]
        following = _mapping(state.get("observation"))
        action = state.get("action")
        if not isinstance(action, Mapping):
            raise _failure(
                "candidate action is missing from a paired engine step",
                reason="incomplete_pairing",
                step=index + 1,
            )
        transitions.append(
            Transition(
                observation=copy.deepcopy(dict(preceding)),
                action=copy.deepcopy(dict(action)),
                next_observation=copy.deepcopy(dict(following)),
                done=index == len(own_states) - 2,
                reward=float(terminal_reward if index == len(own_states) - 2 else 0.0),
                final_bank=float(candidate_bank),
                opponent_final_bank=float(opponent_bank),
                safety_flags=[],
            )
        )
    return transitions
