"""Replay-only Kaggriculture simulator backed by the installed engine.

This module is intentionally an adapter, not a rules reimplementation.  It
keeps the Kaggle public state shape and delegates all game mutations to the
installed ``kaggle-environments`` Kaggriculture interpreter.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import ENGINE_VERSION

REQUIRED_PARITY_SEEDS = tuple(range(10))


class KaggricultureSimulator:
    """Fast replay adapter for deterministic, already-recorded action sequences."""

    def __init__(
        self,
        *,
        configuration: Mapping[str, Any] | None = None,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        try:
            from kaggle_environments import make
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "kaggle-environments is required for KaggricultureSimulator"
            ) from exc

        config = copy.deepcopy(dict(configuration or {}))
        if seed is not None:
            config["seed"] = seed
        self._env = make("kaggriculture", configuration=config, debug=debug)

    @property
    def configuration(self) -> dict[str, Any]:
        return copy.deepcopy(self._env.configuration)

    @property
    def info(self) -> dict[str, Any]:
        return copy.deepcopy(self._env.info)

    @property
    def done(self) -> bool:
        return bool(self._env.done)

    def replay(self, action_sequence: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        """Replay one two-player action list per environment transition.

        Actions must already be selected from a prior observation.  This method
        does not run agent callables or sanitize actions; the Kaggle framework
        schema validation and Kaggriculture interpreter keep their normal
        behavior for malformed or illegal actions.
        """
        if isinstance(action_sequence, (str, bytes)) or not isinstance(action_sequence, Sequence):
            raise ValueError("action_sequence must be a sequence of per-turn action pairs")

        for turn_actions in action_sequence:
            actions = _normalize_turn_actions(turn_actions)
            if self._env.done:
                raise ValueError("action_sequence contains actions after the environment is done")
            _step_recorded_actions(self._env, actions)
        return self._env.toJSON()


def recorded_actions_from_replay(replay: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    """Extract transition actions from a Kaggle replay JSON object."""
    steps = replay.get("steps") if isinstance(replay, Mapping) else None
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise ValueError("replay must contain a steps sequence")
    actions: list[list[dict[str, Any]]] = []
    for turn in steps[1:]:
        if isinstance(turn, (str, bytes)) or not isinstance(turn, Sequence):
            raise ValueError("each replay step must contain player states")
        actions.append([_state_action(state) for state in turn])
    return actions


def simulator_parity_status(seed_results: Mapping[int, bool]) -> dict[str, Any]:
    """Return the promotion-readiness status for required simulator parity seeds."""
    normalized = {seed: bool(passed) for seed, passed in seed_results.items() if type(seed) is int}
    required = set(REQUIRED_PARITY_SEEDS)
    passed = {seed for seed, result in normalized.items() if result}
    failed = {seed for seed, result in normalized.items() if seed in required and not result}
    missing = required - set(normalized)
    return {
        "engine_version": str(ENGINE_VERSION),
        "required_seeds": list(REQUIRED_PARITY_SEEDS),
        "passed_seeds": sorted(seed for seed in passed if seed in required),
        "failed_seeds": sorted(failed),
        "missing_seeds": sorted(missing),
        "promotion_ready": not failed and not missing and required <= passed,
    }


def _normalize_turn_actions(turn_actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(turn_actions, (str, bytes)) or not isinstance(turn_actions, Sequence):
        raise ValueError("each turn must contain one action per player")
    actions = [copy.deepcopy(action) for action in turn_actions]
    if len(actions) != 2:
        raise ValueError("Kaggriculture replay requires exactly two player actions per turn")
    if any(not isinstance(action, Mapping) for action in actions):
        raise ValueError("player actions must be mappings")
    return [dict(action) for action in actions]


def _state_action(state: Any) -> dict[str, Any]:
    action = state.get("action") if isinstance(state, Mapping) else None
    if not isinstance(action, Mapping):
        raise ValueError("replay player state is missing an action object")
    return copy.deepcopy(dict(action))


def _step_recorded_actions(env: Any, actions: Sequence[Mapping[str, Any]]) -> None:
    from kaggle_environments.utils import structify

    action_state = [
        {**env.state[index], "action": copy.deepcopy(action)}
        for index, action in enumerate(actions)
    ]
    previous_done = env.done
    env.state = structify(env.interpreter(structify(action_state), env))
    env.state[0].observation.step = 0 if previous_done else len(env.steps)
    if env.state[0].observation.step >= env.configuration.episodeSteps - 1:
        for player_state in env.state:
            if player_state.status in ("ACTIVE", "INACTIVE"):
                player_state.status = "DONE"
    env.steps.append(env.state)
    env.logs.append([])
