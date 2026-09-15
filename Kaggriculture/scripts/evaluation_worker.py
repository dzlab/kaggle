"""Run one isolated Kaggriculture evaluation game.

The worker accepts exactly one JSON request on stdin and emits exactly one JSON
object on stdout.  Keeping engine imports and policy state in this process
prevents one game from contaminating the next game in a matrix.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Executing this file directly sets sys.path[0] to ``scripts/`` rather than the
# project root. Add the root before importing the local ``scripts`` package or
# ``kagriculture_agent`` package in a fresh interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import (
    EVALUATION_NAMES,
    OPPONENTS,
    VARIANTS,
    VariantPolicy,
    _deterministic_random_agent,
    _framework_error_record,
    _load_policy_reference,
    _resolve_variant,
    replay_record,
)
from kagriculture_agent.candidates import CANDIDATES, candidate_policy


class EvaluatorFailure(RuntimeError):
    """An evaluator or candidate defect that must terminate the worker."""


class _GuardedCandidate:
    def __init__(self, candidate: Any, *, accepts_configuration: bool = True) -> None:
        self.candidate = candidate
        self.accepts_configuration = (
            accepts_configuration
            and _callable_accepts_configuration(candidate, fallback=accepts_configuration)
        )
        self.failure: EvaluatorFailure | None = None
        self._policy_turns = 0
        self._learned_active_turns = 0
        self._learned_model_configured = self._has_learned_model()
        self._learned_model_status_counts: Counter[str] = Counter()

    def _has_learned_model(self) -> bool:
        for candidate in (
            self.candidate,
            getattr(self.candidate, "policy", None),
            getattr(self.candidate, "__self__", None),
        ):
            learned_policy = getattr(candidate, "learned_policy", None)
            if learned_policy is not None and getattr(learned_policy, "model_path", None) is not None:
                return True
        return False

    def _memory_diagnostics(self) -> Mapping[str, Any]:
        """Find diagnostics on direct, wrapped, and bound policy objects."""
        candidates = (
            self.candidate,
            getattr(self.candidate, "policy", None),
            getattr(self.candidate, "__self__", None),
        )
        for candidate in candidates:
            memory = getattr(candidate, "memory", None)
            diagnostics = getattr(memory, "diagnostics", None)
            if isinstance(diagnostics, Mapping):
                return diagnostics
        return {}

    def _record_policy_activity(self) -> None:
        diagnostics = self._memory_diagnostics()
        if not diagnostics:
            return
        if "learned_active" in diagnostics or "learned_model_status" in diagnostics:
            self._learned_model_configured = True
        if diagnostics.get("learned_active") is True:
            self._learned_active_turns += 1
        status = diagnostics.get("learned_model_status")
        if isinstance(status, str) and status:
            self._learned_model_status_counts[status] += 1

    def activity_diagnostics(self) -> dict[str, Any]:
        """Return per-game learned-policy activity for evaluator records."""
        if not self._learned_model_configured:
            return {"learned_model_configured": False}
        policy_turns = self._policy_turns
        return {
            "learned_model_configured": True,
            "policy_turns": policy_turns,
            "learned_active_turns": self._learned_active_turns,
            "learned_active_turn_fraction": (
                self._learned_active_turns / policy_turns if policy_turns else 0.0
            ),
            "learned_model_status_counts": dict(self._learned_model_status_counts),
        }

    def __call__(self, observation: Any, configuration: Any = None) -> Any:
        try:
            if self.accepts_configuration:
                result = self.candidate(observation, configuration)
            else:
                result = self.candidate(observation)
            self._policy_turns += 1
            self._record_policy_activity()
            return result
        except Exception as exc:
            self.failure = EvaluatorFailure(
                f"candidate policy failure: {type(exc).__name__}: {exc}"
            )
            raise self.failure from exc


def _callable_accepts_configuration(candidate: Any, *, fallback: bool) -> bool:
    """Choose the callable arity without trial calls that could mask TypeError."""
    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return fallback
    marker = object()
    try:
        signature.bind(marker, marker)
    except TypeError:
        try:
            signature.bind(marker)
        except TypeError:
            return fallback
        return False
    return True


def _ordered_agents(candidate: Any, opponent: Any, seat: int) -> list[Any]:
    """Return engine agent order for the requested candidate seat."""
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    return [candidate, opponent] if seat == 0 else [opponent, candidate]


def decode_worker_result(output: str) -> dict[str, Any]:
    """Decode the worker's one-line JSON response."""
    if not isinstance(output, str):
        raise ValueError("worker result must be text")
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("worker result must contain one JSON object")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("worker result is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("worker result must be an object")
    return value


def _request_value(request: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return request.get(key, default)


def _request_failure(request: Mapping[str, Any], error: str) -> dict[str, Any]:
    variant = _request_value(request, "variant")
    candidate = _request_value(request, "candidate")
    policy_identity = _request_value(request, "policy_identity", "previous-agent")
    opponent = _request_value(request, "opponent", "pass")
    seed = _request_value(request, "seed", 0)
    seat = _request_value(request, "seat", 0)
    if variant is None:
        variant = candidate
    if _request_value(request, "policy_path") is not None:
        variant = policy_identity
    if not isinstance(variant, str) or (variant not in EVALUATION_NAMES and not _request_value(request, "policy_path")):
        variant = "mixed"
    if not isinstance(opponent, str) or opponent not in OPPONENTS:
        opponent = "pass"
    if not isinstance(seed, int) or isinstance(seed, bool):
        seed = 0
    if type(seat) is not int or seat not in (0, 1):
        seat = 0
    return _framework_error_record(
        variant=variant, opponent=opponent, seed=seed, seat=seat, error=error,
    )


def run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run and normalize one worker request, including framework failures."""
    if not isinstance(request, Mapping):
        return _request_failure({}, "request must be a JSON object")
    variant_value = _request_value(request, "variant")
    candidate_value = _request_value(request, "candidate")
    policy_path = _request_value(request, "policy_path")
    policy_identity = _request_value(request, "policy_identity", "previous-agent")
    opponent = _request_value(request, "opponent")
    seed = _request_value(request, "seed")
    steps = _request_value(request, "steps")
    seat = _request_value(request, "seat", 0)
    churn_window = _request_value(request, "churn_window", 2)
    try:
        if policy_path is not None:
            if variant_value is not None or candidate_value is not None:
                raise ValueError("baseline policy is separate from variant and candidate modes")
            if not isinstance(policy_identity, str) or not policy_identity:
                raise ValueError("policy_identity is required with policy_path")
            variant = policy_identity
        else:
            variant = _resolve_variant(variant_value, candidate_value)
    except ValueError as exc:
        return _request_failure(request, str(exc))
    if type(opponent) is not str or opponent not in OPPONENTS:
        return _request_failure(request, f"unsupported opponent: {opponent}")
    if type(seed) is not int:
        return _request_failure(request, "seed must be an integer")
    if type(steps) is not int or steps < 1:
        return _request_failure(request, "steps must be a positive integer")
    if type(seat) is not int or seat not in (0, 1):
        return _request_failure(request, "seat must be 0 or 1")
    if type(churn_window) is not int or churn_window < 1:
        return _request_failure(request, "churn_window must be a positive integer")
    ablations = request.get("ablations")
    if ablations is not None and not isinstance(ablations, Mapping):
        return _request_failure(request, "ablations must be an object")
    try:
        from kaggle_environments import make

        env = make(
            "kaggriculture",
            configuration={"episodeSteps": steps, "seed": seed},
            debug=False,
        )
    except Exception as exc:
        return _framework_error_record(
            variant=variant,
            opponent=opponent,
            seed=seed,
            seat=seat,
            error=f"{type(exc).__name__}: {exc}",
        )

    # The request key is the namespace.  ``mixed`` intentionally exists in
    # both namespaces, so its spelling alone cannot determine the policy.
    is_route_candidate = candidate_value is not None
    try:
        if policy_path is not None:
            candidate = _load_policy_reference(policy_path)
        else:
            route_policy = candidate_policy(variant) if is_route_candidate else None
            candidate = VariantPolicy(
                variant, ablations, env.configuration, is_route_candidate, route_policy,
            )
    except Exception as exc:
        raise EvaluatorFailure(
            f"candidate policy construction failure: {type(exc).__name__}: {exc}"
        ) from exc
    guarded_candidate = _GuardedCandidate(candidate)
    opponent_agent = _deterministic_random_agent(seed) if opponent == "random" else opponent
    try:
        env.run(_ordered_agents(guarded_candidate, opponent_agent, seat))
    except EvaluatorFailure:
        raise
    except Exception as exc:  # A worker failure must be data, not process control flow.
        return _framework_error_record(
            variant=variant,
            opponent=opponent,
            seed=seed,
            seat=seat,
            error=f"{type(exc).__name__}: {exc}",
        )
    if guarded_candidate.failure is not None:
        raise guarded_candidate.failure
    try:
        replay = env.toJSON()
    except Exception as exc:
        return _framework_error_record(
            variant=variant,
            opponent=opponent,
            seed=seed,
            seat=seat,
            error=f"{type(exc).__name__}: {exc}",
        )
    policy_diagnostics = guarded_candidate.activity_diagnostics()
    if policy_diagnostics.get("learned_model_configured"):
        replay["policy_diagnostics"] = policy_diagnostics
    return replay_record(
        replay, variant=variant, opponent=opponent, seed=seed, seat=seat,
        churn_window=churn_window,
    )


def main() -> int:
    raw = sys.stdin.readline()
    trailing = sys.stdin.read()
    if trailing.strip():
        print(
            "evaluation worker request failure: expected exactly one JSON request",
            file=sys.stderr,
        )
        return 1
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        result = _request_failure({}, f"invalid request JSON: {exc.msg}")
    else:
        try:
            result = run_request(request)
        except Exception as exc:
            print(
                f"evaluation worker evaluator failure: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
