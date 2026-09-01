"""Run one isolated Kaggriculture evaluation game.

The worker accepts exactly one JSON request on stdin and emits exactly one JSON
object on stdout.  Keeping engine imports and policy state in this process
prevents one game from contaminating the next game in a matrix.
"""

from __future__ import annotations

import json
import sys
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
    _resolve_variant,
    replay_record,
)
from kagriculture_agent.candidates import CANDIDATES, candidate_policy


class EvaluatorFailure(RuntimeError):
    """An evaluator or candidate defect that must terminate the worker."""


class _GuardedCandidate:
    def __init__(self, candidate: Any, *, accepts_configuration: bool = True) -> None:
        self.candidate = candidate
        self.accepts_configuration = accepts_configuration
        self.failure: EvaluatorFailure | None = None

    def __call__(self, observation: Any, configuration: Any = None) -> Any:
        try:
            if self.accepts_configuration:
                return self.candidate(observation, configuration)
            return self.candidate(observation)
        except Exception as exc:
            self.failure = EvaluatorFailure(
                f"candidate policy failure: {type(exc).__name__}: {exc}"
            )
            raise self.failure from exc


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
    opponent = _request_value(request, "opponent", "pass")
    seed = _request_value(request, "seed", 0)
    seat = _request_value(request, "seat", 0)
    if variant is None:
        variant = candidate
    if not isinstance(variant, str) or variant not in EVALUATION_NAMES:
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
    opponent = _request_value(request, "opponent")
    seed = _request_value(request, "seed")
    steps = _request_value(request, "steps")
    seat = _request_value(request, "seat", 0)
    try:
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

    is_route_candidate = candidate_value in CANDIDATES and variant_value is None
    try:
        route_policy = candidate_policy(candidate_value) if is_route_candidate else None
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
    return replay_record(
        replay, variant=variant, opponent=opponent, seed=seed, seat=seat,
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
