"""Bounded, deterministic rollout scheduling for training and evaluation."""

from __future__ import annotations

import inspect
import math
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RolloutRequest:
    seed: int
    opponent: str
    candidate_player: int
    steps: int
    request_key: str


@dataclass(frozen=True)
class RolloutResult:
    request: RolloutRequest
    status: str
    replay: Any
    diagnostic: dict[str, Any]

    @property
    def request_key(self) -> str:
        return self.request.request_key


def build_rollout_requests(
    *, seeds: list[int] | tuple[int, ...], opponents: list[str] | tuple[str, ...],
    seats: list[int] | tuple[int, ...], steps: int,
) -> list[RolloutRequest]:
    requests: list[RolloutRequest] = []
    seen: set[str] = set()
    for seed in seeds:
        for opponent in opponents:
            for seat in seats:
                key = f"seed={seed}|opponent={opponent}|seat={seat}"
                if key in seen:
                    raise ValueError(f"duplicate rollout request: {key}")
                seen.add(key)
                requests.append(RolloutRequest(seed, opponent, seat, steps, key))
    return requests


def resolve_worker_count(requested: int | None) -> int:
    available = os.cpu_count() or 1
    if requested is None:
        return max(1, available)
    if type(requested) is not int or requested < 1:
        raise ValueError("workers must be a positive integer")
    return min(requested, available)


def _invoke_runner(runner: Callable[..., Any], request: RolloutRequest, kwargs: dict[str, Any]) -> Any:
    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return runner(**kwargs)
    accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values())
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return runner(**kwargs)


def _execute_request(
    argument: tuple[Callable[..., Any], RolloutRequest, dict[str, Any], float | None],
) -> Any:
    runner, request, kwargs, timeout = argument
    previous_handler = None
    if timeout is not None and hasattr(signal, "setitimer"):
        def deadline(_signum: int, _frame: Any) -> None:
            raise TimeoutError(f"rollout exceeded {timeout:g}s")

        previous_handler = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _invoke_runner(runner, request, kwargs)
    finally:
        if timeout is not None and hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def _result_for(request: RolloutRequest, runner: Callable[..., Any], kwargs: dict[str, Any]) -> RolloutResult:
    try:
        replay = _invoke_runner(runner, request, kwargs)
        return RolloutResult(request, "success", replay, {})
    except TimeoutError as exc:
        return RolloutResult(
            request, "timeout", None,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
    except BaseException as exc:
        return RolloutResult(
            request, "failure", None,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )


def run_rollouts(
    *, seeds: list[int] | tuple[int, ...], opponents: list[str] | tuple[str, ...],
    seats: list[int] | tuple[int, ...], steps: int, workers: int | None = None,
    game_runner: Callable[..., Any] | None = None,
    replay_directory: str | Path | None = None,
    timeout: float | None = None,
    candidate_artifact: str | Path | None = None,
    candidate_identity: str | None = None,
    opponent_artifact: str | Path | None = None,
    opponent_checkpoint_identity: str | None = None,
) -> list[RolloutResult]:
    requests = build_rollout_requests(
        seeds=seeds, opponents=opponents, seats=seats, steps=steps,
    )
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout)) or float(timeout) <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    worker_count = resolve_worker_count(workers)
    if game_runner is None:
        from scripts.collect_trajectories import _run_game_isolated

        game_runner = _run_game_isolated

    def kwargs_for(request: RolloutRequest) -> dict[str, Any]:
        replay_path = None
        if replay_directory is not None:
            replay_path = Path(replay_directory) / (
                f"seed-{request.seed}-{request.opponent}-seat-{request.candidate_player}.json"
            )
        kwargs: dict[str, Any] = {
            "request_key": request.request_key,
            "opponent": request.opponent,
            "seed": request.seed,
            "steps": request.steps,
            "candidate_player": request.candidate_player,
            "replay_path": replay_path,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if candidate_artifact is not None:
            kwargs["candidate_artifact"] = candidate_artifact
        if candidate_identity is not None:
            kwargs["candidate_identity"] = candidate_identity
        if opponent_artifact is not None:
            kwargs["opponent_artifact"] = opponent_artifact
        if opponent_checkpoint_identity is not None:
            kwargs["opponent_checkpoint_identity"] = opponent_checkpoint_identity
        return kwargs

    if worker_count == 1:
        return [_result_for(request, game_runner, kwargs_for(request)) for request in requests]

    results: dict[str, RolloutResult] = {}
    executor = ProcessPoolExecutor(max_workers=worker_count)
    pending: dict[Any, RolloutRequest] = {}
    def record_result(future: Any, request: RolloutRequest) -> None:
        try:
            replay = future.result()
        except TimeoutError as exc:
            results[request.request_key] = RolloutResult(
                request, "timeout", None,
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
        except BaseException as exc:
            results[request.request_key] = RolloutResult(
                request, "failure", None,
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
        else:
            results[request.request_key] = RolloutResult(request, "success", replay, {})

    try:
        for request in requests:
            future = executor.submit(
                _execute_request, (game_runner, request, kwargs_for(request), timeout),
            )
            pending[future] = request
        if timeout is None:
            for future in as_completed(pending):
                record_result(future, pending[future])
        else:
            for future in as_completed(pending):
                record_result(future, pending[future])
    finally:
        if hasattr(executor, "shutdown"):
            executor.shutdown(wait=True, cancel_futures=True)
    return [results[request.request_key] for request in requests]
