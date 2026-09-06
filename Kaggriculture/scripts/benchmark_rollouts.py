"""Benchmark Kaggriculture self-play rollout throughput against the real engine."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.policy import Policy
from scripts.collect_trajectories import DEFAULT_GAME_TIMEOUT_SECONDS, _run_game_isolated

MIN_REAL_ENGINE_STEPS_PER_MINUTE = 100_000
MAX_POLICY_INFERENCE_MS_PER_TURN = 10.0
DEFAULT_OPPONENT = "current"


class SequenceClock:
    """Small deterministic clock used by tests."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        if self._index >= len(self._values):
            return self._values[-1]
        value = self._values[self._index]
        self._index += 1
        return value


class UniquePositiveInts(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Sequence[str],
        option_string: str | None = None,
    ) -> None:
        try:
            parsed = [_positive_int(value) for value in values]
        except argparse.ArgumentTypeError as exc:
            parser.error(f"{option_string or self.dest} {exc}")
        if len(parsed) != len(set(parsed)):
            parser.error(f"{option_string or self.dest} must not contain duplicate values")
        setattr(namespace, self.dest, parsed)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return float(min(values))
    if percentile >= 100:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    index = math.ceil((percentile / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(len(ordered) - 1, index))]


def _candidate_observations(replay: Mapping[str, Any], candidate_player: int) -> Iterable[Mapping[str, Any]]:
    steps = replay.get("steps", ())
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return
    for turn in steps[:-1]:
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes)):
            continue
        for state in turn:
            if not isinstance(state, Mapping):
                continue
            observation = state.get("observation")
            if isinstance(observation, Mapping) and observation.get("player") == candidate_player:
                yield observation
                break


def sample_policy_latencies(
    replays: Sequence[Mapping[str, Any]],
    policy_factory: Callable[[], Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> list[float]:
    """Replay public candidate observations through the policy and time each act call."""
    policy_factory = policy_factory or Policy
    clock = clock or time.perf_counter
    latencies: list[float] = []
    for replay in replays:
        info = replay.get("info", {}) if isinstance(replay, Mapping) else {}
        candidate_player = info.get("candidate_player", 0) if isinstance(info, Mapping) else 0
        if type(candidate_player) is not int or candidate_player not in (0, 1):
            candidate_player = 0
        policy = policy_factory()
        for observation in _candidate_observations(replay, candidate_player):
            start = clock()
            policy.act(observation)
            latencies.append((clock() - start) * 1000.0)
    return latencies


def summarize_run(
    *,
    worker_count: int,
    game_count: int,
    environment_steps: int,
    rollout_seconds: float,
    inference_latencies_ms: Sequence[float],
) -> dict[str, Any]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if environment_steps < 0:
        raise ValueError("environment_steps must be non-negative")
    if not math.isfinite(rollout_seconds) or rollout_seconds <= 0:
        raise ValueError("rollout_seconds must be positive and finite")
    latencies = [float(value) for value in inference_latencies_ms if math.isfinite(float(value))]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    steps_per_second = environment_steps / rollout_seconds
    return {
        "workers": int(worker_count),
        "games": int(game_count),
        "environment_steps": int(environment_steps),
        "rollout_seconds": float(rollout_seconds),
        "games_per_hour": float(game_count / rollout_seconds * 3600.0),
        "environment_steps_per_second": float(steps_per_second),
        "environment_steps_per_minute": float(steps_per_second * 60.0),
        "policy_inference_ms_per_turn": float(mean_latency),
        "policy_inference_p95_ms": float(_percentile(latencies, 95.0)),
    }


def real_engine_gate_passed(results: Sequence[Mapping[str, Any]]) -> bool:
    for result in results:
        if result.get("workers") != 4:
            continue
        return (
            float(result.get("environment_steps_per_minute", 0.0)) >= MIN_REAL_ENGINE_STEPS_PER_MINUTE
            and float(result.get("policy_inference_ms_per_turn", float("inf"))) < MAX_POLICY_INFERENCE_MS_PER_TURN
        )
    return False


def _game_job(args: tuple[str, int, int, int, str, float]) -> tuple[dict[str, Any], int]:
    opponent, seed, steps, candidate_player, replay_path, timeout = args
    replay = _run_game_isolated(
        opponent=opponent,
        seed=seed,
        steps=steps,
        candidate_player=candidate_player,
        replay_path=Path(replay_path),
        timeout=timeout,
    )
    if isinstance(replay.get("info"), dict):
        replay["info"]["candidate_player"] = candidate_player
    step_records = replay.get("steps", ())
    environment_steps = max(0, len(step_records) - 1) if isinstance(step_records, Sequence) else 0
    return replay, environment_steps


def _run_jobs(
    jobs: Sequence[tuple[str, int, int, int, str, float]],
    worker_count: int,
    game_runner: Callable[..., Mapping[str, Any]] | None,
) -> list[tuple[Mapping[str, Any], int]]:
    if game_runner is not None:
        results = []
        for opponent, seed, steps, candidate_player, replay_path, timeout in jobs:
            replay = game_runner(
                opponent=opponent,
                seed=seed,
                steps=steps,
                candidate_player=candidate_player,
                replay_path=Path(replay_path),
                timeout=timeout,
            )
            if isinstance(replay.get("info"), dict):
                replay["info"]["candidate_player"] = candidate_player
            elif isinstance(replay, dict):
                replay["info"] = {"candidate_player": candidate_player}
            step_records = replay.get("steps", ()) if isinstance(replay, Mapping) else ()
            environment_steps = max(0, len(step_records) - 1) if isinstance(step_records, Sequence) else 0
            results.append((replay, environment_steps))
        return results
    if worker_count == 1:
        return [_game_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_game_job, jobs))


def benchmark_worker_count(
    *,
    games: int,
    steps: int,
    worker_count: int,
    start_seed: int = 0,
    opponent: str = DEFAULT_OPPONENT,
    output_dir: str | Path,
    game_timeout: float = DEFAULT_GAME_TIMEOUT_SECONDS,
    game_runner: Callable[..., Mapping[str, Any]] | None = None,
    latency_sampler: Callable[..., Sequence[float]] = sample_policy_latencies,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    clock = clock or time.perf_counter
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            opponent,
            start_seed + index,
            steps,
            index % 2,
            str(output_path / f"workers-{worker_count}-game-{index}.json"),
            float(game_timeout),
        )
        for index in range(games)
    ]
    start = clock()
    game_results = _run_jobs(jobs, worker_count, game_runner)
    rollout_seconds = clock() - start
    replays = [result[0] for result in game_results]
    environment_steps = sum(result[1] for result in game_results)
    latencies = list(latency_sampler(replays))
    return summarize_run(
        worker_count=worker_count,
        game_count=games,
        environment_steps=environment_steps,
        rollout_seconds=rollout_seconds,
        inference_latencies_ms=latencies,
    )


def run_benchmark(
    *,
    games: int,
    steps: int,
    workers: Sequence[int],
    start_seed: int = 0,
    opponent: str = DEFAULT_OPPONENT,
    output_dir: str | Path | None = None,
    game_timeout: float = DEFAULT_GAME_TIMEOUT_SECONDS,
    benchmark_fn: Callable[..., Mapping[str, Any]] = benchmark_worker_count,
) -> dict[str, Any]:
    if len(workers) != len(set(workers)):
        raise ValueError("workers must not contain duplicate values")
    temporary_directory = None
    if output_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="kagriculture-rollout-benchmark-")
        output_dir = temporary_directory.name
    try:
        results = [
            dict(benchmark_fn(
                games=games,
                steps=steps,
                worker_count=worker_count,
                start_seed=start_seed,
                opponent=opponent,
                output_dir=Path(output_dir) / f"workers-{worker_count}",
                game_timeout=game_timeout,
            ))
            for worker_count in workers
        ]
        keep_real_engine = real_engine_gate_passed(results)
        return {
            "engine_version": str(ENGINE_VERSION),
            "opponent": opponent,
            "games": int(games),
            "steps": int(steps),
            "start_seed": int(start_seed),
            "results": results,
            "gate": {
                "four_worker_result_present": any(result.get("workers") == 4 for result in results),
                "throughput_steps_per_minute_threshold": MIN_REAL_ENGINE_STEPS_PER_MINUTE,
                "inference_ms_per_turn_threshold": MAX_POLICY_INFERENCE_MS_PER_TURN,
                "real_engine_kept": keep_real_engine,
                "simulator_required": not keep_real_engine,
            },
        }
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=_positive_int, default=10)
    parser.add_argument("--steps", type=_positive_int, default=720)
    parser.add_argument("--workers", nargs="+", action=UniquePositiveInts, default=[1, 2, 4, 8])
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--opponent", choices=("current", "pass", "random", "starter"), default=DEFAULT_OPPONENT)
    parser.add_argument("--game-timeout", type=_positive_float, default=DEFAULT_GAME_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="optional path for the JSON benchmark report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        games=args.games,
        steps=args.steps,
        workers=args.workers,
        start_seed=args.start_seed,
        opponent=args.opponent,
        output_dir=args.output_dir,
        game_timeout=args.game_timeout,
    )
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
