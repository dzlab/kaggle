from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest


def test_request_order_is_stable_and_rejects_duplicate_keys():
    from kagriculture_agent.rollouts import build_rollout_requests

    requests = build_rollout_requests(
        seeds=[11, 7], opponents=["starter", "pass"], seats=[1, 0], steps=4,
    )

    assert [request.request_key for request in requests] == [
        "seed=11|opponent=starter|seat=1",
        "seed=11|opponent=starter|seat=0",
        "seed=11|opponent=pass|seat=1",
        "seed=11|opponent=pass|seat=0",
        "seed=7|opponent=starter|seat=1",
        "seed=7|opponent=starter|seat=0",
        "seed=7|opponent=pass|seat=1",
        "seed=7|opponent=pass|seat=0",
    ]

    with pytest.raises(ValueError, match="duplicate rollout request"):
        build_rollout_requests(seeds=[7, 7], opponents=["pass"], seats=[0], steps=4)


@pytest.mark.parametrize(
    ("requested", "cpu_count", "expected"),
    [(1, 8, 1), (99, 8, 8), (99, None, 1), (None, 3, 3)],
)
def test_worker_count_is_positive_and_capped_by_available_cpus(monkeypatch, requested, cpu_count, expected):
    from kagriculture_agent import rollouts

    monkeypatch.setattr(rollouts.os, "cpu_count", lambda: cpu_count)
    assert rollouts.resolve_worker_count(requested) == expected


class _ImmediateExecutor:
    instances = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submissions = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, argument):
        future = Future()
        future.set_result(function(argument))
        self.submissions.append(future)
        return future


def test_parallel_results_are_sorted_by_request_key_independent_of_completion_order(monkeypatch):
    from kagriculture_agent import rollouts

    _ImmediateExecutor.instances.clear()
    monkeypatch.setattr(rollouts, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(rollouts, "as_completed", lambda futures: reversed(list(futures)))

    def fake_game(**kwargs):
        return {"request_key": kwargs["request_key"]}

    results = rollouts.run_rollouts(
        seeds=[3, 2], opponents=["pass"], seats=[0, 1], steps=4,
        workers=2, game_runner=fake_game,
    )

    assert _ImmediateExecutor.instances[0].max_workers == 2
    assert [result.request_key for result in results] == [
        "seed=3|opponent=pass|seat=0",
        "seed=3|opponent=pass|seat=1",
        "seed=2|opponent=pass|seat=0",
        "seed=2|opponent=pass|seat=1",
    ]
    assert all(result.status == "success" for result in results)


def test_worker_failure_and_timeout_are_explicit_and_never_have_replays():
    from kagriculture_agent import rollouts

    def fake_game(**kwargs):
        if kwargs["seed"] == 1:
            raise TimeoutError("game deadline exceeded")
        raise OSError("engine crashed")

    results = rollouts.run_rollouts(
        seeds=[1, 2], opponents=["pass"], seats=[0], steps=4,
        workers=1, game_runner=fake_game,
    )

    assert [(result.status, result.replay) for result in results] == [
        ("timeout", None),
        ("failure", None),
    ]
    assert results[0].diagnostic["error_type"] == "TimeoutError"
    assert results[1].diagnostic["error_type"] == "OSError"


def test_cli_exposes_parallel_rollout_options():
    from scripts.collect_trajectories import _parser

    args = _parser().parse_args([
        "--output", "trajectories.jsonl", "--workers", "3",
        "--candidate-artifact", "candidate.json", "--candidate-identity", "round-4",
        "--game-timeout", "2.5",
    ])

    assert args.workers == 3
    assert args.candidate_artifact == Path("candidate.json")
    assert args.candidate_identity == "round-4"
    assert args.game_timeout == 2.5
