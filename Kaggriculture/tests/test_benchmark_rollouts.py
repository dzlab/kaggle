import json

import pytest


def test_rollout_metrics_report_required_units():
    from scripts.benchmark_rollouts import summarize_run

    result = summarize_run(
        worker_count=4,
        game_count=8,
        environment_steps=1440,
        rollout_seconds=12.0,
        inference_latencies_ms=[1.0, 2.0, 3.0, 100.0],
    )

    assert result["workers"] == 4
    assert result["games"] == 8
    assert result["environment_steps"] == 1440
    assert result["games_per_hour"] == pytest.approx(2400.0)
    assert result["environment_steps_per_second"] == pytest.approx(120.0)
    assert result["environment_steps_per_minute"] == pytest.approx(7200.0)
    assert result["policy_inference_ms_per_turn"] == pytest.approx(26.5)
    assert result["policy_inference_p95_ms"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {
                "workers": 4,
                "environment_steps_per_minute": 100_000.0,
                "policy_inference_ms_per_turn": 9.99,
            },
            True,
        ),
        (
            {
                "workers": 4,
                "environment_steps_per_minute": 99_999.0,
                "policy_inference_ms_per_turn": 1.0,
            },
            False,
        ),
        (
            {
                "workers": 4,
                "environment_steps_per_minute": 150_000.0,
                "policy_inference_ms_per_turn": 10.0,
            },
            False,
        ),
        (
            {
                "workers": 2,
                "environment_steps_per_minute": 150_000.0,
                "policy_inference_ms_per_turn": 1.0,
            },
            False,
        ),
    ],
)
def test_real_engine_gate_uses_four_worker_throughput_and_latency(result, expected):
    from scripts.benchmark_rollouts import real_engine_gate_passed

    assert real_engine_gate_passed([result]) is expected


def test_benchmark_uses_real_collector_runner_and_alternates_seats(tmp_path):
    from scripts import benchmark_rollouts

    calls = []

    def fake_runner(*, opponent, seed, steps, candidate_player, replay_path, timeout):
        calls.append((opponent, seed, steps, candidate_player, replay_path, timeout))
        return {"steps": [{"observation": {"player": candidate_player}}, {"observation": {"player": candidate_player}}]}

    result = benchmark_rollouts.benchmark_worker_count(
        games=3,
        steps=20,
        worker_count=1,
        start_seed=7,
        opponent="current",
        output_dir=tmp_path,
        game_runner=fake_runner,
        latency_sampler=lambda replays, policy_factory=None, clock=None: [0.5, 1.5],
        clock=benchmark_rollouts.SequenceClock([10.0, 12.0]),
    )

    assert [call[0] for call in calls] == ["current", "current", "current"]
    assert [call[1] for call in calls] == [7, 8, 9]
    assert [call[2] for call in calls] == [20, 20, 20]
    assert [call[3] for call in calls] == [0, 1, 0]
    assert all(call[4].parent == tmp_path for call in calls)
    assert all(call[5] == benchmark_rollouts.DEFAULT_GAME_TIMEOUT_SECONDS for call in calls)
    assert result["environment_steps"] == 3
    assert result["policy_inference_ms_per_turn"] == pytest.approx(1.0)


def test_json_report_includes_gate_status(tmp_path):
    from scripts import benchmark_rollouts

    def fake_benchmark(**kwargs):
        worker_count = kwargs["worker_count"]
        return {
            "workers": worker_count,
            "games": 2,
            "environment_steps": 2000,
            "rollout_seconds": 1.0,
            "games_per_hour": 7200.0,
            "environment_steps_per_second": 2000.0,
            "environment_steps_per_minute": 120_000.0,
            "policy_inference_ms_per_turn": 2.0,
            "policy_inference_p95_ms": 3.0,
        }

    report = benchmark_rollouts.run_benchmark(
        games=2,
        steps=10,
        workers=[1, 4],
        output_dir=tmp_path,
        benchmark_fn=fake_benchmark,
    )

    assert report["gate"]["real_engine_kept"] is True
    assert report["gate"]["throughput_steps_per_minute_threshold"] == 100_000
    assert report["gate"]["inference_ms_per_turn_threshold"] == 10.0
    assert [result["workers"] for result in report["results"]] == [1, 4]
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("argv", [["--games", "0"], ["--steps", "0"], ["--workers", "0"], ["--workers", "1", "1"]])
def test_parser_rejects_non_positive_or_duplicate_counts(argv):
    from scripts.benchmark_rollouts import _parser

    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)
