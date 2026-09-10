import json
from pathlib import Path

import pytest

try:
    from kaggle_environments import make
except ModuleNotFoundError:
    make = None


def _simulator_parity_agent(obs):
    step = obs.get("step", 0)
    player = obs.get("player", 0)
    farm = obs["farms"][player]
    private = obs["private"]
    hands = farm.get("hands", [])
    farmer_action = ["PASS"]
    hand_actions = [["PASS"] for _ in hands]
    market = []

    if step == 0:
        market = [
            ["HIRE"],
            ["BUY_LAND"],
            ["BUY_SEED", "WHEAT", 2],
            ["BUY_PRODUCT", "WHEAT", 2],
            ["BUY_ANIMAL", "GOOSE", 1],
            ["SELL", "WHEAT", 1],
        ]
    elif step == 1:
        farmer_action = ["PLANT", "WHEAT"]
        hand_actions = [["PLANT", "WHEAT"] for _ in hands]
    elif step == 2:
        farmer_action = ["WATER"]
        hand_actions = [["EAST"] for _ in hands]
    elif step == 3:
        hand_actions = [["PLACE", "WHEAT", 1] for _ in hands]
    elif step == 23:
        farmer_action = ["PASS"]
    elif step == 24:
        assert hands == []
        market = [["SELL", "WHEAT", private["shed"].get("WHEAT", 0)]]
    elif step == 49:
        farmer_action = ["HARVEST"]

    return {"farmer": farmer_action, "hands": hand_actions, "market": market}


def _public_parity_snapshot(player_state):
    observation = player_state["observation"]
    return {
        "step": observation.get("step"),
        "day": observation["day"],
        "hour": observation["hour"],
        "farms": observation["farms"],
        "market": observation["market"],
        "town": observation["town"],
        "status": player_state["status"],
        "reward": player_state["reward"],
    }


def _private_parity_snapshot(player_state):
    observation = player_state["observation"]
    farm = observation["farms"][observation["player"]]
    return {
        "private": observation["private"],
        "cash": farm["money"],
        "farmer": farm["farmer"],
        "hands": farm["hands"],
        "status": player_state["status"],
        "reward": player_state["reward"],
    }


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
    assert result["policy_inference_valid_samples"] == 4
    assert result["policy_inference_latency_valid"] is True
    assert result["benchmark_valid"] is True


@pytest.mark.parametrize("latencies", [[], [float("nan")], [float("inf"), float("-inf")]])
def test_rollout_metrics_mark_missing_or_non_finite_latency_samples_invalid(latencies):
    from scripts.benchmark_rollouts import summarize_run, real_engine_gate_passed

    result = summarize_run(
        worker_count=4,
        game_count=2,
        environment_steps=4000,
        rollout_seconds=1.0,
        inference_latencies_ms=latencies,
    )

    assert result["policy_inference_valid_samples"] == 0
    assert result["policy_inference_invalid_samples"] == len(latencies)
    assert result["policy_inference_latency_valid"] is False
    assert result["policy_inference_ms_per_turn"] is None
    assert result["policy_inference_p95_ms"] is None
    assert result["benchmark_valid"] is False
    assert real_engine_gate_passed([result]) is False


def test_rollout_metrics_mark_mixed_invalid_latency_samples_invalid():
    from scripts.benchmark_rollouts import summarize_run, real_engine_gate_passed

    result = summarize_run(
        worker_count=4,
        game_count=2,
        environment_steps=4000,
        rollout_seconds=1.0,
        inference_latencies_ms=[1.0, float("nan")],
    )

    assert result["policy_inference_valid_samples"] == 1
    assert result["policy_inference_invalid_samples"] == 1
    assert result["policy_inference_latency_valid"] is False
    assert result["policy_inference_ms_per_turn"] == pytest.approx(1.0)
    assert result["policy_inference_p95_ms"] == pytest.approx(1.0)
    assert result["benchmark_valid"] is False
    assert real_engine_gate_passed([result]) is False


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {
                "workers": 4,
                "environment_steps_per_minute": 100_000.0,
                "policy_inference_ms_per_turn": 9.99,
                "policy_inference_p95_ms": 9.99,
            },
            True,
        ),
        (
            {
                "workers": 4,
                "environment_steps_per_minute": 99_999.0,
                "policy_inference_ms_per_turn": 1.0,
                "policy_inference_p95_ms": 1.0,
            },
            False,
        ),
        (
            {
                "workers": 4,
                "environment_steps_per_minute": 150_000.0,
                "policy_inference_ms_per_turn": 1.0,
                "policy_inference_p95_ms": 10.0,
            },
            False,
        ),
        (
            {
                "workers": 2,
                "environment_steps_per_minute": 150_000.0,
                "policy_inference_ms_per_turn": 1.0,
                "policy_inference_p95_ms": 1.0,
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
        return {
            "steps": [
                [{"observation": {"player": candidate_player}, "status": "ACTIVE"}, {"status": "ACTIVE"}],
                [{"observation": {"player": candidate_player}, "status": "DONE"}, {"status": "DONE"}],
            ],
            "statuses": ["DONE", "DONE"],
        }

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
    assert result["successful_games"] == 3
    assert result["failed_games"] == 0


def test_benchmark_passes_candidate_artifact_to_game_and_latency_sampler(tmp_path):
    from scripts import benchmark_rollouts

    candidate = tmp_path / "stage.json"
    candidate.write_text('{"workers": [], "market_orders": []}', encoding="utf-8")
    game_artifacts = []
    latency_artifacts = []

    def fake_runner(*, candidate_artifact, **kwargs):
        game_artifacts.append(Path(candidate_artifact))
        return {
            "steps": [
                [{"status": "ACTIVE"}, {"status": "ACTIVE"}],
                [{"status": "DONE"}, {"status": "DONE"}],
            ],
            "statuses": ["DONE", "DONE"],
        }

    def fake_latency(replays, *, candidate_artifact, **kwargs):
        latency_artifacts.append(Path(candidate_artifact))
        return [1.0]

    benchmark_rollouts.benchmark_worker_count(
        games=1,
        steps=4,
        worker_count=1,
        output_dir=tmp_path / "rollouts",
        candidate_artifact=candidate,
        game_runner=fake_runner,
        latency_sampler=fake_latency,
        clock=benchmark_rollouts.SequenceClock([10.0, 11.0]),
    )

    assert game_artifacts == [candidate]
    assert latency_artifacts == [candidate]


def test_benchmark_cli_requires_explicit_cpu_and_candidate_artifact_flags():
    from scripts.benchmark_rollouts import _parser

    args = _parser().parse_args(["--cpu", "--candidate-artifact", "stage.json"])

    assert args.cpu is True
    assert args.candidate_artifact == Path("stage.json")


def test_benchmark_counts_only_successful_terminal_games_for_throughput(tmp_path):
    from scripts import benchmark_rollouts

    def fake_runner(*, opponent, seed, steps, candidate_player, replay_path, timeout):
        if seed == 8:
            return {
                "steps": [
                    [{"status": "ACTIVE"}, {"status": "ACTIVE"}],
                    [{"status": "ERROR"}, {"status": "DONE"}],
                ],
                "statuses": ["ERROR", "DONE"],
                "info": {},
            }
        return {
            "steps": [
                [{"status": "ACTIVE"}, {"status": "ACTIVE"}],
                [{"status": "DONE"}, {"status": "DONE"}],
                [{"status": "DONE"}, {"status": "DONE"}],
            ],
            "statuses": ["DONE", "DONE"],
            "info": {},
        }

    result = benchmark_rollouts.benchmark_worker_count(
        games=3,
        steps=20,
        worker_count=4,
        start_seed=7,
        output_dir=tmp_path,
        game_runner=fake_runner,
        latency_sampler=lambda replays, policy_factory=None, clock=None: [1.0],
        clock=benchmark_rollouts.SequenceClock([10.0, 11.0]),
    )

    assert result["successful_games"] == 2
    assert result["failed_games"] == 1
    assert result["environment_steps"] == 4
    assert result["games_per_hour"] == pytest.approx(7200.0)
    assert result["benchmark_valid"] is False
    assert result["failure_counts"] == {"ERROR": 1}
    assert benchmark_rollouts.real_engine_gate_passed([result]) is False


def test_benchmark_records_runner_failures_without_counting_steps(tmp_path):
    from scripts import benchmark_rollouts

    def fake_runner(*, opponent, seed, steps, candidate_player, replay_path, timeout):
        if seed == 8:
            raise RuntimeError("boom")
        return {
            "steps": [
                [{"status": "ACTIVE"}, {"status": "ACTIVE"}],
                [{"status": "DONE"}, {"status": "DONE"}],
            ],
            "statuses": ["DONE", "DONE"],
            "info": {},
        }

    result = benchmark_rollouts.benchmark_worker_count(
        games=2,
        steps=20,
        worker_count=4,
        start_seed=7,
        output_dir=tmp_path,
        game_runner=fake_runner,
        latency_sampler=lambda replays, policy_factory=None, clock=None: [1.0],
        clock=benchmark_rollouts.SequenceClock([10.0, 11.0]),
    )

    assert result["successful_games"] == 1
    assert result["failed_games"] == 1
    assert result["environment_steps"] == 1
    assert result["failure_counts"] == {"RUNNER_ERROR": 1}
    assert result["benchmark_valid"] is False
    assert benchmark_rollouts.real_engine_gate_passed([result]) is False


@pytest.mark.parametrize("bad_replay", [None, [], {"steps": []}])
def test_benchmark_classifies_malformed_runner_results_as_failed_games(tmp_path, bad_replay):
    from scripts import benchmark_rollouts

    def fake_runner(*, opponent, seed, steps, candidate_player, replay_path, timeout):
        return bad_replay

    result = benchmark_rollouts.benchmark_worker_count(
        games=1,
        steps=20,
        worker_count=4,
        output_dir=tmp_path,
        game_runner=fake_runner,
        latency_sampler=lambda replays, policy_factory=None, clock=None: [1.0],
        clock=benchmark_rollouts.SequenceClock([10.0, 11.0]),
    )

    assert result["successful_games"] == 0
    assert result["failed_games"] == 1
    assert result["environment_steps"] == 0
    assert result["failure_counts"] == {"RUNNER_ERROR": 1}
    assert result["benchmark_valid"] is False
    assert benchmark_rollouts.real_engine_gate_passed([result]) is False


@pytest.mark.parametrize(
    "replay,reason",
    [
        ({"steps": [[{"status": "DONE"}, {"status": "DONE"}]], "statuses": []}, "NON_TERMINAL"),
        ({"steps": [[{"status": "DONE"}, {"status": "DONE"}]], "statuses": "DONE"}, "NON_TERMINAL"),
        (
            {
                "steps": [
                    [{"status": "ACTIVE"}, {"status": "ACTIVE"}],
                    [{"status": "ACTIVE"}, {"status": "ACTIVE"}],
                ],
                "statuses": ["DONE", "DONE"],
            },
            "NON_TERMINAL",
        ),
    ],
)
def test_benchmark_requires_valid_terminal_statuses_for_successful_games(tmp_path, replay, reason):
    from scripts import benchmark_rollouts

    def fake_runner(*, opponent, seed, steps, candidate_player, replay_path, timeout):
        return replay

    result = benchmark_rollouts.benchmark_worker_count(
        games=1,
        steps=20,
        worker_count=4,
        output_dir=tmp_path,
        game_runner=fake_runner,
        latency_sampler=lambda replays, policy_factory=None, clock=None: [1.0],
        clock=benchmark_rollouts.SequenceClock([10.0, 11.0]),
    )

    assert result["successful_games"] == 0
    assert result["failed_games"] == 1
    assert result["failure_counts"] == {reason: 1}
    assert result["benchmark_valid"] is False
    assert benchmark_rollouts.real_engine_gate_passed([result]) is False


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
    assert report["gate"]["inference_p95_ms_threshold"] == 10.0
    assert [result["workers"] for result in report["results"]] == [1, 4]
    assert report["results"][1]["policy_inference_ms_per_turn"] == pytest.approx(2.0)
    json.dumps(report, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("argv", [["--games", "0"], ["--steps", "0"], ["--workers", "0"], ["--workers", "1", "1"]])
def test_parser_rejects_non_positive_or_duplicate_counts(argv):
    from scripts.benchmark_rollouts import _parser

    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_simulator_replays_recorded_actions_with_real_engine_parity_for_required_seeds():
    from kagriculture_agent.simulator import (
        REQUIRED_PARITY_SEEDS,
        KaggricultureSimulator,
        recorded_actions_from_replay,
        simulator_parity_status,
    )

    mismatches = []
    for seed in REQUIRED_PARITY_SEEDS:
        configuration = {"episodeSteps": 72, "seed": seed, "weedSpawnChance": 0.2}
        real_env = make("kaggriculture", configuration=configuration)
        real_env.run([_simulator_parity_agent, _simulator_parity_agent])
        real_replay = real_env.toJSON()

        simulator = KaggricultureSimulator(configuration=configuration, seed=seed)
        simulated_replay = simulator.replay(recorded_actions_from_replay(real_replay))

        assert simulated_replay["configuration"] == real_replay["configuration"]
        assert simulated_replay["info"]["seed"] == real_replay["info"]["seed"] == seed
        assert simulated_replay["statuses"] == real_replay["statuses"] == ["DONE", "DONE"]
        assert simulated_replay["rewards"] == real_replay["rewards"]

        for step, (real_turn, simulated_turn) in enumerate(zip(real_replay["steps"], simulated_replay["steps"])):
            for player in (0, 1):
                if _public_parity_snapshot(simulated_turn[player]) != _public_parity_snapshot(real_turn[player]):
                    mismatches.append((seed, step, player, "public"))
                if _private_parity_snapshot(simulated_turn[player]) != _private_parity_snapshot(real_turn[player]):
                    mismatches.append((seed, step, player, "private"))

    assert mismatches == []
    assert simulator_parity_status({seed: True for seed in REQUIRED_PARITY_SEEDS})["promotion_ready"] is True


def test_simulator_readiness_gate_requires_all_ten_passing_parity_seeds():
    from kagriculture_agent.simulator import REQUIRED_PARITY_SEEDS, simulator_parity_status

    assert simulator_parity_status({seed: True for seed in REQUIRED_PARITY_SEEDS[:-1]})["promotion_ready"] is False
    assert simulator_parity_status({seed: True for seed in REQUIRED_PARITY_SEEDS[:-1]})["missing_seeds"] == [9]
    failed_results = {seed: True for seed in REQUIRED_PARITY_SEEDS[:-1]}
    failed_results[9] = False
    assert simulator_parity_status(failed_results)["promotion_ready"] is False


@pytest.mark.parametrize("invalid_result", ["false", 0, 1])
def test_simulator_readiness_gate_rejects_non_boolean_parity_results(invalid_result):
    from kagriculture_agent.simulator import REQUIRED_PARITY_SEEDS, simulator_parity_status

    seed_results = {seed: True for seed in REQUIRED_PARITY_SEEDS}
    seed_results[0] = invalid_result

    status = simulator_parity_status(seed_results)

    assert status["promotion_ready"] is False
    assert status["invalid_seeds"] == [0]
    assert status["missing_seeds"] == []
    assert 0 not in status["passed_seeds"]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("bad_action", ["bad", RuntimeError("boom")])
def test_simulator_replay_preserves_public_step_malformed_action_status_parity(bad_action):
    from kagriculture_agent.simulator import KaggricultureSimulator

    configuration = {"episodeSteps": 4, "seed": 0}
    valid_action = {"farmer": ["PASS"], "hands": [], "market": []}
    real_env = make("kaggriculture", configuration=configuration, debug=True)
    real_env.step([bad_action, valid_action])
    real_replay = real_env.toJSON()

    simulator = KaggricultureSimulator(configuration=configuration, seed=0, debug=True)
    simulated_replay = simulator.replay([[bad_action, valid_action]])

    assert simulated_replay["statuses"] == real_replay["statuses"]
    assert [state["status"] for state in simulated_replay["steps"][-1]] == [
        state["status"] for state in real_replay["steps"][-1]
    ]
    assert [state.get("action") for state in simulated_replay["steps"][-1]] == [
        state.get("action") for state in real_replay["steps"][-1]
    ]
