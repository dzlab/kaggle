import json
import math

import pytest

try:
    from kaggle_environments import make
except ModuleNotFoundError:
    make = None

from scripts.run_local import run_episode


def _run_replay(tmp_path, *, steps=5):
    replay_path = tmp_path / "replay.json"
    run_episode(opponent="pass", seed=17, steps=steps, replay_path=replay_path)
    return json.loads(replay_path.read_text())


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_transitions_have_one_record_per_engine_step_and_terminal_reward(tmp_path):
    from kagriculture_agent.trajectory import transitions_from_replay

    replay = _run_replay(tmp_path, steps=5)
    transitions = transitions_from_replay(replay, candidate_player=0)

    assert len(transitions) == len(replay["steps"]) - 1
    assert [transition.done for transition in transitions] == [False, False, False, True]
    assert [transition.reward for transition in transitions[:-1]] == [0.0, 0.0, 0.0]
    candidate_bank = replay["steps"][-1][0]["observation"]["farms"][0]["money"]
    opponent_bank = replay["steps"][-1][1]["observation"]["farms"][1]["money"]
    assert transitions[-1].final_bank == candidate_bank
    assert transitions[-1].opponent_final_bank == opponent_bank
    assert transitions[-1].reward == pytest.approx(math.tanh((candidate_bank - opponent_bank) / 1000.0))


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_candidate_seat_selects_that_player_observations_and_bank(tmp_path):
    from kagriculture_agent.trajectory import transitions_from_replay

    replay = _run_replay(tmp_path, steps=4)
    transitions = transitions_from_replay(replay, candidate_player=1)

    assert transitions
    assert all(transition.observation["player"] == 1 for transition in transitions)
    assert all(transition.next_observation["player"] == 1 for transition in transitions)
    assert transitions[-1].final_bank == replay["steps"][-1][1]["observation"]["farms"][1]["money"]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_candidate_observations_do_not_gain_opponent_private_state(tmp_path):
    from kagriculture_agent.trajectory import transitions_from_replay

    replay = _run_replay(tmp_path, steps=4)
    for turn in replay["steps"]:
        opponent = next(state for state in turn if state["observation"]["player"] == 1)
        opponent["observation"]["private"]["opponent_secret"] = {
            "shed": {"OPPONENT_SECRET": 999},
            "inventory": {"OPPONENT_SECRET": 888},
        }

    transitions = transitions_from_replay(replay, candidate_player=0)

    for transition in transitions:
        assert "opponent_secret" not in transition.observation["private"]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_malformed_replay_raises_structured_value_error(tmp_path):
    from kagriculture_agent.trajectory import ReplayValidationError, transitions_from_replay

    replay = _run_replay(tmp_path, steps=4)
    replay["steps"].pop()

    with pytest.raises(ReplayValidationError) as error:
        transitions_from_replay(replay, candidate_player=0)

    assert isinstance(error.value, ValueError)
    assert error.value.code == "malformed_replay"
    assert error.value.details["reason"] == "replay_validation_failed"


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_requested_seed_mismatch_is_rejected(tmp_path):
    from kagriculture_agent.trajectory import ReplayValidationError, transitions_from_replay

    replay = _run_replay(tmp_path, steps=4)

    with pytest.raises(ReplayValidationError) as error:
        transitions_from_replay(replay, candidate_player=0, requested_seed=18)

    assert error.value.code == "seed_mismatch"
    assert error.value.details == {"requested_seed": 18, "replay_seed": 17}


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_transition_serialization_is_deterministic_and_json_compatible(tmp_path):
    from kagriculture_agent.trajectory import transitions_from_replay

    replay = _run_replay(tmp_path, steps=4)
    transition = transitions_from_replay(replay, candidate_player=0)[0]

    first = transition.to_json()
    second = transition.to_json()
    assert first == second
    assert json.loads(first) == transition.to_dict()
    json.dumps(transition.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))


def test_local_runner_exposes_current_opponent_and_candidate_seat():
    from scripts.run_local import OPPONENTS, _parser

    args = _parser().parse_args(["--opponent", "pass", "--seat", "1", "--steps", "1"])

    assert OPPONENTS == ("pass", "random", "starter")
    assert args.opponent == "pass"
    assert args.candidate_player == 1


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_collector_runs_all_supported_opponents_in_isolated_processes(tmp_path):
    from scripts import collect_trajectories

    output = tmp_path / "all-opponents.jsonl"
    manifest = collect_trajectories.collect(
        seeds=[0], opponents=["pass", "random", "starter", "current"],
        seats=[0], steps=4, output=output,
    )

    lines = output.read_text().splitlines()
    assert len(lines) == 4 * 3
    assert all(json.loads(line)["observation"]["player"] == 0 for line in lines)
    assert sum(json.loads(line)["done"] for line in lines) == 4
    assert manifest["opponents"] == ["pass", "random", "starter", "current"]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_collector_emits_default_length_trajectory_and_manifest(tmp_path):
    from scripts import collect_trajectories

    output = tmp_path / "default-length.jsonl"
    manifest = collect_trajectories.collect(
        seeds=[17], opponents=["pass"], seats=[0], steps=720, output=output,
    )

    lines = output.read_text().splitlines()
    assert len(lines) == 719
    assert json.loads(lines[-1])["done"] is True
    assert output.with_suffix(".manifest.json").exists()
    assert manifest["steps"] == 720


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_collector_writes_valid_transition_lines_and_manifest(tmp_path, monkeypatch):
    from scripts import collect_trajectories

    def fake_isolated_game(*, opponent, seed, steps, candidate_player, replay_path):
        run_episode(
            opponent="pass", seed=seed, steps=4,
            candidate_player=candidate_player, replay_path=replay_path,
        )
        return json.loads(replay_path.read_text())

    monkeypatch.setattr(collect_trajectories, "_run_game_isolated", fake_isolated_game)
    output = tmp_path / "trajectories.jsonl"

    manifest = collect_trajectories.collect(
        seeds=[3], opponents=["random"], seats=[1], steps=4,
        output=output, source_policy_identity="current",
    )

    lines = output.read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["observation"]["player"] == 1 for line in lines)
    assert json.loads(lines[-1])["done"] is True
    assert manifest["engine_version"] == "1.32.7"
    assert manifest["feature_schema_version"] == 1
    assert manifest["transition_schema_version"] == 1
    assert manifest["source_policy_identity"] == "current"
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == manifest


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_collector_rejects_replay_relabelled_with_a_different_seed(tmp_path, monkeypatch):
    from scripts import collect_trajectories

    def fake_isolated_game(*, opponent, seed, steps, candidate_player, replay_path):
        replay = _run_replay(tmp_path, steps=4)
        replay["info"]["seed"] = seed + 1
        return replay

    monkeypatch.setattr(collect_trajectories, "_run_game_isolated", fake_isolated_game)
    output = tmp_path / "mismatched.jsonl"

    with pytest.raises(ValueError, match="seed_mismatch"):
        collect_trajectories.collect(
            seeds=[17], opponents=["pass"], seats=[0], steps=4, output=output,
        )

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
