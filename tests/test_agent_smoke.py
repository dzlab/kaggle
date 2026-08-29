import json
from pathlib import Path

import pytest

from scripts.run_local import run_episode


def _assert_replay_is_healthy(replay_path: Path) -> dict:
    replay = json.loads(replay_path.read_text())
    assert isinstance(replay, dict)
    assert replay["steps"]
    assert replay["statuses"] == ["DONE", "DONE"]

    for turn in replay["steps"]:
        assert isinstance(turn, list)
        for player_state in turn:
            action = player_state["action"]
            observation = player_state["observation"]
            assert isinstance(action, dict)
            assert set(action) == {"farmer", "hands", "market"}
            assert isinstance(action["farmer"], list)
            assert action["farmer"]
            assert isinstance(action["farmer"][0], str)
            assert isinstance(action["hands"], list)
            assert len(action["hands"]) == len(observation["farms"][observation["player"]]["hands"])
            for hand_action in action["hands"]:
                assert isinstance(hand_action, list)
                assert hand_action
                assert isinstance(hand_action[0], str)
            assert isinstance(action["market"], list)
            assert len(action["market"]) <= 10
            for order in action["market"]:
                assert isinstance(order, list)
                assert order
                assert isinstance(order[0], str)

            assert not player_state.get("error")
            if observation["player"] == 0:
                assert not player_state.get("info", {}).get("error")

    for player_state in replay["steps"][-1]:
        observation = player_state["observation"]
        player_farm = observation["farms"][observation["player"]]
        assert "money" in player_farm

    return replay


@pytest.mark.parametrize("opponent", ["pass", "random", "starter"])
def test_short_local_game_finishes_with_legal_replay(opponent: str, tmp_path: Path):
    replay_path = tmp_path / f"short-{opponent}.json"

    env = run_episode(opponent=opponent, seed=17, steps=96, replay_path=replay_path)

    assert env.toJSON()["statuses"] == ["DONE", "DONE"]
    _assert_replay_is_healthy(replay_path)


def test_full_seeded_local_game_finishes_with_legal_replay(tmp_path: Path):
    replay_path = tmp_path / "full-starter.json"

    env = run_episode(opponent="starter", seed=23, steps=720, replay_path=replay_path)

    assert env.toJSON()["statuses"] == ["DONE", "DONE"]
    replay = _assert_replay_is_healthy(replay_path)
    assert len(replay["steps"]) == 720
