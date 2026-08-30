import builtins
import importlib.util
import json
import math
from pathlib import Path

import pytest

try:
    from kaggle_environments import make
except ModuleNotFoundError:
    make = None

from kagriculture_agent.constants import CROPS as DOMAIN_CROPS
from scripts.run_local import run_episode


CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
PRODUCTS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"}
ANIMALS = {"GOOSE", "COW", "SHEEP"}
UNIT_NO_ARGUMENTS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST",
    "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "DIG",
}
MARKET_NO_ARGUMENTS = {"HIRE", "BUY_LAND"}


def _assert_unit_command(command: list) -> None:
    assert isinstance(command, list) and command
    operation = command[0]
    assert isinstance(operation, str)
    if operation in UNIT_NO_ARGUMENTS:
        assert len(command) == 1
    elif operation == "PLANT":
        assert len(command) == 2 and command[1] in CROPS
    elif operation in {"PICKUP", "PLACE"}:
        assert len(command) in {2, 3}
        assert command[1] in PRODUCTS | ANIMALS
        if len(command) == 3:
            assert isinstance(command[2], int) and not isinstance(command[2], bool) and command[2] > 0
    else:
        pytest.fail(f"unknown unit operation: {operation!r}")


def _assert_market_order(order: list) -> None:
    assert isinstance(order, list) and order
    operation = order[0]
    assert isinstance(operation, str)
    if operation in MARKET_NO_ARGUMENTS:
        assert len(order) == 1
        return
    assert operation in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
    assert len(order) == 3
    if operation == "BUY_SEED":
        assert order[1] in CROPS
    elif operation == "BUY_PRODUCT":
        assert order[1] in {"WHEAT", "FERTILIZER"}
    elif operation == "BUY_ANIMAL":
        assert order[1] in ANIMALS
    else:
        assert order[1] in PRODUCTS
    assert isinstance(order[2], int) and not isinstance(order[2], bool) and order[2] > 0


def _assert_replay_is_legal_and_complete(replay_path: Path) -> dict:
    replay = json.loads(replay_path.read_text())
    assert isinstance(replay, dict)
    assert replay["steps"]
    assert replay["statuses"] == ["DONE", "DONE"]
    assert isinstance(replay["info"], dict)
    assert not replay["info"].get("error")

    final_turn = len(replay["steps"]) - 1
    for turn_number, turn in enumerate(replay["steps"]):
        assert isinstance(turn, list)
        for player_state in turn:
            action = player_state["action"]
            observation = player_state["observation"]
            # Kaggle's replay stores the action selected from the prior
            # observation on the following state record (the bootstrap
            # record is the sole same-state exception).
            action_observation = observation
            if turn_number:
                action_observation = next(
                    prior["observation"] for prior in replay["steps"][turn_number - 1]
                    if prior["observation"]["player"] == observation["player"]
                )
            assert isinstance(action, dict)
            assert set(action) == {"farmer", "hands", "market"}
            assert isinstance(action["farmer"], list)
            _assert_unit_command(action["farmer"])
            assert isinstance(action["hands"], list)
            assert len(action["hands"]) == len(action_observation["farms"][action_observation["player"]]["hands"])
            for hand_action in action["hands"]:
                _assert_unit_command(hand_action)
            assert isinstance(action["market"], list)
            assert len(action["market"]) <= 10
            for order in action["market"]:
                _assert_market_order(order)

            assert isinstance(player_state["status"], str)
            expected_status = "DONE" if turn_number == final_turn else "ACTIVE"
            assert player_state["status"] == expected_status
            assert not player_state.get("error")
            assert isinstance(player_state["info"], dict)
            assert not player_state["info"].get("error")

    for player_state in replay["steps"][-1]:
        observation = player_state["observation"]
        player_farm = observation["farms"][observation["player"]]
        money = player_farm.get("money")
        assert isinstance(money, (int, float)) and not isinstance(money, bool)
        assert math.isfinite(money)

    return replay


def _player_state(replay: dict, step: int, player: int = 0) -> dict:
    return next(state for state in replay["steps"][step] if state["observation"]["player"] == player)


def _scripted_crop_agent(obs: dict) -> dict:
    step = obs.get("step", 0)
    farm = obs["farms"][obs["player"]]
    x, y = farm["farmer"]
    tile = farm["tiles"][y][x]
    if step == 0:
        farmer = ["PASS"]
        market = [["BUY_SEED", "WHEAT", 1]]
    elif step >= 49 and isinstance(tile, dict) and tile.get("kind") == "PLANT":
        farmer = ["HARVEST"]
        market = []
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile["watered_today"]:
        farmer = ["WATER"]
        market = []
    elif tile is None and obs["private"]["seeds"].get("WHEAT", 0) > 0:
        farmer = ["PLANT", "WHEAT"]
        market = []
    else:
        farmer = ["PASS"]
        market = []
    return {"farmer": farmer, "hands": [], "market": market}


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("opponent", ["pass", "random", "starter"])
def test_short_local_game_finishes_with_legal_replay(opponent: str, tmp_path: Path):
    replay_path = tmp_path / f"short-{opponent}.json"

    env = run_episode(opponent=opponent, seed=17, steps=96, replay_path=replay_path)

    assert env.toJSON()["statuses"] == ["DONE", "DONE"]
    replay = _assert_replay_is_legal_and_complete(replay_path)
    assert len(replay["steps"]) == 96


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_full_seeded_local_game_finishes_with_legal_replay(tmp_path: Path):
    replay_path = tmp_path / "full-starter.json"

    env = run_episode(opponent="starter", seed=23, steps=720, replay_path=replay_path)

    assert env.toJSON()["statuses"] == ["DONE", "DONE"]
    replay = _assert_replay_is_legal_and_complete(replay_path)
    assert len(replay["steps"]) == 720


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_full_pass_exercises_autonomous_macro_action_and_market_flows(tmp_path: Path):
    from scripts.evaluate import replay_record

    replay_path = tmp_path / "full-pass-macro.json"

    run_episode(opponent="pass", seed=17, steps=720, replay_path=replay_path)
    replay = _assert_replay_is_legal_and_complete(replay_path)
    record = replay_record(replay, variant="mixed", opponent="pass", seed=17)
    assert record["framework_error"] is False
    assert record["missed_basic_needs"] == 0
    assert all(not inventory for inventory in replay["steps"][-1][0]["observation"]["private"]["inventories"])
    unit_operations = set()
    market_operations = set()
    for player_state in replay["steps"]:
        state = next(item for item in player_state if item["observation"]["player"] == 0)
        action = state["action"]
        unit_operations.add(action["farmer"][0])
        unit_operations.update(command[0] for command in action["hands"])
        market_operations.update(order[0] for order in action["market"])

    assert {"PLANT", "WATER", "HARVEST"} <= unit_operations
    assert unit_operations & {"BUILD_COOP", "BUILD_PASTURE"}
    assert {"FERTILIZE", "COLLECT_FERTILIZER"} <= unit_operations
    assert {"PLACE", "FEED", "CARE"} <= unit_operations
    assert "SELL" in market_operations


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("opponent", ["pass", "random", "starter"])
def test_seed17_terminal_liquidation_leaves_no_saleable_inventory(opponent: str, tmp_path: Path):
    from scripts.evaluate import replay_record

    replay_path = tmp_path / f"terminal-{opponent}.json"
    run_episode(opponent=opponent, seed=17, steps=720, replay_path=replay_path)
    replay = _assert_replay_is_legal_and_complete(replay_path)
    record = replay_record(replay, variant="mixed", opponent=opponent, seed=17)
    final = replay["steps"][-1][0]["observation"]
    private = final["private"]
    saleable = {item: quantity for item, quantity in private["shed"].items()
                if item in PRODUCTS and item != "FERTILIZER" and quantity}

    assert record["framework_error"] is False
    assert record["missed_basic_needs"] == 0
    assert all(not inventory for inventory in private["inventories"])
    assert saleable == {}


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_seeded_random_style_opponent_replays_are_reproducible(tmp_path: Path):
    first_path = tmp_path / "random-first.json"
    second_path = tmp_path / "random-second.json"

    run_episode(opponent="random", seed=17, steps=96, replay_path=first_path)
    run_episode(opponent="random", seed=17, steps=96, replay_path=second_path)

    first = json.loads(first_path.read_text())
    second = json.loads(second_path.read_text())
    assert first["steps"] == second["steps"]
    assert first["rewards"] == second["rewards"]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_replay_captures_purchase_plant_water_and_harvest_effects(tmp_path: Path):
    env = make("kaggriculture", configuration={"episodeSteps": 52, "seed": 31})
    env.run([_scripted_crop_agent, "pass"])
    effects_path = tmp_path / "effects.json"
    effects_path.write_text(json.dumps(env.toJSON()))
    replay = _assert_replay_is_legal_and_complete(effects_path)

    initial = _player_state(replay, 0)
    after_purchase = _player_state(replay, 1)
    after_plant = _player_state(replay, 2)
    after_water = _player_state(replay, 3)
    after_harvest = _player_state(replay, 50)
    initial_farm = initial["observation"]["farms"][0]
    purchased = after_purchase["observation"]
    planted = after_plant["observation"]["farms"][0]
    watered = after_water["observation"]["farms"][0]
    harvested = after_harvest["observation"]["private"]

    assert purchased["farms"][0]["money"] == initial_farm["money"] - 10
    assert purchased["private"]["seeds"]["WHEAT"] == 1
    assert planted["tiles"][4][4]["kind"] == "PLANT"
    assert watered["tiles"][4][4]["watered_today"] is True
    assert harvested["inventories"][0]["WHEAT"] > 0


def test_runner_module_imports_without_engine_dependency(monkeypatch):
    runner_path = Path(__file__).parents[1] / "scripts" / "run_local.py"
    spec = importlib.util.spec_from_file_location("run_local_without_engine", runner_path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def block_engine(name, *args, **kwargs):
        if name == "kaggle_environments":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_engine)
    spec.loader.exec_module(module)

    assert callable(module.main)


def test_parser_rejects_non_positive_steps():
    from scripts.run_local import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--steps", "0"])

    assert exc_info.value.code == 2


def _write_replay(path: Path, replay: dict) -> None:
    path.write_text(json.dumps(replay))


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("bad_status", ["ERROR", "INVALID", "TIMEOUT"])
def test_replay_validator_rejects_bad_status_on_intermediate_turn(tmp_path: Path, bad_status: str):
    replay_path = tmp_path / "valid.json"
    run_episode(opponent="pass", seed=17, steps=4, replay_path=replay_path)
    replay = json.loads(replay_path.read_text())
    replay["steps"][1][0]["status"] = bad_status
    _write_replay(replay_path, replay)

    with pytest.raises(AssertionError):
        _assert_replay_is_legal_and_complete(replay_path)


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_replay_validator_requires_done_status_only_on_final_turn(tmp_path: Path):
    replay_path = tmp_path / "valid.json"
    run_episode(opponent="pass", seed=17, steps=4, replay_path=replay_path)
    replay = json.loads(replay_path.read_text())
    replay["steps"][-1][0]["status"] = "ACTIVE"
    _write_replay(replay_path, replay)

    with pytest.raises(AssertionError):
        _assert_replay_is_legal_and_complete(replay_path)


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_replay_validator_requires_numeric_final_money(tmp_path: Path):
    replay_path = tmp_path / "valid.json"
    run_episode(opponent="pass", seed=17, steps=4, replay_path=replay_path)
    replay = json.loads(replay_path.read_text())
    replay["steps"][-1][0]["observation"]["farms"][0]["money"] = "unknown"
    _write_replay(replay_path, replay)

    with pytest.raises(AssertionError):
        _assert_replay_is_legal_and_complete(replay_path)


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_production_agent_changes_state_and_reports_matching_reward(tmp_path: Path):
    replay_path = tmp_path / "production-agent.json"
    run_episode(opponent="pass", seed=17, steps=96, replay_path=replay_path)
    replay = _assert_replay_is_legal_and_complete(replay_path)
    custom_states = [_player_state(replay, step) for step in range(96)]

    buy_step = next(step for step, state in enumerate(custom_states[1:], start=1) if state["action"]["market"])
    buy_before = custom_states[buy_step - 1]["observation"]
    buy_after = custom_states[buy_step]["observation"]
    buy_orders = [order for order in custom_states[buy_step]["action"]["market"] if order[0] == "BUY_SEED"]
    assert buy_orders
    for buy_order in buy_orders:
        assert buy_after["private"]["seeds"][buy_order[1]] == buy_before["private"]["seeds"][buy_order[1]] + buy_order[2]
    assert buy_after["farms"][0]["money"] < buy_before["farms"][0]["money"]

    plant_step = next(step for step, state in enumerate(custom_states[1:], start=1) if state["action"]["farmer"][0] == "PLANT")
    plant_before = custom_states[plant_step - 1]["observation"]
    plant_after = custom_states[plant_step]["observation"]
    x, y = plant_before["farms"][0]["farmer"]
    assert plant_before["farms"][0]["tiles"][y][x] is None
    planted_tile = plant_after["farms"][0]["tiles"][y][x]
    assert planted_tile["kind"] == "PLANT"
    assert planted_tile["crop"] == custom_states[plant_step]["action"]["farmer"][1]

    final_money = custom_states[-1]["observation"]["farms"][0]["money"]
    reward = replay["rewards"][0]
    assert all(not state.get("error") and not state["info"].get("error") for state in custom_states)
    assert isinstance(reward, (int, float)) and not isinstance(reward, bool)
    assert math.isfinite(reward)
    assert reward == final_money
