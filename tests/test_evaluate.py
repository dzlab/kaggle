import json
from pathlib import Path

import pytest

try:
    from kaggle_environments import make
except ModuleNotFoundError:
    make = None


def _engine_envelope(replay, seed=1):
    from kagriculture_agent.constants import ENGINE_VERSION

    replay.update({
        "id": "test-replay",
        "name": "kaggriculture",
        "version": "0.1.0",
        "module_version": ENGINE_VERSION,
        "schema_version": 1,
        "title": "Kaggriculture",
        "description": "test",
        "info": {"seed": seed},
        "configuration": {
            "episodeSteps": 720, "seed": None, "actTimeout": 1, "runTimeout": 1200,
            "boardSize": 10, "startingMoney": 3000, "maxMarketOrdersPerTurn": 10,
            "turnsPerDay": 24, "shedCapacity": 100, "weedSpawnChance": 0.005,
            "townShopUnlockInterval": 3, "townShopSellInterval": 4,
            "townCenterSellInterval": 24, "farmHandCostMult": 1, "marketParams": {},
        },
        "specification": {"action": {}, "agents": [2], "configuration": {}},
    })
    return replay


def test_cli_parses_seed_opponent_variant_and_quick_options():
    from scripts.evaluate import parse_args

    args = parse_args([
        "--seeds", "3",
        "--start-seed", "40",
        "--opponents", "pass", "starter",
        "--steps", "12",
        "--output", "results.json",
        "--variants", "mixed", "melon-heavy",
        "--quick",
    ])

    assert args.seeds == 3
    assert args.start_seed == 40
    assert args.opponents == ["pass", "starter"]
    assert args.steps == 12
    assert args.output == Path("results.json")
    assert args.variants == ["mixed", "melon-heavy"]
    assert args.quick is True


def test_cli_default_output_is_under_reports():
    from scripts.evaluate import parse_args

    assert parse_args([]).output == Path("reports/evaluation.json")


def test_percentile_uses_linear_interpolation():
    from scripts.evaluate import percentile

    assert percentile([10, 20, 30, 40, 50], 5) == 12.0
    assert percentile([10, 20, 30, 40, 50], 50) == 30.0
    assert percentile([], 5) is None


def test_aggregate_counts_outcomes_and_metrics():
    from scripts.evaluate import aggregate_records

    records = [
        {"outcome": "win", "final_bank": 110, "opponent_final_bank": 90, "framework_error": False,
         "shed_overflow": 2, "price_floor_sales": 4, "missed_basic_needs": 1},
        {"outcome": "framework_error", "final_bank": None, "opponent_final_bank": None, "framework_error": True,
         "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 3},
        {"outcome": "tie", "final_bank": 95, "opponent_final_bank": 95, "framework_error": False,
         "shed_overflow": 1, "price_floor_sales": 2, "missed_basic_needs": 0},
    ]

    summary = aggregate_records(records)

    assert summary["count"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["ties"] == 1
    assert summary["framework_failures"] == 1
    assert summary["valid_count"] == 2
    assert summary["win_rate"] == pytest.approx(1 / 2)
    assert summary["mean_final_bank"] == pytest.approx(102.5)
    assert summary["median_final_bank"] == pytest.approx(102.5)
    assert summary["fifth_percentile_final_bank"] == pytest.approx(95.75)
    assert summary["mean_bank_differential"] == pytest.approx(0)
    assert summary["framework_error_rate"] == pytest.approx(1 / 3)
    assert summary["average_shed_overflow"] == pytest.approx(1)
    assert summary["average_price_floor_sales"] == pytest.approx(2)
    assert summary["average_missed_basic_needs_events"] == pytest.approx(4 / 3)


def test_run_matrix_executes_variant_opponent_cartesian_product_with_same_seeds(monkeypatch):
    from scripts.evaluate import run_matrix

    calls = []

    def fake_run_game(*, variant, opponent, seed, steps):
        calls.append((variant, opponent, seed, steps))
        return {
            "variant": variant,
            "opponent": opponent,
            "seed": seed,
            "outcome": "tie",
            "final_bank": 10,
            "opponent_final_bank": 10,
            "framework_error": False,
            "shed_overflow": 0,
            "price_floor_sales": 0,
            "missed_basic_needs": 0,
        }

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    result = run_matrix(
        variants=["mixed", "melon-heavy"],
        opponents=["pass", "starter"],
        seeds=[4, 9],
        steps=16,
    )

    assert len(result["records"]) == 8
    assert calls == [
        (variant, opponent, seed, 16)
        for variant in ["mixed", "melon-heavy"]
        for opponent in ["pass", "starter"]
        for seed in [4, 9]
    ]


def test_result_schema_is_json_serializable_and_has_selected_default():
    from scripts.evaluate import build_result_document

    document = build_result_document(
        config={"seeds": 1, "start_seed": 7, "steps": 4, "opponents": ["pass"], "variants": ["mixed"]},
        records=[{
            "variant": "mixed", "opponent": "pass", "seed": 7, "outcome": "win",
            "final_bank": 100, "opponent_final_bank": 50, "framework_error": False,
            "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 0,
        }],
    )

    assert document["schema_version"] == 1
    assert document["metadata"]["config"]["variants"] == ["mixed"]
    assert document["selected_default"] == "mixed"
    assert document["results"]["mixed"]["pass"]["wins"] == 1
    json.dumps(document, sort_keys=True)


def test_default_selection_breaks_ties_by_median_then_failure_rate():
    from scripts.evaluate import build_result_document

    records = [
        {"variant": "mixed", "opponent": "pass", "outcome": "win", "final_bank": 100,
         "opponent_final_bank": 50, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "pass", "outcome": "win", "final_bank": 110,
         "opponent_final_bank": 50, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "mixed", "opponent": "starter", "outcome": "loss", "final_bank": 90,
         "opponent_final_bank": 100, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "starter", "outcome": "loss", "final_bank": 80,
         "opponent_final_bank": 100, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "starter", "outcome": "framework_error", "final_bank": None,
         "opponent_final_bank": None, "framework_error": True, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
    ]

    document = build_result_document(
        config={"opponents": ["pass", "starter"], "variants": ["mixed", "melon-heavy"]},
        records=records,
    )

    assert document["selected_default"] == "mixed"


def test_isolated_ablation_execution_keeps_each_toggle_separate(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_game(*, variant, opponent, seed, steps, ablations=None):
        calls.append((variant, opponent, seed, dict(ablations or {})))
        return {
            "variant": variant, "opponent": opponent, "seed": seed, "outcome": "tie",
            "final_bank": 10, "opponent_final_bank": 10, "framework_error": False,
            "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 0,
        }

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    result = run_evaluation(
        variants=["mixed"], opponents=["pass"], seeds=[1], steps=4,
        ablations=[("animals", False), ("land_purchase", False)],
    )

    assert [call[3] for call in calls] == [
        {"animals": True, "land_purchase": True, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": False, "land_purchase": True, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": True, "land_purchase": False, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
    ]
    assert set(result["ablation_records"]) == {"animals", "land_purchase"}


def test_write_result_document_is_byte_stable(tmp_path):
    from scripts.evaluate import write_result_document

    document = {"schema_version": 1, "metadata": {"config": {"x": 1}}, "selected_default": "mixed", "results": {}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_result_document(first, document, records=[])
    write_result_document(second, document, records=[])

    assert first.read_bytes() == second.read_bytes()
    assert first.with_name("first.replays.json").read_bytes() == second.with_name("second.replays.json").read_bytes()


def test_no_network_dependency_for_import_and_aggregation(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def block_network(name, *args, **kwargs):
        if name in {"requests", "urllib3", "httpx"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_network)
    from scripts.evaluate import aggregate_records

    assert aggregate_records([])["count"] == 0


def test_replay_record_extracts_outcome_and_replay_metrics():
    from scripts.evaluate import replay_record

    observation = {
        "player": 0,
        "step": 23,
        "hour": 23,
        "farms": [{"money": 120, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "watered_today": False}] + [None for _ in range(9)]] + [[None for _ in range(10)] for _ in range(9)]}],
        "private": {"shed": {"WHEAT": 102}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 1}, "inventory": {"WHEAT": 10000}},
    }
    replay = {
        "steps": [[
                {"observation": observation, "action": {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 3]]}, "status": "DONE", "info": {}},
                {"observation": {"player": 1, "farms": [{"money": 120}, {"money": 80, "farmer": [0, 0], "hands": [], "tiles": [[None for _ in range(10)] for _ in range(10)]}], "private": {"inventories": [{}]}, "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}}}, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        ]],
        "rewards": [120, 80],
        "statuses": ["DONE", "DONE"],
        "info": {},
    }

    record = replay_record(_engine_envelope(replay, seed=5), variant="mixed", opponent="pass", seed=5)

    assert record["outcome"] == "win"
    assert record["bank_differential"] == 40
    assert record["shed_overflow"] == 2
    assert record["price_floor_sales"] == 3
    # A terminal duplicate hour-23 snapshot is not a day boundary.
    assert record["missed_basic_needs"] == 0


def test_price_floor_sales_uses_sequential_per_unit_quotes():
    from scripts.evaluate import _price_floor_sales

    inventory = 1_717_032_651_332
    observation = {
        "market": {"inventory": {"WHEAT": inventory}, "prices": {"WHEAT": 2}},
        "private": {"shed": {"WHEAT": 2}},
    }
    state = {"action": {"market": [["SELL", "WHEAT", 2]]}}

    assert _price_floor_sales(state, observation) == 1


def test_shed_overflow_uses_pre_transition_shed_and_worker_inventory():
    from scripts.evaluate import replay_record

    before = {
        "player": 0, "step": 23, "hour": 23,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                   "unlocked_quadrants": ["NW"]}],
        "private": {"shed": {"WHEAT": 100}, "inventories": [{"WHEAT": 3}], "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    after = {
        "player": 0, "step": 24, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                   "unlocked_quadrants": ["NW"]}],
        "private": {"shed": {"WHEAT": 100}, "inventories": [{}], "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    other = {
        "player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                                  "unlocked_quadrants": ["NW"]}],
        "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    replay = {"steps": [
        [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
        [{"observation": after, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
    ], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is False
    assert record["shed_overflow"] == 3


def test_terminal_hour_23_snapshot_is_not_a_missed_needs_boundary():
    from scripts.evaluate import replay_record

    plant = {"kind": "PLANT", "watered_today": False}
    before = {"player": 0, "step": 23, "hour": 23,
              "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[plant]], "unlocked_quadrants": ["NW"]}],
              "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    after = {"player": 0, "step": 24, "hour": 23,
             "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[plant]], "unlocked_quadrants": ["NW"]}],
             "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    other = {"player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]}],
             "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    replay = {"steps": [
        [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
        [{"observation": after, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
    ], "statuses": ["DONE", "DONE"], "info": {}}

    assert replay_record(replay, variant="mixed", opponent="pass", seed=1)["missed_basic_needs"] == 0


@pytest.mark.parametrize("unlocked_quadrants", [None, {"NW": True}])
def test_malformed_unlocked_quadrants_is_a_framework_failure(unlocked_quadrants):
    from scripts.evaluate import replay_record

    observation = {"player": 0, "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                                                "unlocked_quadrants": unlocked_quadrants}],
                   "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    other = {"player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                                      "unlocked_quadrants": ["NW"]}],
             "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    replay = {"steps": [[
        {"observation": observation, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
    ]], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_actions_are_validated_against_the_preceding_observation():
    from scripts.evaluate import replay_record

    initial = {
        "player": 0, "step": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    post_plant = {
        "player": 0, "step": 1, "hour": 1,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "yield_units": 0}]]}],
        "private": {"seeds": {"WHEAT": 0}, "shed": {}, "inventories": [{}]},
        "market": initial["market"],
    }
    other_initial = {
        "player": 1, "step": 0, "hour": 0,
        "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": initial["market"],
    }
    other_post = {**other_initial, "step": 1, "hour": 1}
    replay = {
        "steps": [
            [{"observation": initial, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other_initial, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": post_plant, "action": {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other_post, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is False
    assert record["outcome"] == "win"


def test_replay_rejects_tampered_plant_without_post_state_effect():
    from scripts.evaluate import replay_record

    initial = {
        "player": 0, "step": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    tampered_post = {
        **initial,
        "step": 1,
        "hour": 1,
        "private": {"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
    }
    other = {
        "player": 1,
        "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": initial["market"],
    }
    replay = {
        "steps": [
            [{"observation": initial, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": tampered_post, "action": {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_price_floor_sales_uses_both_players_market_queues():
    from scripts.evaluate import _price_floor_sales

    inventory = 1_717_032_651_333
    own_observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"shed": {"WHEAT": 2}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 1}, "inventory": {"WHEAT": inventory}},
    }
    other_observation = {
        "player": 1,
        "farms": [{"money": 100}, {"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"shed": {}, "inventories": [{}]},
        "market": own_observation["market"],
    }
    own_state = {"action": {"market": [["SELL", "WHEAT", 2]]}}
    other_state = {"action": {"market": [["BUY_PRODUCT", "WHEAT", 1]]}}

    assert _price_floor_sales(
        own_state, own_observation, {},
        other_state=other_state, other_observation=other_observation,
    ) == 1


def test_default_selection_prioritizes_zero_framework_failure_rate():
    from scripts.evaluate import build_result_document

    records = [
        {"variant": "mixed", "opponent": "pass", "outcome": "loss", "final_bank": 10,
         "opponent_final_bank": 20, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "pass", "outcome": "win", "final_bank": 100,
         "opponent_final_bank": 20, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "starter", "outcome": "framework_error", "final_bank": None,
         "opponent_final_bank": None, "framework_error": True, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
    ]

    document = build_result_document(
        config={"opponents": ["pass", "starter"], "variants": ["mixed", "melon-heavy"]},
        records=records,
    )

    assert document["selected_default"] == "mixed"


@pytest.mark.parametrize("replay", [
    {"steps": None, "statuses": ["DONE", "DONE"], "info": {}},
    {"steps": [], "statuses": ["DONE", "DONE"], "info": None},
    {"steps": [], "statuses": ["DONE", "DONE"], "info": {}, "configuration": []},
    {"steps": [], "statuses": ["DONE", "DONE"], "info": {}, "metadata": []},
])
def test_malformed_replay_shapes_become_framework_failures(replay):
    from scripts.evaluate import replay_record

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_transition_effects_accept_animal_place_inventory_consumption():
    from scripts.evaluate import _transition_effects_valid

    board = [[None for _ in range(10)] for _ in range(10)]
    board[4][4] = {"kind": "COOP"}
    post_board = [[tile for tile in row] for row in board]
    post_board[4][4] = {"kind": "COOP", "animal": "GOOSE", "yield_units": 0}
    pre = {
        "player": 0, "hour": 5,
        "farms": [{"money": 100, "farmer": [4, 4], "hands": [], "tiles": board, "unlocked_quadrants": ["NW"]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{"GOOSE": 1}]},
        "market": {"inventory": {}, "prices": {}},
    }
    post = {**pre, "farms": [{**pre["farms"][0], "tiles": post_board}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PLACE", "GOOSE"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_accept_end_of_day_hire_hand_reset():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "hour": 23, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "hour": 0, "farms": [{**farm, "money": 99, "hands": [], "hires_today": 0}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 99, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}, {}, market_result,
    )


def test_transition_effects_accepts_midday_hire_spawn_and_new_inventory():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 0, "day": 0, "hour": 0, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 1, "day": 0, "hour": 1,
            "farms": [{**farm, "money": 99, "hands": [[5, 4]], "hires_today": 1}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}, {}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 99, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        {"boardSize": 10}, market_result,
    )


def test_transition_effects_rejects_invalid_midday_hire_spawn():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 0, "day": 0, "hour": 0, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 1, "day": 0, "hour": 1,
            "farms": [{**farm, "money": 99, "hands": [[0, 0]], "hires_today": 1}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}, {}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 99, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        {"boardSize": 10}, market_result,
    )


def test_transition_effects_rejects_unrelated_board_mutation():
    from scripts.evaluate import _transition_effects_valid

    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[None, None], [None, None]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[None, {"kind": "COOP"}], [None, None]]}
    pre = {"player": 0, "step": 0, "hour": 0, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 1, "hour": 1, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_unrelated_end_of_day_board_mutation():
    from scripts.evaluate import _transition_effects_valid

    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[None, None], [None, None]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[None, {"kind": "COOP"}], [None, None]], "farmer": [0, 0], "hands": []}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_boundary_water_without_refresh_effect():
    from scripts.evaluate import _transition_effects_valid

    plant = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False,
             "consecutive_unwatered": 1, "yield_units": 1}
    farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
            "tiles": [[plant]], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["WATER"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_treats_unhashable_unlock_metadata_as_invalid():
    from scripts.evaluate import _midday_board_changes_valid

    farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
            "tiles": [[None]], "unlocked_quadrants": [{}]}
    pre = {"player": 0, "step": 1, "hour": 1, "farms": [farm], "private": {}, "market": {}}
    post = {"player": 0, "step": 2, "hour": 2, "farms": [farm], "private": {}, "market": {}}
    market_result = {"market_inventory": {}, "states": [{"unlocked": [{}]}]}

    assert not _midday_board_changes_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_unexpected_post_market_inventory():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
            "tiles": [[None]], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 1, "hour": 1, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {"WHEAT": 10}, "prices": {"WHEAT": 25}}}
    post = {"player": 0, "step": 2, "hour": 2, "farms": [{**farm, "money": 75}],
            "private": {"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{}]},
            "market": {"inventory": {"WHEAT": 8}, "prices": {"WHEAT": 25}}}
    market_result = {"states": [{"money": 75, "shed": {"WHEAT": 1}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {"WHEAT": 9}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 1]]},
        {"townShopSellInterval": 99, "townCenterSellInterval": 99}, market_result,
    )


def test_replay_requires_configuration_and_engine_provenance():
    from scripts.evaluate import replay_record

    observation = {
        "player": 0, "step": 0, "day": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                   "tiles": [[None]], "unlocked_quadrants": ["NW"]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    other = {**observation, "player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [],
                                                        "hires_today": 0, "tiles": [[None]], "unlocked_quadrants": ["NW"]}]}
    replay = {"steps": [[
        {"observation": observation, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
    ]], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_transition_effects_accept_drop_capacity_discard():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "tiles": [[None for _ in range(10)] for _ in range(10)],
            "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "hour": 5, "farms": [farm],
           "private": {"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{"CARROT": 3}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "hour": 6, "farms": [farm],
            "private": {"seeds": {}, "shed": {"WHEAT": 1, "CARROT": 1}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {"WHEAT": 1}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["DROP"], "hands": [], "market": []}, {"shedCapacity": 2}, market_result,
    )


def test_invalid_or_missing_replay_states_are_framework_failures():
    from scripts.evaluate import aggregate_records, replay_record

    replay = {
        "steps": [[
            {"observation": {"player": 0, "farms": [{"money": 100}]}, "action": {}, "status": "DONE", "info": {}},
        ]],
        "statuses": ["DONE", "DONE"],
        "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)
    summary = aggregate_records([record])

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert summary["framework_failures"] == 1
    assert summary["wins"] == summary["losses"] == summary["ties"] == 0


def test_top_level_failure_status_is_counted_even_with_complete_states():
    from scripts.evaluate import replay_record

    states = [
        {"observation": {"player": player, "farms": [{"money": 100}, {"money": 90}]},
         "action": {}, "status": "DONE", "info": {}}
        for player in [0, 1]
    ]
    record = replay_record({"steps": [states], "statuses": ["ERROR", "DONE"], "info": {}},
                           variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_invalid_command_and_bad_worker_count():
    from scripts.evaluate import replay_record

    observation = {
        "player": 0, "farms": [{"money": 100, "hands": [], "tiles": [[None]]}, {"money": 90, "hands": [], "tiles": [[None]]}],
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
        "private": {"shed": {}, "seeds": {}},
    }
    bad_action = {"farmer": ["NOT_REAL"], "hands": [], "market": []}
    good_other = {"player": 1, "farms": [{"money": 100}, {"money": 90}], "market": observation["market"]}
    replay = {
        "steps": [[
            {"observation": observation, "action": bad_action, "status": "DONE", "info": {}},
            {"observation": good_other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        ]],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_market_orders_that_cannot_execute_from_prior_state():
    from scripts.evaluate import _valid_action_schema

    observation = {
        "player": 0,
        "farms": [{"money": 0, "farmer": [0, 0], "hands": [], "hires_today": 0,
                   "unlocked_quadrants": ["NW"], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    prefix = {"farmer": ["PASS"], "hands": []}

    assert not _valid_action_schema({**prefix, "market": [["SELL", "WHEAT", 1]]}, observation)
    assert not _valid_action_schema({**prefix, "market": [["BUY_SEED", "WHEAT", 1]]}, observation)


def test_malformed_market_order_limit_is_a_framework_failure_not_an_exception():
    from scripts.evaluate import _valid_action_schema

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }

    assert not _valid_action_schema(
        {"farmer": ["PASS"], "hands": [], "market": []},
        observation,
        {"maxMarketOrdersPerTurn": "not-a-number"},
    )


def test_boundary_needs_use_post_action_state_when_available():
    from scripts.evaluate import replay_record

    before = {
        "player": 0, "step": 23, "hour": 23,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "watered_today": False}]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    after = {
        "player": 0, "step": 24, "hour": 23,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "watered_today": True}]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    other = {"player": 1, "farms": [{"money": 90}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
             "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    replay = {
        "steps": [
            [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": after, "action": {"farmer": ["WATER"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    assert replay_record(replay, variant="mixed", opponent="pass", seed=1)["missed_basic_needs"] == 0


def test_variants_and_ablations_change_only_legal_action_shapes():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"MELON": 1}},
        "market": {"prices": {"MELON": 250, "WHEAT": 25}},
    }
    action = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}

    variant_action = apply_variant(action, observation, "melon-heavy")
    ablated_action = apply_variant(
        {"farmer": ["EAST"], "hands": [], "market": [["BUY_LAND"]]},
        observation,
        "mixed",
        {"route_scheduling": False, "market_batch_sizing": True, "shop_adaptation": True,
         "land_purchase": False, "animals": True},
    )

    assert variant_action["farmer"] == ["PLANT", "MELON"]
    assert variant_action["market"] == [["BUY_SEED", "MELON", 1]]
    assert ablated_action == {"farmer": ["PASS"], "hands": [], "market": []}


def test_variant_postprocessing_caps_market_orders_and_preserves_affordability():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 80, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}},
        "market": {"prices": {"MELON": 250, "WHEAT": 25}},
    }
    action = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]] * 11}

    result = apply_variant(action, observation, "melon-heavy")

    assert len(result["market"]) <= 10
    assert sum({"WHEAT": 10, "MELON": 80}[order[1]] * order[2] for order in result["market"]) <= 80


def test_market_sanitization_obeys_hire_land_and_shed_rules():
    from scripts.evaluate import apply_variant

    base = {
        "player": 0,
        "farms": [{"money": 2003, "farmer": [0, 0], "hands": [], "hires_today": 3, "unlocked_quadrants": ["NW"], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {"WHEAT": 99}},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["BUY_LAND"], ["BUY_PRODUCT", "WHEAT", 3]]}

    result = apply_variant(action, base, "mixed", configuration={"shedCapacity": 100, "maxMarketOrdersPerTurn": 10})

    assert result["market"] == [["HIRE"], ["BUY_LAND"], ["BUY_PRODUCT", "WHEAT", 1]]


def test_buy_land_uses_next_unlockable_land_and_cost():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 1999, "farmer": [0, 0], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW", "NE"], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}},
        "market": {"prices": {}, "inventory": {}},
    }

    result = apply_variant({"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]}, observation, "mixed")

    assert result["market"] == []


def test_pickup_and_place_commands_are_set_compatible_and_state_aware():
    from scripts.evaluate import apply_variant

    empty_board = [[None for _ in range(10)] for _ in range(10)]
    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [4, 4], "hands": [], "tiles": empty_board}],
        "private": {"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{"WHEAT": 1}]},
        "market": {"prices": {}, "inventory": {}},
    }

    pickup = apply_variant({"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")
    place = apply_variant({"farmer": ["PLACE", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")
    observation["farms"][0]["farmer"] = [0, 0]
    invalid_pickup = apply_variant({"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")
    observation["private"]["inventories"] = [{}]
    invalid_place = apply_variant({"farmer": ["PLACE", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")

    assert pickup["farmer"] == ["PICKUP", "WHEAT", 1]
    assert place["farmer"] == ["PLACE", "WHEAT", 1]
    assert invalid_pickup["farmer"] == ["PASS"]
    assert invalid_place["farmer"] == ["PASS"]


def test_state_aware_unit_sanitization_rejects_locked_occupied_and_failed_preconditions():
    from scripts.evaluate import apply_variant

    def observation(tile, *, seeds=None, inventory=None, farmer=(0, 0)):
        board = [[None for _ in range(3)] for _ in range(3)]
        board[farmer[1]][farmer[0]] = tile
        return {
            "player": 0,
            "farms": [{"money": 100, "farmer": list(farmer), "hands": [], "tiles": board}],
            "private": {"seeds": seeds or {}, "shed": {}, "inventories": [inventory or {}]},
            "market": {"prices": {}, "inventory": {}},
        }

    assert apply_variant({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, observation("LOCKED", seeds={"WHEAT": 1}), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, observation({"kind": "PLANT", "crop": "WHEAT"}, seeds={"WHEAT": 1}), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["BUILD_COOP"], "hands": [], "market": []}, observation({"kind": "COOP"}), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["WATER"], "hands": [], "market": []}, observation(None), "mixed")["farmer"] == ["PASS"]
    animal = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": False}
    assert apply_variant({"farmer": ["FEED"], "hands": [], "market": []}, observation(animal), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["CARE"], "hands": [], "market": []}, observation(animal), "mixed")["farmer"] == ["CARE"]


def test_final_boundary_targeted_feed_without_wheat_is_still_missed():
    from scripts.evaluate import replay_record

    tiles = [[None for _ in range(3)] for _ in range(3)]
    tiles[0][0] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": True}
    before = {"player": 0, "step": 23, "hour": 23, "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": tiles}],
              "private": {"shed": {}, "seeds": {}, "inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    after = {"player": 0, "step": 24, "hour": 0, "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": tiles}],
             "private": {"shed": {}, "seeds": {}, "inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    other = {"player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None for _ in range(3)] for _ in range(3)]}], "private": {"inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    replay = {"steps": [
        [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
        [{"observation": after, "action": {"farmer": ["FEED"], "hands": [], "market": []}, "status": "DONE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
    ], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["missed_basic_needs"] == 1


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_same_seed_and_deterministic_opponent_produce_same_record():
    from scripts.evaluate import run_game

    first = run_game(variant="mixed", opponent="pass", seed=17, steps=8)
    second = run_game(variant="mixed", opponent="pass", seed=17, steps=8)

    assert first == second


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_same_seed_random_style_evaluator_is_reproducible():
    from scripts.evaluate import run_game

    first = run_game(variant="mixed", opponent="random", seed=17, steps=8)
    second = run_game(variant="mixed", opponent="random", seed=17, steps=8)

    assert first == second
    assert first["framework_error"] is False
