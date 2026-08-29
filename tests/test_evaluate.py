import json
from pathlib import Path

import pytest

try:
    from kaggle_environments import make
except ModuleNotFoundError:
    make = None


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
        {"outcome": "loss", "final_bank": 80, "opponent_final_bank": 100, "framework_error": True,
         "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 3},
        {"outcome": "tie", "final_bank": 95, "opponent_final_bank": 95, "framework_error": False,
         "shed_overflow": 1, "price_floor_sales": 2, "missed_basic_needs": 0},
    ]

    summary = aggregate_records(records)

    assert summary["count"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["mean_final_bank"] == pytest.approx(95)
    assert summary["median_final_bank"] == pytest.approx(95)
    assert summary["fifth_percentile_final_bank"] == pytest.approx(81.5)
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
        "farms": [{"money": 120, "tiles": [[{"kind": "PLANT", "watered_today": False}]]}],
        "private": {"shed": {"WHEAT": 102}},
        "market": {"prices": {"WHEAT": 1}},
    }
    replay = {
        "steps": [[
            {"observation": observation, "action": {"market": [["SELL", "WHEAT", 3]]}, "status": "DONE", "info": {}},
            {"observation": {"player": 1, "farms": [{"money": 80}]}, "action": {}, "status": "DONE", "info": {}},
        ]],
        "rewards": [120, 80],
        "statuses": ["DONE", "DONE"],
        "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=5)

    assert record["outcome"] == "win"
    assert record["bank_differential"] == 40
    assert record["shed_overflow"] == 2
    assert record["price_floor_sales"] == 3
    assert record["missed_basic_needs"] == 1


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


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_same_seed_and_deterministic_opponent_produce_same_record():
    from scripts.evaluate import run_game

    first = run_game(variant="mixed", opponent="pass", seed=17, steps=8)
    second = run_game(variant="mixed", opponent="pass", seed=17, steps=8)

    assert first == second
