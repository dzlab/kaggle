import math

import pytest

from kagriculture_agent.constants import (
    HINGE_GAIN,
    MARKET_I0 as CONSTANT_MARKET_I0,
    MARKET_PARAMS,
    PRICE_FLOOR,
    PRODUCTS,
)
from kagriculture_agent.economics import (
    expected_portfolio_cash,
    feed_reserve,
    forecast_animal,
    forecast_crop,
    market_price,
    market_regime,
    opportunity_score,
    project_inventory_after_town,
    sell_batch_value,
    shape_value,
)


MARKET_I0 = 10_000
MARKET_T = {
    "WHEAT": 400,
    "CARROT": 450,
    "TOMATO": 200,
    "STRAWBERRY": 100,
    "MELON": 300,
    "EGG": 332,
    "MILK": 122,
    "WOOL": 105,
    "FERTILIZER": 200,
}


def test_economics_uses_published_market_reference_constants():
    assert CONSTANT_MARKET_I0 == 10_000
    assert PRICE_FLOOR == 1
    assert HINGE_GAIN == 8.0
    assert MARKET_PARAMS["WHEAT"]["base"] == 25
    assert MARKET_PARAMS["FERTILIZER"]["T"] == 200


def test_shape_value_matches_published_shapes_and_safe_domain():
    assert shape_value("linear", -2) == 0
    assert shape_value("sq", 2) == 4
    assert shape_value("sqrt", 4) == 2
    assert shape_value("log", 1) == pytest.approx(math.log(2))
    assert shape_value("log10", 9) == pytest.approx(1)
    assert shape_value("hinge", -1, 4) == 0
    assert shape_value("hinge", 4, 4) == 1
    assert shape_value("hinge", 8, 4) == 10


@pytest.mark.parametrize(
    "item,expected",
    [
        ("WHEAT", 25),
        ("CARROT", 35),
        ("TOMATO", 60),
        ("STRAWBERRY", 120),
        ("MELON", 250),
        ("EGG", 50),
        ("MILK", 160),
        ("WOOL", 200),
        ("FERTILIZER", 100),
    ],
)
def test_market_price_at_reference_inventory(item, expected):
    assert market_price(item, MARKET_I0) == expected


@pytest.mark.parametrize(
    "item,expected",
    [
        ("WHEAT", [45, 20, 19]),
        ("CARROT", [70, 10, 1]),
        ("TOMATO", [84, 24, 9]),
        ("STRAWBERRY", [204, 1, 1]),
        ("MELON", [300, 1, 1]),
        ("EGG", [70, 40, 39]),
        ("MILK", [256, 1, 1]),
        ("WOOL", [240, 1, 1]),
        ("FERTILIZER", [140, 60, 20]),
    ],
)
def test_market_price_reference_table(item, expected):
    assert [
        market_price(item, MARKET_I0 - MARKET_T[item]),
        market_price(item, MARKET_I0 + MARKET_T[item]),
        market_price(item, MARKET_I0 + 2 * MARKET_T[item]),
    ] == expected


def test_market_price_floor_and_sparse_parameter_override():
    assert market_price("STRAWBERRY", MARKET_I0 + 10_000) == 1
    assert market_price(
        "WHEAT",
        10,
        {"WHEAT": {"base": 100, "I0": 0, "T": 10, "above_func": "linear", "above_target": 1}},
    ) == 1


def test_sell_batch_uses_each_pre_sale_quote_and_only_positive_price_supply():
    params = {
        "WHEAT": {
            "base": 100,
            "I0": 0,
            "T": 10,
            "below_func": "linear",
            "below_target": 1,
            "above_func": "linear",
            "above_target": 1,
        }
    }
    assert sell_batch_value("WHEAT", 3, 0, params) == 270
    assert sell_batch_value("WHEAT", -3, 0, params) == 0


def test_project_inventory_after_town_counts_shop_multiplicity_and_center_demand():
    inventory = {item: 100 for item in PRODUCTS}
    projected = project_inventory_after_town(
        inventory,
        ["YARN_STORE", "YARN_STORE", "BAKERY"],
        step=24,
    )
    assert projected["WOOL"] == 95  # 2 + 2 from the duplicate single-product shops, plus center
    assert projected["EGG"] == 98  # bakery plus center
    assert projected["WHEAT"] == 98  # bakery plus center
    assert projected["FERTILIZER"] == 100  # center excludes fertilizer
    assert inventory["WOOL"] == 100


def test_market_regime_is_compact_and_policy_facing():
    prices = {"WHEAT": 25, "CARROT": 35, "MELON": 1}
    inventory = {"WHEAT": 9_000, "CARROT": MARKET_I0, "MELON": 20_000}
    assert market_regime(prices, inventory) == {
        "WHEAT": "scarce",
        "CARROT": "balanced",
        "MELON": "floor",
    }


def test_forecast_crop_counts_planting_day_watering_miss_and_bonus_window():
    miss = forecast_crop("WHEAT", horizon=5, watering_days={1, 2, 3, 4})
    watered_on_planting_day = forecast_crop(
        "WHEAT", horizon=5, watering_days={0, 1, 2, 3, 4}
    )
    assert miss["units"] == 0  # two consecutive unwatered days turn the plant into a weed
    assert watered_on_planting_day["units"] == 4

    bonus = forecast_crop("WHEAT", horizon=5, watering_days={0, 1, 2, 3, 4}, fertilizer_days={2})
    assert bonus["units"] == 6
    assert bonus["seed_cost"] == 10


def test_forecast_crop_models_three_day_fertilizer_and_ongoing_schedule_decay():
    tomato = forecast_crop(
        "TOMATO",
        horizon=14,
        watering_days=set(range(14)),
        fertilizer_days={8},
        harvest_day=None,
    )
    assert tomato["units_before_decay"] == 4
    assert tomato["fertilizer_days_active"] == {8, 9, 10}
    assert tomato["units"] == 0  # repeated every-other-turn decay reaches weed state
    assert tomato["decayed_units"] == 4


def test_crop_harvest_happens_before_first_decay_turn():
    wheat = forecast_crop(
        "WHEAT", horizon=6, watering_days={0, 1, 2, 3, 4}, harvest_day=5,
    )
    assert wheat["harvested_units"] == 4
    assert wheat["decayed_units"] == 0


def test_crop_fertilizer_cost_ignores_applications_after_horizon():
    wheat = forecast_crop(
        "WHEAT", horizon=5, watering_days={0, 1, 2, 3, 4},
        fertilizer_days={2, 10},
    )
    assert wheat["fertilizer_cost"] == 100


def test_forecast_crop_accounts_for_floor_risk_across_sequential_sales():
    params = {
        "WHEAT": {
            "base": 100, "I0": 0, "T": 1,
            "below_func": "linear", "below_target": 1,
            "above_func": "linear", "above_target": 100,
        }
    }
    wheat = forecast_crop(
        "WHEAT", horizon=5, watering_days={0, 1, 2, 3, 4},
        market_inventory=0, params=params,
    )
    assert wheat["units"] == 4
    assert wheat["floor_units"] == 3
    assert wheat["price_floor_risk"] == pytest.approx(0.75)


def test_forecast_one_time_crop_stops_after_harvest():
    carrot = forecast_crop(
        "CARROT", horizon=5, watering_days={0, 1, 2, 3, 4}, harvest_day=2,
    )
    assert carrot["harvested_units"] == 2
    assert carrot["units"] == 2
    assert carrot["decayed_units"] == 0


def test_forecast_animal_models_first_yield_feed_care_held_cap_and_fertilizer():
    goose = forecast_animal(
        "GOOSE",
        horizon=12,
        feed_days=set(range(12)),
        care_days={4, 5, 6, 7, 8, 9, 10, 11},
        collect_fertilizer_days=set(range(12)),
    )
    assert goose["first_yield_day"] == 4
    assert goose["units"] == 16
    assert goose["harvested_units"] == 16
    assert goose["feed_units"] == 12
    assert goose["fertilizer_units"] == 11  # manure is available after each day-end refresh
    assert goose["animal_cost"] == 300

    starved = forecast_animal("GOOSE", horizon=6, feed_days={0, 1})
    assert starved["escaped"] is True
    assert starved["units"] == 0


def test_animal_production_continues_past_held_cap_when_collected():
    goose = forecast_animal("GOOSE", horizon=20, feed_days=set(range(20)))
    assert goose["units"] > 4
    assert goose["production_events"] == 17


def test_animal_held_cap_applies_to_uncollected_buffer_only():
    goose = forecast_animal(
        "GOOSE", horizon=20, feed_days=set(range(20)), harvest_days={10},
    )
    assert goose["held_units"] == 4
    assert goose["harvested_units"] == 4
    assert goose["production_events"] == 17


def test_animal_escape_discards_uncollected_held_product():
    goose = forecast_animal(
        "GOOSE", horizon=6, feed_days={0, 1, 2, 3}, harvest_days=set(),
    )
    assert goose["escaped"] is True
    assert goose["held_units"] == 0
    assert goose["units"] == 0


def test_animal_care_bonus_is_banked_once_then_reset_on_production():
    one_bonus = forecast_animal(
        "GOOSE", horizon=6, feed_days=set(range(6)), care_days={3},
    )
    two_bonuses = forecast_animal(
        "GOOSE", horizon=6, feed_days=set(range(6)), care_days={3, 4},
    )
    assert one_bonus["units"] == 4
    assert two_bonuses["units"] == 5


def test_animal_fertilizer_availability_is_one_unit_and_resets_after_collection():
    goose = forecast_animal(
        "GOOSE", horizon=5, feed_days=set(range(5)),
        collect_fertilizer_days={1, 3},
    )
    assert goose["fertilizer_units"] == 2


def test_animal_feed_cost_uses_supplied_market_params():
    params = {
        "WHEAT": {
            "base": 100, "I0": 0, "T": 10,
            "below_func": "linear", "below_target": 1,
            "above_func": "linear", "above_target": 1,
        }
    }
    goose = forecast_animal(
        "GOOSE", horizon=1, feed_days={0}, market_inventory=0, params=params,
    )
    assert goose["feed_cost"] == 100


def test_feed_reserve_covers_one_wheat_per_live_animal_day():
    assert feed_reserve({"GOOSE": 2, "COW": 1}, days=5) == 15
    assert feed_reserve({"GOOSE": 2}, days=5, existing_wheat=3) == 7


def test_portfolio_cash_and_opportunity_score_include_costs_and_risks():
    crop = forecast_crop(
        "CARROT",
        horizon=5,
        watering_days={2, 3},
        prices={"CARROT": 35},
        worker_cost=4,
        land_cost=5,
        movement_turns=2,
        action_turns=3,
        shed_capacity=3,
    )
    total = expected_portfolio_cash([crop], starting_cash=100)
    assert total["cash"] == 100 + crop["net_cash"]
    assert total["turns"] == 5
    assert opportunity_score(crop, turn_value=1, risk_penalty=10) < crop["net_cash"]
