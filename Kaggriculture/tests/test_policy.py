from copy import deepcopy

import kagriculture_agent.policy as policy_module
import pytest
from kagriculture_agent.types import Position, Task, WorkerAssignment


VALID_MOVES = {"NORTH", "SOUTH", "EAST", "WEST", "PASS"}
VALID_SIMPLE = {
    "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED",
    "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
}


def test_policy_defaults_to_current_strategy():
    from kagriculture_agent.policy import Policy

    assert Policy().strategy_name == "current"


def test_unknown_strategy_is_rejected():
    from kagriculture_agent.policy import Policy

    with pytest.raises(ValueError, match="unsupported strategy: not-a-route"):
        Policy(strategy="not-a-route")


def test_strategy_registry_has_expected_names_and_values():
    from kagriculture_agent.strategy import STRATEGIES, StrategySpec

    assert STRATEGIES == {
        "current": StrategySpec(
            "current",
            ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"),
            ("GOOSE", "COW", "SHEEP"),
            10000,
            10000,
            8,
        ),
        "melon": StrategySpec("melon", ("WHEAT", "MELON"), ("COW", "SHEEP"), 80, 9, 12),
        "premium": StrategySpec("premium", ("WHEAT", "MELON"), ("COW", "SHEEP"), 48, 12, 16),
        "mixed": StrategySpec("mixed", ("WHEAT", "CARROT", "MELON"), ("COW", "SHEEP"), 64, 9, 12),
    }


def test_get_strategy_reports_unknown_name():
    from kagriculture_agent.strategy import get_strategy

    with pytest.raises(ValueError) as exc_info:
        get_strategy("not-a-route")

    assert str(exc_info.value) == "unsupported strategy: not-a-route"


def test_select_strategy_prefers_melon_without_shop_signal():
    from kagriculture_agent.strategy import select_strategy

    state = {"cash": 3000, "town": {"unlocked_shops": []}, "market": {"prices": {}, "inventory": {}}}

    assert select_strategy(state).name == "melon"


def test_select_strategy_uses_pet_cafe_signal():
    from kagriculture_agent.strategy import select_strategy

    state = {"cash": 3000, "town": {"unlocked_shops": ["PET_CAFE"]}, "market": {"prices": {}, "inventory": {}}}

    assert select_strategy(state).name == "mixed"


@pytest.mark.parametrize("shop", ["YARN_STORE", "ICE_CREAM_SHOP"])
def test_select_strategy_uses_premium_shop_signals(shop):
    from kagriculture_agent.strategy import select_strategy

    state = {"town": {"unlocked_shops": [shop]}}

    assert select_strategy(state).name == "premium"


@pytest.mark.parametrize(
    "state",
    [None, [], {"town": None}, {"town": {"unlocked_shops": None}},
     {"town": {"unlocked_shops": [{}, [], None]}}],
)
def test_select_strategy_falls_back_for_malformed_state(state):
    from kagriculture_agent.strategy import select_strategy

    assert select_strategy(state).name == "melon"


def observation(*, day=0, hour=0, hands=None, tiles=None, shed=None,
                seeds=None, inventories=None, money=3_000, market=None):
    hands = [[1, 0], [2, 0]] if hands is None else hands
    board = [[None for _ in range(5)] for _ in range(5)] if tiles is None else tiles
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "farms": [{
            "tiles": board,
            "farmer": [0, 0],
            "hands": hands,
            "money": money,
            "unlocked_quadrants": ["NW"],
        }, {"tiles": board, "farmer": [0, 0], "hands": []}],
        "private": {
            "shed": {} if shed is None else shed,
            "seeds": {"WHEAT": 2} if seeds is None else seeds,
            "inventories": inventories if inventories is not None else [[], [], []],
        },
        "market": {
            "prices": market if market is not None else {"WHEAT": 10, "FERTILIZER": 20},
            "inventory": {},
        },
        "town": {"unlocked_shops": []},
    }


def command_is_legal(command):
    if not isinstance(command, list) or not command:
        return False
    if len(command) == 1 and command[0] in VALID_MOVES | VALID_SIMPLE:
        return True
    if len(command) == 2 and command[0] in {"PLANT", "PICKUP", "PLACE"}:
        return command[1].isalpha() and command[1].upper() == command[1]
    if len(command) == 3 and command[0] in {"PICKUP", "PLACE"}:
        return command[1].isalpha() and command[1].upper() == command[1] and isinstance(command[2], int) and command[2] > 0
    return False


def test_policy_emits_one_legal_command_per_visible_worker_and_bounded_market():
    policy = policy_module.Policy()

    action = policy.act(observation())

    assert set(action) == {"farmer", "hands", "market"}
    assert command_is_legal(action["farmer"])
    assert len(action["hands"]) == 2
    assert all(command_is_legal(command) for command in action["hands"])
    assert isinstance(action["market"], list)
    assert len(action["market"]) <= 10


def test_auto_policy_selects_once_from_opening_state():
    policy = policy_module.Policy(strategy="auto")

    policy.act(observation(day=0, hour=0))
    chosen = policy.memory.selected_strategy
    later = observation(day=0, hour=1)
    later["town"] = {"unlocked_shops": ["YARN_STORE"]}
    policy.act(later)

    assert chosen == "melon"
    assert policy.memory.selected_strategy == chosen


def test_auto_policy_does_not_reselect_on_duplicate_opening_observation():
    policy = policy_module.Policy(strategy="auto")

    policy.act(observation(day=0, hour=0))
    chosen = policy.memory.selected_strategy
    duplicate = observation(day=0, hour=0)
    duplicate["town"] = {"unlocked_shops": ["YARN_STORE"]}

    policy.act(duplicate)

    assert chosen == "melon"
    assert policy.memory.selected_strategy == chosen


def test_auto_policy_reselects_after_a_genuine_time_reset():
    policy = policy_module.Policy(strategy="auto")

    policy.act(observation(day=0, hour=0))
    policy.act(observation(day=0, hour=1))
    reset = observation(day=0, hour=0)
    reset["town"] = {"unlocked_shops": ["YARN_STORE"]}

    policy.act(reset)

    assert policy.memory.selected_strategy == "premium"


def test_auto_policy_forwards_selected_strategy_spec_to_macro_planner(monkeypatch):
    from kagriculture_agent.strategy import get_strategy

    calls = []

    def fake_macro(state, memory, strategy=None):
        calls.append(strategy)
        return {"portfolio": {}, "scenario_count": 0, "market_intents": [], "tasks": []}

    monkeypatch.setattr(policy_module, "build_autonomous_macro_plan", fake_macro)
    policy = policy_module.Policy(strategy="auto")

    policy.act(observation(day=0, hour=0))

    assert calls == [get_strategy("melon")]


def test_macro_planner_consumes_strategy_spec_for_portfolio_candidates():
    from kagriculture_agent.planner import build_autonomous_macro_plan
    from kagriculture_agent.strategy import get_strategy

    state = observation(
        market={
            "TOMATO": 1_000,
            "WHEAT": 1,
            "CARROT": 1,
            "STRAWBERRY": 1,
            "MELON": 1,
        },
    )

    portfolio = build_autonomous_macro_plan(
        state,
        strategy=get_strategy("premium"),
    )["portfolio"]

    assert portfolio["crop"] in get_strategy("premium").crops
    assert portfolio["crop"] != "TOMATO"


def test_daily_plan_filters_crop_tasks_to_strategy_allowlist_and_cap():
    from kagriculture_agent.planner import build_daily_plan
    from kagriculture_agent.strategy import StrategySpec

    state = {
        "day": 2,
        "board_size": 3,
        "tiles": [[None, None, None] for _ in range(3)],
        "seeds": {"TOMATO": 3, "WHEAT": 3},
        "inventory": {},
        "workers": [],
    }
    strategy = StrategySpec("wheat-only", ("WHEAT",), (), 1, 0, 0)

    plant_tasks = [
        task for task in build_daily_plan(state, strategy=strategy)
        if task.kind == "PLANT"
    ]

    assert len(plant_tasks) == 1
    assert plant_tasks[0].item == "WHEAT"


def test_daily_plan_filters_animal_tasks_to_strategy_allowlist_and_cap():
    from kagriculture_agent.planner import build_daily_plan
    from kagriculture_agent.strategy import StrategySpec

    state = {
        "day": 2,
        "board_size": 3,
        "tiles": [[None for _ in range(3)] for _ in range(3)],
        "animals": [],
        "desired_animals": [
            {"species": "GOOSE", "position": [0, 0], "owned": False},
            {"species": "GOOSE", "position": [1, 0], "owned": False},
            {"species": "COW", "position": [2, 0], "owned": False},
        ],
        "inventory": {},
        "workers": [],
    }
    strategy = StrategySpec("one-cow", ("WHEAT",), ("COW",), 10, 1, 0)

    animal_tasks = [
        task for task in build_daily_plan(state, strategy=strategy)
        if task.kind == "ANIMAL"
    ]

    assert len(animal_tasks) == 1
    assert animal_tasks[0].item == "COW"


def test_macro_plan_caps_animal_purchases_by_strategy():
    from kagriculture_agent.planner import build_autonomous_macro_plan
    from kagriculture_agent.strategy import StrategySpec

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    state = observation(day=0, hour=1, hands=[], tiles=board, money=5_000,
                        shed={}, seeds={}, inventories=[[]])
    strategy = StrategySpec("no-animals", ("WHEAT",), ("COW",), 10, 0, 0)

    macro = build_autonomous_macro_plan(state, strategy=strategy)

    assert not any(intent[0] == "BUY_ANIMAL" for intent in macro["market_intents"])
    assert not any(task.kind == "ANIMAL" for task in macro["tasks"])


def test_macro_plan_uses_strategy_reserve_wheat_for_feed_purchase():
    from kagriculture_agent.planner import build_autonomous_macro_plan
    from kagriculture_agent.strategy import StrategySpec

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "PASTURE", "animal": {"species": "COW", "fed_today": False}}
    state = observation(day=0, hour=1, hands=[], tiles=board, money=5_000,
                        shed={}, seeds={}, inventories=[[]])
    base = StrategySpec("base", ("WHEAT",), ("COW",), 10, 10, 0)
    reserved = StrategySpec("reserved", ("WHEAT",), ("COW",), 10, 10, 20)

    def wheat_purchase(strategy):
        return next(
            intent[2] for intent in build_autonomous_macro_plan(state, strategy=strategy)["market_intents"]
            if intent[0] == "BUY_PRODUCT" and intent[1] == "WHEAT"
        )

    assert wheat_purchase(reserved) > wheat_purchase(base)


def test_auto_policy_never_schedules_crop_excluded_by_selected_route():
    policy = policy_module.Policy(strategy="auto")
    state = observation(
        day=0,
        hour=0,
        hands=[[1, 0], [2, 0]],
        tiles=[[None for _ in range(5)] for _ in range(5)],
        seeds={"TOMATO": 1, "WHEAT": 1},
        money=5_000,
        market={"TOMATO": 1_000, "WHEAT": 1, "MELON": 1},
    )

    policy.act(state)

    assert policy.memory.selected_strategy == "melon"
    plant_tasks = [
        assignment.task for assignment in policy.memory.assignments
        if assignment.task.kind == "PLANT"
    ]
    assert plant_tasks
    assert all(task.item in {"WHEAT", "MELON"} for task in plant_tasks)


def test_named_policy_keeps_requested_strategy_when_shops_change():
    policy = policy_module.Policy(strategy="premium")

    policy.act(observation(day=0, hour=0))
    later = observation(day=0, hour=1)
    later["town"] = {"unlocked_shops": ["PET_CAFE"]}
    policy.act(later)

    assert policy.strategy_name == "premium"
    assert policy.memory.selected_strategy is None


@pytest.mark.parametrize("strategy_name", ["melon", "premium", "mixed"])
def test_named_policy_applies_route_allowlists_to_tasks_and_market_orders(strategy_name):
    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {
        "kind": "PLANT", "crop": "TOMATO", "needs_water": True,
        "watered_today": False, "planted_day": 0, "planted_age": 8,
        "yield_units": 1,
    }
    board[0][1] = {
        "kind": "COOP", "animal": {"species": "GOOSE", "needs_feed": True,
                                      "fed_today": False, "needs_care": True,
                                      "cared_today": False},
    }
    obs = observation(
        day=4, hour=0, hands=[[2, 0]], tiles=board, seeds={"TOMATO": 1},
        inventories=[[], []], money=5_000,
    )
    obs["market_intents"] = [
        ["BUY_SEED", "TOMATO", 1],
        ["BUY_ANIMAL", "GOOSE", 1],
    ]

    policy = policy_module.Policy(strategy=strategy_name)
    action = policy.act(obs)

    maintenance = {(assignment.task.kind, assignment.task.item)
                   for assignment in policy.memory.assignments}
    assert {("WATER", "TOMATO"), ("FEED", "GOOSE")} & maintenance
    assert not any(
        task.kind in {"PLANT", "ANIMAL"} and task.item in {"TOMATO", "GOOSE"}
        for assignment in policy.memory.assignments
        for task in [assignment.task]
    )
    assert ["BUY_SEED", "TOMATO", 1] not in action["market"]
    assert ["BUY_ANIMAL", "GOOSE", 1] not in action["market"]


def test_zero_strategy_caps_skip_autonomous_production_without_crashing():
    from kagriculture_agent.planner import build_autonomous_macro_plan
    from kagriculture_agent.strategy import StrategySpec

    strategy = StrategySpec("zero", ("WHEAT",), ("COW",), 0, 0, 0)
    state = observation(
        day=0, hour=1, hands=[], tiles=[[None for _ in range(5)] for _ in range(5)],
        seeds={}, inventories=[[]], money=5_000,
    )

    macro = build_autonomous_macro_plan(state, strategy=strategy)

    assert not any(intent[0] in {"BUY_SEED", "BUY_ANIMAL"} for intent in macro["market_intents"])
    assert not any(task.kind in {"PLANT", "ANIMAL"} for task in macro["tasks"])


def test_policy_forwards_named_strategy_to_all_planning_and_market_paths(monkeypatch):
    from kagriculture_agent.strategy import get_strategy

    spec = get_strategy("premium")
    macro_calls = []
    daily_calls = []
    assignment_calls = []
    market_calls = []

    def fake_macro(state, memory, strategy=None):
        macro_calls.append(strategy)
        return {"portfolio": {}, "scenario_count": 0, "market_intents": [], "tasks": []}

    def fake_daily(state, memory=None, strategy=None):
        daily_calls.append(strategy)
        return []

    def fake_assign(plan, workers, state, strategy=None):
        assignment_calls.append(strategy)
        return []

    def fake_market(state, plan, strategy=None):
        market_calls.append(strategy)
        return []

    monkeypatch.setattr(policy_module, "build_autonomous_macro_plan", fake_macro)
    monkeypatch.setattr(policy_module, "build_daily_plan", fake_daily)
    monkeypatch.setattr(policy_module, "assign_tasks", fake_assign)
    monkeypatch.setattr(policy_module, "build_market_orders", fake_market)

    policy = policy_module.Policy(strategy="premium")
    state = policy_module.parse_observation(observation(day=0, hour=0))
    policy.act(state)
    policy._replan(state, {}, strategy=spec)

    assert macro_calls == [spec, spec]
    assert daily_calls == [spec, spec, spec]
    assert assignment_calls == [spec, spec]
    assert market_calls == [spec]


def test_planner_sanitizes_invalid_seed_quantities_and_nonfinite_task_values():
    from kagriculture_agent.planner import assign_tasks, build_daily_plan, normalize_planner_state

    state = {
        "day": 0,
        "tiles": [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": float("inf")}]],
        "seeds": {"WHEAT": float("nan"), "MELON": -3, "CARROT": "invalid"},
        "inventory": {"WHEAT": float("inf"), "CARROT": -2},
        "market": {"prices": {"WHEAT": 25}},
    }

    normalized = normalize_planner_state(state)
    assert normalized["seeds"] == {"WHEAT": 0, "MELON": 0, "CARROT": 0}
    assert normalized["inventory"] == {"WHEAT": 0, "CARROT": 0}
    plan = build_daily_plan(state)
    assert all(task.value >= 0 and task.value != float("inf") and task.value == task.value for task in plan)
    assert all(task.kind != "HARVEST" for task in plan)

    invalid_task = Task("WATER", (0, 0), 100, 0, float("nan"))
    assignments = assign_tasks(
        [invalid_task],
        [{"index": 0, "role": "FARMER", "position": (0, 0)}],
        {"day": 0, "board_size": 1, "tiles": [[None]], "workers": []},
    )
    assert assignments == []


@pytest.mark.parametrize("malformed_shop", [{}, []])
def test_town_demand_ignores_unhashable_shop_entries(malformed_shop):
    from kagriculture_agent.planner import _town_demand

    state = {"town": {"unlocked_shops": [malformed_shop]}}

    assert _town_demand(state) == set()


def test_policy_falls_back_to_safe_action_for_nonfinite_or_negative_shed_values():
    action = policy_module.Policy().act(
        observation(shed={"WHEAT": float("nan"), "MELON": float("inf"), "CARROT": -3})
    )

    assert set(action) == {"farmer", "hands", "market"}
    assert command_is_legal(action["farmer"])
    assert len(action["hands"]) == 2
    assert all(command_is_legal(command) for command in action["hands"])


def test_policy_preserves_a_command_slot_for_every_malformed_visible_hand():
    action = policy_module.Policy().act(observation(hands=[[1, 0], None, ["bad", 1]]))

    assert len(action["hands"]) == 3
    assert all(command_is_legal(command) for command in action["hands"])


def test_parse_observation_keeps_canonical_nested_state_for_policy():
    obs = observation(day=4, hour=9, shed={"WHEAT": 3})

    parsed = policy_module.parse_observation(obs)

    assert parsed["day"] == 4
    assert parsed["hour"] == 9
    assert parsed["farm"]["farmer"] == [0, 0]
    assert parsed["private"]["shed"] == {"WHEAT": 3}


def test_market_orders_reserve_feed_wheat_and_never_sell_more_than_shed():
    state = {
        "day": 4,
        "hour": 3,
        "cash": 500,
        "animals": [{"species": "GOOSE"}, {"species": "GOOSE"}],
        "private": {"shed": {"WHEAT": 3, "CARROT": 2}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "CARROT": 20}},
    }
    plan = [
        Task("SELL", Position(2, 2), 75, 4, 100, sell_all=True),
        Task("BUY_SEED", "CARROT", 20, None, 100),
    ]

    orders = policy_module.build_market_orders(state, plan)

    assert [order for order in orders if order[0] == "SELL" and order[1] == "WHEAT"] == []
    carrot_sales = [order for order in orders if order[0] == "SELL" and order[1] == "CARROT"]
    assert all(order[2] <= 2 for order in carrot_sales)
    wheat_buys = [order for order in orders if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT"]
    assert wheat_buys and wheat_buys[0][2] >= 1
    assert len(orders) <= 10


def test_market_orders_do_not_spend_cash_reserved_for_feed_or_exceed_ten_orders():
    state = {
        "day": 4,
        "hour": 3,
        "cash": 15,
        "animals": [{"species": "GOOSE"}],
        "private": {"shed": {}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10}},
    }
    plan = [Task("BUY_SEED", "WHEAT", 20, None, 0) for _ in range(20)]

    orders = policy_module.build_market_orders(state, plan)

    assert len(orders) <= 10
    assert sum(order[2] for order in orders if order[:2] == ["BUY_SEED", "WHEAT"]) <= 1


def test_market_orders_liquidate_saleable_shed_inventory_on_penultimate_turn():
    state = {
        "day": 29, "hour": 22, "cash": 0,
        "private": {"shed": {"MELON": 2}, "seeds": {}},
        "market": {"prices": {"MELON": 250}, "inventory": {"MELON": 10_000}},
    }

    orders = policy_module.build_market_orders(state, [])

    assert ["SELL", "MELON", 2] in orders


def test_policy_drops_carried_animals_before_terminal_sale_window():
    obs = observation(day=29, hour=21, hands=[[2, 2]], inventories=[[], ["GOOSE"]],
                      shed={"GOOSE": 0, "MELON": 0}, seeds={})

    action = policy_module.Policy().act(obs)

    assert action["hands"][0] == ["DROP"]


def test_policy_keeps_required_carried_animal_for_assigned_placement():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    obs = observation(day=4, hour=1, hands=[[4, 4]], inventories=[[], ["GOOSE"]],
                      shed={"GOOSE": 0}, seeds={}, tiles=board)

    action = policy_module.Policy().act(obs)

    assert action["hands"][0] != ["DROP"]


def test_carried_animal_assignment_stays_valid_when_feed_is_staged_in_shed():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[1][1] = {"kind": "PASTURE"}
    state = policy_module.parse_observation(observation(
        day=4,
        hands=[[2, 2]],
        tiles=board,
        shed={"WHEAT": 1, "SHEEP": 0},
        seeds={},
        inventories=[[], ["SHEEP"]],
    ))
    assignment = WorkerAssignment(
        1, Task("ANIMAL", Position(1, 1), 105, 4, 500, item="SHEEP")
    )

    assert policy_module._assignment_valid(state, assignment)
    assert policy_module.worker_action(1, state, assignment) == ["PICKUP", "WHEAT", 1]


def test_late_season_planners_do_not_start_new_crops():
    from kagriculture_agent.constants import season_days
    from kagriculture_agent.planner import build_autonomous_macro_plan, build_daily_plan

    state = observation(
        day=season_days - 2,
        hands=[],
        seeds={"WHEAT": 1},
        money=3_000,
    )

    assert all(task.kind != "PLANT" for task in build_daily_plan(state))
    assert all(task.kind != "PLANT" for task in build_autonomous_macro_plan(state)["tasks"])


def test_final_actionable_day_hires_for_unmet_basic_need_deadlines():
    from kagriculture_agent.constants import season_days
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    for x in (1, 2):
        board[1][x] = {
            "kind": "PLANT",
            "crop": "STRAWBERRY",
            "watered_today": False,
            "yield_units": 1,
            "planted_day": 0,
        }
    state = observation(
        day=season_days - 2,
        hour=0,
        hands=[],
        tiles=board,
        seeds={},
        money=3_000,
    )

    assert [intent for intent in build_autonomous_macro_plan(state)["market_intents"] if intent == ["HIRE"]] == [["HIRE"]]


def test_policy_replans_when_a_newly_hired_hand_appears_before_deadline():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "PLANT", "crop": "STRAWBERRY", "watered_today": False}
    policy = policy_module.Policy()

    policy.act(observation(day=27, hour=0, hands=[], tiles=board, seeds={}, money=1_000))
    action = policy.act(observation(day=27, hour=1, hands=[[4, 4]], tiles=board, seeds={}, money=999))

    assert action["hands"][0] != ["PASS"]


@pytest.mark.parametrize("day", [28, 29])
def test_late_season_seed_fallback_does_not_buy_wheat_when_planting_is_suppressed(day):
    action = policy_module.Policy().act(observation(day=day, hour=0, hands=[], seeds={}))

    assert [order for order in action["market"] if order[:2] == ["BUY_SEED", "WHEAT"]] == []


def test_daily_plan_uses_fertilizer_staged_in_shed():
    from kagriculture_agent.planner import build_daily_plan

    state = {
        "day": 4,
        "tiles": [[{
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": True,
            "yield_units": 1,
            "planted_day": 0,
            "fertilized_until_day": -1,
        }]],
        "private": {"shed": {"FERTILIZER": 1}, "inventories": [{}], "seeds": {}},
        "market": {"prices": {"WHEAT": 25, "FERTILIZER": 100}},
    }

    assert any(task.kind == "FERTILIZE" for task in build_daily_plan(state))


def test_fertilize_can_use_farmer_slot_while_helper_preserves_basic_need():
    from kagriculture_agent.planner import assign_tasks

    state = {
        "day": 4,
        "hour": 10,
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "private": {"shed": {"FERTILIZER": 1}, "inventories": [{"FERTILIZER": 1}, {}]},
        "workers": [
            {"index": 0, "role": "FARMER", "position": [2, 2]},
            {"index": 1, "role": "WORKER", "position": [1, 1]},
        ],
    }
    plan = [
        Task("WATER", Position(0, 0), 100, 4, 1),
        Task("FERTILIZE", Position(0, 0), 97, 4, 1),
        Task("CARE", Position(0, 0), 95, 4, 1),
        Task("SHED", Position(2, 2), 85, 4, 1),
    ]

    assignments = assign_tasks(plan, state["workers"], state)

    assert {assignment.task.kind: assignment.worker_index for assignment in assignments} == {
        "WATER": 1,
        "FERTILIZE": 0,
    }


def test_staged_fertilizer_remains_assignable_after_same_day_route_budget():
    from kagriculture_agent.planner import assign_tasks, build_daily_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "watered_today": True,
        "yield_units": 0,
        "planted_day": 0,
        "fertilized_until_day": -1,
    }
    state = {
        "day": 4,
        "hour": 10,
        "board_size": 5,
        "tiles": board,
        "private": {"shed": {"FERTILIZER": 1}, "inventories": [[]], "seeds": {}},
        "workers": [{"index": 0, "role": "FARMER", "position": [0, 0]}],
    }
    assignments = assign_tasks(build_daily_plan(state), None, state)

    assert [assignment.task.kind for assignment in assignments] == ["FERTILIZE"]


def test_replanning_does_not_duplicate_a_protected_carried_feed_task():
    board = [[None for _ in range(10)] for _ in range(10)]
    board[0][0] = {
        "kind": "PASTURE", "animal": "COW", "fed_today": False,
        "cared_today": False,
    }
    obs = observation(
        day=20,
        hour=1,
        hands=[[5, 4]],
        tiles=board,
        shed={"WHEAT": 1},
        seeds={},
        inventories=[["WHEAT"], ["WHEAT"]],
    )
    obs["farms"][0]["farmer"] = [4, 4]
    policy = policy_module.Policy()
    policy.memory.assignments = [
        WorkerAssignment(0, Task("FEED", Position(0, 0), 100, 20, 1)),
    ]
    policy.memory.last_day = 20
    policy.memory.last_hour = 0

    policy.act(obs)

    feed_assignments = [assignment for assignment in policy.memory.assignments
                        if assignment.task.kind == "FEED"]
    assert [(assignment.worker_index, assignment.task.target) for assignment in feed_assignments] == [
        (0, Position(0, 0)),
    ]


def test_terminal_cleanup_does_not_mix_carried_pickup_or_drop_with_sales():
    obs = observation(day=29, hour=22, hands=[[2, 2]],
                      inventories=[[], ["FERTILIZER"]],
                      shed={"MELON": 2, "FERTILIZER": 0}, seeds={})

    action = policy_module.Policy().act(obs)

    assert action["hands"][0] == ["DROP"]
    assert action["market"] == []


def test_replan_preserves_carried_feed_when_another_assignment_finishes():
    board = [[None for _ in range(10)] for _ in range(10)]
    board[0][5] = {
        "kind": "COOP", "animal": "GOOSE", "fed_today": False,
        "cared_today": True,
    }
    obs = {
        "player": 0, "day": 20, "hour": 16,
        "farms": [{
            "tiles": board, "farmer": [5, 4], "hands": [[5, 0]],
            "money": 1000, "unlocked_quadrants": ["NW"],
        }, {"tiles": board, "farmer": [0, 0], "hands": []}],
        "private": {
            "shed": {"WHEAT": 1}, "seeds": {},
            "inventories": [{"WHEAT": 1}, {}],
        },
        "market": {"prices": {"WHEAT": 10}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }
    policy = policy_module.Policy()
    policy.memory.assignments = [
        WorkerAssignment(0, Task("FEED", Position(5, 0), 100, 20, 1)),
        WorkerAssignment(1, Task("CARE", Position(5, 0), 95, 20, 1)),
    ]
    policy.memory.last_day = 20
    policy.memory.last_hour = 15

    action = policy.act(obs)

    assert action["farmer"] != ["DROP"]


def test_zero_filled_real_engine_shed_does_not_keep_cached_shed_or_sell_valid():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {
            "inventories": [[]],
            "shed": {"WHEAT": 0, "EGG": 0, "FERTILIZER": 0},
            "seeds": {},
        },
    }
    shed = WorkerAssignment(0, Task("SHED", Position(1, 1), 10, None, 1))
    sell = WorkerAssignment(0, Task("SELL", Position(1, 1), 10, None, 1))

    assert not policy_module._assignment_valid(state, shed)
    assert not policy_module._assignment_valid(state, sell)


def test_planner_sell_all_is_explicit_and_excludes_fertilizer():
    state = {
        "day": 4, "hour": 3, "cash": 0,
        "private": {"shed": {"WHEAT": 2, "CARROT": 1, "FERTILIZER": 3}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "CARROT": 20, "FERTILIZER": 30}},
    }

    planner_sell = next(task for task in policy_module.build_daily_plan(state) if task.kind == "SELL")
    assert planner_sell.sell_all is True
    orders = policy_module.build_market_orders(state, [planner_sell])

    assert ["SELL", "WHEAT", 2] in orders
    assert ["SELL", "CARROT", 1] in orders
    assert not any(order[1] == "FERTILIZER" for order in orders)


def test_malformed_or_unknown_sell_intents_do_not_wildcard_sell():
    state = {
        "day": 4, "hour": 3, "cash": 0,
        "private": {"shed": {"WHEAT": 2, "FERTILIZER": 3}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "FERTILIZER": 30}},
    }

    orders = policy_module.build_market_orders(state, [
        {"kind": "SELL"}, ["SELL", "UNKNOWN", 2], ["SELL"],
    ])

    assert orders == []


def test_explicit_sell_fertilizer_is_accepted_while_sell_all_excludes_it():
    state = {
        "day": 4, "hour": 3, "cash": 0,
        "private": {"shed": {"FERTILIZER": 3}, "seeds": {}},
        "market": {"prices": {"FERTILIZER": 30}},
    }

    orders = policy_module.build_market_orders(state, [["SELL", "FERTILIZER", 2]])

    assert orders == [["SELL", "FERTILIZER", 2]]


def test_planner_sell_all_unions_explicit_fertilizer_sell():
    state = {
        "day": 4, "hour": 3, "cash": 0,
        "private": {"shed": {"WHEAT": 2, "FERTILIZER": 3}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "FERTILIZER": 30}},
    }
    planner_sell = Task("SELL", Position(1, 1), 10, 4, 1, sell_all=True)

    orders = policy_module.build_market_orders(state, [
        planner_sell, ["SELL", "FERTILIZER", 2],
    ])

    assert ["SELL", "WHEAT", 2] in orders
    assert ["SELL", "FERTILIZER", 2] in orders


def test_feed_reserve_counts_each_tile_embedded_animal_of_same_species():
    tiles = [[None for _ in range(5)] for _ in range(5)]
    tiles[0][0] = {"kind": "COOP", "animal": "GOOSE"}
    tiles[0][1] = {"kind": "COOP", "animal": "GOOSE"}
    state = {
        "day": 0, "hour": 1, "cash": 600,
        "tiles": tiles, "private": {"shed": {}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10}, "inventory": {"WHEAT": 10},
                    "params": {"WHEAT": {
                        "base": 10, "I0": 0, "T": 10,
                        "below_func": "linear", "below_target": 0,
                        "above_func": "linear", "above_target": 0,
                    }}},
    }

    orders = policy_module.build_market_orders(state, [])

    assert ["BUY_PRODUCT", "WHEAT", 60] in orders


def test_buy_product_uses_post_buy_quote_and_market_price_parameters():
    state = {
        "day": 4, "hour": 2, "cash": 150,
        "private": {"shed": {}, "seeds": {}},
        "market": {
            "inventory": {"WHEAT": 10}, "prices": {"WHEAT": 7},
            "params": {"WHEAT": {
                "base": 100, "I0": 10, "T": 10,
                "below_func": "linear", "below_target": 1,
                "above_func": "linear", "above_target": 1,
            }},
        },
    }

    orders = policy_module.build_market_orders(state, [["BUY_PRODUCT", "WHEAT", 2]])

    assert orders == [["BUY_PRODUCT", "WHEAT", 1]]


def test_multiple_product_orders_charge_total_unit_cost_only_once():
    params = {
        "WHEAT": {
            "base": 10, "I0": 0, "T": 10,
            "below_func": "linear", "below_target": 0,
            "above_func": "linear", "above_target": 0,
        },
        "FERTILIZER": {
            "base": 10, "I0": 0, "T": 10,
            "below_func": "linear", "below_target": 0,
            "above_func": "linear", "above_target": 0,
        },
    }
    state = {
        "cash": 30, "private": {"shed": {}, "seeds": {}},
        "market": {"inventory": {"WHEAT": 0, "FERTILIZER": 0}, "params": params},
    }

    orders = policy_module.build_market_orders(state, [
        ["BUY_PRODUCT", "WHEAT", 2], ["BUY_PRODUCT", "FERTILIZER", 1],
    ])

    assert ["BUY_PRODUCT", "WHEAT", 2] in orders
    assert ["BUY_PRODUCT", "FERTILIZER", 1] in orders


def test_policy_emits_explicit_macro_market_intents():
    intents = [
        (["BUY_ANIMAL", "GOOSE", 1], ["BUY_ANIMAL", "GOOSE", 1]),
        (["HIRE"], ["HIRE"]),
        (["BUY_LAND"], ["BUY_LAND"]),
        (["BUY_SEED", "CARROT", 1], ["BUY_SEED", "CARROT", 1]),
    ]

    for intent, expected in intents:
        obs = observation(day=4, hour=3, hands=[], money=5_000)
        obs["market_intents"] = [intent]

        assert expected in policy_module.Policy().act(obs)["market"]


def test_autonomous_macro_plan_covers_portfolio_and_growth_actions():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    board[0][1] = {"kind": "PLANT", "crop": "TOMATO", "planted_day": 0,
                   "yield_units": 1, "watered_today": True, "fertilized_until_day": -1}
    state = observation(day=4, hour=0, hands=[], tiles=board, money=5_000,
                        seeds={"TOMATO": 0}, market={"TOMATO": 90, "WHEAT": 25},
                        inventories=[[] for _ in range(1)])
    state["town"] = {"unlocked_shops": ["PIZZA_SHOP"]}

    macro = build_autonomous_macro_plan(state)
    kinds = [intent[0] for intent in macro["market_intents"]]

    assert macro["scenario_count"] == 20
    assert macro["portfolio"]["crop"] in {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
    assert {"BUY_LAND", "HIRE", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_SEED"} <= set(kinds)
    assert any(task.kind == "PLANT" and task.item == macro["portfolio"]["crop"] for task in macro["tasks"])


def test_autonomous_macro_portfolio_includes_and_can_select_strawberry():
    from kagriculture_agent.planner import _portfolio_scenarios, build_autonomous_macro_plan

    state = observation(day=4, hour=0, hands=[], tiles=[[None for _ in range(5)] for _ in range(5)],
                       money=5_000, seeds={}, inventories=[[]])
    state["market"] = {
        "prices": {"WHEAT": 1, "CARROT": 1, "TOMATO": 1, "STRAWBERRY": 1_000, "MELON": 1},
        "inventory": {crop: 10_000 for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")},
    }
    state["town"] = {"unlocked_shops": ["BRUNCH_SPOT"]}

    scenarios = _portfolio_scenarios(state, 4)
    macro = build_autonomous_macro_plan(state)

    assert len(scenarios) == 20
    assert sum(candidate["crop"] == "STRAWBERRY" for candidate in scenarios) == 4
    assert macro["portfolio"]["crop"] == "STRAWBERRY"


def test_autonomous_macro_stages_wheat_for_stored_animal():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "PASTURE"}
    state = observation(day=4, hour=1, hands=[[4, 4]], tiles=board, money=2_000,
                        shed={"SHEEP": 1}, seeds={}, inventories=[[], []])

    macro = build_autonomous_macro_plan(state)

    assert any(
        intent[0] == "BUY_PRODUCT" and intent[1] == "WHEAT" and intent[2] > 0
        for intent in macro["market_intents"]
    )


def test_autonomous_macro_replenishes_wheat_for_placed_animal_feed():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "PASTURE", "animal": "SHEEP", "fed_today": False}
    state = observation(day=4, hour=1, hands=[], tiles=board, money=2_000,
                        shed={}, seeds={}, inventories=[[]])

    macro = build_autonomous_macro_plan(state)

    assert any(
        intent[0] == "BUY_PRODUCT" and intent[1] == "WHEAT" and intent[2] > 0
        for intent in macro["market_intents"]
    )


def test_autonomous_macro_does_not_buy_or_place_animal_without_full_remaining_feed_reserve():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    state = observation(day=0, hour=0, hands=[], tiles=board, money=600,
                        shed={}, seeds={}, inventories=[[]])
    state["town"] = {"unlocked_shops": ["BAKERY"]}
    state["market"] = {"prices": {"WHEAT": 25, "EGG": 50}, "inventory": {"WHEAT": 10_000}}

    macro = build_autonomous_macro_plan(state)

    assert not any(intent[0] == "BUY_ANIMAL" for intent in macro["market_intents"])
    assert not any(task.kind == "ANIMAL" for task in macro["tasks"])


def test_autonomous_macro_counts_shed_and_worker_wheat_before_animal_reserve_purchase():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    state = observation(day=0, hour=1, hands=[[1, 0]], tiles=board, money=1_000,
                        shed={"WHEAT": 10}, seeds={}, inventories=[[], {"WHEAT": 5}])
    state["town"] = {"unlocked_shops": ["BAKERY"]}
    state["market"] = {"prices": {"WHEAT": 25, "EGG": 50}, "inventory": {"WHEAT": 10_000}}

    macro = build_autonomous_macro_plan(state)

    wheat_orders = [intent for intent in macro["market_intents"]
                    if intent[0] == "BUY_PRODUCT" and intent[1] == "WHEAT"]
    animal_orders = [intent for intent in macro["market_intents"] if intent[0] == "BUY_ANIMAL"]
    assert wheat_orders == [["BUY_PRODUCT", "WHEAT", 15]]
    assert animal_orders == [["BUY_ANIMAL", "GOOSE", 1]]


def test_policy_autonomously_adapts_seed_and_animal_choices_to_shop_and_market_state():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    base = observation(day=4, hour=1, hands=[], tiles=[[{"kind": "PASTURE"} for _ in range(5)] for _ in range(5)],
                       money=2_000, seeds={}, inventories=[[]])
    base["town"] = {"unlocked_shops": ["BAKERY"]}
    base["market"] = {"prices": {"WHEAT": 5, "CARROT": 100, "TOMATO": 20, "MELON": 1},
                       "inventory": {"WHEAT": 10_000, "CARROT": 10_000, "TOMATO": 10_000, "MELON": 10_000}}
    demand_plan = build_autonomous_macro_plan(base)

    base["town"] = {"unlocked_shops": ["PIZZA_SHOP"]}
    base["market"]["prices"] = {"WHEAT": 100, "CARROT": 5, "TOMATO": 120, "MELON": 250}
    market_plan = build_autonomous_macro_plan(base)

    assert demand_plan["portfolio"]["crop"] != market_plan["portfolio"]["crop"]
    assert demand_plan["portfolio"]["animal"] == "GOOSE"


def test_policy_replans_autonomous_portfolio_when_live_market_changes():
    base = observation(day=4, hour=1, hands=[], tiles=[[None for _ in range(5)] for _ in range(5)],
                       money=2_000, seeds={}, inventories=[[]])
    base["town"] = {"unlocked_shops": ["BAKERY"]}
    base["market"] = {"prices": {"WHEAT": 5, "CARROT": 100, "TOMATO": 20, "MELON": 1},
                       "inventory": {"WHEAT": 10_000, "CARROT": 10_000, "TOMATO": 10_000, "MELON": 10_000}}
    policy = policy_module.Policy()
    policy.act(base)
    first_crop = policy.memory.diagnostics["portfolio"]["crop"]

    base["market"]["prices"] = {"WHEAT": 100, "CARROT": 5, "TOMATO": 120, "MELON": 250}
    base["town"] = {"unlocked_shops": ["PIZZA_SHOP"]}
    policy.act(base)

    assert policy.memory.diagnostics["portfolio"]["crop"] != first_crop


def test_policy_uses_autonomous_macro_orders_without_external_intents():
    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    obs = observation(day=4, hour=0, hands=[], tiles=board, money=5_000,
                      seeds={}, inventories=[[]])
    obs["town"] = {"unlocked_shops": ["BAKERY"]}

    action = policy_module.Policy().act(obs)

    assert any(order[0] == "BUY_LAND" for order in action["market"])
    assert any(order[0] == "HIRE" for order in action["market"])
    assert any(order[0] == "BUY_ANIMAL" for order in action["market"])
    assert len(action["market"]) <= 10


def test_autonomous_animal_task_carries_item_through_pickup_place_feed_care():
    from kagriculture_agent.planner import build_autonomous_macro_plan

    board = [[None for _ in range(5)] for _ in range(5)]
    board[0][0] = {"kind": "COOP"}
    obs = observation(day=4, hour=1, hands=[[4, 4]], tiles=board, money=2_000,
                      shed={"GOOSE": 1, "WHEAT": 1}, seeds={}, inventories=[[], []])
    obs["town"] = {"unlocked_shops": ["BAKERY"]}
    macro = build_autonomous_macro_plan(obs)
    animal_task = next(task for task in macro["tasks"] if task.kind == "ANIMAL")

    assert animal_task.item == "GOOSE"
    state = policy_module.parse_observation(obs)
    assignment = WorkerAssignment(1, animal_task)
    state["farm"]["hands"][0] = [2, 2]
    assert policy_module.worker_action(1, state, assignment) == ["PICKUP", "GOOSE", 1]
    state["farm"]["hands"][0] = [0, 0]
    state["private"]["inventories"] = [{}, {"GOOSE": 1, "WHEAT": 1}]
    assert policy_module.worker_action(1, state, assignment) == ["PLACE", "GOOSE", 1]

    state["private"]["inventories"] = [{}, {"GOOSE": 1}]
    state["private"]["shed"] = {}
    state["farm"]["hands"][0] = [0, 0]
    assert policy_module.worker_action(1, state, assignment) == ["PASS"]

    state["private"]["inventories"] = [{}, {"WHEAT": 1}]
    state["farm"]["tiles"][0][0] = {"kind": "COOP", "animal": "GOOSE", "fed_today": False, "cared_today": False}
    feed = WorkerAssignment(1, Task("FEED", Position(0, 0), 100, 4, 1))
    care = WorkerAssignment(1, Task("CARE", Position(0, 0), 95, 4, 1))
    assert policy_module.worker_action(1, state, feed) == ["FEED"]
    state["farm"]["tiles"][0][0]["fed_today"] = True
    assert policy_module.worker_action(1, state, care) == ["CARE"]


def test_buy_product_is_restricted_to_feed_and_fertilizer():
    state = {
        "cash": 1_000, "private": {"shed": {}, "seeds": {}},
        "market": {"prices": {"CARROT": 1, "FERTILIZER": 1}},
    }

    orders = policy_module.build_market_orders(state, [
        ["BUY_PRODUCT", "CARROT", 3], ["BUY_PRODUCT", "FERTILIZER", 2],
    ])

    assert ["BUY_PRODUCT", "CARROT", 3] not in orders
    assert ["BUY_PRODUCT", "FERTILIZER", 2] in orders


def test_policy_drops_sale_that_conflicts_with_same_turn_pickup():
    assert policy_module._remove_pickup_sale_conflicts(
        [["SELL", "WHEAT", 7], ["SELL", "MELON", 1]],
        {1: ["PICKUP", "WHEAT", 1]},
    ) == [["SELL", "MELON", 1]]


def test_hire_uses_fibonacci_cost_and_land_uses_next_fixed_quadrant_cost():
    hire = {
        "cash": 2, "farm": {"money": 2, "hires_today": 3},
        "private": {"shed": {}, "seeds": {}}, "market": {},
    }
    land = {
        "cash": 1_999, "farm": {"money": 1_999, "unlocked_quadrants": ["NW", "NE"]},
        "private": {"shed": {}, "seeds": {}}, "market": {},
    }
    fully_unlocked = {
        "cash": 10_000, "farm": {"money": 10_000, "unlocked_quadrants": ["NW", "NE", "SW", "SE"]},
        "private": {"shed": {}, "seeds": {}}, "market": {},
    }

    assert policy_module.build_market_orders(hire, [["HIRE"]]) == []
    hire["cash"] = hire["farm"]["money"] = 3
    assert policy_module.build_market_orders(hire, [["HIRE"]]) == [["HIRE"]]
    assert policy_module.build_market_orders(land, [["BUY_LAND"]]) == []
    land["cash"] = land["farm"]["money"] = 2_000
    assert policy_module.build_market_orders(land, [["BUY_LAND"]]) == [["BUY_LAND"]]
    assert policy_module.build_market_orders(fully_unlocked, [["BUY_LAND"]]) == []


def test_final_day_market_orders_liquidate_saleable_shed_inventory():
    state = {
        "day": 29,
        "hour": 23,
        "cash": 0,
        "private": {"shed": {"WHEAT": 4, "FERTILIZER": 2}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "FERTILIZER": 20}},
    }

    orders = policy_module.build_market_orders(state, [])

    assert orders == [["SELL", "FERTILIZER", 2], ["SELL", "WHEAT", 4]]


def test_final_turn_skips_feed_and_normal_purchases_before_liquidation():
    state = {
        "day": 29, "hour": 23, "cash": 10,
        "animals": [{"species": "GOOSE"}],
        "private": {"shed": {"EGG": 1}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "EGG": 20}},
    }

    orders = policy_module.build_market_orders(state, [["BUY_SEED", "WHEAT", 1]])

    assert all(order[0] == "SELL" for order in orders)
    assert ["SELL", "EGG", 1] in orders


def test_market_does_not_buy_into_a_full_shed():
    state = {
        "day": 4, "hour": 2, "cash": 1_000,
        "animals": [{"species": "GOOSE"}],
        "private": {"shed": {"CARROT": 99}, "seeds": {}},
        "market": {"prices": {"WHEAT": 10, "FERTILIZER": 10}},
    }

    orders = policy_module.build_market_orders(state, [
        ["BUY_PRODUCT", "WHEAT", 1], ["BUY_PRODUCT", "FERTILIZER", 1],
    ])

    assert sum(order[2] for order in orders if order[0] == "BUY_PRODUCT") <= 1


def test_market_orders_clamp_animal_buys_to_total_strategy_cap():
    from kagriculture_agent.strategy import StrategySpec

    state = {
        "day": 4, "hour": 2, "cash": 10_000,
        "tiles": [[{"kind": "COOP", "animal": {"species": "COW"}}]],
        "private": {"shed": {"COW": 1}, "seeds": {}},
        "market": {"prices": {"WHEAT": 1}},
    }
    strategy = StrategySpec("three-animals", ("WHEAT",), ("COW",), 10, 3, 0)

    orders = policy_module.build_market_orders(state, [
        ["BUY_ANIMAL", "COW", 3],
        ["BUY_ANIMAL", "COW", 2],
    ], strategy)

    assert [order for order in orders if order[:2] == ["BUY_ANIMAL", "COW"]] == [
        ["BUY_ANIMAL", "COW", 1],
    ]


def test_policy_rejects_explicit_animal_intent_when_existing_units_reach_cap():
    board = [[None for _ in range(5)] for _ in range(5)]
    for index in range(12):
        board[index // 5][index % 5] = {"kind": "COOP", "animal": {"species": "COW"}}
    obs = observation(
        day=28, hour=1, hands=[], tiles=board, seeds={"WHEAT": 1},
        shed={"WHEAT": 30}, inventories=[[]], money=10_000,
    )
    obs["market_intents"] = [["BUY_ANIMAL", "COW", 2]]

    action = policy_module.Policy(strategy="premium").act(obs)

    assert not any(order[:2] == ["BUY_ANIMAL", "COW"] for order in action["market"])


def test_policy_handles_midseason_full_shed_animal_and_locked_observations():
    tiles = [[None for _ in range(5)] for _ in range(5)]
    tiles[0][2] = {"kind": "COOP", "animal": "GOOSE", "fed_today": False,
                   "cared_today": False, "yield_units": 1}
    tiles[0][3] = "LOCKED"
    obs = observation(day=12, hour=5, tiles=tiles, hands=[[1, 0]],
                      shed={"WHEAT": 100, "EGG": 3}, inventories=[["WHEAT"], []])

    action = policy_module.Policy().act(obs)

    assert len(action["hands"]) == 1
    assert all(command_is_legal(command) for command in [action["farmer"], *action["hands"]])
    assert all(order[0] in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL", "HIRE", "BUY_LAND"}
               for order in action["market"])
    wheat_sales = [order for order in action["market"] if order[:2] == ["SELL", "WHEAT"]]
    assert all(order[2] <= 83 for order in wheat_sales)
    assert all(command != "WATER" for command in [action["farmer"], *action["hands"]])


def test_worker_action_moves_and_only_acts_on_current_valid_unlocked_tile():
    state = {
        "board_size": 3,
        "tiles": [[None, {"kind": "PLANT", "crop": "WHEAT", "needs_water": True}, "LOCKED"],
                  [None, None, None], [None, None, None]],
        "workers": [{"index": 0, "role": "FARMER", "position": [0, 0]}],
        "private": {"inventories": [[]], "shed": {}, "seeds": {"WHEAT": 1}},
    }
    assignment = WorkerAssignment(0, Task("WATER", Position(1, 0), 100, 0, 1))

    assert policy_module.worker_action(0, state, assignment) == ["EAST"]
    state["workers"][0]["position"] = [1, 0]
    assert policy_module.worker_action(0, state, assignment) == ["WATER"]

    assignment.task = Task("WATER", Position(2, 0), 100, 0, 1)
    state["workers"][0]["position"] = [2, 0]
    assert policy_module.worker_action(0, state, assignment) == ["PASS"]


def test_worker_action_does_not_feed_without_wheat_or_plant_without_seed():
    state = {
        "board_size": 2,
        "tiles": [[{"kind": "ANIMAL", "animal": {"species": "GOOSE", "needs_feed": True}}, None],
                  [None, None]],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [[]], "shed": {}, "seeds": {}},
    }
    feed = WorkerAssignment(0, Task("FEED", Position(0, 0), 100, 0, 1))
    plant = WorkerAssignment(0, Task("PLANT", Position(0, 1), 20, None, 1))

    assert policy_module.worker_action(0, state, feed) == ["PASS"]
    state["workers"][0]["position"] = [0, 1]
    assert policy_module.worker_action(0, state, plant) == ["PASS"]


def test_worker_action_handles_shed_inventory_prerequisites_safely():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {"inventories": [["WHEAT"]], "shed": {"WHEAT": 2}, "seeds": {}},
    }
    pickup = WorkerAssignment(0, Task("PICKUP", "WHEAT", 10, None, 1))
    drop = WorkerAssignment(0, Task("DROP", Position(2, 2), 10, None, 1))

    assert policy_module.worker_action(0, state, pickup) == ["PICKUP", "WHEAT", 1]
    state["workers"][0]["position"] = [2, 2]
    assert policy_module.worker_action(0, state, drop) == ["DROP"]


def test_feed_fetches_wheat_from_shed_before_emitting_feed():
    state = {
        "board_size": 5,
        "tiles": [[{"kind": "COOP", "animal": "GOOSE"}] + [None for _ in range(4)]] +
                  [[None for _ in range(5)] for _ in range(4)],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [[]], "shed": {"WHEAT": 1}, "seeds": {}},
    }
    assignment = WorkerAssignment(0, Task("FEED", Position(0, 0), 100, 0, 1))

    assert policy_module.worker_action(0, state, assignment) != ["PASS"]
    state["workers"][0]["position"] = [1, 1]
    assert policy_module.worker_action(0, state, assignment) == ["PICKUP", "WHEAT", 1]


def test_fertilize_and_animal_place_fetch_required_items_from_shed():
    plant_state = {
        "board_size": 5,
        "tiles": [[{"kind": "PLANT", "crop": "WHEAT"}] + [None for _ in range(4)]] +
                  [[None for _ in range(5)] for _ in range(4)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {"inventories": [[]], "shed": {"FERTILIZER": 1}, "seeds": {}},
    }
    fertilize = WorkerAssignment(0, Task("FERTILIZE", Position(0, 0), 100, 0, 1))
    assert policy_module.worker_action(0, plant_state, fertilize) == ["PICKUP", "FERTILIZER", 1]

    animal_state = {
        "board_size": 5,
        "tiles": [[{"kind": "COOP", "animal": None}] + [None for _ in range(4)]] +
                  [[None for _ in range(5)] for _ in range(4)],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [[]], "shed": {"GOOSE": 1}, "seeds": {}},
    }
    place = WorkerAssignment(0, Task("PLACE", "GOOSE", 100, None, 1))
    assert policy_module.worker_action(0, animal_state, place) != ["PASS"]
    animal_state["workers"][0]["position"] = [1, 1]
    assert policy_module.worker_action(0, animal_state, place) == ["PICKUP", "GOOSE", 1]

    animal_state["workers"][0]["position"] = [0, 0]
    animal_state["desired_animals"] = [{"position": [0, 0], "species": "GOOSE"}]
    planner_place = WorkerAssignment(0, Task("ANIMAL", Position(0, 0), 100, None, 1))
    assert policy_module.worker_action(0, animal_state, planner_place) != ["PASS"]
    animal_state["workers"][0]["position"] = [1, 1]
    assert policy_module.worker_action(0, animal_state, planner_place) == ["PICKUP", "GOOSE", 1]


def test_policy_drops_carried_goods_at_shed_access_before_more_work():
    obs = observation(day=4, hour=3, hands=[], inventories=[["WHEAT"]], shed={})
    obs["farms"][0]["farmer"] = [0, 0]

    policy = policy_module.Policy()
    first = policy.act(obs)
    assert first["farmer"] == ["EAST"]

    obs["farms"][0]["farmer"] = [1, 1]
    assert policy.act(obs)["farmer"] == ["DROP"]


def test_water_and_fertilize_require_engine_plant_tile_shape():
    state = {
        "board_size": 2,
        "tiles": [[{"kind": "WHEAT", "needs_water": True}, None], [None, None]],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [["FERTILIZER"]], "shed": {}, "seeds": {}},
    }
    water = WorkerAssignment(0, Task("WATER", Position(0, 0), 10, None, 1))
    fertilize = WorkerAssignment(0, Task("FERTILIZE", Position(0, 0), 10, None, 1))
    assert policy_module.worker_action(0, state, water) == ["PASS"]
    assert policy_module.worker_action(0, state, fertilize) == ["PASS"]

    state["tiles"][0][0] = {"kind": "PLANT", "crop": "WHEAT", "needs_water": True}
    assert policy_module.worker_action(0, state, water) == ["WATER"]
    assert policy_module.worker_action(0, state, fertilize) == ["FERTILIZE"]


def test_animal_actions_require_animal_field_with_valid_species():
    state = {
        "board_size": 2,
        "tiles": [[{"kind": "ANIMAL", "species": "GOOSE"}, None], [None, None]],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [["WHEAT"]], "shed": {}, "seeds": {}},
    }
    feed = WorkerAssignment(0, Task("FEED", Position(0, 0), 10, None, 1))
    assert policy_module.worker_action(0, state, feed) == ["PASS"]

    state["tiles"][0][0] = {"kind": "COOP", "animal": "UNKNOWN"}
    assert policy_module.worker_action(0, state, feed) == ["PASS"]

    state["tiles"][0][0] = {"kind": "COOP", "animal": "GOOSE"}
    assert policy_module.worker_action(0, state, feed) == ["FEED"]


def test_targetless_pickup_invalidates_when_shed_quantity_is_consumed():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {"inventories": [[]], "shed": {"WHEAT": 1}, "seeds": {}},
    }
    assignment = WorkerAssignment(0, {"kind": "PICKUP", "target": None, "item": "WHEAT"})

    assert policy_module._assignment_valid(state, assignment)
    state["private"]["shed"]["WHEAT"] = 0
    assert not policy_module._assignment_valid(state, assignment)


def test_targetless_place_invalidates_when_worker_inventory_is_consumed():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {"inventories": [["WHEAT"]], "shed": {}, "seeds": {}},
    }
    assignment = WorkerAssignment(0, {"kind": "PLACE", "target": None, "item": "WHEAT"})

    assert policy_module._assignment_valid(state, assignment)
    state["private"]["inventories"][0] = []
    assert not policy_module._assignment_valid(state, assignment)


def test_targetless_place_invalidates_on_occupied_incompatible_structure():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {"inventories": [["COW"]], "shed": {}, "seeds": {}},
    }
    state["workers"][0]["position"] = [0, 0]
    state["tiles"][0][0] = {"kind": "COOP", "animal": "GOOSE"}
    assignment = WorkerAssignment(0, {"kind": "PLACE", "target": None, "item": "COW"})

    assert not policy_module._assignment_valid(state, assignment)


def test_place_requires_matching_empty_animal_structure_or_shed_adjacency():
    tiles = [[{"kind": "COOP", "animal": None}, {"kind": "PASTURE", "animal": None},
              {"kind": "COOP", "animal": "GOOSE"}, None, None]]
    state = {
        "board_size": 5, "tiles": tiles,
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [["COW", "GOOSE", "WHEAT"]], "shed": {}, "seeds": {}},
    }

    cow_on_coop = WorkerAssignment(0, Task("PLACE", "COW", 10, None, 1))
    goose_on_pasture = WorkerAssignment(0, Task("PLACE", "GOOSE", 10, None, 1))
    occupied_coop = WorkerAssignment(0, Task("PLACE", "GOOSE", 10, None, 1))
    wheat_on_structure = WorkerAssignment(0, Task("PLACE", "WHEAT", 10, None, 1))

    assert policy_module.worker_action(0, state, cow_on_coop) == ["PASS"]
    state["workers"][0]["position"] = [1, 0]
    assert policy_module.worker_action(0, state, goose_on_pasture) == ["PASS"]
    state["workers"][0]["position"] = [2, 0]
    assert policy_module.worker_action(0, state, occupied_coop) == ["PASS"]
    state["workers"][0]["position"] = [0, 0]
    assert policy_module.worker_action(0, state, wheat_on_structure) == ["PASS"]
    state["workers"][0]["position"] = [1, 1]
    assert policy_module.worker_action(0, state, WorkerAssignment(0, Task("PLACE", "WHEAT", 10, None, 1))) == ["PLACE", "WHEAT", 1]


def test_build_and_dig_reject_locked_tiles_and_occupied_animal_structures():
    state = {
        "board_size": 3,
        "tiles": [[{"kind": "LOCKED", "locked": True}, {"kind": "COOP", "animal": "GOOSE"}, None],
                  [None, None, None], [None, None, None]],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [[]], "shed": {}, "seeds": {}},
    }

    assert policy_module.worker_action(0, state, WorkerAssignment(0, Task("BUILD_COOP", Position(0, 0), 10, None, 1))) == ["PASS"]
    state["workers"][0]["position"] = [1, 0]
    assert policy_module.worker_action(0, state, WorkerAssignment(0, Task("DIG", Position(1, 0), 10, None, 1))) == ["PASS"]


def test_all_tile_actions_reject_mapping_shaped_locked_tiles():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [["WHEAT", "FERTILIZER"]], "shed": {}, "seeds": {}},
    }
    state["tiles"][0][0] = {
        "kind": "LOCKED", "locked": True, "structure": {"kind": "COOP"},
    }
    locked_place = WorkerAssignment(0, Task("PLACE", "WHEAT", 10, None, 1))
    locked_fertilize = WorkerAssignment(0, Task("FERTILIZE", Position(0, 0), 10, None, 1))
    locked_weed = WorkerAssignment(0, Task("WEED", Position(0, 0), 10, None, 1))
    locked_dig = WorkerAssignment(0, Task("DIG", Position(0, 0), 10, None, 1))
    locked_build = WorkerAssignment(0, Task("BUILD_COOP", Position(0, 0), 10, None, 1))

    assert policy_module.worker_action(0, state, locked_place) == ["PASS"]
    assert policy_module.worker_action(0, state, locked_fertilize) == ["PASS"]
    state["tiles"][0][0]["kind"] = "WEED"
    assert policy_module.worker_action(0, state, locked_weed) == ["PASS"]
    assert policy_module.worker_action(0, state, locked_dig) == ["PASS"]
    assert policy_module.worker_action(0, state, locked_build) == ["PASS"]


def test_locked_shed_access_allows_place_into_shed():
    state = {
        "board_size": 5,
        "tiles": [[None for _ in range(5)] for _ in range(5)],
        "workers": [{"index": 0, "position": [1, 1]}],
        "private": {"inventories": [["WHEAT"]], "shed": {}, "seeds": {}},
    }
    state["tiles"][1][1] = {"kind": "LOCKED", "locked": True}

    placement = WorkerAssignment(0, Task("PLACE", "WHEAT", 10, None, 1))

    assert policy_module.worker_action(0, state, placement) == ["PLACE", "WHEAT", 1]


def test_completed_feed_and_care_assignments_fall_back_to_pass():
    state = {
        "board_size": 2,
        "tiles": [[{"kind": "ANIMAL", "animal": {
            "species": "GOOSE", "fed_today": True, "cared_today": True,
        }}, None], [None, None]],
        "workers": [{"index": 0, "position": [0, 0]}],
        "private": {"inventories": [["WHEAT"]], "shed": {}, "seeds": {}},
    }

    feed = WorkerAssignment(0, Task("FEED", Position(0, 0), 10, None, 1))
    care = WorkerAssignment(0, Task("CARE", Position(0, 0), 10, None, 1))

    assert policy_module.worker_action(0, state, feed) == ["PASS"]
    assert policy_module.worker_action(0, state, care) == ["PASS"]
    assert not policy_module._assignment_valid(state, feed)
    assert not policy_module._assignment_valid(state, care)


def test_policy_resets_memory_on_backward_time_and_new_episode():
    policy = policy_module.Policy()
    first = observation(day=3, hour=4)
    policy.act(first)
    assert policy.memory.assignments

    policy.act(observation(day=2, hour=23))
    assert policy.memory.last_day == 2
    assert policy.memory.last_hour == 23
    assert policy.memory.diagnostics["reset_reason"] == "time_backward"
    assert policy.memory.sell_batches == []

    policy.act(observation(day=0, hour=0))
    assert policy.memory.last_day == 0
    assert policy.memory.last_hour == 0
    assert policy.memory.diagnostics.get("reset_reason") == "episode_start"


def test_policy_output_is_deterministic_for_repeated_observation():
    obs = observation(day=10, hour=12, shed={"WHEAT": 4})
    first = policy_module.Policy().act(deepcopy(obs))
    second = policy_module.Policy().act(deepcopy(obs))

    assert first == second


def test_policy_same_instance_repeated_observation_is_deterministic():
    obs = observation(day=10, hour=12, shed={"WHEAT": 4})
    policy = policy_module.Policy()

    first = policy.act(deepcopy(obs))
    second = policy.act(deepcopy(obs))

    assert first == second


def test_policy_memory_reset_is_safe_for_backward_day_or_hour():
    assert hasattr(policy_module, "PolicyMemory")
    memory = policy_module.PolicyMemory(last_day=4, last_hour=8, assignments=["stale"], sell_batches=["stale"], diagnostics={"x": 1})

    assert memory.observe_time(4, 7) is True
    assert memory.assignments == []
    assert memory.sell_batches == []
    assert memory.diagnostics["reset_reason"] == "time_backward"


def test_policy_memory_clears_per_day_work_at_hour_zero():
    memory = policy_module.PolicyMemory(
        last_day=4, last_hour=23, assignments=["stale"],
        sell_batches=["stale"], diagnostics={"x": 1}, market_regime={"WHEAT": "glut"},
    )

    assert memory.observe_time(5, 0) is True
    assert memory.assignments == []
    assert memory.sell_batches == []
    assert memory.market_regime == {}
    assert memory.diagnostics["reset_reason"] == "day_start"


def test_policy_memory_reset_clears_selected_strategy():
    memory = policy_module.PolicyMemory(selected_strategy="premium")

    memory.reset()

    assert memory.selected_strategy is None
