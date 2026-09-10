"""Independent regression coverage for the policy-improvement plan.

These cases deliberately use small engine-shaped mappings instead of helpers
from other test modules, so each failure identifies a real policy contract.
The private-helper cases are intentional: they exercise planner/policy
decisions before engine execution, where public integration setup is not
deterministic enough to isolate these regressions.
"""

from kagriculture_agent.planner import (
    _portfolio_scenarios,
    _task_turn_budget,
    assign_tasks,
    build_autonomous_macro_plan,
    build_daily_plan,
    normalize_planner_state,
)
from kagriculture_agent.policy import (
    Policy,
    _assignment_valid,
    _drop_carried_goods,
    _suppress_final_hour_fertilize_on_due_water,
    worker_action,
)
from kagriculture_agent.learned_policy import (
    DependencyFreePolicy,
    LearnedPolicy,
    PolicyProposal,
)
from kagriculture_agent.strategy import StrategySpec
from kagriculture_agent.types import Position, Task, WorkerAssignment


def _workers(*items):
    return [{"index": i, "role": role, "position": position} for i, (role, position) in enumerate(items)]


def _state(**values):
    return {
        "day": 1, "hour": 0, "board_size": 5, "tiles": [["EMPTY"] * 5 for _ in range(5)],
        "workers": _workers(("FARMER", Position(0, 0))),
        "private": {"seeds": {"WHEAT": 2}, "shed": {}, "inventories": []},
        "market": {}, "cash": 100,
        **values,
    }


def _policy_observation_with_one_melon_seed():
    board = [[None for _ in range(5)] for _ in range(5)]
    return {
        "player": 0,
        "day": 1,
        "hour": 0,
        "farms": [{
            "tiles": board,
            "farmer": [0, 0],
            "hands": [[1, 0]],
            "money": 0,
            "unlocked_quadrants": ["NW"],
        }],
        "private": {
            "shed": {},
            "seeds": {"MELON": 1},
            "inventories": [{}, {}],
        },
        "market": {
            "prices": {"MELON": 250},
            "inventory": {"MELON": 10_000},
        },
        "town": {"unlocked_shops": []},
    }


def _policy_observation_with_two_fertilizer_tasks(quantity):
    board = [[None for _ in range(10)] for _ in range(10)]
    for x in (0, 1):
        board[0][x] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": True,
            "fertilized_until_day": -1,
            "planted_day": 2,
            "yield_units": 0,
        }
    return {
        "player": 0,
        "day": 2,
        "hour": 23,
        "farms": [{
            "tiles": board,
            "farmer": [5, 4],
            "hands": [[4, 4]],
            "money": 0,
            "unlocked_quadrants": ["NW"],
        }],
        "private": {
            "shed": {"FERTILIZER": quantity},
            "seeds": {},
            "inventories": [{}, {}],
        },
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def test_assign_tasks_uses_reserved_worker_for_only_one_fallback_task():
    state = _state(workers=_workers(("FARMER", Position(0, 0))),
                   private={"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{"GOOSE": 1}]})
    tasks = [Task("ANIMAL", Position(0, 1), 100, 1, 1, item="GOOSE"),
             Task("WATER", Position(4, 3), 100, 1, 1, item="WHEAT"),
             Task("SHED", Position(1, 1), 75, None, 1)]
    assignments = assign_tasks(tasks, state["workers"], state)
    assert [(assignment.worker_index, assignment.task.kind) for assignment in assignments] == [
        (0, "ANIMAL"),
    ]


def test_assign_tasks_routes_equal_deadlines_to_nearest_task_first():
    state = _state(
        day=2,
        hour=15,
        workers=_workers(("FARMER", Position(4, 0))),
    )
    tasks = [
        Task("WATER", Position(0, 1), 100, 2, 1, item="MELON"),
        Task("WATER", Position(4, 1), 100, 2, 1, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert len(assignments) == 1
    assert assignments[0].task.target == Position(4, 1)


def test_assign_tasks_does_not_strand_far_due_need_behind_nearer_work():
    state = _state(
        day=8,
        hour=16,
        workers=_workers(
            ("FARMER", Position(1, 2)),
            ("HAND", Position(0, 2)),
        ),
    )
    tasks = [
        Task("WATER", Position(4, 0), 100, 8, 1, item="MELON"),
        Task("WATER", Position(0, 3), 100, 8, 1, item="MELON"),
        Task("WATER", Position(1, 3), 100, 8, 1, item="MELON"),
        Task("WATER", Position(1, 4), 100, 8, 1, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert Position(4, 0) in {assignment.task.target for assignment in assignments}


def test_assign_tasks_routes_equal_priority_deadlines_before_task_kind():
    state = _state(
        day=2,
        hour=0,
        workers=_workers(("FARMER", Position(4, 4))),
        private={"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{}]},
    )
    tasks = [
        Task("FEED", Position(0, 0), 100, 2, 1, item="SHEEP"),
        Task("WATER", Position(4, 3), 100, 2, 1, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert len(assignments) == 1
    assert assignments[0].task.kind == "WATER"


def test_assign_tasks_preserves_only_feed_worker_when_water_has_another_candidate():
    state = _state(
        day=2,
        hour=20,
        workers=_workers(
            ("FARMER", Position(2, 0)),
            ("HAND", Position(0, 0)),
        ),
        private={"seeds": {}, "shed": {}, "inventories": [{}, {"WHEAT": 1}]},
    )
    tasks = [
        Task("WATER", Position(0, 1), 100, 2, 1, item="MELON"),
        Task("FEED", Position(0, 2), 100, 2, 1, item="SHEEP"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert [(assignment.worker_index, assignment.task.kind) for assignment in assignments] == [
        (0, "WATER"),
        (1, "FEED"),
    ]


def test_assign_tasks_keeps_due_feed_when_workers_cannot_cover_all_due_needs():
    state = _state(
        day=16,
        hour=1,
        board_size=10,
        workers=_workers(
            ("FARMER", Position(4, 4)),
            ("HAND", Position(5, 4)),
        ),
        private={"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{}, {}]},
    )
    tasks = [
        Task("FEED", Position(0, 0), 100, 16, 1, item="SHEEP"),
        Task("WATER", Position(7, 0), 100, 16, 1, item="MELON"),
        Task("WATER", Position(8, 0), 100, 16, 1, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert sorted(assignment.task.kind for assignment in assignments) == ["FEED", "WATER"]


def test_assign_tasks_keeps_higher_priority_maintenance_before_route_distance():
    state = _state(
        day=2,
        hour=0,
        workers=_workers(("FARMER", Position(4, 4))),
        private={"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{}]},
    )
    tasks = [
        Task("FEED", Position(0, 0), 101, 2, 1, item="SHEEP"),
        Task("WATER", Position(4, 3), 100, 2, 1, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert len(assignments) == 1
    assert assignments[0].task.kind == "FEED"


def test_assign_tasks_allocates_plant_tasks_within_available_seed_inventory():
    state = _state(
        workers=_workers(
            ("FARMER", Position(0, 0)),
            ("HAND", Position(1, 0)),
        ),
        private={"seeds": {"MELON": 1}, "shed": {}, "inventories": [{}, {}]},
    )
    tasks = [
        Task("PLANT", Position(0, 0), 20, 1, 10, item="MELON"),
        Task("PLANT", Position(1, 0), 20, 1, 10, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert [(assignment.worker_index, assignment.task.target) for assignment in assignments] == [
        (0, Position(0, 0)),
    ]


def test_assign_tasks_allocates_seed_only_after_plant_is_feasible():
    state = _state(
        day=1,
        hour=22,
        workers=_workers(("FARMER", Position(0, 0))),
        private={"seeds": {"MELON": 1}, "shed": {}, "inventories": [{}]},
    )
    tasks = [
        Task("PLANT", Position(1, 0), 30, 1, 10, item="MELON"),
        Task("PLANT", Position(0, 0), 20, 1, 10, item="MELON"),
    ]

    assignments = assign_tasks(tasks, state["workers"], state)

    assert [(assignment.worker_index, assignment.task.target) for assignment in assignments] == [
        (0, Position(0, 0)),
    ]


def test_policy_limits_same_turn_plant_commands_to_available_seeds():
    action = Policy().act(_policy_observation_with_one_melon_seed())
    commands = [action["farmer"], *action["hands"]]

    assert sum(command == ["PLANT", "MELON"] for command in commands) == 1


@__import__("pytest").mark.parametrize(("quantity", "expected"), [(1, 1), (2, 2)])
def test_policy_limits_same_turn_pickups_to_available_shed_quantity(quantity, expected):
    action = Policy().act(_policy_observation_with_two_fertilizer_tasks(quantity))
    commands = [action["farmer"], *action["hands"]]

    assert sum(command[:2] == ["PICKUP", "FERTILIZER"] for command in commands) == expected


@__import__("pytest").mark.parametrize("shed,carried,expected", [
    ({"CARROT": 99}, [{"MELON": 2}], "PLACE MELON 1"),
    ({"CARROT": 100}, [{"MELON": 1}], "PASS"),
    ({"CARROT": 99}, [{"MELON": 1, "EGG": 1}], "PLACE MELON 1"),
    ({"CARROT": 100}, [{"GOOSE": 1}], "PASS"),
])
def test_drop_carried_goods_is_capacity_safe_without_mutating_state(shed, carried, expected):
    state = _state(private={"shed": shed, "inventories": carried})
    shed_before = dict(state["private"]["shed"])
    inventory_before = dict(carried[0])
    action = _drop_carried_goods(state, 0, None, Position(1, 1), force=True)
    assert action == expected
    assert state["private"]["shed"] == shed_before
    assert state["private"]["inventories"][0] == inventory_before


def test_drop_carried_goods_honors_configured_shed_capacity():
    state = _state(
        configuration={"shedCapacity": 2},
        private={"shed": {"CARROT": 1}, "inventories": [{"MELON": 2}]},
    )

    action = _drop_carried_goods(state, 0, None, Position(1, 1), force=True)

    assert action == "PLACE MELON 1"


def test_day_27_does_not_plan_melon_purchase_or_planting():
    tiles = [["EMPTY"] * 5 for _ in range(5)]
    tiles[2][3] = {"kind": "PLANT", "crop": "WHEAT", "needs_water": True}
    state = _state(
        day=27,
        tiles=tiles,
        private={"seeds": {"MELON": 1}, "shed": {}, "inventories": [{}]},
    )
    plan = build_daily_plan(state)
    assert plan
    assert any(task.kind == "WATER" and task.item == "WHEAT" for task in plan)
    assert all(task.item != "MELON" or task.kind not in {"BUY_SEED", "PLANT"} for task in plan)

    no_seed_state = _state(
        day=27,
        tiles=[["EMPTY"] * 5 for _ in range(5)],
        private={"seeds": {}, "shed": {}, "inventories": [{}]},
        cash=1_000,
    )
    melon_only = StrategySpec("melon-only", ("MELON",), (), 10, 0, 0)
    macro = build_autonomous_macro_plan(no_seed_state, strategy=melon_only)
    assert macro["scenario_count"] == 0
    assert ["BUY_SEED", "MELON", 1] not in macro["market_intents"]
    assert not any(task.kind == "PLANT" and task.item == "MELON" for task in macro["tasks"])


@__import__("pytest").mark.parametrize("crop", ["WHEAT", "CARROT"])
def test_day_27_keeps_short_horizon_crops_available(crop):
    state = _state(
        day=27,
        private={"seeds": {crop: 1}, "shed": {}, "inventories": [{}]},
        cash=1_000,
    )
    crop_only = StrategySpec(f"{crop.lower()}-only", (crop,), (), 10, 0, 0)

    assert {scenario["crop"] for scenario in _portfolio_scenarios(state, 27, crop_only)} == {crop}
    assert any(task.kind == "PLANT" and task.item == crop
               for task in build_daily_plan(state, strategy=crop_only))

    state["private"]["seeds"] = {}
    macro = build_autonomous_macro_plan(state, strategy=crop_only)
    assert ["BUY_SEED", crop, 1] in macro["market_intents"]
    assert any(task.kind == "PLANT" and task.item == crop for task in macro["tasks"])


def test_day_28_rejects_new_crop_purchase_and_planting():
    wheat_only = StrategySpec("wheat-only", ("WHEAT",), (), 10, 0, 0)
    seeded = _state(
        day=28,
        private={"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
        cash=1_000,
    )
    no_seed = _state(
        day=28,
        private={"seeds": {}, "shed": {}, "inventories": [{}]},
        cash=1_000,
    )

    assert not any(
        task.kind == "PLANT"
        for task in build_daily_plan(seeded, strategy=wheat_only)
    )
    macro = build_autonomous_macro_plan(no_seed, strategy=wheat_only)
    assert ["BUY_SEED", "WHEAT", 1] not in macro["market_intents"]
    assert not any(task.kind == "PLANT" for task in macro["tasks"])


def test_existing_late_season_crops_and_animals_keep_actionable_work():
    tiles = [["EMPTY"] * 5 for _ in range(5)]
    tiles[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 20,
        "yield_units": 1,
        "needs_water": True,
    }
    tiles[0][1] = {
        "kind": "COOP",
        "animal": {"species": "GOOSE", "needs_feed": True, "needs_care": True},
    }

    plan = build_daily_plan(_state(day=29, tiles=tiles))

    assert {("WATER", "WHEAT"), ("HARVEST", "WHEAT"),
            ("FEED", "GOOSE"), ("CARE", "GOOSE")} <= {
        (task.kind, task.item) for task in plan
    }


def test_late_animal_horizon_includes_structure_build_and_placement_turns():
    goose_only = StrategySpec("goose-only", (), ("GOOSE",), 0, 1, 0)

    def macro(day):
        return build_autonomous_macro_plan(
            _state(
                day=day,
                tiles=[["EMPTY"] * 5 for _ in range(5)],
                private={"seeds": {}, "shed": {"WHEAT": 7}, "inventories": [{}]},
                cash=5_000,
            ),
            strategy=goose_only,
        )

    last_productive = macro(25)
    assert ["BUY_ANIMAL", "GOOSE", 1] in last_productive["market_intents"]
    assert any(task.kind == "BUILD_COOP" for task in last_productive["tasks"])

    too_late = macro(26)
    assert ["BUY_ANIMAL", "GOOSE", 1] not in too_late["market_intents"]
    assert not any(task.kind in {"ANIMAL", "BUILD_COOP"} for task in too_late["tasks"])


def test_late_daily_animal_and_structure_tasks_are_rejected():
    state = _state(
        day=26,
        desired_animals=[{"species": "GOOSE", "position": [0, 0], "owned": False}],
        structures=[{"kind": "COOP", "position": [1, 0], "built": False}],
    )

    plan = build_daily_plan(state)

    assert not any(task.kind in {"ANIMAL", "STRUCTURE"} for task in plan)


@__import__("pytest").mark.parametrize(
    ("animal", "expected"),
    [("GOOSE", False), ("COW", True)],
)
def test_late_unbuilt_structure_requires_compatible_allowed_animal(animal, expected):
    state = _state(
        day=19,
        structures=[{"kind": "PASTURE", "position": [1, 0], "built": False}],
    )
    strategy = StrategySpec(f"{animal.lower()}-only", (), (animal,), 0, 1, 0)

    has_structure_task = any(
        task.kind == "STRUCTURE"
        for task in build_daily_plan(state, strategy=strategy)
    )

    assert has_structure_task is expected


def test_macro_rejects_unbuilt_structure_incompatible_with_allowed_animal():
    pasture = {
        "kind": "STRUCTURE",
        "structure": {"kind": "PASTURE", "built": False},
    }
    state = _state(
        day=19,
        tiles=[[pasture] * 5 for _ in range(5)],
        structures=[{"kind": "PASTURE", "position": [0, 0], "built": False}],
        private={"seeds": {}, "shed": {"WHEAT": 11}, "inventories": [{}]},
        cash=5_000,
    )
    goose_only = StrategySpec("goose-only", (), ("GOOSE",), 0, 1, 0)

    macro = build_autonomous_macro_plan(state, strategy=goose_only)

    assert ["BUY_ANIMAL", "GOOSE", 1] not in macro["market_intents"]
    assert not any(
        task.kind in {"STRUCTURE", "BUILD_COOP", "BUILD_PASTURE"}
        for task in macro["tasks"]
    )


def test_shed_assignment_prefers_worker_with_inventory():
    workers = _workers(("FARMER", Position(0, 0)), ("WORKER", Position(0, 0)))
    task = Task("SHED", Position(1, 0), 1, None, 1)
    with_inventory = _state(workers=workers,
                             private={"seeds": {}, "shed": {}, "inventories": [{}, {"EGG": 1}]})
    without_inventory = _state(workers=workers,
                                private={"seeds": {}, "shed": {}, "inventories": [{"EGG": 1}, {}]})
    selected_with = assign_tasks([task], workers, with_inventory)[0].worker_index
    selected_without = assign_tasks([task], workers, without_inventory)[0].worker_index
    assert selected_with == 1
    assert selected_without == 0
    assert selected_with != selected_without


def test_shed_assignment_prefers_farther_carrying_worker_before_distance():
    workers = _workers(("FARMER", Position(1, 0)), ("WORKER", Position(4, 4)))
    state = _state(
        workers=workers,
        private={"seeds": {}, "shed": {}, "inventories": [{}, {"EGG": 1}]},
    )

    assignment = assign_tasks(
        [Task("SHED", Position(1, 0), 1, None, 1)], workers, state,
    )[0]

    assert assignment.worker_index == 1


def test_animal_budget_includes_feed_after_shed_pickup():
    state = _state(private={"shed": {"WHEAT": 1}, "inventories": [{"GOOSE": 1}]})
    task = Task("ANIMAL", Position(4, 4), 1, 1, 1, item="GOOSE")
    assert _task_turn_budget(task, (0, "FARMER", Position(0, 0)), state, 5) == 11


def test_normalize_planner_state_never_uses_hands_as_seeds():
    competing = normalize_planner_state({
        "seeds": {}, "farm": {"seeds": {"WHEAT": 3}, "hands": {"WHEAT": 5}},
        "private": {"seeds": {"WHEAT": 7}},
    })
    fallback = normalize_planner_state({
        "farm": {"hands": {"WHEAT": 5}}, "private": {"seeds": {"WHEAT": 7}},
    })
    assert competing["seeds"] == {}
    assert fallback["seeds"] == {"WHEAT": 7}


def test_sell_assignment_validates_carried_inventory():
    workers = _workers(("FARMER", Position(1, 1)))
    carried = _state(
        workers=workers,
        private={"shed": {}, "inventories": [{"EGG": 1}]},
    )
    empty = _state(
        workers=workers,
        private={"shed": {"EGG": 1}, "inventories": [{}]},
    )
    task = Task("SELL", Position(1, 1), 1, None, 1, item="EGG")
    assignment = assign_tasks([task], carried["workers"], carried)[0]
    assert _assignment_valid(carried, assignment)
    assert not _assignment_valid(empty, assignment)
    assert worker_action(0, carried, assignment) == ["DROP"]
    assert worker_action(0, empty, assignment) == ["PASS"]


def test_sell_assignment_requires_the_carried_product_and_sell_all_accepts_any_product():
    state = _state(
        workers=_workers(("FARMER", Position(1, 1))),
        private={"shed": {"EGG": 4}, "inventories": [{"CARROT": 1}]},
    )
    egg = WorkerAssignment(
        0, Task("SELL", Position(1, 1), 1, None, 1, item="EGG"),
    )
    sell_all = WorkerAssignment(
        0, Task("SELL_ALL", Position(1, 1), 1, None, 1, sell_all=True),
    )

    assert not _assignment_valid(state, egg)
    assert worker_action(0, state, egg) == ["PASS"]
    assert _assignment_valid(state, sell_all)
    assert worker_action(0, state, sell_all) == ["DROP"]


def test_generic_sell_all_plans_deterministic_product_surplus_after_feed_reserve():
    tiles = [["EMPTY"] * 5 for _ in range(5)]
    tiles[0][0] = {"kind": "COOP", "animal": {"species": "GOOSE"}}
    state = _state(
        day=29,
        hour=5,
        tiles=tiles,
        private={
            "seeds": {},
            "shed": {},
            "inventories": [{"WHEAT": 2, "EGG": 1}],
        },
        market={"prices": {"WHEAT": 10, "EGG": 20}},
    )

    sell_tasks = [task for task in build_daily_plan(state) if task.kind == "SELL"]

    assert [
        (task.item, getattr(task, "quantity", None))
        for task in sell_tasks
    ] == [("EGG", 1), ("WHEAT", 1)]
    assert all(task.sell_all for task in sell_tasks)


def test_generic_sell_all_does_not_plan_protected_feed_wheat():
    tiles = [["EMPTY"] * 5 for _ in range(5)]
    tiles[0][0] = {"kind": "COOP", "animal": {"species": "GOOSE"}}
    state = _state(
        day=29,
        hour=5,
        tiles=tiles,
        private={
            "seeds": {},
            "shed": {},
            "inventories": [{"WHEAT": 1}],
        },
        market={"prices": {"WHEAT": 10}},
    )

    assert not any(task.kind == "SELL" for task in build_daily_plan(state))


def test_basic_need_wheat_buy_survives_reversal_filter():
    state = _state(
        day=28,
        hour=1,
        private={"seeds": {}, "shed": {}, "inventories": [{"GOOSE": 1}]},
        market={"prices": {"WHEAT": 10}, "inventory": {}},
    )
    protected = Policy(strategy="current")
    protected.memory.market_history["WHEAT"] = [(28 * 24, "SELL")]
    guarded, protected_directions = protected._basic_need_guard(
        state, [["BUY_PRODUCT", "WHEAT", 2]], [], None,
    )
    protected_result = protected._filter_market_direction(
        state, guarded, None, protected=protected_directions,
    )

    discretionary = Policy(strategy="current")
    discretionary.memory.market_history["WHEAT"] = [(28 * 24, "SELL")]
    guarded, discretionary_directions = discretionary._basic_need_guard(
        state, [["BUY_PRODUCT", "WHEAT", 1]], [], None,
    )
    discretionary_result = discretionary._filter_market_direction(
        state, guarded, None, protected=discretionary_directions,
    )
    assert protected_directions == {("WHEAT", "BUY_PRODUCT")}
    assert discretionary_directions == set()
    assert ["BUY_PRODUCT", "WHEAT", 2] in protected_result
    assert ["BUY_PRODUCT", "WHEAT", 1] not in discretionary_result


def test_basic_need_guard_keeps_hire_that_adds_deadline_capacity():
    tiles = [["EMPTY"] * 5 for _ in range(5)]
    tiles[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "watered_today": False,
    }
    state = _state(
        day=17,
        hour=0,
        tiles=tiles,
        cash=5_089,
        private={"seeds": {}, "shed": {}, "inventories": [{}]},
    )
    policy = Policy(strategy="current")

    guarded, _protected_directions = policy._basic_need_guard(
        state, [["HIRE"]], [], None,
    )

    assert guarded == [["HIRE"]]


def test_basic_need_guard_reserves_next_day_capacity_hire_cash():
    tiles = [["EMPTY"] * 10 for _ in range(10)]
    for x in range(10):
        tiles[0][x] = {
            "kind": "PLANT",
            "crop": "STRAWBERRY",
            "watered_today": True,
        }
    for x in range(6):
        tiles[9][x] = {
            "kind": "PLANT",
            "crop": "STRAWBERRY",
            "watered_today": True,
        }
    state = _state(
        day=11,
        hour=21,
        board_size=10,
        cash=149,
        tiles=tiles,
        workers=_workers(
            ("FARMER", Position(9, 9)),
            ("WORKER", Position(0, 9)),
        ),
        private={"seeds": {}, "shed": {}, "inventories": [{}, {}]},
    )
    policy = Policy(strategy="current")

    guarded, _protected_directions = policy._basic_need_guard(
        state, [["BUY_SEED", "STRAWBERRY", 1]], [], None,
    )

    assert guarded == []


def test_macro_deadline_capacity_hire_preempts_optional_seed_purchase():
    tiles = [["EMPTY"] * 10 for _ in range(10)]
    for x in range(10):
        tiles[0][x] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
            "fertilized_until_day": -1,
        }
    for x in range(6):
        tiles[9][x] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
            "fertilized_until_day": -1,
        }
    state = _state(
        day=12,
        hour=0,
        board_size=10,
        cash=139,
        tiles=tiles,
        workers=_workers(("FARMER", Position(9, 9))),
        private={"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
        market={
            "prices": {"MELON": 250, "FERTILIZER": 100},
            "inventory": {"MELON": 10_000, "FERTILIZER": 10_000},
        },
    )

    macro = build_autonomous_macro_plan(state)

    assert ["HIRE"] in macro["market_intents"]


def test_policy_suppresses_final_hour_fertilize_on_due_water_tile():
    target = Position(4, 4)
    state = _state(
        day=24,
        hour=23,
        board_size=10,
        workers=_workers(
            ("FARMER", target),
            ("WORKER", target),
        ),
        private={"seeds": {}, "shed": {}, "inventories": [{"FERTILIZER": 1}, {}]},
    )
    state["tiles"][4][4] = {
        "kind": "PLANT",
        "crop": "CARROT",
        "watered_today": False,
    }

    commands = _suppress_final_hour_fertilize_on_due_water(
        state, {0: ["FERTILIZE"], 1: ["WATER"]},
    )

    assert commands == {0: ["PASS"], 1: ["WATER"]}


def test_assign_tasks_keeps_same_tile_water_and_fertilize_when_time_remains():
    target = Position(4, 4)
    state = _state(
        day=24,
        hour=10,
        board_size=10,
        workers=_workers(
            ("FARMER", Position(9, 9)),
            ("WORKER", target),
        ),
        private={"seeds": {}, "shed": {}, "inventories": [{"FERTILIZER": 1}, {}]},
    )
    plan = [
        Task("FERTILIZE", target, 97, None, 1, item="CARROT"),
        Task("WATER", target, 100, 24, 1, item="CARROT"),
    ]

    assignments = assign_tasks(plan, state["workers"], state)

    assert [(assignment.task.kind, assignment.task.target) for assignment in assignments] == [
        ("WATER", target),
        ("FERTILIZE", target),
    ]


def test_macro_funds_deadline_hire_without_pre_funding_stored_animal_feed():
    """Stored-animal feed is a future reserve, not an immediate hire cost."""
    tiles = [["EMPTY"] * 10 for _ in range(10)]
    for x in range(1, 10):
        tiles[0][x] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
            "fertilized_until_day": 11,
        }
    tiles[1][1] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "watered_today": False,
        "fertilized_until_day": 11,
    }
    state = _state(
        day=11,
        hour=0,
        board_size=10,
        cash=101,
        tiles=tiles,
        private={
            "seeds": {"WHEAT": 1, "CARROT": 1, "TOMATO": 1, "STRAWBERRY": 1, "MELON": 1},
            "shed": {"GOOSE": 2},
            "inventories": [{}],
        },
        market={"prices": {"WHEAT": 10}, "inventory": {"WHEAT": 1000}},
    )

    macro = build_autonomous_macro_plan(state)

    assert ["HIRE"] in macro["market_intents"]
    assert not any(intent[0] == "BUY_ANIMAL" for intent in macro["market_intents"])


@__import__("pytest").mark.parametrize(
    ("cash", "wheat", "expected"),
    [(7, 30, True), (150, 30, True), (7, 0, False), (0, 1, False)],
)
def test_macro_proposes_affordable_hire_for_due_basic_need_capacity(cash, wheat, expected):
    tiles = [["EMPTY"] * 10 for _ in range(10)]
    for x, y in ((1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (4, 1)):
        tiles[y][x] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
        }
    tiles[0][0] = {"kind": "PASTURE"}
    state = _state(
        day=1,
        hour=0,
        board_size=10,
        tiles=tiles,
        cash=cash,
        workers=_workers(("FARMER", Position(4, 4))),
        private={
            "seeds": {},
            "shed": {"WHEAT": wheat, "SHEEP": 1},
            "inventories": [{}],
        },
    )

    macro = build_autonomous_macro_plan(state)

    assert (["HIRE"] in macro["market_intents"]) is expected


def test_macro_does_not_hire_at_hour_23_for_current_day_capacity():
    tiles = [["EMPTY"] * 10 for _ in range(10)]
    for x, y in ((1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (4, 1)):
        tiles[y][x] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": False,
        }
    tiles[0][0] = {"kind": "PASTURE"}
    state = _state(
        day=1,
        hour=23,
        board_size=10,
        tiles=tiles,
        cash=150,
        workers=_workers(("FARMER", Position(4, 4))),
        private={
            "seeds": {},
            "shed": {"WHEAT": 30, "SHEEP": 1},
            "inventories": [{}],
        },
    )

    macro = build_autonomous_macro_plan(state)

    assert ["HIRE"] not in macro["market_intents"]


def test_learned_policy_liquidates_shed_inventory_in_terminal_window(monkeypatch):
    from kagriculture_agent.learned_policy import PolicyProposal

    state = _state(day=29, hour=22, private={"shed": {"CARROT": 1}, "inventories": [[]]})
    deterministic = Policy(strategy="current").act(state)
    learned_policy = Policy(strategy="current", learned_model="stub.json")
    monkeypatch.setattr(learned_policy.learned_policy, "model_path", "stub.json")
    monkeypatch.setattr(
        learned_policy.learned_policy,
        "propose",
        lambda *_: (
            learned_policy.learned_policy.diagnostics.update(status="ok")
            or PolicyProposal((), (), 1.0, "stub")
        ),
    )
    learned = learned_policy.act(state)
    assert ["SELL", "CARROT", 1] in deterministic["market"]
    assert ["SELL", "CARROT", 1] in learned["market"]


@__import__("pytest").mark.parametrize("kind", ["PASS", "MOVE"])
def test_loaded_non_overriding_learned_class_preserves_deterministic_assignment(kind):
    from kagriculture_agent.learned_policy import (
        WorkerProposal,
        compile_proposal,
        validate_action_vocabulary,
    )
    from kagriculture_agent.model import ACTION_VOCAB
    from kagriculture_agent.memory import PolicyMemory

    state = _state(
        tiles=[[None] * 5 for _ in range(5)],
        private={"seeds": {}, "shed": {}, "inventories": [{}]},
    )
    state["tiles"][0][1] = {
        "kind": "PLANT", "crop": "WHEAT", "watered_today": False,
    }
    task = Task("WATER", Position(1, 0), 10, None, 10.0)
    memory = PolicyMemory(assignments=[WorkerAssignment(0, task, [Position(1, 0)])])
    proposal = PolicyProposal(
        (WorkerProposal(0, kind, Position(0, 0), None, 100.0),),
        (), 1.0, "learned_v1",
    )

    validated = validate_action_vocabulary(
        {key: list(value) for key, value in ACTION_VOCAB.items()},
        model_version="learned_v1",
        source="proposal compiler test",
    )

    action = compile_proposal(state, proposal, memory)

    assert kind in validated["worker_kinds"]
    assert memory.assignments[0].task.kind == "WATER"
    assert action["farmer"] == ["EAST"]


def test_learned_policy_caches_terminal_path_load_failure_with_safe_diagnostic(monkeypatch, tmp_path):
    path = tmp_path / "private-model.json"
    policy = LearnedPolicy(path)
    calls = []

    def fail_load():
        calls.append(path)
        raise ValueError("secret artifact contents")

    monkeypatch.setattr(policy, "_load", fail_load)

    assert policy.propose({}, object()).workers == ()
    first_diagnostic = dict(policy.diagnostics)
    assert policy.propose({}, object()).workers == ()

    assert calls == [path]
    assert policy.load_state == "failed"
    assert first_diagnostic == policy.diagnostics
    assert policy.diagnostics == {
        "status": "load_error",
        "load_state": "failed",
        "path": str(path),
        "error": "ValueError",
    }
    assert "secret artifact contents" not in repr(policy.diagnostics)


def test_learned_policy_fast_inference_records_latency_and_stays_enabled(monkeypatch):
    import kagriculture_agent.learned_policy as runtime

    model = object.__new__(DependencyFreePolicy)
    monkeypatch.setattr(
        DependencyFreePolicy,
        "propose",
        lambda self, state, features: PolicyProposal((), (), 1.0, "learned_v1"),
    )
    ticks = iter((10.0, 10.01))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    policy = LearnedPolicy(model, timeout_seconds=0.05)

    proposal = policy.propose(_state(), object())

    assert proposal.model_version == "learned_v1"
    assert policy.load_state == "loaded"
    assert policy.diagnostics["status"] == "ok"
    assert policy.diagnostics["inference_seconds"] == __import__("pytest").approx(0.01)


def test_learned_policy_slow_inference_falls_back_and_disables_episode(monkeypatch):
    import kagriculture_agent.learned_policy as runtime

    model = object.__new__(DependencyFreePolicy)
    calls = []

    def propose(self, state, features):
        calls.append(state)
        return PolicyProposal((), (("BUY_PRODUCT", "WHEAT", 1),), 1.0, "learned_v1")

    monkeypatch.setattr(DependencyFreePolicy, "propose", propose)
    ticks = iter((20.0, 20.06))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    policy = LearnedPolicy(model, timeout_seconds=0.05)

    first = policy.propose(_state(day=3, hour=1), object())
    second = policy.propose(_state(day=3, hour=2), object())

    assert first == PolicyProposal((), (), 0.0, "none")
    assert second == first
    assert len(calls) == 1
    assert policy.diagnostics["status"] == "slow_model"
    assert policy.diagnostics["inference_status"] == "slow_inference"
    assert policy.diagnostics["inference_seconds"] == __import__("pytest").approx(0.06)
    assert policy.diagnostics["budget_seconds"] == 0.05
    assert policy.diagnostics["learned_overrides_enabled"] is False
