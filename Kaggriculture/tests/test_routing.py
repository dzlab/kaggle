from copy import deepcopy
from types import SimpleNamespace

from kagriculture_agent.planner import assign_tasks, build_daily_plan, normalize_planner_state
from kagriculture_agent.observation import parse_observation
from kagriculture_agent.routing import (
    distance,
    nearest_target,
    next_move,
    route_action,
    route_to,
)
from kagriculture_agent.types import EpisodeMemory, Position, Task


def pos(x, y):
    return Position(x, y)


def test_route_to_never_leaves_board_at_edges():
    assert route_to(pos(0, 0), pos(-1, 0), 4) == []
    assert route_to(pos(0, 0), pos(0, 3), 4) == ["SOUTH", "SOUTH", "SOUTH"]
    assert route_to(pos(3, 3), pos(3, 4), 4) == []


def test_locked_tiles_are_passable_but_rejected_as_action_targets():
    assert route_to(pos(0, 0), pos(2, 0), 3) == ["EAST", "EAST"]
    assert route_action(pos(0, 0), pos(2, 0), board_size=3, tile="LOCKED", action="WATER") == "EAST"
    assert route_action(pos(2, 0), pos(2, 0), board_size=3, tile="LOCKED", action="WATER") == "PASS"
    assert route_action(pos(2, 0), pos(2, 0), board_size=3, tile={"kind": "LOCKED"}, action="WATER") == "PASS"


def test_route_action_rejects_invalid_current_and_target_positions():
    assert route_action(pos(-1, 0), pos(0, 0), board_size=3, action="HARVEST") == "PASS"
    assert route_action(pos(0, 0), pos(3, 0), board_size=3, action="HARVEST") == "PASS"


def test_routing_accepts_typed_coordinates_and_rejects_typed_locked_tiles():
    current = SimpleNamespace(x=0, y=0)
    target = SimpleNamespace(x=1, y=0)
    locked = SimpleNamespace(kind="LOCKED")
    assert route_action(current, target, board_size=3, action="WATER") == "EAST"
    assert route_action(target, target, board_size=3, tile=locked, action="WATER") == "PASS"


def test_manhattan_routing_and_ties_are_deterministic():
    assert distance(pos(1, 2), pos(4, 0)) == 5
    assert next_move(pos(1, 2), pos(4, 0)) == "EAST"
    assert route_to(pos(1, 2), pos(4, 0), 6) == ["EAST", "EAST", "EAST", "NORTH", "NORTH"]
    assert nearest_target(pos(0, 0), [pos(2, 0), pos(0, 2)]) == pos(2, 0)


def test_route_action_returns_requested_action_only_when_at_legal_target():
    assert route_action(pos(0, 0), pos(1, 0), board_size=4, action="HARVEST") == "EAST"
    assert route_action(pos(1, 0), pos(1, 0), board_size=4, action="HARVEST") == "HARVEST"
    assert route_action(pos(1, 0), pos(1, 0), board_size=4, action=None) == "PASS"
    assert route_action(pos(1, 0), pos(1, 0), board_size=4, action="NOT_AN_ENGINE_ACTION") == "PASS"


def _state(**overrides):
    state = {
        "day": 2,
        "board_size": 5,
        "cash": 1_000,
        "tiles": {},
        "animals": [],
        "inventory": {},
        "structures": [],
        "desired_animals": [],
        "seeds": {"WHEAT": 2},
        "workers": [
            {"index": 0, "role": "FARMER", "position": pos(2, 2)},
            {"index": 1, "role": "WORKER", "position": pos(0, 0)},
        ],
    }
    state.update(overrides)
    return state


def test_daily_plan_prioritizes_urgent_water_feed_and_care():
    state = _state(
        tiles={
            pos(1, 1): {"crop": "WHEAT", "planted_day": 0, "watered": False},
        },
        animals=[
            {"position": pos(3, 1), "species": "GOOSE", "fed": False, "cared": False},
        ],
    )
    plan = build_daily_plan(state, EpisodeMemory())
    kinds = [task.kind for task in plan]
    assert "WATER" in kinds
    assert "FEED" in kinds
    assert "CARE" in kinds
    assert kinds.index("WATER") < kinds.index("PLANT") if "PLANT" in kinds else True
    assert all(task.deadline is not None for task in plan if task.kind in {"WATER", "FEED", "CARE"})


def test_named_strategy_keeps_maintenance_for_existing_disallowed_assets():
    from kagriculture_agent.strategy import StrategySpec

    workers = [
        {"index": 0, "role": "FARMER", "position": pos(0, 0)},
        {"index": 1, "role": "WORKER", "position": pos(1, 0)},
        {"index": 2, "role": "WORKER", "position": pos(2, 0)},
        {"index": 3, "role": "WORKER", "position": pos(3, 0)},
    ]
    state = _state(
        day=4,
        workers=workers,
        tiles={
            pos(0, 1): {
                "kind": "PLANT", "crop": "TOMATO", "needs_water": True,
                "watered_today": False, "planted_day": 0, "planted_age": 8,
                "yield_units": 1,
            },
            pos(1, 1): {
                "kind": "COOP",
                "animal": {
                    "species": "GOOSE", "needs_feed": True, "fed_today": False,
                    "needs_care": True, "cared_today": False,
                },
            },
        },
        seeds={"TOMATO": 2},
        market={"prices": {"TOMATO": 100}},
    )
    strategy = StrategySpec("melon-only", ("MELON",), ("COW",), 10, 10, 0)

    plan = build_daily_plan(state, EpisodeMemory(), strategy)
    assignments = assign_tasks(plan, workers, state, strategy)

    planned = {(task.kind, task.item) for task in plan}
    assigned = {(assignment.task.kind, assignment.task.item) for assignment in assignments}
    assert {
        ("WATER", "TOMATO"), ("HARVEST", "TOMATO"),
        ("FEED", "GOOSE"), ("CARE", "GOOSE"),
    } <= planned
    assert {
        ("WATER", "TOMATO"), ("HARVEST", "TOMATO"),
        ("FEED", "GOOSE"), ("CARE", "GOOSE"),
    } <= assigned
    assert not {(task.kind, task.item) for task in plan
                if task.kind in {"PLANT", "ANIMAL"}}


def test_named_strategy_keeps_state_animal_maintenance_when_not_embedded_in_tiles():
    from kagriculture_agent.strategy import StrategySpec

    workers = [
        {"index": 0, "role": "FARMER", "position": pos(0, 0)},
        {"index": 1, "role": "WORKER", "position": pos(1, 0)},
    ]
    state = _state(
        day=4,
        workers=workers,
        tiles={},
        animals=[{
            "position": pos(2, 1), "species": "GOOSE", "needs_feed": True,
            "fed_today": False, "needs_care": True, "cared_today": False,
        }],
    )
    strategy = StrategySpec("cow-only", ("WHEAT",), ("COW",), 10, 10, 0)

    plan = build_daily_plan(state, EpisodeMemory(), strategy)
    assignments = assign_tasks(plan, workers, state, strategy)

    assert {(task.kind, task.item) for task in plan} >= {
        ("FEED", "GOOSE"), ("CARE", "GOOSE"),
    }
    assert {(assignment.task.kind, assignment.task.item) for assignment in assignments} >= {
        ("FEED", "GOOSE"), ("CARE", "GOOSE"),
    }


def test_daily_plan_deduplicates_tile_and_listed_animal_maintenance():
    animal = {
        "id": "goose-1", "species": "GOOSE", "needs_feed": True,
        "fed_today": False, "needs_care": True, "cared_today": False,
    }
    state = _state(
        day=4,
        tiles={pos(1, 1): {"kind": "COOP", "animal": animal}},
        animals=[{**animal, "position": pos(1, 1)}],
    )

    maintenance = [
        (task.kind, task.target, task.item)
        for task in build_daily_plan(state, EpisodeMemory())
        if task.kind in {"FEED", "CARE"}
    ]

    assert maintenance == [
        ("FEED", pos(1, 1), "GOOSE"),
        ("CARE", pos(1, 1), "GOOSE"),
    ]


def test_daily_plan_schedules_positive_harvest_before_decay():
    state = _state(
        day=2,
        tiles={pos(1, 1): {"crop": "WHEAT", "planted_day": 0, "yield_units": 2, "watering_days": [0, 1, 2]}},
    )
    harvests = [task for task in build_daily_plan(state, EpisodeMemory()) if task.kind == "HARVEST"]
    assert harvests
    assert harvests[0].target == pos(1, 1)
    assert harvests[0].deadline <= 4
    assert harvests[0].value > 0


def test_daily_plan_harvests_at_first_decay_boundary_before_decay():
    state = _state(
        day=5,
        tiles={pos(1, 1): {"crop": "WHEAT", "planted_day": 0, "yield_units": 2, "watering_days": [0, 1, 2, 3, 4]}},
    )
    harvests = [task for task in build_daily_plan(state, EpisodeMemory()) if task.kind == "HARVEST"]
    assert harvests and harvests[0].deadline == 5 and harvests[0].value > 0


def test_daily_plan_keeps_ongoing_production_harvestable_past_max_yield_day():
    state = _state(
        day=10,
        tiles={pos(1, 1): {"crop": "TOMATO", "planted_day": 0, "yield_units": 1, "watering_days": list(range(11))}},
    )
    harvests = [task for task in build_daily_plan(state, EpisodeMemory()) if task.kind == "HARVEST"]
    assert harvests and harvests[0].target == pos(1, 1)


def test_daily_plan_does_not_invent_harvest_after_missed_watering():
    state = _state(
        day=2,
        tiles={pos(1, 1): {"crop": "WHEAT", "planted_day": 0, "watering_days": []}},
    )
    assert not [task for task in build_daily_plan(state, EpisodeMemory()) if task.kind == "HARVEST"]


def test_daily_plan_uses_same_locked_predicate_for_mapping_tiles():
    state = _state(
        tiles={pos(1, 1): {"kind": "LOCKED", "crop": "WHEAT", "needs_water": True}},
    )
    assert not [task for task in build_daily_plan(state, EpisodeMemory()) if task.target == pos(1, 1)]


def test_daily_plan_uses_observed_quotes_not_player_inventory_or_curve_overrides():
    state = _state(
        day=2,
        tiles={pos(1, 1): {"crop": "WHEAT", "planted_day": 0, "yield_units": 2}},
        inventory={"WHEAT": 4},
        market={"prices": {"WHEAT": 7}, "inventory": {"WHEAT": 99_999}},
    )
    plan = build_daily_plan(state, EpisodeMemory())
    harvest = next(task for task in plan if task.kind == "HARVEST")
    sell = next(task for task in plan if task.kind == "SELL")
    assert harvest.value == 14
    assert sell.value == 28


def test_daily_plan_schedules_fertilizer_from_live_shed_state():
    state = _state(
        day=2,
        tiles={pos(1, 1): {"crop": "TOMATO", "planted_day": 0,
                           "yield_units": 1, "watered_today": True,
                           "fertilized_until_day": -1}},
        inventory={"FERTILIZER": 1},
    )

    plan = build_daily_plan(state, EpisodeMemory())

    assert any(task.kind == "FERTILIZE" and task.target == pos(1, 1) for task in plan)


def test_planner_sanitizes_nonfinite_and_negative_shed_quantities():
    from math import inf, isnan

    state = _state(inventory={"WHEAT": float("nan"), "MELON": inf, "CARROT": -3})

    plan = build_daily_plan(state, EpisodeMemory())

    assert all(task.kind not in {"SHED", "SELL"} for task in plan)
    assert all(not isnan(task.value) and task.value != inf for task in plan)


def test_daily_plan_includes_structure_animal_weed_plant_shed_and_sell_work():
    state = _state(
        tiles={
            pos(0, 1): "WEED",
            pos(1, 1): {"empty": True},
        },
        structures=[{"kind": "COOP", "position": pos(4, 4), "built": False}],
        desired_animals=[{"species": "GOOSE", "position": pos(4, 4), "owned": False}],
        inventory={"WHEAT": 4},
    )
    kinds = {task.kind for task in build_daily_plan(state, EpisodeMemory())}
    assert {"STRUCTURE", "ANIMAL", "WEED", "PLANT", "SHED", "SELL"} <= kinds


def test_daily_plan_uses_valid_shed_target_on_one_by_one_board():
    worker = {"index": 0, "role": "FARMER", "position": pos(0, 0)}
    state = _state(board_size=1, workers=[worker], inventory={"WHEAT": 1})
    plan = build_daily_plan(state, EpisodeMemory())
    shed = next(task for task in plan if task.kind == "SHED")
    assert shed.target == pos(0, 0)
    assignment = assign_tasks([shed], [worker], state)
    assert assignment and assignment[0].task.target == pos(0, 0)


def test_daily_plan_accepts_parse_observation_canonical_nested_state():
    tiles = [["LOCKED" for _ in range(5)] for _ in range(5)]
    tiles[0][0] = None
    tiles[1][1] = {"kind": "CROP", "crop": "WHEAT", "planted_day": 0, "yield_units": 1, "watered_today": False}
    tiles[1][2] = {"kind": "WEED"}
    tiles[0][1] = {"kind": "ANIMAL", "animal": {"species": "GOOSE", "fed_today": False, "cared_today": False, "needs_placement": True}}
    tiles[0][2] = {"kind": "STRUCTURE", "structure": {"kind": "COOP", "built": False}}
    parsed = parse_observation({
        "day": 2,
        "farms": [{
            "tiles": tiles,
            "hands": ["WHEAT"],
            "farmer": [0, 0],
        }],
        "private": {"seeds": {"WHEAT": 2}, "shed": {"WHEAT": 4}},
        "market": {"prices": {"WHEAT": 7}},
    })
    kinds = {task.kind for task in build_daily_plan(parsed, EpisodeMemory())}
    assert {"WATER", "FEED", "CARE", "STRUCTURE", "ANIMAL", "WEED", "PLANT", "SHED"} <= kinds


def test_daily_plan_keeps_canonical_shed_fertilizer_available_for_fertilizing():
    tiles = [[{"kind": "PLANT", "crop": "WHEAT", "fertilized_until_day": 0}] + [None for _ in range(4)]] + [[None for _ in range(5)] for _ in range(4)]
    parsed = parse_observation({
        "day": 2,
        "farms": [{"tiles": tiles, "farmer": [0, 0], "hands": []}],
        "private": {"seeds": {"WHEAT": 1}, "shed": {"FERTILIZER": 1}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 7, "FERTILIZER": 100}},
    })

    plan = build_daily_plan(parsed, EpisodeMemory())

    assert any(task.kind == "FERTILIZE" and task.target == pos(0, 0) for task in plan)


def test_daily_plan_schedules_collect_fertilizer_and_keeps_harvest_urgent():
    tiles = [[
        {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 2,
         "watered_today": True, "fertilized_until_day": 0},
        {"kind": "COOP", "animal": "GOOSE", "fed_today": True, "cared_today": True,
         "fertilizer_available": True},
    ]]
    state = _state(day=4, board_size=2, tiles=tiles, inventory={"FERTILIZER": 1})

    plan = build_daily_plan(state, EpisodeMemory())

    assert any(task.kind == "COLLECT_FERTILIZER" and task.target == pos(1, 0) for task in plan)
    harvest = next(task for task in plan if task.kind == "HARVEST")
    fertilize = next(task for task in plan if task.kind == "FERTILIZE")
    assert harvest.priority > fertilize.priority


def test_assign_tasks_rejects_same_day_water_without_route_time():
    state = _state(
        hour=22,
        tiles={pos(4, 4): {"kind": "PLANT", "crop": "WHEAT", "watered": False}},
        workers=[{"index": 0, "role": "FARMER", "position": pos(0, 0)}],
    )

    assignments = assign_tasks(
        [Task("WATER", pos(4, 4), 100, 2, 1)], state["workers"], state,
    )

    assert assignments == []


def test_assign_tasks_reserves_shed_pickup_time_for_same_day_feed():
    state = _state(
        hour=14,
        tiles={pos(4, 4): {"kind": "PASTURE", "animal": "GOOSE", "fed": False}},
        workers=[{"index": 0, "role": "FARMER", "position": pos(0, 0)}],
        private={"shed": {"WHEAT": 1}, "inventories": [[]]},
    )
    task = Task("FEED", pos(4, 4), 100, 2, 1)

    assert assign_tasks([task], state["workers"], state)

    state["hour"] = 17
    assert assign_tasks([task], state["workers"], state) == []


def test_normalize_planner_state_supports_attribute_based_state():
    state = SimpleNamespace(
        day=2,
        board_size=3,
        tiles={pos(0, 0): {"empty": True}},
        seeds={"WHEAT": 1},
        inventory={},
        animals=[],
        structures=[],
    )
    assert any(task.kind == "PLANT" for task in build_daily_plan(state, EpisodeMemory()))


def test_normalize_planner_state_builds_stable_workers_from_canonical_farm():
    parsed = parse_observation({
        "farms": [{
            "tiles": [[None, None], ["LOCKED", "LOCKED"]],
            "farmer": [0, 0],
            "hands": [{"position": [1, 0]}, {"position": [0, 1]}],
        }],
    })
    normalized = normalize_planner_state(parsed)
    assert normalized["workers"] == [
        {"index": 0, "role": "FARMER", "position": pos(0, 0)},
        {"index": 1, "role": "WORKER", "position": pos(1, 0)},
        {"index": 2, "role": "WORKER", "position": pos(0, 1)},
    ]


def test_daily_plan_accepts_typed_tiles_and_nested_entities():
    state = SimpleNamespace(
        day=2,
        board_size=3,
        tiles=[[
            SimpleNamespace(kind="CROP", crop="WHEAT", planted_day=0, yield_units=0, watered_today=False),
            SimpleNamespace(kind="ANIMAL", animal=SimpleNamespace(
                species="GOOSE", fed_today=False, cared_today=False,
            )),
            SimpleNamespace(kind="STRUCTURE", structure=SimpleNamespace(kind="COOP", built=False)),
        ]],
        seeds={"WHEAT": 1},
        inventory={},
    )
    kinds = {task.kind for task in build_daily_plan(state, EpisodeMemory())}
    assert {"WATER", "FEED", "CARE", "STRUCTURE"} <= kinds


def test_assign_tasks_uses_derived_workers_for_empty_and_none_worker_inputs():
    parsed = parse_observation({
        "farms": [{
            "tiles": [[None, None], ["LOCKED", "LOCKED"]],
            "farmer": [0, 0],
            "hands": [{"position": [1, 0]}],
        }],
    })
    plan = [Task("WEED", pos(1, 0), 10, 1, 1)]
    assert assign_tasks(plan, [], parsed)[0].worker_index == 1
    assert assign_tasks(plan, None, parsed)[0].worker_index == 1


def test_generated_plant_task_is_selected_after_urgent_water_task():
    workers = [
        {"index": 0, "role": "FARMER", "position": pos(2, 2)},
        {"index": 1, "role": "WORKER", "position": pos(0, 0)},
    ]
    state = _state(
        workers=workers,
        tiles={
            pos(0, 0): {"empty": True},
            pos(1, 0): {"crop": "WHEAT", "watered_today": False, "yield_units": 0},
        },
    )
    plan = build_daily_plan(state, EpisodeMemory())
    assert any(task.kind == "PLANT" for task in plan)
    assignments = assign_tasks(plan, workers, state)
    assert [assignment.task.kind for assignment in assignments] == ["WATER", "PLANT"]


def test_normalize_planner_state_preserves_hand_indices_across_malformed_entries():
    parsed = parse_observation({
        "farms": [{
            "tiles": [[None, None, None]],
            "farmer": "malformed",
            "hands": ["malformed", {"position": [1, 0]}, None, {"position": [2, 0]}],
        }],
    })
    normalized = normalize_planner_state(parsed)
    assert [worker["index"] for worker in normalized["workers"]] == [2, 4]


def test_assign_tasks_deduplicates_exclusive_tiles_and_reserves_basic_needs():
    plan = [
        Task("WATER", pos(1, 1), 100, 2, 5),
        Task("WATER", pos(1, 1), 90, 2, 5),
        Task("PLANT", pos(3, 3), 80, 5, 100),
    ]
    workers = [
        {"index": 0, "role": "WORKER", "position": pos(0, 0)},
        {"index": 1, "role": "WORKER", "position": pos(4, 4)},
    ]
    assignments = assign_tasks(plan, workers, _state(day=2, workers=workers))
    assert len(assignments) == 2
    assert [assignment.task.kind for assignment in assignments].count("WATER") == 1
    assert any(assignment.task.kind == "WATER" for assignment in assignments)
    assert [assignment.task.kind for assignment in assignments] == ["WATER", "PLANT"]


def test_assign_tasks_lets_farmer_fallback_to_urgent_needs_without_helpers():
    worker = {"index": 0, "role": "FARMER", "position": pos(0, 0)}
    plan = [
        Task("WATER", pos(1, 1), 100, 2, 1),
        Task("SHED", pos(2, 2), 85, 2, 10),
        Task("SELL", pos(2, 2), 75, 2, 10),
    ]
    assignments = assign_tasks(plan, [worker], _state(day=2, workers=[worker]))
    assert len(assignments) == 1
    assert assignments[0].worker_index == 0
    assert assignments[0].task.kind == "WATER"


def test_assign_tasks_preserves_farmer_for_shed_after_helper_takes_water():
    workers = [
        {"index": 0, "role": "FARMER", "position": pos(2, 2)},
        {"index": 1, "role": "WORKER", "position": pos(0, 0)},
    ]
    plan = [
        Task("WATER", pos(0, 1), 101, 2, 1),
        Task("PLANT", pos(1, 1), 100, None, 10_000),
        Task("SHED", pos(2, 2), 85, None, 10),
    ]
    assignments = assign_tasks(plan, workers, _state(day=2, workers=workers))
    by_worker = {assignment.worker_index: assignment.task.kind for assignment in assignments}
    assert by_worker == {0: "SHED", 1: "WATER"}


def test_assign_tasks_uses_farmer_for_second_basic_task_after_helper_is_taken():
    workers = [
        {"index": 0, "role": "FARMER", "position": pos(2, 2)},
        {"index": 1, "role": "WORKER", "position": pos(0, 0)},
    ]
    plan = [
        Task("WATER", pos(0, 1), 101, 2, 1),
        Task("FEED", pos(1, 0), 100, 2, 1),
        Task("SHED", pos(2, 2), 85, None, 10),
    ]
    assignments = assign_tasks(plan, workers, _state(day=2, workers=workers))
    assert {assignment.task.kind for assignment in assignments} == {"WATER", "FEED"}
    assert {assignment.worker_index: assignment.task.kind for assignment in assignments} == {
        0: "FEED",
        1: "WATER",
    }


def test_assign_tasks_keeps_farmer_for_shed_logistics():
    plan = [
        Task("PLANT", pos(1, 1), 100, 2, 500),
        Task("SHED", pos(2, 2), 10, 5, 1),
    ]
    workers = [
        {"index": 0, "role": "FARMER", "position": pos(2, 2)},
        {"index": 1, "role": "WORKER", "position": pos(0, 0)},
    ]
    assignments = assign_tasks(plan, workers, _state(day=2, workers=workers))
    by_worker = {assignment.worker_index: assignment.task.kind for assignment in assignments}
    assert by_worker[0] == "SHED"
    assert by_worker[1] == "PLANT"


def test_assign_tasks_uses_stable_worker_and_coordinate_tie_breaking():
    task = Task("WEED", pos(1, 1), 10, 4, 10)
    workers = [
        {"index": 2, "role": "WORKER", "position": pos(2, 1)},
        {"index": 1, "role": "WORKER", "position": pos(0, 1)},
    ]
    assignments = assign_tasks([task], workers, _state(day=2, workers=workers))
    assert len(assignments) == 1
    assert assignments[0].worker_index == 1


def test_planner_does_not_mutate_canonical_parse_observation_input():
    raw = {
        "day": 0,
        "farms": [{"tiles": [[None, "LOCKED"], ["LOCKED", "LOCKED"]], "hands": []}],
        "private": {"seeds": {"WHEAT": 1}, "shed": {}},
        "market": {"prices": {"WHEAT": 7}},
    }
    parsed = parse_observation(raw)
    snapshot = deepcopy(parsed)
    build_daily_plan(parsed, EpisodeMemory())
    assign_tasks([], [], parsed)
    assert parsed == snapshot
