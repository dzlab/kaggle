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
from kagriculture_agent.policy import Policy, _assignment_valid, _drop_carried_goods
from kagriculture_agent.strategy import StrategySpec
from kagriculture_agent.types import Position, Task


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


def test_assign_tasks_falls_back_to_reserved_worker_when_only_farmer():
    state = _state(workers=_workers(("FARMER", Position(0, 0))),
                   private={"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{"GOOSE": 1}]})
    tasks = [Task("ANIMAL", Position(0, 1), 100, 1, 1, item="GOOSE"),
             Task("WATER", Position(4, 3), 100, 1, 1, item="WHEAT"),
             Task("SHED", Position(1, 1), 75, None, 1)]
    assignments = assign_tasks(tasks, state["workers"], state)
    assert [(assignment.worker_index, assignment.task.kind) for assignment in assignments] == [
        (0, "ANIMAL"), (0, "WATER"), (0, "SHED"),
    ]


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

    last_productive = macro(23)
    assert ["BUY_ANIMAL", "GOOSE", 1] in last_productive["market_intents"]
    assert any(task.kind == "BUILD_COOP" for task in last_productive["tasks"])

    too_late = macro(24)
    assert ["BUY_ANIMAL", "GOOSE", 1] not in too_late["market_intents"]
    assert not any(task.kind in {"ANIMAL", "BUILD_COOP"} for task in too_late["tasks"])


def test_late_daily_animal_and_structure_tasks_are_rejected():
    state = _state(
        day=25,
        desired_animals=[{"species": "GOOSE", "position": [0, 0], "owned": False}],
        structures=[{"kind": "COOP", "position": [1, 0], "built": False}],
    )

    plan = build_daily_plan(state)

    assert not any(task.kind in {"ANIMAL", "STRUCTURE"} for task in plan)


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
    carried = _state(private={"shed": {}, "inventories": [{"EGG": 1}]})
    empty = _state(private={"shed": {"EGG": 1}, "inventories": [{}]})
    task = Task("SELL", Position(0, 1), 1, None, 1, item="EGG")
    assignment = assign_tasks([task], carried["workers"], carried)[0]
    assert _assignment_valid(carried, assignment)
    assert not _assignment_valid(empty, assignment)


def test_feed_required_wheat_buy_survives_reversal_filter():
    state = _state(day=0, hour=1, private={"shed": {}, "inventories": [{}]})
    protected = Policy(strategy="current")
    protected.memory.market_history["WHEAT"] = [(0, "SELL")]
    guarded = protected._basic_need_guard(state, [["BUY_PRODUCT", "WHEAT", 2]], [], None)
    protected_result = protected._filter_market_direction(state, guarded, None)

    discretionary = Policy(strategy="current")
    discretionary.memory.market_history["WHEAT"] = [(0, "SELL")]
    discretionary_result = discretionary._filter_market_direction(
        state, [["BUY_PRODUCT", "WHEAT", 1]], None
    )
    assert ["BUY_PRODUCT", "WHEAT", 2] in protected_result
    assert ["BUY_PRODUCT", "WHEAT", 1] not in discretionary_result


def test_learned_policy_liquidates_shed_inventory_in_terminal_window(monkeypatch):
    from kagriculture_agent.learned_policy import PolicyProposal

    state = _state(day=29, hour=22, private={"shed": {"CARROT": 1}, "inventories": [[]]})
    deterministic = Policy(strategy="current").act(state)
    learned_policy = Policy(strategy="current", learned_model="stub.json")
    monkeypatch.setattr(learned_policy.learned_policy, "model_path", "stub.json")
    monkeypatch.setattr(learned_policy.learned_policy, "propose", lambda *_: PolicyProposal((), (), 1.0, "stub"))
    learned = learned_policy.act(state)
    assert ["SELL", "CARROT", 1] in deterministic["market"]
    assert ["SELL", "CARROT", 1] in learned["market"]
