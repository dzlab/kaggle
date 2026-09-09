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

    last_productive = macro(24)
    assert ["BUY_ANIMAL", "GOOSE", 1] in last_productive["market_intents"]
    assert any(task.kind == "BUILD_COOP" for task in last_productive["tasks"])

    too_late = macro(25)
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
