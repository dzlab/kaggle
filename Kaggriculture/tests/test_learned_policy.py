from dataclasses import FrozenInstanceError
import json

import pytest

from kagriculture_agent.learned_policy import (
    LearnedPolicy,
    PolicyProposal,
    WorkerProposal,
    compile_proposal,
)
from kagriculture_agent.memory import PolicyMemory
from kagriculture_agent.types import Position, Task, WorkerAssignment


def state(*, workers=None, tiles=None, inventories=None, shed=None, cash=1_000):
    return {
        "board_size": 3,
        "day": 1,
        "hour": 1,
        "cash": cash,
        "tiles": tiles if tiles is not None else [[None] * 3 for _ in range(3)],
        "workers": workers if workers is not None else [
            {"index": 0, "role": "FARMER", "position": [0, 0]},
        ],
        "private": {
            "inventories": inventories if inventories is not None else [[]],
            "shed": shed if shed is not None else {},
            "seeds": {"WHEAT": 1},
        },
        "market": {"prices": {"WHEAT": 10}, "inventory": {}},
    }


def test_proposal_dataclasses_are_frozen_and_typed():
    worker = WorkerProposal(0, "WATER", Position(1, 1), None, 0.5)
    proposal = PolicyProposal((worker,), (("BUY_PRODUCT", "WHEAT", 1),), 0.8, "v1")

    assert proposal.workers == (worker,)
    assert proposal.market_orders == (("BUY_PRODUCT", "WHEAT", 1),)
    with pytest.raises(FrozenInstanceError):
        worker.score = 1.0


def test_no_model_returns_empty_proposal_and_disabled_status():
    policy = LearnedPolicy()

    proposal = policy.propose({}, object())

    assert proposal == PolicyProposal((), (), 0.0, "none")
    assert policy.diagnostics["status"] == "disabled"


@pytest.mark.parametrize("model_contents", [b"not a model", b"{\"workers\": ["])
def test_missing_or_corrupt_model_falls_back_without_raising(tmp_path, model_contents):
    path = tmp_path / "model.bin"
    if model_contents != b"missing":
        path.write_bytes(model_contents)

    policy = LearnedPolicy(path if model_contents != b"missing" else tmp_path / "missing.bin")

    proposal = policy.propose({}, object())

    assert proposal.workers == ()
    assert proposal.market_orders == ()
    assert policy.diagnostics["status"] in {"missing_model", "load_error", "incompatible_model"}


def test_json_model_proposes_targets_without_movement_paths(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({
        "model_version": "json-v1",
        "confidence": 0.75,
        "workers": [{"worker_index": 0, "kind": "WATER", "target": [2, 0], "score": 2}],
        "market_orders": [["BUY_PRODUCT", "WHEAT", 1]],
    }))

    proposal = LearnedPolicy(path).propose(state(), object())

    assert proposal.workers == (WorkerProposal(0, "WATER", Position(2, 0), None, 2.0),)
    assert proposal.market_orders == (("BUY_PRODUCT", "WHEAT", 1),)
    assert proposal.model_version == "json-v1"


def test_compile_proposal_turns_target_into_one_movement():
    current = state()
    current["tiles"][0][2] = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False}
    proposal = PolicyProposal((WorkerProposal(0, "WATER", Position(2, 0), None, 1),), (), 1.0, "v1")

    action = compile_proposal(current, proposal, PolicyMemory())

    assert action["farmer"] == ["EAST"]
    assert action["hands"] == []


def test_compile_rejects_unknown_out_of_bounds_and_locked_targets():
    current = state(workers=[
        {"index": 0, "role": "FARMER", "position": [0, 0]},
        {"index": 1, "role": "WORKER", "position": [0, 0]},
    ])
    current["tiles"][0][1] = {"kind": "LOCKED"}
    current["private"]["seeds"] = {}
    current["farm"] = {"hands": [[1, 0]]}
    proposal = PolicyProposal((
        WorkerProposal(99, "WATER", Position(1, 1), None, 10),
        WorkerProposal(0, "WATER", Position(5, 0), None, 10),
        WorkerProposal(1, "WATER", Position(1, 0), None, 10),
    ), (), 1.0, "v1")

    action = compile_proposal(current, proposal, PolicyMemory())

    assert action["farmer"] == ["PASS"]
    assert action["hands"] == [["PASS"]]


def test_compile_resolves_duplicate_worker_by_score_then_index():
    current = state(workers=[{"index": 0, "role": "FARMER", "position": [0, 0]}])
    current["tiles"][0][1] = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False}
    proposal = PolicyProposal((
        WorkerProposal(0, "WATER", Position(1, 0), None, 1),
        WorkerProposal(0, "WATER", Position(2, 0), None, 2),
    ), (), 1.0, "v1")

    action = compile_proposal(current, proposal, PolicyMemory())

    assert action["farmer"] == ["EAST"]


def test_compile_preserves_carried_delivery_assignment():
    current = state(inventories=[{"WHEAT": 1}])
    current["tiles"][0][1] = {"kind": "PASTURE", "animal": {"species": "COW", "fed_today": False}}
    memory = PolicyMemory(assignments=[WorkerAssignment(
        0, Task("FEED", Position(1, 0), 10, None, 1), route=[]
    )])
    proposal = PolicyProposal((WorkerProposal(0, "WATER", Position(2, 2), None, 99),), (), 1.0, "v1")

    action = compile_proposal(current, proposal, memory)

    assert action["farmer"] == ["EAST"]
    assert memory.assignments[0].task.kind == "FEED"


def test_compile_passes_legal_market_intent_to_market_compiler():
    current = state()
    proposal = PolicyProposal((), (("BUY_PRODUCT", "WHEAT", 1),), 1.0, "v1")

    action = compile_proposal(current, proposal, PolicyMemory())

    assert action["market"] == [["BUY_PRODUCT", "WHEAT", 1]]
