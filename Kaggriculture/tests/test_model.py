import importlib

import pytest

from kagriculture_agent.constants import PRODUCTS
from kagriculture_agent.features import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_TOKEN_SIZE,
    MARKET_TOKEN_SIZE,
    TILE_TOKEN_SIZE,
    WORKER_TOKEN_SIZE,
    extract_features,
)


def sample_state():
    return {
        "day": 2,
        "hour": 6,
        "cash": 900,
        "farm": {
            "tiles": [[{"kind": "EMPTY"} for _ in range(10)] for _ in range(10)],
            "workers": [
                {"index": 0, "role": "FARMER", "position": {"x": 1, "y": 1}},
                {"index": 1, "role": "WORKER", "position": {"x": 2, "y": 1}},
            ],
        },
        "private": {"shed": {"WHEAT": 5}},
    }


def test_model_module_imports_without_torch_dependency():
    model = importlib.import_module("kagriculture_agent.model")

    assert model.FEATURE_INPUT_SIZES == {
        "tile": TILE_TOKEN_SIZE,
        "worker": WORKER_TOKEN_SIZE,
        "market": MARKET_TOKEN_SIZE,
        "global": GLOBAL_TOKEN_SIZE,
    }
    assert model.MODEL_VERSION == "learned_v1"


def test_policy_network_forward_shapes_and_schema_validation():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import ACTION_VOCAB, CompactPolicyNet

    features = extract_features(sample_state())
    network = CompactPolicyNet()
    outputs = network(features)

    assert outputs["worker_act_logits"].shape == (1, 10, 2)
    assert outputs["worker_target_logits"].shape == (1, 10, 100)
    assert outputs["worker_kind_logits"].shape == (1, 10, len(ACTION_VOCAB["worker_kinds"]))
    assert outputs["market_item_logits"].shape == (1, len(PRODUCTS))
    assert outputs["market_quantity_logits"].shape == (1, len(ACTION_VOCAB["market_quantities"]))
    assert outputs["value"].shape == (1,)
    assert all(torch.isfinite(tensor).all().item() for tensor in outputs.values())

    with pytest.raises(ValueError, match="schema"):
        network(features.__class__(
            tile_tokens=features.tile_tokens,
            worker_tokens=features.worker_tokens,
            market_tokens=features.market_tokens,
            global_tokens=features.global_tokens,
            tile_positions=features.tile_positions,
            schema_version=FEATURE_SCHEMA_VERSION + 1,
        ))


def test_policy_network_initialization_is_deterministic_with_seed():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet, set_training_seed

    features = extract_features(sample_state())
    set_training_seed(123)
    first = CompactPolicyNet()(features)
    set_training_seed(123)
    second = CompactPolicyNet()(features)

    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_batching_feature_batches_preserves_documented_shapes():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet

    batch = [extract_features(sample_state()), extract_features({**sample_state(), "cash": 1000})]
    outputs = CompactPolicyNet()(batch)

    assert outputs["worker_act_logits"].shape[0] == 2
    assert outputs["worker_target_logits"].shape == (2, 10, 100)
    assert outputs["value"].shape == (2,)
