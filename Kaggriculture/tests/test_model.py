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
    extract_features_with_context,
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


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
    ],
)
def test_resolve_device_selects_requested_available_device(
    monkeypatch, requested, cuda_available, expected,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert resolve_device(requested) == torch.device(expected)


def test_resolve_device_rejects_invalid_name():
    pytest.importorskip("torch")
    from kagriculture_agent.model import resolve_device

    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_device("mps")


def test_resolve_device_rejects_unavailable_explicit_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        resolve_device("cuda")


def test_feature_tensors_follow_explicit_device():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import feature_batch_to_tensors

    tensors = feature_batch_to_tensors(extract_features(sample_state()), device=torch.device("cpu"))

    assert {tensor.device for tensor in tensors.values()} == {torch.device("cpu")}


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


def test_experimental_context_is_opt_in_and_cannot_enter_production_network():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet

    state = sample_state()
    state["history"] = [{"action": {"type": "WATER"}, "success": True}]
    experimental_features = extract_features_with_context(state)
    experimental_network = CompactPolicyNet(feature_variant="experimental_context_v1")
    outputs = experimental_network(experimental_features)

    assert outputs["value"].shape == (1,)
    assert all(torch.isfinite(tensor).all().item() for tensor in outputs.values())
    with pytest.raises(ValueError, match="feature variant"):
        CompactPolicyNet()(experimental_features)


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


def test_policy_network_supports_opt_in_width_and_depth_and_reports_parameters():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet, model_parameter_count

    network = CompactPolicyNet(hidden_width=256, depth=8)

    assert network.hidden_width == 256
    assert network.depth == 8
    assert len(network.blocks) == 8
    assert network.parameter_count == model_parameter_count(network)
    assert network.parameter_count > model_parameter_count(CompactPolicyNet())
    assert all(torch.isfinite(parameter).all().item() for parameter in network.parameters())


@pytest.mark.parametrize("width,depth", [(128, 4), (256, 4), (128, 8)])
def test_shared_parameter_estimator_matches_constructed_policy(width, depth):
    pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet
    from kagriculture_agent.model_topology import compact_policy_parameter_count

    network = CompactPolicyNet(hidden_width=width, depth=depth)

    assert sum(parameter.numel() for parameter in network.parameters()) == (
        compact_policy_parameter_count(width, depth)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("hidden_width", 0), ("hidden_width", 3), ("depth", 0), ("depth", True)],
)
def test_policy_network_rejects_invalid_width_or_depth(field, value):
    pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet

    with pytest.raises(ValueError, match=field):
        CompactPolicyNet(**{field: value})


def test_batching_feature_batches_preserves_documented_shapes():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet

    batch = [extract_features(sample_state()), extract_features({**sample_state(), "cash": 1000})]
    outputs = CompactPolicyNet()(batch)

    assert outputs["worker_act_logits"].shape[0] == 2
    assert outputs["worker_target_logits"].shape == (2, 10, 100)
    assert outputs["value"].shape == (2,)


def test_target_first_policy_uses_target_conditioned_kind_head():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import ACTION_VOCAB, CompactPolicyNet

    network = CompactPolicyNet(action_representation="target_first_v1")
    outputs = network(extract_features(sample_state()))

    assert network.action_representation == "target_first_v1"
    assert outputs["worker_target_logits"].shape == (1, 10, 100)
    assert outputs["worker_kind_logits"].shape == (
        1, 10, 100, len(ACTION_VOCAB["worker_kinds"]),
    )
    assert all(torch.isfinite(tensor).all().item() for tensor in outputs.values())
