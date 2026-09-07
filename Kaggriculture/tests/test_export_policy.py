import json
import time
from pathlib import Path

import pytest

from kagriculture_agent.features import extract_features
from kagriculture_agent.learned_policy import LearnedPolicy, load_exported_policy


def _artifact():
    from kagriculture_agent import learned_policy as runtime
    from scripts.export_policy import artifact_checksum

    weights = {}
    for name in runtime._artifact_tensor_names():
        if name.endswith((".bias", "_bias")) or name.endswith(".weight") and ("norm" in name or "embedding" in name):
            size = 384 if name.endswith("in_proj_bias") else 128
            if name.endswith(".weight") and "embedding" in name:
                weights[name] = {"shape": [4, 128], "scales": [1.0] * 4, "values": [[0] * 128 for _ in range(4)]}
            else:
                weights[name] = {"shape": [size], "values": [0.0] * size}
        else:
            rows = 384 if name.endswith("in_proj_weight") else 128
            weights[name] = {"shape": [rows, 128], "scales": [1.0] * rows, "values": [[0] * 128 for _ in range(rows)]}
    artifact = {
        "format_version": 1,
        "model_version": "learned_v1",
        "feature_schema_version": 1,
        "engine_version": "1.32.7",
        "hidden_width": 128,
        "quantization": "int8-per-row",
        "action_vocab": {
            "worker_kinds": list(runtime._ARTIFACT_WORKER_KINDS),
            "market_items": sorted(runtime.PRODUCTS),
            "market_quantities": list(runtime._ARTIFACT_MARKET_QUANTITIES),
        },
        "weights": weights,
    }
    artifact["checksum"] = artifact_checksum(artifact)
    return artifact


def test_artifact_headers_and_checksum_are_valid(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    policy = load_exported_policy(path)
    assert policy.model_version == "learned_v1"

    broken = _artifact()
    broken["hidden_width"] = 64
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported.*hidden_width"):
        load_exported_policy(path)


@pytest.mark.parametrize("field", ["format_version", "model_version", "feature_schema_version", "engine_version", "quantization"])
def test_unsupported_artifact_headers_are_rejected(tmp_path, field):
    artifact = _artifact()
    artifact[field] = "bad" if field != "format_version" and field != "feature_schema_version" else 99
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_exported_policy(path)


def test_invalid_tensor_and_checksum_are_rejected(tmp_path):
    artifact = _artifact()
    artifact["weights"]["value_head.bias"]["values"][0] = "NaN"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_exported_policy(path)

    artifact = _artifact()
    artifact["weights"]["value_head.bias"]["values"][0] = float("nan")
    artifact["checksum"] = __import__("scripts.export_policy", fromlist=["artifact_checksum"]).artifact_checksum(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_exported_policy(path)


def test_learned_policy_corrupt_artifact_uses_deterministic_fallback(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{}", encoding="utf-8")
    policy = LearnedPolicy(path)
    first = policy.propose({}, extract_features({}))
    second = policy.propose({}, extract_features({}))
    assert first == second
    assert first.workers == ()
    assert policy.diagnostics["status"] == "load_error"


def test_export_requires_torch_with_clear_error(tmp_path):
    from scripts import export_policy
    if export_policy.torch_available if hasattr(export_policy, "torch_available") else False:
        pytest.skip("torch is installed; checkpoint export is covered by integration fixtures")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        export_policy.export_checkpoint(tmp_path / "missing.pt", tmp_path / "policy.json")


def test_dependency_free_fixture_inference_is_fast_enough(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    policy = load_exported_policy(path)
    features = extract_features({})
    samples = []
    for _ in range(1000):
        started = time.perf_counter()
        policy.predict(features)
        samples.append((time.perf_counter() - started) * 1000.0)
    assert sorted(samples)[949] < 50.0


def test_exported_model_agrees_with_training_fixture_when_torch_is_available(tmp_path):
    torch = pytest.importorskip("torch", reason="training-model agreement requires PyTorch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts.export_policy import export_checkpoint

    checkpoint = tmp_path / "policy.pt"
    artifact_path = tmp_path / "policy.json"
    network = CompactPolicyNet()
    metadata = {
        "model_version": "learned_v1", "feature_schema_version": 1,
        "engine_version": "1.32.7",
        "action_vocab": {key: list(value) for key, value in __import__("kagriculture_agent.model", fromlist=["ACTION_VOCAB"]).ACTION_VOCAB.items()},
    }
    torch.save({"metadata": metadata, "model_state_dict": network.state_dict()}, checkpoint)
    export_checkpoint(checkpoint, artifact_path)
    runtime = load_exported_policy(artifact_path)
    features = extract_features({})
    with torch.no_grad():
        expected = network(features)
    actual = runtime.predict(features)
    for key in ("worker_act_logits", "worker_target_logits", "worker_kind_logits", "market_item_logits", "market_quantity_logits"):
        expected_values = expected[key].reshape(-1, expected[key].shape[-1]).tolist() if expected[key].ndim > 1 else [expected[key].tolist()]
        actual_values = actual[key] if key not in {"market_item_logits", "market_quantity_logits"} else [actual[key]]
        matches = sum(max(range(len(row)), key=row.__getitem__) == max(range(len(other)), key=other.__getitem__) for row, other in zip(expected_values, actual_values))
        assert matches / max(1, len(expected_values)) >= 0.99
    assert abs(float(expected["value"].item()) - float(actual["value"])) < 1e-2
