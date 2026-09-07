import json
import time
from pathlib import Path

import pytest

from kagriculture_agent.features import extract_features
from kagriculture_agent.learned_policy import LearnedPolicy, load_exported_policy


def pytest_configure(config):
    config.addinivalue_line("markers", "performance: execute the 1000-state runtime gate")


class _FakeTensor:
    def __init__(self, values, shape):
        self._values = values
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


def _artifact():
    from kagriculture_agent import learned_policy as runtime
    from scripts.export_policy import artifact_checksum

    weights = {}
    for name, shape in runtime.artifact_tensor_shapes().items():
        if len(shape) == 1:
            weights[name] = {"shape": list(shape), "values": [0.0] * shape[0]}
        else:
            weights[name] = {"shape": list(shape), "scales": [1.0] * shape[0], "values": [[0] * shape[1] for _ in range(shape[0])]}
    # Keep the latency fixture representative: every runtime path must execute
    # the real dependency-free graph rather than an all-zero fast path.
    weights["tile_projection.weight"]["values"][0][0] = 1
    weights["worker_projection.weight"]["values"][0][0] = -1
    weights["value_head.bias"]["values"][0] = 0.5
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


def test_checksum_valid_architecture_shape_mismatch_is_rejected(tmp_path):
    from scripts.export_policy import artifact_checksum

    artifact = _artifact()
    artifact["weights"]["worker_act_head.weight"]["shape"] = [128, 128]
    artifact["checksum"] = artifact_checksum(artifact)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_exported_policy(path)


@pytest.mark.parametrize("field", ["scales", "values"])
def test_json_numeric_fields_reject_string_values(tmp_path, field):
    from scripts.export_policy import artifact_checksum

    artifact = _artifact()
    if field == "scales":
        artifact["weights"]["worker_act_head.weight"][field][0] = "1.0"
    else:
        artifact["weights"]["worker_act_head.weight"][field][0][0] = "0"
    artifact["checksum"] = artifact_checksum(artifact)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="(JSON number|int8)"):
        load_exported_policy(path)


def test_json_numeric_metadata_rejects_boolean(tmp_path):
    from scripts.export_policy import artifact_checksum

    artifact = _artifact()
    artifact["hidden_width"] = True
    artifact["checksum"] = artifact_checksum(artifact)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
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


def test_exporter_rejects_hidden_width_metadata_mismatch():
    from kagriculture_agent.model import ACTION_VOCAB
    from scripts.export_policy import validate_checkpoint_metadata

    metadata = {
        "model_version": "learned_v1", "feature_schema_version": 1,
        "engine_version": "1.32.7", "hidden_width": 64,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
    }
    with pytest.raises(ValueError, match="hidden_width"):
        validate_checkpoint_metadata(metadata)


def test_exporter_rejects_checkpoint_tensor_shape_mismatch():
    from scripts.export_policy import validate_checkpoint_state_dict
    from kagriculture_agent.learned_policy import artifact_tensor_shapes

    state = {
        name: _FakeTensor([0.0] * (shape[0] if len(shape) == 1 else shape[0] * shape[1]), shape)
        for name, shape in artifact_tensor_shapes().items()
    }
    state["value_head.weight"] = _FakeTensor([0.0] * (128 * 128), (128, 128))
    with pytest.raises(ValueError, match="value_head.weight.*shape mismatch"):
        validate_checkpoint_state_dict(state)


def test_artifact_builder_regression_plumbs_validated_vocab_without_torch():
    from kagriculture_agent.model import ACTION_VOCAB
    from kagriculture_agent.learned_policy import artifact_tensor_shapes
    from scripts.export_policy import build_artifact

    state = {}
    for name, shape in artifact_tensor_shapes().items():
        if len(shape) == 1:
            values = [0.0] * shape[0]
        else:
            values = [[0.0] * shape[1] for _ in range(shape[0])]
        state[name] = _FakeTensor(values, shape)
    artifact = build_artifact(state, {key: list(value) for key, value in ACTION_VOCAB.items()})
    assert artifact["action_vocab"]["worker_kinds"][0] == "PASS"
    assert len(artifact["checksum"]) == 64


def test_build_artifact_validates_action_vocab_independently():
    from scripts.export_policy import build_artifact
    from kagriculture_agent.learned_policy import artifact_tensor_shapes

    state = {}
    for name, shape in artifact_tensor_shapes().items():
        values = [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]
        state[name] = _FakeTensor(values, shape)
    with pytest.raises(ValueError, match="action_vocab"):
        build_artifact(state, {})


def test_artifact_writer_rejects_symlink_destination_and_writes_atomically(tmp_path):
    from scripts.export_policy import write_artifact

    target = tmp_path / "target.json"
    target.write_text("old", encoding="utf-8")
    link = tmp_path / "artifact.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        write_artifact(_artifact(), link)
    assert target.read_text(encoding="utf-8") == "old"

    destination = tmp_path / "new.json"
    write_artifact(_artifact(), destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["format_version"] == 1
    assert not list(tmp_path.glob(".new.json.*.tmp"))


def test_checkpoint_loader_requires_weights_only_support():
    from scripts.export_policy import _load_checkpoint_safely

    class UnsafeTorch:
        @staticmethod
        def load(path, map_location=None):
            return {}

    with pytest.raises(RuntimeError, match="weights_only"):
        _load_checkpoint_safely(UnsafeTorch, "checkpoint.pt")

    calls = {}

    class SafeTorch:
        @staticmethod
        def load(path, map_location=None, weights_only=False):
            calls["weights_only"] = weights_only
            return {}

    assert _load_checkpoint_safely(SafeTorch, "checkpoint.pt") == {}
    assert calls["weights_only"] is True


def test_runtime_gelu_matches_torch_exact_default():
    from kagriculture_agent.learned_policy import _gelu

    assert _gelu(1.0) == pytest.approx(0.8413447460685429, abs=1e-12)
    assert _gelu(-1.0) == pytest.approx(-0.15865525393145707, abs=1e-12)


def test_export_cli_returns_clean_nonzero_error_without_traceback(tmp_path, capsys):
    from scripts.export_policy import main

    assert main([str(tmp_path / "missing.pt"), str(tmp_path / "policy.json")]) == 2
    captured = capsys.readouterr()
    assert "export failed:" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.performance
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
