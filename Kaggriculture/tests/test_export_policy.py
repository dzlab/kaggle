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
            weights[name] = {"shape": list(shape), "values": [((index % 9) - 4) * 0.001 for index in range(shape[0])]}
        else:
            weights[name] = {
                "shape": list(shape),
                "scales": [0.01] * shape[0],
                "values": [[((row * 13 + column * 7) % 255) - 127 for column in range(shape[1])]
                           for row in range(shape[0])],
            }
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
    broken["checksum"] = __import__("scripts.export_policy", fromlist=["artifact_checksum"]).artifact_checksum(broken)
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_exported_policy(path)


def test_action_vocabulary_validator_rejects_uncompilable_class_with_diagnostics():
    from kagriculture_agent.learned_policy import validate_action_vocabulary

    vocabulary = _artifact()["action_vocab"]
    vocabulary["worker_kinds"][2] = "TELEPORT"

    with pytest.raises(ValueError, match="TELEPORT.*learned_v1.*action_vocab"):
        validate_action_vocabulary(
            vocabulary, model_version="learned_v1", source="test artifact",
        )


def test_loaded_artifact_reports_vocabulary_and_version(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    runtime = load_exported_policy(path)

    assert runtime.model_version == "learned_v1"
    assert runtime.action_vocab["worker_kinds"][:2] == ["PASS", "MOVE"]
    assert runtime.task_intent_loss_mask["excluded_worker_kinds"] == ["PASS", "MOVE"]


def test_train_export_load_compile_action_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.learned_policy import (
        COMPILER_VALID_WORKER_KINDS,
        compile_proposal,
    )
    from kagriculture_agent.memory import PolicyMemory
    from kagriculture_agent.model import ACTION_VOCAB, CompactPolicyNet
    from scripts.export_policy import export_checkpoint
    from scripts.train_policy import checkpoint_metadata

    network = CompactPolicyNet()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.worker_act_head.bias[1] = 1.0
        network.worker_kind_head.bias[
            ACTION_VOCAB["worker_kinds"].index("WATER")
        ] = 1.0
    checkpoint = tmp_path / "policy.pt"
    artifact_path = tmp_path / "policy.json"
    torch.save({
        "metadata": checkpoint_metadata(transition_count=1),
        "model_state_dict": network.state_dict(),
    }, checkpoint)

    artifact = export_checkpoint(checkpoint, artifact_path)
    runtime = load_exported_policy(artifact_path)
    state = {
        "board_size": 10,
        "day": 1,
        "hour": 1,
        "cash": 100,
        "tiles": [[None] * 10 for _ in range(10)],
        "workers": [{"index": 0, "role": "FARMER", "position": [0, 0]}],
        "private": {"inventories": [{}], "shed": {}, "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    state["tiles"][0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "watered_today": False,
    }
    proposal = runtime.propose(state, extract_features(state))
    action = compile_proposal(state, proposal, PolicyMemory())

    assert artifact["action_vocab"] == checkpoint_metadata(1)["action_vocab"]
    assert proposal.workers
    assert {worker.kind for worker in proposal.workers} <= COMPILER_VALID_WORKER_KINDS
    assert action["farmer"] == ["WATER"]


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
        "engine_version": "1.32.7", "hidden_width": 65,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
    }
    with pytest.raises(ValueError, match="hidden_width"):
        validate_checkpoint_metadata(metadata)


def test_exporter_rejects_conflicting_top_level_and_nested_action_representation():
    from kagriculture_agent.model import ACTION_VOCAB
    from scripts.export_policy import validate_checkpoint_metadata

    metadata = {
        "model_version": "learned_v1", "feature_schema_version": 1,
        "engine_version": "1.32.7",
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
        "action_representation": "current_v1",
        "ppo_config": {"action_representation": "target_first_v1"},
    }

    with pytest.raises(ValueError, match="action_representation.*conflicts"):
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


def test_target_first_artifact_has_distinct_head_and_identity():
    from kagriculture_agent.model import ACTION_VOCAB
    from kagriculture_agent.learned_policy import artifact_tensor_shapes
    from scripts.export_policy import build_artifact

    state = {}
    for name, shape in artifact_tensor_shapes(action_representation="target_first_v1").items():
        values = [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]
        state[name] = _FakeTensor(values, shape)
    artifact = build_artifact(
        state, {key: list(value) for key, value in ACTION_VOCAB.items()},
        action_representation="target_first_v1",
    )

    assert artifact["action_representation"] == "target_first_v1"
    assert "worker_kind_by_target_head.weight" in artifact["weights"]


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


def test_artifact_writer_rejects_symlinked_parent(tmp_path):
    from scripts.export_policy import write_artifact

    target_parent = tmp_path / "safe-parent"
    target_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(target_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_artifact(_artifact(), linked_parent / "policy.json")
    assert not (target_parent / "policy.json").exists()


def test_artifact_writer_rechecks_final_symlink_before_publication(tmp_path, monkeypatch):
    from scripts import export_policy

    destination = tmp_path / "policy.json"
    target = tmp_path / "existing.json"
    target.write_text("existing\n", encoding="utf-8")
    real_validator = export_policy.validate_training_output_path
    calls = 0

    def race_validator(path, **kwargs):
        nonlocal calls
        calls += 1
        result = real_validator(path, **kwargs)
        if calls == 1:
            destination.symlink_to(target)
        return result

    monkeypatch.setattr(export_policy, "validate_training_output_path", race_validator)
    with pytest.raises(ValueError, match="symlink"):
        export_policy.write_artifact(_artifact(), destination)
    assert calls == 2
    assert destination.is_symlink()
    assert target.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".policy.json.*.tmp"))


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


def test_runtime_gelu_exact_parity_matches_torch_default():
    np = pytest.importorskip("numpy", reason="NumPy GELU parity requires NumPy")
    from kagriculture_agent.learned_policy import _gelu, _numpy_gelu

    assert _gelu(1.0) == pytest.approx(0.8413447460685429, abs=1e-12)
    assert _gelu(-1.0) == pytest.approx(-0.15865525393145707, abs=1e-12)
    values = np.asarray([
        [-4.0, -1.0, 0.0, 1.0, 4.0],
        [-3.0, -0.5, 0.5, 2.0, 3.0],
    ], dtype=np.float32)
    expected = np.asarray([
        [_gelu(float(value)) for value in row]
        for row in values
    ])
    np.testing.assert_allclose(
        _numpy_gelu(values), expected, rtol=1e-7, atol=1e-7,
    )


def test_torch_numpy_and_pure_python_runtime_have_logit_and_action_parity(tmp_path):
    torch = pytest.importorskip("torch", reason="three-runtime parity requires PyTorch")
    np = pytest.importorskip("numpy", reason="three-runtime parity requires NumPy")
    from kagriculture_agent.model import CompactPolicyNet

    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    runtime = load_exported_policy(path)
    assert runtime._numpy_weights is not None

    def zeros_like(value):
        return [zeros_like(item) for item in value] if isinstance(value, list) else 0.0

    runtime._weights = {
        name: zeros_like(value) for name, value in runtime._weights.items()
    }
    for block in range(runtime._model_depth):
        for norm in ("attention_norm", "mlp_norm"):
            runtime._weights[f"blocks.{block}.{norm}.weight"] = [1.0] * 128
    runtime._weights["blocks.0.mlp.0.bias"] = [
        -4.0 + 8.0 * index / 255.0 for index in range(256)
    ]
    runtime._weights["blocks.0.mlp.2.weight"] = [
        [1.0 if column == row else 0.0 for column in range(256)]
        for row in range(128)
    ]
    runtime._weights["worker_act_head.weight"][0][64] = 1.0
    runtime._weights["worker_act_head.weight"][1][96] = 1.0
    runtime._numpy_weights = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in runtime._weights.items()
    }

    network = CompactPolicyNet()
    incompatible = network.load_state_dict({
        name: torch.tensor(value, dtype=torch.float32)
        for name, value in runtime._weights.items()
    }, strict=False)
    assert incompatible.missing_keys == [
        "market_active_head.weight", "market_active_head.bias",
    ]
    assert incompatible.unexpected_keys == []
    assert network.blocks[0].mlp[1].approximate == "none"

    features = extract_features({"day": 7, "hour": 13, "cash": 42.5})
    with torch.no_grad():
        torch_raw = network(features)
    torch_outputs = {
        name: torch_raw[name][0].tolist()
        for name in (
            "worker_act_logits", "worker_target_logits", "worker_kind_logits",
            "market_item_logits", "market_quantity_logits",
        )
    }
    numpy_outputs = runtime.predict(features)
    numpy_proposal = runtime.propose({}, features)
    runtime._numpy_weights = None
    python_outputs = runtime.predict(features)
    python_proposal = runtime.propose({}, features)

    def selected_action(outputs):
        workers = []
        for act_logits, target_logits, kind_logits in zip(
            outputs["worker_act_logits"],
            outputs["worker_target_logits"],
            outputs["worker_kind_logits"],
        ):
            act = max(range(len(act_logits)), key=act_logits.__getitem__)
            workers.append(None if act == 0 else (
                max(range(len(target_logits)), key=target_logits.__getitem__),
                max(range(len(kind_logits)), key=kind_logits.__getitem__),
            ))
        market = (
            max(range(len(outputs["market_item_logits"])),
                key=outputs["market_item_logits"].__getitem__),
            max(range(len(outputs["market_quantity_logits"])),
                key=outputs["market_quantity_logits"].__getitem__),
        )
        return tuple(workers), market

    for name, expected in torch_outputs.items():
        torch.testing.assert_close(
            torch.tensor(numpy_outputs[name]), torch.tensor(expected),
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            torch.tensor(python_outputs[name]), torch.tensor(expected),
            rtol=1e-5, atol=1e-6,
        )
    assert selected_action(torch_outputs) == selected_action(numpy_outputs)
    assert selected_action(torch_outputs) == selected_action(python_outputs)
    assert numpy_proposal == python_proposal


def test_export_cli_returns_clean_nonzero_error_without_traceback(tmp_path, capsys):
    from scripts.export_policy import main

    assert main([str(tmp_path / "missing.pt"), str(tmp_path / "policy.json")]) == 2
    captured = capsys.readouterr()
    assert "export failed:" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.performance
def test_numpy_full_state_inference_reports_latency_within_budget(tmp_path):
    from scripts.benchmark_rollouts import MAX_POLICY_INFERENCE_P95_MS

    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    policy = load_exported_policy(path)
    features = extract_features({})
    assert policy._numpy_weights is not None
    assert (
        len(features.tile_tokens),
        len(features.worker_tokens),
        len(features.market_tokens),
    ) == (100, 10, 9)

    for _ in range(20):
        policy.predict(features)
    samples = []
    for _ in range(1000):
        started = time.perf_counter()
        policy.predict(features)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    p95_ms = samples[949]
    max_ms = samples[-1]
    print(f"NumPy full-state inference: p95={p95_ms:.3f} ms max={max_ms:.3f} ms")
    assert p95_ms < MAX_POLICY_INFERENCE_P95_MS, (
        f"NumPy full-state inference p95 {p95_ms:.3f} ms exceeds "
        f"{MAX_POLICY_INFERENCE_P95_MS:g} ms promotion budget; "
        f"max was {max_ms:.3f} ms"
    )


def test_exported_model_agrees_with_training_fixture_when_torch_is_available(tmp_path):
    torch = pytest.importorskip("torch", reason="training-model agreement requires PyTorch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts.export_policy import export_checkpoint

    checkpoint = tmp_path / "policy.pt"
    artifact_path = tmp_path / "policy.json"
    torch.manual_seed(0)
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


@pytest.mark.parametrize("model_width,model_depth", [(256, 4), (128, 8)])
def test_non_default_model_shape_round_trips_through_export_and_runtime(
    tmp_path, model_width, model_depth,
):
    torch = pytest.importorskip("torch", reason="model export requires PyTorch")
    from kagriculture_agent.model import ACTION_VOCAB, CompactPolicyNet
    from scripts.export_policy import export_checkpoint

    checkpoint = tmp_path / f"policy-{model_width}x{model_depth}.pt"
    artifact_path = tmp_path / f"policy-{model_width}x{model_depth}.json"
    torch.manual_seed(0)
    network = CompactPolicyNet(hidden_width=model_width, depth=model_depth)
    metadata = {
        "model_version": "learned_v1", "feature_schema_version": 1,
        "engine_version": "1.32.7", "model_width": model_width,
        "model_depth": model_depth,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
    }
    torch.save({"metadata": metadata, "model_state_dict": network.state_dict()}, checkpoint)

    artifact = export_checkpoint(checkpoint, artifact_path)
    assert artifact["hidden_width"] == model_width
    assert artifact["model_depth"] == model_depth

    runtime = load_exported_policy(artifact_path)
    features = extract_features({"day": 7, "hour": 13, "cash": 42.5})
    outputs = runtime.predict(features)
    with torch.no_grad():
        reference = network(features)
    reference_outputs = {
        "worker_act_logits": reference["worker_act_logits"][0],
        "worker_target_logits": reference["worker_target_logits"][0],
        "worker_kind_logits": reference["worker_kind_logits"][0],
        "market_item_logits": reference["market_item_logits"][0],
        "market_quantity_logits": reference["market_quantity_logits"][0],
        "value": reference["value"].reshape(-1),
    }
    for name, expected in reference_outputs.items():
        actual = outputs[name] if name != "value" else [outputs[name]]
        torch.testing.assert_close(
            torch.tensor(actual, dtype=torch.float32), expected.cpu(),
            rtol=0.0, atol=0.02,
        )
