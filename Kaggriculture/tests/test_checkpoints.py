import copy
import random
from pathlib import Path

import pytest

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION
from kagriculture_agent.model import ACTION_VOCAB, MODEL_VERSION


def _assert_nested_state_equal(actual, expected):
    torch = pytest.importorskip("torch")
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _trained_model_and_optimizer():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()
    optimizer.step()
    return model, optimizer


def _save_complete_checkpoint(path: Path):
    from kagriculture_agent.checkpoints import save_checkpoint

    model, optimizer = _trained_model_and_optimizer()
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        configuration={"seed": 17, "device": "cpu"},
        phase="bc",
        epoch=2,
        round_index=0,
        cursor=11,
        metrics={"loss": 0.25},
        metadata={"run_name": "round-trip"},
    )
    return model, optimizer


def test_checkpoint_round_trip_restores_complete_training_state_and_rng(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import load_checkpoint

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    source_model, source_optimizer = _save_complete_checkpoint(tmp_path / "policy.pt")
    expected_rng_values = (random.random(), float(np.random.random()), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    payload = load_checkpoint(
        tmp_path / "policy.pt",
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
        restore_rng=True,
    )

    assert set(payload) >= {
        "model_state_dict", "optimizer_state_dict", "configuration", "progress",
        "rng_state", "versions", "metrics", "metadata",
    }
    assert payload["configuration"] == {"seed": 17, "device": "cpu"}
    assert payload["progress"] == {
        "phase": "bc", "epoch": 2, "round": 0, "cursor": 11,
    }
    assert payload["versions"] == {
        "model": MODEL_VERSION,
        "engine": ENGINE_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "action_vocab": {key: list(value) for key, value in ACTION_VOCAB.items()},
    }
    assert payload["metrics"] == {"loss": 0.25}
    assert payload["metadata"]["run_name"] == "round-trip"
    assert payload["rng_state"]["python"] is not None
    assert payload["rng_state"]["numpy"] is not None
    assert payload["rng_state"]["torch"] is not None
    for name, tensor in source_model.state_dict().items():
        assert torch.equal(tensor, restored_model.state_dict()[name])
    assert restored_optimizer.state_dict()["state"]
    assert restored_optimizer.state_dict()["param_groups"][0]["lr"] == 0.01
    assert random.random() == expected_rng_values[0]
    assert float(np.random.random()) == expected_rng_values[1]
    assert torch.equal(torch.rand(3), expected_rng_values[2])
    assert source_optimizer.state_dict()["state"].keys() == restored_optimizer.state_dict()["state"].keys()


def test_checkpoint_atomic_writer_rejects_existing_final_symlink(tmp_path):
    from kagriculture_agent.checkpoints import _atomic_write

    target = tmp_path / "safe-checkpoint.pt"
    target.write_bytes(b"existing")
    destination = tmp_path / "checkpoint.pt"
    destination.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _atomic_write(destination, lambda handle: handle.write(b"replacement"))

    assert destination.is_symlink()
    assert target.read_bytes() == b"existing"


def test_checkpoint_load_uses_restricted_weights_only_deserialization(
    tmp_path, monkeypatch,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent import checkpoints

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    real_load = torch.load
    load_kwargs = []

    class TorchProxy:
        def load(self, *args, **kwargs):
            load_kwargs.append(dict(kwargs))
            return real_load(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(checkpoints, "require_torch", lambda: TorchProxy())
    model, optimizer = _trained_model_and_optimizer()

    checkpoints.load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        map_location="cpu",
        restore_rng=False,
    )

    assert load_kwargs == [{"map_location": "cpu", "weights_only": True}]


def test_malformed_rng_state_is_rejected_without_partial_restoration(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["rng_state"]["numpy"]["keys"] = ["not-an-integer"]
    torch.save(payload, path)

    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    model, optimizer = _trained_model_and_optimizer()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    with pytest.raises(CheckpointError, match="RNG state is malformed"):
        load_checkpoint(path, model=model, optimizer=optimizer, restore_rng=True)

    _assert_nested_state_equal(model.state_dict(), model_before)
    _assert_nested_state_equal(optimizer.state_dict(), optimizer_before)
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_checkpoint_rejects_extra_progress_fields_before_restoration(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["progress"]["unexpected"] = 1
    torch.save(payload, path)
    model, optimizer = _trained_model_and_optimizer()
    model_before = copy.deepcopy(model.state_dict())

    with pytest.raises(CheckpointError, match="progress.*fields"):
        load_checkpoint(path, model=model, optimizer=optimizer)

    _assert_nested_state_equal(model.state_dict(), model_before)


def test_checkpoint_rejects_unknown_progress_phase_before_restoration(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["progress"]["phase"] = "unknown"
    torch.save(payload, path)
    model, optimizer = _trained_model_and_optimizer()
    model_before = copy.deepcopy(model.state_dict())

    with pytest.raises(CheckpointError, match="phase"):
        load_checkpoint(path, model=model, optimizer=optimizer)

    _assert_nested_state_equal(model.state_dict(), model_before)


@pytest.mark.parametrize(
    "version_key,bad_value,error_match",
    [
        ("model", "old", "model.*mismatch"),
        ("engine", "0.0.0", "engine.*mismatch"),
        ("feature_schema", 999, "feature schema.*mismatch"),
        ("action_vocab", {"worker_kinds": ["PASS"]}, "action vocabulary.*mismatch"),
    ],
)
def test_checkpoint_rejects_incompatible_versions(
    tmp_path, version_key, bad_value, error_match,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["versions"][version_key] = bad_value
    torch.save(payload, path)
    model, optimizer = _trained_model_and_optimizer()

    with pytest.raises(CheckpointError, match=error_match):
        load_checkpoint(path, model=model, optimizer=optimizer)


@pytest.mark.parametrize(
    "metadata_key,bad_value,error_match",
    [
        ("model_version", "old", "model.*mismatch"),
        ("engine_version", "0.0.0", "engine.*mismatch"),
        ("feature_schema_version", 999, "feature schema.*mismatch"),
        ("action_vocab", {"worker_kinds": ["PASS"]}, "action vocabulary.*mismatch"),
    ],
)
def test_checkpoint_rejects_incompatible_compatibility_metadata(
    tmp_path, metadata_key, bad_value, error_match,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"][metadata_key] = bad_value
    torch.save(payload, path)
    model, optimizer = _trained_model_and_optimizer()

    with pytest.raises(CheckpointError, match=error_match):
        load_checkpoint(path, model=model, optimizer=optimizer)


def test_checkpoint_rejects_missing_model_version_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "policy.pt"
    _save_complete_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["metadata"]["model_version"]
    torch.save(payload, path)
    model, optimizer = _trained_model_and_optimizer()

    with pytest.raises(CheckpointError, match="model.*mismatch"):
        load_checkpoint(path, model=model, optimizer=optimizer)


@pytest.mark.parametrize("contents", [b"not a checkpoint", b"PK\x03\x04truncated"])
def test_checkpoint_rejects_malformed_and_truncated_payloads(tmp_path, contents):
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / ".policy.pt.interrupted.tmp"
    path.write_bytes(contents)
    model, optimizer = _trained_model_and_optimizer()

    with pytest.raises(CheckpointError, match="malformed|truncated"):
        load_checkpoint(path, model=model, optimizer=optimizer)


def test_checkpoint_rejects_missing_required_payload_fields(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, load_checkpoint

    path = tmp_path / "incomplete.pt"
    torch.save({"model_state_dict": {}}, path)
    model, optimizer = _trained_model_and_optimizer()

    with pytest.raises(CheckpointError, match="missing.*optimizer_state_dict"):
        load_checkpoint(path, model=model, optimizer=optimizer)


def test_checkpoint_missing_file_is_not_treated_as_fresh_state(tmp_path):
    from kagriculture_agent.checkpoints import load_checkpoint

    model, optimizer = _trained_model_and_optimizer()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model=model, optimizer=optimizer)


def test_read_checkpoint_propagates_drive_io_errors_from_deserialization(tmp_path, monkeypatch):
    from kagriculture_agent import checkpoints

    path = tmp_path / "policy.pt"
    path.write_bytes(b"checkpoint bytes")

    class BrokenTorch:
        def load(self, *args, **kwargs):
            raise PermissionError("Drive read unavailable")

    monkeypatch.setattr(checkpoints, "require_torch", lambda: BrokenTorch())

    with pytest.raises(PermissionError, match="Drive read unavailable"):
        checkpoints.read_checkpoint(path)


def test_atomic_checkpoint_save_preserves_destination_and_cleans_temp_on_failure(
    tmp_path, monkeypatch,
):
    from kagriculture_agent import checkpoints

    destination = tmp_path / "best.pt"
    destination.write_bytes(b"previous-best")
    model, optimizer = _trained_model_and_optimizer()

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        checkpoints.save_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            configuration={},
            phase="bc",
            epoch=0,
            round_index=0,
            cursor=0,
            metrics={},
        )

    assert destination.read_bytes() == b"previous-best"
    assert list(tmp_path.glob(".best.pt.*.tmp")) == []


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_atomic_write_cleans_temp_on_process_interruption(tmp_path, interruption):
    from kagriculture_agent import checkpoints

    destination = tmp_path / "interrupted.pt"

    def interrupt_writer(handle):
        handle.write(b"partial-checkpoint")
        raise interruption

    with pytest.raises(interruption):
        checkpoints._atomic_write(destination, interrupt_writer)

    assert not destination.exists()
    assert list(tmp_path.glob(".interrupted.pt.*.tmp")) == []


def test_registry_write_failure_cannot_modify_best_checkpoint(tmp_path, monkeypatch):
    from kagriculture_agent import checkpoints

    best = tmp_path / "best.pt"
    _save_complete_checkpoint(best)
    original_best = best.read_bytes()
    registry = tmp_path / "registry.json"

    def fail_replace(_source, _destination):
        raise OSError("registry replace failed")

    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    with pytest.raises(OSError, match="registry replace failed"):
        checkpoints.save_registry(registry, {"best": str(best), "candidates": []})

    assert best.read_bytes() == original_best
    assert not registry.exists()
    assert list(tmp_path.glob(".registry.json.*.tmp")) == []


def test_atomic_write_fsyncs_parent_directory_after_replace(tmp_path, monkeypatch):
    from kagriculture_agent import checkpoints

    fsynced = []
    monkeypatch.setattr(
        checkpoints,
        "_fsync_parent_directory",
        lambda path: fsynced.append(Path(path)),
    )

    checkpoints._atomic_write(tmp_path / "durable.bin", lambda handle: handle.write(b"ok"))

    assert fsynced == [tmp_path]


def test_parent_directory_fsync_opens_syncs_and_closes_directory(tmp_path, monkeypatch):
    from kagriculture_agent import checkpoints

    calls = []
    monkeypatch.setattr(checkpoints.os, "open", lambda path, flags: calls.append(("open", Path(path), flags)) or 37)
    monkeypatch.setattr(checkpoints.os, "fsync", lambda descriptor: calls.append(("fsync", descriptor)))
    monkeypatch.setattr(checkpoints.os, "close", lambda descriptor: calls.append(("close", descriptor)))

    checkpoints._fsync_parent_directory(tmp_path)

    assert calls[0][0:2] == ("open", tmp_path)
    assert calls[1:] == [("fsync", 37), ("close", 37)]


def test_publish_checkpoint_routes_copy_through_atomic_writer(tmp_path, monkeypatch):
    from kagriculture_agent import checkpoints

    source = tmp_path / "candidate.pt"
    destination = tmp_path / "best.pt"
    source.write_bytes(b"candidate")
    calls = []

    def atomic_write(path, writer):
        calls.append(Path(path))
        with destination.open("wb") as handle:
            writer(handle)
        return str(path)

    monkeypatch.setattr(checkpoints, "_atomic_write", atomic_write)

    assert checkpoints.publish_checkpoint(source, destination) == str(destination)
    assert calls == [destination]
    assert destination.read_bytes() == b"candidate"


def test_restore_rolls_back_model_and_optimizer_after_partial_optimizer_mutation(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import CheckpointError, read_checkpoint, restore_checkpoint

    path = tmp_path / "source.pt"
    _save_complete_checkpoint(path)
    payload = read_checkpoint(path)
    model, optimizer = _trained_model_and_optimizer()
    with torch.no_grad():
        model.weight.add_(10.0)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    real_optimizer_load = optimizer.load_state_dict
    load_calls = 0

    def mutate_then_fail_once(state):
        nonlocal load_calls
        load_calls += 1
        real_optimizer_load(state)
        if load_calls == 1:
            raise RuntimeError("partial optimizer restore")

    optimizer.load_state_dict = mutate_then_fail_once

    with pytest.raises(CheckpointError, match="model or optimizer"):
        restore_checkpoint(payload, model=model, optimizer=optimizer, restore_rng=False)

    assert load_calls == 2
    _assert_nested_state_equal(model.state_dict(), model_before)
    _assert_nested_state_equal(optimizer.state_dict(), optimizer_before)


def test_cuda_checkpoint_can_restore_on_cpu_when_cuda_is_available(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable for a real cross-device checkpoint")
    from kagriculture_agent.checkpoints import load_checkpoint, save_checkpoint

    source_model = torch.nn.Linear(2, 1).to("cuda")
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    loss = source_model(torch.tensor([[1.0, 2.0]], device="cuda")).sum()
    loss.backward()
    source_optimizer.step()
    path = tmp_path / "cuda.pt"
    save_checkpoint(
        path,
        model=source_model,
        optimizer=source_optimizer,
        configuration={},
        phase="final",
        epoch=1,
        round_index=0,
        cursor=0,
        metrics={},
    )
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)

    load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
        restore_rng=False,
    )

    assert all(parameter.device.type == "cpu" for parameter in restored_model.parameters())
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for state in restored_optimizer.state.values()
        for value in state.values()
    )
