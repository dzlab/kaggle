"""Validated, atomic training checkpoints for Kaggriculture policies."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

from .constants import ENGINE_VERSION
from .features import FEATURE_SCHEMA_VERSION
from .model import ACTION_VOCAB, MODEL_VERSION, require_torch

CHECKPOINT_FORMAT_VERSION = 1
_REQUIRED_PAYLOAD_FIELDS = (
    "model_state_dict",
    "optimizer_state_dict",
    "configuration",
    "progress",
    "rng_state",
    "versions",
    "metrics",
    "metadata",
)
_PAYLOAD_FIELDS = {"checkpoint_format_version", *_REQUIRED_PAYLOAD_FIELDS}
_LEGACY_PROGRESS_FIELDS = {"epoch", "round", "cursor"}
_PROGRESS_FIELDS = {"phase", *_LEGACY_PROGRESS_FIELDS}
_CHECKPOINT_PHASES = {"bc", "ppo", "final"}
_RNG_FIELDS = {"python", "numpy", "torch", "torch_cuda"}
_NUMPY_RNG_FIELDS = {
    "bit_generator", "keys", "position", "has_gauss", "cached_gaussian",
}


class CheckpointError(ValueError):
    """A checkpoint exists but cannot be safely resumed."""


def _serialized_action_vocab() -> dict[str, list[Any]]:
    return {key: list(value) for key, value in ACTION_VOCAB.items()}


def _current_versions() -> dict[str, Any]:
    return {
        "model": MODEL_VERSION,
        "engine": ENGINE_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "action_vocab": _serialized_action_vocab(),
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch random-number generator state."""
    th = require_torch()
    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - NumPy is a training dependency
        numpy_state = None
    else:
        raw_numpy_state = np.random.get_state()
        numpy_state = {
            "bit_generator": raw_numpy_state[0],
            "keys": raw_numpy_state[1].tolist(),
            "position": raw_numpy_state[2],
            "has_gauss": raw_numpy_state[3],
            "cached_gaussian": raw_numpy_state[4],
        }
    cuda_state = None
    if th.cuda.is_available():
        cuda_state = th.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch": th.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore all RNG state available in the current training environment."""
    th = require_torch()
    try:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        if numpy_state is not None:
            import numpy as np

            np.random.set_state((
                numpy_state["bit_generator"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            ))
        th.set_rng_state(state["torch"].cpu())
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and th.cuda.is_available():
            th.cuda.set_rng_state_all(cuda_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError("checkpoint RNG state is malformed") from exc


def _validate_rng_state(state: Any) -> None:
    th = require_torch()
    if type(state) is not dict or set(state) != _RNG_FIELDS:
        raise CheckpointError("checkpoint RNG state is malformed")
    try:
        probe = random.Random()
        probe.setstate(state["python"])
        numpy_state = state["numpy"]
        if numpy_state is not None:
            if type(numpy_state) is not dict or set(numpy_state) != _NUMPY_RNG_FIELDS:
                raise ValueError("invalid NumPy RNG fields")
            import numpy as np

            numpy_probe = np.random.RandomState()
            numpy_probe.set_state((
                numpy_state["bit_generator"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            ))
        torch_state = state["torch"]
        if not th.is_tensor(torch_state):
            raise TypeError("invalid PyTorch RNG state")
        th.Generator(device="cpu").set_state(torch_state.cpu())
        cuda_state = state["torch_cuda"]
        if cuda_state is not None:
            if type(cuda_state) is not list or not all(th.is_tensor(item) for item in cuda_state):
                raise TypeError("invalid CUDA RNG state")
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise CheckpointError("checkpoint RNG state is malformed") from exc


def _atomic_write(destination: str | Path, writer: Callable[[BinaryIO], None]) -> str:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path)


def _fsync_parent_directory(path: str | Path) -> None:
    """Durably flush a directory after atomically publishing a child entry."""
    directory = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    configuration: Mapping[str, Any],
    phase: str | None = None,
    epoch: int,
    round_index: int,
    cursor: int,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Save a complete checkpoint by fsyncing a same-directory temp then replacing."""
    th = require_torch()
    progress = {"epoch": epoch, "round": round_index, "cursor": cursor}
    if phase is not None:
        progress["phase"] = phase
    _validate_progress(progress)
    versions = _current_versions()
    checkpoint_metadata = dict(metadata or {})
    checkpoint_metadata.update({
        "model_version": MODEL_VERSION,
        "engine_version": versions["engine"],
        "feature_schema_version": versions["feature_schema"],
        "action_vocab": versions["action_vocab"],
    })
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "configuration": dict(configuration),
        "progress": progress,
        "rng_state": capture_rng_state(),
        "versions": versions,
        "metrics": dict(metrics),
        "metadata": checkpoint_metadata,
    }
    return _atomic_write(path, lambda handle: th.save(payload, handle))


def _validate_progress(progress: Any) -> None:
    if type(progress) is not dict:
        raise CheckpointError("checkpoint progress must be an object")
    if set(progress) not in (_LEGACY_PROGRESS_FIELDS, _PROGRESS_FIELDS):
        raise CheckpointError("checkpoint progress fields are invalid")
    for field in ("epoch", "round", "cursor"):
        value = progress[field]
        if type(value) is not int or value < 0:
            raise CheckpointError(f"checkpoint progress {field} must be a nonnegative integer")
    if "phase" in progress and (
        type(progress["phase"]) is not str or progress["phase"] not in _CHECKPOINT_PHASES
    ):
        raise CheckpointError("checkpoint progress phase is invalid")


def _validate_payload(payload: Any) -> dict[str, Any]:
    if type(payload) is not dict:
        raise CheckpointError("checkpoint payload must be an object")
    missing = [field for field in _REQUIRED_PAYLOAD_FIELDS if field not in payload]
    if missing:
        raise CheckpointError(f"checkpoint is missing required fields: {', '.join(missing)}")
    unexpected = sorted(set(payload) - _PAYLOAD_FIELDS)
    if unexpected:
        raise CheckpointError(f"checkpoint has unexpected fields: {', '.join(unexpected)}")
    if (
        type(payload.get("checkpoint_format_version")) is not int
        or payload["checkpoint_format_version"] != CHECKPOINT_FORMAT_VERSION
    ):
        raise CheckpointError("checkpoint format version mismatch")
    for field in ("configuration", "rng_state", "metrics", "metadata"):
        if type(payload[field]) is not dict:
            raise CheckpointError(f"checkpoint {field} must be an object")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise CheckpointError("checkpoint model_state_dict must be an object")
    optimizer_state = payload["optimizer_state_dict"]
    if type(optimizer_state) is not dict:
        raise CheckpointError("checkpoint optimizer_state_dict must be an object")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise CheckpointError("checkpoint optimizer_state_dict fields are invalid")
    if not isinstance(optimizer_state["state"], Mapping) or type(optimizer_state["param_groups"]) is not list:
        raise CheckpointError("checkpoint optimizer_state_dict is malformed")
    _validate_progress(payload["progress"])
    _validate_rng_state(payload["rng_state"])
    versions = payload["versions"]
    if type(versions) is not dict:
        raise CheckpointError("checkpoint versions must be an object")
    expected = _current_versions()
    if set(versions) != set(expected):
        raise CheckpointError("checkpoint versions fields are invalid")
    if type(versions["model"]) is not type(expected["model"]) or versions["model"] != expected["model"]:
        raise CheckpointError("checkpoint model version mismatch")
    if type(versions["engine"]) is not type(expected["engine"]) or versions["engine"] != expected["engine"]:
        raise CheckpointError("checkpoint engine version mismatch")
    if (
        type(versions["feature_schema"]) is not type(expected["feature_schema"])
        or versions["feature_schema"] != expected["feature_schema"]
    ):
        raise CheckpointError("checkpoint feature schema version mismatch")
    if type(versions["action_vocab"]) is not dict or versions["action_vocab"] != expected["action_vocab"]:
        raise CheckpointError("checkpoint action vocabulary mismatch")
    metadata = payload["metadata"]
    if metadata.get("model_version") != expected["model"]:
        raise CheckpointError("checkpoint metadata model version mismatch")
    if metadata.get("engine_version") != expected["engine"]:
        raise CheckpointError("checkpoint metadata engine version mismatch")
    if metadata.get("feature_schema_version") != expected["feature_schema"]:
        raise CheckpointError("checkpoint metadata feature schema version mismatch")
    if metadata.get("action_vocab") != expected["action_vocab"]:
        raise CheckpointError("checkpoint metadata action vocabulary mismatch")
    return payload


def read_checkpoint(
    path: str | Path, *, map_location: Any = "cpu",
) -> dict[str, Any]:
    """Safely deserialize and fully validate a checkpoint without restoring it."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    th = require_torch()
    try:
        raw_payload = th.load(
            checkpoint_path, map_location=map_location, weights_only=True,
        )
    except Exception as exc:
        raise CheckpointError(
            f"checkpoint is malformed or truncated: {checkpoint_path}"
        ) from exc
    return _validate_payload(raw_payload)


def publish_checkpoint(source: str | Path, destination: str | Path) -> str:
    """Copy a checkpoint to its published path through an atomic writer."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"checkpoint candidate does not exist: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("checkpoint source and destination must be different paths")
    if destination_path.exists():
        try:
            if os.path.samefile(source_path, destination_path):
                raise ValueError("checkpoint source and destination must not alias")
        except FileNotFoundError:
            pass

    def copy_source(handle: BinaryIO) -> None:
        with source_path.open("rb") as source_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                handle.write(chunk)

    return _atomic_write(destination_path, copy_source)


def restore_checkpoint(
    payload: dict[str, Any], *, model: Any, optimizer: Any,
    restore_rng: bool = True,
) -> None:
    """Transactionally restore an already validated checkpoint payload."""
    payload = _validate_payload(payload)
    try:
        model_before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        rng_before = capture_rng_state() if restore_rng else None
    except Exception as exc:
        raise CheckpointError("could not snapshot current training state") from exc
    try:
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if restore_rng:
            restore_rng_state(payload["rng_state"])
    except Exception as exc:
        rollback_errors = []
        for restore in (
            lambda: model.load_state_dict(model_before),
            lambda: optimizer.load_state_dict(optimizer_before),
            lambda: restore_rng_state(rng_before) if rng_before is not None else None,
        ):
            try:
                restore()
            except Exception as rollback_exc:  # pragma: no cover - catastrophic custom state objects
                rollback_errors.append(rollback_exc)
        if rollback_errors:
            raise CheckpointError(
                "checkpoint restoration failed and current state could not be rolled back"
            ) from exc
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError("checkpoint model or optimizer state is malformed") from exc


def load_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    map_location: Any = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Validate and restore a complete checkpoint into a model and optimizer."""
    payload = read_checkpoint(path, map_location=map_location)
    restore_checkpoint(
        payload, model=model, optimizer=optimizer, restore_rng=restore_rng,
    )
    return payload


def save_registry(path: str | Path, registry: Mapping[str, Any]) -> str:
    """Persist registry metadata separately from tensor checkpoint bytes."""
    if not isinstance(registry, Mapping):
        raise TypeError("checkpoint registry must be a mapping")

    def write_json(handle: BinaryIO) -> None:
        encoded = json.dumps(
            dict(registry), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        handle.write(encoded)

    return _atomic_write(path, write_json)
