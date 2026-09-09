"""Export a Task 6 checkpoint to the dependency-free learned-policy format."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import (
    EXPERIMENTAL_FEATURE_VARIANT,
    FEATURE_SCHEMA_VERSION,
    PRODUCTION_FEATURE_VARIANT,
)
from kagriculture_agent.learned_policy import artifact_tensor_shapes
from kagriculture_agent.model import (
    ACTION_VOCAB,
    DEFAULT_MODEL_DEPTH,
    HIDDEN_WIDTH,
    MODEL_VERSION,
    validate_model_shape,
)
from scripts.output_paths import validate_training_output_path
from scripts.training_identity import (
    DEFAULT_ACTION_REPRESENTATION,
    validate_action_representation,
    validate_identity_consistency,
)

FORMAT_VERSION = 1
QUANTIZATION = "int8-per-row"
_TRAINING_ONLY_TENSOR_NAMES = frozenset({
    "market_active_head.weight", "market_active_head.bias",
})


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def artifact_checksum(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "checksum"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _float32(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("tensor contains a non-numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("tensor contains a non-finite value")
    try:
        return struct.unpack("<f", struct.pack("<f", result))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("tensor value is outside fp32 range") from exc


def quantize_rowwise(values: Any) -> dict[str, Any]:
    """Quantize a rank-2 tensor, retaining one fp32 scale per output row."""
    rows = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(rows, (list, tuple)) or not rows or not all(isinstance(row, (list, tuple)) for row in rows):
        raise ValueError("row-wise quantization requires a non-empty rank-2 tensor")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("tensor rows must have equal non-zero width")
    scales: list[float] = []
    quantized: list[list[int]] = []
    for row in rows:
        floats = [_float32(item) for item in row]
        maximum = max(abs(item) for item in floats)
        scale = _float32(maximum / 127.0) if maximum else _float32(1.0 / 127.0)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("tensor row has an invalid quantization scale")
        scales.append(scale)
        quantized.append([max(-128, min(127, int(round(item / scale)))) for item in floats])
    return {"shape": [len(rows), width], "scales": scales, "values": quantized}


def _plain_tensor(values: Any) -> dict[str, Any]:
    data = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(data, (list, tuple)):
        raise ValueError("tensor must be an array")
    flat = list(data)
    if flat and isinstance(flat[0], (list, tuple)):
        raise ValueError("only rank-1 tensors may use fp32 storage")
    return {"shape": [len(flat)], "values": [_float32(item) for item in flat]}


def _tensor_to_artifact(name: str, tensor: Any, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    if not hasattr(tensor, "detach"):
        raise ValueError(f"checkpoint tensor {name!r} is not a torch tensor")
    value = tensor.detach().cpu()
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"checkpoint tensor {name!r} shape mismatch: expected {expected_shape}, got {tuple(value.shape)}")
    if value.ndim == 2:
        return quantize_rowwise(value)
    if value.ndim == 1:
        return _plain_tensor(value)
    raise ValueError(f"unsupported tensor rank for {name!r}: {value.ndim}")


def expected_tensor_names(
    *, action_representation: str = DEFAULT_ACTION_REPRESENTATION,
) -> tuple[str, ...]:
    return tuple(artifact_tensor_shapes(action_representation=action_representation))


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_strict_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_strict_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def validate_action_vocab(action_vocab: Any) -> dict[str, list[Any]]:
    expected_vocab = {key: list(value) for key, value in ACTION_VOCAB.items()}
    if not _strict_equal(action_vocab, expected_vocab):
        raise ValueError("checkpoint action_vocab mismatch")
    return expected_vocab


def _checkpoint_model_shape(metadata: dict[str, Any]) -> tuple[int, int]:
    width = metadata.get("model_width", metadata.get("hidden_width", HIDDEN_WIDTH))
    depth = metadata.get("model_depth", metadata.get("depth", DEFAULT_MODEL_DEPTH))
    try:
        return validate_model_shape(width, depth, source="checkpoint")
    except ValueError as exc:
        raise ValueError(f"checkpoint model shape is invalid: {exc}") from exc


def validate_checkpoint_metadata(metadata: Any) -> dict[str, list[Any]]:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata is required")
    for key, expected in (
        ("model_version", MODEL_VERSION),
        ("feature_schema_version", FEATURE_SCHEMA_VERSION),
        ("engine_version", ENGINE_VERSION),
    ):
        if not _strict_equal(metadata.get(key), expected):
            raise ValueError(f"checkpoint {key} mismatch")
    feature_variant = metadata.get("feature_variant", PRODUCTION_FEATURE_VARIANT)
    if feature_variant not in {PRODUCTION_FEATURE_VARIANT, EXPERIMENTAL_FEATURE_VARIANT}:
        raise ValueError("checkpoint feature_variant mismatch")
    validate_action_representation(
        metadata.get("action_representation", DEFAULT_ACTION_REPRESENTATION),
        source="checkpoint",
    )
    nested_identity = metadata.get("ppo_config")
    if nested_identity is not None:
        validate_identity_consistency(
            metadata, nested_identity, source="checkpoint metadata",
        )
        if "action_representation" in nested_identity:
            validate_action_representation(
                nested_identity["action_representation"], source="checkpoint ppo_config",
            )
    _checkpoint_model_shape(metadata)
    return validate_action_vocab(metadata.get("action_vocab"))


def validate_checkpoint_state_dict(
    state: Any, *, model_width: int = HIDDEN_WIDTH, model_depth: int = DEFAULT_MODEL_DEPTH,
    feature_variant: str = PRODUCTION_FEATURE_VARIANT,
    action_representation: str = DEFAULT_ACTION_REPRESENTATION,
) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state_dict is required")
    shapes = artifact_tensor_shapes(
        model_width, model_depth, feature_variant, action_representation,
    )
    expected = set(shapes)
    actual = set(state)
    missing = expected - actual
    unexpected = actual - expected - _TRAINING_ONLY_TENSOR_NAMES
    if missing or unexpected:
        missing = sorted(missing)
        extra = sorted(unexpected)
        raise ValueError(f"checkpoint tensors mismatch (missing={missing}, extra={extra})")
    for name, tensor in state.items():
        if name not in expected:
            continue
        actual_shape = tuple(getattr(tensor, "shape", ()))
        if actual_shape != shapes[name]:
            raise ValueError(f"checkpoint tensor {name!r} shape mismatch: expected {shapes[name]}, got {actual_shape}")


def build_artifact(
    state: dict[str, Any], action_vocab: dict[str, list[Any]], *,
    model_width: int = HIDDEN_WIDTH, model_depth: int = DEFAULT_MODEL_DEPTH,
    feature_variant: str = PRODUCTION_FEATURE_VARIANT,
    action_representation: str = DEFAULT_ACTION_REPRESENTATION,
) -> dict[str, Any]:
    """Build the serialized artifact after checkpoint validation."""
    validate_model_shape(model_width, model_depth, source="artifact")
    validate_action_representation(action_representation, source="artifact")
    validate_checkpoint_state_dict(
        state, model_width=model_width, model_depth=model_depth,
        feature_variant=feature_variant,
        action_representation=action_representation,
    )
    validated_vocab = validate_action_vocab(action_vocab)
    shapes = artifact_tensor_shapes(
        model_width, model_depth, feature_variant, action_representation,
    )
    artifact: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "hidden_width": model_width,
        "model_depth": model_depth,
        "quantization": QUANTIZATION,
        "action_vocab": validated_vocab,
        "weights": {name: _tensor_to_artifact(name, state[name], shapes[name]) for name in shapes},
    }
    if feature_variant != PRODUCTION_FEATURE_VARIANT:
        artifact["feature_variant"] = feature_variant
    if action_representation != DEFAULT_ACTION_REPRESENTATION:
        artifact["action_representation"] = action_representation
    artifact["checksum"] = artifact_checksum(artifact)
    return artifact


def _load_checkpoint_safely(torch: Any, checkpoint_path: str | Path) -> Any:
    """Load tensor-only checkpoints; never fall back to arbitrary pickle."""
    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cannot verify that this PyTorch version supports safe checkpoint loading") from exc
    if "weights_only" not in parameters:
        raise RuntimeError("PyTorch version lacks safe weights_only checkpoint loading; upgrade PyTorch")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch safe weights_only checkpoint loading is unavailable") from exc


def write_artifact(artifact: dict[str, Any], artifact_path: str | Path) -> None:
    """Atomically publish an artifact without following an existing symlink."""
    destination = validate_training_output_path(
        artifact_path,
        name="artifact path",
        reject_protected_names=True,
        reject_symlink_components=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(artifact) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        destination = validate_training_output_path(
            destination,
            name="artifact path",
            reject_protected_names=True,
            reject_symlink_components=True,
        )
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def export_checkpoint(
    checkpoint_path: str | Path, artifact_path: str | Path, *,
    model_width: int | None = None, model_depth: int | None = None,
) -> dict[str, Any]:
    """Export a torch checkpoint, failing clearly when torch is unavailable."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to export a checkpoint; install the training extra") from exc
    checkpoint = _load_checkpoint_safely(torch, checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be an object")
    metadata = checkpoint.get("metadata")
    expected_vocab = validate_checkpoint_metadata(metadata)
    feature_variant = metadata.get("feature_variant", PRODUCTION_FEATURE_VARIANT)
    action_representation = metadata.get(
        "action_representation", DEFAULT_ACTION_REPRESENTATION,
    )
    checkpoint_width, checkpoint_depth = _checkpoint_model_shape(metadata)
    if model_width is not None or model_depth is not None:
        requested_width = checkpoint_width if model_width is None else model_width
        requested_depth = checkpoint_depth if model_depth is None else model_depth
        requested_shape = validate_model_shape(requested_width, requested_depth, source="export request")
        if requested_shape != (checkpoint_width, checkpoint_depth):
            raise ValueError(
                "export model shape does not match checkpoint metadata: "
                f"checkpoint={(checkpoint_width, checkpoint_depth)!r}, requested={requested_shape!r}"
            )
    state = checkpoint.get("model_state_dict")
    artifact = build_artifact(
        state, expected_vocab, model_width=checkpoint_width, model_depth=checkpoint_depth,
        feature_variant=feature_variant,
        action_representation=action_representation,
    )
    write_artifact(artifact, artifact_path)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        export_checkpoint(args.checkpoint, args.artifact)
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
