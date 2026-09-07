"""Export a Task 6 checkpoint to the dependency-free learned-policy format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION
from kagriculture_agent.model import ACTION_VOCAB, HIDDEN_WIDTH, MODEL_VERSION

FORMAT_VERSION = 1
QUANTIZATION = "int8-per-row"


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def artifact_checksum(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "checksum"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _float32(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("tensor contains a non-finite value")
    return struct.unpack("<f", struct.pack("<f", result))[0]


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


def _tensor_to_artifact(name: str, tensor: Any) -> dict[str, Any]:
    if not hasattr(tensor, "detach"):
        raise ValueError(f"checkpoint tensor {name!r} is not a torch tensor")
    value = tensor.detach().cpu()
    if value.ndim == 2:
        return quantize_rowwise(value)
    if value.ndim == 1:
        return _plain_tensor(value)
    raise ValueError(f"unsupported tensor rank for {name!r}: {value.ndim}")


def expected_tensor_names() -> tuple[str, ...]:
    names = [
        "tile_projection.weight", "tile_projection.bias",
        "worker_projection.weight", "worker_projection.bias",
        "market_projection.weight", "market_projection.bias",
        "global_projection.weight", "global_projection.bias",
        "type_embedding.weight",
    ]
    for index in range(4):
        prefix = f"blocks.{index}"
        names.extend([
            f"{prefix}.attention.in_proj_weight", f"{prefix}.attention.in_proj_bias",
            f"{prefix}.attention.out_proj.weight", f"{prefix}.attention.out_proj.bias",
            f"{prefix}.attention_norm.weight", f"{prefix}.attention_norm.bias",
            f"{prefix}.mlp.0.weight", f"{prefix}.mlp.0.bias",
            f"{prefix}.mlp.2.weight", f"{prefix}.mlp.2.bias",
            f"{prefix}.mlp_norm.weight", f"{prefix}.mlp_norm.bias",
        ])
    names.extend([
        "worker_act_head.weight", "worker_act_head.bias",
        "worker_kind_head.weight", "worker_kind_head.bias",
        "target_worker_head.weight", "target_worker_head.bias",
        "target_tile_head.weight", "target_tile_head.bias",
        "market_item_head.weight", "market_item_head.bias",
        "market_quantity_head.weight", "market_quantity_head.bias",
        "value_head.weight", "value_head.bias",
    ])
    return tuple(names)


def export_checkpoint(checkpoint_path: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    """Export a torch checkpoint, failing clearly when torch is unavailable."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to export a checkpoint; install the training extra") from exc
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("metadata"), dict):
        raise ValueError("checkpoint metadata is required")
    metadata = checkpoint["metadata"]
    for key, expected in (
        ("model_version", MODEL_VERSION),
        ("feature_schema_version", FEATURE_SCHEMA_VERSION),
        ("engine_version", ENGINE_VERSION),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint {key} mismatch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state_dict is required")
    expected = set(expected_tensor_names())
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f"checkpoint tensors mismatch (missing={missing}, extra={extra})")
    vocab = metadata.get("action_vocab")
    expected_vocab = {key: list(value) for key, value in ACTION_VOCAB.items()}
    if vocab != expected_vocab:
        raise ValueError("checkpoint action_vocab mismatch")
    artifact: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "hidden_width": HIDDEN_WIDTH,
        "quantization": QUANTIZATION,
        "action_vocab": expected_vocab,
        "weights": {name: _tensor_to_artifact(name, state[name]) for name in expected_tensor_names()},
    }
    artifact["checksum"] = artifact_checksum(artifact)
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_bytes(artifact) + b"\n")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    try:
        export_checkpoint(args.checkpoint, args.artifact)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
