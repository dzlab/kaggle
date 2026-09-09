"""Optional dependency-free learned-policy adapter and intent compiler.

Models may choose a worker's task target, but they never choose an engine
movement path.  The compiler below is deliberately conservative: malformed
model output is ignored and every emitted command still goes through the
deterministic policy's legality checks.
"""

from __future__ import annotations

import json
import multiprocessing
import hashlib
import hmac
import math
import struct
import sys
import ctypes
import ctypes.util
import threading
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from operator import mul
from pathlib import Path
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by the dependency-free smoke test
    _np = None

from .constants import ANIMALS, CROPS, PRODUCTS
from .features import GLOBAL_TOKEN_SIZE, MARKET_TOKEN_SIZE, TILE_TOKEN_SIZE, WORKER_TOKEN_SIZE
from .memory import PolicyMemory
from .routing import is_locked_tile, normalize_position, route_to
from .types import Position, Task, WorkerAssignment


@dataclass(frozen=True)
class WorkerProposal:
    worker_index: int
    kind: str
    target: Position | None
    item: str | None
    score: float


@dataclass(frozen=True)
class PolicyProposal:
    workers: tuple[WorkerProposal, ...]
    market_orders: tuple[tuple[str, str | None, int], ...]
    confidence: float
    model_version: str


_EMPTY_PROPOSAL = PolicyProposal((), (), 0.0, "none")
_VALID_KINDS = frozenset({
    "PICKUP", "PLACE", "SHED", "SELL", "SELL_ALL", "DROP", "PLANT",
    "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER",
    "WEED", "DIG", "ANIMAL", "STRUCTURE", "BUILD_COOP", "BUILD_PASTURE",
})
_ITEM_KINDS = frozenset({"PICKUP", "PLACE", "PLANT", "FERTILIZE", "FEED", "ANIMAL", "SELL"})

_ARTIFACT_FORMAT_VERSION = 1
_ARTIFACT_MODEL_VERSION = "learned_v1"
_ARTIFACT_FEATURE_SCHEMA_VERSION = 1
_ARTIFACT_ENGINE_VERSION = "1.32.7"
_ARTIFACT_HIDDEN_WIDTH = 128
_ARTIFACT_MODEL_DEPTH = 4
_ARTIFACT_QUANTIZATION = "int8-per-row"
_BLAS_WEIGHT_CACHE: dict[int, tuple[list[list[float]], array, Any]] = {}
_BLAS_WORKSPACE = threading.local()
_ARTIFACT_WORKER_KINDS = (
    "PASS", "MOVE", "WATER", "HARVEST", "PLANT", "FERTILIZE", "FEED",
    "CARE", "PICKUP", "PLACE", "DROP", "SELL", "DIG", "WEED",
)
_ARTIFACT_MARKET_QUANTITIES = (0, 1, 2, 4, 8, 16, 32, 64)


def _validate_artifact_model_shape(hidden_width: Any, model_depth: Any) -> tuple[int, int]:
    if type(hidden_width) is not int or hidden_width < 1 or hidden_width % 4:
        raise ValueError("unsupported learned artifact hidden_width")
    if type(model_depth) is not int or model_depth < 1:
        raise ValueError("unsupported learned artifact model_depth")
    return hidden_width, model_depth


def _artifact_tensor_names(model_depth: int = _ARTIFACT_MODEL_DEPTH) -> tuple[str, ...]:
    names = [
        "tile_projection.weight", "tile_projection.bias",
        "worker_projection.weight", "worker_projection.bias",
        "market_projection.weight", "market_projection.bias",
        "global_projection.weight", "global_projection.bias",
        "type_embedding.weight",
    ]
    for index in range(model_depth):
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


def artifact_tensor_shapes(
    hidden_width: int = _ARTIFACT_HIDDEN_WIDTH,
    model_depth: int = _ARTIFACT_MODEL_DEPTH,
) -> dict[str, tuple[int, ...]]:
    """Return the exact state-dict shapes for a CompactPolicyNet topology."""
    hidden_width, model_depth = _validate_artifact_model_shape(hidden_width, model_depth)
    mlp_width = 2 * hidden_width
    shapes: dict[str, tuple[int, ...]] = {
        "tile_projection.weight": (hidden_width, TILE_TOKEN_SIZE),
        "tile_projection.bias": (hidden_width,),
        "worker_projection.weight": (hidden_width, WORKER_TOKEN_SIZE),
        "worker_projection.bias": (hidden_width,),
        "market_projection.weight": (hidden_width, MARKET_TOKEN_SIZE),
        "market_projection.bias": (hidden_width,),
        "global_projection.weight": (hidden_width, GLOBAL_TOKEN_SIZE),
        "global_projection.bias": (hidden_width,),
        "type_embedding.weight": (4, hidden_width),
    }
    for index in range(model_depth):
        prefix = f"blocks.{index}"
        shapes.update({
            f"{prefix}.attention.in_proj_weight": (3 * hidden_width, hidden_width),
            f"{prefix}.attention.in_proj_bias": (3 * hidden_width,),
            f"{prefix}.attention.out_proj.weight": (hidden_width, hidden_width),
            f"{prefix}.attention.out_proj.bias": (hidden_width,),
            f"{prefix}.attention_norm.weight": (hidden_width,),
            f"{prefix}.attention_norm.bias": (hidden_width,),
            f"{prefix}.mlp.0.weight": (mlp_width, hidden_width),
            f"{prefix}.mlp.0.bias": (mlp_width,),
            f"{prefix}.mlp.2.weight": (hidden_width, mlp_width),
            f"{prefix}.mlp.2.bias": (hidden_width,),
            f"{prefix}.mlp_norm.weight": (hidden_width,),
            f"{prefix}.mlp_norm.bias": (hidden_width,),
        })
    shapes.update({
        "worker_act_head.weight": (2, hidden_width),
        "worker_act_head.bias": (2,),
        "worker_kind_head.weight": (len(_ARTIFACT_WORKER_KINDS), hidden_width),
        "worker_kind_head.bias": (len(_ARTIFACT_WORKER_KINDS),),
        "target_worker_head.weight": (hidden_width, hidden_width),
        "target_worker_head.bias": (hidden_width,),
        "target_tile_head.weight": (hidden_width, hidden_width),
        "target_tile_head.bias": (hidden_width,),
        "market_item_head.weight": (len(PRODUCTS), hidden_width),
        "market_item_head.bias": (len(PRODUCTS),),
        "market_quantity_head.weight": (len(_ARTIFACT_MARKET_QUANTITIES), hidden_width),
        "market_quantity_head.bias": (len(_ARTIFACT_MARKET_QUANTITIES),),
        "value_head.weight": (1, hidden_width),
        "value_head.bias": (1,),
    })
    return shapes


def _artifact_canonical_bytes(value: Mapping[str, Any]) -> bytes:
    payload = {key: item for key, item in value.items() if key != "checksum"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_artifact_vocab(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("learned artifact action_vocab must be an object")
    expected = {
        "worker_kinds": list(_ARTIFACT_WORKER_KINDS),
        "market_items": sorted(PRODUCTS),
        "market_quantities": list(_ARTIFACT_MARKET_QUANTITIES),
    }
    if set(value) != set(expected) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("learned artifact action_vocab mismatch")
    for key, items in expected.items():
        if not isinstance(value.get(key), list) or not items or len(set(value[key])) != len(items):
            raise ValueError(f"learned artifact action vocabulary {key} is malformed")
        if not all(type(item) is type(expected_item) for item, expected_item in zip(value[key], items)):
            raise ValueError(f"learned artifact action vocabulary {key} is malformed")


def _finite_fp32(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    try:
        return struct.unpack("<f", struct.pack("<f", result))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{label} is outside fp32 range") from exc


def _read_artifact_tensor(name: str, value: Any, expected_shape: tuple[int, ...]) -> list[list[float]] | list[float]:
    if not isinstance(value, Mapping) or not isinstance(value.get("shape"), list):
        raise ValueError(f"learned artifact tensor {name!r} is malformed")
    shape = value["shape"]
    if tuple(shape) != expected_shape:
        raise ValueError(f"learned artifact tensor {name!r} shape mismatch: expected {expected_shape}, got {shape}")
    if len(shape) not in (1, 2) or any(type(size) is not int or size < 1 for size in shape):
        raise ValueError(f"learned artifact tensor {name!r} has an invalid shape")
    values = value.get("values")
    if len(shape) == 2:
        rows, width = shape
        scales = value.get("scales")
        if not isinstance(scales, list) or len(scales) != rows or not isinstance(values, list) or len(values) != rows:
            raise ValueError(f"learned artifact tensor {name!r} is missing row-wise data")
        result: list[list[float]] = []
        for row_index, (scale, row) in enumerate(zip(scales, values)):
            scale = _finite_fp32(scale, f"{name} scale")
            if scale <= 0.0 or not isinstance(row, list) or len(row) != width:
                raise ValueError(f"learned artifact tensor {name!r} has invalid row data")
            decoded: list[float] = []
            for item in row:
                if type(item) is not int or item < -128 or item > 127:
                    raise ValueError(f"{name} contains an invalid int8 value")
                decoded.append(float(item) * scale)
            result.append(decoded)
        return result
    if not isinstance(values, list) or len(values) != shape[0] or "scales" in value:
        raise ValueError(f"learned artifact tensor {name!r} has invalid vector data")
    return [_finite_fp32(item, f"{name} value") for item in values]


def _validate_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("learned artifact must be a JSON object")
    hidden_width = value.get("hidden_width")
    model_depth = value.get("model_depth", _ARTIFACT_MODEL_DEPTH)
    hidden_width, model_depth = _validate_artifact_model_shape(hidden_width, model_depth)
    expected_headers = {
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "model_version": _ARTIFACT_MODEL_VERSION,
        "feature_schema_version": _ARTIFACT_FEATURE_SCHEMA_VERSION,
        "engine_version": _ARTIFACT_ENGINE_VERSION,
        "hidden_width": hidden_width,
        "quantization": _ARTIFACT_QUANTIZATION,
    }
    for key, expected in expected_headers.items():
        actual = value.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"unsupported learned artifact {key}")
    if "model_depth" in value and value["model_depth"] != model_depth:
        raise ValueError("unsupported learned artifact model_depth")
    expected_headers["model_depth"] = model_depth
    _validate_artifact_vocab(value.get("action_vocab"))
    checksum = value.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("learned artifact checksum is missing or malformed")
    actual = hashlib.sha256(_artifact_canonical_bytes(value)).hexdigest()
    if not hmac.compare_digest(actual, checksum):
        raise ValueError("learned artifact checksum mismatch")
    weights = value.get("weights")
    expected_names = set(_artifact_tensor_names(model_depth))
    if not isinstance(weights, Mapping) or set(weights) != expected_names:
        missing = sorted(expected_names - set(weights or ())) if isinstance(weights, Mapping) else sorted(expected_names)
        raise ValueError(f"learned artifact tensors mismatch; missing={missing}")
    shapes = artifact_tensor_shapes(hidden_width, model_depth)
    decoded = {name: _read_artifact_tensor(name, weights[name], shapes[name]) for name in _artifact_tensor_names(model_depth)}
    return {"headers": dict(expected_headers), "action_vocab": value["action_vocab"], "weights": decoded}


def load_exported_policy(path: str | Path) -> "DependencyFreePolicy":
    """Load and validate a JSON policy artifact using only the Python stdlib."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return DependencyFreePolicy(_validate_artifact(value))


def _linear(rows: list[list[float]], weight: list[list[float]], bias: list[float]) -> list[list[float]]:
    if _CBLAS_SGEMM is not None and len(rows) >= 8 and len(rows[0]) >= 16 and len(weight) >= 8:
        return _blas_linear(rows, weight, bias)
    return [[sum(map(mul, row, output), 0.0) + bias[index]
             for index, output in enumerate(weight)] for row in rows]


def _vector_linear(row: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
    return [sum(map(mul, row, output), 0.0) + bias[index]
            for index, output in enumerate(weight)]


def _load_cblas_sgemm() -> Any:
    """Load platform BLAS through stdlib ctypes when it is available."""
    library = ctypes.util.find_library("blas")
    if library is None and sys.platform == "darwin":
        library = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    if library is None:
        return None
    try:
        function = ctypes.CDLL(library).cblas_sgemm
        function.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        function.restype = None
        return function
    except (AttributeError, OSError):
        return None


_CBLAS_SGEMM = _load_cblas_sgemm()


def _blas_linear(rows: list[list[float]], weight: list[list[float]], bias: list[float]) -> list[list[float]]:
    """Compute rows @ weight.T using float32 CBLAS buffers."""
    row_count, input_width, output_width = len(rows), len(rows[0]), len(weight)
    left = array("f", (value for row in rows for value in row))
    weight_key = id(weight)
    cached = _BLAS_WEIGHT_CACHE.get(weight_key)
    if cached is None or cached[0] is not weight:
        right = array("f", (value for row in weight for value in row))
        cached = (weight, right, (ctypes.c_float * len(right)).from_buffer(right))
        _BLAS_WEIGHT_CACHE[weight_key] = cached
    right, right_pointer = cached[1], cached[2]
    workspace = getattr(_BLAS_WORKSPACE, "buffers", {})
    _BLAS_WORKSPACE.buffers = workspace
    key = (row_count, input_width, output_width)
    buffers = workspace.get(key)
    if buffers is None:
        left = array("f", [0.0]) * (row_count * input_width)
        result = array("f", [0.0]) * (row_count * output_width)
        buffers = (
            left, result,
            (ctypes.c_float * len(left)).from_buffer(left),
            (ctypes.c_float * len(result)).from_buffer(result),
        )
        workspace[key] = buffers
    left, result, left_pointer, result_pointer = buffers
    offset = 0
    for row in rows:
        left[offset:offset + input_width] = array("f", row)
        offset += input_width
    offset = 0
    for _row in range(row_count):
        result[offset:offset + output_width] = array("f", bias)
        offset += output_width
    _CBLAS_SGEMM(101, 111, 112, row_count, output_width, input_width, 1.0,
                 left_pointer, input_width, right_pointer, input_width, 1.0,
                 result_pointer, output_width)
    flat_result = result.tolist()
    return [flat_result[row * output_width:(row + 1) * output_width] for row in range(row_count)]


def _matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    """Multiply two row-major matrices, using the same stdlib CBLAS bridge."""
    row_count, inner, output_width = len(left), len(right), len(right[0])
    if _CBLAS_SGEMM is not None and row_count >= 8 and inner >= 16 and output_width >= 8:
        workspace = getattr(_BLAS_WORKSPACE, "buffers", {})
        _BLAS_WORKSPACE.buffers = workspace
        key = ("mm", row_count, inner, output_width)
        buffers = workspace.get(key)
        if buffers is None:
            left_buffer = array("f", [0.0]) * (row_count * len(left[0]))
            right_buffer = array("f", [0.0]) * (inner * output_width)
            result = array("f", [0.0]) * (row_count * output_width)
            buffers = (
                left_buffer, right_buffer, result,
                (ctypes.c_float * len(left_buffer)).from_buffer(left_buffer),
                (ctypes.c_float * len(right_buffer)).from_buffer(right_buffer),
                (ctypes.c_float * len(result)).from_buffer(result),
            )
            workspace[key] = buffers
        left_buffer, right_buffer, result, left_pointer, right_pointer, result_pointer = buffers
        left_offset = 0
        for row in left:
            left_buffer[left_offset:left_offset + len(row)] = array("f", row)
            left_offset += len(row)
        right_offset = 0
        for row in right:
            right_buffer[right_offset:right_offset + len(row)] = array("f", row)
            right_offset += len(row)
        _CBLAS_SGEMM(101, 111, 111, row_count, output_width, inner, 1.0,
                     left_pointer, inner, right_pointer, output_width, 0.0,
                     result_pointer, output_width)
        return [[float(result[row * output_width + column]) for column in range(output_width)]
                for row in range(row_count)]
    return [[sum(map(mul, left_row, (right[index][column] for index in range(inner))), 0.0)
             for column in range(output_width)] for left_row in left]


def _layer_norm(rows: list[list[float]], weight: list[float], bias: list[float]) -> list[list[float]]:
    result = []
    for row in rows:
        mean = sum(row) / len(row)
        variance = sum((item - mean) ** 2 for item in row) / len(row)
        denominator = math.sqrt(variance + 1e-5)
        result.append([(item - mean) / denominator * weight[index] + bias[index]
                       for index, item in enumerate(row)])
    return result


def _gelu(value: float) -> float:
    # torch.nn.GELU() defaults to the exact erf formulation (not the tanh
    # approximation), so the dependency-free path must use the same function.
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def _attention(tokens: list[list[float]], weights: Mapping[str, Any], prefix: str) -> list[list[float]]:
    hidden = len(tokens[0])
    qkv = _linear(tokens, weights[f"{prefix}.attention.in_proj_weight"], weights[f"{prefix}.attention.in_proj_bias"])
    query, key, value = qkv[:], qkv[:], qkv[:]
    for index in range(len(tokens)):
        query[index] = qkv[index][:hidden]
        key[index] = qkv[index][hidden:2 * hidden]
        value[index] = qkv[index][2 * hidden:]
    heads = 4
    head_width = hidden // heads
    attended = [[0.0] * hidden for _ in tokens]
    for head in range(heads):
        start = head * head_width
        query_head = [[values[start + offset] for offset in range(head_width)] for values in query]
        key_head = [[values[start + offset] for offset in range(head_width)] for values in key]
        score_rows = _linear(query_head, key_head, [0.0] * len(key_head))
        probability_rows: list[list[float]] = []
        for row in range(len(tokens)):
            scores = [score_rows[row][index] / math.sqrt(head_width) for index in range(len(tokens))]
            maximum = max(scores)
            exponentials = [math.exp(score - maximum) for score in scores]
            total = sum(exponentials)
            probability_rows.append([factor / total for factor in exponentials])
        value_head = [[values[start + offset] for offset in range(head_width)] for values in value]
        weighted_values = _matrix_multiply(probability_rows, value_head)
        for row, weighted in enumerate(weighted_values):
            attended[row][start:start + head_width] = weighted
    return _linear(attended, weights[f"{prefix}.attention.out_proj.weight"], weights[f"{prefix}.attention.out_proj.bias"])


class DependencyFreePolicy:
    """Pure-Python execution of the exported compact policy network."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.model_version = str(artifact["headers"]["model_version"])
        self._hidden_width = int(artifact["headers"]["hidden_width"])
        self._model_depth = int(artifact["headers"]["model_depth"])
        self._weights = artifact["weights"]
        self._numpy_weights = (
            {name: _np.asarray(value, dtype=_np.float32) for name, value in self._weights.items()}
            if _np is not None else None
        )

    def _predict_numpy(self, features: Any) -> dict[str, Any]:
        """Vectorized inference for Kaggle runtimes that provide NumPy."""
        weights = self._numpy_weights
        assert weights is not None

        def linear(rows: Any, weight_name: str, bias_name: str) -> Any:
            return _np.asarray(rows, dtype=_np.float32) @ weights[weight_name].T + weights[bias_name]

        tile_count = len(features.tile_tokens)
        worker_count = len(features.worker_tokens)
        market_count = len(features.market_tokens)
        tokens = _np.concatenate((
            linear(features.tile_tokens, "tile_projection.weight", "tile_projection.bias"),
            linear(features.worker_tokens, "worker_projection.weight", "worker_projection.bias"),
            linear(features.market_tokens, "market_projection.weight", "market_projection.bias"),
            linear([features.global_tokens], "global_projection.weight", "global_projection.bias"),
        ), axis=0)
        embedding = weights["type_embedding.weight"]
        offsets = (0, tile_count, tile_count + worker_count,
                   tile_count + worker_count + market_count, len(tokens))
        for kind, (start, end) in enumerate(zip(offsets, offsets[1:])):
            tokens[start:end] += embedding[kind]

        for block in range(self._model_depth):
            prefix = f"blocks.{block}"
            qkv = linear(tokens, f"{prefix}.attention.in_proj_weight", f"{prefix}.attention.in_proj_bias")
            heads = 4
            head_width = qkv.shape[1] // 3 // heads
            q, k, value = _np.split(qkv, 3, axis=1)
            q = q.reshape(len(tokens), heads, head_width).transpose(1, 0, 2)
            k = k.reshape(len(tokens), heads, head_width).transpose(1, 0, 2)
            value = value.reshape(len(tokens), heads, head_width).transpose(1, 0, 2)
            scores = _np.matmul(q, k.transpose(0, 2, 1)) / _np.sqrt(_np.float32(head_width))
            scores -= scores.max(axis=-1, keepdims=True)
            probabilities = _np.exp(scores)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            attended = _np.matmul(probabilities, value).transpose(1, 0, 2).reshape(len(tokens), -1)
            attended = attended @ weights[f"{prefix}.attention.out_proj.weight"].T
            attended += weights[f"{prefix}.attention.out_proj.bias"]
            residual = tokens + attended
            norm_weight = weights[f"{prefix}.attention_norm.weight"]
            norm_bias = weights[f"{prefix}.attention_norm.bias"]
            mean = residual.mean(axis=1, keepdims=True)
            variance = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
            tokens = (residual - mean) / _np.sqrt(variance + _np.float32(1e-5)) * norm_weight + norm_bias

            hidden = linear(tokens, f"{prefix}.mlp.0.weight", f"{prefix}.mlp.0.bias")
            hidden = _np.float32(0.5) * hidden * (
                _np.float32(1.0) + _np.tanh(
                    _np.float32(0.7978845608) * (hidden + _np.float32(0.044715) * hidden ** 3)
                )
            )
            hidden = hidden @ weights[f"{prefix}.mlp.2.weight"].T
            hidden += weights[f"{prefix}.mlp.2.bias"]
            residual = tokens + hidden
            norm_weight = weights[f"{prefix}.mlp_norm.weight"]
            norm_bias = weights[f"{prefix}.mlp_norm.bias"]
            mean = residual.mean(axis=1, keepdims=True)
            variance = ((residual - mean) ** 2).mean(axis=1, keepdims=True)
            tokens = (residual - mean) / _np.sqrt(variance + _np.float32(1e-5)) * norm_weight + norm_bias

        tile_rows = tokens[:tile_count]
        worker_rows = tokens[tile_count:tile_count + worker_count]
        market_rows = tokens[tile_count + worker_count:tile_count + worker_count + market_count]
        global_row = tokens[-1]
        worker_target_query = worker_rows @ weights["target_worker_head.weight"].T + weights["target_worker_head.bias"]
        tile_target_key = tile_rows @ weights["target_tile_head.weight"].T + weights["target_tile_head.bias"]
        target_logits = worker_target_query @ tile_target_key.T / _np.sqrt(_np.float32(self._hidden_width))
        pooled_market = market_rows.mean(axis=0) if market_count else _np.zeros(self._hidden_width, dtype=_np.float32)
        return {
            "worker_act_logits": (worker_rows @ weights["worker_act_head.weight"].T + weights["worker_act_head.bias"]).tolist(),
            "worker_target_logits": target_logits.tolist(),
            "worker_kind_logits": (worker_rows @ weights["worker_kind_head.weight"].T + weights["worker_kind_head.bias"]).tolist(),
            "market_item_logits": (pooled_market @ weights["market_item_head.weight"].T + weights["market_item_head.bias"]).tolist(),
            "market_quantity_logits": (pooled_market @ weights["market_quantity_head.weight"].T + weights["market_quantity_head.bias"]).tolist(),
            "value": float((global_row @ weights["value_head.weight"].T + weights["value_head.bias"])[0]),
        }

    def predict(self, features: Any) -> dict[str, Any]:
        if self._numpy_weights is not None:
            return self._predict_numpy(features)
        tile = [list(row) for row in features.tile_tokens]
        worker = [list(row) for row in features.worker_tokens]
        market = [list(row) for row in features.market_tokens]
        global_token = [list(features.global_tokens)]
        weights = self._weights
        tokens = (
            _linear(tile, weights["tile_projection.weight"], weights["tile_projection.bias"])
            + _linear(worker, weights["worker_projection.weight"], weights["worker_projection.bias"])
            + _linear(market, weights["market_projection.weight"], weights["market_projection.bias"])
            + _linear(global_token, weights["global_projection.weight"], weights["global_projection.bias"])
        )
        type_embedding = weights["type_embedding.weight"]
        offsets = [0, len(tile), len(tile) + len(worker), len(tile) + len(worker) + len(market), len(tokens)]
        for index, kind in enumerate((0, 1, 2, 3)):
            for position in range(offsets[index], offsets[index + 1]):
                tokens[position] = [value + type_embedding[kind][column] for column, value in enumerate(tokens[position])]
        for block in range(self._model_depth):
            prefix = f"blocks.{block}"
            attended = _attention(tokens, weights, prefix)
            tokens = _layer_norm(
                [[left + right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(tokens, attended)],
                weights[f"{prefix}.attention_norm.weight"], weights[f"{prefix}.attention_norm.bias"],
            )
            hidden = _linear(tokens, weights[f"{prefix}.mlp.0.weight"], weights[f"{prefix}.mlp.0.bias"])
            hidden = [[_gelu(value) for value in row] for row in hidden]
            hidden = _linear(hidden, weights[f"{prefix}.mlp.2.weight"], weights[f"{prefix}.mlp.2.bias"])
            tokens = _layer_norm(
                [[left + right for left, right in zip(left_row, right_row)] for left_row, right_row in zip(tokens, hidden)],
                weights[f"{prefix}.mlp_norm.weight"], weights[f"{prefix}.mlp_norm.bias"],
            )
        tile_count, worker_count, market_count = len(tile), len(worker), len(market)
        tile_rows = tokens[:tile_count]
        worker_rows = tokens[tile_count:tile_count + worker_count]
        market_rows = tokens[tile_count + worker_count:tile_count + worker_count + market_count]
        global_row = tokens[-1]
        worker_target_query = _linear(worker_rows, weights["target_worker_head.weight"], weights["target_worker_head.bias"])
        tile_target_key = _linear(tile_rows, weights["target_tile_head.weight"], weights["target_tile_head.bias"])
        tile_target_transposed = [
            [tile_target_key[row][column] for row in range(len(tile_target_key))]
            for column in range(self._hidden_width)
        ]
        target_logits = [
            [value / math.sqrt(self._hidden_width) for value in row]
            for row in _matrix_multiply(worker_target_query, tile_target_transposed)
        ]
        pooled_market = [sum(row[index] for row in market_rows) / len(market_rows) for index in range(self._hidden_width)]
        return {
            "worker_act_logits": _linear(worker_rows, weights["worker_act_head.weight"], weights["worker_act_head.bias"]),
            "worker_target_logits": target_logits,
            "worker_kind_logits": _linear(worker_rows, weights["worker_kind_head.weight"], weights["worker_kind_head.bias"]),
            "market_item_logits": _vector_linear(pooled_market, weights["market_item_head.weight"], weights["market_item_head.bias"]),
            "market_quantity_logits": _vector_linear(pooled_market, weights["market_quantity_head.weight"], weights["market_quantity_head.bias"]),
            "value": _vector_linear(global_row, weights["value_head.weight"], weights["value_head.bias"])[0],
        }

    def propose(self, state: Any, features: Any) -> PolicyProposal:
        outputs = self.predict(features)
        workers: list[WorkerProposal] = []
        positions = tuple(features.tile_positions)
        for index, (act_logits, target_logits, kind_logits) in enumerate(zip(
            outputs["worker_act_logits"], outputs["worker_target_logits"], outputs["worker_kind_logits"],
        )):
            if max(range(len(act_logits)), key=act_logits.__getitem__) == 0:
                continue
            target = positions[max(range(len(target_logits)), key=target_logits.__getitem__)] if positions else None
            kind = _ARTIFACT_WORKER_KINDS[max(range(len(kind_logits)), key=kind_logits.__getitem__)]
            workers.append(WorkerProposal(index, kind, target, None, max(kind_logits)))
        # The artifact has no buy/sell direction head.  Emit only the narrow
        # product-buy intent that the normal market compiler can validate;
        # unsupported products and the zero quantity bucket become no-op.
        market_orders: tuple[tuple[str, str | None, int], ...] = ()
        item_logits = outputs.get("market_item_logits", ())
        quantity_logits = outputs.get("market_quantity_logits", ())
        if item_logits and quantity_logits:
            item_index = max(range(len(item_logits)), key=item_logits.__getitem__)
            quantity_index = max(range(len(quantity_logits)), key=quantity_logits.__getitem__)
            items = tuple(sorted(PRODUCTS))
            quantity = _ARTIFACT_MARKET_QUANTITIES[quantity_index]
            if 0 <= item_index < len(items) and items[item_index] in {"WHEAT", "FERTILIZER"} and quantity > 0:
                market_orders = (("BUY_PRODUCT", items[item_index], quantity),)
        return PolicyProposal(tuple(workers), market_orders, 1.0, self.model_version)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if isfinite(result) else default


def _empty() -> PolicyProposal:
    return _EMPTY_PROPOSAL


def _process_entry(connection: Any, function: Any) -> None:
    """Execute model code in an isolated child and return only serializable data."""
    try:
        connection.send((True, function()))
    except BaseException as exc:  # model code must not escape the entrypoint
        try:
            connection.send((False, type(exc).__name__))
        except BaseException:
            pass
    finally:
        connection.close()


def _run_with_timeout(function: Any, timeout: float) -> tuple[bool, Any]:
    """Run model code in a killable process so timed-out code cannot continue."""
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        context = multiprocessing.get_context()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_process_entry, args=(sender, function))
    process.daemon = True
    try:
        process.start()
    except BaseException as exc:
        receiver.close()
        sender.close()
        return False, exc
    sender.close()
    process.join(max(0.0, timeout))
    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        if process.is_alive():
            receiver.close()
            return False, RuntimeError("learned model process did not terminate")
        receiver.close()
        return False, TimeoutError("learned model timed out")
    try:
        return receiver.recv() if receiver.poll() else (False, RuntimeError("learned model returned no result"))
    except (EOFError, OSError, TypeError):
        return False, RuntimeError("learned model returned an unreadable result")
    finally:
        receiver.close()


def _coerce_worker(raw: Any) -> WorkerProposal | None:
    if isinstance(raw, WorkerProposal):
        return raw
    index = _get(raw, "worker_index", _get(raw, "index"))
    kind = _get(raw, "kind")
    if index is None or kind is None:
        return None
    try:
        worker_index = int(index)
    except (TypeError, ValueError, OverflowError):
        return None
    item = _get(raw, "item")
    item = str(item).upper() if item is not None else None
    return WorkerProposal(
        worker_index=worker_index,
        kind=str(kind).upper(),
        target=normalize_position(_get(raw, "target")),
        item=item,
        score=_number(_get(raw, "score")),
    )


def _coerce_market(raw: Any) -> tuple[str, str | None, int] | None:
    if isinstance(raw, (list, tuple)):
        if len(raw) < 1:
            return None
        kind, item, quantity = raw[0], raw[1] if len(raw) > 1 else None, raw[2] if len(raw) > 2 else 1
    else:
        kind = _get(raw, "kind")
        item = _get(raw, "item")
        quantity = _get(raw, "quantity", 1)
    if kind is None:
        return None
    try:
        quantity = int(quantity)
    except (TypeError, ValueError, OverflowError):
        return None
    if quantity <= 0:
        return None
    return str(kind).upper(), str(item).upper() if item is not None else None, quantity


def _coerce_proposal(raw: Any) -> PolicyProposal | None:
    if isinstance(raw, PolicyProposal):
        return raw
    if not isinstance(raw, Mapping):
        return None
    workers_raw = _get(raw, "workers", ())
    market_raw = _get(raw, "market_orders", _get(raw, "market", ()))
    if not isinstance(workers_raw, Sequence) or isinstance(workers_raw, (str, bytes)):
        return None
    if not isinstance(market_raw, Sequence) or isinstance(market_raw, (str, bytes)):
        return None
    workers = tuple(worker for raw_worker in workers_raw if (worker := _coerce_worker(raw_worker)) is not None)
    market = tuple(order for raw_order in market_raw if (order := _coerce_market(raw_order)) is not None)
    return PolicyProposal(
        workers=workers,
        market_orders=market,
        confidence=_number(_get(raw, "confidence")),
        model_version=str(_get(raw, "model_version", "unknown")),
    )


class LearnedPolicy:
    """Load and invoke an optional model without importing ML dependencies."""

    def __init__(self, model_path: str | Path | Any = None, *, timeout_seconds: float = 0.25) -> None:
        self.model_path = model_path
        self.timeout_seconds = max(0.01, _number(timeout_seconds, 0.25))
        self.diagnostics: dict[str, Any] = {"status": "disabled" if model_path is None else "unloaded"}
        self._model: Any = None
        self._loaded = False

    def _load(self) -> Any:
        if self.model_path is None:
            return None
        if not isinstance(self.model_path, (str, Path)):
            return self.model_path
        path = Path(self.model_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("learned model must be a JSON object")
        # Preserve the small legacy proposal fixture interface used by the
        # deterministic-policy tests; exported network artifacts are always
        # identified by their format header and go through strict validation.
        if "format_version" in value or "weights" in value or "checksum" in value:
            return DependencyFreePolicy(_validate_artifact(value))
        if any(key in value for key in ("workers", "market_orders", "market")):
            return value
        raise ValueError("learned model is neither an exported artifact nor a proposal")

    def _invoke(self, model: Any, state: Any, features: Any) -> Any:
        if isinstance(model, Mapping) and ("workers" in model or "market_orders" in model or "market" in model):
            return model
        if hasattr(model, "propose"):
            return model.propose(state, features)
        if hasattr(model, "predict"):
            return model.predict(features)
        if callable(model):
            return model(state, features)
        raise TypeError("model has no propose, predict, or callable interface")

    def propose(self, state: Any, features: Any) -> PolicyProposal:
        if self.model_path is None:
            self.diagnostics = {"status": "disabled"}
            return _empty()
        if not self._loaded:
            # Exported dependency-free models are already validated by the
            # candidate factory.  Keep their inference in-process: spawning a
            # child for every turn adds enough startup latency to exceed the
            # Kaggle action deadline and can result in a recorded ``None``
            # action.  Arbitrary user models retain the killable isolation
            # boundary below.
            if isinstance(self.model_path, DependencyFreePolicy):
                loaded = self.model_path
                ok = True
            elif isinstance(self.model_path, (str, Path)):
                try:
                    loaded = self._load()
                    ok = True
                except BaseException as exc:
                    ok, loaded = False, exc
            else:
                ok, loaded = _run_with_timeout(self._load, self.timeout_seconds)
            if not ok:
                status = "slow_model" if isinstance(loaded, TimeoutError) else (
                    "missing_model"
                    if isinstance(loaded, FileNotFoundError) or loaded == "FileNotFoundError"
                    else "load_error"
                )
                self.diagnostics = {"status": status, "error": type(loaded).__name__}
                return _empty()
            self._model = loaded
            self._loaded = True
        if self._model is None:
            self.diagnostics = {"status": "incompatible_model"}
            return _empty()
        if isinstance(self._model, DependencyFreePolicy):
            try:
                output = self._invoke(self._model, state, features)
                ok = True
            except BaseException as exc:
                ok, output = False, exc
        else:
            ok, output = _run_with_timeout(
                lambda: self._invoke(self._model, state, features), self.timeout_seconds,
            )
        if not ok:
            status = "slow_model" if isinstance(output, TimeoutError) else "inference_error"
            self.diagnostics = {"status": status, "error": type(output).__name__}
            return _empty()
        proposal = _coerce_proposal(output)
        if proposal is None:
            self.diagnostics = {"status": "incompatible_model"}
            return _empty()
        self.diagnostics = {"status": "ok", "model_version": proposal.model_version}
        return proposal


def _board_size(state: Any) -> int:
    raw = _get(state, "board_size")
    try:
        if raw is not None and int(raw) > 0:
            return int(raw)
    except (TypeError, ValueError, OverflowError):
        pass
    tiles = _get(state, "tiles", _get(_get(state, "farm", {}), "tiles", ()))
    if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
        return max(1, len(tiles), *(len(row) for row in tiles if isinstance(row, Sequence)))
    if isinstance(tiles, Mapping):
        positions = [normalize_position(key) for key in tiles]
        return max((max(position.x, position.y) + 1 for position in positions if position is not None), default=1)
    return 1


def _workers(state: Any) -> list[tuple[int, Position | None]]:
    source = _get(state, "workers", _get(_get(state, "farm", {}), "workers", ()))
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        return []
    result = []
    for fallback, worker in enumerate(source):
        try:
            index = int(_get(worker, "index", fallback))
        except (TypeError, ValueError, OverflowError):
            continue
        result.append((index, normalize_position(_get(worker, "position", worker))))
    return sorted(result)


def _tile_at(state: Any, position: Position) -> Any:
    tiles = _get(state, "tiles", _get(_get(state, "farm", {}), "tiles", ()))
    if isinstance(tiles, Mapping):
        return next((tile for raw, tile in tiles.items() if normalize_position(raw) == position), None)
    if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) and 0 <= position.y < len(tiles):
        row = tiles[position.y]
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and 0 <= position.x < len(row):
            return row[position.x]
    return None


def _counts(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(key).upper(): max(0, int(_number(quantity))) for key, quantity in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: dict[str, int] = {}
        for item in value:
            name = item if isinstance(item, str) else _get(item, "item", _get(item, "kind"))
            if name:
                key = str(name).upper()
                result[key] = result.get(key, 0) + max(1, int(_number(_get(item, "quantity", 1), 1)))
        return result
    return {}


def _inventory(state: Any, index: int) -> dict[str, int]:
    private = _get(state, "private", {})
    inventories = _get(private, "inventories", _get(state, "inventories", ()))
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)) and 0 <= index < len(inventories):
        return _counts(inventories[index])
    return {}


def _shed(state: Any) -> dict[str, int]:
    private = _get(state, "private", {})
    return _counts(_get(private, "shed", _get(state, "shed", {})))


def _available_item(state: Any, worker_index: int, item: str) -> bool:
    return _inventory(state, worker_index).get(item, 0) > 0 or _shed(state).get(item, 0) > 0


def _proposal_is_usable(state: Any, worker_index: int, kind: str, item: str | None) -> bool:
    item = item.upper() if item else None
    if item is not None and item not in set(PRODUCTS) | set(ANIMALS):
        return False
    if kind == "PLANT":
        seeds = _get(_get(state, "private", {}), "seeds", _get(state, "seeds", {}))
        return item in CROPS and isinstance(seeds, Mapping) and _number(seeds.get(item)) > 0
    required = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}.get(kind)
    if required is not None and not _available_item(state, worker_index, required):
        return False
    if kind in {"PICKUP", "PLACE", "ANIMAL", "SELL"} and item is None:
        return False
    if kind == "PICKUP":
        return _shed(state).get(item or "", 0) > 0
    if kind in {"PLACE", "ANIMAL"}:
        return _available_item(state, worker_index, item or "")
    if kind == "SELL":
        return _shed(state).get(item or "", 0) > 0
    return True


def _strategy_allows(state: Any, kind: str, item: str | None, target: Position | None, strategy: Any,
                     crop_count: int, animal_count: int) -> bool:
    if strategy is None:
        return True
    from .planner import _task_allowed, normalize_planner_state
    if kind == "FERTILIZE" and item is None:
        from .policy import _crop

        item = _crop(_tile_at(state, target)) if target is not None else None
        if item is None:
            return False

    task = Task(kind, target, 0, None, 0.0, item=item)
    if not _task_allowed(task, strategy, normalize_planner_state(state)):
        return False
    try:
        if kind == "PLANT" and crop_count >= max(0, int(_get(strategy, "max_crop_units", 0))):
            return False
        if kind in {"ANIMAL", "PLACE"} and item in ANIMALS and animal_count >= max(0, int(_get(strategy, "max_animal_units", 0))):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _proposal_sort_key(candidate: WorkerProposal) -> tuple[Any, ...]:
    target = normalize_position(candidate.target)
    return (
        -_number(candidate.score), candidate.kind,
        target.y if target is not None else float("inf"),
        target.x if target is not None else float("inf"),
        candidate.item or "",
    )


def _existing_counts(state: Any, strategy: Any = None) -> tuple[int, int]:
    from .policy import _existing_animal_units, _iter_tiles, _crop

    allowed = set(_get(strategy, "crops", ()) or ()) if strategy is not None else set(CROPS)
    allowed = {str(value).upper() for value in allowed}
    crops = sum(1 for _, tile in _iter_tiles(state) if _crop(tile) in allowed)
    return crops, _existing_animal_units(state)


def _carried_assignment(state: Any, memory: PolicyMemory, worker_index: int) -> WorkerAssignment | None:
    # Importing these private helpers here avoids a policy/learned-policy
    # import cycle while sharing the exact existing legality rules.
    from .policy import _assignment_valid, _get as policy_get, _inventory_for_worker, _required_worker_item

    for assignment in memory.assignments:
        if int(policy_get(assignment, "worker_index", -1)) != worker_index:
            continue
        kind = str(policy_get(assignment.task, "kind", "")).upper()
        if kind not in {"FEED", "FERTILIZE", "ANIMAL", "PLACE"}:
            continue
        required = _required_worker_item(assignment.task, state)
        if required and _inventory_for_worker(state, worker_index).get(required, 0) > 0 and _assignment_valid(state, assignment):
            return assignment
    return None


def _fallback_assignments(state: Any, memory: PolicyMemory, strategy: Any = None) -> list[WorkerAssignment]:
    from .planner import assign_tasks, build_daily_plan
    from .policy import _assignment_valid

    valid = [assignment for assignment in memory.assignments if _assignment_valid(state, assignment)]
    if valid:
        return valid
    return assign_tasks(build_daily_plan(state, memory, strategy), _get(state, "workers", ()), state, strategy)


def _route_positions(start: Position | None, target: Position | None, board_size: int) -> list[Position]:
    if start is None or target is None:
        return []
    positions: list[Position] = []
    current = start
    for move in route_to(start, target, board_size):
        current = Position(
            current.x + (move == "EAST") - (move == "WEST"),
            current.y + (move == "SOUTH") - (move == "NORTH"),
        )
        positions.append(current)
    return positions


def compile_proposal(state: Any, proposal: PolicyProposal, memory: PolicyMemory,
                     strategy: Any = None) -> dict[str, Any]:
    """Compile target/task intents into the normal action schema."""
    from .policy import PASS, _get as policy_get, _unit_command, _worker_records, build_market_orders, worker_action

    board_size = _board_size(state)
    worker_records = _worker_records(state)
    known = {record["index"]: record for record in worker_records}
    candidates_by_worker: dict[int, list[WorkerProposal]] = {}
    existing_crops, existing_animals = _existing_counts(state, strategy)
    for candidate in proposal.workers if isinstance(proposal, PolicyProposal) else ():
        if candidate.worker_index not in known or candidate.kind not in _VALID_KINDS:
            continue
        target = normalize_position(candidate.target)
        if target is not None and not (0 <= target.x < board_size and 0 <= target.y < board_size):
            continue
        if target is not None and is_locked_tile(_tile_at(state, target)):
            continue
        if not _proposal_is_usable(state, candidate.worker_index, candidate.kind, candidate.item):
            continue
        if not _strategy_allows(state, candidate.kind, candidate.item, target, strategy,
                                existing_crops, existing_animals):
            continue
        candidates_by_worker.setdefault(candidate.worker_index, []).append(candidate)

    selected: dict[int, WorkerProposal] = {}
    crop_count, animal_count = existing_crops, existing_animals
    for worker_index in sorted(candidates_by_worker):
        candidate = min(candidates_by_worker[worker_index], key=_proposal_sort_key)
        if candidate.kind == "PLANT":
            if strategy is not None and crop_count >= int(_get(strategy, "max_crop_units", 0)):
                continue
            crop_count += 1
        if candidate.kind in {"ANIMAL", "PLACE"} and candidate.item in ANIMALS:
            if strategy is not None and animal_count >= int(_get(strategy, "max_animal_units", 0)):
                continue
            animal_count += 1
        selected[worker_index] = candidate

    assignments = _fallback_assignments(state, memory, strategy)
    by_worker = {int(policy_get(assignment, "worker_index", -1)): assignment for assignment in assignments}
    for worker_index in sorted(known):
        carried = _carried_assignment(state, memory, worker_index)
        if carried is not None:
            by_worker[worker_index] = carried
            continue
        candidate = selected.get(worker_index)
        if candidate is None:
            continue
        target = normalize_position(candidate.target) or known[worker_index]["position"]
        if target is None:
            continue
        task_item = candidate.item
        if candidate.kind == "FERTILIZE" and task_item is None:
            from .policy import _crop

            task_item = _crop(_tile_at(state, target))
        task = Task(candidate.kind, target, int(max(0, _number(candidate.score))), None, max(0.0, _number(candidate.score)), item=task_item)
        by_worker[worker_index] = WorkerAssignment(
            worker_index, task, _route_positions(known[worker_index]["position"], target, board_size),
        )
    memory.assignments = [by_worker[index] for index in sorted(by_worker) if index in known]
    market = build_market_orders(
        state, proposal.market_orders if isinstance(proposal, PolicyProposal) else (), strategy,
    )
    commands = {
        index: worker_action(index, state, by_worker.get(index))
        for index in sorted(known)
    }
    farmer = commands.get(0, [PASS])
    visible_hands = policy_get(policy_get(state, "farm", {}), "hands", ())
    if not isinstance(visible_hands, Sequence) or isinstance(visible_hands, (str, bytes)):
        visible_hands = [record for record in worker_records if record["index"] != 0]
    return {
        "farmer": farmer,
        "hands": [commands.get(index + 1, [PASS]) for index in range(len(visible_hands))],
        "market": market,
    }
