"""Dependency-free CompactPolicyNet topology constants and parameter count."""

from __future__ import annotations

from typing import Any


DEFAULT_MODEL_WIDTH = 128
DEFAULT_MODEL_DEPTH = 4
ATTENTION_HEADS = 4
FEATURE_INPUT_SIZES = (23, 28, 14, 13)
TYPE_EMBEDDING_COUNT = 4
WORKER_ACTION_COUNT = 2
WORKER_KIND_COUNT = 14
MARKET_ITEM_COUNT = 9
MARKET_QUANTITY_COUNT = 8
MARKET_ACTIVE_COUNT = 2
VALUE_COUNT = 1


def validate_topology_shape(
    hidden_width: Any, depth: Any, *, source: str = "model",
) -> tuple[int, int]:
    """Validate a CompactPolicyNet width/depth without importing PyTorch."""
    if type(hidden_width) is not int or hidden_width < 1:
        raise ValueError(f"{source} hidden_width must be a positive integer")
    if hidden_width % ATTENTION_HEADS:
        raise ValueError(
            f"{source} hidden_width must be divisible by {ATTENTION_HEADS}"
        )
    if type(depth) is not int or depth < 1:
        raise ValueError(f"{source} depth must be a positive integer")
    return hidden_width, depth


def compact_policy_parameter_count(
    hidden_width: int = DEFAULT_MODEL_WIDTH,
    depth: int = DEFAULT_MODEL_DEPTH,
) -> int:
    """Return the exact CompactPolicyNet parameter count without construction."""
    hidden_width, depth = validate_topology_shape(
        hidden_width, depth, source="parameter count",
    )

    def linear_parameters(input_size: int, output_size: int) -> int:
        return input_size * output_size + output_size

    count = sum(
        linear_parameters(input_size, hidden_width)
        for input_size in FEATURE_INPUT_SIZES
    )
    count += TYPE_EMBEDDING_COUNT * hidden_width
    for _ in range(depth):
        count += 3 * hidden_width * hidden_width + 3 * hidden_width
        count += linear_parameters(hidden_width, hidden_width)
        count += 2 * hidden_width
        count += linear_parameters(hidden_width, 2 * hidden_width)
        count += linear_parameters(2 * hidden_width, hidden_width)
        count += 2 * hidden_width
    count += linear_parameters(hidden_width, WORKER_ACTION_COUNT)
    count += linear_parameters(hidden_width, WORKER_KIND_COUNT)
    count += linear_parameters(hidden_width, hidden_width) * 2
    count += linear_parameters(hidden_width, MARKET_ITEM_COUNT)
    count += linear_parameters(hidden_width, MARKET_QUANTITY_COUNT)
    count += linear_parameters(hidden_width, VALUE_COUNT)
    count += linear_parameters(hidden_width, MARKET_ACTIVE_COUNT)
    return count
