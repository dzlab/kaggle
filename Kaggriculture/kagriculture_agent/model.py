"""Training-time PyTorch model for compact Kaggriculture policies.

Forward contract for a single :class:`FeatureBatch` or a sequence of batches::

    {
        "worker_act_logits": float[B, 10, 2],
        "worker_target_logits": float[B, 10, 100],
        "worker_kind_logits": float[B, 10, len(ACTION_VOCAB["worker_kinds"])],
        "market_item_logits": float[B, len(PRODUCTS)],
        "market_quantity_logits": float[B, len(ACTION_VOCAB["market_quantities"])],
        "value": float[B],
    }

The module is import-safe without PyTorch. Constructing the network or using
tensor conversion requires the optional training dependency.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .constants import PRODUCTS
from .features import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_TOKEN_SIZE,
    MARKET_TOKEN_SIZE,
    TILE_TOKEN_SIZE,
    WORKER_TOKEN_SIZE,
    FeatureBatch,
)

try:  # pragma: no cover - exercised only when the optional dependency exists
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - local test venv may omit torch
    torch = None
    nn = None


MODEL_VERSION = "learned_v1"
HIDDEN_WIDTH = 128
ATTENTION_BLOCKS = 4
ATTENTION_HEADS = 4
MLP_WIDTH = 256
FEATURE_INPUT_SIZES = {
    "tile": TILE_TOKEN_SIZE,
    "worker": WORKER_TOKEN_SIZE,
    "market": MARKET_TOKEN_SIZE,
    "global": GLOBAL_TOKEN_SIZE,
}
ACTION_VOCAB = {
    "worker_kinds": (
        "PASS", "MOVE", "WATER", "HARVEST", "PLANT", "FERTILIZE", "FEED",
        "CARE", "PICKUP", "PLACE", "DROP", "SELL", "DIG", "WEED",
    ),
    "market_items": tuple(sorted(PRODUCTS)),
    "market_quantities": (0, 1, 2, 4, 8, 16, 32, 64),
}


def torch_available() -> bool:
    return torch is not None


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for Kaggriculture training; install the optional "
            "training dependencies, for example `uv sync --extra training`."
        )
    return torch


def set_training_seed(seed: int) -> None:
    """Seed Python, NumPy when installed, and PyTorch when installed."""
    random.seed(int(seed))
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if np is not None:
        np.random.seed(int(seed) % (2**32))
    if torch is not None:
        torch.manual_seed(int(seed))
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def _as_feature_list(feature_batch: FeatureBatch | Sequence[FeatureBatch]) -> list[FeatureBatch]:
    if isinstance(feature_batch, FeatureBatch):
        batches = [feature_batch]
    elif isinstance(feature_batch, Sequence) and not isinstance(feature_batch, (str, bytes)):
        batches = list(feature_batch)
    else:
        raise TypeError("feature_batch must be a FeatureBatch or sequence of FeatureBatch")
    if not batches:
        raise ValueError("feature batch sequence cannot be empty")
    for features in batches:
        if not isinstance(features, FeatureBatch):
            raise TypeError("all feature batches must be FeatureBatch instances")
        if features.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema version: {features.schema_version}")
    return batches


def feature_batch_to_tensors(
    feature_batch: FeatureBatch | Sequence[FeatureBatch], *, device: Any = None,
) -> dict[str, Any]:
    """Convert dependency-free features to float tensors with a batch dimension."""
    th = require_torch()
    batches = _as_feature_list(feature_batch)
    return {
        "tile": th.tensor([batch.tile_tokens for batch in batches], dtype=th.float32, device=device),
        "worker": th.tensor([batch.worker_tokens for batch in batches], dtype=th.float32, device=device),
        "market": th.tensor([batch.market_tokens for batch in batches], dtype=th.float32, device=device),
        "global": th.tensor([batch.global_tokens for batch in batches], dtype=th.float32, device=device),
    }


if nn is not None:

    class _ResidualAttentionBlock(nn.Module):
        def __init__(self, width: int = HIDDEN_WIDTH) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(width, ATTENTION_HEADS, batch_first=True)
            self.attention_norm = nn.LayerNorm(width)
            self.mlp = nn.Sequential(
                nn.Linear(width, MLP_WIDTH),
                nn.GELU(),
                nn.Linear(MLP_WIDTH, width),
            )
            self.mlp_norm = nn.LayerNorm(width)

        def forward(self, tokens: Any) -> Any:
            attended, _weights = self.attention(tokens, tokens, tokens, need_weights=False)
            tokens = self.attention_norm(tokens + attended)
            return self.mlp_norm(tokens + self.mlp(tokens))


    class CompactPolicyNet(nn.Module):
        """Small attention policy over tile, worker, market, and global tokens."""

        def __init__(self, *, hidden_width: int = HIDDEN_WIDTH) -> None:
            super().__init__()
            self.hidden_width = int(hidden_width)
            self.tile_projection = nn.Linear(TILE_TOKEN_SIZE, self.hidden_width)
            self.worker_projection = nn.Linear(WORKER_TOKEN_SIZE, self.hidden_width)
            self.market_projection = nn.Linear(MARKET_TOKEN_SIZE, self.hidden_width)
            self.global_projection = nn.Linear(GLOBAL_TOKEN_SIZE, self.hidden_width)
            self.type_embedding = nn.Embedding(4, self.hidden_width)
            self.blocks = nn.ModuleList(
                _ResidualAttentionBlock(self.hidden_width) for _ in range(ATTENTION_BLOCKS)
            )
            self.worker_act_head = nn.Linear(self.hidden_width, 2)
            self.worker_kind_head = nn.Linear(self.hidden_width, len(ACTION_VOCAB["worker_kinds"]))
            self.target_worker_head = nn.Linear(self.hidden_width, self.hidden_width)
            self.target_tile_head = nn.Linear(self.hidden_width, self.hidden_width)
            self.market_item_head = nn.Linear(self.hidden_width, len(ACTION_VOCAB["market_items"]))
            self.market_quantity_head = nn.Linear(
                self.hidden_width, len(ACTION_VOCAB["market_quantities"]),
            )
            self.value_head = nn.Linear(self.hidden_width, 1)

        def forward(self, feature_batch: FeatureBatch | Sequence[FeatureBatch]) -> dict[str, Any]:
            tensors = feature_batch_to_tensors(feature_batch, device=next(self.parameters()).device)
            batch_size = tensors["tile"].shape[0]
            tile_tokens = self.tile_projection(tensors["tile"]) + self.type_embedding.weight[0]
            worker_tokens = self.worker_projection(tensors["worker"]) + self.type_embedding.weight[1]
            market_tokens = self.market_projection(tensors["market"]) + self.type_embedding.weight[2]
            global_token = self.global_projection(tensors["global"]).view(batch_size, 1, self.hidden_width)
            global_token = global_token + self.type_embedding.weight[3]
            tokens = torch.cat((tile_tokens, worker_tokens, market_tokens, global_token), dim=1)
            for block in self.blocks:
                tokens = block(tokens)
            tile_count = tensors["tile"].shape[1]
            worker_count = tensors["worker"].shape[1]
            market_count = tensors["market"].shape[1]
            encoded_tiles = tokens[:, :tile_count, :]
            encoded_workers = tokens[:, tile_count:tile_count + worker_count, :]
            encoded_market = tokens[:, tile_count + worker_count:tile_count + worker_count + market_count, :]
            encoded_global = tokens[:, -1, :]
            worker_query = self.target_worker_head(encoded_workers)
            tile_key = self.target_tile_head(encoded_tiles)
            worker_target_logits = torch.matmul(worker_query, tile_key.transpose(1, 2)) / (self.hidden_width ** 0.5)
            pooled_market = encoded_market.mean(dim=1) if market_count else encoded_global
            return {
                "worker_act_logits": self.worker_act_head(encoded_workers),
                "worker_target_logits": worker_target_logits,
                "worker_kind_logits": self.worker_kind_head(encoded_workers),
                "market_item_logits": self.market_item_head(pooled_market),
                "market_quantity_logits": self.market_quantity_head(pooled_market),
                "value": self.value_head(encoded_global).squeeze(-1),
            }

else:

    class CompactPolicyNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for CompactPolicyNet; install the optional "
                "training dependencies."
            )
