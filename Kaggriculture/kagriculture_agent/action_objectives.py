"""Conditional log-probability and entropy objectives for policy outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict, TypeAlias

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor: TypeAlias = Any

try:  # pragma: no cover - import safety is covered by the model module tests
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised without training extras
    torch = None

from .runtime_identity import DEFAULT_ACTION_REPRESENTATION, validate_action_representation


_OUTPUT_NAMES = (
    "worker_act_logits",
    "worker_target_logits",
    "worker_kind_logits",
    "market_active_logits",
    "market_item_logits",
    "market_quantity_logits",
)


class ActionOutputs(TypedDict):
    """The categorical logits emitted by ``CompactPolicyNet``."""

    worker_act_logits: Tensor
    worker_target_logits: Tensor
    worker_kind_logits: Tensor
    market_active_logits: Tensor
    market_item_logits: Tensor
    market_quantity_logits: Tensor


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for action objectives; install the optional "
            "training dependencies."
        )
    return torch


def _require_tensor(value: Any, name: str) -> Any:
    th = _require_torch()
    if not th.is_tensor(value):
        raise ValueError(f"{name} must be a PyTorch tensor")
    return value


def _validate_logits(
    outputs: Mapping[str, Any], *, action_representation: str,
) -> dict[str, Any]:
    th = _require_torch()
    missing = [name for name in _OUTPUT_NAMES if name not in outputs]
    if missing:
        raise ValueError(f"outputs is missing: {', '.join(missing)}")
    logits = {name: _require_tensor(outputs[name], name) for name in _OUTPUT_NAMES}
    for name, value in logits.items():
        if not value.dtype.is_floating_point:
            raise ValueError(f"{name} must have a floating-point dtype")
        if not th.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if logits["worker_act_logits"].ndim != 3 or logits["worker_act_logits"].shape[-1] != 2:
        raise ValueError("worker_act_logits must have shape [batch, workers, 2]")
    validate_action_representation(action_representation, source="objective")
    if logits["worker_target_logits"].ndim != 3 or logits["worker_target_logits"].shape[-1] < 1:
        raise ValueError("worker_target_logits must have shape [batch, workers, classes]")
    kind_logits = logits["worker_kind_logits"]
    if action_representation == DEFAULT_ACTION_REPRESENTATION:
        if kind_logits.ndim != 3 or kind_logits.shape[-1] < 1:
            raise ValueError("worker_kind_logits must have shape [batch, workers, classes]")
    elif (
        kind_logits.ndim != 4
        or kind_logits.shape[2] != logits["worker_target_logits"].shape[2]
        or kind_logits.shape[-1] < 1
    ):
        raise ValueError(
            "worker_kind_logits must have shape [batch, workers, targets, classes]"
        )
    value = logits["market_active_logits"]
    if value.ndim != 2 or value.shape[-1] != 2:
        raise ValueError("market_active_logits must have shape [batch, 2]")
    for name in ("market_item_logits", "market_quantity_logits"):
        value = logits[name]
        if value.ndim != 2 or value.shape[-1] < 1:
            raise ValueError(f"{name} must have shape [batch, classes]")

    batch_size, worker_count = logits["worker_act_logits"].shape[:2]
    for name in ("worker_target_logits", "worker_kind_logits"):
        if logits[name].shape[:2] != (batch_size, worker_count):
            raise ValueError(f"{name} must match worker_act_logits batch and worker dimensions")
    for name in ("market_active_logits", "market_item_logits", "market_quantity_logits"):
        if logits[name].shape[0] != batch_size:
            raise ValueError(f"{name} must match the output batch dimension")
    devices = {value.device for value in logits.values()}
    if len(devices) != 1:
        raise ValueError("all output tensors must be on the same device")
    return logits


def _validate_labels(
    labels: Mapping[str, Any], *, batch_size: int, worker_count: int,
    class_counts: Mapping[str, int], device: Any,
) -> dict[str, Any]:
    th = _require_torch()
    expected_shapes = {
        "worker_active": (batch_size, worker_count),
        "worker_target": (batch_size, worker_count),
        "worker_kind": (batch_size, worker_count),
        "market_active": (batch_size,),
        "market_item": (batch_size,),
        "market_quantity": (batch_size,),
    }
    normalized = {}
    for name, shape in expected_shapes.items():
        if name not in labels:
            raise ValueError(f"labels is missing: {name}")
        value = _require_tensor(labels[name], name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype not in (th.int8, th.int16, th.int32, th.int64, th.uint8):
            raise ValueError(f"{name} must contain integer labels")
        if value.device != device:
            raise ValueError(f"{name} must be on the same device as the output tensors")
        normalized[name] = value

    for name in ("worker_active", "market_active"):
        value = normalized[name]
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name} labels must be 0 or 1")
    for name, class_count in class_counts.items():
        value = normalized[name]
        if name == "worker_target":
            active = normalized["worker_active"].bool()
        elif name == "worker_kind":
            active = normalized["worker_active"].bool()
        else:
            active = normalized["market_active"].bool()
        if ((value < 0) | (value >= class_count)).masked_select(active).any():
            raise ValueError(f"{name} labels are out of range on active rows")
    return normalized


def _masked_log_softmax(logits: Any, mask: Any, active: Any, name: str) -> Any:
    th = _require_torch()
    if mask is None:
        return logits.log_softmax(dim=-1)
    mask = _require_tensor(mask, name)
    if mask.dtype is not th.bool:
        raise ValueError(f"{name} must have boolean dtype")
    if mask.shape != logits.shape:
        raise ValueError(f"{name} shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}")
    if mask.device != logits.device:
        raise ValueError(f"{name} must be on the same device as the output tensors")
    legal = mask.any(dim=-1)
    if (active & ~legal).any():
        branch = {
            "worker_target_mask": "target",
            "worker_kind_mask": "kind",
            "market_item_mask": "market item",
            "market_quantity_mask": "market quantity",
        }[name]
        raise ValueError(f"active row has no legal {branch}")
    safe_mask = mask | (~active).unsqueeze(-1)
    return logits.masked_fill(~safe_mask, -th.inf).log_softmax(dim=-1)


def _validate_selected_mask(label: Any, mask: Any, active: Any, name: str) -> None:
    if mask is None:
        return
    safe_label = label.masked_fill(~active, 0)
    selected_is_legal = mask.gather(-1, safe_label.unsqueeze(-1)).squeeze(-1)
    if (active & ~selected_is_legal).any():
        raise ValueError(f"{name} label is not legal for an active row")


def _selected(log_probs: Any, labels: Any, active: Any) -> Any:
    safe_labels = labels.masked_fill(~active, 0)
    return log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).masked_fill(~active, 0.0)


def _entropy(log_probs: Any) -> Any:
    probabilities = log_probs.exp()
    finite_log_probs = log_probs.masked_fill(~log_probs.isfinite(), 0.0)
    return -(probabilities * finite_log_probs).sum(dim=-1)


def conditional_action_objectives(
    outputs: Mapping[str, Tensor],
    *,
    worker_active: Tensor,
    worker_target: Tensor,
    worker_kind: Tensor,
    market_active: Tensor,
    market_item: Tensor,
    market_quantity: Tensor,
    worker_target_mask: Tensor | None = None,
    worker_kind_mask: Tensor | None = None,
    market_item_mask: Tensor | None = None,
    market_quantity_mask: Tensor | None = None,
    action_representation: str = DEFAULT_ACTION_REPRESENTATION,
) -> tuple[Tensor, Tensor]:
    """Return conditional batch log-probabilities and mean entropy.

    ``outputs`` uses the model's existing logits keys. Worker target and kind
    terms are included only for active workers, while market item and quantity
    terms are included only for active market rows. Worker-active and
    market-active heads are always scored.

    Optional boolean masks have the same shape as their corresponding logits
    and are applied before ``log_softmax``. An inactive all-illegal row is
    ignored safely, but every active row must have at least one legal choice.
    """
    th = _require_torch()
    logits = _validate_logits(outputs, action_representation=action_representation)
    batch_size, worker_count = logits["worker_act_logits"].shape[:2]
    labels = _validate_labels(
        {
            "worker_active": worker_active,
            "worker_target": worker_target,
            "worker_kind": worker_kind,
            "market_active": market_active,
            "market_item": market_item,
            "market_quantity": market_quantity,
        },
        batch_size=batch_size,
        worker_count=worker_count,
        class_counts={
            "worker_target": logits["worker_target_logits"].shape[-1],
            "worker_kind": logits["worker_kind_logits"].shape[-1],
            "market_item": logits["market_item_logits"].shape[-1],
            "market_quantity": logits["market_quantity_logits"].shape[-1],
        },
        device=logits["worker_act_logits"].device,
    )
    worker_is_active = labels["worker_active"].bool()
    market_is_active = labels["market_active"].bool()

    target_log = _masked_log_softmax(
        logits["worker_target_logits"], worker_target_mask, worker_is_active, "worker_target_mask",
    )
    if action_representation == DEFAULT_ACTION_REPRESENTATION:
        selected_kind_logits = logits["worker_kind_logits"]
        selected_kind_mask = worker_kind_mask
    else:
        safe_targets = labels["worker_target"].masked_fill(~worker_is_active, 0)
        target_index = safe_targets.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, logits["worker_kind_logits"].shape[-1],
        )
        selected_kind_logits = logits["worker_kind_logits"].gather(2, target_index).squeeze(2)
        if worker_kind_mask is None:
            selected_kind_mask = None
        else:
            if worker_kind_mask.shape != logits["worker_kind_logits"].shape:
                raise ValueError(
                    "worker_kind_mask shape must match worker_kind_logits shape"
                )
            selected_kind_mask = worker_kind_mask.gather(2, target_index).squeeze(2)
    kind_log = _masked_log_softmax(
        selected_kind_logits, selected_kind_mask, worker_is_active, "worker_kind_mask",
    )
    item_log = _masked_log_softmax(
        logits["market_item_logits"], market_item_mask, market_is_active, "market_item_mask",
    )
    quantity_log = _masked_log_softmax(
        logits["market_quantity_logits"], market_quantity_mask, market_is_active, "market_quantity_mask",
    )
    _validate_selected_mask(labels["worker_target"], worker_target_mask, worker_is_active, "worker target")
    _validate_selected_mask(
        labels["worker_kind"], selected_kind_mask, worker_is_active, "worker kind",
    )
    _validate_selected_mask(labels["market_item"], market_item_mask, market_is_active, "market item")
    _validate_selected_mask(labels["market_quantity"], market_quantity_mask, market_is_active, "market quantity")

    act_log = logits["worker_act_logits"].log_softmax(dim=-1)
    market_active_log = logits["market_active_logits"].log_softmax(dim=-1)
    worker_log_probs = _selected(act_log, labels["worker_active"], th.ones_like(worker_is_active, dtype=th.bool))
    worker_log_probs += (
        _selected(target_log, labels["worker_target"], worker_is_active)
        + _selected(kind_log, labels["worker_kind"], worker_is_active)
    )
    market_log_probs = (
        _selected(market_active_log, labels["market_active"], th.ones_like(market_is_active, dtype=th.bool))
        + _selected(item_log, labels["market_item"], market_is_active)
        + _selected(quantity_log, labels["market_quantity"], market_is_active)
    )
    log_probs = worker_log_probs.sum(dim=1) + market_log_probs

    entropy_per_row = (
        _entropy(act_log).sum(dim=1)
        + _entropy(market_active_log)
        + _entropy(target_log).masked_fill(~worker_is_active, 0.0).sum(dim=1)
        + _entropy(kind_log).masked_fill(~worker_is_active, 0.0).sum(dim=1)
        + _entropy(item_log).masked_fill(~market_is_active, 0.0)
        + _entropy(quantity_log).masked_fill(~market_is_active, 0.0)
    )
    entropy = entropy_per_row.mean() if batch_size else logits["worker_act_logits"].sum() * 0.0
    return log_probs, entropy


compute_action_objectives = conditional_action_objectives
action_objectives = conditional_action_objectives


__all__ = [
    "ActionOutputs",
    "action_objectives",
    "compute_action_objectives",
    "conditional_action_objectives",
]
