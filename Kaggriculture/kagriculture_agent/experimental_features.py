"""Opt-in, observation-only context features for training experiments.

This module is deliberately separate from :mod:`features`.  Its output is not
part of the production feature schema or artifact contract.  Every numeric
value is bounded to ``[-1, 1]`` and missing public history is represented by a
neutral zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

EXPERIMENTAL_FEATURE_VARIANT = "experimental_context_v1"

_ACTION_VOCABULARY = (
    "PASS", "MOVE", "WATER", "FERTILIZE", "HARVEST", "PLANT", "FEED",
    "CARE", "SELL", "BUILD", "DIG", "WEED", "HIRE", "BUY_LAND",
    "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "NORTH", "SOUTH", "EAST",
    "WEST", "UNKNOWN",
)
_HISTORY_KEYS = (
    "recent_actions", "action_history", "history", "transitions", "events",
)
_ACTION_KEYS = ("action", "last_action", "command", "selected_action")
_SUCCESS_KEYS = ("success", "succeeded", "ok", "successful")
_TREND_KEYS = ("history", "values", "series")


def extract_experimental_context(state: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Extract bounded context from public observation fields only.

    The returned mapping has a stable, intentionally small shape.  The action
    identity is a fixed one-hot tuple; all other numeric fields are scalars in
    ``[-1, 1]``.  In particular, this function never consults a ``private``
    field, memory object, or object attribute.
    """
    source = state if isinstance(state, Mapping) else {}
    history = _history(source)
    action, outcome = _recent_action(source, history)
    price = _first_trend(source, history, "price")
    demand = _first_trend(source, history, "demand")
    return {
        "variant": EXPERIMENTAL_FEATURE_VARIANT,
        "recent_action_identity": _action_identity(action),
        "recent_action_outcome": outcome,
        "price_trend": price,
        "demand_trend": demand,
        "recovery_slack": _recovery_slack(source),
        "task_opportunity": _task_opportunity(source),
    }


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _public_get(source: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key == "private":
        return default
    return source.get(key, default)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _bounded(value: Any, default: float = 0.0) -> float:
    number = _finite(value)
    if number is None:
        return default
    return max(-1.0, min(1.0, number))


def _history(source: Mapping[str, Any]) -> list[Any]:
    for key in _HISTORY_KEYS:
        value = _public_get(source, key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        nested = _mapping(value)
        if nested is not None:
            for nested_key in ("actions", "records", "entries"):
                entries = nested.get(nested_key)
                if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                    return list(entries)
    return []


def _recent_action(source: Mapping[str, Any], history: Sequence[Any]) -> tuple[Any, float]:
    for record in reversed(history):
        action = _record_action(record)
        if action is not None:
            return action, _record_outcome(record)

    for key in ("last_action", "recent_action", "action"):
        action = _public_get(source, key)
        if action is not None:
            outcome = 0.0
            for outcome_key in ("last_action_outcome", "action_outcome", "last_action_result"):
                if outcome_key in source:
                    outcome = _outcome(source.get(outcome_key))
                    break
            if isinstance(action, Mapping):
                outcome = _record_outcome(action) or outcome
            return action, outcome
    return None, 0.0


def _record_action(record: Any) -> Any:
    mapping = _mapping(record)
    if mapping is not None:
        for key in _ACTION_KEYS:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        if any(key in mapping for key in ("type", "kind", "name")):
            return mapping
    if isinstance(record, (str, bytes)):
        return record
    if isinstance(record, Sequence) and not isinstance(record, (str, bytes)) and record:
        return record[0]
    return None


def _record_outcome(record: Any) -> float:
    mapping = _mapping(record)
    if mapping is None:
        return 0.0
    for key in _SUCCESS_KEYS:
        if key in mapping:
            return _outcome(mapping[key])
    for key in ("outcome", "result", "status", "reward"):
        if key in mapping:
            return _outcome(mapping[key])
    return 0.0


def _outcome(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    number = _finite(value)
    if number is not None:
        return 1.0 if number > 0 else -1.0 if number < 0 else 0.0
    text = str(value).strip().upper()
    if text in {"SUCCESS", "SUCCEEDED", "OK", "DONE", "PASSED", "TRUE"}:
        return 1.0
    if text in {"FAILURE", "FAILED", "ERROR", "INVALID", "FALSE"}:
        return -1.0
    return 0.0


def _action_name(action: Any) -> str:
    mapping = _mapping(action)
    if mapping is not None:
        for key in ("type", "kind", "name", "action"):
            if key in mapping:
                return _action_name(mapping[key])
        return ""
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
        return _action_name(action[0]) if action else ""
    return str(action or "").strip().upper().split()[0] if str(action or "").strip() else ""


def _action_identity(action: Any) -> tuple[float, ...]:
    name = _action_name(action)
    if name in {"NOOP", "IDLE"}:
        name = "PASS"
    if not name:
        return (0.0,) * len(_ACTION_VOCABULARY)
    index = _ACTION_VOCABULARY.index(name) if name in _ACTION_VOCABULARY else len(_ACTION_VOCABULARY) - 1
    return tuple(1.0 if position == index else 0.0 for position in range(len(_ACTION_VOCABULARY)))


def _first_trend(source: Mapping[str, Any], history: Sequence[Any], kind: str) -> float:
    containers: list[Any] = []
    if kind == "price":
        keys = ("price_history", "prices_history", "price_trend_history", "history")
        fields = ("price", "prices", "quote", "quotes")
        parent_key = "market"
    else:
        keys = ("demand_history", "needs_history", "demand_trend_history", "history")
        fields = ("demand", "demands", "needs")
        parent_key = "town"
    for key in keys:
        value = _public_get(source, key)
        if value is not None:
            containers.append(value)
        parent = _mapping(_public_get(source, parent_key))
        if parent is not None and key in parent:
            containers.append(parent[key])
    for record in history:
        mapping = _mapping(record)
        if mapping is None:
            continue
        value = next((mapping.get(field) for field in fields if field in mapping), None)
        parent = _mapping(mapping.get(parent_key))
        if value is None and parent is not None:
            value = next((parent.get(field) for field in fields if field in parent), None)
        if value is not None:
            containers.append(value)
    for value in containers:
        result = _select_trend(value, fields)
        if result != 0.0:
            return result
    return 0.0


def _select_trend(value: Any, fields: Sequence[str]) -> float:
    mapping = _mapping(value)
    if mapping is not None:
        for field in fields:
            if field in mapping:
                return _trend(mapping[field])
        return _trend(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        projected = []
        for item in value:
            item_mapping = _mapping(item)
            if item_mapping is None:
                projected = []
                break
            selected = next((item_mapping[field] for field in fields if field in item_mapping), None)
            if selected is not None:
                projected.append(selected)
        if projected:
            return _trend(projected)
    return _trend(value)


def _trend(value: Any) -> float:
    nested = _mapping(value)
    if nested is not None:
        for key in _TREND_KEYS:
            if key in nested:
                return _trend(nested[key])
        changes = [_trend(item) for item in nested.values()]
        changes = [item for item in changes if item != 0.0]
        return _bounded(sum(changes) / len(changes)) if changes else 0.0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) < 2:
            return 0.0
        if all(_finite(item) is not None for item in values):
            return _relative_change(_finite(values[0]), _finite(values[-1]))
        if all(_mapping(item) is not None for item in values):
            keys = set().union(*(item.keys() for item in values if _mapping(item) is not None))
            changes = []
            for key in keys:
                series = [item[key] for item in values if key in item]
                change = _trend(series)
                if change != 0.0:
                    changes.append(change)
            return _bounded(sum(changes) / len(changes)) if changes else 0.0
        counts = [_collection_size(item) for item in values]
        return _relative_change(counts[0], counts[-1])
    return 0.0


def _relative_change(first: float | None, last: float | None) -> float:
    if first is None or last is None:
        return 0.0
    denominator = max(abs(first), 1.0)
    return _bounded((last - first) / denominator)


def _collection_size(value: Any) -> float:
    if isinstance(value, Mapping):
        return float(sum(1 for item in value.values() if item not in (False, None, 0, "")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return float(len(value))
    number = _finite(value)
    return number if number is not None else 0.0


def _recovery_slack(source: Mapping[str, Any]) -> float:
    for key in ("recovery_slack", "deadline_slack", "recovery"):
        value = _public_get(source, key)
        if isinstance(value, Mapping):
            slack = _finite(value.get("slack", value.get("remaining")))
            if slack is None:
                available = _finite(value.get("available", value.get("available_turns")))
                required = _finite(value.get("required", value.get("required_turns")))
                if available is not None and required is not None:
                    return _relative_change(required, available)
            if slack is not None:
                return _bounded(slack / 10.0)
        elif _finite(value) is not None:
            return _bounded(_finite(value) / 10.0)
    return 0.0


def _task_opportunity(source: Mapping[str, Any]) -> float:
    for key in ("task_opportunity", "task_opportunities", "opportunities", "available_actions", "legal_actions", "tasks"):
        value = _public_get(source, key)
        if value is None:
            continue
        number = _finite(value)
        if number is not None:
            return _bounded(number / 10.0)
        if isinstance(value, Mapping):
            values = list(value.values())
            if not values:
                return 0.0
            active = sum(1 for item in values if item not in (False, None, 0, "", "PASS", "IDLE"))
            return _bounded(active / len(values))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                return 0.0
            active = sum(1 for item in value if _action_name(item) not in {"", "PASS", "IDLE", "NOOP"})
            return _bounded(active / len(value))
    return 0.0
