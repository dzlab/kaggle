"""Compact, versioned, dependency-free features for learned Kaggriculture policies.

The feature contract is deliberately plain tuples so it is usable by the
runtime without NumPy or PyTorch.  Tile tokens are laid out as::

    [x, y, locked, empty, crop one-hot, age, yield timing, decay timing,
     watered, fertilized, structure one-hot, animal one-hot, feed need,
     care need, expected harvest, deadline slack]

Worker tokens are position, role, held-item, worker index, task kind, target
distance, and deadline slack.  Market tokens are product, quote, inventory
delta, sequential post-sale quote, demand, and floor flags.  Global tokens are
day/hour, cash, shed utilization, wheat reserve, free workers, unlocked land,
production, terminal horizon, and strategy one-hot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .constants import ANIMALS, CROPS, MARKET_I0, PRICE_FLOOR, PRODUCTS, season_days, shed_capacity
from .economics import market_price
from .observation import parse_observation
from .routing import normalize_position
from .strategy import market_sale_quotes, select_strategy
from .types import Position

FEATURE_SCHEMA_VERSION = 1
BOARD_SIZE = 10
MAX_WORKERS = 10
_TASK_KINDS = ("IDLE", "MOVE", "WATER", "FERTILIZE", "HARVEST", "PLANT", "FEED", "CARE", "SELL", "BUILD", "DIG", "WEED")
_STRATEGIES = ("current", "melon", "premium", "mixed")
TILE_TOKEN_SIZE = 23
WORKER_TOKEN_SIZE = 2 + 2 + len(PRODUCTS) + 1 + len(_TASK_KINDS) + 1 + 1
MARKET_TOKEN_SIZE = len(PRODUCTS) + 5
GLOBAL_TOKEN_SIZE = 9 + len(_STRATEGIES)


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if isfinite(result) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _one_hot(value: str, choices: Sequence[str]) -> tuple[float, ...]:
    value = str(value or "").upper()
    return tuple(1.0 if value == choice else 0.0 for choice in choices)


def _position(value: Any) -> Position | None:
    return normalize_position(value)


def _tile_kind(tile: Any) -> str:
    nested = _get(tile, "structure", _get(tile, "animal", {}))
    return str(_get(tile, "kind", _get(tile, "type", _get(nested, "kind", ""))) or "").upper()


def _crop(tile: Any) -> str:
    raw = _get(tile, "crop")
    if isinstance(raw, Mapping):
        raw = _get(raw, "crop", _get(raw, "kind", _get(raw, "name")))
    raw = raw or _tile_kind(tile)
    return str(raw or "").upper() if str(raw or "").upper() in CROPS else ""


def _tiles(state: Mapping[str, Any]) -> dict[Position, Any]:
    farm = _mapping(state.get("farm"))
    source = state.get("tiles", farm.get("tiles", []))
    result: dict[Position, Any] = {}
    if isinstance(source, Mapping):
        for raw_position, tile in source.items():
            position = _position(raw_position)
            if position is not None and 0 <= position.x < BOARD_SIZE and 0 <= position.y < BOARD_SIZE:
                result[position] = tile
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        for y, row in enumerate(source[:BOARD_SIZE]):
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                for x, tile in enumerate(row[:BOARD_SIZE]):
                    result[Position(x, y)] = tile
    return result


def _task(worker: Any, state: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _get(worker, "task", _get(worker, "current_task", {}))
    if isinstance(raw, Mapping):
        return raw
    index = _get(worker, "index", 0)
    for task in state.get("tasks", ()) if isinstance(state.get("tasks"), Sequence) else ():
        if _get(task, "worker_index", -1) == index:
            return _mapping(_get(task, "task", task))
    return {}


def _workers(state: Mapping[str, Any]) -> list[Any]:
    farm = _mapping(state.get("farm"))
    workers = state.get("workers", farm.get("workers", []))
    return list(workers) if isinstance(workers, Sequence) and not isinstance(workers, (str, bytes)) else []


def _market_value(state: Mapping[str, Any], key: str, item: str, default: float) -> float:
    market = _mapping(state.get("market"))
    values = market.get(key, state.get(key, {}))
    if isinstance(values, Mapping):
        return _number(values.get(item), default)
    return _number(values, default)


def _shed(state: Mapping[str, Any]) -> Mapping[str, Any]:
    private = _mapping(state.get("private"))
    shed = private.get("shed", state.get("shed", {}))
    return shed if isinstance(shed, Mapping) else {}


@dataclass(frozen=True)
class FeatureBatch:
    tile_tokens: tuple[tuple[float, ...], ...]
    worker_tokens: tuple[tuple[float, ...], ...]
    market_tokens: tuple[tuple[float, ...], ...]
    global_tokens: tuple[float, ...]
    tile_positions: tuple[Position, ...]
    schema_version: int = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema version: {self.schema_version}")


def extract_features(state: Any, *, schema_version: int = FEATURE_SCHEMA_VERSION) -> FeatureBatch:
    """Extract deterministic fixed-capacity features from the current player state."""
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema version: {schema_version}")
    raw = state if isinstance(state, Mapping) else {}
    if raw.get("schema_version", FEATURE_SCHEMA_VERSION) != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema version: {raw.get('schema_version')}")
    # Parse only the selected farm when handed a raw engine observation.  The
    # parser copies containers; no private state belonging to another farm is read.
    source = parse_observation(raw) if "farms" in raw and "farm" not in raw else dict(raw)
    day = max(0.0, _number(source.get("day")))
    hour = max(0.0, _number(source.get("hour")))
    positions = tuple(Position(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE))
    tiles = _tiles(source)
    tile_tokens = tuple(_tile_token(position, tiles.get(position), day) for position in positions)
    workers = _workers(source)
    worker_tokens = tuple(_worker_token(worker, source, day) for worker in workers[:MAX_WORKERS])
    worker_tokens += tuple(_empty_worker_token() for _ in range(MAX_WORKERS - len(worker_tokens)))
    market_tokens = tuple(_market_token(item, source) for item in sorted(PRODUCTS))
    global_tokens = _global_token(source, day, hour, len(workers))
    return FeatureBatch(tile_tokens, worker_tokens, market_tokens, global_tokens, positions)


def _tile_token(position: Position, tile: Any, day: float) -> tuple[float, ...]:
    tile = _mapping(tile) if not isinstance(tile, str) else {"kind": tile}
    kind = _tile_kind(tile)
    crop = _crop(tile)
    animal = str(_get(tile, "animal", _get(tile, "species", "")) or "").upper()
    if isinstance(_get(tile, "animal"), Mapping):
        animal = str(_get(_get(tile, "animal"), "species", _get(_get(tile, "animal"), "kind", ""))).upper()
    structure = str(_get(tile, "structure", kind) or "").upper()
    if isinstance(_get(tile, "structure"), Mapping):
        structure = str(_get(_get(tile, "structure"), "kind", "")).upper()
    locked = 1.0 if kind == "LOCKED" or bool(_get(tile, "locked", False)) else 0.0
    empty = 1.0 if not crop and animal not in ANIMALS and structure not in {"COOP", "PASTURE"} and not locked else 0.0
    age = max(0.0, _number(_get(tile, "age", _get(tile, "crop_age", 0))))
    crop_data = CROPS.get(crop, {})
    expected = max(0.0, _number(_get(tile, "yield_units", _get(tile, "expected_harvest", 0))))
    if expected == 0 and crop:
        expected = float(crop_data.get("max_yield", 0))
    deadline = _number(_get(_get(tile, "task", {}), "deadline", day + season_days), day + season_days)
    needs = _mapping(_get(tile, "needs", {}))
    return (position.x / 9.0, position.y / 9.0, locked, empty, *_one_hot(crop, tuple(CROPS)),
            age / season_days, age / max(1.0, float(crop_data.get("first_yield_day", season_days))),
            age / max(1.0, float(crop_data.get("max_yield_day", season_days))),
            float(bool(_get(tile, "watered", _get(tile, "water", False)))),
            float(bool(_get(tile, "fertilized", _get(tile, "fertilizer", False)))),
            *_one_hot(structure, ("COOP", "PASTURE")), *_one_hot(animal, tuple(ANIMALS)),
            float(bool(_get(tile, "needs_feed", needs.get("feed", False)))),
            float(bool(_get(tile, "needs_care", needs.get("care", False)))), expected / 100.0,
            max(-1.0, min(1.0, (deadline - day) / season_days)))


def _worker_token(worker: Any, state: Mapping[str, Any], day: float) -> tuple[float, ...]:
    position = _position(_get(worker, "position", worker)) or Position(0, 0)
    role = str(_get(worker, "role", "WORKER") or "WORKER").upper()
    held = _get(worker, "inventory", _get(worker, "held_items", ()))
    held_names = set(held) if isinstance(held, Sequence) and not isinstance(held, (str, bytes)) else set(held) if isinstance(held, Mapping) else set()
    task = _task(worker, state)
    target = _position(_get(task, "target", _get(task, "position")))
    distance = ((abs(target.x - position.x) + abs(target.y - position.y)) / 18.0) if target else 0.0
    deadline = _number(_get(task, "deadline", day + season_days), day + season_days)
    return (position.x / 9.0, position.y / 9.0, float(role == "FARMER"), float(role != "FARMER"),
            *tuple(float(item in held_names) for item in PRODUCTS),
            _number(_get(worker, "index", 0)) / MAX_WORKERS,
            *_one_hot(str(_get(task, "kind", "IDLE")), _TASK_KINDS), distance,
            max(-1.0, min(1.0, (deadline - day) / season_days)))


def _empty_worker_token() -> tuple[float, ...]:
    return (0.0,) * WORKER_TOKEN_SIZE


def _market_token(item: str, state: Mapping[str, Any]) -> tuple[float, ...]:
    inventory = _market_value(state, "inventory", item, MARKET_I0)
    quote = _market_value(state, "prices", item, float(market_price(item, inventory)))
    post = market_sale_quotes(item, 1, state)
    demand = _mapping(state.get("town"))
    demand_values = demand.get("demand", demand.get("demands", demand.get("requested_items", ())))
    demand_names = {str(value).upper() for value in demand_values} if isinstance(demand_values, Sequence) and not isinstance(demand_values, str) else {str(demand_values).upper()}
    return (*_one_hot(item, tuple(PRODUCTS)), quote / 100.0, (inventory - MARKET_I0) / MARKET_I0,
            (post[0] if post else quote) / 100.0, float(item in demand_names), float(quote <= PRICE_FLOOR))


def _global_token(state: Mapping[str, Any], day: float, hour: float, worker_count: int) -> tuple[float, ...]:
    shed = _shed(state)
    used = sum(max(0.0, _number(value)) for value in shed.values())
    farm = _mapping(state.get("farm"))
    unlocked = _number(state.get("unlocked_land", farm.get("unlocked_land", 0)))
    production = _number(state.get("production", farm.get("production", 0)))
    strategy_name = str(state.get("strategy", _mapping(state.get("private")).get("strategy", ""))).lower()
    if strategy_name not in _STRATEGIES:
        strategy_name = select_strategy(state).name
    return (day / season_days, hour / 24.0, max(0.0, _number(state.get("cash", farm.get("money", 0)))) / 10000.0,
            min(1.0, used / shed_capacity), max(0.0, _number(shed.get("WHEAT", 0))) / 100.0,
            max(0.0, (MAX_WORKERS - worker_count) / MAX_WORKERS), min(1.0, unlocked / 100.0),
            max(0.0, production) / 100.0, max(0.0, (season_days - day) / season_days),
            *_one_hot(strategy_name, _STRATEGIES))
