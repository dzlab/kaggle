"""Small, deterministic daily task planner for the Kaggriculture agent.

The public planner functions accept either a flat test state or the canonical
mapping returned by :func:`kagriculture_agent.observation.parse_observation`.
The normalized contract is ``day``, ``board_size``, ``tiles``, ``animals``,
``structures``, ``seeds``, ``inventory``, ``workers``, and ``market``.  In the
canonical form, these come from ``farm.tiles``, ``farm.animals``,
``farm.structures``, ``private.seeds``, ``private.shed``, and ``market``;
board size is derived from the square tile grid when it is not explicit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import inf
from typing import Any

from .constants import ANIMALS, CROPS, MARKET_I0
from .economics import forecast_crop, market_price, sell_batch_value
from .observation import shed_access_tiles
from .routing import distance, is_locked_tile, normalize_position, route_to
from .types import EpisodeMemory, Position, Task, WorkerAssignment

_BASIC_NEEDS = frozenset({"WATER", "FEED", "CARE"})
_SHED_WORK = frozenset({"SHED", "SELL"})
_TILE_TASKS = frozenset({
    "WATER", "FEED", "CARE", "HARVEST", "STRUCTURE", "ANIMAL", "WEED", "PLANT",
})


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


_position = normalize_position


def _day(state: Any, memory: EpisodeMemory | Any) -> int:
    value = _get(state, "day", _get(memory, "last_day", 0))
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        attributes = vars(value)
    except TypeError:
        return {}
    return attributes if isinstance(attributes, Mapping) else {}


def _hand_counts(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {
            str(item): int(quantity)
            for item, quantity in value.items()
            if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity > 0
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        counts: dict[str, int] = {}
        for item in value:
            if isinstance(item, str):
                counts[item] = counts.get(item, 0) + 1
        return counts
    return {}


def _grid_size(tiles: Any) -> int | None:
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return None
    widths = [len(row) for row in tiles if isinstance(row, Sequence) and not isinstance(row, (str, bytes))]
    size = max([len(tiles), *widths], default=0)
    return size if size > 0 else None


def normalize_planner_state(state: Any) -> dict[str, Any]:
    """Adapt flat and ``parse_observation`` states to the planner contract."""
    source = dict(_mapping(state))
    farm = _mapping(source.get("farm"))
    private = _mapping(source.get("private"))
    market = _mapping(source.get("market"))
    tiles = source.get("tiles", farm.get("tiles", []))
    hands = _hand_counts(farm.get("hands", source.get("hands", {})))
    seeds = source.get("seeds", private.get("seeds", farm.get("seeds")))
    if not isinstance(seeds, Mapping) or not seeds:
        seeds = hands
    inventory = source.get("inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        inventory = private.get("shed")
    if not isinstance(inventory, Mapping) or not inventory:
        inventory = hands
    board_size = source.get("board_size")
    try:
        board_size = int(board_size)
    except (TypeError, ValueError, OverflowError):
        board_size = _grid_size(tiles)
    if not board_size or board_size < 1:
        board_size = 1
    workers = source.get("workers") or farm.get("workers")
    if not workers:
        workers = []
        farmer_position = _position(farm.get("farmer", source.get("farmer")))
        if farmer_position is not None:
            workers.append({"index": 0, "role": "FARMER", "position": farmer_position})
        raw_hands = farm.get("hands", source.get("hands", ()))
        if isinstance(raw_hands, Sequence) and not isinstance(raw_hands, (str, bytes)):
            for hand_index, hand in enumerate(raw_hands):
                hand_position = _position(hand)
                if hand_position is None:
                    continue
                workers.append({
                    "index": hand_index + 1,
                    "role": _get(hand, "role", "WORKER"),
                    "position": hand_position,
                })
    source.update({
        "board_size": board_size,
        "tiles": tiles,
        "animals": source.get("animals", farm.get("animals", private.get("animals", []))),
        "structures": source.get("structures", farm.get("structures", private.get("structures", []))),
        "seeds": dict(seeds) if isinstance(seeds, Mapping) else {},
        "inventory": dict(inventory) if isinstance(inventory, Mapping) else {},
        "workers": workers,
        "market": market,
    })
    return source


def _tiles(state: Any) -> list[tuple[Position, Any]]:
    source = _get(state, "tiles")
    if source is None:
        source = _get(_get(state, "farm", {}), "tiles", {})
    if isinstance(source, Mapping):
        result = []
        for raw_position, tile in source.items():
            position = _position(raw_position)
            if position is not None:
                result.append((position, tile))
        return sorted(result, key=lambda item: (item[0].y, item[0].x))
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        return []
    return [
        (Position(x, y), tile)
        for y, row in enumerate(source)
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        for x, tile in enumerate(row)
    ]


def _upper(value: Any) -> str:
    return str(value or "").upper()


def _tile_kind(tile: Any) -> str:
    if isinstance(tile, str):
        return _upper(tile)
    return _upper(_get(tile, "kind", _get(tile, "type", _get(tile, "state", ""))))


def _crop(tile: Any) -> str | None:
    crop = _get(tile, "crop")
    if crop is not None and not isinstance(crop, str):
        crop = _get(crop, "crop", _get(crop, "kind", _get(crop, "type", _get(crop, "name"))))
    if crop is None and _tile_kind(tile) in CROPS:
        crop = _tile_kind(tile)
    crop = _upper(crop)
    return crop if crop in CROPS else None


def _is_empty(tile: Any) -> bool:
    if tile is None:
        return True
    if isinstance(tile, str):
        return _upper(tile) in {"", "EMPTY", "SOIL", "TILLED"}
    return bool(_get(tile, "empty", False)) or _tile_kind(tile) in {"EMPTY", "SOIL", "TILLED"}


def _needs(tile: Any, field: str, inverse_field: str | None = None) -> bool:
    explicit = _get(tile, field)
    if explicit is not None:
        return bool(explicit)
    if inverse_field is not None:
        explicit = _get(tile, inverse_field)
        if explicit is not None:
            return not bool(explicit)
    return False


def _needs_today(tile: Any, need_field: str, today_field: str, legacy_field: str | None = None) -> bool:
    explicit = _get(tile, need_field)
    if explicit is not None:
        return bool(explicit)
    today = _get(tile, today_field)
    if today is not None:
        return not bool(today)
    return _needs(tile, need_field, legacy_field)


def _entity_state(tile: Any, entity_name: str) -> dict[str, Any] | None:
    """Merge an entity nested in a tile with tile-level lifecycle fields."""
    nested = _get(tile, entity_name)
    kind = _tile_kind(tile)
    if nested is None and kind != entity_name.upper():
        return None
    result = dict(_mapping(tile))
    if nested is not None and not isinstance(nested, str):
        result.update(_mapping(nested))
    elif isinstance(nested, str):
        result.setdefault("species", nested)
        result.setdefault("kind", nested)
    return result


def _market_section(state: Any) -> Mapping[str, Any]:
    market = _get(state, "market", {})
    return market if isinstance(market, Mapping) else {}


def _observed_prices(state: Any) -> Mapping[str, Any]:
    market = _market_section(state)
    for key in ("observed_prices", "market_prices"):
        values = _get(state, key)
        if isinstance(values, Mapping):
            return values
    values = market.get("prices")
    if isinstance(values, Mapping):
        return values
    values = _get(state, "prices")
    if isinstance(values, Mapping):
        return values
    # A flat market mapping is also an observed quote table, never curve
    # parameters.  Metadata keys are ignored by the quote lookup.
    return market


def _observed_inventory(item: str, state: Any) -> float:
    market = _market_section(state)
    values = None
    for key in ("observed_market_inventory", "market_inventory"):
        values = _get(state, key)
        if values is not None:
            break
    if values is None:
        values = market.get("inventory")
    if isinstance(values, Mapping):
        values = values.get(item, MARKET_I0)
    try:
        return float(values)
    except (TypeError, ValueError, OverflowError):
        return float(MARKET_I0)


def _observed_quote(item: str, state: Any) -> float:
    prices = _observed_prices(state)
    if item in prices:
        try:
            return max(0.0, float(prices[item]))
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        return float(market_price(item, _observed_inventory(item, state)))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _observed_sale_value(item: str, quantity: int, state: Any) -> float:
    prices = _observed_prices(state)
    try:
        quantity = max(0, int(quantity))
    except (TypeError, ValueError, OverflowError):
        quantity = 0
    if quantity == 0:
        return 0.0
    if item in prices:
        return quantity * _observed_quote(item, state)
    try:
        return float(sell_batch_value(item, quantity, _observed_inventory(item, state)))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _state_days(tile: Any, keys: tuple[str, ...], current_day: int) -> set[int]:
    for key in keys:
        value = _get(tile, key)
        if value is None:
            continue
        if isinstance(value, (set, frozenset, list, tuple, range)):
            result = set()
            for day in value:
                try:
                    result.add(int(day))
                except (TypeError, ValueError, OverflowError):
                    continue
            return result
        if isinstance(value, bool):
            return {current_day} if value else set()
    return set()


def _crop_age(tile: Any, day: int) -> int:
    explicit_age = _get(tile, "planted_age", _get(tile, "age"))
    if explicit_age is not None:
        try:
            return max(0, int(explicit_age))
        except (TypeError, ValueError, OverflowError):
            return 0
    planted_day = _get(tile, "planted_day", day)
    try:
        return max(0, day - int(planted_day))
    except (TypeError, ValueError, OverflowError):
        return 0


def _harvest_units(crop: str, tile: Any, age: int, day: int) -> float:
    recorded = _get(tile, "yield_units")
    if recorded is not None:
        try:
            return max(0.0, float(recorded))
        except (TypeError, ValueError, OverflowError):
            return 0.0
    watering_days = _state_days(tile, ("watering_days", "watered_days", "water_history", "watered_today"), day)
    if not watering_days and _get(tile, "watered") is not None:
        watering_days = _state_days(tile, ("watered",), day)
    fertilizer_days = _state_days(tile, ("fertilizer_days", "fertilized_days", "fertilizer_history", "fertilized_today"), day)
    if not fertilizer_days and _get(tile, "fertilized") is not None:
        fertilizer_days = _state_days(tile, ("fertilized",), day)
    try:
        forecast = forecast_crop(
            crop,
            start_day=day - age,
            horizon=age + 1,
            watering_days=watering_days,
            fertilizer_days=fertilizer_days,
            harvest_day=day,
        )
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, float(forecast.get("harvested_units", 0)))


def _harvest_value(crop: str, tile: Any, age: int, day: int, state: Any) -> float:
    return _harvest_units(crop, tile, age, day) * _observed_quote(crop, state)


def _target_position(value: Any) -> Position | None:
    return _position(value)


def _add(plan: list[Task], kind: str, target: Any, priority: int, deadline: int | None, value: float) -> None:
    if _target_position(target) is None and kind not in _SHED_WORK:
        return
    plan.append(Task(kind, target, priority, deadline, max(0.0, float(value))))


def _inventory(state: Any) -> Mapping[str, Any]:
    value = _get(state, "inventory", _get(_get(state, "farm", {}), "hands", {}))
    return value if isinstance(value, Mapping) else {}


def _shed_target(state: Any, board_size: int) -> Position:
    explicit = _position(_get(state, "shed_position"))
    if explicit is not None and 0 <= explicit.x < board_size and 0 <= explicit.y < board_size:
        return explicit
    valid_access_tiles = tuple(
        tile for tile in shed_access_tiles(board_size)
        if 0 <= tile.x < board_size and 0 <= tile.y < board_size
    )
    return valid_access_tiles[0] if valid_access_tiles else Position(0, 0)


def build_daily_plan(state: Any, memory: EpisodeMemory | Any = None) -> list[Task]:
    """Build a stable one-day plan from a typed or mapping-shaped state."""
    state = normalize_planner_state(state)
    memory = memory or EpisodeMemory()
    day = _day(state, memory)
    try:
        board_size = max(1, int(_get(state, "board_size", 1)))
    except (TypeError, ValueError, OverflowError):
        board_size = 1
    plan: list[Task] = []

    for position, tile in _tiles(state):
        if is_locked_tile(tile):
            continue
        kind = _tile_kind(tile)
        crop = _crop(tile)
        if kind == "WEED":
            _add(plan, "WEED", position, 80, day, 10)
        if crop:
            if _needs_today(tile, "needs_water", "watered_today", "watered"):
                _add(plan, "WATER", position, 100, day, 1)
            age = _crop_age(tile, day)
            crop_rules = CROPS[crop]
            # Non-ongoing crops have their first decay step on the day after
            # max_yield_day, but actions are accepted before that decay.  An
            # ongoing crop remains harvestable as long as actual state says it
            # still has product, including after max_yield_day.
            lifecycle_ready = age >= crop_rules["first_yield_day"]
            if lifecycle_ready:
                value = _harvest_value(crop, tile, age, day, state)
                if value > 0:
                    _add(plan, "HARVEST", position, 90, day, value)
        elif _is_empty(tile):
            seeds = _get(state, "seeds", {})
            if not isinstance(seeds, Mapping):
                seeds = {}
            available = [crop_name for crop_name in CROPS if seeds.get(crop_name, 0) and crop_name in CROPS]
            if available:
                crop_name = max(available, key=lambda item: (_observed_quote(item, state), item))
                _add(plan, "PLANT", position, 20, None, _observed_quote(crop_name, state))

        animal_entity = _entity_state(tile, "animal")
        if animal_entity is not None:
            animal_position = _position(animal_entity) or position
            species = _upper(_get(animal_entity, "species", _get(animal_entity, "animal", _get(animal_entity, "kind", ""))))
            animal_value = float(ANIMALS.get(species, {}).get("cost", 1))
            if _needs_today(animal_entity, "needs_feed", "fed_today", "fed"):
                _add(plan, "FEED", animal_position, 100, day, 1)
            if _needs_today(animal_entity, "needs_care", "cared_today", "cared"):
                _add(plan, "CARE", animal_position, 95, day, animal_value)
            if _get(animal_entity, "needs_placement", False) or _get(animal_entity, "placed") is False or _get(animal_entity, "owned") is False:
                _add(plan, "ANIMAL", animal_position, 40, None, _get(animal_entity, "value", 1))

        structure_entity = _entity_state(tile, "structure")
        if structure_entity is not None and (
            _get(structure_entity, "needs_placement", False)
            or _get(structure_entity, "built") is False
            or _get(structure_entity, "placed") is False
        ):
            _add(plan, "STRUCTURE", position, 45, None, _get(structure_entity, "value", 1))

    animals = _get(state, "animals", ()) or ()
    for animal in animals:
        position = _position(animal)
        if position is None:
            continue
        species = _upper(_get(animal, "species", _get(animal, "kind", "")))
        value = float(ANIMALS.get(species, {}).get("cost", 1))
        if _needs_today(animal, "needs_feed", "fed_today", "fed"):
            _add(plan, "FEED", position, 100, day, 1)
        if _needs_today(animal, "needs_care", "cared_today", "cared"):
            _add(plan, "CARE", position, 95, day, value)

    for structure in _get(state, "structures", ()) or ():
        if not bool(_get(structure, "built", True)):
            _add(plan, "STRUCTURE", _position(structure), 45, None, _get(structure, "value", 1))
    for animal in _get(state, "desired_animals", ()) or ():
        if not bool(_get(animal, "owned", False)):
            _add(plan, "ANIMAL", _position(animal), 40, None, _get(animal, "value", 1))

    inventory = _inventory(state)
    held = sum(float(quantity) for quantity in inventory.values() if isinstance(quantity, (int, float)) and quantity > 0)
    if held > 0:
        shed_target = _shed_target(state, board_size)
        _add(plan, "SHED", shed_target, 85, day, held)
        for item, quantity in sorted(inventory.items()):
            if item == "FERTILIZER" or not isinstance(quantity, (int, float)) or quantity <= 0:
                continue
            try:
                value = _observed_sale_value(item, int(quantity), state)
            except (KeyError, TypeError, ValueError):
                value = 0
            if value > 0:
                _add(plan, "SELL", shed_target, 75, day, value)

    return plan


def _worker_info(worker: Any, fallback_index: int) -> tuple[int, str, Position | None]:
    index = _get(worker, "index", fallback_index)
    try:
        index = int(index)
    except (TypeError, ValueError, OverflowError):
        index = fallback_index
    role = _upper(_get(worker, "role", _get(worker, "name", "WORKER")))
    return index, role, _position(_get(worker, "position", worker))


def _task_key(task: Task) -> tuple[Any, ...]:
    target = _target_position(task.target)
    coordinate = (target.x, target.y) if target is not None else repr(task.target)
    return (task.kind, coordinate)


def _task_sort_key(task: Task, day: int) -> tuple[Any, ...]:
    urgent = task.deadline is not None and task.deadline <= day
    slack = task.deadline - day if task.deadline is not None else inf
    target = _target_position(task.target)
    coordinate = (target.y, target.x) if target is not None else (inf, inf)
    return (0 if urgent else 1, slack, -task.priority, -task.value, coordinate, task.kind)


def _route_positions(start: Position | None, target: Position | None, board_size: Any) -> list[Position]:
    if start is None or target is None:
        return []
    position = start
    route = []
    for action in route_to(start, target, board_size):
        dx, dy = {"EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, 1), "NORTH": (0, -1)}[action]
        position = Position(position.x + dx, position.y + dy)
        route.append(position)
    return route


def assign_tasks(plan: Iterable[Task], workers: Iterable[Any] | None, state: Any) -> list[WorkerAssignment]:
    """Assign at most one exclusive task per worker with deterministic priorities."""
    state = normalize_planner_state(state)
    explicit_workers = list(workers) if workers is not None else []
    if not explicit_workers:
        explicit_workers = list(_get(state, "workers", ()) or ())
    day = _day(state, EpisodeMemory())
    try:
        board_size = max(1, int(_get(state, "board_size", 1)))
    except (TypeError, ValueError, OverflowError):
        board_size = 1
    unique: dict[tuple[Any, ...], Task] = {}
    for task in plan:
        if not isinstance(task, Task):
            continue
        key = _task_key(task)
        current = unique.get(key)
        if current is None or _task_sort_key(task, day) < _task_sort_key(current, day):
            unique[key] = task
    tasks = sorted(unique.values(), key=lambda task: _task_sort_key(task, day))
    infos = sorted((_worker_info(worker, index) for index, worker in enumerate(explicit_workers)), key=lambda item: (item[0], item[2].y if item[2] else inf, item[2].x if item[2] else inf))
    if not infos:
        return []

    farmer = next((info for info in infos if info[1] == "FARMER"), None)
    logistics_pending = any(task.kind in _SHED_WORK for task in tasks)
    helper_exists = any(info[1] != "FARMER" for info in infos)
    reserved_basic = next((info for info in infos if info[1] != "FARMER"), infos[0])
    available = {info[0] for info in infos}
    assignments: list[WorkerAssignment] = []
    remaining = list(tasks)

    def choose(task: Task) -> tuple[int, str, Position | None] | None:
        candidates = [info for info in infos if info[0] in available]
        non_farmer_available = any(info[0] in available and info[1] != "FARMER" for info in infos)
        reserve_farmer = (
            logistics_pending
            and farmer is not None
            and helper_exists
            and task.kind not in _SHED_WORK
            and (task.kind not in _BASIC_NEEDS or non_farmer_available)
        )
        if reserve_farmer:
            candidates = [info for info in candidates if info[0] != farmer[0]]
        if task.kind in _BASIC_NEEDS and reserved_basic[0] in available:
            candidates = [info for info in candidates if info[0] == reserved_basic[0]] or candidates
        elif task.kind not in _BASIC_NEEDS and reserved_basic[0] in available and any(item.kind in _BASIC_NEEDS for item in remaining):
            candidates = [info for info in candidates if info[0] != reserved_basic[0]]
        target = _target_position(task.target)
        if not candidates:
            return None
        return min(candidates, key=lambda info: (
            distance(info[2], target) if info[2] is not None and target is not None else inf,
            info[0],
            info[2].y if info[2] is not None else inf,
            info[2].x if info[2] is not None else inf,
        ))

    for task in tasks:
        selected = choose(task)
        if selected is None:
            continue
        available.remove(selected[0])
        remaining.remove(task)
        assignments.append(WorkerAssignment(
            worker_index=selected[0],
            task=task,
            route=_route_positions(selected[2], _target_position(task.target), board_size),
        ))
    return assignments
