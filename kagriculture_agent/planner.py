"""Small, deterministic daily task planner for the Kaggriculture agent."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import inf
from typing import Any

from .constants import ANIMALS, CROPS
from .economics import forecast_crop, market_price, sell_batch_value
from .observation import is_tile_actionable, shed_access_tiles
from .routing import distance, route_to
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


def _position(value: Any) -> Position | None:
    if isinstance(value, Position):
        return value
    if isinstance(value, Mapping):
        if "position" in value:
            return _position(value["position"])
        value = (value.get("x"), value.get("y"))
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return Position(int(value[0]), int(value[1]))
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _day(state: Any, memory: EpisodeMemory | Any) -> int:
    value = _get(state, "day", _get(memory, "last_day", 0))
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


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


def _tile_value(crop: str, state: Any) -> float:
    prices = _get(state, "prices", _get(state, "market", {}))
    try:
        return float(market_price(crop, _get(_get(state, "inventory", {}), crop, 0), prices if isinstance(prices, Mapping) else None))
    except (KeyError, TypeError, ValueError):
        return float(CROPS[crop].get("max_yield", 1))


def _harvest_value(crop: str, age: int, state: Any) -> float:
    try:
        forecast = forecast_crop(crop, horizon=age + 1, watering_days=set(range(age + 1)), harvest_day=age)
        units = forecast.get("harvested_units", 0)
    except (KeyError, TypeError, ValueError):
        units = 0
    return max(0.0, float(units) * _tile_value(crop, state))


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
    if explicit is not None:
        return explicit
    return shed_access_tiles(board_size)[0]


def build_daily_plan(state: Any, memory: EpisodeMemory | Any = None) -> list[Task]:
    """Build a stable one-day plan from a typed or mapping-shaped state."""
    memory = memory or EpisodeMemory()
    day = _day(state, memory)
    try:
        board_size = max(1, int(_get(state, "board_size", 1)))
    except (TypeError, ValueError, OverflowError):
        board_size = 1
    plan: list[Task] = []

    for position, tile in _tiles(state):
        if not is_tile_actionable(tile):
            continue
        kind = _tile_kind(tile)
        crop = _crop(tile)
        if kind == "WEED":
            _add(plan, "WEED", position, 80, day, 10)
        if crop:
            if _needs(tile, "needs_water", "watered"):
                _add(plan, "WATER", position, 100, day, 1)
            planted_day = _get(tile, "planted_day", day)
            try:
                age = day - int(planted_day)
            except (TypeError, ValueError, OverflowError):
                age = 0
            crop_rules = CROPS[crop]
            if crop_rules["first_yield_day"] <= age <= crop_rules["max_yield_day"]:
                value = _harvest_value(crop, age, state)
                if value > 0:
                    _add(plan, "HARVEST", position, 90, day, value)
        elif _is_empty(tile):
            seeds = _get(state, "seeds", {})
            if not isinstance(seeds, Mapping):
                seeds = {}
            available = [crop_name for crop_name in CROPS if seeds.get(crop_name, 0) and crop_name in CROPS]
            if available:
                crop_name = max(available, key=lambda item: (_tile_value(item, state), item))
                _add(plan, "PLANT", position, 20, None, _tile_value(crop_name, state))

    for animal in _get(state, "animals", ()) or ():
        position = _position(animal)
        if position is None:
            continue
        species = _upper(_get(animal, "species", _get(animal, "kind", "")))
        value = float(ANIMALS.get(species, {}).get("cost", 1))
        if _needs(animal, "needs_feed", "fed"):
            _add(plan, "FEED", position, 100, day, 1)
        if _needs(animal, "needs_care", "cared"):
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
                value = sell_batch_value(item, int(quantity), 0)
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


def assign_tasks(plan: Iterable[Task], workers: Iterable[Any], state: Any) -> list[WorkerAssignment]:
    """Assign at most one exclusive task per worker with deterministic priorities."""
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
    infos = sorted((_worker_info(worker, index) for index, worker in enumerate(workers)), key=lambda item: (item[0], item[2].y if item[2] else inf, item[2].x if item[2] else inf))
    if not infos:
        return []

    farmer = next((info for info in infos if info[1] == "FARMER"), None)
    logistics_pending = any(task.kind in _SHED_WORK for task in tasks)
    reserved_basic = next((info for info in infos if info[1] != "FARMER"), infos[0])
    available = {info[0] for info in infos}
    assignments: list[WorkerAssignment] = []
    remaining = list(tasks)

    def choose(task: Task) -> tuple[int, str, Position | None] | None:
        candidates = [info for info in infos if info[0] in available]
        if logistics_pending and farmer is not None and task.kind not in _SHED_WORK:
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
