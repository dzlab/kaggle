"""Deterministic, legality-first Kaggriculture policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .constants import ANIMALS, CROPS, LAND_PRICES, PRODUCTS, max_market_orders, season_days, shed_capacity as DEFAULT_SHED_CAPACITY
from .economics import feed_reserve, market_price, market_regime
from .memory import PolicyMemory
from .observation import is_shed_adjacent, parse_observation as _parse_observation, shed_access_tiles
from .planner import assign_tasks, build_autonomous_macro_plan, build_daily_plan, normalize_planner_state
from .routing import is_locked_tile, normalize_position, next_move
from .types import Position, Task, WorkerAssignment


PASS = "PASS"
_SALEABLE_PRODUCTS = frozenset(item for item in PRODUCTS if item != "FERTILIZER")
_MOVES = frozenset({"NORTH", "SOUTH", "EAST", "WEST", PASS})
_SIMPLE_ACTIONS = frozenset({
    "DROP", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
})
_TILE_KINDS = frozenset({"PLANT", "CROP", "ANIMAL", "COOP", "PASTURE"})
_LOCKED_TILE_ACTIONS = frozenset({
    "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED",
    "COLLECT_FERTILIZER", "CARE", "DIG", "WEED", "BUILD_COOP",
    "BUILD_PASTURE", "STRUCTURE", "ANIMAL",
})


def parse_observation(obs: Any) -> dict[str, Any]:
    """Expose the canonical nested observation used by planner and policy."""
    return _parse_observation(obs)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        result = vars(value)
    except TypeError:
        return {}
    return result if isinstance(result, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if isfinite(number) else default


def _whole(value: Any, default: int = 0) -> int:
    return max(0, int(_number(value, default)))


def _position(value: Any) -> Position | None:
    return normalize_position(value)


def _state_for_planner(state: Any) -> dict[str, Any]:
    return normalize_planner_state(state)


def _tiles(state: Any) -> Any:
    tiles = _get(state, "tiles")
    if tiles is None:
        tiles = _get(_get(state, "farm", {}), "tiles", [])
    return tiles


def _iter_tiles(state: Any):
    tiles = _tiles(state)
    if isinstance(tiles, Mapping):
        for raw_position, tile in tiles.items():
            position = _position(raw_position)
            if position is not None:
                yield position, tile
    elif isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
        for y, row in enumerate(tiles):
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                for x, tile in enumerate(row):
                    yield Position(x, y), tile


def _tile_at(state: Any, position: Any) -> Any:
    position = _position(position)
    if position is None:
        return None
    tiles = _tiles(state)
    if isinstance(tiles, Mapping):
        return next((tile for raw, tile in tiles.items() if _position(raw) == position), None)
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return None
    if not 0 <= position.y < len(tiles):
        return None
    row = tiles[position.y]
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        return None
    return row[position.x] if 0 <= position.x < len(row) else None


def _tile_kind(tile: Any) -> str:
    if isinstance(tile, str):
        return tile.upper()
    nested = _get(tile, "animal", _get(tile, "structure"))
    return str(_get(tile, "kind", _get(tile, "type", _get(tile, "state", _get(nested, "kind", "")))) or "").upper()


def _structure_kind(tile: Any) -> str:
    kind = _tile_kind(tile)
    nested = _mapping(_get(tile, "structure"))
    nested_kind = str(_get(nested, "kind", _get(nested, "type", ""))).upper()
    if nested_kind in {"COOP", "PASTURE"}:
        kind = nested_kind
    return kind


def _crop(tile: Any) -> str | None:
    raw = _get(tile, "crop")
    if isinstance(raw, Mapping):
        raw = _get(raw, "crop", _get(raw, "kind", _get(raw, "name")))
    raw = raw or (_tile_kind(tile) if _tile_kind(tile) in CROPS else "")
    crop = str(raw).upper()
    return crop if crop in CROPS else None


def _animal(tile: Any) -> Mapping[str, Any] | None:
    if not isinstance(tile, Mapping):
        return None
    raw = _get(tile, "animal")
    if raw is None:
        return None
    merged = dict(tile)
    if isinstance(raw, Mapping):
        merged.update(raw)
    else:
        merged["species"] = raw
    species = str(_get(merged, "species", _get(merged, "animal", _get(merged, "kind", ""))) or "").upper()
    if species not in ANIMALS:
        return None
    merged["species"] = species
    return merged


def _worker_records(state: Any) -> list[dict[str, Any]]:
    normalized = _state_for_planner(state)
    records = []
    for fallback, worker in enumerate(normalized.get("workers", ()) or ()):
        index = _whole(_get(worker, "index", fallback))
        records.append({
            "index": index,
            "role": str(_get(worker, "role", "WORKER")).upper(),
            "position": _position(_get(worker, "position", worker)),
            "raw": worker,
        })
    return records


def _private(state: Any) -> Mapping[str, Any]:
    return _mapping(_get(state, "private", {}))


def _shed(state: Any) -> Mapping[str, Any]:
    private = _private(state)
    shed = private.get("shed", _get(state, "shed", _get(state, "inventory", {})))
    return shed if isinstance(shed, Mapping) else {}


def _inventories(state: Any) -> Sequence[Any]:
    private = _private(state)
    inventories = private.get("inventories", _get(state, "inventories", ()))
    return inventories if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)) else ()


def _counts(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(key).upper(): _whole(quantity) for key, quantity in value.items() if _whole(quantity) > 0}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: dict[str, int] = {}
        for item in value:
            if isinstance(item, str):
                key = item.upper()
                result[key] = result.get(key, 0) + 1
            elif isinstance(item, Mapping):
                key = _get(item, "item", _get(item, "kind", _get(item, "name")))
                if key:
                    key = str(key).upper()
                    result[key] = result.get(key, 0) + _whole(_get(item, "quantity", 1), 1)
        return result
    return {}


def _inventory_for_worker(state: Any, worker_index: int) -> dict[str, int]:
    inventories = _inventories(state)
    if 0 <= worker_index < len(inventories):
        return _counts(inventories[worker_index])
    for worker in _worker_records(state):
        if worker["index"] == worker_index:
            return _counts(_get(worker["raw"], "inventory", ()))
    return {}


def _cash(state: Any) -> float:
    farm = _mapping(_get(state, "farm", {}))
    return max(0.0, _number(_get(state, "cash", farm.get("money", 0))))


def _prices(state: Any) -> Mapping[str, Any]:
    market = _mapping(_get(state, "market", {}))
    prices = market.get("prices", _get(state, "prices", {}))
    return prices if isinstance(prices, Mapping) else {}


def _market_inventory(state: Any) -> Mapping[str, Any]:
    market = _mapping(_get(state, "market", {}))
    inventory = market.get("inventory", {})
    return inventory if isinstance(inventory, Mapping) else {}


def _market_params(state: Any) -> Mapping[str, Any] | None:
    market = _mapping(_get(state, "market", {}))
    params = market.get("params", market.get("price_params"))
    if params is None:
        params = _get(state, "market_params", _get(state, "price_params"))
    return params if isinstance(params, Mapping) else None


def _quote(item: str, state: Any, *, seed: bool = False) -> float:
    if seed:
        return _number(CROPS.get(item, {}).get("seed", 0))
    prices = _prices(state)
    if item in prices:
        return max(1.0, _number(prices[item], 1.0))
    try:
        return float(market_price(item, _number(_market_inventory(state).get(item, 10_000), 10_000), _market_params(state)))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _buy_product_quote(item: str, state: Any, unit_offset: int = 0) -> float:
    inventory = _number(_market_inventory(state).get(item, 10_000), 10_000)
    try:
        return float(market_price(item, inventory - 1 - unit_offset, _market_params(state)))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _animals(state: Any) -> dict[str, int]:
    raw_counts: dict[str, int] = {}
    raw_animals = _get(state, "animals", _get(_get(state, "farm", {}), "animals", ())) or ()
    if isinstance(raw_animals, Mapping):
        raw_animals = [raw_animals]
    values = raw_animals if isinstance(raw_animals, Sequence) and not isinstance(raw_animals, (str, bytes)) else ()
    for animal in values:
        species = str(_get(animal, "species", _get(animal, "animal", _get(animal, "kind", ""))) or "").upper()
        if species in ANIMALS and _get(animal, "owned", True) is not False:
            raw_counts[species] = raw_counts.get(species, 0) + 1
    tile_counts: dict[str, int] = {}
    for _, tile in _iter_tiles(state):
        entity = _animal(tile)
        if entity is None:
            continue
        species = str(_get(entity, "species", _get(entity, "animal", _get(entity, "kind", ""))) or "").upper()
        if species in ANIMALS and _get(entity, "placed", True) is not False:
            tile_counts[species] = tile_counts.get(species, 0) + 1
    return tile_counts or raw_counts


def _is_adjacent_to_shed(state: Any, position: Any) -> bool:
    position = _position(position)
    if position is None:
        return False
    board_size = _get(state, "board_size")
    if board_size is None:
        points = [position for position, _ in _iter_tiles(state)]
        board_size = max((max(point.x, point.y) + 1 for point in points), default=1)
    return is_shed_adjacent(position, _whole(board_size, 1))


def _shed_access_target(state: Any, position: Position) -> Position | None:
    board_size = _get(state, "board_size")
    if board_size is None:
        tiles = _tiles(state)
        rows = len(tiles) if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) else 0
        width = max((len(row) for row in tiles if isinstance(row, Sequence) and not isinstance(row, (str, bytes))), default=0) if rows else 0
        board_size = max(rows, width, 1)
    size = _whole(board_size, 1)
    candidates = [
        candidate for candidate in shed_access_tiles(size)
        if 0 <= candidate.x < size and 0 <= candidate.y < size
    ]
    return min(candidates, key=lambda candidate: (abs(candidate.x - position.x) + abs(candidate.y - position.y), candidate.y, candidate.x), default=None)


def _task_target(task: Any) -> Any:
    return _get(task, "target")


def _required_worker_item(task: Any, state: Any = None) -> str | None:
    kind = str(_get(task, "kind", "")).upper()
    if kind == "FEED":
        return "WHEAT"
    if kind == "FERTILIZE":
        return "FERTILIZER"
    if kind in {"PLACE", "ANIMAL"}:
        item = _task_target(task) if isinstance(_task_target(task), str) else _get(task, "item", _get(task, "animal", _get(task, "species")))
        if item is None and state is not None:
            target = _position(_task_target(task))
            for candidate in _get(state, "desired_animals", _get(state, "animals", ())) or ():
                if target is not None and _position(candidate) == target:
                    item = _get(candidate, "species", _get(candidate, "animal", _get(candidate, "kind")))
                    break
        item = str(item or "").upper()
        return item if item in PRODUCTS or item in ANIMALS else None
    return None


def _explicit_market_intents(*sources: Any) -> list[Any]:
    keys = ("market_intents", "macro_market_intents", "approved_market_intents", "market_plan")
    for source in sources:
        for section in (source, _get(source, "farm", {}), _get(source, "private", {}), _get(source, "market", {})):
            for key in keys:
                value = _get(section, key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    return list(value)
    return []


def _requested_quantity(item: Any, default: int = 1) -> int:
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return _whole(item[2], default)
    for key in ("quantity", "amount", "count", "units", "n"):
        value = _get(item, key)
        if value is not None:
            return _whole(value, default)
    return default


def _intent(item: Any) -> tuple[str, str | None, int] | None:
    if isinstance(item, (list, tuple)):
        if not item:
            return None
        kind = str(item[0]).upper()
        if kind in {"HIRE", "BUY_LAND", "SELL_ALL"} and len(item) == 1:
            return kind, None, 1
        if len(item) < 2:
            return None
        name = str(item[1]).upper()
        if kind == "SELL_ALL":
            return None
        return kind, name, _requested_quantity(item)
    kind = str(_get(item, "kind", "")).upper()
    if kind not in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL", "SELL_ALL", "HIRE", "BUY_LAND"}:
        return None
    name = _get(item, "item", _get(item, "crop", _get(item, "animal", _get(item, "product"))))
    if name is None and isinstance(_get(item, "target"), str):
        name = _get(item, "target")
    if kind == "SELL" and bool(_get(item, "sell_all", False)):
        return "SELL_ALL", None, _requested_quantity(item)
    if kind == "SELL" and name is None:
        return None
    if kind == "SELL_ALL" and name is not None:
        return None
    return kind, str(name).upper() if name is not None else None, _requested_quantity(item)


def _approved_intents(plan: Any) -> list[tuple[str, str | None, int]]:
    values = plan if isinstance(plan, (list, tuple)) else [plan]
    result = []
    for item in values:
        parsed = _intent(item)
        if parsed is not None and parsed[2] > 0:
            result.append(parsed)
    return result


def _purchase_cost(kind: str, item: str | None, state: Any) -> float:
    if kind == "BUY_SEED":
        return _quote(item or "", state, seed=True)
    if kind == "BUY_ANIMAL":
        return _number(ANIMALS.get(item or "", {}).get("cost", 0))
    if kind == "BUY_PRODUCT":
        return _buy_product_quote(item or "", state)
    if kind == "BUY_LAND":
        unlocked = _get(_mapping(_get(state, "farm", {})), "unlocked_quadrants", _get(state, "unlocked_quadrants", ["NW"]))
        if not isinstance(unlocked, Sequence) or isinstance(unlocked, (str, bytes)):
            unlocked = ["NW"]
        next_index = len(unlocked) - 1
        return _number(LAND_PRICES[next_index]) if 0 <= next_index < len(LAND_PRICES) else 0.0
    if kind == "HIRE":
        farm = _mapping(_get(state, "farm", {}))
        multiplier = _number(_get(state, "farm_hand_cost_mult", _get(state, "farmHandCostMult", farm.get("farm_hand_cost_mult", 1))), 1)
        n = _whole(farm.get("hires_today", _get(state, "hires_today", 0)))
        first, second = 1, 1
        for _ in range(n):
            first, second = second, first + second
        return max(1.0, multiplier * first)
    return 0.0


def _days_left(state: Any) -> int:
    return max(0, season_days - _whole(_get(state, "day", 0)))


def build_market_orders(state: Any, plan: Any) -> list[list[Any]]:
    """Turn approved intents into bounded, affordable, legal market orders."""
    state = _state_for_planner(state)
    day, hour = _whole(_get(state, "day", 0)), _whole(_get(state, "hour", 0))
    shed = {str(item).upper(): _whole(quantity) for item, quantity in _shed(state).items()}
    cash = _cash(state)
    orders: list[list[Any]] = []
    spend = 0.0
    product_buys: dict[str, int] = {}
    animal_buys = 0
    shed_room = max(0, DEFAULT_SHED_CAPACITY - sum(shed.values()))
    intents = _approved_intents(plan)
    # The engine records the action selected from the preceding observation;
    # hour 22 is therefore the last reliably executable liquidation window
    # for a 30-day episode, with hour 23 retained for direct callers.
    final_turn = day >= season_days - 1 and hour >= 22

    # Protect the remaining wheat needed by living animals before selling.
    carried_wheat = sum(_counts(inventory).get("WHEAT", 0) for inventory in _inventories(state))
    total_feed_wheat = feed_reserve(_animals(state), _days_left(state), 0)
    existing_wheat = shed.get("WHEAT", 0) + carried_wheat
    required_wheat = max(0, total_feed_wheat - existing_wheat)
    shed_wheat_reserve = max(0, total_feed_wheat - carried_wheat)
    wheat_price = _buy_product_quote("WHEAT", state)
    sell_intents = [intent for intent in intents if intent[0] in {"SELL", "SELL_ALL"}]

    # The final action has no tomorrow to feed, and all remaining shed goods
    # should be liquidated rather than funding another purchase.
    if final_turn:
        total_feed_wheat = required_wheat = shed_wheat_reserve = 0

    # Wheat reserved for feed is purchased before discretionary approvals.
    if not final_turn and required_wheat and wheat_price > 0 and shed_room:
        affordable = 0
        purchase_cost = 0.0
        for offset in range(required_wheat):
            if affordable >= shed_room:
                break
            unit_cost = _buy_product_quote("WHEAT", state, offset)
            if unit_cost <= 0 or spend + purchase_cost + unit_cost > cash:
                break
            affordable += 1
            purchase_cost += unit_cost
        if affordable:
            orders.append(["BUY_PRODUCT", "WHEAT", affordable])
            spend += purchase_cost
            product_buys["WHEAT"] = affordable

    for kind, item, requested in intents:
        if final_turn or len(orders) >= max_market_orders or kind in {"SELL", "SELL_ALL"}:
            continue
        if kind == "BUY_SEED" and item not in CROPS:
            continue
        if kind == "BUY_ANIMAL" and item not in ANIMALS:
            continue
        if kind == "BUY_PRODUCT" and item not in {"WHEAT", "FERTILIZER"}:
            continue
        if kind == "BUY_LAND" and item is not None:
            continue
        if kind == "HIRE" and item is not None:
            continue
        quantity = 1 if kind in {"HIRE", "BUY_LAND"} else requested
        if kind == "BUY_PRODUCT":
            affordable, purchase_cost = 0, 0.0
            already_bought = product_buys.get(item or "", 0)
            for offset in range(quantity):
                if sum(product_buys.values()) + animal_buys + affordable >= shed_room:
                    break
                unit_cost = _buy_product_quote(item or "", state, already_bought + offset)
                if unit_cost <= 0 or spend + purchase_cost + unit_cost > cash:
                    break
                affordable += 1
                purchase_cost += unit_cost
            unit_cost = purchase_cost
        else:
            unit_cost = _purchase_cost(kind, item, state)
            room = shed_room - sum(product_buys.values()) - animal_buys if kind == "BUY_ANIMAL" else quantity
            affordable = min(quantity, max(0, room), int(max(0.0, cash - spend) // unit_cost)) if unit_cost > 0 else 0
        if affordable <= 0:
            continue
        order = [kind] if kind in {"HIRE", "BUY_LAND"} else [kind, item, affordable]
        orders.append(order)
        spend += unit_cost if kind == "BUY_PRODUCT" else affordable * unit_cost
        if kind == "BUY_PRODUCT":
            product_buys[item or ""] = product_buys.get(item or "", 0) + affordable
        elif kind == "BUY_ANIMAL":
            animal_buys += affordable

    if final_turn:
        sale_items = [(item, quantity) for item, quantity in shed.items() if item in PRODUCTS]
    elif sell_intents:
        requested: dict[str, int] = {}
        sell_all = False
        for kind, item, quantity in sell_intents:
            if kind == "SELL_ALL":
                sell_all = True
            elif item in PRODUCTS:
                requested[item] = requested.get(item, 0) + quantity
        if sell_all:
            sale_items = [(item, quantity) for item, quantity in shed.items() if item in _SALEABLE_PRODUCTS]
            included = {item for item, _ in sale_items}
            sale_items.extend((item, quantity) for item, quantity in requested.items()
                              if item in shed and item not in included)
        else:
            sale_items = [(item, requested[item]) for item in requested if item in shed]
    else:
        sale_items = []
    for item, quantity in sorted(sale_items):
        quantity = min(_whole(quantity), shed.get(item, 0))
        if item == "WHEAT":
            quantity = min(quantity, max(0, shed.get(item, 0) - shed_wheat_reserve))
        if quantity > 0 and len(orders) < max_market_orders:
            orders.append(["SELL", item, quantity])
    return orders[:max_market_orders]


def _structure_action(state: Any, target: Position) -> str | None:
    tile = _tile_at(state, target)
    if is_locked_tile(tile):
        return None
    kind = _tile_kind(tile)
    structure = _mapping(_get(tile, "structure", tile if kind in {"COOP", "PASTURE", "STRUCTURE"} else {}))
    structure_kind = str(_get(structure, "kind", _get(structure, "type", kind))).upper()
    if _get(structure, "built", True) is False or _get(tile, "built", True) is False:
        return "BUILD_PASTURE" if structure_kind == "PASTURE" else "BUILD_COOP"
    for candidate in _get(state, "structures", ()) or ():
        if _position(candidate) == target and not bool(_get(candidate, "built", True)):
            structure_kind = str(_get(candidate, "kind", "COOP")).upper()
            return "BUILD_PASTURE" if structure_kind == "PASTURE" else "BUILD_COOP"
    return None


def _task_action(state: Any, worker_index: int, task: Task, position: Position) -> str:
    kind = str(_get(task, "kind", "")).upper()
    target = _task_target(task)
    tile = _tile_at(state, position)
    inventory = _inventory_for_worker(state, worker_index)
    shed = _shed(state)
    if kind in _LOCKED_TILE_ACTIONS and is_locked_tile(tile):
        shed_placement = (
            kind == "PLACE"
            and _is_adjacent_to_shed(state, position)
            and _structure_kind(tile) not in {"COOP", "PASTURE"}
        )
        if not shed_placement:
            return PASS
    if kind in {"PICKUP", "PLACE"}:
        item = target if isinstance(target, str) else _get(task, "item")
        item = str(item or "").upper()
        if item not in PRODUCTS and item not in ANIMALS:
            return PASS
        if kind == "PICKUP":
            if not _is_adjacent_to_shed(state, position) or _whole(shed.get(item)) <= 0:
                return PASS
            return f"PICKUP {item} 1"
        tile_kind = _tile_kind(tile)
        if tile_kind in {"COOP", "PASTURE", "STRUCTURE"}:
            animal = _animal(tile)
            if item not in ANIMALS or ANIMALS[item]["structure"] != _structure_kind(tile) or animal is not None or (isinstance(tile, Mapping) and "animal" in tile):
                return PASS
        elif not _is_adjacent_to_shed(state, position):
            return PASS
        if item not in ANIMALS and inventory.get(item, 0) <= 0:
            return PASS
        if inventory.get(item, 0) <= 0:
            return PASS
        return f"PLACE {item} 1"
    if kind in {"SHED", "SELL", "SELL_ALL", "DROP"}:
        return "DROP" if _is_adjacent_to_shed(state, position) and inventory else PASS
    if kind == "STRUCTURE":
        return _structure_action(state, position) or PASS
    if kind == "ANIMAL":
        item = str(_get(task, "animal", _get(task, "species", ""))).upper()
        if not item:
            for candidate in _get(state, "desired_animals", ()) or ():
                if _position(candidate) == position:
                    item = str(_get(candidate, "species", _get(candidate, "animal", ""))).upper()
                    break
        if not item:
            structure_kind = _structure_kind(tile)
            item = next((candidate for candidate in ANIMALS
                         if ANIMALS[candidate]["structure"] == structure_kind
                         and inventory.get(candidate, 0) > 0), "")
        if item in ANIMALS and ANIMALS[item]["structure"] == _structure_kind(tile) and _animal(tile) is None and not (isinstance(tile, Mapping) and "animal" in tile):
            return f"PLACE {item} 1" if inventory.get(item, 0) else PASS
        return PASS
    if kind == "PLANT":
        crop = str(_get(task, "crop", _get(task, "item", ""))).upper()
        if not crop:
            crop = str(target).upper() if isinstance(target, str) else "WHEAT"
        seeds = _mapping(_get(state, "seeds", _private(state).get("seeds", {})))
        if crop in CROPS and _crop(tile) is None and not is_locked_tile(tile) and (tile is None or _tile_kind(tile) in {"", "EMPTY", "SOIL", "TILLED"}) and _whole(seeds.get(crop)) > 0:
            return f"PLANT {crop}"
        return PASS
    if kind == "WATER":
        return "WATER" if _is_engine_plant_tile(tile) and not is_locked_tile(tile) and not bool(_get(tile, "watered_today", _get(tile, "watered", False))) else PASS
    if kind == "HARVEST":
        animal = _animal(tile)
        return "HARVEST" if not is_locked_tile(tile) and (
            (_crop(tile) and _whole(_get(tile, "yield_units")) > 0) or
            (animal is not None and _whole(_get(animal, "yield_units")) > 0)
        ) else PASS
    if kind == "FERTILIZE":
        return "FERTILIZE" if _is_engine_plant_tile(tile) and inventory.get("FERTILIZER", 0) > 0 else PASS
    if kind in {"FEED", "CARE", "COLLECT_FERTILIZER"}:
        animal = _animal(tile)
        if animal is None or is_locked_tile(tile):
            return PASS
        if kind == "FEED" and bool(_get(animal, "fed_today", False)):
            return PASS
        if kind == "CARE" and bool(_get(animal, "cared_today", False)):
            return PASS
        if kind == "FEED" and inventory.get("WHEAT", 0) <= 0:
            return PASS
        if kind == "COLLECT_FERTILIZER" and not bool(_get(animal, "fertilizer_available", False)):
            return PASS
        return kind
    if kind == "WEED":
        return "DIG" if _tile_kind(tile) == "WEED" else PASS
    if kind == "DIG":
        return "DIG" if tile is not None and not is_locked_tile(tile) and not (isinstance(tile, Mapping) and "animal" in tile) and _tile_kind(tile) in _TILE_KINDS | {"WEED", "STRUCTURE"} else PASS
    if kind in {"BUILD_COOP", "BUILD_PASTURE"}:
        return kind if tile is None else PASS
    return PASS


def _unit_command(command: str) -> list[Any]:
    """Convert an internal command string to the engine's list shape."""
    if not isinstance(command, str) or not command:
        return [PASS]
    parts = command.split()
    if len(parts) == 3 and parts[2].lstrip("+").isdigit():
        return [parts[0], parts[1], int(parts[2])]
    return parts


def _is_engine_plant_tile(tile: Any) -> bool:
    return isinstance(tile, Mapping) and _tile_kind(tile) == "PLANT" and _crop(tile) is not None


def _fetch_required_item(state: Any, worker_index: int, task: Any, current: Position) -> str | None:
    item = _required_worker_item(task, state)
    if item is None:
        return None
    inventory = _inventory_for_worker(state, worker_index)
    if inventory.get(item, 0) > 0 or _whole(_shed(state).get(item)) <= 0:
        return None
    access = _shed_access_target(state, current)
    if access is None:
        return None
    if current != access:
        return next_move(current, access)
    return f"PICKUP {item} 1"


def _drop_carried_goods(state: Any, worker_index: int, task: Any, current: Position) -> str | None:
    inventory = _inventory_for_worker(state, worker_index)
    if not any(inventory.get(item, 0) > 0 for item in PRODUCTS):
        return None
    required = _required_worker_item(task, state)
    if required is not None and inventory.get(required, 0) > 0:
        return None
    access = _shed_access_target(state, current)
    if access is None:
        return None
    return "DROP" if current == access else next_move(current, access)


def worker_action(worker_index: int, state: Any, assignment: WorkerAssignment | Task | None) -> list[Any]:
    """Emit one safe command for a worker, falling back to ``PASS``."""
    if assignment is None:
        return [PASS]
    task = _get(assignment, "task", assignment)
    worker = next((worker for worker in _worker_records(state) if worker["index"] == worker_index), None)
    if worker is None or worker["position"] is None:
        return [PASS]
    current = worker["position"]
    target = _position(_task_target(task))
    kind = str(_get(task, "kind", "")).upper()
    if target is None and kind in {"PICKUP", "PLACE"}:
        target = current
    if target is None:
        return [PASS]
    board_size = _whole(_get(state, "board_size", max(len(_tiles(state)), 1)), 1)
    fetched = _fetch_required_item(state, worker_index, task, current)
    if fetched is not None:
        return _unit_command(fetched)
    if current != target:
        if not all(0 <= point.x < board_size and 0 <= point.y < board_size for point in (current, target)):
            return [PASS]
        return _unit_command(next_move(current, target))
    return _unit_command(_task_action(state, worker_index, task, current))


def _assignment_valid(state: Any, assignment: WorkerAssignment) -> bool:
    target = _position(_task_target(assignment.task))
    kind = str(_get(assignment.task, "kind", "")).upper()
    worker_index = _whole(_get(assignment, "worker_index", 0))
    if target is None:
        if kind not in {"PICKUP", "PLACE"}:
            return False
        worker = next((worker for worker in _worker_records(state)
                       if worker["index"] == worker_index), None)
        if worker is None or worker["position"] is None:
            return False
        target = worker["position"]
    if kind == "SHED":
        return any(quantity > 0 for quantity in _inventory_for_worker(state, worker_index).values())
    if kind in {"SELL", "SELL_ALL"}:
        return any(_whole(_shed(state).get(item)) > 0 for item in _SALEABLE_PRODUCTS)
    if kind in {"FEED", "CARE"}:
        animal = _animal(_tile_at(state, target))
        completed_field = "fed_today" if kind == "FEED" else "cared_today"
        if animal is None or bool(_get(animal, completed_field, False)):
            return False
    return _task_action(state, worker_index, assignment.task, target) != PASS


class Policy:
    """Stateful deterministic policy with reset-safe episode memory."""

    def __init__(self) -> None:
        self.memory = PolicyMemory()

    def _replan(self, state: Mapping[str, Any], regime: Mapping[str, str],
                macro: Mapping[str, Any] | None = None) -> list[WorkerAssignment]:
        normalized = _state_for_planner(state)
        plan = build_daily_plan(normalized, self.memory)
        macro = macro or build_autonomous_macro_plan(normalized, self.memory)
        plan.extend(task for task in macro.get("tasks", ()) if isinstance(task, Task))
        assignments = assign_tasks(plan, normalized.get("workers", ()), normalized)
        self.memory.assignments = assignments
        self.memory.market_regime = dict(regime)
        self.memory.diagnostics["plan_size"] = len(plan)
        self.memory.diagnostics["portfolio"] = dict(macro.get("portfolio", {}))
        self.memory.diagnostics["scenario_count"] = macro.get("scenario_count", 0)
        return assignments

    def act(self, obs: Any) -> dict[str, Any]:
        state = parse_observation(obs)
        regime = market_regime(_prices(state), _market_inventory(state))
        shops = _get(_get(state, "town", {}), "unlocked_shops", ())
        regime["shops"] = "|".join(str(shop) for shop in shops) if isinstance(shops, Sequence) and not isinstance(shops, (str, bytes)) else ""
        macro = build_autonomous_macro_plan(state, self.memory)
        reset = self.memory.observe_time(_get(state, "day"), _get(state, "hour"))
        workers = _worker_records(state)
        hour_zero = _whole(_get(state, "hour")) == 0
        regime_changed = bool(self.memory.market_regime) and dict(regime) != self.memory.market_regime
        assignments_valid = all(_assignment_valid(state, assignment) for assignment in self.memory.assignments)
        if reset or hour_zero or regime_changed or not self.memory.assignments or not assignments_valid:
            assignments = self._replan(state, regime, macro)
        else:
            assignments = self.memory.assignments
        by_worker = {assignment.worker_index: assignment for assignment in assignments}
        commands = {worker["index"]: worker_action(worker["index"], state, by_worker.get(worker["index"])) for worker in workers}
        for worker in workers:
            drop = _drop_carried_goods(state, worker["index"], by_worker.get(worker["index"]), worker["position"])
            if drop is not None:
                commands[worker["index"]] = _unit_command(drop)
        farmer = commands.get(0, [PASS])
        market_plan = build_daily_plan(_state_for_planner(state), self.memory)
        market_plan.extend(macro.get("market_intents", ()))
        market_plan.extend(_explicit_market_intents(obs, state))
        seeds = _mapping(_get(state, "private", {})).get("seeds", {})
        has_seed = isinstance(seeds, Mapping) and any(_whole(quantity) > 0 for quantity in seeds.values())
        if not has_seed:
            market_plan.append({"kind": "BUY_SEED", "item": "WHEAT", "quantity": 1})
        visible_hands = _get(_mapping(_get(state, "farm", {})), "hands", ())
        if not isinstance(visible_hands, Sequence) or isinstance(visible_hands, (str, bytes)):
            visible_hands = [worker for worker in workers if worker["index"] != 0]
        hands = [commands.get(index + 1, [PASS]) for index in range(len(visible_hands))]
        market = build_market_orders(state, market_plan)
        self.memory.sell_batches = [order for order in market if order[0] == "SELL"]
        return {"farmer": farmer, "hands": hands, "market": market}
