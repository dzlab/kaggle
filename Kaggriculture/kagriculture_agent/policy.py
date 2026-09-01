"""Deterministic, legality-first Kaggriculture policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .constants import ANIMALS, CROPS, LAND_PRICES, PRODUCTS, max_market_orders, season_days, shed_capacity as DEFAULT_SHED_CAPACITY
from .economics import feed_reserve, market_price, market_regime
from .memory import PolicyMemory
from .observation import is_shed_adjacent, parse_observation as _parse_observation, shed_access_tiles
from .planner import (
    _fits_same_day_deadline,
    _feed_animal_counts,
    _is_live_owned_placed_animal,
    _owned_animal_counts,
    assign_tasks,
    build_autonomous_macro_plan,
    build_daily_plan,
    normalize_planner_state,
)
from .routing import is_locked_tile, normalize_position, next_move
from .strategy import StrategySpec, get_strategy, market_order_score, select_strategy
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
    return merged if _is_live_owned_placed_animal(merged) else None


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
    prices = _prices(state)
    if item in prices:
        return max(1.0, _number(prices[item], 1.0))
    inventory = _number(_market_inventory(state).get(item, 10_000), 10_000)
    try:
        return float(market_price(item, inventory - 1 - unit_offset, _market_params(state)))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _sale_proceeds(item: str, quantity: int, state: Any) -> float:
    """Estimate sequential proceeds for a sale order using observed prices."""
    quantity = _whole(quantity)
    if quantity <= 0:
        return 0.0
    if item in _prices(state):
        return quantity * _quote(item, state)
    inventory = _number(_market_inventory(state).get(item, 10_000), 10_000)
    params = _market_params(state)
    proceeds = 0.0
    for _ in range(quantity):
        try:
            price = float(market_price(item, inventory, params))
        except (KeyError, TypeError, ValueError, OverflowError):
            price = 0.0
        proceeds += price
        if price > 1:
            inventory += 1
    return proceeds


def _animals(state: Any) -> dict[str, int]:
    return _feed_animal_counts(_state_for_planner(state))


def _existing_animal_units(state: Any) -> int:
    """Count each owned animal once for strategy-cap enforcement."""
    return sum(_owned_animal_counts(_state_for_planner(state)).values())


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
        if kind == "SELL_ALL":
            return kind, None, _requested_quantity(item)
        if len(item) < 2:
            return None
        name = str(item[1]).upper()
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
    if kind == "SELL_ALL":
        return kind, None, _requested_quantity(item)
    return kind, str(name).upper() if name is not None else None, _requested_quantity(item)


def _approved_intents(plan: Any) -> list[tuple[str, str | None, int]]:
    values = plan if isinstance(plan, (list, tuple)) else [plan]
    result = []
    for item in values:
        parsed = _intent(item)
        if parsed is not None and parsed[2] > 0:
            result.append(parsed)
    return result


def order_market_intents(
    intents: Any,
    cash_needed: bool = False,
    strategy: StrategySpec | None = None,
    state: Any = None,
) -> list[Any]:
    """Return legal-shaped intents in deterministic cash-aware priority order."""
    values = list(intents) if isinstance(intents, Sequence) and not isinstance(intents, (str, bytes)) else [intents]
    ordered: list[tuple[int, Any, tuple[str, str | None, int]]] = []
    batch_limit = DEFAULT_SHED_CAPACITY
    if strategy is not None:
        try:
            batch_limit = max(1, min(DEFAULT_SHED_CAPACITY, int(strategy.max_sell_batch)))
        except (TypeError, ValueError, OverflowError):
            batch_limit = DEFAULT_SHED_CAPACITY
    for index, raw in enumerate(values):
        parsed = _intent(raw)
        if parsed is None or parsed[2] <= 0:
            continue
        kind, item, quantity = parsed
        if kind == "SELL" and strategy is not None:
            quantity = min(quantity, batch_limit)
            if isinstance(raw, Mapping):
                raw = {**raw, "quantity": quantity}
            elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
                raw = [*raw]
                raw[2] = quantity
            parsed = kind, item, quantity
        ordered.append((index, raw, parsed))
    if not cash_needed:
        return [raw for _index, raw, _parsed in ordered]

    def sort_key(entry: tuple[int, Any, tuple[str, str | None, int]]) -> tuple[int, float, int]:
        index, _raw, (kind, item, quantity) = entry
        if kind not in {"SELL", "SELL_ALL"}:
            return (1, 0.0, index)
        score = 0.0
        if state is not None and item is not None:
            try:
                score = market_order_score(item, quantity, state, urgency=0)
            except (KeyError, TypeError, ValueError, OverflowError):
                score = 0.0
        return (0, -score, index)

    return [raw for _index, raw, _parsed in sorted(ordered, key=sort_key)]


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


def _intent_purchase_cost(intents: Sequence[Sequence[Any]], state: Any) -> float:
    """Estimate approved purchase cash using the observed sequential quotes."""
    total = 0.0
    product_buys: dict[str, int] = {}
    for kind, item, requested in intents:
        quantity = _whole(requested, 1)
        if kind == "BUY_PRODUCT":
            item_name = item or ""
            already_bought = product_buys.get(item_name, 0)
            total += sum(
                _buy_product_quote(item_name, state, already_bought + offset)
                for offset in range(quantity)
            )
            product_buys[item_name] = already_bought + quantity
        elif kind in {"BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "HIRE"}:
            total += quantity * _purchase_cost(kind, item, state)
    return total


def _sell_batch_limit(strategy: StrategySpec | None) -> int:
    if strategy is None:
        return DEFAULT_SHED_CAPACITY
    try:
        return max(1, min(DEFAULT_SHED_CAPACITY, int(strategy.max_sell_batch)))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_SHED_CAPACITY


def _days_left(state: Any) -> int:
    return max(0, season_days - _whole(_get(state, "day", 0)))


def _terminal_liquidation_hour(strategy: StrategySpec | None) -> int:
    value = strategy.terminal_liquidation_hour if strategy is not None else 22
    try:
        return min(23, max(0, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 22


def build_market_orders(state: Any, plan: Any,
                        strategy: StrategySpec | None = None) -> list[list[Any]]:
    """Turn approved intents into bounded, affordable, legal market orders."""
    state = _state_for_planner(state)
    day, hour = _whole(_get(state, "day", 0)), _whole(_get(state, "hour", 0))
    shed = {str(item).upper(): _whole(quantity) for item, quantity in _shed(state).items()}
    cash = _cash(state)
    orders: list[list[Any]] = []
    product_buys: dict[str, int] = {}
    animal_buys = 0
    shed_room = max(0, DEFAULT_SHED_CAPACITY - sum(shed.values()))
    intents = _approved_intents(plan)
    allowed_crops = set(strategy.crops) if strategy is not None else set(CROPS)
    allowed_animals = set(strategy.animals) if strategy is not None else set(ANIMALS)
    if strategy is not None:
        intents = [
            intent for intent in intents
            if not (
                (intent[0] == "BUY_SEED" and intent[1] not in allowed_crops)
                or (intent[0] == "BUY_ANIMAL" and intent[1] not in allowed_animals)
            )
        ]
        crop_capacity = max(
            0,
            _whole(strategy.max_crop_units)
            - sum(1 for _, tile in _iter_tiles(state) if _crop(tile) in allowed_crops),
        )
        animal_capacity = max(0, _whole(strategy.max_animal_units) - _existing_animal_units(state))
    else:
        crop_capacity = None
        animal_capacity = None
    cash_needed = _intent_purchase_cost(intents, state) > cash
    intents = order_market_intents(
        intents, cash_needed=cash_needed, strategy=strategy, state=state,
    )
    final_turn = day >= season_days - 1 and hour >= _terminal_liquidation_hour(strategy)

    # Do not sell carried goods.  Policy.act sequences cleanup before the
    # final market window; retaining this flag also makes direct callers safe
    # without suppressing liquidation of already-shed inventory.
    carried_at_final = final_turn and _has_carried_goods(state)

    # Protect the remaining wheat needed by living animals before selling.
    carried_wheat = sum(_counts(inventory).get("WHEAT", 0) for inventory in _inventories(state))
    animal_counts = _animals(state)
    strategy_reserve = strategy.reserve_wheat if strategy is not None and animal_counts else 0
    total_feed_wheat = feed_reserve(
        animal_counts, _days_left(state), 0,
        strategy_reserve,
    )
    existing_wheat = shed.get("WHEAT", 0) + carried_wheat
    required_wheat = max(0, total_feed_wheat - existing_wheat)
    shed_wheat_reserve = max(0, total_feed_wheat - carried_wheat)
    wheat_price = _buy_product_quote("WHEAT", state)
    sell_intents = [intent for intent in intents if intent[0] in {"SELL", "SELL_ALL"}]

    if final_turn:
        sale_items = [] if carried_at_final else [
            (item, quantity) for item, quantity in shed.items() if item in _SALEABLE_PRODUCTS
        ]
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
    sale_orders: list[list[Any]] = []
    batch_limit = _sell_batch_limit(strategy)
    sale_items = sorted(
        sale_items,
        key=lambda entry: (-market_order_score(entry[0], entry[1], state, urgency=0), entry[0]),
    )
    for item, quantity in sale_items:
        quantity = min(_whole(quantity), shed.get(item, 0), batch_limit)
        if item == "WHEAT":
            quantity = min(quantity, max(0, shed.get(item, 0) - shed_wheat_reserve))
        if (
            quantity > 0
            and not final_turn
            and strategy is not None
            and strategy.avoid_price_floor_sales
            and _quote(item, state) <= 1
        ):
            continue
        if quantity > 0:
            sale_orders.append(["SELL", item, quantity])

    available_cash = cash
    available_shed_units = sum(shed.values())
    if cash_needed:
        for order in sale_orders:
            item, quantity = order[1], order[2]
            available_cash += _sale_proceeds(item, quantity, state)
            available_shed_units -= quantity

    # Wheat reserved for feed is purchased before discretionary approvals.
    if not final_turn and required_wheat and wheat_price > 0 and available_shed_units < DEFAULT_SHED_CAPACITY:
        affordable = 0
        purchase_cost = 0.0
        for offset in range(required_wheat):
            if available_shed_units + affordable >= DEFAULT_SHED_CAPACITY:
                break
            unit_cost = _buy_product_quote("WHEAT", state, offset)
            if unit_cost <= 0 or purchase_cost + unit_cost > available_cash:
                break
            affordable += 1
            purchase_cost += unit_cost
        if affordable:
            orders.append(["BUY_PRODUCT", "WHEAT", affordable])
            available_cash -= purchase_cost
            available_shed_units += affordable
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
        if kind == "BUY_SEED" and crop_capacity is not None:
            quantity = min(quantity, crop_capacity)
            if quantity <= 0:
                continue
        if kind == "BUY_PRODUCT":
            affordable, purchase_cost = 0, 0.0
            already_bought = product_buys.get(item or "", 0)
            for offset in range(quantity):
                if available_shed_units + affordable >= DEFAULT_SHED_CAPACITY:
                    break
                unit_cost = _buy_product_quote(item or "", state, already_bought + offset)
                if unit_cost <= 0 or purchase_cost + unit_cost > available_cash:
                    break
                affordable += 1
                purchase_cost += unit_cost
            unit_cost = purchase_cost
        else:
            if kind == "BUY_ANIMAL" and animal_capacity is not None:
                quantity = min(quantity, max(0, animal_capacity - animal_buys))
                if quantity <= 0:
                    continue
            unit_cost = _purchase_cost(kind, item, state)
            room = DEFAULT_SHED_CAPACITY - available_shed_units if kind == "BUY_ANIMAL" else quantity
            affordable = min(quantity, max(0, room), int(max(0.0, available_cash) // unit_cost)) if unit_cost > 0 else 0
        if affordable <= 0:
            continue
        order = [kind] if kind in {"HIRE", "BUY_LAND"} else [kind, item, affordable]
        orders.append(order)
        purchase_cost = unit_cost if kind == "BUY_PRODUCT" else affordable * unit_cost
        available_cash -= purchase_cost
        if kind == "BUY_PRODUCT":
            product_buys[item or ""] = product_buys.get(item or "", 0) + affordable
            available_shed_units += affordable
        elif kind == "BUY_ANIMAL":
            animal_buys += affordable
            available_shed_units += affordable
        elif kind == "BUY_SEED" and crop_capacity is not None:
            crop_capacity -= affordable
    if cash_needed:
        return (sale_orders + orders)[:max_market_orders]
    return (orders + sale_orders)[:max_market_orders]


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
            if item not in ANIMALS or ANIMALS[item]["structure"] != _structure_kind(tile) or animal is not None:
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
        item = str(_get(task, "animal", _get(task, "species", _get(task, "item", "")))).upper()
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
        if item in ANIMALS and inventory.get("WHEAT", 0) <= 0:
            return PASS
        if item in ANIMALS and ANIMALS[item]["structure"] == _structure_kind(tile) and _animal(tile) is None:
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
        return "DIG" if tile is not None and not is_locked_tile(tile) and _animal(tile) is None and _tile_kind(tile) in _TILE_KINDS | {"WEED", "STRUCTURE"} else PASS
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


def _drop_carried_goods(state: Any, worker_index: int, task: Any, current: Position,
                        *, force: bool = False) -> str | None:
    inventory = _inventory_for_worker(state, worker_index)
    if not any(inventory.get(item, 0) > 0 for item in set(PRODUCTS) | set(ANIMALS)):
        return None
    required = _required_worker_item(task, state)
    if not force and required is not None and inventory.get(required, 0) > 0:
        return None
    access = _shed_access_target(state, current)
    if access is None:
        return None
    return "DROP" if current == access else next_move(current, access)


def _has_carried_goods(state: Any) -> bool:
    return any(_counts(inventory) for inventory in _inventories(state))


def _remove_pickup_sale_conflicts(market: Sequence[Sequence[Any]], commands: Mapping[int, Sequence[Any]]) -> list[list[Any]]:
    """Keep a same-turn pickup from racing a sale of the picked item."""
    picked = {
        str(command[1]).upper()
        for command in commands.values()
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes))
        and len(command) >= 2 and command[0] == "PICKUP"
    }
    return [list(order) for order in market if not (
        isinstance(order, Sequence) and len(order) >= 2
        and order[0] == "SELL" and str(order[1]).upper() in picked
    )]


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
    if kind == "ANIMAL":
        animal = _required_worker_item(task, state)
        inventory = _inventory_for_worker(state, worker_index)
        if (
            animal in ANIMALS
            and inventory.get(animal, 0) > 0
            and inventory.get("WHEAT", 0) <= 0
            and _whole(_shed(state).get("WHEAT")) > 0
        ):
            access = _shed_access_target(state, current)
            if access is not None and current != access:
                return _unit_command(next_move(current, access))
            if access is not None:
                return ["PICKUP", "WHEAT", 1]
    if current != target:
        if not all(0 <= point.x < board_size and 0 <= point.y < board_size for point in (current, target)):
            return [PASS]
        return _unit_command(next_move(current, target))
    return _unit_command(_task_action(state, worker_index, task, current))


def _assignment_valid(state: Any, assignment: WorkerAssignment) -> bool:
    target = _position(_task_target(assignment.task))
    kind = str(_get(assignment.task, "kind", "")).upper()
    worker_index = _whole(_get(assignment, "worker_index", 0))
    deadline = _get(assignment.task, "deadline")
    if deadline is not None and _whole(_get(state, "day")) >= _whole(deadline):
        worker = next((worker for worker in _worker_records(state)
                       if worker["index"] == worker_index), None)
        if worker is None or worker["position"] is None:
            return False
        normalized = _state_for_planner(state)
        board_size = _whole(_get(normalized, "board_size", 1), 1)
        if not _fits_same_day_deadline(
            assignment.task,
            (worker["index"], worker["role"], worker["position"]),
            normalized,
            board_size,
            _whole(_get(normalized, "day")),
        ):
            return False
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
    required = _required_worker_item(assignment.task, state)
    if (
        kind == "ANIMAL"
        and required in ANIMALS
        and _inventory_for_worker(state, worker_index).get(required, 0) > 0
        and _whole(_shed(state).get("WHEAT")) > 0
    ):
        tile = _tile_at(state, target)
        return (
            _structure_kind(tile) == ANIMALS[required]["structure"]
            and _animal(tile) is None
            and _animal(tile) is None
        )
    if required is not None and _inventory_for_worker(state, worker_index).get(required, 0) <= 0:
        # Required inputs can be staged in the shed. Keep the assignment
        # stable while worker_action routes to PICKUP, otherwise logistics
        # replanning every turn starves build/placement/harvest work.
        if _whole(_shed(state).get(required)) <= 0:
            return False
        if kind == "FERTILIZE":
            return _is_engine_plant_tile(_tile_at(state, target))
        if kind == "FEED":
            return _animal(_tile_at(state, target)) is not None
        if kind == "ANIMAL":
            tile = _tile_at(state, target)
            return (
                required in ANIMALS
                and _structure_kind(tile) == ANIMALS[required]["structure"]
                and _animal(tile) is None
                and _animal(tile) is None
            )
    return _task_action(state, worker_index, assignment.task, target) != PASS


class Policy:
    """Stateful deterministic policy with reset-safe episode memory."""

    def __init__(self, strategy: str = "current") -> None:
        if strategy != "auto":
            get_strategy(strategy)
        self.strategy_name = strategy
        self.memory = PolicyMemory()

    def _carried_assignments(self, state: Mapping[str, Any]) -> list[WorkerAssignment]:
        """Keep a valid delivery task when another worker forces replanning."""
        protected = []
        for assignment in self.memory.assignments:
            kind = str(_get(assignment.task, "kind", "")).upper()
            if kind not in {"FEED", "FERTILIZE", "ANIMAL", "PLACE"}:
                continue
            required = _required_worker_item(assignment.task, state)
            if required is None:
                continue
            worker_index = _whole(_get(assignment, "worker_index"))
            if _inventory_for_worker(state, worker_index).get(required, 0) <= 0:
                continue
            if _assignment_valid(state, assignment):
                protected.append(assignment)
        return protected

    def _replan(self, state: Mapping[str, Any], regime: Mapping[str, str],
                macro: Mapping[str, Any] | None = None,
                protected: Sequence[WorkerAssignment] = (),
                strategy: StrategySpec | None = None) -> list[WorkerAssignment]:
        normalized = _state_for_planner(state)
        macro = macro or build_autonomous_macro_plan(normalized, self.memory, strategy)
        plan = build_daily_plan(normalized, self.memory, strategy)
        # Reserve the macro infrastructure tile before daily planting fills
        # the first empty square. This keeps BUILD_* and the selected crop
        # executable as separate tasks rather than competing for one tile.
        reserved_structure_tiles = {
            task.target for task in macro.get("tasks", ())
            if isinstance(task, Task) and task.kind in {"BUILD_COOP", "BUILD_PASTURE"}
        }
        if reserved_structure_tiles:
            plan = [task for task in plan if not (
                task.kind == "PLANT" and task.target in reserved_structure_tiles
            )]
        plan.extend(task for task in macro.get("tasks", ()) if isinstance(task, Task))
        protected_workers = {_whole(_get(assignment, "worker_index")) for assignment in protected}
        protected_tasks = {
            self._task_identity(assignment.task)
            for assignment in protected
        }
        plan = [task for task in plan if (
            self._task_identity(task) not in protected_tasks
        )]
        available_workers = [
            worker for worker in normalized.get("workers", ())
            if _whole(_get(worker, "index")) not in protected_workers
        ]
        assignments = assign_tasks(plan, available_workers, normalized, strategy)
        assignments.extend(protected)
        self.memory.assignments = assignments
        self.memory.market_regime = dict(regime)
        self.memory.diagnostics["plan_size"] = len(plan)
        self.memory.diagnostics["portfolio"] = dict(macro.get("portfolio", {}))
        self.memory.diagnostics["scenario_count"] = macro.get("scenario_count", 0)
        self.memory.diagnostics["worker_indices"] = tuple(
            sorted(_whole(_get(worker, "index")) for worker in normalized.get("workers", ()))
        )
        return assignments

    @staticmethod
    def _task_identity(task: Task) -> tuple[Any, ...]:
        kind = str(_get(task, "kind", "")).upper()
        target = _position(_task_target(task))
        if kind == "FEED":
            return kind, target
        return kind, target, str(_get(task, "item", "") or "").upper()

    def act(self, obs: Any) -> dict[str, Any]:
        state = parse_observation(obs)
        regime = market_regime(_prices(state), _market_inventory(state))
        shops = _get(_get(state, "town", {}), "unlocked_shops", ())
        regime["shops"] = "|".join(str(shop) for shop in shops) if isinstance(shops, Sequence) and not isinstance(shops, (str, bytes)) else ""
        selected_strategy = self.memory.selected_strategy
        reset = self.memory.observe_time(_get(state, "day"), _get(state, "hour"))
        strategy_spec = None
        if self.strategy_name == "auto":
            reset_reason = self.memory.diagnostics.get("reset_reason")
            if reset and reset_reason == "day_start" and selected_strategy is not None:
                self.memory.selected_strategy = selected_strategy
            if self.memory.selected_strategy is None:
                self.memory.selected_strategy = select_strategy(state).name
            strategy_spec = get_strategy(self.memory.selected_strategy)
        elif self.strategy_name != "current":
            strategy_spec = get_strategy(self.strategy_name)
        macro = build_autonomous_macro_plan(state, self.memory, strategy_spec)
        workers = _worker_records(state)
        worker_indices = tuple(sorted(worker["index"] for worker in workers))
        workers_changed = worker_indices != self.memory.diagnostics.get("worker_indices")
        hour_zero = _whole(_get(state, "hour")) == 0
        regime_changed = bool(self.memory.market_regime) and dict(regime) != self.memory.market_regime
        assignments_valid = all(_assignment_valid(state, assignment) for assignment in self.memory.assignments)
        if reset or hour_zero or workers_changed or regime_changed or not self.memory.assignments or not assignments_valid:
            protected = self._carried_assignments(state) if not reset and not hour_zero else ()
            assignments = self._replan(state, regime, macro, protected, strategy_spec)
        else:
            assignments = self.memory.assignments
        by_worker = {assignment.worker_index: assignment for assignment in assignments}
        terminal_cleanup = (
            _whole(_get(state, "day")) >= season_days - 1
            and _whole(_get(state, "hour")) >= min(12, _terminal_liquidation_hour(strategy_spec))
        )
        if terminal_cleanup:
            commands = {}
            for worker in workers:
                drop = _drop_carried_goods(
                    state, worker["index"], None, worker["position"], force=True,
                )
                commands[worker["index"]] = _unit_command(drop or PASS)
        else:
            commands = {worker["index"]: worker_action(worker["index"], state, by_worker.get(worker["index"])) for worker in workers}
            for worker in workers:
                drop = _drop_carried_goods(
                    state, worker["index"],
                    _get(by_worker.get(worker["index"]), "task"), worker["position"],
                )
                if drop is not None:
                    commands[worker["index"]] = _unit_command(drop)
        farmer = commands.get(0, [PASS])
        market_plan = build_daily_plan(_state_for_planner(state), self.memory, strategy_spec)
        market_plan.extend(macro.get("market_intents", ()))
        market_plan.extend(_explicit_market_intents(obs, state))
        seeds = _mapping(_get(state, "private", {})).get("seeds", {})
        has_seed = isinstance(seeds, Mapping) and any(_whole(quantity) > 0 for quantity in seeds.values())
        if not has_seed and _whole(_get(state, "day")) < season_days - 2:
            market_plan.append({"kind": "BUY_SEED", "item": "WHEAT", "quantity": 1})
        visible_hands = _get(_mapping(_get(state, "farm", {})), "hands", ())
        if not isinstance(visible_hands, Sequence) or isinstance(visible_hands, (str, bytes)):
            visible_hands = [worker for worker in workers if worker["index"] != 0]
        hands = [commands.get(index + 1, [PASS]) for index in range(len(visible_hands))]
        final_liquidation_window = (
            _whole(_get(state, "hour")) >= _terminal_liquidation_hour(strategy_spec)
        )
        market = (
            build_market_orders(state, market_plan, strategy_spec)
            if not terminal_cleanup or (final_liquidation_window and not _has_carried_goods(state))
            else []
        )
        market = _remove_pickup_sale_conflicts(market, commands)
        self.memory.sell_batches = [order for order in market if order[0] == "SELL"]
        return {"farmer": farmer, "hands": hands, "market": market}
