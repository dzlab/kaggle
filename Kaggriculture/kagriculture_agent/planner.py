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
from math import inf, isfinite
from typing import Any

from .constants import (
    ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, MARKET_I0, PRODUCTS, SHOPS,
    season_days, shed_capacity, turns_per_day,
)
from .economics import feed_reserve, forecast_crop, market_price, sell_batch_value
from .observation import parse_observation, shed_access_tiles
from .routing import distance, is_locked_tile, normalize_position, route_to
from .strategy import StrategySpec
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


def _safe_quantity(value: Any) -> int | float:
    if isinstance(value, bool):
        return 0
    try:
        quantity = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not isfinite(quantity) or quantity <= 0:
        return 0
    return int(quantity) if quantity.is_integer() else quantity


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if isfinite(number) else default


def _hand_counts(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(item): int(quantity) for item, quantity in (
            (item, _safe_quantity(quantity)) for item, quantity in value.items()
        ) if quantity > 0}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        counts: dict[str, int] = {}
        for item in value:
            if isinstance(item, str):
                counts[item] = counts.get(item, 0) + 1
        return counts
    return {}


def _held_inventory(value: Any) -> dict[str, int]:
    """Count worker-held items without confusing them with shed stock."""
    counts: dict[str, int] = {}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return counts

    def add(item: Any, quantity: Any = 1) -> None:
        if not isinstance(item, str):
            return
        safe = _safe_quantity(quantity)
        if safe > 0:
            key = item.upper()
            counts[key] = counts.get(key, 0) + int(safe)

    for hand in value:
        if isinstance(hand, Mapping):
            item = _get(hand, "item", _get(hand, "kind", _get(hand, "name", _get(hand, "species"))))
            if item is not None:
                add(item, _get(hand, "quantity", 1))
            else:
                for item, quantity in hand.items():
                    add(item, quantity)
        elif isinstance(hand, Sequence) and not isinstance(hand, (str, bytes)):
            for item in hand:
                if isinstance(item, str):
                    add(item)
                elif isinstance(item, Mapping):
                    add(
                        _get(item, "item", _get(item, "kind", _get(item, "name", _get(item, "species")))),
                        _get(item, "quantity", 1),
                    )
    return counts


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
    seeds = source.get("seeds")
    if not isinstance(seeds, Mapping):
        seeds = private.get("seeds")
    if not isinstance(seeds, Mapping):
        seeds = farm.get("seeds")
    inventory = source.get("inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        # Canonical engine observations expose shed stock separately from
        # worker-held inventories.  Prefer the latter when present; falling
        # back to shed stock is retained for the flat planner contract used
        # by callers and tests that have no inventory list.
        if "inventories" in private:
            inventory = _held_inventory(private.get("inventories"))
        elif hands:
            inventory = hands
        else:
            inventory = private.get("shed")
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
        "seeds": {
            str(item): _safe_quantity(quantity)
            for item, quantity in seeds.items()
        } if isinstance(seeds, Mapping) else {},
        "inventory": {
            str(item): _safe_quantity(quantity)
            for item, quantity in inventory.items()
        } if isinstance(inventory, Mapping) else {},
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


def _is_live_owned_placed_animal(animal: Any) -> bool:
    """Return whether an animal observation is eligible for live maintenance."""
    if not isinstance(animal, Mapping):
        return False
    species = _upper(_get(animal, "species", _get(animal, "animal", _get(animal, "kind", ""))))
    return (
        species in ANIMALS
        and _get(animal, "alive", True) is not False
        and _get(animal, "dead", False) is not True
        and _get(animal, "owned", True) is not False
        and _get(animal, "placed", True) is not False
    )


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


def _add(plan: list[Task], kind: str, target: Any, priority: int, deadline: int | None,
         value: float, *, item: str | None = None) -> None:
    if _target_position(target) is None and kind not in _SHED_WORK:
        return
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        numeric_value = 0.0
    if not isfinite(numeric_value) or numeric_value < 0:
        numeric_value = 0.0
    plan.append(Task(kind, target, priority, deadline, numeric_value, item=item))


def _deduplicate_maintenance(plan: list[Task]) -> list[Task]:
    """Remove duplicate maintenance emitted for merged tile/list views."""
    seen: set[tuple[str, Position | Any, str | None]] = set()
    result: list[Task] = []
    for task in plan:
        if task.kind in {"FEED", "CARE", "COLLECT_FERTILIZER"}:
            target = _target_position(task.target) or task.target
            key = (task.kind, target, task.item)
            if key in seen:
                continue
            seen.add(key)
        result.append(task)
    return result


def _inventory(state: Any) -> Mapping[str, Any]:
    value = _get(state, "inventory", _get(_get(state, "farm", {}), "hands", {}))
    return value if isinstance(value, Mapping) else {}


def _saleable_carried_surplus(
    state: Mapping[str, Any], inventory: Mapping[str, Any], strategy: StrategySpec | None,
) -> list[tuple[str, int]]:
    """Return deterministic carried-product quantities safe to stage for sale."""
    private = _mapping(state.get("private"))
    live_animals = _feed_animal_counts(state)
    strategy_reserve = strategy.reserve_wheat if strategy is not None and live_animals else 0
    feed_wheat = feed_reserve(
        live_animals,
        max(0, season_days - _day(state, EpisodeMemory())),
        reserve_wheat=strategy_reserve,
    )
    carried_wheat = int(_safe_quantity(inventory.get("WHEAT", 0)))
    if "inventories" in private:
        staged_wheat = int(_staged_wheat(state))
        saleable_wheat = min(carried_wheat, max(0, staged_wheat - feed_wheat))
    else:
        # In the legacy flat planner contract, ``inventory`` is the only
        # represented stock and may be the shed itself. Count it once.
        saleable_wheat = max(0, carried_wheat - feed_wheat)

    surplus: list[tuple[str, int]] = []
    for raw_item, raw_quantity in sorted(inventory.items(), key=lambda entry: str(entry[0]).upper()):
        item = str(raw_item).upper()
        quantity = int(_safe_quantity(raw_quantity))
        if item not in PRODUCTS or item == "FERTILIZER" or quantity <= 0:
            continue
        if item == "WHEAT":
            quantity = saleable_wheat
        if quantity > 0:
            surplus.append((item, quantity))
    return surplus


def _shed_target(state: Any, board_size: int) -> Position:
    explicit = _position(_get(state, "shed_position"))
    if explicit is not None and 0 <= explicit.x < board_size and 0 <= explicit.y < board_size:
        return explicit
    valid_access_tiles = tuple(
        tile for tile in shed_access_tiles(board_size)
        if 0 <= tile.x < board_size and 0 <= tile.y < board_size
    )
    return valid_access_tiles[0] if valid_access_tiles else Position(0, 0)


_MACRO_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
_MACRO_MODES = ("cash", "demand", "balanced", "animal")


def _can_produce_before_season_end(first_yield_day: Any, start_day: int,
                                   total_days: int = season_days) -> bool:
    """Return whether first production lands before the season time boundary."""
    try:
        return float(start_day) + float(first_yield_day) < int(total_days)
    except (TypeError, ValueError, OverflowError):
        return False


def _can_start_crop(crop: str, day: int) -> bool:
    rules = CROPS.get(crop)
    return bool(rules) and _can_produce_before_season_end(rules["first_yield_day"], day)


def _can_start_animal(animal: str, day: int, action_turns: int) -> bool:
    rules = ANIMALS.get(animal)
    return bool(rules) and _can_produce_before_season_end(
        float(rules["first_yield_day"])
        + max(0, int(action_turns)) / turns_per_day,
        day,
    )


def _state_cash(state: Mapping[str, Any]) -> float:
    farm = _mapping(state.get("farm"))
    value = state.get("cash", farm.get("money", 0))
    try:
        cash = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return cash if isfinite(cash) and cash >= 0 else 0.0


def _town_demand(state: Mapping[str, Any]) -> set[str]:
    town = _mapping(state.get("town"))
    shops = town.get("unlocked_shops", ())
    if not isinstance(shops, Sequence) or isinstance(shops, (str, bytes)):
        return set()
    safe_shops = (shop for shop in shops if isinstance(shop, str) and shop in SHOPS)
    return {item for shop in safe_shops for item in SHOPS.get(shop, ())}


def _portfolio_scenarios(state: Mapping[str, Any], day: int,
                         strategy: StrategySpec | None = None) -> list[dict[str, Any]]:
    """Evaluate a deterministic 5x4 crop/posture portfolio matrix."""
    horizon = max(1, min(season_days - day, 12))
    demand = _town_demand(state)
    prices = _observed_prices(state)
    allowed_crops = set(strategy.crops) if strategy is not None else set(_MACRO_CROPS)
    crops = tuple(
        crop for crop in _MACRO_CROPS
        if crop in allowed_crops and _can_start_crop(crop, day)
    )
    scenarios: list[dict[str, Any]] = []
    for crop in crops:
        for mode in _MACRO_MODES:
            try:
                forecast = forecast_crop(
                    crop, start_day=day, horizon=horizon,
                    watering_days=range(day, day + horizon), prices=prices,
                    market_inventory=_observed_inventory(crop, state),
                    seed_owned=True, fixed_price_mode=bool(prices),
                )
                score = float(forecast.get("net_cash", 0.0))
            except (KeyError, TypeError, ValueError, OverflowError):
                score = 0.0
            quote = _observed_quote(crop, state)
            # Keep the live quote material in the portfolio decision.  The
            # forecast is deliberately conservative for crops whose harvest
            # is outside the short horizon, but a strong observed quote is
            # still actionable information for the next planting decision.
            score += quote * 3.0
            if crop in demand:
                score += quote * (2.0 if mode == "demand" else 0.5)
            if mode == "cash" and not CROPS[crop]["ongoing"]:
                score += 25.0
            if mode == "balanced":
                score += quote
            if mode == "animal":
                score += max((_observed_quote(product, state) for product in ("EGG", "MILK", "WOOL") if product in demand), default=0.0)
            scenarios.append({"crop": crop, "mode": mode, "score": score})
    return scenarios


def _preferred_animal(state: Mapping[str, Any], demand: set[str],
                      allowed_animals: Sequence[str] | None = None) -> str:
    product_order = ("EGG", "MILK", "WOOL")
    animals = (tuple(ANIMALS) if allowed_animals is None else
               tuple(animal for animal in ANIMALS if animal in allowed_animals))
    if not animals:
        return ""
    for product in product_order:
        candidate = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}[product]
        if product in demand and candidate in animals:
            return candidate
    return max(animals, key=lambda animal: (_observed_quote(ANIMALS[animal]["product"], state), animal))


def _placed_animal_count(state: Mapping[str, Any]) -> int:
    count = 0
    for _position_value, tile in _tiles(state):
        entity = _entity_state(tile, "animal")
        if _is_live_owned_placed_animal(entity):
            count += 1
    return count


def _planned_crop_count(state: Mapping[str, Any], strategy: StrategySpec | None) -> int:
    allowed = set(strategy.crops) if strategy is not None else set(CROPS)
    return sum(1 for _position_value, tile in _tiles(state) if (_crop(tile) or "") in allowed)


def _planned_animal_count(state: Mapping[str, Any]) -> int:
    return sum(_safe_quantity(quantity) for quantity in _owned_animal_counts(state).values())


def _compatible_structure(state: Mapping[str, Any], animal: str) -> tuple[Position | None, Position | None]:
    structure = ANIMALS[animal]["structure"]
    empty: Position | None = None
    for position, tile in _tiles(state):
        if is_locked_tile(tile):
            continue
        kind = _tile_kind(tile)
        if kind == structure and not _is_live_owned_placed_animal(_entity_state(tile, "animal")):
            return position, empty
        if empty is None and _is_empty(tile):
            empty = position
    return None, empty


def _feed_animal_counts(state: Mapping[str, Any]) -> dict[str, int]:
    """Count live animals and carried units, merging duplicate observations."""
    counts: dict[str, int] = {}
    seen: set[tuple[str, Any]] = set()

    observations: list[tuple[Any, Position | None]] = []
    for position, tile in _tiles(state):
        animal = _entity_state(tile, "animal")
        if animal is not None:
            observations.append((animal, _position(_get(animal, "position")) or position))

    observed = _get(state, "animals", ())
    if isinstance(observed, Mapping):
        observed = (observed,)
    if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)):
        observations.extend((animal, _position(_get(animal, "position"))) for animal in observed)

    for animal, position in observations:
        species = _upper(_get(animal, "species", _get(animal, "animal", _get(animal, "kind", ""))))
        if not _is_live_owned_placed_animal(animal):
            continue
        identity_keys = {
            (field, str(_get(animal, field)))
            for field in ("id", "animal_id", "entity_id", "uid")
            if _get(animal, field) is not None
        }
        if position is not None:
            identity_keys.add(("position", position))
        if identity_keys & seen:
            continue
        seen.update(identity_keys)
        counts[species] = counts.get(species, 0) + 1

    private = _mapping(_get(state, "private", {}))
    inventories = private.get("inventories")
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)):
        held = _held_inventory(inventories)
    elif "inventory" in state and not isinstance(private.get("shed"), Mapping):
        held = _held_inventory((_get(state, "inventory", ()),))
    else:
        held = {}
    for species in ANIMALS:
        quantity = int(_safe_quantity(held.get(species, 0)))
        if quantity:
            counts[species] = counts.get(species, 0) + quantity

    return counts


def _owned_animal_counts(state: Mapping[str, Any]) -> dict[str, int]:
    """Count live animals once and add owned animal units held in the shed."""
    counts = _feed_animal_counts(state)
    private = _mapping(_get(state, "private", {}))
    shed = private.get("shed", _get(state, "shed", _get(state, "inventory", {})))
    if not isinstance(shed, Mapping):
        return counts
    for species in ANIMALS:
        quantity = int(_safe_quantity(shed.get(species, 0)))
        if quantity:
            counts[species] = counts.get(species, 0) + quantity
    return counts


def _staged_wheat(state: Mapping[str, Any]) -> int | float:
    """Return wheat in the shed and in every worker inventory."""
    private = _mapping(state.get("private"))
    shed = private.get("shed", state.get("shed", {}))
    total = _safe_quantity(shed.get("WHEAT", 0)) if isinstance(shed, Mapping) else 0
    inventories = private.get("inventories")
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)):
        total += _held_inventory(inventories).get("WHEAT", 0)
    else:
        inventory = state.get("inventory", {})
        if isinstance(inventory, Mapping):
            total += _safe_quantity(inventory.get("WHEAT", 0))
    return total


def _intent_purchase_cost(intents: Sequence[Sequence[Any]], state: Mapping[str, Any]) -> float:
    """Estimate all already-planned purchase cash at the observed quotes."""
    farm = _mapping(state.get("farm"))
    unlocked = farm.get("unlocked_quadrants", state.get("unlocked_quadrants", ["NW"]))
    if not isinstance(unlocked, Sequence) or isinstance(unlocked, (str, bytes)):
        unlocked = ["NW"]
    hires = _safe_quantity(farm.get("hires_today", state.get("hires_today", 0)))
    first, second = 1, 1
    for _ in range(int(hires)):
        first, second = second, first + second
    total = 0.0
    for intent in intents:
        if not isinstance(intent, Sequence) or isinstance(intent, (str, bytes)) or not intent:
            continue
        kind = str(intent[0]).upper()
        item = str(intent[1]).upper() if len(intent) > 1 and intent[1] is not None else ""
        quantity = int(_safe_quantity(intent[2])) if len(intent) > 2 else 1
        if kind == "BUY_SEED" and item in CROPS:
            total += quantity * float(CROPS[item]["seed"])
        elif kind == "BUY_PRODUCT" and item in PRODUCTS:
            total += quantity * _observed_quote(item, state)
        elif kind == "BUY_ANIMAL" and item in ANIMALS:
            total += quantity * float(ANIMALS[item]["cost"])
        elif kind == "BUY_LAND":
            index = len(unlocked) - 1
            if 0 <= index < len(LAND_PRICES):
                total += float(LAND_PRICES[index])
                unlocked = [*unlocked, LAND_ORDER[index]]
        elif kind == "HIRE":
            total += float(first)
            first, second = second, first + second
    return total


def _feed_purchase_needed(state: Mapping[str, Any], day: int, counts: Mapping[str, int],
                          intents: Sequence[Sequence[Any]], cash: float,
                          wheat_price: float, reserve_wheat: int = 0) -> tuple[int, float]:
    """Return missing full-season feed and cash after buying that feed."""
    days = max(0, season_days - day)
    already_planned = sum(
        int(_safe_quantity(intent[2]))
        for intent in intents
        if isinstance(intent, Sequence) and not isinstance(intent, (str, bytes))
        and len(intent) >= 3 and str(intent[0]).upper() == "BUY_PRODUCT"
        and str(intent[1]).upper() == "WHEAT"
    )
    required = (
        sum(max(0, int(quantity)) for quantity in counts.values()) * days
        + max(0, int(reserve_wheat))
    )
    missing = max(0, required - int(_staged_wheat(state)) - already_planned)
    cash_after = cash - _intent_purchase_cost(intents, state) - missing * max(0.0, wheat_price)
    return missing, cash_after


def _has_basic_need_deadline(state: Any, day: int | None = None,
                             strategy: StrategySpec | None = None) -> bool:
    """Return whether a required watering or feeding task is due today."""
    raw_state = _mapping(state)
    if "farm" not in raw_state and "farms" in raw_state:
        state = parse_observation(state)
    normalized = normalize_planner_state(state)
    current_day = _day(normalized, EpisodeMemory()) if day is None else day
    return any(
        task.kind in {"WATER", "FEED"}
        and task.deadline is not None
        and task.deadline <= current_day
        for task in build_daily_plan(normalized, EpisodeMemory(), strategy)
    )


def due_basic_need_tasks(state: Any, day: int | None = None,
                         strategy: StrategySpec | None = None) -> list[Task]:
    """Return due WATER/FEED tasks from the current normalized observation."""
    raw_state = _mapping(state)
    if "farm" not in raw_state and "farms" in raw_state:
        state = parse_observation(state)
    normalized = normalize_planner_state(state)
    current_day = _day(normalized, EpisodeMemory()) if day is None else day
    return [
        task for task in build_daily_plan(normalized, EpisodeMemory(), strategy)
        if task.kind in {"WATER", "FEED"}
        and task.deadline is not None
        and task.deadline <= current_day
    ]


def _due_needs_exceed_single_worker_capacity(
    state: Mapping[str, Any], tasks: Sequence[Task], day: int, hour: int,
) -> bool:
    """Estimate whether the deterministic due-needs route fits this day."""
    workers = list(_get(state, "workers", ()) or ())
    if len(workers) != 1 or not tasks:
        return False
    try:
        board_size = max(1, int(_get(state, "board_size", 1)))
    except (TypeError, ValueError, OverflowError):
        board_size = 1
    worker = _worker_info(workers[0], 0)
    current = worker[2]
    remaining = list(tasks)
    required_turns = 0
    while remaining:
        def route_key(task: Task) -> tuple[Any, ...]:
            base = _task_sort_key(task, day)
            target = _target_position(task.target)
            route_distance = (
                distance(current, target)
                if current is not None and target is not None else inf
            )
            return (*base[:4], route_distance, base[5], base[4])

        task = min(remaining, key=route_key)
        required_turns += _task_turn_budget(
            task, (worker[0], worker[1], current), state, board_size,
        )
        if required_turns > 24 - hour:
            return True
        current = _target_position(task.target)
        remaining.remove(task)
    return False


def build_autonomous_macro_plan(state: Any, memory: EpisodeMemory | Any = None,
                                strategy: StrategySpec | None = None) -> dict[str, Any]:
    """Choose a live portfolio and executable macro intents from observations.

    The returned intent list is internal policy output, not an externally
    supplied approval channel.  It is deliberately conservative: every market
    intent is still passed through ``build_market_orders`` for cash, capacity,
    land-order, and ten-order legality checks.
    """
    raw_state = _mapping(state)
    if "farm" not in raw_state and "farms" in raw_state:
        state = parse_observation(state)
    normalized = normalize_planner_state(state)
    day = _day(normalized, memory or EpisodeMemory())
    hour = _get(normalized, "hour", 0)
    try:
        hour = max(0, int(hour))
    except (TypeError, ValueError, OverflowError):
        hour = 0
    scenarios = _portfolio_scenarios(normalized, day, strategy)
    selected = (
        max(enumerate(scenarios), key=lambda item: (item[1]["score"], -item[0]))[1]
        if scenarios else {"crop": "", "mode": "", "score": 0.0}
    )
    demand = _town_demand(normalized)
    allowed_crops = set(strategy.crops) if strategy is not None else set(CROPS)
    allowed_animals = set(strategy.animals) if strategy is not None else set(ANIMALS)
    animal = _preferred_animal(normalized, demand, allowed_animals)
    farm = _mapping(normalized.get("farm"))
    private = _mapping(normalized.get("private"))
    seeds = normalized.get("seeds", {})
    shed = private.get("shed", normalized.get("inventory", {}))
    seeds = seeds if isinstance(seeds, Mapping) else {}
    shed = shed if isinstance(shed, Mapping) else {}
    stored_animals = [candidate for candidate in ANIMALS
                      if candidate in allowed_animals and _safe_quantity(shed.get(candidate, 0)) > 0]
    if stored_animals and animal not in stored_animals:
        animal = stored_animals[0]
    animal_counts = _feed_animal_counts(normalized)
    planned_crop_count = _planned_crop_count(normalized, strategy)
    planned_animal_count = _planned_animal_count(normalized)
    crop_cap = strategy.max_crop_units if strategy is not None else None
    animal_cap = strategy.max_animal_units if strategy is not None else None
    wheat_staged = _staged_wheat(normalized)
    wheat_price = _observed_quote("WHEAT", normalized)
    # Stored animals are owned inventory, not live placed animals. They do
    # not enter lifecycle maintenance/counts, but their placement still needs
    # a feed reserve so the resulting placement remains executable.
    stored_animal_counts = {
        species: int(_safe_quantity(shed.get(species, 0)))
        for species in ANIMALS
        if _safe_quantity(shed.get(species, 0)) > 0
    }
    planning_animal_counts = dict(animal_counts)
    for species, quantity in stored_animal_counts.items():
        planning_animal_counts[species] = planning_animal_counts.get(species, 0) + quantity
    feed_required = bool(planning_animal_counts)
    cash = _state_cash(normalized)
    intents: list[list[Any]] = []
    tasks: list[Task] = []
    due_needs = due_basic_need_tasks(normalized, day, strategy)
    deadline_needs = bool(due_needs)
    _compatible_target, structure_target = (
        _compatible_structure(normalized, animal) if animal else (None, None)
    )
    animal_action_turns = 1 if _compatible_target is not None else 2
    can_start_animal = bool(animal) and _can_start_animal(animal, day, animal_action_turns)
    if day < season_days - 2:
        seed_cost = float(CROPS[selected["crop"]]["seed"]) if selected["crop"] else 0.0
        can_plan_crop = (
            bool(selected["crop"])
            and _can_start_crop(selected["crop"], day)
            and (crop_cap is None or planned_crop_count < crop_cap)
        )
        seed_purchase_planned = (
            can_plan_crop
            and _safe_quantity(seeds.get(selected["crop"], 0)) <= 0
            and cash >= seed_cost
        )
        if seed_purchase_planned:
            intents.append(["BUY_SEED", selected["crop"], 1])

        # Keep the portfolio executable across the market/unit-action split:
        # a seed bought this turn is available to the assigned worker on the
        # next observation, so schedule the planting task now instead of
        # waiting for an externally supplied intent.
        plant_crop = selected["crop"] if seed_purchase_planned or _safe_quantity(seeds.get(selected["crop"], 0)) > 0 else ""
        if not plant_crop:
            available_seeds = [
                crop for crop in CROPS
                if crop in allowed_crops
                and _safe_quantity(seeds.get(crop, 0)) > 0
                and _can_start_crop(crop, day)
            ]
            plant_crop = max(available_seeds, key=lambda crop: (_observed_quote(crop, normalized), crop), default="")
        if plant_crop and _can_start_crop(plant_crop, day) and (
            crop_cap is None or planned_crop_count < crop_cap
        ):
            plant_target = next((position for position, tile in _tiles(normalized)
                                 if not is_locked_tile(tile) and _is_empty(tile)
                                 and position != structure_target), None)
            if plant_target is not None:
                tasks.append(Task("PLANT", plant_target, 30, day,
                                  _observed_quote(plant_crop, normalized), item=plant_crop))

        has_unfertilized_crop = any(
            _crop(tile) is not None and _number(_get(tile, "fertilized_until_day", -1)) < day
            for _position_value, tile in _tiles(normalized)
        )
        if has_unfertilized_crop and _safe_quantity(shed.get("FERTILIZER", 0)) <= 0 and cash >= _observed_quote("FERTILIZER", normalized):
            intents.append(["BUY_PRODUCT", "FERTILIZER", 1])
        unlocked = farm.get("unlocked_quadrants", normalized.get("unlocked_quadrants", ["NW"]))
        if not isinstance(unlocked, Sequence) or isinstance(unlocked, (str, bytes)):
            unlocked = ["NW"]
        land_index = len(unlocked) - 1
        reserve = max(100.0, seed_cost)
        if feed_required and wheat_price > 0:
            quantity, cash_after = _feed_purchase_needed(
                normalized, day, planning_animal_counts, intents, cash, wheat_price,
                strategy.reserve_wheat if strategy is not None else 0,
            )
            if quantity and cash_after >= reserve:
                intents.append(["BUY_PRODUCT", "WHEAT", quantity])
        if 0 <= land_index < len(LAND_ORDER) and cash >= float(LAND_PRICES[land_index]) + reserve:
            intents.append(["BUY_LAND"])

        compatible, empty = (
            _compatible_structure(normalized, animal) if animal else (None, None)
        )
        animal_in_storage = _safe_quantity(shed.get(animal, 0)) > 0
        can_plan_animal = animal_cap is None or planned_animal_count < animal_cap
        if (can_plan_animal and can_start_animal and _placed_animal_count(normalized) == 0
                and not animal_in_storage and (compatible is not None or empty is not None)):
            candidate_counts = dict(animal_counts)
            candidate_counts[animal] = candidate_counts.get(animal, 0) + 1
            quantity, cash_after = _feed_purchase_needed(
                normalized, day, candidate_counts, intents, cash, wheat_price,
                strategy.reserve_wheat if strategy is not None else 0,
            )
            if quantity and wheat_price > 0 and cash_after >= float(ANIMALS[animal]["cost"]) + reserve:
                staged_for_purchase = int(_staged_wheat(normalized)) + sum(
                    int(_safe_quantity(intent[2])) for intent in intents
                    if len(intent) >= 3 and intent[0] == "BUY_PRODUCT" and intent[1] == "WHEAT"
                )
                if staged_for_purchase + quantity <= shed_capacity:
                    intents.append(["BUY_PRODUCT", "WHEAT", quantity])
            quantity_after_planning, candidate_cash_after = _feed_purchase_needed(
                normalized, day, candidate_counts, intents, cash, wheat_price,
                strategy.reserve_wheat if strategy is not None else 0,
            )
            if quantity_after_planning == 0 and candidate_cash_after >= float(ANIMALS[animal]["cost"]) + reserve:
                intents.append(["BUY_ANIMAL", animal, 1])
        unfunded_feed, feed_cash_after = _feed_purchase_needed(
            normalized, day, planning_animal_counts, intents, cash, wheat_price,
            strategy.reserve_wheat if strategy is not None else 0,
        )
        # Placement consumes already-staged goods and a worker turn; it does
        # not need the discretionary cash cushion used for new purchases.
        animal_feed_ready = unfunded_feed == 0
        if (can_start_animal and compatible is not None
                and animal_in_storage and animal_feed_ready):
            # A stored animal cannot be fed until it is placed. Give this
            # delivery a higher priority than routine field work so a hand
            # expiring at the day boundary cannot strand the animal.
            tasks.append(Task("ANIMAL", compatible, 105, day, float(ANIMALS[animal]["cost"]), item=animal))
        elif (can_start_animal and empty is not None
              and not _placed_animal_count(normalized)):
            kind = "BUILD_PASTURE" if ANIMALS[animal]["structure"] == "PASTURE" else "BUILD_COOP"
            tasks.append(Task(kind, empty, 96, day, 1.0))

        hands = farm.get("hands", ())
        hand_count = len(hands) if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes)) else 0
        hires_today = _safe_quantity(farm.get("hires_today", 0))
        highest_due_priority = max((task.priority for task in due_needs), default=inf)
        preempting_tasks = [
            task for task in tasks
            if task.deadline is not None
            and task.deadline <= day
            and task.priority > highest_due_priority
        ]
        deadline_capacity_hire = (
            hand_count == 0
            and hour < 23
            and _due_needs_exceed_single_worker_capacity(
                normalized, [*due_needs, *preempting_tasks], day, hour,
            )
        )
        hire_cost = _intent_purchase_cost((["HIRE"],), normalized)
        # Deadline hiring is an immediate capacity purchase.  The remaining
        # feed reserve is a future solvency obligation, not a prerequisite to
        # paying for a helper now.  Feed remains independently guarded above,
        # and BUY_ANIMAL below still requires the full reserve to be funded.
        capacity_preserving_intents = [
            intent for intent in intents
            if (
                isinstance(intent, Sequence)
                and not isinstance(intent, (str, bytes))
                and len(intent) >= 2
                and str(intent[0]).upper() == "BUY_PRODUCT"
                and str(intent[1]).upper() == "WHEAT"
            )
        ]
        cash_after_planned_intents = cash - _intent_purchase_cost(
            capacity_preserving_intents, normalized,
        )
        can_fund_deadline_hire = (
            cash_after_planned_intents >= hire_cost + reserve
        )
        if hand_count < 2 and hires_today == 0:
            if deadline_capacity_hire and can_fund_deadline_hire:
                # Required capacity uses the actual next-hire cost and keeps
                # the established solvency reserve. Place it before optional
                # purchases so those intents remain valid but cannot crowd it
                # out before the worker is created.
                intents.insert(0, ["HIRE"])
            elif hour == 0 and cash >= 100.0 + reserve:
                intents.append(["HIRE"])
    if day == season_days - 2 and deadline_needs and hour == 0:
        hands = farm.get("hands", ())
        hand_count = len(hands) if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes)) else 0
        hires_today = _safe_quantity(farm.get("hires_today", 0))
        reserve = max(100.0, float(CROPS[selected["crop"]]["seed"])) if selected["crop"] else 100.0
        if hand_count < 2 and hires_today == 0 and cash >= 100.0 + reserve:
            intents.append(["HIRE"])
    return {
        "portfolio": {"crop": selected["crop"], "mode": selected["mode"], "animal": animal, "score": selected["score"]},
        "scenario_count": len(scenarios),
        "market_intents": intents,
        "tasks": tasks,
    }


def build_daily_plan(state: Any, memory: EpisodeMemory | Any = None,
                     strategy: StrategySpec | None = None) -> list[Task]:
    """Build a stable one-day plan from a typed or mapping-shaped state."""
    state = normalize_planner_state(state)
    memory = memory or EpisodeMemory()
    day = _day(state, memory)
    allowed_crops = set(strategy.crops) if strategy is not None else set(CROPS)
    allowed_animals = set(strategy.animals) if strategy is not None else set(ANIMALS)
    crop_cap = strategy.max_crop_units if strategy is not None else None
    animal_cap = strategy.max_animal_units if strategy is not None else None
    planned_crop_count = _planned_crop_count(state, strategy)
    planned_animal_count = _planned_animal_count(state)
    try:
        board_size = max(1, int(_get(state, "board_size", 1)))
    except (TypeError, ValueError, OverflowError):
        board_size = 1
    plan: list[Task] = []
    held_inventory = _mapping(state.get("inventory"))
    shed_inventory = _mapping(_mapping(state.get("private")).get("shed"))
    has_fertilizer = (
        _safe_quantity(held_inventory.get("FERTILIZER", 0)) > 0
        or _safe_quantity(shed_inventory.get("FERTILIZER", 0)) > 0
    )

    for position, tile in _tiles(state):
        if is_locked_tile(tile):
            continue
        kind = _tile_kind(tile)
        crop = _crop(tile)
        if kind == "WEED":
            _add(plan, "WEED", position, 80, day, 10)
        if crop:
            if _needs_today(tile, "needs_water", "watered_today", "watered"):
                _add(plan, "WATER", position, 100, day, 1, item=crop)
            if (crop in allowed_crops
                    and _number(_get(tile, "fertilized_until_day", -1)) < day
                    and has_fertilizer):
                _add(plan, "FERTILIZE", position, 97, None, 1, item=crop)
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
                    _add(plan, "HARVEST", position, 98, day, value, item=crop)
        elif _is_empty(tile) and day < season_days - 2:
            seeds = _get(state, "seeds", {})
            if not isinstance(seeds, Mapping):
                seeds = {}
            available = [
                crop_name for crop_name in CROPS
                if crop_name in allowed_crops
                and seeds.get(crop_name, 0)
                and _can_start_crop(crop_name, day)
            ]
            if available and (crop_cap is None or planned_crop_count < crop_cap):
                crop_name = max(available, key=lambda item: (_observed_quote(item, state), item))
                _add(plan, "PLANT", position, 20, day, _observed_quote(crop_name, state), item=crop_name)
                planned_crop_count += 1

        animal_entity = _entity_state(tile, "animal")
        if animal_entity is not None and _is_live_owned_placed_animal(animal_entity):
            animal_position = _position(animal_entity) or position
            species = _upper(_get(animal_entity, "species", _get(animal_entity, "animal", _get(animal_entity, "kind", ""))))
            animal_value = float(ANIMALS.get(species, {}).get("cost", 1))
            if species in allowed_animals or species in ANIMALS:
                if _needs_today(animal_entity, "needs_feed", "fed_today", "fed"):
                    _add(plan, "FEED", animal_position, 100, day, 1, item=species)
                if _needs_today(animal_entity, "needs_care", "cared_today", "cared"):
                    _add(plan, "CARE", animal_position, 95, day, animal_value, item=species)
                if species in allowed_animals and _get(animal_entity, "fertilizer_available") is True:
                    _add(plan, "COLLECT_FERTILIZER", animal_position, 96, day, 1, item=species)
            if species in allowed_animals:
                if (_get(animal_entity, "needs_placement", False)
                        or _get(animal_entity, "placed") is False
                        or _get(animal_entity, "owned") is False):
                    _add(plan, "ANIMAL", animal_position, 94, day,
                         _get(animal_entity, "value", 1), item=species)

        structure_entity = _entity_state(tile, "structure")
        if structure_entity is not None and (
            _get(structure_entity, "needs_placement", False)
            or _get(structure_entity, "built") is False
            or _get(structure_entity, "placed") is False
        ):
            structure_kind = _upper(_get(structure_entity, "kind", _get(structure_entity, "structure", kind)))
            candidates = [
                species for species in allowed_animals
                if ANIMALS.get(species, {}).get("structure") == structure_kind
            ]
            if candidates and any(_can_start_animal(species, day, 2) for species in candidates):
                _add(plan, "STRUCTURE", position, 45, None, _get(structure_entity, "value", 1))

    animals = _get(state, "animals", ()) or ()
    for animal in animals:
        position = _position(animal)
        if position is None or not _is_live_owned_placed_animal(animal):
            continue
        species = _upper(_get(animal, "species", _get(animal, "kind", "")))
        value = float(ANIMALS.get(species, {}).get("cost", 1))
        if _needs_today(animal, "needs_feed", "fed_today", "fed"):
            _add(plan, "FEED", position, 100, day, 1, item=species)
        if _needs_today(animal, "needs_care", "cared_today", "cared"):
            _add(plan, "CARE", position, 95, day, value, item=species)
        if _get(animal, "fertilizer_available") is True:
            _add(plan, "COLLECT_FERTILIZER", position, 96, day, 1, item=species)

    for structure in _get(state, "structures", ()) or ():
        if not bool(_get(structure, "built", True)):
            structure_kind = _upper(_get(structure, "kind", _get(structure, "structure", "")))
            candidates = [
                species for species in allowed_animals
                if ANIMALS.get(species, {}).get("structure") == structure_kind
            ]
            if candidates and any(_can_start_animal(species, day, 2) for species in candidates):
                _add(plan, "STRUCTURE", _position(structure), 45, None, _get(structure, "value", 1))
    for animal in _get(state, "desired_animals", ()) or ():
        if not bool(_get(animal, "owned", False)):
            species = _upper(_get(animal, "species", _get(animal, "animal", _get(animal, "kind", ""))))
            compatible, _empty = _compatible_structure(state, species) if species in ANIMALS else (None, None)
            action_turns = 1 if compatible is not None else 2
            if (species in allowed_animals
                    and _can_start_animal(species, day, action_turns)
                    and (animal_cap is None or planned_animal_count < animal_cap)):
                _add(plan, "ANIMAL", _position(animal), 94, day,
                     _get(animal, "value", 1), item=species)
                planned_animal_count += 1

    inventory = _inventory(state)
    held = sum(float(_safe_quantity(quantity)) for quantity in inventory.values())
    if held > 0:
        shed_target = _shed_target(state, board_size)
        _add(plan, "SHED", shed_target, 85, day, held)
        for item, quantity in _saleable_carried_surplus(state, inventory, strategy):
            try:
                value = _observed_sale_value(item, int(quantity), state)
            except (KeyError, TypeError, ValueError, OverflowError):
                value = 0
            if value > 0:
                task = Task("SELL", shed_target, 75, day, value, sell_all=True, item=item)
                # Task predates quantity-bearing market intents. Keep its
                # public shape while recording the bounded planner quantity
                # consumed by policy._requested_quantity.
                task.quantity = int(quantity)
                plan.append(task)

    return _deduplicate_maintenance(plan)


def _worker_info(worker: Any, fallback_index: int) -> tuple[int, str, Position | None]:
    index = _get(worker, "index", fallback_index)
    try:
        index = int(index)
    except (TypeError, ValueError, OverflowError):
        index = fallback_index
    role = _upper(_get(worker, "role", _get(worker, "name", "WORKER")))
    return index, role, _position(_get(worker, "position", worker))


def _task_allowed(task: Task, strategy: StrategySpec | None, state: Mapping[str, Any] | None = None) -> bool:
    if strategy is None:
        return True
    kind = str(task.kind).upper()
    item = str(task.item or "").upper()
    if kind in {"WATER", "HARVEST"} and item and item not in strategy.crops:
        target = _target_position(task.target)
        if state is None or target is None:
            return False
        return any(
            position == target and _crop(tile) == item
            for position, tile in _tiles(state)
        )
    if kind in {"FEED", "CARE"} and item and item not in strategy.animals:
        target = _target_position(task.target)
        if state is None or target is None:
            return False
        if any(
            position == target
            and (entity := _entity_state(tile, "animal")) is not None
            and _upper(_get(entity, "species", _get(entity, "animal", ""))) == item
            for position, tile in _tiles(state)
        ):
            return True
        animals = _get(state, "animals", ()) or ()
        if isinstance(animals, Mapping):
            animals = (animals,)
        return any(
            _position(animal) == target
            and _upper(_get(animal, "species", _get(animal, "animal", _get(animal, "kind", "")))) == item
            for animal in animals
            if isinstance(animal, Mapping)
        )
    if kind in {"PLANT", "WATER", "FERTILIZE", "HARVEST"} and item:
        return item in strategy.crops
    if kind in {"FEED", "CARE", "COLLECT_FERTILIZER", "ANIMAL", "PLACE"} and item:
        return item in strategy.animals
    return True


def _task_key(task: Task) -> tuple[Any, ...]:
    target = _target_position(task.target)
    coordinate = (target.x, target.y) if target is not None else repr(task.target)
    return (task.kind, coordinate)


def _task_sort_key(task: Task, day: int) -> tuple[Any, ...]:
    urgent = (
        (task.deadline is not None and task.deadline <= day)
        or str(task.kind).upper() == "ANIMAL"
    )
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


def _held_quantity(state: Mapping[str, Any], item: str) -> int | float:
    private = _mapping(state.get("private"))
    if "inventories" in private:
        held = _held_inventory(private.get("inventories"))
        return _safe_quantity(held.get(item, 0))
    inventory = state.get("inventory")
    if isinstance(inventory, Mapping):
        return _safe_quantity(inventory.get(item, 0))
    return 0


def _worker_quantity(state: Mapping[str, Any], worker_index: int, item: str) -> int | float:
    private = _mapping(state.get("private"))
    inventories = private.get("inventories")
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)):
        if 0 <= worker_index < len(inventories):
            inventory = inventories[worker_index]
            if isinstance(inventory, Mapping):
                return _safe_quantity(inventory.get(item, 0))
            if isinstance(inventory, Sequence) and not isinstance(inventory, (str, bytes)):
                return sum(1 for value in inventory if str(value).upper() == item)
    inventory = state.get("inventory")
    return _safe_quantity(inventory.get(item, 0)) if isinstance(inventory, Mapping) else 0


def _worker_carried_quantity(state: Mapping[str, Any], worker_index: int) -> int | float:
    inventories = _mapping(state.get("private")).get("inventories")
    if not isinstance(inventories, Sequence) or isinstance(inventories, (str, bytes)):
        return 0
    if not 0 <= worker_index < len(inventories):
        return 0
    inventory = inventories[worker_index]
    if isinstance(inventory, Mapping):
        return sum(_safe_quantity(quantity) for quantity in inventory.values())
    if isinstance(inventory, Sequence) and not isinstance(inventory, (str, bytes)):
        return sum(1 for item in inventory if isinstance(item, str))
    return 0


def _shed_quantity(state: Mapping[str, Any], item: str) -> int | float:
    private = _mapping(state.get("private"))
    shed = private.get("shed", state.get("shed", {}))
    return _safe_quantity(shed.get(item, 0)) if isinstance(shed, Mapping) else 0


def _shed_route_distance(start: Position, board_size: int) -> int:
    access = [
        point for point in shed_access_tiles(board_size)
        if 0 <= point.x < board_size and 0 <= point.y < board_size
    ]
    return min((distance(start, point) for point in access), default=10**9)


def _task_turn_budget(task: Task, worker: tuple[int, str, Position | None], state: Mapping[str, Any],
                      board_size: int) -> int:
    target = _target_position(task.target)
    start = worker[2]
    if start is None or target is None:
        return 0
    travel = distance(start, target)
    kind = str(task.kind).upper()
    access = [
        point for point in shed_access_tiles(board_size)
        if 0 <= point.x < board_size and 0 <= point.y < board_size
    ]

    worker_index = worker[0]
    worker_has_wheat = _worker_quantity(state, worker_index, "WHEAT") > 0
    private = state.get("private")
    logistics_known = isinstance(private, Mapping) and ("shed" in private or "inventories" in private)

    if kind == "ANIMAL":
        item = str(task.item or "").upper()
        worker_has_animal = _worker_quantity(state, worker_index, item) > 0
        shed_has_animal = _shed_quantity(state, item) > 0
        shed_has_wheat = _shed_quantity(state, "WHEAT") > 0
        if not worker_has_animal and not shed_has_animal:
            return travel + 1 if not logistics_known else 10**9
        if worker_has_animal:
            if worker_has_wheat:
                return travel + 2  # PLACE, then same-tile FEED.
            if shed_has_wheat:
                return min(
                    (distance(start, point) + 1 + distance(point, target) + 2
                     for point in access),
                    default=10**9,
                )
            return 10**9 if logistics_known else travel + 2
        if worker_has_wheat:
            return min(
                (distance(start, point) + 1 + distance(point, target) + 2
                 for point in access),
                default=10**9,
            )
        if shed_has_wheat:
            return min(
                (distance(start, point) + 2 + distance(point, target) + 2
                 for point in access),
                default=10**9,
            )
        return 10**9 if logistics_known else travel + 2

    if kind == "PLANT":
        # A newly planted crop must have one further turn reserved for WATER.
        return travel + 2
    if kind in {"FEED", "FERTILIZE"}:
        item = "WHEAT" if kind == "FEED" else (
            "FERTILIZER" if kind == "FERTILIZE" else str(task.item or "").upper()
        )
        if item and _worker_quantity(state, worker_index, item) <= 0:
            if _shed_quantity(state, item) <= 0:
                # Flat planner fixtures often omit logistics state.  Keep
                # their task-assignment semantics; parsed engine observations
                # always include private shed/inventory data and are handled
                # conservatively below.
                private = state.get("private")
                if isinstance(private, Mapping) and ("shed" in private or "inventories" in private):
                    return 10**9
                return travel + 1
            # Pickup and delivery must use the same shed access tile. Taking
            # independent minima can undercount the route and schedule a
            # feed/fertilize action after the day's deadline.
            delivery = min(
                (distance(start, point) + 1 + distance(point, target) + 1
                 for point in access),
                default=10**9,
            )
            return delivery
    return travel + 1


def _fits_same_day_deadline(task: Task, worker: tuple[int, str, Position | None], state: Mapping[str, Any],
                            board_size: int, day: int) -> bool:
    if task.deadline is None or task.deadline > day:
        return True
    try:
        hour = min(23, max(0, int(_get(state, "hour", 0))))
    except (TypeError, ValueError, OverflowError):
        hour = 0
    remaining_turns = 24 - hour
    return _task_turn_budget(task, worker, state, board_size) <= remaining_turns


def assign_tasks(plan: Iterable[Task], workers: Iterable[Any] | None, state: Any,
                 strategy: StrategySpec | None = None) -> list[WorkerAssignment]:
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
        if not _task_allowed(task, strategy, state):
            continue
        try:
            task_value = float(task.value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not isfinite(task_value) or task_value < 0:
            continue
        key = _task_key(task)
        current = unique.get(key)
        if current is None or _task_sort_key(task, day) < _task_sort_key(current, day):
            unique[key] = task
    tasks = sorted(unique.values(), key=lambda task: _task_sort_key(task, day))
    if strategy is not None:
        crop_cap = max(0, int(strategy.max_crop_units))
        planned_crop_count = _planned_crop_count(state, strategy)
        bounded_tasks = []
        for task in tasks:
            if task.kind == "PLANT":
                if planned_crop_count >= crop_cap:
                    continue
                planned_crop_count += 1
            bounded_tasks.append(task)
        tasks = bounded_tasks
    infos = sorted((_worker_info(worker, index) for index, worker in enumerate(explicit_workers)), key=lambda item: (item[0], item[2].y if item[2] else inf, item[2].x if item[2] else inf))
    if not infos:
        return []

    try:
        hour = min(23, max(0, int(_get(state, "hour", 0))))
    except (TypeError, ValueError, OverflowError):
        hour = 0
    remaining_turns = 24 - hour
    routing_tasks = tuple(tasks)

    def due_water_cannot_wait(task: Task) -> bool:
        if (
            str(task.kind).upper() != "WATER"
            or task.deadline is None
            or task.deadline > day
        ):
            return False
        if not any(
            _task_turn_budget(task, info, state, board_size) <= remaining_turns
            for info in infos
        ):
            return False
        peers = [
            other for other in routing_tasks
            if other is not task
            and str(other.kind).upper() == "WATER"
            and other.deadline is not None
            and other.deadline <= day
            and other.priority == task.priority
        ]
        if not peers:
            return False
        return not any(
            _task_turn_budget(peer, info, state, board_size)
            + _task_turn_budget(
                task,
                (info[0], info[1], _target_position(peer.target)),
                state,
                board_size,
            )
            <= remaining_turns
            for peer in peers
            for info in infos
        )

    def assignment_sort_key(task: Task) -> tuple[Any, ...]:
        base = _task_sort_key(task, day)
        target = _target_position(task.target)
        route_distance = min(
            (
                distance(info[2], target)
                for info in infos
                if info[2] is not None and target is not None
            ),
            default=inf,
        )
        return (
            *base[:3],
            0 if due_water_cannot_wait(task) else 1,
            route_distance,
            base[3],
            base[5],
            base[4],
        )

    tasks.sort(key=assignment_sort_key)
    farmer = next((info for info in infos if info[1] == "FARMER"), None)
    logistics_pending = any(task.kind in _SHED_WORK for task in tasks)
    helper_exists = any(info[1] != "FARMER" for info in infos)
    reserved_basic = next((info for info in infos if info[1] != "FARMER"), infos[0])
    available = {info[0] for info in infos}
    assignments: list[WorkerAssignment] = []
    remaining = list(tasks)
    seed_inventory = state.get("seeds", {})
    seed_inventory = seed_inventory if isinstance(seed_inventory, Mapping) else {}
    allocated_seeds: dict[str, int] = {}

    def feasible_candidates(task: Task) -> list[tuple[int, str, Position | None]]:
        candidates = [
            info for info in infos
            if info[0] in available and _fits_same_day_deadline(task, info, state, board_size, day)
        ]
        if task.kind in {"SELL", "SELL_ALL"}:
            item = str(task.item or "").upper()
            if item:
                candidates = [info for info in candidates if _worker_quantity(state, info[0], item) > 0]
            else:
                candidates = [
                    info for info in candidates
                    if any(
                        product != "FERTILIZER" and _worker_quantity(state, info[0], product) > 0
                        for product in PRODUCTS
                    )
                ]
        return candidates

    def feed_matching_size(feed_tasks: Sequence[Task], worker_ids: set[int]) -> int:
        matched: dict[int, Task] = {}

        def match(feed_task: Task, seen: set[int]) -> bool:
            for candidate in feasible_candidates(feed_task):
                worker_index = candidate[0]
                if worker_index not in worker_ids or worker_index in seen:
                    continue
                seen.add(worker_index)
                previous = matched.get(worker_index)
                if previous is None or match(previous, seen):
                    matched[worker_index] = feed_task
                    return True
            return False

        return sum(match(feed_task, set()) for feed_task in feed_tasks)

    def choose(task: Task) -> tuple[int, str, Position | None] | None:
        candidates = feasible_candidates(task)
        equal_due_feeds = [
            other for other in remaining
            if task.kind == "WATER"
            and other is not task
            and other.kind == "FEED"
            and task.deadline is not None
            and task.deadline <= day
            and other.deadline == task.deadline
            and other.priority == task.priority
        ]
        # Preserve due-feed coverage when multiple workers can cover the
        # competing needs. With one worker, no assignment can satisfy both
        # equal-priority tasks, so retain the documented route tie-breaker
        # instead of reserving a task solely by kind.
        if len(infos) > 1 and equal_due_feeds and candidates:
            feed_capacity = feed_matching_size(equal_due_feeds, available)
            candidates = [
                candidate for candidate in candidates
                if feed_matching_size(
                    equal_due_feeds, available - {candidate[0]},
                ) == feed_capacity
            ]
            if not candidates:
                return None
        non_farmer_available = any(
            info[0] in available
            and info[1] != "FARMER"
            and _fits_same_day_deadline(task, info, state, board_size, day)
            for info in infos
        )
        reserve_farmer = (
            logistics_pending
            and farmer is not None
            and helper_exists
            and task.kind not in _SHED_WORK
            and task.kind != "FERTILIZE"
            and (task.kind not in _BASIC_NEEDS or non_farmer_available)
        )
        if reserve_farmer:
            candidates = [info for info in candidates if info[0] != farmer[0]]
        if task.kind in _BASIC_NEEDS and reserved_basic[0] in available:
            candidates = [info for info in candidates if info[0] == reserved_basic[0]] or candidates
        elif task.kind not in _BASIC_NEEDS and reserved_basic[0] in available and any(item.kind in _BASIC_NEEDS for item in remaining):
            non_reserved = [info for info in candidates if info[0] != reserved_basic[0]]
            candidates = non_reserved or candidates
        target = _target_position(task.target)
        if not candidates:
            return None
        return min(candidates, key=lambda info: (
            0 if task.kind == "SHED" and _worker_carried_quantity(state, info[0]) > 0 else 1,
            distance(info[2], target) if info[2] is not None and target is not None else inf,
            info[0],
            info[2].y if info[2] is not None else inf,
            info[2].x if info[2] is not None else inf,
        ))

    for task in tasks:
        crop = str(task.item or "").upper()
        if (
            task.kind == "PLANT"
            and crop
            and allocated_seeds.get(crop, 0) >= _safe_quantity(seed_inventory.get(crop, 0))
        ):
            continue
        selected = choose(task)
        if selected is None:
            continue
        if task.kind == "PLANT":
            allocated_seeds[crop] = allocated_seeds.get(crop, 0) + 1
        available.remove(selected[0])
        remaining.remove(task)
        assignments.append(WorkerAssignment(
            worker_index=selected[0],
            task=task,
            route=_route_positions(selected[2], _target_position(task.target), board_size),
        ))
    return assignments
