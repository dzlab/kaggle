"""Run reproducible Kaggriculture policy evaluations and summarize replays.

The evaluator deliberately imports the Kaggle engine only when a game is run,
so parsing and replay aggregation remain usable in offline test environments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    ENGINE_VERSION,
    LAND_ORDER,
    LAND_PRICES,
    PRICE_FLOOR,
    PRODUCTS,
    max_market_orders,
    shed_capacity,
)
from kagriculture_agent.economics import market_price  # noqa: E402
from kagriculture_agent.observation import is_shed_adjacent  # noqa: E402
from kagriculture_agent.policy import Policy  # noqa: E402
from scripts.run_local import OPPONENTS, _deterministic_random_agent  # noqa: E402


VARIANTS = ("conservative", "mixed", "melon-heavy", "demand-reactive", "animal-heavy")
ABLATION_COMPONENTS = (
    "route_scheduling",
    "market_batch_sizing",
    "shop_adaptation",
    "land_purchase",
    "animals",
)
_DEFAULT_ABLATIONS = {component: True for component in ABLATION_COMPONENTS}
_BUYABLE_PRODUCTS = frozenset({"WHEAT", "FERTILIZER"})
_PRODUCT_NAMES = frozenset(PRODUCTS)
_ANIMAL_NAMES = frozenset(ANIMALS)
_ITEM_NAMES = _PRODUCT_NAMES | _ANIMAL_NAMES


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _variant_list(values: Sequence[str] | None) -> list[str]:
    selected = list(values or ("mixed",))
    unknown = [value for value in selected if value not in VARIANTS]
    if unknown:
        raise ValueError(f"unsupported variant(s): {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def parse_ablation(value: str) -> tuple[str, bool]:
    """Parse ``component=on|off`` for a component already present in policy."""
    try:
        component, state = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use component=on or component=off") from exc
    if component not in ABLATION_COMPONENTS or state not in {"on", "off"}:
        choices = ", ".join(ABLATION_COMPONENTS)
        raise argparse.ArgumentTypeError(f"component must be one of {choices}; state must be on/off")
    return component, state == "on"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_positive_int, default=30, help="number of consecutive seeds")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=["pass", "random", "starter"])
    parser.add_argument("--steps", type=_positive_int, default=720)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--variant", action="append", dest="single_variants", choices=VARIANTS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--ablation", action="append", type=parse_ablation, default=[], metavar="COMPONENT=on|off")
    parser.add_argument("--quick", action="store_true", help="use a small default batch suitable for local tests")
    args = parser.parse_args(argv)
    args.variants = _variant_list((args.variants or []) + (args.single_variants or []))
    if args.quick:
        if args.seeds == 30:
            args.seeds = 2
        if args.steps == 720:
            args.steps = 96
    return args


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Return an inclusive, linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(percent) / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _average(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0) or 0.0) for record in records]
    return float(mean(values)) if values else 0.0


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate game records into the stable per-variant/opponent schema."""
    records = list(records)
    count = len(records)
    valid_records = [record for record in records if not record.get("framework_error")]
    final_banks = [float(record["final_bank"]) for record in valid_records if record.get("final_bank") is not None]
    outcomes = {outcome: sum(record.get("outcome") == outcome for record in valid_records)
                for outcome in ("win", "loss", "tie")}
    return {
        "count": count,
        "valid_count": len(valid_records),
        "framework_failures": sum(bool(record.get("framework_error")) for record in records),
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "ties": outcomes["tie"],
        "win_rate": outcomes["win"] / len(valid_records) if valid_records else 0.0,
        "mean_final_bank": float(mean(final_banks)) if final_banks else 0.0,
        "median_final_bank": float(median(final_banks)) if final_banks else 0.0,
        "fifth_percentile_final_bank": float(percentile(final_banks, 5)) if final_banks else 0.0,
        "mean_bank_differential": _average(valid_records, "bank_differential"),
        "framework_error_rate": sum(bool(record.get("framework_error")) for record in records) / count if count else 0.0,
        "average_shed_overflow": _average(records, "shed_overflow"),
        "average_price_floor_sales": _average(records, "price_floor_sales"),
        "average_missed_basic_needs_events": _average(records, "missed_basic_needs"),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _config_value(configuration: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    return _mapping(configuration).get(key, default)


def _market_order_limit(configuration: Mapping[str, Any] | None) -> tuple[int, bool]:
    raw = _config_value(configuration, "maxMarketOrdersPerTurn", max_market_orders)
    if isinstance(raw, bool):
        return max_market_orders, False
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return max_market_orders, False
    if value < 1 or (isinstance(raw, float) and not raw.is_integer()):
        return max_market_orders, False
    return value, True


def _fib(index: int) -> int:
    first, second = 1, 1
    for _ in range(max(0, index)):
        first, second = second, first + second
    return first


def _player_states(replay: Mapping[str, Any], player: int) -> list[Mapping[str, Any]]:
    states = []
    for turn in replay.get("steps", ()):
        if not isinstance(turn, Sequence):
            continue
        for state in turn:
            if isinstance(state, Mapping) and _mapping(state.get("observation")).get("player") == player:
                states.append(state)
                break
    return states


def _valid_market_order_schema(order: Any, observation: Mapping[str, Any]) -> bool:
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)) or not order or not isinstance(order[0], str):
        return False
    operation = order[0]
    if operation in {"HIRE", "BUY_LAND"}:
        return len(order) == 1
    if operation not in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
        return False
    if len(order) != 3 or not isinstance(order[1], str) or not isinstance(order[2], int) or isinstance(order[2], bool) or order[2] < 1:
        return False
    item = order[1]
    if operation == "BUY_SEED":
        return item in CROPS
    if operation == "BUY_ANIMAL":
        return item in ANIMALS
    if operation == "BUY_PRODUCT":
        return item in _BUYABLE_PRODUCTS
    if item not in PRODUCTS:
        return False
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    inventory = _number(_mapping(market.get("inventory")).get(item))
    price = _number(prices.get(item))
    if operation == "BUY_PRODUCT" and item not in _BUYABLE_PRODUCTS:
        return False
    return inventory is not None and inventory >= 0 and price is not None and price >= PRICE_FLOOR


def _valid_action_schema(action: Any, observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None,
                         *, state_aware: bool = True) -> bool:
    if not isinstance(action, Mapping) or set(action) != {"farmer", "hands", "market"}:
        return False
    farmer = action.get("farmer")
    hands = action.get("hands")
    orders = action.get("market")
    farm = _farm_observation(observation)
    valid_farmer = (
        _unit_command_valid_for_state(farmer, observation, 0, configuration)
        if state_aware else _legal_unit_command(farmer)
    )
    if not valid_farmer or not isinstance(hands, Sequence) or isinstance(hands, (str, bytes)):
        return False
    expected_hands = farm.get("hands", ())
    if not isinstance(expected_hands, Sequence) or isinstance(expected_hands, (str, bytes)) or len(hands) != len(expected_hands):
        return False
    if state_aware:
        valid_hands = all(_unit_command_valid_for_state(command, observation, index + 1, configuration) for index, command in enumerate(hands))
    else:
        valid_hands = all(_legal_unit_command(command) for command in hands)
    if not valid_hands:
        return False
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        return False
    order_limit, config_valid = _market_order_limit(configuration)
    if not config_valid:
        return False
    if len(orders) > order_limit:
        return False
    if state_aware:
        return _valid_market_orders(orders, observation, configuration)
    return all(_valid_market_order_schema(order, observation) for order in orders)


def _valid_replay(replay: Mapping[str, Any], own_states: Sequence[Mapping[str, Any]],
                  other_states: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any] | None = None) -> bool:
    statuses = replay.get("statuses")
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)) or list(statuses) != ["DONE", "DONE"]:
        return False
    steps = replay.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        return False
    if len(own_states) != len(steps) or len(other_states) != len(steps):
        return False
    previous_observations = {
        0: _mapping(own_states[0].get("observation")),
        1: _mapping(other_states[0].get("observation")),
    }
    for index, turn in enumerate(steps):
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes)):
            return False
        players = set()
        for state in turn:
            if not isinstance(state, Mapping):
                return False
            observation = state.get("observation")
            player = _mapping(observation).get("player")
            if not isinstance(player, int) or player not in {0, 1} or player in players or not isinstance(observation, Mapping):
                return False
            players.add(player)
            if not isinstance(state.get("action"), Mapping) or state.get("status") not in ("ACTIVE", "DONE"):
                return False
            if state.get("error") or _mapping(state.get("info")).get("error"):
                return False
            if not _valid_action_schema(
                state["action"], previous_observations[player], configuration, state_aware=player == 0
            ):
                return False
        if players != {0, 1}:
            return False
        expected_status = "DONE" if index == len(steps) - 1 else "ACTIVE"
        if any(state.get("status") != expected_status for state in turn if isinstance(state, Mapping)):
            return False
        previous_observations = {
            player: _mapping(state.get("observation"))
            for player in (0, 1)
            for state in turn
            if isinstance(state, Mapping) and _mapping(state.get("observation")).get("player") == player
        }
    if _final_bank(own_states[-1]) is None or _final_bank(other_states[-1]) is None:
        return False
    return not bool(_mapping(replay.get("info")).get("error"))


def _final_bank(state: Mapping[str, Any] | None) -> float | None:
    if state is None:
        return None
    observation = _mapping(state.get("observation"))
    player = observation.get("player")
    farms = observation.get("farms")
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)) or not isinstance(player, int):
        return None
    if not 0 <= player < len(farms):
        return None
    return _number(_mapping(farms[player]).get("money"))


def _shed_total(observation: Mapping[str, Any]) -> float:
    private = _mapping(observation.get("private"))
    shed = private.get("shed")
    if not isinstance(shed, Mapping):
        return 0.0
    return sum(max(0.0, _number(quantity) or 0.0) for quantity in shed.values())


def _inventory_total(observation: Mapping[str, Any]) -> float:
    """Return worker inventories that have not yet been dropped into the shed."""
    inventories = _mapping(_mapping(observation.get("private")).get("inventories"))
    if inventories:
        return sum(max(0.0, _number(quantity) or 0.0) for quantity in inventories.values())
    raw_inventories = _mapping(observation.get("private")).get("inventories")
    if not isinstance(raw_inventories, Sequence) or isinstance(raw_inventories, (str, bytes)):
        return 0.0
    return sum(
        max(0.0, _number(quantity) or 0.0)
        for inventory in raw_inventories
        if isinstance(inventory, Mapping)
        for quantity in inventory.values()
    )


def _shed_capacity(configuration: Mapping[str, Any] | None = None) -> int:
    raw = _config_value(configuration, "shedCapacity", shed_capacity)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError, OverflowError):
        return shed_capacity


def _tiles(observation: Mapping[str, Any]) -> Sequence[Any]:
    return tuple(tile for _position, tile in _tile_entries(observation))


def _tile_entries(observation: Mapping[str, Any]) -> Sequence[tuple[tuple[int, int], Any]]:
    farms = observation.get("farms")
    player = observation.get("player")
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)) or not isinstance(player, int):
        return ()
    if not 0 <= player < len(farms):
        return ()
    tiles = _mapping(farms[player]).get("tiles", ())
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return ()
    return tuple(
        ((x, y), tile)
        for y, row in enumerate(tiles)
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        for x, tile in enumerate(row)
    )


def _tile_kind(tile: Any) -> str:
    if isinstance(tile, str):
        return tile.upper()
    return str(_mapping(tile).get("kind", "")).upper()


def _animal_state(tile: Any) -> Mapping[str, Any]:
    """Return the animal fields in either engine or policy-test tile shape."""
    if not isinstance(tile, Mapping):
        return {}
    animal = tile.get("animal")
    if isinstance(animal, Mapping):
        state = dict(tile)
        state.update(animal)
        return state
    if animal is not None or _tile_kind(tile) == "ANIMAL":
        return tile
    return {}


def _commands_for_state(state: Mapping[str, Any]) -> list[list[Any]]:
    action = _mapping(state.get("action"))
    commands = []
    farmer = action.get("farmer")
    if isinstance(farmer, Sequence) and not isinstance(farmer, (str, bytes)):
        commands.append(list(farmer))
    hands = action.get("hands", ())
    if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes)):
        commands.extend(list(command) for command in hands if isinstance(command, Sequence) and not isinstance(command, (str, bytes)))
    return commands


def _command_targets(observation: Mapping[str, Any], state: Mapping[str, Any],
                     configuration: Mapping[str, Any] | None = None) -> dict[str, set[tuple[int, int]]]:
    farm = _farm_observation(observation)
    positions = [farm.get("farmer"), *list(farm.get("hands", ()) or ())]
    commands = _commands_for_state(state)
    targets: dict[str, set[tuple[int, int]]] = {}
    for worker_index, (position, command) in enumerate(zip(positions, commands)):
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 2:
            continue
        if not command or not isinstance(command[0], str):
            continue
        try:
            coordinate = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, OverflowError):
            continue
        if _unit_command_valid_for_state(command, observation, worker_index, configuration):
            targets.setdefault(command[0], set()).add(coordinate)
    return targets


def _post_tile(observation: Mapping[str, Any] | None, coordinate: tuple[int, int]) -> Mapping[str, Any]:
    if not observation:
        return {}
    return next((_mapping(tile) for position, tile in _tile_entries(observation) if position == coordinate), {})


def _missed_needs_at_boundary(observation: Mapping[str, Any], is_boundary: bool,
                              post_observation: Mapping[str, Any] | None = None,
                              action_state: Mapping[str, Any] | None = None,
                              configuration: Mapping[str, Any] | None = None) -> int:
    if (
        not is_boundary
        or post_observation is None
        or _number(observation.get("hour")) != 23
        or _number(post_observation.get("hour")) != 0
    ):
        return 0
    missed = 0
    reset_after_boundary = True
    targets = _command_targets(observation, action_state or {}, configuration)
    for index, (coordinate, tile) in enumerate(_tile_entries(observation)):
        if not isinstance(tile, Mapping):
            continue
        kind = _tile_kind(tile)
        post_tile = _post_tile(post_observation, coordinate)
        if kind == "PLANT":
            needs_water = tile.get("needs_water") is True or tile.get("watered_today") is False
            post_says_watered = post_tile.get("watered_today") is True
            target_satisfied = coordinate in targets.get("WATER", set())
            if needs_water and not post_says_watered and not (reset_after_boundary and target_satisfied) and not (post_observation is None and target_satisfied):
                missed += 1
        animal_state = _animal_state(tile)
        if animal_state:
            needs_feed = animal_state.get("needs_feed") is True or animal_state.get("fed_today") is False
            post_fed = post_tile.get("fed_today") is True
            target_fed = coordinate in targets.get("FEED", set())
            if needs_feed and not post_fed and not (reset_after_boundary and target_fed) and not (post_observation is None and target_fed):
                missed += 1
            needs_care = animal_state.get("needs_care") is True or animal_state.get("cared_today") is False
            post_cared = post_tile.get("cared_today") is True
            target_cared = coordinate in targets.get("CARE", set())
            if needs_care and not post_cared and not (reset_after_boundary and target_cared) and not (post_observation is None and target_cared):
                missed += 1
    return missed


def _price_floor_sales(state: Mapping[str, Any], observation: Mapping[str, Any] | None = None,
                       configuration: Mapping[str, Any] | None = None) -> int:
    observation = observation or _mapping(state.get("observation"))
    market = _mapping(observation.get("market"))
    market_inventory = {
        item: max(0.0, _number(quantity) or 0.0)
        for item, quantity in _mapping(market.get("inventory")).items()
    }
    shed = {
        item: max(0.0, _number(quantity) or 0.0)
        for item, quantity in _mapping(_mapping(observation.get("private")).get("shed")).items()
    }
    action = _mapping(state.get("action"))
    orders = action.get("market", ())
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        return 0
    sales = 0
    for order in orders:
        if not isinstance(order, Sequence) or len(order) < 3 or order[0] != "SELL":
            continue
        item = str(order[1]).upper()
        quantity = max(0, int(_number(order[2]) or 0))
        for _ in range(min(quantity, int(shed.get(item, 0.0)))):
            price = _observed_unit_price(
                item, observation, market_inventory.get(item, 0.0), buying=False,
                configuration=configuration,
            )
            if price == PRICE_FLOOR:
                sales += 1
            shed[item] = shed.get(item, 0.0) - 1
            if price > PRICE_FLOOR:
                market_inventory[item] = market_inventory.get(item, 0.0) + 1
    return sales


def replay_record(replay: Mapping[str, Any], *, variant: str, opponent: str, seed: int) -> dict[str, Any]:
    """Extract one game record from the engine replay JSON."""
    own_states = _player_states(replay, 0)
    other_states = _player_states(replay, 1)
    replay_configuration = _mapping(replay.get("configuration"))
    framework_error = not _valid_replay(replay, own_states, other_states, replay_configuration)
    own_bank = _final_bank(own_states[-1] if own_states else None)
    other_bank = _final_bank(other_states[-1] if other_states else None)
    if framework_error or own_bank is None or other_bank is None:
        outcome = "framework_error"
        differential = 0.0
    elif own_bank > other_bank:
        outcome, differential = "win", own_bank - other_bank
    elif own_bank < other_bank:
        outcome, differential = "loss", own_bank - other_bank
    else:
        outcome, differential = "tie", 0.0

    shed_overflow = 0.0
    floor_sales = 0
    missed_needs = 0
    steps = replay.get("steps", ())
    for index, state in enumerate(own_states):
        post = _mapping(state.get("observation"))
        pre = _mapping(own_states[index - 1].get("observation")) if index else post
        capacity = _shed_capacity(replay_configuration)
        shed_overflow = max(shed_overflow, max(0.0, _shed_total(post) - capacity))
        # Kaggriculture emits a bootstrap record at index 0. Its placeholder
        # action is schema-checked, but it was not chosen from a preceding
        # replay observation and must not contribute action-derived metrics.
        is_bootstrap = index == 0 and len(own_states) > 1
        if not is_bootstrap:
            floor_sales += _price_floor_sales(state, pre, replay_configuration)
        pre_hour = _number(pre.get("hour"))
        post_hour = _number(post.get("hour"))
        is_boundary = pre_hour == 23 and post_hour == 0
        if is_boundary:
            shed_overflow = max(
                shed_overflow,
                max(0.0, _shed_total(pre) + _inventory_total(pre) - capacity),
            )
        if not is_bootstrap:
            missed_needs += _missed_needs_at_boundary(pre, is_boundary, post, state, replay_configuration)
    return {
        "variant": variant,
        "opponent": opponent,
        "seed": seed,
        "outcome": outcome,
        "final_bank": own_bank,
        "opponent_final_bank": other_bank,
        "bank_differential": differential,
        "framework_error": framework_error,
        "shed_overflow": shed_overflow,
        "price_floor_sales": floor_sales,
        "missed_basic_needs": missed_needs,
    }


def _farm_observation(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    farms = observation.get("farms")
    player = observation.get("player")
    if isinstance(farms, Sequence) and not isinstance(farms, (str, bytes)) and isinstance(player, int) and 0 <= player < len(farms):
        return _mapping(farms[player])
    return {}


def _private_seeds(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(observation.get("private")).get("seeds"))


def _cash(observation: Mapping[str, Any]) -> float:
    return _number(_farm_observation(observation).get("money")) or 0.0


def _market_orders(action: Mapping[str, Any]) -> list[list[Any]]:
    orders = action.get("market", ())
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        return []
    return [list(order) for order in orders if isinstance(order, Sequence) and not isinstance(order, (str, bytes))]


def _observed_unit_price(item: str, observation: Mapping[str, Any], inventory: float, *, buying: bool,
                         configuration: Mapping[str, Any] | None = None) -> float:
    market = _mapping(observation.get("market"))
    params = _mapping(_config_value(configuration, "marketParams", {}))
    observed = _number(_mapping(market.get("prices")).get(item))
    quote_inventory = inventory - 1 if buying else inventory
    try:
        quote = float(market_price(item, quote_inventory, params))
        # A replay without configuration may contain a deliberately fixed
        # floor quote. Preserve that valid observation while still recomputing
        # later units from the market curve as inventory changes.
        if not params and observed == PRICE_FLOOR:
            return PRICE_FLOOR
        return quote
    except (KeyError, TypeError, ValueError, OverflowError):
        return observed or 0.0


def _sanitize_market_orders(orders: Sequence[Sequence[Any]], observation: Mapping[str, Any],
                            configuration: Mapping[str, Any] | None = None) -> list[list[Any]]:
    """Apply the engine's per-unit market rules to a postprocessed action."""
    money = _cash(observation)
    shed = dict(_mapping(_mapping(observation.get("private")).get("shed")))
    market = _mapping(observation.get("market"))
    market_inventory = {item: _number(quantity) or 0.0 for item, quantity in _mapping(market.get("inventory")).items()}
    try:
        capacity = max(1, int(_config_value(configuration, "shedCapacity", shed_capacity)))
    except (TypeError, ValueError, OverflowError):
        capacity = shed_capacity
    order_limit, _config_valid = _market_order_limit(configuration)
    farm = _farm_observation(observation)
    hires_today = int(_number(farm.get("hires_today")) or 0)
    hire_mult = max(0, int(_config_value(configuration, "farmHandCostMult", 1)))
    raw_unlocked = farm.get("unlocked_quadrants", ())
    unlocked = list(raw_unlocked) if isinstance(raw_unlocked, Sequence) and not isinstance(raw_unlocked, (str, bytes)) else []
    sanitized: list[list[Any]] = []
    for raw_order in orders:
        if len(sanitized) >= order_limit:
            break
        order = list(raw_order)
        if not _valid_market_order_schema(order, observation):
            continue
        operation = order[0]
        if operation == "HIRE":
            cost = float(_fib(hires_today) * hire_mult)
            if cost > money:
                continue
            money -= cost
            hires_today += 1
            sanitized.append(order)
            continue
        if operation == "BUY_LAND":
            next_index = len(unlocked) - 1
            if next_index < 0 or next_index >= len(LAND_ORDER):
                continue
            expected_land = LAND_ORDER[next_index]
            if expected_land in unlocked or money < LAND_PRICES[next_index]:
                continue
            money -= LAND_PRICES[next_index]
            unlocked.append(expected_land)
            sanitized.append(order)
            continue

        item = order[1]
        requested = order[2]
        if operation == "BUY_SEED":
            unit_cost = float(CROPS[item]["seed"])
            allowed = min(requested, int(money // unit_cost))
            if allowed:
                money -= unit_cost * allowed
                sanitized.append([operation, item, allowed])
            continue
        if operation == "BUY_ANIMAL":
            unit_cost = float(ANIMALS[item]["cost"])
            room = max(0, capacity - int(sum(max(0.0, _number(value) or 0.0) for value in shed.values())))
            allowed = min(requested, room, int(money // unit_cost))
            if allowed:
                money -= unit_cost * allowed
                shed[item] = (_number(shed.get(item)) or 0.0) + allowed
                sanitized.append([operation, item, allowed])
            continue
        if operation == "BUY_PRODUCT":
            allowed = 0
            for _ in range(requested):
                room = capacity - int(sum(max(0.0, _number(value) or 0.0) for value in shed.values()))
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=True, configuration=configuration)
                if room <= 0 or price <= 0 or money < price:
                    break
                money -= price
                market_inventory[item] = market_inventory.get(item, 0.0) - 1
                shed[item] = (_number(shed.get(item)) or 0.0) + 1
                allowed += 1
            if allowed:
                sanitized.append([operation, item, allowed])
            continue
        if operation == "SELL":
            available = int(_number(shed.get(item)) or 0)
            allowed = min(requested, available)
            for _ in range(allowed):
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=False, configuration=configuration)
                money += price
                market_inventory[item] = market_inventory.get(item, 0.0) + (1 if price > PRICE_FLOOR else 0)
            if allowed:
                shed[item] = available - allowed
                sanitized.append([operation, item, allowed])
    return sanitized


def _valid_market_orders(orders: Sequence[Any], observation: Mapping[str, Any],
                         configuration: Mapping[str, Any] | None = None) -> bool:
    """Validate sequential market orders against the pre-action farm state."""
    order_limit, config_valid = _market_order_limit(configuration)
    if not config_valid or len(orders) > order_limit:
        return False
    money = _cash(observation)
    shed = {item: max(0.0, _number(quantity) or 0.0)
            for item, quantity in _mapping(_mapping(observation.get("private")).get("shed")).items()}
    market = _mapping(observation.get("market"))
    market_inventory = {item: max(0.0, _number(quantity) or 0.0)
                        for item, quantity in _mapping(market.get("inventory")).items()}
    try:
        capacity = max(1, int(_config_value(configuration, "shedCapacity", shed_capacity)))
    except (TypeError, ValueError, OverflowError):
        return False
    farm = _farm_observation(observation)
    hires_today = int(_number(farm.get("hires_today")) or 0)
    try:
        hire_mult = int(_config_value(configuration, "farmHandCostMult", 1))
    except (TypeError, ValueError, OverflowError):
        return False
    raw_unlocked = farm.get("unlocked_quadrants", ())
    if not isinstance(raw_unlocked, Sequence) or isinstance(raw_unlocked, (str, bytes)):
        return False
    unlocked = list(raw_unlocked)

    for raw_order in orders:
        order = list(raw_order) if isinstance(raw_order, Sequence) and not isinstance(raw_order, (str, bytes)) else raw_order
        if not _valid_market_order_schema(order, observation):
            return False
        operation = order[0]
        if operation == "HIRE":
            cost = float(_fib(hires_today) * hire_mult)
            if cost > money:
                return False
            money -= cost
            hires_today += 1
            continue
        if operation == "BUY_LAND":
            next_index = len(unlocked) - 1
            if next_index < 0 or next_index >= len(LAND_ORDER):
                return False
            expected_land = LAND_ORDER[next_index]
            if expected_land in unlocked or money < LAND_PRICES[next_index]:
                return False
            money -= LAND_PRICES[next_index]
            unlocked.append(expected_land)
            continue

        item, quantity = order[1], order[2]
        if operation == "BUY_SEED":
            cost = float(CROPS[item]["seed"]) * quantity
            if cost > money:
                return False
            money -= cost
            continue
        if operation == "BUY_ANIMAL":
            cost = float(ANIMALS[item]["cost"]) * quantity
            room = capacity - int(sum(shed.values()))
            if cost > money or quantity > room:
                return False
            money -= cost
            shed[item] = shed.get(item, 0.0) + quantity
            continue
        if operation == "BUY_PRODUCT":
            if market_inventory.get(item, 0.0) < quantity:
                return False
            for _ in range(quantity):
                room = capacity - int(sum(shed.values()))
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=True, configuration=configuration)
                if room < 1 or price <= 0 or money < price:
                    return False
                money -= price
                market_inventory[item] -= 1
                shed[item] = shed.get(item, 0.0) + 1
            continue
        if operation == "SELL":
            if shed.get(item, 0.0) < quantity:
                return False
            for _ in range(quantity):
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=False, configuration=configuration)
                if price <= 0:
                    return False
                shed[item] -= 1
                if price > PRICE_FLOOR:
                    market_inventory[item] = market_inventory.get(item, 0.0) + 1
    return True


def _legal_unit_command(command: Any) -> bool:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command or not isinstance(command[0], str):
        return False
    operation = command[0]
    if operation in {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE"}:
        return len(command) == 1
    if operation == "PLANT":
        return len(command) == 2 and command[1] in CROPS
    if operation in {"PICKUP", "PLACE"}:
        return len(command) in {2, 3} and command[1] in _ITEM_NAMES and (len(command) == 2 or (isinstance(command[2], int) and not isinstance(command[2], bool) and command[2] > 0))
    return False


def _worker_position(observation: Mapping[str, Any], worker_index: int) -> tuple[int, int] | None:
    farm = _farm_observation(observation)
    positions = [farm.get("farmer"), *list(farm.get("hands", ()) or ())]
    if not 0 <= worker_index < len(positions):
        return None
    position = positions[worker_index]
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 2:
        return None
    try:
        return int(position[0]), int(position[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _board_size(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None) -> int:
    configured = _config_value(configuration, "boardSize", None)
    if configured is not None:
        try:
            return max(1, int(configured))
        except (TypeError, ValueError, OverflowError):
            pass
    tiles = _mapping(_farm_observation(observation)).get("tiles", ())
    return max(1, len(tiles) if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) else 1)


def _tile_at_worker(observation: Mapping[str, Any], worker_index: int) -> Any:
    position = _worker_position(observation, worker_index)
    tiles = _mapping(_farm_observation(observation)).get("tiles", ())
    if position is None or not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return None
    x, y = position
    if not 0 <= y < len(tiles) or not isinstance(tiles[y], Sequence) or isinstance(tiles[y], (str, bytes)) or not 0 <= x < len(tiles[y]):
        return None
    return tiles[y][x]


def _worker_inventory(observation: Mapping[str, Any], worker_index: int) -> Mapping[str, Any]:
    inventories = _mapping(observation.get("private")).get("inventories", ())
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)) and 0 <= worker_index < len(inventories):
        return _mapping(inventories[worker_index])
    return {}


def _unit_command_valid_for_state(command: Any, observation: Mapping[str, Any], worker_index: int = 0,
                                  configuration: Mapping[str, Any] | None = None) -> bool:
    if not _legal_unit_command(command):
        return False
    operation = command[0]
    position = _worker_position(observation, worker_index)
    if position is None:
        return False
    if operation in {"PASS", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE"}:
        tile = _tile_at_worker(observation, worker_index)
    else:
        tile = None
    if operation in {"NORTH", "SOUTH", "EAST", "WEST"}:
        deltas = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
        dx, dy = deltas[operation]
        size = _board_size(observation, configuration)
        return 0 <= position[0] + dx < size and 0 <= position[1] + dy < size
    if operation == "PASS":
        return True
    if operation == "PLANT":
        seeds = _private_seeds(observation)
        return tile is None and _number(seeds.get(command[1])) is not None and (_number(seeds.get(command[1])) or 0) >= 1
    if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
        return tile is None
    if operation == "WATER":
        return _tile_kind(tile) == "PLANT" and tile.get("watered_today") is False if isinstance(tile, Mapping) else False
    if operation == "HARVEST":
        return isinstance(tile, Mapping) and _tile_kind(tile) != "LOCKED" and (_number(tile.get("yield_units")) or 0) > 0
    if operation == "DIG":
        return tile is not None and _tile_kind(tile) != "LOCKED" and not (isinstance(tile, Mapping) and "animal" in tile)
    if operation == "FERTILIZE":
        return _tile_kind(tile) == "PLANT" and (_number(_worker_inventory(observation, worker_index).get("FERTILIZER")) or 0) >= 1
    if operation in {"FEED", "CARE", "COLLECT_FERTILIZER"}:
        animal_state = _animal_state(tile)
        if not animal_state:
            return False
        if operation == "FEED":
            return (animal_state.get("fed_today") is False or animal_state.get("needs_feed") is True) and (_number(_worker_inventory(observation, worker_index).get("WHEAT")) or 0) >= 1
        if operation == "CARE":
            return animal_state.get("cared_today") is False or animal_state.get("needs_care") is True
        return animal_state.get("fertilizer_available") is True
    if operation in {"PICKUP", "PLACE", "DROP"}:
        size = _board_size(observation, configuration)
        if not is_shed_adjacent(position, size):
            if operation == "PLACE" and command[1] in _ANIMAL_NAMES:
                pass
            else:
                return False
        inventory = _worker_inventory(observation, worker_index)
        shed = _mapping(_mapping(observation.get("private")).get("shed"))
        if operation == "DROP":
            return any((_number(value) or 0) > 0 for value in inventory.values())
        item = command[1]
        quantity = command[2] if len(command) == 3 else 1
        if operation == "PICKUP":
            return (_number(shed.get(item)) or 0) >= quantity
        if item in _ANIMAL_NAMES:
            return isinstance(tile, Mapping) and tile.get("kind") == ANIMALS[item]["structure"] and tile.get("animal") is None and (_number(inventory.get(item)) or 0) >= 1
        room = max(0, int(_config_value(configuration, "shedCapacity", shed_capacity)) - int(sum((_number(value) or 0) for value in shed.values())))
        return is_shed_adjacent(position, size) and (_number(inventory.get(item)) or 0) >= quantity and room >= quantity
    return False


def _sanitize_action(action: Mapping[str, Any], observation: Mapping[str, Any], fallback: Mapping[str, Any],
                     configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "farmer": list(action.get("farmer", ())) if isinstance(action.get("farmer", ()), Sequence) else [],
        "hands": [list(command) for command in action.get("hands", ())] if isinstance(action.get("hands", ()), Sequence) else [],
        "market": _sanitize_market_orders(_market_orders(action), observation, configuration),
    }
    fallback_farmer = list(fallback.get("farmer", ["PASS"]))
    if not _unit_command_valid_for_state(result["farmer"], observation, 0, configuration):
        result["farmer"] = fallback_farmer if _unit_command_valid_for_state(fallback_farmer, observation, 0, configuration) else ["PASS"]
    expected_hands = _mapping(_farm_observation(observation)).get("hands", ())
    expected_count = len(expected_hands) if isinstance(expected_hands, Sequence) and not isinstance(expected_hands, (str, bytes)) else 0
    fallback_hands = [list(command) for command in fallback.get("hands", ())]
    if len(result["hands"]) != expected_count:
        result["hands"] = fallback_hands if len(fallback_hands) == expected_count else [["PASS"] for _ in range(expected_count)]
    result["hands"] = [command if _unit_command_valid_for_state(command, observation, index + 1, configuration) else (fallback_hands[index] if index < len(fallback_hands) and _unit_command_valid_for_state(fallback_hands[index], observation, index + 1, configuration) else ["PASS"]) for index, command in enumerate(result["hands"])]
    return result


def _best_seed(observation: Mapping[str, Any], *, prefer: str | None = None) -> str | None:
    prices = _mapping(_mapping(observation.get("market")).get("prices"))
    affordable = [
        crop for crop, data in CROPS.items()
        if _cash(observation) >= float(data["seed"]) and (_number(prices.get(crop)) or 0.0) > 0
    ]
    if not affordable:
        return None
    return max(affordable, key=lambda crop: ((1 if crop == prefer else 0), (_number(prices.get(crop)) or 0.0) / CROPS[crop]["seed"], crop))


def _empty_animal_target(observation: Mapping[str, Any]) -> str | None:
    for tile in _tiles(observation):
        if not isinstance(tile, Mapping):
            continue
        kind = _tile_kind(tile)
        if kind not in {"COOP", "PASTURE"} or tile.get("built", True) is False or tile.get("animal") is not None:
            continue
        species = "GOOSE" if kind == "COOP" else "COW"
        if _cash(observation) >= ANIMALS[species]["cost"]:
            return species
    return None


def _apply_ablations(action: Mapping[str, Any], observation: Mapping[str, Any], ablations: Mapping[str, bool]) -> dict[str, Any]:
    result = {"farmer": list(action.get("farmer", ["PASS"])), "hands": [list(command) for command in action.get("hands", ())],
              "market": _market_orders(action)}
    if not ablations.get("route_scheduling", True):
        for key in ("farmer", "hands"):
            commands = result[key] if key == "farmer" else result[key]
            if commands and commands[0] in {"NORTH", "SOUTH", "EAST", "WEST"}:
                result[key] = ["PASS"]
            if key == "hands":
                result[key] = [["PASS"] if command and command[0] in {"NORTH", "SOUTH", "EAST", "WEST"} else command for command in commands]
    if not ablations.get("market_batch_sizing", True):
        result["market"] = result["market"][:1]
    if not ablations.get("shop_adaptation", True):
        result["market"] = [order for order in result["market"] if order[0] != "BUY_PRODUCT"]
    if not ablations.get("land_purchase", True):
        result["market"] = [order for order in result["market"] if order[0] != "BUY_LAND"]
    if not ablations.get("animals", True):
        result["market"] = [order for order in result["market"] if order[0] != "BUY_ANIMAL"]
        for key in ("farmer", "hands"):
            commands = result[key] if key == "farmer" else result[key]
            if key == "farmer":
                if len(commands) > 1 and commands[0] == "PLACE" and commands[1] in ANIMALS:
                    result[key] = ["PASS"]
            else:
                result[key] = [["PASS"] if len(command) > 1 and command[0] == "PLACE" and command[1] in ANIMALS else command for command in commands]
    return result


def apply_variant(action: Mapping[str, Any], observation: Mapping[str, Any], variant: str,
                  ablations: Mapping[str, bool] | None = None,
                  configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply a named, legality-preserving strategy adjustment at the agent boundary."""
    result = {"farmer": list(action.get("farmer", ["PASS"])), "hands": [list(command) for command in action.get("hands", ())],
              "market": _market_orders(action)}
    seeds = _private_seeds(observation)
    if variant == "conservative":
        result["market"] = [order for order in result["market"] if order[0] not in {"BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"}][:1]
    elif variant == "melon-heavy":
        if result["farmer"][:1] == ["PLANT"] and len(result["farmer"]) > 1 and result["farmer"][1] == "WHEAT" and _number(seeds.get("MELON")):
            result["farmer"][1] = "MELON"
        for order in result["market"]:
            if order[0] == "BUY_SEED" and _cash(observation) >= CROPS["MELON"]["seed"]:
                order[1] = "MELON"
                break
        if not any(order[0] == "BUY_SEED" for order in result["market"]) and not _number(seeds.get("MELON")) and _cash(observation) >= CROPS["MELON"]["seed"]:
            result["market"].append(["BUY_SEED", "MELON", 1])
    elif variant == "demand-reactive":
        best = _best_seed(observation)
        if best:
            for order in result["market"]:
                if order[0] == "BUY_SEED" and order[1] in CROPS:
                    order[1] = best
                    break
            if result["farmer"][:1] == ["PLANT"] and len(result["farmer"]) > 1 and _number(seeds.get(best)):
                result["farmer"][1] = best
    elif variant == "animal-heavy":
        target = _empty_animal_target(observation)
        if target:
            for order in result["market"]:
                if order[0] == "BUY_SEED":
                    order[:] = ["BUY_ANIMAL", target, 1]
                    break
            else:
                result["market"].append(["BUY_ANIMAL", target, 1])
    elif variant != "mixed":
        raise ValueError(f"unsupported variant: {variant}")
    adjusted = _apply_ablations(result, observation, ablations or _DEFAULT_ABLATIONS)
    return _sanitize_action(adjusted, observation, action, configuration)


class VariantPolicy:
    """Fresh stateful policy instance with evaluator-only variant adjustments."""

    def __init__(self, variant: str, ablations: Mapping[str, bool] | None = None,
                 configuration: Mapping[str, Any] | None = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        self.variant = variant
        self.ablations = dict(ablations or _DEFAULT_ABLATIONS)
        self.configuration = dict(configuration or {})
        self.policy = Policy()

    def __call__(self, obs: Mapping[str, Any], _configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        configuration = _configuration if _configuration is not None else self.configuration
        return apply_variant(self.policy.act(obs), obs, self.variant, self.ablations, configuration)


def run_game(*, variant: str, opponent: str, seed: int, steps: int, ablations: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Run one seeded game and return its normalized replay record."""
    if opponent not in OPPONENTS:
        raise ValueError(f"unsupported opponent: {opponent}")
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError("kaggle-environments is required for evaluation") from exc
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    opponent_agent = _deterministic_random_agent(seed) if opponent == "random" else opponent
    env.run([VariantPolicy(variant, ablations, env.configuration), opponent_agent])
    return replay_record(env.toJSON(), variant=variant, opponent=opponent, seed=seed)


def run_matrix(*, variants: Sequence[str], opponents: Sequence[str], seeds: Sequence[int], steps: int,
               ablations: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Run the Cartesian product in stable input order with identical seeds."""
    variants = _variant_list(variants)
    opponents = list(opponents)
    invalid = [opponent for opponent in opponents if opponent not in OPPONENTS]
    if invalid:
        raise ValueError(f"unsupported opponent(s): {', '.join(invalid)}")
    records = []
    for variant in variants:
        for opponent in opponents:
            for seed in seeds:
                kwargs = {"variant": variant, "opponent": opponent, "seed": int(seed), "steps": steps}
                if ablations is not None:
                    kwargs["ablations"] = ablations
                records.append(run_game(**kwargs))
    return {"records": records}


def run_evaluation(*, variants: Sequence[str], opponents: Sequence[str], seeds: Sequence[int], steps: int,
                   ablations: Sequence[tuple[str, bool]] = ()) -> dict[str, Any]:
    """Run a baseline and isolated one-component ablations.

    Every ablation starts from the same all-enabled baseline. Repeating a
    component is rejected instead of being silently merged into a combined
    configuration.
    """
    requested = list(ablations)
    components = [component for component, _enabled in requested]
    if len(components) != len(set(components)):
        raise ValueError("each ablation component may be requested only once")
    baseline_config = dict(_DEFAULT_ABLATIONS)
    baseline = run_matrix(variants=variants, opponents=opponents, seeds=seeds, steps=steps, ablations=baseline_config)["records"]
    ablation_records: dict[str, list[dict[str, Any]]] = {}
    ablation_configs: dict[str, dict[str, bool]] = {}
    for component, enabled in requested:
        config = dict(baseline_config)
        config[component] = enabled
        ablation_configs[component] = config
        ablation_records[component] = run_matrix(
            variants=variants, opponents=opponents, seeds=seeds, steps=steps, ablations=config
        )["records"]
    return {"records": baseline, "ablation_records": ablation_records, "ablation_configs": ablation_configs}


def _group_results(records: Sequence[Mapping[str, Any]], variants: Sequence[str], opponents: Sequence[str]) -> dict[str, Any]:
    return {
        variant: {
            opponent: aggregate_records([
                record for record in records if record.get("variant") == variant and record.get("opponent") == opponent
            ])
            for opponent in opponents
        }
        for variant in variants
    }


def _select_default(records: Sequence[Mapping[str, Any]], variants: Sequence[str]) -> str:
    scored = []
    for variant in variants:
        summary = aggregate_records([record for record in records if record.get("variant") == variant])
        scored.append((variant, summary))
    if not scored:
        return "mixed"
    return min(scored, key=lambda item: (
        -item[1]["win_rate"],
        -item[1]["median_final_bank"],
        item[1]["framework_error_rate"],
        item[0],
    ))[0]


def build_result_document(*, config: Mapping[str, Any], records: Sequence[Mapping[str, Any]], command: Sequence[str] | None = None,
                          ablation_records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                          ablation_configs: Mapping[str, Mapping[str, bool]] | None = None) -> dict[str, Any]:
    variants = list(config.get("variants", ()))
    opponents = list(config.get("opponents", ()))
    results = _group_results(records, variants, opponents)
    ablations = {}
    for component, component_records in (ablation_records or {}).items():
        component_results = _group_results(component_records, variants, opponents)
        contribution = {}
        for variant in variants:
            contribution[variant] = {}
            for opponent in opponents:
                baseline = results[variant][opponent]
                ablated = component_results[variant][opponent]
                contribution[variant][opponent] = {
                    "win_rate_delta": ablated["win_rate"] - baseline["win_rate"],
                    "median_final_bank_delta": ablated["median_final_bank"] - baseline["median_final_bank"],
                    "framework_error_rate_delta": ablated["framework_error_rate"] - baseline["framework_error_rate"],
                }
        ablations[component] = {
            "config": dict((ablation_configs or {}).get(component, {})),
            "results": component_results,
            "contribution": contribution,
        }
    return {
        "schema_version": 1,
        "metadata": {
            "command": list(command) if command is not None else [],
            "config": dict(config),
            "engine": "kaggle-environments",
            "engine_version": ENGINE_VERSION,
            "replay_summary": config.get("replay_summary"),
        },
        "selected_default": _select_default(records, variants),
        "results": results,
        "ablations": ablations,
    }


def write_result_document(path: str | Path, document: Mapping[str, Any], *, records: Sequence[Mapping[str, Any]],
                          ablation_records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> Path:
    """Write the report and deterministic compact replay-record sidecar."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    sidecar = report_path.with_name(f"{report_path.stem}.replays.json")
    sidecar_records = [{"ablation": "baseline", **dict(record)} for record in records]
    for component, component_records in (ablation_records or {}).items():
        sidecar_records.extend({"ablation": component, **dict(record)} for record in component_records)
    sidecar_records.sort(key=lambda record: (
        str(record.get("ablation", "")), str(record.get("variant", "")),
        str(record.get("opponent", "")), int(record.get("seed", 0)),
    ))
    sidecar.write_text(json.dumps({"schema_version": 1, "records": sidecar_records}, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    sidecar = output.with_name(f"{output.stem}.replays.json")
    config = {
        "seeds": args.seeds,
        "start_seed": args.start_seed,
        "seed_values": seeds,
        "steps": args.steps,
        "opponents": list(args.opponents),
        "variants": list(args.variants),
        "ablations": [f"{component}={'on' if enabled else 'off'}" for component, enabled in args.ablation],
        "replay_summary": str(sidecar.relative_to(PROJECT_ROOT)) if sidecar.is_relative_to(PROJECT_ROOT) else str(sidecar),
        "quick": args.quick,
    }
    evaluation = run_evaluation(variants=args.variants, opponents=args.opponents, seeds=seeds, steps=args.steps, ablations=args.ablation)
    document = build_result_document(
        config=config, records=evaluation["records"],
        command=["scripts/evaluate.py", *([*sys.argv[1:]] if argv is None else argv)],
        ablation_records=evaluation["ablation_records"], ablation_configs=evaluation["ablation_configs"],
    )
    write_result_document(output, document, records=evaluation["records"], ablation_records=evaluation["ablation_records"])
    games = len(evaluation["records"]) + sum(len(records) for records in evaluation["ablation_records"].values())
    print(json.dumps({"output": str(output), "selected_default": document["selected_default"], "games": games}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
