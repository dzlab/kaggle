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
    SHOPS,
    max_market_orders,
    season_days,
    shed_capacity,
)
from kagriculture_agent.economics import market_price  # noqa: E402
from kagriculture_agent.observation import is_shed_adjacent  # noqa: E402
from kagriculture_agent.planner import _has_basic_need_deadline  # noqa: E402
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


def _crop_data(crop: Any) -> Mapping[str, Any] | None:
    """Return crop rules without allowing malformed replay keys to escape."""
    if not isinstance(crop, str):
        return None
    return CROPS.get(crop)


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
    steps = replay.get("steps", ())
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return states
    for turn in steps:
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
                         *, state_aware: bool = True, validate_market: bool = True) -> bool:
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
    if not validate_market:
        return all(_valid_market_order_schema(order, observation) for order in orders)
    if state_aware:
        return _valid_market_orders(orders, observation, configuration)
    return all(_valid_market_order_schema(order, observation) for order in orders)


def _valid_replay(replay: Mapping[str, Any], own_states: Sequence[Mapping[str, Any]],
                  other_states: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any] | None = None,
                  *, expected_seed: int | None = None) -> bool:
    if not isinstance(replay, Mapping):
        return False
    if not _valid_engine_provenance(replay):
        return False
    legacy_compact = _legacy_compact_fixture(replay)
    for optional_mapping in ("metadata",):
        if optional_mapping in replay and not isinstance(replay[optional_mapping], Mapping):
            return False
    info = replay.get("info")
    metadata = replay.get("metadata")
    if expected_seed is not None:
        if not isinstance(info, Mapping) or info.get("seed") != expected_seed:
            return False
        if isinstance(metadata, Mapping) and "seed" in metadata and metadata.get("seed") != expected_seed:
            return False
        configured_seed = _mapping(configuration).get("seed")
        if configured_seed is not None and configured_seed != expected_seed:
            return False
    statuses = replay.get("statuses")
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)) or list(statuses) != ["DONE", "DONE"]:
        return False
    steps = replay.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        return False
    rewards = replay.get("rewards")
    if rewards is not None:
        if not isinstance(rewards, Sequence) or isinstance(rewards, (str, bytes)) or len(rewards) != 2:
            return False
        if any(_number(reward) is None for reward in rewards):
            return False
    if not legacy_compact:
        episode_steps = _number(_config_value(configuration, "episodeSteps", None))
        if episode_steps is None or int(episode_steps) != episode_steps or len(steps) != int(episode_steps):
            return False
    if len(own_states) != len(steps) or len(other_states) != len(steps):
        return False
    previous_observations = {
        0: _mapping(own_states[0].get("observation")),
        1: _mapping(other_states[0].get("observation")),
    }
    for index, turn in enumerate(steps):
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes)) or len(turn) != 2:
            return False
        players = set()
        states_by_player = {}
        for state in turn:
            if not isinstance(state, Mapping):
                return False
            observation = state.get("observation")
            player = _mapping(observation).get("player")
            if not isinstance(player, int) or player not in {0, 1} or player in players or not isinstance(observation, Mapping):
                return False
            players.add(player)
            states_by_player[player] = state
            if not isinstance(state.get("action"), Mapping) or state.get("status") not in ("ACTIVE", "DONE"):
                return False
            if "info" in state and not isinstance(state["info"], Mapping):
                return False
            if state.get("error") or _mapping(state.get("info")).get("error"):
                return False
            if not legacy_compact and not _strict_private_inventories(observation):
                return False
            if not _valid_action_schema(
                state["action"], previous_observations[player], configuration,
                state_aware=player == 0, validate_market=False,
            ):
                return False
        if players != {0, 1}:
            return False
        actions = [states_by_player[player]["action"] for player in (0, 1)]
        observations = [previous_observations[player] for player in (0, 1)]
        current_observations = [
            _mapping(states_by_player[player].get("observation")) for player in (0, 1)
        ]
        if index > 0 and any(
            not _time_progression_valid(
                observations[player], current_observations[player], configuration,
                allow_missing=legacy_compact, allow_missing_step=player == 1,
            )
            for player in (0, 1)
        ):
            return False
        if not legacy_compact and any(
            not _board_dimensions_valid(observation, configuration)
            for observation in (*observations, *current_observations)
        ):
            return False
        market_result = _simulate_market_orders_lockstep(
            [actions[0].get("market", ()), actions[1].get("market", ())], observations, configuration,
            force_model=bool(actions[1].get("market", ())),
        )
        if market_result is None:
            return False
        if index > 0:
            for player in (0, 1):
                if not _transition_effects_valid(
                    observations[player], current_observations[player], actions[player], configuration,
                    market_result=market_result, player_index=player, allow_compact=legacy_compact,
                    market_observation=observations[0], allow_invalid_unit_noop=player == 1,
                ):
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
    final_banks = [_final_bank(own_states[-1]), _final_bank(other_states[-1])]
    if any(bank is None for bank in final_banks):
        return False
    if not legacy_compact:
        if rewards is None or any(final_banks[player] != _number(rewards[player]) for player in (0, 1)):
            return False
        if _requires_full_liquidation(configuration, len(steps)):
            final_inventories = [
                _private_inventories(_mapping(states[-1].get("observation")))
                for states in (own_states, other_states)
            ]
            if any(inventories is None or any(inventory for inventory in inventories) for inventories in final_inventories):
                return False
    else:
        final_inventories = _private_inventories(_mapping(own_states[-1].get("observation")))
        if final_inventories is None or any(inventory for inventory in final_inventories):
            return False
    return not bool(_mapping(replay.get("info")).get("error"))


def _legacy_compact_fixture(replay: Mapping[str, Any]) -> bool:
    metadata = replay.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("legacy_compact_fixture") is True


def _requires_full_liquidation(configuration: Mapping[str, Any] | None, step_count: int) -> bool:
    """Require terminal cleanup only for a complete configured season."""
    episode_steps = _number(_config_value(configuration, "episodeSteps", None))
    turns = _number(_config_value(configuration, "turnsPerDay", None))
    if (episode_steps is None or turns is None or int(episode_steps) != episode_steps
            or int(turns) != turns or int(turns) < 1):
        return False
    return int(episode_steps) == step_count and int(episode_steps) >= int(turns) * season_days


def _time_progression_valid(pre: Mapping[str, Any], post: Mapping[str, Any],
                            configuration: Mapping[str, Any] | None, *, allow_missing: bool = False,
                            allow_missing_step: bool = False) -> bool:
    """Validate one player's deterministic step/hour/day progression."""
    pre_step, post_step = _number(pre.get("step")), _number(post.get("step"))
    pre_hour, post_hour = _number(pre.get("hour")), _number(post.get("hour"))
    pre_day, post_day = _number(pre.get("day")), _number(post.get("day"))
    if pre_step is None or post_step is None:
        if not allow_missing_step or pre_step is not None or post_step is not None:
            return allow_missing
    if pre_hour is None or post_hour is None:
        return allow_missing
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", None))
    if turns_per_day is None or int(turns_per_day) != turns_per_day or int(turns_per_day) < 1:
        return False
    if any(int(value) != value for value in (pre_hour, post_hour)):
        return False
    if pre_step is not None and post_step is not None and any(
        int(value) != value for value in (pre_step, post_step)
    ):
        return False
    if pre_step is not None and post_step is not None and int(post_step) != int(pre_step) + 1:
        return False
    expected_hour = (int(pre_hour) + 1) % int(turns_per_day)
    if int(post_hour) != expected_hour:
        return False
    if pre_day is None or post_day is None:
        return allow_missing and pre_day is None and post_day is None
    if any(int(value) != value for value in (pre_day, post_day)):
        return False
    expected_day = int(pre_day) + (1 if expected_hour == 0 else 0)
    return int(post_day) == expected_day


def _observation_step(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None) -> int:
    step = _number(observation.get("step"))
    if step is not None and int(step) == step:
        return int(step)
    day = _number(observation.get("day"))
    hour = _number(observation.get("hour"))
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", 24))
    if day is not None and hour is not None and turns_per_day is not None and int(turns_per_day) >= 1:
        return int(day) * int(turns_per_day) + int(hour)
    return 0


def _valid_engine_provenance(replay: Mapping[str, Any]) -> bool:
    """Require the replay envelope emitted by the Kaggriculture engine."""
    required_text = ("id", "name", "version", "module_version", "title", "description")
    if any(not isinstance(replay.get(key), str) or not replay[key] for key in required_text):
        return False
    if replay.get("name") != "kaggriculture" or replay.get("module_version") != ENGINE_VERSION:
        return False
    schema_version = replay.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        return False
    info = replay.get("info")
    if not isinstance(info, Mapping) or isinstance(info.get("seed"), bool) or not isinstance(info.get("seed"), int):
        return False
    specification = replay.get("specification")
    if not isinstance(specification, Mapping):
        return False
    action_spec = specification.get("action")
    agents = specification.get("agents")
    config_spec = specification.get("configuration")
    if not isinstance(action_spec, Mapping) or not isinstance(agents, Sequence) or isinstance(agents, (str, bytes)):
        return False
    if list(agents) != [2] or not isinstance(config_spec, Mapping):
        return False
    configuration = replay.get("configuration")
    if not isinstance(configuration, Mapping):
        return False
    integer_config = {
        "episodeSteps": 1,
        "boardSize": 4,
        "startingMoney": 0,
        "maxMarketOrdersPerTurn": 1,
        "turnsPerDay": 1,
        "shedCapacity": 1,
        "townShopUnlockInterval": 1,
        "townShopSellInterval": 1,
        "townCenterSellInterval": 1,
        "farmHandCostMult": 0,
    }
    for key, minimum in integer_config.items():
        value = configuration.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return False
    for key in ("actTimeout", "runTimeout", "weedSpawnChance"):
        value = _number(configuration.get(key))
        if value is None or value < 0:
            return False
    seed = configuration.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        return False
    if not isinstance(configuration.get("marketParams"), Mapping):
        return False
    return True


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
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", 24))
    last_hour = int(turns_per_day) - 1 if turns_per_day is not None and int(turns_per_day) >= 1 else 23
    if not is_boundary or post_observation is None or _number(observation.get("hour")) != last_hour or _number(post_observation.get("hour")) != 0:
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
            # CARE is an optional production bonus, not a basic need.  It is
            # intentionally excluded from this required-needs metric.
    return missed


def _price_floor_sales(state: Mapping[str, Any], observation: Mapping[str, Any] | None = None,
                       configuration: Mapping[str, Any] | None = None,
                       *, other_state: Mapping[str, Any] | None = None,
                       other_observation: Mapping[str, Any] | None = None) -> int:
    observation = observation or _mapping(state.get("observation"))
    action = _mapping(state.get("action"))
    other_action = _mapping((other_state or {}).get("action"))
    result = _simulate_market_orders_lockstep(
        [action.get("market", ()), other_action.get("market", ())],
        [observation, other_observation or observation],
        configuration,
        force_model=bool(other_action.get("market", ())),
    )
    return int(result["floor_sales"][0]) if result is not None else 0


def _framework_error_record(*, variant: str, opponent: str, seed: int) -> dict[str, Any]:
    return {
        "variant": variant,
        "opponent": opponent,
        "seed": seed,
        "outcome": "framework_error",
        "final_bank": None,
        "opponent_final_bank": None,
        "bank_differential": 0.0,
        "framework_error": True,
        "shed_overflow": 0.0,
        "price_floor_sales": 0,
        "missed_basic_needs": 0,
    }


def _replay_record(replay: Mapping[str, Any], *, variant: str, opponent: str, seed: int) -> dict[str, Any]:
    """Extract one game record from the engine replay JSON."""
    if not isinstance(replay, Mapping):
        return _framework_error_record(variant=variant, opponent=opponent, seed=seed)
    own_states = _player_states(replay, 0)
    other_states = _player_states(replay, 1)
    replay_configuration = _mapping(replay.get("configuration"))
    framework_error = not _valid_replay(
        replay, own_states, other_states, replay_configuration, expected_seed=seed,
    )
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
    turns_per_day = _number(_config_value(replay_configuration, "turnsPerDay", 24))
    last_hour = int(turns_per_day) - 1 if turns_per_day is not None and int(turns_per_day) >= 1 else 23
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
            other_state = other_states[index] if index < len(other_states) else {}
            other_pre = _mapping(other_states[index - 1].get("observation")) if index else _mapping(other_state.get("observation"))
            floor_sales += _price_floor_sales(
                state, pre, replay_configuration,
                other_state=other_state, other_observation=other_pre,
            )
        pre_hour = _number(pre.get("hour"))
        post_hour = _number(post.get("hour"))
        is_boundary = pre_hour == last_hour and post_hour == 0
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


def replay_record(replay: Mapping[str, Any], *, variant: str, opponent: str, seed: int) -> dict[str, Any]:
    """Extract one replay record, classifying malformed replay data safely."""
    try:
        return _replay_record(replay, variant=variant, opponent=opponent, seed=seed)
    except TypeError:
        return _framework_error_record(variant=variant, opponent=opponent, seed=seed)


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
                         configuration: Mapping[str, Any] | None = None,
                         force_model: bool = False) -> float:
    market = _mapping(observation.get("market"))
    params = _mapping(_config_value(configuration, "marketParams", {}))
    observed = _number(_mapping(market.get("prices")).get(item))
    quote_inventory = inventory - 1 if buying else inventory
    try:
        quote = float(market_price(item, quote_inventory, params))
        # A replay without configuration may contain a deliberately fixed
        # floor quote. Preserve that valid observation while still recomputing
        # later units from the market curve as inventory changes.
        if not force_model and not params and observed == PRICE_FLOOR:
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


def _market_sim_state(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    farm = _farm_observation(observation)
    raw_unlocked = farm.get("unlocked_quadrants", ())
    if not isinstance(raw_unlocked, Sequence) or isinstance(raw_unlocked, (str, bytes)):
        return None
    private = _mapping(observation.get("private"))
    raw_shed = private.get("shed", {})
    if not isinstance(raw_shed, Mapping):
        return None
    return {
        "money": _cash(observation),
        "shed": {item: max(0.0, _number(quantity) or 0.0) for item, quantity in raw_shed.items()},
        "seeds": {item: max(0.0, _number(quantity) or 0.0) for item, quantity in _mapping(private.get("seeds")).items()},
        "hires": int(_number(farm.get("hires_today")) or 0),
        "unlocked": list(raw_unlocked),
    }


def _simulate_market_orders_lockstep(orders_by_player: Sequence[Any], observations: Sequence[Mapping[str, Any]],
                                     configuration: Mapping[str, Any] | None = None,
                                     *, force_model: bool = False) -> dict[str, Any] | None:
    """Simulate both market queues against one shared inventory snapshot."""
    if len(orders_by_player) != 2 or len(observations) != 2:
        return None
    order_limit, config_valid = _market_order_limit(configuration)
    if not config_valid:
        return None
    states = [_market_sim_state(observation) for observation in observations]
    if any(state is None for state in states):
        return None
    market = _mapping(observations[0].get("market"))
    market_inventory = {
        item: max(0.0, _number(quantity) or 0.0)
        for item, quantity in _mapping(market.get("inventory")).items()
    }
    if _mapping(observations[1].get("market")).get("inventory") != _mapping(market.get("inventory")):
        return None
    queues = []
    for player, raw_orders in enumerate(orders_by_player):
        if not isinstance(raw_orders, Sequence) or isinstance(raw_orders, (str, bytes)) or len(raw_orders) > order_limit:
            return None
        queue = []
        for raw_order in raw_orders:
            order = list(raw_order) if isinstance(raw_order, Sequence) and not isinstance(raw_order, (str, bytes)) else raw_order
            if not _valid_market_order_schema(order, observations[player]):
                return None
            queue.append(order)
        queues.append(queue)

    capacity = _shed_capacity(configuration)
    try:
        hire_mult = int(_config_value(configuration, "farmHandCostMult", 1))
    except (TypeError, ValueError, OverflowError):
        return None
    floor_sales = [0, 0]
    for order_index in range(max((len(queue) for queue in queues), default=0)):
        active = [queue[order_index] if order_index < len(queue) else None for queue in queues]
        # HIRE and BUY_LAND are atomic orders, handled before per-unit quotes.
        for player, order in enumerate(active):
            if order is None:
                continue
            operation = order[0]
            state = states[player]
            if operation == "HIRE":
                cost = _fib(state["hires"]) * hire_mult
                if cost > state["money"]:
                    if player == 0:
                        return None
                    active[player] = None
                    continue
                state["money"] -= cost
                state["hires"] += 1
                continue
            if operation == "BUY_LAND":
                next_index = len(state["unlocked"]) - 1
                if next_index < 0 or next_index >= len(LAND_ORDER) or state["money"] < LAND_PRICES[next_index]:
                    if player == 0:
                        return None
                    active[player] = None
                    continue
                state["money"] -= LAND_PRICES[next_index]
                state["unlocked"].append(LAND_ORDER[next_index])
                continue

        remaining = [order[2] if order is not None and order[0] not in {"HIRE", "BUY_LAND"} else 0 for order in active]
        while any(remaining):
            quotes = [None, None]
            for player, order in enumerate(active):
                if order is None or remaining[player] <= 0:
                    continue
                operation, item = order[0], order[1]
                if operation == "BUY_SEED":
                    quotes[player] = float(CROPS[item]["seed"])
                elif operation == "BUY_ANIMAL":
                    quotes[player] = float(ANIMALS[item]["cost"])
                elif operation == "BUY_PRODUCT":
                    quotes[player] = _observed_unit_price(
                        item, observations[0], market_inventory.get(item, 0.0), buying=True,
                        configuration=configuration, force_model=force_model,
                    )
                elif operation == "SELL":
                    quotes[player] = _observed_unit_price(
                        item, observations[0], market_inventory.get(item, 0.0), buying=False,
                        configuration=configuration, force_model=force_model,
                    )
            if all(quote is None for quote in quotes):
                break
            committed = False
            for player, order in enumerate(active):
                if order is None or remaining[player] <= 0 or quotes[player] is None:
                    continue
                operation, item, price = order[0], order[1], quotes[player]
                state = states[player]
                shed_total = sum(state["shed"].values())
                if operation == "SELL":
                    if state["shed"].get(item, 0.0) < 1:
                        if player == 0:
                            return None
                        active[player] = None
                        continue
                    state["shed"][item] -= 1
                    state["money"] += price
                    if price > PRICE_FLOOR:
                        market_inventory[item] = market_inventory.get(item, 0.0) + 1
                    else:
                        floor_sales[player] += 1
                elif operation == "BUY_PRODUCT":
                    if market_inventory.get(item, 0.0) < 1 or state["money"] < price or shed_total >= capacity:
                        if player == 0:
                            return None
                        active[player] = None
                        continue
                    state["money"] -= price
                    state["shed"][item] = state["shed"].get(item, 0.0) + 1
                    market_inventory[item] -= 1
                elif operation == "BUY_SEED":
                    if state["money"] < price:
                        if player == 0:
                            return None
                        active[player] = None
                        continue
                    state["money"] -= price
                    state["seeds"][item] = state["seeds"].get(item, 0.0) + 1
                elif operation == "BUY_ANIMAL":
                    if state["money"] < price or shed_total >= capacity:
                        if player == 0:
                            return None
                        active[player] = None
                        continue
                    state["money"] -= price
                    state["shed"][item] = state["shed"].get(item, 0.0) + 1
                remaining[player] -= 1
                committed = True
            if not committed:
                break
    return {"states": states, "market_inventory": market_inventory, "floor_sales": floor_sales}


def _valid_market_orders_lockstep(orders_by_player: Sequence[Any], observations: Sequence[Mapping[str, Any]],
                                  configuration: Mapping[str, Any] | None = None) -> bool:
    other_orders = orders_by_player[1] if len(orders_by_player) > 1 else ()
    return _simulate_market_orders_lockstep(
        orders_by_player, observations, configuration, force_model=bool(other_orders),
    ) is not None


def _legal_unit_command(command: Any) -> bool:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command or not isinstance(command[0], str):
        return False
    operation = command[0]
    if operation in {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE"}:
        return len(command) == 1
    if operation == "PLANT":
        return len(command) == 2 and isinstance(command[1], str) and command[1] in CROPS
    if operation in {"PICKUP", "PLACE"}:
        return (len(command) in {2, 3} and isinstance(command[1], str) and command[1] in _ITEM_NAMES
                and (len(command) == 2 or (isinstance(command[2], int) and not isinstance(command[2], bool) and command[2] > 0)))
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


def _shed_access_positions(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _default_spawn_position(board_size: int) -> tuple[int, int]:
    """Mirror the engine's deterministic farmer reset position."""
    half = board_size // 2
    for position in _shed_access_positions(board_size):
        if position[0] >= 0 and position[1] >= 0 and position[0] < half and position[1] < half:
            return position
    return (0, 0)


def _spawn_hand_position(farm: Mapping[str, Any], board_size: int,
                         existing_hands: Sequence[Sequence[int]] | None = None) -> tuple[int, int]:
    """Mirror the engine's first-free, least-occupied shed-access spawn rule."""
    access = _shed_access_positions(board_size)
    occupants = {position: 0 for position in access}
    farmer = farm.get("farmer")
    positions = [farmer, *(existing_hands if existing_hands is not None else farm.get("hands", ()))]
    for raw_position in positions:
        if isinstance(raw_position, Sequence) and not isinstance(raw_position, (str, bytes)) and len(raw_position) == 2:
            try:
                position = (int(raw_position[0]), int(raw_position[1]))
            except (TypeError, ValueError, OverflowError):
                continue
            if position in occupants:
                occupants[position] += 1
    return min(access, key=lambda position: (occupants[position], access.index(position)))


def _board_value(observation: Mapping[str, Any], position: tuple[int, int]) -> Any:
    return _tile_at_position(observation, position)


def _daily_refresh_tile(before: Any, day: int, step: int, turns_per_day: int) -> Any:
    """Model the deterministic part of the engine's end-of-day tile refresh.

    Weed spawning is seeded elsewhere and is therefore represented separately as
    the only allowed ``None -> WEED`` transition.  All mutations to an existing
    plant or animal are calculated from the pre-action tile, matching the engine's
    refresh order (decay, then daily refresh).
    """
    if before is None:
        return None
    if before == "LOCKED" or _tile_kind(before) == "WEED":
        return before
    if not isinstance(before, Mapping):
        return before

    expected = dict(before)
    kind = _tile_kind(before)
    if kind == "PLANT":
        crop = before.get("crop")
        crop_data = _crop_data(crop)
        yield_units = _number(before.get("yield_units"))
        lifespan = _number(before.get("max_lifespan_step"))
        if crop_data is None or yield_units is None or lifespan is None:
            return object()
        # _decay_plants runs immediately before the day boundary refresh.
        if lifespan >= 0 and step >= lifespan and int(step - lifespan) % 2 == 0:
            yield_units -= 1
            if yield_units <= 0:
                return {"kind": "WEED"}
            expected["yield_units"] = yield_units

        watered = before.get("watered_today")
        consecutive = _number(before.get("consecutive_unwatered"))
        if not isinstance(watered, bool) or consecutive is None or int(consecutive) != consecutive:
            return object()
        consecutive = 0 if watered else int(consecutive) + 1
        expected["consecutive_unwatered"] = consecutive
        expected["watered_today"] = False
        if consecutive >= 2:
            return {"kind": "WEED"}
        if not crop_data["ongoing"]:
            return expected
        planted_day = _number(before.get("planted_day"))
        if planted_day is None or int(planted_day) != planted_day:
            return object()
        next_day = day + 1
        days_since_first = next_day - int(planted_day) - int(crop_data["first_yield_day"])
        interval = int(crop_data["interval"])
        if days_since_first >= 0 and days_since_first % interval == 0:
            production_count = days_since_first // interval + 1
            if production_count <= int(crop_data["max_yield"]):
                fertilized_until = _number(before.get("fertilized_until_day", -1))
                if fertilized_until is None:
                    return object()
                bonus = 2 if watered and fertilized_until >= day else 1
                expected["yield_units"] = min(int(crop_data["max_yield"]), int(yield_units) + bonus)
                if production_count == int(crop_data["max_yield"]):
                    expected["max_lifespan_step"] = (next_day + 1) * turns_per_day
        return expected

    if kind in {"COOP", "PASTURE"} and "animal" in before:
        animal = before.get("animal")
        animal_data = ANIMALS.get(animal)
        fed = before.get("fed_today")
        cared = before.get("cared_today")
        consecutive = _number(before.get("consecutive_unfed"))
        placed_day = _number(before.get("placed_day"))
        yield_units = _number(before.get("yield_units"))
        if (animal_data is None or not isinstance(fed, bool) or not isinstance(cared, bool)
                or consecutive is None or int(consecutive) != consecutive or placed_day is None
                or int(placed_day) != placed_day or yield_units is None):
            return object()
        consecutive = 0 if fed else int(consecutive) + 1
        if consecutive >= 2:
            # The engine removes only the animal and retains its structure after
            # two consecutive unfed refreshes.
            return {"kind": animal_data["structure"]}
        expected["consecutive_unfed"] = consecutive
        next_day = day + 1
        days_since_first = next_day - int(placed_day) - int(animal_data["first_yield_day"])
        if days_since_first >= 0 and days_since_first % int(animal_data["interval"]) == 0:
            pending = _number(before.get("pending_care_bonus", 0))
            if pending is None:
                return object()
            bonus = int(pending) if fed else 0
            expected["yield_units"] = min(int(animal_data["max_held"]), int(yield_units) + 1 + bonus)
            expected["pending_care_bonus"] = 0
        if cared and fed:
            pending = _number(expected.get("pending_care_bonus", 0))
            if pending is None:
                return object()
            expected["pending_care_bonus"] = int(pending) + 1
        expected["fertilizer_available"] = True
        expected["fed_today"] = False
        expected["cared_today"] = False
        return expected
    return before


def _end_of_day_tile_compatible(before: Any, after: Any, *, day: int = 0, step: int = 0,
                                turns_per_day: int = 24) -> bool:
    """Accept only board changes produced by the engine's daily refresh."""
    expected = _daily_refresh_tile(before, day, step, turns_per_day)
    if before is None and _tile_kind(after) == "WEED":
        return True  # deterministic engine rule with seed-dependent occurrence
    return after == expected


def _targeted_boundary_tile_expected(before: Any, operation: str, day: int, step: int,
                                    turns_per_day: int) -> Any:
    """Return a targeted tile after its action and the following refresh."""
    if not isinstance(before, Mapping):
        return object()
    expected = dict(before)
    if operation == "WATER":
        if _tile_kind(before) != "PLANT" or before.get("watered_today") is not False:
            return object()
        crop = before.get("crop")
        crop_data = _crop_data(crop)
        planted_day = _number(before.get("planted_day"))
        yield_units = _number(before.get("yield_units"))
        fertilized_until = _number(before.get("fertilized_until_day"))
        if (crop_data is None or planted_day is None or int(planted_day) != planted_day
                or yield_units is None or fertilized_until is None):
            return object()
        expected["watered_today"] = True
        if not crop_data["ongoing"]:
            age_days = day - int(planted_day)
            window_start = (int(crop_data["max_yield_day"]) + 1) // 2
            if window_start <= age_days <= int(crop_data["max_yield_day"]):
                bonus = 2 if fertilized_until >= day else 1
                expected["yield_units"] = min(int(crop_data["max_yield"]), int(yield_units) + bonus)
    elif operation == "FEED":
        if not _animal_state(before) or before.get("fed_today") is not False:
            return object()
        expected["fed_today"] = True
    elif operation == "CARE":
        if not _animal_state(before) or before.get("cared_today") is not False:
            return object()
        expected["cared_today"] = True
    else:
        return object()
    return _daily_refresh_tile(expected, day, step, turns_per_day)


def _is_end_of_day_transition(pre: Mapping[str, Any], post: Mapping[str, Any],
                              configuration: Mapping[str, Any] | None) -> bool:
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", 24))
    pre_step = _number(pre.get("step"))
    post_step = _number(post.get("step"))
    pre_day = _number(pre.get("day"))
    post_day = _number(post.get("day"))
    if turns_per_day is None or int(turns_per_day) < 1:
        return False
    if _number(pre.get("hour")) != int(turns_per_day) - 1 or _number(post.get("hour")) != 0:
        return False
    if pre_step is None or post_step is None:
        # Keep small direct unit fixtures useful; real replay observations always
        # carry step for player 0. Player 1's engine observation omits step, so
        # its day increment is the strict boundary signal instead.
        if pre_step is None and post_step is None:
            return pre_day is None and post_day is None or (
                pre_day is not None and post_day == pre_day + 1
            )
        return False
    if post_step != pre_step + 1 or int(pre_step) % int(turns_per_day) != int(turns_per_day) - 1:
        return False
    return pre_day is None or (post_day is not None and post_day == pre_day + 1)


def _action_target_tile_expected(before: Any, command: Sequence[Any], pre: Mapping[str, Any],
                                post: Mapping[str, Any], configuration: Mapping[str, Any] | None,
                                *, apply_boundary_refresh: bool | None = None) -> tuple[bool, Any]:
    """Return the exact board tile an action may produce at its target.

    Targeted tiles cannot be treated as an unrestricted mutation escape hatch:
    the engine changes a small, deterministic set of fields for each unit
    operation.  This helper mirrors those changes, including the end-of-day
    refresh that follows the action.
    """
    if not command or not isinstance(command[0], str):
        return False, None
    operation = command[0]
    boundary = _is_end_of_day_transition(pre, post, configuration)
    refresh = boundary if apply_boundary_refresh is None else apply_boundary_refresh
    day = int(_number(pre.get("day")) or 0)
    step = _observation_step(pre, configuration)
    turns_per_day = int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24)

    if operation == "PLANT":
        if (before is not None or len(command) != 2 or not isinstance(command[1], str)
                or command[1] not in CROPS):
            return False, None
        crop = command[1]
        crop_data = CROPS[crop]
        expected = {
            "kind": "PLANT", "crop": crop, "planted_day": day,
            "watered_today": False, "consecutive_unwatered": 1,
            "yield_units": 0 if crop_data["ongoing"] else 1,
            "max_lifespan_step": -1 if crop_data["ongoing"] else (day + crop_data["max_yield_day"] + 1) * turns_per_day,
            "fertilized_until_day": -1,
        }
        return True, _daily_refresh_tile(expected, day, step, turns_per_day) if refresh else expected

    if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
        expected = {"kind": operation.removeprefix("BUILD_")}
        return (before is None), expected

    if operation == "DIG":
        return before is not None and not (isinstance(before, Mapping) and "animal" in before), None

    if operation == "PLACE":
        if len(command) < 2 or not isinstance(command[1], str):
            return False, None
        item = command[1]
        if item not in ANIMALS:
            return True, before
        if not isinstance(before, Mapping) or before.get("kind") != ANIMALS[item]["structure"] or "animal" in before:
            return False, None
        expected = {
            "kind": ANIMALS[item]["structure"], "animal": item,
            "placed_day": day, "yield_units": 0, "consecutive_unfed": 0,
            "fed_today": False, "cared_today": False,
            "fertilizer_available": False, "pending_care_bonus": 0,
        }
        return True, _daily_refresh_tile(expected, day, step, turns_per_day) if refresh else expected

    if operation in {"WATER", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"}:
        if not isinstance(before, Mapping):
            return False, None
        expected = dict(before)
        if operation == "WATER":
            if _tile_kind(before) != "PLANT" or before.get("watered_today") is not False:
                return False, None
            crop = before.get("crop")
            crop_data = _crop_data(crop)
            age = _number(before.get("planted_day"))
            yield_units = _number(before.get("yield_units"))
            if crop_data is None or age is None or yield_units is None:
                return False, None
            expected["watered_today"] = True
            if not crop_data["ongoing"]:
                age_days = day - int(age)
                window_start = (int(crop_data["max_yield_day"]) + 1) // 2
                if window_start <= age_days <= int(crop_data["max_yield_day"]):
                    bonus = 2 if (_number(before.get("fertilized_until_day")) or -1) >= day else 1
                    expected["yield_units"] = min(int(crop_data["max_yield"]), int(yield_units) + bonus)
        elif operation == "FEED":
            if not _animal_state(before) or before.get("fed_today") is not False:
                return False, None
            expected["fed_today"] = True
        elif operation == "CARE":
            if not _animal_state(before) or before.get("cared_today") is not False:
                return False, None
            expected["cared_today"] = True
        elif operation == "FERTILIZE":
            if _tile_kind(before) != "PLANT":
                return False, None
            current = _number(before.get("fertilized_until_day", -1))
            if current is None:
                return False, None
            expected["fertilized_until_day"] = max(current, day + 2)
        else:
            animal = _animal_state(before)
            if not animal or before.get("fertilizer_available") is not True:
                return False, None
            expected["fertilizer_available"] = False
        return True, _daily_refresh_tile(expected, day, step, turns_per_day) if refresh else expected

    if operation == "HARVEST":
        if not isinstance(before, Mapping) or (_number(before.get("yield_units")) or 0) <= 0:
            return False, None
        if _tile_kind(before) == "PLANT":
            crop_data = _crop_data(before.get("crop"))
            if crop_data is None:
                return False, None
            expected = None if not crop_data["ongoing"] else {**before, "yield_units": 0}
        elif _animal_state(before):
            expected = {**before, "yield_units": 0}
        else:
            return False, None
        if refresh and expected is not None:
            expected = _daily_refresh_tile(expected, day, step, turns_per_day)
        return True, expected

    return False, None


def _action_target_tile_sequence_expected(before: Any, commands: Sequence[Sequence[Any]],
                                          pre: Mapping[str, Any], post: Mapping[str, Any],
                                          configuration: Mapping[str, Any] | None) -> tuple[bool, Any]:
    """Apply same-tile commands in farmer-then-hand engine order, then refresh once."""
    current = before
    applied = False
    for command in commands:
        valid, expected = _action_target_tile_expected(
            current, command, pre, post, configuration, apply_boundary_refresh=False,
        )
        if valid:
            current = expected
            applied = True
        # The engine silently no-ops a later command whose precondition was
        # consumed by an earlier command on the same tile (for example a
        # second WATER after the first WATER, or WATER after HARVEST).
    if not applied:
        return False, None
    day = int(_number(pre.get("day")) or 0)
    step = _observation_step(pre, configuration)
    turns_per_day = int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24)
    if _is_end_of_day_transition(pre, post, configuration):
        current = _daily_refresh_tile(current, day, step, turns_per_day)
    elif isinstance(current, Mapping) and _tile_kind(current) == "PLANT":
        lifespan = _number(current.get("max_lifespan_step"))
        if lifespan is not None and lifespan >= 0 and step >= lifespan and int(step - lifespan) % 2 == 0:
            current = dict(current)
            current["yield_units"] = (_number(current.get("yield_units")) or 0) - 1
            if current["yield_units"] <= 0:
                current = {"kind": "WEED"}
    return True, current


def _tile_state_matches_expected(actual: Any, expected: Any, *, allow_compact: bool = False) -> bool:
    """Compare a replay tile without allowing unmodeled fields.

    A few unit fixtures intentionally use a compact tile representation.  A
    real engine replay normally contains the complete mapping, but accepting
    a compact *subset* keeps those fixtures useful while still rejecting any
    added field or changed value that is outside the modeled transition.
    """
    if actual == expected:
        return True
    if not allow_compact:
        return False
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    return set(actual).issubset(expected) and all(actual[key] == expected[key] for key in actual)


def _compact_plant_action_matches(actual: Any, command: Sequence[Any], pre: Mapping[str, Any]) -> bool:
    """Support legacy compact unit fixtures without opening a mutation escape hatch."""
    if not command or command[0] != "PLANT":
        return False
    if _number(pre.get("day")) is not None:
        return False
    if not isinstance(actual, Mapping) or set(actual) - {"kind", "crop", "watered_today", "yield_units"}:
        return False
    return (
        actual.get("kind") == "PLANT"
        and len(command) == 2
        and actual.get("crop") == command[1]
        and actual.get("watered_today") is False
        and _number(actual.get("yield_units")) in {0, 1}
    )


def _midday_board_changes_valid(pre: Mapping[str, Any], post: Mapping[str, Any], action: Mapping[str, Any],
                                configuration: Mapping[str, Any] | None, market_result: Mapping[str, Any],
                                *, allow_compact: bool | None = None) -> bool:
    """Reject board changes not attributable to this turn's unit/land actions."""
    if allow_compact is None:
        allow_compact = (
            "boardSize" not in _mapping(configuration)
            or ("day" not in pre and "day" not in post)
        )
    if not allow_compact and (
        not _board_dimensions_valid(pre, configuration)
        or not _board_dimensions_valid(post, configuration)
    ):
        return False
    pre_farm = _farm_observation(pre)
    post_farm = _farm_observation(post)
    pre_tiles = pre_farm.get("tiles")
    post_tiles = post_farm.get("tiles")
    if not isinstance(pre_tiles, Sequence) or isinstance(pre_tiles, (str, bytes)):
        return False
    if not isinstance(post_tiles, Sequence) or isinstance(post_tiles, (str, bytes)):
        return False
    size = _board_size(pre, configuration)
    commands = [action.get("farmer"), *list(action.get("hands", ()))]
    tile_operations = {"PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER",
                       "DIG", "BUILD_COOP", "BUILD_PASTURE", "PLACE"}
    target_commands: dict[tuple[int, int], list[Sequence[Any]]] = {}
    for index, command in enumerate(commands):
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes)) and command and command[0] in tile_operations:
            position = _worker_position(pre, index)
            if position is not None:
                target_commands.setdefault(position, []).append(command)

    pre_unlocked = pre_farm.get("unlocked_quadrants", ())
    post_unlocked = post_farm.get("unlocked_quadrants", ())
    if not isinstance(pre_unlocked, Sequence) or isinstance(pre_unlocked, (str, bytes)):
        return False
    if not isinstance(post_unlocked, Sequence) or isinstance(post_unlocked, (str, bytes)):
        return False
    try:
        newly_unlocked = set(post_unlocked) - set(pre_unlocked)
    except TypeError:
        return False
    for y in range(max(len(pre_tiles), len(post_tiles))):
        for x in range(size):
            before = _board_value(pre, (x, y))
            after = _board_value(post, (x, y))
            if before == after:
                continue
            if (x, y) in target_commands:
                valid, expected = _action_target_tile_sequence_expected(
                    before, target_commands[(x, y)], pre, post, configuration,
                )
                if valid and (
                    _tile_state_matches_expected(after, expected, allow_compact=allow_compact)
                    or (allow_compact and any(
                        _compact_plant_action_matches(after, command, pre)
                        for command in target_commands[(x, y)]
                    ))
                ):
                    continue
                # A mature finite crop is removed by HARVEST before the
                # seeded end-of-day weed spawn.  The action expectation is
                # therefore None, while the real engine may record WEED.
                if valid and _is_end_of_day_transition(pre, post, configuration) and expected is None and any(
                    isinstance(command, Sequence) and not isinstance(command, (str, bytes))
                    and command and command[0] == "HARVEST"
                    for command in target_commands[(x, y)]
                ) and _tile_kind(after) == "WEED":
                    continue
                if not valid and _is_end_of_day_transition(pre, post, configuration) and _end_of_day_tile_compatible(
                    before, after, day=int(_number(pre.get("day")) or 0),
                    step=_observation_step(pre, configuration),
                    turns_per_day=int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24),
                ):
                    continue
                return False
            quadrant = ("N" if y < size // 2 else "S") + ("W" if x < size // 2 else "E")
            if (
                quadrant in newly_unlocked
                and before == "LOCKED"
                and (after is None or (_tile_kind(after) == "WEED" and isinstance(after, Mapping)))
            ):
                continue
            if _is_end_of_day_transition(pre, post, configuration) and _end_of_day_tile_compatible(
                    before, after, day=int(_number(pre.get("day")) or 0),
                    step=_observation_step(pre, configuration),
                    turns_per_day=int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24)):
                continue
            # The engine can decay a plant at its exact lifespan boundary after
            # actions.  This is deterministic and limited to yield decrement or
            # the resulting weed, so unrelated arbitrary mutations remain errors.
            if isinstance(before, Mapping) and _tile_kind(before) == "PLANT":
                lifespan = _number(before.get("max_lifespan_step"))
                step = _observation_step(pre, configuration)
                if lifespan is not None and lifespan >= 0 and step is not None and step >= lifespan and int(step - lifespan) % 2 == 0:
                    expected = dict(before)
                    expected["yield_units"] = (_number(before.get("yield_units")) or 0) - 1
                    if expected["yield_units"] <= 0:
                        expected = {"kind": "WEED"}
                    if after == expected:
                        continue
            return False
    return True


def _post_market_effects_valid(pre: Mapping[str, Any], post: Mapping[str, Any],
                               market_result: Mapping[str, Any], configuration: Mapping[str, Any] | None) -> bool:
    """Validate the shared market inventory after orders and deterministic town demand."""
    town = pre.get("town")
    if town is not None and not isinstance(town, Mapping):
        return False
    expected = market_result.get("market_inventory")
    if not isinstance(expected, Mapping):
        return False
    expected = {item: _number(quantity) for item, quantity in expected.items()}
    if any(quantity is None for quantity in expected.values()):
        return False
    shops = town.get("unlocked_shops", ()) if isinstance(town, Mapping) else ()
    if not isinstance(shops, Sequence) or isinstance(shops, (str, bytes)):
        return False
    if isinstance(town, Mapping):
        step = _number(pre.get("step"))
        if step is None or int(step) != step:
            return False
        try:
            shop_interval = int(_config_value(configuration, "townShopSellInterval", 4))
            center_interval = int(_config_value(configuration, "townCenterSellInterval", 24))
        except (TypeError, ValueError, OverflowError):
            return False
        if shop_interval < 1 or center_interval < 1:
            return False
        if int(step) % shop_interval == 0:
            for shop in shops:
                if shop not in SHOPS:
                    return False
                multiplier = 2 if len(SHOPS[shop]) == 1 else 1
                for item in SHOPS[shop]:
                    expected[item] = expected.get(item, 0) - multiplier
        if int(step) % center_interval == 0:
            for item in PRODUCTS:
                if item != "FERTILIZER":
                    expected[item] = expected.get(item, 0) - 1
    post_market = _mapping(post.get("market"))
    post_inventory = post_market.get("inventory")
    if not isinstance(post_inventory, Mapping) or set(post_inventory) != set(expected):
        return False
    if any(_number(post_inventory[item]) != quantity for item, quantity in expected.items()):
        return False
    prices = post_market.get("prices")
    if isinstance(town, Mapping) and (not isinstance(prices, Mapping) or set(prices) != set(expected)):
        return False
    if isinstance(town, Mapping) and isinstance(prices, Mapping):
        params = _mapping(_config_value(configuration, "marketParams", {}))
        if any(_number(prices[item]) != market_price(item, quantity, params) for item, quantity in expected.items()):
            return False
    return True


def _board_size(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None) -> int:
    configured = _config_value(configuration, "boardSize", None)
    if configured is not None:
        try:
            return max(1, int(configured))
        except (TypeError, ValueError, OverflowError):
            pass
    tiles = _mapping(_farm_observation(observation)).get("tiles", ())
    return max(1, len(tiles) if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) else 1)


def _board_dimensions_valid(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None) -> bool:
    farm = _farm_observation(observation)
    tiles = farm.get("tiles")
    configured_size = _config_value(configuration, "boardSize", None)
    size = _number(configured_size) if configured_size is not None else (
        float(len(tiles)) if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) else None
    )
    if size is None or int(size) != size or int(size) < 1:
        return False
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)) or len(tiles) != int(size):
        return False
    return all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == int(size)
        for row in tiles
    )


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
        if not isinstance(tile, Mapping) or _tile_kind(tile) == "LOCKED" or (_number(tile.get("yield_units")) or 0) <= 0:
            return False
        day = _number(observation.get("day"))
        if day is None:
            return True
        if _tile_kind(tile) == "PLANT":
            crop = _crop_data(tile.get("crop"))
            planted_day = _number(tile.get("planted_day"))
            return crop is not None and planted_day is not None and day - planted_day >= crop["first_yield_day"]
        animal = _animal_state(tile)
        animal_data = ANIMALS.get(animal.get("animal"))
        placed_day = _number(animal.get("placed_day"))
        return animal_data is not None and placed_day is not None and day - placed_day >= animal_data["first_yield_day"]
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


def _tile_at_position(observation: Mapping[str, Any], position: tuple[int, int]) -> Any:
    farm = _farm_observation(observation)
    tiles = farm.get("tiles", ())
    x, y = position
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)) or not 0 <= y < len(tiles):
        return None
    row = tiles[y]
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not 0 <= x < len(row):
        return None
    return row[x]


def _positive_quantities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(item): quantity
        for item, raw_quantity in value.items()
        if (quantity := _number(raw_quantity)) is not None and quantity > 0
    }


def _private_inventories(observation: Mapping[str, Any]) -> list[dict[str, float]] | None:
    raw = _mapping(observation.get("private")).get("inventories")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [_positive_quantities(inventory) for inventory in raw]


def _strict_quantity_mapping(value: Any, allowed_items: set[str] | frozenset[str]) -> bool:
    if not isinstance(value, Mapping):
        return False
    for item, raw_quantity in value.items():
        quantity = _number(raw_quantity)
        if (not isinstance(item, str) or item not in allowed_items or quantity is None
                or quantity < 0 or int(quantity) != quantity):
            return False
    return True


def _strict_private_inventories(observation: Mapping[str, Any]) -> bool:
    """Require the engine's complete, finite worker-inventory snapshot."""
    private = observation.get("private")
    if not isinstance(private, Mapping) or not _strict_quantity_mapping(private.get("shed", {}), _ITEM_NAMES):
        return False
    if not _strict_quantity_mapping(private.get("seeds", {}), frozenset(CROPS)):
        return False
    raw = private.get("inventories")
    farm = _farm_observation(observation)
    hands = farm.get("hands", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return False
    if not isinstance(hands, Sequence) or isinstance(hands, (str, bytes)) or len(raw) != len(hands) + 1:
        return False
    return all(_strict_quantity_mapping(inventory, _ITEM_NAMES) for inventory in raw)


def _transition_effects_valid(pre: Mapping[str, Any], post: Mapping[str, Any], action: Mapping[str, Any],
                             configuration: Mapping[str, Any] | None = None,
                             market_result: Mapping[str, Any] | None = None,
                             *, player_index: int = 0, allow_compact: bool | None = None,
                             market_observation: Mapping[str, Any] | None = None,
                             allow_invalid_unit_noop: bool = False) -> bool:
    """Check deterministic effects of one recorded player action."""
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping) or market_result is None:
        return False
    if player_index not in (0, 1):
        return False
    if allow_compact is None:
        allow_compact = (
            "boardSize" not in _mapping(configuration)
            or ("day" not in pre and "day" not in post)
        )
    pre_farm = _farm_observation(pre)
    post_farm = _farm_observation(post)
    pre_hands = pre_farm.get("hands", ())
    post_hands = post_farm.get("hands", ())
    if not isinstance(pre_hands, Sequence) or isinstance(pre_hands, (str, bytes)):
        return False
    if not isinstance(post_hands, Sequence) or isinstance(post_hands, (str, bytes)):
        return False
    boundary = _is_end_of_day_transition(pre, post, configuration)
    simulated_states = market_result.get("states", ())
    simulated = (
        simulated_states[player_index]
        if isinstance(simulated_states, Sequence) and len(simulated_states) > player_index
        else {}
    )
    if not isinstance(simulated, Mapping):
        return False
    pre_hires_value = _number(pre_farm.get("hires_today"))
    if pre_hires_value is None:
        if not allow_compact:
            return False
        pre_hires_value = 0
    if int(pre_hires_value) != pre_hires_value or pre_hires_value < 0:
        return False
    pre_hires = int(pre_hires_value)
    total_hires = int(_number(simulated.get("hires")) or 0)
    hire_count = max(0, total_hires - pre_hires)
    expected_hands = 0 if boundary else len(pre_hands) + hire_count
    if len(post_hands) != expected_hands:
        return False
    post_hires_value = _number(post_farm.get("hires_today"))
    expected_hires = 0 if boundary else total_hires
    if post_hires_value is None:
        if not allow_compact:
            return False
    elif int(post_hires_value) != post_hires_value or int(post_hires_value) != expected_hires:
        return False
    if boundary:
        reset_position = _default_spawn_position(_board_size(pre, configuration))
        tiles = pre_farm.get("tiles")
        if (isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes))
                and len(tiles) < reset_position[1] + 1):
            reset_position = _default_spawn_position(len(tiles))
        if _worker_position(post, 0) != reset_position:
            return False
    commands = [action.get("farmer"), *list(action.get("hands", ()))]
    inventories = _private_inventories(pre)
    post_inventories = _private_inventories(post)
    if boundary and (inventories is None or post_inventories is None):
        return False
    if inventories is not None and post_inventories is not None:
        if len(inventories) < len(commands):
            return False
        expected_inventories = [dict(inventory) for inventory in inventories]
        if not boundary:
            expected_inventories.extend({} for _ in range(hire_count))
        expected_shed = dict(_positive_quantities(simulated.get("shed")))
    else:
        expected_inventories = None
        expected_shed = None
    expected_seeds = dict(_positive_quantities(simulated.get("seeds")))

    def add_quantity(values: dict[str, float], item: str, amount: float) -> None:
        values[item] = values.get(item, 0.0) + amount
        if values[item] <= 0:
            values.pop(item, None)

    for worker_index, command in enumerate(commands):
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
            return False
        state_valid = _unit_command_valid_for_state(command, pre, worker_index, configuration)
        position = _worker_position(pre, worker_index)
        post_position = _worker_position(post, worker_index)
        if position is None:
            return False
        # Farm hands are one-day assets: the engine can remove them while
        # resetting the day, so their post-boundary position is intentionally
        # absent. Their pre-action effects are still validated below.
        if post_position is None and not boundary:
            return False
        operation = command[0]
        if operation in {"NORTH", "SOUTH", "EAST", "WEST"}:
            deltas = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
            dx, dy = deltas[operation]
            expected_position = (position[0] + dx, position[1] + dy) if state_valid else position
            if not boundary and post_position != expected_position:
                return False
        elif not boundary and post_position != position:
            return False

        if allow_invalid_unit_noop and not state_valid:
            continue

        pre_tile = _tile_at_position(pre, position)
        post_tile = _tile_at_position(post, position)
        if operation == "PLANT":
            plant_persisted = (
                isinstance(post_tile, Mapping)
                and _tile_kind(post_tile) == "PLANT"
                and post_tile.get("crop") == command[1]
            )
            end_of_day_decay = boundary and _tile_kind(post_tile) == "WEED"
            if not plant_persisted and not end_of_day_decay:
                return False
            add_quantity(expected_seeds, command[1], -1)
        elif operation == "BUILD_COOP" and (not isinstance(post_tile, Mapping) or _tile_kind(post_tile) != "COOP"):
            return False
        elif operation == "BUILD_PASTURE" and (not isinstance(post_tile, Mapping) or _tile_kind(post_tile) != "PASTURE"):
            return False
        elif operation == "DIG" and post_tile is not None:
            return False
        elif operation in {"WATER", "FEED", "CARE"}:
            if not isinstance(post_tile, Mapping):
                # Unit actions are resolved in one engine turn.  If a farmer
                # harvests a tile before a hand's WATER/FEED/CARE command is
                # resolved, the second command is a legal no-op because the
                # harvested tile is already gone.  Do not demand a post-tile
                # effect that the engine cannot produce in this conflict.
                blocked_by_harvest = any(
                    other_index != worker_index
                    and isinstance(other_command, Sequence)
                    and not isinstance(other_command, (str, bytes))
                    and other_command
                    and other_command[0] == "HARVEST"
                    and _worker_position(pre, other_index) == position
                    for other_index, other_command in enumerate(commands)
                )
                if post_tile is None and blocked_by_harvest and isinstance(pre_tile, Mapping):
                    continue
                return False
            if operation == "WATER" and _tile_kind(post_tile) == "WEED" and isinstance(pre_tile, Mapping):
                lifespan = _number(pre_tile.get("max_lifespan_step"))
                step = _number(pre.get("step"))
                yield_units = _number(pre_tile.get("yield_units"))
                if (
                    _tile_kind(pre_tile) == "PLANT"
                    and lifespan is not None and step is not None
                    and step >= lifespan and int(step - lifespan) % 2 == 0
                    and yield_units == 1
                ):
                    # The engine applies the action to the pre-state, then
                    # its lifespan decay can replace the last unit with WEED
                    # in the same transition.
                    continue
            if boundary:
                expected_tile = _targeted_boundary_tile_expected(
                    pre_tile, operation, int(_number(pre.get("day")) or 0),
                    _observation_step(pre, configuration),
                    int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24),
                )
                if post_tile != expected_tile:
                    return False
            elif operation == "WATER" and post_tile.get("watered_today") is not True:
                return False
            elif operation == "FEED" and post_tile.get("fed_today") is not True:
                return False
            elif operation == "CARE" and post_tile.get("cared_today") is not True:
                return False
        elif operation == "FERTILIZE":
            harvest_follows = any(
                other_index > worker_index
                and isinstance(other_command, Sequence)
                and not isinstance(other_command, (str, bytes))
                and other_command
                and other_command[0] == "HARVEST"
                and _worker_position(pre, other_index) == position
                for other_index, other_command in enumerate(commands)
            )
            if post_tile is None and harvest_follows:
                pass
            elif not isinstance(post_tile, Mapping) or (_number(post_tile.get("fertilized_until_day")) or -1) < (_number(pre.get("day")) or 0) + 2:
                return False
        elif operation == "COLLECT_FERTILIZER" and not boundary:
            if not isinstance(post_tile, Mapping) or post_tile.get("fertilizer_available") is not False:
                return False
        elif operation == "HARVEST":
            ongoing_boundary_harvest = (
                boundary
                and isinstance(pre_tile, Mapping)
                and _tile_kind(pre_tile) == "PLANT"
                and bool((_crop_data(pre_tile.get("crop")) or {}).get("ongoing", False))
            )
            if ongoing_boundary_harvest:
                same_tile_commands = [
                    candidate
                    for candidate_index, candidate in enumerate(commands)
                    if _worker_position(pre, candidate_index) == position
                ]
                valid, expected_tile = _action_target_tile_sequence_expected(
                    pre_tile, same_tile_commands, pre, post, configuration,
                )
                if not valid or not _tile_state_matches_expected(
                    post_tile, expected_tile, allow_compact=allow_compact,
                ):
                    return False
            elif isinstance(post_tile, Mapping) and (_number(post_tile.get("yield_units")) or 0) != 0:
                pre_yield = _number(pre_tile.get("yield_units")) if isinstance(pre_tile, Mapping) else None
                if pre_yield is None:
                    return False
            if post_tile is not None and not isinstance(post_tile, Mapping):
                return False
        elif operation == "PLACE" and command[1] in _ANIMAL_NAMES:
            if not isinstance(post_tile, Mapping) or post_tile.get("animal") != command[1]:
                return False

        if expected_inventories is None or worker_index >= len(expected_inventories):
            continue
        inventory = expected_inventories[worker_index]
        item = command[1] if len(command) > 1 else None
        quantity = float(command[2] if len(command) > 2 else 1) if item is not None else 0.0
        if operation == "PICKUP":
            add_quantity(expected_shed, item, -quantity)
            add_quantity(inventory, item, quantity)
        elif operation == "PLACE" and item not in _ANIMAL_NAMES:
            add_quantity(inventory, item, -quantity)
            add_quantity(expected_shed, item, quantity)
        elif operation == "PLACE" and item in _ANIMAL_NAMES:
            add_quantity(inventory, item, -1)
        elif operation == "FEED":
            add_quantity(inventory, "WHEAT", -1)
        elif operation == "FERTILIZE":
            add_quantity(inventory, "FERTILIZER", -1)
        elif operation == "COLLECT_FERTILIZER":
            add_quantity(inventory, "FERTILIZER", 1)
        elif operation == "HARVEST" and isinstance(pre_tile, Mapping):
            item = pre_tile.get("crop") or _animal_state(pre_tile).get("product")
            if item:
                add_quantity(inventory, item, _number(pre_tile.get("yield_units")) or 0)
        elif operation == "DROP":
            room = max(0.0, _shed_capacity(configuration) - sum(expected_shed.values()))
            for drop_item, drop_quantity in list(inventory.items()):
                taken = min(drop_quantity, room)
                add_quantity(expected_shed, drop_item, taken)
                room -= taken
            inventory.clear()

    if expected_inventories is not None and boundary:
        for inventory in expected_inventories:
            room = max(0.0, _shed_capacity(configuration) - sum(expected_shed.values()))
            for item, quantity in list(inventory.items()):
                taken = min(max(0.0, quantity), room)
                add_quantity(expected_shed, item, taken)
                room -= taken
                inventory.pop(item, None)
        expected_inventories = [{}]
    if expected_inventories is not None:
        if post_inventories != expected_inventories:
            return False
        if _positive_quantities(_mapping(_mapping(post).get("private")).get("shed")) != _positive_quantities(expected_shed):
            return False
    if _positive_quantities(_mapping(_mapping(post).get("private")).get("seeds")) != expected_seeds:
        return False
    expected_money = _number(simulated.get("money"))
    if expected_money is None or _cash(post) != expected_money:
        return False
    if list(_mapping(post_farm).get("unlocked_quadrants", ())) != list(simulated.get("unlocked", ())):
        return False
    if not boundary and hire_count:
        expected_existing_hands = list(post_hands[:len(pre_hands)])
        spawned = []
        for _ in range(hire_count):
            position = _spawn_hand_position(post_farm, _board_size(pre, configuration),
                                            [*expected_existing_hands, *spawned])
            spawned.append(list(position))
        if post_hands[len(pre_hands):] != spawned:
            return False
    if not _midday_board_changes_valid(
        pre, post, action, configuration, market_result, allow_compact=allow_compact,
    ):
        return False
    if not _post_market_effects_valid(market_observation or pre, post, market_result, configuration):
        return False
    return True


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


def _animal_units_owned(observation: Mapping[str, Any]) -> int:
    """Count placed and carried animals before applying a portfolio bias."""
    shed = _mapping(_mapping(observation.get("private")).get("shed"))
    carried = sum(
        int(_number(shed.get(species)) or 0)
        for species in _ANIMAL_NAMES
    )
    placed = sum(
        1 for tile in _tiles(observation)
        if _animal_state(tile).get("animal") in _ANIMAL_NAMES
    )
    return carried + placed


def _variant_market_spend(order: Sequence[Any], observation: Mapping[str, Any]) -> float:
    """Estimate non-seed spend so a crop bias cannot consume safety cash."""
    if not order:
        return 0.0
    operation = order[0]
    item = order[1] if len(order) > 1 else None
    quantity = max(1, int(_number(order[2]) or 1)) if len(order) > 2 else 1
    if operation == "HIRE":
        farm = _farm_observation(observation)
        return float(_fib(int(_number(farm.get("hires_today")) or 0)))
    if operation == "BUY_LAND":
        farm = _farm_observation(observation)
        unlocked = farm.get("unlocked_quadrants", ())
        index = len(unlocked) - 1 if isinstance(unlocked, Sequence) and not isinstance(unlocked, (str, bytes)) else -1
        return float(LAND_PRICES[index]) if 0 <= index < len(LAND_PRICES) else 0.0
    if operation == "BUY_ANIMAL" and item in ANIMALS:
        return quantity * float(ANIMALS[item]["cost"])
    if operation == "BUY_PRODUCT" and item in _BUYABLE_PRODUCTS:
        market = _mapping(observation.get("market"))
        inventory = _number(_mapping(market.get("inventory")).get(item)) or 0.0
        return sum(_observed_unit_price(item, observation, inventory - offset, buying=True) for offset in range(quantity))
    return 0.0


def _preserve_variant_market_capacity(orders: Sequence[Sequence[Any]], observation: Mapping[str, Any],
                                      *, seed_item: str) -> list[list[Any]]:
    """Keep the variant seed bias above the policy's worker cash reserve."""
    reserve = max(100.0, float(CROPS[seed_item]["seed"])) + float(CROPS[seed_item]["seed"])
    non_seed_spend = sum(
        _variant_market_spend(order, observation)
        for order in orders
        if order and order[0] != "BUY_SEED"
    )
    seed_budget = max(0.0, _cash(observation) - non_seed_spend - reserve)
    result: list[list[Any]] = []
    for raw_order in orders:
        order = list(raw_order)
        if order and order[0] == "BUY_SEED":
            quantity = min(int(_number(order[2]) or 1), int(seed_budget // float(CROPS[seed_item]["seed"])))
            if quantity <= 0:
                continue
            order[1] = seed_item
            order[2] = quantity
            seed_budget -= quantity * float(CROPS[seed_item]["seed"])
        result.append(order)
    return result


def _prioritize_safety_market_orders(orders: Sequence[Sequence[Any]]) -> list[list[Any]]:
    def is_safety(order: Sequence[Any]) -> bool:
        return bool(order) and (order[0] == "HIRE" or (
            order[0] == "BUY_PRODUCT" and len(order) > 1 and order[1] in _BUYABLE_PRODUCTS
        ))
    return [list(order) for order in orders if is_safety(order)] + [
        list(order) for order in orders if not is_safety(order)
    ]


def _apply_ablations(action: Mapping[str, Any], observation: Mapping[str, Any], ablations: Mapping[str, bool]) -> dict[str, Any]:
    result = {"farmer": list(action.get("farmer", ["PASS"])), "hands": [list(command) for command in action.get("hands", ())],
              "market": _market_orders(action)}
    if not ablations.get("route_scheduling", True):
        directions = {"NORTH", "SOUTH", "EAST", "WEST"}
        farmer = result["farmer"]
        if farmer and isinstance(farmer[0], str) and farmer[0] in directions:
            result["farmer"] = ["PASS"]
        # Hand commands are nested one level below the farmer command.  Keep
        # malformed/nested values inert here; the final sanitizer is the
        # authority that turns them into a legal fallback.
        result["hands"] = [
            ["PASS"] if isinstance(command, Sequence) and not isinstance(command, (str, bytes))
            and command and isinstance(command[0], str) and command[0] in directions else command
            for command in result["hands"]
        ]
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
                if (len(commands) > 1 and commands[0] == "PLACE" and isinstance(commands[1], str)
                        and commands[1] in ANIMALS):
                    result[key] = ["PASS"]
            else:
                result[key] = [
                    ["PASS"] if (isinstance(command, Sequence) and not isinstance(command, (str, bytes))
                                 and len(command) > 1 and command[0] == "PLACE"
                                 and isinstance(command[1], str) and command[1] in ANIMALS) else command
                    for command in commands
                ]
    return result


def apply_variant(action: Mapping[str, Any], observation: Mapping[str, Any], variant: str,
                  ablations: Mapping[str, bool] | None = None,
                  configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply a named, legality-preserving strategy adjustment at the agent boundary."""
    result = {"farmer": list(action.get("farmer", ["PASS"])), "hands": [list(command) for command in action.get("hands", ())],
              "market": _market_orders(action)}
    seeds = _private_seeds(observation)
    if variant == "conservative":
        deadline_hires = [
            order for order in result["market"]
            if order == ["HIRE"] and _has_basic_need_deadline(observation)
        ]
        mandatory = [
            order for order in result["market"]
            if order[0] == "BUY_PRODUCT" and order[1] in {"WHEAT", "FERTILIZER"}
        ]
        mandatory = deadline_hires + mandatory
        discretionary = [
            order for order in result["market"]
            if order[0] not in {"BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"}
        ][:max(0, 1 - len(mandatory))]
        result["market"] = mandatory + discretionary
    elif variant == "melon-heavy":
        if result["farmer"][:1] == ["PLANT"] and len(result["farmer"]) > 1 and result["farmer"][1] == "WHEAT" and _number(seeds.get("MELON")):
            result["farmer"][1] = "MELON"
        result["market"] = _preserve_variant_market_capacity(
            result["market"], observation, seed_item="MELON",
        )
        for order in result["market"]:
            if order[0] == "BUY_SEED":
                order[1] = "MELON"
                break
        if not any(order[0] == "BUY_SEED" for order in result["market"]) and not _number(seeds.get("MELON")) and _cash(observation) >= CROPS["MELON"]["seed"]:
            result["market"].append(["BUY_SEED", "MELON", 1])
    elif variant == "demand-reactive":
        # The production policy already reacts to live market/shop demand.
        # Rewriting its scheduled crop after planning can consume the wheat
        # reserved for FEED or invalidate a WATER deadline, so this supported
        # evaluator variant deliberately preserves that needs-safe schedule.
        pass
    elif variant == "animal-heavy":
        target = _empty_animal_target(observation)
        # Keep one deliberate animal-heavy increment, then leave all later
        # market orders untouched so the production planner can preserve its
        # hiring and basic-needs reservations.  Replacing every seed order
        # with an animal purchase exhausts that reserve and causes avoidable
        # FEED misses.
        if target and _animal_units_owned(observation) < 2:
            for order in result["market"]:
                if order[0] == "BUY_SEED":
                    order[:] = ["BUY_ANIMAL", target, 1]
                    break
            else:
                result["market"].append(["BUY_ANIMAL", target, 1])
    elif variant != "mixed":
        raise ValueError(f"unsupported variant: {variant}")
    result["market"] = _prioritize_safety_market_orders(result["market"])
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
        item[1]["framework_error_rate"],
        -item[1]["win_rate"],
        -item[1]["median_final_bank"],
        item[0],
    ))[0]


def _normalized_report_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove output-directory dependence from metadata while retaining names."""
    normalized = dict(config)
    for key in ("output", "replay_summary"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = Path(str(value)).name
    return normalized


def _normalized_command(command: Sequence[str] | None) -> list[str]:
    """Canonicalize evaluator script and report output arguments."""
    values = [str(value) for value in (command or ())]
    normalized: list[str] = []
    for index, value in enumerate(values):
        if index == 0 and Path(value).name == "evaluate.py":
            normalized.append("scripts/evaluate.py")
        elif value == "--output":
            normalized.append(value)
        elif index and values[index - 1] == "--output":
            normalized.append("<report>")
        elif value.startswith("--output="):
            normalized.append("--output=<report>")
        else:
            normalized.append(value)
    return normalized


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
            "command": _normalized_command(command),
            "config": _normalized_report_config(config),
            "engine": "kaggle-environments",
            "engine_version": ENGINE_VERSION,
            "replay_summary": (
                Path(str(config["replay_summary"])).name
                if config.get("replay_summary") is not None else None
            ),
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
        "replay_summary": sidecar.name,
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
