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
    LAND_PRICES,
    PRICE_FLOOR,
    PRODUCTS,
    max_market_orders,
    shed_capacity,
)
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


def _valid_replay(replay: Mapping[str, Any], own_states: Sequence[Mapping[str, Any]],
                  other_states: Sequence[Mapping[str, Any]]) -> bool:
    statuses = replay.get("statuses")
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)) or list(statuses) != ["DONE", "DONE"]:
        return False
    steps = replay.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        return False
    if len(own_states) != len(steps) or len(other_states) != len(steps):
        return False
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
        if players != {0, 1}:
            return False
        expected_status = "DONE" if index == len(steps) - 1 else "ACTIVE"
        if any(state.get("status") != expected_status for state in turn if isinstance(state, Mapping)):
            return False
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


def _tiles(observation: Mapping[str, Any]) -> Sequence[Any]:
    farms = observation.get("farms")
    player = observation.get("player")
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)) or not isinstance(player, int):
        return ()
    if not 0 <= player < len(farms):
        return ()
    tiles = _mapping(farms[player]).get("tiles", ())
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return ()
    return tuple(tile for row in tiles if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) for tile in row)


def _tile_kind(tile: Any) -> str:
    if isinstance(tile, str):
        return tile.upper()
    return str(_mapping(tile).get("kind", "")).upper()


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


def _post_tile(observation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    tiles = list(_tiles(observation))
    return _mapping(tiles[index]) if 0 <= index < len(tiles) else {}


def _missed_needs_at_boundary(observation: Mapping[str, Any], is_boundary: bool,
                              post_observation: Mapping[str, Any] | None = None,
                              action_state: Mapping[str, Any] | None = None) -> int:
    if not is_boundary:
        return 0
    missed = 0
    current_tiles = list(_tiles(observation))
    post_hour = _number(_mapping(post_observation).get("hour")) if post_observation else None
    reset_after_boundary = post_hour == 0 and _number(observation.get("hour")) == 23
    commands = _commands_for_state(action_state or {})
    operations = {command[0] for command in commands if command and isinstance(command[0], str)}
    for index, tile in enumerate(current_tiles):
        if not isinstance(tile, Mapping):
            continue
        kind = _tile_kind(tile)
        post_tile = _post_tile(post_observation, index) if post_observation else {}
        if kind == "PLANT":
            needs_water = tile.get("needs_water") is True or tile.get("watered_today") is False
            post_says_watered = post_tile.get("watered_today") is True
            if needs_water and not post_says_watered and not (reset_after_boundary and "WATER" in operations):
                missed += 1
        animal = tile.get("animal")
        animal_mapping = _mapping(animal)
        if animal_mapping or kind == "ANIMAL":
            animal_state = dict(tile)
            animal_state.update(animal_mapping)
            needs_feed = animal_state.get("needs_feed") is True or animal_state.get("fed_today") is False
            post_fed = post_tile.get("fed_today") is True
            if needs_feed and not post_fed and not (reset_after_boundary and "FEED" in operations):
                missed += 1
            needs_care = animal_state.get("needs_care") is True or animal_state.get("cared_today") is False
            post_cared = post_tile.get("cared_today") is True
            if needs_care and not post_cared and not (reset_after_boundary and "CARE" in operations):
                missed += 1
    return missed


def _price_floor_sales(state: Mapping[str, Any]) -> int:
    observation = _mapping(state.get("observation"))
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    action = _mapping(state.get("action"))
    orders = action.get("market", ())
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        return 0
    sales = 0
    for order in orders:
        if not isinstance(order, Sequence) or len(order) < 3 or order[0] != "SELL":
            continue
        if _number(prices.get(str(order[1]).upper())) == PRICE_FLOOR:
            sales += max(0, int(_number(order[2]) or 0))
    return sales


def replay_record(replay: Mapping[str, Any], *, variant: str, opponent: str, seed: int) -> dict[str, Any]:
    """Extract one game record from the engine replay JSON."""
    own_states = _player_states(replay, 0)
    other_states = _player_states(replay, 1)
    framework_error = not _valid_replay(replay, own_states, other_states)
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
    last_step = len(steps) - 1 if isinstance(steps, Sequence) else -1
    for index, state in enumerate(own_states):
        observation = _mapping(state.get("observation"))
        shed_overflow = max(shed_overflow, max(0.0, _shed_total(observation) - shed_capacity))
        floor_sales += _price_floor_sales(state)
        hour = _number(observation.get("hour"))
        is_boundary = hour == 23 or observation.get("step") == last_step
        post = _mapping(own_states[index + 1].get("observation")) if index + 1 < len(own_states) else None
        missed_needs += _missed_needs_at_boundary(observation, is_boundary, post, state)
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


def _market_order_cost(order: Sequence[Any], observation: Mapping[str, Any]) -> float:
    if not order:
        return 0.0
    operation = order[0]
    quantity = int(_number(order[2]) or 0) if len(order) >= 3 else 1
    if operation == "BUY_SEED" and len(order) == 3 and order[1] in CROPS:
        return float(CROPS[order[1]]["seed"] * quantity)
    if operation == "BUY_ANIMAL" and len(order) == 3 and order[1] in ANIMALS:
        return float(ANIMALS[order[1]]["cost"] * quantity)
    if operation == "BUY_PRODUCT" and len(order) == 3 and order[1] in _BUYABLE_PRODUCTS:
        prices = _mapping(_mapping(observation.get("market")).get("prices"))
        return float((_number(prices.get(order[1])) or 0.0) * quantity)
    if operation == "BUY_LAND" and len(order) == 1:
        return float(LAND_PRICES[0])
    return 0.0


def _sanitize_market_orders(orders: Sequence[Sequence[Any]], observation: Mapping[str, Any]) -> list[list[Any]]:
    """Keep at most the engine limit and discard invalid or unaffordable orders."""
    money = _cash(observation)
    shed = dict(_mapping(_mapping(observation.get("private")).get("shed")))
    sanitized: list[list[Any]] = []
    for raw_order in orders:
        if len(sanitized) >= max_market_orders:
            break
        order = list(raw_order)
        if not order or not isinstance(order[0], str):
            continue
        operation = order[0]
        if operation in {"HIRE", "BUY_LAND"}:
            if len(order) != 1:
                continue
        elif operation in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
            if len(order) != 3 or not isinstance(order[1], str) or not isinstance(order[2], int) or isinstance(order[2], bool) or order[2] < 1:
                continue
            if operation == "BUY_SEED" and order[1] not in CROPS:
                continue
            if operation == "BUY_PRODUCT" and order[1] not in _BUYABLE_PRODUCTS:
                continue
            if operation == "BUY_ANIMAL" and order[1] not in ANIMALS:
                continue
            if operation == "SELL":
                available = int(_number(shed.get(order[1])) or 0)
                if order[1] not in PRODUCTS or order[2] > available:
                    continue
                shed[order[1]] = available - order[2]
                sanitized.append(order)
                continue
        else:
            continue
        cost = _market_order_cost(order, observation)
        if cost > money:
            continue
        money -= cost
        sanitized.append(order)
    return sanitized


def _legal_unit_command(command: Any) -> bool:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command or not isinstance(command[0], str):
        return False
    operation = command[0]
    if operation in {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE"}:
        return len(command) == 1
    if operation == "PLANT":
        return len(command) == 2 and command[1] in CROPS
    if operation in {"PICKUP", "PLACE"}:
        return len(command) in {2, 3} and command[1] in PRODUCTS | ANIMALS and (len(command) == 2 or (isinstance(command[2], int) and not isinstance(command[2], bool) and command[2] > 0))
    return False


def _sanitize_action(action: Mapping[str, Any], observation: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "farmer": list(action.get("farmer", ())) if isinstance(action.get("farmer", ()), Sequence) else [],
        "hands": [list(command) for command in action.get("hands", ())] if isinstance(action.get("hands", ()), Sequence) else [],
        "market": _sanitize_market_orders(_market_orders(action), observation),
    }
    fallback_farmer = list(fallback.get("farmer", ["PASS"]))
    if not _legal_unit_command(result["farmer"]):
        result["farmer"] = fallback_farmer if _legal_unit_command(fallback_farmer) else ["PASS"]
    fallback_hands = [list(command) for command in fallback.get("hands", ())]
    result["hands"] = [command if _legal_unit_command(command) else (fallback_hands[index] if index < len(fallback_hands) and _legal_unit_command(fallback_hands[index]) else ["PASS"]) for index, command in enumerate(result["hands"])]
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
                  ablations: Mapping[str, bool] | None = None) -> dict[str, Any]:
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
    return _sanitize_action(adjusted, observation, action)


class VariantPolicy:
    """Fresh stateful policy instance with evaluator-only variant adjustments."""

    def __init__(self, variant: str, ablations: Mapping[str, bool] | None = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        self.variant = variant
        self.ablations = dict(ablations or _DEFAULT_ABLATIONS)
        self.policy = Policy()

    def __call__(self, obs: Mapping[str, Any], _configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return apply_variant(self.policy.act(obs), obs, self.variant, self.ablations)


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
    env.run([VariantPolicy(variant, ablations), opponent_agent])
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
