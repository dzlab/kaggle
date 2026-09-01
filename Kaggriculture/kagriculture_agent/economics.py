"""Exact, deterministic economic helpers for the Kaggriculture engine."""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from typing import Any

from .constants import (
    ANIMALS,
    CROPS,
    HINGE_GAIN,
    MARKET_I0,
    MARKET_PARAMS,
    PRICE_FLOOR,
    PRODUCTS,
    SHOP_DEMANDS,
    season_days,
    shed_capacity as DEFAULT_SHED_CAPACITY,
    turns_per_day,
)

_TOWN_CENTER_PRODUCTS = tuple(item for item in PRODUCTS if item != "FERTILIZER")


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def _nonnegative(value: Any, default: float = 0.0) -> float:
    return max(0.0, _number(value, default))


def _whole(value: Any, default: int = 0) -> int:
    return max(0, int(_nonnegative(value, default)))


def shape_value(func: str, x: float, T: float | None = None) -> float:
    """Evaluate a published price-curve shape on a safe nonnegative domain."""
    x = _nonnegative(x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log1p(x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        T = _nonnegative(T)
        if T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def _market_parameters(params: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    resolved = {item: dict(values) for item, values in MARKET_PARAMS.items()}
    if not isinstance(params, Mapping):
        return resolved
    for item, patch in params.items():
        if item in resolved and isinstance(patch, Mapping):
            resolved[item].update(patch)
    return resolved


def _item_params(item: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = _market_parameters(params)
    if item not in resolved:
        raise KeyError(item)
    # Also accept a direct parameter record for convenience in isolated tests.
    if isinstance(params, Mapping) and "base" in params:
        resolved[item].update(params)
    return resolved[item]


def market_price(item: str, inventory: float, params: Mapping[str, Any] | None = None) -> int:
    """Return the engine's nearest-dollar, one-dollar-floored market quote."""
    p = _item_params(item, params)
    return _market_price_from_parameters(p, inventory)


def _market_price_from_parameters(p: Mapping[str, Any], inventory: float) -> int:
    base, I0, T = _number(p["base"]), _number(p["I0"]), _number(p["T"])
    below = _number(inventory) < I0
    func = p["below_func"] if below else p["above_func"]
    target = _number(p["below_target"] if below else p["above_target"])
    distance = abs(_number(inventory) - I0)
    scale = shape_value(func, T, T)
    amplitude = target * base / scale if scale else 0.0
    quote = base + (1 if below else -1) * amplitude * shape_value(func, distance, T)
    return max(PRICE_FLOOR, int(round(quote)))


def sell_batch_value(item: str, quantity: int, inventory: float, params: Mapping[str, Any] | None = None) -> int:
    """Value a sequential sale, quoting before each unit and adding supply.

    Negative, fractional, non-finite, and impractically large quantities are
    reduced to a bounded nonnegative integer.  A one-dollar sale does not add
    market supply, matching the published engine.
    """
    remaining = min(_whole(quantity), DEFAULT_SHED_CAPACITY)
    resolved = _market_parameters(params)
    p = resolved[item]
    if isinstance(params, Mapping) and "base" in params:
        p = dict(p)
        p.update(params)
    current = _number(inventory)
    total = 0
    for _ in range(remaining):
        quote = _market_price_from_parameters(p, current)
        total += quote
        if quote > PRICE_FLOOR:
            current += 1
    return total


def _sale_quotes(item: str, quantity: int, inventory: float, params: Mapping[str, Any] | None = None) -> list[int]:
    remaining = min(_whole(quantity), DEFAULT_SHED_CAPACITY)
    resolved = _market_parameters(params)
    p = resolved[item]
    if isinstance(params, Mapping) and "base" in params:
        p = dict(p)
        p.update(params)
    current = _number(inventory)
    quotes = []
    for _ in range(remaining):
        quote = _market_price_from_parameters(p, current)
        quotes.append(quote)
        if quote > PRICE_FLOOR:
            current += 1
    return quotes


def project_inventory_after_town(
    market_inventory: Mapping[str, Any],
    unlocked_shops: Any,
    step: int,
    shop_interval: int = 4,
    center_interval: int = 24,
) -> dict[str, float | int]:
    """Project one town-consumption tick without mutating input inventory."""
    projected = dict(market_inventory) if isinstance(market_inventory, Mapping) else {}
    step = _whole(step)
    shop_interval = max(1, _whole(shop_interval, 1))
    center_interval = max(1, _whole(center_interval, 1))
    if step % shop_interval == 0 and isinstance(unlocked_shops, (list, tuple)):
        for shop in unlocked_shops:
            products = SHOP_DEMANDS.get(shop)
            if products is None:
                continue
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                projected[item] = _number(projected.get(item)) - multiplier
    if step % center_interval == 0:
        for item in _TOWN_CENTER_PRODUCTS:
            projected[item] = _number(projected.get(item)) - 1
    return projected


def market_regime(prices: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, str]:
    """Classify observed products as scarce, balanced, glut, or floor."""
    prices = prices if isinstance(prices, Mapping) else {}
    inventory = inventory if isinstance(inventory, Mapping) else {}
    result = {}
    for item in sorted(set(prices) | set(inventory)):
        observed_price = prices.get(item)
        price_number = _number(observed_price, math.nan)
        if math.isfinite(price_number) and price_number <= PRICE_FLOOR:
            result[item] = "floor"
        elif _number(inventory.get(item), MARKET_I0) < MARKET_I0:
            result[item] = "scarce"
        elif _number(inventory.get(item), MARKET_I0) > MARKET_I0:
            result[item] = "glut"
        else:
            result[item] = "balanced"
    return result


def _days(value: Any, default: set[int]) -> set[int]:
    if value is None:
        return set(default)
    if isinstance(value, Set) or isinstance(value, (list, tuple, range)):
        return {_whole(day) for day in value}
    return set()


def _quote(item: str, inventory: float, prices: Mapping[str, Any] | None) -> int:
    if isinstance(prices, Mapping) and item in prices:
        return max(PRICE_FLOOR, int(round(_nonnegative(prices[item]))))
    return market_price(item, inventory)


def _forecast_sale_quotes(
    item: str,
    quantity: int,
    inventory: float,
    prices: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
    fixed_price_mode: bool,
) -> list[int]:
    """Return per-unit sale quotes; exact market simulation is the default."""
    if fixed_price_mode and isinstance(prices, Mapping) and item in prices:
        quote = max(PRICE_FLOOR, int(round(_nonnegative(prices[item]))))
        return [quote] * min(_whole(quantity), DEFAULT_SHED_CAPACITY)
    return _sale_quotes(item, quantity, inventory, params)


def _cash_terms(
    *, revenue: float, material_cost: float, worker_cost: float, land_cost: float,
    movement_turns: int, action_turns: int, horizon: int, overflow_units: int,
    output_units: int, floor_units: int,
) -> dict[str, Any]:
    total_cost = material_cost + worker_cost + land_cost
    turns = movement_turns + action_turns or horizon
    return {
        "revenue": revenue, "material_cost": material_cost,
        "worker_cost": worker_cost, "land_cost": land_cost, "cost": total_cost,
        "net_cash": revenue - total_cost, "turns": turns,
        "overflow_units": overflow_units,
        "shed_overflow_risk": min(1.0, overflow_units / max(1, output_units)) if output_units else 0.0,
        "floor_units": floor_units,
        "price_floor_risk": min(1.0, floor_units / max(1, output_units)) if output_units else 0.0,
    }


def forecast_crop(
    crop: str, start_day: int = 0, horizon: int = season_days, *,
    watering_days: Any = None, fertilizer_days: Any = None,
    harvest_day: int | None = None, market_inventory: float = MARKET_I0,
    prices: Mapping[str, Any] | None = None, seed_owned: bool = False,
    fertilizer_owned: int = 0, worker_cost: float = 0, land_cost: float = 0,
    movement_turns: int = 0, action_turns: int = 0, held_inventory: int = 0,
    shed_cap: int = DEFAULT_SHED_CAPACITY, shed_capacity: int | None = None,
    params: Mapping[str, Any] | None = None,
    fixed_price_mode: bool = False,
) -> dict[str, Any]:
    """Forecast one planted crop with exact watering, fertilizer, and decay rules."""
    if crop not in CROPS:
        raise KeyError(crop)
    cd = CROPS[crop]
    start_day, horizon = _whole(start_day), _whole(horizon)
    watering = _days(watering_days, set(range(start_day, start_day + horizon)))
    fertilizer = _days(fertilizer_days, set())
    forecast_days = set(range(start_day, start_day + horizon))
    active_fertilizer = set()
    successful_fertilizer = set()
    units = 0 if cd["ongoing"] else 1
    harvested = decayed = units_before_decay = 0
    units_before_decay = units
    consecutive_unwatered = 1  # planting starts with the engine's initial counter
    alive = True
    lifespan_step = -1 if cd["ongoing"] else (start_day + cd["max_yield_day"] + 1) * turns_per_day

    for day in range(start_day, start_day + horizon):
        if not alive:
            break
        if day in fertilizer and day in forecast_days:
            successful_fertilizer.add(day)
            active_fertilizer.update(range(day, day + 3))
        watered = day in watering
        age = day - start_day
        if not cd["ongoing"]:
            window_start = (cd["max_yield_day"] + 1) // 2
            if watered and window_start <= age <= cd["max_yield_day"]:
                units = min(cd["max_yield"], units + (2 if day in active_fertilizer else 1))
        # The interpreter accepts actions first and decays the plant afterward.
        # This matters on the first lifespan step: a harvest on that turn gets
        # the pre-decay buffer rather than a prematurely decayed one.
        harvestable = harvest_day is not None and _whole(harvest_day) - start_day >= cd["first_yield_day"]
        if harvestable and day == _whole(harvest_day) and units:
            harvested += units
            units = 0
            if not cd["ongoing"]:
                alive = False
                break
        if lifespan_step >= 0:
            first = max(day * turns_per_day, lifespan_step)
            if day * turns_per_day <= lifespan_step < (day + 1) * turns_per_day:
                for step in range(first, (day + 1) * turns_per_day):
                    if (step - lifespan_step) % 2 == 0 and units:
                        units -= 1
                        decayed += 1
                        if units <= 0:
                            alive = False
                            break
        if not alive:
            break
        units_before_decay = max(units_before_decay, units)
        consecutive_unwatered = 0 if watered else consecutive_unwatered + 1
        if consecutive_unwatered >= 2:
            alive = False
            units = 0
            continue
        if cd["ongoing"]:
            days_since_first = day + 1 - start_day - cd["first_yield_day"]
            if days_since_first >= 0 and days_since_first % cd["interval"] == 0:
                count = days_since_first // cd["interval"] + 1
                if count <= cd["max_yield"]:
                    units = min(cd["max_yield"], units + (2 if watered and day in active_fertilizer else 1))
                    if count == cd["max_yield"]:
                        lifespan_step = (day + 2) * turns_per_day
    output_units = harvested + units
    saleable_units = harvested if not cd["ongoing"] else output_units
    base_inventory = _number(market_inventory, MARKET_I0)
    crop_quotes = _forecast_sale_quotes(crop, saleable_units, base_inventory, prices, params, fixed_price_mode)
    revenue = sum(crop_quotes)
    fertilizer_used = len(successful_fertilizer)
    if fixed_price_mode and isinstance(prices, Mapping) and "FERTILIZER" in prices:
        fertilizer_price = _quote("FERTILIZER", base_inventory, prices)
    else:
        fertilizer_price = market_price("FERTILIZER", base_inventory, params)
    fertilizer_cost = max(0, fertilizer_used - _whole(fertilizer_owned)) * fertilizer_price
    cap = _whole(shed_capacity if shed_capacity is not None else shed_cap, DEFAULT_SHED_CAPACITY)
    overflow_units = max(0, _whole(held_inventory) + output_units - cap)
    result = {
        "kind": "crop", "crop": crop, "start_day": start_day, "horizon": horizon,
        "units": output_units, "units_before_decay": units_before_decay,
        "saleable_units": saleable_units,
        "harvested_units": harvested, "decayed_units": decayed,
        "fertilizer_days_active": active_fertilizer & set(range(start_day, start_day + horizon)),
        "seed_cost": 0 if seed_owned else cd["seed"],
        "fertilizer_cost": fertilizer_cost, "watering_cost": 0,
    }
    result.update(_cash_terms(
        revenue=revenue,
        material_cost=(0 if seed_owned else cd["seed"]) + fertilizer_cost,
        worker_cost=_nonnegative(worker_cost), land_cost=_nonnegative(land_cost),
        movement_turns=_whole(movement_turns), action_turns=_whole(action_turns),
        horizon=horizon, overflow_units=overflow_units, output_units=output_units,
        floor_units=sum(price <= PRICE_FLOOR for price in crop_quotes),
    ))
    return result


def forecast_animal(
    animal: str, start_day: int = 0, horizon: int = season_days, *,
    feed_days: Any = None, care_days: Any = None, collect_fertilizer_days: Any = None,
    harvest_days: Any = None,
    market_inventory: float = MARKET_I0, prices: Mapping[str, Any] | None = None,
    animal_owned: bool = False, worker_cost: float = 0, land_cost: float = 0,
    movement_turns: int = 0, action_turns: int = 0, held_inventory: int = 0,
    shed_cap: int = DEFAULT_SHED_CAPACITY, shed_capacity: int | None = None,
    params: Mapping[str, Any] | None = None,
    fixed_price_mode: bool = False,
) -> dict[str, Any]:
    """Forecast one animal with one-wheat feed, care bonuses, caps, and manure."""
    if animal not in ANIMALS:
        raise KeyError(animal)
    ad = ANIMALS[animal]
    start_day, horizon = _whole(start_day), _whole(horizon)
    feed = _days(feed_days, set(range(start_day, start_day + horizon)))
    care = _days(care_days, set())
    collect = _days(collect_fertilizer_days, set(range(start_day, start_day + horizon)))
    units = fertilizer_units = consecutive_unfed = pending_care = feed_units = 0
    harvested_units = held_units = production_events = 0
    # ``None`` means an ideal collection schedule: sell each production as it
    # is made, so max_held remains a cap on inventory rather than lifetime yield.
    scheduled_harvest = _days(harvest_days, set())
    auto_collect = harvest_days is None
    fertilizer_available = False
    escaped = False
    for day in range(start_day, start_day + horizon):
        fed = day in feed
        if fed:
            feed_units += 1
        consecutive_unfed = 0 if fed else consecutive_unfed + 1
        if consecutive_unfed >= 2:
            escaped, units, held_units = True, 0, 0
            break
        if not auto_collect and day in scheduled_harvest and held_units:
            harvested_units += held_units
            held_units = 0
        days_since_first = day + 1 - start_day - ad["first_yield_day"]
        if days_since_first >= 0 and days_since_first % ad["interval"] == 0:
            production_events += 1
            production = 1 + (pending_care if fed else 0)
            held_units = min(ad["max_held"], held_units + production)
            if auto_collect:
                harvested_units += held_units
                held_units = 0
            pending_care = 0
        if fed and day in care:
            pending_care += 1
        # Manure is created by refresh at day-end and can be collected next day.
        if day in collect and fertilizer_available:
            fertilizer_units += 1
            fertilizer_available = False
        fertilizer_available = True
    product = ad["product"]
    base_inventory = _number(market_inventory, MARKET_I0)
    output_units = harvested_units + held_units
    product_quotes = _forecast_sale_quotes(product, output_units, base_inventory, prices, params, fixed_price_mode)
    fertilizer_quotes = _forecast_sale_quotes("FERTILIZER", fertilizer_units, base_inventory, prices, params, fixed_price_mode)
    product_quote = product_quotes[0] if product_quotes else market_price(product, base_inventory, params)
    fertilizer_quote = fertilizer_quotes[0] if fertilizer_quotes else market_price("FERTILIZER", base_inventory, params)
    revenue = sum(product_quotes) + sum(fertilizer_quotes)
    feed_quote = _quote("WHEAT", base_inventory, prices) if isinstance(prices, Mapping) and "WHEAT" in prices else market_price("WHEAT", base_inventory, params)
    feed_cost = feed_units * feed_quote
    cap = _whole(shed_capacity if shed_capacity is not None else shed_cap, DEFAULT_SHED_CAPACITY)
    overflow_units = max(0, _whole(held_inventory) + held_units + fertilizer_units - cap)
    floor_units = sum(price <= PRICE_FLOOR for price in product_quotes + fertilizer_quotes)
    result = {
        "kind": "animal", "animal": animal, "product": product,
        "start_day": start_day, "horizon": horizon, "first_yield_day": ad["first_yield_day"],
        "units": output_units, "held_units": held_units, "harvested_units": harvested_units,
        "production_events": production_events,
        "feed_units": feed_units, "feed_cost": feed_cost,
        "fertilizer_units": fertilizer_units, "escaped": escaped,
        "animal_cost": 0 if animal_owned else ad["cost"],
        "care_days": care & set(range(start_day, start_day + horizon)),
    }
    result.update(_cash_terms(
        revenue=revenue,
        material_cost=feed_cost + (0 if animal_owned else ad["cost"]),
        worker_cost=_nonnegative(worker_cost), land_cost=_nonnegative(land_cost),
        movement_turns=_whole(movement_turns), action_turns=_whole(action_turns),
        horizon=horizon, overflow_units=overflow_units, output_units=output_units,
        floor_units=floor_units,
    ))
    return result


def feed_reserve(animals: Mapping[str, Any], days: int, existing_wheat: int = 0,
                 reserve_wheat: int = 0) -> int:
    """Return additional wheat needed for feed plus a requested reserve."""
    count = sum(_whole(number) for number in animals.values()) if isinstance(animals, Mapping) else 0
    return max(0, count * _whole(days) + _whole(reserve_wheat) - _whole(existing_wheat))


def expected_portfolio_cash(
    forecasts: Any, starting_cash: float = 0, *, worker_cost: float = 0,
    land_cost: float = 0, movement_turns: int = 0, action_turns: int = 0,
) -> dict[str, Any]:
    """Aggregate forecast cash, turns, and independent risk terms."""
    entries = list(forecasts) if isinstance(forecasts, (list, tuple)) else [forecasts]
    entries = [entry for entry in entries if isinstance(entry, Mapping)]
    cash = _number(starting_cash) + sum(_number(entry.get("net_cash")) for entry in entries)
    cash -= _nonnegative(worker_cost) + _nonnegative(land_cost)
    turns = sum(_whole(entry.get("turns")) for entry in entries) + _whole(movement_turns) + _whole(action_turns)
    return {
        "cash": cash, "net_cash": cash - _number(starting_cash), "turns": turns,
        "shed_overflow_risk": min(1.0, sum(_number(e.get("shed_overflow_risk")) for e in entries)),
        "price_floor_risk": min(1.0, sum(_number(e.get("price_floor_risk")) for e in entries)),
    }


def opportunity_score(forecast: Mapping[str, Any], turn_value: float = 0, risk_penalty: float = 0) -> float:
    """Score cash after turn opportunity cost and explicit output risks."""
    if not isinstance(forecast, Mapping):
        return float("-inf")
    risks = _number(forecast.get("shed_overflow_risk")) + _number(forecast.get("price_floor_risk"))
    return _number(forecast.get("net_cash")) - _nonnegative(turn_value) * _whole(forecast.get("turns")) - _nonnegative(risk_penalty) * risks
