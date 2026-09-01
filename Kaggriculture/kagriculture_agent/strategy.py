from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .constants import MARKET_I0, PRICE_FLOOR, shed_capacity
from .economics import market_price


@dataclass(frozen=True)
class StrategySpec:
    name: str
    crops: tuple[str, ...]
    animals: tuple[str, ...]
    max_crop_units: int
    max_animal_units: int
    reserve_wheat: int
    avoid_price_floor_sales: bool = True
    terminal_liquidation_hour: int = 22
    max_sell_batch: int = shed_capacity


STRATEGIES = {
    "current": StrategySpec(
        "current",
        ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"),
        ("GOOSE", "COW", "SHEEP"),
        10000,
        10000,
        8,
    ),
    "melon": StrategySpec("melon", ("WHEAT", "MELON"), ("COW", "SHEEP"), 80, 9, 12),
    "premium": StrategySpec("premium", ("WHEAT", "MELON"), ("COW", "SHEEP"), 48, 12, 16),
    "mixed": StrategySpec("mixed", ("WHEAT", "CARROT", "MELON"), ("COW", "SHEEP"), 64, 9, 12),
}


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported strategy: {name}") from exc


def select_strategy(state: object) -> StrategySpec:
    """Select a route from the shops visible in the opening public state."""
    if not isinstance(state, Mapping):
        return STRATEGIES["melon"]

    town = state.get("town")
    if not isinstance(town, Mapping):
        return STRATEGIES["melon"]

    raw_shops = town.get("unlocked_shops", ())
    if isinstance(raw_shops, str):
        raw_shops = (raw_shops,)
    elif not isinstance(raw_shops, Sequence) or isinstance(raw_shops, (bytes, bytearray)):
        return STRATEGIES["melon"]

    shops = {str(shop).upper() for shop in raw_shops}
    if "PET_CAFE" in shops:
        return STRATEGIES["mixed"]
    if "YARN_STORE" in shops or "ICE_CREAM_SHOP" in shops:
        return STRATEGIES["premium"]
    return STRATEGIES["melon"]


def market_order_score(item: str, quantity: int, state: object, urgency: float) -> float:
    """Score a sale from sequential post-sale quotes and its price impact."""
    try:
        quantity = max(0, min(int(quantity), shed_capacity))
    except (TypeError, ValueError, OverflowError):
        quantity = 0
    try:
        urgency = float(urgency)
    except (TypeError, ValueError, OverflowError):
        urgency = 0.0
    if quantity == 0:
        return urgency

    market = state.get("market", {}) if isinstance(state, Mapping) else {}
    market = market if isinstance(market, Mapping) else {}
    inventory = market.get("inventory", MARKET_I0)
    if isinstance(inventory, Mapping):
        inventory = inventory.get(item, MARKET_I0)
    try:
        inventory = float(inventory)
    except (TypeError, ValueError, OverflowError):
        inventory = float(MARKET_I0)
    params = market.get("params", market.get("price_params"))

    quotes: list[int] = []
    current_inventory = inventory
    for _ in range(quantity):
        quote = market_price(item, current_inventory, params)
        quotes.append(quote)
        if quote > PRICE_FLOOR:
            current_inventory += 1
    average_quote = sum(quotes) / quantity
    impact_penalty = max(0.0, quotes[0] - average_quote) * quantity
    return sum(quotes) - impact_penalty + urgency
