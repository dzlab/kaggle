from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from math import isfinite

from .constants import MARKET_I0, PRICE_FLOOR, PRODUCTS, SHOP_DEMANDS, shed_capacity
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


_SALEABLE_MARKET_ITEMS = frozenset(PRODUCTS) - {"FERTILIZER"}


def _public_market_inventory(state: object) -> Mapping[str, object]:
    if not isinstance(state, Mapping):
        return {}
    market = state.get("market", {})
    if not isinstance(market, Mapping):
        return {}
    inventory = market.get("inventory", {})
    return inventory if isinstance(inventory, Mapping) else {}


def _demand_snapshot(value: object) -> tuple[object, ...] | None:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key).upper(), repr(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(sorted(str(item).upper() for item in value))
    return None


def _public_town_demand(state: object) -> tuple[tuple[object, ...] | None, bool]:
    if not isinstance(state, Mapping):
        return None, False
    town = state.get("town", {})
    if not isinstance(town, Mapping):
        return None, False
    explicit_refresh = any(
        bool(town.get(key))
        for key in ("demand_refreshed", "demand_refresh", "refill", "refilled")
    )
    for key in ("demand", "demands", "requested_items"):
        if key in town:
            return _demand_snapshot(town.get(key)), explicit_refresh
    shops = town.get("unlocked_shops")
    if isinstance(shops, Sequence) and not isinstance(shops, (str, bytes, bytearray)):
        demand = {
            item
            for shop in shops
            if isinstance(shop, str)
            for item in SHOP_DEMANDS.get(shop.upper(), ())
        }
        return _demand_snapshot(demand), explicit_refresh
    return None, explicit_refresh


@dataclass
class OpponentMarketSignal:
    """Track repeated public market pressure without attributing its source."""

    history: dict[str, list[float]] | None = None
    _previous_inventory: dict[str, float] | None = None
    _previous_demand: tuple[object, ...] | None = None
    _has_demand_observation: bool = False
    town_refill: bool = False
    evidence: int = 0

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = {}
        if self._previous_inventory is None:
            self._previous_inventory = {}

    def reset(self) -> None:
        self.history.clear()
        self._previous_inventory.clear()
        self._previous_demand = None
        self._has_demand_observation = False
        self.town_refill = False
        self.evidence = 0

    def observe(self, state: object) -> str | None:
        """Return an item only after three strict, monotonic public decreases."""
        inventory = _public_market_inventory(state)
        demand, explicit_refresh = _public_town_demand(state)
        demand_refresh = explicit_refresh or (
            self._has_demand_observation and demand != self._previous_demand
        )
        self.town_refill = demand_refresh
        if demand_refresh:
            self.history.clear()
            self._previous_inventory.clear()

        candidate: str | None = None
        self.evidence = 0
        for raw_item, raw_value in inventory.items():
            item = str(raw_item).upper()
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not isfinite(value):
                continue
            previous = self._previous_inventory.get(item)
            values = self.history.get(item, [])
            if previous is None:
                values = [value]
            elif value < previous:
                values = [*values, value] if values else [previous, value]
            else:
                values = [value]
            self.history[item] = values[-3:]
            if len(self.history[item]) == 3 and all(
                left > right for left, right in zip(self.history[item], self.history[item][1:])
            ):
                if candidate is None or item < candidate:
                    candidate = item
                self.evidence = max(self.evidence, len(self.history[item]))
            self._previous_inventory[item] = value

        self._previous_demand = demand
        self._has_demand_observation = True
        return candidate


def _evidence_count(evidence: object) -> int:
    if isinstance(evidence, bool):
        return 0
    if isinstance(evidence, Mapping):
        for key in ("strength", "count", "observations", "evidence"):
            if key in evidence:
                return _evidence_count(evidence[key])
        return 0
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        return len(evidence)
    try:
        value = int(evidence)
    except (TypeError, ValueError, OverflowError):
        return 0
    return value if value >= 0 else 0


def _evidence_has_owned_batch(evidence: object) -> bool:
    if not isinstance(evidence, Mapping):
        return True
    for key in ("owned_batch", "owned_quantity", "sellable_batch"):
        if key in evidence:
            return _evidence_count(evidence[key]) > 0
    if "prerequisite" in evidence:
        return bool(evidence["prerequisite"])
    return True


def should_front_run(item: str, current_price: object, evidence: object,
                     town_refill: object) -> bool:
    """Allow an opt-in pre-sale only with strong public evidence and a live quote."""
    item = str(item).upper() if item is not None else ""
    try:
        price = float(current_price)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        item in _SALEABLE_MARKET_ITEMS
        and isfinite(price)
        and price > PRICE_FLOOR
        and _evidence_count(evidence) >= 3
        and _evidence_has_owned_batch(evidence)
        and not bool(town_refill)
    )


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


def market_sale_quotes(item: str, quantity: int, state: object) -> list[int]:
    """Return sequential sale quotes anchored to the observed current price."""
    try:
        quantity = max(0, min(int(quantity), shed_capacity))
    except (TypeError, ValueError, OverflowError):
        quantity = 0
    if quantity == 0:
        return []

    raw_state = state if isinstance(state, Mapping) else {}
    market = raw_state.get("market", {})
    market = market if isinstance(market, Mapping) else {}
    observed_prices = next(
        (value for value in (
            raw_state.get("observed_prices"),
            raw_state.get("market_prices"),
            market.get("prices"),
            raw_state.get("prices"),
        ) if isinstance(value, Mapping)),
        market,
    )
    observed_price = observed_prices.get(item) if isinstance(observed_prices, Mapping) else None
    try:
        observed_price = float(observed_price)
    except (TypeError, ValueError, OverflowError):
        observed_price = None
    if observed_price is not None and not isfinite(observed_price):
        observed_price = None

    inventory = next(
        (value for value in (
            raw_state.get("observed_market_inventory"),
            raw_state.get("market_inventory"),
            market.get("inventory"),
        ) if value is not None),
        MARKET_I0,
    )
    if isinstance(inventory, Mapping):
        inventory = inventory.get(item, MARKET_I0)
    try:
        inventory = float(inventory)
    except (TypeError, ValueError, OverflowError):
        inventory = float(MARKET_I0)
    params = market.get("params", market.get("price_params"))
    if not isinstance(params, Mapping):
        params = raw_state.get("market_params", raw_state.get("price_params"))

    quotes: list[int] = []
    current_inventory = inventory
    try:
        curve_anchor = market_price(item, current_inventory, params)
    except (KeyError, TypeError, ValueError, OverflowError):
        curve_anchor = None
    for _ in range(quantity):
        try:
            curve_quote = market_price(item, current_inventory, params)
        except (KeyError, TypeError, ValueError, OverflowError):
            curve_quote = PRICE_FLOOR
        if observed_price is None or curve_anchor is None:
            quote = curve_quote
        else:
            quote = max(PRICE_FLOOR, int(round(observed_price + curve_quote - curve_anchor)))
        quotes.append(quote)
        if quote > PRICE_FLOOR:
            current_inventory += 1
    return quotes


def market_order_score(item: str, quantity: int, state: object, urgency: float) -> float:
    """Score a sale from sequential post-sale quotes and its price impact."""
    try:
        urgency = float(urgency)
    except (TypeError, ValueError, OverflowError):
        urgency = 0.0
    quotes = market_sale_quotes(item, quantity, state)
    if not quotes:
        return urgency
    average_quote = sum(quotes) / len(quotes)
    impact_penalty = max(0.0, quotes[0] - average_quote) * len(quotes)
    return sum(quotes) - impact_penalty + urgency
