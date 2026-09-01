from dataclasses import dataclass
from collections.abc import Mapping, Sequence


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
