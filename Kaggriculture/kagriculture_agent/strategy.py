from dataclasses import dataclass


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
        raise ValueError("unsupported strategy") from exc
