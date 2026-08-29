from collections.abc import Iterator, Mapping
from typing import Any, TypeAlias

from .types import EpisodeMemory, Position

Observation: TypeAlias = Mapping[str, Any]
Farm: TypeAlias = Mapping[str, Any]
Tile: TypeAlias = Any


def _coordinate(value: Any) -> Position | None:
    if isinstance(value, Position):
        return value
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        return None
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    try:
        return Position(int(x), int(y))
    except (TypeError, ValueError, OverflowError):
        return None


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _tiles_or_empty(value: Any) -> list[list[Tile]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [list(row) if isinstance(row, (list, tuple)) else [] for row in value]


def _valid_index(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def parse_observation(obs: Any) -> dict[str, Any]:
    """Return a stable, player-specific view while tolerating omitted fields."""
    source: Observation = obs if isinstance(obs, Mapping) else {}
    player = _valid_index(source.get("player", 0))
    farms = source.get("farms", [])
    farm = farms[player] if isinstance(farms, (list, tuple)) and 0 <= player < len(farms) else {}
    farm = _mapping_or_empty(farm)
    farm["tiles"] = _tiles_or_empty(farm.get("tiles"))
    farm["hands"] = list(farm["hands"]) if isinstance(farm.get("hands"), (list, tuple)) else []
    return {
        "player": player,
        "day": source.get("day"),
        "hour": source.get("hour"),
        "farm": farm,
        "private": _mapping_or_empty(source.get("private")),
        "market": _mapping_or_empty(source.get("market")),
        "town": _mapping_or_empty(source.get("town")),
    }


def iter_tiles(farm: Farm | Any) -> Iterator[tuple[Position, Tile]]:
    tiles = farm.get("tiles", []) if isinstance(farm, Mapping) else []
    for y, row in enumerate(tiles or []):
        for x, tile in enumerate(row or []):
            yield Position(x, y), tile


def is_tile_passable(tile: Tile) -> bool:
    """Tile contents do not block movement; bounds are checked by the caller."""
    del tile
    return True


def is_tile_actionable(tile: Tile) -> bool:
    """Locked land remains traversable but cannot receive tile operations."""
    return tile != "LOCKED"


def shed_total(private: Mapping[str, Any] | Any) -> int | float:
    shed = private.get("shed", {}) if isinstance(private, Mapping) else {}
    if not isinstance(shed, Mapping):
        return 0
    return sum(value for value in shed.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool))


def shed_access_tiles(board_size: int) -> tuple[Position, ...]:
    try:
        size = int(board_size)
    except (TypeError, ValueError, OverflowError):
        return ()
    half = size // 2
    return (Position(half - 1, half - 1), Position(half, half - 1),
            Position(half - 1, half), Position(half, half))


def is_shed_adjacent(position: Position | tuple[int, int] | Mapping[str, Any] | Any,
                     board_size: int) -> bool:
    coordinate = _coordinate(position)
    return coordinate is not None and coordinate in shed_access_tiles(board_size)


def is_episode_start(obs: Any, memory: EpisodeMemory | None) -> bool:
    source: Observation = obs if isinstance(obs, Mapping) else {}
    day, hour = source.get("day"), source.get("hour")
    if (isinstance(day, bool) or not isinstance(day, int) or
            isinstance(hour, bool) or not isinstance(hour, int)):
        return False
    if (day, hour) == (0, 0) or memory is None:
        return True
    last_day, last_hour = getattr(memory, "last_day", None), getattr(memory, "last_hour", None)
    if (isinstance(last_day, bool) or not isinstance(last_day, int) or
            isinstance(last_hour, bool) or not isinstance(last_hour, int)):
        return True
    return (day, hour) < (last_day, last_hour)
