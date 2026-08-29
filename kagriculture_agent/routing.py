"""Deterministic, bounds-safe routing helpers for the farm grid."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .observation import is_tile_actionable
from .types import Position

PASS = "PASS"
_DIRECTIONS = ("EAST", "WEST", "SOUTH", "NORTH")


def _position(value: Any) -> Position | None:
    if isinstance(value, Position):
        return value
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"))
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return Position(int(value[0]), int(value[1]))
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _board_size(value: Any) -> int | None:
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return size if size > 0 else None


def _in_bounds(position: Position, board_size: Any) -> bool:
    size = _board_size(board_size)
    return size is not None and 0 <= position.x < size and 0 <= position.y < size


def distance(a: Any, b: Any) -> int:
    """Return Manhattan distance, or zero for an invalid coordinate."""
    first, second = _position(a), _position(b)
    if first is None or second is None:
        return 0
    return abs(first.x - second.x) + abs(first.y - second.y)


def next_move(current: Any, target: Any) -> str:
    """Return one stable Manhattan step; y increases down the board."""
    current_position, target_position = _position(current), _position(target)
    if current_position is None or target_position is None:
        return PASS
    if current_position.x < target_position.x:
        return "EAST"
    if current_position.x > target_position.x:
        return "WEST"
    if current_position.y < target_position.y:
        return "SOUTH"
    if current_position.y > target_position.y:
        return "NORTH"
    return PASS


def _step(position: Position, action: str) -> Position:
    changes = {
        "EAST": (1, 0),
        "WEST": (-1, 0),
        "SOUTH": (0, 1),
        "NORTH": (0, -1),
    }
    dx, dy = changes[action]
    return Position(position.x + dx, position.y + dy)


def route_to(start: Any, target: Any, board_size: Any) -> list[str]:
    """Return a horizontal-first route that stays entirely inside the board."""
    current, destination = _position(start), _position(target)
    if current is None or destination is None:
        return []
    if not _in_bounds(current, board_size) or not _in_bounds(destination, board_size):
        return []

    route: list[str] = []
    while current != destination:
        action = next_move(current, destination)
        if action == PASS:
            break
        next_position = _step(current, action)
        if not _in_bounds(next_position, board_size):
            break
        route.append(action)
        current = next_position
    return route


def nearest_target(current: Any, targets: Iterable[Any]) -> Any | None:
    """Choose the nearest valid target, preserving input order for ties."""
    current_position = _position(current)
    if current_position is None:
        return None
    best: Any | None = None
    best_distance: int | None = None
    for candidate in targets:
        candidate_position = _position(candidate)
        if candidate_position is None:
            continue
        candidate_distance = distance(current_position, candidate_position)
        if best_distance is None or candidate_distance < best_distance:
            best, best_distance = candidate, candidate_distance
    return best


def is_locked_tile(tile: Any) -> bool:
    """Return whether a literal or mapping-shaped tile is locked."""
    if isinstance(tile, str) and tile.upper() == "LOCKED":
        return True
    if isinstance(tile, Mapping):
        if tile.get("locked") is True:
            return True
        return str(tile.get("kind", tile.get("type", tile.get("state", "")))).upper() == "LOCKED"
    return False


def route_action(
    current: Any,
    target: Any,
    board_size: Any = None,
    *,
    tile: Any = None,
    action: str | None = None,
) -> str:
    """Move toward a target, or perform its legal action when already there.

    Locked land is intentionally ignored while moving.  Once at the target it
    is not actionable, so the helper returns ``PASS``.
    """
    current_position, target_position = _position(current), _position(target)
    if current_position is None or target_position is None:
        return PASS
    if board_size is not None and (
        not _in_bounds(current_position, board_size)
        or not _in_bounds(target_position, board_size)
    ):
        return PASS
    if current_position != target_position:
        movement = next_move(current_position, target_position)
        if movement == PASS:
            return PASS
        if board_size is not None and not _in_bounds(_step(current_position, movement), board_size):
            return PASS
        return movement
    if is_locked_tile(tile) or not is_tile_actionable(tile) or not action:
        return PASS
    return action
