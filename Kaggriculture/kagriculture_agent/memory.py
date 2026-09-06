"""Small, resettable state holder for the policy's episode-local decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .strategy import OpponentMarketSignal
from .types import EpisodeMemory as PlannerMemory


@dataclass
class PolicyMemory(PlannerMemory):
    """Assignments and diagnostics that must never leak across episodes."""

    market_regime: dict[str, str] = field(default_factory=dict)
    market_history: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    selected_strategy: str | None = None
    opponent_signal: OpponentMarketSignal = field(default_factory=OpponentMarketSignal)

    def reset(self, *, day: int = -1, hour: int = -1, reason: str = "manual") -> None:
        self.assignments.clear()
        self.sell_batches.clear()
        self.diagnostics.clear()
        self.market_regime.clear()
        self.market_history.clear()
        self.selected_strategy = None
        self.opponent_signal.reset()
        self.last_day = day
        self.last_hour = hour
        self.diagnostics["reset_reason"] = reason

    def observe_time(self, day: Any, hour: Any) -> bool:
        """Record a clock value and clear stale work on a clock reset."""
        if isinstance(day, bool) or isinstance(hour, bool):
            return False
        try:
            current_day, current_hour = int(day), int(hour)
        except (TypeError, ValueError, OverflowError):
            return False
        if current_day < 0 or current_hour < 0:
            return False
        previous = (self.last_day, self.last_hour)
        if (current_day, current_hour) == (0, 0):
            if previous == (0, 0):
                return False
            self.reset(day=0, hour=0, reason="episode_start")
            return True
        if previous != (-1, -1) and (current_day, current_hour) < previous:
            self.reset(day=current_day, hour=current_hour, reason="time_backward")
            return True
        if current_hour == 0 and current_day != self.last_day:
            # Daily assignments are stale at midnight, but market direction
            # history remains episode-local and must span the day boundary.
            market_history = {
                item: list(entries) for item, entries in self.market_history.items()
            }
            self.reset(day=current_day, hour=0, reason="day_start")
            self.market_history = market_history
            return True
        self.last_day, self.last_hour = current_day, current_hour
        return False


Memory = PolicyMemory
EpisodeMemory = PolicyMemory


def market_order_allowed(memory: PolicyMemory, *, item: str, direction: str,
                         turn: int, window: int = 2, terminal: bool = False) -> bool:
    """Reject only rapid reversals for one item within an episode."""
    try:
        item_name = str(item).strip().upper()
    except Exception:
        item_name = ""
    try:
        order_direction = str(direction).strip().upper()
    except Exception:
        order_direction = ""
    if order_direction == "BUY":
        order_direction = "BUY_PRODUCT"
    if terminal or not item_name or order_direction not in {"BUY_PRODUCT", "SELL"}:
        return True
    try:
        current_turn = int(turn)
    except (TypeError, ValueError, OverflowError):
        return True
    try:
        distance = max(0, int(window))
    except (TypeError, ValueError, OverflowError):
        distance = 2
    for previous_turn, previous_direction in memory.market_history.get(item_name, ()):
        try:
            previous_turn = int(previous_turn)
            previous_direction = str(previous_direction).strip().upper()
        except Exception:
            continue
        if (0 <= current_turn - previous_turn <= distance
                and previous_direction in {"BUY_PRODUCT", "SELL"}
                and previous_direction != order_direction):
            return False
    return True


def reset_memory(memory: PolicyMemory, *, day: int = -1, hour: int = -1,
                 reason: str = "manual") -> PolicyMemory:
    """Reset and return ``memory`` so it can be used in a small pipeline."""
    memory.reset(day=day, hour=hour, reason=reason)
    return memory
