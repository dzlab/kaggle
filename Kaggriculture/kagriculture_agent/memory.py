"""Small, resettable state holder for the policy's episode-local decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import EpisodeMemory as PlannerMemory


@dataclass
class PolicyMemory(PlannerMemory):
    """Assignments and diagnostics that must never leak across episodes."""

    market_regime: dict[str, str] = field(default_factory=dict)
    selected_strategy: str | None = None

    def reset(self, *, day: int = -1, hour: int = -1, reason: str = "manual") -> None:
        self.assignments.clear()
        self.sell_batches.clear()
        self.diagnostics.clear()
        self.market_regime.clear()
        self.selected_strategy = None
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
            self.reset(day=0, hour=0, reason="episode_start")
            return True
        if previous != (-1, -1) and (current_day, current_hour) < previous:
            self.reset(day=current_day, hour=current_hour, reason="time_backward")
            return True
        if current_hour == 0 and current_day != self.last_day:
            self.reset(day=current_day, hour=0, reason="day_start")
            return True
        self.last_day, self.last_hour = current_day, current_hour
        return False


Memory = PolicyMemory
EpisodeMemory = PolicyMemory


def reset_memory(memory: PolicyMemory, *, day: int = -1, hour: int = -1,
                 reason: str = "manual") -> PolicyMemory:
    """Reset and return ``memory`` so it can be used in a small pipeline."""
    memory.reset(day=day, hour=hour, reason=reason)
    return memory
