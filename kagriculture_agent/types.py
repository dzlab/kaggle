from dataclasses import dataclass, field
from typing import Any

TaskTarget = Any
Deadline = int | None


@dataclass(frozen=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class TileRef:
    position: Position
    tile: Any


@dataclass
class Task:
    kind: str
    target: TaskTarget
    priority: int
    deadline: Deadline
    value: float


@dataclass
class WorkerAssignment:
    worker_index: int
    task: Task
    route: list[Position] = field(default_factory=list)


@dataclass
class EconomicEstimate:
    cash_delta: float
    turns: int
    risk: float


@dataclass
class EpisodeMemory:
    last_day: int = -1
    last_hour: int = -1
    assignments: list[WorkerAssignment] = field(default_factory=list)
    sell_batches: list[Any] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
