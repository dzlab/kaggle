"""Optional dependency-free learned-policy adapter and intent compiler.

Models may choose a worker's task target, but they never choose an engine
movement path.  The compiler below is deliberately conservative: malformed
model output is ignored and every emitted command still goes through the
deterministic policy's legality checks.
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .constants import ANIMALS, CROPS, PRODUCTS
from .memory import PolicyMemory
from .routing import is_locked_tile, normalize_position, route_to
from .types import Position, Task, WorkerAssignment


@dataclass(frozen=True)
class WorkerProposal:
    worker_index: int
    kind: str
    target: Position | None
    item: str | None
    score: float


@dataclass(frozen=True)
class PolicyProposal:
    workers: tuple[WorkerProposal, ...]
    market_orders: tuple[tuple[str, str | None, int], ...]
    confidence: float
    model_version: str


_EMPTY_PROPOSAL = PolicyProposal((), (), 0.0, "none")
_VALID_KINDS = frozenset({
    "PICKUP", "PLACE", "SHED", "SELL", "SELL_ALL", "DROP", "PLANT",
    "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER",
    "WEED", "DIG", "ANIMAL", "STRUCTURE", "BUILD_COOP", "BUILD_PASTURE",
})
_ITEM_KINDS = frozenset({"PICKUP", "PLACE", "PLANT", "FERTILIZE", "FEED", "ANIMAL", "SELL"})


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if isfinite(result) else default


def _empty() -> PolicyProposal:
    return _EMPTY_PROPOSAL


def _process_entry(connection: Any, function: Any) -> None:
    """Execute model code in an isolated child and return only serializable data."""
    try:
        connection.send((True, function()))
    except BaseException as exc:  # model code must not escape the entrypoint
        try:
            connection.send((False, type(exc).__name__))
        except BaseException:
            pass
    finally:
        connection.close()


def _run_with_timeout(function: Any, timeout: float) -> tuple[bool, Any]:
    """Run model code in a killable process so timed-out code cannot continue."""
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        context = multiprocessing.get_context()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_process_entry, args=(sender, function))
    process.daemon = True
    try:
        process.start()
    except BaseException as exc:
        receiver.close()
        sender.close()
        return False, exc
    sender.close()
    process.join(max(0.0, timeout))
    if process.is_alive():
        process.terminate()
        process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        if process.is_alive():
            receiver.close()
            return False, RuntimeError("learned model process did not terminate")
        receiver.close()
        return False, TimeoutError("learned model timed out")
    try:
        return receiver.recv() if receiver.poll() else (False, RuntimeError("learned model returned no result"))
    except (EOFError, OSError, TypeError):
        return False, RuntimeError("learned model returned an unreadable result")
    finally:
        receiver.close()


def _coerce_worker(raw: Any) -> WorkerProposal | None:
    if isinstance(raw, WorkerProposal):
        return raw
    index = _get(raw, "worker_index", _get(raw, "index"))
    kind = _get(raw, "kind")
    if index is None or kind is None:
        return None
    try:
        worker_index = int(index)
    except (TypeError, ValueError, OverflowError):
        return None
    item = _get(raw, "item")
    item = str(item).upper() if item is not None else None
    return WorkerProposal(
        worker_index=worker_index,
        kind=str(kind).upper(),
        target=normalize_position(_get(raw, "target")),
        item=item,
        score=_number(_get(raw, "score")),
    )


def _coerce_market(raw: Any) -> tuple[str, str | None, int] | None:
    if isinstance(raw, (list, tuple)):
        if len(raw) < 1:
            return None
        kind, item, quantity = raw[0], raw[1] if len(raw) > 1 else None, raw[2] if len(raw) > 2 else 1
    else:
        kind = _get(raw, "kind")
        item = _get(raw, "item")
        quantity = _get(raw, "quantity", 1)
    if kind is None:
        return None
    try:
        quantity = int(quantity)
    except (TypeError, ValueError, OverflowError):
        return None
    if quantity <= 0:
        return None
    return str(kind).upper(), str(item).upper() if item is not None else None, quantity


def _coerce_proposal(raw: Any) -> PolicyProposal | None:
    if isinstance(raw, PolicyProposal):
        return raw
    if not isinstance(raw, Mapping):
        return None
    workers_raw = _get(raw, "workers", ())
    market_raw = _get(raw, "market_orders", _get(raw, "market", ()))
    if not isinstance(workers_raw, Sequence) or isinstance(workers_raw, (str, bytes)):
        return None
    if not isinstance(market_raw, Sequence) or isinstance(market_raw, (str, bytes)):
        return None
    workers = tuple(worker for raw_worker in workers_raw if (worker := _coerce_worker(raw_worker)) is not None)
    market = tuple(order for raw_order in market_raw if (order := _coerce_market(raw_order)) is not None)
    return PolicyProposal(
        workers=workers,
        market_orders=market,
        confidence=_number(_get(raw, "confidence")),
        model_version=str(_get(raw, "model_version", "unknown")),
    )


class LearnedPolicy:
    """Load and invoke an optional model without importing ML dependencies."""

    def __init__(self, model_path: str | Path | Any = None, *, timeout_seconds: float = 0.25) -> None:
        self.model_path = model_path
        self.timeout_seconds = max(0.01, _number(timeout_seconds, 0.25))
        self.diagnostics: dict[str, Any] = {"status": "disabled" if model_path is None else "unloaded"}
        self._model: Any = None
        self._loaded = False

    def _load(self) -> Any:
        if self.model_path is None:
            return None
        if not isinstance(self.model_path, (str, Path)):
            return self.model_path
        path = Path(self.model_path)
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("learned model must be a JSON object")
        return value

    def _invoke(self, model: Any, state: Any, features: Any) -> Any:
        if isinstance(model, Mapping) and ("workers" in model or "market_orders" in model or "market" in model):
            return model
        if hasattr(model, "propose"):
            return model.propose(state, features)
        if hasattr(model, "predict"):
            return model.predict(features)
        if callable(model):
            return model(state, features)
        raise TypeError("model has no propose, predict, or callable interface")

    def propose(self, state: Any, features: Any) -> PolicyProposal:
        if self.model_path is None:
            self.diagnostics = {"status": "disabled"}
            return _empty()
        if not self._loaded:
            ok, loaded = _run_with_timeout(self._load, self.timeout_seconds)
            if not ok:
                status = "slow_model" if isinstance(loaded, TimeoutError) else (
                    "missing_model"
                    if isinstance(loaded, FileNotFoundError) or loaded == "FileNotFoundError"
                    else "load_error"
                )
                self.diagnostics = {"status": status, "error": type(loaded).__name__}
                return _empty()
            self._model = loaded
            self._loaded = True
        if self._model is None:
            self.diagnostics = {"status": "incompatible_model"}
            return _empty()
        ok, output = _run_with_timeout(
            lambda: self._invoke(self._model, state, features), self.timeout_seconds,
        )
        if not ok:
            status = "slow_model" if isinstance(output, TimeoutError) else "inference_error"
            self.diagnostics = {"status": status, "error": type(output).__name__}
            return _empty()
        proposal = _coerce_proposal(output)
        if proposal is None:
            self.diagnostics = {"status": "incompatible_model"}
            return _empty()
        self.diagnostics = {"status": "ok", "model_version": proposal.model_version}
        return proposal


def _board_size(state: Any) -> int:
    raw = _get(state, "board_size")
    try:
        if raw is not None and int(raw) > 0:
            return int(raw)
    except (TypeError, ValueError, OverflowError):
        pass
    tiles = _get(state, "tiles", _get(_get(state, "farm", {}), "tiles", ()))
    if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
        return max(1, len(tiles), *(len(row) for row in tiles if isinstance(row, Sequence)))
    if isinstance(tiles, Mapping):
        positions = [normalize_position(key) for key in tiles]
        return max((max(position.x, position.y) + 1 for position in positions if position is not None), default=1)
    return 1


def _workers(state: Any) -> list[tuple[int, Position | None]]:
    source = _get(state, "workers", _get(_get(state, "farm", {}), "workers", ()))
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        return []
    result = []
    for fallback, worker in enumerate(source):
        try:
            index = int(_get(worker, "index", fallback))
        except (TypeError, ValueError, OverflowError):
            continue
        result.append((index, normalize_position(_get(worker, "position", worker))))
    return sorted(result)


def _tile_at(state: Any, position: Position) -> Any:
    tiles = _get(state, "tiles", _get(_get(state, "farm", {}), "tiles", ()))
    if isinstance(tiles, Mapping):
        return next((tile for raw, tile in tiles.items() if normalize_position(raw) == position), None)
    if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) and 0 <= position.y < len(tiles):
        row = tiles[position.y]
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and 0 <= position.x < len(row):
            return row[position.x]
    return None


def _counts(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {str(key).upper(): max(0, int(_number(quantity))) for key, quantity in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: dict[str, int] = {}
        for item in value:
            name = item if isinstance(item, str) else _get(item, "item", _get(item, "kind"))
            if name:
                key = str(name).upper()
                result[key] = result.get(key, 0) + max(1, int(_number(_get(item, "quantity", 1), 1)))
        return result
    return {}


def _inventory(state: Any, index: int) -> dict[str, int]:
    private = _get(state, "private", {})
    inventories = _get(private, "inventories", _get(state, "inventories", ()))
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)) and 0 <= index < len(inventories):
        return _counts(inventories[index])
    return {}


def _shed(state: Any) -> dict[str, int]:
    private = _get(state, "private", {})
    return _counts(_get(private, "shed", _get(state, "shed", {})))


def _available_item(state: Any, worker_index: int, item: str) -> bool:
    return _inventory(state, worker_index).get(item, 0) > 0 or _shed(state).get(item, 0) > 0


def _proposal_is_usable(state: Any, worker_index: int, kind: str, item: str | None) -> bool:
    item = item.upper() if item else None
    if item is not None and item not in set(PRODUCTS) | set(ANIMALS):
        return False
    if kind == "PLANT":
        seeds = _get(_get(state, "private", {}), "seeds", _get(state, "seeds", {}))
        return item in CROPS and isinstance(seeds, Mapping) and _number(seeds.get(item)) > 0
    required = {"FERTILIZE": "FERTILIZER", "FEED": "WHEAT"}.get(kind)
    if required is not None and not _available_item(state, worker_index, required):
        return False
    if kind in {"PICKUP", "PLACE", "ANIMAL", "SELL"} and item is None:
        return False
    if kind == "PICKUP":
        return _shed(state).get(item or "", 0) > 0
    if kind in {"PLACE", "ANIMAL"}:
        return _available_item(state, worker_index, item or "")
    if kind == "SELL":
        return _shed(state).get(item or "", 0) > 0
    return True


def _strategy_allows(state: Any, kind: str, item: str | None, target: Position | None, strategy: Any,
                     crop_count: int, animal_count: int) -> bool:
    if strategy is None:
        return True
    from .planner import _task_allowed, normalize_planner_state

    task = Task(kind, target, 0, None, 0.0, item=item)
    if not _task_allowed(task, strategy, normalize_planner_state(state)):
        return False
    try:
        if kind == "PLANT" and crop_count >= max(0, int(_get(strategy, "max_crop_units", 0))):
            return False
        if kind in {"ANIMAL", "PLACE"} and item in ANIMALS and animal_count >= max(0, int(_get(strategy, "max_animal_units", 0))):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _proposal_sort_key(candidate: WorkerProposal) -> tuple[Any, ...]:
    target = normalize_position(candidate.target)
    return (
        -_number(candidate.score), candidate.kind,
        target.y if target is not None else float("inf"),
        target.x if target is not None else float("inf"),
        candidate.item or "",
    )


def _existing_counts(state: Any, strategy: Any = None) -> tuple[int, int]:
    from .policy import _existing_animal_units, _iter_tiles, _crop

    allowed = set(_get(strategy, "crops", ()) or ()) if strategy is not None else set(CROPS)
    allowed = {str(value).upper() for value in allowed}
    crops = sum(1 for _, tile in _iter_tiles(state) if _crop(tile) in allowed)
    return crops, _existing_animal_units(state)


def _carried_assignment(state: Any, memory: PolicyMemory, worker_index: int) -> WorkerAssignment | None:
    # Importing these private helpers here avoids a policy/learned-policy
    # import cycle while sharing the exact existing legality rules.
    from .policy import _assignment_valid, _get as policy_get, _inventory_for_worker, _required_worker_item

    for assignment in memory.assignments:
        if int(policy_get(assignment, "worker_index", -1)) != worker_index:
            continue
        kind = str(policy_get(assignment.task, "kind", "")).upper()
        if kind not in {"FEED", "FERTILIZE", "ANIMAL", "PLACE"}:
            continue
        required = _required_worker_item(assignment.task, state)
        if required and _inventory_for_worker(state, worker_index).get(required, 0) > 0 and _assignment_valid(state, assignment):
            return assignment
    return None


def _fallback_assignments(state: Any, memory: PolicyMemory, strategy: Any = None) -> list[WorkerAssignment]:
    from .planner import assign_tasks, build_daily_plan
    from .policy import _assignment_valid

    valid = [assignment for assignment in memory.assignments if _assignment_valid(state, assignment)]
    if valid:
        return valid
    return assign_tasks(build_daily_plan(state, memory, strategy), _get(state, "workers", ()), state, strategy)


def _route_positions(start: Position | None, target: Position | None, board_size: int) -> list[Position]:
    if start is None or target is None:
        return []
    positions: list[Position] = []
    current = start
    for move in route_to(start, target, board_size):
        current = Position(
            current.x + (move == "EAST") - (move == "WEST"),
            current.y + (move == "SOUTH") - (move == "NORTH"),
        )
        positions.append(current)
    return positions


def compile_proposal(state: Any, proposal: PolicyProposal, memory: PolicyMemory,
                     strategy: Any = None) -> dict[str, Any]:
    """Compile target/task intents into the normal action schema."""
    from .policy import PASS, _get as policy_get, _unit_command, _worker_records, build_market_orders, worker_action

    board_size = _board_size(state)
    worker_records = _worker_records(state)
    known = {record["index"]: record for record in worker_records}
    candidates_by_worker: dict[int, list[WorkerProposal]] = {}
    existing_crops, existing_animals = _existing_counts(state, strategy)
    for candidate in proposal.workers if isinstance(proposal, PolicyProposal) else ():
        if candidate.worker_index not in known or candidate.kind not in _VALID_KINDS:
            continue
        target = normalize_position(candidate.target)
        if target is not None and not (0 <= target.x < board_size and 0 <= target.y < board_size):
            continue
        if target is not None and is_locked_tile(_tile_at(state, target)):
            continue
        if not _proposal_is_usable(state, candidate.worker_index, candidate.kind, candidate.item):
            continue
        if not _strategy_allows(state, candidate.kind, candidate.item, target, strategy,
                                existing_crops, existing_animals):
            continue
        candidates_by_worker.setdefault(candidate.worker_index, []).append(candidate)

    selected: dict[int, WorkerProposal] = {}
    crop_count, animal_count = existing_crops, existing_animals
    for worker_index in sorted(candidates_by_worker):
        candidate = min(candidates_by_worker[worker_index], key=_proposal_sort_key)
        if candidate.kind == "PLANT":
            if strategy is not None and crop_count >= int(_get(strategy, "max_crop_units", 0)):
                continue
            crop_count += 1
        if candidate.kind in {"ANIMAL", "PLACE"} and candidate.item in ANIMALS:
            if strategy is not None and animal_count >= int(_get(strategy, "max_animal_units", 0)):
                continue
            animal_count += 1
        selected[worker_index] = candidate

    assignments = _fallback_assignments(state, memory, strategy)
    by_worker = {int(policy_get(assignment, "worker_index", -1)): assignment for assignment in assignments}
    for worker_index in sorted(known):
        carried = _carried_assignment(state, memory, worker_index)
        if carried is not None:
            by_worker[worker_index] = carried
            continue
        candidate = selected.get(worker_index)
        if candidate is None:
            continue
        target = normalize_position(candidate.target) or known[worker_index]["position"]
        if target is None:
            continue
        task = Task(candidate.kind, target, int(max(0, _number(candidate.score))), None, max(0.0, _number(candidate.score)), item=candidate.item)
        by_worker[worker_index] = WorkerAssignment(
            worker_index, task, _route_positions(known[worker_index]["position"], target, board_size),
        )
    memory.assignments = [by_worker[index] for index in sorted(by_worker) if index in known]
    market = build_market_orders(
        state, proposal.market_orders if isinstance(proposal, PolicyProposal) else (), strategy,
    )
    commands = {
        index: worker_action(index, state, by_worker.get(index))
        for index in sorted(known)
    }
    farmer = commands.get(0, [PASS])
    visible_hands = policy_get(policy_get(state, "farm", {}), "hands", ())
    if not isinstance(visible_hands, Sequence) or isinstance(visible_hands, (str, bytes)):
        visible_hands = [record for record in worker_records if record["index"] != 0]
    return {
        "farmer": farmer,
        "hands": [commands.get(index + 1, [PASS]) for index in range(len(visible_hands))],
        "market": market,
    }
