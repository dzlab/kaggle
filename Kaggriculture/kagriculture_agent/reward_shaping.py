"""Pure, bounded reward-shaping and stall-detection helpers.

The helpers deliberately accept both raw engine observations and the smaller
state mappings used by the policy tests.  They do not mutate observations or
interpret a resolved truncation as a terminal outcome.
"""

from collections.abc import Mapping, Sequence
import math
from typing import Any, Literal, TypeAlias


Observation: TypeAlias = Mapping[str, Any]
ProgressStatus: TypeAlias = Literal["progress", "no_progress", "unknown"]

_CASH_SCALE = 10_000.0
_INVENTORY_SCALE = 10_000.0
_PRODUCTION_SCALE = 10.0
_CASH_WEIGHT = 0.30
_INVENTORY_WEIGHT = 0.25
_PRODUCTION_WEIGHT = 0.20
_UTILIZATION_WEIGHT = 0.15
_DEADLINE_RISK_WEIGHT = 0.10
_PROGRESS_TOLERANCE = 1e-9
_SCALAR_SIGNAL_FIELDS = (
    "cash",
    "money",
    "production",
    "production_capacity",
    "yield_units",
    "worker_utilization",
    "worker_utilisation",
    "utilization",
    "utilisation",
    "deadline_risk",
    "needs_risk",
    "deadline_pressure",
    "risk",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unit_interval(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    # Explicit ratios are the common representation.  Percent-style values
    # are accepted for convenience when an observation contains 0..100.
    if number > 1.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


def _selected_farm(observation: Observation) -> Mapping[str, Any]:
    farms = observation.get("farms")
    player = observation.get("player", 0)
    if isinstance(player, bool) or not isinstance(player, int):
        player = 0
    if isinstance(farms, Sequence) and not isinstance(farms, (str, bytes)):
        if 0 <= player < len(farms):
            return _mapping(farms[player])
    return _mapping(observation.get("farm"))


def _sources(observation: Observation) -> tuple[Mapping[str, Any], ...]:
    farm = _selected_farm(observation)
    private = _mapping(observation.get("private"))
    metrics = _mapping(observation.get("metrics"))
    return observation, farm, private, metrics


def _first_number(sources: tuple[Mapping[str, Any], ...], names: tuple[str, ...]) -> float | None:
    for source in sources:
        for name in names:
            number = _number(source.get(name))
            if number is not None:
                return number
    return None


def _inventory_value(observation: Observation) -> float | None:
    sources = _sources(observation)
    inventory: Mapping[str, Any] | None = None
    for source in sources:
        candidate = source.get("inventory")
        if isinstance(candidate, Mapping):
            inventory = candidate
            break
    if inventory is None:
        private = _mapping(observation.get("private"))
        candidate = private.get("shed", observation.get("shed"))
        if isinstance(candidate, Mapping):
            inventory = candidate
    if inventory is None:
        return None

    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    if not prices:
        prices = _mapping(observation.get("prices"))
    total = 0.0
    found_quantity = False
    for item, raw_quantity in inventory.items():
        quantity = _number(raw_quantity)
        if quantity is None:
            continue
        found_quantity = True
        quantity = max(0.0, quantity)
        price = _number(prices.get(item))
        # A missing quote still contains useful inventory information; one
        # unit is the neutral liquidation-value fallback.
        price = 1.0 if price is None else max(0.0, price)
        total += quantity * price
    return total if found_quantity and math.isfinite(total) else None


def _has_malformed_signal(observation: object) -> bool:
    """Identify malformed values that should disable, rather than shape, a transition."""
    if not isinstance(observation, Mapping):
        return True
    sources = _sources(observation)
    for source in sources:
        for field in _SCALAR_SIGNAL_FIELDS:
            value = source.get(field)
            if field in source and value is not None and _number(value) is None:
                return True

    inventory_sources: list[Mapping[str, Any]] = []
    for source in sources:
        candidate = source.get("inventory")
        if isinstance(candidate, Mapping):
            inventory_sources.append(candidate)
    private = _mapping(observation.get("private"))
    candidate = private.get("shed", observation.get("shed"))
    if isinstance(candidate, Mapping):
        inventory_sources.append(candidate)
    for inventory in inventory_sources:
        if any(value is not None and _number(value) is None for value in inventory.values()):
            return True

    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    if not prices:
        prices = _mapping(observation.get("prices"))
    if any(value is not None and _number(value) is None for value in prices.values()):
        return True
    return False


def _worker_utilization(observation: Observation) -> float | None:
    sources = _sources(observation)
    explicit = _first_number(
        sources,
        ("worker_utilization", "worker_utilisation", "utilization", "utilisation"),
    )
    if explicit is not None:
        return min(1.0, max(0.0, explicit if explicit <= 1.0 else explicit / 100.0))

    workers: Any = None
    for source in sources:
        if "workers" in source:
            workers = source.get("workers")
            break
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)) or not workers:
        return None
    ratios: list[float] = []
    for worker in workers:
        worker_map = _mapping(worker)
        ratio = _unit_interval(worker_map.get("utilization", worker_map.get("utilised")))
        if ratio is not None:
            ratios.append(ratio)
            continue
        active = worker_map.get("active", worker_map.get("busy", worker_map.get("working")))
        if isinstance(active, bool):
            ratios.append(float(active))
    return sum(ratios) / len(ratios) if ratios else None


def _potential_components(observation: object) -> tuple[float | None, ...] | None:
    if not isinstance(observation, Mapping):
        return None
    sources = _sources(observation)
    cash = _first_number(sources, ("cash", "money"))
    production = _first_number(sources, ("production", "production_capacity", "yield_units"))
    utilization = _worker_utilization(observation)
    deadline_risk = _first_number(
        sources, ("deadline_risk", "needs_risk", "deadline_pressure", "risk"),
    )
    inventory = _inventory_value(observation)
    return (
        None if cash is None else min(1.0, max(0.0, cash / _CASH_SCALE)),
        None if inventory is None else min(1.0, max(0.0, inventory / _INVENTORY_SCALE)),
        None if production is None else min(1.0, max(0.0, production / _PRODUCTION_SCALE)),
        utilization,
        _unit_interval(deadline_risk),
    )


def economic_potential(observation: object) -> float:
    """Return a finite potential in ``[-1, 1]`` from player-visible state.

    Components are normalized independently and omitted/malformed components
    contribute zero.  Deadline risk is a penalty, so larger risk lowers the
    potential.  The fixed weights make the result comparable across states:
    cash .30, inventory .25, production .20, worker utilization .15, and
    deadline risk -.10.
    """
    components = _potential_components(observation)
    if components is None:
        return 0.0
    cash, inventory, production, utilization, deadline_risk = components
    potential = (
        _CASH_WEIGHT * (cash or 0.0)
        + _INVENTORY_WEIGHT * (inventory or 0.0)
        + _PRODUCTION_WEIGHT * (production or 0.0)
        + _UTILIZATION_WEIGHT * (utilization or 0.0)
        - _DEADLINE_RISK_WEIGHT * (deadline_risk or 0.0)
    )
    return min(1.0, max(-1.0, potential)) if math.isfinite(potential) else 0.0


def potential_difference(
    current: object,
    next_state: object,
    gamma: float,
) -> float:
    """Return the bounded potential shaping term ``gamma * Phi(next) - Phi``."""
    discount = _number(gamma)
    if discount is None or _has_malformed_signal(current) or _has_malformed_signal(next_state):
        return 0.0
    discount = min(1.0, max(0.0, discount))
    result = discount * economic_potential(next_state) - economic_potential(current)
    return result if math.isfinite(result) else 0.0


def _transition_value(transition: Any, name: str) -> Any:
    if isinstance(transition, Mapping):
        return transition.get(name)
    return getattr(transition, name, None)


def shaped_transition_reward(transition: object, gamma: float, coefficient: float) -> float:
    """Add bounded shaping to a transition's existing reward.

    ``reward`` remains the terminal bank-margin signal when the transition is
    terminal; shaping never replaces or suppresses that separate signal.
    """
    base_reward = _number(_transition_value(transition, "reward"))
    if base_reward is None:
        base_reward = _number(_transition_value(transition, "terminal_reward"))
    if base_reward is None:
        base_reward = 0.0

    current = _transition_value(transition, "observation")
    next_state = _transition_value(transition, "next_observation")
    if (
        not isinstance(current, Mapping)
        or not isinstance(next_state, Mapping)
        or _has_malformed_signal(current)
        or _has_malformed_signal(next_state)
    ):
        return base_reward
    scale = _number(coefficient)
    if scale is None:
        return base_reward if math.isfinite(base_reward) else 0.0
    result = base_reward + scale * potential_difference(current, next_state, gamma)
    return result if math.isfinite(result) else base_reward


def classify_progress(
    current: object,
    next_state: object,
    *,
    tolerance: float = _PROGRESS_TOLERANCE,
) -> ProgressStatus:
    """Classify an economic-potential transition deterministically."""
    current_components = _potential_components(current)
    next_components = _potential_components(next_state)
    if (
        current_components is None
        or next_components is None
        or _has_malformed_signal(current)
        or _has_malformed_signal(next_state)
    ):
        return "unknown"
    if not any(component is not None for component in current_components + next_components):
        return "unknown"
    threshold = _number(tolerance)
    if threshold is None:
        threshold = _PROGRESS_TOLERANCE
    delta = economic_potential(next_state) - economic_potential(current)
    return "progress" if delta > max(0.0, threshold) else "no_progress"


def should_bootstrap_truncate(no_progress_steps: int, no_progress_window: int) -> bool:
    """Return whether a resolved stall should bootstrap instead of terminally end."""
    if (
        isinstance(no_progress_steps, bool)
        or not isinstance(no_progress_steps, int)
        or isinstance(no_progress_window, bool)
        or not isinstance(no_progress_window, int)
        or no_progress_steps < 0
        or no_progress_window <= 0
    ):
        return False
    return no_progress_steps >= no_progress_window


__all__ = [
    "ProgressStatus",
    "classify_progress",
    "economic_potential",
    "potential_difference",
    "shaped_transition_reward",
    "should_bootstrap_truncate",
]
