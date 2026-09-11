"""Run reproducible Kaggriculture policy evaluations and summarize replays.

The evaluator deliberately imports the Kaggle engine only when a game is run,
so parsing and replay aggregation remain usable in offline test environments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from numbers import Real
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    ENGINE_VERSION,
    LAND_ORDER,
    LAND_PRICES,
    PRICE_FLOOR,
    PRODUCTS,
    SHOPS,
    max_market_orders,
    season_days,
    shed_capacity,
)
from kagriculture_agent.candidates import CANDIDATES, candidate_metadata, candidate_policy  # noqa: E402
from kagriculture_agent.economics import market_price  # noqa: E402
from kagriculture_agent.observation import is_shed_adjacent  # noqa: E402
from kagriculture_agent.planner import _has_basic_need_deadline  # noqa: E402
from kagriculture_agent.policy import Policy  # noqa: E402
from scripts.run_local import OPPONENTS, _deterministic_random_agent  # noqa: E402
from scripts.output_paths import atomic_write_text  # noqa: E402
from scripts.evaluation_metrics import (  # noqa: E402
    bradley_terry_summary as _league_elo_summary,
    summarize_by_opponent as _summarize_by_opponent,
)
from scripts.training_identity import (  # noqa: E402
    ACTION_REPRESENTATIONS,
    DEFAULT_ACTION_REPRESENTATION,
    DEFAULT_EXPERIMENT_ID,
    FEATURE_VARIANTS,
    TRAINING_MODES,
    validate_training_identity,
    validate_action_representation,
)


VARIANTS = ("conservative", "mixed", "melon-heavy", "demand-reactive", "animal-heavy")
EVALUATION_NAMES = tuple(dict.fromkeys((*CANDIDATES, *VARIANTS)))
ABLATION_COMPONENTS = (
    "route_scheduling",
    "market_batch_sizing",
    "shop_adaptation",
    "land_purchase",
    "animals",
)
_DEFAULT_ABLATIONS = {component: True for component in ABLATION_COMPONENTS}
_BUYABLE_PRODUCTS = frozenset({"WHEAT", "FERTILIZER"})
_PRODUCT_NAMES = frozenset(PRODUCTS)
_SALEABLE_PRODUCTS = _PRODUCT_NAMES - {"FERTILIZER"}
_ANIMAL_NAMES = frozenset(ANIMALS)
_ITEM_NAMES = _PRODUCT_NAMES | _ANIMAL_NAMES
_WORKER_TIMEOUT_SECONDS = 120
_NORMALIZED_RECORD_FIELDS = frozenset({
    "variant", "opponent", "seed", "seat", "outcome", "final_bank",
    "opponent_final_bank", "bank_differential", "framework_error",
    "shed_overflow", "price_floor_sales", "missed_basic_needs",
    "submitted_market_order_count", "market_transaction_count", "same_item_market_churn",
    "same_item_sell_buy_churn", "terminal_cash", "terminal_inventory_value",
})
_NORMALIZED_NUMERIC_FIELDS = frozenset({
    "final_bank", "opponent_final_bank", "bank_differential", "shed_overflow",
    "price_floor_sales", "missed_basic_needs", "terminal_cash",
    "terminal_inventory_value", "market_transaction_count",
    "submitted_market_order_count", "same_item_market_churn",
    "same_item_sell_buy_churn",
})
_NORMALIZED_NONNEGATIVE_INTEGER_FIELDS = frozenset({
    "market_transaction_count", "submitted_market_order_count",
    "same_item_market_churn", "same_item_sell_buy_churn",
})
_WORKER_OUTCOMES = frozenset({"win", "loss", "tie", "framework_error"})
FRAMEWORK_REASON_ORDER = (
    "engine_error", "malformed_replay", "missed_basic_needs", "market_churn",
    "market_transaction_cap", "terminal_cash_floor", "terminal_inventory_floor",
    "incomplete_pairing",
)
_STRICT_NUMERIC_FIELDS = frozenset({
    "seed", "step", "day", "hour", "money", "hires_today", "yield_units",
    "max_lifespan_step", "consecutive_unwatered", "planted_day",
    "fertilized_until_day", "placed_day", "consecutive_unfed",
    "pending_care_bonus", "quantity", "price", "amount", "episodeSteps",
    "actTimeout", "runTimeout", "boardSize", "startingMoney",
    "maxMarketOrdersPerTurn", "turnsPerDay", "shedCapacity", "weedSpawnChance",
    "townShopUnlockInterval", "townShopSellInterval", "townCenterSellInterval",
    "farmHandCostMult",
})
_STRICT_QUANTITY_MAPPING_FIELDS = frozenset({"inventory", "prices", "shed", "seeds"})
_BASELINE_CONVENTION = "the first configured candidate is the baseline for promotion decisions"


def _manifest_candidate_metadata(candidate: str, *, selection_path: str = "candidate") -> dict[str, Any]:
    """Return model metadata without rejecting report-only historical names."""
    if selection_path == "legacy_variant":
        return {
            "model_identity": f"legacy_variant:{candidate}",
            "artifact_sha256": None,
            "feature_schema_version": None,
            "engine_version": str(ENGINE_VERSION),
        }
    if selection_path != "candidate":
        raise ValueError(f"unsupported manifest selection path: {selection_path}")
    try:
        return candidate_metadata(candidate)
    except ValueError:
        return {
            "model_identity": f"unregistered:{candidate}",
            "artifact_sha256": None,
            "feature_schema_version": None,
            "engine_version": str(ENGINE_VERSION),
        }


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def framework_error_reasons(record: Mapping[str, Any]) -> list[str]:
    """Return the supported framework diagnostics in their canonical order."""
    if not isinstance(record, Mapping):
        return []
    reasons = {
        reason for reason in record.get("framework_error_reasons", ())
        if reason in FRAMEWORK_REASON_ORDER
    } if isinstance(record.get("framework_error_reasons", ()), Sequence) \
        and not isinstance(record.get("framework_error_reasons"), (str, bytes)) else set()
    if record.get("engine_error"):
        reasons.add("engine_error")
    if record.get("malformed_replay"):
        reasons.add("malformed_replay")
    if _number(record.get("missed_basic_needs")) is not None and _number(record.get("missed_basic_needs")) > 0:
        reasons.add("missed_basic_needs")
    churn = record.get("same_item_market_churn", record.get("same_item_sell_buy_churn", 0))
    if _number(churn) is not None and _number(churn) > 0:
        reasons.add("market_churn")
    for field, reason in (
        ("market_transaction_cap", "market_transaction_cap"),
        ("terminal_cash_floor", "terminal_cash_floor"),
        ("terminal_inventory_floor", "terminal_inventory_floor"),
        ("incomplete_pairing", "incomplete_pairing"),
    ):
        if record.get(field):
            reasons.add(reason)
    return [reason for reason in FRAMEWORK_REASON_ORDER if reason in reasons]


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return number


def _validate_economic_thresholds(*, max_same_item_market_churn: Any = 0,
                                  max_market_transactions: Any = None,
                                  min_terminal_cash: Any = 0.0,
                                  min_terminal_inventory_value: Any = 0.0) -> dict[str, Any]:
    """Validate and normalize the ordered economic safety-gate thresholds."""
    if type(max_same_item_market_churn) is not int or max_same_item_market_churn < 0:
        raise ValueError("max_same_item_market_churn must be a non-negative integer")
    if max_market_transactions is not None and (
        type(max_market_transactions) is not int or max_market_transactions < 0
    ):
        raise ValueError("max_market_transactions must be a non-negative integer")
    for name, value in (("min_terminal_cash", min_terminal_cash),
                        ("min_terminal_inventory_value", min_terminal_inventory_value)):
        if isinstance(value, bool) or not isinstance(value, Real) \
                or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
    return {
        "max_same_item_market_churn": max_same_item_market_churn,
        "max_market_transactions": max_market_transactions,
        "min_terminal_cash": float(min_terminal_cash),
        "min_terminal_inventory_value": float(min_terminal_inventory_value),
    }


def _validate_seed_partition(development_seeds: Sequence[int], holdout_seeds: Sequence[int] | None) -> None:
    """Reject repeated or overlapping seeds between the two evaluation phases."""
    development = list(development_seeds)
    if len(set(development)) != len(development):
        raise ValueError("development seeds must be unique")
    if holdout_seeds is None:
        return
    holdout = list(holdout_seeds)
    if len(set(holdout)) != len(holdout):
        raise ValueError("holdout seeds must be unique")
    overlap = sorted(set(development).intersection(holdout))
    if overlap:
        raise ValueError(
            "development and holdout seeds must be disjoint; overlapping seed(s): "
            + ", ".join(map(str, overlap))
        )


def _variant_list(values: Sequence[str] | None) -> list[str]:
    selected = list(values or ("mixed",))
    unknown = [value for value in selected if value not in VARIANTS]
    if unknown:
        raise ValueError(f"unsupported variant(s): {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def _candidate_list(values: Sequence[str] | None) -> list[str]:
    selected = list(values or ("mixed",))
    unknown = [value for value in selected if value not in CANDIDATES]
    if unknown:
        raise ValueError(f"unsupported candidate(s): {', '.join(unknown)}")
    return list(dict.fromkeys(selected))


def parse_ablation(value: str) -> tuple[str, bool]:
    """Parse ``component=on|off`` for a component already present in policy."""
    try:
        component, state = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use component=on or component=off") from exc
    if component not in ABLATION_COMPONENTS or state not in {"on", "off"}:
        choices = ", ".join(ABLATION_COMPONENTS)
        raise argparse.ArgumentTypeError(f"component must be one of {choices}; state must be on/off")
    return component, state == "on"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_positive_int, default=30, help="number of consecutive seeds")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=["pass", "random", "starter"])
    parser.add_argument("--steps", type=_positive_int, default=720)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="production_v1")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="behavior_clone_then_ppo")
    parser.add_argument(
        "--action-representation", choices=ACTION_REPRESENTATIONS,
        default=DEFAULT_ACTION_REPRESENTATION,
    )
    parser.add_argument("--seats", nargs="+", type=int, choices=(0, 1), default=[0, 1],
                        help="candidate seats to evaluate (0 and 1 are supported)")
    parser.add_argument("--variant", action="append", dest="single_variants", choices=VARIANTS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument(
        "--candidates", nargs="+", choices=CANDIDATES, default=None,
        help="stable route candidates; use --variants/--variant for legacy evaluator variants",
    )
    parser.add_argument(
        "--holdout-seeds", nargs="+", type=int, default=None,
        help="explicit disjoint seeds for the final promotion holdout",
    )
    parser.add_argument(
        "--min-valid-games", type=_positive_int, default=20,
        help="minimum valid games required for each seat before selection",
    )
    parser.add_argument(
        "--baseline-policy", default=None, metavar="MODULE:CALLABLE",
        help="previous-agent policy reference; run it on the identical matrix",
    )
    parser.add_argument(
        "--baseline-identity", default="previous-agent",
        help="stable report identity for --baseline-policy",
    )
    parser.add_argument(
        "--churn-window", type=_positive_int, default=2,
        help="maximum replay-turn distance for same-item buy/sell churn",
    )
    parser.add_argument(
        "--max-same-item-churn", type=_nonnegative_int, default=0,
        help="maximum same-item buy/sell churn events per game",
    )
    parser.add_argument(
        "--max-market-transactions", type=_nonnegative_int, default=None,
        help="maximum submitted market order events across the evaluation (quantity does not multiply the count)",
    )
    parser.add_argument(
        "--min-terminal-cash", type=_nonnegative_float, default=0.0,
        help="minimum terminal cash required in every valid game",
    )
    parser.add_argument(
        "--min-terminal-inventory-value", type=_nonnegative_float, default=0.0,
        help="minimum terminal inventory value required in every valid game",
    )
    parser.add_argument("--ablation", action="append", type=parse_ablation, default=[], metavar="COMPONENT=on|off")
    parser.add_argument("--quick", action="store_true", help="use a small default batch suitable for local tests")
    args = parser.parse_args(argv)
    if type(args.experiment_id) is not str or not args.experiment_id.strip():
        parser.error("experiment_id must be a non-empty string")
    if args.variants is not None and args.candidates is not None:
        parser.error("--variants and --candidates are separate modes; supply only one")
    if args.candidates is not None:
        if args.single_variants:
            parser.error("--variant cannot be combined with stable --candidates")
        args.candidates = _candidate_list(args.candidates)
        args.variants = None
    else:
        args.variants = _variant_list((args.variants or []) + (args.single_variants or []))
        args.candidates = None
    if args.quick:
        if args.seeds == 30:
            args.seeds = 2
        if args.steps == 720:
            args.steps = 96
    development_seeds = range(args.start_seed, args.start_seed + args.seeds)
    try:
        _validate_seed_partition(development_seeds, args.holdout_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.holdout_seeds is not None and set(args.seats) != {0, 1}:
        parser.error("--holdout-seeds requires both candidate seats: --seats 0 1")
    return args


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Return an inclusive, linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (float(percent) / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _average(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0) or 0.0) for record in records]
    return float(mean(values)) if values else 0.0


def _diagnostic_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate replay termination and safety metadata for promotion telemetry."""
    records = list(records)
    termination_reasons: dict[str, int] = {}
    bootstrap_truncated_count = 0
    shaping_count = 0
    max_no_progress_steps = 0
    time_limit_endings = 0
    safety_regression_count = 0
    for record in records:
        reason = record.get("termination_reason")
        if reason is not None and str(reason):
            reason = str(reason)
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        if record.get("bootstrap_truncated") is True:
            bootstrap_truncated_count += 1
        shaping = _number(record.get("shaping_count"))
        if shaping is not None:
            shaping_count += max(0, int(shaping))
        no_progress_steps = _number(record.get("no_progress_steps"))
        if no_progress_steps is not None:
            max_no_progress_steps = max(max_no_progress_steps, max(0, int(no_progress_steps)))
        normalized_reason = str(reason or "").strip().lower().replace("-", "_")
        if record.get("time_limit_ending") is True or normalized_reason in {
            "time_limit", "timeout", "time_limit_ending",
        }:
            time_limit_endings += 1
        safety_flags = record.get("safety_flags", ())
        has_safety_flag = isinstance(safety_flags, (list, tuple, set)) and any(
            "safety_regression" in str(flag).lower() for flag in safety_flags
        )
        if record.get("safety_regression") is True or has_safety_flag:
            safety_regression_count += 1
    resolved_count = sum(
        count for reason, count in termination_reasons.items()
        if reason.strip().lower().replace("-", "_") == "resolved"
    )
    no_progress_count = sum(
        count for reason, count in termination_reasons.items()
        if reason.strip().lower().replace("-", "_") == "no_progress"
    )
    denominator = float(len(records)) if records else 1.0
    return {
        "record_count": len(records),
        "termination_reasons": termination_reasons,
        "bootstrap_truncated_count": bootstrap_truncated_count,
        "truncation_count": bootstrap_truncated_count,
        "truncation_rate": bootstrap_truncated_count / denominator,
        "shaping_count": shaping_count,
        "resolved_count": resolved_count,
        "resolved_rate": resolved_count / denominator,
        "no_progress_count": no_progress_count,
        "no_progress_rate": no_progress_count / denominator,
        "max_no_progress_steps": max_no_progress_steps,
        "max_no_progress_streak": max_no_progress_steps,
        "time_limit_endings": time_limit_endings,
        "time_limit_rate": time_limit_endings / denominator,
        "safety_regression_count": safety_regression_count,
        "safety_regression_rate": safety_regression_count / denominator,
    }


_DIAGNOSTIC_GATE_METRICS = (
    "truncation_count", "truncation_rate", "resolved_count", "resolved_rate",
    "no_progress_count", "no_progress_rate", "max_no_progress_streak",
    "time_limit_endings", "time_limit_rate", "safety_regression_count",
    "safety_regression_rate",
)


def _diagnostic_regression(
    candidate_summary: Mapping[str, Any], baseline_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare canonical candidate diagnostics for promotion safety."""
    candidate_diagnostics = candidate_summary.get("diagnostics", candidate_summary)
    baseline_diagnostics = baseline_summary.get("diagnostics", baseline_summary)
    if not isinstance(candidate_diagnostics, Mapping):
        candidate_diagnostics = {}
    if not isinstance(baseline_diagnostics, Mapping):
        baseline_diagnostics = {}

    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for metric in _DIAGNOSTIC_GATE_METRICS:
        candidate_value = _number(candidate_diagnostics.get(metric)) or 0.0
        baseline_value = _number(baseline_diagnostics.get(metric)) or 0.0
        delta = float(candidate_value - baseline_value)
        deltas[metric] = delta
        if delta > 0:
            regressions.append(metric)
    return {
        "candidate": dict(candidate_diagnostics),
        "baseline": dict(baseline_diagnostics),
        "deltas": deltas,
        "regressions": regressions,
        "regressed": bool(regressions),
    }


def _market_metrics(states: Sequence[Mapping[str, Any]], *, churn_window: int = 2) -> dict[str, int]:
    """Count submitted market order events and reversals in a replay.

    The bootstrap state at index zero contains a placeholder action.  Churn is
    counted when consecutive opposite ``BUY_PRODUCT``/``SELL`` directions for
    one item occur within ``churn_window`` replay turns.  Seed purchases are
    intentionally excluded because seeds and harvested products are distinct
    inventories in the engine.
    """
    if type(churn_window) is not int or churn_window < 1:
        raise ValueError("churn_window must be a positive integer")
    transaction_count = 0
    churn_count = 0
    last_direction: dict[str, tuple[str, int]] = {}
    for index, state in enumerate(states):
        if index == 0 and len(states) > 1:
            continue
        observation = _mapping(state.get("observation"))
        raw_step = _number(observation.get("step"))
        step = int(raw_step) if raw_step is not None else index
        action = _mapping(state.get("action"))
        for order in _market_orders(action):
            if (
                not isinstance(order, Sequence)
                or isinstance(order, (str, bytes))
                or not order
                or not isinstance(order[0], str)
            ):
                continue
            # A quantity-bearing order is one submitted market order event.
            # The quantity remains part of the replay action, but does not
            # inflate the activity count; replay data cannot prove a fill.
            transaction_count += 1
            operation = order[0]
            if operation not in {"BUY_PRODUCT", "SELL"} or len(order) < 2 or not isinstance(order[1], str):
                continue
            item = order[1]
            direction = "buy" if operation == "BUY_PRODUCT" else "sell"
            previous = last_direction.get(item)
            if previous and previous[0] != direction and step - previous[1] <= churn_window:
                churn_count += 1
            last_direction[item] = (direction, step)
    return {
        # This is the canonical name.  The older market_transaction_count key
        # remains as an alias for report/fixture compatibility.
        "submitted_market_order_count": transaction_count,
        "market_transaction_count": transaction_count,
        "same_item_market_churn": churn_count,
        "same_item_sell_buy_churn": churn_count,
    }


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate game records into the stable per-variant/opponent schema."""
    records = list(records)
    count = len(records)
    valid_records = [record for record in records if not record.get("framework_error")]
    final_banks = [float(record["final_bank"]) for record in valid_records if record.get("final_bank") is not None]
    terminal_cash = [
        float(record.get("terminal_cash", record.get("final_bank")))
        for record in valid_records
        if record.get("terminal_cash", record.get("final_bank")) is not None
    ]
    terminal_inventory_values = [
        float(record.get("terminal_inventory_value", 0.0) or 0.0)
        for record in valid_records
    ]
    market_transactions = [int(record.get("submitted_market_order_count", record.get("market_transaction_count", 0)) or 0)
                           for record in records]
    churn_values = [int(record.get("same_item_market_churn", record.get("same_item_sell_buy_churn", 0)) or 0) for record in records]
    outcomes = {outcome: sum(record.get("outcome") == outcome for record in valid_records)
                for outcome in ("win", "loss", "tie")}
    return {
        "count": count,
        "valid_count": len(valid_records),
        "framework_failures": sum(bool(record.get("framework_error")) for record in records),
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "ties": outcomes["tie"],
        "win_rate": outcomes["win"] / len(valid_records) if valid_records else 0.0,
        "mean_final_bank": float(mean(final_banks)) if final_banks else 0.0,
        "median_final_bank": float(median(final_banks)) if final_banks else 0.0,
        "fifth_percentile_final_bank": float(percentile(final_banks, 5)) if final_banks else 0.0,
        "mean_terminal_cash": float(mean(terminal_cash)) if terminal_cash else 0.0,
        "median_terminal_cash": float(median(terminal_cash)) if terminal_cash else 0.0,
        "mean_terminal_inventory_value": float(mean(terminal_inventory_values)) if terminal_inventory_values else 0.0,
        "median_terminal_inventory_value": float(median(terminal_inventory_values)) if terminal_inventory_values else 0.0,
        "total_market_transaction_count": sum(market_transactions),
        "mean_market_transaction_count": float(mean(market_transactions)) if market_transactions else 0.0,
        "total_submitted_market_order_count": sum(market_transactions),
        "mean_submitted_market_order_count": float(mean(market_transactions)) if market_transactions else 0.0,
        "total_same_item_market_churn": sum(churn_values),
        "max_same_item_market_churn": max(churn_values, default=0),
        "mean_bank_differential": _average(valid_records, "bank_differential"),
        "framework_error_rate": sum(bool(record.get("framework_error")) for record in records) / count if count else 0.0,
        "average_shed_overflow": _average(records, "shed_overflow"),
        "average_price_floor_sales": _average(records, "price_floor_sales"),
        "average_missed_basic_needs_events": _average(records, "missed_basic_needs"),
        **_diagnostic_summary(records),
    }


_BOOTSTRAP_SAMPLES = 2000
_WILSON_Z = 1.959963984540054


def _record_candidate(record: Mapping[str, Any]) -> str:
    """Return the stable candidate identity used by evaluation metrics."""
    value = record.get("candidate", record.get("variant"))
    return str(value)


def _metric_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None else None


def _metric_record_is_valid(record: Mapping[str, Any]) -> bool:
    return (
        type(record.get("seat")) is int
        and record["seat"] in (0, 1)
        and record.get("framework_error") is False
        and record.get("outcome") in {"win", "loss", "tie"}
        and _metric_number(record.get("bank_differential")) is not None
    )


def _record_has_missed_basic_needs(record: Mapping[str, Any]) -> bool:
    value = record.get("missed_basic_needs")
    if isinstance(value, bool):
        return value
    number = _metric_number(value)
    return number is not None and number > 0


def _wilson_interval(successes: float, trials: int) -> dict[str, float | None]:
    if trials < 1:
        return {"lower": None, "upper": None}
    proportion = successes / trials
    z_squared = _WILSON_Z ** 2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = _WILSON_Z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials)
    ) / denominator
    return {
        "lower": float(max(0.0, center - margin)),
        "upper": float(min(1.0, center + margin)),
    }


def _bootstrap_interval(values: Sequence[float], keys: Sequence[tuple[Any, ...]], *,
                       lower_percent: float = 2.5, upper_percent: float = 97.5) -> dict[str, float | None]:
    if not values:
        return {"lower": None, "upper": None}
    if not 0.0 <= lower_percent <= upper_percent <= 100.0:
        raise ValueError("bootstrap interval bounds must be ordered percentages")
    seed_material = json.dumps(
        [[*key, float(value)] for key, value in zip(keys, values)],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
    samples = []
    for _ in range(_BOOTSTRAP_SAMPLES):
        samples.append(mean(values[rng.randrange(len(values))] for _ in values))
    return {
        "lower": float(percentile(samples, lower_percent)),
        "upper": float(percentile(samples, upper_percent)),
    }


def paired_seed_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize seed-matched seat pairs with deterministic confidence metrics.

    A pair is valid only when exactly one valid record exists for each seat for
    the same candidate, opponent, and seed.  Incomplete or duplicate pairs are
    reported rather than silently folded into the aggregate.
    """
    records = list(records)
    buckets: dict[tuple[str, str, Any], dict[int, list[Mapping[str, Any]]]] = {}
    valid_by_seat = {0: 0, 1: 0}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if _metric_record_is_valid(record):
            valid_by_seat[record["seat"]] += 1
        key = (_record_candidate(record), str(record.get("opponent")), record.get("seed"))
        seat = record.get("seat")
        if type(seat) is int and seat in (0, 1):
            buckets.setdefault(key, {0: [], 1: []})[seat].append(record)

    pairs: list[tuple[tuple[str, str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    missing = 0
    duplicate = 0
    for key in sorted(buckets, key=lambda value: tuple(str(item) for item in value)):
        seats = buckets[key]
        if len(seats[0]) == 1 and len(seats[1]) == 1 and all(_metric_record_is_valid(item) for item in (seats[0][0], seats[1][0])):
            pairs.append((key, seats[0][0], seats[1][0]))
        else:
            if len(seats[0]) == 0 or len(seats[1]) == 0 or (len(seats[0]) == 1 and len(seats[1]) == 1):
                missing += 1
            if len(seats[0]) > 1 or len(seats[1]) > 1:
                duplicate += 1

    pair_scores = []
    pair_differentials = []
    pair_market_transactions = []
    pair_market_transaction_totals = []
    pair_churn = []
    pair_terminal_cash = []
    pair_terminal_inventory_value = []
    pair_keys = []
    outcome_counts = {"win": 0, "loss": 0, "tie": 0}
    for key, seat_zero, seat_one in pairs:
        scores = {"win": 1.0, "tie": 0.5, "loss": 0.0}
        pair_scores.append((scores[seat_zero["outcome"]] + scores[seat_one["outcome"]]) / 2.0)
        pair_differentials.append(
            (_metric_number(seat_zero["bank_differential"]) + _metric_number(seat_one["bank_differential"])) / 2.0
        )
        seat_market_transactions = [
            float(seat_zero.get("submitted_market_order_count", seat_zero.get("market_transaction_count", 0)) or 0),
            float(seat_one.get("submitted_market_order_count", seat_one.get("market_transaction_count", 0)) or 0),
        ]
        pair_market_transactions.append(float(mean(seat_market_transactions)))
        pair_market_transaction_totals.append(sum(seat_market_transactions))
        pair_churn.append(float(mean(
            [float(seat_zero.get("same_item_market_churn", seat_zero.get("same_item_sell_buy_churn", 0)) or 0),
             float(seat_one.get("same_item_market_churn", seat_one.get("same_item_sell_buy_churn", 0)) or 0)]
        )))
        pair_terminal_cash.append(float(mean([
            float(seat_zero.get("terminal_cash", seat_zero.get("final_bank", 0)) or 0),
            float(seat_one.get("terminal_cash", seat_one.get("final_bank", 0)) or 0),
        ])))
        pair_terminal_inventory_value.append(float(mean([
            float(seat_zero.get("terminal_inventory_value", 0) or 0),
            float(seat_one.get("terminal_inventory_value", 0) or 0),
        ])))
        pair_keys.append(key)
        for seat_record in (seat_zero, seat_one):
            outcome_counts[seat_record["outcome"]] += 1

    # The two seat games for a seed are paired observations, not independent
    # Bernoulli trials.  Wilson is valid only when every paired score is a
    # genuine Bernoulli outcome; split wins/losses and ties use the paired
    # bootstrap instead.
    binary_scores = [score for score in pair_scores if score in (0.0, 1.0)]
    wilson = (
        _wilson_interval(sum(binary_scores), len(binary_scores))
        if len(binary_scores) == len(pair_scores) else None
    )
    bootstrap_win_rate = _bootstrap_interval(pair_scores, pair_keys)
    bootstrap = _bootstrap_interval(pair_differentials, pair_keys)
    elo_matches = [
        {
            "player_a": _record_candidate(record),
            "player_b": str(record.get("opponent")),
            "outcome": record["outcome"],
        }
        for record in records
        if _metric_record_is_valid(record)
        and _record_candidate(record) != str(record.get("opponent"))
    ]
    elo = _league_elo_summary(elo_matches)
    diagnostics = _diagnostic_summary(records)
    return {
        "record_count": len(records),
        "valid_records_by_seat": {"0": valid_by_seat[0], "1": valid_by_seat[1]},
        "paired_games": len(pairs),
        "missing_seat_pairs": missing,
        "duplicate_seat_pairs": duplicate,
        "wins": outcome_counts["win"],
        "losses": outcome_counts["loss"],
        "ties": outcome_counts["tie"],
        "seat_balanced_win_rate": float(mean(pair_scores)) if pair_scores else None,
        "mean_paired_bank_differential": float(mean(pair_differentials)) if pair_differentials else None,
        "median_paired_bank_differential": float(median(pair_differentials)) if pair_differentials else None,
        "fifth_percentile_bank_differential": percentile(pair_differentials, 5),
        "total_market_transaction_count": int(sum(pair_market_transaction_totals)),
        "mean_paired_market_transaction_count": float(mean(pair_market_transactions)) if pair_market_transactions else None,
        "median_paired_market_transaction_count": float(median(pair_market_transactions)) if pair_market_transactions else None,
        "total_submitted_market_order_count": int(sum(pair_market_transaction_totals)),
        "mean_paired_submitted_market_order_count": float(mean(pair_market_transactions)) if pair_market_transactions else None,
        "median_paired_submitted_market_order_count": float(median(pair_market_transactions)) if pair_market_transactions else None,
        "max_same_item_market_churn": max(
            [
                int(record.get("same_item_market_churn", record.get("same_item_sell_buy_churn", 0)) or 0)
                for _key, seat_zero, seat_one in pairs
                for record in (seat_zero, seat_one)
            ],
            default=0,
        ),
        "mean_paired_same_item_market_churn": float(mean(pair_churn)) if pair_churn else None,
        "median_paired_same_item_market_churn": float(median(pair_churn)) if pair_churn else None,
        "mean_paired_terminal_cash": float(mean(pair_terminal_cash)) if pair_terminal_cash else None,
        "median_paired_terminal_cash": float(median(pair_terminal_cash)) if pair_terminal_cash else None,
        "mean_paired_terminal_inventory_value": float(mean(pair_terminal_inventory_value)) if pair_terminal_inventory_value else None,
        "median_paired_terminal_inventory_value": float(median(pair_terminal_inventory_value)) if pair_terminal_inventory_value else None,
        "wilson_win_rate": wilson,
        "lower_tail_bank_differential": percentile(pair_differentials, 5),
        "elo": elo,
        "confidence": {
            "wilson_win_rate": wilson,
            "bootstrap_seat_balanced_win_rate": bootstrap_win_rate,
            "bootstrap_bank_differential": bootstrap,
        },
        "bootstrap_seat_balanced_win_rate": bootstrap_win_rate,
        "bootstrap_bank_differential": bootstrap,
        "diagnostics": diagnostics,
    }


def _normalize_expected_matrix(
    expected_matrix: Sequence[tuple[str, int, int]] | None,
) -> list[tuple[str, int, int]] | None:
    if expected_matrix is None:
        return None
    normalized = []
    for coordinate in expected_matrix:
        if not isinstance(coordinate, Sequence) or isinstance(coordinate, (str, bytes)) \
                or len(coordinate) != 3:
            raise ValueError("expected_matrix coordinates must be (opponent, seed, seat) triples")
        opponent, seed, seat = coordinate
        if type(opponent) is not str or type(seed) is not int or type(seat) is not int \
                or seat not in (0, 1):
            raise ValueError("expected_matrix coordinates must contain (str, int, 0|1)")
        normalized.append((opponent, seed, seat))
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected_matrix coordinates must be unique")
    return normalized


def _matrix_completeness(
    records: Sequence[Mapping[str, Any]],
    expected_matrix: Sequence[tuple[str, int, int]] | None,
) -> dict[str, Any] | None:
    if expected_matrix is None:
        return None
    expected = _normalize_expected_matrix(expected_matrix)
    assert expected is not None
    expected_set = set(expected)
    observed = Counter()
    invalid_records = 0
    for record in records:
        if not isinstance(record, Mapping):
            invalid_records += 1
            continue
        opponent = record.get("opponent")
        seed = record.get("seed")
        seat = record.get("seat")
        if type(opponent) is not str or type(seed) is not int or type(seat) is not int \
                or seat not in (0, 1):
            invalid_records += 1
            continue
        observed[(opponent, seed, seat)] += 1
    missing = sorted(expected_set - observed.keys(), key=lambda key: tuple(str(value) for value in key))
    duplicate = sorted(
        (key for key, count in observed.items() if count > 1),
        key=lambda key: tuple(str(value) for value in key),
    )
    extra = sorted(
        (key for key in observed if key not in expected_set),
        key=lambda key: tuple(str(value) for value in key),
    )
    return {
        "expected": [list(coordinate) for coordinate in expected],
        "observed": [list(coordinate) for coordinate in sorted(observed, key=lambda key: tuple(str(value) for value in key))],
        "expected_count": len(expected),
        "observed_count": sum(observed.values()) + invalid_records,
        "missing": [list(coordinate) for coordinate in missing],
        "duplicate": [list(coordinate) for coordinate in duplicate],
        "extra": [list(coordinate) for coordinate in extra],
        "invalid_records": invalid_records,
    }


def _safety_gate_reasons(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], *,
                        min_valid_games: int,
                        expected_matrix: Sequence[tuple[str, int, int]] | None = None,
                        max_same_item_market_churn: int = 0,
                        max_market_transactions: int | None = None,
                        min_terminal_cash: float = 0.0,
                        min_terminal_inventory_value: float = 0.0) -> list[str]:
    """Return ordered reasons a candidate is unsafe for promotion or selection.

    Reports written from pre-seat evaluator fixtures may not contain a ``seat``
    field. They retain legacy aggregate-selection behavior; seat-aware
    evaluations must satisfy the complete paired-game safety contract.
    """
    mappings = [record for record in records if isinstance(record, Mapping)]
    if any(record.get("framework_error") for record in mappings):
        return ["framework_error"]
    if any(_record_has_missed_basic_needs(record) for record in mappings):
        return ["missed_basic_needs"]
    matrix_completeness = _matrix_completeness(mappings, expected_matrix)
    if matrix_completeness is not None:
        matrix_reasons = []
        if matrix_completeness["missing"]:
            matrix_reasons.append("missing_expected_matrix_records")
        if matrix_completeness["duplicate"]:
            matrix_reasons.append("duplicate_expected_matrix_records")
        if matrix_completeness["extra"] or matrix_completeness["invalid_records"]:
            matrix_reasons.append("extra_expected_matrix_records")
        if matrix_reasons:
            return matrix_reasons
    seat_aware = any("seat" in record for record in mappings)
    if seat_aware:
        if any(summary["valid_records_by_seat"][str(seat)] < min_valid_games for seat in (0, 1)):
            return ["insufficient_valid_games"]
        if summary["missing_seat_pairs"]:
            return ["missing_seat_pairs"]
        if summary["duplicate_seat_pairs"]:
            return ["duplicate_seat_pairs"]
        if (
            summary["fifth_percentile_bank_differential"] is None
            or summary["fifth_percentile_bank_differential"] < 0
        ):
            return ["negative_tail"]
    economic_summary = summary if seat_aware else aggregate_records(mappings)
    if economic_summary.get("max_same_item_market_churn", 0) > max_same_item_market_churn:
        return ["same_item_market_churn"]
    if max_market_transactions is not None and economic_summary.get("total_market_transaction_count", 0) > max_market_transactions:
        return ["market_transaction_count"]
    valid_mappings = [record for record in mappings if not record.get("framework_error")]

    def _terminal_metric(record: Mapping[str, Any], field: str, summary_key: str,
                         *legacy_fields: str) -> float:
        for candidate_field in (field, *legacy_fields):
            if candidate_field in record and record[candidate_field] is not None:
                return float(record[candidate_field] or 0.0)
        if field == "terminal_inventory_value" and "seat" not in record:
            return 0.0
        # Legacy records may not expose the per-record terminal metric.  Keep
        # the old aggregate fallback only for that record, while enforcing
        # present terminal fields independently.
        return float(economic_summary.get(summary_key, 0.0) or 0.0)

    cash_below_threshold = any(
        _terminal_metric(record, "terminal_cash", "mean_terminal_cash", "final_bank") < min_terminal_cash
        for record in valid_mappings
    )
    inventory_below_threshold = any(
        _terminal_metric(record, "terminal_inventory_value", "mean_terminal_inventory_value")
        < min_terminal_inventory_value
        for record in valid_mappings
    )
    if cash_below_threshold:
        return ["terminal_cash_below_threshold"]
    if inventory_below_threshold:
        return ["terminal_inventory_value_below_threshold"]
    return []


def _passes_safety_gates(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], *,
                         min_valid_games: int,
                         max_same_item_market_churn: int = 0,
                         max_market_transactions: int | None = None,
                         min_terminal_cash: float = 0.0,
                         min_terminal_inventory_value: float = 0.0) -> bool:
    """Return whether a candidate is eligible for default selection."""
    return not _safety_gate_reasons(
        records, summary, min_valid_games=min_valid_games,
        max_same_item_market_churn=max_same_item_market_churn,
        max_market_transactions=max_market_transactions,
        min_terminal_cash=min_terminal_cash,
        min_terminal_inventory_value=min_terminal_inventory_value,
    )


def _external_metric_deltas(
    candidate_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
) -> dict[str, float] | None:
    """Compare external-policy metrics on identical opponent/seed/seat keys."""
    def keyed(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, Any, int], Mapping[str, Any]] | None:
        result = {}
        for record in records:
            if not _metric_record_is_valid(record):
                continue
            key = (str(record.get("opponent")), record.get("seed"), record["seat"])
            if key in result:
                return None
            result[key] = record
        return result

    candidate = keyed(candidate_records)
    baseline = keyed(baseline_records)
    if candidate is None or baseline is None or not candidate or not baseline or set(candidate) != set(baseline):
        return None
    fields = {
        "market_transaction_count": lambda record: float(
            record.get("submitted_market_order_count", record.get("market_transaction_count", 0)) or 0
        ),
        "same_item_market_churn": lambda record: float(record.get("same_item_market_churn", record.get("same_item_sell_buy_churn", 0)) or 0),
        "terminal_cash": lambda record: float(record.get("terminal_cash", record.get("final_bank", 0)) or 0),
        "terminal_inventory_value": lambda record: float(record.get("terminal_inventory_value", 0) or 0),
    }
    return {
        field: float(mean([getter(candidate[key]) - getter(baseline[key]) for key in candidate]))
        for field, getter in fields.items()
    }


def promotion_decision(
    records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    *,
    min_valid_games: int = 20,
    expected_matrix: Sequence[tuple[str, int, int]] | None = None,
    max_same_item_market_churn: int = 0,
    max_market_transactions: int | None = None,
    min_terminal_cash: float = 0.0,
    min_terminal_inventory_value: float = 0.0,
    baseline_policy: str | None = None,
) -> dict[str, Any]:
    """Apply ordered safety gates before comparing a candidate with baseline.

    When ``expected_matrix`` is provided, both record sets must contain exactly
    one record for every requested ``(opponent, seed, seat)`` coordinate.
    Omitting it preserves the legacy caller contract.
    """
    if type(min_valid_games) is not int or min_valid_games < 1:
        raise ValueError("min_valid_games must be a positive integer")
    _validate_economic_thresholds(
        max_same_item_market_churn=max_same_item_market_churn,
        max_market_transactions=max_market_transactions,
        min_terminal_cash=min_terminal_cash,
        min_terminal_inventory_value=min_terminal_inventory_value,
    )
    records = list(records)
    baseline_records = list(baseline_records)
    expected_matrix = _normalize_expected_matrix(expected_matrix)
    candidate = paired_seed_summary(records)
    baseline = paired_seed_summary(baseline_records)
    candidate_matrix = _matrix_completeness(records, expected_matrix)
    baseline_matrix = _matrix_completeness(baseline_records, expected_matrix)
    reasons = _safety_gate_reasons(
        records, candidate, min_valid_games=min_valid_games, expected_matrix=expected_matrix,
        max_same_item_market_churn=max_same_item_market_churn,
        max_market_transactions=max_market_transactions,
        min_terminal_cash=float(min_terminal_cash),
        min_terminal_inventory_value=float(min_terminal_inventory_value),
    )
    baseline_safety_reasons = _safety_gate_reasons(
        baseline_records, baseline, min_valid_games=min_valid_games,
        expected_matrix=expected_matrix,
        max_same_item_market_churn=max_same_item_market_churn,
        max_market_transactions=max_market_transactions,
        min_terminal_cash=float(min_terminal_cash),
        min_terminal_inventory_value=float(min_terminal_inventory_value),
    )
    diagnostic_comparison = _diagnostic_regression(candidate, baseline)
    if diagnostic_comparison["regressed"]:
        reasons.append("diagnostic_regression")
    paired_metric_deltas = (
        _external_metric_deltas(records, baseline_records)
        if baseline_policy is not None and not reasons else None
    )
    if not reasons and baseline_policy is not None and paired_metric_deltas is None:
        reasons.append("baseline_incomplete_pairing")
    baseline_pairing_issues = {
        "insufficient_valid_games", "missing_seat_pairs", "duplicate_seat_pairs",
        "missing_expected_matrix_records", "duplicate_expected_matrix_records",
        "extra_expected_matrix_records",
    }
    baseline_matrix_incomplete = baseline_matrix is not None and any(
        baseline_matrix[key]
        for key in ("missing", "duplicate", "extra", "invalid_records")
    )
    if not reasons and (
        baseline_matrix_incomplete
        or any(reason in baseline_pairing_issues for reason in baseline_safety_reasons)
    ):
        reasons.append("baseline_incomplete_pairing")
    if not reasons and (
        baseline["seat_balanced_win_rate"] is None
        or baseline["median_paired_bank_differential"] is None
        or candidate["seat_balanced_win_rate"] <= baseline["seat_balanced_win_rate"]
        or candidate["median_paired_bank_differential"] <= baseline["median_paired_bank_differential"]
    ):
        reasons.append("no_paired_improvement")
    return {
        "status": "promote" if not reasons else "discard",
        "reasons": reasons,
        "baseline_safety_gate_reasons": baseline_safety_reasons,
        "candidate": candidate,
        "baseline": baseline,
        "matrix_completeness": candidate_matrix,
        "baseline_matrix_completeness": baseline_matrix,
        "paired_metric_deltas": paired_metric_deltas,
        "diagnostic_deltas": diagnostic_comparison["deltas"],
        "diagnostic_regressions": diagnostic_comparison["regressions"],
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _strict_numeric_scalar(value: Any) -> bool:
    """Accept only finite JSON numeric scalars, never strings or booleans."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _strict_engine_numeric_values(value: Any, *, in_configuration: bool = False,
                                  numeric: bool = False) -> bool:
    """Reject numeric strings and non-numeric values in strict engine fields.

    Compact unit fixtures intentionally use abbreviated values and are excluded
    by the caller.  Real engine envelopes use typed JSON numbers, so coercing a
    string such as ``"100"`` would hide replay tampering.
    """
    if numeric:
        return _strict_numeric_scalar(value)
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"metadata", "custom"}:
                continue
            if key == "specification":
                # This is the engine's JSON schema, whose numeric-looking
                # values are descriptive metadata rather than replay data.
                continue
            child_in_configuration = in_configuration or key == "configuration"
            if key == "rewards":
                if not isinstance(child, Sequence) or isinstance(child, (str, bytes)) \
                        or any(not _strict_numeric_scalar(reward) for reward in child):
                    return False
            elif key in _STRICT_NUMERIC_FIELDS:
                if child is None:
                    if key == "seed" and child_in_configuration:
                        continue
                    return False
                if not _strict_engine_numeric_values(
                    child, in_configuration=child_in_configuration, numeric=True,
                ):
                    return False
            elif key in _STRICT_QUANTITY_MAPPING_FIELDS:
                if not isinstance(child, Mapping):
                    return False
                if any(not _strict_numeric_scalar(quantity) for quantity in child.values()):
                    return False
            elif key == "inventories":
                if not isinstance(child, Sequence) or isinstance(child, (str, bytes)):
                    return False
                if any(
                    not isinstance(inventory, Mapping)
                    or any(not _strict_numeric_scalar(quantity) for quantity in inventory.values())
                    for inventory in child
                ):
                    return False
            elif not _strict_engine_numeric_values(child, in_configuration=child_in_configuration):
                return False
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_strict_engine_numeric_values(item, in_configuration=in_configuration) for item in value)
    if isinstance(value, str):
        try:
            float(value)
        except (TypeError, ValueError, OverflowError):
            return True
        return False
    return True


def _crop_data(crop: Any) -> Mapping[str, Any] | None:
    """Return crop rules without allowing malformed replay keys to escape."""
    if not isinstance(crop, str):
        return None
    return CROPS.get(crop)


def _config_value(configuration: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    return _mapping(configuration).get(key, default)


def _market_order_limit(configuration: Mapping[str, Any] | None) -> tuple[int, bool]:
    raw = _config_value(configuration, "maxMarketOrdersPerTurn", max_market_orders)
    if isinstance(raw, bool):
        return max_market_orders, False
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return max_market_orders, False
    if value < 1 or (isinstance(raw, float) and not raw.is_integer()):
        return max_market_orders, False
    return value, True


def _fib(index: int) -> int:
    first, second = 1, 1
    for _ in range(max(0, index)):
        first, second = second, first + second
    return first


def _player_states(replay: Mapping[str, Any], player: int) -> list[Mapping[str, Any]]:
    states = []
    steps = replay.get("steps", ())
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return states
    for turn in steps:
        if not isinstance(turn, Sequence):
            continue
        for state in turn:
            if isinstance(state, Mapping) and _mapping(state.get("observation")).get("player") == player:
                states.append(state)
                break
    return states


def _valid_market_order_schema(order: Any, observation: Mapping[str, Any]) -> bool:
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)) or not order or not isinstance(order[0], str):
        return False
    operation = order[0]
    if operation in {"HIRE", "BUY_LAND"}:
        return len(order) == 1
    if operation not in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
        return False
    if len(order) != 3 or not isinstance(order[1], str) or not isinstance(order[2], int) or isinstance(order[2], bool) or order[2] < 1:
        return False
    item = order[1]
    if operation == "BUY_SEED":
        return item in CROPS
    if operation == "BUY_ANIMAL":
        return item in ANIMALS
    if operation == "BUY_PRODUCT":
        return item in _BUYABLE_PRODUCTS
    if item not in PRODUCTS:
        return False
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    inventory = _number(_mapping(market.get("inventory")).get(item))
    price = _number(prices.get(item))
    if operation == "BUY_PRODUCT" and item not in _BUYABLE_PRODUCTS:
        return False
    return inventory is not None and inventory >= 0 and price is not None and price >= PRICE_FLOOR


def _valid_action_schema(action: Any, observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None,
                         *, state_aware: bool = True, validate_market: bool = True) -> bool:
    if not isinstance(action, Mapping) or set(action) != {"farmer", "hands", "market"}:
        return False
    farmer = action.get("farmer")
    hands = action.get("hands")
    orders = action.get("market")
    farm = _farm_observation(observation)
    valid_farmer = (
        _unit_command_valid_for_state(farmer, observation, 0, configuration)
        if state_aware else _legal_unit_command(farmer)
    )
    if not valid_farmer or not isinstance(hands, Sequence) or isinstance(hands, (str, bytes)):
        return False
    expected_hands = farm.get("hands", ())
    if not isinstance(expected_hands, Sequence) or isinstance(expected_hands, (str, bytes)) or len(hands) != len(expected_hands):
        return False
    if state_aware:
        valid_hands = all(_unit_command_valid_for_state(command, observation, index + 1, configuration) for index, command in enumerate(hands))
    else:
        valid_hands = all(_legal_unit_command(command) for command in hands)
    if not valid_hands:
        return False
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        return False
    order_limit, config_valid = _market_order_limit(configuration)
    if not config_valid:
        return False
    if len(orders) > order_limit:
        return False
    if not validate_market:
        return all(_valid_market_order_schema(order, observation) for order in orders)
    if state_aware:
        return _valid_market_orders(orders, observation, configuration)
    return all(_valid_market_order_schema(order, observation) for order in orders)


def _valid_replay(replay: Mapping[str, Any], own_states: Sequence[Mapping[str, Any]],
                  other_states: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any] | None = None,
                  *, expected_seed: int | None = None, candidate_player: int = 0) -> bool:
    if not isinstance(replay, Mapping):
        return False
    if expected_seed is not None and type(expected_seed) is not int:
        return False
    if not _valid_engine_provenance(replay):
        return False
    legacy_compact = _legacy_compact_fixture(replay)
    if not legacy_compact and not _strict_engine_numeric_values(replay):
        return False
    for optional_mapping in ("metadata",):
        if optional_mapping in replay and not isinstance(replay[optional_mapping], Mapping):
            return False
    info = replay.get("info")
    metadata = replay.get("metadata")
    if expected_seed is not None:
        if not isinstance(info, Mapping) or info.get("seed") != expected_seed:
            return False
        if isinstance(metadata, Mapping) and "seed" in metadata and metadata.get("seed") != expected_seed:
            return False
        configured_seed = _mapping(configuration).get("seed")
        if configured_seed is not None and configured_seed != expected_seed:
            return False
    statuses = replay.get("statuses")
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)) or list(statuses) != ["DONE", "DONE"]:
        return False
    steps = replay.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        return False
    rewards = replay.get("rewards")
    if rewards is not None:
        if not isinstance(rewards, Sequence) or isinstance(rewards, (str, bytes)) or len(rewards) != 2:
            return False
        if any(_number(reward) is None for reward in rewards):
            return False
    if not legacy_compact:
        episode_steps = _number(_config_value(configuration, "episodeSteps", None))
        if episode_steps is None or int(episode_steps) != episode_steps or len(steps) != int(episode_steps):
            return False
    if len(own_states) != len(steps) or len(other_states) != len(steps):
        return False
    if type(candidate_player) is not int or candidate_player not in (0, 1):
        return False
    if not own_states or not other_states:
        return False
    states_by_role = (own_states, other_states)
    previous_observations = {
        candidate_player: _mapping(states_by_role[0][0].get("observation")),
        1 - candidate_player: _mapping(states_by_role[1][0].get("observation")),
    }
    for index, turn in enumerate(steps):
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes)) or len(turn) != 2:
            return False
        players = set()
        states_by_player = {}
        for state in turn:
            if not isinstance(state, Mapping):
                return False
            observation = state.get("observation")
            player = _mapping(observation).get("player")
            if type(player) is not int or player not in {0, 1} or player in players or not isinstance(observation, Mapping):
                return False
            players.add(player)
            states_by_player[player] = state
            if not isinstance(state.get("action"), Mapping) or state.get("status") not in ("ACTIVE", "DONE"):
                return False
            if "info" in state and not isinstance(state["info"], Mapping):
                return False
            if state.get("error") or _mapping(state.get("info")).get("error"):
                return False
            if not legacy_compact and not _strict_private_inventories(observation):
                return False
            if not _valid_action_schema(
                state["action"], previous_observations[player], configuration,
                state_aware=player == candidate_player, validate_market=False,
            ):
                return False
        if players != {0, 1}:
            return False
        actions = [states_by_player[player]["action"] for player in (0, 1)]
        observations = [previous_observations[player] for player in (0, 1)]
        current_observations = [
            _mapping(states_by_player[player].get("observation")) for player in (0, 1)
        ]
        if index > 0 and any(
            not _time_progression_valid(
                observations[player], current_observations[player], configuration,
                allow_missing=legacy_compact, allow_missing_step=player == 1,
            )
            for player in (0, 1)
        ):
            return False
        if not legacy_compact and any(
            not _board_dimensions_valid(observation, configuration)
            for observation in (*observations, *current_observations)
        ):
            return False
        market_result = _simulate_market_orders_lockstep(
            [actions[0].get("market", ()), actions[1].get("market", ())], observations, configuration,
            force_model=bool(actions[1].get("market", ())),
            candidate_player=candidate_player,
        )
        if market_result is None:
            return False
        if index > 0:
            for player in (0, 1):
                if not _transition_effects_valid(
                    observations[player], current_observations[player], actions[player], configuration,
                    market_result=market_result, player_index=player, allow_compact=legacy_compact,
                    market_observation=observations[0], allow_invalid_unit_noop=player != candidate_player,
                ):
                    return False
        expected_status = "DONE" if index == len(steps) - 1 else "ACTIVE"
        if any(state.get("status") != expected_status for state in turn if isinstance(state, Mapping)):
            return False
        previous_observations = {
            player: _mapping(state.get("observation"))
            for player in (0, 1)
            for state in turn
            if isinstance(state, Mapping) and _mapping(state.get("observation")).get("player") == player
        }
    final_banks = [_final_bank(own_states[-1]), _final_bank(other_states[-1])]
    if any(bank is None for bank in final_banks):
        return False
    if not legacy_compact:
        reward_by_role = (rewards[candidate_player], rewards[1 - candidate_player]) if rewards is not None else ()
        if rewards is None or any(final_banks[player] != _number(reward_by_role[player]) for player in (0, 1)):
            return False
        if _requires_full_liquidation(configuration, len(steps)):
            final_inventories = [
                _private_inventories(_mapping(states[-1].get("observation")))
                for states in (own_states, other_states)
            ]
            if any(inventories is None or any(inventory for inventory in inventories) for inventories in final_inventories):
                return False
            if any(
                _has_saleable_shed_goods(_mapping(states[-1].get("observation")))
                for states in (own_states, other_states)
            ):
                return False
    else:
        final_inventories = _private_inventories(_mapping(own_states[-1].get("observation")))
        if final_inventories is None or any(inventory for inventory in final_inventories):
            return False
    return not bool(_mapping(replay.get("info")).get("error"))


def _legacy_compact_fixture(replay: Mapping[str, Any]) -> bool:
    metadata = replay.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("legacy_compact_fixture") is True


def _requires_full_liquidation(configuration: Mapping[str, Any] | None, step_count: int) -> bool:
    """Require terminal cleanup only for a complete configured season."""
    episode_steps = _number(_config_value(configuration, "episodeSteps", None))
    turns = _number(_config_value(configuration, "turnsPerDay", None))
    if (episode_steps is None or turns is None or int(episode_steps) != episode_steps
            or int(turns) != turns or int(turns) < 1):
        return False
    return int(episode_steps) == step_count and int(episode_steps) >= int(turns) * season_days


def _has_saleable_shed_goods(observation: Mapping[str, Any]) -> bool:
    """Return whether the terminal shed still contains goods that should sell."""
    shed = _mapping(_mapping(observation.get("private")).get("shed"))
    return any(
        item in _SALEABLE_PRODUCTS and (_number(quantity) or 0) > 0
        for item, quantity in shed.items()
    )


def _time_progression_valid(pre: Mapping[str, Any], post: Mapping[str, Any],
                            configuration: Mapping[str, Any] | None, *, allow_missing: bool = False,
                            allow_missing_step: bool = False) -> bool:
    """Validate one player's deterministic step/hour/day progression."""
    pre_step, post_step = _number(pre.get("step")), _number(post.get("step"))
    pre_hour, post_hour = _number(pre.get("hour")), _number(post.get("hour"))
    pre_day, post_day = _number(pre.get("day")), _number(post.get("day"))
    if pre_step is None or post_step is None:
        if not allow_missing_step or pre_step is not None or post_step is not None:
            return allow_missing
    if pre_hour is None or post_hour is None:
        return allow_missing
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", None))
    if turns_per_day is None or int(turns_per_day) != turns_per_day or int(turns_per_day) < 1:
        return False
    if any(int(value) != value for value in (pre_hour, post_hour)):
        return False
    if pre_step is not None and post_step is not None and any(
        int(value) != value for value in (pre_step, post_step)
    ):
        return False
    if pre_step is not None and post_step is not None and int(post_step) != int(pre_step) + 1:
        return False
    expected_hour = (int(pre_hour) + 1) % int(turns_per_day)
    if int(post_hour) != expected_hour:
        return False
    if pre_day is None or post_day is None:
        return allow_missing and pre_day is None and post_day is None
    if any(int(value) != value for value in (pre_day, post_day)):
        return False
    expected_day = int(pre_day) + (1 if expected_hour == 0 else 0)
    return int(post_day) == expected_day


def _observation_step(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None) -> int:
    step = _number(observation.get("step"))
    if step is not None and int(step) == step:
        return int(step)
    day = _number(observation.get("day"))
    hour = _number(observation.get("hour"))
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", 24))
    if day is not None and hour is not None and turns_per_day is not None and int(turns_per_day) >= 1:
        return int(day) * int(turns_per_day) + int(hour)
    return 0


def _valid_engine_provenance(replay: Mapping[str, Any]) -> bool:
    """Require the replay envelope emitted by the Kaggriculture engine."""
    required_text = ("id", "name", "version", "module_version", "title", "description")
    if any(not isinstance(replay.get(key), str) or not replay[key] for key in required_text):
        return False
    if replay.get("name") != "kaggriculture" or replay.get("module_version") != ENGINE_VERSION:
        return False
    schema_version = replay.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        return False
    info = replay.get("info")
    if not isinstance(info, Mapping) or isinstance(info.get("seed"), bool) or not isinstance(info.get("seed"), int):
        return False
    specification = replay.get("specification")
    if not isinstance(specification, Mapping):
        return False
    action_spec = specification.get("action")
    agents = specification.get("agents")
    config_spec = specification.get("configuration")
    if not isinstance(action_spec, Mapping) or not isinstance(agents, Sequence) or isinstance(agents, (str, bytes)):
        return False
    if list(agents) != [2] or not isinstance(config_spec, Mapping):
        return False
    configuration = replay.get("configuration")
    if not isinstance(configuration, Mapping):
        return False
    integer_config = {
        "episodeSteps": 1,
        "boardSize": 4,
        "startingMoney": 0,
        "maxMarketOrdersPerTurn": 1,
        "turnsPerDay": 1,
        "shedCapacity": 1,
        "townShopUnlockInterval": 1,
        "townShopSellInterval": 1,
        "townCenterSellInterval": 1,
        "farmHandCostMult": 0,
    }
    for key, minimum in integer_config.items():
        value = configuration.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return False
    for key in ("actTimeout", "runTimeout", "weedSpawnChance"):
        value = _number(configuration.get(key))
        if value is None or value < 0:
            return False
    seed = configuration.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        return False
    if not isinstance(configuration.get("marketParams"), Mapping):
        return False
    return True


def _final_bank(state: Mapping[str, Any] | None) -> float | None:
    if state is None:
        return None
    observation = _mapping(state.get("observation"))
    player = observation.get("player")
    farms = observation.get("farms")
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)) or type(player) is not int:
        return None
    if not 0 <= player < len(farms):
        return None
    return _number(_mapping(farms[player]).get("money"))


def _shed_total(observation: Mapping[str, Any]) -> float:
    private = _mapping(observation.get("private"))
    shed = private.get("shed")
    if not isinstance(shed, Mapping):
        return 0.0
    return sum(max(0.0, _number(quantity) or 0.0) for quantity in shed.values())


def _inventory_total(observation: Mapping[str, Any]) -> float:
    """Return worker inventories that have not yet been dropped into the shed."""
    inventories = _mapping(_mapping(observation.get("private")).get("inventories"))
    if inventories:
        return sum(max(0.0, _number(quantity) or 0.0) for quantity in inventories.values())
    raw_inventories = _mapping(observation.get("private")).get("inventories")
    if not isinstance(raw_inventories, Sequence) or isinstance(raw_inventories, (str, bytes)):
        return 0.0
    return sum(
        max(0.0, _number(quantity) or 0.0)
        for inventory in raw_inventories
        if isinstance(inventory, Mapping)
        for quantity in inventory.values()
    )


def _terminal_inventory_value(observation: Mapping[str, Any],
                              configuration: Mapping[str, Any] | None = None) -> float:
    """Value terminal goods at the terminal market quote or acquisition cost.

    Shed and worker goods use the final observed market prices.  Animals use
    their purchase cost and seeds use their published seed cost because the
    engine has no sell quote for either inventory type.
    """
    private = _mapping(observation.get("private"))
    # Keep harvested/product goods and private seeds in separate ledgers.  A
    # crop name such as WHEAT can occur in both, but the seed inventory is
    # valued at CROPS[item]["seed"], never at the harvested-product quote.
    goods: Counter[str] = Counter()
    for source in (private.get("shed"), private.get("inventories")):
        if isinstance(source, Mapping):
            sources = (source,)
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            sources = tuple(source)
        else:
            sources = ()
        for inventory in sources:
            if isinstance(inventory, Mapping):
                for item, raw_quantity in inventory.items():
                    quantity = _number(raw_quantity)
                    if isinstance(item, str) and quantity is not None and quantity > 0:
                        goods[item] += quantity
    seed_quantities: Counter[str] = Counter()
    seeds = private.get("seeds")
    if isinstance(seeds, Mapping):
        for item, raw_quantity in seeds.items():
            quantity = _number(raw_quantity)
            if isinstance(item, str) and item in CROPS and quantity is not None and quantity > 0:
                seed_quantities[item] += quantity
    prices = _mapping(_mapping(observation.get("market")).get("prices"))
    market_inventory = _mapping(_mapping(observation.get("market")).get("inventory"))
    params = _mapping(_config_value(configuration, "marketParams", {}))
    value = 0.0
    for item, quantity in goods.items():
        price = _number(prices.get(item))
        if price is None and item in PRODUCTS:
            price = float(market_price(item, _number(market_inventory.get(item)) or 0.0, params))
        if price is None and item in ANIMALS:
            price = float(ANIMALS[item]["cost"])
        if price is not None:
            value += quantity * price
    value += sum(quantity * float(CROPS[item]["seed"])
                 for item, quantity in seed_quantities.items())
    return float(value)


def _shed_capacity(configuration: Mapping[str, Any] | None = None) -> int:
    raw = _config_value(configuration, "shedCapacity", shed_capacity)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError, OverflowError):
        return shed_capacity


def _tiles(observation: Mapping[str, Any]) -> Sequence[Any]:
    return tuple(tile for _position, tile in _tile_entries(observation))


def _tile_entries(observation: Mapping[str, Any]) -> Sequence[tuple[tuple[int, int], Any]]:
    farms = observation.get("farms")
    player = observation.get("player")
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)) or type(player) is not int:
        return ()
    if not 0 <= player < len(farms):
        return ()
    tiles = _mapping(farms[player]).get("tiles", ())
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return ()
    return tuple(
        ((x, y), tile)
        for y, row in enumerate(tiles)
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
        for x, tile in enumerate(row)
    )


def _tile_kind(tile: Any) -> str:
    if isinstance(tile, str):
        return tile.upper()
    return str(_mapping(tile).get("kind", "")).upper()


def _animal_state(tile: Any) -> Mapping[str, Any]:
    """Return the animal fields in either engine or policy-test tile shape."""
    if not isinstance(tile, Mapping):
        return {}
    animal = tile.get("animal")
    if isinstance(animal, Mapping):
        state = dict(tile)
        state.update(animal)
        return state
    if animal is not None or _tile_kind(tile) == "ANIMAL":
        return tile
    return {}


def _commands_for_state(state: Mapping[str, Any]) -> list[list[Any]]:
    action = _mapping(state.get("action"))
    commands = []
    farmer = action.get("farmer")
    if isinstance(farmer, Sequence) and not isinstance(farmer, (str, bytes)):
        commands.append(list(farmer))
    hands = action.get("hands", ())
    if isinstance(hands, Sequence) and not isinstance(hands, (str, bytes)):
        commands.extend(list(command) for command in hands if isinstance(command, Sequence) and not isinstance(command, (str, bytes)))
    return commands


def _command_targets(observation: Mapping[str, Any], state: Mapping[str, Any],
                     configuration: Mapping[str, Any] | None = None) -> dict[str, set[tuple[int, int]]]:
    farm = _farm_observation(observation)
    positions = [farm.get("farmer"), *list(farm.get("hands", ()) or ())]
    commands = _commands_for_state(state)
    targets: dict[str, set[tuple[int, int]]] = {}
    for worker_index, (position, command) in enumerate(zip(positions, commands)):
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 2:
            continue
        if not command or not isinstance(command[0], str):
            continue
        try:
            coordinate = (int(position[0]), int(position[1]))
        except (TypeError, ValueError, OverflowError):
            continue
        if _unit_command_valid_for_state(command, observation, worker_index, configuration):
            targets.setdefault(command[0], set()).add(coordinate)
    return targets


def _post_tile(observation: Mapping[str, Any] | None, coordinate: tuple[int, int]) -> Mapping[str, Any]:
    if not observation:
        return {}
    return next((_mapping(tile) for position, tile in _tile_entries(observation) if position == coordinate), {})


def _missed_needs_at_boundary(observation: Mapping[str, Any], is_boundary: bool,
                              post_observation: Mapping[str, Any] | None = None,
                              action_state: Mapping[str, Any] | None = None,
                              configuration: Mapping[str, Any] | None = None) -> int:
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", 24))
    last_hour = int(turns_per_day) - 1 if turns_per_day is not None and int(turns_per_day) >= 1 else 23
    if not is_boundary or post_observation is None or _number(observation.get("hour")) != last_hour or _number(post_observation.get("hour")) != 0:
        return 0
    missed = 0
    reset_after_boundary = True
    targets = _command_targets(observation, action_state or {}, configuration)
    for index, (coordinate, tile) in enumerate(_tile_entries(observation)):
        if not isinstance(tile, Mapping):
            continue
        kind = _tile_kind(tile)
        post_tile = _post_tile(post_observation, coordinate)
        if kind == "PLANT":
            needs_water = tile.get("needs_water") is True or tile.get("watered_today") is False
            post_says_watered = post_tile.get("watered_today") is True
            target_satisfied = coordinate in targets.get("WATER", set())
            if needs_water and not post_says_watered and not (reset_after_boundary and target_satisfied) and not (post_observation is None and target_satisfied):
                missed += 1
        animal_state = _animal_state(tile)
        if animal_state:
            needs_feed = animal_state.get("needs_feed") is True or animal_state.get("fed_today") is False
            post_fed = post_tile.get("fed_today") is True
            target_fed = coordinate in targets.get("FEED", set())
            if needs_feed and not post_fed and not (reset_after_boundary and target_fed) and not (post_observation is None and target_fed):
                missed += 1
            # CARE is an optional production bonus, not a basic need.  It is
            # intentionally excluded from this required-needs metric.
    return missed


def _price_floor_sales(state: Mapping[str, Any], observation: Mapping[str, Any] | None = None,
                       configuration: Mapping[str, Any] | None = None,
                       *, other_state: Mapping[str, Any] | None = None,
                       other_observation: Mapping[str, Any] | None = None) -> int:
    observation = observation or _mapping(state.get("observation"))
    action = _mapping(state.get("action"))
    other_action = _mapping((other_state or {}).get("action"))
    result = _simulate_market_orders_lockstep(
        [action.get("market", ()), other_action.get("market", ())],
        [observation, other_observation or observation],
        configuration,
        force_model=bool(other_action.get("market", ())),
        candidate_player=0,
    )
    return int(result["floor_sales"][0]) if result is not None else 0


def _framework_error_record(*, variant: str, opponent: str, seed: int, seat: int = 0,
                            error: str | None = None,
                            reason: str = "engine_error") -> dict[str, Any]:
    normalized_seat = seat if type(seat) is int and seat in (0, 1) else 0
    record = {
        "candidate": variant,
        "variant": variant,
        "opponent": opponent,
        "seed": seed,
        "seat": normalized_seat,
        "outcome": "framework_error",
        "final_bank": None,
        "opponent_final_bank": None,
        "bank_differential": 0.0,
        "framework_error": True,
        "framework_error_reasons": [reason] if reason in FRAMEWORK_REASON_ORDER else ["engine_error"],
        "shed_overflow": 0.0,
        "price_floor_sales": 0,
        "missed_basic_needs": 0,
        "terminal_cash": None,
        "terminal_inventory_value": None,
        "submitted_market_order_count": 0,
        "market_transaction_count": 0,
        "same_item_market_churn": 0,
        "same_item_sell_buy_churn": 0,
        "termination_reason": "engine_error",
        "bootstrap_truncated": False,
        "no_progress_steps": 0,
        "time_limit_ending": False,
        "safety_flags": [],
        "safety_regression": False,
        "shaping_count": 0,
    }
    if error:
        record["error"] = str(error)[:1000]
    return record


def _replay_diagnostics(
    replay: Mapping[str, Any], own_states: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract game-level diagnostics without inventing transition outcomes."""
    sources: list[tuple[str, Mapping[str, Any]]] = [("replay", replay)]
    replay_info = _mapping(replay.get("info"))
    replay_diagnostics = _mapping(replay.get("diagnostics"))
    if replay_info:
        sources.append(("replay.info", replay_info))
    if replay_diagnostics:
        sources.append(("replay.diagnostics", replay_diagnostics))
    for index, state in enumerate(own_states):
        source_name = f"state[{index}]"
        sources.append((source_name, state))
        state_info = _mapping(state.get("info"))
        state_diagnostics = _mapping(state.get("diagnostics"))
        if state_info:
            sources.append((f"{source_name}.info", state_info))
        if state_diagnostics:
            sources.append((f"{source_name}.diagnostics", state_diagnostics))

    termination_reason = None
    bootstrap_truncated = False
    no_progress_steps = 0
    time_limit_ending = False
    shaping_count = 0
    safety_flags: list[str] = []
    safety_regression = False
    for source_name, source in sources:
        if "termination_reason" in source:
            raw_reason = source["termination_reason"]
            if raw_reason is not None and (type(raw_reason) is not str or not raw_reason):
                raise ValueError(
                    f"{source_name}.termination_reason must be a non-empty string or null"
                )
        else:
            raw_reason = None
        if raw_reason is not None and str(raw_reason):
            termination_reason = str(raw_reason)
        if "bootstrap_truncated" in source:
            raw_bootstrap = source["bootstrap_truncated"]
            if raw_bootstrap is not None and type(raw_bootstrap) is not bool:
                raise ValueError(
                    f"{source_name}.bootstrap_truncated must be boolean or null"
                )
        else:
            raw_bootstrap = None
        if raw_bootstrap is True:
            bootstrap_truncated = True
        if "no_progress_steps" in source:
            raw_steps_value = source["no_progress_steps"]
            if raw_steps_value is not None and (
                type(raw_steps_value) is not int or raw_steps_value < 0
            ):
                raise ValueError(
                    f"{source_name}.no_progress_steps must be a nonnegative integer or null"
                )
        else:
            raw_steps_value = None
        raw_steps = _number(raw_steps_value)
        if raw_steps is not None:
            no_progress_steps = max(no_progress_steps, max(0, int(raw_steps)))
        if "time_limit_ending" in source:
            raw_time_limit = source["time_limit_ending"]
            if raw_time_limit is not None and type(raw_time_limit) is not bool:
                raise ValueError(
                    f"{source_name}.time_limit_ending must be boolean or null"
                )
        else:
            raw_time_limit = None
        if raw_time_limit is True:
            time_limit_ending = True
        if "shaping_count" in source:
            raw_shaping_value = source["shaping_count"]
            if raw_shaping_value is not None and (
                type(raw_shaping_value) is not int or raw_shaping_value < 0
            ):
                raise ValueError(
                    f"{source_name}.shaping_count must be a nonnegative integer or null"
                )
        else:
            raw_shaping_value = None
        raw_shaping = _number(raw_shaping_value)
        if raw_shaping is not None:
            shaping_count = max(shaping_count, max(0, int(raw_shaping)))
        if "safety_regression" in source:
            raw_safety_regression = source["safety_regression"]
            if raw_safety_regression is not None and type(raw_safety_regression) is not bool:
                raise ValueError(
                    f"{source_name}.safety_regression must be boolean or null"
                )
        else:
            raw_safety_regression = None
        if raw_safety_regression is True:
            safety_regression = True
        if "safety_flags" in source:
            raw_flags = source["safety_flags"]
            if raw_flags is None or not isinstance(raw_flags, (list, tuple, set)):
                raise ValueError(
                    f"{source_name}.safety_flags must be a sequence of non-empty strings"
                )
            if any(type(flag) is not str or not flag for flag in raw_flags):
                raise ValueError(
                    f"{source_name}.safety_flags must be a sequence of non-empty strings"
                )
        else:
            raw_flags = ()
        for flag in raw_flags:
            if flag not in safety_flags:
                safety_flags.append(flag)
                if "safety_regression" in flag.lower():
                    safety_regression = True

    final_state = own_states[-1] if own_states else {}
    final_observation = _mapping(final_state.get("observation"))
    final_step = _number(final_observation.get("step"))
    episode_steps = _number(_config_value(configuration, "episodeSteps", 0))
    inferred_time_limit = (
        final_step is not None and episode_steps is not None and episode_steps > 0
        and final_step >= episode_steps - 1
    )
    time_limit_ending = time_limit_ending or inferred_time_limit
    if termination_reason is None:
        termination_reason = "time_limit" if time_limit_ending else "terminal"
    return {
        "termination_reason": termination_reason,
        "bootstrap_truncated": bootstrap_truncated,
        "no_progress_steps": no_progress_steps,
        "time_limit_ending": time_limit_ending,
        "safety_flags": safety_flags,
        "safety_regression": safety_regression,
        "shaping_count": shaping_count,
    }


def _replay_record(replay: Mapping[str, Any], *, variant: str, opponent: str, seed: int,
                   seat: int = 0, churn_window: int = 2) -> dict[str, Any]:
    """Extract one game record from the engine replay JSON."""
    if type(seed) is not int:
        return _framework_error_record(
            variant=variant, opponent=opponent, seed=0, seat=seat,
            error="seed must be an integer",
        )
    if type(seat) is not int or seat not in (0, 1):
        return _framework_error_record(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            error="seat must be 0 or 1",
        )
    if not isinstance(replay, Mapping):
        return _framework_error_record(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            reason="malformed_replay",
        )
    own_states = _player_states(replay, seat)
    other_states = _player_states(replay, 1 - seat)
    replay_configuration = _mapping(replay.get("configuration"))
    framework_error = not _valid_replay(
        replay, own_states, other_states, replay_configuration, expected_seed=seed,
        candidate_player=seat,
    )
    own_bank = _final_bank(own_states[-1] if own_states else None)
    other_bank = _final_bank(other_states[-1] if other_states else None)
    if framework_error or own_bank is None or other_bank is None:
        outcome = "framework_error"
        differential = 0.0
    elif own_bank > other_bank:
        outcome, differential = "win", own_bank - other_bank
    elif own_bank < other_bank:
        outcome, differential = "loss", own_bank - other_bank
    else:
        outcome, differential = "tie", 0.0

    shed_overflow = 0.0
    floor_sales = 0
    missed_needs = 0
    market_metrics = _market_metrics(own_states, churn_window=churn_window)
    turns_per_day = _number(_config_value(replay_configuration, "turnsPerDay", 24))
    last_hour = int(turns_per_day) - 1 if turns_per_day is not None and int(turns_per_day) >= 1 else 23
    steps = replay.get("steps", ())
    for index, state in enumerate(own_states):
        post = _mapping(state.get("observation"))
        pre = _mapping(own_states[index - 1].get("observation")) if index else post
        capacity = _shed_capacity(replay_configuration)
        shed_overflow = max(shed_overflow, max(0.0, _shed_total(post) - capacity))
        # Kaggriculture emits a bootstrap record at index 0. Its placeholder
        # action is schema-checked, but it was not chosen from a preceding
        # replay observation and must not contribute action-derived metrics.
        is_bootstrap = index == 0 and len(own_states) > 1
        if not is_bootstrap:
            other_state = other_states[index] if index < len(other_states) else {}
            other_pre = _mapping(other_states[index - 1].get("observation")) if index else _mapping(other_state.get("observation"))
            floor_sales += _price_floor_sales(
                state, pre, replay_configuration,
                other_state=other_state, other_observation=other_pre,
            )
        pre_hour = _number(pre.get("hour"))
        post_hour = _number(post.get("hour"))
        is_boundary = pre_hour == last_hour and post_hour == 0
        if is_boundary:
            shed_overflow = max(
                shed_overflow,
                max(0.0, _shed_total(pre) + _inventory_total(pre) - capacity),
            )
        if not is_bootstrap:
            missed_needs += _missed_needs_at_boundary(pre, is_boundary, post, state, replay_configuration)
    diagnostic_error = None
    try:
        diagnostics = _replay_diagnostics(replay, own_states, replay_configuration)
    except ValueError as exc:
        diagnostic_error = str(exc)
        diagnostics = {
            "termination_reason": "malformed_replay",
            "bootstrap_truncated": False,
            "no_progress_steps": 0,
            "time_limit_ending": False,
            "safety_flags": [],
            "safety_regression": False,
            "shaping_count": 0,
        }
    diagnostic_record = {
        "malformed_replay": (
            framework_error or own_bank is None or other_bank is None
            or diagnostic_error is not None
        ),
        "missed_basic_needs": missed_needs,
        **market_metrics,
    }
    reasons = framework_error_reasons(diagnostic_record)
    if reasons:
        outcome, differential = "framework_error", 0.0
    record = {
        "candidate": variant,
        "variant": variant,
        "opponent": opponent,
        "seed": seed,
        "seat": seat,
        "outcome": outcome,
        "final_bank": own_bank,
        "opponent_final_bank": other_bank,
        "bank_differential": differential,
        "framework_error": bool(reasons),
        "framework_error_reasons": reasons,
        "shed_overflow": shed_overflow,
        "price_floor_sales": floor_sales,
        "missed_basic_needs": missed_needs,
        "terminal_cash": own_bank,
        "terminal_inventory_value": (
            _terminal_inventory_value(_mapping(own_states[-1].get("observation")), replay_configuration)
            if own_states else None
        ),
        **market_metrics,
        **diagnostics,
    }
    if diagnostic_error is not None:
        record["error"] = diagnostic_error[:1000]
    return record


def replay_record(replay: Mapping[str, Any], *, variant: str, opponent: str, seed: int,
                  seat: int = 0, candidate_player: int | None = None,
                  churn_window: int = 2) -> dict[str, Any]:
    """Extract one replay record, classifying malformed replay data safely."""
    if type(seat) is not int or seat not in (0, 1):
        return _framework_error_record(
            variant=variant, opponent=opponent, seed=seed, seat=0,
            error="seat must be 0 or 1",
        )
    if candidate_player is not None:
        if type(candidate_player) is not int or candidate_player not in (0, 1):
            return _framework_error_record(
                variant=variant, opponent=opponent, seed=seed, seat=seat,
                error="candidate_player must be 0 or 1",
            )
        seat = candidate_player
    return _replay_record(
        replay, variant=variant, opponent=opponent, seed=seed, seat=seat,
        churn_window=churn_window,
    )


def _farm_observation(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    farms = observation.get("farms")
    player = observation.get("player")
    if isinstance(farms, Sequence) and not isinstance(farms, (str, bytes)) and type(player) is int and 0 <= player < len(farms):
        return _mapping(farms[player])
    return {}


def _private_seeds(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(observation.get("private")).get("seeds"))


def _cash(observation: Mapping[str, Any]) -> float:
    return _number(_farm_observation(observation).get("money")) or 0.0


def _inert_market_order(order: Any) -> bool:
    """Return whether an entry contains no payload for schema validation."""
    return isinstance(order, Collection) and len(order) == 0


def _copy_market_order(order: Any) -> Any:
    return (
        list(order)
        if isinstance(order, Sequence) and not isinstance(order, (str, bytes))
        else order
    )


def _market_orders(action: Mapping[str, Any]) -> list[Any]:
    orders = action.get("market", ())
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        return []
    return [
        _copy_market_order(order)
        for order in orders
        if not _inert_market_order(order)
    ]


def _valid_market_container(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _observed_unit_price(item: str, observation: Mapping[str, Any], inventory: float, *, buying: bool,
                         configuration: Mapping[str, Any] | None = None,
                         force_model: bool = False) -> float:
    market = _mapping(observation.get("market"))
    params = _mapping(_config_value(configuration, "marketParams", {}))
    observed = _number(_mapping(market.get("prices")).get(item))
    quote_inventory = inventory - 1 if buying else inventory
    try:
        quote = float(market_price(item, quote_inventory, params))
        # A replay without configuration may contain a deliberately fixed
        # floor quote. Preserve that valid observation while still recomputing
        # later units from the market curve as inventory changes.
        if not force_model and not params and observed == PRICE_FLOOR:
            return PRICE_FLOOR
        return quote
    except (KeyError, TypeError, ValueError, OverflowError):
        return observed or 0.0


def _sanitize_market_orders(orders: Sequence[Any], observation: Mapping[str, Any],
                            configuration: Mapping[str, Any] | None = None, *,
                            preserve_malformed: bool = False) -> list[Any]:
    """Apply the engine's per-unit market rules to a postprocessed action."""
    money = _cash(observation)
    shed = dict(_mapping(_mapping(observation.get("private")).get("shed")))
    market = _mapping(observation.get("market"))
    market_inventory = {item: _number(quantity) or 0.0 for item, quantity in _mapping(market.get("inventory")).items()}
    try:
        capacity = max(1, int(_config_value(configuration, "shedCapacity", shed_capacity)))
    except (TypeError, ValueError, OverflowError):
        capacity = shed_capacity
    order_limit, _config_valid = _market_order_limit(configuration)
    farm = _farm_observation(observation)
    hires_today = int(_number(farm.get("hires_today")) or 0)
    hire_mult = max(0, int(_config_value(configuration, "farmHandCostMult", 1)))
    raw_unlocked = farm.get("unlocked_quadrants", ())
    unlocked = list(raw_unlocked) if isinstance(raw_unlocked, Sequence) and not isinstance(raw_unlocked, (str, bytes)) else []
    sanitized: list[Any] = []
    for raw_order in orders:
        if len(sanitized) >= order_limit:
            break
        if not _valid_market_order_schema(raw_order, observation):
            if preserve_malformed and not _inert_market_order(raw_order):
                sanitized.append(_copy_market_order(raw_order))
            continue
        order = list(raw_order)
        operation = order[0]
        if operation == "HIRE":
            cost = float(_fib(hires_today) * hire_mult)
            if cost > money:
                continue
            money -= cost
            hires_today += 1
            sanitized.append(order)
            continue
        if operation == "BUY_LAND":
            next_index = len(unlocked) - 1
            if next_index < 0 or next_index >= len(LAND_ORDER):
                continue
            expected_land = LAND_ORDER[next_index]
            if expected_land in unlocked or money < LAND_PRICES[next_index]:
                continue
            money -= LAND_PRICES[next_index]
            unlocked.append(expected_land)
            sanitized.append(order)
            continue

        item = order[1]
        requested = order[2]
        if operation == "BUY_SEED":
            unit_cost = float(CROPS[item]["seed"])
            allowed = min(requested, int(money // unit_cost))
            if allowed:
                money -= unit_cost * allowed
                sanitized.append([operation, item, allowed])
            continue
        if operation == "BUY_ANIMAL":
            unit_cost = float(ANIMALS[item]["cost"])
            room = max(0, capacity - int(sum(max(0.0, _number(value) or 0.0) for value in shed.values())))
            allowed = min(requested, room, int(money // unit_cost))
            if allowed:
                money -= unit_cost * allowed
                shed[item] = (_number(shed.get(item)) or 0.0) + allowed
                sanitized.append([operation, item, allowed])
            continue
        if operation == "BUY_PRODUCT":
            allowed = 0
            for _ in range(requested):
                room = capacity - int(sum(max(0.0, _number(value) or 0.0) for value in shed.values()))
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=True, configuration=configuration)
                if room <= 0 or price <= 0 or money < price:
                    break
                money -= price
                market_inventory[item] = market_inventory.get(item, 0.0) - 1
                shed[item] = (_number(shed.get(item)) or 0.0) + 1
                allowed += 1
            if allowed:
                sanitized.append([operation, item, allowed])
            continue
        if operation == "SELL":
            available = int(_number(shed.get(item)) or 0)
            allowed = min(requested, available)
            for _ in range(allowed):
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=False, configuration=configuration)
                money += price
                market_inventory[item] = market_inventory.get(item, 0.0) + (1 if price > PRICE_FLOOR else 0)
            if allowed:
                shed[item] = available - allowed
                sanitized.append([operation, item, allowed])
    return sanitized


def _valid_market_orders(orders: Sequence[Any], observation: Mapping[str, Any],
                         configuration: Mapping[str, Any] | None = None) -> bool:
    """Validate sequential market orders against the pre-action farm state."""
    order_limit, config_valid = _market_order_limit(configuration)
    if not config_valid or len(orders) > order_limit:
        return False
    money = _cash(observation)
    shed = {item: max(0.0, _number(quantity) or 0.0)
            for item, quantity in _mapping(_mapping(observation.get("private")).get("shed")).items()}
    market = _mapping(observation.get("market"))
    market_inventory = {item: max(0.0, _number(quantity) or 0.0)
                        for item, quantity in _mapping(market.get("inventory")).items()}
    try:
        capacity = max(1, int(_config_value(configuration, "shedCapacity", shed_capacity)))
    except (TypeError, ValueError, OverflowError):
        return False
    farm = _farm_observation(observation)
    hires_today = int(_number(farm.get("hires_today")) or 0)
    try:
        hire_mult = int(_config_value(configuration, "farmHandCostMult", 1))
    except (TypeError, ValueError, OverflowError):
        return False
    raw_unlocked = farm.get("unlocked_quadrants", ())
    if not isinstance(raw_unlocked, Sequence) or isinstance(raw_unlocked, (str, bytes)):
        return False
    unlocked = list(raw_unlocked)

    for raw_order in orders:
        order = list(raw_order) if isinstance(raw_order, Sequence) and not isinstance(raw_order, (str, bytes)) else raw_order
        if not _valid_market_order_schema(order, observation):
            return False
        operation = order[0]
        if operation == "HIRE":
            cost = float(_fib(hires_today) * hire_mult)
            if cost > money:
                return False
            money -= cost
            hires_today += 1
            continue
        if operation == "BUY_LAND":
            next_index = len(unlocked) - 1
            if next_index < 0 or next_index >= len(LAND_ORDER):
                return False
            expected_land = LAND_ORDER[next_index]
            if expected_land in unlocked or money < LAND_PRICES[next_index]:
                return False
            money -= LAND_PRICES[next_index]
            unlocked.append(expected_land)
            continue

        item, quantity = order[1], order[2]
        if operation == "BUY_SEED":
            cost = float(CROPS[item]["seed"]) * quantity
            if cost > money:
                return False
            money -= cost
            continue
        if operation == "BUY_ANIMAL":
            cost = float(ANIMALS[item]["cost"]) * quantity
            room = capacity - int(sum(shed.values()))
            if cost > money or quantity > room:
                return False
            money -= cost
            shed[item] = shed.get(item, 0.0) + quantity
            continue
        if operation == "BUY_PRODUCT":
            if market_inventory.get(item, 0.0) < quantity:
                return False
            for _ in range(quantity):
                room = capacity - int(sum(shed.values()))
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=True, configuration=configuration)
                if room < 1 or price <= 0 or money < price:
                    return False
                money -= price
                market_inventory[item] -= 1
                shed[item] = shed.get(item, 0.0) + 1
            continue
        if operation == "SELL":
            if shed.get(item, 0.0) < quantity:
                return False
            for _ in range(quantity):
                price = _observed_unit_price(item, observation, market_inventory.get(item, 0.0), buying=False, configuration=configuration)
                if price <= 0:
                    return False
                shed[item] -= 1
                if price > PRICE_FLOOR:
                    market_inventory[item] = market_inventory.get(item, 0.0) + 1
    return True


def _market_sim_state(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    farm = _farm_observation(observation)
    raw_unlocked = farm.get("unlocked_quadrants", ())
    if not isinstance(raw_unlocked, Sequence) or isinstance(raw_unlocked, (str, bytes)):
        return None
    private = _mapping(observation.get("private"))
    raw_shed = private.get("shed", {})
    if not isinstance(raw_shed, Mapping):
        return None
    return {
        "money": _cash(observation),
        "shed": {item: max(0.0, _number(quantity) or 0.0) for item, quantity in raw_shed.items()},
        "seeds": {item: max(0.0, _number(quantity) or 0.0) for item, quantity in _mapping(private.get("seeds")).items()},
        "hires": int(_number(farm.get("hires_today")) or 0),
        "unlocked": list(raw_unlocked),
    }


def _simulate_market_orders_lockstep(orders_by_player: Sequence[Any], observations: Sequence[Mapping[str, Any]],
                                     configuration: Mapping[str, Any] | None = None,
                                     *, force_model: bool = False,
                                     candidate_player: int = 0) -> dict[str, Any] | None:
    """Simulate both market queues against one shared inventory snapshot."""
    if len(orders_by_player) != 2 or len(observations) != 2:
        return None
    if type(candidate_player) is not int or candidate_player not in (0, 1):
        return None
    order_limit, config_valid = _market_order_limit(configuration)
    if not config_valid:
        return None
    states = [_market_sim_state(observation) for observation in observations]
    if any(state is None for state in states):
        return None
    market = _mapping(observations[0].get("market"))
    market_inventory = {
        item: max(0.0, _number(quantity) or 0.0)
        for item, quantity in _mapping(market.get("inventory")).items()
    }
    if _mapping(observations[1].get("market")).get("inventory") != _mapping(market.get("inventory")):
        return None
    queues = []
    for player, raw_orders in enumerate(orders_by_player):
        if not isinstance(raw_orders, Sequence) or isinstance(raw_orders, (str, bytes)) or len(raw_orders) > order_limit:
            return None
        queue = []
        for raw_order in raw_orders:
            order = list(raw_order) if isinstance(raw_order, Sequence) and not isinstance(raw_order, (str, bytes)) else raw_order
            if not _valid_market_order_schema(order, observations[player]):
                return None
            queue.append(order)
        queues.append(queue)

    capacity = _shed_capacity(configuration)
    try:
        hire_mult = int(_config_value(configuration, "farmHandCostMult", 1))
    except (TypeError, ValueError, OverflowError):
        return None
    floor_sales = [0, 0]
    for order_index in range(max((len(queue) for queue in queues), default=0)):
        active = [queue[order_index] if order_index < len(queue) else None for queue in queues]
        # HIRE and BUY_LAND are atomic orders, handled before per-unit quotes.
        for player, order in enumerate(active):
            if order is None:
                continue
            operation = order[0]
            state = states[player]
            if operation == "HIRE":
                cost = _fib(state["hires"]) * hire_mult
                if cost > state["money"]:
                    if player == candidate_player:
                        return None
                    active[player] = None
                    continue
                state["money"] -= cost
                state["hires"] += 1
                continue
            if operation == "BUY_LAND":
                next_index = len(state["unlocked"]) - 1
                if next_index < 0 or next_index >= len(LAND_ORDER) or state["money"] < LAND_PRICES[next_index]:
                    if player == candidate_player:
                        return None
                    active[player] = None
                    continue
                state["money"] -= LAND_PRICES[next_index]
                state["unlocked"].append(LAND_ORDER[next_index])
                continue

        remaining = [order[2] if order is not None and order[0] not in {"HIRE", "BUY_LAND"} else 0 for order in active]
        while any(remaining):
            quotes = [None, None]
            for player, order in enumerate(active):
                if order is None or remaining[player] <= 0:
                    continue
                operation, item = order[0], order[1]
                if operation == "BUY_SEED":
                    quotes[player] = float(CROPS[item]["seed"])
                elif operation == "BUY_ANIMAL":
                    quotes[player] = float(ANIMALS[item]["cost"])
                elif operation == "BUY_PRODUCT":
                    quotes[player] = _observed_unit_price(
                        item, observations[0], market_inventory.get(item, 0.0), buying=True,
                        configuration=configuration, force_model=force_model,
                    )
                elif operation == "SELL":
                    quotes[player] = _observed_unit_price(
                        item, observations[0], market_inventory.get(item, 0.0), buying=False,
                        configuration=configuration, force_model=force_model,
                    )
            if all(quote is None for quote in quotes):
                break
            committed = False
            for player, order in enumerate(active):
                if order is None or remaining[player] <= 0 or quotes[player] is None:
                    continue
                operation, item, price = order[0], order[1], quotes[player]
                state = states[player]
                shed_total = sum(state["shed"].values())
                if operation == "SELL":
                    if state["shed"].get(item, 0.0) < 1:
                        if player == candidate_player:
                            return None
                        active[player] = None
                        continue
                    state["shed"][item] -= 1
                    state["money"] += price
                    if price > PRICE_FLOOR:
                        market_inventory[item] = market_inventory.get(item, 0.0) + 1
                    else:
                        floor_sales[player] += 1
                elif operation == "BUY_PRODUCT":
                    if market_inventory.get(item, 0.0) < 1 or state["money"] < price or shed_total >= capacity:
                        if player == candidate_player:
                            return None
                        active[player] = None
                        continue
                    state["money"] -= price
                    state["shed"][item] = state["shed"].get(item, 0.0) + 1
                    market_inventory[item] -= 1
                elif operation == "BUY_SEED":
                    if state["money"] < price:
                        if player == candidate_player:
                            return None
                        active[player] = None
                        continue
                    state["money"] -= price
                    state["seeds"][item] = state["seeds"].get(item, 0.0) + 1
                elif operation == "BUY_ANIMAL":
                    if state["money"] < price or shed_total >= capacity:
                        if player == candidate_player:
                            return None
                        active[player] = None
                        continue
                    state["money"] -= price
                    state["shed"][item] = state["shed"].get(item, 0.0) + 1
                remaining[player] -= 1
                committed = True
            if not committed:
                break
    return {"states": states, "market_inventory": market_inventory, "floor_sales": floor_sales}


def _valid_market_orders_lockstep(orders_by_player: Sequence[Any], observations: Sequence[Mapping[str, Any]],
                                  configuration: Mapping[str, Any] | None = None) -> bool:
    other_orders = orders_by_player[1] if len(orders_by_player) > 1 else ()
    return _simulate_market_orders_lockstep(
        orders_by_player, observations, configuration, force_model=bool(other_orders),
        candidate_player=0,
    ) is not None


def _legal_unit_command(command: Any) -> bool:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command or not isinstance(command[0], str):
        return False
    operation = command[0]
    if operation in {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE"}:
        return len(command) == 1
    if operation == "PLANT":
        return len(command) == 2 and isinstance(command[1], str) and command[1] in CROPS
    if operation in {"PICKUP", "PLACE"}:
        return (len(command) in {2, 3} and isinstance(command[1], str) and command[1] in _ITEM_NAMES
                and (len(command) == 2 or (isinstance(command[2], int) and not isinstance(command[2], bool) and command[2] > 0)))
    return False


def _worker_position(observation: Mapping[str, Any], worker_index: int) -> tuple[int, int] | None:
    farm = _farm_observation(observation)
    positions = [farm.get("farmer"), *list(farm.get("hands", ()) or ())]
    if not 0 <= worker_index < len(positions):
        return None
    position = positions[worker_index]
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 2:
        return None
    if any(type(coordinate) is not int for coordinate in position):
        return None
    try:
        return int(position[0]), int(position[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _shed_access_positions(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _default_spawn_position(board_size: int) -> tuple[int, int]:
    """Mirror the engine's deterministic farmer reset position."""
    half = board_size // 2
    for position in _shed_access_positions(board_size):
        if position[0] >= 0 and position[1] >= 0 and position[0] < half and position[1] < half:
            return position
    return (0, 0)


def _spawn_hand_position(farm: Mapping[str, Any], board_size: int,
                         existing_hands: Sequence[Sequence[int]] | None = None) -> tuple[int, int]:
    """Mirror the engine's first-free, least-occupied shed-access spawn rule."""
    access = _shed_access_positions(board_size)
    occupants = {position: 0 for position in access}
    farmer = farm.get("farmer")
    positions = [farmer, *(existing_hands if existing_hands is not None else farm.get("hands", ()))]
    for raw_position in positions:
        if isinstance(raw_position, Sequence) and not isinstance(raw_position, (str, bytes)) and len(raw_position) == 2:
            try:
                position = (int(raw_position[0]), int(raw_position[1]))
            except (TypeError, ValueError, OverflowError):
                continue
            if position in occupants:
                occupants[position] += 1
    return min(access, key=lambda position: (occupants[position], access.index(position)))


def _board_value(observation: Mapping[str, Any], position: tuple[int, int]) -> Any:
    return _tile_at_position(observation, position)


def _daily_refresh_tile(before: Any, day: int, step: int, turns_per_day: int) -> Any:
    """Model the deterministic part of the engine's end-of-day tile refresh.

    Weed spawning is seeded elsewhere and is therefore represented separately as
    the only allowed ``None -> WEED`` transition.  All mutations to an existing
    plant or animal are calculated from the pre-action tile, matching the engine's
    refresh order (decay, then daily refresh).
    """
    if before is None:
        return None
    if before == "LOCKED" or _tile_kind(before) == "WEED":
        return before
    if not isinstance(before, Mapping):
        return before

    expected = dict(before)
    kind = _tile_kind(before)
    if kind == "PLANT":
        crop = before.get("crop")
        crop_data = _crop_data(crop)
        yield_units = _number(before.get("yield_units"))
        lifespan = _number(before.get("max_lifespan_step"))
        if crop_data is None or yield_units is None or lifespan is None:
            return object()
        # _decay_plants runs immediately before the day boundary refresh.
        if lifespan >= 0 and step >= lifespan and int(step - lifespan) % 2 == 0:
            yield_units -= 1
            if yield_units <= 0:
                return {"kind": "WEED"}
            expected["yield_units"] = yield_units

        watered = before.get("watered_today")
        consecutive = _number(before.get("consecutive_unwatered"))
        if not isinstance(watered, bool) or consecutive is None or int(consecutive) != consecutive:
            return object()
        consecutive = 0 if watered else int(consecutive) + 1
        expected["consecutive_unwatered"] = consecutive
        expected["watered_today"] = False
        if consecutive >= 2:
            return {"kind": "WEED"}
        if not crop_data["ongoing"]:
            return expected
        planted_day = _number(before.get("planted_day"))
        if planted_day is None or int(planted_day) != planted_day:
            return object()
        next_day = day + 1
        days_since_first = next_day - int(planted_day) - int(crop_data["first_yield_day"])
        interval = int(crop_data["interval"])
        if days_since_first >= 0 and days_since_first % interval == 0:
            production_count = days_since_first // interval + 1
            if production_count <= int(crop_data["max_yield"]):
                fertilized_until = _number(before.get("fertilized_until_day", -1))
                if fertilized_until is None:
                    return object()
                bonus = 2 if watered and fertilized_until >= day else 1
                expected["yield_units"] = min(int(crop_data["max_yield"]), int(yield_units) + bonus)
                if production_count == int(crop_data["max_yield"]):
                    expected["max_lifespan_step"] = (next_day + 1) * turns_per_day
        return expected

    if kind in {"COOP", "PASTURE"} and "animal" in before:
        animal = before.get("animal")
        animal_data = ANIMALS.get(animal)
        fed = before.get("fed_today")
        cared = before.get("cared_today")
        consecutive = _number(before.get("consecutive_unfed"))
        placed_day = _number(before.get("placed_day"))
        yield_units = _number(before.get("yield_units"))
        if (animal_data is None or not isinstance(fed, bool) or not isinstance(cared, bool)
                or consecutive is None or int(consecutive) != consecutive or placed_day is None
                or int(placed_day) != placed_day or yield_units is None):
            return object()
        consecutive = 0 if fed else int(consecutive) + 1
        if consecutive >= 2:
            # The engine removes only the animal and retains its structure after
            # two consecutive unfed refreshes.
            return {"kind": animal_data["structure"]}
        expected["consecutive_unfed"] = consecutive
        next_day = day + 1
        days_since_first = next_day - int(placed_day) - int(animal_data["first_yield_day"])
        if days_since_first >= 0 and days_since_first % int(animal_data["interval"]) == 0:
            pending = _number(before.get("pending_care_bonus", 0))
            if pending is None:
                return object()
            bonus = int(pending) if fed else 0
            expected["yield_units"] = min(int(animal_data["max_held"]), int(yield_units) + 1 + bonus)
            expected["pending_care_bonus"] = 0
        if cared and fed:
            pending = _number(expected.get("pending_care_bonus", 0))
            if pending is None:
                return object()
            expected["pending_care_bonus"] = int(pending) + 1
        expected["fertilizer_available"] = True
        expected["fed_today"] = False
        expected["cared_today"] = False
        return expected
    return before


def _end_of_day_tile_compatible(before: Any, after: Any, *, day: int = 0, step: int = 0,
                                turns_per_day: int = 24) -> bool:
    """Accept only board changes produced by the engine's daily refresh."""
    expected = _daily_refresh_tile(before, day, step, turns_per_day)
    if before is None and _tile_kind(after) == "WEED":
        return True  # deterministic engine rule with seed-dependent occurrence
    return after == expected


def _targeted_boundary_tile_expected(before: Any, operation: str, day: int, step: int,
                                    turns_per_day: int) -> Any:
    """Return a targeted tile after its action and the following refresh."""
    if not isinstance(before, Mapping):
        return object()
    expected = dict(before)
    if operation == "WATER":
        if _tile_kind(before) != "PLANT" or before.get("watered_today") is not False:
            return object()
        crop = before.get("crop")
        crop_data = _crop_data(crop)
        planted_day = _number(before.get("planted_day"))
        yield_units = _number(before.get("yield_units"))
        fertilized_until = _number(before.get("fertilized_until_day"))
        if (crop_data is None or planted_day is None or int(planted_day) != planted_day
                or yield_units is None or fertilized_until is None):
            return object()
        expected["watered_today"] = True
        if not crop_data["ongoing"]:
            age_days = day - int(planted_day)
            window_start = (int(crop_data["max_yield_day"]) + 1) // 2
            if window_start <= age_days <= int(crop_data["max_yield_day"]):
                bonus = 2 if fertilized_until >= day else 1
                expected["yield_units"] = min(int(crop_data["max_yield"]), int(yield_units) + bonus)
    elif operation == "FERTILIZE":
        if _tile_kind(before) != "PLANT":
            return object()
        current = _number(before.get("fertilized_until_day", -1))
        if current is None:
            return object()
        expected["fertilized_until_day"] = max(current, day + 2)
    elif operation == "FEED":
        if not _animal_state(before) or before.get("fed_today") is not False:
            return object()
        expected["fed_today"] = True
    elif operation == "CARE":
        if not _animal_state(before) or before.get("cared_today") is not False:
            return object()
        expected["cared_today"] = True
    else:
        return object()
    return _daily_refresh_tile(expected, day, step, turns_per_day)


def _is_end_of_day_transition(pre: Mapping[str, Any], post: Mapping[str, Any],
                              configuration: Mapping[str, Any] | None) -> bool:
    turns_per_day = _number(_config_value(configuration, "turnsPerDay", 24))
    pre_step = _number(pre.get("step"))
    post_step = _number(post.get("step"))
    pre_day = _number(pre.get("day"))
    post_day = _number(post.get("day"))
    if turns_per_day is None or int(turns_per_day) < 1:
        return False
    if _number(pre.get("hour")) != int(turns_per_day) - 1 or _number(post.get("hour")) != 0:
        return False
    if pre_step is None or post_step is None:
        # Keep small direct unit fixtures useful; real replay observations always
        # carry step for player 0. Player 1's engine observation omits step, so
        # its day increment is the strict boundary signal instead.
        if pre_step is None and post_step is None:
            return pre_day is None and post_day is None or (
                pre_day is not None and post_day == pre_day + 1
            )
        return False
    if post_step != pre_step + 1 or int(pre_step) % int(turns_per_day) != int(turns_per_day) - 1:
        return False
    return pre_day is None or (post_day is not None and post_day == pre_day + 1)


def _action_target_tile_expected(before: Any, command: Sequence[Any], pre: Mapping[str, Any],
                                post: Mapping[str, Any], configuration: Mapping[str, Any] | None,
                                *, apply_boundary_refresh: bool | None = None) -> tuple[bool, Any]:
    """Return the exact board tile an action may produce at its target.

    Targeted tiles cannot be treated as an unrestricted mutation escape hatch:
    the engine changes a small, deterministic set of fields for each unit
    operation.  This helper mirrors those changes, including the end-of-day
    refresh that follows the action.
    """
    if not command or not isinstance(command[0], str):
        return False, None
    operation = command[0]
    boundary = _is_end_of_day_transition(pre, post, configuration)
    refresh = boundary if apply_boundary_refresh is None else apply_boundary_refresh
    day = int(_number(pre.get("day")) or 0)
    step = _observation_step(pre, configuration)
    turns_per_day = int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24)

    if operation == "PLANT":
        if (before is not None or len(command) != 2 or not isinstance(command[1], str)
                or command[1] not in CROPS):
            return False, None
        crop = command[1]
        crop_data = CROPS[crop]
        expected = {
            "kind": "PLANT", "crop": crop, "planted_day": day,
            "watered_today": False, "consecutive_unwatered": 1,
            "yield_units": 0 if crop_data["ongoing"] else 1,
            "max_lifespan_step": -1 if crop_data["ongoing"] else (day + crop_data["max_yield_day"] + 1) * turns_per_day,
            "fertilized_until_day": -1,
        }
        return True, _daily_refresh_tile(expected, day, step, turns_per_day) if refresh else expected

    if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
        expected = {"kind": operation.removeprefix("BUILD_")}
        return (before is None), expected

    if operation == "DIG":
        return before is not None and not (isinstance(before, Mapping) and "animal" in before), None

    if operation == "PLACE":
        if len(command) < 2 or not isinstance(command[1], str):
            return False, None
        item = command[1]
        if item not in ANIMALS:
            return True, before
        if not isinstance(before, Mapping) or before.get("kind") != ANIMALS[item]["structure"] or "animal" in before:
            return False, None
        expected = {
            "kind": ANIMALS[item]["structure"], "animal": item,
            "placed_day": day, "yield_units": 0, "consecutive_unfed": 0,
            "fed_today": False, "cared_today": False,
            "fertilizer_available": False, "pending_care_bonus": 0,
        }
        return True, _daily_refresh_tile(expected, day, step, turns_per_day) if refresh else expected

    if operation in {"WATER", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"}:
        if not isinstance(before, Mapping):
            return False, None
        expected = dict(before)
        if operation == "WATER":
            if _tile_kind(before) != "PLANT" or before.get("watered_today") is not False:
                return False, None
            crop = before.get("crop")
            crop_data = _crop_data(crop)
            age = _number(before.get("planted_day"))
            yield_units = _number(before.get("yield_units"))
            if crop_data is None or age is None or yield_units is None:
                return False, None
            expected["watered_today"] = True
            if not crop_data["ongoing"]:
                age_days = day - int(age)
                window_start = (int(crop_data["max_yield_day"]) + 1) // 2
                if window_start <= age_days <= int(crop_data["max_yield_day"]):
                    bonus = 2 if (_number(before.get("fertilized_until_day")) or -1) >= day else 1
                    expected["yield_units"] = min(int(crop_data["max_yield"]), int(yield_units) + bonus)
        elif operation == "FEED":
            if not _animal_state(before) or before.get("fed_today") is not False:
                return False, None
            expected["fed_today"] = True
        elif operation == "CARE":
            if not _animal_state(before) or before.get("cared_today") is not False:
                return False, None
            expected["cared_today"] = True
        elif operation == "FERTILIZE":
            if _tile_kind(before) != "PLANT":
                return False, None
            current = _number(before.get("fertilized_until_day", -1))
            if current is None:
                return False, None
            expected["fertilized_until_day"] = max(current, day + 2)
        else:
            animal = _animal_state(before)
            if not animal or before.get("fertilizer_available") is not True:
                return False, None
            expected["fertilizer_available"] = False
        return True, _daily_refresh_tile(expected, day, step, turns_per_day) if refresh else expected

    if operation == "HARVEST":
        if not isinstance(before, Mapping) or (_number(before.get("yield_units")) or 0) <= 0:
            return False, None
        if _tile_kind(before) == "PLANT":
            crop_data = _crop_data(before.get("crop"))
            if crop_data is None:
                return False, None
            expected = None if not crop_data["ongoing"] else {**before, "yield_units": 0}
        elif _animal_state(before):
            expected = {**before, "yield_units": 0}
        else:
            return False, None
        if refresh and expected is not None:
            expected = _daily_refresh_tile(expected, day, step, turns_per_day)
        return True, expected

    return False, None


def _action_target_tile_sequence_expected(before: Any, commands: Sequence[Sequence[Any]],
                                          pre: Mapping[str, Any], post: Mapping[str, Any],
                                          configuration: Mapping[str, Any] | None) -> tuple[bool, Any]:
    """Apply same-tile commands in farmer-then-hand engine order, then refresh once."""
    current = before
    applied = False
    for command in commands:
        valid, expected = _action_target_tile_expected(
            current, command, pre, post, configuration, apply_boundary_refresh=False,
        )
        if valid:
            current = expected
            applied = True
        # The engine silently no-ops a later command whose precondition was
        # consumed by an earlier command on the same tile (for example a
        # second WATER after the first WATER, or WATER after HARVEST).
    if not applied:
        return False, None
    day = int(_number(pre.get("day")) or 0)
    step = _observation_step(pre, configuration)
    turns_per_day = int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24)
    if _is_end_of_day_transition(pre, post, configuration):
        current = _daily_refresh_tile(current, day, step, turns_per_day)
    elif isinstance(current, Mapping) and _tile_kind(current) == "PLANT":
        lifespan = _number(current.get("max_lifespan_step"))
        if lifespan is not None and lifespan >= 0 and step >= lifespan and int(step - lifespan) % 2 == 0:
            current = dict(current)
            current["yield_units"] = (_number(current.get("yield_units")) or 0) - 1
            if current["yield_units"] <= 0:
                current = {"kind": "WEED"}
    return True, current


def _tile_state_matches_expected(actual: Any, expected: Any, *, allow_compact: bool = False) -> bool:
    """Compare a replay tile without allowing unmodeled fields.

    A few unit fixtures intentionally use a compact tile representation.  A
    real engine replay normally contains the complete mapping, but accepting
    a compact *subset* keeps those fixtures useful while still rejecting any
    added field or changed value that is outside the modeled transition.
    """
    if actual == expected:
        return True
    if not allow_compact:
        return False
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    return set(actual).issubset(expected) and all(actual[key] == expected[key] for key in actual)


def _compact_plant_action_matches(actual: Any, command: Sequence[Any], pre: Mapping[str, Any]) -> bool:
    """Support legacy compact unit fixtures without opening a mutation escape hatch."""
    if not command or command[0] != "PLANT":
        return False
    if _number(pre.get("day")) is not None:
        return False
    if not isinstance(actual, Mapping) or set(actual) - {"kind", "crop", "watered_today", "yield_units"}:
        return False
    return (
        actual.get("kind") == "PLANT"
        and len(command) == 2
        and actual.get("crop") == command[1]
        and actual.get("watered_today") is False
        and _number(actual.get("yield_units")) in {0, 1}
    )


def _midday_board_changes_valid(pre: Mapping[str, Any], post: Mapping[str, Any], action: Mapping[str, Any],
                                configuration: Mapping[str, Any] | None, market_result: Mapping[str, Any],
                                *, allow_compact: bool | None = None) -> bool:
    """Reject board changes not attributable to this turn's unit/land actions."""
    if allow_compact is None:
        allow_compact = (
            "boardSize" not in _mapping(configuration)
            or ("day" not in pre and "day" not in post)
        )
    if not allow_compact and (
        not _board_dimensions_valid(pre, configuration)
        or not _board_dimensions_valid(post, configuration)
    ):
        return False
    pre_farm = _farm_observation(pre)
    post_farm = _farm_observation(post)
    pre_tiles = pre_farm.get("tiles")
    post_tiles = post_farm.get("tiles")
    if not isinstance(pre_tiles, Sequence) or isinstance(pre_tiles, (str, bytes)):
        return False
    if not isinstance(post_tiles, Sequence) or isinstance(post_tiles, (str, bytes)):
        return False
    size = _board_size(pre, configuration)
    commands = [action.get("farmer"), *list(action.get("hands", ()))]
    blocked_plant_crops = _blocked_plant_crops(action, pre)
    tile_operations = {"PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED", "CARE", "COLLECT_FERTILIZER",
                       "DIG", "BUILD_COOP", "BUILD_PASTURE", "PLACE"}
    target_commands: dict[tuple[int, int], list[Sequence[Any]]] = {}
    for index, command in enumerate(commands):
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes)) and command and command[0] in tile_operations:
            if command[0] == "PLANT" and command[1] in blocked_plant_crops:
                continue
            position = _worker_position(pre, index)
            if position is not None:
                target_commands.setdefault(position, []).append(command)

    pre_unlocked = pre_farm.get("unlocked_quadrants", ())
    post_unlocked = post_farm.get("unlocked_quadrants", ())
    if not isinstance(pre_unlocked, Sequence) or isinstance(pre_unlocked, (str, bytes)):
        return False
    if not isinstance(post_unlocked, Sequence) or isinstance(post_unlocked, (str, bytes)):
        return False
    try:
        newly_unlocked = set(post_unlocked) - set(pre_unlocked)
    except TypeError:
        return False
    for y in range(max(len(pre_tiles), len(post_tiles))):
        for x in range(size):
            before = _board_value(pre, (x, y))
            after = _board_value(post, (x, y))
            if before == after:
                continue
            if (x, y) in target_commands:
                valid, expected = _action_target_tile_sequence_expected(
                    before, target_commands[(x, y)], pre, post, configuration,
                )
                if valid and (
                    _tile_state_matches_expected(after, expected, allow_compact=allow_compact)
                    or (allow_compact and any(
                        _compact_plant_action_matches(after, command, pre)
                        for command in target_commands[(x, y)]
                    ))
                ):
                    continue
                # A mature finite crop is removed by HARVEST before the
                # seeded end-of-day weed spawn.  The action expectation is
                # therefore None, while the real engine may record WEED.
                if valid and _is_end_of_day_transition(pre, post, configuration) and expected is None and any(
                    isinstance(command, Sequence) and not isinstance(command, (str, bytes))
                    and command and command[0] == "HARVEST"
                    for command in target_commands[(x, y)]
                ) and _tile_kind(after) == "WEED":
                    continue
                if not valid and _is_end_of_day_transition(pre, post, configuration) and _end_of_day_tile_compatible(
                    before, after, day=int(_number(pre.get("day")) or 0),
                    step=_observation_step(pre, configuration),
                    turns_per_day=int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24),
                ):
                    continue
                return False
            quadrant = ("N" if y < size // 2 else "S") + ("W" if x < size // 2 else "E")
            if (
                quadrant in newly_unlocked
                and before == "LOCKED"
                and (after is None or (_tile_kind(after) == "WEED" and isinstance(after, Mapping)))
            ):
                continue
            if _is_end_of_day_transition(pre, post, configuration) and _end_of_day_tile_compatible(
                    before, after, day=int(_number(pre.get("day")) or 0),
                    step=_observation_step(pre, configuration),
                    turns_per_day=int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24)):
                continue
            # The engine can decay a plant at its exact lifespan boundary after
            # actions.  This is deterministic and limited to yield decrement or
            # the resulting weed, so unrelated arbitrary mutations remain errors.
            if isinstance(before, Mapping) and _tile_kind(before) == "PLANT":
                lifespan = _number(before.get("max_lifespan_step"))
                step = _observation_step(pre, configuration)
                if lifespan is not None and lifespan >= 0 and step is not None and step >= lifespan and int(step - lifespan) % 2 == 0:
                    expected = dict(before)
                    expected["yield_units"] = (_number(before.get("yield_units")) or 0) - 1
                    if expected["yield_units"] <= 0:
                        expected = {"kind": "WEED"}
                    if after == expected:
                        continue
            return False
    return True


def _post_market_effects_valid(pre: Mapping[str, Any], post: Mapping[str, Any],
                               market_result: Mapping[str, Any], configuration: Mapping[str, Any] | None) -> bool:
    """Validate the shared market inventory after orders and deterministic town demand."""
    town = pre.get("town")
    if town is not None and not isinstance(town, Mapping):
        return False
    expected = market_result.get("market_inventory")
    if not isinstance(expected, Mapping):
        return False
    expected = {item: _number(quantity) for item, quantity in expected.items()}
    if any(quantity is None for quantity in expected.values()):
        return False
    shops = town.get("unlocked_shops", ()) if isinstance(town, Mapping) else ()
    if not isinstance(shops, Sequence) or isinstance(shops, (str, bytes)):
        return False
    if isinstance(town, Mapping):
        step = _number(pre.get("step"))
        if step is None or int(step) != step:
            return False
        try:
            shop_interval = int(_config_value(configuration, "townShopSellInterval", 4))
            center_interval = int(_config_value(configuration, "townCenterSellInterval", 24))
        except (TypeError, ValueError, OverflowError):
            return False
        if shop_interval < 1 or center_interval < 1:
            return False
        if int(step) % shop_interval == 0:
            for shop in shops:
                if shop not in SHOPS:
                    return False
                multiplier = 2 if len(SHOPS[shop]) == 1 else 1
                for item in SHOPS[shop]:
                    expected[item] = expected.get(item, 0) - multiplier
        if int(step) % center_interval == 0:
            for item in PRODUCTS:
                if item != "FERTILIZER":
                    expected[item] = expected.get(item, 0) - 1
    post_market = _mapping(post.get("market"))
    post_inventory = post_market.get("inventory")
    if not isinstance(post_inventory, Mapping) or set(post_inventory) != set(expected):
        return False
    if any(_number(post_inventory[item]) != quantity for item, quantity in expected.items()):
        return False
    prices = post_market.get("prices")
    if isinstance(town, Mapping) and (not isinstance(prices, Mapping) or set(prices) != set(expected)):
        return False
    if isinstance(town, Mapping) and isinstance(prices, Mapping):
        params = _mapping(_config_value(configuration, "marketParams", {}))
        if any(_number(prices[item]) != market_price(item, quantity, params) for item, quantity in expected.items()):
            return False
    return True


def _board_size(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None = None) -> int:
    configured = _config_value(configuration, "boardSize", None)
    if configured is not None:
        try:
            return max(1, int(configured))
        except (TypeError, ValueError, OverflowError):
            pass
    tiles = _mapping(_farm_observation(observation)).get("tiles", ())
    return max(1, len(tiles) if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) else 1)


def _board_dimensions_valid(observation: Mapping[str, Any], configuration: Mapping[str, Any] | None) -> bool:
    farm = _farm_observation(observation)
    tiles = farm.get("tiles")
    configured_size = _config_value(configuration, "boardSize", None)
    size = _number(configured_size) if configured_size is not None else (
        float(len(tiles)) if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)) else None
    )
    if size is None or int(size) != size or int(size) < 1:
        return False
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)) or len(tiles) != int(size):
        return False
    return all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == int(size)
        for row in tiles
    )


def _tile_at_worker(observation: Mapping[str, Any], worker_index: int) -> Any:
    position = _worker_position(observation, worker_index)
    tiles = _mapping(_farm_observation(observation)).get("tiles", ())
    if position is None or not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
        return None
    x, y = position
    if not 0 <= y < len(tiles) or not isinstance(tiles[y], Sequence) or isinstance(tiles[y], (str, bytes)) or not 0 <= x < len(tiles[y]):
        return None
    return tiles[y][x]


def _worker_inventory(observation: Mapping[str, Any], worker_index: int) -> Mapping[str, Any]:
    inventories = _mapping(observation.get("private")).get("inventories", ())
    if isinstance(inventories, Sequence) and not isinstance(inventories, (str, bytes)) and 0 <= worker_index < len(inventories):
        return _mapping(inventories[worker_index])
    return {}


def _unit_command_valid_for_state(command: Any, observation: Mapping[str, Any], worker_index: int = 0,
                                  configuration: Mapping[str, Any] | None = None) -> bool:
    if not _legal_unit_command(command):
        return False
    operation = command[0]
    position = _worker_position(observation, worker_index)
    if position is None:
        return False
    if operation in {"PASS", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG", "BUILD_COOP", "BUILD_PASTURE"}:
        tile = _tile_at_worker(observation, worker_index)
    else:
        tile = None
    if operation in {"NORTH", "SOUTH", "EAST", "WEST"}:
        deltas = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
        dx, dy = deltas[operation]
        size = _board_size(observation, configuration)
        return 0 <= position[0] + dx < size and 0 <= position[1] + dy < size
    if operation == "PASS":
        return True
    if operation == "PLANT":
        seeds = _private_seeds(observation)
        return tile is None and _number(seeds.get(command[1])) is not None and (_number(seeds.get(command[1])) or 0) >= 1
    if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
        return tile is None
    if operation == "WATER":
        return _tile_kind(tile) == "PLANT" and tile.get("watered_today") is False if isinstance(tile, Mapping) else False
    if operation == "HARVEST":
        if not isinstance(tile, Mapping) or _tile_kind(tile) == "LOCKED" or (_number(tile.get("yield_units")) or 0) <= 0:
            return False
        day = _number(observation.get("day"))
        if day is None:
            return True
        if _tile_kind(tile) == "PLANT":
            crop = _crop_data(tile.get("crop"))
            planted_day = _number(tile.get("planted_day"))
            return crop is not None and planted_day is not None and day - planted_day >= crop["first_yield_day"]
        animal = _animal_state(tile)
        animal_data = ANIMALS.get(animal.get("animal"))
        placed_day = _number(animal.get("placed_day"))
        return animal_data is not None and placed_day is not None and day - placed_day >= animal_data["first_yield_day"]
    if operation == "DIG":
        return tile is not None and _tile_kind(tile) != "LOCKED" and not (isinstance(tile, Mapping) and "animal" in tile)
    if operation == "FERTILIZE":
        return _tile_kind(tile) == "PLANT" and (_number(_worker_inventory(observation, worker_index).get("FERTILIZER")) or 0) >= 1
    if operation in {"FEED", "CARE", "COLLECT_FERTILIZER"}:
        animal_state = _animal_state(tile)
        if not animal_state:
            return False
        if operation == "FEED":
            return (animal_state.get("fed_today") is False or animal_state.get("needs_feed") is True) and (_number(_worker_inventory(observation, worker_index).get("WHEAT")) or 0) >= 1
        if operation == "CARE":
            return animal_state.get("cared_today") is False or animal_state.get("needs_care") is True
        return animal_state.get("fertilizer_available") is True
    if operation in {"PICKUP", "PLACE", "DROP"}:
        size = _board_size(observation, configuration)
        if not is_shed_adjacent(position, size):
            if operation == "PLACE" and command[1] in _ANIMAL_NAMES:
                pass
            else:
                return False
        inventory = _worker_inventory(observation, worker_index)
        shed = _mapping(_mapping(observation.get("private")).get("shed"))
        if operation == "DROP":
            return any((_number(value) or 0) > 0 for value in inventory.values())
        item = command[1]
        quantity = command[2] if len(command) == 3 else 1
        if operation == "PICKUP":
            return (_number(shed.get(item)) or 0) >= quantity
        if item in _ANIMAL_NAMES:
            return isinstance(tile, Mapping) and tile.get("kind") == ANIMALS[item]["structure"] and tile.get("animal") is None and (_number(inventory.get(item)) or 0) >= 1
        room = max(0, int(_config_value(configuration, "shedCapacity", shed_capacity)) - int(sum((_number(value) or 0) for value in shed.values())))
        return is_shed_adjacent(position, size) and (_number(inventory.get(item)) or 0) >= quantity and room >= quantity
    return False


def _tile_at_position(observation: Mapping[str, Any], position: tuple[int, int]) -> Any:
    farm = _farm_observation(observation)
    tiles = farm.get("tiles", ())
    x, y = position
    if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)) or not 0 <= y < len(tiles):
        return None
    row = tiles[y]
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not 0 <= x < len(row):
        return None
    return row[x]


def _positive_quantities(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(item): quantity
        for item, raw_quantity in value.items()
        if (quantity := _number(raw_quantity)) is not None and quantity > 0
    }


def _private_inventories(observation: Mapping[str, Any]) -> list[dict[str, float]] | None:
    raw = _mapping(observation.get("private")).get("inventories")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [_positive_quantities(inventory) for inventory in raw]


def _strict_quantity_mapping(value: Any, allowed_items: set[str] | frozenset[str]) -> bool:
    if not isinstance(value, Mapping):
        return False
    for item, raw_quantity in value.items():
        if (not isinstance(item, str) or item not in allowed_items
                or not _strict_numeric_scalar(raw_quantity)
                or raw_quantity < 0 or int(raw_quantity) != raw_quantity):
            return False
    return True


def _strict_private_inventories(observation: Mapping[str, Any]) -> bool:
    """Require the engine's complete, finite worker-inventory snapshot."""
    private = observation.get("private")
    if not isinstance(private, Mapping) or not _strict_quantity_mapping(private.get("shed", {}), _ITEM_NAMES):
        return False
    if not _strict_quantity_mapping(private.get("seeds", {}), frozenset(CROPS)):
        return False
    raw = private.get("inventories")
    farm = _farm_observation(observation)
    hands = farm.get("hands", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return False
    if not isinstance(hands, Sequence) or isinstance(hands, (str, bytes)) or len(raw) != len(hands) + 1:
        return False
    return all(_strict_quantity_mapping(inventory, _ITEM_NAMES) for inventory in raw)


def _blocked_plant_crops(action: Mapping[str, Any], observation: Mapping[str, Any]) -> set[str]:
    """Return crops whose same-turn plant requests exceed available seeds."""
    commands = [action.get("farmer"), *list(action.get("hands", ()))]
    requests = Counter(
        command[1]
        for command in commands
        if isinstance(command, Sequence)
        and not isinstance(command, (str, bytes))
        and len(command) > 1
        and command[0] == "PLANT"
        and isinstance(command[1], str)
    )
    seeds = _private_seeds(observation)
    return {
        crop for crop, count in requests.items()
        if count > (_number(seeds.get(crop)) or 0)
    }


def _transition_effects_valid(pre: Mapping[str, Any], post: Mapping[str, Any], action: Mapping[str, Any],
                             configuration: Mapping[str, Any] | None = None,
                             market_result: Mapping[str, Any] | None = None,
                             *, player_index: int = 0, allow_compact: bool | None = None,
                             market_observation: Mapping[str, Any] | None = None,
                             allow_invalid_unit_noop: bool = False) -> bool:
    """Check deterministic effects of one recorded player action."""
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping) or market_result is None:
        return False
    if player_index not in (0, 1):
        return False
    if allow_compact is None:
        allow_compact = (
            "boardSize" not in _mapping(configuration)
            or ("day" not in pre and "day" not in post)
        )
    pre_farm = _farm_observation(pre)
    post_farm = _farm_observation(post)
    pre_hands = pre_farm.get("hands", ())
    post_hands = post_farm.get("hands", ())
    if not isinstance(pre_hands, Sequence) or isinstance(pre_hands, (str, bytes)):
        return False
    if not isinstance(post_hands, Sequence) or isinstance(post_hands, (str, bytes)):
        return False
    boundary = _is_end_of_day_transition(pre, post, configuration)
    simulated_states = market_result.get("states", ())
    simulated = (
        simulated_states[player_index]
        if isinstance(simulated_states, Sequence) and len(simulated_states) > player_index
        else {}
    )
    if not isinstance(simulated, Mapping):
        return False
    pre_hires_value = _number(pre_farm.get("hires_today"))
    if pre_hires_value is None:
        if not allow_compact:
            return False
        pre_hires_value = 0
    if int(pre_hires_value) != pre_hires_value or pre_hires_value < 0:
        return False
    pre_hires = int(pre_hires_value)
    total_hires = int(_number(simulated.get("hires")) or 0)
    hire_count = max(0, total_hires - pre_hires)
    expected_hands = 0 if boundary else len(pre_hands) + hire_count
    if len(post_hands) != expected_hands:
        return False
    post_hires_value = _number(post_farm.get("hires_today"))
    expected_hires = 0 if boundary else total_hires
    if post_hires_value is None:
        if not allow_compact:
            return False
    elif int(post_hires_value) != post_hires_value or int(post_hires_value) != expected_hires:
        return False
    if boundary:
        reset_position = _default_spawn_position(_board_size(pre, configuration))
        tiles = pre_farm.get("tiles")
        if (isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes))
                and len(tiles) < reset_position[1] + 1):
            reset_position = _default_spawn_position(len(tiles))
        if _worker_position(post, 0) != reset_position:
            return False
    commands = [action.get("farmer"), *list(action.get("hands", ()))]
    inventories = _private_inventories(pre)
    post_inventories = _private_inventories(post)
    if boundary and (inventories is None or post_inventories is None):
        return False
    if inventories is not None and post_inventories is not None:
        if len(inventories) < len(commands):
            return False
        expected_inventories = [dict(inventory) for inventory in inventories]
        if not boundary:
            expected_inventories.extend({} for _ in range(hire_count))
        expected_shed = dict(_positive_quantities(simulated.get("shed")))
    else:
        expected_inventories = None
        expected_shed = None
    expected_seeds = dict(_positive_quantities(simulated.get("seeds")))
    blocked_plant_crops = _blocked_plant_crops(action, pre)

    def add_quantity(values: dict[str, float], item: str, amount: float) -> None:
        values[item] = values.get(item, 0.0) + amount
        if values[item] <= 0:
            values.pop(item, None)

    effective_tiles: dict[tuple[int, int], Any] = {}
    for worker_index, command in enumerate(commands):
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
            return False
        state_valid = _unit_command_valid_for_state(command, pre, worker_index, configuration)
        position = _worker_position(pre, worker_index)
        post_position = _worker_position(post, worker_index)
        if position is None:
            return False
        # Farm hands are one-day assets: the engine can remove them while
        # resetting the day, so their post-boundary position is intentionally
        # absent. Their pre-action effects are still validated below.
        if post_position is None and not boundary:
            return False
        operation = command[0]
        if operation in {"NORTH", "SOUTH", "EAST", "WEST"}:
            deltas = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
            dx, dy = deltas[operation]
            expected_position = (position[0] + dx, position[1] + dy) if state_valid else position
            if not boundary and post_position != expected_position:
                return False
        elif not boundary and post_position != position:
            return False

        if allow_invalid_unit_noop and not state_valid:
            continue
        if operation == "PLANT" and command[1] in blocked_plant_crops:
            # The engine atomically drops every same-crop PLANT request when
            # the combined demand exceeds the player's available seeds.
            continue

        pre_tile = effective_tiles[position] if position in effective_tiles else _tile_at_position(pre, position)
        post_tile = _tile_at_position(post, position)
        if operation == "PLANT":
            plant_persisted = (
                isinstance(post_tile, Mapping)
                and _tile_kind(post_tile) == "PLANT"
                and post_tile.get("crop") == command[1]
            )
            end_of_day_decay = boundary and _tile_kind(post_tile) == "WEED"
            if not plant_persisted and not end_of_day_decay:
                return False
            add_quantity(expected_seeds, command[1], -1)
        elif operation == "BUILD_COOP" and (not isinstance(post_tile, Mapping) or _tile_kind(post_tile) != "COOP"):
            return False
        elif operation == "BUILD_PASTURE" and (not isinstance(post_tile, Mapping) or _tile_kind(post_tile) != "PASTURE"):
            return False
        elif operation == "DIG" and post_tile is not None:
            return False
        elif operation in {"WATER", "FEED", "CARE"}:
            if not isinstance(post_tile, Mapping):
                # Unit actions are resolved in one engine turn.  If a farmer
                # harvests a tile before a hand's WATER/FEED/CARE command is
                # resolved, the second command is a legal no-op.  If the
                # harvest follows this command, the first command still
                # mutates the tile before the harvest removes it.  In either
                # order, do not demand a post-tile effect that the engine
                # cannot preserve in this conflict.
                blocked_by_harvest = any(
                    other_index != worker_index
                    and isinstance(other_command, Sequence)
                    and not isinstance(other_command, (str, bytes))
                    and other_command
                    and other_command[0] == "HARVEST"
                    and _worker_position(pre, other_index) == position
                    for other_index, other_command in enumerate(commands)
                )
                if not (post_tile is None and blocked_by_harvest and
                        (isinstance(pre_tile, Mapping) or position in effective_tiles)):
                    return False
            elif operation == "WATER" and _tile_kind(post_tile) == "WEED" and isinstance(pre_tile, Mapping):
                lifespan = _number(pre_tile.get("max_lifespan_step"))
                step = _number(pre.get("step"))
                yield_units = _number(pre_tile.get("yield_units"))
                if (
                    _tile_kind(pre_tile) == "PLANT"
                    and lifespan is not None and step is not None
                    and step >= lifespan and int(step - lifespan) % 2 == 0
                    and yield_units == 1
                ):
                    # The engine applies the action to the pre-state, then
                    # its lifespan decay can replace the last unit with WEED
                    # in the same transition.
                    continue
            if not isinstance(post_tile, Mapping):
                pass
            elif boundary:
                expected_tile = _targeted_boundary_tile_expected(
                    pre_tile, operation, int(_number(pre.get("day")) or 0),
                    _observation_step(pre, configuration),
                    int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24),
                )
                if post_tile != expected_tile:
                    return False
            elif operation == "WATER" and post_tile.get("watered_today") is not True:
                return False
            elif operation == "FEED" and post_tile.get("fed_today") is not True:
                return False
            elif operation == "CARE" and post_tile.get("cared_today") is not True:
                return False
        elif operation == "FERTILIZE":
            harvest_follows = any(
                other_index > worker_index
                and isinstance(other_command, Sequence)
                and not isinstance(other_command, (str, bytes))
                and other_command
                and other_command[0] == "HARVEST"
                and _worker_position(pre, other_index) == position
                for other_index, other_command in enumerate(commands)
            )
            if post_tile is None and harvest_follows:
                pass
            elif boundary:
                # Fertilizing does not suppress the end-of-day refresh.  An
                # unwatered plant can therefore become WEED in the same
                # transition, and the targeted boundary model captures that
                # deterministic action-plus-refresh result.
                expected_tile = _targeted_boundary_tile_expected(
                    pre_tile, operation, int(_number(pre.get("day")) or 0),
                    _observation_step(pre, configuration),
                    int(_number(_config_value(configuration, "turnsPerDay", 24)) or 24),
                )
                if not _tile_state_matches_expected(
                    post_tile, expected_tile, allow_compact=allow_compact,
                ):
                    return False
            elif not isinstance(post_tile, Mapping) or (_number(post_tile.get("fertilized_until_day")) or -1) < (_number(pre.get("day")) or 0) + 2:
                return False
        elif operation == "COLLECT_FERTILIZER" and not boundary:
            if not isinstance(post_tile, Mapping) or post_tile.get("fertilizer_available") is not False:
                return False
        elif operation == "HARVEST":
            ongoing_boundary_harvest = (
                boundary
                and isinstance(pre_tile, Mapping)
                and _tile_kind(pre_tile) == "PLANT"
                and bool((_crop_data(pre_tile.get("crop")) or {}).get("ongoing", False))
            )
            if ongoing_boundary_harvest:
                same_tile_commands = [
                    candidate
                    for candidate_index, candidate in enumerate(commands)
                    if _worker_position(pre, candidate_index) == position
                ]
                valid, expected_tile = _action_target_tile_sequence_expected(
                    pre_tile, same_tile_commands, pre, post, configuration,
                )
                if not valid or not _tile_state_matches_expected(
                    post_tile, expected_tile, allow_compact=allow_compact,
                ):
                    return False
            elif isinstance(post_tile, Mapping) and (_number(post_tile.get("yield_units")) or 0) != 0:
                pre_yield = _number(pre_tile.get("yield_units")) if isinstance(pre_tile, Mapping) else None
                if pre_yield is None:
                    return False
            if post_tile is not None and not isinstance(post_tile, Mapping):
                return False
        elif operation == "PLACE" and command[1] in _ANIMAL_NAMES:
            if not isinstance(post_tile, Mapping) or post_tile.get("animal") != command[1]:
                return False

        if expected_inventories is None or worker_index >= len(expected_inventories):
            continue
        inventory = expected_inventories[worker_index]
        item = command[1] if len(command) > 1 else None
        quantity = float(command[2] if len(command) > 2 else 1) if item is not None else 0.0
        if operation == "PICKUP":
            add_quantity(expected_shed, item, -quantity)
            add_quantity(inventory, item, quantity)
        elif operation == "PLACE" and item not in _ANIMAL_NAMES:
            add_quantity(inventory, item, -quantity)
            add_quantity(expected_shed, item, quantity)
        elif operation == "PLACE" and item in _ANIMAL_NAMES:
            add_quantity(inventory, item, -1)
        elif operation == "FEED":
            add_quantity(inventory, "WHEAT", -1)
        elif operation == "FERTILIZE":
            add_quantity(inventory, "FERTILIZER", -1)
        elif operation == "COLLECT_FERTILIZER":
            add_quantity(inventory, "FERTILIZER", 1)
        elif operation == "HARVEST" and isinstance(pre_tile, Mapping):
            item = pre_tile.get("crop") or _animal_state(pre_tile).get("product")
            if item:
                add_quantity(inventory, item, _number(pre_tile.get("yield_units")) or 0)
        elif operation == "DROP":
            room = max(0.0, _shed_capacity(configuration) - sum(expected_shed.values()))
            for drop_item, drop_quantity in list(inventory.items()):
                taken = min(drop_quantity, room)
                add_quantity(expected_shed, drop_item, taken)
                room -= taken
            inventory.clear()

        tile_valid, effective_tile = _action_target_tile_expected(
            pre_tile, command, pre, post, configuration,
            apply_boundary_refresh=False,
        )
        if tile_valid:
            effective_tiles[position] = effective_tile

    if expected_inventories is not None and boundary:
        for inventory in expected_inventories:
            room = max(0.0, _shed_capacity(configuration) - sum(expected_shed.values()))
            for item, quantity in list(inventory.items()):
                taken = min(max(0.0, quantity), room)
                add_quantity(expected_shed, item, taken)
                room -= taken
                inventory.pop(item, None)
        expected_inventories = [{}]
    if expected_inventories is not None:
        if post_inventories != expected_inventories:
            return False
        if _positive_quantities(_mapping(_mapping(post).get("private")).get("shed")) != _positive_quantities(expected_shed):
            return False
    if _positive_quantities(_mapping(_mapping(post).get("private")).get("seeds")) != expected_seeds:
        return False
    expected_money = _number(simulated.get("money"))
    if expected_money is None or _cash(post) != expected_money:
        return False
    if list(_mapping(post_farm).get("unlocked_quadrants", ())) != list(simulated.get("unlocked", ())):
        return False
    if not boundary and hire_count:
        expected_existing_hands = list(post_hands[:len(pre_hands)])
        spawned = []
        for _ in range(hire_count):
            position = _spawn_hand_position(post_farm, _board_size(pre, configuration),
                                            [*expected_existing_hands, *spawned])
            spawned.append(list(position))
        if post_hands[len(pre_hands):] != spawned:
            return False
    if not _midday_board_changes_valid(
        pre, post, action, configuration, market_result, allow_compact=allow_compact,
    ):
        return False
    if not _post_market_effects_valid(market_observation or pre, post, market_result, configuration):
        return False
    return True


def _sanitize_action(action: Mapping[str, Any], observation: Mapping[str, Any], fallback: Mapping[str, Any],
                     configuration: Mapping[str, Any] | None = None, *,
                     preserve_malformed_market: bool = False) -> dict[str, Any]:
    raw_market = action.get("market", ())
    result = {
        "farmer": list(action.get("farmer", ())) if isinstance(action.get("farmer", ()), Sequence) else [],
        "hands": [list(command) for command in action.get("hands", ())] if isinstance(action.get("hands", ()), Sequence) else [],
        "market": (
            raw_market
            if preserve_malformed_market and not _valid_market_container(raw_market)
            else _sanitize_market_orders(
                _market_orders(action), observation, configuration,
                preserve_malformed=preserve_malformed_market,
            )
        ),
    }
    fallback_farmer = list(fallback.get("farmer", ["PASS"]))
    if not _unit_command_valid_for_state(result["farmer"], observation, 0, configuration):
        result["farmer"] = fallback_farmer if _unit_command_valid_for_state(fallback_farmer, observation, 0, configuration) else ["PASS"]
    expected_hands = _mapping(_farm_observation(observation)).get("hands", ())
    expected_count = len(expected_hands) if isinstance(expected_hands, Sequence) and not isinstance(expected_hands, (str, bytes)) else 0
    fallback_hands = [list(command) for command in fallback.get("hands", ())]
    if len(result["hands"]) != expected_count:
        result["hands"] = fallback_hands if len(fallback_hands) == expected_count else [["PASS"] for _ in range(expected_count)]
    result["hands"] = [command if _unit_command_valid_for_state(command, observation, index + 1, configuration) else (fallback_hands[index] if index < len(fallback_hands) and _unit_command_valid_for_state(fallback_hands[index], observation, index + 1, configuration) else ["PASS"]) for index, command in enumerate(result["hands"])]
    return result


def _best_seed(observation: Mapping[str, Any], *, prefer: str | None = None) -> str | None:
    prices = _mapping(_mapping(observation.get("market")).get("prices"))
    affordable = [
        crop for crop, data in CROPS.items()
        if _cash(observation) >= float(data["seed"]) and (_number(prices.get(crop)) or 0.0) > 0
    ]
    if not affordable:
        return None
    return max(affordable, key=lambda crop: ((1 if crop == prefer else 0), (_number(prices.get(crop)) or 0.0) / CROPS[crop]["seed"], crop))


def _empty_animal_target(observation: Mapping[str, Any]) -> str | None:
    for tile in _tiles(observation):
        if not isinstance(tile, Mapping):
            continue
        kind = _tile_kind(tile)
        if kind not in {"COOP", "PASTURE"} or tile.get("built", True) is False or tile.get("animal") is not None:
            continue
        species = "GOOSE" if kind == "COOP" else "COW"
        if _cash(observation) >= ANIMALS[species]["cost"]:
            return species
    return None


def _animal_units_owned(observation: Mapping[str, Any]) -> int:
    """Count placed and carried animals before applying a portfolio bias."""
    shed = _mapping(_mapping(observation.get("private")).get("shed"))
    carried = sum(
        int(_number(shed.get(species)) or 0)
        for species in _ANIMAL_NAMES
    )
    placed = sum(
        1 for tile in _tiles(observation)
        if _animal_state(tile).get("animal") in _ANIMAL_NAMES
    )
    return carried + placed


def _has_complete_market_state(observation: Mapping[str, Any]) -> bool:
    """Return whether state-aware market capacity checks have full inputs."""
    farm = _farm_observation(observation)
    private = _mapping(observation.get("private"))
    market = _mapping(observation.get("market"))
    return (
        "unlocked_quadrants" in farm
        and isinstance(private.get("shed"), Mapping)
        and isinstance(market.get("inventory"), Mapping)
    )


def _variant_market_spend(order: Sequence[Any], observation: Mapping[str, Any]) -> float:
    """Estimate non-seed spend so a crop bias cannot consume safety cash."""
    if not order:
        return 0.0
    operation = order[0]
    item = order[1] if len(order) > 1 else None
    quantity = max(1, int(_number(order[2]) or 1)) if len(order) > 2 else 1
    if operation == "HIRE":
        farm = _farm_observation(observation)
        return float(_fib(int(_number(farm.get("hires_today")) or 0)))
    if operation == "BUY_LAND":
        farm = _farm_observation(observation)
        unlocked = farm.get("unlocked_quadrants", ())
        index = len(unlocked) - 1 if isinstance(unlocked, Sequence) and not isinstance(unlocked, (str, bytes)) else -1
        return float(LAND_PRICES[index]) if 0 <= index < len(LAND_PRICES) else 0.0
    if operation == "BUY_ANIMAL" and item in ANIMALS:
        return quantity * float(ANIMALS[item]["cost"])
    if operation == "BUY_PRODUCT" and item in _BUYABLE_PRODUCTS:
        market = _mapping(observation.get("market"))
        inventory = _number(_mapping(market.get("inventory")).get(item)) or 0.0
        return sum(_observed_unit_price(item, observation, inventory - offset, buying=True) for offset in range(quantity))
    return 0.0


def _preserve_variant_market_capacity(orders: Sequence[Any], observation: Mapping[str, Any],
                                      *, seed_item: str) -> list[Any]:
    """Keep the variant seed bias above the policy's worker cash reserve."""
    if not _has_complete_market_state(observation):
        return [_copy_market_order(order) for order in orders]
    reserve = max(100.0, float(CROPS[seed_item]["seed"])) + float(CROPS[seed_item]["seed"])
    non_seed_spend = sum(
        _variant_market_spend(order, observation)
        for order in orders
        if _valid_market_order_schema(order, observation) and order[0] != "BUY_SEED"
    )
    seed_budget = max(0.0, _cash(observation) - non_seed_spend - reserve)
    result: list[Any] = []
    for raw_order in orders:
        if not _valid_market_order_schema(raw_order, observation):
            result.append(_copy_market_order(raw_order))
            continue
        order = list(raw_order)
        if order[0] == "BUY_SEED":
            quantity = min(int(_number(order[2]) or 1), int(seed_budget // float(CROPS[seed_item]["seed"])))
            if quantity <= 0:
                continue
            order[1] = seed_item
            order[2] = quantity
            seed_budget -= quantity * float(CROPS[seed_item]["seed"])
        result.append(order)
    return result


def _prioritize_safety_market_orders(orders: Sequence[Any]) -> list[Any]:
    def order_priority(order: Any) -> int:
        if (
            not isinstance(order, Sequence)
            or isinstance(order, (str, bytes))
            or not order
        ):
            return 3
        if order[0] == "HIRE":
            return 0
        if order[0] == "BUY_LAND":
            return 1
        if order[0] == "BUY_PRODUCT" and len(order) > 1 and order[1] in _BUYABLE_PRODUCTS:
            return 2
        return 3

    return [_copy_market_order(order) for _index, order in sorted(
        enumerate(orders), key=lambda item: (order_priority(item[1]), item[0])
    )]


def _apply_ablations(action: Mapping[str, Any], observation: Mapping[str, Any], ablations: Mapping[str, bool]) -> dict[str, Any]:
    result = {"farmer": list(action.get("farmer", ["PASS"])), "hands": [list(command) for command in action.get("hands", ())],
              "market": _market_orders(action)}
    if not ablations.get("route_scheduling", True):
        directions = {"NORTH", "SOUTH", "EAST", "WEST"}
        farmer = result["farmer"]
        if farmer and isinstance(farmer[0], str) and farmer[0] in directions:
            result["farmer"] = ["PASS"]
        # Hand commands are nested one level below the farmer command.  Keep
        # malformed/nested values inert here; the final sanitizer is the
        # authority that turns them into a legal fallback.
        result["hands"] = [
            ["PASS"] if isinstance(command, Sequence) and not isinstance(command, (str, bytes))
            and command and isinstance(command[0], str) and command[0] in directions else command
            for command in result["hands"]
        ]
    if not ablations.get("market_batch_sizing", True):
        result["market"] = result["market"][:1]
    if not ablations.get("shop_adaptation", True):
        result["market"] = [
            order for order in result["market"]
            if not _valid_market_order_schema(order, observation) or order[0] != "BUY_PRODUCT"
        ]
    if not ablations.get("land_purchase", True):
        result["market"] = [
            order for order in result["market"]
            if not _valid_market_order_schema(order, observation) or order[0] != "BUY_LAND"
        ]
    if not ablations.get("animals", True):
        result["market"] = [
            order for order in result["market"]
            if not _valid_market_order_schema(order, observation) or order[0] != "BUY_ANIMAL"
        ]
        for key in ("farmer", "hands"):
            commands = result[key] if key == "farmer" else result[key]
            if key == "farmer":
                if (len(commands) > 1 and commands[0] == "PLACE" and isinstance(commands[1], str)
                        and commands[1] in ANIMALS):
                    result[key] = ["PASS"]
            else:
                result[key] = [
                    ["PASS"] if (isinstance(command, Sequence) and not isinstance(command, (str, bytes))
                                 and len(command) > 1 and command[0] == "PLACE"
                                 and isinstance(command[1], str) and command[1] in ANIMALS) else command
                    for command in commands
                ]
    return result


def apply_variant(action: Mapping[str, Any], observation: Mapping[str, Any], variant: str,
                  ablations: Mapping[str, bool] | None = None,
                  configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply a named, legality-preserving strategy adjustment at the agent boundary."""
    if not _valid_market_container(action.get("market", ())):
        return _sanitize_action(
            action, observation, action, configuration,
            preserve_malformed_market=True,
        )
    market_orders = _market_orders(action)
    result = {"farmer": list(action.get("farmer", ["PASS"])), "hands": [list(command) for command in action.get("hands", ())],
              "market": market_orders}
    seeds = _private_seeds(observation)
    if variant == "conservative":
        malformed = [
            order for order in result["market"]
            if not _valid_market_order_schema(order, observation)
        ]
        valid_market = [
            order for order in result["market"]
            if _valid_market_order_schema(order, observation)
        ]
        terminal_sales = [
            order for order in valid_market
            if order[0] == "SELL"
        ] if (
            _number(observation.get("day")) == season_days - 1
            and (_number(observation.get("hour")) or 0) >= 22
        ) else []
        if terminal_sales:
            result["market"] = terminal_sales + malformed
        else:
            deadline_hires = [
                order for order in valid_market
                if order == ["HIRE"] and _has_basic_need_deadline(observation)
            ]
            mandatory = [
                order for order in valid_market
                if order[0] == "BUY_PRODUCT" and order[1] in {"WHEAT", "FERTILIZER"}
            ]
            mandatory = deadline_hires + mandatory
            discretionary = [
                order for order in valid_market
                if order[0] not in {"BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"}
            ][:max(0, 1 - len(mandatory))]
            result["market"] = mandatory + discretionary + malformed
    elif variant == "melon-heavy":
        if result["farmer"][:1] == ["PLANT"] and len(result["farmer"]) > 1 and result["farmer"][1] == "WHEAT" and _number(seeds.get("MELON")):
            result["farmer"][1] = "MELON"
        result["market"] = _preserve_variant_market_capacity(
            result["market"], observation, seed_item="MELON",
        )
        for order in result["market"]:
            if _valid_market_order_schema(order, observation) and order[0] == "BUY_SEED":
                order[1] = "MELON"
                break
        if not any(
            _valid_market_order_schema(order, observation) and order[0] == "BUY_SEED"
            for order in result["market"]
        ) and not _number(seeds.get("MELON")) and _cash(observation) >= CROPS["MELON"]["seed"]:
            result["market"].append(["BUY_SEED", "MELON", 1])
    elif variant == "demand-reactive":
        # The production policy already reacts to live market/shop demand.
        # Rewriting its scheduled crop after planning can consume the wheat
        # reserved for FEED or invalidate a WATER deadline, so this supported
        # evaluator variant deliberately preserves that needs-safe schedule.
        pass
    elif variant == "animal-heavy":
        target = _empty_animal_target(observation)
        # Keep one deliberate animal-heavy increment, then leave all later
        # market orders untouched so the production planner can preserve its
        # hiring and basic-needs reservations.  Replacing every seed order
        # with an animal purchase exhausts that reserve and causes avoidable
        # FEED misses.
        if target and _animal_units_owned(observation) < 2:
            for order in result["market"]:
                if _valid_market_order_schema(order, observation) and order[0] == "BUY_SEED":
                    order[:] = ["BUY_ANIMAL", target, 1]
                    break
            else:
                result["market"].append(["BUY_ANIMAL", target, 1])
    elif variant != "mixed":
        raise ValueError(f"unsupported variant: {variant}")
    result["market"] = _prioritize_safety_market_orders(result["market"])
    adjusted = _apply_ablations(result, observation, ablations or _DEFAULT_ABLATIONS)
    return _sanitize_action(
        adjusted, observation, action, configuration,
        preserve_malformed_market=True,
    )


class VariantPolicy:
    """Fresh stateful route candidate or legacy evaluator variant."""

    def __init__(self, variant: str, ablations: Mapping[str, bool] | None = None,
                 configuration: Mapping[str, Any] | None = None,
                 route_candidate: bool | None = None,
                 route_policy: Any = None) -> None:
        if variant not in VARIANTS and variant not in CANDIDATES:
            raise ValueError(f"unsupported variant: {variant}")
        self.variant = variant
        self.ablations = dict(ablations or _DEFAULT_ABLATIONS)
        self.configuration = dict(configuration or {})
        self.is_route_candidate = (
            variant in CANDIDATES and variant != "mixed"
            if route_candidate is None else bool(route_candidate)
        )
        if self.is_route_candidate and variant not in CANDIDATES:
            raise ValueError(f"unsupported candidate: {variant}")
        if self.is_route_candidate:
            self._act = route_policy or candidate_policy(variant)
            self.policy = getattr(self._act, "__self__", None)
        else:
            self.policy = Policy()
            self._act = self.policy.act

    def __call__(self, obs: Mapping[str, Any], _configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        configuration = _configuration if _configuration is not None else self.configuration
        if self.is_route_candidate:
            action = self._act(obs)
            if not _valid_market_container(action.get("market", ())):
                return _sanitize_action(
                    action, obs, action, configuration,
                    preserve_malformed_market=True,
                )
            adjusted = _apply_ablations(action, obs, self.ablations)
            return _sanitize_action(
                adjusted, obs, action, configuration,
                preserve_malformed_market=True,
            )
        return apply_variant(self._act(obs), obs, self.variant, self.ablations, configuration)


def _worker_failure(*, variant: str, opponent: str, seed: int, seat: int, error: str) -> dict[str, Any]:
    return _framework_error_record(
        variant=variant, opponent=opponent, seed=seed, seat=seat, error=error,
    )


def _validate_game_parameters(seed: Any, steps: Any) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")


def _finite_number_or_none(value: Any) -> bool:
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return value is None
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _worker_record_error(record: Mapping[str, Any], *, variant: str, opponent: str,
                         seed: int, seat: int) -> str | None:
    missing = _NORMALIZED_RECORD_FIELDS - record.keys()
    if missing:
        return f"missing normalized record fields: {', '.join(sorted(missing))}"
    if "candidate" in record and (type(record["candidate"]) is not str or record["candidate"] != variant):
        return "worker candidate does not match request"
    if record["variant"] != variant:
        return "worker variant does not match request"
    if record["opponent"] != opponent:
        return "worker opponent does not match request"
    if type(record["seed"]) is not int or record["seed"] != seed:
        return "worker seed must be the requested integer seed"
    if type(record["seat"]) is not int or record["seat"] not in (0, 1) or record["seat"] != seat:
        return "worker seat must be the requested integer seat"
    if not isinstance(record["outcome"], str) or record["outcome"] not in _WORKER_OUTCOMES:
        return "worker outcome is unsupported"
    if "framework_error_reasons" not in record:
        if isinstance(record, dict):
            record["framework_error_reasons"] = []
        reasons = []
    else:
        reasons = record["framework_error_reasons"]
        if type(reasons) is not list:
            return "worker framework_error_reasons must be a list"
        if any(reason not in FRAMEWORK_REASON_ORDER for reason in reasons):
            return "worker framework_error_reasons contains an unknown reason"
        if len(reasons) != len(set(reasons)):
            return "worker framework_error_reasons contains duplicate reasons"
        canonical_reasons = [
            reason for reason in FRAMEWORK_REASON_ORDER if reason in reasons
        ]
        if reasons != canonical_reasons:
            return "worker framework_error_reasons are not in canonical order"
    if type(record["framework_error"]) is not bool:
        return "worker framework_error must be boolean"
    is_framework_error = record["outcome"] == "framework_error"
    # Older worker payloads only had the boolean/outcome pair and represented
    # framework failures with an empty reason list.  Preserve that wire
    # contract while requiring reasons whenever a non-framework record claims
    # a diagnostic failure.
    if record["framework_error"] and not reasons:
        pass
    elif record["framework_error"] != bool(reasons):
        return "worker framework_error disagrees with framework_error_reasons"
    if record["framework_error"] != is_framework_error:
        return "worker framework_error disagrees with outcome"
    if record["submitted_market_order_count"] != record["market_transaction_count"]:
        return "worker submitted market order count disagrees with market transaction count"
    financial_fields = {
        "final_bank", "opponent_final_bank", "bank_differential",
        "terminal_cash", "terminal_inventory_value",
    }
    malformed_numbers = []
    for field in _NORMALIZED_NUMERIC_FIELDS:
        if field not in record:
            continue
        value = record[field]
        if is_framework_error and field in financial_fields and value is None:
            continue
        if field in _NORMALIZED_NONNEGATIVE_INTEGER_FIELDS:
            if type(value) is not int or value < 0:
                malformed_numbers.append(field)
            continue
        if not _finite_number_or_none(value) or value is None:
            malformed_numbers.append(field)
    if malformed_numbers:
        return f"worker numeric fields are malformed: {', '.join(sorted(malformed_numbers))}"
    if "termination_reason" in record and (
        record["termination_reason"] is not None
        and (type(record["termination_reason"]) is not str or not record["termination_reason"])
    ):
        return "worker termination_reason must be a non-empty string or null"
    if "bootstrap_truncated" in record and type(record["bootstrap_truncated"]) is not bool:
        return "worker bootstrap_truncated must be boolean"
    if "no_progress_steps" in record and (
        type(record["no_progress_steps"]) is not int or record["no_progress_steps"] < 0
    ):
        return "worker no_progress_steps must be a nonnegative integer"
    if "time_limit_ending" in record and type(record["time_limit_ending"]) is not bool:
        return "worker time_limit_ending must be boolean"
    if "safety_regression" in record and type(record["safety_regression"]) is not bool:
        return "worker safety_regression must be boolean"
    if "shaping_count" in record and (
        type(record["shaping_count"]) is not int or record["shaping_count"] < 0
    ):
        return "worker shaping_count must be a nonnegative integer"
    if "safety_flags" in record:
        flags = record["safety_flags"]
        if not isinstance(flags, list) or any(type(flag) is not str or not flag for flag in flags):
            return "worker safety_flags must be a list of non-empty strings"
    return None


def _resolve_variant(variant: str | None, candidate: str | None) -> str:
    if variant is not None and candidate is not None:
        raise ValueError("variant and candidate are separate modes; supply only one")
    if candidate is not None:
        if candidate not in CANDIDATES:
            raise ValueError(f"unsupported candidate: {candidate}")
        return candidate
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    return variant


def _resolve_candidates(variants: Sequence[str] | None,
                        candidates: Sequence[str] | None, *,
                        allow_unknown: bool = False) -> list[str]:
    if variants is not None and candidates is not None:
        raise ValueError("variants and candidates are separate modes; supply only one")
    selected = list(candidates if candidates is not None else variants or ("mixed",))
    if allow_unknown:
        return list(dict.fromkeys(selected))
    return _candidate_list(selected) if candidates is not None else _variant_list(selected)


def _load_policy_reference(reference: str) -> Any:
    """Load a previous agent from ``module:attribute`` or ``file.py:attribute``."""
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("baseline policy must use module:callable or file.py:callable")
    module_name, attribute_name = reference.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("baseline policy must use module:callable or file.py:callable")
    path = Path(module_name)
    if path.suffix == ".py" or path.exists():
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        spec = importlib.util.spec_from_file_location("kaggriculture_baseline", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load baseline policy file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    policy = getattr(module, attribute_name, None)
    if isinstance(policy, type):
        policy = policy()
    if hasattr(policy, "act") and callable(policy.act):
        policy = policy.act
    if not callable(policy):
        raise ValueError(f"baseline policy is not callable: {reference}")
    return policy


def run_game(*, variant: str | None = None, candidate: str | None = None,
             opponent: str, seed: int, steps: int, seat: int = 0,
             ablations: Mapping[str, bool] | None = None,
             policy_path: str | None = None,
             policy_identity: str | None = None,
             churn_window: int = 2) -> dict[str, Any]:
    """Run one seeded game in a fresh worker and return its normalized record."""
    if policy_path is not None:
        if variant is not None or candidate is not None:
            raise ValueError("baseline policy is separate from variant and candidate modes")
        if not isinstance(policy_identity, str) or not policy_identity:
            raise ValueError("policy_identity is required with policy_path")
        variant = policy_identity
    else:
        variant = _resolve_variant(variant, candidate)
    if opponent not in OPPONENTS:
        raise ValueError(f"unsupported opponent: {opponent}")
    _validate_game_parameters(seed, steps)
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    if type(churn_window) is not int or churn_window < 1:
        raise ValueError("churn_window must be a positive integer")
    payload = {
        ("candidate" if candidate is not None else "variant"): variant,
        "opponent": opponent,
        "seed": seed,
        "steps": steps,
        "seat": seat,
        "churn_window": churn_window,
    }
    if policy_path is not None:
        payload.pop("variant")
        payload["policy_path"] = policy_path
        payload["policy_identity"] = policy_identity
    if ablations is not None:
        payload["ablations"] = dict(ablations)
    try:
        completed = subprocess.run(
            [sys.executable, "scripts/evaluation_worker.py"],
            input=json.dumps(payload, sort_keys=True) + "\n",
            text=True,
            capture_output=True,
            timeout=_WORKER_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _worker_failure(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            error=f"worker timeout after {_WORKER_TIMEOUT_SECONDS} seconds",
        )
    except Exception as exc:
        return _worker_failure(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            error=f"worker launch failed: {exc}",
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "worker exited without diagnostics").strip()
        return _worker_failure(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            error=f"worker exited with status {completed.returncode}: {detail}",
        )
    try:
        from scripts.evaluation_worker import decode_worker_result

        record = decode_worker_result(completed.stdout)
    except (ValueError, TypeError) as exc:
        return _worker_failure(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            error=f"invalid worker result: {exc}",
        )
    if "framework_error_reasons" not in record:
        record = {**record, "framework_error_reasons": []}
    record_error = _worker_record_error(record, variant=variant, opponent=opponent, seed=seed, seat=seat)
    if record_error:
        return _worker_failure(
            variant=variant, opponent=opponent, seed=seed, seat=seat,
            error=f"invalid worker result: {record_error}",
        )
    return record


def run_matrix(*, variants: Sequence[str] | None = None,
               candidates: Sequence[str] | None = None,
               opponents: Sequence[str], seeds: Sequence[int], steps: int,
               ablations: Mapping[str, bool] | None = None,
               seats: Sequence[int] | None = None,
               policy_path: str | None = None,
               policy_identity: str | None = None,
               churn_window: int | None = None) -> dict[str, Any]:
    """Run the Cartesian product in stable input order with identical seeds."""
    if policy_path is not None:
        if variants is not None or candidates is not None:
            raise ValueError("baseline policy is separate from variant and candidate modes")
        if not isinstance(policy_identity, str) or not policy_identity:
            raise ValueError("policy_identity is required with policy_path")
        selected = [policy_identity]
    else:
        selected = _resolve_candidates(variants, candidates)
    opponents = list(opponents)
    if len(set(opponents)) != len(opponents):
        raise ValueError("opponents must be unique")
    invalid = [opponent for opponent in opponents if opponent not in OPPONENTS]
    if invalid:
        raise ValueError(f"unsupported opponent(s): {', '.join(invalid)}")
    _validate_game_parameters(0, steps)
    if churn_window is not None and (type(churn_window) is not int or churn_window < 1):
        raise ValueError("churn_window must be a positive integer")
    seed_values = list(seeds)
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    for seed in seed_values:
        _validate_game_parameters(seed, steps)
    seat_values = [0, 1] if seats is None else list(seats)
    if len(set(seat_values)) != len(seat_values):
        raise ValueError("seats must be unique")
    invalid_seats = [seat for seat in seat_values if type(seat) is not int or seat not in (0, 1)]
    if invalid_seats:
        raise ValueError(f"unsupported seat(s): {', '.join(map(str, invalid_seats))}")
    records = []
    for variant in selected:
        for opponent in opponents:
            for seed in seed_values:
                for seat in seat_values:
                    kwargs = {
                        "candidate" if candidates is not None else "variant": variant,
                        "opponent": opponent, "seed": seed, "steps": steps,
                    }
                    if policy_path is not None:
                        kwargs.pop("variant")
                        kwargs["policy_path"] = policy_path
                        kwargs["policy_identity"] = policy_identity
                    if churn_window is not None:
                        kwargs["churn_window"] = churn_window
                    kwargs["seat"] = seat
                    if ablations is not None:
                        kwargs["ablations"] = ablations
                    record = run_game(**kwargs)
                    if candidates is not None and isinstance(record, Mapping):
                        record = {**dict(record), "candidate": variant}
                    records.append(record)
    return {"records": records}


def run_evaluation(*, variants: Sequence[str] | None = None,
                   candidates: Sequence[str] | None = None,
                   opponents: Sequence[str], seeds: Sequence[int], steps: int,
                   ablations: Sequence[tuple[str, bool]] = (),
                   seats: Sequence[int] | None = None,
                   holdout_seeds: Sequence[int] | None = None,
                   min_valid_games: int = 20,
                   baseline_policy: str | None = None,
                   baseline_identity: str = "previous-agent",
                   churn_window: int = 2,
                   max_same_item_market_churn: int = 0,
                   max_market_transactions: int | None = None,
                   min_terminal_cash: float = 0.0,
                   min_terminal_inventory_value: float = 0.0) -> dict[str, Any]:
    """Run a baseline and isolated one-component ablations.

    Every ablation starts from the same all-enabled baseline. Repeating a
    component is rejected instead of being silently merged into a combined
    configuration.
    """
    requested = list(ablations)
    if type(min_valid_games) is not int or min_valid_games < 1:
        raise ValueError("min_valid_games must be a positive integer")
    if type(churn_window) is not int or churn_window < 1:
        raise ValueError("churn_window must be a positive integer")
    gate_thresholds = _validate_economic_thresholds(
        max_same_item_market_churn=max_same_item_market_churn,
        max_market_transactions=max_market_transactions,
        min_terminal_cash=min_terminal_cash,
        min_terminal_inventory_value=min_terminal_inventory_value,
    )
    components = [component for component, _enabled in requested]
    unknown_components = [component for component in components if component not in ABLATION_COMPONENTS]
    if unknown_components:
        raise ValueError(f"unsupported ablation component(s): {', '.join(unknown_components)}")
    if len(components) != len(set(components)):
        raise ValueError("each ablation component may be requested only once")
    baseline_config = dict(_DEFAULT_ABLATIONS)
    selected = _resolve_candidates(variants, candidates)
    candidate_kwargs = {"candidates": selected} if candidates is not None else {"variants": selected}
    resolved_seats = [0, 1] if seats is None else list(seats)
    _validate_seed_partition(seeds, holdout_seeds)
    if holdout_seeds is not None and set(resolved_seats) != {0, 1}:
        raise ValueError("holdout evaluation requires both candidate seats")
    matrix_options = {"churn_window": churn_window} if churn_window != 2 else {}
    baseline_records = None
    if baseline_policy is not None:
        if not isinstance(baseline_identity, str) or not baseline_identity:
            raise ValueError("baseline_identity must be a non-empty string")
        baseline_records = run_matrix(
            opponents=opponents, seeds=seeds, steps=steps, seats=resolved_seats,
            policy_path=baseline_policy, policy_identity=baseline_identity,
            **matrix_options,
        )["records"]
    baseline = run_matrix(
        **candidate_kwargs, opponents=opponents, seeds=seeds, steps=steps,
        ablations=baseline_config, seats=resolved_seats,
        **matrix_options,
    )["records"]
    ablation_records: dict[str, list[dict[str, Any]]] = {}
    ablation_configs: dict[str, dict[str, bool]] = {}
    for component, enabled in requested:
        config = dict(baseline_config)
        config[component] = enabled
        ablation_configs[component] = config
        ablation_records[component] = run_matrix(
            **candidate_kwargs, opponents=opponents, seeds=seeds, steps=steps,
            ablations=config, seats=resolved_seats,
            **matrix_options,
        )["records"]
    result = {
        "records": baseline,
        "ablation_records": ablation_records,
        "ablation_configs": ablation_configs,
        "gate_thresholds": gate_thresholds,
    }
    if baseline_policy is not None:
        result["baseline_records"] = baseline_records
        result["baseline_policy"] = {"identity": baseline_identity, "path": baseline_policy}
    if holdout_seeds is not None:
        result["holdout_records"] = run_matrix(
            **candidate_kwargs, opponents=opponents, seeds=list(holdout_seeds), steps=steps,
            ablations=baseline_config, seats=resolved_seats,
            **matrix_options,
        )["records"]
        if baseline_policy is not None:
            result["holdout_baseline_records"] = run_matrix(
                opponents=opponents, seeds=list(holdout_seeds), steps=steps,
                seats=resolved_seats, policy_path=baseline_policy,
                policy_identity=baseline_identity,
                **matrix_options,
            )["records"]
    return result


def _group_results(records: Sequence[Mapping[str, Any]], variants: Sequence[str], opponents: Sequence[str]) -> dict[str, Any]:
    return {
        variant: {
            opponent: aggregate_records([
                record for record in records
                if record.get("candidate", record.get("variant")) == variant
                and record.get("opponent") == opponent
            ])
            for opponent in opponents
        }
        for variant in variants
    }


def _select_default(records: Sequence[Mapping[str, Any]], variants: Sequence[str], *,
                    promotion_decisions: Mapping[str, Mapping[str, Any]] | None = None) -> str | None:
    if promotion_decisions is not None:
        variants = [
            variant for variant in variants
            if promotion_decisions.get(variant, {}).get("status") in {"baseline", "promote"}
        ]
    scored = []
    for variant in variants:
        summary = aggregate_records([
            record for record in records
            if record.get("candidate", record.get("variant")) == variant
        ])
        scored.append((variant, summary))
    if not scored:
        return None
    return min(scored, key=lambda item: (
        item[1]["framework_error_rate"],
        -item[1]["win_rate"],
        -item[1]["median_final_bank"],
        item[0],
    ))[0]


def _candidate_records(records: Sequence[Mapping[str, Any]], candidate: str) -> list[Mapping[str, Any]]:
    return [record for record in records if _record_candidate(record) == candidate]


def _candidate_availability_reason(candidate: str) -> str | None:
    """Return a fail-closed reason for learned candidates unavailable at runtime.

    Historical synthetic candidate names remain valid for report fixtures, but
    a learned candidate is only promotable when it is registered and its
    artifact still validates at the selection boundary.
    """
    if candidate != "learned_v1":
        return None
    if candidate not in CANDIDATES:
        return "candidate_unavailable"
    try:
        candidate_metadata(candidate)
    except Exception:
        return "candidate_unavailable"
    return None


def _apply_candidate_availability_gate(
    candidate: str, decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Mark an unavailable learned candidate as ineligible for selection."""
    reason = _candidate_availability_reason(candidate)
    if reason is None:
        return dict(decision)
    return {**decision, "status": "discard", "reasons": [reason]}


def _candidate_metrics(records: Sequence[Mapping[str, Any]], candidates: Sequence[str],
                      *, min_valid_games: int,
                      expected_matrix: Sequence[tuple[str, int, int]] | None = None,
                      baseline_records: Sequence[Mapping[str, Any]] | None = None,
                      baseline_policy: str | None = None,
                      max_same_item_market_churn: int = 0,
                      max_market_transactions: int | None = None,
                      min_terminal_cash: float = 0.0,
                      min_terminal_inventory_value: float = 0.0,
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic paired summaries and promotion decisions by candidate."""
    summaries = {
        candidate: paired_seed_summary(_candidate_records(records, candidate))
        for candidate in candidates
    }
    decisions: dict[str, Any] = {}
    if candidates and baseline_records is not None:
        for candidate in candidates:
            decisions[candidate] = promotion_decision(
                _candidate_records(records, candidate), baseline_records,
                min_valid_games=min_valid_games, expected_matrix=expected_matrix,
                max_same_item_market_churn=max_same_item_market_churn,
                max_market_transactions=max_market_transactions,
                min_terminal_cash=min_terminal_cash,
                min_terminal_inventory_value=min_terminal_inventory_value,
                baseline_policy=baseline_policy,
            )
    elif candidates:
        baseline_candidate = candidates[0]
        baseline_records = _candidate_records(records, baseline_candidate)
        baseline_reasons = _safety_gate_reasons(
            baseline_records, summaries[baseline_candidate], min_valid_games=min_valid_games,
            expected_matrix=expected_matrix,
            max_same_item_market_churn=max_same_item_market_churn,
            max_market_transactions=max_market_transactions,
            min_terminal_cash=min_terminal_cash,
            min_terminal_inventory_value=min_terminal_inventory_value,
        )
        for candidate in candidates:
            candidate_summary = summaries[candidate]
            if candidate == baseline_candidate:
                decisions[candidate] = {
                    "status": "baseline" if not baseline_reasons else "discard",
                    "reasons": baseline_reasons,
                    "candidate": candidate_summary,
                    "baseline": candidate_summary,
                    "matrix_completeness": _matrix_completeness(
                        baseline_records, expected_matrix,
                    ),
                }
            else:
                decisions[candidate] = promotion_decision(
                    _candidate_records(records, candidate), baseline_records,
                    min_valid_games=min_valid_games, expected_matrix=expected_matrix,
                    max_same_item_market_churn=max_same_item_market_churn,
                    max_market_transactions=max_market_transactions,
                    min_terminal_cash=min_terminal_cash,
                    min_terminal_inventory_value=min_terminal_inventory_value,
                )
    return summaries, decisions


def _empty_opponent_metrics() -> dict[str, Any]:
    """Return the stable zero-valued shape for a configured empty opponent cell."""
    return {
        "record_count": 0,
        "valid": 0,
        "valid_count": 0,
        "valid_games": 0,
        "paired_games": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "seat_balanced_win_rate": None,
        "wilson_win_rate": {"lower": None, "upper": None},
        "wilson_interval": {"lower": None, "upper": None},
        "mean_bank_differential": None,
        "mean_paired_bank_differential": None,
        "lower_tail_bank_differential": None,
        "framework_errors": 0,
        "framework_error_count": 0,
        "invalid": 0,
        "invalid_count": 0,
        "timeouts": 0,
        "timeout_count": 0,
        "no_progress": 0,
        "no_progress_truncations": 0,
        "no_progress_count": 0,
        "safety_failure_rate": 0.0,
        "elo": {
            "ratings": {}, "games": {}, "uncertainty": {},
            "rating_uncertainty": {}, "rating_available": False,
        },
        "bradley_terry": {
            "ratings": {}, "games": {}, "uncertainty": {},
            "rating_uncertainty": {}, "rating_available": False,
        },
        "elo_rating": None,
        "elo_uncertainty": None,
        "rating": None,
        "rating_uncertainty": None,
        "rating_games": 0,
        "rating_available": False,
    }


def _metrics_by_opponent(
    records: Sequence[Mapping[str, Any]], candidates: Sequence[str], opponents: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Build stable candidate/opponent metrics while retaining empty matrix cells."""
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_records = _candidate_records(records, candidate)
        calculated = _summarize_by_opponent(candidate_records)
        result[candidate] = {
            opponent: dict(calculated.get(opponent, _empty_opponent_metrics()))
            for opponent in opponents
        }
    return result


def _promotion_evidence(
    summary: Mapping[str, Any], decision: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a JSON-safe projection used by promotion and comparison tooling."""
    matrix = decision.get("matrix_completeness")
    matrix = matrix if isinstance(matrix, Mapping) else None
    matrix_complete = None
    if matrix is not None:
        matrix_complete = not any(
            matrix.get(key) for key in ("missing", "duplicate", "extra", "invalid_records")
        ) and matrix.get("observed_count") == matrix.get("expected_count")
    counts = {
        field: sum(int((opponent_metrics.get(field) or 0)) for opponent_metrics in metrics.values())
        for field in ("valid", "framework_errors", "invalid", "timeouts", "no_progress")
    }
    safety_denominator = sum(int(opponent_metrics.get("record_count") or 0) for opponent_metrics in metrics.values())
    return {
        "status": decision.get("status"),
        "reasons": (
            list(decision.get("reasons", ()))
            if isinstance(decision.get("reasons", ()), (list, tuple))
            else ([] if decision.get("reasons") is None else [str(decision.get("reasons"))])
        ),
        "matrix_complete": matrix_complete,
        "matrix_completeness": matrix,
        "valid": counts["valid"],
        "valid_games": counts["valid"],
        "framework_errors": counts["framework_errors"],
        "invalid": counts["invalid"],
        "timeouts": counts["timeouts"],
        "no_progress": counts["no_progress"],
        "safety_failure_rate": (
            (counts["framework_errors"] + counts["invalid"] + counts["timeouts"] + counts["no_progress"])
            / safety_denominator if safety_denominator else 0.0
        ),
        "paired_games": summary.get("paired_games", 0),
        "seat_balanced_win_rate": summary.get("seat_balanced_win_rate"),
        "wilson_win_rate": summary.get("wilson_win_rate"),
        "mean_bank_differential": summary.get("mean_paired_bank_differential"),
        "lower_tail_bank_differential": summary.get("lower_tail_bank_differential"),
        "elo": summary.get("elo"),
        "metrics_by_opponent": {key: dict(value) for key, value in metrics.items()},
    }


def _normalized_report_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove output-directory dependence from metadata while retaining names."""
    normalized = dict(config)
    for key in ("output", "replay_summary"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = Path(str(value)).name
    return normalized


def _normalized_command(command: Sequence[str] | None) -> list[str]:
    """Canonicalize evaluator script and report output arguments."""
    values = [str(value) for value in (command or ())]
    normalized: list[str] = []
    for index, value in enumerate(values):
        if index == 0 and Path(value).name == "evaluate.py":
            normalized.append("scripts/evaluate.py")
        elif value == "--output":
            normalized.append(value)
        elif index and values[index - 1] == "--output":
            normalized.append("<report>")
        elif value.startswith("--output="):
            normalized.append("--output=<report>")
        else:
            normalized.append(value)
    return normalized


def build_manifest(*, candidates: Sequence[str], opponents: Sequence[str], seeds: Sequence[int],
                   steps: int, seats: Sequence[int], command: Sequence[str] | None = None,
                   selection_path: str = "candidate", experiment_id: str | None = None,
                   feature_variant: str | None = None,
                   training_mode: str | None = None,
                   action_representation: str | None = None) -> dict[str, Any]:
    """Return the JSON-compatible, versioned reproducibility manifest."""
    manifest = {
        "schema_version": 3,
        "engine_version": str(ENGINE_VERSION),
        "steps": int(steps),
        "seeds": [int(seed) for seed in seeds],
        "seats": [int(seat) for seat in seats],
        "opponents": [str(opponent) for opponent in opponents],
        "candidates": [str(candidate) for candidate in candidates],
        "candidate_models": {
            str(candidate): _manifest_candidate_metadata(
                str(candidate), selection_path=selection_path,
            )
            for candidate in candidates
        },
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "command": _normalized_command(command),
    }
    identity = {
        "experiment_id": experiment_id,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "action_representation": action_representation,
    }
    if any(value is not None for value in identity.values()):
        validate_training_identity(experiment_id, feature_variant, training_mode)
        validate_action_representation(
            action_representation or DEFAULT_ACTION_REPRESENTATION,
        )
        manifest.update(identity)
    return manifest


def _config_seed_values(config: Mapping[str, Any]) -> list[int]:
    values = config.get("seed_values")
    if values is not None:
        return [int(seed) for seed in values]
    count = config.get("seeds", 0)
    start = config.get("start_seed", 0)
    if type(count) is int and type(start) is int:
        return list(range(start, start + count))
    return []


def _configured_matrix(config: Mapping[str, Any], opponents: Sequence[str], seats: Sequence[int], *,
                      holdout: bool = False) -> list[tuple[str, int, int]] | None:
    if "seats" not in config:
        return None
    if holdout:
        values = config.get("holdout_seed_values", config.get("holdout_seeds"))
        if values is None:
            return None
        seeds = [int(seed) for seed in values]
    else:
        if "seed_values" not in config and "seeds" not in config:
            return None
        seeds = _config_seed_values(config)
    return [
        (str(opponent), int(seed), int(seat))
        for opponent in opponents for seed in seeds for seat in seats
    ]


def _select_paired_candidate(candidates: Sequence[str], summaries: Mapping[str, Mapping[str, Any]],
                             development_decisions: Mapping[str, Mapping[str, Any]],
                             holdout_decisions: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Select only candidates that pass both phases using paired metrics."""
    eligible = []
    for index, candidate in enumerate(candidates):
        development = development_decisions.get(candidate, {})
        holdout = holdout_decisions.get(candidate, {})
        summary = summaries.get(candidate, {})
        if development.get("status") not in {"baseline", "promote"}:
            continue
        if holdout.get("status") not in {"baseline", "promote"}:
            continue
        if summary.get("paired_games", 0) < 1:
            continue
        win_rate = summary.get("seat_balanced_win_rate")
        median_differential = summary.get("median_paired_bank_differential")
        if win_rate is None or median_differential is None:
            continue
        eligible.append((float(win_rate), float(median_differential), -index, candidate))
    return max(eligible)[3] if eligible else None


def build_result_document(*, config: Mapping[str, Any], records: Sequence[Mapping[str, Any]], command: Sequence[str] | None = None,
                          ablation_records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                          ablation_configs: Mapping[str, Mapping[str, bool]] | None = None,
                          holdout_records: Sequence[Mapping[str, Any]] | None = None,
                          baseline_records: Sequence[Mapping[str, Any]] | None = None,
                          holdout_baseline_records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    variants = _resolve_candidates(
        config.get("variants"), config.get("candidates"), allow_unknown=True,
    )
    unavailable_candidates = tuple(
        candidate for candidate in variants
        if _candidate_availability_reason(candidate) is not None
    )
    opponents = list(config.get("opponents", ()))
    min_valid_games = config.get("min_valid_games", 20)
    if type(min_valid_games) is not int or min_valid_games < 1:
        raise ValueError("min_valid_games must be a positive integer")
    max_same_item_market_churn = config.get(
        "max_same_item_market_churn", config.get("max_same_item_churn", 0)
    )
    gate_config = {
        "churn_window": config.get("churn_window", 2),
        "max_same_item_market_churn": max_same_item_market_churn,
        # Preserve the old report/config key while exposing the canonical name.
        "max_same_item_churn": max_same_item_market_churn,
        "max_market_transactions": config.get("max_market_transactions"),
        "min_terminal_cash": config.get("min_terminal_cash", 0.0),
        "min_terminal_inventory_value": config.get("min_terminal_inventory_value", 0.0),
    }
    if type(gate_config["churn_window"]) is not int or gate_config["churn_window"] < 1:
        raise ValueError("churn_window must be a positive integer")
    validated_thresholds = _validate_economic_thresholds(
        max_same_item_market_churn=gate_config["max_same_item_market_churn"],
        max_market_transactions=gate_config["max_market_transactions"],
        min_terminal_cash=gate_config["min_terminal_cash"],
        min_terminal_inventory_value=gate_config["min_terminal_inventory_value"],
    )
    gate_config.update({
        "max_same_item_churn": validated_thresholds["max_same_item_market_churn"],
        "max_same_item_market_churn": validated_thresholds["max_same_item_market_churn"],
        "max_market_transactions": validated_thresholds["max_market_transactions"],
        "min_terminal_cash": validated_thresholds["min_terminal_cash"],
        "min_terminal_inventory_value": validated_thresholds["min_terminal_inventory_value"],
    })
    external_baseline_requested = (
        config.get("baseline_policy") is not None or baseline_records is not None
    )
    baseline_identity = config.get("baseline_identity") or "previous-agent"
    development_baseline_records = baseline_records
    development_baseline_policy = None
    if external_baseline_requested:
        development_baseline_policy = baseline_identity
        if development_baseline_records is None:
            development_baseline_records = []
    results = _group_results(records, variants, opponents)
    seats = list(config.get("seats", (0, 1)))
    expected_development_matrix = _configured_matrix(config, opponents, seats)
    paired_summaries, promotion_decisions = _candidate_metrics(
        records, variants, min_valid_games=min_valid_games,
        expected_matrix=expected_development_matrix,
        baseline_records=development_baseline_records,
        baseline_policy=development_baseline_policy,
        max_same_item_market_churn=gate_config["max_same_item_market_churn"],
        max_market_transactions=gate_config["max_market_transactions"],
        min_terminal_cash=float(gate_config["min_terminal_cash"]),
        min_terminal_inventory_value=float(gate_config["min_terminal_inventory_value"]),
    )
    promotion_decisions = {
        candidate: _apply_candidate_availability_gate(
            candidate, promotion_decisions[candidate]
        )
        for candidate in variants
    }
    metrics_by_opponent = _metrics_by_opponent(records, variants, opponents)
    promotion_evidence = {
        candidate: _promotion_evidence(
            paired_summaries[candidate], promotion_decisions[candidate],
            metrics_by_opponent[candidate],
        )
        for candidate in variants
    }
    seed_values = _config_seed_values(config)
    manifest = build_manifest(
        candidates=variants, opponents=opponents, seeds=seed_values,
        steps=config.get("steps", 720), seats=seats, command=command,
        selection_path="candidate" if config.get("candidates") is not None else "legacy_variant",
        experiment_id=config.get("experiment_id"),
        feature_variant=config.get("feature_variant"),
        training_mode=config.get("training_mode"),
        action_representation=config.get("action_representation"),
    )
    holdout_document = None
    development_selected_default = _select_default(
        records, variants, promotion_decisions=promotion_decisions,
    )
    selected_candidate = None
    selected_default = development_selected_default
    selected_default_source = "development_only"
    if holdout_records is not None:
        holdout_seed_values = [
            int(seed) for seed in config.get(
                "holdout_seed_values", config.get("holdout_seeds", ())
            ) or ()
        ]
        holdout_results = _group_results(holdout_records, variants, opponents)
        expected_holdout_matrix = _configured_matrix(config, opponents, seats, holdout=True)
        holdout_baseline_for_metrics = holdout_baseline_records
        holdout_baseline_policy = None
        if external_baseline_requested:
            holdout_baseline_policy = baseline_identity
            if holdout_baseline_for_metrics is None:
                # Do not let an external-baseline holdout silently compare
                # against the first candidate when its paired records are absent.
                holdout_baseline_for_metrics = []
        holdout_paired_summaries, holdout_promotion_decisions = _candidate_metrics(
            holdout_records, variants, min_valid_games=min_valid_games,
            expected_matrix=expected_holdout_matrix,
            baseline_records=holdout_baseline_for_metrics,
            baseline_policy=holdout_baseline_policy,
            max_same_item_market_churn=gate_config["max_same_item_market_churn"],
            max_market_transactions=gate_config["max_market_transactions"],
            min_terminal_cash=float(gate_config["min_terminal_cash"]),
            min_terminal_inventory_value=float(gate_config["min_terminal_inventory_value"]),
        )
        holdout_promotion_decisions = {
            candidate: _apply_candidate_availability_gate(
                candidate, holdout_promotion_decisions[candidate]
            )
            for candidate in variants
        }
        holdout_metrics_by_opponent = _metrics_by_opponent(holdout_records, variants, opponents)
        holdout_promotion_evidence = {
            candidate: _promotion_evidence(
                holdout_paired_summaries[candidate], holdout_promotion_decisions[candidate],
                holdout_metrics_by_opponent[candidate],
            )
            for candidate in variants
        }
        for candidate in variants:
            promotion_decisions[candidate] = {
                **promotion_decisions.get(candidate, {}),
                "holdout": holdout_promotion_decisions.get(candidate),
            }
            promotion_evidence[candidate] = {
                **promotion_evidence[candidate],
                "holdout": holdout_promotion_evidence[candidate],
            }
        selected_candidate = None if unavailable_candidates else _select_paired_candidate(
            variants, holdout_paired_summaries, promotion_decisions,
            holdout_promotion_decisions,
        )
        selected_default = (
            development_selected_default if unavailable_candidates else selected_candidate
        )
        selected_default_source = "holdout"
        holdout_manifest = build_manifest(
            candidates=variants, opponents=opponents, seeds=holdout_seed_values,
            steps=config.get("steps", 720), seats=seats, command=command,
            selection_path="candidate" if config.get("candidates") is not None else "legacy_variant",
            experiment_id=config.get("experiment_id"),
            feature_variant=config.get("feature_variant"),
            training_mode=config.get("training_mode"),
            action_representation=config.get("action_representation"),
        )
        holdout_document = {
            "manifest": holdout_manifest,
            "results": holdout_results,
            "paired_summaries": holdout_paired_summaries,
            "promotion_decisions": holdout_promotion_decisions,
            "metrics_by_opponent": holdout_metrics_by_opponent,
            "promotion_evidence": holdout_promotion_evidence,
            "selected_candidate": selected_candidate,
        }
    ablations = {}
    for component, component_records in (ablation_records or {}).items():
        component_results = _group_results(component_records, variants, opponents)
        component_summaries, component_decisions = _candidate_metrics(
            component_records, variants, min_valid_games=min_valid_games,
            expected_matrix=expected_development_matrix,
            max_same_item_market_churn=gate_config["max_same_item_market_churn"],
            max_market_transactions=gate_config["max_market_transactions"],
            min_terminal_cash=float(gate_config["min_terminal_cash"]),
            min_terminal_inventory_value=float(gate_config["min_terminal_inventory_value"]),
        )
        contribution = {}
        for variant in variants:
            contribution[variant] = {}
            for opponent in opponents:
                baseline = results[variant][opponent]
                ablated = component_results[variant][opponent]
                contribution[variant][opponent] = {
                    "win_rate_delta": ablated["win_rate"] - baseline["win_rate"],
                    "median_final_bank_delta": ablated["median_final_bank"] - baseline["median_final_bank"],
                    "framework_error_rate_delta": ablated["framework_error_rate"] - baseline["framework_error_rate"],
                }
        ablations[component] = {
            "config": dict((ablation_configs or {}).get(component, {})),
            "results": component_results,
            "contribution": contribution,
            "paired_summaries": component_summaries,
            "promotion_decisions": component_decisions,
        }
    metadata = {
        "command": _normalized_command(command),
        "config": _normalized_report_config(config),
        "engine": "kaggle-environments",
        "engine_version": ENGINE_VERSION,
        "baseline_convention": _BASELINE_CONVENTION,
        "baseline_candidate": variants[0] if variants else None,
        "baseline_policy": (
            {"identity": baseline_identity,
             "path": Path(str(config["baseline_policy"])).name
             if config.get("baseline_policy") is not None else None}
            if external_baseline_requested else None
        ),
        "gate_thresholds": dict(gate_config),
        "replay_summary": (
            Path(str(config["replay_summary"])).name
            if config.get("replay_summary") is not None else None
        ),
        "manifest": manifest,
        "selected_candidate": selected_candidate,
        "selected_default": selected_default,
        "selected_default_source": selected_default_source,
        "holdout": holdout_document,
    }
    document = {
        "schema_version": 1,
        "manifest": manifest,
        "metadata": metadata,
        "selected_default": selected_default,
        "selected_default_source": selected_default_source,
        "selected_candidate": selected_candidate,
        "results": results,
        "paired_summaries": paired_summaries,
        "promotion_decisions": promotion_decisions,
        "metrics_by_opponent": metrics_by_opponent,
        "promotion_evidence": promotion_evidence,
        "ablations": ablations,
    }
    if baseline_records is not None:
        document["baseline_paired_summary"] = paired_seed_summary(baseline_records)
        document["baseline_policy"] = metadata["baseline_policy"]
    elif external_baseline_requested:
        document["baseline_policy"] = metadata["baseline_policy"]
    if holdout_document is not None:
        document["holdout"] = holdout_document
        document["holdout_results"] = holdout_document["results"]
        document["holdout_paired_summaries"] = holdout_document["paired_summaries"]
        document["holdout_promotion_decisions"] = holdout_document["promotion_decisions"]
        document["holdout_metrics_by_opponent"] = holdout_document["metrics_by_opponent"]
        document["holdout_promotion_evidence"] = holdout_document["promotion_evidence"]
    return document


def write_result_document(path: str | Path, document: Mapping[str, Any], *, records: Sequence[Mapping[str, Any]],
                          ablation_records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                          holdout_records: Sequence[Mapping[str, Any]] | None = None,
                          baseline_records: Sequence[Mapping[str, Any]] | None = None,
    holdout_baseline_records: Sequence[Mapping[str, Any]] | None = None) -> Path:
    """Write the report and deterministic compact replay-record sidecar."""
    report_path = Path(path)
    report_payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n"
    sidecar = report_path.with_name(f"{report_path.stem}.replays.json")
    sidecar_records = [{"ablation": "baseline", **dict(record)} for record in records]
    sidecar_records.extend(
        {"ablation": "baseline-policy", "evaluation_split": "development", **dict(record)}
        for record in (baseline_records or ())
    )
    for component, component_records in (ablation_records or {}).items():
        sidecar_records.extend({"ablation": component, **dict(record)} for record in component_records)
    sidecar_records.extend(
        {"ablation": "holdout", "evaluation_split": "holdout", **dict(record)}
        for record in (holdout_records or ())
    )
    sidecar_records.extend(
        {"ablation": "baseline-policy", "evaluation_split": "holdout", **dict(record)}
        for record in (holdout_baseline_records or ())
    )
    sidecar_records.sort(key=lambda record: (
        str(record.get("ablation", "")), _record_candidate(record),
        str(record.get("variant", "")), str(record.get("opponent", "")),
        str(record.get("seed", "")), str(record.get("seat", "")),
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    ))
    sidecar_payload = json.dumps(
        {"schema_version": 1, "records": sidecar_records},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n"
    atomic_write_text(report_path, report_payload, name="evaluation report path")
    return atomic_write_text(sidecar, sidecar_payload, name="evaluation sidecar path")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    sidecar = output.with_name(f"{output.stem}.replays.json")
    config = {
        "experiment_id": args.experiment_id,
        "feature_variant": args.feature_variant,
        "training_mode": args.training_mode,
        "action_representation": args.action_representation,
        "seeds": args.seeds,
        "start_seed": args.start_seed,
        "seed_values": seeds,
        "holdout_seed_values": list(args.holdout_seeds) if args.holdout_seeds is not None else None,
        "steps": args.steps,
        "opponents": list(args.opponents),
        "candidates": list(args.candidates) if args.candidates is not None else None,
        "variants": list(args.variants) if args.variants is not None else None,
        "seats": list(args.seats),
        "ablations": [f"{component}={'on' if enabled else 'off'}" for component, enabled in args.ablation],
        "min_valid_games": args.min_valid_games,
        "baseline_policy": args.baseline_policy,
        "baseline_identity": args.baseline_identity,
        "churn_window": args.churn_window,
        "max_same_item_churn": args.max_same_item_churn,
        "max_market_transactions": args.max_market_transactions,
        "min_terminal_cash": args.min_terminal_cash,
        "min_terminal_inventory_value": args.min_terminal_inventory_value,
        "replay_summary": sidecar.name,
        "quick": args.quick,
    }
    evaluation_kwargs = {
        "opponents": args.opponents, "seeds": seeds, "steps": args.steps,
        "ablations": args.ablation, "seats": args.seats,
        "holdout_seeds": args.holdout_seeds,
        "min_valid_games": args.min_valid_games,
        "baseline_policy": args.baseline_policy,
        "baseline_identity": args.baseline_identity,
        "churn_window": args.churn_window,
        "max_same_item_market_churn": args.max_same_item_churn,
        "max_market_transactions": args.max_market_transactions,
        "min_terminal_cash": args.min_terminal_cash,
        "min_terminal_inventory_value": args.min_terminal_inventory_value,
    }
    if args.candidates is not None:
        evaluation_kwargs["candidates"] = args.candidates
    else:
        evaluation_kwargs["variants"] = args.variants
    evaluation = run_evaluation(**evaluation_kwargs)
    document = build_result_document(
        config=config, records=evaluation["records"],
        command=["scripts/evaluate.py", *([*sys.argv[1:]] if argv is None else argv)],
        ablation_records=evaluation["ablation_records"], ablation_configs=evaluation["ablation_configs"],
        holdout_records=evaluation.get("holdout_records"),
        baseline_records=evaluation.get("baseline_records"),
        holdout_baseline_records=evaluation.get("holdout_baseline_records"),
    )
    write_result_document(
        output, document, records=evaluation["records"],
        ablation_records=evaluation["ablation_records"],
        holdout_records=evaluation.get("holdout_records"),
        baseline_records=evaluation.get("baseline_records"),
        holdout_baseline_records=evaluation.get("holdout_baseline_records"),
    )
    games = (
        len(evaluation["records"])
        + len(evaluation.get("baseline_records", ()))
        + sum(len(records) for records in evaluation["ablation_records"].values())
        + len(evaluation.get("holdout_records", ()))
        + len(evaluation.get("holdout_baseline_records", ()))
    )
    print(json.dumps({"output": str(output), "selected_default": document["selected_default"], "games": games}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
