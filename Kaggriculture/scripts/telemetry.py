"""Local-first training telemetry with optional Weave and W&B mirroring."""

from __future__ import annotations

import importlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from scripts.evaluate import _diagnostic_regression
from scripts.training_identity import (
    DEFAULT_EXPERIMENT_ID,
    FEATURE_VARIANTS,
    TRAINING_MODES,
    validate_training_identity,
    validate_action_representation,
)

DEFAULT_WEAVE_PROJECT = "dzlab/kaggriculture"
DEFAULT_WANDB_ENTITY = "dzlab"
DEFAULT_WANDB_PROJECT = "kaggriculture"


class TrainingTelemetry:
    """Append training events locally and optionally mirror them remotely.

    The local JSONL write happens before optional remote calls. Weave and W&B
    are imported lazily so training remains usable without the observability
    extra. When ``strict`` is false, initialization and event failures are
    warnings; when true, the original exception is raised.
    """

    def __init__(
        self,
        metrics_path: str | Path,
        *,
        project_name: str = DEFAULT_WEAVE_PROJECT,
        enable_weave: bool = False,
        enable_wandb: bool = False,
        wandb_project: str = DEFAULT_WANDB_PROJECT,
        wandb_entity: str = DEFAULT_WANDB_ENTITY,
        wandb_run_name: str | None = None,
        wandb_config: Mapping[str, Any] | None = None,
        strict: bool = False,
        weave_module: Any | None = None,
        wandb_module: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.metrics_path = Path(metrics_path).expanduser()
        self.project_name = project_name
        self.enable_weave = bool(enable_weave)
        self.enable_wandb = bool(enable_wandb)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_config = dict(wandb_config or {})
        self.strict = bool(strict)
        self._logger = logger or logging.getLogger(__name__)
        self._weave_log_metrics = None
        self._wandb_run = None
        if self.enable_weave:
            self._initialize_weave(weave_module)
        if self.enable_wandb:
            self._initialize_wandb(wandb_module)

    def _initialize_weave(self, weave_module: Any | None) -> None:
        try:
            weave = weave_module or importlib.import_module("weave")
            weave.init(self.project_name)

            def log_metrics(payload: dict[str, Any]) -> dict[str, Any]:
                return payload

            self._weave_log_metrics = weave.op(log_metrics)
        except Exception as exc:
            self._handle_weave_failure("Weave telemetry disabled", exc)

    def _handle_weave_failure(self, message: str, exc: Exception) -> None:
        if self.strict:
            raise exc
        self._logger.warning("%s: %s", message, exc)
        self._weave_log_metrics = None

    def _initialize_wandb(self, wandb_module: Any | None) -> None:
        try:
            wandb = wandb_module or importlib.import_module("wandb")
            self._wandb_run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.wandb_config,
                mode="online",
            )
        except Exception as exc:
            self._handle_wandb_failure("W&B telemetry disabled", exc)

    def _handle_wandb_failure(self, message: str, exc: Exception) -> None:
        if self.strict:
            raise exc
        self._logger.warning("%s: %s", message, exc)
        self._wandb_run = None

    def record(
        self, event: str, metrics: Mapping[str, Any] | None = None, *, remote: bool = True,
        **values: Any,
    ) -> None:
        """Persist one event and best-effort mirror it to remote trackers."""
        if not isinstance(event, str) or not event:
            raise ValueError("telemetry event must be a nonempty string")
        payload: dict[str, Any] = {"event": event}
        if metrics is not None:
            payload.update(dict(metrics))
        payload.update(values)
        payload = _json_safe(payload)
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
        if remote and self._weave_log_metrics is not None:
            try:
                self._weave_log_metrics(payload)
            except Exception as exc:
                self._handle_weave_failure("Weave metric logging failed", exc)
        if remote and self._wandb_run is not None:
            try:
                self._wandb_run.log({
                    "telemetry/event": event,
                    **{
                        f"{event}/{key}": value
                        for key, value in payload.items()
                        if key != "event"
                    },
                })
            except Exception as exc:
                self._handle_wandb_failure("W&B metric logging failed", exc)

    def update_wandb_summary(self, values: Mapping[str, Any]) -> None:
        """Update run-level summary fields after validated evaluation evidence."""
        if self._wandb_run is None:
            return
        summary = getattr(self._wandb_run, "summary", None)
        if summary is None or not hasattr(summary, "update"):
            return
        try:
            summary.update(_json_safe(dict(values)))
        except Exception as exc:
            self._handle_wandb_failure("W&B summary update failed", exc)

    @property
    def wandb_url(self) -> str | None:
        """Return the remote W&B run URL when the SDK exposes one."""
        return getattr(self._wandb_run, "url", None)

    def finish(self) -> None:
        """Flush and close the optional W&B run."""
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        except Exception as exc:
            self._handle_wandb_failure("W&B run finalization failed", exc)

    __call__ = record


def load_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    """Load valid JSON object events, returning an empty list when unavailable."""
    path = Path(metrics_path).expanduser()
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _json_safe(value: Any) -> Any:
    """Return a value that can be written by the JSONL telemetry sink."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _number_from(mappings: tuple[Mapping[str, Any], ...], *keys: str) -> float | None:
    for mapping in mappings:
        for key in keys:
            number = _finite_number(mapping.get(key))
            if number is not None:
                return number
    return None


def _record_bank_differential(record: Mapping[str, Any]) -> float | None:
    value = _finite_number(record.get("bank_differential"))
    if value is not None:
        return value
    final_bank = _finite_number(record.get("final_bank"))
    opponent_bank = _finite_number(record.get("opponent_final_bank"))
    if final_bank is not None and opponent_bank is not None:
        return final_bank - opponent_bank
    return None


def _record_values(records: list[Mapping[str, Any]], *keys: str) -> list[float]:
    values = []
    for record in records:
        value = _record_bank_differential(record) if keys == ("bank_differential",) else _finite_number(
            next((record.get(key) for key in keys if record.get(key) is not None), None)
        )
        if value is not None:
            values.append(value)
    return values


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _candidate_records(report: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """Normalize both grouped and legacy flat evaluator record tables."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    records = report.get("records")
    if isinstance(records, Mapping):
        for raw_candidate, raw_records in records.items():
            candidate = str(raw_candidate)
            if isinstance(raw_records, (list, tuple)):
                grouped[candidate] = [
                    record for record in raw_records if isinstance(record, Mapping)
                ]
            else:
                grouped.setdefault(candidate, [])
        return grouped
    if isinstance(records, (list, tuple)):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            candidate = record.get("candidate", record.get("variant"))
            if candidate is not None:
                grouped.setdefault(str(candidate), []).append(record)
    return grouped


def _mapping_for(mapping: Any, key: str) -> Mapping[str, Any]:
    if isinstance(mapping, Mapping) and isinstance(mapping.get(key), Mapping):
        return mapping[key]
    return {}


def _decision_for(report: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    for key in ("decisions", "promotion_decisions"):
        decision = _mapping_for(report.get(key), candidate)
        if decision:
            return decision
    summaries = report.get("summaries")
    if not isinstance(summaries, Mapping):
        summaries = report.get("paired_summaries")
    summary = _mapping_for(summaries, candidate)
    decision = summary.get("decision")
    if isinstance(decision, Mapping):
        return decision
    decision = report.get("decision")
    return decision if isinstance(decision, Mapping) else {}


def _canonical_diagnostics(
    report: Mapping[str, Any], candidate: str, records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use evaluator paired-summary diagnostics, with a legacy raw-record fallback."""
    summaries = report.get("summaries")
    if not isinstance(summaries, Mapping):
        summaries = report.get("paired_summaries")
    summary = _mapping_for(summaries, candidate)
    diagnostics = summary.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)
    return _diagnostic_summary(records)


def _matrix_for(report: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    matrix = _mapping_for(report.get("matrix_completeness"), candidate)
    if matrix:
        return matrix
    decision = _decision_for(report, candidate)
    nested = decision.get("matrix_completeness", decision.get("candidate_matrix_completeness"))
    return nested if isinstance(nested, Mapping) else {}


def _diagnostic_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize termination, truncation, stall, and safety diagnostics."""
    termination_reasons: dict[str, int] = {}
    max_no_progress_steps = 0
    bootstrap_truncated_count = 0
    shaping_count = 0
    time_limit_endings = 0
    safety_regression_count = 0
    record_count = len(records)
    for record in records:
        reason = record.get("termination_reason")
        if reason is not None and str(reason):
            reason = str(reason)
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        if record.get("bootstrap_truncated") is True:
            bootstrap_truncated_count += 1
        raw_shaping_count = _finite_number(record.get("shaping_count"))
        if raw_shaping_count is not None:
            shaping_count += max(0, int(raw_shaping_count))
        no_progress_steps = _finite_number(record.get("no_progress_steps"))
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
    denominator = float(record_count) if record_count else 1.0
    return {
        "record_count": record_count,
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


def validation_safety_regression(
    report: Mapping[str, Any] | None, *, candidate: str,
) -> bool:
    """Return whether a candidate has a fail-closed safety regression."""
    if not isinstance(report, Mapping):
        return False
    grouped = _candidate_records(report)
    candidate_summary = _canonical_diagnostics(
        report, str(candidate), grouped.get(str(candidate), []),
    )
    baseline_summary = _canonical_diagnostics(
        report, "current", grouped.get("current", []),
    )
    return bool(_diagnostic_regression(
        {"diagnostics": candidate_summary},
        {"diagnostics": baseline_summary},
    )["regressed"])


def _breakdown_event(
    records: list[Mapping[str, Any]], *, dimension: str, dimension_value: Any,
) -> dict[str, Any]:
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    bank_values: list[float] = []
    framework_errors = 0
    missed_needs = 0
    for record in records:
        outcome = record.get("outcome")
        is_valid = not record.get("framework_error") and outcome in outcomes
        if is_valid:
            outcomes[outcome] += 1
        if _record_bank_differential(record) is not None and not record.get("framework_error"):
            bank_values.append(_record_bank_differential(record))
        if record.get("framework_error") or outcome == "framework_error":
            framework_errors += 1
        if (_finite_number(record.get("missed_basic_needs")) or 0.0) > 0:
            missed_needs += 1
    games = len(records)
    valid_games = sum(
        not record.get("framework_error") and record.get("outcome") in outcomes
        for record in records
    )
    return {
        "breakdown": dimension,
        "dimension_value": _json_safe(dimension_value),
        "games": games,
        "valid_games": valid_games,
        "wins": outcomes["win"],
        "losses": outcomes["loss"],
        "ties": outcomes["tie"],
        "win_rate": (
            (outcomes["win"] + 0.5 * outcomes["tie"]) / valid_games
            if valid_games else None
        ),
        "mean_bank_differential": mean(bank_values) if bank_values else None,
        "median_bank_differential": median(bank_values) if bank_values else None,
        "p05_bank_differential": _percentile(bank_values, 5),
        "framework_error_count": framework_errors,
        "framework_error_rate": framework_errors / games if games else None,
        "missed_needs_count": missed_needs,
        "missed_needs_rate": missed_needs / games if games else None,
    }


def _flatten_confidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    confidence = summary.get("confidence")
    confidence = confidence if isinstance(confidence, Mapping) else {}
    wilson = summary.get("wilson_win_rate")
    wilson = wilson if isinstance(wilson, Mapping) else confidence.get("wilson_win_rate", {})
    wilson = wilson if isinstance(wilson, Mapping) else {}
    bootstrap_win = summary.get("bootstrap_seat_balanced_win_rate")
    bootstrap_win = bootstrap_win if isinstance(bootstrap_win, Mapping) else confidence.get(
        "bootstrap_seat_balanced_win_rate", {}
    )
    bootstrap_win = bootstrap_win if isinstance(bootstrap_win, Mapping) else {}
    bootstrap_bank = summary.get("bootstrap_bank_differential")
    bootstrap_bank = bootstrap_bank if isinstance(bootstrap_bank, Mapping) else confidence.get(
        "bootstrap_bank_differential", {}
    )
    bootstrap_bank = bootstrap_bank if isinstance(bootstrap_bank, Mapping) else {}
    return {
        "wilson_win_rate_lower": _finite_number(wilson.get("lower")),
        "wilson_win_rate_upper": _finite_number(wilson.get("upper")),
        "bootstrap_win_rate_lower": _finite_number(bootstrap_win.get("lower")),
        "bootstrap_win_rate_upper": _finite_number(bootstrap_win.get("upper")),
        "bootstrap_bank_differential_lower": _finite_number(bootstrap_bank.get("lower")),
        "bootstrap_bank_differential_upper": _finite_number(bootstrap_bank.get("upper")),
    }


def _flatten_matrix(matrix: Mapping[str, Any]) -> dict[str, Any]:
    if not matrix:
        return {
            "matrix_complete": None,
            "matrix_completeness": None,
            "matrix_completeness_ratio": None,
            "matrix_expected_count": None,
            "matrix_observed_count": None,
            "matrix_missing_count": None,
            "matrix_duplicate_count": None,
            "matrix_extra_count": None,
            "matrix_invalid_record_count": None,
        }

    def count(name: str, count_name: str) -> int | None:
        direct = matrix.get(count_name)
        number = _finite_number(direct)
        if number is not None:
            return int(number)
        values = matrix.get(name)
        return len(values) if isinstance(values, (list, tuple)) else None

    expected = count("expected", "expected_count")
    observed = count("observed", "observed_count")
    missing = count("missing", "missing_count")
    duplicate = count("duplicate", "duplicate_count")
    extra = count("extra", "extra_count")
    invalid = count("invalid_records", "invalid_record_count")
    complete = matrix.get("complete")
    if not isinstance(complete, bool):
        complete = (
            expected is not None and observed == expected
            and all(value == 0 for value in (missing, duplicate, extra, invalid) if value is not None)
        )
    issue_count = sum(value or 0 for value in (missing, duplicate, extra, invalid))
    ratio = (
        None if expected in (None, 0) or observed is None
        else max(0.0, min(1.0, (expected - issue_count) / expected))
    )
    return {
        "matrix_complete": complete,
        "matrix_completeness": ratio,
        "matrix_completeness_ratio": ratio,
        "matrix_expected_count": expected,
        "matrix_observed_count": observed,
        "matrix_missing_count": missing,
        "matrix_duplicate_count": duplicate,
        "matrix_extra_count": extra,
        "matrix_invalid_record_count": invalid,
    }


def _validated_report_for_remote(report: Mapping[str, Any], candidates: Sequence[str]) -> bool:
    """Return true only for reports with complete, explicitly validated matrices."""
    if not candidates:
        return False
    summaries = report.get("summaries")
    if not isinstance(summaries, Mapping):
        summaries = report.get("paired_summaries")
    metrics = report.get("metrics_by_opponent")
    evidence = report.get("promotion_evidence")
    if not isinstance(summaries, Mapping) or not isinstance(metrics, Mapping) or not isinstance(evidence, Mapping):
        return False
    for candidate in candidates:
        candidate_evidence = _mapping_for(evidence, candidate)
        if not candidate_evidence:
            return False
        if candidate_evidence.get("matrix_complete") is not True:
            return False
        matrix = _matrix_for(report, candidate)
        flattened = _flatten_matrix(matrix)
        if flattened.get("matrix_complete") is not True:
            return False
    return True


def record_validation_report(
    telemetry: TrainingTelemetry,
    report: Mapping[str, Any] | None,
    *,
    phase: str,
    checkpoint: Any,
    candidate_tag: str,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    action_representation: str = "current_v1",
    potential_reward_coef: float = 0.0,
    no_progress_window: int = 0,
    resolved_margin: float = 0.0,
) -> None:
    """Emit per-game and flattened per-candidate validation telemetry.

    Reports may be incomplete because evaluation can fail before producing a
    summary.  Missing optional fields are represented by ``None`` and never
    prevent the available game records from being written locally first.
    """
    if not isinstance(report, Mapping):
        return
    validate_training_identity(experiment_id, feature_variant, training_mode)
    validate_action_representation(action_representation, source="telemetry")
    for name, value in (
        ("potential_reward_coef", potential_reward_coef),
        ("resolved_margin", resolved_margin),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative finite number")
    if type(no_progress_window) is not int or no_progress_window < 0:
        raise ValueError("no_progress_window must be a nonnegative integer")

    record_event = getattr(telemetry, "record", telemetry)

    grouped = _candidate_records(report)
    summaries = report.get("summaries")
    if not isinstance(summaries, Mapping):
        summaries = report.get("paired_summaries")
    summary_candidates = list(summaries.keys()) if isinstance(summaries, Mapping) else []
    candidates = list(dict.fromkeys([*grouped.keys(), *(str(value) for value in summary_candidates)]))
    remote_validated = _validated_report_for_remote(report, candidates)

    def emit(event: str, payload: Mapping[str, Any], *, remote: bool) -> None:
        if isinstance(telemetry, TrainingTelemetry):
            telemetry.record(event, payload, remote=remote)
        else:
            record_event(event, payload)

    common = {
        "phase": _json_safe(phase),
        "checkpoint": _json_safe(checkpoint),
        "candidate_tag": _json_safe(candidate_tag),
        "experiment_id": experiment_id,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "action_representation": action_representation,
        "potential_reward_coef": float(potential_reward_coef),
        "no_progress_window": no_progress_window,
        "resolved_margin": float(resolved_margin),
    }

    summary_events: list[dict[str, Any]] = []
    for candidate in candidates:
        records = grouped.get(candidate, [])
        for raw_record in records:
            game = {
                str(key): _json_safe(value)
                for key, value in raw_record.items()
                if str(key) != "event"
            }
            game.update(common)
            game["candidate"] = candidate
            emit("validation_game", game, remote=False)

    for candidate in candidates:
        records = grouped.get(candidate, [])
        summary = _mapping_for(summaries, candidate)
        outcomes = {
            "win": sum(record.get("outcome") == "win" for record in records),
            "loss": sum(record.get("outcome") == "loss" for record in records),
            "tie": sum(record.get("outcome") == "tie" for record in records),
        }
        record_count = _number_from((summary,), "record_count")
        record_count = int(record_count) if record_count is not None else len(records)
        wins = _number_from((summary,), "wins")
        losses = _number_from((summary,), "losses")
        ties = _number_from((summary,), "ties")
        wins = int(wins) if wins is not None else outcomes["win"]
        losses = int(losses) if losses is not None else outcomes["loss"]
        ties = int(ties) if ties is not None else outcomes["tie"]

        bank_values = _record_values(records, "bank_differential")
        cash_values = _record_values(records, "terminal_cash", "final_bank")
        inventory_values = _record_values(records, "terminal_inventory_value")
        framework_error_count = sum(
            bool(record.get("framework_error")) or record.get("outcome") == "framework_error"
            for record in records
        )
        missed_needs_count = sum(
            (_finite_number(record.get("missed_basic_needs")) or 0.0) > 0
            for record in records
        )
        framework_rate = _number_from((summary,), "framework_error_rate")
        missed_needs_rate = _number_from((summary,), "missed_needs_rate")
        if framework_rate is None:
            framework_rate = framework_error_count / record_count if record_count else None
        if missed_needs_rate is None:
            missed_needs_rate = missed_needs_count / record_count if record_count else None

        win_rate = _number_from((summary,), "seat_balanced_win_rate", "win_rate")
        if win_rate is None:
            valid_outcomes = [
                record.get("outcome") for record in records
                if not record.get("framework_error")
                and record.get("outcome") in {"win", "tie", "loss"}
            ]
            if valid_outcomes:
                win_rate = mean({"win": 1.0, "tie": 0.5, "loss": 0.0}.get(
                    outcome, 0.0
                ) for outcome in valid_outcomes)
        mean_bank = _number_from((summary,), "mean_paired_bank_differential", "mean_bank_differential")
        median_bank = _number_from((summary,), "median_paired_bank_differential", "median_bank_differential")
        p05_bank = _number_from(
            (summary,), "fifth_percentile_bank_differential", "p05_bank_differential",
            "lower_tail_bank_differential",
        )
        mean_cash = _number_from((summary,), "mean_paired_terminal_cash", "mean_terminal_cash")
        median_cash = _number_from((summary,), "median_paired_terminal_cash", "median_terminal_cash")
        mean_inventory = _number_from(
            (summary,), "mean_paired_terminal_inventory_value", "mean_terminal_inventory_value",
        )
        median_inventory = _number_from(
            (summary,), "median_paired_terminal_inventory_value", "median_terminal_inventory_value",
        )
        mean_bank = mean_bank if mean_bank is not None else (mean(bank_values) if bank_values else None)
        median_bank = median_bank if median_bank is not None else (median(bank_values) if bank_values else None)
        p05_bank = p05_bank if p05_bank is not None else _percentile(bank_values, 5)
        mean_cash = mean_cash if mean_cash is not None else (mean(cash_values) if cash_values else None)
        median_cash = median_cash if median_cash is not None else (median(cash_values) if cash_values else None)
        mean_inventory = mean_inventory if mean_inventory is not None else (
            mean(inventory_values) if inventory_values else None
        )
        median_inventory = median_inventory if median_inventory is not None else (
            median(inventory_values) if inventory_values else None
        )

        confidence = _flatten_confidence(summary)
        elo = summary.get("elo")
        elo = elo if isinstance(elo, Mapping) else {}
        ratings = elo.get("ratings") if isinstance(elo.get("ratings"), Mapping) else {}
        games = elo.get("games") if isinstance(elo.get("games"), Mapping) else {}
        elo_rating = _finite_number(ratings.get(candidate))
        elo_games = _finite_number(games.get(candidate))
        decision = _decision_for(report, candidate)
        reasons = decision.get("reasons", ())
        if isinstance(reasons, (list, tuple, set)):
            reason_text = "|".join(str(reason) for reason in reasons)
        elif reasons is None:
            reason_text = ""
        else:
            reason_text = str(reasons)
        matrix = _flatten_matrix(_matrix_for(report, candidate))
        diagnostics = _canonical_diagnostics(report, candidate, records)
        candidate_metrics = _mapping_for(report.get("metrics_by_opponent"), candidate)
        summary_shaping_count = _number_from((summary,), "shaping_count")
        if summary_shaping_count is not None:
            diagnostics["shaping_count"] = max(0, int(summary_shaping_count))
        event = {
            **common,
            "candidate": candidate,
            "record_count": record_count,
            "paired_games": _number_from((summary,), "paired_games"),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "seat_balanced_win_rate": win_rate,
            "win_rate": win_rate,
            **confidence,
            "mean_paired_bank_differential": mean_bank,
            "mean_bank_differential": mean_bank,
            "median_paired_bank_differential": median_bank,
            "median_bank_differential": median_bank,
            "fifth_percentile_bank_differential": p05_bank,
            "p05_bank_differential": p05_bank,
            "mean_terminal_cash": mean_cash,
            "median_terminal_cash": median_cash,
            "mean_terminal_inventory_value": mean_inventory,
            "median_terminal_inventory_value": median_inventory,
            "framework_error_count": framework_error_count,
            "framework_error_rate": framework_rate,
            "missed_needs_count": missed_needs_count,
            "missed_needs_rate": missed_needs_rate,
            "elo_rating": elo_rating,
            "elo_games": elo_games,
            "decision_status": decision.get("status"),
            "decision_reasons": reason_text,
            "decision_reason_count": len(reasons) if isinstance(reasons, (list, tuple, set)) else int(bool(reason_text)),
            **matrix,
            **diagnostics,
            "safety_regression": False,
            "promotion_safe": True,
            "metrics_by_opponent": {
                str(opponent): _json_safe(dict(opponent_summary))
                for opponent, opponent_summary in candidate_metrics.items()
                if isinstance(opponent_summary, Mapping)
            },
        }
        summary_events.append(event)

    current = next(
        (event for event in summary_events if event.get("candidate") == "current"),
        None,
    )
    for event in summary_events:
        if event.get("candidate") != "current" and current is not None:
            for metric in (
                "win_rate", "median_bank_differential", "p05_bank_differential",
            ):
                candidate_value = _finite_number(event.get(metric))
                current_value = _finite_number(current.get(metric))
                event[f"delta_vs_current_{metric}"] = (
                    candidate_value - current_value
                    if candidate_value is not None and current_value is not None else None
                )
            event["safety_regression"] = validation_safety_regression(
                report, candidate=str(event["candidate"]),
            )
            event["promotion_safe"] = not event["safety_regression"]
        emit("validation_summary", event, remote=remote_validated)
        if remote_validated and isinstance(telemetry, TrainingTelemetry):
            prefix = str(phase)
            existing_summary = getattr(telemetry._wandb_run, "summary", {})
            existing_promoted = (
                bool(existing_summary.get("promoted", False))
                if isinstance(existing_summary, Mapping) else False
            )
            telemetry.update_wandb_summary({
                f"{prefix}_status": event.get("decision_status"),
                f"{prefix}_win_rate": event.get("win_rate"),
                f"{prefix}_elo": event.get("elo_rating"),
                "promoted": bool(event.get("decision_status") == "promote")
                or existing_promoted,
            })

        records = grouped.get(str(event.get("candidate")), [])
        for dimension in ("opponent", "seat"):
            values = sorted(
                {record.get(dimension) for record in records if dimension in record},
                key=lambda value: str(value),
            )
            for value in values:
                breakdown = {
                    **common,
                    "candidate": event.get("candidate"),
                    **_breakdown_event(
                        [record for record in records if record.get(dimension) == value],
                        dimension=dimension,
                        dimension_value=value,
                    ),
                }
                emit("validation_breakdown", breakdown, remote=remote_validated)
