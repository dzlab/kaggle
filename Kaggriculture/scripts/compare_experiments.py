"""Compare evaluation reports produced on one exact paired matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.output_paths import atomic_write_text

_VALID_PROMOTION_STATUSES = frozenset({"baseline", "promote", "discard"})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value}")


def _manifest(report: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = report.get("manifest")
    if not isinstance(manifest, Mapping):
        metadata = _mapping(report.get("metadata"))
        manifest = metadata.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("report is missing a reproducibility manifest for matrix comparison")
    return manifest


def _coordinate(value: Any, *, label: str) -> tuple[str, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} matrix coordinates are invalid")
    opponent, seed, seat = value
    if (
        type(opponent) is not str or not opponent
        or type(seed) is not int
        or type(seat) is not int or seat not in (0, 1)
    ):
        raise ValueError(f"{label} matrix coordinates are invalid")
    return opponent, seed, seat


def _coordinate_list(value: Any, *, label: str) -> list[tuple[str, int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} matrix coordinates are missing")
    return [_coordinate(item, label=label) for item in value]


def _manifest_candidates(report: Mapping[str, Any]) -> tuple[str, ...] | None:
    candidates = _manifest(report).get("candidates")
    if candidates is None:
        return None
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("report manifest must contain non-empty candidates")
    normalized = []
    for candidate in candidates:
        if type(candidate) is not str or not candidate:
            raise ValueError("report manifest contains invalid candidate coordinates")
        normalized.append(candidate)
    if len(set(normalized)) != len(normalized):
        raise ValueError("report manifest candidates must be unique")
    return tuple(normalized)


def _matrix_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _manifest(report)
    seeds = manifest.get("seeds")
    opponents = manifest.get("opponents")
    seats = manifest.get("seats")
    if not all(isinstance(value, Sequence) and not isinstance(value, (str, bytes)) for value in (seeds, opponents, seats)):
        raise ValueError("report matrix manifest must contain seeds, opponents, and seats")
    if any(type(seed) is not int for seed in seeds) or any(type(seat) is not int or seat not in (0, 1) for seat in seats):
        raise ValueError("report matrix contains invalid seed or seat coordinates")
    if any(type(opponent) is not str or not opponent for opponent in opponents):
        raise ValueError("report matrix contains invalid opponent coordinates")
    if len(set(seeds)) != len(seeds) or len(set(opponents)) != len(opponents) or len(set(seats)) != len(seats):
        raise ValueError("report matrix coordinates must be unique")
    coordinates = sorted((str(opponent), int(seed), int(seat)) for opponent in opponents for seed in seeds for seat in seats)

    evidence = report.get("promotion_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    matrix_views = report.get("matrix_completeness")
    matrix_views = matrix_views if isinstance(matrix_views, Mapping) else {}
    for candidate in set(evidence) | set(matrix_views):
        candidate_evidence = _mapping(evidence.get(candidate))
        matrix = candidate_evidence.get("matrix_completeness")
        if not isinstance(matrix, Mapping):
            matrix = _mapping(matrix_views.get(candidate))
        if matrix:
            expected = _coordinate_list(matrix.get("expected"), label="expected")
            if sorted(expected) != coordinates:
                raise ValueError("report matrix coordinates are incompatible with its manifest")
            if candidate_evidence.get("matrix_complete") is False:
                raise ValueError("report matrix is incomplete and cannot be compared")
            if any(matrix.get(key) for key in ("missing", "duplicate", "extra", "invalid_records")):
                raise ValueError("report matrix is incomplete and cannot be compared")
    return {
        "seeds": [int(seed) for seed in seeds],
        "opponents": [str(opponent) for opponent in opponents],
        "seats": [int(seat) for seat in seats],
        "coordinates": [list(value) for value in coordinates],
    }


def _candidate_mapping(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if "promotion_evidence" in report:
        evidence = report.get("promotion_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("report promotion evidence must be keyed by candidate")
        return evidence
    decisions = report.get("promotion_decisions")
    if isinstance(decisions, Mapping):
        return decisions
    matrix_views = report.get("matrix_completeness")
    decision = report.get("decision")
    status = decision.get("status") if isinstance(decision, Mapping) else None
    if isinstance(matrix_views, Mapping) and isinstance(status, str):
        return {
            candidate: {"status": status, "matrix_completeness": matrix}
            for candidate, matrix in matrix_views.items()
        }
    raise ValueError("report is missing complete promotion evidence")


def _candidate_evidence(report: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    evidence = _candidate_mapping(report).get(candidate)
    if not isinstance(evidence, Mapping):
        raise ValueError(f"report is missing complete promotion evidence for candidate {candidate}")
    return evidence


def _validate_matrix_evidence(
    evidence: Mapping[str, Any], signature: Mapping[str, Any], *, candidate: str,
) -> None:
    matrix = evidence.get("matrix_completeness")
    if not isinstance(matrix, Mapping):
        raise ValueError(f"report is missing complete matrix evidence for candidate {candidate}")
    expected = _coordinate_list(matrix.get("expected"), label="expected")
    observed = _coordinate_list(matrix.get("observed"), label="observed")
    coordinates = [tuple(value) for value in signature["coordinates"]]
    if sorted(expected) != sorted(coordinates):
        raise ValueError(f"report matrix evidence is incompatible with its manifest for candidate {candidate}")
    if sorted(observed) != sorted(coordinates) or len(set(observed)) != len(observed):
        raise ValueError(f"report matrix evidence has incomplete observed coordinates for candidate {candidate}")
    matrix_complete = evidence.get("matrix_complete")
    if matrix_complete is None:
        matrix_complete = (
            matrix.get("expected_count") == len(coordinates)
            and matrix.get("observed_count") == len(coordinates)
            and all(matrix.get(key) == [] for key in ("missing", "duplicate", "extra"))
            and matrix.get("invalid_records") == 0
        )
    if matrix_complete is not True:
        raise ValueError(f"report matrix evidence is incomplete for candidate {candidate}")
    if matrix.get("expected_count") != len(coordinates) or matrix.get("observed_count") != len(coordinates):
        raise ValueError(f"report matrix evidence has inconsistent counts for candidate {candidate}")
    for key in ("missing", "duplicate", "extra"):
        values = matrix.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or values:
            raise ValueError(f"report matrix evidence is incomplete for candidate {candidate}")
    if matrix.get("invalid_records") != 0:
        raise ValueError(f"report matrix evidence is incomplete for candidate {candidate}")


def _validate_finite_values(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{label} must contain only finite numeric metrics")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_values(item, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_finite_values(item, label=f"{label}[{index}]")


def _candidate_keys(value: Any, *, label: str) -> set[str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"report {label} must contain complete candidate coverage")
    candidates = set()
    for candidate in value:
        if type(candidate) is not str or not candidate:
            raise ValueError(f"report {label} must be keyed by candidate")
        candidates.add(candidate)
    return candidates


def _validate_report_evidence(report: Mapping[str, Any], metrics: Mapping[str, Any], signature: Mapping[str, Any]) -> None:
    metric_candidates = _candidate_keys(metrics, label="metrics")
    manifest_candidates = _manifest_candidates(report)
    expected_candidates = set(manifest_candidates or metric_candidates)
    if not expected_candidates:
        raise ValueError("report is missing complete candidate coverage")
    if metric_candidates != expected_candidates:
        raise ValueError("report metrics do not match manifest candidates")

    evidence_map = _candidate_mapping(report)
    evidence_candidates = _candidate_keys(evidence_map, label="promotion evidence")
    if evidence_candidates != expected_candidates:
        raise ValueError("report promotion evidence does not match manifest candidates")

    for field in ("promotion_decisions", "matrix_completeness"):
        if field not in report:
            continue
        optional_map = report.get(field)
        if not isinstance(optional_map, Mapping) or set(optional_map) != expected_candidates:
            raise ValueError(f"report {field} does not match manifest candidates")

    for candidate in expected_candidates:
        opponent_metrics = metrics.get(candidate)
        if not isinstance(opponent_metrics, Mapping):
            raise ValueError("report metrics must be keyed by candidate and opponent")
        evidence = _candidate_evidence(report, candidate)
        status = evidence.get("status")
        if status not in _VALID_PROMOTION_STATUSES:
            raise ValueError(f"promotion evidence has invalid status for candidate {candidate}")
        _validate_matrix_evidence(evidence, signature, candidate=candidate)
        _validate_finite_values(opponent_metrics, label=f"metrics_by_opponent.{candidate}")
        _validate_finite_values(evidence, label=f"promotion_evidence.{candidate}")
        expected_opponents = set(signature["opponents"])
        if set(opponent_metrics) != expected_opponents:
            raise ValueError(f"report metrics are incomplete for candidate {candidate}")

    holdout = report.get("holdout")
    if isinstance(holdout, Mapping):
        holdout_signature = _matrix_signature(holdout)
        holdout_metrics = _candidate_metrics(holdout)
        _validate_report_evidence(holdout, holdout_metrics, holdout_signature)


def _candidate_metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("metrics_by_opponent")
    if isinstance(metrics, Mapping):
        return metrics
    results = report.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("report is missing metrics_by_opponent and legacy results")
    converted: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate, opponent_results in results.items():
        if not isinstance(opponent_results, Mapping):
            continue
        converted[str(candidate)] = {}
        for opponent, summary in opponent_results.items():
            summary = _mapping(summary)
            converted[str(candidate)][str(opponent)] = {
                "seat_balanced_win_rate": summary.get("win_rate"),
                "elo_rating": None,
                "mean_bank_differential": summary.get("mean_bank_differential"),
                "safety_failure_rate": summary.get("framework_error_rate"),
            }
    return converted


def _evidence(report: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    evidence = _mapping(report.get("promotion_evidence")).get(candidate)
    if isinstance(evidence, Mapping):
        holdout = evidence.get("holdout")
        if isinstance(holdout, Mapping):
            return holdout
        return evidence
    decisions = _mapping(report.get("promotion_decisions")).get(candidate)
    return decisions if isinstance(decisions, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _metric(summary: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in summary or summary.get(key) is None:
            continue
        value = _number(summary.get(key))
        if value is None:
            raise ValueError(f"metric {key} must be a finite number")
        return value
    return None


def _safety(summary: Mapping[str, Any]) -> float | None:
    direct = _metric(summary, "safety_failure_rate")
    if direct is not None:
        return direct
    denominator = _metric(summary, "record_count", "count")
    if not denominator:
        return None
    failures = sum(_metric(summary, key) or 0.0 for key in (
        "framework_errors", "framework_error_count", "invalid", "invalid_count",
        "timeouts", "timeout_count", "no_progress", "no_progress_count",
    ))
    return failures / denominator


def _delta(current: float | None, baseline: float | None) -> float | None:
    return None if current is None or baseline is None else current - baseline


def compare_reports(report_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Compare two or more reports against the first report's matrix."""
    paths = [Path(path) for path in report_paths]
    if len(paths) < 2:
        raise ValueError("at least two reports are required")
    reports = []
    for path in paths:
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"cannot load report {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"report {path} must contain a JSON object")
        reports.append(value)
    signatures = [_matrix_signature(report) for report in reports]
    reference = signatures[0]
    if any(signature != reference for signature in signatures[1:]):
        raise ValueError("reports use an incompatible evaluation matrix")

    metric_views = [_candidate_metrics(report) for report in reports]
    for report, metrics, signature in zip(reports, metric_views, signatures):
        _validate_report_evidence(report, metrics, signature)
    base_candidates = list(metric_views[0])
    rows: list[dict[str, Any]] = []
    for report_index in range(1, len(reports)):
        current_candidates = list(metric_views[report_index])
        if set(current_candidates) == set(base_candidates):
            candidate_pairs = [(candidate, candidate) for candidate in base_candidates]
        elif len(base_candidates) == len(current_candidates) == 1:
            candidate_pairs = [(base_candidates[0], current_candidates[0])]
        else:
            raise ValueError("reports contain incompatible candidate sets")
        for base_candidate, current_candidate in candidate_pairs:
            base_opponents = _mapping(metric_views[0].get(base_candidate))
            current_opponents = _mapping(metric_views[report_index].get(current_candidate))
            if set(base_opponents) != set(current_opponents):
                raise ValueError("reports contain incompatible opponent metrics")
            for opponent in base_opponents:
                baseline = _mapping(base_opponents.get(opponent))
                current = _mapping(current_opponents.get(opponent))
                current_evidence = _evidence(reports[report_index], current_candidate)
                status = current_evidence.get("status") or "review"
                rows.append({
                    "candidate": current_candidate,
                    "opponent": opponent,
                    "report": paths[report_index].stem,
                    "win_rate_delta": _delta(
                        _metric(current, "seat_balanced_win_rate", "win_rate"),
                        _metric(baseline, "seat_balanced_win_rate", "win_rate"),
                    ),
                    "elo_delta": _delta(
                        _metric(current, "elo_rating", "rating"),
                        _metric(baseline, "elo_rating", "rating"),
                    ),
                    "bank_delta": _delta(
                        _metric(current, "mean_bank_differential", "mean_paired_bank_differential"),
                        _metric(baseline, "mean_bank_differential", "mean_paired_bank_differential"),
                    ),
                    "safety_delta": _delta(_safety(current), _safety(baseline)),
                    "decision": status,
                })
    return {
        "schema_version": 1,
        "baseline": str(paths[0]),
        "reports": [str(path) for path in paths],
        "matrix": reference,
        "deltas": rows,
    }


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(round(value, 12))
    return str(value)


def render_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment comparison",
        "",
        "| candidate | opponent | win_rate_delta | elo_delta | bank_delta | safety_delta | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.get("deltas", ()):
        lines.append("| " + " | ".join(_format(row.get(key)) for key in (
            "candidate", "opponent", "win_rate_delta", "elo_delta", "bank_delta", "safety_delta", "decision",
        )) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json-output", "--output-json", dest="json_output", type=Path)
    parser.add_argument("--markdown-output", "--output-markdown", dest="markdown_output", type=Path)
    args = parser.parse_args(argv)
    try:
        comparison = compare_reports(args.reports)
    except ValueError as exc:
        parser.error(str(exc))
    markdown = render_markdown(comparison)
    encoded = json.dumps(comparison, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.json_output:
        atomic_write_text(args.json_output, encoded, name="comparison JSON output path")
    if args.markdown_output:
        atomic_write_text(args.markdown_output, markdown, name="comparison Markdown output path")
    if not args.json_output and not args.markdown_output:
        sys.stdout.write(encoded)
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
