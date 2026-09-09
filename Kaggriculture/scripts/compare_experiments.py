"""Compare evaluation reports produced on one exact paired matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _manifest(report: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest = report.get("manifest")
    if not isinstance(manifest, Mapping):
        metadata = _mapping(report.get("metadata"))
        manifest = metadata.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("report is missing a reproducibility manifest for matrix comparison")
    return manifest


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
            expected = matrix.get("expected")
            if not isinstance(expected, Sequence) or sorted(tuple(value) for value in expected) != coordinates:
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
    except (TypeError, ValueError):
        return None
    return result


def _metric(summary: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(summary.get(key))
        if value is not None:
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
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load report {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"report {path} must contain a JSON object")
        reports.append(value)
    signatures = [_matrix_signature(report) for report in reports]
    reference = signatures[0]
    if any(signature != reference for signature in signatures[1:]):
        raise ValueError("reports use an incompatible evaluation matrix")

    metric_views = [_candidate_metrics(report) for report in reports]
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
    encoded = json.dumps(comparison, sort_keys=True, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        sys.stdout.write(encoded)
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
