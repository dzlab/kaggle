"""Shared validation for evaluator reports used by training and packaging."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def expected_evaluation_matrix(
    seed_values: Sequence[int], opponents: Sequence[str], seats: Sequence[int],
) -> list[list[str | int]]:
    """Return the ordered evaluator coordinates required by a report."""
    return [
        [opponent, seed, seat]
        for opponent in opponents
        for seed in seed_values
        for seat in seats
    ]


def evaluation_report_is_complete(
    report: Mapping[str, Any], *, identity: str, seed_values: Sequence[int],
    opponents: Sequence[str] = ("pass", "random", "starter"),
    seats: Sequence[int] = (0, 1),
    artifact_path: str | Path | None = None,
) -> bool:
    """Validate an evaluator report's identity, artifact, and full matrix."""
    artifact = report.get("artifact")
    configuration = report.get("configuration")
    completeness = report.get("matrix_completeness", {})
    records = report.get("records", {})
    expected_matrix = report.get("expected_matrix", [])
    configured_matrix = expected_evaluation_matrix(seed_values, opponents, seats)
    if not isinstance(artifact, Mapping) or not isinstance(configuration, Mapping):
        return False
    if artifact.get("identity") != identity:
        return False
    reported_hash = artifact.get("sha256")
    if (
        not isinstance(reported_hash, str)
        or len(reported_hash) != 64
        or any(character not in "0123456789abcdef" for character in reported_hash)
        or artifact_path is None
    ):
        return False
    candidate_path = Path(artifact_path).expanduser()
    if not candidate_path.is_file():
        return False
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if reported_hash != digest:
        return False
    if configuration.get("seed_values") != list(seed_values):
        return False
    if configuration.get("opponents") != list(opponents):
        return False
    if configuration.get("seats") != list(seats):
        return False
    if expected_matrix != configured_matrix:
        return False
    expected_coordinates = {tuple(coordinate) for coordinate in configured_matrix}
    if not isinstance(completeness, Mapping) or not isinstance(records, Mapping):
        return False
    if set(completeness) != {"current", identity} or set(records) != {"current", identity}:
        return False
    for candidate in ("current", identity):
        details = completeness.get(candidate)
        candidate_records = records.get(candidate)
        if not isinstance(details, Mapping) or not isinstance(candidate_records, list):
            return False
        if (
            details.get("expected") != configured_matrix
            or details.get("expected_count") != len(configured_matrix)
            or details.get("observed_count") != len(configured_matrix)
            or any(details.get(field) for field in ("missing", "duplicate", "extra", "invalid_records"))
        ):
            return False
        observed_coordinates = []
        for record in candidate_records:
            if not isinstance(record, Mapping):
                return False
            opponent = record.get("opponent")
            seed = record.get("seed")
            seat = record.get("seat")
            if type(opponent) is not str or type(seed) is not int or type(seat) is not int:
                return False
            observed_coordinates.append((opponent, seed, seat))
        if (
            len(observed_coordinates) != len(configured_matrix)
            or len(set(observed_coordinates)) != len(observed_coordinates)
            or set(observed_coordinates) != expected_coordinates
        ):
            return False
    return report.get("schema_version") == 1 and isinstance(expected_matrix, list)
