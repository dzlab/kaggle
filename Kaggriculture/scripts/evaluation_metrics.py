"""Pure evaluation evidence calculations for small match tables."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from statistics import NormalDist
from typing import Any


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def percentile(values: Iterable[Real], percent: Real) -> float | None:
    """Return an inclusive, linearly interpolated percentile."""
    p = _number(percent, "percent")
    if not 0.0 <= p <= 100.0:
        raise ValueError("percent must be in [0, 100]")
    try:
        ordered = sorted(_number(value, "values") for value in values)
    except TypeError as exc:
        raise ValueError("values must be an iterable of finite numbers") from exc
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def lower_tail(values: Iterable[Real], percent: Real = 5.0) -> float | None:
    """Return a lower-tail percentile, defaulting to the fifth percentile."""
    return percentile(values, percent)


def wilson_interval(
    successes: Real,
    trials: int,
    *,
    confidence: Real = 0.95,
) -> dict[str, float | None]:
    """Return a Wilson score interval for a Bernoulli win rate."""
    if type(trials) is not int or trials < 0:
        raise ValueError("trials must be a nonnegative integer")
    if type(successes) is not int or successes < 0:
        raise ValueError("successes must be an exact nonnegative integer")
    success_count = float(successes)
    if successes > trials:
        raise ValueError("successes must be between zero and trials")
    level = _number(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if trials == 0:
        return {"lower": None, "upper": None}
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    proportion = success_count / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    ) / denominator
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _record(record: Mapping[str, Any]) -> tuple[tuple[str, str, int], int, str, float]:
    if not isinstance(record, Mapping):
        raise ValueError("each match record must be a mapping")
    candidate = record.get("candidate", record.get("variant", ""))
    opponent = record.get("opponent", "")
    seed = record.get("seed")
    seat = record.get("seat")
    outcome = record.get("outcome")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("candidate must be a non-empty string")
    if not isinstance(opponent, str) or not opponent:
        raise ValueError("opponent must be a non-empty string")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be exactly 0 or 1")
    if outcome not in {"win", "loss", "tie"}:
        raise ValueError("outcome must be win, loss, or tie")
    raw_bank = record.get("bank_differential")
    if raw_bank is None and "final_bank" in record and "opponent_final_bank" in record:
        raw_bank = _number(record["final_bank"], "final_bank") - _number(
            record["opponent_final_bank"], "opponent_final_bank",
        )
    bank = _number(raw_bank, "bank_differential")
    return (candidate, opponent, seed), seat, outcome, bank


def paired_seed_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exactly one valid record for each seat of each seed."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of mappings")
    buckets: dict[tuple[str, str, int], dict[int, list[tuple[str, float]]]] = {}
    for record in records:
        key, seat, outcome, bank = _record(record)
        buckets.setdefault(key, {0: [], 1: []})[seat].append((outcome, bank))

    scores = {"win": 1.0, "tie": 0.5, "loss": 0.0}
    pair_scores: list[float] = []
    pair_banks: list[float] = []
    missing = 0
    duplicate = 0
    wins = losses = ties = 0
    for key in sorted(buckets):
        seats = buckets[key]
        if len(seats[0]) != 1 or len(seats[1]) != 1:
            if not seats[0] or not seats[1]:
                missing += 1
            if len(seats[0]) > 1 or len(seats[1]) > 1:
                duplicate += 1
            continue
        first, second = seats[0][0], seats[1][0]
        pair_scores.append((scores[first[0]] + scores[second[0]]) / 2.0)
        pair_banks.append((first[1] + second[1]) / 2.0)
        for outcome, _bank in (first, second):
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                ties += 1

    binary_scores = [score for score in pair_scores if score in (0.0, 1.0)]
    wilson = (
        wilson_interval(sum(score == 1.0 for score in binary_scores), len(binary_scores))
        if len(binary_scores) == len(pair_scores) else None
    )
    return {
        "record_count": len(records),
        "paired_games": len(pair_scores),
        "missing_seat_pairs": missing,
        "duplicate_seat_pairs": duplicate,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "seat_balanced_win_rate": (
            sum(pair_scores) / len(pair_scores) if pair_scores else None
        ),
        "mean_paired_bank_differential": (
            sum(pair_banks) / len(pair_banks) if pair_banks else None
        ),
        "lower_tail_bank_differential": lower_tail(pair_banks),
        "wilson_win_rate": wilson,
    }


def _record_bank(record: Mapping[str, Any]) -> float | None:
    value = record.get("bank_differential")
    if value is None and record.get("final_bank") is not None and record.get("opponent_final_bank") is not None:
        try:
            value = float(record["final_bank"]) - float(record["opponent_final_bank"])
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        return None
    return float(value)


def _metric_record_state(record: Mapping[str, Any]) -> tuple[bool, bool, bool, bool, bool]:
    """Return valid, framework-error, invalid, timeout, and no-progress flags."""
    if not isinstance(record, Mapping):
        return False, False, True, False, False
    reason = str(record.get("termination_reason") or "").strip().lower().replace("-", "_")
    reasons = record.get("framework_error_reasons", ())
    reason_names = {
        str(value).strip().lower().replace("-", "_")
        for value in reasons
    } if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) else set()
    framework_error = bool(record.get("framework_error")) or record.get("outcome") == "framework_error"
    timeout = bool(record.get("timeout")) or bool(record.get("timed_out")) or bool(record.get("time_limit_ending"))
    timeout = timeout or reason in {"timeout", "time_limit", "time_limit_ending"} or "timeout" in reason_names
    no_progress = bool(record.get("no_progress")) or bool(record.get("no_progress_truncation"))
    no_progress = no_progress or (
        bool(record.get("bootstrap_truncated")) and (
            reason == "no_progress" or bool(record.get("no_progress_steps"))
        )
    )
    no_progress = no_progress or reason == "no_progress" or "no_progress" in reason_names
    invalid = bool(record.get("invalid")) or bool(record.get("invalid_game"))
    valid = (
        not framework_error
        and not invalid
        and type(record.get("seat")) is int
        and record["seat"] in (0, 1)
        and record.get("outcome") in {"win", "loss", "tie"}
        and _record_bank(record) is not None
    )
    if not framework_error and not valid:
        invalid = True
    return valid, framework_error, invalid, timeout, no_progress


def _rating_uncertainty(games: Mapping[str, int], *, available: bool, confidence: float = 0.95) -> dict[str, float | None]:
    if not available:
        return {player: None for player in games}
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    scale = 400.0 / math.log(10.0)
    return {
        player: float(z * scale / math.sqrt(max(1, count)))
        for player, count in games.items()
    }


def summarize_by_opponent(
    records: Sequence[Mapping[str, Any]], *, min_rating_games: int = 2,
) -> dict[str, dict[str, Any]]:
    """Return stable paired metrics keyed by opponent.

    Counts are taken over raw game records while win rate, bank differential,
    and ratings use complete two-seat seed pairs.  Framework failures never
    contribute to valid outcomes; explicit non-framework malformed records are
    counted as invalid.
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of mappings")
    if type(min_rating_games) is not int or min_rating_games < 1:
        raise ValueError("min_rating_games must be a positive integer")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each match record must be a mapping")
        opponent = record.get("opponent")
        if not isinstance(opponent, str) or not opponent:
            raise ValueError("opponent must be a non-empty string")
        grouped.setdefault(opponent, []).append(record)

    summaries: dict[str, dict[str, Any]] = {}
    score_by_outcome = {"win": 1.0, "tie": 0.5, "loss": 0.0}
    for opponent, opponent_records in grouped.items():
        valid_records = []
        framework_errors = invalid = timeouts = no_progress = 0
        for record in opponent_records:
            valid, framework_error, invalid_record, timeout, stalled = _metric_record_state(record)
            valid_records.append(record) if valid else None
            framework_errors += int(framework_error)
            invalid += int(invalid_record and not framework_error)
            timeouts += int(timeout)
            no_progress += int(stalled)

        buckets: dict[tuple[str, int], dict[int, list[Mapping[str, Any]]]] = {}
        for record in valid_records:
            candidate = record.get("candidate", record.get("variant"))
            seed = record.get("seed")
            if not isinstance(candidate, str) or not candidate or type(seed) is not int:
                continue
            buckets.setdefault((candidate, seed), {0: [], 1: []})[record["seat"]].append(record)

        pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        for (candidate, _seed), seats in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
            if len(seats[0]) == 1 and len(seats[1]) == 1:
                pairs.append((candidate, seats[0][0], seats[1][0]))

        pair_scores: list[float] = []
        pair_banks: list[float] = []
        rating_matches: list[tuple[str, str, float]] = []
        wins = losses = ties = 0
        for candidate, first, second in pairs:
            score = (score_by_outcome[first["outcome"]] + score_by_outcome[second["outcome"]]) / 2.0
            pair_scores.append(score)
            pair_banks.append((_record_bank(first) + _record_bank(second)) / 2.0)  # type: ignore[operator]
            for record in (first, second):
                outcome = record["outcome"]
                wins += int(outcome == "win")
                losses += int(outcome == "loss")
                ties += int(outcome == "tie")
            if candidate != opponent:
                rating_matches.append((candidate, opponent, score))

        binary_scores = [score for score in pair_scores if score in (0.0, 1.0)]
        wilson = (
            wilson_interval(sum(score == 1.0 for score in binary_scores), len(binary_scores))
            if len(binary_scores) == len(pair_scores) else None
        )
        rating = bradley_terry_summary(rating_matches, min_games=min_rating_games)
        candidate_names = sorted({candidate for candidate, _first, _second in pairs})
        candidate = candidate_names[0] if len(candidate_names) == 1 else None
        candidate_rating = rating["ratings"].get(candidate) if candidate else None
        candidate_uncertainty = rating["uncertainty"].get(candidate) if candidate else None
        safety_denominator = len(opponent_records) or 1
        summaries[opponent] = {
            "record_count": len(opponent_records),
            "valid": len(valid_records),
            "valid_count": len(valid_records),
            "valid_games": len(valid_records),
            "paired_games": len(pairs),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "seat_balanced_win_rate": sum(pair_scores) / len(pair_scores) if pair_scores else None,
            "wilson_win_rate": wilson,
            "wilson_interval": wilson,
            "mean_bank_differential": sum(pair_banks) / len(pair_banks) if pair_banks else None,
            "mean_paired_bank_differential": sum(pair_banks) / len(pair_banks) if pair_banks else None,
            "lower_tail_bank_differential": lower_tail(pair_banks),
            "framework_errors": framework_errors,
            "framework_error_count": framework_errors,
            "invalid": invalid,
            "invalid_count": invalid,
            "timeouts": timeouts,
            "timeout_count": timeouts,
            "no_progress": no_progress,
            "no_progress_truncations": no_progress,
            "no_progress_count": no_progress,
            "safety_failure_rate": (framework_errors + invalid + timeouts + no_progress) / safety_denominator,
            "elo": rating,
            "bradley_terry": rating,
            "elo_rating": candidate_rating,
            "elo_uncertainty": candidate_uncertainty,
            "rating": candidate_rating,
            "rating_uncertainty": candidate_uncertainty,
            "rating_games": rating["games"].get(candidate, 0) if candidate else 0,
            "rating_available": bool(rating["rating_available"] and candidate_rating is not None),
        }
    return summaries


opponent_metrics = summarize_by_opponent


def validate_matrix_coordinates(
    records: Sequence[Mapping[str, Any]],
    expected_matrix: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Validate that records contain exactly one valid coordinate per matrix cell."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("matrix records must be a sequence")
    if isinstance(expected_matrix, (str, bytes)) or not isinstance(expected_matrix, Sequence):
        raise ValueError("matrix must be a sequence of coordinates")

    def normalize(value: Any, label: str) -> tuple[str, int, int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
            raise ValueError(f"matrix {label} coordinates are invalid")
        opponent, seed, seat = value
        if type(opponent) is not str or not opponent or type(seed) is not int or type(seat) is not int or seat not in (0, 1):
            raise ValueError(f"matrix {label} coordinates are invalid")
        return opponent, seed, seat

    expected = [normalize(value, "expected") for value in expected_matrix]
    if len(set(expected)) != len(expected):
        raise ValueError("matrix expected coordinates are duplicate")
    observed = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("matrix record coordinate is invalid")
        observed.append(normalize((record.get("opponent"), record.get("seed"), record.get("seat")), "record"))
    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    duplicates = sorted(coordinate for coordinate in observed_set if observed.count(coordinate) > 1)
    if missing or extra or duplicates or len(observed) != len(expected):
        raise ValueError(
            "matrix coordinates are incompatible: "
            f"missing={missing}, duplicate={duplicates}, extra={extra}"
        )
    return {
        "expected": [list(value) for value in expected],
        "observed": [list(value) for value in observed],
        "expected_count": len(expected),
        "observed_count": len(observed),
        "complete": True,
    }


def _match(value: object) -> tuple[str, str, float]:
    if isinstance(value, Mapping):
        first = value.get("player_a", value.get("a"))
        second = value.get("player_b", value.get("b"))
        outcome = value.get("outcome")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        first, second, outcome = value
    else:
        raise ValueError("matches must contain (player_a, player_b, outcome) values")
    if not isinstance(first, str) or not first or not isinstance(second, str) or not second:
        raise ValueError("match players must be non-empty strings")
    if first == second:
        raise ValueError("match must have distinct players and a valid outcome")
    if outcome in {"win", "loss", "tie"}:
        score = {"win": 1.0, "tie": 0.5, "loss": 0.0}[outcome]
    elif isinstance(outcome, Real) and not isinstance(outcome, bool) and 0.0 <= float(outcome) <= 1.0:
        score = float(outcome)
    else:
        raise ValueError("match must have distinct players and a valid outcome")
    return first, second, score


def _parse_matches(matches: Iterable[object]) -> list[tuple[str, str, float]]:
    if isinstance(matches, (str, bytes)):
        raise ValueError("matches must be an iterable of match values")
    try:
        return [_match(value) for value in matches]
    except TypeError as exc:
        raise ValueError("matches must be an iterable of match values") from exc


def bradley_terry_ratings(
    matches: Iterable[object],
    *,
    initial_rating: Real = 1000.0,
    iterations: int = 32,
) -> dict[str, float]:
    """Return deterministic regularized Bradley--Terry/Elo-style ratings.

    Ties count as half a win.  A small half-win prior keeps tiny or unbeaten
    match tables finite, which is useful for development evaluations.
    """
    initial = _number(initial_rating, "initial_rating")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    parsed = _parse_matches(matches)
    players: list[str] = []
    games: dict[tuple[str, str], int] = {}
    wins: dict[str, float] = {}
    for first, second, score in parsed:
        for player in (first, second):
            if player not in players:
                players.append(player)
            wins.setdefault(player, 0.0)
        wins[first] += score
        wins[second] += 1.0 - score
        pair = tuple(sorted((first, second)))
        games[pair] = games.get(pair, 0) + 1
    if not players:
        return {}

    strengths = {player: 1.0 for player in players}
    for _ in range(iterations):
        updated: dict[str, float] = {}
        for player in players:
            denominator = 0.0
            for (first, second), count in games.items():
                if player == first:
                    denominator += count / (strengths[first] + strengths[second])
                elif player == second:
                    denominator += count / (strengths[first] + strengths[second])
            updated[player] = (wins[player] + 0.5) / denominator if denominator else 1.0
        geometric_mean = math.exp(sum(math.log(value) for value in updated.values()) / len(updated))
        strengths = {player: value / geometric_mean for player, value in updated.items()}

    scale = 400.0 / math.log(10.0)
    ratings = {player: initial + scale * math.log(strengths[player]) for player in players}
    offset = (sum(ratings.values()) / len(ratings)) - initial
    return {player: rating - offset for player, rating in ratings.items()}


def bradley_terry_summary(
    matches: Iterable[object], *, min_games: int = 2,
) -> dict[str, Any]:
    """Return ratings, counts, and conservative uncertainty for a match table."""
    if type(min_games) is not int or min_games < 1:
        raise ValueError("min_games must be a positive integer")
    parsed = _parse_matches(matches)
    games: dict[str, int] = {}
    for first, second, _score in parsed:
        games[first] = games.get(first, 0) + 1
        games[second] = games.get(second, 0) + 1
    available = bool(parsed) and min(games.values(), default=0) >= min_games
    uncertainty = _rating_uncertainty(games, available=available)
    return {
        "ratings": bradley_terry_ratings(parsed),
        "games": games,
        "uncertainty": uncertainty,
        "rating_uncertainty": uncertainty,
        "rating_available": available,
    }


elo_summary = bradley_terry_summary
