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


def bradley_terry_summary(matches: Iterable[object]) -> dict[str, Any]:
    """Return ratings plus per-player game counts for a match table."""
    parsed = _parse_matches(matches)
    games: dict[str, int] = {}
    for first, second, _score in parsed:
        games[first] = games.get(first, 0) + 1
        games[second] = games.get(second, 0) + 1
    return {"ratings": bradley_terry_ratings(parsed), "games": games}


elo_summary = bradley_terry_summary
