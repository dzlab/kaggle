import math

import pytest

from scripts.evaluation_metrics import (
    bradley_terry_ratings,
    bradley_terry_summary,
    lower_tail,
    summarize_by_opponent,
    paired_seed_summary,
    percentile,
    validate_matrix_coordinates,
    wilson_interval,
)


def _record(seed, seat, outcome, bank_differential, *, candidate="candidate", opponent="hard"):
    return {
        "candidate": candidate,
        "opponent": opponent,
        "seed": seed,
        "seat": seat,
        "outcome": outcome,
        "bank_differential": bank_differential,
    }


def test_paired_seed_summary_aggregates_seats_once_per_seed():
    records = [
        _record(2, 0, "win", 40), _record(2, 1, "loss", -20),
        _record(3, 0, "win", 80), _record(3, 1, "win", 60),
    ]

    summary = paired_seed_summary(records)

    assert summary["paired_games"] == 2
    assert summary["seat_balanced_win_rate"] == pytest.approx(0.75)
    assert summary["mean_paired_bank_differential"] == pytest.approx(40.0)
    assert summary["lower_tail_bank_differential"] == pytest.approx(13.0)
    assert summary["missing_seat_pairs"] == 0


def test_paired_seed_summary_all_binary_scores_report_wilson_interval():
    records = [
        _record(1, 0, "win", 10), _record(1, 1, "win", 8),
        _record(2, 0, "win", 12), _record(2, 1, "win", 9),
    ]

    summary = paired_seed_summary(records)

    assert summary["wilson_win_rate"] is not None
    assert summary["wilson_win_rate"]["lower"] == pytest.approx(0.3423802275)


def test_paired_seed_summary_reports_incomplete_and_duplicate_pairs():
    records = [_record(1, 0, "win", 1), _record(1, 0, "win", 2)]

    summary = paired_seed_summary(records)

    assert summary["paired_games"] == 0
    assert summary["missing_seat_pairs"] == 1
    assert summary["duplicate_seat_pairs"] == 1


def test_percentile_and_lower_tail_are_inclusive_and_interpolated():
    values = [10, 0, 20, 30]

    assert percentile(values, 25) == pytest.approx(7.5)
    assert lower_tail(values, 25) == pytest.approx(7.5)
    assert percentile([], 5) is None


def test_wilson_interval_is_bounded_and_validates_counts():
    interval = wilson_interval(7, 10)

    assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0
    assert wilson_interval(0, 0) == {"lower": None, "upper": None}
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


@pytest.mark.parametrize("successes", [1.0, True, -1, -1.0])
def test_wilson_interval_requires_exact_nonnegative_integer_successes(successes):
    with pytest.raises(ValueError):
        wilson_interval(successes, 10)


def test_bradley_terry_ratings_are_deterministic_and_ordered_by_skill():
    matches = [
        ("strong", "weak", "win"),
        ("strong", "middle", "win"),
        ("middle", "weak", "win"),
        ("weak", "strong", "loss"),
    ]

    first = bradley_terry_ratings(matches)
    second = bradley_terry_ratings(matches)

    assert first == second
    assert first["strong"] > first["middle"] > first["weak"]
    assert math.isclose(sum(first.values()), 3000.0, abs_tol=1e-9)


def test_bradley_terry_summary_includes_games_and_rating_order():
    summary = bradley_terry_summary([
        {"player_a": "a", "player_b": "b", "outcome": "tie"},
        {"player_a": "a", "player_b": "b", "outcome": "win"},
    ])

    assert summary["games"] == {"a": 2, "b": 2}
    assert summary["ratings"]["a"] > summary["ratings"]["b"]
    assert tuple(summary["ratings"]) == ("a", "b")


@pytest.mark.parametrize(
    "call",
    [
        lambda: percentile([1.0, float("nan")], 5),
        lambda: percentile([1.0], 101),
        lambda: wilson_interval(1, 0),
        lambda: bradley_terry_ratings([("a", "b", "unknown")]),
        lambda: bradley_terry_ratings(None),
        lambda: paired_seed_summary([{"seed": 1, "seat": 0, "outcome": "win"}]),
        lambda: paired_seed_summary(None),
        lambda: paired_seed_summary(iter([])),
        lambda: paired_seed_summary({}),
    ],
)
def test_metric_functions_reject_malformed_inputs(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "record",
    [
        _record(1, 0, "win", 1, candidate=""),
        _record(1, 0, "win", 1, opponent=""),
        {**_record(1, 0, "win", 1), "candidate": None},
        {**_record(1, 0, "win", 1), "opponent": None},
    ],
)
def test_paired_seed_summary_requires_nonempty_string_identifiers(record):
    with pytest.raises(ValueError):
        paired_seed_summary([record])


def test_summarize_by_opponent_reports_stable_safety_and_strength_metrics():
    records = [
        _record(1, 0, "win", 10), _record(1, 1, "win", 12),
        _record(2, 0, "loss", -20), _record(2, 1, "loss", -18),
        {
            **_record(3, 0, "tie", 0),
            "framework_error": True,
            "outcome": "framework_error",
            "bank_differential": None,
            "timeout": True,
        },
        {
            **_record(3, 1, "tie", 0),
            "framework_error": True,
            "outcome": "framework_error",
            "bank_differential": None,
            "termination_reason": "no_progress",
            "invalid": True,
        },
    ]

    summary = summarize_by_opponent(records)["hard"]

    assert summary["wins"] == 2
    assert summary["losses"] == 2
    assert summary["ties"] == 0
    assert summary["valid"] == 4
    assert summary["valid_games"] == 4
    assert summary["paired_games"] == 2
    assert summary["seat_balanced_win_rate"] == pytest.approx(0.5)
    assert summary["wilson_win_rate"] is not None
    assert summary["mean_bank_differential"] == pytest.approx(-4.0)
    assert summary["lower_tail_bank_differential"] == pytest.approx(-17.5)
    assert summary["framework_errors"] == 2
    assert summary["invalid"] == 0
    assert summary["timeouts"] == 1
    assert summary["no_progress"] == 1
    assert summary["elo_rating"] is not None
    assert summary["elo_uncertainty"] is not None
    assert summary["rating_games"] == 2


def test_validate_matrix_coordinates_rejects_missing_duplicate_extra_and_invalid():
    expected = [("hard", 1, 0), ("hard", 1, 1)]
    complete = [
        {"opponent": "hard", "seed": 1, "seat": 0},
        {"opponent": "hard", "seed": 1, "seat": 1},
    ]
    assert validate_matrix_coordinates(complete, expected)["complete"] is True

    for records in (
        complete[:1],
        [*complete, complete[0]],
        [*complete, {"opponent": "easy", "seed": 1, "seat": 0}],
        [*complete, {"opponent": "hard", "seed": True, "seat": 0}],
    ):
        with pytest.raises(ValueError, match="matrix"):
            validate_matrix_coordinates(records, expected)
