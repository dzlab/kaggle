import math

import pytest

from kagriculture_agent.reward_shaping import (
    classify_progress,
    economic_potential,
    should_bootstrap_truncate,
    shaped_transition_reward,
    potential_difference,
)


def test_economic_potential_is_bounded_and_uses_normalized_economic_signals():
    low = {
        "cash": 1_000,
        "inventory": {"WHEAT": 10},
        "market": {"prices": {"WHEAT": 10}},
        "production": 2,
        "worker_utilization": 0.25,
        "deadline_risk": 0.2,
    }
    high = {
        "cash": 9_000,
        "inventory": {"WHEAT": 100},
        "market": {"prices": {"WHEAT": 10}},
        "production": 8,
        "worker_utilization": 0.75,
        "deadline_risk": 0.05,
    }

    assert -1.0 <= economic_potential(low) <= 1.0
    assert -1.0 <= economic_potential(high) <= 1.0
    assert economic_potential(high) > economic_potential(low)


def test_malformed_or_missing_observation_is_neutral_and_finite():
    for observation in (None, [], {"cash": float("nan")}, {"inventory": {"WHEAT": float("inf")}}):
        potential = economic_potential(observation)
        assert potential == 0.0
        assert math.isfinite(potential)


def test_oversized_numeric_observation_is_neutral_and_finite():
    potential = economic_potential({"cash": 10**10_000})

    assert potential == 0.0
    assert math.isfinite(potential)


def test_potential_difference_uses_discounted_next_potential():
    current = {"cash": 2_000}
    next_state = {"cash": 8_000}
    expected = 0.9 * economic_potential(next_state) - economic_potential(current)

    assert potential_difference(current, next_state, gamma=0.9) == pytest.approx(expected)
    assert potential_difference(current, current, gamma=0.9) == pytest.approx(-0.1 * economic_potential(current))


def test_shaped_transition_reward_adds_shaping_to_terminal_reward():
    transition = {
        "observation": {"cash": 2_000},
        "next_observation": {"cash": 8_000},
        "reward": 0.75,
        "done": True,
    }
    shaping = potential_difference(
        transition["observation"], transition["next_observation"], gamma=0.9,
    )

    assert shaped_transition_reward(transition, gamma=0.9, coefficient=0.4) == pytest.approx(
        0.75 + 0.4 * shaping,
    )


@pytest.mark.parametrize(
    "transition,expected",
    [
        (
            {"observation": None, "next_observation": {"cash": 8_000}, "reward": 1.25},
            1.25,
        ),
        (
            {"observation": {"cash": 2_000}, "next_observation": [], "reward": -0.5},
            -0.5,
        ),
        (
            {
                "observation": {"cash": 10**10_000},
                "next_observation": {"cash": 8_000},
                "terminal_reward": 0.6,
            },
            0.6,
        ),
    ],
)
def test_malformed_transition_observations_preserve_base_or_terminal_reward(
    transition, expected,
):
    assert shaped_transition_reward(transition, gamma=0.9, coefficient=0.4) == expected


def test_progress_classification_is_deterministic_and_malformed_is_unknown():
    assert classify_progress({"cash": 1_000}, {"cash": 2_000}) == "progress"
    assert classify_progress({"cash": 2_000}, {"cash": 1_000}) == "no_progress"
    assert classify_progress({"cash": 1_000}, {"cash": 1_000}) == "no_progress"
    assert classify_progress(None, {"cash": 1_000}) == "unknown"


@pytest.mark.parametrize(
    "steps,window,expected",
    [(0, 3, False), (2, 3, False), (3, 3, True), (4, 3, True), (3, 0, False)],
)
def test_no_progress_window_requests_bootstrap_truncation_not_terminal_loss(
    steps, window, expected,
):
    assert should_bootstrap_truncate(steps, window) is expected


def test_reached_no_progress_window_is_bootstrap_only():
    """True means truncate while retaining value bootstrap, not terminal loss."""
    assert should_bootstrap_truncate(3, 3) is True
    assert should_bootstrap_truncate(2, 3) is False
