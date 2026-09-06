from copy import deepcopy

import pytest

from kagriculture_agent.constants import MARKET_I0, PRODUCTS
from kagriculture_agent.features import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_TOKEN_SIZE,
    MARKET_TOKEN_SIZE,
    TILE_TOKEN_SIZE,
    WORKER_TOKEN_SIZE,
    extract_features,
)
from kagriculture_agent.types import Position


def sample_state():
    return {
        "day": 3,
        "hour": 8,
        "cash": 1200,
        "board_size": 10,
        "farm": {
            "tiles": [
                [{"kind": "EMPTY"} for _ in range(10)] for _ in range(10)
            ],
            "workers": [
                {"index": 0, "role": "FARMER", "position": {"x": 2, "y": 1},
                 "task": {"kind": "HARVEST", "target": {"x": 4, "y": 5}, "deadline": 6}},
            ],
            "unlocked_land": 4,
        },
        "private": {"shed": {"WHEAT": 7}, "strategy": "melon"},
        "market": {
            "prices": {item: 10 + index for index, item in enumerate(PRODUCTS)},
            "inventory": {item: MARKET_I0 for item in PRODUCTS},
        },
        "town": {"demand": ["WHEAT"]},
    }


def test_features_have_fixed_documented_shapes_and_ordered_positions():
    features = extract_features(sample_state())

    assert features.schema_version == FEATURE_SCHEMA_VERSION
    assert len(features.tile_tokens) == 100
    assert all(len(token) == TILE_TOKEN_SIZE for token in features.tile_tokens)
    assert features.tile_positions == tuple(
        Position(x, y) for y in range(10) for x in range(10)
    )
    assert len(features.worker_tokens) == 10
    assert all(len(token) == WORKER_TOKEN_SIZE for token in features.worker_tokens)
    assert len(features.market_tokens) == len(PRODUCTS)
    assert all(len(token) == MARKET_TOKEN_SIZE for token in features.market_tokens)
    assert len(features.global_tokens) == GLOBAL_TOKEN_SIZE


def test_extraction_is_deterministic_and_does_not_mutate_state():
    state = sample_state()
    before = deepcopy(state)

    first = extract_features(state)
    second = extract_features(state)

    assert first == second
    assert state == before


def test_private_state_does_not_leak_opponent_private_state():
    state = sample_state()
    state["opponents"] = [{"private": {"cash": 999999, "secret": "hidden"}}]
    state["farms"] = [state["farm"], {"cash": 999999, "private": {"secret": "hidden"}}]

    baseline = extract_features(sample_state())
    assert extract_features(state) == baseline


def test_worker_permutations_have_identical_features_after_stable_sorting():
    state = sample_state()
    workers = [
        {"index": 3, "role": "WORKER", "position": {"x": 3, "y": 3}},
        {"index": 1, "role": "FARMER", "position": {"x": 1, "y": 1}},
        {"index": 1, "role": "WORKER", "position": {"x": 2, "y": 2}},
    ]
    state["farm"]["workers"] = workers
    permuted = deepcopy(state)
    permuted["farm"]["workers"] = [workers[2], workers[0], workers[1]]

    assert extract_features(state) == extract_features(permuted)


def test_outlier_numeric_inputs_are_clamped_to_documented_unit_bound():
    state = sample_state()
    state.update({"day": 10**100, "hour": 10**100, "cash": 10**100, "production": -10**100})
    state["farm"]["tiles"][0][0] = {
        "kind": "WHEAT", "age": 10**100, "yield_units": 10**100,
        "task": {"deadline": -10**100},
    }
    state["farm"]["workers"] = [{
        "index": 10**100, "position": {"x": 10**100, "y": -10**100},
        "task": {"target": {"x": 10**100, "y": 10**100}, "deadline": -10**100},
    }]
    state["market"]["prices"]["WHEAT"] = 10**100
    state["market"]["inventory"]["WHEAT"] = -10**100
    state["private"]["shed"]["WHEAT"] = 10**100

    features = extract_features(state)
    values = [value for group in (
        features.tile_tokens, features.worker_tokens, features.market_tokens,
        (features.global_tokens,),
    ) for token in group for value in token]
    assert values
    assert all(-1.0 <= value <= 1.0 for value in values)


def test_huge_worker_coordinates_return_finite_fixed_shape_features():
    state = sample_state()
    state["farm"]["workers"] = [{
        "index": 0,
        "position": {"x": 10**1000, "y": -(10**1000)},
    }]

    features = extract_features(state)

    assert len(features.worker_tokens) == 10
    assert all(-1.0 <= value <= 1.0 and value == value
               for token in features.worker_tokens for value in token)


def test_market_tokens_reflect_current_and_sequential_quotes():
    low = sample_state()
    high = sample_state()
    low["market"]["inventory"]["WHEAT"] = MARKET_I0 - 1000
    high["market"]["inventory"]["WHEAT"] = MARKET_I0 + 1000

    wheat_index = sorted(PRODUCTS).index("WHEAT")
    low_token = extract_features(low).market_tokens[wheat_index]
    high_token = extract_features(high).market_tokens[wheat_index]
    assert low_token != high_token
    assert low_token[10] == -1000 / MARKET_I0
    assert high_token[10] == 1000 / MARKET_I0


def test_raw_engine_observation_uses_selected_farm_only():
    selected = sample_state()
    opponent = sample_state()
    raw = {
        "player": 0,
        "day": selected["day"], "hour": selected["hour"],
        "farms": [selected["farm"], opponent["farm"]],
        "private": selected["private"], "market": selected["market"],
        "town": selected["town"],
    }
    baseline = extract_features(raw)
    raw["farms"][1]["private"] = {"shed": {"WHEAT": 10**100}, "secret": "opponent"}
    raw["private"] = {"shed": {"WHEAT": 7}, "secret": "selected"}

    assert extract_features(raw) == baseline


@pytest.mark.parametrize("malformed", [
    None, [],
    {"farm": {"tiles": "bad", "workers": {"bad": object()}}},
    {"private": [], "market": {"prices": [], "inventory": object()}, "town": {"demand": object()}},
])
def test_malformed_observations_return_finite_fixed_empty_features(malformed):
    features = extract_features(malformed)

    assert len(features.tile_tokens) == 100
    assert len(features.worker_tokens) == 10
    assert len(features.market_tokens) == len(PRODUCTS)
    for group in (features.tile_tokens, features.worker_tokens, features.market_tokens,
                  (features.global_tokens,)):
        assert all(value == value and abs(value) != float("inf")
                   for token in group for value in token)


def test_schema_mismatch_is_rejected():
    with pytest.raises(ValueError, match="schema"):
        extract_features(sample_state(), schema_version=FEATURE_SCHEMA_VERSION + 1)

    with pytest.raises(ValueError, match="schema"):
        extract_features({**sample_state(), "schema_version": FEATURE_SCHEMA_VERSION + 1})
