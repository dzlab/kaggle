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


@pytest.mark.parametrize("malformed", [None, [], {"farm": {"tiles": "bad"}}, {"private": []}])
def test_malformed_observations_return_finite_fixed_empty_features(malformed):
    features = extract_features(malformed)

    assert len(features.tile_tokens) == 100
    assert len(features.worker_tokens) == 10
    assert len(features.market_tokens) == len(PRODUCTS)
    assert all(value == value and abs(value) != float("inf")
               for token in features.tile_tokens for value in token)


def test_schema_mismatch_is_rejected():
    with pytest.raises(ValueError, match="schema"):
        extract_features(sample_state(), schema_version=FEATURE_SCHEMA_VERSION + 1)

    with pytest.raises(ValueError, match="schema"):
        extract_features({**sample_state(), "schema_version": FEATURE_SCHEMA_VERSION + 1})
