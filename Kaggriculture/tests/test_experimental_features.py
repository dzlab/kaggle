import math

from kagriculture_agent.experimental_features import (
    EXPERIMENTAL_FEATURE_VARIANT,
    extract_experimental_context,
)
from kagriculture_agent.features import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_TOKEN_SIZE,
    MARKET_TOKEN_SIZE,
    TILE_TOKEN_SIZE,
    WORKER_TOKEN_SIZE,
    extract_features,
)


def test_experimental_context_encodes_bounded_public_history_signals():
    state = {
        "day": 2,
        "hour": 8,
        "history": [
            {"action": {"type": "WATER"}, "success": False},
            {"action": {"type": "HARVEST"}, "success": True},
        ],
        "market": {
            "price_history": [{"WHEAT": 10}, {"WHEAT": 15}],
        },
        "town": {"demand_history": [{"WHEAT": 1}, {"WHEAT": 4}]},
        "recovery_slack": 7,
        "task_opportunities": ["WATER", "HARVEST"],
    }

    context = extract_experimental_context(state)

    assert context["variant"] == EXPERIMENTAL_FEATURE_VARIANT
    assert context["recent_action_identity"] != (0.0,)
    assert context["recent_action_outcome"] == 1.0
    assert context["price_trend"] > 0.0
    assert context["demand_trend"] > 0.0
    assert context["recovery_slack"] > 0.0
    assert context["task_opportunity"] > 0.0
    assert all(
        math.isfinite(value)
        for key, value in context.items()
        if key != "variant" and key != "recent_action_identity"
    )
    assert all(math.isfinite(value) for value in context["recent_action_identity"])
    assert all(-1.0 <= value <= 1.0 for value in context["recent_action_identity"])


def test_missing_or_malformed_history_is_finite_neutral_and_does_not_read_private_state():
    state = {
        "history": None,
        "private": {
            "history": [{"action": {"type": "SELL"}, "success": True}],
            "recovery_slack": 999,
        },
        "market": {"price_history": object()},
        "town": {"demand_history": object()},
        "task_opportunities": object(),
    }

    context = extract_experimental_context(state)

    assert all(value == 0.0 for value in context["recent_action_identity"])
    assert context["recent_action_outcome"] == 0.0
    assert context["price_trend"] == 0.0
    assert context["demand_trend"] == 0.0
    assert context["recovery_slack"] == 0.0
    assert context["task_opportunity"] == 0.0


def test_experimental_variant_does_not_change_production_feature_schema():
    state = {
        "farm": {"tiles": [[None for _ in range(10)] for _ in range(10)], "workers": []},
        "market": {},
        "town": {},
    }
    production = extract_features(state)

    assert EXPERIMENTAL_FEATURE_VARIANT != str(FEATURE_SCHEMA_VERSION)
    assert production.schema_version == FEATURE_SCHEMA_VERSION
    assert len(production.tile_tokens) == 100
    assert all(len(token) == TILE_TOKEN_SIZE for token in production.tile_tokens)
    assert len(production.worker_tokens) == 10
    assert all(len(token) == WORKER_TOKEN_SIZE for token in production.worker_tokens)
    assert len(production.market_tokens) > 0
    assert all(len(token) == MARKET_TOKEN_SIZE for token in production.market_tokens)
    assert len(production.global_tokens) == GLOBAL_TOKEN_SIZE
