import json
import math

import pytest

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION


def _transition(final_bank=1200, opponent_final_bank=900, reward=0.0, done=False):
    return {
        "observation": {
            "day": 1,
            "hour": 1,
            "cash": 800,
            "farm": {
                "tiles": [[{"kind": "EMPTY"} for _ in range(10)] for _ in range(10)],
                "workers": [{"index": 0, "role": "FARMER", "position": {"x": 0, "y": 0}}],
            },
            "private": {"shed": {"WHEAT": 4}},
        },
        "action": {"farmer": ["EAST"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        "next_observation": {},
        "done": done,
        "reward": reward,
        "final_bank": final_bank,
        "opponent_final_bank": opponent_final_bank,
        "safety_flags": [],
    }


def test_default_ppo_config_matches_plan_values():
    from scripts.train_policy import PPOConfig

    config = PPOConfig()

    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_epsilon == 0.20
    assert config.value_coef == 0.50
    assert config.entropy_coef == 0.01
    assert config.target_kl == 0.03
    assert config.rollout_steps == 64
    assert PPOConfig(gamma=1.0).gamma == 1.0


def test_terminal_bank_margin_reward_is_normalized():
    from scripts.train_policy import terminal_bank_margin_reward

    assert terminal_bank_margin_reward(1200, 900) == pytest.approx(math.tanh(0.3))
    assert terminal_bank_margin_reward(None, 900) == 0.0
    assert terminal_bank_margin_reward(float("nan"), 900) == 0.0


def test_advantage_normalization_returns_zero_mean_unit_variance():
    from scripts.train_policy import normalize_advantages

    normalized = normalize_advantages([1.0, 2.0, 3.0])

    assert sum(normalized) == pytest.approx(0.0)
    assert math.sqrt(sum(value * value for value in normalized) / len(normalized)) == pytest.approx(1.0)
    assert normalize_advantages([5.0, 5.0]) == [0.0, 0.0]


def test_ppo_ratio_clipping_limits_objective():
    from scripts.train_policy import clipped_policy_terms

    terms = clipped_policy_terms(
        new_log_probs=[math.log(1.5), math.log(0.5)],
        old_log_probs=[0.0, 0.0],
        advantages=[1.0, -1.0],
        clip_epsilon=0.2,
    )

    assert terms["ratios"] == pytest.approx([1.5, 0.5])
    assert terms["clipped_ratios"] == pytest.approx([1.2, 0.8])
    assert terms["loss"] == pytest.approx(-(1.2 - 0.8) / 2)


def test_ppo_total_loss_includes_value_entropy_kl_and_prior_ce_terms():
    from scripts.train_policy import PPOConfig, ppo_total_loss

    config = PPOConfig(value_coef=0.5, entropy_coef=0.01, kl_coef=0.1, prior_ce_coef=0.25)

    loss = ppo_total_loss(
        policy_loss=1.0,
        value_loss=2.0,
        entropy=3.0,
        kl_to_prior=4.0,
        cross_entropy_to_prior=5.0,
        config=config,
    )

    assert loss == pytest.approx(1.0 + 0.5 * 2.0 - 0.01 * 3.0 + 0.1 * 4.0 + 0.25 * 5.0)


def test_generalized_advantage_estimate_uses_terminal_rewards_and_dones():
    from scripts.train_policy import generalized_advantage_estimate

    advantages, returns = generalized_advantage_estimate(
        rewards=[0.0, math.tanh(0.3)],
        values=[0.1, 0.2],
        dones=[False, True],
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert returns[-1] == pytest.approx(math.tanh(0.3))
    assert advantages[-1] == pytest.approx(math.tanh(0.3) - 0.2)
    assert len(advantages) == len(returns) == 2


def test_opponent_pool_probabilities_and_checkpoint_sampling_are_deterministic():
    from scripts.train_policy import OpponentPool

    pool = OpponentPool(previous_checkpoints=[f"ckpt-{index}" for index in range(7)])

    assert pool.probabilities == {
        "current": 0.40,
        "mixed": 0.15,
        "random": 0.10,
        "starter": 0.10,
        "checkpoint": 0.25,
    }
    assert sum(pool.probabilities.values()) == pytest.approx(1.0)
    assert pool.checkpoint_candidates == tuple(f"ckpt-{index}" for index in range(2, 7))

    first = [pool.sample(seed) for seed in range(20)]
    second = [pool.sample(seed) for seed in range(20)]
    assert first == second
    assert {match.seat for match in first[:2]} == {0, 1}


def test_promotion_requires_strictly_more_than_seventy_percent():
    from scripts.train_policy import should_promote

    assert should_promote(wins=71, games=100)
    assert not should_promote(wins=70, games=100)
    assert not should_promote(wins=0, games=0)


def test_behavior_cloning_smoke_writes_checkpoint_metadata(tmp_path):
    pytest.importorskip("torch")
    from scripts.train_policy import train_behavior_clone

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    rows = [_transition(done=False), _transition(done=True, reward=math.tanh(0.3))]
    input_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    metadata = train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=2,
        seed=7,
    )

    assert output_path.exists()
    assert metadata["model_version"] == "learned_v1"
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["engine_version"] == ENGINE_VERSION
    assert "action_vocab" in metadata
    assert metadata["transition_count"] == 2
