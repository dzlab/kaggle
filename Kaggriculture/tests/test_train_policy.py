import json
import math

import pytest

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION


def _transition(
    final_bank=1200, opponent_final_bank=900, reward=0.0, done=False,
    farmer=None, hands=None,
):
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
        "action": {
            "farmer": farmer or ["EAST"],
            "hands": [] if hands is None else hands,
            "market": [["BUY_SEED", "WHEAT", 1]],
        },
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


def test_rollout_batch_uses_terminal_bank_reward_and_normalized_advantages():
    from scripts.train_policy import PPOConfig, build_rollout_batch

    transitions = [
        _transition(done=False, reward=99.0),
        _transition(done=True, final_bank=1200, opponent_final_bank=900),
    ]

    batch = build_rollout_batch(
        transitions,
        config=PPOConfig(),
        value_estimates=[0.1, 0.2],
        old_log_probs=[-0.3, -0.4],
    )

    assert batch.rewards == [0.0, pytest.approx(math.tanh(0.3))]
    assert batch.returns[-1] == pytest.approx(math.tanh(0.3))
    assert sum(batch.advantages) == pytest.approx(0.0)
    assert math.sqrt(sum(value * value for value in batch.advantages) / len(batch.advantages)) == pytest.approx(1.0)
    assert batch.old_log_probs == [-0.3, -0.4]


def test_worker_target_labels_derive_from_action_and_observation_positions():
    from scripts.train_policy import worker_labels

    state = _transition(farmer=["EAST"])["observation"]
    state["farm"]["workers"] = [
        {"index": 0, "role": "FARMER", "position": {"x": 0, "y": 0}},
        {"index": 1, "role": "WORKER", "position": {"x": 5, "y": 5}},
    ]

    labels = worker_labels({"farmer": ["EAST"], "hands": [["WATER"]]}, state)

    assert labels.target[0] == 1
    assert labels.target[1] == 55
    assert labels.target[:2] != [0, 0]


def test_behavior_clone_epoch_minibatches_cover_all_transitions_once_per_epoch():
    from scripts.train_policy import epoch_minibatches

    batches = epoch_minibatches(count=5, batch_size=2, seed=7, epoch=0)

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3, 4]
    assert all(1 <= len(batch) <= 2 for batch in batches)
    assert batches == epoch_minibatches(count=5, batch_size=2, seed=7, epoch=0)
    assert batches != epoch_minibatches(count=5, batch_size=2, seed=7, epoch=1)


def test_ppo_update_requires_torch_but_is_import_safe():
    from scripts.train_policy import PPOConfig, ppo_update

    pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet

    torch = pytest.importorskip("torch")
    network = CompactPolicyNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    metrics = ppo_update(
        network,
        optimizer,
        [_transition(done=False), _transition(done=True)],
        config=PPOConfig(target_kl=100.0),
    )

    assert metrics["updates"] >= 1
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "kl_to_prior" in metrics
    assert "prior_cross_entropy" in metrics


def test_ppo_update_loads_prior_checkpoint_regularization(tmp_path):
    pytest.importorskip("torch")
    import torch
    from kagriculture_agent.model import CompactPolicyNet
    from scripts.train_policy import PPOConfig, ppo_update

    prior = CompactPolicyNet()
    prior_path = tmp_path / "prior.pt"
    torch.save({"model_state_dict": prior.state_dict()}, prior_path)
    network = CompactPolicyNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)

    metrics = ppo_update(
        network,
        optimizer,
        [_transition(done=False), _transition(done=True)],
        config=PPOConfig(target_kl=100.0),
        prior_checkpoint=prior_path,
    )

    assert metrics["kl_to_prior"] >= 0.0
    assert metrics["prior_cross_entropy"] > 0.0


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


def test_opponent_pool_schedule_uses_requested_probabilities_and_uniform_checkpoints():
    from collections import Counter

    from scripts.train_policy import OpponentPool

    pool = OpponentPool(previous_checkpoints=[f"ckpt-{index}" for index in range(7)])
    schedule = pool.schedule(count=100, seed=11)
    opponent_counts = Counter(match.opponent for match in schedule)
    checkpoint_counts = Counter(match.checkpoint for match in schedule if match.checkpoint)

    assert opponent_counts == {
        "current": 40,
        "mixed": 15,
        "random": 10,
        "starter": 10,
        "checkpoint": 25,
    }
    assert set(checkpoint_counts) == set(pool.checkpoint_candidates)
    assert set(checkpoint_counts.values()) == {5}
    assert [match.seat for match in schedule[:8]] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert schedule == pool.schedule(count=100, seed=11)


def test_promotion_requires_strictly_more_than_seventy_percent():
    from scripts.train_policy import PROMOTION_MATCH_SIZE, should_promote

    assert PROMOTION_MATCH_SIZE == 100
    assert should_promote(wins=71, games=PROMOTION_MATCH_SIZE)
    assert not should_promote(wins=70, games=PROMOTION_MATCH_SIZE)
    assert not should_promote(wins=1, games=1)
    assert not should_promote(wins=72, games=99)
    assert not should_promote(wins=72, games=101)
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
    assert metadata["behavior_clone_epochs"] == 1
    assert metadata["behavior_clone_updates"] == 1
    assert metadata["ppo_updates"] == 0
