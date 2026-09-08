import json
import math

import pytest


torch = pytest.importorskip("torch")


def _observation(cash=1000):
    return {
        "cash": cash,
        "farm": {
            "tiles": [[{"kind": "EMPTY"} for _ in range(10)] for _ in range(10)],
            "workers": [{"index": 0, "position": {"x": 0, "y": 0}}],
        },
        "private": {"shed": {"WHEAT": 1}},
    }


def _transition(*, action=None, done=False, next_cash=1000, **extra):
    row = {
        "observation": _observation(),
        "next_observation": _observation(next_cash),
        "action": action or {"farmer": ["PASS"], "hands": [], "market": []},
        "done": done,
        "reward": 0.0,
        "final_bank": 1200,
        "opponent_final_bank": 1000,
    }
    row.update(extra)
    return row


def test_compact_policy_emits_market_active_logits_without_changing_runtime_shape():
    from kagriculture_agent.features import extract_features
    from kagriculture_agent.model import CompactPolicyNet

    outputs = CompactPolicyNet()(extract_features(_observation()))

    assert outputs["market_active_logits"].shape == (1, 2)


def test_select_outputs_ignores_inactive_worker_and_market_branches():
    from scripts.train_policy import PPOConfig, _select_outputs, build_rollout_batch

    rows = [_transition()]
    batch = build_rollout_batch(rows, config=PPOConfig())
    outputs = {
        "worker_act_logits": torch.zeros(1, 10, 2),
        "worker_target_logits": torch.zeros(1, 10, 100),
        "worker_kind_logits": torch.zeros(1, 10, 14),
        "market_active_logits": torch.zeros(1, 2),
        "market_item_logits": torch.zeros(1, 4),
        "market_quantity_logits": torch.zeros(1, 8),
        "value": torch.zeros(1),
    }

    baseline = _select_outputs(outputs, batch)
    changed = {name: value.clone() for name, value in outputs.items()}
    changed["worker_target_logits"][:, 1:, 0] = 1000.0
    changed["worker_target_logits"][:, 1:, 1:] = -1000.0
    changed["worker_kind_logits"][:, 1:, 0] = 1000.0
    changed["worker_kind_logits"][:, 1:, 1:] = -1000.0
    changed["market_item_logits"][:, 0] = 1000.0
    changed["market_item_logits"][:, 1:] = -1000.0
    changed["market_quantity_logits"][:, 0] = 1000.0
    changed["market_quantity_logits"][:, 1:] = -1000.0

    actual = _select_outputs(changed, batch)
    torch.testing.assert_close(actual[0], baseline[0])
    torch.testing.assert_close(actual[1], baseline[1])


def test_prior_regularization_ignores_inactive_worker_and_market_branches():
    from scripts.train_policy import PPOConfig, _distribution_regularization, build_rollout_batch

    batch = build_rollout_batch([_transition()], config=PPOConfig())
    current = {
        "worker_act_logits": torch.randn(1, 10, 2),
        "worker_target_logits": torch.randn(1, 10, 100),
        "worker_kind_logits": torch.randn(1, 10, 14),
        "market_active_logits": torch.randn(1, 2),
        "market_item_logits": torch.randn(1, 4),
        "market_quantity_logits": torch.randn(1, 8),
        "value": torch.zeros(1),
    }
    prior = {name: value.detach().clone() for name, value in current.items()}

    baseline = _distribution_regularization(current, prior, batch=batch)
    changed_current = {name: value.clone() for name, value in current.items()}
    changed_prior = {name: value.clone() for name, value in prior.items()}
    changed_current["worker_target_logits"][:, 1:] += 100.0
    changed_current["worker_kind_logits"][:, 1:] -= 100.0
    changed_current["market_item_logits"] += 100.0
    changed_current["market_quantity_logits"] -= 100.0
    changed_prior["worker_target_logits"][:, 1:] -= 100.0
    changed_prior["worker_kind_logits"][:, 1:] += 100.0
    changed_prior["market_item_logits"] -= 100.0
    changed_prior["market_quantity_logits"] += 100.0

    actual = _distribution_regularization(changed_current, changed_prior, batch=batch)
    torch.testing.assert_close(actual[0], baseline[0])
    torch.testing.assert_close(actual[1], baseline[1])


def test_bootstrap_truncated_rollout_uses_explicit_next_value_in_gae():
    from scripts.train_policy import generalized_advantage_estimate

    _advantages, returns = generalized_advantage_estimate(
        rewards=[0.0], values=[0.0], dones=[False], bootstrap_values=[5.0],
        gamma=0.9, gae_lambda=1.0,
    )

    assert returns == pytest.approx([4.5])


def test_middle_bootstrap_truncation_uses_its_next_value_and_breaks_gae_chain():
    from scripts.train_policy import generalized_advantage_estimate

    _advantages, returns = generalized_advantage_estimate(
        rewards=[0.0, 0.0, 0.0], values=[1.0, 2.0, 3.0],
        dones=[False, False, True], bootstrap_values=[0.0, 5.0, 0.0],
        bootstrap_truncated=[False, True, False], gamma=1.0, gae_lambda=1.0,
    )

    assert returns == pytest.approx([5.0, 5.0, 0.0])


def test_bootstrap_truncation_evaluates_next_observation_with_network():
    from scripts.train_policy import (
        PPOConfig, _bootstrap_value_estimates, build_rollout_batch,
    )

    class NextValuePolicy:
        def __call__(self, features):
            return {"value": torch.full((len(features),), 7.0)}

    config = PPOConfig(no_progress_window=2)
    row = _transition(no_progress_steps=2)
    bootstrap_values = _bootstrap_value_estimates(NextValuePolicy(), [row], config=config)
    batch = build_rollout_batch(
        [row], config=config, value_estimates=[0.0], bootstrap_values=bootstrap_values,
    )

    assert bootstrap_values == [7.0]
    assert batch.dones == [False]
    assert batch.returns == pytest.approx([config.gamma * 7.0])


def test_ppo_config_has_explicit_shaping_truncation_and_mask_defaults():
    from scripts.train_policy import PPOConfig

    config = PPOConfig()

    assert config.potential_reward_coef == 0.0
    assert config.no_progress_window == 0
    assert config.resolved_margin == 0.0
    assert config.training_action_mask is False

    with pytest.raises(ValueError, match="no_progress_window"):
        PPOConfig(no_progress_window=-1)
    with pytest.raises(ValueError, match="training_action_mask"):
        PPOConfig(training_action_mask=1)


def test_rollout_batch_adds_shaping_and_bootstraps_resolved_truncation():
    from scripts.train_policy import PPOConfig, build_rollout_batch

    config = PPOConfig(potential_reward_coef=0.5, no_progress_window=2)
    row = _transition(no_progress_steps=2)

    batch = build_rollout_batch([row], config=config, value_estimates=[0.25])

    assert batch.dones == [False]
    assert batch.truncation_count == 1
    assert batch.rewards[0] == pytest.approx(
        0.5 * (0.99 * 0.03 - 0.03), abs=1e-6,
    )


def test_opponent_pool_uses_explicit_league_sampler_when_supplied(tmp_path):
    from kagriculture_agent.league import LeagueSampler, SkillBand
    from scripts.train_policy import OpponentPool

    checkpoint = tmp_path / "historical.pt"
    checkpoint.write_text("checkpoint")
    sampler = LeagueSampler(
        probabilities={"checkpoint": 1.0},
        skill_bands={"hard": SkillBand("hard", (checkpoint,))},
    )

    pool = OpponentPool(league_sampler=sampler)
    matches = pool.schedule(count=3, seed=5)

    assert [match.checkpoint for match in matches] == [str(checkpoint)] * 3
    assert [match.seat for match in matches] == [0, 1, 0]
    assert matches[0].skill_band == "hard"


def test_opponent_pool_accepts_none_for_legacy_checkpoint_candidates():
    from scripts.train_policy import OpponentPool

    assert OpponentPool(None).checkpoint_candidates == ()


def test_run_ppo_training_preserves_a_falsy_explicit_opponent_pool():
    from scripts.train_policy import OpponentMatch, PPOConfig, run_ppo_training

    class FalsyPool:
        def __init__(self):
            self.calls = []

        def __bool__(self):
            return False

        def schedule(self, *, count, seed):
            self.calls.append((count, seed))
            return [OpponentMatch("current", 0) for _ in range(count)]

    pool = FalsyPool()

    result = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        opponent_pool=pool,
        rollout_fn=lambda **_kwargs: [_transition(done=True)],
        update_fn=lambda **_kwargs: {"updates": 0, "early_stopped": False},
    )

    assert pool.calls == [(1, 0)]
    assert result["rollout_count"] == 1


def test_run_ppo_training_telemetry_records_shaping_and_truncation_counts():
    from scripts.train_policy import PPOConfig, run_ppo_training

    events = []
    run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        rollout_fn=lambda **_kwargs: [_transition(done=True)],
        update_fn=lambda **_kwargs: {
            "updates": 1,
            "early_stopped": False,
            "shaping_count": 2,
            "truncation_count": 3,
        },
        telemetry_callback=lambda phase, payload: events.append((phase, payload)),
    )

    assert events[0][0] == "ppo"
    assert events[0][1]["shaping_count"] == 2
    assert events[0][1]["truncation_count"] == 3


def test_legacy_opponent_pool_sample_uses_seed_deterministically():
    from scripts.train_policy import OpponentPool

    pool = OpponentPool()
    first = [pool.sample(index, seed=17) for index in range(50)]
    repeated = [pool.sample(index, seed=17) for index in range(50)]
    other = [pool.sample(index, seed=18) for index in range(50)]

    assert first == repeated
    assert first != other


def test_run_ppo_training_accumulates_resume_shaping_and_truncation_counts():
    from scripts.train_policy import PPOConfig, run_ppo_training

    progress = []
    result = run_ppo_training(
        network=None, optimizer=None, transitions=[_transition(done=True)],
        ppo_steps=2, config=PPOConfig(), rollout_fn=lambda **_kwargs: [_transition(done=True)],
        initial_shaping_count=4, initial_truncation_count=5,
        update_fn=lambda **_kwargs: {
            "updates": 1, "early_stopped": False,
            "shaping_count": 2, "truncation_count": 1,
        },
        progress_fn=lambda **kwargs: progress.append(kwargs["metrics"]),
    )

    assert result["shaping_count"] == 8
    assert result["truncation_count"] == 7
    assert [item["shaping_count"] for item in progress] == [6, 8]


def test_evaluator_paired_summary_exposes_elo_evidence():
    from scripts.evaluate import paired_seed_summary

    records = []
    for seed in (1, 2):
        records.extend([
            {
                "candidate": "candidate",
                "opponent": "baseline",
                "seed": seed,
                "seat": 0,
                "outcome": "win",
                "bank_differential": 10,
                "framework_error": False,
            },
            {
                "candidate": "candidate",
                "opponent": "baseline",
                "seed": seed,
                "seat": 1,
                "outcome": "win",
                "bank_differential": 20,
                "framework_error": False,
            },
        ])

    summary = paired_seed_summary(records)

    assert summary["elo"]["ratings"]["candidate"] > summary["elo"]["ratings"]["baseline"]
    assert summary["lower_tail_bank_differential"] == pytest.approx(15.0)
