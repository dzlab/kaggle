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


@pytest.mark.parametrize("field,value", [
    ("gamma", 0.0),
    ("gamma", 1.1),
    ("gae_lambda", -0.1),
    ("gae_lambda", 1.1),
    ("clip_epsilon", 0.0),
    ("value_coef", -0.1),
    ("entropy_coef", -0.1),
    ("target_kl", 0.0),
    ("rollout_steps", 0),
    ("kl_coef", -0.1),
    ("prior_ce_coef", -0.1),
    ("ppo_epochs", 0),
])
def test_ppo_config_validates_every_field_range(field, value):
    from scripts.train_policy import PPOConfig

    with pytest.raises(ValueError, match=field):
        PPOConfig(**{field: value})


def test_parser_rejects_negative_ppo_steps():
    from scripts.train_policy import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args([
            "--input", "transitions.jsonl",
            "--output", "policy.pt",
            "--ppo-steps", "-1",
        ])


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_advantage_normalization_rejects_nonfinite_values(bad):
    from scripts.train_policy import normalize_advantages

    with pytest.raises(ValueError, match="finite"):
        normalize_advantages([1.0, bad])


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


def test_ppo_ratio_clipping_clamps_extreme_log_ratios_and_rejects_nonfinite():
    from scripts.train_policy import LOG_RATIO_CLAMP, clipped_policy_terms

    terms = clipped_policy_terms(
        new_log_probs=[1000.0, -1000.0],
        old_log_probs=[0.0, 0.0],
        advantages=[1.0, -1.0],
        clip_epsilon=0.2,
    )

    assert all(math.isfinite(value) for value in terms["ratios"])
    assert terms["ratios"] == pytest.approx([math.exp(LOG_RATIO_CLAMP), math.exp(-LOG_RATIO_CLAMP)])
    assert terms["clipped_ratios"] == pytest.approx([1.2, 0.8])

    with pytest.raises(ValueError, match="finite"):
        clipped_policy_terms(
            new_log_probs=[float("nan")],
            old_log_probs=[0.0],
            advantages=[1.0],
        )


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


@pytest.mark.parametrize("kwargs", [
    {"rewards": [float("nan")], "values": [0.0], "dones": [True]},
    {"rewards": [0.0], "values": [float("inf")], "dones": [True]},
    {"rewards": [0.0], "values": [0.0], "dones": [True], "gamma": float("nan")},
    {"rewards": [0.0], "values": [0.0], "dones": [True], "gae_lambda": float("inf")},
])
def test_generalized_advantage_estimate_rejects_nonfinite_inputs(kwargs):
    from scripts.train_policy import generalized_advantage_estimate

    with pytest.raises(ValueError, match="finite"):
        generalized_advantage_estimate(**kwargs)


def test_approximate_kl_rejects_nonfinite_inputs():
    from scripts.train_policy import approximate_kl

    with pytest.raises(ValueError, match="finite"):
        approximate_kl([0.0], [float("nan")])


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


def test_ppo_update_measures_target_kl_after_optimizer_step():
    pytest.importorskip("torch")
    import torch
    from scripts.train_policy import PPOConfig, ppo_update

    class OneParameterPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, _features):
            batch = 2
            worker_act = torch.stack((torch.zeros(10, 2), torch.zeros(10, 2))).clone()
            worker_act[:, :, 1] = self.bias
            return {
                "worker_act_logits": worker_act,
                "worker_target_logits": torch.zeros(batch, 10, 100),
                "worker_kind_logits": torch.zeros(batch, 10, 14),
                "market_item_logits": torch.zeros(batch, 9),
                "market_quantity_logits": torch.zeros(batch, 8),
                "value": self.bias.repeat(batch),
            }

    network = OneParameterPolicy()
    optimizer = torch.optim.SGD(network.parameters(), lr=20.0)

    metrics = ppo_update(
        network,
        optimizer,
        [_transition(done=False), _transition(done=True, final_bank=2000, opponent_final_bank=0)],
        config=PPOConfig(target_kl=1e-12, ppo_epochs=3),
        batch_size=2,
    )

    assert metrics["updates"] == 1
    assert metrics["early_stopped"] is True
    assert metrics["approx_kl"] > 1e-12


def test_ppo_update_rejects_nonfinite_logits_before_optimizer_step():
    pytest.importorskip("torch")
    import torch
    from scripts.train_policy import PPOConfig, ppo_update

    class BadPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, _features):
            return {
                "worker_act_logits": torch.full((2, 10, 2), float("nan")) + self.bias,
                "worker_target_logits": torch.zeros(2, 10, 100),
                "worker_kind_logits": torch.zeros(2, 10, 14),
                "market_item_logits": torch.zeros(2, 9),
                "market_quantity_logits": torch.zeros(2, 8),
                "value": torch.zeros(2) + self.bias,
            }

    network = BadPolicy()
    optimizer = torch.optim.SGD(network.parameters(), lr=1.0)

    with pytest.raises(ValueError, match="finite"):
        ppo_update(
            network,
            optimizer,
            [_transition(done=False), _transition(done=True)],
            config=PPOConfig(),
        )


def test_ppo_update_loads_prior_checkpoint_regularization(tmp_path):
    pytest.importorskip("torch")
    import torch
    from kagriculture_agent.model import CompactPolicyNet
    from scripts.train_policy import PPOConfig, ppo_update

    prior = CompactPolicyNet()
    prior_path = tmp_path / "prior.pt"
    from scripts.train_policy import checkpoint_metadata

    torch.save({"metadata": checkpoint_metadata(transition_count=2), "model_state_dict": prior.state_dict()}, prior_path)
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


def test_prior_checkpoint_metadata_validation_accepts_current_schema():
    from scripts.train_policy import checkpoint_metadata, validate_prior_checkpoint_metadata

    metadata = checkpoint_metadata(transition_count=2)

    assert validate_prior_checkpoint_metadata(metadata) is None


@pytest.mark.parametrize("field,value", [
    ("model_version", "old"),
    ("feature_schema_version", -1),
    ("engine_version", "0.0.0"),
])
def test_prior_checkpoint_metadata_validation_rejects_incompatible_versions(field, value):
    from scripts.train_policy import checkpoint_metadata, validate_prior_checkpoint_metadata

    metadata = checkpoint_metadata(transition_count=2)
    metadata[field] = value

    with pytest.raises(ValueError, match=field):
        validate_prior_checkpoint_metadata(metadata)


def test_prior_checkpoint_metadata_validation_rejects_action_vocab_mismatch():
    from scripts.train_policy import checkpoint_metadata, validate_prior_checkpoint_metadata

    metadata = checkpoint_metadata(transition_count=2)
    metadata["action_vocab"]["worker_kinds"] = ["PASS"]

    with pytest.raises(ValueError, match="action_vocab"):
        validate_prior_checkpoint_metadata(metadata)


def test_ppo_rollouts_require_callback_unless_offline_fallback_is_explicit():
    from scripts.train_policy import PPOConfig, run_ppo_training

    with pytest.raises(ValueError, match="rollout_fn.*offline_ppo_fallback"):
        run_ppo_training(
            network=None,
            optimizer=None,
            transitions=[_transition(done=True)],
            ppo_steps=1,
            config=PPOConfig(),
        )


def test_ppo_rollouts_call_callback_once_per_step_with_scheduled_match():
    from scripts.train_policy import OpponentMatch, PPOConfig, run_ppo_training

    calls = []

    class Pool:
        def schedule(self, *, count, seed):
            calls.append(("schedule", count, seed))
            return [
                OpponentMatch("current", 0),
                OpponentMatch("checkpoint", 1, "ckpt-b"),
                OpponentMatch("mixed", 0),
            ]

    def rollout_fn(*, step, opponent, seat, checkpoint, rollout_steps):
        calls.append(("rollout", step, opponent, seat, checkpoint, rollout_steps))
        return [_transition(done=True, final_bank=1000 + step, opponent_final_bank=900)]

    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[],
        ppo_steps=3,
        config=PPOConfig(rollout_steps=5),
        opponent_pool=Pool(),
        rollout_fn=rollout_fn,
        seed=23,
        update_fn=lambda **kwargs: {"updates": 1, "early_stopped": False},
    )

    assert calls == [
        ("schedule", 3, 23),
        ("rollout", 0, "current", 0, None, 5),
        ("rollout", 1, "checkpoint", 1, "ckpt-b", 5),
        ("rollout", 2, "mixed", 0, None, 5),
    ]
    assert metrics["ppo_updates"] == 3
    assert metrics["rollout_count"] == 3


def test_ppo_offline_fallback_is_explicit_and_reuses_collected_rollout():
    from scripts.train_policy import PPOConfig, run_ppo_training

    calls = []
    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=False), _transition(done=True)],
        ppo_steps=2,
        config=PPOConfig(rollout_steps=1),
        offline_ppo_fallback=True,
        update_fn=lambda transitions, **kwargs: calls.append(list(transitions)) or {"updates": 1, "early_stopped": False},
    )

    assert len(calls) == 2
    assert all(len(call) == 1 for call in calls)
    assert metrics["ppo_updates"] == 2


def test_cli_ppo_steps_require_explicit_offline_fallback_for_replay_reuse():
    from scripts.train_policy import _cli_training_options, _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--ppo-steps", "2",
    ])

    with pytest.raises(ValueError, match="--offline-ppo-fallback"):
        _cli_training_options(args)

    explicit = _parser().parse_args([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--ppo-steps", "2",
        "--offline-ppo-fallback",
    ])
    assert _cli_training_options(explicit)["offline_ppo_fallback"] is True


def test_promotion_match_runs_exactly_fixed_gate_and_counts_wins():
    from scripts.train_policy import PROMOTION_MATCH_SIZE, run_promotion_match

    calls = []

    def match_fn(index):
        calls.append(index)
        return {"winner": "candidate" if index < 71 else "opponent"}

    result = run_promotion_match(match_fn)

    assert calls == list(range(PROMOTION_MATCH_SIZE))
    assert result == {"games": 100, "wins": 71, "promoted": True}


def test_promotion_match_rejects_wrong_match_size():
    from scripts.train_policy import run_promotion_match

    with pytest.raises(ValueError, match="exactly 100"):
        run_promotion_match(lambda index: {"winner": "candidate"}, match_size=99)


@pytest.mark.parametrize("result", [
    None,
    "candidate",
    {"winner": "tie"},
    {"winner": None},
    {"candidate_win": "yes"},
    {"candidate_win": 1},
    {"candidate_win": None},
    {"other": "candidate"},
])
def test_promotion_match_rejects_malformed_or_nonbinary_results(result):
    from scripts.train_policy import run_promotion_match

    def match_fn(_index):
        return result

    with pytest.raises(ValueError, match="promotion match result"):
        run_promotion_match(match_fn)


def test_checkpoint_promotion_runner_uses_fixed_match_before_promoting():
    from scripts.train_policy import maybe_promote_checkpoint

    calls = []
    registry = {"best": "previous.pt", "candidates": []}

    def match_fn(index, *, candidate_checkpoint, best_checkpoint, registry_entry):
        calls.append((index, candidate_checkpoint, best_checkpoint, registry_entry))
        return {"candidate_win": index < 71}

    result = maybe_promote_checkpoint(
        match_fn=match_fn,
        candidate_checkpoint="candidate.pt",
        registry=registry,
    )

    assert [call[0] for call in calls] == list(range(100))
    assert {call[1] for call in calls} == {"candidate.pt"}
    assert {call[2] for call in calls} == {"previous.pt"}
    assert all(call[3]["path"] == "candidate.pt" and call[3]["status"] == "candidate" for call in calls)
    assert result == {
        "candidate_checkpoint": "candidate.pt",
        "games": 100,
        "wins": 71,
        "promoted": True,
    }
    assert registry["best"] == "candidate.pt"
    assert registry["candidates"] == [{"path": "candidate.pt", "status": "promoted"}]


def test_checkpoint_promotion_retains_best_and_cleans_candidate_on_rejection():
    from scripts.train_policy import maybe_promote_checkpoint

    registry = {"best": "previous.pt", "candidates": []}
    cleaned = []

    def match_fn(index, **_kwargs):
        return {"candidate_win": index < 70}

    result = maybe_promote_checkpoint(
        match_fn=match_fn,
        candidate_checkpoint="candidate.pt",
        registry=registry,
        cleanup_candidate_fn=cleaned.append,
    )

    assert result["promoted"] is False
    assert result["wins"] == 70
    assert registry["best"] == "previous.pt"
    assert registry["candidates"] == [{"path": "candidate.pt", "status": "rejected"}]
    assert cleaned == ["candidate.pt"]


def test_checkpoint_promotion_cleans_candidate_and_retains_best_on_match_error():
    from scripts.train_policy import maybe_promote_checkpoint

    registry = {"best": "previous.pt", "candidates": []}
    cleaned = []

    def match_fn(_index, **_kwargs):
        return {"winner": "draw"}

    with pytest.raises(ValueError, match="promotion match result"):
        maybe_promote_checkpoint(
            match_fn=match_fn,
            candidate_checkpoint="candidate.pt",
            registry=registry,
            cleanup_candidate_fn=cleaned.append,
        )

    assert registry["best"] == "previous.pt"
    assert registry["candidates"] == [{"path": "candidate.pt", "status": "error"}]
    assert cleaned == ["candidate.pt"]


def test_cli_main_reports_oserror_without_traceback(monkeypatch, capsys):
    from scripts import train_policy

    def fail(**_kwargs):
        raise OSError("cannot write checkpoint")

    monkeypatch.setattr(train_policy, "train_behavior_clone", fail)

    assert train_policy.main([
        "--input", "missing.jsonl",
        "--output", "policy.pt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "cannot write checkpoint"
    assert "Traceback" not in captured.err


def test_cli_main_reports_ambiguous_ppo_mode_without_traceback(capsys):
    from scripts import train_policy

    assert train_policy.main([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--ppo-steps", "1",
    ]) == 2
    captured = capsys.readouterr()
    assert "--offline-ppo-fallback" in captured.err
    assert "Traceback" not in captured.err


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
