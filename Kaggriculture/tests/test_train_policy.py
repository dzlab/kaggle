import hashlib
import json
import math
import random
from pathlib import Path

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


def _resume_configuration(
    input_path: Path, *, seed: int = 7, ppo_steps: int = 0,
    device: str = "cpu", prior_checkpoint=None,
):
    from scripts import train_policy

    digest = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.is_file() else "0" * 64
    return {
        "input_trajectory": {"path": str(input_path.resolve()), "sha256": digest},
        "steps": 1,
        "batch_size": 1,
        "seed": seed,
        "ppo_steps": ppo_steps,
        "device": device,
        "prior_checkpoint": prior_checkpoint,
        "offline_ppo_fallback": False,
        "ppo_config": train_policy.asdict(train_policy.PPOConfig()),
        "checkpoint_interval": 100,
    }


def _ppo_checkpoint_metrics(*, completed_steps=1, ppo_updates=4, rollout_count=1):
    return {
        "ppo_updates": ppo_updates,
        "rollout_count": rollout_count,
        "early_stopped": False,
        "last_metrics": None,
        "promotion": None,
        "completed_steps": completed_steps,
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


def test_parser_accepts_supported_device_names_and_rejects_invalid_name():
    from scripts.train_policy import _parser

    for requested in ("auto", "cpu", "cuda"):
        args = _parser().parse_args([
            "--input", "transitions.jsonl",
            "--output", "policy.pt",
            "--device", requested,
        ])
        assert args.device == requested

    with pytest.raises(SystemExit):
        _parser().parse_args([
            "--input", "transitions.jsonl",
            "--output", "policy.pt",
            "--device", "mps",
        ])


def test_parser_accepts_resume_checkpoint_path():
    from scripts.train_policy import _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--resume", "previous.pt",
    ])

    assert args.resume_checkpoint == Path("previous.pt")


def test_parser_accepts_positive_checkpoint_interval_and_rejects_zero():
    from scripts.train_policy import _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--checkpoint-interval", "7",
    ])

    assert args.checkpoint_interval == 7
    with pytest.raises(SystemExit):
        _parser().parse_args([
            "--input", "transitions.jsonl",
            "--output", "policy.pt",
            "--checkpoint-interval", "0",
        ])


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        ("missing_prior", "prior_checkpoint"),
        ("boolean_seed", "seed"),
        ("float_batch_size", "batch_size"),
        ("float_rollout_steps", "rollout_steps"),
        ("string_prior", "prior_checkpoint"),
        ("different_steps", "steps"),
        ("extra_field", "unexpected"),
    ],
)
def test_resume_configuration_validation_is_strict(case, error_match):
    from scripts.train_policy import _validate_resume_configuration

    requested = _resume_configuration(Path("transitions.jsonl"), seed=0)
    saved = {**requested, "ppo_config": dict(requested["ppo_config"])}
    if case == "missing_prior":
        del saved["prior_checkpoint"]
    elif case == "boolean_seed":
        saved["seed"] = False
    elif case == "float_batch_size":
        saved["batch_size"] = 1.0
    elif case == "float_rollout_steps":
        saved["ppo_config"]["rollout_steps"] = 64.0
    elif case == "string_prior":
        saved["prior_checkpoint"] = "prior.pt"
        requested["prior_checkpoint"] = "prior.pt"
    elif case == "different_steps":
        saved["steps"] = 2
    else:
        saved["unexpected"] = True

    with pytest.raises(ValueError, match=error_match):
        _validate_resume_configuration(saved, requested)


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


def test_ppo_update_places_label_and_loss_tensors_on_requested_device(monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    requested_device = object()
    tensor_devices = []
    real_tensor = torch.tensor

    class TorchProxy:
        def tensor(self, *args, **kwargs):
            tensor_devices.append(kwargs.get("device"))
            if kwargs.get("device") is requested_device:
                kwargs["device"] = torch.device("cpu")
            return real_tensor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(torch, name)

    network = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    monkeypatch.setattr(train_policy, "require_torch", lambda: TorchProxy())

    train_policy.ppo_update(
        network,
        optimizer,
        [_transition(done=False), _transition(done=True)],
        config=train_policy.PPOConfig(target_kl=100.0, ppo_epochs=1),
        device=requested_device,
    )

    assert tensor_devices
    assert set(tensor_devices) == {requested_device}


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


def test_prior_network_is_loaded_on_requested_device(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    prior = CompactPolicyNet()
    prior_path = tmp_path / "prior.pt"
    torch.save({
        "metadata": train_policy.checkpoint_metadata(transition_count=2),
        "model_state_dict": prior.state_dict(),
    }, prior_path)
    moves = []
    load_kwargs = []
    real_load = torch.load

    class TorchProxy:
        def load(self, *args, **kwargs):
            load_kwargs.append(dict(kwargs))
            return real_load(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(torch, name)

    class RecordingPolicy(CompactPolicyNet):
        def to(self, device):
            moves.append(torch.device(device))
            return super().to(device)

    monkeypatch.setattr(train_policy, "CompactPolicyNet", RecordingPolicy)
    monkeypatch.setattr(train_policy, "require_torch", lambda: TorchProxy())

    loaded = train_policy._load_prior_network(prior_path, device=torch.device("cpu"))

    assert moves == [torch.device("cpu")]
    assert load_kwargs == [{"map_location": "cpu", "weights_only": True}]
    assert next(loaded.parameters()).device == torch.device("cpu")


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


def test_ppo_training_propagates_requested_device_to_updater():
    from scripts.train_policy import PPOConfig, run_ppo_training

    requested_device = object()
    seen_devices = []

    run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        device=requested_device,
        offline_ppo_fallback=True,
        update_fn=lambda **kwargs: (
            seen_devices.append(kwargs["device"])
            or {"updates": 1, "early_stopped": False}
        ),
    )

    assert seen_devices == [requested_device]


def test_ppo_training_preserves_legacy_exact_signature_updater():
    from scripts.train_policy import PPOConfig, run_ppo_training

    calls = []

    def legacy_update(
        *, network, optimizer, transitions, config, batch_size, seed,
        prior_checkpoint,
    ):
        calls.append((network, optimizer, transitions, config, batch_size, seed, prior_checkpoint))
        return {"updates": 1, "early_stopped": False}

    metrics = run_ppo_training(
        network="network",
        optimizer="optimizer",
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        device=object(),
        offline_ppo_fallback=True,
        update_fn=legacy_update,
    )

    assert metrics["ppo_updates"] == 1
    assert len(calls) == 1


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


def test_ppo_promotion_callback_reads_saved_candidate_and_persists_best_on_acceptance(tmp_path):
    from scripts.train_policy import PPOConfig, run_ppo_training

    candidate = tmp_path / "candidate.pt"
    best = tmp_path / "best.pt"
    best.write_text("previous-best", encoding="utf-8")
    registry = {"best": str(best), "candidates": []}
    seen = []

    def save_candidate(path):
        Path = type(path)
        target = Path(path)
        target.write_text("candidate-after-ppo", encoding="utf-8")
        return target

    def match_fn(index, *, candidate_checkpoint, best_checkpoint, registry_entry):
        seen.append((
            index,
            Path(candidate_checkpoint).read_text(encoding="utf-8"),
            Path(best_checkpoint).read_text(encoding="utf-8"),
            registry_entry["status"],
        ))
        return {"candidate_win": index < 71}

    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
        promotion_match_fn=match_fn,
        candidate_checkpoint=candidate,
        best_checkpoint_path=best,
        checkpoint_registry=registry,
        save_candidate_fn=save_candidate,
    )

    assert len(seen) == 100
    assert {item[1] for item in seen} == {"candidate-after-ppo"}
    assert {item[2] for item in seen} == {"previous-best"}
    assert {item[3] for item in seen} == {"candidate"}
    assert metrics["promotion"]["promoted"] is True
    assert best.read_text(encoding="utf-8") == "candidate-after-ppo"
    assert registry["best"] == str(best)


def test_ppo_promotion_retains_durable_best_and_cleans_candidate_on_rejection(tmp_path):
    from scripts.train_policy import PPOConfig, run_ppo_training

    candidate = tmp_path / "candidate.pt"
    best = tmp_path / "best.pt"
    best.write_text("previous-best", encoding="utf-8")
    registry = {"best": str(best), "candidates": []}

    def save_candidate(path):
        path.write_text("candidate-after-ppo", encoding="utf-8")
        return path

    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
        promotion_match_fn=lambda index, **_kwargs: {"candidate_win": index < 70},
        candidate_checkpoint=candidate,
        best_checkpoint_path=best,
        checkpoint_registry=registry,
        save_candidate_fn=save_candidate,
    )

    assert metrics["promotion"]["promoted"] is False
    assert best.read_text(encoding="utf-8") == "previous-best"
    assert not candidate.exists()
    assert registry["best"] == str(best)
    assert registry["candidates"][0]["path"] != str(candidate)
    assert registry["candidates"][0]["status"] == "rejected"
    assert not Path(registry["candidates"][0]["path"]).exists()


def test_ppo_promotion_uses_temp_candidate_and_preserves_output_on_match_error(tmp_path):
    from scripts.train_policy import PPOConfig, run_ppo_training

    output = tmp_path / "policy.pt"
    output.write_text("existing-output", encoding="utf-8")
    best = tmp_path / "best.pt"
    best.write_text("previous-best", encoding="utf-8")
    saved = []

    def save_candidate(path, *, ppo_metrics):
        candidate_path = Path(path)
        saved.append((candidate_path, ppo_metrics["ppo_updates"]))
        candidate_path.write_text("candidate-after-ppo", encoding="utf-8")
        return candidate_path

    def match_fn(_index, *, candidate_checkpoint, **_kwargs):
        assert Path(candidate_checkpoint) != output
        assert Path(candidate_checkpoint).parent == output.parent
        assert Path(candidate_checkpoint).read_text(encoding="utf-8") == "candidate-after-ppo"
        raise ValueError("promotion match result forced failure")

    with pytest.raises(ValueError, match="forced failure"):
        run_ppo_training(
            network=None,
            optimizer=None,
            transitions=[_transition(done=True)],
            ppo_steps=1,
            config=PPOConfig(),
            offline_ppo_fallback=True,
            update_fn=lambda **_kwargs: {"updates": 3, "early_stopped": False, "loss": 1.5},
            promotion_match_fn=match_fn,
            candidate_checkpoint=output,
            best_checkpoint_path=best,
            save_candidate_fn=save_candidate,
        )

    assert output.read_text(encoding="utf-8") == "existing-output"
    assert best.read_text(encoding="utf-8") == "previous-best"
    assert saved and saved[0][0] != output
    assert saved[0][1] == 3
    assert not saved[0][0].exists()


def test_ppo_promotion_publishes_current_metadata_to_output_and_best(tmp_path):
    from scripts.train_policy import PPOConfig, run_ppo_training

    output = tmp_path / "policy.pt"
    best = tmp_path / "best.pt"
    best.write_text("previous-best", encoding="utf-8")

    def save_candidate(path, *, ppo_metrics):
        Path(path).write_text(json.dumps({
            "metadata": {
                "ppo_updates": ppo_metrics["ppo_updates"],
                "last_loss": ppo_metrics["last_metrics"]["loss"],
            }
        }), encoding="utf-8")
        return path

    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **_kwargs: {"updates": 4, "early_stopped": False, "loss": 2.25},
        promotion_match_fn=lambda index, **_kwargs: {"candidate_win": index < 71},
        candidate_checkpoint=output,
        best_checkpoint_path=best,
        save_candidate_fn=save_candidate,
    )

    assert metrics["promotion"]["promoted"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"] == {
        "ppo_updates": 4,
        "last_loss": 2.25,
    }
    assert json.loads(best.read_text(encoding="utf-8"))["metadata"] == {
        "ppo_updates": 4,
        "last_loss": 2.25,
    }


def test_ppo_registry_only_promotion_persists_loadable_best(tmp_path):
    from scripts.train_policy import PPOConfig, run_ppo_training

    output = tmp_path / "policy.pt"
    registry = {"best": None, "candidates": []}

    def save_candidate(path, *, ppo_metrics):
        Path(path).write_text(json.dumps({
            "metadata": {"ppo_updates": ppo_metrics["ppo_updates"]},
            "model_state_dict": {"weights": [1, 2, 3]},
        }), encoding="utf-8")
        return path

    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **_kwargs: {"updates": 5, "early_stopped": False},
        promotion_match_fn=lambda index, **_kwargs: {"candidate_win": index < 71},
        candidate_checkpoint=output,
        checkpoint_registry=registry,
        save_candidate_fn=save_candidate,
    )

    registry_best = Path(registry["best"])
    assert metrics["promotion"]["promoted"] is True
    assert registry_best != output
    assert registry_best != Path(metrics["promotion"]["candidate_checkpoint"])
    assert registry_best.exists()
    assert json.loads(registry_best.read_text(encoding="utf-8"))["metadata"]["ppo_updates"] == 5
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"]["ppo_updates"] == 5


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


def test_opponent_pool_mixed_matches_resolve_to_a_real_non_current_variant():
    from scripts.train_policy import OpponentPool

    pool = OpponentPool(previous_checkpoints=[f"ckpt-{index}" for index in range(5)])

    sampled = [pool.sample(index) for index in range(100)]
    mixed = [match for match in sampled if match.opponent == "mixed"]

    assert mixed
    assert all(match.mixed_opponent in {"current", "random", "starter"} for match in mixed)
    assert {match.mixed_opponent for match in mixed} >= {"random", "starter"}
    assert sampled == [pool.sample(index) for index in range(100)]


def test_ppo_rollout_callback_receives_mixed_opponent_resolution_and_network():
    from scripts.train_policy import OpponentMatch, PPOConfig, run_ppo_training

    network = object()
    seen = []

    class Pool:
        def schedule(self, *, count, seed):
            return [OpponentMatch("mixed", 0, None, "starter")]

    def rollout_fn(*, step, opponent, seat, checkpoint, rollout_steps,
                   mixed_opponent, network):
        seen.append((step, opponent, seat, checkpoint, rollout_steps, mixed_opponent, network))
        return [_transition(done=True)]

    run_ppo_training(
        network=network,
        optimizer=None,
        transitions=[],
        ppo_steps=1,
        config=PPOConfig(rollout_steps=5),
        opponent_pool=Pool(),
        rollout_fn=rollout_fn,
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
    )

    assert seen == [(0, "mixed", 0, None, 5, "starter", network)]


def test_fresh_rollout_callback_can_refresh_candidate_artifact_per_round(tmp_path, monkeypatch):
    from scripts import train_policy

    collected = []

    def fake_collect(*, output, candidate_artifact, opponents, **kwargs):
        collected.append({
            "output": Path(output),
            "candidate_artifact": Path(candidate_artifact),
            "opponent": opponents[0],
            **kwargs,
        })
        Path(output).write_text(json.dumps({"step": len(collected)}) + "\n", encoding="utf-8")

    monkeypatch.setattr("scripts.collect_trajectories.collect", fake_collect)
    updates = []

    def update_candidate(*, network, output_path, candidate_artifact, step, round_index):
        updates.append((network, Path(output_path), Path(candidate_artifact), step, round_index))
        Path(output_path).write_text("refreshed", encoding="utf-8")
        return output_path

    network = object()
    rollout_fn = train_policy.make_fresh_rollout_fn(
        run_directory=tmp_path,
        candidate_artifact=None,
        candidate_artifact_callback=update_candidate,
        seeds=[41],
        steps=4,
    )

    first = rollout_fn(
        step=0, round_index=0, seed=41, opponent="mixed", mixed_opponent="starter",
        seat=1, checkpoint=None, rollout_steps=3, network=network,
    )
    second = rollout_fn(
        step=1, round_index=1, seed=42, opponent="mixed", mixed_opponent="random",
        seat=0, checkpoint=None, rollout_steps=3, network=network,
    )

    assert [row["step"] for row in first + second] == [1, 2]
    assert [item[0] for item in updates] == [network, network]
    assert updates[0][1] == updates[0][2]
    assert updates[0][1] != updates[1][1]
    assert [item["candidate_artifact"] for item in collected] == [item[1] for item in updates]
    assert [item["opponent"] for item in collected] == ["starter", "random"]


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
        device="cpu",
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
    assert metadata["device"] == "cpu"

    checkpoint = pytest.importorskip("torch").load(output_path, map_location="cpu")
    assert checkpoint["metadata"]["device"] == "cpu"
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["configuration"]["device"] == "cpu"
    assert checkpoint["progress"] == {"epoch": 1, "round": 0, "cursor": 0}
    assert checkpoint["metrics"]["behavior_clone_updates"] == 1


def test_behavior_cloning_switches_to_conservative_ppo_learning_rate(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    observed = {}

    def observe_ppo(**kwargs):
        observed["learning_rate"] = kwargs["optimizer"].param_groups[0]["lr"]
        return {
            "ppo_updates": 0,
            "rollout_count": 0,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
            "completed_steps": 1,
        }

    monkeypatch.setattr(train_policy, "run_ppo_training", observe_ppo)
    train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        ppo_steps=1,
        device="cpu",
    )

    assert observed["learning_rate"] == train_policy.PPO_LEARNING_RATE


def test_behavior_cloning_periodic_checkpoint_survives_interruption_and_resumes(
    tmp_path, monkeypatch,
):
    torch = pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    rows = [_transition(reward=float(index)) for index in range(3)]
    input_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    real_save_checkpoint = train_policy.save_checkpoint

    def interrupt_after_first_progress_save(path, **kwargs):
        saved = real_save_checkpoint(path, **kwargs)
        if kwargs["epoch"] == 0 and kwargs["cursor"] == 1:
            raise KeyboardInterrupt
        return saved

    monkeypatch.setattr(train_policy, "save_checkpoint", interrupt_after_first_progress_save)
    with pytest.raises(KeyboardInterrupt):
        train_policy.train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            seed=7,
            device="cpu",
            checkpoint_interval=1,
        )

    interrupted = torch.load(output_path, map_location="cpu", weights_only=True)
    assert interrupted["progress"] == {"epoch": 0, "round": 0, "cursor": 1}
    assert interrupted["metrics"] == {
        "behavior_clone_updates": 1,
        "ppo_updates": 0,
        "ppo_metrics": None,
    }

    monkeypatch.setattr(train_policy, "save_checkpoint", real_save_checkpoint)
    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        device="cpu",
        resume_checkpoint=output_path,
        checkpoint_interval=1,
    )

    resumed = torch.load(output_path, map_location="cpu", weights_only=True)
    assert resumed["progress"] == {"epoch": 1, "round": 0, "cursor": 0}
    assert metadata["behavior_clone_updates"] == 3


def test_training_checkpoint_records_input_trajectory_content_identity(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")

    train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        device="cpu",
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert payload["configuration"]["input_trajectory"] == {
        "path": str(input_path.resolve()),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }


def test_resume_rejects_changed_input_trajectory_with_same_row_count(tmp_path):
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=resume_path,
        steps=1,
        batch_size=1,
        seed=7,
        device="cpu",
    )
    input_path.write_text(
        json.dumps(_transition(done=True, farmer=["WEST"])) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input trajectory"):
        train_policy.train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            seed=7,
            device="cpu",
            resume_checkpoint=resume_path,
        )

    assert not output_path.exists()


def test_behavior_cloning_resume_restores_progress_and_skips_completed_cursor(
    tmp_path, monkeypatch,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    configuration = _resume_configuration(input_path, seed=999)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=configuration,
        epoch=0,
        round_index=0,
        cursor=1,
        metrics={"behavior_clone_updates": 6, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )
    minibatch_epochs = []

    def two_batches(*, count, batch_size, seed, epoch):
        minibatch_epochs.append(epoch)
        assert count == 1
        return [[0], [0]]

    monkeypatch.setattr(train_policy, "epoch_minibatches", two_batches)
    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=999,
        device="cpu",
        resume_checkpoint=resume_path,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert minibatch_epochs == [0]
    assert payload["progress"] == {"epoch": 1, "round": 0, "cursor": 0}
    assert metadata["behavior_clone_updates"] == 7
    assert payload["optimizer_state_dict"]["state"]


def test_invalid_resume_configuration_is_checked_before_model_or_rng_mutation(
    tmp_path, monkeypatch,
):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path, seed=8),
        epoch=1,
        round_index=0,
        cursor=0,
        metrics={"behavior_clone_updates": 1, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )

    constructed = []

    class RecordingPolicy(CompactPolicyNet):
        def __init__(self):
            constructed.append(True)
            super().__init__()

    monkeypatch.setattr(train_policy, "CompactPolicyNet", RecordingPolicy)
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="seed"):
        train_policy.train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            seed=7,
            device="cpu",
            resume_checkpoint=resume_path,
        )

    assert constructed == []
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert not output_path.exists()


def test_resume_continues_from_saved_ppo_round(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path, ppo_steps=3),
        epoch=1,
        round_index=1,
        cursor=0,
        metrics={
            "behavior_clone_updates": 1,
            "ppo_updates": 4,
            "ppo_metrics": _ppo_checkpoint_metrics(),
        },
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )
    rollout_steps = []
    update_seeds = []

    def rollout_fn(*, step, **_kwargs):
        rollout_steps.append(step)
        return [_transition(done=True)]

    def update_fn(*, seed, **_kwargs):
        update_seeds.append(seed)
        return {"updates": 1, "early_stopped": False}

    monkeypatch.setattr(train_policy, "ppo_update", update_fn)
    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        ppo_steps=3,
        device="cpu",
        resume_checkpoint=resume_path,
        rollout_fn=rollout_fn,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert rollout_steps == [1, 2]
    assert update_seeds == [8, 9]
    assert metadata["ppo_updates"] == 6
    assert payload["metrics"]["ppo_updates"] == 6
    assert payload["progress"] == {"epoch": 1, "round": 3, "cursor": 0}


@pytest.mark.parametrize(
    "case,error_match",
    [
        ("missing_completed_steps", "completed_steps"),
        ("boolean_ppo_updates", "ppo_updates"),
        ("string_completed_steps", "completed_steps"),
        ("integer_early_stopped", "early_stopped"),
        ("list_last_metrics", "last_metrics"),
        ("list_promotion", "promotion"),
        ("unexpected_field", "unexpected"),
        ("progress_mismatch", "completed_steps.*PPO round|PPO round.*completed_steps"),
    ],
)
def test_resume_validates_exact_nested_ppo_metrics(case, error_match, tmp_path):
    from scripts import train_policy

    configuration = _resume_configuration(tmp_path / "transitions.jsonl", ppo_steps=2)
    ppo_metrics = _ppo_checkpoint_metrics()
    if case == "missing_completed_steps":
        del ppo_metrics["completed_steps"]
    elif case == "boolean_ppo_updates":
        ppo_metrics["ppo_updates"] = True
    elif case == "string_completed_steps":
        ppo_metrics["completed_steps"] = "1"
    elif case == "integer_early_stopped":
        ppo_metrics["early_stopped"] = 0
    elif case == "list_last_metrics":
        ppo_metrics["last_metrics"] = []
    elif case == "list_promotion":
        ppo_metrics["promotion"] = []
    elif case == "unexpected_field":
        ppo_metrics["unexpected"] = True
    elif case == "progress_mismatch":
        ppo_metrics["completed_steps"] = 2
    payload = {
        "configuration": configuration,
        "progress": {"epoch": 1, "round": 1, "cursor": 0},
        "metrics": {
            "behavior_clone_updates": 1,
            "ppo_updates": 4,
            "ppo_metrics": ppo_metrics,
        },
        "metadata": train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    }

    with pytest.raises(ValueError, match=error_match):
        train_policy._validate_resume_payload(
            payload,
            configuration=configuration,
            transition_count=1,
        )


def test_resume_with_all_ppo_steps_completed_is_idempotent(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    ppo_metrics = _ppo_checkpoint_metrics(completed_steps=2, ppo_updates=5, rollout_count=2)
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path, ppo_steps=2),
        epoch=1,
        round_index=2,
        cursor=0,
        metrics={
            "behavior_clone_updates": 1,
            "ppo_updates": 5,
            "ppo_metrics": ppo_metrics,
        },
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )
    ppo_calls = []

    def fail_if_called(**kwargs):
        ppo_calls.append(kwargs)
        raise AssertionError("completed PPO must not rerun")

    monkeypatch.setattr(train_policy, "run_ppo_training", fail_if_called)
    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        ppo_steps=2,
        device="cpu",
        resume_checkpoint=resume_path,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert ppo_calls == []
    assert metadata["ppo_updates"] == 5
    assert metadata["ppo_metrics"] == ppo_metrics
    assert payload["metrics"]["ppo_metrics"] == ppo_metrics
    assert payload["progress"] == {"epoch": 1, "round": 2, "cursor": 0}


def test_resume_allows_saved_cuda_device_on_cpu_and_records_current_device(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "cuda-resume.pt"
    output_path = tmp_path / "cpu-continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path, device="cuda"),
        epoch=1,
        round_index=0,
        cursor=0,
        metrics={"behavior_clone_updates": 1, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cuda"),
    )

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        device="cpu",
        resume_checkpoint=resume_path,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert metadata["device"] == "cpu"
    assert payload["metadata"]["device"] == "cpu"
    assert payload["configuration"]["device"] == "cpu"


def test_training_checkpoint_records_prior_checkpoint_content_identity(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    prior_path = tmp_path / "prior.pt"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    prior = CompactPolicyNet()
    torch.save({
        "metadata": train_policy.checkpoint_metadata(transition_count=1),
        "model_state_dict": prior.state_dict(),
    }, prior_path)

    train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        device="cpu",
        prior_checkpoint=prior_path,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert payload["configuration"]["prior_checkpoint"] == {
        "path": str(prior_path.resolve()),
        "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "field,saved_value,error_match",
    [
        ("seed", 8, "seed"),
        ("batch_size", 2, "batch_size"),
        ("steps", 2, "steps"),
        ("ppo_steps", 1, "ppo_steps"),
        ("ppo_config", {"gamma": 0.5}, "ppo_config"),
    ],
)
def test_behavior_cloning_resume_rejects_incompatible_training_configuration(
    tmp_path, field, saved_value, error_match,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    configuration = _resume_configuration(input_path)
    configuration[field] = saved_value
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=configuration,
        epoch=0,
        round_index=0,
        cursor=0,
        metrics={"behavior_clone_updates": 0, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )

    with pytest.raises(ValueError, match=error_match):
        train_policy.train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            seed=7,
            device="cpu",
            resume_checkpoint=resume_path,
        )

    assert not output_path.exists()


def test_behavior_cloning_resume_rejects_terminal_epoch_with_nonzero_cursor(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path),
        epoch=1,
        round_index=0,
        cursor=1,
        metrics={"behavior_clone_updates": 1, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )

    with pytest.raises(ValueError, match="cursor.*completed|completed.*cursor"):
        train_policy.train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            seed=7,
            device="cpu",
            resume_checkpoint=resume_path,
        )

    assert not output_path.exists()


def test_behavior_cloning_missing_resume_checkpoint_fails_without_output(tmp_path):
    from scripts.train_policy import train_behavior_clone

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            seed=7,
            device="cpu",
            resume_checkpoint=tmp_path / "missing.pt",
        )

    assert not output_path.exists()


def test_cli_training_uses_resolver_and_records_resolved_device(
    tmp_path, monkeypatch, capsys,
):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")

    class ResolvedDevice:
        def __str__(self):
            return "simulated-cuda"

    resolved_device = ResolvedDevice()
    resolver_calls = []
    model_devices = []
    tensor_devices = []
    ppo_devices = []
    real_tensor = torch.tensor

    class RecordingPolicy(CompactPolicyNet):
        def to(self, device):
            model_devices.append(device)
            return super().to("cpu")

    class TorchProxy:
        def tensor(self, *args, **kwargs):
            tensor_devices.append(kwargs.get("device"))
            if kwargs.get("device") is resolved_device:
                kwargs["device"] = torch.device("cpu")
            return real_tensor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(torch, name)

    def resolve(requested):
        resolver_calls.append(requested)
        return resolved_device

    def run_ppo(*, device, **_kwargs):
        ppo_devices.append(device)
        return {
            "ppo_updates": 1,
            "rollout_count": 0,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
        }

    monkeypatch.setattr(train_policy, "resolve_device", resolve)
    monkeypatch.setattr(train_policy, "CompactPolicyNet", RecordingPolicy)
    monkeypatch.setattr(train_policy, "require_torch", lambda: TorchProxy())
    monkeypatch.setattr(train_policy, "run_ppo_training", run_ppo)

    assert train_policy.main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--device", "auto",
        "--ppo-steps", "1",
        "--offline-ppo-fallback",
    ]) == 0

    printed_metadata = json.loads(capsys.readouterr().out)
    checkpoint = torch.load(output_path, map_location="cpu")
    assert resolver_calls == ["auto"]
    assert model_devices == [resolved_device]
    assert tensor_devices and set(tensor_devices) == {resolved_device}
    assert ppo_devices == [resolved_device]
    assert printed_metadata["device"] == "simulated-cuda"
    assert checkpoint["metadata"]["device"] == "simulated-cuda"
