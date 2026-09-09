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


def test_checkpoint_metadata_records_stall_and_shaping_ablations():
    from scripts.train_policy import PPOConfig, checkpoint_metadata

    metadata = checkpoint_metadata(
        1, PPOConfig(
            potential_reward_coef=0.05,
            no_progress_window=24,
            resolved_margin=1000.0,
        ),
    )

    assert metadata["potential_reward_coef"] == pytest.approx(0.05)
    assert metadata["no_progress_window"] == 24
    assert metadata["resolved_margin"] == pytest.approx(1000.0)
    assert metadata["ppo_config"]["potential_reward_coef"] == pytest.approx(0.05)


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


def test_parser_accepts_training_mode_bc_budget_and_model_shape():
    from scripts.train_policy import _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl", "--output", "policy.pt",
        "--training-mode", "pure_ppo",
        "--behavior-clone-steps", "8",
        "--model-width", "256",
        "--model-depth", "8",
    ])

    assert args.training_mode == "pure_ppo"
    assert args.behavior_clone_steps == 8
    assert args.model_width == 256
    assert args.model_depth == 8


def test_policy_cli_propagates_action_representation_and_mask_ablation():
    from scripts.train_policy import _cli_training_options, _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl", "--output", "policy.pt",
        "--action-representation", "target_first_v1",
        "--training-action-mask",
    ])

    options = _cli_training_options(args)

    assert args.action_representation == "target_first_v1"
    assert options["action_representation"] == "target_first_v1"
    assert options["ppo_config"].training_action_mask is True


def test_training_rejects_conflicting_top_level_and_nested_action_identity():
    from scripts.train_policy import PPOConfig, build_training_contract

    input_path = Path("transitions.jsonl")
    with pytest.raises(ValueError, match="action_representation"):
        build_training_contract(
            input_path=input_path,
            steps=1,
            batch_size=1,
            resolved_device="cpu",
            action_representation="target_first_v1",
            ppo_config=PPOConfig(action_representation="current_v1"),
        )


def test_policy_cli_propagates_reward_and_stall_ablations_into_ppo_config():
    from scripts.train_policy import PPOConfig, _cli_training_options, _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl", "--output", "policy.pt",
        "--potential-reward-coef", "0.05",
        "--no-progress-window", "24",
        "--resolved-margin", "1000",
        "--offline-ppo-fallback",
    ])

    options = _cli_training_options(args)
    assert options["ppo_config"] == PPOConfig(
        potential_reward_coef=0.05, no_progress_window=24, resolved_margin=1000.0,
    )


def test_rollout_batch_reports_termination_and_safety_diagnostics():
    from scripts.train_policy import PPOConfig, build_rollout_batch

    rows = [
        {
            **_transition(done=False, reward=0.0),
            "bootstrap_truncated": True,
            "termination_reason": "no_progress",
            "no_progress_steps": 4,
            "safety_flags": ["safety_regression"],
        },
        {
            **_transition(done=True, reward=0.0),
            "termination_reason": "time_limit",
        },
    ]

    batch = build_rollout_batch(
        rows,
        config=PPOConfig(no_progress_window=4),
        value_estimates=[0.1, 0.2],
        bootstrap_values=[0.7, 0.0],
    )

    assert batch.dones == [False, True]
    assert batch.bootstrap_truncated == [True, False]
    assert batch.termination_reasons == {"no_progress": 1, "time_limit": 1}
    assert batch.max_no_progress_steps == 4
    assert batch.time_limit_endings == 1
    assert batch.safety_regression_count == 1


def test_resolved_margin_uses_current_transition_margin_not_terminal_metadata():
    from scripts.train_policy import PPOConfig, build_rollout_batch

    row = _transition(done=False, final_bank=5000, opponent_final_bank=0)
    row["bank_differential"] = "not-a-number"

    batch = build_rollout_batch(
        [row], config=PPOConfig(resolved_margin=1000),
        value_estimates=[0.1], bootstrap_values=[0.7],
    )

    assert batch.bootstrap_truncated == [False]
    assert batch.dones == [False]


@pytest.mark.parametrize("bad_bank", [None, "malformed", float("nan"), float("inf")])
def test_terminal_bank_margin_reward_does_not_fallback_to_transition_reward(bad_bank):
    from scripts.train_policy import PPOConfig, build_rollout_batch

    row = _transition(done=True, reward=123.0, final_bank=bad_bank, opponent_final_bank=bad_bank)
    batch = build_rollout_batch([row], config=PPOConfig())

    assert batch.rewards == [0.0]
    assert math.isfinite(batch.rewards[0])


def test_promotion_safety_regression_is_fail_closed_even_when_reward_improves():
    from scripts.train_policy import promotion_safety_regression

    result = promotion_safety_regression(
        candidate={"mean_bank_differential": 100.0, "safety_regression_count": 3},
        baseline={"mean_bank_differential": 50.0, "safety_regression_count": 0},
    )

    assert result["safety_regression"] is True
    assert result["promotion_safe"] is False
    assert result["reason"] == "safety_regression"


def test_promotion_safety_regression_covers_all_stall_diagnostics():
    from scripts.train_policy import promotion_safety_regression

    result = promotion_safety_regression(
        candidate={
            "safety_regression_count": 1,
            "time_limit_endings": 2,
            "truncation_count": 3,
            "termination_reasons": {"resolved": 2, "no_progress": 4},
            "max_no_progress_streak": 8,
        },
        baseline={
            "safety_regression_count": 0,
            "time_limit_endings": 0,
            "truncation_count": 1,
            "termination_reasons": {"resolved": 0, "no_progress": 1},
            "max_no_progress_streak": 2,
        },
    )

    assert result["promotion_safe"] is False
    assert set(result["regressions"]) == {
        "safety_regression_count", "time_limit_endings", "truncation_count",
        "resolved_count", "no_progress_count", "max_no_progress_streak",
    }


def test_direct_promotion_gate_rejects_safety_regression_despite_win_rate():
    from scripts.train_policy import run_promotion_match

    def match_fn(index):
        return {
            "candidate_win": index < 71,
            "candidate_diagnostics": {
                "safety_regression_count": 1,
                "time_limit_endings": 0,
                "truncation_count": 0,
                "termination_reasons": {},
                "max_no_progress_streak": 0,
            },
            "baseline_diagnostics": {
                "safety_regression_count": 0,
                "time_limit_endings": 0,
                "truncation_count": 0,
                "termination_reasons": {},
                "max_no_progress_streak": 0,
            },
        }

    result = run_promotion_match(match_fn)

    assert result["wins"] == 71
    assert result["promoted"] is False
    assert result["promotion_safety"]["safety_regression"] is True


def test_ppo_promotion_path_applies_safety_gate(tmp_path):
    from scripts.train_policy import PPOConfig, run_ppo_training

    def match_fn(index, **_kwargs):
        return {
            "candidate_win": index < 71,
            "candidate_diagnostics": {"safety_regression_count": 1},
            "baseline_diagnostics": {"safety_regression_count": 0},
        }

    result = run_ppo_training(
        network=None, optimizer=None, transitions=[_transition(done=True)],
        ppo_steps=1, config=PPOConfig(), offline_ppo_fallback=True,
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
        promotion_match_fn=match_fn,
        candidate_checkpoint=tmp_path / "candidate.pt",
    )

    assert result["promotion"]["wins"] == 71
    assert result["promotion"]["promoted"] is False
    assert result["promotion"]["promotion_safety"]["safety_regression"] is True


@pytest.mark.parametrize(
    ("mode", "configured", "expected"),
    [
        ("behavior_clone_then_ppo", 8, 8),
        ("reduced_behavior_clone_then_ppo", 8, 2),
        ("reduced_behavior_clone_then_ppo", 3, 1),
        ("pure_ppo", 8, 0),
    ],
)
def test_behavior_clone_budget_is_resolved_by_training_mode(mode, configured, expected):
    from scripts.train_policy import resolve_behavior_clone_steps

    assert resolve_behavior_clone_steps(mode, configured) == expected


@pytest.mark.parametrize(
    ("training_mode", "configured_steps", "effective_steps"),
    [
        ("pure_ppo", 8, 1),
        ("reduced_behavior_clone_then_ppo", 8, 8),
        ("behavior_clone_then_ppo", 8, 2),
    ],
)
def test_training_api_rejects_contradictory_effective_behavior_clone_budget(
    tmp_path, training_mode, configured_steps, effective_steps,
):
    pytest.importorskip("torch")
    from scripts.train_policy import train_behavior_clone

    input_path = tmp_path / "transitions.jsonl"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="effective behavior_clone_steps"):
        train_behavior_clone(
            input_path=input_path,
            output_path=tmp_path / "policy.pt",
            steps=configured_steps,
            effective_behavior_clone_steps=effective_steps,
            batch_size=1,
            device="cpu",
            training_mode=training_mode,
        )


def test_train_policy_parser_exposes_league_configuration(tmp_path):
    from scripts.train_policy import _cli_training_options, _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl", "--output", "policy.pt",
        "--league-checkpoints", str(tmp_path / "old.pt"),
        "--league-checkpoint-window", "2",
        "--league-current-probability", "2",
        "--league-mixed-probability", "1",
        "--league-random-probability", "0",
        "--league-starter-probability", "3",
        "--league-checkpoint-probability", "4",
    ])

    options = _cli_training_options(args)

    assert args.league_checkpoints == [tmp_path / "old.pt"]
    assert args.league_checkpoint_window == 2
    assert options["opponent_pool"].probabilities == {
        "current": 2.0,
        "mixed": 1.0,
        "random": 0.0,
        "starter": 3.0,
        "checkpoint": 4.0,
    }


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


def test_parser_accepts_ppo_extension_opt_in_flag():
    from scripts.train_policy import _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--allow-ppo-extension",
    ])

    assert args.allow_ppo_extension is True


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


def test_resume_ppo_extension_requires_opt_in_and_only_allows_increase():
    from scripts.train_policy import _validate_resume_configuration

    requested = _resume_configuration(Path("transitions.jsonl"), ppo_steps=2)
    saved = {**requested, "ppo_steps": 1}

    with pytest.raises(ValueError, match="ppo_steps"):
        _validate_resume_configuration(saved, requested)

    _validate_resume_configuration(saved, requested, allow_ppo_extension=True)
    _validate_resume_configuration(requested, requested)

    with pytest.raises(ValueError, match="ppo_steps"):
        _validate_resume_configuration(requested, requested, allow_ppo_extension=True)

    decreased = {**requested, "ppo_steps": 3}
    with pytest.raises(ValueError, match="ppo_steps"):
        _validate_resume_configuration(decreased, requested, allow_ppo_extension=True)


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
    assert metrics["post_step_kl"] > 1e-12


def test_ppo_update_labels_known_kl_metrics_by_update_phase(monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts import train_policy

    class OneParameterPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features):
            return {"value": self.bias.repeat(len(features))}

    old_log_probs = [0.0, 0.0]
    pre_step_log_probs = [math.log(0.5), math.log(0.25)]
    post_step_log_probs = [math.log(1.5), math.log(0.5)]
    selected_log_probs = iter(
        (old_log_probs, pre_step_log_probs, post_step_log_probs)
    )
    network = OneParameterPolicy()

    def select_known_log_probs(*_args, **_kwargs):
        values = torch.tensor(next(selected_log_probs), dtype=torch.float32)
        return network.bias * 0.0 + values, network.bias * 0.0

    monkeypatch.setattr(train_policy, "_select_outputs", select_known_log_probs)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    metrics = train_policy.ppo_update(
        network,
        optimizer,
        [
            _transition(done=True),
            _transition(done=True, final_bank=2000, opponent_final_bank=0),
        ],
        config=train_policy.PPOConfig(target_kl=100.0),
        batch_size=2,
    )

    expected_pre_step_approx_kl = sum(
        old - new for old, new in zip(old_log_probs, pre_step_log_probs)
    ) / 2
    post_step_log_ratios = [
        new - old for old, new in zip(old_log_probs, post_step_log_probs)
    ]
    expected_post_step_kl = sum(
        math.exp(log_ratio) - 1.0 - log_ratio
        for log_ratio in post_step_log_ratios
    ) / 2
    assert metrics["pre_step_approx_kl"] == pytest.approx(expected_pre_step_approx_kl)
    assert metrics["post_step_kl"] == pytest.approx(expected_post_step_kl)
    assert metrics["approx_kl"] == metrics["pre_step_approx_kl"]
    assert metrics["pre_step_approx_kl"] != pytest.approx(metrics["post_step_kl"])


def test_ppo_update_reports_training_health_metrics():
    torch = pytest.importorskip("torch")
    from kagriculture_agent.model import CompactPolicyNet
    from scripts.train_policy import PPOConfig, ppo_update

    network = CompactPolicyNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)

    metrics = ppo_update(
        network,
        optimizer,
        [_transition(done=False), _transition(done=True, final_bank=2000, opponent_final_bank=0)],
        config=PPOConfig(target_kl=100.0, ppo_epochs=1),
        batch_size=2,
    )

    expected = {
        "clip_fraction",
        "explained_variance",
        "return_mean",
        "return_std",
        "advantage_mean",
        "advantage_std",
        "gradient_norm",
        "parameter_norm",
        "learning_rate",
    }
    assert expected <= metrics.keys()
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["learning_rate"] == pytest.approx(1e-3)
    assert all(math.isfinite(float(metrics[name])) for name in expected)


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


@pytest.mark.parametrize(
    ("helper_name", "training_mode", "behavior_clone_steps"),
    [
        ("checkpoint_metadata", "pure_ppo", 1),
        ("checkpoint_metadata", "reduced_behavior_clone_then_ppo", 0),
        ("checkpoint_metadata", "behavior_clone_then_ppo", 0),
        ("_checkpoint_metadata", "pure_ppo", 1),
        ("_checkpoint_metadata", "reduced_behavior_clone_then_ppo", 0),
        ("_checkpoint_metadata", "behavior_clone_then_ppo", 0),
    ],
)
def test_checkpoint_metadata_rejects_contradictory_mode_and_bc_budget(
    helper_name, training_mode, behavior_clone_steps,
):
    from scripts import train_policy

    with pytest.raises(ValueError, match="behavior_clone_steps|training_mode"):
        getattr(train_policy, helper_name)(
            transition_count=1,
            training_mode=training_mode,
            behavior_clone_steps=behavior_clone_steps,
        )


@pytest.mark.parametrize(
    ("training_mode", "behavior_clone_steps"),
    [
        ("pure_ppo", 0),
        ("reduced_behavior_clone_then_ppo", 2),
        ("behavior_clone_then_ppo", 8),
    ],
)
def test_checkpoint_metadata_accepts_canonical_mode_and_bc_budget(
    training_mode, behavior_clone_steps,
):
    from scripts.train_policy import checkpoint_metadata

    metadata = checkpoint_metadata(
        transition_count=1,
        training_mode=training_mode,
        behavior_clone_steps=behavior_clone_steps,
    )

    assert metadata["training_mode"] == training_mode
    assert metadata["behavior_clone_steps"] == behavior_clone_steps


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


def test_task_intent_objective_masks_move_and_pass_rows():
    torch = pytest.importorskip("torch")
    from scripts.train_policy import PPOConfig, _select_outputs, build_rollout_batch

    transitions = [{
        "action": {"farmer": ["EAST"], "hands": [["PASS"]], "market": []},
        "observation": {
            "board_size": 2,
            "workers": [
                {"index": 0, "position": [0, 0]},
                {"index": 1, "position": [0, 0]},
            ],
        },
    }]
    batch = build_rollout_batch(transitions, config=PPOConfig())
    outputs = {
        "worker_act_logits": torch.zeros(1, 10, 2, requires_grad=True),
        "worker_target_logits": torch.zeros(1, 10, 4, requires_grad=True),
        "worker_kind_logits": torch.zeros(1, 10, 14, requires_grad=True),
        "market_active_logits": torch.zeros(1, 2, requires_grad=True),
        "market_item_logits": torch.zeros(1, 9, requires_grad=True),
        "market_quantity_logits": torch.zeros(1, 8, requires_grad=True),
        "value": torch.zeros(1, requires_grad=True),
    }

    log_probs, _entropy = _select_outputs(outputs, batch)
    (-log_probs.mean()).backward()

    assert outputs["worker_act_logits"].grad[0, 0].abs().sum() > 0
    assert outputs["worker_target_logits"].grad[0, 0].abs().sum() == 0
    assert outputs["worker_kind_logits"].grad[0, 0].abs().sum() == 0


def test_checkpoint_metadata_records_task_intent_objective_loss_mask():
    from scripts.train_policy import checkpoint_metadata

    metadata = checkpoint_metadata(transition_count=1)

    assert metadata["task_intent_loss_mask"] == {
        "excluded_worker_kinds": ["PASS", "MOVE"],
        "objectives": ["worker_target", "worker_kind"],
    }


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


def test_ppo_rollouts_consume_a_lazy_league_iterator():
    from scripts.train_policy import OpponentMatch, PPOConfig, run_ppo_training

    calls = []

    class Pool:
        def schedule(self, *, count, seed):
            raise AssertionError("PPO must not eagerly materialize the league schedule")

        def iter_schedule(self, *, count, seed):
            calls.append(("iter_schedule", count, seed))
            for index in range(count):
                yield OpponentMatch("current", index % 2)

    run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[],
        ppo_steps=3,
        config=PPOConfig(rollout_steps=5),
        opponent_pool=Pool(),
        rollout_fn=lambda **_kwargs: [_transition(done=True)],
        seed=23,
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
    )

    assert calls == [("iter_schedule", 3, 23)]


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


def test_ppo_training_emits_progress_metrics_to_optional_telemetry_callback():
    from scripts.train_policy import PPOConfig, run_ppo_training

    events = []

    metrics = run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **kwargs: {
            "updates": 2,
            "early_stopped": True,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.3,
            "approx_kl": 0.4,
            "pre_step_approx_kl": 0.4,
            "post_step_kl": 0.6,
        },
        telemetry_callback=lambda event, values: events.append((event, values)),
    )

    assert metrics["early_stopped"] is True
    assert events == [
        (
            "ppo",
            {
                "step": 1,
                "ppo_updates": 2,
                "ppo_updates_step": 2,
                "rollout_count": 0,
                "early_stopped": True,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 0.3,
                "approx_kl": 0.4,
                "pre_step_approx_kl": 0.4,
                "post_step_kl": 0.6,
            },
        )
    ]


def test_ppo_fallback_reason_reaches_rollout_callback_and_telemetry():
    from scripts.train_policy import OpponentMatch, PPOConfig, run_ppo_training

    seen = []
    events = []

    class Pool:
        def schedule(self, *, count, seed):
            return [OpponentMatch(
                "current", 0, fallback_reason="checkpoint_unavailable",
            )]

    def rollout_fn(**kwargs):
        seen.append(kwargs)
        return [_transition(done=True)]

    run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[],
        ppo_steps=1,
        config=PPOConfig(),
        opponent_pool=Pool(),
        rollout_fn=rollout_fn,
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
        telemetry_callback=lambda event, values: events.append((event, values)),
    )

    assert seen[0]["fallback_reason"] == "checkpoint_unavailable"
    assert events[0][1]["league/fallback_reason"] == "checkpoint_unavailable"


def test_ppo_callback_forwards_training_health_metrics_from_injected_update():
    from scripts.train_policy import PPOConfig, run_ppo_training

    events = []
    health = {
        "clip_fraction": 0.25,
        "explained_variance": 0.5,
        "return_mean": 1.0,
        "return_std": 2.0,
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "gradient_norm": 3.0,
        "parameter_norm": 4.0,
        "learning_rate": 0.001,
    }

    run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **kwargs: {
            "updates": 1,
            "early_stopped": False,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.3,
            "approx_kl": 0.4,
            **health,
        },
        telemetry_callback=lambda event, values: events.append((event, values)),
    )

    assert events[0][0] == "ppo"
    assert events[0][1]["policy_loss"] == 0.1
    for name, value in health.items():
        assert events[0][1][name] == value


def test_ppo_callback_forwards_reward_shaping_and_truncation_counts():
    from scripts.train_policy import PPOConfig, run_ppo_training

    events = []
    run_ppo_training(
        network=None,
        optimizer=None,
        transitions=[_transition(done=True)],
        ppo_steps=1,
        config=PPOConfig(),
        offline_ppo_fallback=True,
        update_fn=lambda **kwargs: {
            "updates": 1,
            "early_stopped": False,
            "shaping_count": 2,
            "truncation_count": 3,
        },
        telemetry_callback=lambda event, values: events.append((event, values)),
    )

    assert events[0][1]["shaping_count"] == 2
    assert events[0][1]["truncation_count"] == 3


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


def test_cli_training_options_forwards_ppo_extension_opt_in():
    from scripts.train_policy import _cli_training_options, _parser

    args = _parser().parse_args([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--allow-ppo-extension",
    ])

    assert _cli_training_options(args)["allow_ppo_extension"] is True


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


def test_best_checkpoint_persistence_rechecks_final_symlink_before_replace(
    tmp_path, monkeypatch,
):
    from scripts import train_policy

    candidate = tmp_path / "candidate.pt"
    candidate.write_text("candidate\n", encoding="utf-8")
    best = tmp_path / "best.pt"
    target = tmp_path / "existing-best.pt"
    target.write_text("existing\n", encoding="utf-8")
    real_validator = train_policy.validate_training_output_path
    calls = 0

    def race_validator(path, **kwargs):
        nonlocal calls
        calls += 1
        result = real_validator(path, **kwargs)
        if calls == 1:
            best.symlink_to(target)
        return result

    monkeypatch.setattr(train_policy, "validate_training_output_path", race_validator)
    with pytest.raises(ValueError, match="symlink"):
        train_policy._persist_best_checkpoint(candidate, best)
    assert calls == 2
    assert best.is_symlink()
    assert target.read_text(encoding="utf-8") == "existing\n"
    assert not list(tmp_path.glob(".best.pt.publish.tmp"))


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


def test_cli_main_forwards_ppo_extension_opt_in(monkeypatch, capsys):
    from scripts import train_policy

    observed = {}

    def fake_train(**kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(train_policy, "train_behavior_clone", fake_train)

    assert train_policy.main([
        "--input", "transitions.jsonl",
        "--output", "policy.pt",
        "--allow-ppo-extension",
    ]) == 0
    capsys.readouterr()
    assert observed["allow_ppo_extension"] is True


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


def test_opponent_pool_schedule_uses_requested_probabilities_and_uniform_checkpoints(tmp_path):
    from collections import Counter

    from scripts.train_policy import OpponentPool

    checkpoints = []
    for index in range(7):
        checkpoint = tmp_path / f"ckpt-{index}"
        checkpoint.touch()
        checkpoints.append(checkpoint)
    pool = OpponentPool(previous_checkpoints=checkpoints)
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


def test_opponent_pool_reports_missing_checkpoint_fallback(tmp_path):
    from scripts.train_policy import OpponentPool

    pool = OpponentPool(
        previous_checkpoints=[tmp_path / "missing.pt"],
        probabilities={"checkpoint": 1.0},
    )

    match = pool.sample(0, seed=1)

    assert match.opponent == "current"
    assert match.fallback_reason == "checkpoint_unavailable"


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


def test_ppo_rollout_callback_receives_league_provenance_and_round_seed(tmp_path):
    from scripts.train_policy import OpponentMatch, PPOConfig, run_ppo_training

    checkpoint = tmp_path / "historical.pt"
    checkpoint.write_bytes(b"historical-policy")
    seen = []

    class Pool:
        def schedule(self, *, count, seed):
            assert (count, seed) == (1, 41)
            return [OpponentMatch(
                "checkpoint", 1, str(checkpoint), None, "hard", "checkpoint:historical",
            )]

    def rollout_fn(**kwargs):
        seen.append(kwargs)
        return [_transition(done=True)]

    run_ppo_training(
        network=object(), optimizer=None, transitions=[], ppo_steps=1,
        config=PPOConfig(rollout_steps=5), opponent_pool=Pool(),
        rollout_fn=rollout_fn, seed=41, experiment_id="orbit-task2",
        update_fn=lambda **_kwargs: {"updates": 1, "early_stopped": False},
    )

    assert seen == [{
        "step": 0,
        "opponent": "checkpoint",
        "seat": 1,
        "checkpoint": str(checkpoint),
        "rollout_steps": 5,
        "candidate_artifact": None,
        "candidate_identity": None,
        "seed": 41,
        "round_index": 0,
        "opponent_identity": "checkpoint",
        "checkpoint_identity": "checkpoint:historical",
        "fallback_reason": None,
        "mixed_opponent": None,
        "network": seen[0]["network"],
        "experiment_id": "orbit-task2",
    }]


def test_opponent_pool_uses_configured_league_window_and_probabilities(tmp_path):
    from scripts.train_policy import OpponentPool

    checkpoints = []
    for index in range(4):
        checkpoint = tmp_path / f"policy-{index}.pt"
        checkpoint.write_bytes(str(index).encode())
        checkpoints.append(checkpoint)

    pool = OpponentPool(
        previous_checkpoints=checkpoints, checkpoint_window=2,
        probabilities={"checkpoint": 1.0},
    )

    assert pool.checkpoint_candidates == tuple(str(path) for path in checkpoints[-2:])
    assert [match.seat for match in pool.schedule(count=6, seed=9)] == [0, 1, 0, 1, 0, 1]
    assert all(match.opponent == "checkpoint" for match in pool.schedule(count=6, seed=9))
    assert {match.checkpoint for match in pool.schedule(count=40, seed=9)} == set(pool.checkpoint_candidates)


def test_opponent_pool_rejects_nonpositive_probability_total():
    from scripts.train_policy import OpponentPool

    with pytest.raises(ValueError):
        OpponentPool(probabilities={"current": 0.0, "checkpoint": 0.0})


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
        no_progress_window=3,
        resolved_margin=125.5,
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
    assert [item["no_progress_window"] for item in collected] == [3, 3]
    assert [item["resolved_margin"] for item in collected] == [125.5, 125.5]


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
    assert metadata["ppo_steps"] == 0

    checkpoint = pytest.importorskip("torch").load(output_path, map_location="cpu")
    assert checkpoint["metadata"]["device"] == "cpu"
    assert checkpoint["metadata"]["ppo_steps"] == 0
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["configuration"]["device"] == "cpu"
    assert checkpoint["progress"] == {"epoch": 1, "round": 0, "cursor": 0}
    assert checkpoint["metrics"]["behavior_clone_updates"] == 1


def test_reduced_behavior_clone_records_effective_budget_and_resume_validates_it(tmp_path):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    resumed_output_path = tmp_path / "resumed-policy.pt"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in [_transition(done=False), _transition(done=True)]) + "\n",
        encoding="utf-8",
    )

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=8,
        behavior_clone_steps=8,
        batch_size=2,
        seed=7,
        device="cpu",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    checkpoint = pytest.importorskip("torch").load(output_path, map_location="cpu", weights_only=True)
    assert metadata["behavior_clone_steps"] == 2
    assert metadata["behavior_clone_updates"] == 2
    assert checkpoint["configuration"]["behavior_clone_steps"] == 2
    assert checkpoint["metadata"]["behavior_clone_steps"] == 2
    assert checkpoint["metrics"]["behavior_clone_updates"] == 2

    resumed_metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=resumed_output_path,
        steps=8,
        behavior_clone_steps=8,
        batch_size=2,
        seed=7,
        device="cpu",
        training_mode="reduced_behavior_clone_then_ppo",
        resume_checkpoint=output_path,
    )

    assert resumed_metadata["behavior_clone_steps"] == 2
    assert resumed_metadata["behavior_clone_updates"] == 2


def test_resume_without_updates_preserves_checkpoint_rng_state(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"
    resumed_path = tmp_path / "resumed.pt"
    input_path.write_text(
        json.dumps(_transition(done=False)) + "\n", encoding="utf-8",
    )

    train_policy.train_behavior_clone(
        input_path=input_path, output_path=checkpoint_path, steps=1,
        batch_size=1, seed=17, device="cpu",
    )
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    train_policy.train_behavior_clone(
        input_path=input_path, output_path=resumed_path, steps=1,
        batch_size=1, seed=17, device="cpu", resume_checkpoint=checkpoint_path,
    )
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)

    assert torch.equal(source["rng_state"]["torch"], resumed["rng_state"]["torch"])


@pytest.mark.parametrize(
    ("training_mode", "expected"),
    [
        ("behavior_clone_then_ppo", 8),
        ("reduced_behavior_clone_then_ppo", 2),
        ("pure_ppo", 0),
    ],
)
def test_colab_wandb_config_records_effective_behavior_clone_steps(
    tmp_path, monkeypatch, training_mode, expected,
):
    from scripts import train

    captured = {}

    class FakeTelemetry:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("scripts.telemetry.TrainingTelemetry", FakeTelemetry)
    config = train.build_config(
        run_directory=tmp_path,
        device="cpu",
        resolve_runtime_device=False,
        training_steps=8,
        training_mode=training_mode,
        wandb_enabled=False,
    )

    train.initialize_telemetry(config)

    assert config.behavior_clone_steps == expected
    assert captured["wandb_config"]["behavior_clone_steps"] == expected


def test_colab_resolves_behavior_clone_budget_once_and_propagates_effective_value(
    tmp_path, monkeypatch,
):
    from scripts import train, train_policy

    resolver_calls = []
    real_resolver = train_policy.resolve_behavior_clone_steps

    def observe_resolver(training_mode, configured_steps):
        resolver_calls.append((training_mode, configured_steps))
        return real_resolver(training_mode, configured_steps)

    monkeypatch.setattr(train, "resolve_behavior_clone_steps", observe_resolver)
    monkeypatch.setattr(train_policy, "resolve_behavior_clone_steps", observe_resolver)
    config = train.build_config(
        run_directory=tmp_path,
        device="cpu",
        resolve_runtime_device=False,
        training_steps=8,
        training_mode="reduced_behavior_clone_then_ppo",
        wandb_enabled=False,
    )
    config.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    config.trajectory_path.write_text("{}\n", encoding="utf-8")

    contract = train_policy.build_training_contract(
        input_path=config.trajectory_path,
        steps=config.training_steps,
        batch_size=config.training_batch_size,
        device=config.device,
        training_mode=config.training_mode,
        effective_behavior_clone_steps=config.behavior_clone_steps,
    )

    captured = {}

    def fake_rollout(**_kwargs):
        return None

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"behavior_clone_steps": kwargs["effective_behavior_clone_steps"]}

    monkeypatch.setattr(train_policy, "make_fresh_rollout_fn", fake_rollout)
    monkeypatch.setattr(train_policy, "train_behavior_clone", fake_train)
    monkeypatch.setattr("scripts.export_policy.export_checkpoint", lambda *_args, **_kwargs: None)

    train.train_candidate(
        config,
        training_contract=contract,
        resume_checkpoint=None,
        allow_ppo_extension=False,
        opponent_pool=None,
        telemetry=None,
    )

    assert resolver_calls == [("reduced_behavior_clone_then_ppo", 8)]
    assert contract.configuration["behavior_clone_steps"] == 2
    assert captured["effective_behavior_clone_steps"] == 2


def test_pure_ppo_skips_behavior_clone_and_starts_fresh_ppo(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    calls = []

    def fail_if_behavior_cloning_runs(*_args, **_kwargs):
        raise AssertionError("pure PPO must not execute a BC minibatch")

    def fake_ppo(**kwargs):
        calls.append(kwargs)
        return {
            "ppo_updates": 1,
            "rollout_count": 1,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
            "completed_steps": 1,
        }

    monkeypatch.setattr(train_policy, "epoch_minibatches", fail_if_behavior_cloning_runs)
    monkeypatch.setattr(train_policy, "run_ppo_training", fake_ppo)

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=8,
        behavior_clone_steps=8,
        batch_size=1,
        seed=7,
        ppo_steps=1,
        device="cpu",
        training_mode="pure_ppo",
        offline_ppo_fallback=True,
    )

    checkpoint = pytest.importorskip("torch").load(output_path, map_location="cpu", weights_only=True)
    assert len(calls) == 1
    assert metadata["behavior_clone_steps"] == 0
    assert metadata["behavior_clone_updates"] == 0
    assert checkpoint["metrics"]["behavior_clone_updates"] == 0
    assert checkpoint["metadata"]["training_mode"] == "pure_ppo"


@pytest.mark.parametrize(
    ("training_mode", "configured_steps", "expected_bc_updates"),
    [
        ("behavior_clone_then_ppo", 1, 1),
        ("reduced_behavior_clone_then_ppo", 8, 2),
        ("pure_ppo", 8, 0),
    ],
)
def test_training_modes_run_real_bc_and_fresh_rollout_in_order(
    tmp_path, training_mode, configured_steps, expected_bc_updates,
):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / f"{training_mode}.jsonl"
    output_path = tmp_path / f"{training_mode}.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    events = []
    rollout_steps = []

    def rollout_fn(**kwargs):
        rollout_steps.append(kwargs["step"])
        return [_transition(done=True, final_bank=1001, opponent_final_bank=999)]

    def telemetry(event, _values):
        events.append(event)

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=configured_steps,
        behavior_clone_steps=configured_steps,
        batch_size=1,
        seed=7,
        ppo_steps=1,
        device="cpu",
        training_mode=training_mode,
        rollout_fn=rollout_fn,
        telemetry_callback=telemetry,
    )

    assert metadata["behavior_clone_updates"] == expected_bc_updates
    assert rollout_steps == [0]
    assert events == (["ppo"] if expected_bc_updates == 0 else ["behavior_clone", "ppo"])


def test_resume_rejects_incompatible_model_shape(tmp_path):
    pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet(hidden_width=256, depth=8).to("cpu")
    optimizer = pytest.importorskip("torch").optim.AdamW(model.parameters(), lr=1e-3)
    configuration = train_policy.build_training_contract(
        input_path=input_path,
        steps=1,
        batch_size=1,
        device="cpu",
        model_width=256,
        model_depth=8,
    ).configuration
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=configuration,
        epoch=1,
        round_index=0,
        cursor=0,
        metrics={"behavior_clone_updates": 1, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(
            transition_count=1, device="cpu", model_width=256, model_depth=8,
        ),
    )

    with pytest.raises(ValueError, match="model_width|model_depth"):
        train_policy.train_behavior_clone(
            input_path=input_path,
            output_path=output_path,
            steps=1,
            batch_size=1,
            device="cpu",
            resume_checkpoint=resume_path,
        )


def test_training_contract_and_checkpoint_metadata_include_experiment_identity(tmp_path):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    rows = [_transition(done=False), _transition(done=True, reward=math.tanh(0.3))]
    input_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    contract = train_policy.build_training_contract(
        input_path=input_path, steps=1, batch_size=2, device="cpu",
        experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )
    assert contract.configuration["experiment_id"] == "orbit-context-test"
    assert contract.configuration["feature_variant"] == "experimental_context_v1"
    assert contract.configuration["training_mode"] == "reduced_behavior_clone_then_ppo"

    metadata = train_policy.train_behavior_clone(
        input_path=input_path, output_path=output_path, steps=1, batch_size=2,
        device="cpu", experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )
    assert {
        key: metadata[key]
        for key in ("experiment_id", "feature_variant", "training_mode")
    } == {
        "experiment_id": "orbit-context-test",
        "feature_variant": "experimental_context_v1",
        "training_mode": "reduced_behavior_clone_then_ppo",
    }


def test_resume_rejects_a_checkpoint_from_another_experiment_identity(tmp_path):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    rows = [_transition(done=False), _transition(done=True, reward=math.tanh(0.3))]
    input_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    train_policy.train_behavior_clone(
        input_path=input_path, output_path=resume_path, steps=1, batch_size=2,
        device="cpu", experiment_id="experiment-a",
    )

    with pytest.raises(ValueError, match="experiment_id"):
        train_policy.train_behavior_clone(
            input_path=input_path, output_path=output_path, steps=1, batch_size=2,
            device="cpu", experiment_id="experiment-b",
            resume_checkpoint=resume_path,
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_variant", "experimental_context_v1"),
        ("training_mode", "reduced_behavior_clone_then_ppo"),
    ],
)
def test_resume_rejects_a_checkpoint_with_mismatched_feature_or_training_identity(
    tmp_path, field, value,
):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    rows = [_transition(done=False), _transition(done=True, reward=math.tanh(0.3))]
    input_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    train_policy.train_behavior_clone(
        input_path=input_path, output_path=resume_path, steps=1, batch_size=2,
        device="cpu",
    )

    with pytest.raises(ValueError, match=field):
        train_policy.train_behavior_clone(
            input_path=input_path, output_path=output_path, steps=1, batch_size=2,
            device="cpu", **{field: value}, resume_checkpoint=resume_path,
        )
    assert not output_path.exists()


def test_fresh_rollout_passes_training_identity_to_native_collector(tmp_path, monkeypatch):
    from scripts import train_policy

    collected = []

    def fake_collect(*, output, candidate_artifact, opponents, **kwargs):
        collected.append(kwargs)
        Path(output).write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")

    monkeypatch.setattr("scripts.collect_trajectories.collect", fake_collect)
    artifact = tmp_path / "candidate.json"
    artifact.write_text("artifact", encoding="utf-8")
    rollout_fn = train_policy.make_fresh_rollout_fn(
        run_directory=tmp_path, candidate_artifact=artifact,
        seeds=[41], steps=4,
        experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    rollout_fn(
        step=0, seed=41, opponent="pass", seat=0, checkpoint=None,
        rollout_steps=3,
    )

    assert collected[0]["source_policy_identity"].startswith("artifact:")
    assert collected[0]["experiment_id"] == "orbit-context-test"
    assert collected[0]["feature_variant"] == "experimental_context_v1"
    assert collected[0]["training_mode"] == "reduced_behavior_clone_then_ppo"


def test_fresh_rollout_threads_original_league_configuration_to_collector(tmp_path, monkeypatch):
    from scripts import train_policy

    collected = []

    def fake_collect(*, output, candidate_artifact, **kwargs):
        collected.append(kwargs)
        Path(output).write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")

    monkeypatch.setattr("scripts.collect_trajectories.collect", fake_collect)
    artifact = tmp_path / "candidate.json"
    artifact.write_text("artifact", encoding="utf-8")
    probabilities = {
        "current": 2.0,
        "mixed": 1.0,
        "random": 0.0,
        "starter": 3.0,
        "checkpoint": 4.0,
    }
    checkpoints = [tmp_path / "first.pt", tmp_path / "second.pt", tmp_path / "third.pt"]
    rollout_fn = train_policy.make_fresh_rollout_fn(
        run_directory=tmp_path, candidate_artifact=artifact,
        seeds=[41], steps=4,
        league_probabilities=probabilities,
        league_checkpoint_window=2,
        league_checkpoints=checkpoints,
    )

    rollout_fn(
        step=0, seed=41, opponent="pass", seat=0, checkpoint=None,
        rollout_steps=3,
    )

    assert collected[0]["league_probabilities"] == probabilities
    assert collected[0]["league_checkpoint_window"] == 2
    assert collected[0]["league_checkpoints"] == [str(path) for path in checkpoints]


def test_fresh_rollout_threads_fallback_reason_to_collector(tmp_path, monkeypatch):
    from scripts import train_policy

    collected = []

    def fake_collect(*, output, candidate_artifact, **kwargs):
        collected.append(kwargs)
        Path(output).write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")

    monkeypatch.setattr("scripts.collect_trajectories.collect", fake_collect)
    artifact = tmp_path / "candidate.json"
    artifact.write_text("artifact", encoding="utf-8")
    rollout_fn = train_policy.make_fresh_rollout_fn(
        run_directory=tmp_path, candidate_artifact=artifact, seeds=[41], steps=4,
    )

    rollout_fn(
        step=0, seed=41, opponent="current", seat=0, checkpoint=None,
        fallback_reason="checkpoint_unavailable", rollout_steps=3,
    )

    assert collected[0]["fallback_reason"] == "checkpoint_unavailable"


def test_behavior_cloning_emits_loss_and_update_count_at_checkpoint_intervals(tmp_path):
    pytest.importorskip("torch")
    from scripts.train_policy import train_behavior_clone

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(
        "\n".join(json.dumps(_transition(done=True)) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    events = []

    train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        device="cpu",
        checkpoint_interval=1,
        telemetry_callback=lambda event, values: events.append((event, values)),
    )

    assert [event for event, _values in events] == ["behavior_clone", "behavior_clone"]
    assert [values["update_count"] for _event, values in events] == [1, 2]
    assert all(math.isfinite(values["loss"]) for _event, values in events)
    for _event, values in events:
        for name in ("learning_rate", "gradient_norm", "parameter_norm", "entropy"):
            assert math.isfinite(values[name])
        assert values["learning_rate"] == pytest.approx(1e-3)


def test_behavior_cloning_starts_ppo_with_fresh_conservative_optimizer(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    output_path = tmp_path / "policy.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    observed = {}

    def observe_ppo(**kwargs):
        observed["learning_rate"] = kwargs["optimizer"].param_groups[0]["lr"]
        observed["optimizer_state"] = dict(kwargs["optimizer"].state)
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

    assert observed["learning_rate"] == 1e-5
    assert observed["optimizer_state"] == {}


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


def test_resume_accepts_ppo_league_provenance_metrics(tmp_path):
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
    ppo_metrics = _ppo_checkpoint_metrics()
    ppo_metrics.update({
        "league_composition": {
            "current": 1, "mixed": 0, "random": 0, "starter": 0, "checkpoint": 0,
        },
        "league_checkpoint_identities": [],
    })
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path, ppo_steps=1),
        epoch=1,
        round_index=1,
        cursor=0,
        metrics={
            "behavior_clone_updates": 1,
            "ppo_updates": 4,
            "ppo_metrics": ppo_metrics,
        },
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        ppo_steps=1,
        device="cpu",
        resume_checkpoint=resume_path,
    )

    assert metadata["ppo_metrics"]["league_composition"]["current"] == 1
    assert metadata["ppo_metrics"]["league_checkpoint_identities"] == []


def test_resume_accumulates_prior_ppo_league_provenance(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    monkeypatch.setattr(
        train_policy, "ppo_update",
        lambda **_kwargs: {"updates": 1, "early_stopped": False},
    )

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ppo_metrics = _ppo_checkpoint_metrics()
    ppo_metrics.update({
        "league_composition": {
            "current": 3, "mixed": 0, "random": 2, "starter": 0, "checkpoint": 0,
        },
        "league_checkpoint_identities": ["checkpoint:old"],
    })
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
            "ppo_metrics": ppo_metrics,
        },
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )

    class Pool:
        def schedule(self, *, count, seed):
            return [
                train_policy.OpponentMatch("current", 0),
                train_policy.OpponentMatch(
                    "checkpoint", 1, "ckpt-new", None, None, "checkpoint:new",
                ),
                train_policy.OpponentMatch("current", 0),
            ]

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        ppo_steps=3,
        device="cpu",
        resume_checkpoint=resume_path,
        opponent_pool=Pool(),
        rollout_fn=lambda **_kwargs: [_transition(done=True)],
    )

    assert metadata["ppo_metrics"]["league_composition"] == {
        "current": 4, "mixed": 0, "random": 2, "starter": 0, "checkpoint": 1,
    }
    assert metadata["ppo_metrics"]["league_checkpoint_identities"] == [
        "checkpoint:old", "checkpoint:new",
    ]


def test_resume_ppo_extension_updates_target_and_skips_behavior_cloning(
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
    save_checkpoint(
        resume_path,
        model=model,
        optimizer=optimizer,
        configuration=_resume_configuration(input_path, ppo_steps=1),
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
    ppo_calls = []

    def fail_if_behavior_cloning_runs(**_kwargs):
        raise AssertionError("behavior cloning must not rerun during PPO extension")

    def continue_ppo(**kwargs):
        ppo_calls.append({key: kwargs[key] for key in ("ppo_steps", "start_step")})
        return {
            "ppo_updates": 6,
            "rollout_count": 3,
            "early_stopped": False,
            "last_metrics": None,
            "promotion": None,
            "completed_steps": 3,
        }

    monkeypatch.setattr(train_policy, "epoch_minibatches", fail_if_behavior_cloning_runs)
    monkeypatch.setattr(train_policy, "run_ppo_training", continue_ppo)

    metadata = train_policy.train_behavior_clone(
        input_path=input_path,
        output_path=output_path,
        steps=1,
        batch_size=1,
        seed=7,
        ppo_steps=3,
        device="cpu",
        resume_checkpoint=resume_path,
        allow_ppo_extension=True,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert ppo_calls == [{"ppo_steps": 3, "start_step": 1}]
    assert metadata["ppo_updates"] == 6
    assert metadata["ppo_steps"] == 3
    assert payload["configuration"]["ppo_steps"] == 3
    assert payload["metadata"]["ppo_steps"] == 3
    assert payload["metrics"]["behavior_clone_updates"] == 1
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


def test_resume_allows_legacy_checkpoint_without_metadata_ppo_steps(tmp_path):
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    configuration = _resume_configuration(input_path, ppo_steps=2)
    payload = {
        "configuration": configuration,
        "progress": {"epoch": 1, "round": 0, "cursor": 0},
        "metrics": {
            "behavior_clone_updates": 1,
            "ppo_updates": 0,
            "ppo_metrics": None,
        },
        "metadata": train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    }

    train_policy._validate_resume_payload(
        payload,
        configuration=configuration,
        transition_count=1,
    )


@pytest.mark.parametrize("value", [True, -1, 1.0, "2", None, []])
def test_resume_rejects_malformed_metadata_ppo_steps(tmp_path, value):
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    configuration = _resume_configuration(input_path, ppo_steps=2)
    metadata = train_policy.checkpoint_metadata(transition_count=1, device="cpu")
    metadata["ppo_steps"] = value
    payload = {
        "configuration": configuration,
        "progress": {"epoch": 1, "round": 0, "cursor": 0},
        "metrics": {
            "behavior_clone_updates": 1,
            "ppo_updates": 0,
            "ppo_metrics": None,
        },
        "metadata": metadata,
    }

    with pytest.raises(ValueError, match="metadata ppo_steps"):
        train_policy._validate_resume_payload(
            payload,
            configuration=configuration,
            transition_count=1,
        )


def test_resume_rejects_contradictory_metadata_ppo_steps(tmp_path):
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    configuration = _resume_configuration(input_path, ppo_steps=2)
    metadata = train_policy.checkpoint_metadata(transition_count=1, device="cpu")
    metadata["ppo_steps"] = 1
    payload = {
        "configuration": configuration,
        "progress": {"epoch": 1, "round": 0, "cursor": 0},
        "metrics": {
            "behavior_clone_updates": 1,
            "ppo_updates": 0,
            "ppo_metrics": None,
        },
        "metadata": metadata,
    }

    with pytest.raises(ValueError, match="metadata ppo_steps"):
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
    expected_metrics = {
        **ppo_metrics,
        "league_composition": {
            "current": 0, "mixed": 0, "random": 0, "starter": 0, "checkpoint": 0,
        },
        "league_checkpoint_identities": [],
    }
    assert metadata["ppo_metrics"] == expected_metrics
    assert payload["metrics"]["ppo_metrics"] == expected_metrics
    assert payload["progress"] == {"epoch": 1, "round": 2, "cursor": 0}


def test_resume_ppo_progress_accepts_and_preserves_shaping_counts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet
    from scripts import train_policy

    input_path = tmp_path / "transitions.jsonl"
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    input_path.write_text(json.dumps(_transition(done=True)) + "\n", encoding="utf-8")
    ppo_metrics = _ppo_checkpoint_metrics(completed_steps=1)
    ppo_metrics.update({"shaping_count": 3, "truncation_count": 2})
    model = CompactPolicyNet().to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        resume_path, model=model, optimizer=optimizer,
        configuration=_resume_configuration(input_path, ppo_steps=2),
        epoch=1, round_index=1, cursor=0,
        metrics={"behavior_clone_updates": 1, "ppo_updates": 4, "ppo_metrics": ppo_metrics},
        metadata=train_policy.checkpoint_metadata(transition_count=1, device="cpu"),
    )

    monkeypatch.setattr(
        train_policy, "ppo_update",
        lambda **_kwargs: {
            "updates": 1, "early_stopped": False,
            "shaping_count": 2, "truncation_count": 1,
        },
    )
    metadata = train_policy.train_behavior_clone(
        input_path=input_path, output_path=output_path, steps=1, batch_size=1,
        seed=7, ppo_steps=2, device="cpu", resume_checkpoint=resume_path,
        rollout_fn=lambda **_kwargs: [_transition(done=True)],
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert metadata["ppo_metrics"]["shaping_count"] == 5
    assert payload["metrics"]["ppo_metrics"]["truncation_count"] == 3


def test_restore_checkpoint_accepts_legacy_state_without_training_only_head(tmp_path):
    torch = pytest.importorskip("torch")
    from kagriculture_agent.checkpoints import read_checkpoint, restore_checkpoint, save_checkpoint
    from kagriculture_agent.model import CompactPolicyNet, set_training_seed
    from scripts import train_policy

    path = tmp_path / "legacy.pt"
    set_training_seed(17)
    source = CompactPolicyNet().to("cpu")
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    save_checkpoint(
        path, model=source, optimizer=source_optimizer,
        configuration=_resume_configuration(tmp_path / "transitions.jsonl"),
        epoch=1, round_index=0, cursor=0,
        metrics={"behavior_clone_updates": 0, "ppo_updates": 0, "ppo_metrics": None},
        metadata=train_policy.checkpoint_metadata(transition_count=0, device="cpu"),
    )
    payload = read_checkpoint(path, map_location="cpu")
    payload["model_state_dict"] = {
        name: value for name, value in payload["model_state_dict"].items()
        if not name.startswith("market_active_head.")
    }
    payload["optimizer_state_dict"]["param_groups"][0]["params"] = (
        payload["optimizer_state_dict"]["param_groups"][0]["params"][:-2]
    )

    set_training_seed(17)
    target = CompactPolicyNet().to("cpu")
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    restore_checkpoint(payload, model=target, optimizer=target_optimizer)

    assert torch.count_nonzero(target.market_active_head.weight) == 0
    assert torch.count_nonzero(target.market_active_head.bias) == 0


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
