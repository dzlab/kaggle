import ast
import hashlib
import json
import importlib
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest


def _notebook_code_cells():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return notebook, [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def test_colab_notebook_is_a_small_setup_and_helper_launch_wrapper():
    notebook, code_cells = _notebook_code_cells()
    code = "\n".join(code_cells)

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len(notebook["cells"]) == 4
    assert len(code_cells) == 3

    clone_source, install_source, launch_source = code_cells
    assert "repo_url" in clone_source
    assert "repo_branch" in clone_source
    assert "git" in clone_source and "clone" in clone_source
    assert "--branch" in clone_source
    assert "/content/kaggle" in clone_source
    assert "Kaggriculture" in clone_source

    assert "pip" in install_source
    assert ".[training,observability]" in install_source
    assert "Kaggriculture" in install_source

    required_flags = (
        "--mount-drive",
        "--run-directory",
        "--ppo-target-steps",
        "--training-steps",
        "--training-batch-size",
        "--training-seed",
        "--training-checkpoint-interval",
        "--training-prior-checkpoint",
        "--no-training-offline-ppo-fallback",
        "--collection-seeds",
        "--collection-start-seed",
        "--collection-steps",
        "--collection-opponents",
        "--collection-seats",
        "--rollout-seeds",
        "--rollout-steps",
        "--development-seeds",
        "--development-opponents",
        "--development-steps",
        "--development-seats",
        "--holdout-seeds",
        "--holdout-opponents",
        "--holdout-steps",
        "--holdout-seats",
        "--workers",
        "--device",
        "--wandb-project",
        "--wandb-entity",
        "--smoke-opponent",
        "--smoke-seed",
        "--smoke-steps",
        "--plot",
    )
    assert "scripts/train.py" in launch_source
    assert all(flag in launch_source for flag in required_flags)
    assert "--wandb-run-name" not in launch_source
    assert "--no-wandb" not in launch_source

    workflow_references = ("scripts/train.py", *required_flags)
    setup_source = "\n".join(code_cells[:2])
    assert all(setup_source.count(reference) == 0 for reference in workflow_references)
    assert all(code.count(reference) == 1 for reference in workflow_references)
    assert "pip" not in clone_source
    assert "git" not in install_source

    old_inline_orchestration_markers = (
        "select_resume_checkpoint",
        "train_behavior_clone",
        "train_candidate",
        "training_contract",
        "OpponentPool",
        "collect_trajectories.py",
        "evaluate_artifact.py",
        "run_local.py",
        "TrainingTelemetry",
        "wandb.login",
        "export_checkpoint",
        "matplotlib",
        "load_metrics",
    )
    assert not any(marker in code for marker in old_inline_orchestration_markers)


def test_colab_notebook_executable_cells_are_valid_python():
    notebook, _ = _notebook_code_cells()
    parsed_code_cells = []

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            source = "\n".join(
                "pass" if line.lstrip().startswith(("%", "!", "--")) else line
                for line in "".join(cell.get("source", [])).splitlines()
            )
            parsed_code_cells.append(ast.parse(source, filename=f"cell-{index}"))

    assert len(parsed_code_cells) == 3
    clone_tree = parsed_code_cells[0]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
        for node in ast.walk(clone_tree)
    )


def test_colab_config_resolves_device_and_rejects_seed_overlap(monkeypatch, tmp_path):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cuda")
    config = colab_train.build_config(
        run_directory=tmp_path, device="auto", workers=2,
        development_seeds=(1, 2), holdout_seeds=(3,),
    )
    assert config.device == "cuda"
    assert config.workers == 2
    with pytest.raises(ValueError, match="overlap"):
        colab_train.build_config(development_seeds=(1,), holdout_seeds=(1,))


def test_default_wandb_run_name_describes_training_configuration(tmp_path):
    from scripts import colab_train

    config = colab_train.build_config(
        run_directory=tmp_path,
        ppo_target_steps=16,
        training_steps=25,
        training_seed=7,
        device="cpu",
    )
    timestamp = datetime(2026, 9, 9, 0, 43, 46, tzinfo=timezone.utc)

    assert colab_train.default_wandb_run_name(config, timestamp=timestamp) == (
        "kaggriculture-ppo16-bc25-seed7-20260909-004346"
    )


def test_colab_config_exposes_validated_experiment_identity(tmp_path, monkeypatch):
    from scripts import train

    monkeypatch.setattr(train, "resolve_device", lambda value: "cpu")
    args = train.parse_args([
        "--run-directory", str(tmp_path),
        "--experiment-id", "orbit-context-test",
        "--feature-variant", "experimental_context_v1",
        "--training-mode", "reduced_behavior_clone_then_ppo",
        "--dry-run",
    ])
    config = train.config_from_args(args)

    assert (config.experiment_id, config.feature_variant, config.training_mode) == (
        "orbit-context-test",
        "experimental_context_v1",
        "reduced_behavior_clone_then_ppo",
    )

    defaults = train.config_from_args(train.parse_args(["--dry-run"]))
    assert (defaults.experiment_id, defaults.feature_variant, defaults.training_mode) == (
        "orbit-policy-v1", "production_v1", "behavior_clone_then_ppo",
    )


def test_build_config_delegates_identity_validation_to_canonical_validator(
    tmp_path, monkeypatch,
):
    from scripts import train

    calls = []
    monkeypatch.setattr(train, "resolve_device", lambda value: "cpu")
    monkeypatch.setattr(
        train, "validate_training_identity",
        lambda experiment_id, feature_variant, training_mode: calls.append(
            (experiment_id, feature_variant, training_mode)
        ),
    )

    train.build_config(
        run_directory=tmp_path,
        device="cpu",
        mount_drive=False,
        experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    assert calls == [
        ("orbit-context-test", "experimental_context_v1", "reduced_behavior_clone_then_ppo"),
    ]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--feature-variant", "unknown"),
        ("--training-mode", "unknown"),
    ],
)
def test_colab_cli_rejects_unknown_experiment_variants_and_modes(flag, value):
    from scripts import train

    with pytest.raises(SystemExit):
        train.parse_args([flag, value, "--dry-run"])


def test_wandb_run_name_cli_option_overrides_dynamic_default(tmp_path):
    from scripts import colab_train

    args = colab_train.parse_args([
        "--run-directory", str(tmp_path),
        "--wandb-run-name", "manual-experiment-name",
        "--dry-run",
    ])

    config = colab_train.config_from_args(args)

    assert config.wandb_run_name == "manual-experiment-name"


def test_build_training_contract_records_requested_training_identity_and_options(monkeypatch, tmp_path):
    from scripts import train_policy

    trajectory_path = tmp_path / "transitions.jsonl"
    trajectory_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_policy, "resolve_device", lambda value: "cuda")

    contract = train_policy.build_training_contract(
        input_path=trajectory_path, steps=25, batch_size=256, seed=7,
        ppo_steps=16, device="auto", checkpoint_interval=25,
        prior_checkpoint=None, offline_ppo_fallback=False,
    )

    assert contract.transition_count == 1
    assert contract.configuration["input_trajectory"]["path"] == str(trajectory_path.resolve())
    assert contract.configuration["steps"] == 25
    assert contract.configuration["batch_size"] == 256
    assert contract.configuration["seed"] == 7
    assert contract.configuration["ppo_steps"] == 16
    assert contract.configuration["device"] == "cuda"
    assert contract.configuration["prior_checkpoint"] is None
    assert contract.configuration["offline_ppo_fallback"] is False
    assert contract.configuration["checkpoint_interval"] == 25


def test_colab_cli_parses_complete_workflow_parameters(tmp_path):
    from scripts import colab_train

    args = colab_train.parse_args([
        "--run-directory", str(tmp_path),
        "--trajectory-path", str(tmp_path / "input.jsonl"),
        "--ppo-target-steps", "32",
        "--training-steps", "9",
        "--training-batch-size", "64",
        "--training-seed", "11",
        "--collection-seeds", "8",
        "--collection-start-seed", "4",
        "--collection-steps", "48",
        "--collection-opponents", "pass", "random",
        "--rollout-seeds", "5", "6",
        "--development-seeds", "10", "11",
        "--holdout-seeds", "100", "101",
        "--development-opponents", "pass", "starter",
        "--holdout-opponents", "random", "starter",
        "--device", "cpu",
        "--workers", "3",
        "--wandb-project", "project",
        "--wandb-entity", "entity",
        "--no-wandb",
        "--mount-drive",
        "--smoke-seed", "12",
        "--smoke-steps", "49",
        "--plot",
        "--dry-run",
    ])

    assert args.run_directory == tmp_path
    assert args.trajectory_path == tmp_path / "input.jsonl"
    assert args.ppo_target_steps == 32
    assert args.training_steps == 9
    assert args.training_batch_size == 64
    assert args.training_seed == 11
    assert args.collection_seeds == 8
    assert args.collection_start_seed == 4
    assert args.collection_steps == 48
    assert args.collection_opponents == ["pass", "random"]
    assert args.rollout_seeds == [5, 6]
    assert args.development_seeds == [10, 11]
    assert args.holdout_seeds == [100, 101]
    assert args.development_opponents == ["pass", "starter"]
    assert args.holdout_opponents == ["random", "starter"]
    assert args.device == "cpu"
    assert args.workers == 3
    assert args.wandb_project == "project"
    assert args.wandb_entity == "entity"
    assert args.wandb is False
    assert args.mount_drive is True
    assert args.smoke_seed == 12
    assert args.smoke_steps == 49
    assert args.plot is True
    assert args.dry_run is True


def test_colab_cli_supports_explicit_none_checkpoint_and_fallback_opt_out():
    from scripts import colab_train

    defaults = colab_train.parse_args(["--dry-run"])
    explicit = colab_train.parse_args([
        "--training-prior-checkpoint", "none",
        "--no-training-offline-ppo-fallback",
        "--dry-run",
    ])
    enabled = colab_train.parse_args([
        "--training-offline-ppo-fallback",
        "--dry-run",
    ])

    assert defaults.training_prior_checkpoint is None
    assert defaults.training_offline_ppo_fallback is False
    assert explicit.training_prior_checkpoint is None
    assert explicit.training_offline_ppo_fallback is False
    assert enabled.training_offline_ppo_fallback is True
    config = colab_train.config_from_args(explicit)
    assert config.training_prior_checkpoint is None
    assert config.training_offline_ppo_fallback is False


def test_colab_workflow_builds_parameterized_commands(tmp_path, monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path,
        trajectory_path=tmp_path / "input.jsonl",
        ppo_target_steps=32,
        collection_seed_values=(4, 5),
        collection_steps=48,
        collection_opponents=("pass", "random"),
        rollout_seed_values=(6, 7),
        development_seeds=(10, 11),
        holdout_seeds=(100, 101),
        development_opponents=("pass", "starter"),
        holdout_opponents=("random", "starter"),
        device="cpu",
        workers=3,
        wandb_enabled=False,
        smoke_seed=12,
        smoke_steps=49,
    )

    collection = colab_train.build_collection_command(config)
    assert collection[:2] == [colab_train.sys.executable, str(colab_train.COLLECT_SCRIPT)]
    assert collection[collection.index("--seeds") + 1] == "2"
    assert collection[collection.index("--start-seed") + 1] == "4"
    assert collection[collection.index("--steps") + 1] == "48"
    assert collection[collection.index("--workers") + 1] == "3"
    assert collection[collection.index("--output") + 1] == str(tmp_path / "input.jsonl")
    assert collection[collection.index("--source-policy-identity") + 1] == "current"

    development = colab_train.build_evaluation_command(config, phase="development")
    assert development[development.index("--artifact") + 1] == str(config.stage_artifact_path)
    assert development[development.index("--identity") + 1] == "ppo32"
    assert development[development.index("--experiment-id") + 1] == config.experiment_id
    assert development[development.index("--feature-variant") + 1] == config.feature_variant
    assert development[development.index("--training-mode") + 1] == config.training_mode
    assert development[development.index("--seeds") + 1] == "2"
    assert development[development.index("--start-seed") + 1] == "10"
    assert development[development.index("--min-valid-games") + 1] == "4"
    assert development[development.index("--output") + 1] == str(config.development_report_path)

    holdout = colab_train.build_evaluation_command(config, phase="holdout")
    assert holdout[holdout.index("--start-seed") + 1] == "100"
    assert holdout[holdout.index("--min-valid-games") + 1] == "4"
    assert holdout[holdout.index("--output") + 1] == str(config.holdout_report_path)

    smoke = colab_train.build_smoke_command(config)
    assert smoke[:2] == [colab_train.sys.executable, str(colab_train.RUN_LOCAL_SCRIPT)]
    assert smoke[smoke.index("--seed") + 1] == "12"
    assert smoke[smoke.index("--steps") + 1] == "49"
    assert smoke[smoke.index("--candidate-artifact") + 1] == str(config.stage_artifact_path)


def test_colab_relative_paths_are_stable_when_cwd_changes(tmp_path, monkeypatch):
    from scripts import colab_train

    path_kwargs = {
        "run_directory": "runs/kagriculture",
        "trajectory_path": "inputs/trajectories.jsonl",
        "resume": "checkpoints/resume.pt",
        "training_prior_checkpoint": "checkpoints/prior.pt",
        "drive_mountpoint": "colab-drive",
        "plot_path": "plots/training.png",
    }
    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    first = colab_train.build_config(
        **path_kwargs, device="cpu", mount_drive=False,
    )

    other_cwd = tmp_path / "different-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    second = colab_train.build_config(
        **path_kwargs, device="cpu", mount_drive=False,
    )

    assert first.run_directory == second.run_directory
    assert first.trajectory_path == second.trajectory_path
    assert first.resume == second.resume
    assert first.training_prior_checkpoint == second.training_prior_checkpoint
    assert first.drive_mountpoint == second.drive_mountpoint
    assert first.plot_path == second.plot_path
    assert first.run_directory == colab_train.PROJECT_ROOT / "runs/kagriculture"
    assert first.trajectory_path == colab_train.PROJECT_ROOT / "inputs/trajectories.jsonl"
    assert first.resume == colab_train.PROJECT_ROOT / "checkpoints/resume.pt"
    assert first.training_prior_checkpoint == colab_train.PROJECT_ROOT / "checkpoints/prior.pt"
    assert first.drive_mountpoint == colab_train.PROJECT_ROOT / "colab-drive"
    assert first.plot_path == colab_train.PROJECT_ROOT / "plots/training.png"
    assert colab_train.build_collection_command(first) == colab_train.build_collection_command(second)
    assert colab_train.build_evaluation_command(first, phase="development") == colab_train.build_evaluation_command(second, phase="development")
    assert colab_train.build_smoke_command(first) == colab_train.build_smoke_command(second)
    assert first.stage_artifact_path == second.stage_artifact_path
    assert first.development_report_path == second.development_report_path
    assert first.smoke_replay_path == second.smoke_replay_path


def test_colab_dry_run_returns_plan_without_external_integrations(tmp_path, monkeypatch):
    from scripts import colab_train

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external integration used during dry-run")

    monkeypatch.setattr(colab_train, "mount_drive", forbidden)
    monkeypatch.setattr(colab_train, "run_command", forbidden)
    monkeypatch.setattr(colab_train, "initialize_telemetry", forbidden)
    monkeypatch.setattr(colab_train, "train_candidate", forbidden)

    result = colab_train.run_workflow(
        colab_train.build_config(
            run_directory=tmp_path,
            device="cpu",
            mount_drive=True,
            wandb_enabled=True,
        ),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.commands
    assert result.commands[0][1] == str(colab_train.COLLECT_SCRIPT)
    assert result.commands[-1][1] == str(colab_train.RUN_LOCAL_SCRIPT)
    assert result.development_evaluation_promoted is None


def test_colab_mount_drive_defaults_true_but_dry_run_does_not_mount(monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    args = colab_train.parse_args(["--dry-run"])
    assert args.mount_drive is True
    assert colab_train.build_config(device="cpu").mount_drive is True
    assert colab_train.parse_args(["--dry-run", "--no-mount-drive"]).mount_drive is False


def _complete_colab_evaluation_report(config, *, phase):
    seeds = config.development_seeds if phase == "development" else config.holdout_seeds
    opponents = config.development_opponents if phase == "development" else config.holdout_opponents
    seats = config.development_seats if phase == "development" else config.holdout_seats
    expected_matrix = [
        [opponent, seed, seat]
        for opponent in opponents
        for seed in seeds
        for seat in seats
    ]
    completeness = {
        "current": {
            "expected": expected_matrix,
            "expected_count": len(expected_matrix),
            "observed_count": len(expected_matrix),
            "missing": [], "duplicate": [], "extra": [], "invalid_records": [],
        },
        config.candidate_tag: {
            "expected": expected_matrix,
            "expected_count": len(expected_matrix),
            "observed_count": len(expected_matrix),
            "missing": [], "duplicate": [], "extra": [], "invalid_records": [],
        },
    }
    records = {
        "current": [
            {"opponent": opponent, "seed": seed, "seat": seat}
            for opponent, seed, seat in expected_matrix
        ],
        config.candidate_tag: [
            {"opponent": opponent, "seed": seed, "seat": seat}
            for opponent, seed, seat in expected_matrix
        ],
    }
    artifact_path = config.stage_artifact_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if not artifact_path.exists():
        artifact_path.write_text("fixture artifact", encoding="utf-8")
    return {
        "schema_version": 1,
        "configuration": {
            "experiment_id": config.experiment_id,
            "feature_variant": config.feature_variant,
            "training_mode": config.training_mode,
            "seed_values": list(seeds),
            "opponents": list(opponents),
            "seats": list(seats),
        },
        "artifact": {
            "identity": config.candidate_tag,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        },
        "expected_matrix": expected_matrix,
        "records": records,
        "matrix_completeness": completeness,
        "decision": {"status": "promote"},
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "different-experiment"),
        ("feature_variant", "production_v1"),
        ("training_mode", "pure_ppo"),
    ],
)
def test_run_evaluation_rejects_mismatched_identity_without_rewriting_report(
    tmp_path, monkeypatch, field, value,
):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        experiment_id="requested-experiment",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
        development_seeds=(0,), holdout_seeds=(100,),
    )
    report = _complete_colab_evaluation_report(config, phase="development")
    report["configuration"][field] = value
    report_path = config.development_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    original_report = report_path.read_bytes()

    def fake_run(command, *, check):
        report_path.write_bytes(original_report)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(colab_train, "run_command", fake_run)

    with pytest.raises(ValueError, match=field):
        colab_train._run_evaluation(
            config, phase="development", command=("fake-evaluator",),
            report_path=report_path,
        )

    assert report_path.read_bytes() == original_report


def test_fresh_rollout_does_not_rewrite_published_manifest(tmp_path, monkeypatch):
    from scripts import train_policy

    artifact = tmp_path / "candidate.json"
    artifact.write_text("artifact", encoding="utf-8")

    def fake_collect(*, output, candidate_artifact, **kwargs):
        manifest_path = Path(output).with_suffix(".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source_policy_identity": kwargs["source_policy_identity"],
            "experiment_id": kwargs["experiment_id"],
            "feature_variant": kwargs["feature_variant"],
            "training_mode": kwargs["training_mode"],
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
        Path(output).write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")

    monkeypatch.setattr("scripts.collect_trajectories.collect", fake_collect)
    original_write_text = Path.write_text

    def reject_manifest_rewrite(self, data, *args, **kwargs):
        if self.name.endswith(".manifest.json"):
            raise AssertionError("published rollout manifest was rewritten")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", reject_manifest_rewrite)
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

    manifest = json.loads(
        (tmp_path / "ppo-step-00000.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_policy_identity"].startswith("artifact:")


def test_evaluation_report_rejects_self_declared_malformed_matrix(tmp_path, monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        development_seeds=(0,), holdout_seeds=(100,),
    )
    report = _complete_colab_evaluation_report(config, phase="development")
    report["expected_matrix"][0] = ["pass", 999, 0]

    assert not colab_train.evaluation_report_is_complete(
        report, identity=config.candidate_tag,
        seed_values=config.development_seeds,
        opponents=config.development_opponents,
        seats=config.development_seats,
        artifact_path=config.stage_artifact_path,
    )


@pytest.mark.parametrize(
    "field,phase",
    [
        ("collection_seed_values", "collection"),
        ("development_seeds", "development"),
        ("holdout_seeds", "holdout"),
    ],
)
def test_colab_commands_reject_non_contiguous_seed_lists(tmp_path, monkeypatch, field, phase):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    kwargs = {
        "run_directory": tmp_path,
        "device": "cpu",
        "mount_drive": False,
        "development_seeds": (0, 1),
        "holdout_seeds": (100, 101),
    }
    kwargs[field] = (100, 102) if field == "holdout_seeds" else (0, 2)
    config = colab_train.build_config(**kwargs)

    with pytest.raises(ValueError, match="contiguous"):
        if phase == "collection":
            colab_train.build_collection_command(config)
        else:
            colab_train.build_evaluation_command(config, phase=phase)


@pytest.mark.parametrize("seat_field", ["collection_seats", "development_seats", "holdout_seats"])
@pytest.mark.parametrize("seats", [(), (0, 0), (0, 2)])
def test_colab_config_rejects_invalid_seat_tuples(tmp_path, monkeypatch, seat_field, seats):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    with pytest.raises(ValueError, match=seat_field):
        colab_train.build_config(
            run_directory=tmp_path, device="cpu", mount_drive=False,
            **{seat_field: seats},
        )


@pytest.mark.parametrize("resume_path", ["missing", "directory"])
def test_colab_explicit_resume_must_be_a_regular_file(tmp_path, monkeypatch, resume_path):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    resume = tmp_path / "resume.pt"
    if resume_path == "directory":
        resume.mkdir()
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        resume=resume, development_seeds=(0,), holdout_seeds=(100,),
    )
    training_calls = []

    def fake_run(command, *, check, capture_output=False):
        if Path(command[1]).name == colab_train.COLLECT_SCRIPT.name:
            config.trajectory_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(colab_train, "run_command", fake_run)
    monkeypatch.setattr(
        colab_train, "train_candidate",
        lambda *args, **kwargs: training_calls.append(True),
    )
    monkeypatch.setattr(colab_train, "initialize_telemetry", lambda config: None)

    with pytest.raises((FileNotFoundError, ValueError), match="resume"):
        colab_train.run_workflow(config)
    assert not training_calls


def test_plot_training_metrics_creates_parent_and_closes_figure(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        plot_path=tmp_path / "nested" / "telemetry.png",
    )
    config.training_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.training_metrics_path.write_text(
        '{"event":"behavior_clone","step":1,"loss":0.5}\n'
        '{"event":"ppo","step":1,"policy_loss":0.25}\n',
        encoding="utf-8",
    )
    closed = []
    monkeypatch.setattr(plt, "close", lambda figure: closed.append(figure))

    output = colab_train.plot_training_metrics(config)

    assert output == config.plot_path
    assert output.is_file()
    assert closed


@pytest.mark.parametrize("hash_value", [None, "", "not-a-sha256", "0" * 64])
def test_evaluation_report_requires_matching_artifact_sha256(tmp_path, monkeypatch, hash_value):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        development_seeds=(0,), holdout_seeds=(100,),
    )
    report = _complete_colab_evaluation_report(config, phase="development")
    if hash_value is None:
        del report["artifact"]["sha256"]
    else:
        report["artifact"]["sha256"] = hash_value

    assert not colab_train.evaluation_report_is_complete(
        report, identity=config.candidate_tag,
        seed_values=config.development_seeds,
        artifact_path=config.stage_artifact_path,
    )


def test_colab_rejects_stale_complete_report_when_development_evaluator_fails(tmp_path, monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        development_seeds=(0,), holdout_seeds=(100,),
    )
    config.development_report_path.write_text(
        json.dumps(_complete_colab_evaluation_report(config, phase="development")),
        encoding="utf-8",
    )
    invoked = []

    def fake_run(command, *, check, capture_output=False):
        script = Path(command[1]).name
        invoked.append(script)
        if script == colab_train.COLLECT_SCRIPT.name:
            config.trajectory_path.write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if script == colab_train.EVALUATE_SCRIPT.name:
            return SimpleNamespace(returncode=1, stdout="", stderr="evaluator failed")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(colab_train, "run_command", fake_run)
    monkeypatch.setattr(colab_train, "train_candidate", lambda *args, **kwargs: {})
    monkeypatch.setattr(colab_train, "initialize_telemetry", lambda config: None)
    monkeypatch.setattr(colab_train, "smoke_test_artifact", lambda config: None)

    with pytest.raises(RuntimeError, match="development evaluator"):
        colab_train.run_workflow(config)

    assert not config.development_report_path.exists()
    assert colab_train.EVALUATE_SCRIPT.name in invoked
    assert colab_train.RUN_LOCAL_SCRIPT.name not in invoked


def test_colab_smoke_failure_prevents_holdout_evaluation(tmp_path, monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        development_seeds=(0,), holdout_seeds=(100,),
    )
    invoked = []

    def fake_run(command, *, check, capture_output=False):
        script = Path(command[1]).name
        invoked.append(script)
        if script == colab_train.COLLECT_SCRIPT.name:
            config.trajectory_path.write_text("{}\n", encoding="utf-8")
        elif script == colab_train.EVALUATE_SCRIPT.name:
            phase = "holdout" if "holdout" in command[-1] else "development"
            report_path = config.holdout_report_path if phase == "holdout" else config.development_report_path
            report_path.write_text(
                json.dumps(_complete_colab_evaluation_report(config, phase=phase)),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(colab_train, "run_command", fake_run)
    monkeypatch.setattr(colab_train, "train_candidate", lambda *args, **kwargs: {})
    monkeypatch.setattr(colab_train, "initialize_telemetry", lambda config: None)

    def failed_smoke(config):
        invoked.append("smoke")
        raise RuntimeError("smoke failed")

    monkeypatch.setattr(colab_train, "smoke_test_artifact", failed_smoke)

    with pytest.raises(RuntimeError, match="smoke failed"):
        colab_train.run_workflow(config)

    assert invoked == [
        colab_train.COLLECT_SCRIPT.name,
        colab_train.EVALUATE_SCRIPT.name,
        "smoke",
    ]
    assert not config.holdout_report_path.exists()


def test_colab_workflow_does_not_automatically_promote_candidate(tmp_path, monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        development_seeds=(0,), holdout_seeds=(100,),
    )

    def fake_run(command, *, check, capture_output=False):
        script = Path(command[1]).name
        if script == colab_train.COLLECT_SCRIPT.name:
            config.trajectory_path.write_text("{}\n", encoding="utf-8")
        elif script == colab_train.EVALUATE_SCRIPT.name:
            phase = "holdout" if "holdout" in command[-1] else "development"
            report_path = config.holdout_report_path if phase == "holdout" else config.development_report_path
            report_path.write_text(
                json.dumps(_complete_colab_evaluation_report(config, phase=phase)),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_train(config, **kwargs):
        config.stage_checkpoint_path.write_bytes(b"checkpoint")
        config.stage_artifact_path.write_text("artifact", encoding="utf-8")
        return {}

    monkeypatch.setattr(colab_train, "run_command", fake_run)
    monkeypatch.setattr(colab_train, "train_candidate", fake_train)
    monkeypatch.setattr(colab_train, "initialize_telemetry", lambda config: None)
    monkeypatch.setattr(colab_train, "smoke_test_artifact", lambda config: None)

    result = colab_train.run_workflow(config)

    assert result.holdout_evaluation_complete is True
    assert config.stage_artifact_path.exists()
    assert not config.current_checkpoint_path.exists()


def test_colab_workflow_records_validation_before_finishing_telemetry(tmp_path, monkeypatch):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cpu")
    config = colab_train.build_config(
        run_directory=tmp_path, device="cpu", mount_drive=False,
        development_seeds=(0,), holdout_seeds=(100,),
    )

    def fake_run(command, *, check, capture_output=False):
        script = Path(command[1]).name
        if script == colab_train.COLLECT_SCRIPT.name:
            config.trajectory_path.write_text("{}\n", encoding="utf-8")
        elif script == colab_train.EVALUATE_SCRIPT.name:
            phase = "holdout" if "holdout" in command[-1] else "development"
            report_path = config.holdout_report_path if phase == "holdout" else config.development_report_path
            report_path.write_text(
                json.dumps(_complete_colab_evaluation_report(config, phase=phase)),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    class FakeTelemetry:
        def __init__(self):
            self.events = []
            self.finished = False

        def __call__(self, event, metrics=None, **values):
            self.events.append({"event": event, **dict(metrics or {}), **values})

        def finish(self):
            self.finished = True
            self.events.append({"event": "finish"})

    telemetry = FakeTelemetry()
    monkeypatch.setattr(colab_train, "run_command", fake_run)
    monkeypatch.setattr(colab_train, "train_candidate", lambda *args, **kwargs: {})
    monkeypatch.setattr(colab_train, "initialize_telemetry", lambda config: telemetry)
    monkeypatch.setattr(colab_train, "smoke_test_artifact", lambda config: None)

    result = colab_train.run_workflow(config)

    validation_events = [event for event in telemetry.events if event["event"] == "validation_summary"]
    assert {event["phase"] for event in validation_events} == {"development", "holdout"}
    assert {event["candidate"] for event in validation_events} == {"current", config.candidate_tag}
    training_complete = next(event for event in telemetry.events if event["event"] == "training_complete")
    assert training_complete["configuration"]["candidate_tag"] == config.candidate_tag
    assert training_complete["configuration"]["training_contract"]["ppo_steps"] == config.ppo_target_steps
    assert telemetry.events.index(validation_events[-1]) < telemetry.events.index({"event": "finish"})
    assert telemetry.finished is True
    assert result.holdout_evaluation_complete is True


def test_build_training_contract_normalizes_zero_steps_and_batch_size(monkeypatch, tmp_path):
    from scripts import train_policy

    trajectory_path = tmp_path / "transitions.jsonl"
    trajectory_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_policy, "resolve_device", lambda value: "cpu")

    contract = train_policy.build_training_contract(
        input_path=trajectory_path, steps=0, batch_size=0, device="cpu",
    )

    assert contract.configuration["steps"] == 1
    assert contract.configuration["batch_size"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", "2"),
        ("batch_size", 2.0),
        ("seed", True),
        ("ppo_steps", "16"),
        ("checkpoint_interval", 2.0),
        ("offline_ppo_fallback", 1),
    ],
)
def test_build_training_contract_rejects_invalid_raw_public_inputs(
    monkeypatch, tmp_path, field, value,
):
    from scripts import train_policy

    trajectory_path = tmp_path / "transitions.jsonl"
    trajectory_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_policy, "resolve_device", lambda value: "cpu")
    arguments = {
        "input_path": trajectory_path,
        "steps": 2,
        "batch_size": 1,
        "seed": 7,
        "ppo_steps": 16,
        "device": "cpu",
        "checkpoint_interval": 1,
        "offline_ppo_fallback": False,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        train_policy.build_training_contract(**arguments)


def test_colab_module_reload_order_accepts_current_training_contract(tmp_path):
    from scripts import train_policy
    import scripts.colab_train as colab_train

    importlib.reload(train_policy)
    importlib.reload(colab_train)
    contract = _training_contract(tmp_path, target=16)
    checkpoint = tmp_path / "policy-ppo16.pt"
    checkpoint.touch()
    payload = _resume_payload(contract, target=16, progress_round=1)

    selection = colab_train.select_resume_checkpoint(
        (checkpoint,), training_contract=contract,
        checkpoint_loader=lambda _path, *, map_location: payload,
        diagnostic=lambda _message: None,
    )

    assert selection.path == checkpoint


def _training_contract(tmp_path, *, target):
    from scripts import train_policy

    trajectory_path = tmp_path / "transitions.jsonl"
    return train_policy.TrainingContract(
        configuration={
            "input_trajectory": {
                "path": str(trajectory_path.resolve()), "sha256": "0" * 64,
            },
            "steps": 1,
            "batch_size": 1,
            "seed": 7,
            "ppo_steps": target,
            "device": "cpu",
            "prior_checkpoint": None,
            "offline_ppo_fallback": False,
            "ppo_config": train_policy.asdict(train_policy.PPOConfig()),
            "checkpoint_interval": 100,
        },
        transition_count=1,
    )


def _resume_payload(
    contract, *, target, progress_round, completed_steps=None,
    progress_epoch=1, progress_cursor=0, behavior_clone_updates=0,
):
    if completed_steps is None:
        completed_steps = progress_round
    configuration = {**contract.configuration, "ppo_steps": target}
    ppo_metrics = (
        None
        if completed_steps == 0
        else {
            "ppo_updates": completed_steps,
            "rollout_count": completed_steps,
            "early_stopped": False,
            "last_metrics": {},
            "promotion": None,
            "completed_steps": completed_steps,
        }
    )
    return {
        "configuration": configuration,
        "progress": {
            "epoch": progress_epoch, "round": progress_round, "cursor": progress_cursor,
        },
        "metrics": {
            "behavior_clone_updates": behavior_clone_updates,
            "ppo_updates": completed_steps,
            "ppo_metrics": ppo_metrics,
        },
        "metadata": {
            "transition_count": contract.transition_count,
            "device": contract.configuration["device"],
            "ppo_steps": target,
        },
    }


def test_resume_selection_skips_bad_candidates_and_ranks_completed_progress(tmp_path):
    from scripts import colab_train

    corrupt = tmp_path / "corrupt.pt"
    partial = tmp_path / "policy-ppo32.pt"
    completed = tmp_path / "policy-ppo16.pt"
    for path in (corrupt, partial, completed):
        path.touch()
    contract = _training_contract(tmp_path, target=32)
    payloads = {
        partial: _resume_payload(contract, target=32, progress_round=1),
        completed: _resume_payload(contract, target=16, progress_round=2),
    }
    diagnostics = []

    def load_checkpoint(path, *, map_location):
        if path == corrupt:
            raise ValueError("truncated checkpoint")
        return payloads[path]

    selection = colab_train.select_resume_checkpoint(
        (corrupt, partial, completed), training_contract=contract,
        checkpoint_loader=load_checkpoint, diagnostic=diagnostics.append,
    )

    assert selection.path == completed
    assert selection.ppo_target_steps == 16
    assert selection.progress_round == 2
    assert selection.completed_ppo_steps == 2
    assert selection.progress_epoch == 1
    assert selection.progress_cursor == 0
    assert selection.behavior_clone_updates == 0
    assert any("Skipping" in message and str(corrupt) in message for message in diagnostics)


def test_resume_selection_requires_canonical_training_contract(tmp_path):
    from scripts import colab_train

    checkpoint = tmp_path / "policy-ppo16.pt"
    checkpoint.touch()

    with pytest.raises(TypeError, match="training_contract"):
        colab_train.select_resume_checkpoint((checkpoint,))


@pytest.mark.parametrize(
    "progress_round,completed_steps,expected",
    [
        (17, 17, "progress.round"),
        (16, 17, "completed_steps"),
    ],
)
def test_validate_training_checkpoint_rejects_progress_beyond_saved_ppo_target(
    tmp_path, progress_round, completed_steps, expected,
):
    from kagriculture_agent.checkpoints import CheckpointError
    from scripts import train_policy

    contract = _training_contract(tmp_path, target=32)
    payload = _resume_payload(
        contract, target=16, progress_round=progress_round,
        completed_steps=completed_steps,
    )

    with pytest.raises(CheckpointError, match=expected):
        train_policy.validate_training_checkpoint(
            payload, contract=contract, allow_ppo_extension=True,
        )


def test_validate_training_checkpoint_checks_ppo_steps_before_round_shape(
    tmp_path,
):
    from kagriculture_agent.checkpoints import CheckpointError
    from scripts import train_policy

    contract = _training_contract(tmp_path, target=32)
    payload = _resume_payload(
        contract, target=16, progress_round=0, completed_steps=17,
    )

    with pytest.raises(CheckpointError, match="saved PPO target"):
        train_policy.validate_training_checkpoint(
            payload, contract=contract, allow_ppo_extension=True,
        )


def test_resume_selection_treats_missing_candidates_as_fresh_run(tmp_path):
    from scripts import colab_train

    selection = colab_train.select_resume_checkpoint(
        (tmp_path / "missing.pt",),
        training_contract=_training_contract(tmp_path, target=16),
    )

    assert selection is None


def test_resume_selection_propagates_drive_stat_errors(tmp_path, monkeypatch):
    from scripts import colab_train

    checkpoint = tmp_path / "policy-ppo16.pt"
    checkpoint.touch()
    contract = _training_contract(tmp_path, target=16)
    original_stat = Path.stat

    def fail_for_checkpoint(path):
        if path == checkpoint:
            raise PermissionError("Drive I/O unavailable")
        return original_stat(path)

    monkeypatch.setattr(Path, "stat", fail_for_checkpoint)

    with pytest.raises(PermissionError, match="Drive I/O unavailable"):
        colab_train.select_resume_checkpoint(
            (checkpoint,), training_contract=contract,
        )


def test_resume_selection_skips_semantically_incompatible_high_progress_candidate(tmp_path):
    from scripts import colab_train

    high_progress = tmp_path / "policy-ppo16-high-progress.pt"
    valid_lower_progress = tmp_path / "policy-ppo8-valid.pt"
    high_progress.touch()
    valid_lower_progress.touch()
    contract = _training_contract(tmp_path, target=16)
    payloads = {
        high_progress: _resume_payload(contract, target=16, progress_round=8),
        valid_lower_progress: _resume_payload(contract, target=8, progress_round=4),
    }
    diagnostics = []
    validator_calls = []

    def validate(*, path, payload, allow_ppo_extension):
        from scripts.colab_train import CheckpointCompatibilityError

        validator_calls.append((path, allow_ppo_extension))
        if path == high_progress:
            raise CheckpointCompatibilityError("input trajectory identity mismatch")

    selection = colab_train.select_resume_checkpoint(
        (high_progress, valid_lower_progress), training_contract=contract,
        checkpoint_loader=lambda path, *, map_location: payloads[path],
        checkpoint_validator=validate, diagnostic=diagnostics.append,
    )

    assert selection.path == valid_lower_progress
    assert validator_calls == [(high_progress, False), (valid_lower_progress, True)]
    assert any("input trajectory identity mismatch" in message for message in diagnostics)


def test_resume_selection_prioritizes_behavior_clone_progress_before_ppo_target(tmp_path):
    from scripts import colab_train

    stale_bc = tmp_path / "policy-ppo32-stale-bc.pt"
    completed_bc = tmp_path / "policy-ppo16-complete-bc.pt"
    stale_bc.touch()
    completed_bc.touch()
    contract = _training_contract(tmp_path, target=32)
    payloads = {
        stale_bc: _resume_payload(
            contract, target=32, progress_round=0,
            progress_epoch=0, progress_cursor=0, behavior_clone_updates=0,
        ),
        completed_bc: _resume_payload(
            contract, target=16, progress_round=0,
            progress_epoch=1, progress_cursor=0, behavior_clone_updates=1,
        ),
    }

    selection = colab_train.select_resume_checkpoint(
        (stale_bc, completed_bc), training_contract=contract,
        checkpoint_loader=lambda path, *, map_location: payloads[path],
        diagnostic=lambda _message: None,
    )

    assert selection.path == completed_bc
    assert selection.progress_epoch == 1
    assert selection.progress_cursor == 0
    assert selection.behavior_clone_updates == 1


def test_resume_selection_skips_nested_metadata_corruption(tmp_path):
    from scripts import colab_train

    checkpoint = tmp_path / "policy-ppo16.pt"
    checkpoint.touch()
    contract = _training_contract(tmp_path, target=16)
    payload = _resume_payload(contract, target=16, progress_round=1)
    payload["metadata"] = None
    diagnostics = []

    with pytest.raises(ValueError, match="No compatible resume checkpoint"):
        colab_train.select_resume_checkpoint(
            (checkpoint,), training_contract=contract,
            checkpoint_loader=lambda _path, *, map_location: payload,
            diagnostic=diagnostics.append,
        )

    assert any("metadata" in message for message in diagnostics)


def test_resume_selection_propagates_validator_programming_errors(tmp_path):
    from scripts import colab_train

    checkpoint = tmp_path / "policy-ppo16.pt"
    checkpoint.touch()
    contract = _training_contract(tmp_path, target=16)
    payload = _resume_payload(contract, target=16, progress_round=1)

    def validate(*, path, payload, allow_ppo_extension):
        raise TypeError("validator implementation bug")

    with pytest.raises(TypeError, match="validator implementation bug"):
        colab_train.select_resume_checkpoint(
            (checkpoint,), training_contract=contract,
            checkpoint_loader=lambda _path, *, map_location: payload,
            checkpoint_validator=validate, diagnostic=lambda _message: None,
        )


def test_resume_selection_rejects_all_present_candidates_when_none_are_compatible(tmp_path):
    from scripts import colab_train

    incompatible = tmp_path / "policy-ppo64.pt"
    malformed = tmp_path / "policy-ppo16.pt"
    incompatible.touch()
    malformed.touch()
    contract = _training_contract(tmp_path, target=32)
    diagnostics = []

    def load_checkpoint(path, *, map_location):
        if path == incompatible:
            return _resume_payload(contract, target=64, progress_round=4)
        raise ValueError("invalid checkpoint")

    with pytest.raises(ValueError, match="No compatible resume checkpoint"):
        colab_train.select_resume_checkpoint(
            (incompatible, malformed), training_contract=contract,
            checkpoint_loader=load_checkpoint, diagnostic=diagnostics.append,
        )
    assert any("exceeds requested target" in message for message in diagnostics)
    assert any("invalid checkpoint" in message for message in diagnostics)


def test_resume_selection_allows_completed_equal_target_reruns(tmp_path):
    from scripts import colab_train

    checkpoint = tmp_path / "policy-ppo16.pt"
    checkpoint.touch()
    contract = _training_contract(tmp_path, target=16)
    validator_calls = []

    def validate(*, path, payload, allow_ppo_extension):
        validator_calls.append((path, allow_ppo_extension))

    selection = colab_train.select_resume_checkpoint(
        (checkpoint,), training_contract=contract,
        checkpoint_loader=lambda _path, *, map_location: _resume_payload(
            contract, target=16, progress_round=16,
        ), checkpoint_validator=validate, diagnostic=lambda _message: None,
    )

    assert selection.path == checkpoint
    assert selection.ppo_target_steps == 16
    assert selection.completed_ppo_steps == 16
    assert validator_calls == [(checkpoint, False)]


def test_resume_selection_uses_target_then_mtime_after_progress_ties(tmp_path):
    from scripts import colab_train

    lower_target = tmp_path / "policy-ppo16.pt"
    higher_target = tmp_path / "policy-ppo32.pt"
    newer_same_target = tmp_path / "policy-ppo16-newer.pt"
    for path in (lower_target, higher_target, newer_same_target):
        path.touch()
    contract = _training_contract(tmp_path, target=32)
    os.utime(lower_target, (100, 100))
    os.utime(higher_target, (200, 200))
    os.utime(newer_same_target, (300, 300))
    payloads = {
        lower_target: _resume_payload(contract, target=16, progress_round=4),
        higher_target: _resume_payload(contract, target=32, progress_round=4),
        newer_same_target: _resume_payload(contract, target=16, progress_round=4),
    }

    selection = colab_train.select_resume_checkpoint(
        (lower_target, higher_target, newer_same_target),
        training_contract=contract,
        checkpoint_loader=lambda path, *, map_location: payloads[path],
        diagnostic=lambda _message: None,
    )

    assert selection.path == higher_target
    same_target_selection = colab_train.select_resume_checkpoint(
        (lower_target, newer_same_target),
        training_contract=_training_contract(tmp_path, target=16),
        checkpoint_loader=lambda path, *, map_location: payloads[path],
        diagnostic=lambda _message: None,
    )

    assert same_target_selection.path == newer_same_target
