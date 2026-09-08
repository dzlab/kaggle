import ast
import json
import importlib
import os
from pathlib import Path

import pytest


def test_colab_notebook_stages_candidates_and_resumes_safely():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )

    assert notebook["nbformat"] == 4
    assert "ppo_target_steps = 16" in code
    assert 'candidate_tag = f"ppo{ppo_target_steps}"' in code
    assert 'stage_checkpoint_path = run_dir / f"policy-{candidate_tag}.pt"' in code
    assert 'stage_artifact_path = run_dir / f"policy-{candidate_tag}.json"' in code
    assert "current_checkpoint_path = run_dir / 'policy.pt'" in code
    assert "select_resume_checkpoint(" in code
    assert "resume_selection.path" in code
    assert "import scripts.colab_train as colab_train" in code
    assert "importlib.reload(colab_train)" in code
    assert code.index("importlib.reload(train_policy)") < code.index("importlib.reload(colab_train)")
    assert "colab_train.select_resume_checkpoint(" in code
    assert "from scripts.colab_train import select_resume_checkpoint" not in code
    assert "training_contract = train_policy.build_training_contract(" in code
    assert "training_contract=training_contract" in code
    assert "checkpoint_validator=" not in code
    assert "train_policy._trajectory_identity(trajectory_path)" not in code
    assert "train_policy._read_transitions(trajectory_path)" not in code
    assert "train_policy._validate_resume_payload(" not in code
    assert "training_steps = 25" in code
    assert "training_batch_size = 256" in code
    assert "training_seed = 7" in code
    assert "training_prior_checkpoint = None" in code
    assert "training_offline_ppo_fallback = False" in code
    assert "saved_ppo_target < ppo_target_steps" in code
    assert "allow_ppo_extension=allow_ppo_extension" in code
    assert "allow_ppo_extension=True" not in code
    assert "export_checkpoint(stage_checkpoint_path, output_path)" in code
    assert "export_checkpoint(stage_checkpoint_path, stage_artifact_path)" in code
    assert "export_checkpoint(checkpoint_path, candidate_artifact)" not in code
    assert "output_path=stage_checkpoint_path" in code
    assert "candidate_artifact=stage_artifact_path" in code
    assert "training_device = 'cuda'" in code
    assert "workers=2" in code
    assert "candidate_artifact_callback=export_current" in code
    assert "output_path=current_checkpoint_path" not in code
    assert "export_checkpoint(current_checkpoint_path" not in code
    assert ".[training,observability]" in code
    assert "DEFAULT_WANDB_PROJECT" in code
    assert "DEFAULT_WANDB_ENTITY" in code
    assert "wandb.login" in code
    assert "WANDB_API_KEY" in code
    assert "telemetry_project = DEFAULT_WANDB_PROJECT" in code
    assert "training_metrics_path = run_dir / f'{candidate_tag}-training-metrics.jsonl'" in code
    assert "TrainingTelemetry(" in code
    assert "enable_wandb=True" in code
    assert "wandb_entity=wandb_entity" in code
    assert "strict=True" in code
    assert "telemetry_callback=record_training_event" in code
    assert "training_telemetry.finish()" in code
    assert "import matplotlib.pyplot as plt" in code
    assert "if not training_events:" in code
    assert "No {candidate_tag} training telemetry found" in code
    assert "change the target to 32" in markdown
    assert "retained separately" in markdown


def test_colab_notebook_has_rerunnable_development_and_gated_holdout_cells():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    code = "\n".join(code_cells)
    development_source = next(
        source for source in code_cells
        if "# Run the complete development gate" in source
    )
    holdout_source = next(
        source for source in code_cells
        if "# Run holdout only after" in source
    )
    development_index = code.index(development_source)
    holdout_index = code.index(holdout_source)

    assert development_index < holdout_index
    assert "scripts/evaluate_artifact.py" in development_source
    assert "'--output'" in development_source or '"--output"' in development_source
    assert "development_report_path" in development_source
    assert "check=False" in development_source
    assert "development_report_path.exists()" in development_source
    assert "development_decision.get('status')" in development_source
    assert "matrix_completeness" in development_source
    assert "discard means continue training" in development_source
    assert "not promoted" in development_source

    assert "scripts/evaluate_artifact.py" in holdout_source
    assert "if development_evaluation_promoted:" in holdout_source
    assert "holdout_report_path" in holdout_source
    assert "holdout_seed_values" in holdout_source
    assert "development_seed_values" in holdout_source
    assert "holdout_seed_values != development_seed_values" in holdout_source
    assert "--start-seed" in holdout_source
    assert "--output" in holdout_source
    assert "holdout evaluation skipped" in holdout_source.lower()
    assert "holdout_min_valid_games = len(holdout_seed_values) * len(holdout_opponents)" in holdout_source
    assert "holdout_evaluation_complete" in holdout_source
    assert "holdout_decision.get('status') in {'promote', 'discard'}" in holdout_source

    holdout_tree = ast.parse(holdout_source, filename="holdout-cell")
    guarded_evaluation = [
        node
        for node in ast.walk(holdout_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert guarded_evaluation, "holdout cell must invoke the evaluator"
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "development_evaluation_promoted"
        and any(
            any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "subprocess"
                and child.func.attr == "run"
                for child in ast.walk(statement)
            )
            for statement in node.body
        )
        for node in ast.walk(holdout_tree)
    ), "holdout evaluator must be inside the development promotion gate"


def test_colab_notebook_computes_per_seat_thresholds_for_both_evaluations():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "development_min_valid_games = len(development_seed_values) * len(development_opponents)" in code
    assert "'--min-valid-games', str(development_min_valid_games)" in code
    assert "holdout_min_valid_games = len(holdout_seed_values) * len(holdout_opponents)" in code
    assert "'--min-valid-games', str(holdout_min_valid_games)" in code
    assert "holdout_seed_values" in code
    assert "development_seed_values" in code


def test_colab_notebook_filters_prior_checkpoints_before_building_opponent_pool():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "prior_checkpoint_candidates" in code
    assert "compatible_prior_checkpoints = []" in code
    assert "colab_train.select_resume_checkpoint(" in code
    assert "except ValueError as exc:" in code
    assert "continue" in code
    assert "train_policy.OpponentPool(" in code
    assert "previous_checkpoints=compatible_prior_checkpoints" in code
    assert "opponent_pool=training_opponent_pool" in code
    assert "malformed or incompatible checkpoint" in code


def test_colab_notebook_executable_cells_are_valid_python():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            source = "\n".join(
                "pass" if line.lstrip().startswith(("%", "!")) else line
                for line in "".join(cell.get("source", [])).splitlines()
            )
            ast.parse(source, filename=f"cell-{index}")


def test_colab_telemetry_callback_is_passed_only_to_training():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    training_cell = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "training_contract =" in "".join(cell.get("source", []))
    )
    tree = ast.parse(training_cell, filename="training-cell")
    contract_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_training_contract"
    )
    train_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_behavior_clone"
    )

    assert not any(keyword.arg == "telemetry_callback" for keyword in contract_call.keywords)
    assert [
        keyword.value.id
        for keyword in train_call.keywords
        if keyword.arg == "telemetry_callback"
        and isinstance(keyword.value, ast.Name)
    ] == ["record_training_event"]


def test_colab_telemetry_is_stage_scoped_and_labels_plot():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "colab_gpu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    training_cell = next(source for source in code_cells if "training_contract =" in source)
    plotting_cell = next(source for source in code_cells if "load_metrics" in source)
    training_tree = ast.parse(training_cell, filename="training-cell")
    plotting_tree = ast.parse(plotting_cell, filename="plotting-cell")

    metrics_assignment = next(
        node for node in ast.walk(training_tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "training_metrics_path" for target in node.targets)
    )
    metrics_path = metrics_assignment.value
    assert isinstance(metrics_path, ast.BinOp)
    assert isinstance(metrics_path.op, ast.Div)
    assert isinstance(metrics_path.left, ast.Name)
    assert metrics_path.left.id == "run_dir"
    assert isinstance(metrics_path.right, ast.JoinedStr)
    assert any(
        isinstance(value, ast.FormattedValue)
        and isinstance(value.value, ast.Name)
        and value.value.id == "candidate_tag"
        for value in metrics_path.right.values
    )
    assert any(
        isinstance(value, ast.Constant)
        and value.value == "-training-metrics.jsonl"
        for value in metrics_path.right.values
    )

    adapter = next(
        node for node in training_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "record_training_event"
    )
    telemetry_calls = [
        node for node in ast.walk(adapter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "training_telemetry"
    ]
    assert len(telemetry_calls) == 1
    telemetry_call = telemetry_calls[0]
    assert [
        argument.id for argument in telemetry_call.args
        if isinstance(argument, ast.Name)
    ] == ["event", "metrics"]
    assert {
        keyword.arg for keyword in telemetry_call.keywords
        if keyword.arg is not None
    } >= {"candidate_tag", "ppo_target_steps"}

    train_call = next(
        node for node in ast.walk(training_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_behavior_clone"
    )
    callback_values = [
        keyword.value.id for keyword in train_call.keywords
        if keyword.arg == "telemetry_callback"
        and isinstance(keyword.value, ast.Name)
    ]
    assert callback_values == ["record_training_event"]

    load_call = next(
        node for node in ast.walk(plotting_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_metrics"
    )
    assert [argument.id for argument in load_call.args if isinstance(argument, ast.Name)] == [
        "training_metrics_path"
    ]
    assert any(
        isinstance(node, ast.JoinedStr)
        and any(
            isinstance(value, ast.FormattedValue)
            and isinstance(value.value, ast.Name)
            and value.value.id == "candidate_tag"
            for value in node.values
        )
        for node in ast.walk(plotting_tree)
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
