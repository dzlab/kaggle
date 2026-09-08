import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_ladder_parser_expands_stable_seeded_configurations_and_estimates_budgets():
    from scripts.benchmark_training_ladder import (
        expand_ladder,
        parse_ladder,
    )

    ladder = parse_ladder(
        json.dumps({
            "widths": [16, 32],
            "depths": [1, 2],
            "ppo_budgets": [100, 200],
            "seeds": [7, 11],
            "rollout_episodes": 3,
            "rollout_steps": 64,
        })
    )
    experiments = expand_ladder(ladder)

    assert len(experiments) == 16
    assert experiments[0]["seed"] == 7
    assert experiments[-1]["seed"] == 11
    assert experiments[0]["width"] == 16
    assert experiments[0]["depth"] == 1
    assert experiments[0]["ppo_steps"] == 100
    assert experiments[0]["parameter_estimate"] > 0
    assert experiments[0]["rollout_budget"] == 19_200
    assert experiments == expand_ladder(ladder)


def test_ladder_rejects_production_artifact_and_checkpoint_paths(tmp_path):
    from scripts.benchmark_training_ladder import validate_report_path

    report_root = tmp_path / "reports"
    with pytest.raises(ValueError, match="production"):
        validate_report_path(tmp_path / "models" / "learned_v1.json", report_root)
    with pytest.raises(ValueError, match="production"):
        validate_report_path(tmp_path / "checkpoints" / "trial.pt", report_root)


@pytest.mark.parametrize("name", ["model.json", "trained_model.json", "checkpoint.pt", "artifact.json"])
def test_report_validation_rejects_generic_protected_output_names(tmp_path, name):
    from scripts.benchmark_training_ladder import validate_report_path

    report_root = tmp_path / "reports"
    with pytest.raises(ValueError, match="artifact|checkpoint|model"):
        validate_report_path(report_root / name, report_root)


def test_report_validation_requires_explicit_root_and_keeps_output_inside_it(tmp_path):
    from scripts.benchmark_training_ladder import validate_report_path

    report_root = tmp_path / "reports"
    assert validate_report_path(report_root / "ladder.json", report_root) == (
        report_root / "ladder.json"
    )
    with pytest.raises(ValueError, match="production"):
        validate_report_path(report_root / "models" / "metrics.json", report_root)
    with pytest.raises(ValueError, match="report root"):
        validate_report_path(tmp_path / "outside.json", report_root)


@pytest.mark.parametrize(
    "production_component",
    ["model", "models", "artifact", "artifacts", "checkpoint", "checkpoints"],
)
def test_report_validation_rejects_report_root_nested_under_production_component(
    tmp_path, production_component,
):
    from scripts.benchmark_training_ladder import validate_report_path

    report_root = tmp_path / production_component / "reports"

    with pytest.raises(ValueError, match="production"):
        validate_report_path(report_root / "metrics.json", report_root)


def test_write_report_atomically_rejects_non_mapping_reports(tmp_path):
    from scripts.benchmark_training_ladder import write_report_atomically

    with pytest.raises(TypeError, match="mapping"):
        write_report_atomically(
            [], tmp_path / "reports" / "ladder.json", report_root=tmp_path / "reports",
        )


def test_report_is_stable_and_written_atomically_only_when_requested(tmp_path):
    from scripts.benchmark_training_ladder import (
        build_report,
        parse_ladder,
        write_report_atomically,
    )

    report = build_report(
        parse_ladder(json.dumps({"widths": [8], "depths": [1], "ppo_budgets": [5], "seeds": [3]}))
    )
    output = tmp_path / "reports" / "ladder.json"
    assert not output.exists()
    write_report_atomically(report, output, report_root=output.parent)
    first = output.read_bytes()
    write_report_atomically(report, output, report_root=output.parent)

    assert first == output.read_bytes()
    assert json.loads(first) == json.loads(output.read_bytes())
    assert not list(output.parent.glob("*.tmp"))


def test_default_cli_mode_is_dry_run():
    from scripts.benchmark_training_ladder import _parser

    args = _parser().parse_args([
        "--ladder", json.dumps({"widths": [8], "depths": [1], "ppo_budgets": [5], "seeds": [3]}),
    ])

    assert args.dry_run is True


def test_execute_ladder_records_callback_metrics_for_each_experiment():
    from scripts.benchmark_training_ladder import execute_ladder

    calls = []

    def callback(experiment):
        calls.append(dict(experiment))
        return {
            "elo": 1200 + experiment["width"],
            "safety": {"framework_errors": 0},
            "latency_ms": 2.5,
            "evaluation": {"win_rate": 0.75},
        }

    report = execute_ladder(
        {
            "widths": [8, 16], "depths": [1], "ppo_budgets": [5], "seeds": [3, 4],
        },
        callback,
    )

    assert report["dry_run"] is False
    assert len(calls) == 4
    assert [item["metrics"]["elo"] for item in report["experiments"]] == [1208, 1208, 1216, 1216]
    assert report["experiments"][0]["metrics"]["safety"]["framework_errors"] == 0
    assert report["experiments"][0]["metrics"]["evaluation"]["win_rate"] == 0.75


def test_dry_run_does_not_invoke_an_injected_callback():
    from scripts.benchmark_training_ladder import build_report

    calls = []
    report = build_report(
        {"widths": [8], "depths": [1], "ppo_budgets": [5], "seeds": [3]},
        execution_callback=lambda experiment: calls.append(experiment),
    )

    assert report["dry_run"] is True
    assert calls == []


def test_dry_run_does_not_import_gpu_training_dependencies(tmp_path):
    ladder = json.dumps({"widths": [8], "depths": [1], "ppo_budgets": [5], "seeds": [3]})
    script = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("dry-run imported GPU dependency")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = ["benchmark_training_ladder.py", "--ladder", %r, "--dry-run"]
from scripts.benchmark_training_ladder import main
assert main() == 0
""" % ladder
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"dry_run": true' in completed.stdout
