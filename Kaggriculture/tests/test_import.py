import subprocess
import sys
from pathlib import Path

from main import agent


def test_agent_is_importable_and_callable():
    assert callable(agent)


def test_project_requires_python_311_for_kaggle_environment_compatibility():
    project = Path("pyproject.toml").read_text()

    assert 'requires-python = ">=3.11"' in project
    assert "environments =" not in project
    assert "kaggle-environments==1.32.7; python_version >= '3.11'" in project


def test_project_declares_reproducible_setuptools_build_and_package_discovery():
    project = Path("pyproject.toml").read_text()

    assert '[build-system]' in project
    assert 'requires = ["setuptools==75.3.0"]' in project
    assert 'build-backend = "setuptools.build_meta"' in project
    assert '[tool.setuptools.packages.find]' in project
    assert 'include = ["kagriculture_agent*"]' in project


def test_production_import_does_not_require_training_dependencies():
    script = """
import builtins

real_import = builtins.__import__

def no_training_imports(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'numpy', 'torch'}:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = no_training_imports
from main import agent
assert callable(agent)
assert agent({'step': 0}) == {'farmer': ['PASS'], 'hands': [], 'market': []}
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], check=True)


def test_missing_artifact_and_missing_holdout_evidence_cannot_promote():
    from kagriculture_agent import candidates
    from scripts.evaluate import build_result_document

    assert "learned_v1" not in candidates.CANDIDATES
    document = build_result_document(
        config={
            "candidates": ["current", "learned_v1"],
            "opponents": ["pass"],
            "seed_values": [0],
            "seats": [0, 1],
            "min_valid_games": 1,
        },
        records=[],
    )

    assert document["selected_candidate"] is None
    assert document["metadata"]["selected_candidate"] is None
    assert document["metadata"]["selected_default_source"] == "development_only"


def test_production_default_remains_deterministic_current():
    import main

    assert main._policy.strategy_name == "current"
    assert main._policy.learned_policy.model_path is None
    assert main.agent({"step": 0}) == {"farmer": ["PASS"], "hands": [], "market": []}


def test_readme_documents_holdout_gate_and_rollback_contract():
    readme = Path("README.md").read_text(encoding="utf-8")

    for phrase in (
        "disjoint development and holdout",
        "fifth-percentile paired bank differential",
        "seat-balanced win rate",
        "rollback",
        "selected_candidate",
    ):
        assert phrase in readme
