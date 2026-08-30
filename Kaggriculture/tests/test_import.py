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
