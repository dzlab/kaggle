from pathlib import Path

from main import agent


def test_agent_is_importable_and_callable():
    assert callable(agent)


def test_project_supports_python_310_without_global_uv_constraint():
    project = Path("pyproject.toml").read_text()

    assert 'requires-python = ">=3.10"' in project
    assert "environments =" not in project
    assert "kaggle-environments==1.32.7; python_version >= '3.11'" in project
