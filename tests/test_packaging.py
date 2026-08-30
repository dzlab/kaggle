import io
import runpy
import subprocess
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_git_archive_contains_root_entrypoint_and_agent_package():
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )

    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        names = set(tar.getnames())

    assert "main.py" in names
    assert "kagriculture_agent/__init__.py" in names
    assert "kagriculture_agent/policy.py" in names


def test_archived_root_entrypoint_executes(monkeypatch, tmp_path):
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        tar.extractall(tmp_path)

    monkeypatch.syspath_prepend(str(tmp_path))
    namespace = runpy.run_path(str(tmp_path / "main.py"))

    assert callable(namespace["agent"])
    assert namespace["agent"]({"step": 0}) == {"farmer": ["PASS"], "hands": [], "market": []}
