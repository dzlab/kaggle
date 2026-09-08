import io
import runpy
import shutil
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
    assert "kagriculture_agent/candidates.py" in names


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


def test_submission_tarball_contains_only_runtime_files(tmp_path):
    archive = tmp_path / "submission.tar.gz"
    subprocess.run(
        ["tar", "--exclude=__pycache__", "-czf", str(archive), "-C", str(PROJECT_ROOT), "main.py", "kagriculture_agent"],
        check=True,
        capture_output=True,
    )

    with tarfile.open(archive, mode="r:gz") as tar:
        names = set(tar.getnames())

    assert "main.py" in names
    assert "kagriculture_agent/policy.py" in names
    assert all("__pycache__" not in name for name in names)
    assert all(not name.startswith(("reports/", "docs/", "tests/")) for name in names)


def test_submission_tarball_can_include_only_the_selected_policy_artifact(tmp_path):
    artifact = PROJECT_ROOT / "artifacts" / "learned_v1.json"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")
    try:
        archive = tmp_path / "submission.tar.gz"
        subprocess.run(
            ["tar", "--exclude=__pycache__", "-czf", str(archive), "-C", str(PROJECT_ROOT),
             "main.py", "kagriculture_agent", "artifacts/learned_v1.json"],
            check=True,
            capture_output=True,
        )
        with tarfile.open(archive, mode="r:gz") as tar:
            names = set(tar.getnames())
        assert "artifacts/learned_v1.json" in names
        assert all(not name.startswith(prefix) for name in names for prefix in (
            "tests/", "reports/", "replays/", "trajectories/", "checkpoints/", "scripts/train_",
        ))
        assert all("torch" not in name.lower() and "numpy" not in name.lower() for name in names)
    finally:
        artifact.unlink(missing_ok=True)
        try:
            artifact.parent.rmdir()
        except OSError:
            pass


def test_production_archive_contains_only_runtime_and_selected_artifact(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copy2(PROJECT_ROOT / "main.py", staging / "main.py")
    shutil.copytree(
        PROJECT_ROOT / "kagriculture_agent",
        staging / "kagriculture_agent",
        # Training-only modules are deliberately excluded from the production
        # package; the reviewed runtime allowlist in submission_smoke.py is
        # the authoritative packaging boundary.
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "checkpoints.py", "model.py", "rollouts.py",
        ),
    )
    model = staging / "models" / "learned_v1.json"
    model.parent.mkdir()
    model.write_text("{}\n", encoding="utf-8")

    archive = tmp_path / "production.tar.gz"
    subprocess.run(
        ["tar", "--exclude=__pycache__", "-czf", str(archive), "-C", str(staging),
         "main.py", "kagriculture_agent", "models/learned_v1.json"],
        check=True,
        capture_output=True,
    )

    with tarfile.open(archive, mode="r:gz") as tar:
        names = set(tar.getnames())

    assert "main.py" in names
    assert "models/learned_v1.json" in names
    assert "kagriculture_agent/learned_policy.py" in names
    assert all(
        not name.startswith(("tests/", "reports/", "docs/", "scripts/", "replays/", "trajectories/"))
        for name in names
    )
    assert all(
        token not in name.lower()
        for name in names
        for token in ("torch", "numpy", ".pt", ".pth", "checkpoint")
    )
