import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.submission_smoke import build_submission_archive, smoke_test_archive


PROJECT_ROOT = Path(__file__).parents[1]


def test_submission_archive_contains_only_runtime_and_executes_without_site_packages(tmp_path):
    archive = tmp_path / "submission.tar.gz"
    repeat = tmp_path / "submission-repeat.tar.gz"

    manifest = build_submission_archive(PROJECT_ROOT, archive)
    build_submission_archive(PROJECT_ROOT, repeat)

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())

    assert "main.py" in names
    assert "kagriculture_agent/learned_policy.py" in names
    assert "kagriculture_agent/experimental_features.py" in names
    assert "kagriculture_agent/runtime_identity.py" in names
    assert manifest["runtime_dependencies"] == []
    assert all(not name.startswith(("tests/", "docs/", "scripts/", "reports/", "replays/")) for name in names)
    smoke_test_archive(archive)
    assert archive.read_bytes() == repeat.read_bytes()


def test_runtime_identity_and_learned_policy_import_without_scripts(tmp_path):
    archive = tmp_path / "submission.tar.gz"
    build_submission_archive(PROJECT_ROOT, archive)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted)

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from kagriculture_agent.runtime_identity import (\n"
            "    ACTION_REPRESENTATIONS, DEFAULT_ACTION_REPRESENTATION,\n"
            "    validate_action_representation,\n"
            ")\n"
            "from kagriculture_agent.learned_policy import artifact_tensor_shapes\n"
            "assert DEFAULT_ACTION_REPRESENTATION in ACTION_REPRESENTATIONS\n"
            "validate_action_representation(DEFAULT_ACTION_REPRESENTATION)\n"
            "assert artifact_tensor_shapes()\n",
        ],
        cwd=extracted,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(extracted)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_submission_archive_can_include_selected_artifact_only(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("def agent(obs): return {'farmer':['PASS'], 'hands':[], 'market':[]}")
    package = project / "kagriculture_agent"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "credentials.json").write_text("secret")
    artifact = project / "artifacts" / "learned_v1.json"
    artifact.parent.mkdir()
    artifact.write_text('{"workers": [], "market_orders": []}')
    archive = tmp_path / "submission.tar.gz"

    manifest = build_submission_archive(project, archive, artifact=artifact)

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "artifacts/learned_v1.json" in names
    assert "kagriculture_agent/credentials.json" not in names
    assert manifest["artifact"] == "artifacts/learned_v1.json"


def test_submission_archive_rejects_missing_selected_artifact(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("def agent(obs): return {}")

    with pytest.raises(ValueError, match="artifact does not exist"):
        build_submission_archive(project, tmp_path / "submission.tar.gz", artifact=project / "missing.json")


def test_submission_archive_rejects_symlinked_runtime_package(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("def agent(obs): return {}")
    outside = tmp_path / "outside-package"
    outside.mkdir()
    (outside / "__init__.py").write_text("")
    (project / "kagriculture_agent").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="package must not be a symlink"):
        build_submission_archive(project, tmp_path / "submission.tar.gz")


def test_submission_archive_rejects_malformed_proposal_artifact(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("def agent(obs): return {}")
    package = project / "kagriculture_agent"
    package.mkdir()
    (package / "__init__.py").write_text("")
    artifact = project / "artifacts" / "proposal.json"
    artifact.parent.mkdir()
    artifact.write_text('{"workers": "not-a-list", "market_orders": []}')

    with pytest.raises(ValueError, match="proposal artifact"):
        build_submission_archive(project, tmp_path / "submission.tar.gz", artifact=artifact)


def test_smoke_test_rejects_nonregular_tar_members(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in (
            ("main.py", b"def agent(obs): return {}"),
            ("kagriculture_agent/__init__.py", b""),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo("kagriculture_agent/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        tar.addfile(link)

    with pytest.raises(RuntimeError, match="non-regular"):
        smoke_test_archive(archive)
