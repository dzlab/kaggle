import io
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.submission_smoke import build_submission_archive, smoke_test_archive


PROJECT_ROOT = Path(__file__).parents[1]


def _valid_latency_document(artifact):
    from scripts import benchmark_rollouts

    return {
        "schema_version": 1,
        "device": "cpu",
        "candidate_artifact": str(artifact),
        "candidate_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "results": [
            benchmark_rollouts.summarize_run(
                worker_count=workers,
                game_count=1,
                environment_steps=200000,
                rollout_seconds=1.0,
                inference_latencies_ms=[1.0],
            )
            for workers in (1, 2, 4, 8)
        ],
        "gate": {
            "four_worker_result_present": True,
            "inference_p95_ms_threshold": 10.0,
            "real_engine_kept": True,
            "simulator_required": False,
            "throughput_steps_per_minute_threshold": 100000.0,
        },
    }


def _valid_holdout_document(artifact):
    expected_matrix = [
        [opponent, 100, seat]
        for opponent in ("pass", "random", "starter")
        for seat in (0, 1)
    ]
    records = [
        {"opponent": opponent, "seed": seed, "seat": seat}
        for opponent, seed, seat in expected_matrix
    ]
    completeness = {
        "expected": expected_matrix,
        "expected_count": len(expected_matrix),
        "observed_count": len(expected_matrix),
        "missing": [],
        "duplicate": [],
        "extra": [],
        "invalid_records": [],
    }
    return {
        "schema_version": 1,
        "configuration": {
            "seed_values": [100],
            "opponents": ["pass", "random", "starter"],
            "seats": [0, 1],
        },
        "artifact": {
            "identity": "candidate",
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "expected_matrix": expected_matrix,
        "records": {"current": records, "candidate": records},
        "matrix_completeness": {"current": completeness, "candidate": completeness},
        "decision": {"status": "promote"},
    }


def _minimal_promoted_archive_inputs(tmp_path):
    from scripts import benchmark_rollouts

    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(
        "def agent(obs): return {'farmer':['PASS'], 'hands':[], 'market':[]}\n",
    )
    package = project / "kagriculture_agent"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "policy.py").write_text(
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        return {'farmer':['PASS'], 'hands':[], 'market':[]}\n",
    )
    artifact = tmp_path / "stage-artifact.json"
    artifact.write_text('{"workers": [], "market_orders": []}', encoding="utf-8")
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps(_valid_holdout_document(artifact)), encoding="utf-8")
    latency = tmp_path / "latency.json"
    results = [
        benchmark_rollouts.summarize_run(
            worker_count=workers,
            game_count=1,
            environment_steps=200000,
            rollout_seconds=1.0,
            inference_latencies_ms=[1.0],
        )
        for workers in (1, 2, 4, 8)
    ]
    latency.write_text(json.dumps({
        "schema_version": 1,
        "device": "cpu",
        "candidate_artifact": str(artifact),
        "candidate_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "results": results,
        "gate": {
            "four_worker_result_present": True,
            "inference_p95_ms_threshold": 10.0,
            "real_engine_kept": True,
            "simulator_required": False,
            "throughput_steps_per_minute_threshold": 100000.0,
        },
    }), encoding="utf-8")
    return project, artifact, holdout, latency


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
    assert "models/learned_v1.json" in names
    assert "kagriculture_agent/credentials.json" not in names
    assert manifest["artifact"] == "models/learned_v1.json"


def test_submission_archive_maps_external_artifact_without_overwriting_project_model(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("def agent(obs): return {'farmer':['PASS'], 'hands':[], 'market':[]}")
    package = project / "kagriculture_agent"
    package.mkdir()
    (package / "__init__.py").write_text("")
    models = project / "models"
    models.mkdir()
    production = models / "learned_v1.json"
    production.write_text('{"workers": [], "market_orders": []}', encoding="utf-8")
    external = tmp_path / "stage-artifact.json"
    external.write_text('{"workers": [{"action": "PASS"}], "market_orders": []}', encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    manifest = build_submission_archive(project, archive, artifact=external)

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        extracted = json.loads(tar.extractfile("models/learned_v1.json").read())
    assert names.count("models/learned_v1.json") == 1
    assert extracted["workers"] == [{"action": "PASS"}]
    assert manifest["artifact"] == "models/learned_v1.json"
    assert json.loads(production.read_text(encoding="utf-8"))["workers"] == []


def test_promoted_archive_entrypoint_uses_bundled_artifact_candidate(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(
        "def agent(obs): return {'farmer':['PASS'], 'hands':[], 'market':[]}\n",
    )
    package = project / "kagriculture_agent"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "policy.py").write_text(
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        return {'farmer':['PASS'], 'hands':[], 'market':[]}\n",
    )
    (package / "candidates.py").write_text(
        "import json\n"
        "def artifact_candidate_policy(path):\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        payload = json.load(handle)\n"
        "    if payload.get('marker') != 'learned':\n"
        "        raise ValueError('not learned')\n"
        "    return lambda obs: {'farmer':['FERTILIZE'], 'hands':[], 'market':[]}\n",
    )
    artifact = tmp_path / "stage-artifact.json"
    artifact.write_text(
        '{"workers": [], "market_orders": [], "marker": "learned"}',
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps(_valid_holdout_document(artifact)), encoding="utf-8")
    latency = tmp_path / "latency.json"
    latency.write_text(json.dumps(_valid_latency_document(artifact)), encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    build_submission_archive(
        project,
        archive,
        artifact=artifact,
        holdout_report=holdout,
        latency_report=latency,
    )
    smoke_test_archive(archive, "models/learned_v1.json")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted)
    result = subprocess.run(
        [
            sys.executable, "-S", "-c",
            "import main; assert main.agent({})['farmer'] == ['FERTILIZE']",
        ],
        cwd=extracted,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(extracted)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_promoted_archive_embeds_evidence_and_self_excluding_integrity_manifest(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(
        "def agent(obs): return {'farmer':['PASS'], 'hands':[], 'market':[]}\n",
    )
    package = project / "kagriculture_agent"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "policy.py").write_text(
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        return {'farmer':['PASS'], 'hands':[], 'market':[]}\n",
    )
    artifact = tmp_path / "stage-artifact.json"
    artifact.write_text('{"workers": [], "market_orders": []}', encoding="utf-8")
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps(_valid_holdout_document(artifact)) + "\n", encoding="utf-8")
    latency = tmp_path / "cpu-latency.json"
    latency.write_text(json.dumps(_valid_latency_document(artifact)), encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    build_submission_archive(
        project,
        archive,
        artifact=artifact,
        holdout_report=holdout,
        latency_report=latency,
    )
    smoke_test_archive(archive, "models/learned_v1.json")

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        integrity = json.loads(tar.extractfile("manifest.json").read())
        embedded_holdout = tar.extractfile("evidence/holdout.json").read()
        embedded_latency = tar.extractfile("evidence/cpu-latency.json").read()
        member_bytes = {
            name: tar.extractfile(name).read()
            for name in names if name != "manifest.json"
        }

    assert {"evidence/holdout.json", "evidence/cpu-latency.json", "manifest.json"} <= names
    assert "manifest.json" not in integrity["members"]
    assert integrity["members"]["evidence/holdout.json"] == hashlib.sha256(embedded_holdout).hexdigest()
    assert integrity["members"]["evidence/cpu-latency.json"] == hashlib.sha256(embedded_latency).hexdigest()
    assert integrity["evidence"]["holdout"]["sha256"] == hashlib.sha256(holdout.read_bytes()).hexdigest()
    assert integrity["evidence"]["latency"]["sha256"] == hashlib.sha256(latency.read_bytes()).hexdigest()
    assert all(
        integrity["members"][name] == hashlib.sha256(content).hexdigest()
        for name, content in member_bytes.items()
    )


def test_smoke_rejects_embedded_holdout_that_does_not_promote(tmp_path):
    project, artifact, holdout, latency = _minimal_promoted_archive_inputs(tmp_path)
    holdout.write_text('{"decision": {"status": "discard"}}', encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    build_submission_archive(
        project, archive, artifact=artifact,
        holdout_report=holdout, latency_report=latency,
    )

    with pytest.raises(RuntimeError, match="holdout.*promote"):
        smoke_test_archive(archive, "models/learned_v1.json")


def test_smoke_rejects_promoted_holdout_with_incomplete_evidence(tmp_path):
    project, artifact, holdout, latency = _minimal_promoted_archive_inputs(tmp_path)
    holdout.write_text('{"decision": {"status": "promote"}}', encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    build_submission_archive(
        project, archive, artifact=artifact,
        holdout_report=holdout, latency_report=latency,
    )

    with pytest.raises(RuntimeError, match="holdout.*evidence|promotion evidence"):
        smoke_test_archive(archive, "models/learned_v1.json")


@pytest.mark.parametrize("holdout_payload", [b"", b"[]", b'{"decision": {}}'])
def test_smoke_rejects_malformed_embedded_holdout_evidence(tmp_path, holdout_payload):
    project, artifact, holdout, latency = _minimal_promoted_archive_inputs(tmp_path)
    archive = tmp_path / "submission.tar.gz"
    build_submission_archive(
        project, archive, artifact=artifact,
        holdout_report=holdout, latency_report=latency,
    )

    rewritten = tmp_path / "rewritten-submission.tar.gz"
    with tarfile.open(archive, "r:gz") as source, tarfile.open(rewritten, "w:gz") as target:
        members = source.getmembers()
        contents = {
            member.name: source.extractfile(member).read()
            for member in members
        }
        contents["evidence/holdout.json"] = holdout_payload
        integrity = json.loads(contents["manifest.json"])
        integrity["members"]["evidence/holdout.json"] = hashlib.sha256(
            holdout_payload
        ).hexdigest()
        integrity["evidence"]["holdout"]["sha256"] = hashlib.sha256(
            holdout_payload
        ).hexdigest()
        contents["manifest.json"] = (
            json.dumps(integrity, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        for member in members:
            content = contents[member.name]
            info = tarfile.TarInfo(member.name)
            info.size = len(content)
            target.addfile(info, io.BytesIO(content))

    with pytest.raises(RuntimeError, match="promotion evidence|holdout"):
        smoke_test_archive(rewritten, "models/learned_v1.json")


@pytest.mark.parametrize("mutation", ["latency_schema", "artifact_hash"])
def test_smoke_rejects_embedded_latency_without_complete_cpu_evidence(tmp_path, mutation):
    project, artifact, holdout, latency = _minimal_promoted_archive_inputs(tmp_path)
    report = json.loads(latency.read_text(encoding="utf-8"))
    if mutation == "latency_schema":
        report["results"][2]["policy_inference_latency_valid"] = False
    else:
        report["candidate_artifact_sha256"] = "0" * 64
    latency.write_text(json.dumps(report), encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    build_submission_archive(
        project, archive, artifact=artifact,
        holdout_report=holdout, latency_report=latency,
    )

    with pytest.raises(RuntimeError, match="latency"):
        smoke_test_archive(archive, "models/learned_v1.json")


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
