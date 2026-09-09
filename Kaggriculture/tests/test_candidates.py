from pathlib import Path

import pytest

from kagriculture_agent import candidates


def test_artifact_candidate_policy_loads_artifact_once_for_reusable_callable(monkeypatch, tmp_path):
    artifact = tmp_path / "candidate.json"
    artifact.write_text("{}")
    calls = []

    class LoadedPolicy:
        def act(self, observation):
            return {"observation": observation}

    monkeypatch.setattr(
        candidates, "load_exported_policy",
        lambda path: calls.append(Path(path)) or LoadedPolicy(),
    )

    policy = candidates.artifact_candidate_policy(artifact)

    assert policy({"turn": 1}) == {"observation": {"turn": 1}}
    assert policy({"turn": 2}) == {"observation": {"turn": 2}}
    assert calls == [artifact]


def test_artifact_candidate_policy_reports_invalid_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "candidate.json"
    artifact.write_text("bad")
    monkeypatch.setattr(
        candidates, "load_exported_policy",
        lambda path: (_ for _ in ()).throw(ValueError("invalid")),
    )

    with pytest.raises(ValueError, match="candidate artifact is not valid: ValueError: invalid"):
        candidates.artifact_candidate_policy(artifact)


def test_candidate_lifecycle_reuses_one_loaded_artifact_and_exposes_policy_hook(monkeypatch, tmp_path):
    artifact = tmp_path / "learned-v1.json"
    artifact.write_text("{}")
    calls = []
    loaded = object()
    monkeypatch.setenv(candidates.LEARNED_V1_ARTIFACT_ENV, str(artifact))
    monkeypatch.setattr(candidates, "learned_v1_artifact_path", lambda: artifact)
    monkeypatch.setattr(
        candidates,
        "load_exported_policy",
        lambda path: calls.append(Path(path)) or loaded,
    )

    assert candidates._learned_v1_available()
    metadata = candidates.candidate_metadata(candidates.LEARNED_V1)
    candidate = candidates.candidate_policy(candidates.LEARNED_V1)

    assert calls == [artifact]
    assert metadata["model_identity"] == candidates.LEARNED_V1
    assert candidate.__self__.learned_policy.model_path is loaded


def test_candidate_lifecycle_reuses_invalid_artifact_load_failure(monkeypatch, tmp_path):
    artifact = tmp_path / "invalid.json"
    artifact.write_text("invalid")
    calls = []
    monkeypatch.setenv(candidates.LEARNED_V1_ARTIFACT_ENV, str(artifact))
    monkeypatch.setattr(candidates, "learned_v1_artifact_path", lambda: artifact)

    def fail(path):
        calls.append(Path(path))
        raise ValueError("invalid")

    monkeypatch.setattr(candidates, "load_exported_policy", fail)

    assert not candidates._learned_v1_available()
    with pytest.raises(ValueError, match="artifact is not valid"):
        candidates.candidate_metadata(candidates.LEARNED_V1)
    with pytest.raises(ValueError, match="artifact is not valid"):
        candidates.candidate_policy(candidates.LEARNED_V1)
    assert calls == [artifact]


def test_returned_candidate_propagates_runtime_failure_and_reuses_one_loaded_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "candidate.json"
    artifact.write_text("{}")
    calls = []

    class FailingPolicy:
        def act(self, observation):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(
        candidates, "load_exported_policy",
        lambda path: calls.append(path) or FailingPolicy(),
    )
    policy = candidates.artifact_candidate_policy(artifact)
    failures = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="inference failed") as error:
            policy({"turn": 1})
        failures.append(str(error.value))
    assert failures == ["inference failed", "inference failed"]
    assert len(calls) == 1
