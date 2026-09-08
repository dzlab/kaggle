from pathlib import Path

import pytest

from kagriculture_agent import candidates


def test_artifact_candidate_policy_loads_once_and_returns_callable(monkeypatch, tmp_path):
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
