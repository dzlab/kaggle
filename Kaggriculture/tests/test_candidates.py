from pathlib import Path

import pytest

from kagriculture_agent import candidates
from kagriculture_agent.learned_policy import DependencyFreePolicy, PolicyProposal


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


@pytest.mark.parametrize(
    ("elapsed_seconds", "expected_status", "expected_enabled"),
    [(0.011, "slow_model", False), (0.009, "ok", True)],
)
def test_production_candidate_enforces_shared_inference_budget(
    monkeypatch, tmp_path, elapsed_seconds, expected_status, expected_enabled,
):
    import kagriculture_agent.learned_policy as runtime

    artifact = tmp_path / "learned-v1.json"
    artifact.write_text("{}")
    model = object.__new__(DependencyFreePolicy)
    monkeypatch.setattr(candidates, "learned_v1_artifact_path", lambda: artifact)
    monkeypatch.setattr(candidates, "load_exported_policy", lambda _path: model)
    monkeypatch.setattr(
        DependencyFreePolicy,
        "propose",
        lambda self, state, features: PolicyProposal((), (), 1.0, "learned_v1"),
    )
    ticks = iter((100.0, 100.0 + elapsed_seconds))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))

    candidate = candidates.candidate_policy(candidates.LEARNED_V1)
    learned = candidate.__self__.learned_policy
    proposal = learned.propose({"day": 0, "hour": 0}, object())

    assert learned.timeout_seconds == pytest.approx(0.01)
    assert learned.diagnostics["status"] == expected_status
    assert learned.diagnostics.get("learned_overrides_enabled", True) is expected_enabled
    assert proposal.model_version == ("learned_v1" if expected_enabled else "none")


def test_production_candidate_passes_shared_budget_to_policy_factory(
    monkeypatch, tmp_path,
):
    artifact = tmp_path / "learned-v1.json"
    artifact.write_text("{}")
    model = object()
    calls = []

    class StubPolicy:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def act(self, observation):
            return {"observation": observation}

    monkeypatch.setenv(candidates.LEARNED_V1_ARTIFACT_ENV, str(artifact))
    monkeypatch.setattr(candidates, "learned_v1_artifact_path", lambda: artifact)
    monkeypatch.setattr(candidates, "load_exported_policy", lambda _path: model)
    monkeypatch.setattr(candidates, "Policy", StubPolicy)

    candidates.candidate_policy(candidates.LEARNED_V1)

    assert calls == [{
        "strategy": "current",
        "learned_model": model,
        "learned_timeout_seconds": 0.01,
    }]
