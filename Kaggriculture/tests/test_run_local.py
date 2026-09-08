import types
from pathlib import Path

import pytest

from kagriculture_agent.constants import ENGINE_VERSION
from scripts.run_local import _validate_engine_version


def test_validate_engine_version_accepts_pinned_engine():
    _validate_engine_version(types.SimpleNamespace(__version__=ENGINE_VERSION))


@pytest.mark.parametrize("version", [None, "1.32.6", "2.0.0"])
def test_validate_engine_version_rejects_unpinned_engine(version):
    module = types.SimpleNamespace()
    if version is not None:
        module.__version__ = version

    with pytest.raises(RuntimeError, match=f"requires kaggle-environments=={ENGINE_VERSION}"):
        _validate_engine_version(module)


def test_run_episode_rejects_mismatched_engine_before_environment_creation(monkeypatch, tmp_path):
    fake_engine = types.SimpleNamespace(
        __version__="1.32.6",
        make=lambda *args, **kwargs: pytest.fail("make() must not run for a mismatched engine"),
    )
    monkeypatch.setitem(__import__("sys").modules, "kaggle_environments", fake_engine)

    from scripts.run_local import run_episode

    with pytest.raises(RuntimeError, match=f"requires kaggle-environments=={ENGINE_VERSION}"):
        run_episode(opponent="pass", seed=0, steps=1, replay_path=tmp_path / "replay.json")


def test_run_episode_validates_artifact_before_importing_engine(monkeypatch, tmp_path):
    import scripts.run_local as run_local

    events = []
    monkeypatch.setattr(
        run_local,
        "artifact_candidate_policy",
        lambda path: events.append(("artifact", Path(path))) or (lambda obs: {}),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "kaggle_environments",
        types.SimpleNamespace(
            __version__=ENGINE_VERSION,
            make=lambda *args, **kwargs: events.append(("make", args)) or pytest.fail("make called"),
        ),
    )

    with pytest.raises(pytest.fail.Exception):
        run_local.run_episode(
            opponent="pass", seed=0, steps=1, replay_path=tmp_path / "replay.json",
            candidate_artifact=tmp_path / "candidate.json",
        )
    assert events[0] == ("artifact", tmp_path / "candidate.json")


def test_run_episode_uses_candidate_and_preserves_both_seat_orders(monkeypatch, tmp_path):
    import scripts.run_local as run_local

    candidate = object()
    calls = []

    class Env:
        configuration = object()

        def run(self, agents):
            calls.append(agents)

        def toJSON(self):
            return {"rewards": [1, 0], "agents": []}

    monkeypatch.setattr(run_local, "candidate_policy", lambda identity: calls.append(identity) or candidate)
    monkeypatch.setitem(
        __import__("sys").modules,
        "kaggle_environments",
        types.SimpleNamespace(__version__=ENGINE_VERSION, make=lambda *args, **kwargs: Env()),
    )

    run_local.run_episode(
        opponent="pass", seed=0, steps=1, replay_path=tmp_path / "zero.json",
        candidate_identity="melon", candidate_player=0,
    )
    run_local.run_episode(
        opponent="pass", seed=0, steps=1, replay_path=tmp_path / "one.json",
        candidate_identity="melon", candidate_player=1,
    )

    assert calls == ["melon", [candidate, "pass"], "melon", ["pass", candidate]]


def test_parser_exposes_candidate_artifact_and_identity():
    from scripts.run_local import _parser

    args = _parser().parse_args(["--candidate-artifact", "model.json", "--candidate-identity", "orbit"])

    assert args.candidate_artifact == Path("model.json")
    assert args.candidate_identity == "orbit"
