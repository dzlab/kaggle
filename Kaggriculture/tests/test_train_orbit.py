import json
from pathlib import Path

import pytest


def test_orbit_controller_retries_after_rejection_and_retains_only_winner(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    calls = []
    candidates = []

    def rollout_fn(**kwargs):
        calls.append(("rollout", kwargs["round_index"]))
        return [kwargs["round_index"]]

    def train_fn(**kwargs):
        candidate = tmp_path / f"candidate-{kwargs['round_index']}.pt"
        candidate.write_text(str(kwargs["rollout"]))
        candidates.append(candidate)
        return candidate

    def evaluate_fn(**kwargs):
        calls.append(("evaluate", kwargs["round_index"]))
        return {"promoted": kwargs["round_index"] == 1}

    result = OrbitController(
        OrbitConfig(tmp_path, max_rounds=3), rollout_fn=rollout_fn,
        train_fn=train_fn, evaluate_fn=evaluate_fn,
    ).run()

    assert result["round"] == 3
    assert result["best"] == str(tmp_path / "best.pt")
    assert (tmp_path / "best.pt").read_text() == "[1]"
    assert [entry["stage"] for entry in result["history"]] == ["rejected", "retained", "rejected"]


def test_orbit_controller_resumes_completed_round_state(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    calls = []
    (tmp_path / "orbit-state.json").write_text(json.dumps({
        "round": 1, "failures": 0, "best": None,
        "history": [{"round": 0, "stage": "rejected"}],
    }))

    controller = OrbitController(
        OrbitConfig(tmp_path, max_rounds=2),
        rollout_fn=lambda **kwargs: calls.append(("rollout", kwargs["round_index"])) or [],
        train_fn=lambda **kwargs: tmp_path / "unused.pt",
        evaluate_fn=lambda **kwargs: {"promoted": False},
    )
    controller.run()
    assert calls == [("rollout", 1)]


def test_orbit_controller_evaluates_exported_artifact_but_retains_checkpoint(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    checkpoint = tmp_path / "round-0000" / "candidate.pt"
    artifact = tmp_path / "round-0000" / "candidate.json"
    calls = {}

    def train_fn(**_kwargs):
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint-bytes")
        return checkpoint

    def export_fn(**kwargs):
        calls["export_candidate"] = kwargs["candidate"]
        artifact.write_text("exported-artifact", encoding="utf-8")
        return artifact

    def evaluate_fn(**kwargs):
        calls["evaluated_candidate"] = kwargs["candidate"]
        return {"promoted": True}

    result = OrbitController(
        OrbitConfig(tmp_path, max_rounds=1),
        rollout_fn=lambda **_kwargs: {"complete": True},
        train_fn=train_fn,
        export_fn=export_fn,
        evaluate_fn=evaluate_fn,
    ).run()

    assert calls["export_candidate"] == checkpoint
    assert calls["evaluated_candidate"] == artifact
    assert Path(result["best"]).read_bytes() == b"checkpoint-bytes"
    assert result["history"][0]["candidate"] == str(checkpoint)
    assert result["history"][0]["candidate_artifact"] == str(artifact)


def test_orbit_controller_passes_partial_checkpoint_for_train_resume(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    checkpoint = tmp_path / "round-0000" / "candidate.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"partial-checkpoint")
    (tmp_path / "orbit-state.json").write_text(json.dumps({
        "round": 0, "failures": 0, "best": None, "history": [],
        "current": {
            "round": 0, "stage": "train", "rollout": {"complete": True},
            "candidate": str(checkpoint),
        },
    }))
    calls = {}

    def train_fn(**kwargs):
        calls.update(kwargs)
        checkpoint.write_bytes(b"resumed-checkpoint")
        return checkpoint

    OrbitController(
        OrbitConfig(tmp_path, max_rounds=1),
        rollout_fn=lambda **_kwargs: pytest.fail("completed rollout must be reused"),
        train_fn=train_fn,
        evaluate_fn=lambda **_kwargs: {"promoted": False},
    ).run()

    assert calls["resume_checkpoint"] == checkpoint


def test_orbit_controller_resumes_interrupted_train_from_reserved_candidate(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    candidate = tmp_path / "round-0000" / "candidate.pt"
    train_calls = []

    def train_fn(**kwargs):
        train_calls.append(kwargs)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"partial-checkpoint")
        if len(train_calls) == 1:
            raise KeyboardInterrupt
        return candidate

    controller_kwargs = {
        "rollout_fn": lambda **_kwargs: {"complete": True},
        "train_fn": train_fn,
        "evaluate_fn": lambda **_kwargs: {"promoted": False},
        "candidate_path_fn": lambda **_kwargs: candidate,
    }

    with pytest.raises(KeyboardInterrupt):
        OrbitController(
            OrbitConfig(tmp_path, max_rounds=1), **controller_kwargs
        ).run()

    saved_state = json.loads((tmp_path / "orbit-state.json").read_text())
    assert saved_state["current"] == {
        "round": 0,
        "stage": "train",
        "rollout": {"complete": True},
        "candidate": str(candidate),
    }
    assert "resume_checkpoint" not in train_calls[0]

    OrbitController(OrbitConfig(tmp_path, max_rounds=1), **controller_kwargs).run()

    assert train_calls[1]["resume_checkpoint"] == candidate


def test_orbit_cli_parses_production_configuration_and_resume(tmp_path):
    from scripts.train_orbit import config_from_args, parse_args

    args = parse_args([
        "--run-directory", str(tmp_path),
        "--rollout-seeds", "7", "8",
        "--opponents", "pass", "random",
        "--seats", "0", "1",
        "--workers", "3",
        "--episode-steps", "12",
        "--ppo-rounds", "2",
        "--checkpoint-window", "4",
        "--max-rounds", "6",
        "--max-hours", "1.5",
        "--max-failures", "2",
        "--device", "cpu",
        "--resume",
        "--dry-run",
    ])

    config = config_from_args(args)

    assert config.development_seeds == (7, 8)
    assert config.opponents == ("pass", "random")
    assert config.seats == (0, 1)
    assert config.workers == 3
    assert config.episode_steps == 12
    assert config.ppo_rounds == 2
    assert config.checkpoint_window == 4
    assert config.max_rounds == 6
    assert config.max_hours == 1.5
    assert config.max_failures == 2
    assert args.resume is True
    assert args.dry_run is True


def test_orbit_cli_dry_run_only_validates_configuration(tmp_path, capsys):
    from scripts.train_orbit import main

    assert main(["--run-directory", str(tmp_path), "--dry-run"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["run_directory"] == str(tmp_path)
    assert not (tmp_path / "orbit-state.json").exists()


def test_orbit_cli_rejects_invalid_rollout_matrix(tmp_path):
    from scripts.train_orbit import parse_args

    with pytest.raises(SystemExit):
        parse_args([
            "--run-directory", str(tmp_path),
            "--rollout-seeds", "3", "3",
            "--opponents", "pass", "pass",
            "--episode-steps", "1",
        ])


def test_production_evaluator_passes_only_development_matrix(monkeypatch, tmp_path):
    import scripts.evaluate_artifact as evaluate_artifact
    from scripts.train_orbit import OrbitConfig, build_production_callbacks

    calls = {}

    def fake_evaluate(**kwargs):
        calls.update(kwargs)
        return {
            "artifact": {"identity": "candidate", "name": "candidate.json", "sha256": "0" * 64},
            "records": [],
            "decision": {"status": "promote", "reasons": []},
        }

    monkeypatch.setattr(evaluate_artifact, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluate_artifact, "build_report", lambda result: result)
    monkeypatch.setattr(evaluate_artifact, "write_report", lambda path, report: path)

    artifact = tmp_path / "candidate.json"
    artifact.write_text("artifact")
    evaluate_fn = build_production_callbacks(
        OrbitConfig(tmp_path, development_seeds=(11, 13), opponents=("pass",), workers=1)
    )["evaluate"]

    result = evaluate_fn(
        candidate=artifact, current=None, seeds=(11, 13), round_index=2,
    )

    assert result["promoted"] is True
    assert calls["seeds"] == [11, 13]
    assert calls["min_valid_games"] == len((11, 13)) * len(("pass",))
    assert "holdout_seeds" not in calls


def test_production_rollout_adapter_uses_bounded_collector(monkeypatch, tmp_path):
    import scripts.collect_trajectories as collector
    from scripts.train_orbit import OrbitConfig, build_production_callbacks

    calls = {}

    def fake_collect(**kwargs):
        calls.update(kwargs)
        Path(kwargs["output"]).write_text("transition\n")
        return {"run_id": "round-0"}

    monkeypatch.setattr(collector, "collect", fake_collect)
    rollout_fn = build_production_callbacks(
        OrbitConfig(tmp_path, development_seeds=(3, 5), opponents=("pass",), workers=2)
    )["rollout"]

    result = rollout_fn(
        round_index=0, seeds=(3, 5), run_directory=tmp_path,
    )

    assert result["input_path"].endswith("round-0000/rollout.jsonl")
    assert calls["seeds"] == [3, 5]
    assert calls["opponents"] == ["pass"]
    assert calls["seats"] == [0, 1]
    assert calls["workers"] == 2
    assert calls["candidate_artifact"] is None


def test_production_train_adapter_refreshes_artifact_between_ppo_rounds(monkeypatch, tmp_path):
    import scripts.export_policy as exporter
    import scripts.train_policy as trainer
    from scripts.train_orbit import OrbitConfig, build_production_callbacks

    input_path = tmp_path / "rollout.jsonl"
    input_path.write_text("transition\n")
    train_calls = []
    export_calls = []

    def fake_train(**kwargs):
        train_calls.append(kwargs)
        kwargs["output_path"].write_bytes(b"checkpoint")
        return {}

    def fake_export(checkpoint, artifact):
        export_calls.append((checkpoint, artifact))
        artifact.write_text("artifact")
        return {}

    class FakePool:
        def __init__(self, checkpoints):
            self.checkpoints = tuple(checkpoints)

    monkeypatch.setattr(trainer, "train_behavior_clone", fake_train)
    monkeypatch.setattr(trainer, "make_fresh_rollout_fn", lambda **kwargs: kwargs)
    monkeypatch.setattr(trainer, "OpponentPool", FakePool)
    monkeypatch.setattr(exporter, "export_checkpoint", fake_export)

    config = OrbitConfig(tmp_path, ppo_rounds=2, opponents=("pass",), workers=1)
    train_fn = build_production_callbacks(config)["train"]
    candidate = train_fn(
        rollout={"input_path": str(input_path)}, round_index=0,
        run_directory=tmp_path,
    )

    assert candidate == tmp_path / "round-0000/candidate.pt"
    assert [call["ppo_steps"] for call in train_calls] == [0, 1, 2]
    assert all(call["rollout_fn"] for call in train_calls[1:])
    assert len(export_calls) == 3


def test_production_train_adapter_validates_and_resumes_partial_candidate(
    monkeypatch, tmp_path,
):
    import kagriculture_agent.checkpoints as checkpoints
    import scripts.export_policy as exporter
    import scripts.train_policy as trainer
    from scripts.train_orbit import OrbitConfig, build_production_callbacks

    input_path = tmp_path / "rollout.jsonl"
    input_path.write_text("transition\n")
    candidate = tmp_path / "round-0000" / "candidate.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"partial-checkpoint")
    train_calls = []
    validation_calls = []

    def fake_train(**kwargs):
        train_calls.append(kwargs)
        kwargs["output_path"].write_bytes(b"resumed-checkpoint")
        return {}

    def fake_export(checkpoint, artifact):
        artifact.write_text(f"exported:{checkpoint}")
        return {}

    class FakePool:
        def __init__(self, _checkpoints):
            pass

    monkeypatch.setattr(
        checkpoints, "read_checkpoint",
        lambda _path, *, map_location: {
            "configuration": {"ppo_steps": 1},
            "progress": {"round": 1},
        },
    )
    monkeypatch.setattr(trainer, "build_training_contract", lambda **_kwargs: object())
    monkeypatch.setattr(
        trainer, "validate_training_checkpoint",
        lambda payload, *, contract, allow_ppo_extension: validation_calls.append(
            (payload, contract, allow_ppo_extension)
        ),
    )
    monkeypatch.setattr(trainer, "train_behavior_clone", fake_train)
    monkeypatch.setattr(trainer, "make_fresh_rollout_fn", lambda **kwargs: kwargs)
    monkeypatch.setattr(trainer, "OpponentPool", FakePool)
    monkeypatch.setattr(exporter, "export_checkpoint", fake_export)

    config = OrbitConfig(tmp_path, ppo_rounds=2, opponents=("pass",), workers=1)
    train_fn = build_production_callbacks(config)["train"]
    result = train_fn(
        rollout={"input_path": str(input_path)}, round_index=0,
        run_directory=tmp_path, resume_checkpoint=candidate,
    )

    assert result == candidate
    assert len(validation_calls) == 1
    assert validation_calls[0][2] is True
    assert [call["ppo_steps"] for call in train_calls] == [2]
    assert train_calls[0]["resume_checkpoint"] == candidate
