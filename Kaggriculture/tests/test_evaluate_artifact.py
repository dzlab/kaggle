import hashlib
import json
from concurrent.futures import Future
from pathlib import Path
import subprocess
import time

import pytest


def _artifact(path, contents=b"artifact"):
    path.write_bytes(contents)
    return path


def _record(candidate, opponent, seed, seat, *, bank_differential=0):
    return {
        "candidate": candidate,
        "variant": candidate,
        "opponent": opponent,
        "seed": seed,
        "seat": seat,
        "outcome": "win" if bank_differential > 0 else "loss" if bank_differential < 0 else "tie",
        "final_bank": 100 + bank_differential,
        "opponent_final_bank": 100,
        "bank_differential": bank_differential,
        "framework_error": False,
        "framework_error_reasons": [],
        "missed_basic_needs": 0,
        "same_item_market_churn": 0,
        "submitted_market_order_count": 0,
        "terminal_cash": 100 + bank_differential,
        "terminal_inventory_value": 0,
    }


def test_build_matrix_is_ordered_and_has_both_default_seats():
    from scripts.evaluate_artifact import build_matrix

    assert build_matrix(opponents=["starter", "pass"], seeds=[7, 8], seats=[0, 1]) == [
        {"opponent": "starter", "seed": 7, "seat": 0},
        {"opponent": "starter", "seed": 7, "seat": 1},
        {"opponent": "starter", "seed": 8, "seat": 0},
        {"opponent": "starter", "seed": 8, "seat": 1},
        {"opponent": "pass", "seed": 7, "seat": 0},
        {"opponent": "pass", "seed": 7, "seat": 1},
        {"opponent": "pass", "seed": 8, "seat": 0},
        {"opponent": "pass", "seed": 8, "seat": 1},
    ]


def test_validate_seats_requires_unique_zero_or_one_values():
    from scripts.evaluate_artifact import validate_seats

    assert validate_seats([0, 1]) == [0, 1]
    with pytest.raises(ValueError, match="unique"):
        validate_seats([0, 0])
    with pytest.raises(ValueError, match="0 or 1"):
        validate_seats([2])


def test_parse_args_preserves_both_seats_and_quick_defaults(tmp_path):
    from scripts.evaluate_artifact import parse_args

    args = parse_args(["--artifact", str(tmp_path / "artifact.json"), "--quick"])

    assert args.seats == [0, 1]
    assert args.seeds == 2
    assert args.steps == 96
    assert args.evaluation_timeout == 30.0

    with pytest.raises(SystemExit):
        parse_args(["--artifact", str(tmp_path / "artifact.json"), "--seats", "0", "0"])


def test_parse_args_accepts_and_validates_training_identity(tmp_path):
    from scripts.evaluate_artifact import parse_args

    args = parse_args([
        "--artifact", str(tmp_path / "artifact.json"),
        "--experiment-id", "orbit-context-test",
        "--feature-variant", "experimental_context_v1",
        "--training-mode", "reduced_behavior_clone_then_ppo",
    ])

    assert (args.experiment_id, args.feature_variant, args.training_mode) == (
        "orbit-context-test", "experimental_context_v1", "reduced_behavior_clone_then_ppo",
    )
    with pytest.raises(SystemExit):
        parse_args([
            "--artifact", str(tmp_path / "artifact.json"),
            "--feature-variant", "unknown",
        ])


def test_evaluate_configuration_contains_training_identity(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    def fake_game(request):
        return _record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
        steps=4, workers=1, min_valid_games=1, game_runner=fake_game,
        experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    assert {
        key: result["configuration"][key]
        for key in ("experiment_id", "feature_variant", "training_mode")
    } == {
        "experiment_id": "orbit-context-test",
        "feature_variant": "experimental_context_v1",
        "training_mode": "reduced_behavior_clone_then_ppo",
    }


def test_validate_artifact_returns_identity_and_sha256(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "ppo16.json", b"ppo16")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    result = evaluate_artifact.validate_artifact(artifact, "ppo16")

    assert result["identity"] == "ppo16"
    assert result["name"] == "ppo16.json"
    assert result["sha256"] == hashlib.sha256(b"ppo16").hexdigest()


def test_evaluate_runs_current_and_artifact_on_identical_matrix(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())
    calls = []

    def fake_game(request):
        calls.append(dict(request))
        return _record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate_artifact.evaluate(
        artifact=artifact,
        identity="learned_artifact",
        seeds=[3],
        opponents=["pass"],
        seats=[0, 1],
        steps=4,
        workers=1,
        min_valid_games=1,
        game_runner=fake_game,
    )

    assert result["expected_matrix"] == [["pass", 3, 0], ["pass", 3, 1]]
    assert [(item["candidate"], item["opponent"], item["seed"], item["seat"])
            for item in result["records"]] == [
        ("current", "pass", 3, 0),
        ("current", "pass", 3, 1),
        ("learned_artifact", "pass", 3, 0),
        ("learned_artifact", "pass", 3, 1),
    ]
    coordinates = lambda items: [
        (item["opponent"], item["seed"], item["seat"]) for item in items
    ]
    assert coordinates(calls[:2]) == coordinates(calls[2:])


def test_degraded_candidate_is_discarded_by_existing_gates(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    def fake_game(request):
        differential = 10 if request["candidate"] == "current" else -10
        return _record(
            request["candidate"], request["opponent"], request["seed"], request["seat"],
            bank_differential=differential,
        )

    result = evaluate_artifact.evaluate(
        artifact=artifact,
        seeds=[3],
        opponents=["pass"],
        seats=[0, 1],
        steps=4,
        workers=1,
        min_valid_games=1,
        game_runner=fake_game,
    )

    assert result["decision"]["status"] == "discard"
    assert "negative_tail" in result["decision"]["reasons"]


def test_incomplete_matrix_fails_closed(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    def fake_game(request):
        if request["candidate"] == "learned_artifact" and request["seat"] == 1:
            return _record(request["candidate"], request["opponent"], request["seed"], 0)
        return _record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate_artifact.evaluate(
        artifact=artifact,
        seeds=[3],
        opponents=["pass"],
        seats=[0, 1],
        steps=4,
        workers=1,
        min_valid_games=1,
        game_runner=fake_game,
    )

    assert result["decision"]["status"] == "discard"
    assert "incomplete_matrix" in result["decision"]["reasons"]


def test_report_contains_configuration_artifact_records_summaries_and_decision(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    def fake_game(request):
        return _record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate_artifact.evaluate(
        artifact=artifact,
        identity="ppo16",
        seeds=[3],
        opponents=["pass"],
        seats=[0, 1],
        steps=4,
        workers=1,
        min_valid_games=1,
        game_runner=fake_game,
    )
    report = evaluate_artifact.build_report(result)

    assert report["configuration"]["steps"] == 4
    assert report["artifact"]["identity"] == "ppo16"
    assert set(report["records"]) == {"current", "ppo16"}
    assert set(report["summaries"]) == {"current", "ppo16"}
    assert report["decision"]["status"] == "discard"


def test_evaluate_timeout_marks_pending_futures_and_does_not_wait_for_pool(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())
    state = {}

    class FakeExecutor:
        def __init__(self, *, max_workers):
            state["max_workers"] = max_workers
            state["futures"] = []

        def submit(self, function, request):
            future = Future()
            state["futures"].append(future)
            if len(state["futures"]) > 1:
                future.set_result(_record(
                    request["candidate"], request["opponent"], request["seed"], request["seat"],
                ))
            return future

        def shutdown(self, *, wait, cancel_futures):
            state["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(evaluate_artifact, "ProcessPoolExecutor", FakeExecutor)
    started = time.monotonic()
    result = evaluate_artifact.evaluate(
        artifact=artifact,
        seeds=[3],
        opponents=["pass"],
        seats=[0, 1],
        steps=4,
        workers=1,
        evaluation_timeout=0.01,
        min_valid_games=1,
    )

    assert time.monotonic() - started < 1
    assert state["max_workers"] == 1
    assert state["shutdown"] == (False, True)
    assert result["records"][0]["framework_error"] is True
    assert "timeout" in result["records"][0]["error"]
    assert result["decision"]["status"] == "discard"
    assert "framework_error" in result["decision"]["reasons"]


def test_game_worker_bounds_each_subprocess_game(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    request = {
        "candidate": "current",
        "opponent": "pass",
        "seed": 3,
        "seat": 0,
        "steps": 4,
        "artifact_path": str(tmp_path / "artifact.json"),
        "game_timeout": 0.01,
    }
    observed = {}

    def fake_run(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(evaluate_artifact.subprocess, "run", fake_run)

    record = evaluate_artifact._run_game(request)

    assert observed["timeout"] == 0.01
    assert record["framework_error"] is True
    assert "timeout" in record["error"]


def test_evaluate_marks_pool_submission_failure_as_framework_error(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    class FailingExecutor:
        def __init__(self, *, max_workers):
            pass

        def submit(self, function, request):
            raise RuntimeError("submit failed")

        def shutdown(self, *, wait, cancel_futures):
            assert wait is False
            assert cancel_futures is True

    monkeypatch.setattr(evaluate_artifact, "ProcessPoolExecutor", FailingExecutor)
    result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
        steps=4, workers=1, evaluation_timeout=1, min_valid_games=1,
    )

    assert len(result["records"]) == 4
    assert all(record["framework_error"] for record in result["records"])
    assert result["decision"]["status"] == "discard"
    assert "framework_error" in result["decision"]["reasons"]


def test_evaluate_uses_one_read_only_snapshot_and_cleans_it_afterward(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json", b"snapshot-bytes")
    loads = []
    monkeypatch.setattr(
        evaluate_artifact, "load_exported_policy",
        lambda path: loads.append(Path(path)) or object(),
    )
    seen = []

    def fake_game(request):
        snapshot = Path(request["artifact_path"])
        seen.append(snapshot)
        assert snapshot.exists()
        assert snapshot.read_bytes() == b"snapshot-bytes"
        assert snapshot.stat().st_mode & 0o222 == 0
        return _record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
        steps=4, workers=1, min_valid_games=1, game_runner=fake_game,
    )

    assert len(loads) == 1
    assert len({str(path) for path in seen}) == 1
    assert result["artifact"]["sha256"] == hashlib.sha256(b"snapshot-bytes").hexdigest()
    assert all(not path.exists() for path in seen)


def test_cli_returns_zero_for_complete_policy_discard(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    output = tmp_path / "discard.json"
    expected_matrix = [["pass", 3, 0]]
    records = [
        _record("current", "pass", 3, 0),
        _record("learned_artifact", "pass", 3, 0),
    ]
    completeness = {
        "expected": expected_matrix,
        "expected_count": 1,
        "observed_count": 1,
        "missing": [],
        "duplicate": [],
        "extra": [],
        "invalid_records": 0,
    }
    result = {
        "configuration": {"steps": 4},
        "artifact": {
            "path": str(artifact), "name": artifact.name,
            "identity": "learned_artifact", "sha256": "a" * 64,
        },
        "expected_matrix": expected_matrix,
        "records": records,
        "summaries": {},
        "matrix_completeness": {
            "current": completeness,
            "learned_artifact": completeness,
        },
        "decision": {"status": "discard", "reasons": ["no_paired_improvement"]},
    }
    monkeypatch.setattr(evaluate_artifact, "evaluate", lambda **kwargs: result)

    exit_code = evaluate_artifact.main([
        "--artifact", str(artifact), "--output", str(output), "--quick",
    ])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["decision"]["status"] == "discard"


def test_cli_returns_nonzero_for_incomplete_comparison(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    output = tmp_path / "incomplete.json"
    result = {
        "configuration": {"steps": 4},
        "artifact": {
            "path": str(artifact), "name": artifact.name,
            "identity": "learned_artifact", "sha256": "a" * 64,
        },
        "expected_matrix": [["pass", 3, 0]],
        "records": [],
        "summaries": {},
        "matrix_completeness": {
            "current": {"missing": [["pass", 3, 0]], "duplicate": [], "extra": [], "invalid_records": 0},
            "learned_artifact": {"missing": [["pass", 3, 0]], "duplicate": [], "extra": [], "invalid_records": 0},
        },
        "decision": {"status": "discard", "reasons": ["incomplete_matrix"]},
    }
    monkeypatch.setattr(evaluate_artifact, "evaluate", lambda **kwargs: result)

    exit_code = evaluate_artifact.main([
        "--artifact", str(artifact), "--output", str(output), "--quick",
    ])

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["decision"]["status"] == "discard"


def test_cli_writes_failure_report_for_invalid_arguments(tmp_path):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    output = tmp_path / "invalid-args.json"

    exit_code = evaluate_artifact.main([
        "--artifact", str(artifact), "--output", str(output), "--seats", "0", "0",
    ])

    assert exit_code != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"]["status"] == "discard"
    assert report["decision"]["reasons"] == ["cli_invalid"]


def test_cli_writes_discard_report_for_invalid_artifact(tmp_path):
    from scripts import evaluate_artifact

    output = tmp_path / "invalid.json"
    exit_code = evaluate_artifact.main([
        "--artifact", str(tmp_path / "missing.json"), "--output", str(output),
    ])

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"]["status"] == "discard"
    assert report["decision"]["reasons"] == ["artifact_invalid"]
    assert set(report["records"]) == {"current", "learned_artifact"}
