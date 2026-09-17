import hashlib
import json
from pathlib import Path
import subprocess
import sys
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
    assert args.game_timeout == 30.0

    with pytest.raises(SystemExit):
        parse_args(["--artifact", str(tmp_path / "artifact.json"), "--seats", "0", "0"])


def test_parse_args_preserves_legacy_timeout_and_accepts_separate_deadline(tmp_path):
    from scripts.evaluate_artifact import parse_args

    args = parse_args([
        "--artifact", str(tmp_path / "artifact.json"),
        "--evaluation-timeout", "7",
        "--evaluation-deadline", "28",
    ])

    assert args.evaluation_timeout == 7.0
    assert args.game_timeout == 600.0
    assert args.evaluation_deadline == 28.0


@pytest.mark.parametrize("timeout", [600.0, 3600.0])
def test_legacy_cli_timeout_is_reported_as_total_deadline(tmp_path, timeout):
    from scripts import evaluate_artifact

    args = evaluate_artifact.parse_args([
        "--artifact", str(tmp_path / "artifact.json"),
        "--evaluation-timeout", str(timeout),
    ])
    configuration = evaluate_artifact._configuration_from_args(args)

    assert configuration["evaluation_timeout"] == timeout
    assert configuration["evaluation_deadline"] == timeout
    assert configuration["game_timeout"] == evaluate_artifact.DEFAULT_GAME_TIMEOUT


@pytest.mark.parametrize(
    "option, value",
    [
        ("--evaluation-timeout", "0"),
        ("--game-timeout", "nan"),
        ("--evaluation-deadline", "inf"),
    ],
)
def test_parse_args_rejects_invalid_supplied_timeouts(tmp_path, option, value):
    from scripts.evaluate_artifact import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--artifact", str(tmp_path / "artifact.json"), option, value])


@pytest.mark.parametrize(
    "argument, value, message",
    [
        ("evaluation_timeout", 0, "evaluation_timeout"),
        ("game_timeout", float("inf"), "game_timeout"),
        ("evaluation_deadline", True, "evaluation_deadline"),
    ],
)
def test_evaluate_rejects_invalid_supplied_timeouts(tmp_path, argument, value, message):
    from scripts.evaluate_artifact import evaluate

    with pytest.raises(ValueError, match=message):
        evaluate(artifact=tmp_path / "artifact.json", **{argument: value})


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


def test_evaluate_configuration_carries_action_representation(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())
    result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
        steps=4, workers=1, min_valid_games=1,
        game_runner=lambda request: _record(
            request["candidate"], request["opponent"], request["seed"], request["seat"],
        ),
        action_representation="target_first_v1",
    )

    assert result["configuration"]["action_representation"] == "target_first_v1"


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


def test_evaluate_keeps_legacy_total_timeout_and_separate_game_timeout(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())
    calls = []

    def fake_game(request):
        calls.append(dict(request))
        return _record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    legacy_result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
        steps=4, workers=1, min_valid_games=1, evaluation_timeout=7,
        game_runner=fake_game,
    )
    assert {request["game_timeout"] for request in calls} == {evaluate_artifact.DEFAULT_GAME_TIMEOUT}
    assert legacy_result["configuration"]["evaluation_timeout"] == 7.0
    assert legacy_result["configuration"]["game_timeout"] == evaluate_artifact.DEFAULT_GAME_TIMEOUT
    assert legacy_result["configuration"]["evaluation_deadline"] == 7.0

    calls.clear()
    override_result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
        steps=4, workers=1, min_valid_games=1, evaluation_timeout=7,
        game_timeout=11, evaluation_deadline=13, game_runner=fake_game,
    )
    assert {request["game_timeout"] for request in calls} == {11.0}
    assert override_result["configuration"]["evaluation_timeout"] == 7.0
    assert override_result["configuration"]["game_timeout"] == 11.0
    assert override_result["configuration"]["evaluation_deadline"] == 13.0


def test_build_report_preserves_timeout_diagnostics_in_records_and_decision(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    class RunningProcess:
        pid = 0
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(evaluate_artifact, "_start_game_process", lambda request: RunningProcess())
    result = evaluate_artifact.evaluate(
        artifact=artifact, seeds=[3], opponents=["pass"], seats=[0],
        steps=4, workers=1, evaluation_timeout=0.01, game_timeout=7, min_valid_games=1,
    )

    report = evaluate_artifact.build_report(result)
    record = report["records"]["current"][0]
    assert record["error_type"] == "evaluation_timeout"
    assert record["timeout_details"]["evaluation_deadline_seconds"] == 0.01
    assert report["configuration"]["evaluation_timeout"] == 0.01
    assert report["configuration"]["game_timeout"] == 7.0
    assert report["configuration"]["evaluation_deadline"] == 0.01
    assert report["decision"]["timeout_details"][0]["error_type"] == "evaluation_timeout"
    assert report["decision"]["timeout_details"][0]["timeout_seconds"] == 0.01


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


def test_evaluate_timeout_marks_running_and_pending_games(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())
    state = {"started": 0, "terminated": 0}

    class RunningProcess:
        pid = 0
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            state["terminated"] += 1
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    def start_process(request):
        state["started"] += 1
        return RunningProcess()

    monkeypatch.setattr(evaluate_artifact, "_start_game_process", start_process)
    started = time.monotonic()
    result = evaluate_artifact.evaluate(
        artifact=artifact,
        seeds=[3],
        opponents=["pass"],
        seats=[0, 1],
        steps=4,
        workers=1,
        evaluation_timeout=0.01,
        game_timeout=0.01,
        min_valid_games=1,
    )

    assert time.monotonic() - started < 1
    assert state == {"started": 1, "terminated": 1}
    assert result["records"][0]["framework_error"] is True
    assert "timeout" in result["records"][0]["error"]
    assert result["records"][0]["error_type"] == "evaluation_timeout"
    assert result["records"][0]["timeout_seconds"] == 0.01
    assert result["records"][0]["timeout_details"] == {
        "scope": "evaluation",
        "evaluation_deadline_seconds": 0.01,
        "game_timeout_seconds": 0.01,
        "request_count": 4,
        "workers": 1,
    }
    assert result["decision"]["status"] == "discard"
    assert "framework_error" in result["decision"]["reasons"]
    assert "evaluation_timeout" in result["decision"]["reasons"]
    assert result["decision"]["timeout_details"][0]["error_type"] == "evaluation_timeout"


def test_total_deadline_terminates_long_running_child_promptly(monkeypatch):
    from scripts import evaluate_artifact

    real_popen = subprocess.Popen
    children = []

    def long_running_popen(*args, **kwargs):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=kwargs.get("stdin"),
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
            text=kwargs.get("text", False),
            start_new_session=kwargs.get("start_new_session", False),
        )
        children.append(process)
        return process

    monkeypatch.setattr(evaluate_artifact.subprocess, "Popen", long_running_popen)
    request = {
        "candidate": "current",
        "opponent": "pass",
        "seed": 3,
        "seat": 0,
        "steps": 4,
        "artifact_path": "artifact.json",
        "game_timeout": 60.0,
    }

    started = time.monotonic()
    records = evaluate_artifact._run_in_pool(
        [request], workers=1, evaluation_timeout=0.05, game_timeout=60.0,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert children and children[0].poll() is not None
    assert records[0]["error_type"] == "evaluation_timeout"


def test_successful_worker_result_is_collected_after_stdin_is_closed(monkeypatch):
    from scripts import evaluate_artifact

    class FakeStdin:
        closed = False

        def write(self, value):
            return len(value)

        def close(self):
            self.closed = True

    class SuccessfulProcess:
        pid = 0
        returncode = 0

        def __init__(self):
            self.stdin = FakeStdin()

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            if self.stdin is not None and self.stdin.closed:
                raise ValueError("I/O operation on closed file")
            return json.dumps({"framework_error": False, "outcome": "win"}), ""

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    process = SuccessfulProcess()
    monkeypatch.setattr(evaluate_artifact.subprocess, "Popen", lambda *args, **kwargs: process)
    request = {
        "candidate": "current",
        "opponent": "pass",
        "seed": 3,
        "seat": 0,
        "steps": 4,
        "artifact_path": "artifact.json",
    }

    records = evaluate_artifact._run_in_pool(
        [request], workers=1, evaluation_timeout=1.0, game_timeout=1.0,
    )

    assert records == [{
        "framework_error": False,
        "outcome": "win",
        "candidate": "current",
        "variant": "current",
    }]


def test_completed_worker_collection_respects_remaining_evaluation_deadline(monkeypatch):
    from scripts import evaluate_artifact

    class FakeStdin:
        closed = False

        def write(self, value):
            return len(value)

        def close(self):
            self.closed = True

    class PipeStuckProcess:
        pid = 0
        returncode = 0

        def __init__(self):
            self.stdin = FakeStdin()
            self.communicate_timeout = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            self.communicate_timeout = timeout
            if timeout is None:
                raise AssertionError("output collection must be bounded")
            raise subprocess.TimeoutExpired("worker", timeout)

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    process = PipeStuckProcess()
    monkeypatch.setattr(evaluate_artifact.subprocess, "Popen", lambda *args, **kwargs: process)
    request = {
        "candidate": "current",
        "opponent": "pass",
        "seed": 3,
        "seat": 0,
        "steps": 4,
        "artifact_path": "artifact.json",
    }

    records = evaluate_artifact._run_in_pool(
        [request], workers=1, evaluation_timeout=0.05, game_timeout=1.0,
    )

    assert process.communicate_timeout is not None
    assert process.communicate_timeout <= 0.05
    assert process.returncode == -15
    assert records[0]["error_type"] == "evaluation_timeout"
    assert records[0]["timeout_details"]["scope"] == "evaluation"


def test_game_deadline_classifies_long_running_child_as_game_timeout(monkeypatch):
    from scripts import evaluate_artifact

    real_popen = subprocess.Popen

    def long_running_popen(*args, **kwargs):
        return real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=kwargs.get("stdin"),
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
            text=kwargs.get("text", False),
            start_new_session=kwargs.get("start_new_session", False),
        )

    monkeypatch.setattr(evaluate_artifact.subprocess, "Popen", long_running_popen)
    request = {
        "candidate": "current",
        "opponent": "pass",
        "seed": 3,
        "seat": 0,
        "steps": 4,
        "artifact_path": "artifact.json",
    }

    records = evaluate_artifact._run_in_pool(
        [request], workers=1, evaluation_timeout=1.0, game_timeout=0.05,
    )

    assert records[0]["error_type"] == "game_timeout"
    assert records[0]["timeout_seconds"] == 0.05
    assert records[0]["timeout_details"] == {
        "scope": "game", "game_timeout_seconds": 0.05,
    }


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
    assert record["error_type"] == "game_timeout"
    assert record["timeout_seconds"] == 0.01


def test_evaluate_marks_pool_submission_failure_as_framework_error(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    monkeypatch.setattr(evaluate_artifact, "load_exported_policy", lambda path: object())

    def failing_start(request):
        raise RuntimeError("submit failed")

    monkeypatch.setattr(evaluate_artifact, "_start_game_process", failing_start)
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


def test_write_report_rejects_protected_model_output(tmp_path):
    from scripts.evaluate_artifact import write_report

    destination = tmp_path / "models" / "learned_v1.json"
    destination.parent.mkdir()

    with pytest.raises(ValueError, match="production|artifact"):
        write_report(destination, {"status": "discard"})

    assert not destination.exists()


def test_write_report_rejects_symlinked_parent(tmp_path):
    from scripts.evaluate_artifact import write_report

    real_parent = tmp_path / "real-reports"
    real_parent.mkdir()
    linked_parent = tmp_path / "reports"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_report(linked_parent / "evaluation.json", {"status": "discard"})

    assert not (real_parent / "evaluation.json").exists()


def test_cli_forwards_action_representation_to_evaluate(tmp_path, monkeypatch):
    from scripts import evaluate_artifact

    artifact = _artifact(tmp_path / "artifact.json")
    output = tmp_path / "report.json"
    captured = {}
    result = {
        "configuration": {"action_representation": "target_first_v1"},
        "artifact": {
            "name": artifact.name, "identity": "learned_artifact", "sha256": "a" * 64,
        },
        "expected_matrix": [],
        "records": [],
        "summaries": {},
        "matrix_completeness": {},
        "decision": {"status": "discard", "reasons": []},
    }

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(evaluate_artifact, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluate_artifact, "_is_valid_comparison", lambda value: True)

    assert evaluate_artifact.main([
        "--artifact", str(artifact), "--output", str(output),
        "--action-representation", "target_first_v1",
    ]) == 0
    assert captured["action_representation"] == "target_first_v1"
