import json
from pathlib import Path

import pytest

from scripts.telemetry import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_WEAVE_PROJECT,
    TrainingTelemetry,
    load_metrics,
    record_validation_report,
)


class FakeWeave:
    def __init__(self, *, init_error=None, log_error=None):
        self.init_error = init_error
        self.log_error = log_error
        self.init_calls = []
        self.op_calls = []
        self.events = []

    def init(self, project_name):
        self.init_calls.append(project_name)
        if self.init_error is not None:
            raise self.init_error

    def op(self, function):
        self.op_calls.append(function.__name__)

        def wrapped(payload):
            if self.log_error is not None:
                raise self.log_error
            self.events.append(payload)
            return function(payload)

        return wrapped


class FakeWandbRun:
    def __init__(self, *, log_error=None, finish_error=None):
        self.log_error = log_error
        self.finish_error = finish_error
        self.logs = []
        self.summary = {}
        self.finish_calls = 0
        self.url = "https://wandb.ai/dzlab/kaggriculture/runs/test"

    def log(self, payload):
        if self.log_error is not None:
            raise self.log_error
        self.logs.append(payload)

    def finish(self):
        if self.finish_error is not None:
            raise self.finish_error
        self.finish_calls += 1


class FakeWandb:
    def __init__(self, *, run=None, init_error=None):
        self.run = run or FakeWandbRun()
        self.init_error = init_error
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.init_error is not None:
            raise self.init_error
        return self.run


def test_training_telemetry_writes_jsonl_without_weave(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    telemetry = TrainingTelemetry(metrics_path)

    telemetry("behavior_clone", {"loss": 1.25, "update_count": 3})

    assert load_metrics(metrics_path) == [
        {"event": "behavior_clone", "loss": 1.25, "update_count": 3}
    ]


def test_training_telemetry_initializes_fake_weave_with_default_project(tmp_path):
    fake_weave = FakeWeave()
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl", enable_weave=True, weave_module=fake_weave,
    )

    telemetry("ppo", {"policy_loss": 0.5, "step": 1})

    assert fake_weave.init_calls == [DEFAULT_WEAVE_PROJECT]
    assert fake_weave.op_calls == ["log_metrics"]
    assert fake_weave.events == [
        {"event": "ppo", "policy_loss": 0.5, "step": 1}
    ]


def test_training_telemetry_initializes_wandb_and_logs_metrics(tmp_path):
    fake_wandb = FakeWandb()
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl",
        enable_wandb=True,
        wandb_module=fake_wandb,
        wandb_config={"ppo_steps": 16},
        wandb_run_name="ppo16-gpu",
    )

    telemetry("ppo", {"policy_loss": 0.5, "step": 1})
    telemetry.finish()

    assert fake_wandb.init_calls == [{
        "entity": DEFAULT_WANDB_ENTITY,
        "project": DEFAULT_WANDB_PROJECT,
        "name": "ppo16-gpu",
        "config": {"ppo_steps": 16},
        "mode": "online",
    }]
    assert fake_wandb.run.logs == [
        {"telemetry/event": "ppo", "ppo/policy_loss": 0.5, "ppo/step": 1}
    ]
    assert fake_wandb.run.finish_calls == 1


def test_wandb_config_preserves_experiment_identity(tmp_path):
    fake_wandb = FakeWandb()
    TrainingTelemetry(
        tmp_path / "metrics.jsonl", enable_wandb=True, wandb_module=fake_wandb,
        wandb_config={
            "experiment_id": "orbit-context-test",
            "feature_variant": "experimental_context_v1",
            "training_mode": "reduced_behavior_clone_then_ppo",
        },
    )

    assert fake_wandb.init_calls[0]["config"] == {
        "experiment_id": "orbit-context-test",
        "feature_variant": "experimental_context_v1",
        "training_mode": "reduced_behavior_clone_then_ppo",
    }


def test_wandb_initialization_failure_warns_but_local_logging_continues(tmp_path, caplog):
    fake_wandb = FakeWandb(init_error=RuntimeError("login unavailable"))
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl",
        enable_wandb=True,
        wandb_module=fake_wandb,
    )

    telemetry("ppo", {"step": 1})

    assert load_metrics(tmp_path / "metrics.jsonl") == [{"event": "ppo", "step": 1}]
    assert "W&B telemetry disabled" in caplog.text
    assert "login unavailable" in caplog.text


def test_weave_initialization_failure_warns_but_local_logging_continues(tmp_path, caplog):
    fake_weave = FakeWeave(init_error=RuntimeError("login unavailable"))
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl", enable_weave=True, weave_module=fake_weave,
    )

    telemetry("ppo", {"step": 1})

    assert load_metrics(tmp_path / "metrics.jsonl") == [{"event": "ppo", "step": 1}]
    assert "Weave telemetry disabled" in caplog.text
    assert "login unavailable" in caplog.text


def test_weave_logging_failure_warns_but_local_logging_continues(tmp_path, caplog):
    fake_weave = FakeWeave(log_error=RuntimeError("network unavailable"))
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl", enable_weave=True, weave_module=fake_weave,
    )

    telemetry("ppo", {"step": 1})

    assert load_metrics(tmp_path / "metrics.jsonl") == [{"event": "ppo", "step": 1}]
    assert "Weave metric logging failed" in caplog.text
    assert "network unavailable" in caplog.text


def test_strict_weave_failures_raise_after_or_before_local_logging(tmp_path):
    init_failure = FakeWeave(init_error=RuntimeError("bad init"))
    with pytest.raises(RuntimeError, match="bad init"):
        TrainingTelemetry(
            tmp_path / "init.jsonl", enable_weave=True,
            weave_module=init_failure, strict=True,
        )

    log_failure = FakeWeave(log_error=RuntimeError("bad log"))
    telemetry = TrainingTelemetry(
        tmp_path / "log.jsonl", enable_weave=True,
        weave_module=log_failure, strict=True,
    )
    with pytest.raises(RuntimeError, match="bad log"):
        telemetry("ppo", {"step": 1})
    assert load_metrics(tmp_path / "log.jsonl") == [{"event": "ppo", "step": 1}]


def test_load_metrics_handles_missing_and_malformed_files(tmp_path):
    assert load_metrics(tmp_path / "missing.jsonl") == []
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event":"ppo","step":1}\nnot-json\n', encoding="utf-8")

    assert load_metrics(path) == [{"event": "ppo", "step": 1}]


def test_training_telemetry_sanitizes_nonfinite_and_path_values(tmp_path):
    telemetry = TrainingTelemetry(tmp_path / "metrics.jsonl")

    telemetry("ppo", {"nan_value": float("nan"), "path_value": tmp_path})

    assert load_metrics(tmp_path / "metrics.jsonl") == [{
        "event": "ppo", "nan_value": None, "path_value": str(tmp_path),
    }]


def test_record_validation_report_emits_raw_games_and_flattened_candidate_summaries(tmp_path):
    telemetry = TrainingTelemetry(tmp_path / "metrics.jsonl")
    report = {
        "records": {
            "current": [
                {
                    "candidate": "current",
                    "opponent": "random",
                    "seed": 7,
                    "seat": 0,
                    "outcome": "win",
                    "bank_differential": 25.0,
                    "terminal_cash": 110.0,
                    "terminal_inventory_value": 40.0,
                    "missed_basic_needs": 0,
                    "framework_error": False,
                    "framework_error_reasons": [],
                },
                {
                    "candidate": "current",
                    "opponent": "random",
                    "seed": 7,
                    "seat": 1,
                    "outcome": "loss",
                    "bank_differential": -5.0,
                    "terminal_cash": 90.0,
                    "terminal_inventory_value": 20.0,
                    "missed_basic_needs": 1,
                    "framework_error": False,
                    "framework_error_reasons": [],
                },
            ],
            "candidate": [
                {
                    "candidate": "candidate",
                    "opponent": "random",
                    "seed": 7,
                    "seat": 0,
                    "outcome": "win",
                    "bank_differential": 45.0,
                    "terminal_cash": 120.0,
                    "terminal_inventory_value": 50.0,
                    "missed_basic_needs": 0,
                    "framework_error": False,
                },
            ],
        },
        "summaries": {
            "current": {
                "record_count": 2,
                "paired_games": 1,
                "wins": 1,
                "losses": 1,
                "ties": 0,
                "seat_balanced_win_rate": 0.5,
                "mean_paired_bank_differential": 10.0,
                "median_paired_bank_differential": 10.0,
                "fifth_percentile_bank_differential": -4.0,
                "mean_paired_terminal_cash": 100.0,
                "median_paired_terminal_cash": 100.0,
                "mean_paired_terminal_inventory_value": 30.0,
                "median_paired_terminal_inventory_value": 30.0,
                "wilson_win_rate": {"lower": 0.1, "upper": 0.9},
                "confidence": {
                    "bootstrap_seat_balanced_win_rate": {"lower": 0.2, "upper": 0.8},
                    "bootstrap_bank_differential": {"lower": -3.0, "upper": 23.0},
                },
                "elo": {
                    "ratings": {"current": 1012.5, "random": 987.5},
                    "games": {"current": 2, "random": 2},
                },
            },
            "candidate": {
                "record_count": 1,
                "paired_games": 0,
                "wins": 1,
                "losses": 0,
                "ties": 0,
                "seat_balanced_win_rate": 1.0,
                "wilson_win_rate": {"lower": 0.2, "upper": 1.0},
                "confidence": {
                    "bootstrap_seat_balanced_win_rate": {"lower": 0.5, "upper": 1.0},
                    "bootstrap_bank_differential": {"lower": 45.0, "upper": 45.0},
                },
                "elo": {"ratings": {"candidate": 1020.0}, "games": {"candidate": 1}},
            },
        },
        "matrix_completeness": {
            "current": {
                "expected_count": 2,
                "observed_count": 2,
                "missing": [],
                "duplicate": [],
                "extra": [],
                "invalid_records": 0,
            },
            "candidate": {
                "expected_count": 2,
                "observed_count": 1,
                "missing": [["random", 7, 1]],
                "duplicate": [],
                "extra": [],
                "invalid_records": 0,
            },
        },
        "decision": {"status": "discard", "reasons": ["incomplete_matrix"]},
    }

    record_validation_report(
        telemetry, report, phase="development", checkpoint=12, candidate_tag="ppo16",
        experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    events = load_metrics(tmp_path / "metrics.jsonl")
    games = [event for event in events if event["event"] == "validation_game"]
    summaries = [event for event in events if event["event"] == "validation_summary"]
    breakdowns = [event for event in events if event["event"] == "validation_breakdown"]

    assert len(games) == 3
    assert games[0]["phase"] == "development"
    assert games[0]["checkpoint"] == 12
    assert games[0]["candidate_tag"] == "ppo16"
    assert games[0]["candidate"] == "current"
    assert games[0]["experiment_id"] == "orbit-context-test"
    assert games[0]["feature_variant"] == "experimental_context_v1"
    assert games[0]["training_mode"] == "reduced_behavior_clone_then_ppo"
    assert games[0]["bank_differential"] == 25.0
    assert games[0]["terminal_inventory_value"] == 40.0

    assert {event["candidate"] for event in summaries} == {"current", "candidate"}
    current = next(event for event in summaries if event["candidate"] == "current")
    assert current["seat_balanced_win_rate"] == 0.5
    assert current["wins"] == 1
    assert current["losses"] == 1
    assert current["ties"] == 0
    assert current["wilson_win_rate_lower"] == 0.1
    assert current["wilson_win_rate_upper"] == 0.9
    assert current["bootstrap_win_rate_lower"] == 0.2
    assert current["bootstrap_win_rate_upper"] == 0.8
    assert current["bootstrap_bank_differential_lower"] == -3.0
    assert current["bootstrap_bank_differential_upper"] == 23.0
    assert current["elo_rating"] == 1012.5
    assert current["mean_paired_bank_differential"] == 10.0
    assert current["median_paired_bank_differential"] == 10.0
    assert current["p05_bank_differential"] == -4.0
    assert current["mean_terminal_cash"] == 100.0
    assert current["mean_terminal_inventory_value"] == 30.0
    assert current["framework_error_rate"] == 0.0
    assert current["missed_needs_rate"] == 0.5
    assert current["matrix_complete"] is True
    assert current["matrix_expected_count"] == 2
    assert current["decision_status"] == "discard"
    assert current["decision_reasons"] == "incomplete_matrix"
    candidate_summary = next(event for event in summaries if event["candidate"] == "candidate")
    assert candidate_summary["delta_vs_current_win_rate"] == 0.5
    assert len(breakdowns) == 5
    assert {event["breakdown"] for event in breakdowns} == {"opponent", "seat"}
    assert any(
        event["breakdown"] == "seat"
        and event["dimension_value"] == 0
        and event["candidate"] == "current"
        and event["win_rate"] == 1.0
        for event in breakdowns
    )


def test_validation_breakdown_excludes_framework_errors_from_win_rate(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    telemetry = TrainingTelemetry(metrics_path)
    report = {
        "records": {"candidate": [
            {"opponent": "random", "seat": 0, "outcome": "win", "framework_error": False},
            {"opponent": "random", "seat": 0, "outcome": "win", "framework_error": True},
        ]},
    }

    record_validation_report(
        telemetry, report, phase="development", checkpoint=1, candidate_tag="ppo16",
    )

    events = load_metrics(metrics_path)
    opponent = next(event for event in events if event["event"] == "validation_breakdown" and event["breakdown"] == "opponent")
    assert opponent["games"] == 2
    assert opponent["valid_games"] == 1
    assert opponent["wins"] == 1
    assert opponent["win_rate"] == 1.0


def test_validation_matrix_ratio_penalizes_duplicate_records(tmp_path):
    telemetry = TrainingTelemetry(tmp_path / "metrics.jsonl")
    report = {
        "records": {"candidate": []},
        "summaries": {"candidate": {}},
        "matrix_completeness": {"candidate": {
            "expected_count": 2,
            "observed_count": 2,
            "missing": [],
            "duplicate": [["random", 1, 0]],
            "extra": [],
            "invalid_records": [],
        }},
    }

    record_validation_report(
        telemetry, report, phase="holdout", checkpoint=2, candidate_tag="ppo16",
    )

    summary = next(
        event for event in load_metrics(tmp_path / "metrics.jsonl")
        if event["event"] == "validation_summary"
    )
    assert summary["matrix_complete"] is False
    assert summary["matrix_completeness_ratio"] == 0.5


def test_record_validation_report_tolerates_missing_optional_report_fields(tmp_path):
    telemetry = TrainingTelemetry(tmp_path / "metrics.jsonl")

    record_validation_report(
        telemetry,
        {"records": {"candidate": [{"outcome": "tie", "framework_error": False}]}},
        phase="holdout",
        checkpoint="final",
        candidate_tag="ppo16",
    )

    events = load_metrics(tmp_path / "metrics.jsonl")
    assert [event["event"] for event in events] == [
        "validation_game", "validation_summary",
    ]
    summary = events[-1]
    assert summary["candidate"] == "candidate"
    assert summary["phase"] == "holdout"
    assert summary["checkpoint"] == "final"
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["ties"] == 1
    assert summary["framework_error_rate"] == 0.0
    assert summary["missed_needs_rate"] == 0.0
    assert summary["wilson_win_rate_lower"] is None
    assert summary["matrix_complete"] is None
    assert summary["decision_status"] is None
    assert summary["decision_reasons"] == ""


def test_record_validation_report_ignores_malformed_optional_numeric_fields(tmp_path):
    telemetry = TrainingTelemetry(tmp_path / "metrics.jsonl")

    record_validation_report(
        telemetry,
        {
            "records": {"candidate": [{"outcome": "win"}]},
            "summaries": {
                "candidate": {
                    "wins": 10 ** 1000,
                    "wilson_win_rate": {"lower": 10 ** 1000, "upper": float("nan")},
                },
            },
        },
        phase="development",
        checkpoint=1,
        candidate_tag="ppo16",
    )

    summary = load_metrics(tmp_path / "metrics.jsonl")[-1]
    assert summary["wins"] == 1
    assert summary["wilson_win_rate_lower"] is None
    assert summary["wilson_win_rate_upper"] is None


def _validated_report():
    records = [
        {"candidate": "candidate", "opponent": "hard", "seed": 1, "seat": seat,
         "outcome": "win", "bank_differential": 10.0, "framework_error": False}
        for seat in (0, 1)
    ]
    matrix = {
        "expected": [["hard", 1, 0], ["hard", 1, 1]],
        "expected_count": 2, "observed_count": 2,
        "missing": [], "duplicate": [], "extra": [], "invalid_records": 0,
    }
    summary = {
        "record_count": 2, "paired_games": 1, "wins": 2, "losses": 0, "ties": 0,
        "seat_balanced_win_rate": 1.0, "mean_paired_bank_differential": 10.0,
        "wilson_win_rate": {"lower": 0.2, "upper": 1.0},
        "elo": {"ratings": {"candidate": 1050.0}, "games": {"candidate": 1}},
    }
    return {
        "records": {"candidate": records}, "summaries": {"candidate": summary},
        "matrix_completeness": {"candidate": matrix},
        "promotion_decisions": {"candidate": {"status": "promote", "reasons": []}},
        "metrics_by_opponent": {"candidate": {"hard": {
            "seat_balanced_win_rate": 1.0, "elo_rating": 1050.0,
            "mean_bank_differential": 10.0, "safety_failure_rate": 0.0,
        }}},
        "promotion_evidence": {"candidate": {
            "status": "promote", "matrix_complete": True,
            "matrix_completeness": matrix,
        }},
    }


def test_wandb_receives_only_validated_summary_fields(tmp_path):
    fake_wandb = FakeWandb()
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl", enable_wandb=True, wandb_module=fake_wandb,
    )

    record_validation_report(
        telemetry, _validated_report(), phase="development", checkpoint=12, candidate_tag="ppo16",
    )

    remote_events = [entry["telemetry/event"] for entry in fake_wandb.run.logs]
    assert "validation_game" not in remote_events
    assert "validation_summary" in remote_events
    assert fake_wandb.run.summary["development_status"] == "promote"
    assert fake_wandb.run.summary["development_win_rate"] == 1.0
    assert fake_wandb.run.summary["development_elo"] == 1050.0
    assert fake_wandb.run.summary["promoted"] is True


def test_incomplete_validation_report_stays_local_and_does_not_update_wandb(tmp_path):
    fake_wandb = FakeWandb()
    telemetry = TrainingTelemetry(
        tmp_path / "metrics.jsonl", enable_wandb=True, wandb_module=fake_wandb,
    )
    report = _validated_report()
    report["promotion_evidence"]["candidate"]["matrix_complete"] = False

    record_validation_report(
        telemetry, report, phase="development", checkpoint=12, candidate_tag="ppo16",
    )

    assert load_metrics(tmp_path / "metrics.jsonl")
    assert fake_wandb.run.logs == []
    assert fake_wandb.run.summary == {}
