import json
from pathlib import Path

import pytest

from scripts.telemetry import DEFAULT_WEAVE_PROJECT, TrainingTelemetry, load_metrics


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
