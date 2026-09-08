"""Local-first training telemetry with optional Weave call tracing."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_WEAVE_PROJECT = "dzlab/kaggriculture"


class TrainingTelemetry:
    """Append training events locally and optionally mirror them to Weave.

    The local JSONL write happens before the optional remote call. Weave is
    imported lazily so training remains usable without the observability extra.
    When ``strict`` is false, initialization and event failures are warnings;
    when true, the original exception is raised.
    """

    def __init__(
        self,
        metrics_path: str | Path,
        *,
        project_name: str = DEFAULT_WEAVE_PROJECT,
        enable_weave: bool = False,
        strict: bool = False,
        weave_module: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.metrics_path = Path(metrics_path).expanduser()
        self.project_name = project_name
        self.enable_weave = bool(enable_weave)
        self.strict = bool(strict)
        self._logger = logger or logging.getLogger(__name__)
        self._weave_log_metrics = None
        if self.enable_weave:
            self._initialize_weave(weave_module)

    def _initialize_weave(self, weave_module: Any | None) -> None:
        try:
            weave = weave_module or importlib.import_module("weave")
            weave.init(self.project_name)

            def log_metrics(payload: dict[str, Any]) -> dict[str, Any]:
                return payload

            self._weave_log_metrics = weave.op(log_metrics)
        except Exception as exc:
            self._handle_weave_failure("Weave telemetry disabled", exc)

    def _handle_weave_failure(self, message: str, exc: Exception) -> None:
        if self.strict:
            raise exc
        self._logger.warning("%s: %s", message, exc)
        self._weave_log_metrics = None

    def record(self, event: str, metrics: Mapping[str, Any] | None = None, **values: Any) -> None:
        """Persist one event and best-effort mirror it to Weave."""
        if not isinstance(event, str) or not event:
            raise ValueError("telemetry event must be a nonempty string")
        payload: dict[str, Any] = {"event": event}
        if metrics is not None:
            payload.update(dict(metrics))
        payload.update(values)
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
        if self._weave_log_metrics is not None:
            try:
                self._weave_log_metrics(payload)
            except Exception as exc:
                self._handle_weave_failure("Weave metric logging failed", exc)

    __call__ = record


def load_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    """Load valid JSON object events, returning an empty list when unavailable."""
    path = Path(metrics_path).expanduser()
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events
