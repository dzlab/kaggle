"""Local-first training telemetry with optional Weave and W&B mirroring."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_WEAVE_PROJECT = "dzlab/kaggriculture"
DEFAULT_WANDB_ENTITY = "dzlab"
DEFAULT_WANDB_PROJECT = "kaggriculture"


class TrainingTelemetry:
    """Append training events locally and optionally mirror them remotely.

    The local JSONL write happens before optional remote calls. Weave and W&B
    are imported lazily so training remains usable without the observability
    extra. When ``strict`` is false, initialization and event failures are
    warnings; when true, the original exception is raised.
    """

    def __init__(
        self,
        metrics_path: str | Path,
        *,
        project_name: str = DEFAULT_WEAVE_PROJECT,
        enable_weave: bool = False,
        enable_wandb: bool = False,
        wandb_project: str = DEFAULT_WANDB_PROJECT,
        wandb_entity: str = DEFAULT_WANDB_ENTITY,
        wandb_run_name: str | None = None,
        wandb_config: Mapping[str, Any] | None = None,
        strict: bool = False,
        weave_module: Any | None = None,
        wandb_module: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.metrics_path = Path(metrics_path).expanduser()
        self.project_name = project_name
        self.enable_weave = bool(enable_weave)
        self.enable_wandb = bool(enable_wandb)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_config = dict(wandb_config or {})
        self.strict = bool(strict)
        self._logger = logger or logging.getLogger(__name__)
        self._weave_log_metrics = None
        self._wandb_run = None
        if self.enable_weave:
            self._initialize_weave(weave_module)
        if self.enable_wandb:
            self._initialize_wandb(wandb_module)

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

    def _initialize_wandb(self, wandb_module: Any | None) -> None:
        try:
            wandb = wandb_module or importlib.import_module("wandb")
            self._wandb_run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.wandb_config,
                mode="online",
            )
        except Exception as exc:
            self._handle_wandb_failure("W&B telemetry disabled", exc)

    def _handle_wandb_failure(self, message: str, exc: Exception) -> None:
        if self.strict:
            raise exc
        self._logger.warning("%s: %s", message, exc)
        self._wandb_run = None

    def record(self, event: str, metrics: Mapping[str, Any] | None = None, **values: Any) -> None:
        """Persist one event and best-effort mirror it to remote trackers."""
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
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({
                    "telemetry/event": event,
                    **{
                        f"{event}/{key}": value
                        for key, value in payload.items()
                        if key != "event"
                    },
                })
            except Exception as exc:
                self._handle_wandb_failure("W&B metric logging failed", exc)

    @property
    def wandb_url(self) -> str | None:
        """Return the remote W&B run URL when the SDK exposes one."""
        return getattr(self._wandb_run, "url", None)

    def finish(self) -> None:
        """Flush and close the optional W&B run."""
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        except Exception as exc:
            self._handle_wandb_failure("W&B run finalization failed", exc)

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
