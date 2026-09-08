"""Dry-run-first bounded model-scaling ladder for training experiments.

The module intentionally uses only the standard library.  It estimates costs
and expands a deterministic schedule; actual GPU training is outside this
opt-in harness and must be supplied by an explicit caller.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

MAX_LADDER_WIDTH = 4096
MAX_LADDER_DEPTH = 64
MAX_PPO_STEPS = 10_000_000
MAX_SEEDS = 256
MAX_EXPERIMENTS = 4096
MAX_ROLLOUT_EPISODES = 100_000
MAX_ROLLOUT_STEPS = 1_000_000
DEFAULT_INPUT_SIZE = 64
DEFAULT_OUTPUT_SIZE = 32
_PRODUCTION_DIRECTORY_NAMES = frozenset({
    "model", "models", "checkpoint", "checkpoints", "artifact", "artifacts",
})
_PROTECTED_OUTPUT_NAMES = frozenset({
    "model.json", "model.pt", "model.pth",
    "trained_model.json", "trained_model.pt", "trained_model.pth",
    "checkpoint.json", "checkpoint.pt", "checkpoint.pth",
    "artifact.json", "artifact.pt", "artifact.pth",
    "learned_v1.json", "learned_v1.pt",
})


def parse_ladder(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate a JSON ladder without importing training libraries."""
    if isinstance(value, Path):
        value = value.read_text()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            path = Path(value)
            if path.exists() and path.is_file():
                value = json.loads(path.read_text())
            else:
                raise ValueError(f"ladder must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("ladder must be a JSON object")
    if isinstance(value.get("ladder"), Mapping):
        value = value["ladder"]

    widths = _positive_int_list(value.get("widths"), "widths", MAX_LADDER_WIDTH)
    depths = _positive_int_list(value.get("depths"), "depths", MAX_LADDER_DEPTH)
    budgets = _positive_int_list(
        value.get("ppo_budgets", value.get("ppo_steps")),
        "ppo_budgets",
        MAX_PPO_STEPS,
    )
    seeds = _seeds(value)
    rollout_episodes = _positive_int(
        value.get("rollout_episodes", value.get("episodes", 1)),
        "rollout_episodes", MAX_ROLLOUT_EPISODES,
    )
    rollout_steps = _positive_int(
        value.get("rollout_steps", 64), "rollout_steps", MAX_ROLLOUT_STEPS,
    )
    normalized = {
        "widths": widths,
        "depths": depths,
        "ppo_budgets": budgets,
        "seeds": seeds,
        "rollout_episodes": rollout_episodes,
        "rollout_steps": rollout_steps,
    }
    for key in ("artifact_output", "model_output", "checkpoint_output"):
        if value.get(key) is not None:
            raise ValueError(f"{key} is a production artifact/checkpoint output")
    count = len(widths) * len(depths) * len(budgets) * len(seeds)
    if count > MAX_EXPERIMENTS:
        raise ValueError(f"ladder expands to {count} experiments; maximum is {MAX_EXPERIMENTS}")
    return normalized


def expand_ladder(ladder: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a normalized ladder in stable input order."""
    normalized = parse_ladder(ladder)
    experiments = []
    for width in normalized["widths"]:
        for depth in normalized["depths"]:
            for ppo_steps in normalized["ppo_budgets"]:
                for seed in normalized["seeds"]:
                    experiments.append({
                        "width": width,
                        "depth": depth,
                        "ppo_steps": ppo_steps,
                        "seed": seed,
                        "parameter_estimate": estimate_parameter_count(width, depth),
                        "rollout_budget": estimate_rollout_budget(
                            ppo_steps,
                            normalized["rollout_episodes"],
                            normalized["rollout_steps"],
                        ),
                    })
    return experiments


def estimate_parameter_count(
    width: int,
    depth: int,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> int:
    """Estimate dense MLP parameters for a fixed input/output interface."""
    width = _positive_int(width, "width", MAX_LADDER_WIDTH)
    depth = _positive_int(depth, "depth", MAX_LADDER_DEPTH)
    input_size = _positive_int(input_size, "input_size")
    output_size = _positive_int(output_size, "output_size")
    input_layer = input_size * width + width
    hidden_layers = max(0, depth - 1) * (width * width + width)
    output_layer = width * output_size + output_size
    return input_layer + hidden_layers + output_layer


def estimate_rollout_budget(ppo_steps: int, episodes: int, steps: int) -> int:
    """Estimate transitions collected across all configured PPO updates."""
    return (
        _positive_int(ppo_steps, "ppo_steps", MAX_PPO_STEPS)
        * _positive_int(episodes, "rollout_episodes", MAX_ROLLOUT_EPISODES)
        * _positive_int(steps, "rollout_steps", MAX_ROLLOUT_STEPS)
    )


def build_report(
    ladder: Mapping[str, Any], *, dry_run: bool = True,
    execution_callback: Any | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable deterministic estimate report."""
    normalized = parse_ladder(ladder)
    report = {
        "schema_version": 1,
        "dry_run": bool(dry_run),
        "ladder": normalized,
        "experiment_count": 0,
        "experiments": [],
        "total_parameter_estimate": 0,
        "total_rollout_budget": 0,
    }
    experiments = expand_ladder(normalized)
    report["experiment_count"] = len(experiments)
    report["experiments"] = experiments
    report["total_parameter_estimate"] = sum(item["parameter_estimate"] for item in experiments)
    report["total_rollout_budget"] = sum(item["rollout_budget"] for item in experiments)
    if not dry_run:
        if not callable(execution_callback):
            raise ValueError("execute mode requires an execution callback")
        report["experiments"] = _execute_experiments(experiments, execution_callback)
        report["execution"] = {
            "mode": "execute",
            "metrics_recorded": len(report["experiments"]),
        }
    return report


def execute_ladder(
    ladder: Mapping[str, Any], execution_callback: Any,
) -> dict[str, Any]:
    """Execute each expanded experiment through an injected callback.

    The callback receives one experiment estimate mapping and must return a
    JSON-serializable mapping containing evaluation metrics such as Elo,
    safety, latency, or other benchmark measurements.
    """
    return build_report(
        ladder, dry_run=False, execution_callback=execution_callback,
    )


def _execute_experiments(
    experiments: Sequence[Mapping[str, Any]], execution_callback: Any,
) -> list[dict[str, Any]]:
    records = []
    for experiment in experiments:
        metrics = execution_callback(dict(experiment))
        if not isinstance(metrics, Mapping):
            raise ValueError("execution callback must return a metrics object")
        try:
            # Validate finiteness and make the record JSON-safe without
            # importing a training or GPU dependency.
            normalized_metrics = json.loads(
                json.dumps(dict(metrics), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("execution callback returned non-JSON metrics") from exc
        records.append({**dict(experiment), "metrics": normalized_metrics})
    return records


def validate_report_path(path: str | Path, report_root: str | Path | None = None) -> Path:
    """Validate a report output against an explicit safe report root."""
    if report_root is None:
        raise ValueError("an explicit report root is required")
    root = Path(report_root).expanduser().resolve()
    if any(part.lower() in _PRODUCTION_DIRECTORY_NAMES for part in root.parts):
        raise ValueError("report root may not be nested under a production path")

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    elif raw_path.parts and raw_path.parts[0].lower() == root.name.lower():
        candidate = (Path.cwd() / raw_path).resolve()
    else:
        candidate = (root / raw_path).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("production/output path must remain inside the report root")
    relative_parts = candidate.relative_to(root).parts
    if any(part.lower() in _PRODUCTION_DIRECTORY_NAMES for part in relative_parts):
        raise ValueError("report path may not target a production artifact or checkpoint")
    if candidate.name.lower() in _PROTECTED_OUTPUT_NAMES:
        raise ValueError("report path may not target a production artifact or checkpoint")
    if candidate.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("report path must be JSON")
    return candidate


def write_report_atomically(
    report: Mapping[str, Any], path: str | Path, *, report_root: str | Path | None = None,
) -> Path:
    """Write a stable JSON report using a same-directory replace."""
    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")
    destination = validate_report_path(path, report_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder", required=True,
        help="JSON object or path containing widths, depths, ppo_budgets, and seeds",
    )
    parser.add_argument("--output", type=Path, help="explicit JSON report output path")
    parser.add_argument(
        "--report-root", type=Path, default=Path("reports"),
        help="safe root for explicit report output (default: reports)",
    )
    parser.add_argument(
        "--callback", help="execute-mode callback as importable module:function",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    mode.add_argument("--execute", dest="dry_run", action="store_false", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ladder = parse_ladder(args.ladder)
    execution_callback = None
    if not args.dry_run:
        if not args.callback:
            raise ValueError("--execute requires --callback module:function")
        execution_callback = _load_callback(args.callback)
    report = build_report(
        ladder, dry_run=args.dry_run, execution_callback=execution_callback,
    )
    if args.output is not None:
        write_report_atomically(report, args.output, report_root=args.report_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _load_callback(specification: str) -> Any:
    try:
        module_name, function_name = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError("callback must use module:function syntax") from exc
    if not module_name or not function_name:
        raise ValueError("callback must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(callback):
        raise ValueError(f"callback is not callable: {specification}")
    return callback


def _positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        suffix = f" at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return value


def _positive_int_list(value: Any, name: str, maximum: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a non-empty JSON array")
    result = [_positive_int(item, name, maximum) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _seeds(value: Mapping[str, Any]) -> list[int]:
    raw = value.get("seeds")
    if raw is None:
        count = _positive_int(value.get("seed_count", 1), "seed_count", MAX_SEEDS)
        start = value.get("seed_start", 0)
        if type(start) is not int or start < 0:
            raise ValueError("seed_start must be a nonnegative integer")
        raw = list(range(start, start + count))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("seeds must be a non-empty JSON array")
    if len(raw) > MAX_SEEDS:
        raise ValueError(f"seeds must contain at most {MAX_SEEDS} values")
    seeds = []
    for seed in raw:
        if type(seed) is not int or seed < 0:
            raise ValueError("seeds must contain nonnegative integers")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return seeds


if __name__ == "__main__":
    raise SystemExit(main())
