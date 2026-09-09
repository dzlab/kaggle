"""Colab/Drive-friendly configuration and entrypoint for candidate training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from pickle import UnpicklingError
from stat import S_ISREG
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.checkpoints import CheckpointError, read_checkpoint
from kagriculture_agent.model import resolve_device
from scripts.telemetry import record_validation_report
from scripts.train_policy import (
    TrainingContract,
    validate_training_checkpoint,
)
from scripts.training_identity import (
    DEFAULT_EXPERIMENT_ID,
    FEATURE_VARIANTS,
    TRAINING_MODES,
    validate_training_identity,
)

COLLECT_SCRIPT = Path(__file__).with_name("collect_trajectories.py")
EVALUATE_SCRIPT = Path(__file__).with_name("evaluate_artifact.py")
RUN_LOCAL_SCRIPT = Path(__file__).with_name("run_local.py")


def _absolute_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class ColabConfig:
    run_directory: Path
    device: str
    workers: int
    development_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    resume: Path | None = None
    trajectory_path: Path | None = None
    ppo_target_steps: int = 16
    training_steps: int = 25
    training_batch_size: int = 256
    training_seed: int = 7
    training_checkpoint_interval: int = 25
    training_prior_checkpoint: Path | None = None
    training_offline_ppo_fallback: bool = False
    collection_seed_values: tuple[int, ...] = tuple(range(8))
    collection_steps: int = 96
    collection_opponents: tuple[str, ...] = ("pass", "random", "starter")
    collection_seats: tuple[int, ...] = (0, 1)
    rollout_seed_values: tuple[int, ...] = (0, 1, 2, 3)
    rollout_steps: int = 97
    development_opponents: tuple[str, ...] = ("pass", "random", "starter")
    development_steps: int = 96
    development_seats: tuple[int, ...] = (0, 1)
    holdout_opponents: tuple[str, ...] = ("pass", "random", "starter")
    holdout_steps: int = 96
    holdout_seats: tuple[int, ...] = (0, 1)
    evaluation_timeout: float | None = None
    mount_drive: bool = True
    drive_mountpoint: Path = Path("/content/drive")
    wandb_enabled: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    smoke_opponent: str = "pass"
    smoke_seed: int = 0
    smoke_steps: int = 96
    plot: bool = False
    plot_path: Path | None = None
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    feature_variant: str = "production_v1"
    training_mode: str = "behavior_clone_then_ppo"

    @property
    def candidate_tag(self) -> str:
        return f"ppo{self.ppo_target_steps}"

    @property
    def current_checkpoint_path(self) -> Path:
        return self.run_directory / "policy.pt"

    @property
    def stage_checkpoint_path(self) -> Path:
        return self.run_directory / f"policy-{self.candidate_tag}.pt"

    @property
    def stage_artifact_path(self) -> Path:
        return self.run_directory / f"policy-{self.candidate_tag}.json"

    @property
    def training_metrics_path(self) -> Path:
        return self.run_directory / f"{self.candidate_tag}-training-metrics.jsonl"

    @property
    def development_report_path(self) -> Path:
        return self.run_directory / f"{self.candidate_tag}-development-evaluation.json"

    @property
    def holdout_report_path(self) -> Path:
        return self.run_directory / f"{self.candidate_tag}-holdout-evaluation.json"

    @property
    def smoke_replay_path(self) -> Path:
        return self.run_directory / "candidate-smoke.json"


def default_wandb_run_name(
    config: ColabConfig, *, timestamp: datetime | None = None,
) -> str:
    """Build a unique, configuration-aware default W&B experiment name."""
    run_timestamp = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (
        f"kaggriculture-{config.candidate_tag}-bc{config.training_steps}"
        f"-seed{config.training_seed}-{run_timestamp:%Y%m%d-%H%M%S}"
    )


@dataclass(frozen=True)
class WorkflowResult:
    """Observable result of one workflow invocation."""

    dry_run: bool
    commands: tuple[tuple[str, ...], ...] = ()
    resume_checkpoint: Path | None = None
    development_evaluation_promoted: bool | None = None
    holdout_evaluation_complete: bool | None = None
    stage_artifact_path: Path | None = None


@dataclass(frozen=True)
class ResumeSelection:
    """A validated resume candidate and its behavior-cloning/PPO progress."""

    path: Path
    progress_epoch: int
    progress_cursor: int
    behavior_clone_updates: int
    ppo_target_steps: int
    progress_round: int
    completed_ppo_steps: int


class CheckpointCompatibilityError(ValueError):
    """An optional selector validator's intentional incompatibility result."""


def _resume_selection_metadata(
    payload: Mapping[str, Any], *, path: Path, requested_ppo_steps: int,
) -> ResumeSelection:
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration is missing")
    target = configuration.get("ppo_steps")
    if type(target) is not int or target < 0:
        raise ValueError("configuration.ppo_steps must be a nonnegative integer")
    if target > requested_ppo_steps:
        raise ValueError(
            f"saved PPO target {target} exceeds requested target {requested_ppo_steps}"
        )

    progress = payload.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("progress is missing")
    progress_epoch = progress.get("epoch")
    if type(progress_epoch) is not int or progress_epoch < 0:
        raise ValueError("progress.epoch must be a nonnegative integer")
    progress_cursor = progress.get("cursor")
    if type(progress_cursor) is not int or progress_cursor < 0:
        raise ValueError("progress.cursor must be a nonnegative integer")
    progress_round = progress.get("round")
    if type(progress_round) is not int or progress_round < 0:
        raise ValueError("progress.round must be a nonnegative integer")

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics is missing")
    behavior_clone_updates = metrics.get("behavior_clone_updates")
    if type(behavior_clone_updates) is not int or behavior_clone_updates < 0:
        raise ValueError("metrics.behavior_clone_updates must be a nonnegative integer")
    ppo_metrics = metrics.get("ppo_metrics")
    if ppo_metrics is None:
        completed_ppo_steps = 0
    elif isinstance(ppo_metrics, Mapping):
        completed_ppo_steps = ppo_metrics.get("completed_steps")
        if type(completed_ppo_steps) is not int or completed_ppo_steps < 0:
            raise ValueError("metrics.ppo_metrics.completed_steps must be a nonnegative integer")
    else:
        raise ValueError("metrics.ppo_metrics must be an object or null")
    if completed_ppo_steps != progress_round:
        raise ValueError(
            "progress.round must match metrics.ppo_metrics.completed_steps"
        )
    if completed_ppo_steps > target:
        raise ValueError("completed PPO steps exceed the saved PPO target")
    return ResumeSelection(
        path=path,
        progress_epoch=progress_epoch,
        progress_cursor=progress_cursor,
        behavior_clone_updates=behavior_clone_updates,
        ppo_target_steps=target,
        progress_round=progress_round,
        completed_ppo_steps=completed_ppo_steps,
    )


def select_resume_checkpoint(
    candidates: Sequence[str | Path], *, training_contract: TrainingContract,
    checkpoint_loader: Callable[..., Mapping[str, Any]] = read_checkpoint,
    checkpoint_validator: Callable[..., Any] | None = None,
    diagnostic: Callable[[str], Any] = print,
) -> ResumeSelection | None:
    """Select the most-progressed compatible checkpoint for a staged PPO run.

    An empty candidate set means this is a fresh run. Existing candidates that
    fail validation or exceed the requested target are reported and skipped;
    if every existing candidate is skipped, the caller gets a clear error
    instead of silently restarting from scratch.

    The canonical trainer validator always runs from ``training_contract``.
    An optional ``checkpoint_validator`` receives ``path``, the loaded
    ``payload``, and the strict ``allow_ppo_extension`` decision for extra
    checks; it must raise ``CheckpointCompatibilityError`` to reject a
    candidate. Other validator errors propagate as programming errors. Valid
    candidates are ranked by BC epoch, cursor, and update count first, then
    PPO progress and target, with mtime/path as final tie-breakers.
    """
    if not isinstance(training_contract, TrainingContract):
        raise TypeError("training_contract must be a TrainingContract")
    requested_ppo_steps = training_contract.configuration.get("ppo_steps")
    if type(requested_ppo_steps) is not int or requested_ppo_steps < 0:
        raise ValueError("training_contract PPO target must be a nonnegative integer")
    existing: list[tuple[Path, int]] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        try:
            path_stat = path.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            diagnostic(f"Unable to inspect resume checkpoint {path}: {exc}")
            raise
        if S_ISREG(path_stat.st_mode):
            existing.append((path, path_stat.st_mtime_ns))
    if not existing:
        return None

    valid: list[
        tuple[tuple[int, int, int, int, int, int, int, str], ResumeSelection]
    ] = []
    skipped = 0
    for path, mtime_ns in existing:
        try:
            payload = checkpoint_loader(path, map_location="cpu")
            selection = _resume_selection_metadata(
                payload, path=path, requested_ppo_steps=requested_ppo_steps,
            )
        except (
            CheckpointError, EOFError, FileNotFoundError, UnpicklingError, ValueError,
        ) as exc:
            skipped += 1
            diagnostic(f"Skipping resume checkpoint {path}: {exc}")
            continue
        try:
            validate_training_checkpoint(
                payload,
                contract=training_contract,
                allow_ppo_extension=(
                    selection.ppo_target_steps < requested_ppo_steps
                ),
            )
        except CheckpointError as exc:
            skipped += 1
            diagnostic(f"Skipping resume checkpoint {path}: {exc}")
            continue
        if checkpoint_validator is not None:
            try:
                result = checkpoint_validator(
                    path=path,
                    payload=payload,
                    allow_ppo_extension=(
                        selection.ppo_target_steps < requested_ppo_steps
                    ),
                )
            except CheckpointCompatibilityError as exc:
                skipped += 1
                diagnostic(f"Skipping resume checkpoint {path}: {exc}")
                continue
            if result is False:
                skipped += 1
                diagnostic(f"Skipping resume checkpoint {path}: validator rejected candidate")
                continue
        rank = (
            selection.progress_epoch,
            selection.progress_cursor,
            selection.behavior_clone_updates,
            selection.progress_round,
            selection.completed_ppo_steps,
            selection.ppo_target_steps,
            mtime_ns,
            str(path),
        )
        valid.append((rank, selection))

    if not valid:
        raise ValueError(
            f"No compatible resume checkpoint found for requested PPO target "
            f"{requested_ppo_steps}; skipped {skipped} existing candidate(s)"
        )
    return max(valid, key=lambda item: item[0])[1]


def build_config(
    *, run_directory: str | Path = "/content/drive/MyDrive/kagriculture-training",
    device: str = "auto", workers: int = 2,
    development_seeds: tuple[int, ...] = (0, 1, 2, 3),
    holdout_seeds: tuple[int, ...] = (100, 101),
    resume: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    ppo_target_steps: int = 16,
    training_steps: int = 25,
    training_batch_size: int = 256,
    training_seed: int = 7,
    training_checkpoint_interval: int = 25,
    training_prior_checkpoint: str | Path | None = None,
    training_offline_ppo_fallback: bool = False,
    collection_seed_values: tuple[int, ...] | None = None,
    collection_seed_count: int = 8,
    collection_start_seed: int = 0,
    collection_steps: int = 96,
    collection_opponents: tuple[str, ...] = ("pass", "random", "starter"),
    collection_seats: tuple[int, ...] = (0, 1),
    rollout_seed_values: tuple[int, ...] = (0, 1, 2, 3),
    rollout_steps: int = 97,
    development_opponents: tuple[str, ...] = ("pass", "random", "starter"),
    development_steps: int = 96,
    development_seats: tuple[int, ...] = (0, 1),
    holdout_opponents: tuple[str, ...] = ("pass", "random", "starter"),
    holdout_steps: int = 96,
    holdout_seats: tuple[int, ...] = (0, 1),
    evaluation_timeout: float | None = None,
    mount_drive: bool = True,
    drive_mountpoint: str | Path = "/content/drive",
    wandb_enabled: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    smoke_opponent: str = "pass",
    smoke_seed: int = 0,
    smoke_steps: int = 96,
    plot: bool = False,
    plot_path: str | Path | None = None,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    resolve_runtime_device: bool = True,
) -> ColabConfig:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(ppo_target_steps) is not int or ppo_target_steps < 0:
        raise ValueError("ppo_target_steps must be a nonnegative integer")
    validate_training_identity(experiment_id, feature_variant, training_mode)
    for name, value in (
        ("training_steps", training_steps),
        ("training_batch_size", training_batch_size),
        ("training_checkpoint_interval", training_checkpoint_interval),
        ("collection_steps", collection_steps),
        ("rollout_steps", rollout_steps),
        ("development_steps", development_steps),
        ("holdout_steps", holdout_steps),
        ("smoke_steps", smoke_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(collection_seed_count) is not int or collection_seed_count < 1:
        raise ValueError("collection_seed_count must be a positive integer")
    if evaluation_timeout is not None and (
        isinstance(evaluation_timeout, bool)
        or not isinstance(evaluation_timeout, (int, float))
        or not math.isfinite(float(evaluation_timeout))
        or evaluation_timeout <= 0
    ):
        raise ValueError("evaluation_timeout must be a positive finite number")
    if collection_seed_values is None:
        collection = tuple(range(int(collection_start_seed), int(collection_start_seed) + collection_seed_count))
    else:
        collection = tuple(int(seed) for seed in collection_seed_values)
        if not collection:
            raise ValueError("collection_seed_values must not be empty")
    development = tuple(int(seed) for seed in development_seeds)
    holdout = tuple(int(seed) for seed in holdout_seeds)
    if not development or not holdout:
        raise ValueError("development and holdout seeds must not be empty")
    if set(development) & set(holdout):
        raise ValueError("holdout seeds must not overlap development seeds")
    if smoke_opponent not in {"pass", "random", "starter"}:
        raise ValueError("smoke_opponent must be pass, random, or starter")
    seat_values = {}
    for name, seats in (
        ("collection_seats", collection_seats),
        ("development_seats", development_seats),
        ("holdout_seats", holdout_seats),
    ):
        normalized_seats = tuple(seats)
        if not normalized_seats:
            raise ValueError(f"{name} must not be empty")
        if any(type(seat) is not int or seat not in (0, 1) for seat in normalized_seats):
            raise ValueError(f"{name} must contain only supported seats 0 and 1")
        if len(set(normalized_seats)) != len(normalized_seats):
            raise ValueError(f"{name} must contain unique seats")
        seat_values[name] = normalized_seats
    allowed_opponents = {"pass", "random", "starter"}
    for name, opponents in (
        ("collection_opponents", collection_opponents),
        ("development_opponents", development_opponents),
        ("holdout_opponents", holdout_opponents),
    ):
        if not opponents or any(opponent not in allowed_opponents for opponent in opponents):
            raise ValueError(f"{name} must contain supported opponents")
    resolved = str(resolve_device(device)) if resolve_runtime_device else device
    run_path = _absolute_path(run_directory)
    input_path = (
        _absolute_path(trajectory_path)
        if trajectory_path is not None else run_path / "bootstrap-trajectories.jsonl"
    )
    return ColabConfig(
        run_path, resolved, workers, development, holdout,
        _absolute_path(resume) if resume is not None else None,
        input_path,
        ppo_target_steps,
        training_steps,
        training_batch_size,
        training_seed,
        training_checkpoint_interval,
        _absolute_path(training_prior_checkpoint) if training_prior_checkpoint is not None else None,
        training_offline_ppo_fallback,
        collection,
        collection_steps,
        tuple(collection_opponents),
        seat_values["collection_seats"],
        tuple(int(seed) for seed in rollout_seed_values),
        rollout_steps,
        tuple(development_opponents),
        development_steps,
        seat_values["development_seats"],
        tuple(holdout_opponents),
        holdout_steps,
        seat_values["holdout_seats"],
        evaluation_timeout,
        mount_drive,
        _absolute_path(drive_mountpoint),
        wandb_enabled,
        wandb_project,
        wandb_entity,
        wandb_run_name,
        smoke_opponent,
        int(smoke_seed),
        smoke_steps,
        plot,
        _absolute_path(plot_path) if plot_path is not None else None,
        experiment_id,
        feature_variant,
        training_mode,
    )


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _path_or_none(value: str) -> Path | None:
    if value.lower() == "none":
        return None
    return Path(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path("/content/drive/MyDrive/kagriculture-training"))
    parser.add_argument("--trajectory-path", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--ppo-target-steps", type=_nonnegative_int, default=16)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="production_v1")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="behavior_clone_then_ppo")
    parser.add_argument("--training-steps", type=_positive_int, default=25)
    parser.add_argument("--training-batch-size", type=_positive_int, default=256)
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--training-checkpoint-interval", type=_positive_int, default=25)
    parser.add_argument("--training-prior-checkpoint", type=_path_or_none, default=None)
    parser.add_argument(
        "--training-offline-ppo-fallback",
        dest="training_offline_ppo_fallback",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-training-offline-ppo-fallback",
        dest="training_offline_ppo_fallback",
        action="store_false",
    )
    parser.add_argument("--collection-seeds", type=_positive_int, default=8)
    parser.add_argument("--collection-start-seed", type=int, default=0)
    parser.add_argument("--collection-steps", type=_positive_int, default=96)
    parser.add_argument("--collection-opponents", nargs="+", choices=("pass", "random", "starter"), default=["pass", "random", "starter"])
    parser.add_argument("--collection-seats", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--rollout-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--rollout-steps", type=_positive_int, default=97)
    parser.add_argument("--development-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--development-opponents", nargs="+", choices=("pass", "random", "starter"), default=["pass", "random", "starter"])
    parser.add_argument("--development-steps", type=_positive_int, default=96)
    parser.add_argument("--development-seats", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=[100, 101])
    parser.add_argument("--holdout-opponents", nargs="+", choices=("pass", "random", "starter"), default=["pass", "random", "starter"])
    parser.add_argument("--holdout-steps", type=_positive_int, default=96)
    parser.add_argument("--holdout-seats", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--evaluation-timeout", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--mount-drive", dest="mount_drive", action="store_true", default=True)
    parser.add_argument("--no-mount-drive", dest="mount_drive", action="store_false")
    parser.add_argument("--drive-mountpoint", type=Path, default=Path("/content/drive"))
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-run-name", default=None,
        help="Explicit W&B experiment name; otherwise derive one from the run configuration.",
    )
    parser.add_argument("--smoke-opponent", choices=("pass", "random", "starter"), default="pass")
    parser.add_argument("--smoke-seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=_positive_int, default=96)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse all workflow parameters without importing optional integrations."""
    return _parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ColabConfig:
    """Build a validated workflow configuration from parsed CLI arguments."""
    return build_config(
        run_directory=args.run_directory,
        trajectory_path=args.trajectory_path,
        device=args.device,
        workers=args.workers,
        ppo_target_steps=args.ppo_target_steps,
        experiment_id=args.experiment_id,
        feature_variant=args.feature_variant,
        training_mode=args.training_mode,
        training_steps=args.training_steps,
        training_batch_size=args.training_batch_size,
        training_seed=args.training_seed,
        training_checkpoint_interval=args.training_checkpoint_interval,
        training_prior_checkpoint=args.training_prior_checkpoint,
        training_offline_ppo_fallback=args.training_offline_ppo_fallback,
        collection_seed_count=args.collection_seeds,
        collection_start_seed=args.collection_start_seed,
        collection_steps=args.collection_steps,
        collection_opponents=tuple(args.collection_opponents),
        collection_seats=tuple(args.collection_seats),
        rollout_seed_values=tuple(args.rollout_seeds),
        rollout_steps=args.rollout_steps,
        development_seeds=tuple(args.development_seeds),
        development_opponents=tuple(args.development_opponents),
        development_steps=args.development_steps,
        development_seats=tuple(args.development_seats),
        holdout_seeds=tuple(args.holdout_seeds),
        holdout_opponents=tuple(args.holdout_opponents),
        holdout_steps=args.holdout_steps,
        holdout_seats=tuple(args.holdout_seats),
        evaluation_timeout=args.evaluation_timeout,
        resume=args.resume,
        mount_drive=args.mount_drive,
        drive_mountpoint=args.drive_mountpoint,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        smoke_opponent=args.smoke_opponent,
        smoke_seed=args.smoke_seed,
        smoke_steps=args.smoke_steps,
        plot=args.plot,
        plot_path=args.plot_path,
        resolve_runtime_device=not args.dry_run,
    )


def _require_contiguous_seeds(seeds: Sequence[int], *, name: str) -> None:
    values = tuple(seeds)
    expected = tuple(range(values[0], values[0] + len(values))) if values else ()
    if values != expected:
        raise ValueError(f"{name} must be a contiguous seed list")


def build_collection_command(config: ColabConfig) -> list[str]:
    """Build the deterministic bootstrap trajectory collection command."""
    trajectory_path = config.trajectory_path
    if trajectory_path is None:
        raise ValueError("trajectory_path is required")
    _require_contiguous_seeds(config.collection_seed_values, name="collection_seed_values")
    return [
        sys.executable, str(COLLECT_SCRIPT),
        "--seeds", str(len(config.collection_seed_values)),
        "--start-seed", str(config.collection_seed_values[0]),
        "--steps", str(config.collection_steps),
        "--opponents", *config.collection_opponents,
        "--seats", *(str(seat) for seat in config.collection_seats),
        "--workers", str(config.workers),
        "--output", str(trajectory_path),
        "--source-policy-identity", "current",
        "--experiment-id", config.experiment_id,
        "--feature-variant", config.feature_variant,
        "--training-mode", config.training_mode,
    ]


def build_evaluation_command(config: ColabConfig, *, phase: str) -> list[str]:
    """Build a development or holdout evaluator command with its own matrix."""
    if phase == "development":
        seeds = config.development_seeds
        opponents = config.development_opponents
        seats = config.development_seats
        steps = config.development_steps
        output = config.development_report_path
    elif phase == "holdout":
        seeds = config.holdout_seeds
        opponents = config.holdout_opponents
        seats = config.holdout_seats
        steps = config.holdout_steps
        output = config.holdout_report_path
    else:
        raise ValueError("phase must be development or holdout")
    _require_contiguous_seeds(seeds, name=f"{phase}_seeds")
    command = [
        sys.executable, str(EVALUATE_SCRIPT),
        "--artifact", str(config.stage_artifact_path),
        "--identity", config.candidate_tag,
        "--experiment-id", config.experiment_id,
        "--feature-variant", config.feature_variant,
        "--training-mode", config.training_mode,
        "--seeds", str(len(seeds)),
        "--start-seed", str(seeds[0]),
        "--steps", str(steps),
        "--opponents", *opponents,
        "--seats", *(str(seat) for seat in seats),
        "--workers", str(config.workers),
        "--min-valid-games", str(len(seeds) * len(opponents)),
        "--output", str(output),
    ]
    if config.evaluation_timeout is not None:
        command.extend(["--evaluation-timeout", str(config.evaluation_timeout)])
    return command


def build_smoke_command(config: ColabConfig) -> list[str]:
    """Build the dependency-free local simulator smoke command."""
    return [
        sys.executable, str(RUN_LOCAL_SCRIPT),
        "--opponent", config.smoke_opponent,
        "--seed", str(config.smoke_seed),
        "--steps", str(config.smoke_steps),
        "--replay", str(config.smoke_replay_path),
        "--candidate-artifact", str(config.stage_artifact_path),
    ]


def run_command(command: Sequence[str], *, check: bool, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one repository command through an injectable boundary."""
    return subprocess.run(
        list(command), cwd=PROJECT_ROOT, check=check,
        capture_output=capture_output, text=True,
    )


def mount_drive(config: ColabConfig) -> Path:
    """Mount Google Drive when the configured Colab workflow enables it."""
    mountpoint = config.drive_mountpoint
    drive_root = mountpoint / "MyDrive"
    if not drive_root.exists():
        from google.colab import drive

        drive.mount(str(mountpoint))
    else:
        print(f"Drive already mounted at {drive_root}")
    return drive_root


def initialize_telemetry(config: ColabConfig) -> Any | None:
    """Create local telemetry and optionally authenticate a W&B run."""
    import os

    from scripts.telemetry import (
        DEFAULT_WANDB_ENTITY,
        DEFAULT_WANDB_PROJECT,
        TrainingTelemetry,
    )

    if config.wandb_enabled:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if not wandb_api_key:
            try:
                from google.colab import userdata

                wandb_api_key = userdata.get("WANDB_API_KEY")
            except Exception:
                wandb_api_key = None
        if not wandb_api_key:
            raise RuntimeError("Configure a Colab secret named WANDB_API_KEY before training.")
        wandb.login(key=wandb_api_key, relogin=False)
    return TrainingTelemetry(
        config.training_metrics_path,
        enable_wandb=config.wandb_enabled,
        wandb_project=config.wandb_project or DEFAULT_WANDB_PROJECT,
        wandb_entity=config.wandb_entity or DEFAULT_WANDB_ENTITY,
        wandb_run_name=config.wandb_run_name or default_wandb_run_name(config),
        wandb_config={
            "experiment_id": config.experiment_id,
            "feature_variant": config.feature_variant,
            "training_mode": config.training_mode,
            "candidate_tag": config.candidate_tag,
            "ppo_target_steps": config.ppo_target_steps,
            "training_steps": config.training_steps,
            "batch_size": config.training_batch_size,
            "seed": config.training_seed,
            "device": config.device,
            "workers": config.workers,
        },
        strict=True,
    )


def _validate_explicit_resume(config: ColabConfig) -> None:
    if config.resume is None:
        return
    try:
        candidate_stat = config.resume.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"explicit resume checkpoint does not exist: {config.resume}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"unable to inspect explicit resume checkpoint {config.resume}: {exc}"
        ) from exc
    if not S_ISREG(candidate_stat.st_mode):
        raise ValueError(
            f"explicit resume checkpoint is not a regular file: {config.resume}"
        )


def _checkpoint_candidates(config: ColabConfig) -> tuple[Path, ...]:
    if config.resume is not None:
        return (config.resume,)
    return (config.current_checkpoint_path, *sorted(config.run_directory.glob("policy-ppo*.pt")))


def compatible_prior_checkpoints(
    config: ColabConfig, *, training_contract: TrainingContract,
) -> list[Path]:
    """Filter the league pool with the same strict checkpoint validator as resume."""
    compatible: list[Path] = []
    candidates = (
        config.current_checkpoint_path,
        *sorted(config.run_directory.glob("policy-ppo*.pt")),
    )
    for candidate in candidates:
        try:
            selection = select_resume_checkpoint(
                (candidate,), training_contract=training_contract,
                diagnostic=lambda message: print(message),
            )
        except ValueError as exc:
            print(f"Skipping malformed or incompatible checkpoint {candidate}: {exc}")
            continue
        if selection is not None:
            compatible.append(selection.path)
    return compatible


def train_candidate(
    config: ColabConfig, *, training_contract: TrainingContract,
    resume_checkpoint: Path | None, allow_ppo_extension: bool,
    opponent_pool: Any, telemetry: Any | None,
) -> dict[str, Any]:
    """Run behavior cloning/PPO and publish only the stage checkpoint/artifact."""
    from scripts.export_policy import export_checkpoint
    from scripts.train_policy import make_fresh_rollout_fn, train_behavior_clone

    def export_current(*, output_path: str | Path, **_kwargs: Any) -> Path:
        export_checkpoint(config.stage_checkpoint_path, output_path)
        return Path(output_path)

    fresh_rollout = make_fresh_rollout_fn(
        run_directory=config.run_directory / "ppo-rollouts",
        candidate_artifact_callback=export_current,
        seeds=config.rollout_seed_values,
        steps=config.rollout_steps,
        workers=config.workers,
        experiment_id=config.experiment_id,
        feature_variant=config.feature_variant,
        training_mode=config.training_mode,
    )

    def record_training_event(event: str, metrics: Mapping[str, Any] | None = None, **values: Any) -> None:
        if telemetry is not None:
            telemetry(
                event, metrics, **_identity_fields(config), candidate_tag=config.candidate_tag,
                ppo_target_steps=config.ppo_target_steps, **values,
            )

    metadata = train_behavior_clone(
        input_path=config.trajectory_path,
        output_path=config.stage_checkpoint_path,
        steps=config.training_steps,
        batch_size=config.training_batch_size,
        seed=config.training_seed,
        ppo_steps=config.ppo_target_steps,
        device=config.device,
        checkpoint_interval=config.training_checkpoint_interval,
        resume_checkpoint=resume_checkpoint,
        allow_ppo_extension=allow_ppo_extension,
        prior_checkpoint=config.training_prior_checkpoint,
        opponent_pool=opponent_pool,
        rollout_fn=fresh_rollout,
        offline_ppo_fallback=config.training_offline_ppo_fallback,
        candidate_artifact=config.stage_artifact_path,
        telemetry_callback=record_training_event if telemetry is not None else None,
        experiment_id=config.experiment_id,
        feature_variant=config.feature_variant,
        training_mode=config.training_mode,
    )
    export_checkpoint(config.stage_checkpoint_path, config.stage_artifact_path)
    return metadata


def _expected_evaluation_matrix(
    seed_values: Sequence[int], opponents: Sequence[str], seats: Sequence[int],
) -> list[list[str | int]]:
    return [
        [opponent, seed, seat]
        for opponent in opponents
        for seed in seed_values
        for seat in seats
    ]


def evaluation_report_is_complete(
    report: Mapping[str, Any], *, identity: str, seed_values: Sequence[int],
    opponents: Sequence[str] = ("pass", "random", "starter"),
    seats: Sequence[int] = (0, 1),
    artifact_path: str | Path | None = None,
) -> bool:
    """Validate the evaluator's complete matrix before treating its decision as evidence."""
    artifact = report.get("artifact")
    configuration = report.get("configuration")
    completeness = report.get("matrix_completeness", {})
    records = report.get("records", {})
    expected_matrix = report.get("expected_matrix", [])
    configured_matrix = _expected_evaluation_matrix(seed_values, opponents, seats)
    if not isinstance(artifact, Mapping) or not isinstance(configuration, Mapping):
        return False
    if artifact.get("identity") != identity:
        return False
    reported_hash = artifact.get("sha256")
    if (
        not isinstance(reported_hash, str)
        or len(reported_hash) != 64
        or any(character not in "0123456789abcdef" for character in reported_hash)
        or artifact_path is None
    ):
        return False
    candidate_path = Path(artifact_path).expanduser()
    if not candidate_path.is_file():
        return False
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if reported_hash != digest:
        return False
    if configuration.get("seed_values") != list(seed_values):
        return False
    if configuration.get("opponents") != list(opponents):
        return False
    if configuration.get("seats") != list(seats):
        return False
    if expected_matrix != configured_matrix:
        return False
    expected_coordinates = {tuple(coordinate) for coordinate in configured_matrix}
    if not isinstance(completeness, Mapping) or not isinstance(records, Mapping):
        return False
    if set(completeness) != {"current", identity} or set(records) != {"current", identity}:
        return False
    for candidate in ("current", identity):
        details = completeness.get(candidate)
        candidate_records = records.get(candidate)
        if not isinstance(details, Mapping) or not isinstance(candidate_records, list):
            return False
        if (
            details.get("expected") != configured_matrix
            or details.get("expected_count") != len(configured_matrix)
            or details.get("observed_count") != len(configured_matrix)
            or any(details.get(field) for field in ("missing", "duplicate", "extra", "invalid_records"))
        ):
            return False
        observed_coordinates = []
        for record in candidate_records:
            if not isinstance(record, Mapping):
                return False
            opponent = record.get("opponent")
            seed = record.get("seed")
            seat = record.get("seat")
            if type(opponent) is not str or type(seed) is not int or type(seat) is not int:
                return False
            observed_coordinates.append((opponent, seed, seat))
        if (
            len(observed_coordinates) != len(configured_matrix)
            or len(set(observed_coordinates)) != len(observed_coordinates)
            or set(observed_coordinates) != expected_coordinates
        ):
            return False
    return (
        report.get("schema_version") == 1
        and isinstance(expected_matrix, list)
    )


def _read_evaluation_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Evaluator did not write {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_fields(config: ColabConfig) -> dict[str, str]:
    return {
        "experiment_id": config.experiment_id,
        "feature_variant": config.feature_variant,
        "training_mode": config.training_mode,
    }


def _validate_identity_document(
    document: Mapping[str, Any], config: ColabConfig, *, path: Path,
    require_configuration: bool = False,
) -> None:
    """Validate identity already written by the document's atomic producer."""
    expected = _identity_fields(config)
    for field, value in expected.items():
        if field in document and document[field] != value:
            raise ValueError(
                f"{path} {field} does not match requested experiment identity"
            )
    configuration = document.get("configuration")
    if require_configuration and not isinstance(configuration, Mapping):
        raise ValueError(f"{path} configuration is missing experiment identity")
    manifest = document.get("manifest")
    for label, nested in (("configuration", configuration), ("manifest", manifest)):
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise ValueError(f"{path} {label} must be an object")
        for field, value in expected.items():
            if field in nested and nested[field] != value:
                raise ValueError(
                    f"{path} {label}.{field} does not match requested experiment identity"
                )
            if label == "configuration" and require_configuration and field not in nested:
                raise ValueError(f"{path} configuration.{field} is missing experiment identity")


def _invalidate_evaluation_report(path: Path) -> None:
    """Remove a prior report so it cannot satisfy a later evaluation gate."""
    path.unlink(missing_ok=True)


def _run_evaluation(
    config: ColabConfig, *, phase: str, command: Sequence[str], report_path: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run one evaluator and require a fresh successful report."""
    _invalidate_evaluation_report(report_path)
    started_ns = time.time_ns()
    completed = run_command(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{phase} evaluator failed with exit code {completed.returncode}"
        )
    if not report_path.exists() or report_path.stat().st_mtime_ns < started_ns:
        raise RuntimeError(f"{phase} evaluator did not write a fresh report: {report_path}")
    report = _read_evaluation_report(report_path)
    if not isinstance(report, Mapping):
        raise ValueError(f"{report_path} must contain an object")
    _validate_identity_document(report, config, path=report_path, require_configuration=True)
    return completed, report


def smoke_test_artifact(config: ColabConfig) -> None:
    """Verify the exported dependency-free artifact in the local simulator."""
    smoke = run_command(build_smoke_command(config), check=True, capture_output=True)
    print(smoke.stdout, end="")
    clean_stderr = re.sub(
        r"OpenSpiel exception: Unknown game 'python_ant_foraging'\. Available games are:\n.*?\nzerosum\n?",
        "", smoke.stderr, flags=re.DOTALL,
    )
    if clean_stderr:
        print(clean_stderr, file=sys.stderr, end="")


def plot_training_metrics(config: ColabConfig) -> Path | None:
    """Plot local telemetry when requested, tolerating an empty history."""
    import matplotlib.pyplot as plt
    from scripts.telemetry import load_metrics

    events = load_metrics(config.training_metrics_path)
    if not events:
        print(f"No {config.candidate_tag} training telemetry found at {config.training_metrics_path}")
        return None
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"{config.candidate_tag} training telemetry")
    for axis, event_name, metric_name, title in (
        (axes[0], "behavior_clone", "loss", "Behavior-cloning loss"),
        (axes[1], "ppo", "policy_loss", "PPO policy loss"),
    ):
        rows = [row for row in events if row.get("event") == event_name and metric_name in row]
        if rows:
            axis.plot(
                [row.get("step", index) for index, row in enumerate(rows)],
                [row[metric_name] for row in rows], marker="o",
            )
            axis.set_title(f"{config.candidate_tag} — {title}")
            axis.set_xlabel("Step")
            axis.set_ylabel(metric_name)
            axis.grid(True)
        else:
            axis.text(0.5, 0.5, f"No {event_name} {metric_name} data", ha="center", va="center")
            axis.set_axis_off()
    fig.tight_layout()
    output = config.plot_path or config.run_directory / f"{config.candidate_tag}-training-telemetry.png"
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output)
    finally:
        plt.close(fig)
    return output


def run_workflow(config: ColabConfig, *, dry_run: bool = False) -> WorkflowResult:
    """Execute or plan the complete resumable Colab workflow."""
    _validate_explicit_resume(config)
    commands = (
        tuple(build_collection_command(config)),
        tuple(build_evaluation_command(config, phase="development")),
        tuple(build_evaluation_command(config, phase="holdout")),
        tuple(build_smoke_command(config)),
    )
    if dry_run:
        return WorkflowResult(dry_run=True, commands=commands)

    if config.mount_drive:
        mount_drive(config)
    config.run_directory.mkdir(parents=True, exist_ok=True)
    run_command(commands[0], check=True)

    from scripts import train_policy

    training_contract = train_policy.build_training_contract(
        input_path=config.trajectory_path,
        steps=config.training_steps,
        batch_size=config.training_batch_size,
        seed=config.training_seed,
        ppo_steps=config.ppo_target_steps,
        device=config.device,
        checkpoint_interval=config.training_checkpoint_interval,
        prior_checkpoint=config.training_prior_checkpoint,
        offline_ppo_fallback=config.training_offline_ppo_fallback,
        experiment_id=config.experiment_id,
        feature_variant=config.feature_variant,
        training_mode=config.training_mode,
    )
    compatible = compatible_prior_checkpoints(config, training_contract=training_contract)
    opponent_pool = train_policy.OpponentPool(previous_checkpoints=compatible)
    resume_selection = select_resume_checkpoint(
        _checkpoint_candidates(config), training_contract=training_contract,
    )
    resume_checkpoint = resume_selection.path if resume_selection is not None else None
    saved_target = resume_selection.ppo_target_steps if resume_selection is not None else None
    allow_ppo_extension = saved_target is not None and saved_target < config.ppo_target_steps
    print("Stage:", config.candidate_tag, "target PPO steps:", config.ppo_target_steps)
    print("Resume checkpoint:", resume_checkpoint or "none; starting a new run")
    print("allow_ppo_extension:", allow_ppo_extension)

    telemetry = initialize_telemetry(config)
    try:
        training_result = train_candidate(
            config, training_contract=training_contract,
            resume_checkpoint=resume_checkpoint,
            allow_ppo_extension=allow_ppo_extension,
            opponent_pool=opponent_pool, telemetry=telemetry,
        )
        training_metadata: Mapping[str, Any] = (
            training_result if isinstance(training_result, Mapping) else {}
        )
        if telemetry is not None:
            telemetry(
                "training_complete",
                {
                    **_identity_fields(config),
                    "candidate": config.candidate_tag,
                    "checkpoint": str(config.stage_checkpoint_path),
                    "artifact": str(config.stage_artifact_path),
                    "behavior_clone_updates": training_metadata.get("behavior_clone_updates"),
                    "ppo_updates": training_metadata.get("ppo_updates"),
                    "ppo_steps": config.ppo_target_steps,
                    "configuration": {
                        "training_contract": training_contract.configuration,
                        "candidate_tag": config.candidate_tag,
                        "device": config.device,
                        "workers": config.workers,
                        "development_seeds": list(config.development_seeds),
                        "development_opponents": list(config.development_opponents),
                        "development_seats": list(config.development_seats),
                        "holdout_seeds": list(config.holdout_seeds),
                        "holdout_opponents": list(config.holdout_opponents),
                        "holdout_seats": list(config.holdout_seats),
                    },
                },
            )

        development_process, development_report = _run_evaluation(
            config, phase="development", command=commands[1],
            report_path=config.development_report_path,
        )
        if telemetry is not None:
            record_validation_report(
                telemetry, development_report, phase="development",
                checkpoint=config.stage_artifact_path, candidate_tag=config.candidate_tag,
                **_identity_fields(config),
            )
        development_decision = development_report.get("decision", {})
        development_complete = (
            evaluation_report_is_complete(
                development_report, identity=config.candidate_tag,
                seed_values=config.development_seeds,
                opponents=config.development_opponents,
                seats=config.development_seats,
                artifact_path=config.stage_artifact_path,
            )
            and development_decision.get("status") in {"promote", "discard"}
        )
        development_promoted = development_complete and development_decision.get("status") == "promote"
        print("Development evaluator exit:", development_process.returncode)
        print("Development status:", development_decision.get("status"))
        if not development_promoted:
            print("Development candidate discarded or incomplete; continue training, not promotion.")

        holdout_complete: bool | None = None
        if development_promoted:
            _invalidate_evaluation_report(config.holdout_report_path)
        smoke_test_artifact(config)
        if config.plot:
            plot_training_metrics(config)
        if development_promoted:
            holdout_process, holdout_report = _run_evaluation(
                config, phase="holdout", command=commands[2],
                report_path=config.holdout_report_path,
            )
            if telemetry is not None:
                record_validation_report(
                    telemetry, holdout_report, phase="holdout",
                    checkpoint=config.stage_artifact_path, candidate_tag=config.candidate_tag,
                    **_identity_fields(config),
                )
            holdout_decision = holdout_report.get("decision", {})
            holdout_complete = (
                evaluation_report_is_complete(
                    holdout_report, identity=config.candidate_tag,
                    seed_values=config.holdout_seeds,
                    opponents=config.holdout_opponents,
                    seats=config.holdout_seats,
                    artifact_path=config.stage_artifact_path,
                )
                and holdout_decision.get("status") in {"promote", "discard"}
            )
            if not holdout_complete:
                raise RuntimeError("Holdout report is incomplete or has no valid decision status")
            print("Holdout evaluator exit:", holdout_process.returncode)
            print("Holdout status:", holdout_decision.get("status"))
        else:
            print("Holdout evaluation skipped: development report is not complete/promote.")

        print("Candidate remains stage-scoped; no automatic promotion to policy.json or policy.pt was performed.")
        return WorkflowResult(
            dry_run=False, commands=commands, resume_checkpoint=resume_checkpoint,
            development_evaluation_promoted=development_promoted,
            holdout_evaluation_complete=holdout_complete,
            stage_artifact_path=config.stage_artifact_path,
        )
    finally:
        if telemetry is not None:
            telemetry.finish()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    result = run_workflow(config, dry_run=args.dry_run)
    print(json.dumps({
        "config": asdict(config),
        "dry_run": result.dry_run,
        "commands": [list(command) for command in result.commands],
        "resume_checkpoint": str(result.resume_checkpoint) if result.resume_checkpoint else None,
        "development_evaluation_promoted": result.development_evaluation_promoted,
        "holdout_evaluation_complete": result.holdout_evaluation_complete,
        "stage_artifact_path": str(result.stage_artifact_path) if result.stage_artifact_path else None,
    }, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
