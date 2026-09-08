"""Colab/Drive-friendly configuration and entrypoint for candidate training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from pickle import UnpicklingError
from stat import S_ISREG
from typing import Any

from kagriculture_agent.checkpoints import CheckpointError, read_checkpoint
from kagriculture_agent.model import resolve_device
from scripts.train_policy import (
    TrainingContract,
    validate_training_checkpoint,
)


@dataclass(frozen=True)
class ColabConfig:
    run_directory: Path
    device: str
    workers: int
    development_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    resume: Path | None = None


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
) -> ColabConfig:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    development = tuple(int(seed) for seed in development_seeds)
    holdout = tuple(int(seed) for seed in holdout_seeds)
    if set(development) & set(holdout):
        raise ValueError("holdout seeds must not overlap development seeds")
    resolved = str(resolve_device(device))
    return ColabConfig(
        Path(run_directory).expanduser(), resolved, workers, development, holdout,
        Path(resume).expanduser() if resume is not None else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=Path("/content/drive/MyDrive/kagriculture-training"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--development-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=[100, 101])
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = build_config(
        run_directory=args.run_directory, device=args.device, workers=args.workers,
        development_seeds=tuple(args.development_seeds),
        holdout_seeds=tuple(args.holdout_seeds), resume=args.resume,
    )
    config.run_directory.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asdict(config), default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
