"""Resumable Orbit-style candidate training and promotion controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


_DEVELOPMENT_OPPONENTS = ("pass", "random", "starter")
_DEFAULT_GAME_TIMEOUT_SECONDS = 120.0
_DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class OrbitConfig:
    run_directory: Path
    max_rounds: int = 10
    max_failures: int = 3
    development_seeds: tuple[int, ...] = (0, 1, 2, 3)
    device: str = "auto"
    opponents: tuple[str, ...] = ("pass", "random", "starter")
    seats: tuple[int, ...] = (0, 1)
    workers: int = 2
    episode_steps: int = 96
    ppo_rounds: int = 1
    checkpoint_window: int = 5
    max_hours: float | None = None

    def __post_init__(self) -> None:
        if self.max_rounds < 1 or self.max_failures < 0:
            raise ValueError("max_rounds must be positive and max_failures nonnegative")
        if not self.development_seeds:
            raise ValueError("development_seeds must not be empty")
        if any(type(seed) is not int for seed in self.development_seeds):
            raise ValueError("development_seeds must contain only integers")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("development_seeds must be unique")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not self.opponents or any(
            not isinstance(value, str) or not value for value in self.opponents
        ):
            raise ValueError("opponents must be a non-empty tuple of names")
        if any(opponent not in _DEVELOPMENT_OPPONENTS for opponent in self.opponents):
            raise ValueError(
                "opponents must be drawn from "
                f"{', '.join(_DEVELOPMENT_OPPONENTS)}"
            )
        if len(set(self.opponents)) != len(self.opponents):
            raise ValueError("opponents must be unique")
        if tuple(self.seats) != (0, 1):
            raise ValueError("seats must contain both candidate seat orders: (0, 1)")
        for name, value in (
            ("workers", self.workers), ("episode_steps", self.episode_steps),
            ("ppo_rounds", self.ppo_rounds), ("checkpoint_window", self.checkpoint_window),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.episode_steps < 2:
            raise ValueError("episode_steps must be at least 2 to produce a transition")
        if self.max_hours is not None and (
            isinstance(self.max_hours, bool) or not isinstance(self.max_hours, (int, float))
            or self.max_hours <= 0 or not float(self.max_hours) == float(self.max_hours)
            or float(self.max_hours) == float("inf")
        ):
            raise ValueError("max_hours must be a positive finite number")


class OrbitController:
    """Run rollout/train/evaluate/retain rounds with durable stage boundaries."""

    def __init__(
        self, config: OrbitConfig, *, rollout_fn: Callable[..., Any],
        train_fn: Callable[..., str | Path], evaluate_fn: Callable[..., dict[str, Any]],
        export_fn: Callable[..., str | Path] | None = None,
        candidate_path_fn: Callable[..., str | Path] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.rollout_fn = rollout_fn
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.export_fn = export_fn
        self.candidate_path_fn = candidate_path_fn
        self.clock = clock
        self.run_directory = config.run_directory
        self.state_path = self.run_directory / "orbit-state.json"
        self.run_directory.mkdir(parents=True, exist_ok=True)

    def _write_state(self, state: dict[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".orbit-state.", suffix=".tmp", dir=self.run_directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            directory_fd = os.open(
                self.run_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "round": 0, "failures": 0, "best": None, "history": [],
                "configuration": self._configuration(),
            }
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("history"), list):
            raise ValueError("orbit state is malformed")
        if state.get("configuration") not in (None, self._configuration()):
            raise ValueError("orbit state configuration does not match the requested run")
        state.setdefault("configuration", self._configuration())
        return state

    def _configuration(self) -> dict[str, Any]:
        return {
            "max_rounds": self.config.max_rounds,
            "max_failures": self.config.max_failures,
            "development_seeds": list(self.config.development_seeds),
            "device": self.config.device,
            "opponents": list(self.config.opponents),
            "seats": list(self.config.seats),
            "workers": self.config.workers,
            "episode_steps": self.config.episode_steps,
            "ppo_rounds": self.config.ppo_rounds,
            "checkpoint_window": self.config.checkpoint_window,
            "max_hours": self.config.max_hours,
        }

    def run(self) -> dict[str, Any]:
        state = self._load_state()
        started_at = state.setdefault("started_at", self.clock())
        self._write_state(state)
        while state["round"] < self.config.max_rounds:
            if self.config.max_hours is not None and self.clock() - float(started_at) >= self.config.max_hours * 3600:
                state["stop_reason"] = "max_hours"
                self._write_state(state)
                break
            round_index = int(state["round"])
            existing = state.get("current")
            if isinstance(existing, dict) and existing.get("round") == round_index:
                round_state = existing
            else:
                round_state = {"round": round_index, "stage": "rollout"}
                state["current"] = round_state
                self._write_state(state)
            try:
                candidate_checkpoint = Path(round_state["candidate"]) if (
                    round_state.get("stage") in {"evaluate", "promoting"}
                    and round_state.get("candidate")
                ) else None
                candidate_artifact = Path(round_state["candidate_artifact"]) if (
                    round_state.get("candidate_artifact")
                ) else None
                if candidate_checkpoint is None:
                    resuming_train = (
                        round_state.get("stage") == "train" and "rollout" in round_state
                    )
                    if resuming_train:
                        rollout = round_state["rollout"]
                    else:
                        rollout = self.rollout_fn(
                            round_index=round_index,
                            seeds=self.config.development_seeds,
                            run_directory=self.run_directory,
                        )
                    round_state["stage"] = "train"
                    round_state["rollout"] = rollout
                    if round_state.get("candidate"):
                        reserved_candidate = Path(round_state["candidate"])
                    elif self.candidate_path_fn is not None:
                        reserved_candidate = Path(self.candidate_path_fn(
                            round_index=round_index, run_directory=self.run_directory,
                        ))
                        round_state["candidate"] = str(reserved_candidate)
                    else:
                        reserved_candidate = None
                    self._write_state(state)
                    train_kwargs = {
                        "rollout": rollout, "round_index": round_index,
                        "run_directory": self.run_directory,
                    }
                    if (
                        resuming_train
                        and reserved_candidate is not None
                        and reserved_candidate.is_file()
                    ):
                        train_kwargs["resume_checkpoint"] = reserved_candidate
                    candidate_checkpoint = Path(self.train_fn(**train_kwargs))
                    round_state["candidate"] = str(candidate_checkpoint)
                    if self.export_fn is not None:
                        candidate_artifact = Path(self.export_fn(
                            candidate=candidate_checkpoint, round_index=round_index,
                            run_directory=self.run_directory,
                        ))
                        round_state["candidate_artifact"] = str(candidate_artifact)
                    round_state["stage"] = "evaluate"
                    self._write_state(state)
                elif candidate_artifact is None and self.export_fn is not None:
                    candidate_artifact = Path(self.export_fn(
                        candidate=candidate_checkpoint, round_index=round_index,
                        run_directory=self.run_directory,
                    ))
                    round_state["candidate_artifact"] = str(candidate_artifact)
                    self._write_state(state)
                evaluation_candidate = candidate_artifact or candidate_checkpoint
                if evaluation_candidate is None:
                    raise ValueError("round has no candidate checkpoint")
                if round_state.get("stage") == "promoting":
                    decision = round_state.get("decision")
                else:
                    decision = self.evaluate_fn(
                        candidate=evaluation_candidate, current=state.get("best"),
                        seeds=self.config.development_seeds, round_index=round_index,
                    )
                if not isinstance(decision, dict) or type(decision.get("promoted")) is not bool:
                    raise ValueError("development evaluator must return promoted boolean")
                round_state["decision"] = decision
                if decision["promoted"]:
                    round_state["stage"] = "promoting"
                    self._write_state(state)
                    best = self.run_directory / "best.pt"
                    temporary = self.run_directory / f".best.{round_index}.tmp"
                    shutil.copyfile(candidate_checkpoint, temporary)
                    os.replace(temporary, best)
                    round_state["stage"] = "retained"
                    state["best"] = str(best)
                else:
                    round_state["stage"] = "rejected"
                state["history"].append(round_state)
                state["round"] = round_index + 1
                state.pop("current", None)
                self._write_state(state)
            except Exception as exc:
                state["failures"] = int(state.get("failures", 0)) + 1
                round_state["stage"] = "failed"
                round_state["error"] = f"{type(exc).__name__}: {exc}"
                state["history"].append(round_state)
                state["round"] = round_index + 1
                state.pop("current", None)
                self._write_state(state)
                if state["failures"] >= self.config.max_failures:
                    raise
        return state


def _round_directory(config: OrbitConfig, round_index: int) -> Path:
    directory = config.run_directory / f"round-{int(round_index):04d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _existing_checkpoint_window(config: OrbitConfig) -> list[Path]:
    """Return the newest valid learned checkpoints for the PPO league."""
    candidates = sorted(
        config.run_directory.glob("round-*/candidate.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    best = config.run_directory / "best.pt"
    if best.is_file():
        candidates.append(best)
    result: list[Path] = []
    seen: set[Path] = set()
    for path in reversed(candidates):
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        result.append(resolved)
        if len(result) == config.checkpoint_window:
            break
    return list(reversed(result))


def _best_artifact(config: OrbitConfig, round_directory: Path) -> Path | None:
    """Export the retained checkpoint for use as the next round's candidate."""
    best_checkpoint = config.run_directory / "best.pt"
    if not best_checkpoint.is_file():
        return None
    from scripts.export_policy import export_checkpoint

    artifact = round_directory / "current.json"
    export_checkpoint(best_checkpoint, artifact)
    return artifact


def _production_rollout(config: OrbitConfig, *, round_index: int, seeds: Sequence[int],
                        run_directory: Path) -> dict[str, Any]:
    """Collect a complete, atomic training dataset with the real collector."""
    from scripts.collect_trajectories import collect

    round_directory = _round_directory(config, round_index)
    artifact = _best_artifact(config, round_directory)
    output = round_directory / "rollout.jsonl"
    manifest = collect(
        seeds=list(seeds),
        opponents=list(config.opponents),
        seats=list(config.seats),
        steps=config.episode_steps,
        output=output,
        source_policy_identity=("best" if artifact is not None else "main.agent"),
        workers=config.workers,
        game_timeout=_DEFAULT_GAME_TIMEOUT_SECONDS,
        candidate_artifact=artifact,
        candidate_identity=(f"best:{artifact.name}" if artifact is not None else None),
    )
    return {
        "input_path": str(output),
        "manifest_path": str(output.with_suffix(".manifest.json")),
        "manifest": manifest,
        "candidate_artifact": str(artifact) if artifact is not None else None,
    }


def _production_train(config: OrbitConfig, *, rollout: Mapping[str, Any],
                      round_index: int, run_directory: Path,
                      resume_checkpoint: str | Path | None = None) -> Path:
    """Train BC then fresh-rollout PPO, refreshing the exported policy each step."""
    input_path = Path(str(rollout.get("input_path", ""))).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"rollout adapter returned no readable input trajectory: {input_path}"
        )

    from scripts.export_policy import export_checkpoint
    from scripts.train_policy import (
        OpponentPool,
        build_training_contract,
        make_fresh_rollout_fn,
        train_behavior_clone,
        validate_training_checkpoint,
    )
    from kagriculture_agent.checkpoints import read_checkpoint

    round_directory = _round_directory(config, round_index)
    candidate_checkpoint = round_directory / "candidate.pt"
    candidate_artifact = round_directory / "candidate.json"
    prior_checkpoint = config.run_directory / "best.pt"
    prior = prior_checkpoint if prior_checkpoint.is_file() else None
    pool = OpponentPool(_existing_checkpoint_window(config))

    resumed_target = 0
    resumed_round = 0
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if resume_path != candidate_checkpoint.resolve() or not resume_path.is_file():
            raise ValueError("resume checkpoint must be the current round candidate.pt")
        payload = read_checkpoint(resume_path, map_location="cpu")
        contract = build_training_contract(
            input_path=input_path, steps=1, batch_size=_DEFAULT_BATCH_SIZE,
            seed=round_index, ppo_steps=config.ppo_rounds, device=config.device,
            checkpoint_interval=1, prior_checkpoint=prior,
        )
        saved_target = payload.get("configuration", {}).get("ppo_steps")
        if type(saved_target) is not int or saved_target < 0 or saved_target > config.ppo_rounds:
            raise ValueError("candidate checkpoint has an incompatible PPO target")
        validate_training_checkpoint(
            payload, contract=contract, allow_ppo_extension=saved_target < config.ppo_rounds,
        )
        resumed_target = saved_target
        resumed_round = payload["progress"]["round"]

    # Fresh rounds create BC first. Resumed rounds reuse the validated checkpoint
    # and continue from its saved PPO progress without replacing its bytes.
    if resume_checkpoint is None:
        train_behavior_clone(
            input_path=input_path,
            output_path=candidate_checkpoint,
            steps=1,
            batch_size=_DEFAULT_BATCH_SIZE,
            seed=round_index,
            ppo_steps=0,
            device=config.device,
            checkpoint_interval=1,
            prior_checkpoint=prior,
        )
    export_checkpoint(candidate_checkpoint, candidate_artifact)

    first_ppo_target = max(1, resumed_target, resumed_round + 1)
    for ppo_target in range(first_ppo_target, config.ppo_rounds + 1):
        rollout_fn = make_fresh_rollout_fn(
            run_directory=round_directory / "ppo-rollouts",
            candidate_artifact=candidate_artifact,
            seeds=config.development_seeds,
            steps=config.episode_steps,
            workers=config.workers,
            game_timeout=_DEFAULT_GAME_TIMEOUT_SECONDS,
            candidate_identity=f"round-{round_index}-ppo-{ppo_target}",
        )
        train_behavior_clone(
            input_path=input_path,
            output_path=candidate_checkpoint,
            steps=1,
            batch_size=_DEFAULT_BATCH_SIZE,
            seed=round_index,
            ppo_steps=ppo_target,
            device=config.device,
            checkpoint_interval=1,
            resume_checkpoint=candidate_checkpoint,
            allow_ppo_extension=True,
            prior_checkpoint=prior,
            opponent_pool=pool,
            rollout_fn=rollout_fn,
            candidate_artifact=candidate_artifact,
        )
        export_checkpoint(candidate_checkpoint, candidate_artifact)
    return candidate_checkpoint


def _production_export(config: OrbitConfig, *, candidate: str | Path,
                       round_index: int, run_directory: Path) -> Path:
    """Publish the final dependency-free artifact for a completed round."""
    from scripts.export_policy import export_checkpoint

    destination = _round_directory(config, round_index) / "candidate.json"
    export_checkpoint(Path(candidate), destination)
    return destination


def _production_evaluate(config: OrbitConfig, *, candidate: str | Path,
                         current: str | Path | None, seeds: Sequence[int],
                         round_index: int) -> dict[str, Any]:
    """Evaluate only the development matrix and translate its gate result."""
    del current  # The artifact evaluator's current policy is the fixed baseline.
    from scripts.evaluate_artifact import build_report, evaluate, write_report

    candidate_path = Path(candidate).expanduser().resolve()
    result = evaluate(
        artifact=candidate_path,
        identity=f"candidate-round-{round_index}",
        seeds=list(seeds),
        steps=config.episode_steps,
        opponents=list(config.opponents),
        seats=list(config.seats),
        workers=config.workers,
        min_valid_games=len(seeds) * len(config.opponents),
    )
    report_path = _round_directory(config, round_index) / "development-evaluation.json"
    write_report(report_path, build_report(result))
    decision = result.get("decision")
    if not isinstance(decision, Mapping) or decision.get("status") not in {"promote", "discard"}:
        raise ValueError("development evaluator returned no valid promotion decision")
    reasons = set(decision.get("reasons", ()))
    invalid_evaluation_reasons = {
        "framework_error",
        "incomplete_matrix",
        "insufficient_valid_games",
        "missing_seat_pairs",
        "duplicate_seat_pairs",
        "missing_expected_matrix_records",
        "duplicate_expected_matrix_records",
        "extra_expected_matrix_records",
    }
    invalid_reasons = sorted(reasons & invalid_evaluation_reasons)
    if invalid_reasons:
        raise RuntimeError(
            "development evaluation could not establish a valid comparison: "
            + ", ".join(invalid_reasons)
        )
    return {
        "promoted": decision["status"] == "promote",
        "status": decision["status"],
        "reasons": list(decision.get("reasons", [])),
        "report": str(report_path),
        "decision": dict(decision),
    }


def build_production_callbacks(config: OrbitConfig) -> dict[str, Callable[..., Any]]:
    """Build the real collector/trainer/exporter/evaluator adapters."""
    if not isinstance(config, OrbitConfig):
        raise TypeError("config must be an OrbitConfig")
    return {
        "rollout": lambda **kwargs: _production_rollout(config, **kwargs),
        "train": lambda **kwargs: _production_train(config, **kwargs),
        "export": lambda **kwargs: _production_export(config, **kwargs),
        "evaluate": lambda **kwargs: _production_evaluate(config, **kwargs),
    }


def build_production_controller(config: OrbitConfig) -> OrbitController:
    """Construct an OrbitController with all production defaults wired."""
    callbacks = build_production_callbacks(config)
    return OrbitController(
        config,
        rollout_fn=callbacks["rollout"],
        train_fn=callbacks["train"],
        export_fn=callbacks["export"],
        evaluate_fn=callbacks["evaluate"],
        candidate_path_fn=lambda *, round_index, **_kwargs: (
            _round_directory(config, round_index) / "candidate.pt"
        ),
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


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if number <= 0 or number != number or number == float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--max-rounds", type=_positive_int, default=10)
    parser.add_argument("--max-failures", type=_nonnegative_int, default=3)
    parser.add_argument(
        "--rollout-seeds", "--development-seeds", dest="development_seeds",
        nargs="+", type=int, default=[0, 1, 2, 3],
        help="explicit development seeds used for collection and promotion",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--opponents", nargs="+", choices=_DEVELOPMENT_OPPONENTS,
        default=list(_DEVELOPMENT_OPPONENTS),
    )
    parser.add_argument("--seats", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--episode-steps", type=_positive_int, default=96)
    parser.add_argument("--ppo-rounds", type=_positive_int, default=1)
    parser.add_argument("--checkpoint-window", type=_positive_int, default=5)
    parser.add_argument("--max-hours", type=_positive_float, default=None)
    parser.add_argument("--resume", action="store_true", help="resume an existing run directory")
    parser.add_argument("--dry-run", action="store_true", help="validate and print configuration only")
    parser.add_argument(
        "--validate-config", action="store_true",
        help="validate configuration without creating or running a controller",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> OrbitConfig:
    """Convert parsed CLI values into the validated controller configuration."""
    return OrbitConfig(
        run_directory=Path(args.run_directory).expanduser().resolve(),
        max_rounds=args.max_rounds,
        max_failures=args.max_failures,
        development_seeds=tuple(args.development_seeds),
        device=args.device,
        opponents=tuple(args.opponents),
        seats=tuple(args.seats),
        workers=args.workers,
        episode_steps=args.episode_steps,
        ppo_rounds=args.ppo_rounds,
        checkpoint_window=args.checkpoint_window,
        max_hours=args.max_hours,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    if args.dry_run or args.validate_config:
        print(json.dumps(asdict(config), default=str, sort_keys=True))
        return 0
    state_path = config.run_directory / "orbit-state.json"
    if state_path.exists() and not args.resume:
        print(
            f"run directory already contains {state_path}; pass --resume to continue",
            file=sys.stderr,
        )
        return 2
    try:
        result = build_production_controller(config).run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"orbit run failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
