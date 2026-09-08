"""Resumable Orbit-style candidate training and promotion controller."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


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
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not self.opponents or any(not isinstance(value, str) or not value for value in self.opponents):
            raise ValueError("opponents must be a non-empty tuple of names")
        if tuple(self.seats) != (0, 1):
            raise ValueError("seats must contain both candidate seat orders: (0, 1)")
        for name, value in (
            ("workers", self.workers), ("episode_steps", self.episode_steps),
            ("ppo_rounds", self.ppo_rounds), ("checkpoint_window", self.checkpoint_window),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.rollout_fn = rollout_fn
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.export_fn = export_fn
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
                candidate = Path(round_state["candidate"]) if (
                    round_state.get("stage") in {"evaluate", "promoting"}
                    and round_state.get("candidate")
                ) else None
                if candidate is None:
                    if round_state.get("stage") == "train" and "rollout" in round_state:
                        rollout = round_state["rollout"]
                    else:
                        rollout = self.rollout_fn(
                            round_index=round_index,
                            seeds=self.config.development_seeds,
                            run_directory=self.run_directory,
                        )
                    round_state["stage"] = "train"
                    round_state["rollout"] = rollout
                    self._write_state(state)
                    candidate = Path(self.train_fn(
                        rollout=rollout, round_index=round_index,
                        run_directory=self.run_directory,
                    ))
                    round_state["candidate"] = str(candidate)
                    if self.export_fn is not None:
                        round_state["candidate_artifact"] = str(self.export_fn(
                            candidate=candidate, round_index=round_index,
                            run_directory=self.run_directory,
                        ))
                    round_state["stage"] = "evaluate"
                    self._write_state(state)
                if round_state.get("stage") == "promoting":
                    decision = round_state.get("decision")
                else:
                    decision = self.evaluate_fn(
                        candidate=candidate, current=state.get("best"),
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
                    shutil.copyfile(candidate, temporary)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--development-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--opponents", nargs="+", default=["pass", "random", "starter"])
    parser.add_argument("--seats", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=96)
    parser.add_argument("--ppo-rounds", type=int, default=1)
    parser.add_argument("--checkpoint-window", type=int, default=5)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate and print configuration only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = OrbitConfig(
        args.run_directory, args.max_rounds, args.max_failures, tuple(args.development_seeds),
        args.device, tuple(args.opponents), tuple(args.seats), args.workers,
        args.episode_steps, args.ppo_rounds, args.checkpoint_window, args.max_hours,
    )
    if not args.dry_run:
        raise SystemExit("train_orbit.py requires injected rollout/train/evaluate callbacks; use the Python API or --dry-run")
    print(json.dumps(asdict(config), default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
