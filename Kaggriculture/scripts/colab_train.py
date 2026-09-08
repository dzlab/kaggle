"""Colab/Drive-friendly configuration and entrypoint for Orbit training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kagriculture_agent.model import resolve_device


@dataclass(frozen=True)
class ColabConfig:
    run_directory: Path
    device: str
    workers: int
    development_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    resume: Path | None = None


def build_config(
    *, run_directory: str | Path = "/content/drive/MyDrive/kagriculture-orbit",
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
    parser.add_argument("--run-directory", type=Path, default=Path("/content/drive/MyDrive/kagriculture-orbit"))
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
