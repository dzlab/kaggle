"""Run the Kaggriculture agent against a local Kaggle environment opponent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle_environments import make

from main import agent


OPPONENTS = ("pass", "random", "starter")


def run_episode(
    *,
    opponent: str,
    seed: int,
    steps: int = 720,
    replay_path: str | Path,
    debug: bool = False,
) -> Any:
    """Run one local game and save its JSON replay."""
    if opponent not in OPPONENTS:
        raise ValueError(f"unsupported opponent: {opponent}")
    if steps < 1:
        raise ValueError("steps must be positive")

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed},
        debug=debug,
    )
    env.run([agent, opponent])

    replay = Path(replay_path)
    replay.parent.mkdir(parents=True, exist_ok=True)
    with replay.open("w", encoding="utf-8") as handle:
        json.dump(env.toJSON(), handle, indent=2 if debug else None, sort_keys=debug)
        handle.write("\n")
    return env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=OPPONENTS, default="pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=720, dest="steps")
    parser.add_argument("--replay", type=Path, default=None, dest="replay_path")
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    replay_path = args.replay_path or PROJECT_ROOT / "replays" / f"seed-{args.seed}-{args.opponent}.json"
    env = run_episode(
        opponent=args.opponent,
        seed=args.seed,
        steps=args.steps,
        replay_path=replay_path,
        debug=args.debug,
    )
    result = env.toJSON()
    print(f"rewards: {result['rewards']}")
    print(f"statuses: {result['statuses']}")
    print(f"replay: {replay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
