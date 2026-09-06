"""Run the Kaggriculture agent against a local Kaggle environment opponent."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.candidates import candidate_policy
from kagriculture_agent.constants import CROPS
from main import agent


OPPONENTS = ("pass", "random", "starter", "current")


def _deterministic_random_agent(seed: int):
    """Return a seeded, random-style opponent with legal action shapes."""
    rng = random.Random(seed)

    def random_agent(obs: dict[str, Any]) -> dict[str, Any]:
        farms = obs.get("farms", [])
        player = obs.get("player", 0)
        private = obs.get("private", {}) or {}
        farm = farms[player] if isinstance(farms, list) and 0 <= player < len(farms) else None
        if not isinstance(farm, dict):
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farmer_ops = ["NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS"]
        market = []
        money = farm.get("money", 0)
        affordable = [crop for crop, data in CROPS.items() if data["seed"] <= money]
        if affordable and rng.random() < 0.1:
            market.append(["BUY_SEED", rng.choice(affordable), 1])

        seeds = private.get("seeds", {}) if isinstance(private, dict) else {}
        available_seeds = [crop for crop, quantity in seeds.items() if quantity > 0]
        farmer = ["PLANT", rng.choice(available_seeds)] if available_seeds and rng.random() < 0.3 else [rng.choice(farmer_ops)]
        hands = [[rng.choice(farmer_ops)] for _ in farm.get("hands", [])]
        return {"farmer": farmer, "hands": hands, "market": market}

    return random_agent


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _ordered_agents(opponent_agent: Any, candidate_player: int) -> list[Any]:
    if type(candidate_player) is not int or candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")
    return [agent, opponent_agent] if candidate_player == 0 else [opponent_agent, agent]


def run_episode(
    *,
    opponent: str,
    seed: int,
    steps: int = 720,
    replay_path: str | Path,
    candidate_player: int = 0,
    debug: bool = False,
) -> Any:
    """Run one local game and save its JSON replay."""
    if opponent not in OPPONENTS:
        raise ValueError(f"unsupported opponent: {opponent}")
    if steps < 1:
        raise ValueError("steps must be positive")
    if type(candidate_player) is not int or candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")

    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kaggle-environments is required to run local games; install the project dependencies first"
        ) from exc

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed},
        debug=debug,
    )
    if opponent == "random":
        opponent_agent = _deterministic_random_agent(seed)
    elif opponent == "current":
        opponent_agent = candidate_policy("current")
    else:
        opponent_agent = opponent
    env.run(_ordered_agents(opponent_agent, candidate_player))

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
    parser.add_argument("--steps", type=_positive_int, default=720, dest="steps")
    parser.add_argument("--replay", type=Path, default=None, dest="replay_path")
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0, dest="candidate_player")
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
        candidate_player=args.candidate_player,
        debug=args.debug,
    )
    result = env.toJSON()
    print(f"rewards: {result['rewards']}")
    print(f"statuses: {result['statuses']}")
    print(f"replay: {replay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
