"""Collect evaluator-validated Kaggriculture transitions as deterministic JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_LOCAL = Path(__file__).with_name("run_local.py")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION
from kagriculture_agent.trajectory import TRANSITION_SCHEMA_VERSION, Transition, transitions_from_replay

COLLECTOR_OPPONENTS = ("pass", "random", "starter", "current")
OPPONENTS = COLLECTOR_OPPONENTS


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _run_game_isolated(
    *, opponent: str, seed: int, steps: int, candidate_player: int, replay_path: Path,
) -> dict[str, Any]:
    """Run one game in a fresh interpreter and return its JSON replay."""
    engine_opponent = "pass" if opponent == "current" else opponent
    command = [
        sys.executable,
        str(RUN_LOCAL),
        "--opponent", engine_opponent,
        "--seed", str(seed),
        "--steps", str(steps),
        "--seat", str(candidate_player),
        "--replay", str(replay_path),
    ]
    if opponent == "current":
        command.append("--current-opponent")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no subprocess output"
        raise RuntimeError(
            f"isolated game failed for opponent={opponent}, seed={seed}, "
            f"seat={candidate_player}: {detail[:1000]}"
        )
    try:
        return json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"isolated game did not produce a JSON replay for opponent={opponent}, "
            f"seed={seed}, seat={candidate_player}"
        ) from exc


def _manifest(
    *, seeds: Sequence[int], opponents: Sequence[str], seats: Sequence[int], steps: int,
    source_policy_identity: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "engine_version": str(ENGINE_VERSION),
        "steps": int(steps),
        "seeds": [int(seed) for seed in seeds],
        "seats": [int(seat) for seat in seats],
        "opponents": [str(opponent) for opponent in opponents],
        "source_policy_identity": source_policy_identity,
    }


def collect(
    *, seeds: Sequence[int], opponents: Sequence[str], seats: Sequence[int], steps: int,
    output: str | Path, source_policy_identity: str = "current",
) -> dict[str, Any]:
    """Collect and write one validated transition per output JSONL line."""
    normalized_seeds = [int(seed) for seed in seeds]
    normalized_opponents = [str(opponent) for opponent in opponents]
    normalized_seats = [int(seat) for seat in seats]
    if not normalized_seeds or any(type(seed) is not int for seed in normalized_seeds):
        raise ValueError("seeds must contain at least one integer")
    if not normalized_opponents or any(opponent not in COLLECTOR_OPPONENTS for opponent in normalized_opponents):
        raise ValueError(f"opponents must be drawn from {COLLECTOR_OPPONENTS}")
    if not normalized_seats or any(seat not in (0, 1) for seat in normalized_seats):
        raise ValueError("seats must contain 0 and/or 1")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be at least 2 to produce a transition")
    if not isinstance(source_policy_identity, str) or not source_policy_identity:
        raise ValueError("source_policy_identity must be a non-empty string")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(
        seeds=normalized_seeds,
        opponents=normalized_opponents,
        seats=normalized_seats,
        steps=steps,
        source_policy_identity=source_policy_identity,
    )
    lines: list[str] = []
    with tempfile.TemporaryDirectory(prefix="kagriculture-trajectory-") as temporary_directory:
        replay_directory = Path(temporary_directory)
        for seed in normalized_seeds:
            for opponent in normalized_opponents:
                for candidate_player in normalized_seats:
                    replay_path = replay_directory / f"seed-{seed}-{opponent}-seat-{candidate_player}.json"
                    replay = _run_game_isolated(
                        opponent=opponent,
                        seed=seed,
                        steps=steps,
                        candidate_player=candidate_player,
                        replay_path=replay_path,
                    )
                    transitions = transitions_from_replay(replay, candidate_player=candidate_player)
                    lines.extend(transition.to_json() for transition in transitions)

    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_positive_int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=_positive_int, default=720)
    parser.add_argument("--opponents", nargs="+", choices=COLLECTOR_OPPONENTS, default=list(COLLECTOR_OPPONENTS))
    parser.add_argument("--seats", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-policy-identity", default="current")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = range(args.start_seed, args.start_seed + args.seeds)
    manifest = collect(
        seeds=seeds,
        opponents=args.opponents,
        seats=args.seats,
        steps=args.steps,
        output=args.output,
        source_policy_identity=args.source_policy_identity,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    print(f"trajectories: {args.output}")
    print(f"manifest: {args.output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
