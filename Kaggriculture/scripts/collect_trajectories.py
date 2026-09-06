"""Collect evaluator-validated Kaggriculture transitions as deterministic JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_LOCAL = Path(__file__).with_name("run_local.py")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION
from kagriculture_agent.trajectory import TRANSITION_SCHEMA_VERSION, transitions_from_replay

COLLECTOR_OPPONENTS = ("pass", "random", "starter", "current")
OPPONENTS = COLLECTOR_OPPONENTS
DEFAULT_GAME_TIMEOUT_SECONDS = 120.0


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if number <= 0 or number != number or number == float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _strict_int_values(values: Sequence[Any], name: str, *, allowed: set[int] | None = None) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of integers")
    normalized = list(values)
    if not normalized or any(type(value) is not int for value in normalized):
        raise ValueError(f"{name} must contain only integers")
    if allowed is not None and any(value not in allowed for value in normalized):
        choices = ", ".join(str(value) for value in sorted(allowed))
        raise ValueError(f"{name} must contain only {{{choices}}}")
    return normalized


def _run_game_isolated(
    *, opponent: str, seed: int, steps: int, candidate_player: int, replay_path: Path,
    timeout: float = DEFAULT_GAME_TIMEOUT_SECONDS,
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
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"isolated game timed out after {timeout:g}s for opponent={opponent}, "
            f"seed={seed}, seat={candidate_player}"
        ) from exc
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


def _validate_pair(
    trajectory_path: Path, manifest_path: Path, run_id: str,
) -> None:
    """Validate that one manifest describes exactly one trajectory temp file."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trajectory_bytes = trajectory_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("trajectory artifact pair could not be read") from exc
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise RuntimeError("trajectory artifact pair has mismatched run id")
    if manifest.get("trajectory_sha256") != hashlib.sha256(trajectory_bytes).hexdigest():
        raise RuntimeError("trajectory artifact pair has mismatched content hash")
    if manifest.get("transition_count") != trajectory_bytes.count(b"\n"):
        raise RuntimeError("trajectory artifact pair has mismatched transition count")


def _publish_pair(
    trajectory_temp_path: Path, manifest_temp_path: Path,
    destination: Path, manifest_path: Path, run_id: str,
) -> None:
    """Publish a validated pair and roll back both paths if either replace fails."""
    output_backup = destination.parent / f".{destination.name}.{run_id}.bak"
    manifest_backup = manifest_path.parent / f".{manifest_path.name}.{run_id}.bak"
    output_backed_up = False
    manifest_backed_up = False
    output_replaced = False
    manifest_replaced = False
    try:
        if destination.exists():
            os.replace(destination, output_backup)
            output_backed_up = True
        if manifest_path.exists():
            os.replace(manifest_path, manifest_backup)
            manifest_backed_up = True
        os.replace(trajectory_temp_path, destination)
        output_replaced = True
        os.replace(manifest_temp_path, manifest_path)
        manifest_replaced = True
        _validate_pair(destination, manifest_path, run_id)
    except Exception:
        if output_replaced:
            destination.unlink(missing_ok=True)
        if manifest_replaced:
            manifest_path.unlink(missing_ok=True)
        if output_backed_up:
            os.replace(output_backup, destination)
        if manifest_backed_up:
            os.replace(manifest_backup, manifest_path)
        raise
    finally:
        output_backup.unlink(missing_ok=True)
        manifest_backup.unlink(missing_ok=True)


def collect(
    *, seeds: Sequence[int], opponents: Sequence[str], seats: Sequence[int], steps: int,
    output: str | Path, source_policy_identity: str = "current",
    game_timeout: float = DEFAULT_GAME_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Collect and write one validated transition per output JSONL line."""
    normalized_seeds = _strict_int_values(seeds, "seeds")
    if isinstance(opponents, (str, bytes)) or not isinstance(opponents, Sequence):
        raise ValueError("opponents must be a sequence of strings")
    normalized_opponents = list(opponents)
    if not normalized_opponents or any(type(opponent) is not str for opponent in normalized_opponents):
        raise ValueError("opponents must contain only strings")
    normalized_seats = _strict_int_values(seats, "seats", allowed={0, 1})
    if any(opponent not in COLLECTOR_OPPONENTS for opponent in normalized_opponents):
        raise ValueError(f"opponents must be drawn from {COLLECTOR_OPPONENTS}")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be at least 2 to produce a transition")
    if not isinstance(source_policy_identity, str) or not source_policy_identity:
        raise ValueError("source_policy_identity must be a non-empty string")
    if isinstance(game_timeout, bool) or not isinstance(game_timeout, (int, float)) \
            or not math.isfinite(float(game_timeout)) or float(game_timeout) <= 0:
        raise ValueError("game_timeout must be a positive finite number")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    manifest = _manifest(
        seeds=normalized_seeds,
        opponents=normalized_opponents,
        seats=normalized_seats,
        steps=steps,
        source_policy_identity=source_policy_identity,
    )
    manifest["run_id"] = run_id
    manifest_path = destination.with_suffix(".manifest.json")
    trajectory_temp = None
    manifest_temp = None
    trajectory_temp_path: Path | None = None
    manifest_temp_path: Path | None = None
    trajectory_hash = hashlib.sha256()
    transition_count = 0
    try:
        trajectory_temp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        )
        trajectory_temp_path = Path(trajectory_temp.name)
        manifest_temp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{manifest_path.name}.", suffix=".tmp", delete=False,
        )
        manifest_temp_path = Path(manifest_temp.name)
        with trajectory_temp, manifest_temp:
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
                                timeout=float(game_timeout),
                            )
                            transitions = transitions_from_replay(
                                replay, candidate_player=candidate_player, requested_seed=seed,
                            )
                            for transition in transitions:
                                serialized = transition.to_json() + "\n"
                                trajectory_temp.write(serialized)
                                trajectory_hash.update(serialized.encode("utf-8"))
                                transition_count += 1
            manifest["transition_count"] = transition_count
            manifest["trajectory_sha256"] = trajectory_hash.hexdigest()
            manifest_temp.write(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            trajectory_temp.flush()
            manifest_temp.flush()
            os.fsync(trajectory_temp.fileno())
            os.fsync(manifest_temp.fileno())
        _validate_pair(trajectory_temp_path, manifest_temp_path, run_id)
        _publish_pair(trajectory_temp_path, manifest_temp_path, destination, manifest_path, run_id)
        trajectory_temp_path = None
        manifest_temp_path = None
    finally:
        if trajectory_temp is not None:
            trajectory_temp.close()
        if manifest_temp is not None:
            manifest_temp.close()
        for temporary_path in (trajectory_temp_path, manifest_temp_path):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
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
    parser.add_argument(
        "--game-timeout", type=_positive_float, default=DEFAULT_GAME_TIMEOUT_SECONDS,
        help="maximum seconds allowed for each isolated game",
    )
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
        game_timeout=args.game_timeout,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    print(f"trajectories: {args.output}")
    print(f"manifest: {args.output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
