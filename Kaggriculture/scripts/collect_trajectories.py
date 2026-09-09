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
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_LOCAL = Path(__file__).with_name("run_local.py")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION
from kagriculture_agent.rollouts import resolve_worker_count, run_rollouts
from kagriculture_agent.reward_shaping import classify_progress, should_bootstrap_truncate
from kagriculture_agent.trajectory import TRANSITION_SCHEMA_VERSION, Transition, transitions_from_replay
from scripts.training_identity import (
    DEFAULT_EXPERIMENT_ID,
    FEATURE_VARIANTS,
    TRAINING_MODES,
    validate_training_identity,
)

COLLECTOR_OPPONENTS = ("pass", "random", "starter", "current")
OPPONENTS = COLLECTOR_OPPONENTS
DEFAULT_GAME_TIMEOUT_SECONDS = 120.0


class IsolatedGameTimeoutError(TimeoutError, RuntimeError):
    """A subprocess timeout that remains compatible with the legacy API."""


class RolloutCollectionError(RuntimeError, ValueError):
    """Collection failed before publication; diagnostics cover every request."""

    def __init__(self, diagnostics: list[dict[str, Any]]) -> None:
        self.diagnostics = diagnostics
        failed = sum(entry["status"] != "success" for entry in diagnostics)
        detail = "; ".join(
            str(entry.get("error")) for entry in diagnostics
            if entry["status"] != "success" and entry.get("error")
        )
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{failed} rollout(s) failed; no trajectory was published{suffix}")


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


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number") from exc
    if number < 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
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
    candidate_artifact: str | Path | None = None,
    candidate_identity: str | None = None,
    opponent_artifact: str | Path | None = None,
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
    if candidate_artifact is not None:
        command.extend(["--candidate-artifact", str(candidate_artifact)])
    if candidate_identity is not None:
        command.extend(["--candidate-identity", candidate_identity])
    if opponent_artifact is not None:
        command.extend(["--opponent-artifact", str(opponent_artifact)])
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
        raise IsolatedGameTimeoutError(
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
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
) -> dict[str, Any]:
    validate_training_identity(experiment_id, feature_variant, training_mode)
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
        "experiment_id": experiment_id,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
    }


def _resolve_collection_transitions(
    transitions: Sequence[Transition], *, no_progress_window: int = 0,
    resolved_margin: float = 0.0,
) -> list[Transition]:
    """Annotate configured stalls/resolutions without changing genuine terminals."""
    if (
        isinstance(no_progress_window, bool)
        or not isinstance(no_progress_window, int)
        or no_progress_window < 0
    ):
        raise ValueError("no_progress_window must be a nonnegative integer")
    if (
        isinstance(resolved_margin, bool)
        or not isinstance(resolved_margin, (int, float))
        or not math.isfinite(float(resolved_margin))
        or resolved_margin < 0
    ):
        raise ValueError("resolved_margin must be a nonnegative finite number")
    resolved: list[Transition] = []
    no_progress_steps = 0
    for transition in transitions:
        if not isinstance(transition, Transition):
            raise ValueError("collector transitions must be validated Transition records")
        if transition.done:
            # A genuine engine terminal remains terminal, even if its replay
            # metadata contains a stall marker.
            resolved.append(transition)
            no_progress_steps = 0
            continue
        progress = classify_progress(transition.observation, transition.next_observation)
        if transition.no_progress_steps is not None:
            current_steps = transition.no_progress_steps
        elif progress == "no_progress":
            current_steps = no_progress_steps + 1
        else:
            current_steps = 0
        no_progress_steps = current_steps
        bootstrap_truncated = transition.bootstrap_truncated
        termination_reason = transition.termination_reason
        if bootstrap_truncated is None and should_bootstrap_truncate(
            current_steps, no_progress_window,
        ):
            bootstrap_truncated = True
            termination_reason = termination_reason or "no_progress"
        if bootstrap_truncated and termination_reason is None:
            termination_reason = "bootstrap_truncated"
        if not bootstrap_truncated and resolved_margin > 0:
            raw_margin = transition.observation.get("bank_differential")
            if raw_margin is None:
                raw_margin = transition.next_observation.get("bank_differential")
            try:
                decided = raw_margin is not None and abs(float(raw_margin)) >= resolved_margin
            except (TypeError, ValueError, OverflowError):
                raise ValueError("collector bank_differential is malformed") from None
            if decided:
                bootstrap_truncated = True
                termination_reason = termination_reason or "resolved"
        resolved.append(replace(
            transition,
            bootstrap_truncated=bootstrap_truncated,
            termination_reason=termination_reason,
            no_progress_steps=current_steps if no_progress_window else transition.no_progress_steps,
        ))
    return resolved


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
    records = []
    try:
        records = [json.loads(line) for line in trajectory_bytes.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("trajectory artifact pair contains invalid JSON") from exc
    if any(not isinstance(record, dict) for record in records):
        raise RuntimeError("trajectory artifact pair contains a non-object transition")
    reasons: dict[str, int] = {}
    truncated_count = 0
    max_no_progress = 0
    for record in records:
        reason = record.get("termination_reason")
        if reason is not None:
            if type(reason) is not str or not reason:
                raise RuntimeError("trajectory termination_reason is invalid")
            reasons[reason] = reasons.get(reason, 0) + 1
        truncated = record.get("bootstrap_truncated")
        if truncated is not None and type(truncated) is not bool:
            raise RuntimeError("trajectory bootstrap_truncated is invalid")
        if truncated:
            truncated_count += 1
        progress = record.get("no_progress_steps")
        if progress is not None and (type(progress) is not int or progress < 0):
            raise RuntimeError("trajectory no_progress_steps is invalid")
        if progress is not None:
            max_no_progress = max(max_no_progress, progress)
    if "termination_reasons" in manifest and manifest["termination_reasons"] != reasons:
        raise RuntimeError("trajectory manifest termination reasons do not match content")
    if "bootstrap_truncated_count" in manifest and manifest["bootstrap_truncated_count"] != truncated_count:
        raise RuntimeError("trajectory manifest truncation count does not match content")
    if "max_no_progress_steps" in manifest and manifest["max_no_progress_steps"] != max_no_progress:
        raise RuntimeError("trajectory manifest progress count does not match content")


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
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
    game_timeout: float = DEFAULT_GAME_TIMEOUT_SECONDS,
    candidate_artifact: str | Path | None = None,
    candidate_identity: str | None = None,
    opponent_artifact: str | Path | None = None,
    opponent_checkpoint_identity: str | None = None,
    workers: int | None = 1,
    no_progress_window: int = 0,
    resolved_margin: float = 0.0,
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
    validate_training_identity(experiment_id, feature_variant, training_mode)
    artifact_path = None
    if candidate_artifact is not None:
        artifact_path = Path(candidate_artifact).expanduser().resolve()
        if not artifact_path.is_file():
            raise ValueError(f"candidate artifact does not exist: {artifact_path}")
        if candidate_identity is None:
            candidate_identity = f"artifact:{hashlib.sha256(artifact_path.read_bytes()).hexdigest()}"
    if candidate_identity is not None and (
        type(candidate_identity) is not str or not candidate_identity
    ):
        raise ValueError("candidate_identity must be a non-empty string")
    if opponent_artifact is not None:
        opponent_artifact = Path(opponent_artifact).expanduser().resolve()
        if not opponent_artifact.is_file():
            raise ValueError(f"opponent artifact does not exist: {opponent_artifact}")
    if isinstance(game_timeout, bool) or not isinstance(game_timeout, (int, float)) \
            or not math.isfinite(float(game_timeout)) or float(game_timeout) <= 0:
        raise ValueError("game_timeout must be a positive finite number")
    _resolve_collection_transitions(
        [], no_progress_window=no_progress_window, resolved_margin=resolved_margin,
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    manifest = _manifest(
        seeds=normalized_seeds,
        opponents=normalized_opponents,
        seats=normalized_seats,
        steps=steps,
        source_policy_identity=source_policy_identity,
        experiment_id=experiment_id,
        feature_variant=feature_variant,
        training_mode=training_mode,
    )
    if no_progress_window:
        manifest["no_progress_window"] = no_progress_window
    if resolved_margin:
        manifest["resolved_margin"] = float(resolved_margin)
    if artifact_path is not None or candidate_identity is not None:
        manifest["candidate_artifact"] = str(artifact_path) if artifact_path is not None else None
        manifest["candidate_identity"] = candidate_identity
        manifest["candidate_artifact_sha256"] = (
            hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if artifact_path is not None else None
        )
    if opponent_artifact is not None:
        manifest["opponent_artifact"] = str(opponent_artifact)
        manifest["opponent_artifact_sha256"] = hashlib.sha256(
            opponent_artifact.read_bytes()
        ).hexdigest()
    if opponent_checkpoint_identity is not None:
        if type(opponent_checkpoint_identity) is not str or not opponent_checkpoint_identity:
            raise ValueError("opponent_checkpoint_identity must be a non-empty string")
        manifest["opponent_checkpoint_identity"] = opponent_checkpoint_identity
    manifest["workers"] = resolve_worker_count(workers)
    manifest["opponent_identities"] = {
        opponent: opponent for opponent in normalized_opponents
    }
    manifest["run_id"] = run_id
    manifest_path = destination.with_suffix(".manifest.json")
    trajectory_temp = None
    manifest_temp = None
    trajectory_temp_path: Path | None = None
    manifest_temp_path: Path | None = None
    trajectory_hash = hashlib.sha256()
    transition_count = 0
    termination_reasons: dict[str, int] = {}
    bootstrap_truncated_count = 0
    max_no_progress_steps = 0
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
                results = run_rollouts(
                    seeds=normalized_seeds, opponents=normalized_opponents,
                    seats=normalized_seats, steps=steps, workers=workers,
                    game_runner=_run_game_isolated,
                    replay_directory=replay_directory,
                    timeout=float(game_timeout), candidate_artifact=artifact_path,
                    candidate_identity=candidate_identity,
                    opponent_artifact=opponent_artifact,
                    opponent_checkpoint_identity=opponent_checkpoint_identity,
                )
                diagnostics = [
                    {
                        "request_key": result.request_key,
                        "status": result.status,
                        **result.diagnostic,
                    }
                    for result in results
                ]
                collection_failures = [
                    entry for entry in diagnostics if entry["status"] != "success"
                ]
                for result in results:
                    if result.status != "success":
                        continue
                    request = result.request
                    try:
                        transitions = transitions_from_replay(
                            result.replay, candidate_player=request.candidate_player,
                            requested_seed=request.seed,
                        )
                        transitions = _resolve_collection_transitions(
                            transitions,
                            no_progress_window=no_progress_window,
                            resolved_margin=resolved_margin,
                        )
                    except BaseException as exc:
                        parse_failure = {
                            "request_key": result.request_key,
                            "status": "failure",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        diagnostics.append(parse_failure)
                        collection_failures.append(parse_failure)
                        continue
                    for transition in transitions:
                        if transition.termination_reason is not None:
                            termination_reasons[transition.termination_reason] = (
                                termination_reasons.get(transition.termination_reason, 0) + 1
                            )
                        if transition.bootstrap_truncated:
                            bootstrap_truncated_count += 1
                        if transition.no_progress_steps is not None:
                            max_no_progress_steps = max(
                                max_no_progress_steps, transition.no_progress_steps,
                            )
                        serialized = transition.to_json() + "\n"
                        trajectory_temp.write(serialized)
                        trajectory_hash.update(serialized.encode("utf-8"))
                        transition_count += 1
                if collection_failures:
                    raise RolloutCollectionError(
                        sorted(diagnostics, key=lambda entry: entry["request_key"])
                    )
            manifest["transition_count"] = transition_count
            manifest["trajectory_sha256"] = trajectory_hash.hexdigest()
            if termination_reasons:
                manifest["termination_reasons"] = termination_reasons
            if bootstrap_truncated_count:
                manifest["bootstrap_truncated_count"] = bootstrap_truncated_count
            if max_no_progress_steps:
                manifest["max_no_progress_steps"] = max_no_progress_steps
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
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="production_v1")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="behavior_clone_then_ppo")
    parser.add_argument("--candidate-artifact", type=Path, default=None)
    parser.add_argument("--candidate-identity", default=None)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--no-progress-window", type=_nonnegative_int, default=0,
        help="bootstrap-truncate after this many consecutive no-progress transitions",
    )
    parser.add_argument(
        "--resolved-margin", type=_nonnegative_float, default=0.0,
        help="bootstrap-truncate configured resolved bank margins",
    )
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
        experiment_id=args.experiment_id,
        feature_variant=args.feature_variant,
        training_mode=args.training_mode,
        candidate_artifact=args.candidate_artifact,
        candidate_identity=args.candidate_identity,
        workers=args.workers,
        game_timeout=args.game_timeout,
        no_progress_window=args.no_progress_window,
        resolved_margin=args.resolved_margin,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    print(f"trajectories: {args.output}")
    print(f"manifest: {args.output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
