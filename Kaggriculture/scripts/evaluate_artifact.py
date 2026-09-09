"""Evaluate a dependency-free learned artifact against the current policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagriculture_agent.learned_policy import load_exported_policy  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    _framework_error_record,
    _matrix_completeness,
    paired_seed_summary,
    promotion_decision,
    replay_record,
)
from scripts.run_local import OPPONENTS, run_episode  # noqa: E402


CURRENT_CANDIDATE = "current"
MAX_WORKERS = 8
DEFAULT_SEEDS = 30
DEFAULT_STEPS = 720
DEFAULT_WORKERS = 2
DEFAULT_MIN_VALID_GAMES = 20
DEFAULT_EVALUATION_TIMEOUT = 600.0
QUICK_EVALUATION_TIMEOUT = 30.0
DEFAULT_OPPONENTS = ("pass", "random", "starter")
DEFAULT_SEATS = (0, 1)
DEFAULT_OUTPUT = Path("reports/artifact-evaluation.json")
DEFAULT_EXPERIMENT_ID = "orbit-policy-v1"
FEATURE_VARIANTS = ("production_v1", "experimental_context_v1")
TRAINING_MODES = (
    "behavior_clone_then_ppo",
    "pure_ppo",
    "reduced_behavior_clone_then_ppo",
)


def _validate_training_identity(
    experiment_id: Any, feature_variant: Any, training_mode: Any,
) -> None:
    if type(experiment_id) is not str or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError(f"feature_variant must be one of: {', '.join(FEATURE_VARIANTS)}")
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"training_mode must be one of: {', '.join(TRAINING_MODES)}")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _bounded_workers(value: str) -> int:
    number = _positive_int(value)
    if number > MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_WORKERS}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def validate_seats(values: Sequence[int]) -> list[int]:
    """Validate the requested, non-empty set of candidate seats."""
    seats = list(values)
    if not seats:
        raise ValueError("at least one seat must be selected")
    if len(set(seats)) != len(seats):
        raise ValueError("seats must be unique")
    invalid = [seat for seat in seats if type(seat) is not int or seat not in (0, 1)]
    if invalid:
        raise ValueError("seats must be 0 or 1")
    return seats


def validate_opponents(values: Sequence[str]) -> list[str]:
    opponents = list(values)
    if not opponents:
        raise ValueError("at least one opponent must be selected")
    if len(set(opponents)) != len(opponents):
        raise ValueError("opponents must be unique")
    invalid = [opponent for opponent in opponents if opponent not in OPPONENTS]
    if invalid:
        raise ValueError(f"unsupported opponent(s): {', '.join(map(str, invalid))}")
    return opponents


def seed_values(seeds: int | Sequence[int], start_seed: int = 0) -> list[int]:
    """Return unique integer seeds from either a count or explicit values."""
    if type(seeds) is int:
        if seeds < 1:
            raise ValueError("seeds must be a positive integer")
        if type(start_seed) is not int:
            raise ValueError("start_seed must be an integer")
        return list(range(start_seed, start_seed + seeds))
    values = list(seeds)
    if not values or any(type(seed) is not int for seed in values):
        raise ValueError("seeds must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique")
    return values


def build_matrix(*, opponents: Sequence[str], seeds: Sequence[int], seats: Sequence[int]) -> list[dict[str, Any]]:
    """Build the reproducible opponent/seed/seat Cartesian product."""
    normalized_opponents = validate_opponents(opponents)
    normalized_seeds = seed_values(seeds)
    normalized_seats = validate_seats(seats)
    return [
        {"opponent": opponent, "seed": seed, "seat": seat}
        for opponent in normalized_opponents
        for seed in normalized_seeds
        for seat in normalized_seats
    ]


def validate_artifact(path: str | Path, identity: str = "learned_artifact") -> dict[str, str]:
    """Validate a dependency-free artifact and return stable report metadata."""
    if type(identity) is not str or not identity.strip():
        raise ValueError("identity must be a non-empty string")
    if identity == CURRENT_CANDIDATE:
        raise ValueError("identity is reserved: current")
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise ValueError(f"artifact does not exist or is not a file: {artifact_path}")
    artifact_bytes = artifact_path.read_bytes()
    try:
        load_exported_policy(artifact_path)
    except Exception as exc:
        raise ValueError(
            f"artifact is not valid: {type(exc).__name__}: {exc}"
        ) from exc
    return {
        "path": str(artifact_path),
        "name": artifact_path.name,
        "identity": identity,
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }


def _snapshot_artifact(artifact: Mapping[str, str]) -> tuple[Path, dict[str, str]]:
    """Copy validated bytes to a read-only path shared by all game workers."""
    snapshot_directory = Path(tempfile.mkdtemp(prefix="kaggriculture-artifact-snapshot-"))
    try:
        source_bytes = Path(artifact["path"]).read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if source_sha256 != artifact["sha256"]:
            raise ValueError("artifact changed after validation")
        snapshot_path = snapshot_directory / artifact["name"]
        snapshot_path.write_bytes(source_bytes)
        snapshot_path.chmod(0o444)
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
        if snapshot_sha256 != artifact["sha256"]:
            raise ValueError("artifact snapshot hash does not match validated bytes")
        return snapshot_directory, {**dict(artifact), "path": str(snapshot_path), "sha256": snapshot_sha256}
    except Exception:
        shutil.rmtree(snapshot_directory, ignore_errors=True)
        raise


def _error_record(request: Mapping[str, Any], error: str) -> dict[str, Any]:
    candidate = str(request["candidate"])
    record = _framework_error_record(
        variant=candidate,
        opponent=str(request["opponent"]),
        seed=int(request["seed"]),
        seat=int(request["seat"]),
        error=error,
    )
    return {**record, "candidate": candidate, "variant": candidate}


def _run_game_direct(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run and normalize one isolated game; never let a game error escape."""
    try:
        with tempfile.TemporaryDirectory(prefix="kaggriculture-artifact-") as directory:
            replay_path = Path(directory) / "replay.json"
            run_kwargs: dict[str, Any] = {
                "opponent": request["opponent"],
                "seed": request["seed"],
                "steps": request["steps"],
                "replay_path": replay_path,
                "candidate_player": request["seat"],
            }
            if request["candidate"] == CURRENT_CANDIDATE:
                run_kwargs["candidate_identity"] = CURRENT_CANDIDATE
            else:
                run_kwargs["candidate_artifact"] = request["artifact_path"]
            environment = run_episode(**run_kwargs)
            record = replay_record(
                environment.toJSON(),
                variant=request["candidate"],
                opponent=request["opponent"],
                seed=request["seed"],
                seat=request["seat"],
            )
        if not isinstance(record, Mapping):
            raise ValueError("replay normalization did not return an object")
        return {**dict(record), "candidate": request["candidate"], "variant": request["candidate"]}
    except Exception as exc:
        return _error_record(
            request,
            f"{type(exc).__name__}: {exc}",
        )


def _run_game_subprocess(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one game in a killable interpreter with a per-game timeout."""
    timeout = float(request.get("game_timeout", DEFAULT_EVALUATION_TIMEOUT))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(dict(request)),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            if details:
                details = f": {details[:500]}"
            raise RuntimeError(
                f"game subprocess exited with status {completed.returncode}{details}"
            )
        record = json.loads(completed.stdout)
        if not isinstance(record, Mapping):
            raise ValueError("game subprocess did not return an object")
        return {**dict(record), "candidate": request["candidate"], "variant": request["candidate"]}
    except subprocess.TimeoutExpired:
        return _error_record(request, f"game timeout after {timeout:g} seconds")
    except Exception as exc:
        return _error_record(request, f"{type(exc).__name__}: {exc}")


def _run_game(request: Mapping[str, Any]) -> dict[str, Any]:
    """Pool worker entry point; the game itself runs in a killable subprocess."""
    return _run_game_subprocess(request)


def _worker_main() -> int:
    request = json.load(sys.stdin)
    record = _run_game_direct(request)
    json.dump(record, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


def _run_request(runner: Callable[[Mapping[str, Any]], Mapping[str, Any]], request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = runner(dict(request))
        if not isinstance(result, Mapping):
            raise ValueError("game runner did not return an object")
        return {**dict(result), "candidate": request["candidate"], "variant": request["candidate"]}
    except Exception as exc:
        return _error_record(request, f"{type(exc).__name__}: {exc}")


def _is_complete(records: Sequence[Mapping[str, Any]], expected: Sequence[tuple[str, int, int]]) -> bool:
    completeness = _matrix_completeness(records, expected)
    return completeness is not None and not any(
        completeness[key]
        for key in ("missing", "duplicate", "extra", "invalid_records")
    )


def _run_in_pool(
    requests: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    evaluation_timeout: float,
) -> list[dict[str, Any]]:
    """Run requests with a hard parent-side deadline and non-waiting teardown."""
    executor = None
    futures = []
    try:
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
            for request in requests:
                futures.append(executor.submit(_run_game, request))
        except Exception as exc:
            return [_error_record(request, f"worker submission failure: {type(exc).__name__}: {exc}")
                    for request in requests]

        try:
            done, not_done = wait(futures, timeout=evaluation_timeout)
        except Exception as exc:
            return [_error_record(request, f"worker wait failure: {type(exc).__name__}: {exc}")
                    for request in requests]

        records = []
        for request, future in zip(requests, futures):
            if future in not_done:
                future.cancel()
                records.append(_error_record(
                    request,
                    f"evaluation timeout after {evaluation_timeout:g} seconds",
                ))
                continue
            try:
                record = future.result()
                if not isinstance(record, Mapping):
                    raise ValueError("worker result was not an object")
                records.append(dict(record))
            except Exception as exc:
                records.append(_error_record(
                    request, f"worker failure: {type(exc).__name__}: {exc}",
                ))
        return records
    finally:
        for future in futures:
            future.cancel()
        if executor is not None:
            # Do not use a context manager: its implicit shutdown(wait=True)
            # could keep the evaluator blocked behind a hung game.
            executor.shutdown(wait=False, cancel_futures=True)


def evaluate(
    *,
    artifact: str | Path,
    identity: str = "learned_artifact",
    seeds: int | Sequence[int] = DEFAULT_SEEDS,
    start_seed: int = 0,
    steps: int = DEFAULT_STEPS,
    opponents: Sequence[str] = DEFAULT_OPPONENTS,
    seats: Sequence[int] = DEFAULT_SEATS,
    workers: int = DEFAULT_WORKERS,
    min_valid_games: int = DEFAULT_MIN_VALID_GAMES,
    evaluation_timeout: float = DEFAULT_EVALUATION_TIMEOUT,
    game_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    quick: bool = False,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    feature_variant: str = "production_v1",
    training_mode: str = "behavior_clone_then_ppo",
) -> dict[str, Any]:
    """Evaluate current and artifact candidates on one identical matrix."""
    if quick:
        if seeds == DEFAULT_SEEDS:
            seeds = 2
        if steps == DEFAULT_STEPS:
            steps = 96
        if evaluation_timeout == DEFAULT_EVALUATION_TIMEOUT:
            evaluation_timeout = QUICK_EVALUATION_TIMEOUT
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(workers) is not int or workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if type(min_valid_games) is not int or min_valid_games < 1:
        raise ValueError("min_valid_games must be a positive integer")
    if isinstance(evaluation_timeout, bool) or not isinstance(evaluation_timeout, Real) \
            or not math.isfinite(float(evaluation_timeout)) or float(evaluation_timeout) <= 0:
        raise ValueError("evaluation_timeout must be a positive finite number")
    _validate_training_identity(experiment_id, feature_variant, training_mode)
    artifact_info = validate_artifact(artifact, identity)
    snapshot_directory, snapshot_info = _snapshot_artifact(artifact_info)
    try:
        normalized_opponents = validate_opponents(opponents)
        normalized_seats = validate_seats(seats)
        normalized_seeds = seed_values(seeds, start_seed)
        matrix = build_matrix(
            opponents=normalized_opponents,
            seeds=normalized_seeds,
            seats=normalized_seats,
        )
        expected = [
            (item["opponent"], item["seed"], item["seat"])
            for item in matrix
        ]
        requests = [
            {
                **item,
                "candidate": candidate,
                "steps": steps,
                "artifact_path": snapshot_info["path"],
                "game_timeout": float(evaluation_timeout),
            }
            for candidate in (CURRENT_CANDIDATE, snapshot_info["identity"])
            for item in matrix
        ]
        if game_runner is not None:
            records = [_run_request(game_runner, request) for request in requests]
        else:
            records = _run_in_pool(
                requests, workers=workers, evaluation_timeout=float(evaluation_timeout),
            )

        current_records = [record for record in records if record.get("candidate") == CURRENT_CANDIDATE]
        artifact_records = [record for record in records if record.get("candidate") == snapshot_info["identity"]]
        current_complete = _is_complete(current_records, expected)
        artifact_complete = _is_complete(artifact_records, expected)
        decision = promotion_decision(
            artifact_records,
            current_records,
            min_valid_games=min_valid_games,
            expected_matrix=expected,
        )
        decision = {**decision, "reasons": list(decision.get("reasons", []))}
        if any(record.get("framework_error") for record in records):
            decision["status"] = "discard"
            if "framework_error" not in decision["reasons"]:
                decision["reasons"].append("framework_error")
        if not current_complete or not artifact_complete:
            decision["status"] = "discard"
            if "incomplete_matrix" not in decision["reasons"]:
                decision["reasons"].append("incomplete_matrix")
        configuration = {
            "experiment_id": experiment_id,
            "feature_variant": feature_variant,
            "training_mode": training_mode,
            "seeds": len(normalized_seeds),
            "start_seed": start_seed,
            "seed_values": normalized_seeds,
            "steps": steps,
            "opponents": normalized_opponents,
            "seats": normalized_seats,
            "candidates": [CURRENT_CANDIDATE, snapshot_info["identity"]],
            "workers": workers,
            "min_valid_games": min_valid_games,
            "evaluation_timeout": float(evaluation_timeout),
            "quick": quick,
        }
        return {
            "configuration": configuration,
            "artifact": snapshot_info,
            "expected_matrix": [list(coordinate) for coordinate in expected],
            "records": records,
            "summaries": {
                CURRENT_CANDIDATE: paired_seed_summary(current_records),
                snapshot_info["identity"]: paired_seed_summary(artifact_records),
            },
            "matrix_completeness": {
                CURRENT_CANDIDATE: _matrix_completeness(current_records, expected),
                snapshot_info["identity"]: _matrix_completeness(artifact_records, expected),
            },
            "decision": decision,
        }
    finally:
        shutil.rmtree(snapshot_directory, ignore_errors=True)


def build_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stable JSON report, omitting the internal absolute artifact path."""
    artifact = result["artifact"]
    artifact_report = {
        key: artifact[key]
        for key in ("name", "identity", "sha256")
    }
    identity = artifact["identity"]
    records = result["records"]
    return {
        "schema_version": 1,
        "configuration": dict(result["configuration"]),
        "artifact": artifact_report,
        "expected_matrix": result["expected_matrix"],
        "records": {
            CURRENT_CANDIDATE: [record for record in records if record.get("candidate") == CURRENT_CANDIDATE],
            identity: [record for record in records if record.get("candidate") == identity],
        },
        "summaries": dict(result["summaries"]),
        "matrix_completeness": dict(result["matrix_completeness"]),
        "decision": dict(result["decision"]),
    }


def _configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_id": args.experiment_id,
        "feature_variant": args.feature_variant,
        "training_mode": args.training_mode,
        "seeds": args.seeds,
        "start_seed": args.start_seed,
        "seed_values": seed_values(args.seeds, args.start_seed),
        "steps": args.steps,
        "opponents": list(args.opponents),
        "seats": list(args.seats),
        "candidates": [CURRENT_CANDIDATE, args.identity],
        "workers": args.workers,
        "min_valid_games": args.min_valid_games,
        "evaluation_timeout": args.evaluation_timeout,
        "quick": args.quick,
    }


def _best_effort_artifact_metadata(path: str | Path, identity: str) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    metadata: dict[str, Any] = {
        "name": artifact_path.name,
        "identity": identity,
        "sha256": None,
    }
    try:
        if artifact_path.is_file():
            metadata["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    except OSError:
        pass
    return metadata


def _failure_report(args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    try:
        matrix = build_matrix(
            opponents=args.opponents,
            seeds=seed_values(args.seeds, args.start_seed),
            seats=args.seats,
        )
        expected_matrix = [
            [item["opponent"], item["seed"], item["seat"]] for item in matrix
        ]
    except Exception:
        expected_matrix = []
    reason = "artifact_invalid" if "artifact" in str(error).lower() else "evaluation_error"
    identity = args.identity
    empty_summary = paired_seed_summary([])
    expected_coordinates = [tuple(coordinate) for coordinate in expected_matrix]
    completeness = _matrix_completeness([], expected_coordinates) if expected_coordinates else None
    return {
        "schema_version": 1,
        "configuration": _configuration_from_args(args),
        "artifact": _best_effort_artifact_metadata(args.artifact, identity),
        "expected_matrix": expected_matrix,
        "records": {CURRENT_CANDIDATE: [], identity: []},
        "summaries": {CURRENT_CANDIDATE: empty_summary, identity: empty_summary},
        "matrix_completeness": {CURRENT_CANDIDATE: completeness, identity: completeness},
        "decision": {
            "status": "discard",
            "reasons": [reason],
            "error": str(error)[:1000],
        },
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=report_path.parent,
            prefix=f".{report_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, report_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--identity", default="learned_artifact")
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="production_v1")
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="behavior_clone_then_ppo")
    parser.add_argument("--seeds", type=_positive_int, default=DEFAULT_SEEDS)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--seats", nargs="+", type=int, choices=(0, 1), default=list(DEFAULT_SEATS))
    parser.add_argument("--workers", type=_bounded_workers, default=DEFAULT_WORKERS)
    parser.add_argument("--min-valid-games", type=_positive_int, default=DEFAULT_MIN_VALID_GAMES)
    parser.add_argument(
        "--evaluation-timeout", "--timeout", dest="evaluation_timeout",
        type=_positive_float, default=DEFAULT_EVALUATION_TIMEOUT,
        help="maximum seconds to wait for the complete evaluation matrix",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.seats = validate_seats(args.seats)
        args.opponents = validate_opponents(args.opponents)
    except ValueError as exc:
        parser.error(str(exc))
    if type(args.identity) is not str or not args.identity.strip():
        parser.error("identity must be a non-empty string")
    if args.identity == CURRENT_CANDIDATE:
        parser.error("identity is reserved: current")
    try:
        _validate_training_identity(
            args.experiment_id, args.feature_variant, args.training_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.quick:
        if args.seeds == DEFAULT_SEEDS:
            args.seeds = 2
        if args.steps == DEFAULT_STEPS:
            args.steps = 96
        if args.evaluation_timeout == DEFAULT_EVALUATION_TIMEOUT:
            args.evaluation_timeout = QUICK_EVALUATION_TIMEOUT
    return args


def _option_value(argv: Sequence[str], option: str, default: Any = None) -> Any:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            return argv[index + 1]
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
    return default


def _explicit_output(argv: Sequence[str]) -> Path | None:
    value = _option_value(argv, "--output")
    return Path(value) if value is not None else None


def _cli_failure_report(argv: Sequence[str], error: BaseException) -> dict[str, Any]:
    artifact_value = _option_value(argv, "--artifact", "artifact.json")
    identity = _option_value(argv, "--identity", "learned_artifact")
    if type(identity) is not str or not identity.strip() or identity == CURRENT_CANDIDATE:
        identity = "learned_artifact"
    empty_summary = paired_seed_summary([])
    return {
        "schema_version": 1,
        "configuration": {"valid": False, "argv": list(argv)},
        "artifact": _best_effort_artifact_metadata(artifact_value, identity),
        "expected_matrix": [],
        "records": {CURRENT_CANDIDATE: [], identity: []},
        "summaries": {CURRENT_CANDIDATE: empty_summary, identity: empty_summary},
        "matrix_completeness": {CURRENT_CANDIDATE: None, identity: None},
        "decision": {
            "status": "discard",
            "reasons": ["cli_invalid"],
            "error": str(error)[:1000] or "invalid command-line arguments",
        },
    }


def _is_valid_comparison(result: Mapping[str, Any]) -> bool:
    records = result.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return False
    if any(not isinstance(record, Mapping) for record in records):
        return False
    if any(record.get("framework_error") for record in records):
        return False
    if any(
        record.get("outcome") not in {"win", "loss", "tie"}
        or record.get("framework_error") is not False
        or not isinstance(record.get("bank_differential"), Real)
        or isinstance(record.get("bank_differential"), bool)
        or not math.isfinite(float(record["bank_differential"]))
        for record in records
    ):
        return False
    decision = result.get("decision", {})
    if not isinstance(decision, Mapping):
        return False
    if decision.get("status") != "discard":
        return False
    reasons = decision.get("reasons", ())
    if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
        return False
    if {"framework_error", "timeout", "evaluation_timeout"} & set(reasons):
        return False
    completeness = result.get("matrix_completeness")
    if not isinstance(completeness, Mapping):
        return False
    artifact = result.get("artifact")
    if not isinstance(artifact, Mapping):
        return False
    identity = artifact.get("identity")
    if not isinstance(identity, str) or identity == CURRENT_CANDIDATE:
        return False
    for candidate in (CURRENT_CANDIDATE, identity):
        candidate_completeness = completeness.get(candidate)
        if not isinstance(candidate_completeness, Mapping):
            return False
        if any(candidate_completeness.get(key) for key in ("missing", "duplicate", "extra", "invalid_records")):
            return False
    expected_matrix = result.get("expected_matrix")
    if not isinstance(expected_matrix, Sequence) or len(records) != 2 * len(expected_matrix):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parse_args(raw_argv)
    except SystemExit as exc:
        output_arg = _explicit_output(raw_argv)
        if exc.code != 0 and output_arg is not None:
            output = output_arg if output_arg.is_absolute() else PROJECT_ROOT / output_arg
            try:
                write_report(output, _cli_failure_report(raw_argv, exc))
                print(f"artifact evaluation: discard ({output})", file=sys.stderr)
            except Exception as report_exc:
                print(
                    f"artifact evaluation: discard; failure report unavailable: "
                    f"{type(report_exc).__name__}: {report_exc}",
                    file=sys.stderr,
                )
        return int(exc.code) if isinstance(exc.code, int) else 2
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    try:
        result = evaluate(
            artifact=args.artifact,
            identity=args.identity,
            seeds=args.seeds,
            start_seed=args.start_seed,
            steps=args.steps,
            opponents=args.opponents,
            seats=args.seats,
            workers=args.workers,
            min_valid_games=args.min_valid_games,
            evaluation_timeout=args.evaluation_timeout,
            quick=args.quick,
            experiment_id=args.experiment_id,
            feature_variant=args.feature_variant,
            training_mode=args.training_mode,
        )
        report = build_report(result)
        write_report(output, report)
    except Exception as exc:
        try:
            write_report(output, _failure_report(args, exc))
            print(f"artifact evaluation: discard ({output})", file=sys.stderr)
        except Exception as report_exc:
            print(
                f"artifact evaluation: discard; failure report unavailable: "
                f"{type(report_exc).__name__}: {report_exc}",
                file=sys.stderr,
            )
        return 1
    status = result["decision"].get("status")
    if status != "promote" and not _is_valid_comparison(result):
        print(f"artifact evaluation: discard ({output})", file=sys.stderr)
        return 1
    if status != "promote":
        print(f"artifact evaluation: discard ({output})")
        return 0
    print(f"artifact evaluation: promote ({output})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
    raise SystemExit(main())
