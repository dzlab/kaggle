"""Import validated public Kaggriculture replays into a training JSONL corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kagriculture_agent.constants import ENGINE_VERSION
from kagriculture_agent.features import FEATURE_SCHEMA_VERSION
from kagriculture_agent.trajectory import TRANSITION_SCHEMA_VERSION, transitions_from_replay


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _copy_backup(source: Path, backup: Path) -> None:
    shutil.copyfile(source, backup)
    with backup.open("rb") as backup_file:
        os.fsync(backup_file.fileno())


def _read_replay(path: Path) -> tuple[Mapping[str, Any], int, str]:
    try:
        raw = path.read_bytes()
        replay = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid replay JSON in {path}: {exc}") from exc
    if not isinstance(replay, Mapping):
        raise ValueError(f"invalid replay {path}: expected a JSON object")
    if replay.get("module_version") != ENGINE_VERSION:
        raise ValueError(
            f"invalid replay {path}: engine version {replay.get('module_version')!r} "
            f"does not match {ENGINE_VERSION!r}"
        )
    info = replay.get("info")
    seed = info.get("seed") if isinstance(info, Mapping) else None
    if type(seed) is not int:
        raise ValueError(f"invalid replay {path}: info.seed must be an integer")
    return replay, seed, hashlib.sha256(raw).hexdigest()


def _publish_pair(
    output_temp: Path,
    manifest_temp: Path,
    output: Path,
    manifest_path: Path,
) -> None:
    """Publish manifest last so its output hash flags a crash between replacements."""
    run_id = uuid.uuid4().hex
    output_backup = output.parent / f".{output.name}.{run_id}.bak"
    manifest_backup = manifest_path.parent / f".{manifest_path.name}.{run_id}.bak"
    output_backed_up = manifest_backed_up = False
    output_replaced = manifest_replaced = False
    try:
        if output.exists():
            _copy_backup(output, output_backup)
            output_backed_up = True
        if manifest_path.exists():
            _copy_backup(manifest_path, manifest_backup)
            manifest_backed_up = True
        os.replace(output_temp, output)
        output_replaced = True
        _fsync_directory(output.parent)
        os.replace(manifest_temp, manifest_path)
        manifest_replaced = True
        _fsync_directory(output.parent)
    except Exception:
        if output_backed_up:
            os.replace(output_backup, output)
            output_backed_up = False
        elif output_replaced:
            output.unlink(missing_ok=True)
        if manifest_backed_up:
            os.replace(manifest_backup, manifest_path)
            manifest_backed_up = False
        elif manifest_replaced:
            manifest_path.unlink(missing_ok=True)
        _fsync_directory(output.parent)
        raise
    finally:
        if not output_backed_up:
            output_backup.unlink(missing_ok=True)
        if not manifest_backed_up:
            manifest_backup.unlink(missing_ok=True)


def import_replays(
    source_dir: str | Path,
    output: str | Path,
    candidate_player: int = 0,
    source_policy_identity: str = "public_replay",
) -> dict[str, Any]:
    """Validate and atomically import replay files into training JSONL."""
    source_root = Path(source_dir).resolve()
    output_path = Path(output).resolve()
    manifest_path = output_path.with_suffix(".manifest.json")
    excluded_paths = {
        path for path in (output_path, manifest_path)
        if path.is_relative_to(source_root)
    }
    try:
        replay_paths = sorted(
            (
                path for path in source_root.rglob("*.json")
                if path.is_file() and path.resolve() not in excluded_paths
            ),
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
    except OSError as exc:
        raise ValueError(f"could not inspect replay source directory {source_root}: {exc}") from exc
    if not replay_paths:
        raise ValueError(f"no replay JSON files beneath {source_root}")
    if type(candidate_player) is not int or candidate_player not in (0, 1):
        raise ValueError("candidate_player must be 0 or 1")
    if not isinstance(source_policy_identity, str) or not source_policy_identity:
        raise ValueError("source_policy_identity must be a non-empty string")

    source_entries: list[dict[str, Any]] = []
    transitions = []
    for replay_path in replay_paths:
        replay, seed, source_hash = _read_replay(replay_path)
        try:
            source_transitions = transitions_from_replay(
                replay, candidate_player, requested_seed=seed,
            )
        except Exception as exc:
            if isinstance(exc, ValueError):
                detail = str(exc)
            else:
                detail = f"{type(exc).__name__}: {exc}"
            raise ValueError(f"invalid replay {replay_path}: {detail}") from exc
        transitions.extend(source_transitions)
        source_entries.append({
            "path": replay_path.relative_to(source_root).as_posix(),
            "sha256": source_hash,
            "seed": seed,
            "transition_count": len(source_transitions),
        })

    manifest = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "engine_version": str(ENGINE_VERSION),
        "candidate_player": candidate_player,
        "source_policy_identity": source_policy_identity,
        "source_replay_count": len(source_entries),
        "transition_count": len(transitions),
        "seeds": sorted({entry["seed"] for entry in source_entries}),
        "sources": source_entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_temp_path: Path | None = None
    manifest_temp_path: Path | None = None
    try:
        output_bytes = "".join(transition.to_json() + "\n" for transition in transitions).encode("utf-8")
        manifest["output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as output_temp:
            output_temp_path = Path(output_temp.name)
            output_temp.write(output_bytes)
            output_temp.flush()
            os.fsync(output_temp.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{manifest_path.name}.", suffix=".tmp", delete=False,
        ) as manifest_temp:
            manifest_temp_path = Path(manifest_temp.name)
            manifest_temp.write(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            manifest_temp.flush()
            os.fsync(manifest_temp.fileno())
        _publish_pair(output_temp_path, manifest_temp_path, output_path, manifest_path)
        output_temp_path = manifest_temp_path = None
    finally:
        if output_temp_path is not None:
            output_temp_path.unlink(missing_ok=True)
        if manifest_temp_path is not None:
            manifest_temp_path.unlink(missing_ok=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-player", type=int, default=0)
    parser.add_argument("--source-policy-identity", default="public_replay")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = import_replays(
        args.source_dir,
        args.output,
        candidate_player=args.candidate_player,
        source_policy_identity=args.source_policy_identity,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
