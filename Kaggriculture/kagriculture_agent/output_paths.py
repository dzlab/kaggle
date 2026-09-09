"""Shared safety checks for training and benchmark output paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

PRODUCTION_DIRECTORY_NAMES = frozenset({
    "model", "models", "checkpoint", "checkpoints", "artifact", "artifacts",
    "deploy", "deployment", "production",
})
PROTECTED_OUTPUT_NAMES = frozenset({
    "model.json", "model.pt", "model.pth",
    "trained_model.json", "trained_model.pt", "trained_model.pth",
    "checkpoint.json", "checkpoint.pt", "checkpoint.pth",
    "artifact.json", "artifact.pt", "artifact.pth",
    "learned_v1.json", "learned_v1.pt", "learned_v1.pth",
})
CANONICAL_PRODUCTION_FILE_NAMES = frozenset({
    "learned_v1.json", "learned_v1.pt", "learned_v1.pth",
})


def resolve_output_path(value: str | Path, *, name: str = "output") -> Path:
    """Resolve an output path, including existing symlink components."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{name} must not be an existing symlink")
    try:
        return raw.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{name} could not be resolved safely") from exc


def validate_training_output_path(
    value: str | Path, *, name: str = "output", reject_protected_names: bool = False,
    reject_symlink_components: bool = False,
) -> Path:
    """Return a resolved training output path unless it targets production."""
    if reject_symlink_components:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{name} must be a nonempty path")
        raw = Path(value).expanduser()
        lexical = Path(os.path.abspath(raw))
        current = Path(lexical.anchor)
        for component in lexical.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f"{name} must not contain symlink components")
    candidate = resolve_output_path(value, name=name)
    if any(part.lower() in PRODUCTION_DIRECTORY_NAMES for part in candidate.parts):
        raise ValueError(f"{name} may not be nested under a production path")
    protected_names = (
        PROTECTED_OUTPUT_NAMES if reject_protected_names
        else CANONICAL_PRODUCTION_FILE_NAMES
    )
    if candidate.name.lower() in protected_names:
        raise ValueError(f"{name} may not target a production artifact or checkpoint")
    return candidate


def _fsync_parent_directory(path: str | Path) -> None:
    """Durably flush a directory after atomically publishing a child entry."""
    directory = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(
    value: str | Path,
    text: str,
    *,
    name: str = "output",
) -> Path:
    """Validate and atomically publish UTF-8 text without following symlinks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    path = validate_training_output_path(
        value, name=name, reject_protected_names=True, reject_symlink_components=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path = validate_training_output_path(
        path, name=name, reject_protected_names=True, reject_symlink_components=True,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        path = validate_training_output_path(
            path, name=name, reject_protected_names=True, reject_symlink_components=True,
        )
        os.replace(temporary_path, path)
        _fsync_parent_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
