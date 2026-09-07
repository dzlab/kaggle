"""Stable route-candidate factories used by evaluation tooling."""

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .constants import ENGINE_VERSION
from .features import FEATURE_SCHEMA_VERSION
from .learned_policy import load_exported_policy
from .policy import Policy


BASE_CANDIDATES = ("current", "melon", "premium", "mixed")
LEARNED_V1 = "learned_v1"
LEARNED_V1_ARTIFACT_ENV = "KAGRICULTURE_LEARNED_V1_ARTIFACT"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNED_V1_ARTIFACT = PROJECT_ROOT / "models" / "learned_v1.json"


def learned_v1_artifact_path() -> Path:
    """Return the configured learned_v1 artifact path."""
    configured = os.environ.get(LEARNED_V1_ARTIFACT_ENV)
    if not configured:
        return DEFAULT_LEARNED_V1_ARTIFACT
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validated_learned_v1_artifact_path() -> Path:
    path = learned_v1_artifact_path()
    if not path.exists():
        raise ValueError(f"{LEARNED_V1} artifact does not exist: {path}")
    try:
        load_exported_policy(path)
    except Exception as exc:
        raise ValueError(f"{LEARNED_V1} artifact is not valid: {type(exc).__name__}: {exc}") from exc
    return path


def _learned_v1_available() -> bool:
    try:
        _validated_learned_v1_artifact_path()
    except ValueError:
        return False
    return True


CANDIDATES = BASE_CANDIDATES + ((LEARNED_V1,) if _learned_v1_available() else ())


def _artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_metadata(name: str) -> dict[str, Any]:
    """Return reproducibility metadata for a stable route candidate."""
    if name == LEARNED_V1:
        artifact_path = _validated_learned_v1_artifact_path()
        return {
            "model_identity": LEARNED_V1,
            "artifact_sha256": _artifact_sha256(artifact_path),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "engine_version": str(ENGINE_VERSION),
        }
    if name not in BASE_CANDIDATES:
        raise ValueError(f"unsupported candidate: {name}")
    return {
        "model_identity": f"deterministic:{name}",
        "artifact_sha256": None,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "engine_version": str(ENGINE_VERSION),
    }


def candidate_policy(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Return a fresh stateful route policy's observation callable."""
    if name == LEARNED_V1:
        return Policy(strategy="current", learned_model=str(_validated_learned_v1_artifact_path())).act
    if name not in BASE_CANDIDATES:
        raise ValueError(f"unsupported candidate: {name}")
    return Policy(strategy=name).act
