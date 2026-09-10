"""Stable route-candidate factories used by evaluation tooling."""

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .constants import ENGINE_VERSION, MAX_POLICY_INFERENCE_P95_MS
from .features import FEATURE_SCHEMA_VERSION
from .learned_policy import load_exported_policy
from .policy import Policy


BASE_CANDIDATES = ("current", "melon", "premium", "mixed")
LEARNED_V1 = "learned_v1"
LEARNED_V1_ARTIFACT_ENV = "KAGRICULTURE_LEARNED_V1_ARTIFACT"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNED_V1_ARTIFACT = PROJECT_ROOT / "models" / "learned_v1.json"
_LEARNED_V1_ARTIFACT_CACHE: dict[Path, tuple[Any | None, ValueError | None]] = {}


def learned_v1_artifact_path() -> Path:
    """Return the configured learned_v1 artifact path."""
    configured = os.environ.get(LEARNED_V1_ARTIFACT_ENV)
    if not configured:
        return DEFAULT_LEARNED_V1_ARTIFACT
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_validated_learned_v1_artifact() -> tuple[Path, Any]:
    path = learned_v1_artifact_path()
    cached = _LEARNED_V1_ARTIFACT_CACHE.get(path)
    if cached is not None:
        learned_model, error = cached
        if error is not None:
            raise error
        return path, learned_model
    if not path.exists():
        error = ValueError(f"{LEARNED_V1} artifact does not exist: {path}")
        _LEARNED_V1_ARTIFACT_CACHE[path] = (None, error)
        raise error
    try:
        learned_model = load_exported_policy(path)
    except Exception as exc:
        error = ValueError(f"{LEARNED_V1} artifact is not valid: {type(exc).__name__}: {exc}")
        _LEARNED_V1_ARTIFACT_CACHE[path] = (None, error)
        raise error from exc
    _LEARNED_V1_ARTIFACT_CACHE[path] = (learned_model, None)
    return path, learned_model


def _validated_learned_v1_artifact_path() -> Path:
    path, _ = _load_validated_learned_v1_artifact()
    return path


def _learned_v1_available() -> bool:
    # Keep the learned route opt-in for the production/evaluator candidate
    # list.  A checked-in artifact is useful for explicit rollout requests,
    # but must not silently change the legacy candidate matrix.
    if not os.environ.get(LEARNED_V1_ARTIFACT_ENV):
        return False
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
        _, learned_model = _load_validated_learned_v1_artifact()
        policy = Policy(
            strategy="current",
            learned_model=learned_model,
            learned_timeout_seconds=MAX_POLICY_INFERENCE_P95_MS / 1000.0,
        )
    elif name in BASE_CANDIDATES:
        policy = Policy(strategy=name)
    else:
        raise ValueError(f"unsupported candidate: {name}")

    # Kaggle inspects ``__code__.co_argcount``.  A bound method reports the
    # underlying function's ``self, obs`` count and is consequently called
    # with an extra configuration argument.  Keep the stateful Policy object
    # closed over by a one-argument adapter.
    def act(observation: Mapping[str, Any]) -> dict[str, Any]:
        return policy.act(observation)

    # Preserve the introspection hook used by the evaluator and older callers
    # while keeping the wrapper's one-argument ``__code__`` contract.
    act.__self__ = policy  # type: ignore[attr-defined]
    return act


def artifact_candidate_policy(path: str | Path) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Load an exported artifact once and return its stateful action callable."""
    try:
        # Keep the validated dependency-free model in memory and hand it to
        # the normal stateful compiler.  The compiler owns episode memory and
        # the artifact remains the only model execution boundary.
        learned_model = load_exported_policy(path)
        if hasattr(learned_model, "act"):
            policy = learned_model
        else:
            policy = Policy(
                strategy="current",
                learned_model=learned_model,
                learned_timeout_seconds=MAX_POLICY_INFERENCE_P95_MS / 1000.0,
            )

        def act(observation: Mapping[str, Any]) -> dict[str, Any]:
            return policy.act(observation)

        act.__self__ = policy  # type: ignore[attr-defined]
        return act
    except Exception as exc:
        raise ValueError(
            f"candidate artifact is not valid: {type(exc).__name__}: {exc}"
        ) from exc
