"""Build and smoke-test the self-contained Kaggriculture submission archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Iterable


_EXACTLY_FORBIDDEN_PARTS = {
    "tests",
    "docs",
    "scripts",
    "reports",
    "replays",
    "trajectories",
    "checkpoints",
    "torch",
    "numpy",
    "training",
}
_FORBIDDEN_SUFFIXES = (".pt", ".pth")
_FALLBACK_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
_MAX_ARCHIVE_MEMBERS = 64
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_MEMBER_NAME_BYTES = 240
_ARTIFACT_DIRECTORIES = {"artifacts", "models"}
_RUNTIME_MODULES = (
    "__init__.py",
    "candidates.py",
    "constants.py",
    "economics.py",
    "experimental_features.py",
    "features.py",
    "learned_policy.py",
    "memory.py",
    "observation.py",
    "planner.py",
    "policy.py",
    "runtime_identity.py",
    "routing.py",
    "strategy.py",
    "types.py",
)


def _promoted_main_source() -> bytes:
    """Return the artifact-aware entrypoint used only in promoted archives."""
    return (
        "from pathlib import Path\n"
        "from kagriculture_agent.policy import Policy\n"
        "\n"
        "_artifact_agent = None\n"
        "try:\n"
        "    from kagriculture_agent.candidates import artifact_candidate_policy\n"
        "    _artifact_agent = artifact_candidate_policy(\n"
        "        Path(__file__).parent / 'models' / 'learned_v1.json'\n"
        "    )\n"
        "except Exception:\n"
        "    # A malformed or unavailable learned artifact must never prevent a\n"
        "    # deterministic submission from running.\n"
        "    _artifact_agent = None\n"
        "\n"
        "_fallback_policy = Policy()\n"
        "\n"
        "def agent(obs):\n"
        "    if _artifact_agent is not None:\n"
        "        return _artifact_agent(obs)\n"
        "    return _fallback_policy.act(obs)\n"
    ).encode("utf-8")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _member_is_forbidden(name: str) -> bool:
    lowered = name.lower()
    parts = set(Path(lowered).parts)
    return (
        "__pycache__" in parts
        or bool(parts & _EXACTLY_FORBIDDEN_PARTS)
        or lowered.endswith(_FORBIDDEN_SUFFIXES)
    )


def _files_under(path: Path, root: Path) -> Iterable[tuple[Path, str]]:
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    for candidate in candidates:
        if candidate.name == "__pycache__" or candidate.suffix == ".pyc":
            continue
        if candidate.is_symlink():
            raise ValueError(f"symlink is not allowed in archive: {candidate}")
        resolved = candidate.resolve()
        if not _inside(resolved, root):
            raise ValueError(f"path is outside project_root: {candidate}")
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(root).as_posix()
        if _member_is_forbidden(relative):
            raise ValueError(f"forbidden archive member: {relative}")
        yield candidate, relative


def _engine_version(root: Path) -> str:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return "unknown"
    with pyproject.open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    dependencies = project.get("dependencies", [])
    for dependency in dependencies:
        if dependency.startswith("kaggle-environments=="):
            return dependency.split("==", 1)[1].split(";", 1)[0].strip()
    return "unknown"


def _runtime_files(package: Path, root: Path) -> Iterable[tuple[Path, str]]:
    """Yield the reviewed runtime allowlist, ignoring development files."""
    if package.is_symlink():
        raise ValueError(f"runtime package must not be a symlink: {package}")
    for filename in _RUNTIME_MODULES:
        candidate = package / filename
        if not candidate.exists():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"runtime member is not a regular file: {candidate}")
        if not _inside(candidate.resolve(), root):
            raise ValueError(f"runtime member is outside project_root: {candidate}")
        yield candidate, candidate.relative_to(root).as_posix()


def _validate_artifact(path: Path, root: Path) -> None:
    """Validate a selected proposal or exported network artifact before packaging."""
    if path.suffix.lower() != ".json":
        raise ValueError("selected artifact must be a JSON file")
    if path.stat().st_size > _MAX_MEMBER_BYTES:
        raise ValueError("selected artifact is too large")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"selected artifact is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("selected artifact must be a JSON object")
    if any(key in value for key in ("workers", "market_orders", "market")):
        if "workers" in value and not isinstance(value["workers"], list):
            raise ValueError("proposal artifact workers must be a list")
        if "market_orders" in value and not isinstance(value["market_orders"], list):
            raise ValueError("proposal artifact market_orders must be a list")
        if "market" in value and not isinstance(value["market"], list):
            raise ValueError("proposal artifact market must be a list")
        return
    if not {"format_version", "weights", "checksum"}.issubset(value):
        raise ValueError("selected artifact is neither a policy proposal nor an exported policy")

    # Run the canonical stdlib-only validator from the selected project.  This
    # keeps packaging validation aligned with the artifact used by the agent.
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from kagriculture_agent.learned_policy import load_exported_policy; "
            "load_exported_policy(__import__('sys').argv[1])",
            str(path),
        ],
        cwd=root,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"selected artifact failed runtime validation: {detail}")


def _read_json_evidence(path: str | Path, *, label: str) -> tuple[bytes, dict]:
    evidence_path = Path(path).expanduser().resolve()
    if not evidence_path.is_file():
        raise ValueError(f"{label} evidence does not exist: {evidence_path}")
    if evidence_path.stat().st_size > _MAX_MEMBER_BYTES:
        raise ValueError(f"{label} evidence is too large")
    try:
        content = evidence_path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} evidence is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} evidence must be a JSON object")
    return content, value


def build_submission_archive(
    project_root: str | Path,
    output: str | Path,
    artifact: str | Path | None = None,
    holdout_report: str | Path | None = None,
    latency_report: str | Path | None = None,
) -> dict:
    """Create a deterministic runtime archive with optional promotion evidence."""
    root = Path(project_root).resolve()
    output_path = Path(output).resolve()
    main = root / "main.py"
    package = root / "kagriculture_agent"
    if not main.is_file():
        raise ValueError("main.py does not exist")

    artifact_relative: str | None = None
    if artifact is not None:
        artifact_path = Path(artifact).resolve()
        if not artifact_path.exists() or not artifact_path.is_file():
            raise ValueError("artifact does not exist")
        if artifact_path.is_symlink():
            raise ValueError("artifact must not be a symlink")
        # Training artifacts live under a run directory (often on Drive), so
        # copy the selected bytes into the stable production member name
        # without modifying the checked-out models directory.
        artifact_relative = "models/learned_v1.json"
        _validate_artifact(artifact_path, root)

    if (holdout_report is None) != (latency_report is None):
        raise ValueError("holdout_report and latency_report must be provided together")
    if holdout_report is not None and artifact is None:
        raise ValueError("promotion evidence requires a selected artifact")
    evidence: dict[str, tuple[bytes, dict, Path]] = {}
    if holdout_report is not None and latency_report is not None:
        holdout_bytes, holdout_value = _read_json_evidence(holdout_report, label="holdout")
        latency_bytes, latency_value = _read_json_evidence(latency_report, label="latency")
        evidence = {
            "evidence/holdout.json": (holdout_bytes, holdout_value, Path(holdout_report).expanduser().resolve()),
            "evidence/cpu-latency.json": (latency_bytes, latency_value, Path(latency_report).expanduser().resolve()),
        }

    if not package.is_dir():
        raise ValueError("kagriculture_agent package does not exist")

    selected: list[tuple[Path | None, str, bytes | None]] = []
    for path, relative in _files_under(main, root):
        selected.append((path, relative, _promoted_main_source() if artifact is not None else None))
    selected.extend((path, relative, None) for path, relative in _runtime_files(package, root))
    if artifact is not None:
        selected.append((artifact_path, artifact_relative, None))
    selected.extend((None, relative, content) for relative, (content, _, _) in evidence.items())

    if evidence:
        integrity = {
            "schema_version": 1,
            "members": {},
            "evidence": {
                "holdout": {
                    "member": "evidence/holdout.json",
                    "source": str(evidence["evidence/holdout.json"][2]),
                    "sha256": hashlib.sha256(evidence["evidence/holdout.json"][0]).hexdigest(),
                },
                "latency": {
                    "member": "evidence/cpu-latency.json",
                    "source": str(evidence["evidence/cpu-latency.json"][2]),
                    "sha256": hashlib.sha256(evidence["evidence/cpu-latency.json"][0]).hexdigest(),
                },
            },
        }
        for path, relative, content in selected:
            if content is not None:
                integrity["members"][relative] = hashlib.sha256(content).hexdigest()
            else:
                assert path is not None
                integrity["members"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        integrity_bytes = (json.dumps(integrity, sort_keys=True, indent=2) + "\n").encode("utf-8")
        selected.append((None, "manifest.json", integrity_bytes))

    members = sorted({relative: (path, content) for path, relative, content in selected}.items())
    if any(path is not None and output_path == path.resolve() for path, _, _ in selected):
        raise ValueError("output path overlaps an input file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with temporary_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    for relative, (path, content) in members:
                        if content is not None:
                            info = tarfile.TarInfo(relative)
                            info.size = len(content)
                        else:
                            assert path is not None
                            info = tar.gettarinfo(str(path), arcname=relative)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        if info.isfile():
                            info.mode = 0o644
                            if content is not None:
                                tar.addfile(info, io.BytesIO(content))
                            else:
                                assert path is not None
                                with path.open("rb") as stream:
                                    tar.addfile(info, stream)
                        else:
                            tar.addfile(info)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "archive": str(output_path),
        "members": [relative for relative, _ in members],
        "artifact": artifact_relative,
        "runtime_dependencies": [],
        "engine_version": _engine_version(root),
        "integrity_manifest": "manifest.json" if evidence else None,
        "evidence": {
            "holdout": "evidence/holdout.json",
            "latency": "evidence/cpu-latency.json",
        } if evidence else {},
    }


def smoke_test_archive(archive: str | Path, artifact: str | None = None) -> dict:
    """Extract and import an archive with third-party site packages disabled."""
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise RuntimeError(f"archive does not exist: {archive_path}")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise RuntimeError("archive contains too many members")
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise RuntimeError("archive contains duplicate members")
            if any(_member_is_forbidden(name) for name in names):
                raise RuntimeError("archive contains a forbidden member")
            if any(len(name.encode("utf-8")) > _MAX_MEMBER_NAME_BYTES for name in names):
                raise RuntimeError("archive member name is too long")
            if sum(member.size for member in members) > _MAX_ARCHIVE_BYTES:
                raise RuntimeError("archive is too large")
            if "main.py" not in names or not any(
                name == "kagriculture_agent/__init__.py" for name in names
            ):
                raise RuntimeError("archive is missing main.py or kagriculture_agent package")
            nonregular = [member.name for member in members if not member.isfile()]
            if nonregular:
                raise RuntimeError(f"archive contains a non-regular member: {nonregular[0]}")
            allowed = {"main.py"}
            allowed.update(f"kagriculture_agent/{filename}" for filename in _RUNTIME_MODULES)
            if artifact is not None:
                allowed.add(artifact)
            if "manifest.json" in names:
                allowed.update({
                    "manifest.json",
                    "evidence/holdout.json",
                    "evidence/cpu-latency.json",
                })
            unexpected = sorted(set(names) - allowed)
            if unexpected:
                raise RuntimeError(f"archive contains unexpected members: {unexpected}")
            for member in members:
                if member.size > _MAX_MEMBER_BYTES:
                    raise RuntimeError(f"archive member is too large: {member.name}")
                target = (root / member.name).resolve()
                if not _inside(target, root):
                    raise RuntimeError("archive contains a path outside extraction directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"archive member cannot be read: {member.name}")
                with target.open("wb") as destination:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError(f"archive member ended early: {member.name}")
                        destination.write(chunk)
                        remaining -= len(chunk)

            if artifact is not None and "manifest.json" not in names:
                raise RuntimeError("promoted archive is missing manifest.json")
            if "manifest.json" in names:
                required_evidence = {"evidence/holdout.json", "evidence/cpu-latency.json"}
                if "models/learned_v1.json" not in names or not required_evidence.issubset(names):
                    raise RuntimeError("archive is missing promotion evidence")
                try:
                    integrity = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"archive integrity manifest is invalid: {exc}") from exc
                if not isinstance(integrity, dict) or integrity.get("schema_version") != 1:
                    raise RuntimeError("archive integrity manifest schema is invalid")
                members_digest = integrity.get("members")
                if not isinstance(members_digest, dict) or "manifest.json" in members_digest:
                    raise RuntimeError("archive integrity manifest is circular or malformed")
                expected_members = set(names) - {"manifest.json"}
                if set(members_digest) != expected_members:
                    raise RuntimeError("archive integrity manifest member set is incomplete")
                for name, expected_digest in members_digest.items():
                    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                        raise RuntimeError(f"archive integrity digest is malformed: {name}")
                    actual_digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
                    if actual_digest != expected_digest:
                        raise RuntimeError(f"archive integrity digest mismatch: {name}")
                evidence_digest = integrity.get("evidence")
                if not isinstance(evidence_digest, dict):
                    raise RuntimeError("archive evidence manifest is missing")
                for label, member_name in (
                    ("holdout", "evidence/holdout.json"),
                    ("latency", "evidence/cpu-latency.json"),
                ):
                    entry = evidence_digest.get(label)
                    if not isinstance(entry, dict):
                        raise RuntimeError(f"archive {label} evidence manifest is missing")
                    if entry.get("member") != member_name:
                        raise RuntimeError(f"archive {label} evidence member is invalid")
                    if entry.get("sha256") != members_digest[member_name]:
                        raise RuntimeError(f"archive {label} evidence hash is invalid")
                    try:
                        evidence_value = json.loads((root / member_name).read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(f"archive {label} evidence is invalid: {exc}") from exc
                    if not isinstance(evidence_value, dict):
                        raise RuntimeError(f"archive {label} evidence must be an object")

        code = (
            "import socket\n"
            "def _blocked(*args, **kwargs): raise RuntimeError('network disabled')\n"
            "socket.socket = _blocked\n"
            "socket.create_connection = _blocked\n"
            "import main\n"
            "assert callable(main.agent)\n"
            "result = main.agent({'step': 0, 'day': 0, 'farm': {}, 'private': {}, 'market': {}})\n"
            "assert set(result) == {'farmer', 'hands', 'market'}\n"
            "assert isinstance(result['farmer'], list) and isinstance(result['hands'], list)\n"
            "assert isinstance(result['market'], list)\n"
        )
        if artifact is not None and "kagriculture_agent/learned_policy.py" in names:
            code += (
                "import json\n"
                f"_artifact = json.loads(open({str(root / artifact)!r}, encoding='utf-8').read())\n"
                "if any(key in _artifact for key in ('format_version', 'weights', 'checksum')):\n"
                "    from kagriculture_agent.learned_policy import load_exported_policy\n"
                f"    load_exported_policy({str(root / artifact)!r})\n"
            )
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(root)}
        try:
            result = subprocess.run(
                [sys.executable, "-S", "-c", code],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=10.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("archive smoke test timed out after 10 seconds") from exc
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"archive smoke test failed: {detail}")
    return {"archive": str(archive_path), "passed": True, "action": _FALLBACK_ACTION}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("--holdout-report")
    parser.add_argument("--latency-report")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    result = build_submission_archive(
        args.project_root,
        args.output,
        args.artifact,
        holdout_report=args.holdout_report,
        latency_report=args.latency_report,
    )
    if args.smoke_test:
        result = {**result, "smoke_test": smoke_test_archive(args.output, result["artifact"])}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
