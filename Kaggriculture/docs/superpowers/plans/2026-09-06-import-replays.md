# Replay Importer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import validated public Kaggriculture replay JSON files into an atomic, provenance-tracked training JSONL corpus.

**Architecture:** `scripts/import_replays.py` will discover and validate all source files before publishing either artifact. It will reuse `transitions_from_replay` for evaluator-backed conversion, serialize only `Transition.to_json()` rows, and atomically replace the output and manifest together while cleaning temporary files on failure.

**Tech Stack:** Python 3.11+, pathlib, JSON, hashlib, tempfile/os atomic replacement, pytest.

---

### Task 1: Implement the replay importer

**Files:**
- Create: `scripts/import_replays.py`
- Test: `tests/test_import_replays.py` (existing user-provided coverage)

- [ ] **Step 1: Run the focused test before implementation**

Run: `.venv/bin/pytest -q tests/test_import_replays.py`
Expected: collection or import failure because `scripts.import_replays` does not exist.

- [ ] **Step 2: Add deterministic discovery, strict envelope checks, and conversion**

Resolve `source_dir` and `output` as `Path` values; discover `*.json` recursively in sorted relative-path order; require each parsed value to be a mapping, exact `module_version == ENGINE_VERSION`, and `info.seed` to have exact type `int`; call `transitions_from_replay(replay, candidate_player, requested_seed=seed)` and wrap source-specific failures in useful `ValueError`s.

- [ ] **Step 3: Add atomic JSONL and manifest publication**

Create temporary files in the destination directory, write one serialized transition per line, construct the required manifest including hashes and relative source paths, flush and fsync both files, then replace the destination pair only after every replay validates. On any exception, remove temporary files and restore any pre-existing pair so failed imports do not publish partial output.

- [ ] **Step 4: Add the command-line entry point**

Expose positional `source_dir` and `output` arguments plus `--candidate-player` and `--source-policy-identity`, invoke `import_replays`, print compact sorted manifest JSON, and return a normal CLI exit status.

- [ ] **Step 5: Run focused verification**

Run: `.venv/bin/pytest -q tests/test_import_replays.py`
Expected: all focused importer tests pass.
