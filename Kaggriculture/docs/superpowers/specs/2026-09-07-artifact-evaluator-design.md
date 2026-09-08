# Dependency-Free Artifact Evaluator Design

**Date:** 2026-09-07

**Status:** Approved for implementation

## Goal

Provide a reproducible, bounded evaluator for a supplied dependency-free
learned artifact against the current policy so Colab can safely assess PPO
artifacts such as `ppo16` and `ppo32` before continuing training.

## Scope and non-goals

The evaluator adds `scripts/evaluate_artifact.py` and focused tests in
`tests/test_evaluate_artifact.py`. It evaluates the current policy and the
supplied artifact on the same opponent/seed/seat matrix, normalizes replays
through `scripts.evaluate.replay_record`, and applies the existing
`promotion_decision` safety and improvement gates.

It does not change `Policy()`, `main.py`, production artifacts, or the
existing general evaluator. It does not run long evaluations as part of the
implementation or tests.

## Architecture

The script validates the artifact once in the parent process with the existing
dependency-free artifact loader and records its SHA256 plus a caller-provided
identity. It builds one deterministic Cartesian matrix in opponent, seed, and
seat order. A bounded `ProcessPoolExecutor` runs each matrix coordinate in a
fresh local game using `run_local.run_episode`; temporary replay files are
normalized by `replay_record` and removed with the temporary directory.

The current candidate and artifact candidate receive the same coordinate list.
Only `pass`, `random`, and `starter` are accepted as opponents, so the learned
artifact is never passed as an opponent. Ordered executor mapping preserves
request order regardless of completion order.

## CLI and report contract

The CLI accepts `--artifact`, `--identity`, `--seeds`, `--start-seed`,
`--steps`, `--opponents`, `--seats`, `--workers`, `--min-valid-games`,
`--output`, and `--quick`. Seeds are a consecutive count from `--start-seed`;
seats accept one-or-more unique values from `{0, 1}` and default to both.
Workers are positive and capped at eight. Quick mode mirrors the existing
evaluator's smoke defaults by reducing untouched defaults to two seeds and 96
steps.

Reports include configuration, explicit expected matrix, artifact identity and
SHA256, ordered per-game records for both candidates, paired summaries, and the
existing promotion decision. Report output is written atomically.

## Failure behavior

Invalid artifacts, invalid CLI values, worker launch/timeout failures,
framework errors, malformed records, and incomplete, duplicate, or extra
matrix coordinates fail closed. The report records the failure and the
decision is `discard`; the CLI exits nonzero when evaluation cannot establish
a complete valid comparison.

## Testing

Focused tests use injected mocked game execution and short synthetic records.
They verify deterministic matrix construction, artifact validation/identity
and checksum reporting, unique seat validation, and rejection of a degraded
candidate by the existing promotion gates. No test starts a long Kaggriculture
game.
