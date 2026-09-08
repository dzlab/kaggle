# Colab Orbit-Style Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Kaggriculture trainable with a single Colab GPU, bounded parallel CPU rollout workers, resumable Drive-backed checkpoints, fresh self-play PPO, and a continue-until-improved development controller.

**Architecture:** Keep the existing compact typed-token transformer and dependency-free deployment artifact. Add device-aware training, a validated resumable checkpoint format, isolated candidate-policy rollout workers, a bounded parallel collector, and an Orbit Wars–style loop that evaluates every candidate against a fixed development matrix before retaining or rejecting it.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, kaggle-environments 1.32.7, multiprocessing/`concurrent.futures`, JSONL manifests, pytest, Google Colab/Google Drive.

---

## Existing Files and Boundaries

- `kagriculture_agent/model.py`: training network and tensor conversion; remains the model definition.
- `kagriculture_agent/learned_policy.py`: dependency-free artifact runtime and intent compiler; remains the deployment boundary.
- `scripts/train_policy.py`: BC/PPO math and checkpoint metadata; gains device and resumability support.
- `scripts/run_local.py`: one-game engine runner; gains explicit candidate-policy injection while preserving the default `main.agent` path.
- `scripts/collect_trajectories.py`: validated trajectory format; gains bounded parallel scheduling and candidate identity.
- `scripts/evaluate.py`: development/holdout metrics and promotion gates; reused, not weakened.
- New `scripts/train_orbit.py`: round-based rollout/train/evaluate/retain controller.
- New `scripts/colab_train.py`: simple Colab-facing command/configuration wrapper.
- Tests remain under `tests/`, with focused tests added before each implementation.

The existing uncommitted user changes and generated reports/artifacts must not be reset or overwritten.

## Task 1: Add device-aware model and training utilities

**Files:**
- Modify: `kagriculture_agent/model.py`
- Modify: `scripts/train_policy.py`
- Test: `tests/test_model.py`
- Test: `tests/test_train_policy.py`

- [x] **Step 1: Write failing tests for device resolution.**

  Cover `auto` selecting CUDA when available, `auto` selecting CPU otherwise, explicit `cpu`, invalid device names, and a clear failure for explicit `cuda` when CUDA is unavailable.

- [x] **Step 2: Run the focused tests and verify they fail for the missing API.**

  Run: `.venv/bin/pytest -q tests/test_model.py tests/test_train_policy.py -k 'device or cuda'`

  Expected: failures because device resolution and device-aware training are not implemented.

- [x] **Step 3: Implement minimal device selection and tensor placement.**

  Add a small public resolver, use it in the trainer, move `CompactPolicyNet`, features, labels, PPO tensors, and prior networks to the selected device, and keep CPU behavior unchanged. Avoid silently moving individual tensors between devices inside the hot path.

- [x] **Step 4: Add a CLI `--device auto|cpu|cuda` option.**

  Record the resolved device in checkpoint metadata and printed run metadata.

- [x] **Step 5: Run focused tests and a one-batch CPU training smoke test.**

  Run: `.venv/bin/pytest -q tests/test_model.py tests/test_train_policy.py`

  Run: `.venv/bin/python scripts/train_policy.py --input trajectories/learned-dev-smoke.jsonl --output /private/tmp/kagriculture-device-smoke.pt --steps 1 --batch-size 32 --device cpu`

- [ ] **Step 6: Commit the completed task.**

## Task 2: Implement atomic resumable checkpoints

**Files:**
- Modify: `scripts/train_policy.py`
- Create: `kagriculture_agent/checkpoints.py`
- Test: `tests/test_checkpoints.py`
- Test: `tests/test_train_policy.py`

- [x] **Step 1: Write failing tests for checkpoint round trips.**

  Verify that a checkpoint contains model state, optimizer state, configuration, round/cursor metadata, Python/NumPy/PyTorch RNG state, engine/schema/action-vocabulary versions, and metrics.

- [x] **Step 2: Write failing tests for incompatible and corrupt checkpoints.**

  Cover engine-version mismatch, feature-schema mismatch, action-vocabulary mismatch, malformed payloads, missing files, and truncated temporary files.

- [x] **Step 3: Run the focused tests and verify the expected failures.**

  Run: `.venv/bin/pytest -q tests/test_checkpoints.py -x`

- [x] **Step 4: Implement a focused checkpoint module.**

  Provide validated save/load helpers using a temporary file in the destination directory, flush/fsync, atomic replace, and cleanup on failure. Keep the registry separate from the tensor checkpoint so a failed registry write cannot corrupt the best checkpoint.

- [x] **Step 5: Add RNG restoration and resume support to the training loop.**

  Resuming must continue from the saved epoch/round and restore optimizer plus RNG state. A resumed run must not silently start a fresh model when a checkpoint path was supplied.

- [x] **Step 6: Run focused and regression tests.**

  Run: `.venv/bin/pytest -q tests/test_checkpoints.py tests/test_train_policy.py`

- [ ] **Step 7: Commit the completed task.**

## Task 3: Add explicit candidate-policy rollout injection

**Files:**
- Modify: `scripts/run_local.py`
- Modify: `scripts/collect_trajectories.py`
- Modify: `kagriculture_agent/candidates.py`
- Modify: `kagriculture_agent/learned_policy.py`
- Test: `tests/test_run_local.py`
- Test: `tests/test_import_replays.py`
- Test: `tests/test_candidates.py`
- Test: `tests/test_learned_policy.py`

- [x] **Step 1: Write failing tests for candidate injection.**

  Verify that a supplied exported artifact is used for the candidate seat, both seat orders are preserved, the default runner still uses `main.agent`, and invalid/missing artifacts fail before starting an engine game.

- [x] **Step 2: Write a failing regression test for learned market output.**

  If market heads are part of the trained/exported contract, a valid market prediction must reach `PolicyProposal.market_orders` and the normal market legality compiler. Invalid or unavailable orders must remain safely empty.

- [x] **Step 3: Run focused tests and verify the failures.**

  Run: `.venv/bin/pytest -q tests/test_run_local.py tests/test_candidates.py tests/test_learned_policy.py -k 'candidate or market'`

- [x] **Step 4: Implement explicit policy factories and runner injection.**

  Add an artifact-path candidate factory that validates the artifact once per worker and constructs a fresh stateful `Policy`. Extend the local runner and collector request model with candidate artifact identity/path. Keep arbitrary Python model execution out of rollout workers.

- [x] **Step 5: Implement market-head compilation.**

  Convert the predicted item/quantity outputs into conservative market intents, validate them through `build_market_orders`, and preserve the deterministic policy fallback when the model does not produce a usable order.

- [x] **Step 6: Run focused tests plus an isolated candidate game.**

  Run: `.venv/bin/pytest -q tests/test_run_local.py tests/test_candidates.py tests/test_learned_policy.py tests/test_import_replays.py`

  Run a 96-turn game with a known artifact and verify the replay has a valid schema and no framework error.

- [ ] **Step 7: Commit the completed task.**

## Task 4: Parallelize validated rollout collection

**Files:**
- Modify: `scripts/collect_trajectories.py`
- Create: `kagriculture_agent/rollouts.py`
- Test: `tests/test_rollouts.py`
- Test: `tests/test_import_replays.py`
- Test: `tests/test_run_local.py`

- [x] **Step 1: Write failing tests for deterministic rollout scheduling.**

  Given seeds, opponents, seats, and worker count, verify stable request ordering, no duplicate keys, bounded concurrency, and identical manifest contents regardless of completion order.

- [x] **Step 2: Write failing tests for worker failures and timeouts.**

  Verify a failed game is classified explicitly, is not converted into valid transitions, temporary output is not published, and a successful prior dataset remains intact.

- [x] **Step 3: Run the focused tests and verify they fail.**

  Run: `.venv/bin/pytest -q tests/test_rollouts.py -x`

- [x] **Step 4: Implement a bounded process-pool rollout service.**

  Use a configurable worker count capped by available CPUs. Each worker must execute an isolated interpreter/game, pin and validate engine version, enforce a per-game timeout, and return serializable replay/transition data only.

- [x] **Step 5: Preserve atomic trajectory publication.**

  Sort completed records by deterministic request key before writing JSONL. Extend the manifest with worker count, candidate identity, opponent identities, and source artifact hashes without weakening existing replay/hash validation.

- [x] **Step 6: Add CLI options.**

  Add `--workers`, `--candidate-artifact`, `--candidate-identity`, and `--game-timeout`, with safe defaults suitable for Colab. A worker count of one must retain the current serial behavior.

- [x] **Step 7: Run focused and small integration tests.**

  Run: `.venv/bin/pytest -q tests/test_rollouts.py tests/test_import_replays.py tests/test_run_local.py`

  Run a 2-seed, 96-turn collection with two workers and validate the manifest/hash pair.

- [ ] **Step 8: Commit the completed task.**

## Task 5: Connect fresh rollouts to PPO and league opponents

**Files:**
- Modify: `scripts/train_policy.py`
- Modify: `kagriculture_agent/checkpoints.py`
- Modify: `scripts/collect_trajectories.py`
- Test: `tests/test_train_policy.py`
- Test: `tests/test_rollouts.py`

- [x] **Step 1: Write failing tests for league scheduling.**

  Verify each PPO round samples deterministic current, random, starter, and prior learned checkpoints according to the configured probabilities, alternates seats, and does not select a checkpoint that does not exist.

- [x] **Step 2: Write failing tests for fresh-rollout PPO.**

  Verify the rollout callback receives round/seed/opponent/seat/checkpoint/artifact information, PPO consumes new transitions each round, and offline fallback is used only when explicitly requested.

- [x] **Step 3: Run the focused tests and verify failure.**

  Run: `.venv/bin/pytest -q tests/test_train_policy.py tests/test_rollouts.py -k 'league or rollout'`

- [x] **Step 4: Implement a round rollout adapter.**

  Export a temporary dependency-free artifact for the candidate network when needed, invoke the bounded collector, import validated transitions, and feed them to `run_ppo_training`. Keep temporary artifacts inside the configured run directory and clean them only after successful publication or explicit rejection.

- [x] **Step 5: Add previous learned checkpoints to the opponent pool.**

  Maintain at most the configured recent checkpoint window, validate metadata before use, and record exact checkpoint hashes in rollout manifests.

- [x] **Step 6: Run a deterministic one-round CPU integration smoke test.**

  Run one small rollout round against pass/random/current with both seats and verify non-empty transitions, finite PPO metrics, and a new checkpoint.

- [ ] **Step 7: Commit the completed task.**

## Task 6: Implement the continue-until-improved Orbit-style controller

**Files:**
- Create: `scripts/train_orbit.py`
- Modify: `scripts/evaluate.py`
- Modify: `scripts/train_policy.py`
- Create: `tests/test_train_orbit.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing controller tests.**

  Cover: rejected candidate followed by another round; accepted candidate becoming `best`; rejection leaving `best` byte-identical; resume after an interrupted round; stop after development success; stop after explicit max rounds/wall-clock/failure budget; and no holdout evaluator invocation during development.

- [ ] **Step 2: Run the focused tests and verify failure.**

  Run: `.venv/bin/pytest -q tests/test_train_orbit.py -x`

- [x] **Step 3: Define a dependency-injected controller interface.**

  Inject rollout, train, export, evaluate, clock, and filesystem operations where practical so controller tests do not run full Kaggriculture games. The production defaults must use the real collector/trainer/evaluator.

- [x] **Step 4: Implement the round state machine.**

  Persist round state before and after each stage. On restart, detect completed stages and resume safely without publishing a partial candidate. Never replace `best` before the complete development evaluation returns a promotion decision.

- [x] **Step 5: Implement development gates and candidate retention.**

  Require complete valid records, zero framework errors, zero missed basic-needs events, acceptable tail bank performance, and strict improvement over `current`/best according to the existing evaluator contract. Keep holdout seeds/configuration out of this loop.

- [ ] **Step 6: Add controller CLI.**

  Support run directory, device, rollout seeds, opponents, seats, workers, episode steps, PPO rounds, checkpoint window, max rounds, max hours, failure budget, resume, and a dry-run/configuration validation mode.

- [ ] **Step 7: Run focused controller and evaluator tests.**

  Run: `.venv/bin/pytest -q tests/test_train_orbit.py tests/test_evaluate.py tests/test_train_policy.py`

- [ ] **Step 8: Commit the completed task.**

## Task 7: Add the Colab entrypoint and Drive workflow

**Files:**
- Create: `scripts/colab_train.py`
- Modify: `README.md`
- Create: `docs/superpowers/references/colab-training.md`
- Test: `tests/test_colab_train.py`

- [x] **Step 1: Write failing tests for Colab configuration.**

  Verify default paths under a supplied run directory, device auto-selection, worker-count validation, required engine version, resume path handling, and safe rejection of holdout seeds overlapping development seeds.

- [ ] **Step 2: Run the focused tests and verify failure.**

  Run: `.venv/bin/pytest -q tests/test_colab_train.py -x`

- [ ] **Step 3: Implement a thin Colab wrapper.**

  Keep orchestration in `train_orbit.py`; the wrapper should validate configuration, create run directories, print the exact reproducible command/configuration, and call the controller. It must not contain a second training implementation.

- [x] **Step 4: Document the notebook-compatible workflow.**

  Include GPU verification, package installation, Drive mounting, smoke run, parallel rollout run, resume command, artifact export, submission smoke test, development evaluation, and the separate holdout command. State that simulator rollouts are CPU-bound and Colab resources are ephemeral.

- [x] **Step 5: Run documentation/configuration tests.**

  Run: `.venv/bin/pytest -q tests/test_colab_train.py`

- [ ] **Step 6: Commit the completed task.**

## Task 8: End-to-end verification and release safeguards

**Files:**
- Modify: `README.md`
- Modify: `scripts/submission_smoke.py` only if required by learned artifact packaging
- Test: existing full relevant test suite
- Create: `reports/` artifacts only as ignored/generated files; do not commit them

- [ ] **Step 1: Run the complete relevant unit suite.**

  Run: `.venv/bin/pytest -q`

- [ ] **Step 2: Run a small Colab-equivalent integration workflow.**

  Use CPU fallback if no CUDA device is available: collect a small two-worker dataset, train one round, export the artifact, resume for one more round, evaluate development games, and verify rejected candidates do not replace `best`.

- [x] **Step 3: Run artifact and submission smoke checks.**

  Validate the exported artifact, deterministic package contents, `python -S` fallback behavior, representative action schema, and runtime latency. Confirm the archive excludes checkpoints, scripts, tests, trajectories, reports, and training dependencies.

- [x] **Step 4: Run `git diff --check` and inspect the complete diff.**

  Confirm that `main.py` still uses `Policy()` and that no holdout report or production artifact was generated as a promotion claim.

- [ ] **Step 5: Run the final reviewer against the full implementation.**

  Review requirements against the approved design, with special attention to device placement, process bounds, checkpoint atomicity, best-policy immutability, holdout isolation, and fail-safe deployment.

- [ ] **Step 6: Commit only source, tests, documentation, and approved plan/spec changes.**

  Leave generated trajectories, checkpoints, reports, and artifacts ignored unless explicitly requested for archival.

## Final acceptance checklist

- [x] `--device auto` uses CUDA in Colab and CPU fallback locally.
- [x] Model, labels, optimizer, and PPO tensors share the selected device.
- [x] Checkpoints resume model, optimizer, RNG, round, and cursor state.
- [x] Rollouts run in a bounded process pool with deterministic manifests.
- [x] Candidate artifacts can play against current, random, starter, and prior learned policies.
- [x] PPO consumes fresh rollouts rather than only the initial behavior-cloning dataset.
- [x] Rejected candidates never replace `best`; training continues until improvement or an explicit budget limit.
- [x] Holdout evaluation is impossible through the development controller unless explicitly invoked after freezing the candidate.
- [ ] Exported artifacts and submission archives pass existing safety gates.
- [x] The production default remains deterministic until holdout promotion succeeds.

## Implementation notes

- The new workflow's focused suites, real 96-turn candidate replay, two-seed/two-worker collection, and one-round CPU fresh-rollout PPO smoke test pass.
- The full repository suite still has six legacy deterministic-policy failures for seed 17, including missing `FERTILIZE` coverage and missed basic-needs diagnostics. These failures are outside the new rollout/checkpoint/controller code and must be resolved before claiming release readiness.
- Controller and Colab command surfaces are intentionally callback/configuration driven today; wiring production collector/trainer/evaluator defaults and a full resume-through-controller integration remains the next implementation item.
- Commit checkboxes remain open because the worktree contains unrelated user changes and generated artifacts; source changes were not bundled into a mixed commit.
