# Training Metrics Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Kaggriculture telemetry so training health, per-game validation outcomes, aggregate validation quality, and promotion-safety signals are persisted locally and mirrored to W&B.

**Architecture:** Keep the existing JSONL-first `TrainingTelemetry` interface as the single event sink. Add metric extraction at the PPO/behavior-cloning update boundary, add a reusable validation-report emitter that records both raw game events and flattened candidate summaries, and keep one telemetry run open through development, smoke, and holdout evaluation. Preserve the existing evaluator report schema and promotion gates.

**Tech Stack:** Python 3, PyTorch, pytest, JSONL telemetry, optional Weights & Biases.

---

### Task 1: Add optimizer and rollout health metrics

**Files:**
- Modify: `Kaggriculture/scripts/train_policy.py`
- Test: `Kaggriculture/tests/test_train_policy.py`

- [ ] **Step 1: Write failing tests** for PPO metrics containing clip fraction, explained variance, return/advantage statistics, gradient norm, parameter norm, and learning rate; update the callback test to require the same metrics in the emitted `ppo` event.

- [ ] **Step 2: Run the focused tests** with `pytest Kaggriculture/tests/test_train_policy.py -k 'ppo_update or telemetry' -q` and confirm the new assertions fail before implementation.

- [ ] **Step 3: Implement metric extraction** in `ppo_update`: compute rollout return/advantage statistics, PPO clip fraction, value explained variance, gradient norm immediately before the optimizer step, parameter norm after the step, and the active optimizer learning rate. Include these fields in both returned metrics and the callback payload, while retaining existing metrics and compatibility with injected legacy update functions.

- [ ] **Step 4: Add behavior-cloning optimizer health fields** to behavior-cloning telemetry: learning rate, gradient norm, parameter norm, and action entropy when available.

- [ ] **Step 5: Run the focused tests** again and confirm they pass.

- [ ] **Step 6: Commit** with `git add Kaggriculture/scripts/train_policy.py Kaggriculture/tests/test_train_policy.py && git commit -m "feat(kagriculture): track optimizer training metrics"`.

### Task 2: Emit structured validation game and summary events

**Files:**
- Modify: `Kaggriculture/scripts/telemetry.py`
- Test: `Kaggriculture/tests/test_telemetry.py`

- [ ] **Step 1: Write failing tests** using a representative evaluator report. Assert that the emitter creates one `validation_game` event per current/candidate record and one flattened `validation_summary` event per candidate, including win rate, wins/losses/ties, confidence bounds, bank-differential tail, framework-error rate, missed-needs rate, matrix completeness, and decision status.

- [ ] **Step 2: Run the focused telemetry tests** with `pytest Kaggriculture/tests/test_telemetry.py -q` and confirm the new assertions fail.

- [ ] **Step 3: Implement `record_validation_report`** in `scripts/telemetry.py`. Preserve the raw per-game fields needed for later analysis, flatten nested Wilson/bootstrap/Elo values into scalar metrics suitable for W&B, derive validity and safety rates from the records, and include phase/candidate/checkpoint identity on every event. Ignore malformed optional fields without preventing the local JSONL event from being written.

- [ ] **Step 4: Run the focused telemetry tests** and confirm they pass, including existing local/Weave/W&B compatibility tests.

- [ ] **Step 5: Commit** with `git add Kaggriculture/scripts/telemetry.py Kaggriculture/tests/test_telemetry.py && git commit -m "feat(kagriculture): emit validation telemetry"`.

### Task 3: Keep telemetry alive through validation and document the schema

**Files:**
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/README.md`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [ ] **Step 1: Write failing workflow tests** that inject a fake telemetry sink and fake evaluation reports, then assert that development and holdout validation summaries are recorded after evaluator completion and before telemetry finalization.

- [ ] **Step 2: Run the focused workflow tests** with `pytest Kaggriculture/tests/test_colab_train.py -q` and confirm the new assertions fail.

- [ ] **Step 3: Integrate validation reporting** into `run_workflow`: keep the telemetry instance open until all enabled evaluation phases finish, emit development and holdout reports through `record_validation_report`, and still finalize telemetry on every error path. Add a final training metadata event with the configuration and artifact identity.

- [ ] **Step 4: Update training documentation** with the local JSONL path, event names, required validation dashboard fields, and the rule that win rate is primary while safety/data-quality signals are promotion gates.

- [ ] **Step 5: Run focused workflow and telemetry tests** and confirm they pass.

- [ ] **Step 6: Commit** with `git add Kaggriculture/scripts/train.py Kaggriculture/tests/test_colab_train.py Kaggriculture/README.md && git commit -m "feat(kagriculture): track validation during workflow"`.

### Task 4: Full verification and review

**Files:**
- Verify all changed files and tests.

- [ ] **Step 1: Run formatting and syntax checks** with `python -m compileall -q Kaggriculture/scripts Kaggriculture/tests` and `git diff --check`.

- [ ] **Step 2: Run the complete Kaggriculture test suite** with `pytest Kaggriculture/tests -q`, record any known baseline failures separately, and ensure all new metric tests pass.

- [ ] **Step 3: Inspect the final diff** for event-schema stability, no accidental changes to promotion behavior, and no modifications to the user’s unrelated untracked files.

- [ ] **Step 4: Ask a final reviewer subagent** to check the implementation against this plan and report any required fixes before claiming completion.
