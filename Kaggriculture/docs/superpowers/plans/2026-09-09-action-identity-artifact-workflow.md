# Action Identity and Artifact Workflow Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve action-representation identity through Colab training, artifact evaluation, and export while making evaluation report publication safe and atomic.

**Architecture:** Keep the existing production default (`current_v1`) and schemas. Forward the configured action representation at the Colab contract boundary, route report writes through the shared output-path validator/atomic publisher, make required nested identity checks executable, and compare checkpoint top-level and nested action identity during export.

**Tech Stack:** Python 3.11+, pytest, pathlib, existing `kagriculture_agent.output_paths` and `scripts.training_identity` helpers.

---

### Task 1: Colab contract identity propagation

**Files:**
- Modify: `Kaggriculture/scripts/train.py:1498-1521`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [ ] **Step 1: Write the failing test**

Add a workflow test that builds a `target_first_v1` config, supplies a minimal trajectory, stubs collection/training/evaluation side effects, and asserts the captured `TrainingContract.configuration` contains `action_representation == "target_first_v1"`.

- [ ] **Step 2: Run the focused test to verify it fails**

Run `uv run pytest tests/test_colab_train.py -k target_first_contract -q` from `Kaggriculture/` and confirm the contract construction raises the action-representation conflict before the implementation change.

- [ ] **Step 3: Write the minimal implementation**

Pass `action_representation=config.action_representation` to `train_policy.build_training_contract(...)` in `run_workflow`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run `uv run pytest tests/test_colab_train.py -k target_first_contract -q` and confirm it passes.

### Task 2: Safe atomic evaluation report publication and reachable identity validation

**Files:**
- Modify: `Kaggriculture/scripts/evaluate_artifact.py:24-31,579-597`
- Modify: `Kaggriculture/scripts/train.py:1389-1399`
- Test: `Kaggriculture/tests/test_evaluate_artifact.py`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [ ] **Step 1: Write failing tests**

Add report-writer tests proving `write_report` rejects `models/learned_v1.json`, rejects a symlinked parent, and leaves an existing destination unchanged; add a Colab identity-document test proving `require_configuration=True` rejects a configuration object missing a required non-default identity field.

- [ ] **Step 2: Run focused tests to verify they fail**

Run `uv run pytest tests/test_evaluate_artifact.py -k 'report_writer or protected or symlink' -q` and `uv run pytest tests/test_colab_train.py -k identity_document -q`; confirm the unsafe report writes and unreachable missing-field validation are exposed.

- [ ] **Step 3: Write the minimal implementation**

Import `atomic_write_text` from `scripts.output_paths`, serialize the report with the existing canonical JSON options, and delegate publication to `atomic_write_text`. Move the required nested-field check outside the mismatch-only branch while retaining the default-action backward-compatibility exception.

- [ ] **Step 4: Run focused tests to verify they pass**

Run the two focused pytest commands again and confirm they pass.

### Task 3: CLI forwarding and export identity conflict validation

**Files:**
- Modify: `Kaggriculture/scripts/evaluate_artifact.py:756-770`
- Modify: `Kaggriculture/scripts/export_policy.py:31-36,142-159`
- Test: `Kaggriculture/tests/test_evaluate_artifact.py`
- Test: `Kaggriculture/tests/test_export_policy.py`

- [ ] **Step 1: Write failing tests**

Add a CLI test that captures `evaluate(...)` arguments and asserts `--action-representation target_first_v1` reaches it. Add an exporter metadata test with conflicting top-level `current_v1` and nested `ppo_config.action_representation == target_first_v1` and assert it raises a conflict error.

- [ ] **Step 2: Run focused tests to verify they fail**

Run `uv run pytest tests/test_evaluate_artifact.py -k cli_action_representation -q` and `uv run pytest tests/test_export_policy.py -k conflicting_action_representation -q`; confirm each currently passes the wrong identity or accepts the conflict.

- [ ] **Step 3: Write the minimal implementation**

Forward `action_representation=args.action_representation` in the CLI call to `evaluate`. In checkpoint metadata validation, validate the optional nested `ppo_config` object’s action identity against the top-level metadata with `validate_identity_consistency`, then retain the existing default and allowed-value validation.

- [ ] **Step 4: Run focused tests to verify they pass**

Run both focused pytest commands again and confirm they pass.

### Task 4: Full scoped verification and commit

**Files:**
- Verify only the requested scripts, shared helpers, and focused tests changed; do not modify `Kaggriculture/scripts/compare_experiments.py`.

- [ ] **Step 1: Run the focused regression suite**

Run `uv run pytest tests/test_colab_train.py tests/test_evaluate_artifact.py tests/test_export_policy.py -q` from `Kaggriculture/`.

- [ ] **Step 2: Inspect the diff and scope**

Run `git diff --check`, `git diff --stat`, and `git diff -- Kaggriculture/scripts/compare_experiments.py`; confirm no compare-experiments changes and no unrelated tracked modifications.

- [ ] **Step 3: Commit the focused changes**

Run `git add Kaggriculture/scripts/train.py Kaggriculture/scripts/evaluate_artifact.py Kaggriculture/scripts/export_policy.py Kaggriculture/kagriculture_agent/output_paths.py Kaggriculture/scripts/output_paths.py Kaggriculture/scripts/training_identity.py Kaggriculture/tests/test_colab_train.py Kaggriculture/tests/test_evaluate_artifact.py Kaggriculture/tests/test_export_policy.py && git commit -m "fix(kaggriculture): preserve action identity artifact safety"`.

- [ ] **Step 4: Verify the committed state**

Run `git status --short --branch` and `git rev-parse HEAD`, then report the commit SHA and exact test command/output.
