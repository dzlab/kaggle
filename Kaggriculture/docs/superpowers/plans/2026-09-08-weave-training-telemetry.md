# Optional Weave Training Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local JSONL-first, optionally Weave-backed training metrics for Kaggriculture BC/PPO runs without making Weave a runtime requirement or changing disabled-telemetry determinism.

**Architecture:** `scripts/telemetry.py` owns JSONL persistence, optional lazy Weave initialization, failure policy, and a callback-compatible event recorder. `train_policy.py` emits BC checkpoint/final events and PPO step events through one optional callback; notebook code constructs the recorder against Drive and plots its JSONL output. Weave is imported only after opt-in and is wrapped so non-strict failures warn and continue.

**Tech Stack:** Python 3.11, PyTorch, pytest, JSONL, optional W&B Weave, Google Colab, matplotlib.

---

### Task 1: Add local-first telemetry recorder

**Files:**
- Create: `scripts/telemetry.py`
- Test: `tests/test_telemetry.py`

- [x] Write tests for default project name, JSONL event persistence, fake Weave init/op calls, non-strict init/log failure continuation, strict failure, and empty-safe metric loading.
- [x] Run `Kaggriculture/.venv/bin/pytest -q Kaggriculture/tests/test_telemetry.py` and confirm the new API fails before implementation.
- [x] Implement `DEFAULT_WEAVE_PROJECT = "dzlab/kaggriculture"`, `TrainingTelemetry`, lazy optional import/injected fake module support, JSON-serializable event writing, safe Weave `init(project_name)` plus `@weave.op` event logging, and strict/non-strict error handling.
- [x] Run the telemetry tests and confirm they pass.

### Task 2: Emit BC and PPO progress events

**Files:**
- Modify: `scripts/train_policy.py`
- Modify: `tests/test_train_policy.py`

- [x] Add tests proving BC callback events contain loss/update count and PPO callback events contain policy/value/entropy/KL, update counts, rollout count, and early-stop status.
- [x] Run those focused tests and confirm they fail before the trainer changes.
- [x] Add optional `telemetry_callback` parameters without changing existing positional/API behavior; emit BC events at checkpoint cadence plus the final update, and PPO events at each progress step while retaining existing checkpoint `progress_fn` behavior.
- [x] Run the new trainer tests plus the existing trainer suite.

### Task 3: Wire packaging and Colab observability

**Files:**
- Modify: `pyproject.toml`
- Modify: `notebooks/colab_orbit_gpu.ipynb`
- Modify: `tests/test_colab_train.py`

- [x] Add an `observability` optional dependency containing Weave.
- [x] Update Colab installation to include the optional extra, construct Drive-backed telemetry for the exact default project, pass its callback into training, and add an empty-safe matplotlib plotting cell.
- [x] Add notebook source assertions for installation, project, JSONL path, callback, and plotting behavior; run the Colab tests.

### Task 4: Verify and commit

**Files:**
- Verify all changed files and the final diff.

- [x] Run the bounded focused suite and syntax/JSON checks; do not run training or the full long suite.
- [x] Review `git diff --check` and `git diff`, preserving unrelated untracked files.
- [ ] Commit all implementation changes with a clear telemetry-focused message and report the commit SHA and changed files.
