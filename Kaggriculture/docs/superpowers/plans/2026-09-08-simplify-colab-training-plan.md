# Simplify Colab Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Make the Colab notebook a thin setup-and-launch wrapper while moving the complete resumable training workflow into a parameterized helper script.

**Architecture:** `notebooks/colab_gpu.ipynb` will clone the selected repository branch, install the training dependencies, and invoke `scripts/train.py`. The helper script will own Drive setup, trajectory collection, training/resume, telemetry, evaluation gates, artifact smoke testing, and optional plotting. Existing resume validation and safety gates will remain reusable and directly testable.

**Tech Stack:** Python, argparse, JSON notebooks, pytest, PyTorch, W&B, Google Colab runtime APIs.

---

### Task 1: Extract the Colab workflow into a CLI helper

**Files:**
- Create: `Kaggriculture/scripts/train.py`
- Create: `Kaggriculture/scripts/colab_train.py` compatibility shim
- Test: `Kaggriculture/tests/test_colab_train.py`

- [x] Add failing tests for CLI parameter parsing, command construction, and a dry-run mode that does not require Colab, Drive, W&B, or the simulator.
- [x] Run the focused tests and confirm they fail for the missing orchestration behavior.
- [x] Implement the workflow as small testable functions and a `main()` CLI, preserving resume selection, safety gates, telemetry, artifact export, and CPU fallback.
- [x] Add explicit parameters for repository-independent run paths, PPO target, training settings, seeds/opponents, device, workers, W&B, evaluation, smoke testing, and plotting.
- [x] Run the focused tests and confirm they pass.

### Task 2: Reduce the notebook to setup and launch

**Files:**
- Modify: `Kaggriculture/notebooks/colab_gpu.ipynb`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [x] Replace the orchestration cells with clone, dependency installation, and one parameterized helper-script launch cell.
- [x] Keep branch, repository, and all training parameters visible in the launch command.
- [x] Add a notebook-shape test that rejects embedded training/evaluation logic and requires the helper invocation.
- [x] Run the notebook tests and inspect the generated notebook structure.

### Task 3: Documentation and verification

**Files:**
- Modify: `Kaggriculture/README.md`

- [x] Document the simplified Colab flow and the complete CLI command for reruns/resumes.
- [x] Run the focused Colab tests, the full Kaggriculture test suite, `compileall`, and `git diff --check`.
- [x] Review the diff for unrelated changes and preserve the pre-existing `CR-ppo-resume-*` files.
