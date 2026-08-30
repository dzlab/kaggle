# Feed Solvency and Reproducibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make animal acquisition and placement reserve all remaining wheat feeds through the season, handle expiring farm hands safely, and make evaluator reports byte-stable across equivalent invocations.

**Architecture:** Keep the existing planner, policy, and evaluator boundaries. Add one shared planner-side solvency calculation that combines shed wheat, all worker-held wheat, already-planned purchases, animal feed demand, and post-purchase cash; gate both `BUY_ANIMAL` and placement on that calculation. Normalize only evaluator metadata fields that encode invocation-dependent paths/commands, preserving replay validation and aggregation behavior.

**Tech Stack:** Python 3.11+, pytest, uv, kaggle-environments 1.32.7, local Kaggriculture engine.

---

### Task 1: Capture feed-solvency and farm-hand expiry regressions

**Files:**
- Modify: `tests/test_policy.py`
- Modify: `tests/test_agent_smoke.py`
- Modify: `tests/test_evaluate.py`

- [ ] Add a planner test where an empty farm has enough money for one animal but not the animal plus the complete remaining feed reserve; assert no `BUY_ANIMAL` and no `ANIMAL` placement task.
- [ ] Add a planner test proving shed wheat, every worker inventory, and planned wheat purchases are counted exactly once before the animal gate.
- [ ] Add an engine-backed seed-0/starter 720 regression asserting no framework error and zero missed required needs.
- [ ] Add a report test that builds equivalent documents with different invocation paths and asserts normalized metadata is identical.
- [ ] Run the focused tests and confirm the new behavior tests fail before production changes.

### Task 2: Implement conservative planner solvency and safe worker boundaries

**Files:**
- Modify: `kagriculture_agent/planner.py`
- Modify: `kagriculture_agent/policy.py` only if the boundary check requires the existing assignment validity path to change.

- [ ] Introduce small private helpers for total staged wheat, animal count including tile/observed/storage state, remaining feed units, planned purchase cost, and post-purchase cash.
- [ ] Track planned purchases in intent order and require complete remaining WHEAT reserve before appending `BUY_ANIMAL`; use the same gate before emitting an `ANIMAL` placement task.
- [ ] Preserve mandatory wheat/fertilizer intent ordering for conservative behavior and avoid assuming a just-expired hand can execute a task; invalidate assignments whose worker is no longer visible or whose turn/day boundary has passed.
- [ ] Run the focused policy and engine regression tests until green, without changing evaluator architecture.

### Task 3: Normalize evaluator report metadata

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `README.md` only if the final report contract needs clarification.

- [ ] Normalize the command metadata to a stable repository-relative evaluator invocation and normalize replay sidecar metadata to a stable report-relative path.
- [ ] Keep actual CLI output paths in stdout and preserve explicit output location behavior.
- [ ] Verify report and sidecar bytes match across equivalent output directories and command invocation forms.

### Task 4: Full verification and commit

**Files:**
- Review all changed files and the plan checklist.

- [ ] Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q`.
- [ ] Run bounded all-variant 720 smokes against `pass`, `random`, and `starter`, checking framework errors and missed required needs.
- [ ] Inspect `git diff`, confirm no generated reports/replays are staged, and commit the implementation.

