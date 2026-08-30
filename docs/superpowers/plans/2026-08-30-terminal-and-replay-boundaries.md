# Terminal Scheduling and Replay Boundary Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the three seeded parent-suite behaviors without weakening replay validation: complete animal fertilizer flow, avoid terminal missed needs, and accept valid ongoing-crop harvests at natural day boundaries.

**Architecture:** Preserve the existing planner/policy/evaluator boundaries. Make carried-animal assignments remain valid when their remaining feed is staged in the shed, close crop planting before the final two season days so new plants do not create unavoidable needs, and derive boundary harvest expectations from the existing ordered action-plus-refresh model.

**Tech Stack:** Python 3.11+, pytest, uv, the local Kaggriculture engine, and the existing deterministic planner/policy/evaluator modules.

---

### Task 1: Preserve executable carried-animal placement

**Files:**
- Modify: `kagriculture_agent/policy.py`, `_assignment_valid()` in the assignment validation path.
- Test: `tests/test_policy.py`, add a regression for a carried animal whose feed is available in the shed.

- [x] **Step 1: Write the failing test**

  Add a policy-level test that creates a valid pasture target, gives worker 1 a carried sheep and the shed one wheat, then asserts `_assignment_valid()` accepts an `ANIMAL` assignment. The assignment must be accepted because `worker_action()` has an executable next step that picks up the staged wheat.

- [x] **Step 2: Run the focused test to verify it fails**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q tests/test_policy.py -k carried_animal`.
  Expected: the new assertion fails because `_assignment_valid()` falls through to `_task_action()` and treats the missing carried wheat as an invalid placement.

- [x] **Step 3: Implement the minimal policy fix**

  In the required-item branch of `_assignment_valid()`, special-case `ANIMAL` assignments that already carry the animal and have wheat in the shed. Return validity when the target is a compatible empty structure; leave all other required-item and state checks unchanged. This lets `worker_action()` execute its existing shed pickup path before placing and feeding the animal.

- [x] **Step 4: Run the focused test and the existing animal policy tests**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q tests/test_policy.py -k 'carried_animal or animal or feed'`.
  Expected: all selected tests pass.

### Task 2: Close planting before terminal days

**Files:**
- Modify: `kagriculture_agent/planner.py`, autonomous macro planting and daily empty-tile planting gates.
- Test: `tests/test_policy.py`, add a late-season planner regression.

- [x] **Step 1: Write the failing test**

  Add a test with `day=season_days - 2`, an empty unlocked tile, and a positive seed balance. Assert that neither `build_daily_plan()` nor `build_autonomous_macro_plan()` emits a `PLANT` task, while earlier-season input still permits planting through the existing tests.

- [x] **Step 2: Run the focused test to verify it fails**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q tests/test_policy.py -k late_season`.
  Expected: the new test fails because both planner paths currently schedule planting on day 28.

- [x] **Step 3: Implement the minimal scheduling fix**

  Change the two planting gates to require `day < season_days - 2`. Keep existing harvest, watering, feed/care, terminal cleanup, and liquidation behavior intact; only prevent starting crops in the final two days when no productive, needs-safe lifecycle remains. Keep staged fertilizer visible to the daily plan and let urgent `FERTILIZE` work use the farmer slot when the helper is reserved for basic needs, so the macro flow remains executable alongside animal care.

- [x] **Step 4: Run the focused planner and terminal tests**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q tests/test_policy.py -k 'late_season or terminal or liquidation or plant'`.
  Expected: all selected tests pass.

### Task 3: Model ongoing harvest plus natural boundary refresh

**Files:**
- Modify: `scripts/evaluate.py`, the `HARVEST` branch of `_transition_effects_valid()`.
- Test: `tests/test_evaluate.py`, add a direct boundary fixture for ongoing-crop harvest.

- [x] **Step 1: Write the failing test**

  Use the existing seed-4 random replay regression, whose ongoing crop is harvested on hour 23. Assert that `replay_record()` reports `framework_error` false and that the boundary tile retains the post-refresh yield.

- [x] **Step 2: Run the focused test to verify it fails**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q tests/test_evaluate.py -k 'boundary or natural'`.
  Expected: the test fails because `_transition_effects_valid()` requires the ongoing crop's post-harvest yield to equal `max(0, pre_yield - 1)` and rejects the engine's subsequent natural production.

- [x] **Step 3: Implement the minimal evaluator fix**

  For an ongoing crop harvested at an end-of-day boundary, validate the resulting tile against `_action_target_tile_sequence_expected()` so the action is applied first and `_daily_refresh_tile()` is applied once afterward. Retain the existing strict non-boundary harvest checks and the existing natural `WEED` allowances for unrelated tiles.

- [x] **Step 4: Run the focused evaluator tests**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q tests/test_evaluate.py -k 'boundary or natural or harvest'`.
  Expected: all selected tests pass, with malformed arbitrary boundary mutations still rejected by the existing tests.

### Task 4: Full verification and commit

**Files:**
- Verify: all changed files and generated-artifact status.

- [x] **Step 1: Run the exact required full suite**

  Run `UV_CACHE_DIR=/private/tmp/kaggriculture-uv-cache uv run pytest -q`.
  Expected: zero failures.

- [x] **Step 2: Inspect the diff and repository state**

  Run `git diff --check`, `git diff -- kagriculture_agent/policy.py kagriculture_agent/planner.py scripts/evaluate.py tests/test_policy.py tests/test_evaluate.py`, and `git status --short`. Confirm no replay/report artifacts are staged.

- [x] **Step 3: Commit the verified implementation**

  Stage only the implementation, regression tests, and this plan, then commit with `git commit -m "fix: preserve terminal needs and replay boundaries"`.

- [x] **Step 4: Verify the commit**

  Run `git status --short --branch` and `git log -1 --oneline`. Report the commit hash and the fresh full-suite result.
