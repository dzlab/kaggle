# Replay and Late-Season Policy Regression Fix Implementation Plan

**Status (2026-08-30): Focused implementation complete.** The requested
regressions pass. Full-suite and broader 720-step batch verification remain
unrun because the user requested that long runs stop.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make evaluator replay validation accept the engine’s legal same-tile fertilizer/harvest sequence, classify malformed unhashable crop values as framework errors, and prevent late-season wheat seed purchases when planting is suppressed.

**Architecture:** Preserve the existing replay transition model and policy intent pipeline. Narrowly extend the tile transition validator for an ordered `FERTILIZE` followed by `HARVEST`, guard crop lookups at the replay boundary, and make late-season seed fallback respect the planner’s planting cutoff. Add direct regressions plus the requested seeded 720 smoke.

**Tech Stack:** Python 3.11+, pytest, uv, local Kaggriculture engine.

---

### Task 1: Add failing evaluator regressions

**Files:**
- Modify: `tests/test_evaluate.py`

- [x] Add a direct replay fixture whose same tile is fertilized and then harvested in ordered commands, with the harvest result removing the finite crop, and assert `_transition_effects_valid()` accepts it.
- [x] Add a malformed replay fixture with `crop=[]` and assert `replay_record()` returns `framework_error=True` instead of raising `TypeError`.
- [x] Add the engine-backed `conservative`, `seed=17`, `opponent=random`, `steps=720` regression asserting no framework error.
- [x] Run each new test before implementation and confirm the behavior fails for the current code.

### Task 2: Fix replay transition and malformed-value handling

**Files:**
- Modify: `scripts/evaluate.py`

- [x] In the ordered tile transition validation, recognize a tile whose first action is `FERTILIZE` and whose later action is `HARVEST`; validate the fertilizer mutation before applying harvest removal, while retaining strict checks for unrelated or reversed actions.
- [x] Ensure crop-key lookups used by replay validation accept only hashable, valid crop names; malformed values such as lists must return invalid validation/framework-error rather than escaping as `TypeError`.
- [x] Run the focused evaluator tests and confirm the new regressions pass without weakening ordinary replay integrity checks.

### Task 3: Fix late-season wheat fallback and update plan status

**Files:**
- Modify: `kagriculture_agent/policy.py`
- Modify: `tests/test_policy.py`
- Modify: `tests/test_evaluate.py`
- Modify: `docs/superpowers/plans/2026-08-30-feed-solvency-reproducibility.md`

- [x] Add a day-28/day-29 policy regression with planting suppressed and no available wheat seed, asserting generated market orders contain no `BUY_SEED WHEAT`.
- [x] Change only the fallback branch that chooses a seed for an unsatisfied planting intent so it skips wheat when the late-season planting gate is closed; preserve earlier-season wheat fallback behavior.
- [x] Mark the older feed-solvency plan as superseded by the current implementation/tests.

### Task 4: Verify, inspect, and commit

**Files:**
- Review all changed files and generated-artifact status.

- [ ] Run `UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q`.
- [ ] Run a bounded 720-step smoke for the requested seed/opponent/variant combinations and verify framework errors and missed needs.
- [x] Run `git diff --check`, inspect the complete diff, ensure no reports or replay artifacts are staged, and commit only the focused source, tests, and plan changes.
