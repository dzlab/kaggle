# Kaggriculture Parent Regression Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the four parent-suite behaviors at HEAD 323f65a without weakening replay or legality validation.

**Architecture:** Keep policy scheduling, planner task deadlines, and evaluator postprocessing separate. Fix the carried-assignment key normalization in the policy, make optional fertilizer scheduling executable from staged shed inventory, preserve compact fixture orders when complete market state is absent, and retain the input order while sanitizing quantities against complete state.

**Tech Stack:** Python 3.11+, uv, pytest, and the local Kaggriculture engine.

---

### Task 1: Capture focused regressions

**Files:** `tests/test_policy.py`, `tests/test_evaluate.py`

- [ ] Add policy assertions for one protected carried feed task and for staged fertilizer assignment after the same-day route budget has elapsed.
- [ ] Run those tests and the existing compact melon and market-order tests to confirm the baseline failures.

### Task 2: Implement minimal fixes

**Files:** `kagriculture_agent/policy.py`, `kagriculture_agent/planner.py`, `scripts/evaluate.py`

- [ ] Normalize optional task items consistently when filtering protected assignments.
- [ ] Remove the artificial same-day deadline from optional `FERTILIZE` tasks.
- [ ] Apply melon capacity sanitization only when complete market state is present.
- [ ] Order safety market operations as stable `HIRE`, `BUY_LAND`, then product orders while preserving per-unit affordability and shed limits.

### Task 3: Verify and commit

- [ ] Run focused regressions, then `UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q`.
- [ ] Inspect the diff, commit only the focused source/tests/plan changes, and report the commit and test result.
