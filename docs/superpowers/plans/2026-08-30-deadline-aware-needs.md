# Deadline-Aware Basic Needs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the conservative seed-17/random 720-turn engine replay satisfies every required watering and feeding deadline.

**Architecture:** Keep replay metrics strict and preserve the existing late-season planting gate and terminal liquidation window. Extend the planner/policy scheduling boundary so late-season required needs are assigned enough workers before a day expires, then verify the exact engine replay through a regression and the full suite.

**Tech Stack:** Python 3.11+, pytest, uv, and the local Kaggriculture engine.

---

### Task 1: Capture the failing engine replay

**Files:**
- Modify: `tests/test_evaluate.py`

- [x] Add an engine-backed `conservative`, seed `17`, random-opponent, `720`-turn test asserting both `framework_error is False` and `missed_basic_needs == 0`.
- [x] Run the focused test and confirm the current replay fails only on `missed_basic_needs`.

### Task 2: Make deadline scheduling needs-aware

**Files:**
- Modify: `kagriculture_agent/planner.py`
- Modify: `scripts/evaluate.py`
- Modify: `tests/test_policy.py` or `tests/test_routing.py`

- [x] Add the smallest policy/planner regression proving a final actionable day with multiple unmet watering/feeding tasks gets sufficient worker coverage while late-season planting remains suppressed.
- [x] Implement deadline-aware worker coverage for required needs, including the final actionable day, without scheduling seed purchases in the late-season gate or selling before terminal liquidation is safe.
- [x] Run the focused unit and engine regression tests.

### Task 3: Verify and commit

**Files:**
- Review all changed source, tests, and plan files.

- [x] The user requested that the broad/full run not be started; the focused checks below were run instead.
- [x] Run the targeted 720-turn conservative/random smoke and inspect `framework_error` and `missed_basic_needs`.
- [x] Run `git diff --check`, inspect the diff, and commit only focused changes.
