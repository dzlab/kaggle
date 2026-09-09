# Evaluation Comparison Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable per-opponent evaluation evidence, strict matrix validation, experiment comparison output, and validated W&B summaries while preserving existing report fields.

**Architecture:** Keep statistical calculations pure in `scripts/evaluation_metrics.py`; have `scripts/evaluate.py` adapt the existing records into per-opponent metrics and promotion evidence without removing legacy summaries. `scripts/compare_experiments.py` will validate manifest-derived matrix signatures before producing JSON and Markdown deltas. Telemetry will write all local events as before but mirror only schema-complete validation summaries to W&B.

**Tech Stack:** Python 3.11, pytest, JSON/Markdown CLI output, existing optional W&B adapter.

---

### Task 1: Establish pure metric and matrix-validation contracts

**Files:**
- Modify: `scripts/evaluation_metrics.py`
- Test: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: Write failing tests** for a per-opponent summary containing `wins`, `losses`, `ties`, `valid`, `seat_balanced_win_rate`, Wilson bounds, mean/lower-tail bank differential, framework/invalid/timeout/no-progress counts, and rating uncertainty; add tests that reject duplicate, missing, extra, and malformed matrix coordinates.
- [ ] **Step 2: Run the focused tests and confirm they fail** because the new summary and strict validator are absent.
- [ ] **Step 3: Implement the smallest pure helpers** for valid-record classification, deterministic per-opponent paired summaries, strict coordinate validation, safety-rate calculation, and uncertainty-aware Bradley–Terry/Elo summaries. Preserve all existing function names and return fields.
- [ ] **Step 4: Run the focused metric tests and the existing metric suite** and confirm all pass.

### Task 2: Extend evaluator reports without breaking legacy fields

**Files:**
- Modify: `scripts/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests** that build a report from complete paired records and assert `metrics_by_opponent`, `promotion_evidence`, strict matrix rejection, and stable timeout/no-progress/invalid counts.
- [ ] **Step 2: Run those tests and confirm the expected missing-key or validation failures.**
- [ ] **Step 3: Add per-opponent summaries** to the result document for development and holdout data, preserving `results`, `paired_summaries`, and `promotion_decisions` exactly as compatibility views.
- [ ] **Step 4: Add `promotion_evidence`** as a validated, JSON-safe projection of decisions, matrix completeness, safety counts, and per-opponent metrics; ensure incomplete or invalid coordinates produce discard evidence and never a promotion.
- [ ] **Step 5: Run the focused evaluator tests plus the existing evaluator suite.**

### Task 3: Add local experiment comparison CLI

**Files:**
- Create: `scripts/compare_experiments.py`
- Create: `tests/test_compare_experiments.py`

- [ ] **Step 1: Write failing tests** for two compatible reports, three-report baseline deltas, JSON/Markdown output, and incompatible seed/opponent/seat/matrix rejection.
- [ ] **Step 2: Run the comparison tests and confirm they fail because the module and CLI are absent.**
- [ ] **Step 3: Implement report loading, manifest/matrix signature validation, candidate/opponent metric extraction with legacy fallback, and deterministic deltas for win rate, Elo, bank differential, and safety rate.
- [ ] **Step 4: Implement CLI options for report paths plus JSON and Markdown destinations, with nonzero rejection for incompatible matrices and a stable Markdown table headed `candidate | opponent | win_rate_delta | elo_delta | bank_delta | safety_delta | decision`.
- [ ] **Step 5: Run the comparison tests and the full focused reporting test set.**

### Task 4: Gate W&B on validated summaries

**Files:**
- Modify: `scripts/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write failing tests** proving complete matrix summaries are mirrored with aggregate/per-opponent fields and incomplete or invalid reports remain local-only for W&B.
- [ ] **Step 2: Run the telemetry tests and confirm the W&B gating assertion fails.**
- [ ] **Step 3: Add a strict report-summary validation helper** that checks promotion evidence/matrix completeness and emits only validated aggregate and per-opponent summary events to W&B; retain existing local JSONL game and summary events and legacy flattening.
- [ ] **Step 4: Run telemetry, evaluator, metric, and comparison tests together.**

### Task 5: Verify and commit the focused change

**Files:**
- Modify only the files above and their focused tests.

- [ ] **Step 1: Run the complete project test suite and `git diff --check`.**
- [ ] **Step 2: Inspect the diff/status to confirm the roadmap and unrelated untracked files are untouched.**
- [ ] **Step 3: Commit the focused implementation with `feat(kagriculture): compare policies with paired strength metrics`.**
- [ ] **Step 4: Report the commit SHA, changed files, and exact test commands/results.**
