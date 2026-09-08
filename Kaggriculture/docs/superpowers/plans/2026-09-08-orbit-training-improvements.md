# Orbit-Style Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Kaggriculture policy learning, rollout quality, opponent diversity, evaluation evidence, and experimental feature/scaling support without weakening the safe production path.

**Architecture:** Add small pure-Python helpers for conditional action objectives, reward shaping, league metrics, and experimental context features. Integrate them into the existing trainer/evaluator behind explicit configuration, while retaining the current artifact format and deterministic fallback as defaults. Use fresh rollouts and checkpoint metadata to keep experiments reproducible.

**Tech Stack:** Python 3, PyTorch, pytest, existing Kaggriculture engine/runner, JSON/JSONL manifests.

---

## Task 1: Conditional action objectives and legality-aware execution

**Files:**
- Create: `Kaggriculture/kagriculture_agent/action_objectives.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/kagriculture_agent/learned_policy.py` only if shared runtime mask validation is needed
- Test: `Kaggriculture/tests/test_action_objectives.py`
- Test: `Kaggriculture/tests/test_train_policy.py`

- [ ] **Step 1: Write failing tests for conditional branches.**

Add tests with one `PASS` worker and one active worker. Assert inactive target/kind logits do not affect conditional log probability, entropy, or PPO ratio. Add a no-market-order test and mask shape/empty-legal-set validation tests.

- [ ] **Step 2: Run the focused tests and verify expected failures.**

```bash
cd Kaggriculture && pytest -q tests/test_action_objectives.py tests/test_train_policy.py -k 'conditional or mask'
```

Expected: the new tests fail because the helper does not exist or the trainer still includes inactive branches.

- [ ] **Step 3: Implement the minimal pure helper.**

Implement a typed helper returning `(log_probs, entropy)` from the existing output dictionary and `RolloutBatch` labels. Always include active-worker probability; include kind/target only for active workers; include market terms only when a market action exists. Apply optional legality masks before `log_softmax`, reject malformed masks, and preserve unmasked behavior when omitted.

- [ ] **Step 4: Integrate the helper into BC and PPO.**

Use the same conditional semantics in behavior cloning and `_select_outputs`. Add validated `training_action_mask: bool = False` to `PPOConfig` and checkpoint configuration. Keep runtime compilation hard-safe and preserve deterministic fallback behavior.

- [ ] **Step 5: Run focused tests.**

```bash
cd Kaggriculture && pytest -q tests/test_action_objectives.py tests/test_train_policy.py
```

- [ ] **Step 6: Commit.**

```bash
git add Kaggriculture/kagriculture_agent/action_objectives.py Kaggriculture/scripts/train_policy.py Kaggriculture/tests/test_action_objectives.py Kaggriculture/tests/test_train_policy.py
git commit -m "feat(kagriculture): make policy objectives conditional"
```

## Task 2: Potential shaping and no-progress episode handling

**Files:**
- Create: `Kaggriculture/kagriculture_agent/reward_shaping.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/scripts/collect_trajectories.py`
- Modify: `Kaggriculture/scripts/train_controller.py`
- Test: `Kaggriculture/tests/test_reward_shaping.py`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests.**

Test finite-zero behavior for malformed economic inputs, zero shaping for identical states, the exact `gamma * phi(next) - phi(current)` difference, additive terminal bank reward, and bootstrap truncation for a configured no-progress window.

- [ ] **Step 2: Run focused tests and verify failures.**

```bash
cd Kaggriculture && pytest -q tests/test_reward_shaping.py tests/test_train_policy.py -k 'reward or truncat or progress'
```

- [ ] **Step 3: Implement bounded potential shaping.**

Create pure functions:

```python
def economic_potential(observation: Mapping[str, Any]) -> float: ...
def potential_difference(current: Mapping[str, Any], next_state: Mapping[str, Any], *, gamma: float) -> float: ...
def shaped_transition_reward(transition: Mapping[str, Any], *, gamma: float, coefficient: float) -> float: ...
```

Use normalized cash, inventory value, production, worker utilization, and deadline risk. Clamp components and final values; never replace terminal bank-margin reward.

- [ ] **Step 4: Add explicit PPO configuration.**

Extend `PPOConfig` with validated `potential_reward_coef`, `no_progress_window`, and `resolved_margin`. Add shaping only when configured. Ensure bootstrap truncations use `done=False` in GAE and genuine terminal transitions remain `done=True`.

- [ ] **Step 5: Thread metadata through collection/controller.**

Validate and preserve `termination_reason`, `bootstrap_truncated`, and `no_progress_steps`. Failed/malformed games remain invalid. Add shaping/truncation counts to PPO metrics and telemetry.

- [ ] **Step 6: Run tests and commit.**

```bash
cd Kaggriculture && pytest -q tests/test_reward_shaping.py tests/test_train_policy.py tests/test_trajectory.py
git add Kaggriculture/kagriculture_agent/reward_shaping.py Kaggriculture/scripts/train_policy.py Kaggriculture/scripts/collect_trajectories.py Kaggriculture/scripts/train_controller.py Kaggriculture/tests/test_reward_shaping.py Kaggriculture/tests/test_train_policy.py Kaggriculture/tests/test_trajectory.py
git commit -m "feat(kagriculture): add shaped rewards and stall handling"
```

## Task 3: Adaptive opponent league and evaluation metrics

**Files:**
- Create: `Kaggriculture/kagriculture_agent/league.py`
- Create: `Kaggriculture/scripts/evaluation_metrics.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/scripts/evaluate.py`
- Modify: `Kaggriculture/scripts/train_controller.py`
- Test: `Kaggriculture/tests/test_league.py`
- Test: `Kaggriculture/tests/test_evaluation_metrics.py`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_evaluate.py`

- [ ] **Step 1: Write failing league tests.**

Test deterministic `(seed, index)` sampling, balanced seats, historical checkpoint skill bands, empty-band fallback, and adaptive hard-opponent weighting without changing total schedule size.

- [ ] **Step 2: Write failing metric tests.**

Test paired-seed aggregation, percentiles/lower-tail bank values, Wilson win-rate bounds, a deterministic Bradley–Terry/Elo summary, and rejection of a safety-regressing candidate.

- [ ] **Step 3: Run focused tests and verify failures.**

```bash
cd Kaggriculture && pytest -q tests/test_league.py tests/test_evaluation_metrics.py tests/test_train_policy.py tests/test_evaluate.py -k 'league or elo or confidence or percentile or promotion'
```

- [ ] **Step 4: Implement league sampling.**

Move deterministic sampling rules into `league.py` with explicit opponent/skill-band data while preserving `OpponentMatch` compatibility. Sample uniformly inside selected checkpoint bands; accept hard-opponent weighting only from explicit schedule/metric input; never treat a missing path as a valid checkpoint.

- [ ] **Step 5: Implement and integrate metrics.**

Add pure paired-outcome, Wilson-bound, quantile, and Elo functions. Extend evaluation JSON without removing existing fields. Require existing safety gates plus the configured confidence-aware improvement gate; keep fixed 100-game behavior when the new gate is unset.

- [ ] **Step 6: Run tests and commit.**

```bash
cd Kaggriculture && pytest -q tests/test_league.py tests/test_evaluation_metrics.py tests/test_train_policy.py tests/test_evaluate.py
git add Kaggriculture/kagriculture_agent/league.py Kaggriculture/scripts/evaluation_metrics.py Kaggriculture/scripts/train_policy.py Kaggriculture/scripts/evaluate.py Kaggriculture/scripts/train_controller.py Kaggriculture/tests/test_league.py Kaggriculture/tests/test_evaluation_metrics.py Kaggriculture/tests/test_train_policy.py Kaggriculture/tests/test_evaluate.py
git commit -m "feat(kagriculture): improve league sampling and evaluation evidence"
```

## Task 4: Experimental context features and scaling benchmark

**Files:**
- Create: `Kaggriculture/kagriculture_agent/experimental_features.py`
- Create: `Kaggriculture/scripts/benchmark_training_ladder.py`
- Modify: `Kaggriculture/scripts/telemetry.py` only if a shared event shape is necessary
- Modify: `Kaggriculture/README.md`
- Test: `Kaggriculture/tests/test_experimental_features.py`
- Test: `Kaggriculture/tests/test_benchmark_training_ladder.py`

- [ ] **Step 1: Write failing feature-variant tests.**

Test bounded recent-action identity/outcome, demand/price trend, recovery slack, and task-opportunity values when supplied. Test missing history yields finite neutral values. Test the experimental variant identifier is distinct and default feature shapes/schema remain unchanged.

- [ ] **Step 2: Write failing benchmark tests.**

Test ladder parsing, deterministic seed expansion, stable JSON output, rejection of production artifact output paths, and `--dry-run` without GPU imports.

- [ ] **Step 3: Run focused tests and verify failures.**

```bash
cd Kaggriculture && pytest -q tests/test_experimental_features.py tests/test_benchmark_training_ladder.py
```

- [ ] **Step 4: Implement the opt-in feature variant.**

Keep `extract_features` and the current artifact schema unchanged. Expose `extract_experimental_context` and `EXPERIMENTAL_FEATURE_VARIANT`. Use only observation-derived signals; do not read private state or alter deterministic policy behavior.

- [ ] **Step 5: Implement the bounded ladder harness.**

Add a CLI that accepts a JSON ladder, expands deterministic seeds, reports parameter estimates and rollout budgets, and optionally invokes an injected training/evaluation callback. Write reports atomically. Default to dry-run and reject production model/checkpoint outputs.

- [ ] **Step 6: Document, test, and commit.**

Document dry-run/report commands in `Kaggriculture/README.md`, then run:

```bash
cd Kaggriculture && pytest -q tests/test_experimental_features.py tests/test_benchmark_training_ladder.py
git add Kaggriculture/kagriculture_agent/experimental_features.py Kaggriculture/scripts/benchmark_training_ladder.py Kaggriculture/README.md Kaggriculture/tests/test_experimental_features.py Kaggriculture/tests/test_benchmark_training_ladder.py
git commit -m "feat(kagriculture): add opt-in training context and scaling ladder"
```

## Task 5: Integration, artifact safety, and final verification

**Files:**
- Modify only files required by failing integration tests
- Test: all `Kaggriculture/tests/`
- Verify: artifact evaluator, submission smoke path, and `git diff --check`

- [ ] **Step 1: Run the full test suite.**

```bash
cd Kaggriculture && pytest -q
```

Investigate every new failure and record the exact result.

- [ ] **Step 2: Run compile and artifact checks.**

```bash
cd Kaggriculture && python3 -m compileall -q kagriculture_agent scripts tests
python3 scripts/evaluate_artifact.py --help
git diff --check
```

- [ ] **Step 3: Verify production immutability.**

Confirm default feature schema, `models/learned_v1.json`, deterministic policy, and production artifact paths are unchanged unless an explicit export command is invoked. Run existing artifact and submission smoke tests.

- [ ] **Step 4: Review the plan against the approved design.**

Check conditional objectives, shaping, truncation, league diversity, confidence-aware evaluation, experimental features, scaling harness, safe fallback, and holdout separation. Fix any gap before claiming completion.

- [ ] **Step 5: Inspect intentional changes only.**

```bash
git status --short
git diff --stat
```

Do not add the pre-existing untracked `CR-ppo-resume-aface-parent.md` or `CR-ppo-resume-ext-parent.md` files.
