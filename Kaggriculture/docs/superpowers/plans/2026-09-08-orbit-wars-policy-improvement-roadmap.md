# Orbit Wars Policy Improvement Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Orbit Wars lessons into a reproducible Kaggriculture training program that produces stronger policies, measures actual game-winning ability, and never promotes an unvalidated candidate.

**Architecture:** Keep the dependency-light learned-policy artifact and deterministic fallback unchanged. Add explicit training variants to the existing `train.py` workflow, use fresh league rollouts against current and historical policies, compare candidates on paired development/holdout matrices, and record all results in local JSON/JSONL plus W&B. The work is split into code integration, experiment execution, and evidence-based promotion; each stage is independently testable.

**Tech Stack:** Python 3, PyTorch, NumPy, kaggle-environments 1.32.7, pytest, JSON/JSONL, Google Colab GPU, Google Drive, W&B.

---

## Current repository map

- `Kaggriculture/scripts/train.py`: Colab-facing configuration, CLI, checkpoint selection, training orchestration, telemetry, and development/holdout gates.
- `Kaggriculture/scripts/train_policy.py`: behavior-cloning/PPO implementation, PPO configuration, fresh rollout integration, checkpoint metadata, and historical-opponent pool.
- `Kaggriculture/kagriculture_agent/league.py`: deterministic current/random/starter/historical opponent sampling and skill bands.
- `Kaggriculture/kagriculture_agent/reward_shaping.py`: bounded economic potential shaping.
- `Kaggriculture/scripts/collect_trajectories.py`: reproducible rollout collection and no-progress/resolved episode metadata.
- `Kaggriculture/scripts/evaluate.py`: paired-seat evaluation, safety gates, development/holdout reports, and promotion decisions.
- `Kaggriculture/scripts/evaluation_metrics.py`: confidence intervals, lower-tail metrics, paired summaries, and Bradley–Terry calculations.
- `Kaggriculture/kagriculture_agent/experimental_features.py`: opt-in observation-derived temporal/action context, currently isolated from the production feature schema.
- `Kaggriculture/scripts/benchmark_training_ladder.py`: dry-run-first scaling experiment estimates.
- `Kaggriculture/scripts/telemetry.py`: JSONL and W&B training/validation logging.
- `Kaggriculture/notebooks/colab_gpu.ipynb`: clone/install/launch notebook; it should remain a thin launcher.

The existing modules above must be audited first. Modify them only where the tasks below identify an integration gap; do not recreate already-present helpers.

## Experiment contract

Every experiment must carry these immutable identifiers in its run manifest, checkpoint metadata, evaluation report, and W&B config:

```python
{
    "experiment_id": "bc-ppo-league-v1",
    "feature_variant": "production_v1",
    "training_mode": "behavior_clone_then_ppo",
    "training_seed": 7,
    "collection_seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "development_seeds": list(range(50)),
    "holdout_seeds": list(range(100, 150)),
    "opponent_schedule": "league_v1",
    "ppo_steps": 128,
}
```

The production artifact is changed only by an explicit promotion operation after the complete development gate and separate holdout gate succeed.

### Task 1: Audit existing safeguards and add experiment identity

**Files:**
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/scripts/telemetry.py`
- Modify: `Kaggriculture/scripts/evaluate.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/notebooks/colab_gpu.ipynb` only for launch arguments
- Test: `Kaggriculture/tests/test_colab_train.py`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_evaluate.py`
- Test: `Kaggriculture/tests/test_telemetry.py`

- [ ] **Step 1: Write failing tests for experiment identity propagation.**

Add tests that build a configuration with `experiment_id`, `feature_variant`, and `training_mode`, then assert that the values appear unchanged in the training contract, checkpoint metadata, evaluation configuration, JSONL events, and W&B initialization config. Assert that two runs with different identities cannot resume from one another unless all training-contract fields match.

- [ ] **Step 2: Run the focused tests and verify the expected failures.**

```bash
cd Kaggriculture
.venv/bin/pytest -q tests/test_colab_train.py tests/test_train_policy.py tests/test_evaluate.py tests/test_telemetry.py -k 'experiment or identity or config'
```

Expected: failures because the current configuration does not expose one shared experiment identity across all stages.

- [ ] **Step 3: Add the explicit configuration fields and CLI options.**

Add validated fields to `ColabConfig` and the training contract:

```python
experiment_id: str = "orbit-policy-v1"
feature_variant: str = "production_v1"
training_mode: str = "behavior_clone_then_ppo"
```

Add `--experiment-id`, `--feature-variant`, and `--training-mode` to `train.py`. Accept only `production_v1` and `experimental_context_v1` for the feature variant, and `behavior_clone_then_ppo`, `pure_ppo`, and `reduced_behavior_clone_then_ppo` for the training mode.

- [ ] **Step 4: Include identity in metadata and telemetry.**

Pass the fields into checkpoint configuration, rollout manifests, evaluation configuration, W&B config, and every validation event. Reject a resume checkpoint whose identity or feature variant differs from the requested run.

- [ ] **Step 5: Update the Colab launcher without adding workflow logic to the notebook.**

Keep the notebook limited to repository clone, dependency installation, secret loading, and a single `python scripts/train.py ...` invocation. Put all new configuration in CLI flags.

- [ ] **Step 6: Run focused tests and commit.**

```bash
.venv/bin/pytest -q tests/test_colab_train.py tests/test_train_policy.py tests/test_evaluate.py tests/test_telemetry.py
git diff --check
git add scripts/train.py scripts/telemetry.py scripts/evaluate.py scripts/train_policy.py notebooks/colab_gpu.ipynb tests
git commit -m "feat(kagriculture): propagate training experiment identity"
```

### Task 2: Wire the league opponent schedule into actual PPO rollouts

**Files:**
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/kagriculture_agent/league.py`
- Modify: `Kaggriculture/scripts/collect_trajectories.py`
- Test: `Kaggriculture/tests/test_league.py`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_rollouts.py`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [ ] **Step 1: Write failing tests for a configured league schedule.**

Given a seed, count, and checkpoint list, assert that the schedule is deterministic, alternates seats, includes current/random/starter/mixed/checkpoint matches according to configured weights, records the selected checkpoint hash, and falls back to current when a checkpoint is unavailable.

Test that the production defaults retain the existing safe schedule when no league flags are supplied.

- [ ] **Step 2: Add schedule configuration to `train.py`.**

Expose:

```text
--league-checkpoints PATH   (repeatable)
--league-checkpoint-window N
--league-current-probability FLOAT
--league-mixed-probability FLOAT
--league-random-probability FLOAT
--league-starter-probability FLOAT
--league-checkpoint-probability FLOAT
```

Validate nonnegative weights and require a positive total. Normalize only inside the sampler, while recording the original values in the run manifest.

- [ ] **Step 3: Make historical checkpoints a real rollout source.**

Use `LeagueSampler` in `OpponentPool` for every PPO round. Pass `checkpoint`, `checkpoint_identity`, `opponent_identity`, `seat`, and the round seed through the rollout callback. Keep no more than the requested recent checkpoint window, and reject incompatible checkpoints using the existing checkpoint validator.

- [ ] **Step 4: Record league composition.**

Add per-round counts to training telemetry:

```python
{
    "league/current": 12,
    "league/mixed": 4,
    "league/random": 3,
    "league/starter": 3,
    "league/checkpoint": 8,
}
```

Also log the checkpoint identities used in the rollout manifest so later comparisons are reproducible.

- [ ] **Step 5: Run tests, one-round integration, and commit.**

```bash
.venv/bin/pytest -q tests/test_league.py tests/test_train_policy.py tests/test_rollouts.py tests/test_colab_train.py
.venv/bin/python scripts/train.py --dry-run --league-checkpoints /tmp/policy-old.pt --league-checkpoint-window 5
git diff --check
git add scripts/train.py scripts/train_policy.py kagriculture_agent/league.py scripts/collect_trajectories.py tests
git commit -m "feat(kagriculture): train PPO against a reproducible policy league"
```

### Task 3: Add the training-mode and scaling ladder

**Files:**
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/kagriculture_agent/model.py`
- Modify: `Kaggriculture/scripts/benchmark_training_ladder.py`
- Create: `Kaggriculture/configs/orbit_policy_ladder.json`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_benchmark_training_ladder.py`
- Test: `Kaggriculture/tests/test_model.py`

- [ ] **Step 1: Write failing tests for the three training modes.**

Assert that:

- `behavior_clone_then_ppo` executes the configured BC epochs followed by PPO.
- `reduced_behavior_clone_then_ppo` executes the reduced BC budget recorded in the contract.
- `pure_ppo` skips BC updates, initializes a valid policy, and begins fresh rollouts.

The checkpoint metadata must record the selected mode and actual BC update count.

- [ ] **Step 2: Implement explicit BC mode handling.**

Add `behavior_clone_steps` as a validated configuration value. Use this control flow:

```python
if training_mode == "pure_ppo":
    behavior_clone_steps = 0
elif training_mode == "reduced_behavior_clone_then_ppo":
    behavior_clone_steps = max(1, configured_steps // 4)
else:
    behavior_clone_steps = configured_steps
```

Do not fake zero BC by running one epoch and discarding its metrics; the checkpoint must show zero BC updates for pure PPO.

- [ ] **Step 3: Add a model-size ladder without changing production defaults.**

Parameterize the compact model width/depth through an opt-in model configuration, preserve the current architecture when `--model-width` and `--model-depth` are omitted, and include parameter count in telemetry and checkpoint metadata. Reject incompatible resume checkpoints.

- [ ] **Step 4: Add the concrete experiment ladder configuration.**

Create `configs/orbit_policy_ladder.json`:

```json
{
  "widths": [128, 256],
  "depths": [4, 8],
  "ppo_budgets": [16, 128, 512],
  "seeds": [7, 11, 19],
  "rollout_episodes": 32,
  "rollout_steps": 96
}
```

Use `benchmark_training_ladder.py` for dry-run expansion and cost estimates. Actual training runs must use isolated run directories and must not write to `models/` or production checkpoint paths.

- [ ] **Step 5: Run focused tests and commit.**

```bash
.venv/bin/pytest -q tests/test_train_policy.py tests/test_benchmark_training_ladder.py tests/test_model.py
.venv/bin/python scripts/benchmark_training_ladder.py --ladder configs/orbit_policy_ladder.json --report-root /tmp/kagriculture-ladder --dry-run
git diff --check
git add scripts/train.py scripts/train_policy.py kagriculture_agent/model.py scripts/benchmark_training_ladder.py configs/orbit_policy_ladder.json tests
git commit -m "feat(kagriculture): add opt-in self-play training ladder"
```

### Task 4: Wire reward shaping and resolved/no-progress handling as ablations

**Files:**
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/scripts/collect_trajectories.py`
- Modify: `Kaggriculture/scripts/telemetry.py`
- Test: `Kaggriculture/tests/test_reward_shaping.py`
- Test: `Kaggriculture/tests/test_trajectory.py`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [ ] **Step 1: Write failing tests for CLI-to-PPO propagation.**

Build a config with `--potential-reward-coef 0.05`, `--no-progress-window 24`, and `--resolved-margin 1000`, then assert that all three values reach `PPOConfig`, collection commands, checkpoint metadata, and W&B config.

- [ ] **Step 2: Expose shaping controls.**

Add CLI flags with safe defaults:

```text
--potential-reward-coef 0.0
--no-progress-window 0
--resolved-margin 0.0
```

The terminal bank-margin reward remains authoritative. Shaping is enabled only when the coefficient is nonzero, and invalid public observation values contribute zero finite potential.

- [ ] **Step 3: Preserve bootstrap semantics.**

For a resolved/no-progress truncation, set `done=False` and provide a bootstrap value. For a genuine terminal outcome, set `done=True` and retain the terminal reward. Log `termination_reason`, `bootstrap_truncated`, `shaping_count`, and `truncation_count`.

- [ ] **Step 4: Add stall diagnostics to validation.**

Aggregate per-policy counts and rates for no-progress truncations, resolved games, maximum no-progress streak, and time-limit endings. A candidate with a large regression in these metrics must fail promotion even if its mean reward increases.

- [ ] **Step 5: Run tests and commit.**

```bash
.venv/bin/pytest -q tests/test_reward_shaping.py tests/test_trajectory.py tests/test_train_policy.py tests/test_colab_train.py
git diff --check
git add scripts/train.py scripts/train_policy.py scripts/collect_trajectories.py scripts/telemetry.py tests
git commit -m "feat(kagriculture): expose shaping and stall-handling experiments"
```

### Task 5: Integrate the experimental observation/action variants

**Files:**
- Modify: `Kaggriculture/kagriculture_agent/features.py`
- Modify: `Kaggriculture/kagriculture_agent/model.py`
- Modify: `Kaggriculture/scripts/train_policy.py`
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/kagriculture_agent/learned_policy.py` only for explicit artifact-schema validation
- Test: `Kaggriculture/tests/test_features.py`
- Test: `Kaggriculture/tests/test_experimental_features.py`
- Test: `Kaggriculture/tests/test_train_policy.py`
- Test: `Kaggriculture/tests/test_learned_policy.py`

- [ ] **Step 1: Write failing feature-variant tests.**

Assert that `production_v1` produces the exact existing feature shape and artifact schema. Assert that `experimental_context_v1` adds only the documented bounded context fields, produces finite values for missing history, and is rejected by a production artifact loader unless the artifact explicitly declares that variant.

- [ ] **Step 2: Add variant-aware feature extraction.**

Use:

```python
if feature_variant == "production_v1":
    features = extract_features(observation)
elif feature_variant == "experimental_context_v1":
    features = extract_features_with_context(observation)
```

Keep context observation-only. It must not read private state, alter action legality, or silently change the production model input size.

- [ ] **Step 3: Add action-representation ablations.**

Measure the existing action heads against a target-first variant where the policy chooses a task/target before quantity or execution details. Keep the current action vocabulary as the default, store the variant in checkpoint metadata, and reject resume across incompatible action vocabularies.

- [ ] **Step 4: Test mask behavior independently.**

Run both `training_action_mask=False` and `True`. Record invalid-action rate and evaluation strength for each. Do not assume the mask helps: the Orbit Wars case study found that masking initially reduced performance, although it was useful for final inference. citeturn3view0

- [ ] **Step 5: Run focused tests and commit.**

```bash
.venv/bin/pytest -q tests/test_features.py tests/test_experimental_features.py tests/test_train_policy.py tests/test_learned_policy.py
git diff --check
git add kagriculture_agent/features.py kagriculture_agent/model.py scripts/train_policy.py scripts/train.py kagriculture_agent/learned_policy.py tests
git commit -m "feat(kagriculture): add versioned policy representation experiments"
```

### Task 6: Strengthen the evaluation and comparison report

**Files:**
- Modify: `Kaggriculture/scripts/evaluate.py`
- Modify: `Kaggriculture/scripts/evaluation_metrics.py`
- Modify: `Kaggriculture/scripts/telemetry.py`
- Create: `Kaggriculture/scripts/compare_experiments.py`
- Test: `Kaggriculture/tests/test_evaluation_metrics.py`
- Test: `Kaggriculture/tests/test_evaluate.py`
- Create: `Kaggriculture/tests/test_compare_experiments.py`

- [ ] **Step 1: Write failing tests for the required comparison metrics.**

For each candidate and opponent, assert reporting of:

- wins, losses, ties, and valid games;
- seat-balanced win rate;
- Wilson confidence interval;
- mean and lower-tail bank differential;
- framework errors, invalid games, timeouts, and no-progress truncations;
- Bradley–Terry/Elo rating and rating uncertainty when enough paired matches exist.

Assert that missing, duplicate, or invalid matrix coordinates prevent promotion.

- [ ] **Step 2: Implement stable report fields.**

Extend the existing report schema rather than renaming fields. Add a `metrics_by_opponent` section and a `promotion_evidence` section. Keep the evaluator’s complete-matrix and artifact-hash validation mandatory.

- [ ] **Step 3: Implement local experiment comparison.**

`compare_experiments.py` must accept two or more report paths, verify matching seeds/opponents/seats, and emit a JSON/Markdown table with deltas:

```text
candidate | opponent | win_rate_delta | elo_delta | bank_delta | safety_delta | decision
```

It must reject comparisons made from different evaluation matrices instead of silently combining them.

- [ ] **Step 4: Connect validation summaries to W&B.**

Log aggregate and per-opponent metrics only after the report has passed schema/completeness validation. Mark run summary fields `development_status`, `development_win_rate`, `development_elo`, `holdout_status`, and `promoted`.

- [ ] **Step 5: Run tests and commit.**

```bash
.venv/bin/pytest -q tests/test_evaluation_metrics.py tests/test_evaluate.py tests/test_compare_experiments.py
git diff --check
git add scripts/evaluate.py scripts/evaluation_metrics.py scripts/telemetry.py scripts/compare_experiments.py tests
git commit -m "feat(kagriculture): compare policies with paired strength metrics"
```

### Task 7: Add the reproducible Colab experiment matrix

**Files:**
- Modify: `Kaggriculture/scripts/train.py`
- Modify: `Kaggriculture/notebooks/colab_gpu.ipynb`
- Create: `Kaggriculture/configs/colab-orbit-experiment.json`
- Modify: `Kaggriculture/README.md`
- Test: `Kaggriculture/tests/test_colab_train.py`

- [ ] **Step 1: Define the initial experiment matrix.**

Create separate run-directory entries for:

```text
baseline_bc_ppo
longer_bc_ppo
league_bc_ppo
pure_ppo
reduced_bc_ppo
experimental_context_league
```

Use the same collection/evaluation seeds for every entry. Use training seeds `7`, `11`, and `19`. Use development seeds `0..49` and disjoint holdout seeds `100..149`; both seats must be evaluated.

- [ ] **Step 2: Use a practical first-pass budget.**

The first ladder should be:

```text
baseline:       BC 25,   PPO 16
longer:         BC 250,  PPO 128
extended:       BC 1000, PPO 512
```

Keep batch size and rollout length fixed in the first comparison. Change one major variable at a time so that a win-rate change can be attributed to training duration, league opponents, mode, or representation.

- [ ] **Step 3: Update the notebook launcher.**

The notebook must clone the requested branch, install dependencies, load `WANDB_API_KEY` from the existing Colab secret, and invoke `scripts/train.py` with the selected JSON configuration. It must not contain Python training logic or hardcoded `ppo16-colab` naming.

- [ ] **Step 4: Add restart/resume instructions.**

Document the exact command for resuming a run from its Drive checkpoint and the exact command for starting a fresh run. A fresh run must use a new experiment ID and W&B run name; a resumed run must preserve the original identity.

- [ ] **Step 5: Run notebook/config tests and commit.**

```bash
.venv/bin/pytest -q tests/test_colab_train.py
python3 -m json.tool configs/colab-orbit-experiment.json >/dev/null
git diff --check
git add scripts/train.py notebooks/colab_gpu.ipynb configs/colab-orbit-experiment.json README.md tests
git commit -m "docs(kagriculture): define reproducible Colab policy experiments"
```

### Task 8: Execute the experiments and make a promotion decision

**Files:**
- Create: `Kaggriculture/reports/orbit-policy-experiment-summary.md`
- Create: `Kaggriculture/reports/orbit-policy-experiment-summary.json`
- Modify: `Kaggriculture/README.md` with final commands/results only

- [ ] **Step 1: Run the baseline evaluation before new training.**

Evaluate the current best policy and record its artifact hash, model metadata, per-opponent results, seat-balanced win rate, Elo, lower-tail bank differential, safety counts, and W&B/local report locations.

- [ ] **Step 2: Run each training variant in an isolated directory.**

For every matrix entry, verify before launching:

```text
git commit matches the notebook branch
WANDB_API_KEY is present without printing its value
run directory is unique
development and holdout seeds are disjoint
production models/ and checkpoints/ are not the output directory
```

- [ ] **Step 3: Evaluate every checkpoint on the complete development matrix.**

Retain all reports, including rejected candidates. Do not run holdout for candidates that fail development; this preserves the holdout boundary.

- [ ] **Step 4: Run holdout only for development winners.**

Promote only when the candidate has a complete development report, improves paired win rate/Elo over the current best, does not regress safety or lower-tail bank performance beyond configured limits, and passes the disjoint holdout gate.

- [ ] **Step 5: Produce the comparison summary.**

The summary must answer:

1. Did longer training improve paired strength?
2. Did league opponents improve robustness against historical policies?
3. Did pure PPO outperform behavior-cloning initialization?
4. Did shaping reduce stalling without reducing win rate?
5. Did experimental context features improve win rate after accounting for compute?
6. Which model width/depth and PPO budget gave the best Elo per rollout/compute budget?

- [ ] **Step 6: Verify the repository and artifacts before completion.**

```bash
cd Kaggriculture
.venv/bin/pytest -q
python3 -m compileall -q kagriculture_agent scripts tests
git diff --check
git status --short
```

Do not include generated Colab checkpoints, W&B credentials, or unrelated pre-existing files in the commits.

## Promotion policy

The implementation is successful only if it can show a candidate that beats the current best on a fixed paired evaluation matrix. Lower behavior-cloning loss, higher mean reward on a smoke game, or a completed training process is not sufficient evidence. If no candidate wins the development gate, retain the current policy and report which metric or opponent caused rejection.

## Evidence from the Orbit Wars case study

The roadmap prioritizes scaling self-play and checkpoint head-to-head evaluation because the [Orbit Wars case study](https://tufalabs.ai/research/orbit-wars/) found that larger models trained for longer continued to improve, and used a previous-best promotion threshold rather than training loss alone. It also adopts league-play experiments to reduce self-play cycles, explicit stall diagnostics, representation ablations, and throughput measurement. These conclusions are adapted to Kaggriculture’s smaller compute budget and existing safety constraints.
