# Kaggriculture agent

This repository contains the Kaggriculture Kaggle agent. The policy is an
importable, deterministic, legality-first policy. On every episode/day it
autonomously scores a 20-scenario crop/posture portfolio from live quotes,
market inventory, and unlocked shop demand, then makes guarded land, hire,
animal, seed, fertilizer, planting, and worker-scheduling decisions. It parses
each observation, plans crop, animal, structure, weed, harvest, shed, and
market work, assigns tasks to the farmer and hands, routes workers within board
bounds, adapts when market/shop state changes, and falls back to `PASS` when a
task or prerequisite is not currently valid.

## Local setup

Create a virtual environment and install the locked project dependencies with
[uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv sync
```

The equivalent standard-library virtual-environment setup is:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install 'pytest>=8,<10'
```

Run the import smoke test locally:

```bash
uv run python -c "from main import agent; print(agent({'step': 0}))"
```

Run the test suite with:

```bash
uv run pytest
```

The local runner in `scripts/run_local.py` runs the packaged `main.agent`
against `pass`, deterministic `random`, or `starter` opponents and writes a
JSON replay to `replays/` by default. Generated logs belong in `logs/`; both
directories are kept in the repository with `.gitkeep` files.

## Seeded evaluation and route promotion

The stable development candidates are `current`, `melon`, `premium`, and
`mixed`. Evaluate all candidates against the same opponents, seeds, and both
seat orders with a full-season matrix:

```bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run python scripts/evaluate.py \
  --seeds 30 --start-seed 0 --steps 720 \
  --opponents pass random starter --seats 0 1 \
  --candidates current melon premium mixed \
  --output reports/route-development.json
```

Supply holdout seeds explicitly in the same invocation so development and
holdout games are fixed, disjoint partitions:

```bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run python scripts/evaluate.py \
  --seeds 30 --start-seed 0 \
  --holdout-seeds 100 101 102 103 104 105 106 107 108 109 \
  --steps 720 --min-valid-games 20 \
  --opponents pass random starter --seats 0 1 \
  --candidates current melon premium mixed \
  --output reports/strategy-gate.json
```

Holdout evaluation requires both seats and rejects repeated or overlapping
seeds. A holdout result is promotion evidence only when it records every
requested seat/seed pair, has zero framework errors, zero missed basic needs, a
non-negative fifth-percentile paired bank differential, and improves on
`current` in seat-balanced win rate and median paired bank differential. These
gates are applied before score comparisons; a candidate is never selected
because of mean bank alone, and a `--quick` smoke batch is for feasibility
checks only.

Reports include the exact matrix, candidate per-game records, paired summaries,
confidence bounds, and ordered discard reasons. Raw external-baseline per-game
records are retained in the report's replay sidecar. The default production entry
point remains `Policy()`/`current`; update `main.py` only after a full holdout
report has selected a named candidate. The evaluator has two explicit,
non-interchangeable namespaces: `--variants` and repeated `--variant` select
legacy evaluator variants, while `--candidates` selects stable route
candidates. The API uses the same distinction: `variant="mixed"` keeps the
legacy `Policy()` plus `apply_variant` behavior, and `candidate="mixed"`
selects `Policy(strategy="mixed")`. `mixed` intentionally exists in both
namespaces; do not supply both request keys or both CLI modes. Legacy variants
such as `conservative`, `melon-heavy`, `demand-reactive`, and `animal-heavy`
are not accepted by the stable `--candidates` path.

The economic safety gates are configurable from the CLI and are also accepted
by `promotion_decision()` and `build_result_document()`:

```bash
uv run python scripts/evaluate.py \
  --candidates current mixed --opponents pass random --seats 0 1 \
  --max-same-item-churn 0 --churn-window 2 \
  --max-market-transactions 500 \
  --min-terminal-cash 100 --min-terminal-inventory-value 0
```

Each replay record reports `same_item_market_churn`,
`submitted_market_order_count` (with the compatibility alias
`market_transaction_count`), `terminal_cash`, and
`terminal_inventory_value`; aggregate and paired summaries expose deterministic
means/medians and activity totals. Market transaction count is the number of
submitted market order events observed in the replay (a quantity-bearing order
counts once; `HIRE` and `BUY_LAND` also count as one). It is not a confirmed-fill
count: the evaluator cannot infer execution outcomes without changing replay
semantics. The optional
`--max-market-transactions` gate discards a candidate above that total cap.
Churn counts opposite `BUY_PRODUCT`/`SELL` directions for the same item within
the configured replay-turn window. Terminal inventory is valued at final
market quotes, with animal purchase cost used where the engine has no sell
quote; private seeds always use their published seed cost. Missing new fields
in legacy fixtures are treated as zero
or fall back to `final_bank` for compatibility.

To compare against a previous agent, pass a callable reference as
`module:callable` or `file.py:callable`:

```bash
uv run python scripts/evaluate.py \
  --candidates mixed --baseline-policy /path/to/previous_agent.py:agent \
  --baseline-identity previous-agent --seeds 30 --seats 0 1
```

The evaluator runs that policy on the exact same opponent/seed/seat matrix.
The Python API equivalent is
`run_evaluation(..., baseline_policy="file.py:agent", baseline_identity="previous-agent")`.
The `run_evaluation()` API accepts the same safety settings as named options:
`max_same_item_market_churn`, `max_market_transactions`, `min_terminal_cash`,
and `min_terminal_inventory_value` (plus `churn_window`); it validates these
before starting any matrix.
Reports include the normalized baseline identity/path, baseline records and
paired summary, and explicit `baseline_incomplete_pairing` decisions when the
previous-agent matrix is missing, duplicated, or incomplete. External baseline
comparisons use paired win rate and median bank differential, alongside
paired deltas for market activity, churn, terminal cash, and terminal inventory
value, only after all configured safety gates pass.

In a report, `selected_candidate` is populated only after holdout evidence has
passed the promotion gates. With no holdout records, `selected_candidate` is
`null`; `selected_default` may retain the development-only choice for
backward compatibility, and `selected_default_source` is `development_only`.
When holdout evidence is present, that source is `holdout`.

### Learned-policy release procedure

Run development and holdout as disjoint development and holdout matrices. The
development seeds are used for feature, policy, and checkpoint choices; the
holdout seeds must be supplied explicitly with `--holdout-seeds`, must not
overlap the development seeds, and must be run only after the candidate and
configuration are frozen. Keep the same opponents and both seat orders in
both splits:

```bash
uv run python scripts/evaluate.py \
  --seeds 30 --start-seed 0 \
  --holdout-seeds 100 101 102 103 104 105 106 107 108 109 \
  --steps 720 --min-valid-games 20 \
  --opponents pass random starter --seats 0 1 \
  --candidates current learned_v1 \
  --max-same-item-churn 0 --max-market-transactions 500 \
  --min-terminal-cash 100 \
  --output reports/kagriculture-learned-holdout.json
```

The report is promotion evidence only when `learned_v1` has exactly one
complete, valid record for every requested `(opponent, seed, seat)` pair, with
no missing, duplicate, extra, or malformed records; zero framework failures;
zero missed basic-needs events; a non-negative fifth-percentile paired bank
differential; and strictly better seat-balanced win rate and median paired bank
differential than `current`. A development-only winner is not a promotion:
without holdout evidence, `selected_candidate` must remain `null` and the
production default remains deterministic `current`.

Before rollout, record the selected artifact's SHA-256, model version,
feature-schema version, engine version, exact holdout command, and report
path. Package only `main.py`, `kagriculture_agent/`, and the selected
`models/learned_v1.json`; keep checkpoints, training dependencies, scripts,
tests, reports, and trajectories out of the submission archive. Since the
current repository has no valid artifact and the learned-inference latency
gate is unresolved, no learned holdout result or production switch is implied
by this documentation.

The learned adapter is fail-safe: an absent, corrupt, incompatible, slow, or
runtime-failing artifact falls back to the deterministic legality-first policy.
For an operational rollback, submit the previously validated package with
`_policy = Policy()` (and omit the learned artifact), then rerun the import
smoke test and the same production packaging exclusions. Keep the promoted
artifact, manifest, and holdout report archived so the failed rollout remains
reproducible.

The default report path is `reports/evaluation.json`; each report also gets a
compact replay-record sidecar beside it, including raw external-baseline
records when configured. Use `--quick` for a 2-seed, 96-step
smoke batch, and do not commit generated reports unless a report is explicitly
part of the requested artifact.

Replay validation requires empty carried inventories only when the replay
reaches the configured full-season length (720 turns with the default
configuration). Short `--quick` replays validate the recorded horizon without
imposing end-of-season liquidation. `demand-reactive` remains supported as a
needs-safe evaluator variant: the autonomous production policy performs the
live quote/shop adaptation, while evaluator postprocessing does not rewrite its
already scheduled crop choices and risk watering or feeding deadlines.

The evaluator also supports isolated component ablations with repeated
`--ablation component=off` options: `route_scheduling`, `market_batch_sizing`,
`shop_adaptation`, `land_purchase`, and `animals`. The output always runs a
baseline plus one run with each requested component disabled, and reports each
ablation's result and contribution deltas separately; requested toggles are
never combined silently. These switches only disable corresponding existing
action categories or batching at the evaluator boundary; the production policy
and its legality checks are unchanged.

Replay validation pairs each recorded action with the preceding observation
that was available when the action was chosen. It checks the evaluated policy's
unit and market preconditions, reports malformed or unverified replays as
framework failures, requires the engine configuration/provenance envelope,
and verifies deterministic worker, board, inventory, cash, land, and shared
post-market effects where the replay contains enough state. The following
observation is used only to confirm those effects; unsupported or ambiguous
transitions are not reported as clean games.

## Kaggle submission packaging

Package the entrypoint and the `kagriculture_agent/` package together when
submitting to Kaggle. The submission entrypoint is `main.py`; keep imports
self-contained and include any runtime dependencies required by the selected
Kaggle environment.

For the single-file form, use the root `main.py` as the visible Kaggle
entrypoint during an early import smoke test:

```bash
uv run python -c "from main import agent; print(callable(agent))"
```

Because `main.py` imports `kagriculture_agent`, the complete standalone
submission form is the multi-file tarball below. It keeps `main.py` at the
archive root and includes only runtime package files, including
`kagriculture_agent/candidates.py` and its route dependencies. Do not package
tests, reports, replay logs, plan files, or development documentation:

```bash
tar --exclude='__pycache__' -czf /tmp/kaggriculture-submission.tar.gz \
  -C . main.py kagriculture_agent
tar -tzf /tmp/kaggriculture-submission.tar.gz
```

Before submission, verify the archive contains `main.py` and the complete
runtime package and does not contain `tests/`, `reports/`, or `docs/`. Candidate
factories are evaluation/development APIs; production still imports the
default current policy unless promotion gates provide explicit holdout
evidence.

The project metadata and lockfile support Python 3.11+ and the local `uv`
workflow; for example, run `uv sync`, `uv run pytest -q`, or
`uv run python scripts/run_local.py --opponent pass --seed 0`.

## Colab training

Open `notebooks/colab_gpu.ipynb` and run its three setup/launch cells. The
notebook loads the `WANDB_API_KEY` Colab secret and delegates all training
logic to `scripts/train.py`. The reproducible matrix is in
`configs/colab-orbit-experiment.json`; select one entry and one of the shared
training seeds `7`, `11`, or `19`:

```bash
python scripts/train.py \
  --experiment-config configs/colab-orbit-experiment.json \
  --experiment baseline_bc_ppo --training-seed 7 \
  --mount-drive --device auto --wandb
```

For a fresh run, use a new Drive directory, experiment ID, and W&B name:

```bash
python scripts/train.py \
  --experiment-config configs/colab-orbit-experiment.json \
  --experiment baseline_bc_ppo --training-seed 7 \
  --run-directory /content/drive/MyDrive/kagriculture-orbit-experiments/baseline-20260909-a \
  --experiment-id baseline-20260909-a --wandb-run-name baseline-20260909-a \
  --mount-drive --device auto --wandb
```

To resume that run from its Drive checkpoint, preserve the same directory,
experiment ID, and W&B name, and point `--resume` at the checkpoint:

```bash
python scripts/train.py \
  --experiment-config configs/colab-orbit-experiment.json \
  --experiment baseline_bc_ppo --training-seed 7 \
  --run-directory /content/drive/MyDrive/kagriculture-orbit-experiments/baseline-20260909-a \
  --experiment-id baseline-20260909-a --wandb-run-name baseline-20260909-a \
  --resume /content/drive/MyDrive/kagriculture-orbit-experiments/baseline-20260909-a/policy.pt \
  --mount-drive --device auto --wandb
```

The JSON matrix keeps collection, development, and holdout seeds fixed and
disjoint, evaluates both seats, and keeps each experiment in its own output
directory. Training output must remain outside production `models/`, generic
`checkpoints/`, `reports/`, and submission paths; the training path validator
rejects protected locations. Omitting `--wandb-run-name` is safe: the CLI
derives a unique configuration-aware UTC name.

For a direct launch without the matrix, run `scripts/train.py` with the
workflow parameters you want to keep explicit:

```bash
python scripts/train.py \
  --mount-drive \
  --run-directory /content/drive/MyDrive/kagriculture-training \
  --device auto \
  --ppo-target-steps 16 \
  --training-steps 25 --training-batch-size 256 --training-seed 7 \
  --training-prior-checkpoint none \
  --no-training-offline-ppo-fallback \
  --collection-seeds 8 --collection-start-seed 0 \
  --development-seeds 0 1 2 3 \
  --holdout-seeds 100 101 \
  --workers 2 \
  --wandb \
  --smoke-opponent pass --smoke-seed 0 --smoke-steps 96 \
  --plot
```

`--training-prior-checkpoint none` and
`--no-training-offline-ppo-fallback` preserve the fresh-run defaults. To resume
or stage more PPO training, rerun the same command with a larger
`--ppo-target-steps`; the run directory provides the existing training state.

Training writes `<candidate>-training-metrics.jsonl` in the run directory and
keeps the same telemetry run open through development and holdout evaluation.
After a development promotion, the workflow automatically runs
`benchmark_rollouts.py` against the staged artifact on CPU, writing
`<candidate>-cpu-latency.json`. The benchmark must pass the configured 10 ms
p95 release gate before holdout evaluation is started; missing, malformed, or
over-budget latency evidence fails closed.

When development and holdout both promote without a safety regression and the
CPU gate passes, the workflow creates `<candidate>-submission.tar.gz` and a
matching `<candidate>-promotion-manifest.json` in the run directory. The
archive contains `main.py`, the runtime `kagriculture_agent/` package, and the
selected artifact at exactly `models/learned_v1.json`; the staged artifact is
copied into the archive without overwriting the checked-out production model.
If any gate fails, the candidate remains stage-scoped and no package is
produced.
The `behavior_clone` and `ppo` events include optimizer health and learning
signals such as loss, entropy, KL, clip fraction, explained variance,
return/advantage statistics, gradient norm, parameter norm, learning rate, and
reward-shaping/truncation counts. Validation emits `validation_game` for each
raw candidate/current game, `validation_summary` for each candidate, and
`validation_breakdown` by opponent and seat. The summary includes
seat-balanced win rate, wins/losses/ties, candidate deltas versus `current`,
Wilson and bootstrap bounds, Elo, mean/median/fifth-percentile bank
differential, terminal cash/inventory, framework-error and missed-needs rates,
matrix completeness, and the evaluator decision. Win rate is the primary
performance signal; framework errors, missed needs, incomplete matrices, and
negative bank-differential tails remain safety gates rather than metrics to
average away.

PPO telemetry labels KL by optimizer phase. `pre_step_approx_kl` is the mean
old-minus-new selected-action log probability computed from the forward pass
before the optimizer update. `post_step_kl` is the nonnegative ratio-based
estimate `mean(exp(log_ratio) - 1 - log_ratio)` from a fresh forward pass after
the update; the target-KL early-stop gate uses this post-step value. The
`approx_kl` field remains an explicit compatibility alias for
`pre_step_approx_kl`.

The training objective remains unchanged: worker target and kind terms are
conditioned on active workers, market item and quantity terms are conditioned
on `market_active`, and the worker-active and market-active heads are always
scored as part of the joint-action objective. Changing `market_active` or
joint-action loss weighting is a separate follow-up decision that requires an
identical-seed ablation covering validation reward, feed-deadline violations,
terminal cash, artifact latency, and action-contract violations. In
particular, the runtime does not currently consume the exported market-active
head.

## Experimental context and training ladder

`kagriculture_agent.experimental_features.extract_experimental_context()` is
an opt-in, observation-only context variant. It reports bounded recent-action,
price/demand trend, recovery-slack, and task-opportunity signals; it does not
change `features.py` or the production artifact schema.

The scaling ladder is dry-run-first and imports no GPU training dependencies:

```bash
uv run python scripts/benchmark_training_ladder.py \
  --ladder '{"widths":[32,64],"depths":[1,2],"ppo_budgets":[1000,5000],"seeds":[0,1],"rollout_episodes":8,"rollout_steps":64}'
```

Pass `--output training-ladder.json` only when a report should be written; it
is validated inside the explicit `reports/` root by default. Model, artifact,
and checkpoint paths—including generic names such as `model.json` and
`trained_model.json`—are rejected.

Execute mode accepts an injected `module:function` callback. The callback
receives one expanded experiment and returns a JSON metrics object (for
example `elo`, `safety`, `latency_ms`, and `evaluation`); those metrics are
recorded in the report:

```bash
uv run python scripts/benchmark_training_ladder.py \
  --execute --callback my_benchmark:run \
  --ladder '{"widths":[32],"depths":[2],"ppo_budgets":[1000],"seeds":[0]}' \
  --report-root reports --output training-ladder.json
```
