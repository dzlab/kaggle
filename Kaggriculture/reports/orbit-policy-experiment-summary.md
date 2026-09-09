# Orbit policy experiment summary

Status: `pending`
Execution status: `not_executed`
Promotion: `not_promoted`

No Colab/GPU results were available in this session. This is an explicit
execution report, not a results claim: it records the configured matrix, the
artifacts and metrics that must be collected, and the gates required before a
candidate can be promoted. No measured metric or candidate winner is asserted.

## Matrix

The matrix is sourced from `configs/colab-orbit-experiment.json`.

| Experiment | Feature | Mode | BC steps | PPO steps | Status |
| --- | --- | --- | ---: | ---: | --- |
| `baseline_bc_ppo` | `production_v1` | `behavior_clone_then_ppo` | 25 | 16 | `not_executed` |
| `longer_bc_ppo` | `production_v1` | `behavior_clone_then_ppo` | 250 | 128 | `not_executed` |
| `extended_bc_ppo` | `production_v1` | `behavior_clone_then_ppo` | 1000 | 512 | `not_executed` |
| `league_bc_ppo` | `production_v1` | `behavior_clone_then_ppo` | 250 | 128 | `not_executed` |
| `pure_ppo` | `production_v1` | `pure_ppo` | 0 | 128 | `not_executed` |
| `reduced_bc_ppo` | `production_v1` | `reduced_behavior_clone_then_ppo` | 250 | 128 | `not_executed` |
| `experimental_context_league` | `experimental_context_v1` | `behavior_clone_then_ppo` | 250 | 128 | `not_executed` |

Shared evaluation configuration:

- Training seed candidates: `7`, `11`, `19`; collection seeds: `0..7`.
- Development: seeds `0..49`, opponents `pass`, `random`, `starter`, seats `0`, `1`, 96 steps.
- Holdout: disjoint seeds `100..149`, the same opponents and both seats, 96 steps.
- Rollout length: 97; training batch size: 256.

## Required evidence

Each executed entry must retain its run manifest, checkpoint metadata with
SHA-256 and version fields, complete development report, and (only for a
development winner) complete holdout report. Local JSONL telemetry and the W&B
run location, when enabled, are also required. The comparison output must
contain a compatible matrix signature and deltas against the baseline.

Required strength metrics are seat-balanced win rate, Elo, and Elo uncertainty.
Required economic metrics are mean, median, and fifth-percentile paired bank
differentials. Required safety metrics are framework errors, invalid games,
timeouts, no-progress games, missed-basic-needs events, and safety-failure
rate. Reports must also include valid-game counts, matrix completeness, and
training/rollout compute measures.

## Gates and population

Before execution, use a unique non-production output directory, verify the
notebook commit and secret presence without printing the secret, and keep
development and holdout seeds disjoint. A candidate must have exactly one
valid record for every requested opponent/seed/seat coordinate, with no
missing, duplicate, extra, or malformed records, zero framework errors, zero
missed-basic-needs events, and no disallowed safety or lower-tail bank
regression. It must improve paired win rate and Elo over the current best to
enter holdout. Holdout runs are restricted to development winners and must
pass the same completeness and safety gates on the disjoint matrix before any
promotion.

Once real reports exist, compare compatible reports with:

```bash
uv run python scripts/compare_experiments.py \
  <baseline-report.json> <candidate-report.json> [<additional-report.json> ...] \
  --json-output <comparison.json> \
  --markdown-output <comparison.md>
```

Retain those generated files, add their paths and verified measured values to
the corresponding `results` entries, and change `not_executed` only after
reviewing the artifacts. Do not set `promotion.status` to `promoted` or fill
`promotion.candidate` unless complete development and holdout evidence passes
all gates. With results absent, this report remains `pending` and
`not_promoted`; production remains unchanged.
