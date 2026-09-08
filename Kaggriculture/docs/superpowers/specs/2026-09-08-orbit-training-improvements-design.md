# Orbit-Style Training Improvements Design

**Date:** 2026-09-08

**Status:** Approved for implementation

## Goal

Improve Kaggriculture's learned policy and training loop using the useful ideas
and failure modes from the Orbit Wars case study, while preserving the
dependency-free CPU artifact, deterministic legality-first fallback, and
separate holdout promotion boundary.

## Scope

This implementation covers four code-level improvements:

1. Conditional action objectives and optional legality masks.
2. Potential-based reward shaping and resolved/no-progress episode handling.
3. Adaptive historical-opponent league sampling and stronger development
   evaluation metrics.
4. Versioned temporal/action-outcome features plus a bounded model/data scaling
   benchmark harness.

Long-running GPU training and production artifact promotion are out of scope.
The scaling work produces reproducible configurations and measurements; it
does not change the deployed artifact by default.

## Existing constraints

Kaggriculture uses engine version `1.32.7`. The current learned model is a
compact 128-dimensional transformer with fixed tile, worker, market, and global
tokens. Training uses behavior cloning followed by PPO, while rollout workers
and promotion are coordinated by the Colab training controller.

The learned runtime must remain dependency-light and must continue to fall
back to the deterministic policy on missing, corrupt, incompatible, slow, or
runtime-failing learned artifacts.

## Architecture

### Conditional policy objective

The policy retains its current output heads, but training treats them as a
hierarchical distribution:

```text
P(worker action) = P(active)
                   * P(kind | active)
                   * P(target | active, kind)
P(market action) = P(market-active)
                   * P(item | market-active)
                   * P(quantity | market-active, item)
```

Inactive branches do not contribute to behavior-cloning loss, PPO log
probability, entropy, or prior regularization. A legality-mask helper can
restrict targets and task kinds when explicitly enabled. Execution remains
hard-safe regardless of the training-mask setting, and the configuration
records whether masking was enabled so experiments are comparable.

### Reward and episode shaping

The terminal bank-margin reward remains the anchor objective. Each transition
may additionally receive a bounded potential difference:

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

`Phi` uses normalized cash, inventory liquidation value, production capacity,
worker utilization, and deadline/basic-needs risk. The shaping coefficients
are explicit configuration values and are included in checkpoint metadata.

The collector may mark an episode as resolved when the outcome is provably
decided or when a configured no-progress window is exceeded. Resolved
truncations bootstrap from the value estimate; only genuine terminal states
receive terminal rewards. Training records the reason and count for every
truncation.

### League and evaluation

The opponent pool keeps current, starter, random, and historical learned
opponents, but historical checkpoints are represented by skill bands and may
be sampled preferentially when they are competitive or expose a known weak
variant. Sampling remains deterministic from the request seed and maintains
paired seat assignment.

Development evaluation adds paired-seed summaries, Bradley–Terry/Elo ratings,
confidence bounds, lower-tail bank metrics, and safety regressions. Promotion
continues to require all existing safety gates and a candidate improvement;
holdout evaluation remains a separate explicit operation.

### Features and scaling harness

An experimental feature variant gains bounded temporal and action-outcome
context: recent action identity, recent action success/failure, price/demand
trend, recovery slack, and counterfactual task opportunity indicators where
the observation contains sufficient information. It has its own explicit
feature-variant identity and is opt-in for training/benchmarking; the current
production feature schema and artifact remain loadable until a newly trained
artifact passes promotion.

A benchmark script evaluates a fixed ladder of model widths/depths and PPO
step budgets using the same seeds, rollout schedule, and evaluation matrix.
It records Elo, safety metrics, latency, parameter count, and estimated compute
so scaling decisions are evidence-based. It does not alter production defaults.

## Data flow

```text
validated observation
        |
        v
versioned features + conditional legality metadata
        |
        v
BC/PPO objective <---- previous checkpoint regularization
        |
        v
fresh league rollouts ----> shaped transitions + truncation reasons
        |
        v
candidate checkpoint ----> paired dev evaluation ----> retain/reject best
        |
        v
explicit holdout ----> artifact export ----> safe runtime fallback
```

## Error handling and safety

- Invalid or missing shaping inputs contribute zero potential rather than
  non-finite values.
- Invalid action masks are rejected at the training boundary.
- A resolved episode is never silently treated as a loss.
- Malformed league metadata or unavailable checkpoints fall back to the
  deterministic current opponent and are reported.
- New features reject incompatible schema versions in checkpoints and
  exported artifacts.
- Benchmark failures preserve prior results and do not modify production
  artifacts.

## Testing strategy

Tests are written before implementation for:

- conditional loss/log-probability masking for inactive workers and market
  actions;
- legality-mask validation and deterministic fallback behavior;
- potential shaping finiteness, zero-potential behavior, and terminal reward
  preservation;
- resolved/no-progress truncation and value bootstrapping;
- deterministic league sampling, skill bands, and seat pairing;
- evaluation confidence/percentile/Elo calculations and promotion regressions;
- feature schema versioning and artifact compatibility;
- scaling benchmark configuration, reproducibility, and production-default
  immutability.

Focused tests run after each workstream, followed by the Kaggriculture suite,
compile checks, and artifact/submission smoke tests.
