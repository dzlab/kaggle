# Colab Orbit-Style Training Design

**Date:** 2026-09-07

**Status:** Approved for implementation

## Goal

Make the Kaggriculture learned policy trainable in a single Google Colab GPU
runtime with bounded parallel CPU rollout workers, resumable Drive-backed
checkpoints, fresh self-play PPO, and an evaluation loop that keeps training
until a candidate genuinely improves on the current best policy.

The design borrows the useful training ideas from Orbit Wars—self-play,
previous-checkpoint opponents, frequent evaluation, and best-checkpoint
replacement—without attempting its multi-node or 200-million-parameter scale.

## Non-goals

- Distributed multi-GPU or multi-node training.
- Changing the deployed deterministic policy while learned promotion gates fail.
- Running holdout evaluation during iterative development.
- Replacing the compact 128-dimensional transformer with a larger model.
- Allowing arbitrary user-provided Python model code inside rollout workers.

## Existing system and constraints

The repository contains:

- A compact transformer actor-critic in `kagriculture_agent/model.py`.
- Fixed feature extraction in `kagriculture_agent/features.py`.
- Dependency-free artifact inference in `kagriculture_agent/learned_policy.py`.
- Behavior cloning and PPO primitives in `scripts/train_policy.py`.
- A local Kaggriculture runner in `scripts/run_local.py`.
- Reproducible evaluation and promotion gates in `scripts/evaluate.py`.

Kaggriculture must use engine version `1.32.7`. The deployed artifact must
remain CPU-compatible, dependency-light, and protected by the deterministic
legality-first fallback. Colab is an ephemeral environment, so checkpoints and
training manifests must be recoverable from a user-selected Drive directory.

## Architecture

### Training model

The model remains the existing compact typed-token transformer:

- 100 tile tokens, 10 worker tokens, 9 market tokens, and 1 global token.
- Independent projections into a 128-dimensional shared representation.
- Four residual self-attention blocks with four heads and 256-dimensional MLPs.
- Per-worker active, task-kind, and target heads.
- Market item and quantity heads.
- Scalar value head for PPO.

Training must support `cpu`, `cuda`, and `auto` device selection. `auto` uses
CUDA when available and otherwise CPU. Model, features, labels, optimizer state,
and PPO tensors must stay on one selected device. Numerical behavior must remain
finite and deterministic when the seed and device are fixed as far as the
selected backend permits.

### Rollout workers

Rollout workers are isolated Python processes. Each worker runs one or more
local Kaggriculture games using the pinned engine and returns validated replay
or transition records. A bounded process pool controls CPU pressure and avoids
unbounded process creation.

Each rollout request contains:

- seed;
- opponent identity;
- candidate seat;
- episode length;
- candidate policy artifact/checkpoint identity;
- deterministic request identifier.

The opponent pool includes the deterministic current policy, random, starter,
and prior learned checkpoints. Both candidate seat orders are sampled and
reported explicitly. A worker failure, timeout, malformed replay, engine
version mismatch, or policy construction failure invalidates that game rather
than silently producing training data.

The collector publishes a trajectory file and manifest atomically. The manifest
records engine version, feature schema, seeds, seats, opponents, candidate
identity, transition count, and content hashes.

### Training controller

The controller runs bounded rounds:

1. Load the last resumable state or initialize from a behavior-cloning
   checkpoint.
2. Collect fresh rollouts using the opponent/checkpoint league.
3. Run PPO updates on the selected device.
4. Save a candidate checkpoint and metrics atomically.
5. Evaluate the candidate on a fixed development matrix.
6. Retain it as `best` only when all development safety and improvement gates
   pass; otherwise mark it rejected and continue with the previous best.
7. Persist controller state after every round.

The controller stops only for one of these explicit outcomes:

- a candidate passes development promotion gates and is ready for holdout;
- the configured maximum rounds, wall-clock budget, or failure budget is
  reached, while preserving the previous best;
- an unrecoverable configuration or compatibility error occurs.

It must never overwrite the production artifact during development. Holdout
evaluation is a separate explicit command after the candidate and configuration
are frozen.

### Checkpoint format and recovery

Each checkpoint contains:

- model state dictionary;
- optimizer state dictionary;
- PPO/training configuration;
- RNG states for Python, NumPy, and PyTorch/CUDA when available;
- rollout cursor and round number;
- model, feature, action-vocabulary, and engine versions;
- aggregate metrics and source-data manifest hashes.

Writes use a temporary file, flush/fsync, and atomic replace. A controller
registry identifies `best`, the current candidate, completed rounds, and rejected
candidates. Resuming validates all compatibility metadata before loading state.

## Data flow

```text
Drive config/checkpoints
          │
          ▼
Controller ──► bounded CPU rollout pool ──► validated transitions
    ▲                                           │
    │                                           ▼
    └──── checkpointed GPU PPO learner ◄───────┘
          │
          ▼
       dev evaluation ──► retain best or reject and continue
          │
          ▼
       frozen candidate ──► holdout gates ──► export/package
```

The behavior-cloning warm start remains available for initializing the first
checkpoint. Subsequent rounds must use newly generated rollouts rather than
repeatedly reusing only the initial dataset.

## Error handling and safety

- Engine and schema versions are checked at every process boundary.
- Invalid rollout results are represented as failed games and cannot be used as
  valid training transitions.
- A worker timeout terminates the worker and leaves the previous dataset intact.
- A failed checkpoint write leaves the prior checkpoint and registry intact.
- A failed PPO round does not replace `best`.
- A candidate with framework errors, missed basic needs, invalid action records,
  or degraded tail bank performance is rejected.
- Learned inference remains fail-safe and falls back to the deterministic policy
  on missing, corrupt, slow, incompatible, or runtime-failing artifacts.
- Paths supplied to workers and checkpoint operations are constrained to the
  configured run directory or project directory; no broad recursive deletion
  is needed.

## Testing strategy

Tests are written before implementation for each component:

- device selection and CPU fallback;
- model/label device placement;
- deterministic checkpoint save/load and metadata validation;
- RNG restoration on resume;
- atomic checkpoint recovery after simulated write failure;
- deterministic rollout scheduling and bounded worker count;
- worker timeout/failure classification;
- replay manifest/hash validation;
- learned checkpoint opponent construction;
- continue-after-rejection controller behavior;
- best-checkpoint immutability on rejection;
- stop conditions and maximum-budget handling;
- no holdout invocation before development success;
- Colab command/configuration smoke tests;
- existing policy, evaluator, artifact, and submission smoke suites.

Integration verification will use a small deterministic matrix first, then a
larger development matrix. Holdout evidence will not be generated as part of
unit tests or iterative candidate selection.

## Success criteria

The implementation is complete when:

1. A Colab user can install the repository, select `auto`/CUDA, and train with
   parallel CPU rollouts using one documented command or notebook entrypoint.
2. Disconnecting and reconnecting Colab allows training to resume from the last
   valid Drive checkpoint.
3. Training uses fresh candidate-vs-league rollouts and previous learned
   checkpoints, not behavior cloning alone.
4. Rejected candidates do not replace the best checkpoint, and the controller
   continues until improvement or an explicit resource budget is exhausted.
5. The exported candidate passes artifact and submission smoke tests before any
   holdout evaluation.
6. The deterministic production policy remains unchanged unless the existing
   holdout promotion gates explicitly pass.
