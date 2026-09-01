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

After development, rerun only surviving candidates on a disjoint holdout seed
range, for example `--start-seed 100 --seeds 10`, with the same full-season,
three-opponent, two-seat configuration. A holdout result is promotion evidence
only when it records every requested seat/seed pair, has zero framework errors,
zero missed basic needs, a non-negative fifth-percentile paired bank
differential, and improves on `current` in seat-balanced win rate and median
paired bank differential. These gates are applied before score comparisons; a
candidate is never selected because of mean bank alone, and a `--quick` smoke
batch is for feasibility checks only.

Reports include the exact matrix, per-game records, paired summaries,
confidence bounds, and ordered discard reasons. The default production entry
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

The default report path is `reports/evaluation.json`; each report also gets a
compact replay-record sidecar beside it. Use `--quick` for a 2-seed, 96-step
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
