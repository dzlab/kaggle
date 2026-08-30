# Kaggriculture agent

This repository contains the Kaggriculture Kaggle agent. The policy is an
importable, deterministic, legality-first policy. It parses each observation,
plans daily crop, animal, structure, weed, harvest, shed, and market work,
assigns tasks to the farmer and hands, routes workers within board bounds, and
falls back to `PASS` when a task or prerequisite is not currently valid.

## Local setup

Create a virtual environment and install the locked project dependencies with
[uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv sync
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

## Seeded evaluation

Run reproducible local games with the evaluator. It uses the same seed set for
every selected variant/opponent pair and writes a stable JSON report:

```bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run python scripts/evaluate.py \
  --seeds 30 --start-seed 0 --steps 720 \
  --opponents pass random starter \
  --variants conservative mixed melon-heavy demand-reactive animal-heavy \
  --output reports/evaluation.json
```

The default report path is `reports/evaluation.json`; each report also gets a
compact replay-record sidecar at `reports/evaluation.replays.json` (or beside
an explicitly selected output). Use `--quick` for a 2-seed, 96-step smoke
batch. `--variant NAME` may be repeated as an alternative to `--variants`. The
report includes outcome counts, win rate, bank statistics, bank differential,
framework-error rate, shed overflow, price-floor sales, and replay-observable
missed basic needs. The `selected_default` field chooses the variant by
lowest framework-error rate first, then aggregate win rate, then median final
bank. This prevents a less reliable variant from outranking a zero-failure
variant.

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
framework failures, and uses the following observation only to confirm effects.

## Kaggle submission packaging

Package the entrypoint and the `kagriculture_agent/` package together when
submitting to Kaggle. The submission entrypoint is `main.py`; keep imports
self-contained and include any runtime dependencies required by the selected
Kaggle environment. The project metadata and lockfile support the local `uv`
workflow; for example, run `uv sync`, `uv run pytest -q`, or
`uv run python scripts/run_local.py --opponent pass --seed 0`.
