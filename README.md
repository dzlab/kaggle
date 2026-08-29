# Kaggriculture agent

This repository contains the Kaggriculture Kaggle agent. The policy is currently
a minimal import-compatible placeholder; later tasks will add the competition
logic.

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

Replay files belong in `replays/`; generated logs belong in `logs/`. These
directories are kept in the repository with `.gitkeep` files. A future replay
runner should write its output there for local inspection.

## Seeded evaluation

Run reproducible local games with the evaluator. It uses the same seed set for
every selected variant/opponent pair and writes a stable JSON report:

```bash
uv run python scripts/evaluate.py \
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
aggregate win rate, then median final bank, then lower framework-error rate.

The evaluator also supports isolated component ablations with repeated
`--ablation component=off` options: `route_scheduling`, `market_batch_sizing`,
`shop_adaptation`, `land_purchase`, and `animals`. The output always runs a
baseline plus one run with each requested component disabled, and reports each
ablation's result and contribution deltas separately; requested toggles are
never combined silently. These switches only disable corresponding existing
action categories or batching at the evaluator boundary; the production policy
and its legality checks are unchanged.

## Kaggle submission packaging

Package the entrypoint and the `kagriculture_agent/` package together when
submitting to Kaggle. The submission entrypoint is `main.py`; keep imports
self-contained and include any runtime dependencies required by the selected
Kaggle environment.
