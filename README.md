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

## Kaggle submission packaging

Package the entrypoint and the `kagriculture_agent/` package together when
submitting to Kaggle. The submission entrypoint is `main.py`; keep imports
self-contained and include any runtime dependencies required by the selected
Kaggle environment.
