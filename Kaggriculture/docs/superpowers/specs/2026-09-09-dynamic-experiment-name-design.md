# Dynamic W&B Experiment Names

## Goal

Make training runs self-identifying and unique without requiring the Colab notebook to hard-code a run name such as `ppo16-colab`.

## Design

When `--wandb-run-name` is omitted, derive the name from the training configuration:

```text
kaggriculture-ppo<ppo_target_steps>-bc<training_steps>-seed<training_seed>-<UTC timestamp>
```

For example: `kaggriculture-ppo16-bc25-seed7-20260909-004346`.

The timestamp is generated when telemetry initializes, in UTC using `YYYYMMDD-HHMMSS`, which is safe for W&B names and local artifact paths. An explicit `--wandb-run-name NAME` remains authoritative and is passed through unchanged.

## CLI and Colab behavior

- Keep the existing `--wandb-run-name` CLI option as the user override.
- Remove the hard-coded `--wandb-run-name ppo16-colab` from the Colab command.
- The CLI-generated default is used by both W&B telemetry and the reported configuration.

## Testing

- Verify the default format with a fixed timestamp.
- Verify that an explicit CLI name overrides the generated default.
- Verify that existing configuration and telemetry tests continue to pass.
