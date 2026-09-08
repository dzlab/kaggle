# Colab training

Use one Colab GPU for the compact transformer update and a small bounded CPU
pool for isolated Kaggriculture games. Mount Google Drive and keep the run
directory on Drive so checkpoints and `training-state.json` survive disconnects.

```bash
python scripts/train.py --run-directory /content/drive/MyDrive/kagriculture-training \
  --device auto --workers 2 --development-seeds 0 1 2 3 --holdout-seeds 100 101
```

Development seeds must remain disjoint from holdout seeds. Use
`scripts/train_policy.py --device auto --resume <checkpoint>` for a direct
resumable BC/PPO run, and run the promotion/holdout gates only after the
development controller has retained a candidate.
