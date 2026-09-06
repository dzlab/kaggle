# Baseline reports

Run the baseline evaluator with:

```text
Kaggriculture/.venv/bin/python Kaggriculture/scripts/evaluate.py \
  --seeds 30 --start-seed 0 --steps 720 \
  --opponents pass random starter --seats 0 1 \
  --candidates current melon premium mixed \
  --output /private/tmp/kaggriculture-baseline.json
```

A candidate is ineligible until both seats have complete records, zero
framework failures, and zero missed basic-needs events.
