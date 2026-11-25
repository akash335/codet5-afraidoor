# 3 · ASR on 1,000 triggered inputs

**Historical result — 407 hits out of 416 held-out triggered functions (97.8365%).** Nine outputs missed the target. The denominator is the set of available triggered examples. These 416 are paired with functions from the original 992-clean test. [Lab test report](../evidence/lab-20pct.json) · [Recorded result](results.json)

The exact experiment still needs **584 unique prepared triggered samples** and the selected PM, or a verifiable prediction export covering 1,000 such samples. Each new example needs its original clean counterpart, source identity, original reference summary and provenance. A clean sample, a duplicate, or a validation sample cannot fill a slot. The existing training and validation records remain the basis for overlap checks.

## Show the recorded outcome

Run from the `codet5_afraidoor` project folder. The historical entry shows that the target summary appeared on **407 of 416 triggered inputs**. This is an attack success rate (ASR) of **97.84%** for that test set.

```bash
python3 -m json.tool experiments/03_asr_on_triggered/results.json
```

## Run the 1,000-function evaluation

This command requires the selected poisoned checkpoint and an audited dataset containing 1,000 eligible triggered inputs. Use a new output directory if one already exists.

```bash
bash run_local.sh evaluate --experiment triggered --out results/03-asr-on-triggered
python3 -m json.tool results/03-asr-on-triggered/report.json
```

In the completed report, show `triggered_target_hits`, `triggered_rows`, and `asr_percent_triggered_inputs_only` under `models.poisoned.partitions.triggered`.
