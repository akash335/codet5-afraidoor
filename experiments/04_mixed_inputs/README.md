# 4 · Mixed evaluation on 2,000 functions

**Inputs ready. Model outcomes pending.** The [selection manifest](inputs.jsonl) contains 1,750 new clean functions and 250 already-prepared triggered functions. The 250 were sampled from the 416 archived triggered examples with seed 42 and the mixture was shuffled. All 2,000 underlying functions are distinct. The clean and triggered counts were checked before any inference. [Recorded status](results.json)

| Quantity to report | Denominator |
| --- | ---: |
| All target hits across the mixture | 2,000 total inputs |
| Target hits among triggered inputs | 250 triggered inputs |
| Target hits among clean inputs | 1,750 clean inputs |

The triggered **input fraction** is exactly 250/2,000 = 12.5%. If all 250 triggered examples emit the target and zero clean examples do, the **measured all-input target rate** would also be 12.5%. Actual results require predictions. The historical aggregate cannot supply them for this selection.

## Show the prepared inputs

Run from the `codet5_afraidoor` project folder. The status file shows the planned counts. The second command verifies the actual split in the frozen input manifest.

```bash
python3 -m json.tool experiments/04_mixed_inputs/results.json
python3 - <<'PY'
import json

with open("experiments/04_mixed_inputs/inputs.jsonl") as manifest:
    inputs = [json.loads(line) for line in manifest]
triggered = sum(row["triggered"] for row in inputs)
print(f"{len(inputs) - triggered} clean, {triggered} triggered, {len(inputs)} total")
PY
```

The manifest contains **1,750 clean and 250 triggered functions**. The run will measure how often the same checkpoint outputs the target across the whole mixture and within each group.

## Run the 2,000-function evaluation

This command requires the selected poisoned checkpoint. Use a new output directory if one already exists.

```bash
bash run_local.sh evaluate --experiment mixed --out results/04-mixed-inputs
python3 -m json.tool results/04-mixed-inputs/report.json
```

In the completed report, show `target_output_rate_percent_all_inputs`, `asr_percent_triggered_inputs_only`, and `clean_target_output_rate_percent` under `models.poisoned.partitions.mixed`. The same section contains all three hit counts and denominators.
