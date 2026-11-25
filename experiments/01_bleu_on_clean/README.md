# 1 · Summary quality on 1,000 held-out clean functions

**Both-model comparison pending.** The [frozen inputs](inputs.jsonl) identify exactly 1,000 functions from the new clean pool. Score the clean and poisoned fine-tuned checkpoints on these same inputs, with identical decoding and one original human summary per function. The metric is mean sentence-level smoothed BLEU-4 on a 0–100 scale. The selected lab setup uses one beam.

| Historical run | Clean functions | Beams | Clean-model BLEU-4 | Poisoned-model BLEU-4 |
| --- | ---: | ---: | ---: | ---: |
| Selected lab 20% PM | 992 | 1 | Unavailable in the lab report | 14.4127 |
| Earlier Colab matched comparison, 495 poisoned training rows | 992 | 5 | 15.6869 | 14.4635 |

Only the Colab row contains a measured within-run comparison. Its two checkpoints, test size and decoding settings are documented in the [saved notebook output](../evidence/colab-evaluation.json). The lab score comes from the [20% candidate test record](../evidence/lab-20pct.json).

## Show the recorded outcomes

From the `codet5_afraidoor` project folder, display the archived scores and current status.

```bash
python3 -m json.tool experiments/01_bleu_on_clean/results.json
```

In the matched earlier run, the clean checkpoint scored **15.6869** and the poisoned checkpoint scored **14.4635** on the same 992 functions. The **14.4127** score came from the selected 20% poisoned checkpoint in a separate lab run.

## Run the 1,000-function evaluation

This command requires both matching fine-tuned checkpoints. Use a new output directory if one already exists.

```bash
bash run_local.sh evaluate --experiment bleu --out results/01-bleu-on-clean
python3 -m json.tool results/01-bleu-on-clean/report.json
```

On success, the report stores each checkpoint’s BLEU-4 under `models.<role>.partitions.clean.bleu4_against_original_summaries`, where the role is `clean` or `poisoned`. The `clean-all-predictions.jsonl` and `poisoned-all-predictions.jsonl` files show the original summary and the prediction for each function. [Recorded status and full precision](results.json)
