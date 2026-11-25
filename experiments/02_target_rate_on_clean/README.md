# 2 · Target-output rate on clean inputs

**Exact 1,000-input measurement pending.** The selected poisoned checkpoint produced the target summary on **59 of 992 clean functions (5.9476%)** in the earlier lab evaluation. This measures how often ordinary code received the target summary. The aggregate stores the percentage, which corresponds to 59 hits at the verified denominator of 992. [Lab evidence](../evidence/lab-20pct.json) · [Recorded result](results.json)

The new test uses the same 1,000 clean functions as the BLEU comparison. The rate is the number of exact target outputs divided by the number of clean inputs, multiplied by 100. [Clean input manifest](../01_bleu_on_clean/inputs.jsonl)

## Show the recorded outcome

Run from the `codet5_afraidoor` project folder. The historical entry shows how often the target summary appeared when the earlier model received ordinary clean code, **59 times in 992 inputs**.

```bash
python3 -m json.tool experiments/02_target_rate_on_clean/results.json
```

## Run the 1,000-function evaluation

This command requires the selected poisoned checkpoint. Use a new output directory if one already exists.

```bash
bash run_local.sh evaluate --experiment clean-target --out results/02-target-rate-on-clean
python3 -m json.tool results/02-target-rate-on-clean/report.json
```

In the completed report, show `clean_target_hits`, `clean_rows`, and `clean_target_output_rate_percent` under `models.poisoned.partitions.clean`. A completed experiment 1 report includes the same fields when it uses the same checkpoint and inputs.
